//! KV cache for attention. Per-layer K/V device buffers with optional sharing.
//!
//! Layout (per physical slot): K and V are `[max_len, h_kv, head_dim]` bf16.
//! Layers that share KV point at another slot; reads/writes route through the
//! `slot_for_layer` map. `cur_len` tracks how many positions have been written
//! so far; append writes new K/V starting at `cur_len`.
//!
//! Memory budget at max context is a real constraint (131K ctx on this GPU
//! doesn't fit even with sharing), but the cache itself is agnostic — the
//! caller picks `max_len`.

use half::bf16;

use crate::cuda::{CudaError, DeviceBuffer, Stream};
use crate::kernels::{inc_i32_device, kv_append_bf16};

/// Dimensions of one physical KV slot.
#[derive(Clone, Copy, Debug)]
pub struct SlotSpec {
    pub h_kv: usize,
    pub head_dim: usize,
}

impl SlotSpec {
    pub fn elements(&self, max_len: usize) -> usize {
        max_len * self.h_kv * self.head_dim
    }
}

/// KV cache for all layers of one attention stack.
pub struct KvCache {
    /// One buffer per physical slot, `[max_len, h_kv, head_dim]` bf16.
    pub k: Vec<DeviceBuffer<bf16>>,
    pub v: Vec<DeviceBuffer<bf16>>,
    /// `slot_for_layer[layer_idx]` = index into `k`/`v`.
    pub slot_for_layer: Vec<usize>,
    pub slot_specs: Vec<SlotSpec>,
    pub max_len: usize,
    pub cur_len: usize,
    /// Device-resident mirror of `cur_len`. Kernels that write the KV cache
    /// or read the cache length (split-KV attention device variant) read
    /// their offset from this pointer so CUDA-graph replays pick up the
    /// live value instead of the capture-time baked-in one.
    pub cur_len_dev: DeviceBuffer<i32>,
}

impl KvCache {
    /// Allocate a cache. `slot_for_layer.len()` is the number of layers; each
    /// entry points into `slot_specs`, which enumerates the physical slots.
    pub fn new(
        slot_specs: Vec<SlotSpec>,
        slot_for_layer: Vec<usize>,
        max_len: usize,
    ) -> Result<Self, CudaError> {
        for &s in &slot_for_layer {
            assert!(s < slot_specs.len(), "slot_for_layer: index out of range");
        }
        let mut k = Vec::with_capacity(slot_specs.len());
        let mut v = Vec::with_capacity(slot_specs.len());
        for spec in &slot_specs {
            k.push(DeviceBuffer::<bf16>::new(spec.elements(max_len))?);
            v.push(DeviceBuffer::<bf16>::new(spec.elements(max_len))?);
        }
        let mut cur_len_dev = DeviceBuffer::<i32>::new(1)?;
        cur_len_dev.copy_from_host(&[0i32])?;
        Ok(Self { k, v, slot_for_layer, slot_specs, max_len, cur_len: 0, cur_len_dev })
    }

    pub fn num_layers(&self) -> usize {
        self.slot_for_layer.len()
    }

    pub fn slot_for(&self, layer: usize) -> usize {
        self.slot_for_layer[layer]
    }

    pub fn slot_spec(&self, layer: usize) -> SlotSpec {
        self.slot_specs[self.slot_for(layer)]
    }

    pub fn k_buf(&self, layer: usize) -> &DeviceBuffer<bf16> {
        &self.k[self.slot_for(layer)]
    }

    pub fn v_buf(&self, layer: usize) -> &DeviceBuffer<bf16> {
        &self.v[self.slot_for(layer)]
    }

    /// Append `n_tokens` of new K and V to the cache for `layer` at
    /// `self.cur_len`. The new K/V must be `[n_tokens, h_kv, head_dim]` bf16
    /// in layout matching the slot's spec.
    ///
    /// Shared layers: this is a no-op if `layer` shares a slot with an already-
    /// written owner; we assume the owner wrote first. Ordering is the caller's
    /// responsibility (iterate layers in order).
    pub fn append(
        &mut self,
        layer: usize,
        new_k: &DeviceBuffer<bf16>,
        new_v: &DeviceBuffer<bf16>,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        let spec = self.slot_spec(layer);
        let row_elts = spec.h_kv * spec.head_dim;
        let n_elts = n_tokens * row_elts;
        // new_k/new_v may be oversized persistent scratch (e.g. sized for the
        // larger head_dim variant); we only read the first `n_elts` elements.
        assert!(new_k.len() >= n_elts, "append: new_k length {} < {}", new_k.len(), n_elts);
        assert!(new_v.len() >= n_elts, "append: new_v length {} < {}", new_v.len(), n_elts);
        assert!(
            self.cur_len + n_tokens <= self.max_len,
            "append: overflows max_len (cur {}, add {}, max {})",
            self.cur_len, n_tokens, self.max_len
        );
        let slot = self.slot_for(layer);
        let offset = self.cur_len * row_elts;
        self.k[slot].copy_region_from_device(offset, new_k, 0, n_elts)?;
        self.v[slot].copy_region_from_device(offset, new_v, 0, n_elts)?;
        Ok(())
    }

    /// Append `n_tokens` worth of K/V starting at the given element offsets
    /// into `new_k` / `new_v`. Avoids needing per-slot row scratch buffers
    /// when the caller already has a batched `[N, h_kv, head_dim]` buffer
    /// and wants to append row `i` of it to slot `i`'s KV cache.
    pub fn append_from_offset(
        &mut self,
        layer: usize,
        new_k: &DeviceBuffer<bf16>,
        k_src_offset: usize,
        new_v: &DeviceBuffer<bf16>,
        v_src_offset: usize,
        n_tokens: usize,
    ) -> Result<(), CudaError> {
        let spec = self.slot_spec(layer);
        let row_elts = spec.h_kv * spec.head_dim;
        let n_elts = n_tokens * row_elts;
        assert!(new_k.len() >= k_src_offset + n_elts, "append_from_offset: new_k too short");
        assert!(new_v.len() >= v_src_offset + n_elts, "append_from_offset: new_v too short");
        assert!(
            self.cur_len + n_tokens <= self.max_len,
            "append_from_offset: overflows max_len (cur {}, add {}, max {})",
            self.cur_len, n_tokens, self.max_len
        );
        let slot = self.slot_for(layer);
        let dst_offset = self.cur_len * row_elts;
        self.k[slot].copy_region_from_device(dst_offset, new_k, k_src_offset, n_elts)?;
        self.v[slot].copy_region_from_device(dst_offset, new_v, v_src_offset, n_elts)?;
        Ok(())
    }

    /// Mark that `n_tokens` positions have been consumed across all layers.
    /// Caller invokes this exactly once after processing a batch of tokens.
    ///
    /// This variant only updates the host counter. Use `advance_device` for
    /// graph-captured paths — that one issues a kernel that bumps the
    /// device-resident counter inside the graph, so replays advance
    /// automatically. If the same cache will later be used via the device-
    /// resident path (e.g. prefill-then-decode flow), call
    /// `sync_device_counter` to copy the host cur_len into `cur_len_dev`
    /// before the first device-path call.
    pub fn advance(&mut self, n_tokens: usize) {
        assert!(self.cur_len + n_tokens <= self.max_len);
        self.cur_len += n_tokens;
    }

    /// Copy the host `cur_len` into the device-resident `cur_len_dev`.
    /// Required once between any host-only `advance()` calls (e.g. serial
    /// prefill via `forward_step`) and the first device-resident append /
    /// attention call (batched decode via `forward_step_batched`).
    pub fn sync_device_counter(&mut self) -> Result<(), CudaError> {
        self.cur_len_dev.copy_from_host(&[self.cur_len as i32])
    }

    /// Kernel-based append: reads the destination offset from the device-
    /// resident `cur_len_dev` counter, so the captured graph picks up the
    /// live value at replay time. Used by the decode path that feeds
    /// `xk_attn_split_kv_bf16_device`.
    ///
    /// Does NOT advance the counter — call `advance_device` after all layers
    /// have appended (or after the whole decode step) to bump both host and
    /// device counters.
    pub fn append_device(
        &mut self,
        layer: usize,
        new_k: &DeviceBuffer<bf16>,
        k_src_offset: usize,
        new_v: &DeviceBuffer<bf16>,
        v_src_offset: usize,
        n_tokens: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        let spec = self.slot_spec(layer);
        let row_elts = spec.h_kv * spec.head_dim;
        let n_elts = n_tokens * row_elts;
        assert!(new_k.len() >= k_src_offset + n_elts, "append_device: new_k too short");
        assert!(new_v.len() >= v_src_offset + n_elts, "append_device: new_v too short");
        let slot = self.slot_for(layer);
        kv_append_bf16(
            &mut self.k[slot], new_k, k_src_offset,
            &self.cur_len_dev, row_elts, n_tokens, Some(stream),
        )?;
        kv_append_bf16(
            &mut self.v[slot], new_v, v_src_offset,
            &self.cur_len_dev, row_elts, n_tokens, Some(stream),
        )?;
        Ok(())
    }

    /// Device-resident counter bump. Issues a 1×1 kernel that increments
    /// `cur_len_dev` by `n_tokens` and also bumps the host mirror so
    /// non-captured callers (slot length queries, kv_len lookups) stay in
    /// sync.
    pub fn advance_device(&mut self, n_tokens: usize, stream: &Stream) -> Result<(), CudaError> {
        assert!(self.cur_len + n_tokens <= self.max_len);
        inc_i32_device(&mut self.cur_len_dev, n_tokens as i32, Some(stream))?;
        self.cur_len += n_tokens;
        Ok(())
    }

    /// Reset both host and device counters to zero (new request).
    pub fn reset(&mut self) -> Result<(), CudaError> {
        self.cur_len = 0;
        self.cur_len_dev.copy_from_host(&[0i32])
    }

    /// Number of valid positions currently held in the cache.
    pub fn len(&self) -> usize {
        self.cur_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_len == 0
    }
}
