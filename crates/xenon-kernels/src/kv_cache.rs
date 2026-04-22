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

use crate::cuda::{CudaError, DeviceBuffer};

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
        Ok(Self { k, v, slot_for_layer, slot_specs, max_len, cur_len: 0 })
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
        assert_eq!(new_k.len(), n_tokens * row_elts, "append: new_k length");
        assert_eq!(new_v.len(), n_tokens * row_elts, "append: new_v length");
        assert!(
            self.cur_len + n_tokens <= self.max_len,
            "append: overflows max_len (cur {}, add {}, max {})",
            self.cur_len, n_tokens, self.max_len
        );
        let slot = self.slot_for(layer);
        let offset = self.cur_len * row_elts;
        self.k[slot].copy_slice_from_device(offset, new_k)?;
        self.v[slot].copy_slice_from_device(offset, new_v)?;
        Ok(())
    }

    /// Mark that `n_tokens` positions have been consumed across all layers.
    /// Caller invokes this exactly once after processing a batch of tokens.
    pub fn advance(&mut self, n_tokens: usize) {
        assert!(self.cur_len + n_tokens <= self.max_len);
        self.cur_len += n_tokens;
    }

    pub fn reset(&mut self) {
        self.cur_len = 0;
    }

    /// Number of valid positions currently held in the cache.
    pub fn len(&self) -> usize {
        self.cur_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_len == 0
    }
}
