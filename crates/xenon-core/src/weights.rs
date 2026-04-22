//! Safetensors header introspection.
//!
//! Phase 0: parse the JSON header and classify tensors into NVFP4 weight+scale
//! pairs vs plain (unquantized) tensors. No tensor bytes are read here — the
//! actual mmap and device upload land in phase 1.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;

use crate::{Error, Result};

/// Per-tensor entry from the safetensors JSON header.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

impl TensorInfo {
    pub fn bytes(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

/// Contents of a safetensors file header (everything before the tensor data).
pub struct SafetensorsHeader {
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: Option<serde_json::Value>,
    pub header_bytes: u64,
    pub file_bytes: u64,
}

impl SafetensorsHeader {
    pub fn from_path(path: &Path) -> Result<Self> {
        let mut f = File::open(path)?;
        let file_bytes = f.metadata()?.len();

        let mut len_bytes = [0u8; 8];
        f.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > file_bytes {
            return Err(Error::Config(format!(
                "safetensors header length {header_len} exceeds file size {file_bytes}"
            )));
        }
        let mut header_buf = vec![0u8; header_len as usize];
        f.read_exact(&mut header_buf)?;

        let mut all: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&header_buf)?;
        let metadata = all.remove("__metadata__");

        let mut tensors = BTreeMap::new();
        for (name, val) in all {
            let info: TensorInfo = serde_json::from_value(val).map_err(|e| {
                Error::Config(format!("bad tensor entry for '{name}': {e}"))
            })?;
            tensors.insert(name, info);
        }

        Ok(Self {
            tensors,
            metadata,
            header_bytes: header_len,
            file_bytes,
        })
    }
}

/// A quantized-linear weight group: packed FP4 weight + per-block scales,
/// optionally plus a second-level (global) scale.
#[derive(Debug, Clone)]
pub struct QuantPair {
    pub module: String,
    pub weight_shape: Vec<usize>,
    pub weight_dtype: String,
    pub weight_bytes: u64,
    pub scale_dtype: String,
    pub scale_shape: Vec<usize>,
    pub scale_bytes: u64,
    pub extra_scale: Option<String>,
}

/// Breakdown of the safetensors file by tensor kind + dtype.
#[derive(Debug)]
pub struct WeightBreakdown {
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub tensor_count: usize,
    pub bytes_by_dtype: BTreeMap<String, u64>,
    pub count_by_dtype: BTreeMap<String, usize>,
    pub quant_pairs: Vec<QuantPair>,
    /// `.weight` tensors with no matching `.weight_scale` (e.g. norms, embeddings,
    /// anything excluded from quant by the producer).
    pub plain_weights: Vec<String>,
    /// Scale tensors with no matching `.weight` — this should stay empty.
    pub orphan_scales: Vec<String>,
    /// Everything else (biases, input_scales, statistics, ...).
    pub other_tensors: Vec<String>,
}

impl WeightBreakdown {
    pub fn from_header(h: &SafetensorsHeader) -> Self {
        let mut bytes_by_dtype: BTreeMap<String, u64> = BTreeMap::new();
        let mut count_by_dtype: BTreeMap<String, usize> = BTreeMap::new();
        for t in h.tensors.values() {
            *bytes_by_dtype.entry(t.dtype.clone()).or_default() += t.bytes();
            *count_by_dtype.entry(t.dtype.clone()).or_default() += 1;
        }

        let mut quant_pairs = Vec::new();
        let mut plain_weights = Vec::new();
        let mut orphan_scales = Vec::new();
        let mut other_tensors = Vec::new();
        let mut consumed: HashSet<String> = HashSet::new();

        for (name, t) in &h.tensors {
            if let Some(base) = name.strip_suffix(".weight") {
                let scale_name = format!("{base}.weight_scale");
                if let Some(scale) = h.tensors.get(&scale_name) {
                    let extra = format!("{base}.weight_scale_2");
                    let extra_present = h.tensors.contains_key(&extra);
                    quant_pairs.push(QuantPair {
                        module: base.to_string(),
                        weight_shape: t.shape.clone(),
                        weight_dtype: t.dtype.clone(),
                        weight_bytes: t.bytes(),
                        scale_dtype: scale.dtype.clone(),
                        scale_shape: scale.shape.clone(),
                        scale_bytes: scale.bytes(),
                        extra_scale: extra_present.then_some(extra.clone()),
                    });
                    consumed.insert(name.clone());
                    consumed.insert(scale_name);
                    if extra_present {
                        consumed.insert(extra);
                    }
                }
            }
        }

        for name in h.tensors.keys() {
            if consumed.contains(name) {
                continue;
            }
            if name.ends_with(".weight") {
                plain_weights.push(name.clone());
            } else if name.ends_with(".weight_scale") || name.ends_with(".weight_scale_2") {
                orphan_scales.push(name.clone());
            } else {
                other_tensors.push(name.clone());
            }
        }

        Self {
            file_bytes: h.file_bytes,
            header_bytes: h.header_bytes,
            tensor_count: h.tensors.len(),
            bytes_by_dtype,
            count_by_dtype,
            quant_pairs,
            plain_weights,
            orphan_scales,
            other_tensors,
        }
    }

    /// Sum of packed FP4 weight bytes across all quant pairs.
    pub fn quant_weight_bytes(&self) -> u64 {
        self.quant_pairs.iter().map(|p| p.weight_bytes).sum()
    }

    /// Sum of block-scale bytes across all quant pairs.
    pub fn quant_scale_bytes(&self) -> u64 {
        self.quant_pairs.iter().map(|p| p.scale_bytes).sum()
    }

    /// Bytes accounted for by plain (unquantized) `.weight` tensors.
    pub fn plain_weight_bytes(&self, h: &SafetensorsHeader) -> u64 {
        self.plain_weights
            .iter()
            .filter_map(|n| h.tensors.get(n).map(|t| t.bytes()))
            .sum()
    }
}

/// Whether a tensor name matches a modelopt exclude-module glob
/// (supports trailing `*`, which is all modelopt emits).
pub fn matches_glob(name: &str, pattern: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix)
    } else {
        name == pattern || name.starts_with(&format!("{pattern}."))
    }
}

pub fn is_excluded(tensor_name: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| matches_glob(tensor_name, p))
}

/// Memory-mapped safetensors file. Holds both the mmap and the parsed header;
/// tensor byte slices point into the mmap'd region.
pub struct MmapWeights {
    pub path: PathBuf,
    pub mmap: Mmap,
    pub header: SafetensorsHeader,
    /// Byte offset in the file where tensor data starts (= 8 + header_bytes).
    pub data_offset: u64,
}

impl MmapWeights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse header out of the mmap'd bytes directly. SafetensorsHeader
        // reads from disk; replicate the parsing here to keep a single owner
        // of the tensor table without re-reading the file.
        if mmap.len() < 8 {
            return Err(Error::Config("safetensors file too short".into()));
        }
        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap());
        let data_offset = 8 + header_len;
        if (data_offset as usize) > mmap.len() {
            return Err(Error::Config(format!(
                "header length {header_len} exceeds file size {}",
                mmap.len()
            )));
        }
        let header_buf = &mmap[8..(data_offset as usize)];
        let mut all: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(header_buf)?;
        let metadata = all.remove("__metadata__");
        let mut tensors = BTreeMap::new();
        for (name, val) in all {
            let info: TensorInfo = serde_json::from_value(val)
                .map_err(|e| Error::Config(format!("bad tensor entry for '{name}': {e}")))?;
            tensors.insert(name, info);
        }
        let header = SafetensorsHeader {
            tensors,
            metadata,
            header_bytes: header_len,
            file_bytes: mmap.len() as u64,
        };

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            header,
            data_offset,
        })
    }

    /// Byte slice for a named tensor's data (points into the mmap).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| Error::Config(format!("tensor '{name}' not found")))?;
        let begin = (self.data_offset + info.data_offsets[0]) as usize;
        let end = (self.data_offset + info.data_offsets[1]) as usize;
        if end > self.mmap.len() {
            return Err(Error::Config(format!(
                "tensor '{name}' offset {end} exceeds mmap len {}",
                self.mmap.len()
            )));
        }
        Ok(&self.mmap[begin..end])
    }

    /// Look up a tensor's metadata.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.get(name)
    }

    /// Does this layer compute its own K/V at inference time? We detect this
    /// by the presence of the `k_proj.input_scale` activation-scale tensor:
    /// layers that share KV under `num_kv_shared_layers` don't actually run
    /// their (still-present) k_proj/v_proj weights, so modelopt never writes
    /// an input_scale for them.
    pub fn layer_owns_kv(&self, layer: usize) -> bool {
        let name = format!(
            "model.language_model.layers.{layer}.self_attn.k_proj.input_scale"
        );
        self.header.tensors.contains_key(&name)
    }

    /// Raw bytes of a plain bf16 tensor, with dtype + length validation.
    pub fn load_bf16(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .info(name)
            .ok_or_else(|| Error::Config(format!("missing tensor '{name}'")))?;
        if info.dtype != "BF16" {
            return Err(Error::Config(format!(
                "'{name}' dtype {} (expected BF16)",
                info.dtype
            )));
        }
        let expected = info.shape.iter().copied().product::<usize>() * 2;
        let bytes = self.tensor_bytes(name)?;
        if bytes.len() != expected {
            return Err(Error::Config(format!(
                "'{name}' has {} bytes but shape {:?} implies {}",
                bytes.len(),
                info.shape,
                expected
            )));
        }
        Ok(bytes)
    }

    /// Load the three tensors making up an NVFP4 quantized linear:
    /// `{prefix}.weight` (U8 packed), `{prefix}.weight_scale` (F8_E4M3 block scales),
    /// `{prefix}.weight_scale_2` (F32 per-tensor global scale).
    ///
    /// Validates shape invariants: `weight` is `[out, in/2]`, `weight_scale` is
    /// `[out, in/16]`, `weight_scale_2` is a scalar. `in_features` must be a
    /// multiple of 16. Returns references tied to the mmap.
    pub fn load_quant_linear(&self, prefix: &str) -> Result<QuantLinearRef<'_>> {
        let w_name  = format!("{prefix}.weight");
        let s_name  = format!("{prefix}.weight_scale");
        let s2_name = format!("{prefix}.weight_scale_2");

        let w_info  = self.info(&w_name).ok_or_else(|| Error::Config(format!("missing '{w_name}'")))?;
        let s_info  = self.info(&s_name).ok_or_else(|| Error::Config(format!("missing '{s_name}'")))?;
        let s2_info = self.info(&s2_name).ok_or_else(|| Error::Config(format!("missing '{s2_name}'")))?;

        if w_info.dtype != "U8" {
            return Err(Error::Config(format!("'{w_name}' dtype {} (want U8)", w_info.dtype)));
        }
        if s_info.dtype != "F8_E4M3" {
            return Err(Error::Config(format!("'{s_name}' dtype {} (want F8_E4M3)", s_info.dtype)));
        }
        if s2_info.dtype != "F32" {
            return Err(Error::Config(format!("'{s2_name}' dtype {} (want F32)", s2_info.dtype)));
        }
        if w_info.shape.len() != 2 {
            return Err(Error::Config(format!("'{w_name}' rank {} (want 2)", w_info.shape.len())));
        }

        let out = w_info.shape[0];
        let in_packed = w_info.shape[1];
        let in_features = in_packed * 2;
        if in_features % 16 != 0 {
            return Err(Error::Config(format!(
                "'{w_name}' in_features {in_features} not a multiple of 16"
            )));
        }

        let expected_scale_shape = vec![out, in_features / 16];
        if s_info.shape != expected_scale_shape {
            return Err(Error::Config(format!(
                "'{s_name}' shape {:?} != expected {:?}",
                s_info.shape, expected_scale_shape
            )));
        }
        if !s2_info.shape.is_empty() && s2_info.shape != vec![1] {
            return Err(Error::Config(format!(
                "'{s2_name}' shape {:?} is not scalar",
                s2_info.shape
            )));
        }

        let packed = self.tensor_bytes(&w_name)?;
        let scales = self.tensor_bytes(&s_name)?;
        let s2_bytes = self.tensor_bytes(&s2_name)?;
        if packed.len() != out * in_packed {
            return Err(Error::Config(format!(
                "'{w_name}' byte length {} != expected {}",
                packed.len(),
                out * in_packed
            )));
        }
        if scales.len() != out * in_features / 16 {
            return Err(Error::Config(format!(
                "'{s_name}' byte length {} != expected {}",
                scales.len(),
                out * in_features / 16
            )));
        }
        if s2_bytes.len() != 4 {
            return Err(Error::Config(format!(
                "'{s2_name}' has {} bytes (want 4 for f32)",
                s2_bytes.len()
            )));
        }
        let global_scale = f32::from_le_bytes([s2_bytes[0], s2_bytes[1], s2_bytes[2], s2_bytes[3]]);

        Ok(QuantLinearRef {
            module: prefix.to_string(),
            out_features: out,
            in_features,
            packed,
            scales,
            global_scale,
        })
    }
}

/// A loaded NVFP4 quantized linear layer. Byte slices point into the mmap.
#[derive(Debug)]
pub struct QuantLinearRef<'a> {
    pub module: String,
    pub out_features: usize,
    pub in_features: usize,
    pub packed: &'a [u8],    // [out, in/2]
    pub scales: &'a [u8],    // [out, in/16] UE4M3
    pub global_scale: f32,   // per-tensor fp32 multiplier
}
