//! Safetensors header introspection.
//!
//! Phase 0: parse the JSON header and classify tensors into NVFP4 weight+scale
//! pairs vs plain (unquantized) tensors. No tensor bytes are read here — the
//! actual mmap and device upload land in phase 1.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

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
