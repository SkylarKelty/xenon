//! Parsing of HuggingFace-style `config.json` and `hf_quant_config.json` for
//! Gemma 4. We accept a superset of fields: unknown keys are ignored so the
//! loader is forward-compatible with minor config churn upstream.

use serde::Deserialize;
use std::path::Path;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GemmaTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    pub vocab_size: usize,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub layer_types: Vec<LayerKind>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub attention_bias: Option<bool>,
}

/// Top-level model config. Multimodal fields are accepted but unused by the
/// text-only v0; they'll be wired up when we add vision/audio in a later phase.
#[derive(Debug, Clone, Deserialize)]
pub struct GemmaConfig {
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub text_config: GemmaTextConfig,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl GemmaConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let cfg: GemmaConfig = serde_json::from_slice(&bytes)?;
        if cfg.model_type != "gemma4" {
            return Err(Error::Unsupported(format!(
                "model_type '{}' (expected 'gemma4')",
                cfg.model_type
            )));
        }
        if cfg.text_config.layer_types.len() != cfg.text_config.num_hidden_layers {
            return Err(Error::Config(format!(
                "layer_types has {} entries but num_hidden_layers is {}",
                cfg.text_config.layer_types.len(),
                cfg.text_config.num_hidden_layers
            )));
        }
        Ok(cfg)
    }

    /// Effective head dim for a given layer (sliding layers use `head_dim`,
    /// full-attention layers use `global_head_dim` when present).
    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        let tc = &self.text_config;
        match tc.layer_types[layer_idx] {
            LayerKind::FullAttention => tc.global_head_dim.unwrap_or(tc.head_dim),
            LayerKind::SlidingAttention => tc.head_dim,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    pub producer: QuantProducer,
    pub quantization: QuantSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantProducer {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantSpec {
    pub quant_algo: String,
    #[serde(default)]
    pub kv_cache_quant_algo: Option<String>,
    pub group_size: usize,
    #[serde(default)]
    pub exclude_modules: Vec<String>,
}

impl QuantConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}
