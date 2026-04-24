//! xenon-core: model, layers, KV cache, weight loading.
//!
//! Phase 0: config parsing + weight-file introspection.
//! Phases 1-3 add layer forward, attention, KV cache, decode.

pub mod config;
pub mod emotion;
pub mod error;
pub mod model;
pub mod tokenizer;
pub mod weights;

pub use config::{GemmaConfig, LayerKind, QuantConfig};
pub use emotion::{EmotionArtifact, EMOTION_ARTIFACT_VERSION};
pub use error::{Error, Result};
pub use tokenizer::Tokenizer;
pub use weights::{
    MmapWeights, QuantLinearRef, QuantPair, SafetensorsHeader, TensorInfo, WeightBreakdown,
};
