//! xenon-core: model, layers, KV cache, weight loading.
//!
//! Phase 0: config parsing + weight-file introspection.
//! Phases 1-3 add layer forward, attention, KV cache, decode.

pub mod config;
pub mod error;
pub mod model;
pub mod weights;

pub use config::{GemmaConfig, LayerKind, QuantConfig};
pub use error::{Error, Result};
