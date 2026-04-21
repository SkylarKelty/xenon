//! Model handle. Skeleton — actual weights, KV cache, and forward pass land
//! in phases 1-3.

use crate::config::GemmaConfig;

pub struct Model {
    pub config: GemmaConfig,
}

impl Model {
    pub fn new(config: GemmaConfig) -> Self {
        Self { config }
    }
}
