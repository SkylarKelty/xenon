//! Thin wrapper around HF `tokenizers` crate. Loads `tokenizer.json` and
//! provides encode/decode for text ↔ token IDs.
//!
//! We don't re-implement BPE/SentencePiece here — the tokenizer is upstream
//! of any GPU work, and HF's crate handles the Gemma pretokenizer + special
//! tokens (including the chat template's turn boundaries) correctly.

use std::path::Path;

use crate::{Error, Result};

#[derive(Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load `tokenizer.json` from a model directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path).map_err(|e| {
            Error::Config(format!("failed to load {}: {e}", path.display()))
        })?;
        Ok(Self { inner })
    }

    /// Encode `text` into a flat list of token IDs. `add_special_tokens`
    /// controls whether the tokenizer prepends/appends BOS/EOS etc.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Config(format!("encode failed: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token IDs back to text. `skip_special_tokens=true` drops BOS/EOS
    /// and similar control tokens from the output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Config(format!("decode failed: {e}")))
    }

    /// Reported vocabulary size. Use to sanity-check against `config.vocab_size`.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
