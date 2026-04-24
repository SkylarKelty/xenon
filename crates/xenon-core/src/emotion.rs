//! Emotion-probe artifact loading.
//!
//! An emotion-probe artifact is a single `.safetensors` file holding:
//!
//! - `vectors`:     `[L, N, H]` bf16 — unit-normed, PCA-confound-stripped
//!                  emotion directions per probed layer.
//! - `global_mean`: `[L, H]`    bf16 — mean residual-stream activation
//!                  subtracted before the dot product.
//! - `z_mean`:      `[N]`       fp32 *(optional)* — per-emotion baseline mean
//!                  for cross-emotion comparability.
//! - `z_std`:       `[N]`       fp32 *(optional)* — per-emotion baseline stddev.
//!
//! Plus the safetensors `__metadata__` map (all values string-encoded per the
//! spec) with:
//!
//! - `version`:    `"1"`
//! - `model_id`:   e.g. `"cosmicproc/gemma-4-E4B-it-NVFP4"`
//! - `hidden`:     `"2560"`
//! - `layers`:     JSON array of layer indices, e.g. `"[28]"`
//! - `emotions`:   JSON array of N emotion names, e.g. `"[\"afraid\", ...]"`
//!
//! The inference-time math is `score_e(t) = (h_{L,t} - global_mean_L) . v_e`.

use std::path::Path;

use crate::{Error, Result, weights::MmapWeights};

pub const EMOTION_ARTIFACT_VERSION: &str = "1";

pub struct EmotionArtifact {
    pub model_id: String,
    pub hidden: usize,
    pub layers: Vec<u32>,
    pub emotions: Vec<String>,
    /// Borrowed mmap backing, in row-major `[L, N, H]` bf16 layout.
    pub vectors_bytes: &'static [u8],
    /// `[L, H]` bf16.
    pub global_mean_bytes: &'static [u8],
    /// `[N]` fp32, if present.
    pub z_mean_bytes: Option<&'static [u8]>,
    /// `[N]` fp32, if present.
    pub z_std_bytes: Option<&'static [u8]>,
    // Keeps the mmap alive for the lifetime of the artifact.
    _mm: Box<MmapWeights>,
}

impl EmotionArtifact {
    pub fn open(path: &Path) -> Result<Self> {
        let mm = Box::new(MmapWeights::open(path)?);

        let meta = mm
            .header
            .metadata
            .as_ref()
            .ok_or_else(|| Error::Config("emotion artifact: missing __metadata__".into()))?;
        let meta_map = meta
            .as_object()
            .ok_or_else(|| Error::Config("emotion artifact: __metadata__ not object".into()))?;

        let meta_str = |k: &str| -> Result<&str> {
            meta_map
                .get(k)
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Config(format!("emotion artifact: missing metadata '{k}'")))
        };

        let version = meta_str("version")?;
        if version != EMOTION_ARTIFACT_VERSION {
            return Err(Error::Config(format!(
                "emotion artifact: version '{version}' != expected '{}'",
                EMOTION_ARTIFACT_VERSION
            )));
        }
        let model_id = meta_str("model_id")?.to_string();
        let hidden: usize = meta_str("hidden")?
            .parse()
            .map_err(|e| Error::Config(format!("emotion artifact: bad 'hidden' ({e})")))?;
        let layers: Vec<u32> = serde_json::from_str(meta_str("layers")?)
            .map_err(|e| Error::Config(format!("emotion artifact: bad 'layers' JSON ({e})")))?;
        let emotions: Vec<String> = serde_json::from_str(meta_str("emotions")?)
            .map_err(|e| Error::Config(format!("emotion artifact: bad 'emotions' JSON ({e})")))?;

        let num_l = layers.len();
        let num_n = emotions.len();

        // Pull tensor bytes. We reinterpret the mmap's lifetime as 'static via
        // the `Box<MmapWeights>` we hold — as long as `self` lives, the mmap
        // does. Caller sees those slices through `&self`, which is bound by
        // the real lifetime.
        let vectors_bytes = mm.tensor_bytes("vectors")?;
        let gmean_bytes = mm.tensor_bytes("global_mean")?;

        let v_info = mm
            .info("vectors")
            .ok_or_else(|| Error::Config("emotion artifact: no 'vectors' entry".into()))?;
        if v_info.dtype != "BF16" && v_info.dtype != "bfloat16" {
            return Err(Error::Config(format!(
                "emotion artifact: 'vectors' dtype is {}, want BF16",
                v_info.dtype
            )));
        }
        if v_info.shape != vec![num_l, num_n, hidden] {
            return Err(Error::Config(format!(
                "emotion artifact: 'vectors' shape {:?} != [{num_l}, {num_n}, {hidden}]",
                v_info.shape
            )));
        }

        let g_info = mm
            .info("global_mean")
            .ok_or_else(|| Error::Config("emotion artifact: no 'global_mean' entry".into()))?;
        if g_info.shape != vec![num_l, hidden] {
            return Err(Error::Config(format!(
                "emotion artifact: 'global_mean' shape {:?} != [{num_l}, {hidden}]",
                g_info.shape
            )));
        }

        let z_mean_bytes = mm.tensor_bytes("z_mean").ok();
        let z_std_bytes = mm.tensor_bytes("z_std").ok();

        // SAFETY: the slices borrow from `mm`, which we own in the returned
        // struct. Transmuting the lifetime is fine as long as `mm` is not
        // moved out or replaced — we never do that.
        let vectors_bytes: &'static [u8] = unsafe { std::mem::transmute(vectors_bytes) };
        let global_mean_bytes: &'static [u8] = unsafe { std::mem::transmute(gmean_bytes) };
        let z_mean_bytes: Option<&'static [u8]> =
            z_mean_bytes.map(|b| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(b) });
        let z_std_bytes: Option<&'static [u8]> =
            z_std_bytes.map(|b| unsafe { std::mem::transmute::<&[u8], &'static [u8]>(b) });

        Ok(Self {
            model_id,
            hidden,
            layers,
            emotions,
            vectors_bytes,
            global_mean_bytes,
            z_mean_bytes,
            z_std_bytes,
            _mm: mm,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn num_emotions(&self) -> usize {
        self.emotions.len()
    }

    /// Resolve a layer index within the probed set. Returns `None` if
    /// `model_layer` has no probes.
    pub fn slot_for_model_layer(&self, model_layer: u32) -> Option<usize> {
        self.layers.iter().position(|&l| l == model_layer)
    }
}
