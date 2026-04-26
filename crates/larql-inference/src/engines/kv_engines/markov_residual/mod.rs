//! MarkovResidualEngine — residual-stream KV-cache replacement.
//!
//! The pre-layer residual vector is the complete Markov state of the transformer.
//! K/V are recomputed from stored residuals at decode time (KL = 0.0 vs full-KV
//! baseline on Gemma 3 4B, validated 2026-04-23).

pub mod compute;
pub mod engine;
pub mod q4k;
pub mod store;

pub use engine::MarkovResidualEngine;
pub use store::RsStore;
pub(crate) use compute::rs_decode_step_profiled;
pub use compute::{RsPrefillResult, rs_prefill, rs_decode_step, recompute_kv, kv_memory_bytes_for_seq};
pub use q4k::ensure_attn_tensors_dequantised;
