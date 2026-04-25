//! KV-cache engine implementations.
//!
//! Each engine in this module implements the [`crate::engines::KvEngine`] trait
//! and manages inference state differently:
//!
//! | Engine | Strategy | Memory @ 370K | Compression |
//! |---|---|---|---|
//! | [`markov_residual`] | Store residuals; recompute K/V on decode | ~193 MB | ~134× |
//! | [`unlimited_context`] | Window K/V checkpoints + token replay | ~30 MB | ~2,000× |
//! | [`turbo_quant`] | WHT + Lloyd-Max K/V compression (4-bit) | ~6.6 GB | ~4× |
//! | [`apollo`] | Single-vector boundary + retrieval injection | ~2.8 MB | ~20,000× |

pub mod apollo;
pub mod markov_residual;
pub mod turbo_quant;
pub mod unlimited_context;
