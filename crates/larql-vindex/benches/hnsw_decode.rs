//! HNSW vs brute-force gate KNN — synthetic-data bench.
//!
//! Validates the 2026-04-25 wiring of HNSW into the decode path
//! (`gate_knn` routes through `gate_knn_hnsw` when `hnsw_enabled`).
//! Two regimes:
//!
//! 1. Dense Gemma-3-4B-shape (10 240 features × 2560 hidden) — brute
//!    BLAS gemv is competitive here; HNSW build cost amortises only
//!    over many queries.
//! 2. Wide MoE-shape (32 768 features × 2560 hidden, ≈ 16-expert
//!    bank) — brute matmul is memory-bound; HNSW search wins.
//!
//! What this measures:
//! - `gate_knn` brute (registry-routed path; baseline)
//! - `gate_knn` with HNSW enabled (graph search + abs re-rank)
//! - HNSW build cost (one-time per layer, reported separately)
//!
//! Recall numbers are validated by `tests/test_hnsw.rs::gate_knn_hnsw_smoke` —
//! this bench measures only timing. The synthetic data has no
//! semantic structure, so HNSW's relative speedup here is a
//! pessimistic ceiling on what real models see.
//!
//! Run: `cargo bench -p larql-vindex --bench hnsw_decode`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_vindex::VectorIndex;
use ndarray::{Array1, Array2};

fn random_query(hidden: usize) -> Array1<f32> {
    let mut state = 0xc0ffeeu64;
    Array1::from_shape_fn(hidden, |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut state = 42u64;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn build_index(features: usize, hidden: usize) -> VectorIndex {
    VectorIndex::new(
        vec![Some(synth_matrix(features, hidden))],
        vec![None],
        1,
        hidden,
    )
}

fn bench_gate_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_knn_brute_vs_hnsw");
    let configs: &[(&str, usize, usize)] = &[
        ("gemma3-4b-dense-10240x2560", 10_240, 2560),
        ("moe-16expert-32768x2560", 32_768, 2560),
    ];

    for &(label, features, hidden) in configs {
        let index = build_index(features, hidden);
        let query = random_query(hidden);

        // Brute baseline (HNSW disabled — registry-routed brute path).
        index.disable_hnsw();
        group.bench_with_input(
            BenchmarkId::new("brute", label),
            &index,
            |b, idx| b.iter(|| idx.gate_knn(0, &query, 10)),
        );

        // HNSW enabled. Build cost is one-shot — first query pays it.
        // Pre-warm so the bench measures steady-state search.
        index.enable_hnsw(200);
        let _warm = index.gate_knn(0, &query, 10);
        group.bench_with_input(
            BenchmarkId::new("hnsw", label),
            &index,
            |b, idx| b.iter(|| idx.gate_knn(0, &query, 10)),
        );

        // Reset for the next config.
        index.disable_hnsw();
    }
    group.finish();
}

/// One-time HNSW build cost — paid on the first query per layer
/// (lazy build via `get_or_build_hnsw`). Reported separately so
/// callers can decide whether HNSW is worth it for their query
/// volume.
fn bench_hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    group.sample_size(10); // construction is slow; fewer samples
    let configs: &[(&str, usize, usize)] = &[
        ("dense-10240x2560", 10_240, 2560),
        ("moe-32768x2560", 32_768, 2560),
    ];

    for &(label, features, hidden) in configs {
        group.bench_with_input(BenchmarkId::from_parameter(label), &(features, hidden), |b, &(f, h)| {
            b.iter(|| {
                let idx = build_index(f, h);
                idx.enable_hnsw(200);
                // Trigger lazy build.
                let q = random_query(h);
                let _ = idx.gate_knn(0, &q, 10);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gate_knn, bench_hnsw_build);
criterion_main!(benches);
