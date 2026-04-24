//! Per-layer CPU↔Metal prefill parity regression guard.
//!
//! The architecture golden tests (`test_arch_golden`) only check the first
//! few generated tokens. That's cheap but loose — a subtle kernel drift
//! can compound for 50 layers and still happen to argmax on the expected
//! token. This suite runs both backends' **prefill** passes through the
//! per-layer residual dump hooks (`LARQL_METAL_DUMP_LAYERS` +
//! `LARQL_CPU_DUMP_LAYERS`) and asserts that every layer's end-of-layer
//! hidden state is bit-compatible (cos ≥ 0.99995) between the two paths.
//!
//! Why prefill only: decode adds a KV-cache layer on Metal (a different
//! code path — `metal/decode/mod.rs`), so "match at every layer" only
//! holds semantically for prefill. Kernel-level parity on that path is a
//! good forcing function — every per-layer delta Metal introduces must
//! be justified against the CPU reference.
//!
//! **Caught regressions.** The Metal `fused_attention` shader's
//! `tid < head_dim` load gate (left `tg_q[256..512]` uninitialised on
//! head_dim=512 layers) produced ~6% drift at every Gemma 4 global layer
//! and compounded to cos ≈ 0.91 by L59. Pure-unit-test exists for that
//! kernel (`test_metal_shaders::fused_attention_head_dim_512`); this
//! suite is the end-to-end cousin that would have caught the bug through
//! a real vindex forward pass even if the unit test hadn't been written.
//!
//! **Skip semantics**: any case whose vindex isn't present in the cache
//! prints a skip and returns Ok — CI stays green. Set `LARQL_ARCH_STRICT=1`
//! to turn missing vindexes into hard failures.

use std::path::{Path, PathBuf};

use larql_inference::encode_prompt;
use larql_inference::layer_graph::generate::generate;
use larql_inference::layer_graph::CachedLayerGraph;
use larql_inference::wrap_chat_prompt;
use larql_vindex::{
    load_model_weights_q4k, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

/// Per-layer cos_sim threshold. Below this, the residual has drifted
/// meaningfully. Anything above is float noise (BF16→f32 dequant,
/// accumulation order, BLAS vs manual scalar summation).
const COS_THRESHOLD: f32 = 0.99995;

/// Relative max-abs threshold: flag when any single element differs by
/// more than this fraction of the Metal vector's L2 norm. Absolute-value
/// thresholds don't travel across architectures (Gemma 3's norms sit at
/// ~400, Gemma 4 31B's at ~1500, Gemma 4 E2B at ~2000), so we normalise
/// — 1% relative is tight enough that the fused_attention head_dim=512
/// regression (which produced ~7% relative drift at L59 on Gemma 4 31B)
/// trips this check immediately, while BF16-dequant + BLAS-ordering
/// noise (empirically up to 0.3 abs on hidden=2560 → <0.08% relative)
/// stays well below.
const MAX_ABS_REL_THRESHOLD: f32 = 0.01;

struct ParityCase {
    name: &'static str,
    vindex_name: &'static str,
}

/// Every vindex we've extracted locally. Add a row per new architecture.
const CASES: &[ParityCase] = &[
    ParityCase { name: "gemma3-4b-it",             vindex_name: "gemma3-4b-q4k-v2" },
    ParityCase { name: "gemma4-31b-it (dense)",    vindex_name: "gemma4-31b-q4k" },
    ParityCase { name: "llama2-7b-hf (base)",      vindex_name: "llama2-7b-q4k" },
    ParityCase { name: "mistral-7b-v0.1 (base)",   vindex_name: "mistral-7b-v0.1-q4k" },
    // gemma-4-26B-A4B-it (MoE) intentionally omitted: Metal's MoE prefill
    // is a token-by-token shim (`metal/trait_impl.rs:215-229`) that goes
    // through `decode_token`, not `dispatch_full_pipeline`, so the
    // per-layer dump hooks don't fire. Re-include when MoE prefill
    // batches for real.
];

fn find_vindex(name: &str) -> Option<PathBuf> {
    let filename = format!("{name}.vindex");
    if let Ok(env_path) = std::env::var(format!(
        "LARQL_VINDEX_{}",
        name.to_uppercase().replace('-', "_")
    )) {
        let p = PathBuf::from(env_path);
        if p.is_dir() {
            return Some(p);
        }
    }
    let chris_models = PathBuf::from("/Users/christopherhay/chris-models").join(&filename);
    if chris_models.is_dir() {
        return Some(chris_models);
    }
    let home = std::env::var("HOME").ok()?;
    [
        PathBuf::from(&home).join(".cache/larql/local").join(&filename),
        PathBuf::from("output").join(&filename),
    ]
    .into_iter()
    .find(|p| p.is_dir())
}

fn strict_mode() -> bool {
    matches!(
        std::env::var("LARQL_ARCH_STRICT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

/// Read a raw `f32[]` little-endian file. Returns `None` on any I/O
/// error or non-multiple-of-4 file size.
fn read_f32(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Layer-level parity stats: cos similarity, max absolute diff, and the
/// Metal vector's L2 norm so callers can compute a relative max_abs.
struct LayerStats {
    cos: f32,
    max_abs: f32,
    metal_norm: f32,
}

fn layer_stats(cpu: &[f32], metal: &[f32]) -> LayerStats {
    assert_eq!(cpu.len(), metal.len(), "shape mismatch");
    let mut dot = 0.0f64;
    let mut cn = 0.0f64;
    let mut mn = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..cpu.len() {
        let a = cpu[i] as f64;
        let b = metal[i] as f64;
        dot += a * b;
        cn += a * a;
        mn += b * b;
        let d = (cpu[i] - metal[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cos = if cn > 0.0 && mn > 0.0 {
        (dot / (cn.sqrt() * mn.sqrt())) as f32
    } else {
        0.0
    };
    LayerStats { cos, max_abs, metal_norm: mn.sqrt() as f32 }
}

/// Drive a single vindex through CPU and Metal prefills with dump
/// hooks enabled. Returns the number of layers successfully compared
/// so the caller can assert we actually exercised the model.
fn run_parity_case(case: &ParityCase) -> Result<usize, String> {
    let Some(vindex_path) = find_vindex(case.vindex_name) else {
        if strict_mode() {
            return Err(format!(
                "[{}] vindex `{}` not found (LARQL_ARCH_STRICT=1)",
                case.name, case.vindex_name
            ));
        }
        eprintln!(
            "[{}] skip: vindex `{}` not found in ~/.cache/larql/local/ or output/",
            case.name, case.vindex_name
        );
        return Ok(0);
    };

    // Disjoint dump dirs per backend — tempfile cleans up when the
    // `TempDir` guard drops at end of scope.
    let cpu_dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let metal_dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    std::env::set_var("LARQL_CPU_DUMP_LAYERS", cpu_dir.path());
    std::env::set_var("LARQL_METAL_DUMP_LAYERS", metal_dir.path());

    let mut cb = SilentLoadCallbacks;
    let cfg = load_vindex_config(&vindex_path)
        .map_err(|e| format!("load_vindex_config: {e}"))?;
    if cfg.quant != QuantFormat::Q4k {
        return Err(format!("expected Q4K vindex (got {:?})", cfg.quant));
    }

    let tokenizer = load_vindex_tokenizer(&vindex_path)
        .map_err(|e| format!("load_vindex_tokenizer: {e}"))?;
    let mut q4_index =
        VectorIndex::load_vindex(&vindex_path, &mut cb).map_err(|e| format!("load vindex: {e}"))?;
    q4_index
        .load_attn_q4k(&vindex_path)
        .map_err(|e| format!("load_attn_q4k: {e}"))?;
    q4_index
        .load_interleaved_q4k(&vindex_path)
        .map_err(|e| format!("load_interleaved_q4k: {e}"))?;
    let _ = q4_index.load_lm_head_q4(&vindex_path);

    // Separate weight copies — CPU's per-layer dequant inserts into
    // `weights.tensors`, which would otherwise race across backends
    // sharing the same handle.
    let mut w_metal = load_model_weights_q4k(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (metal): {e}"))?;
    let mut w_cpu = load_model_weights_q4k(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (cpu): {e}"))?;

    let prompt = "The capital of France is";
    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), prompt);
    let token_ids = encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;
    let num_layers = w_metal.num_layers;

    // max_tokens=1 → single prefill pass per backend, no decode. Keeps
    // the test fast (we only need the layer dumps) and avoids the KV-
    // cache decode path whose per-layer dumps aren't wired.
    let cached = CachedLayerGraph::from_residuals(Vec::new());
    let metal_backend = larql_compute::metal::MetalBackend::new()
        .ok_or("Metal backend unavailable — rebuild with --features metal")?;
    let _ = generate(
        &mut w_metal, &tokenizer, &token_ids, 1,
        &q4_index, &metal_backend, &cached, 0..num_layers,
    );
    let cpu_backend = larql_compute::CpuBackend;
    let _ = generate(
        &mut w_cpu, &tokenizer, &token_ids, 1,
        &q4_index, &cpu_backend, &cached, 0..num_layers,
    );

    // Compare every layer's end-of-layer hidden state. Missing files
    // count as a test failure — if the backend ran but no dump appeared
    // the test would otherwise pass vacuously.
    let mut compared = 0usize;
    for l in 0..num_layers {
        let cpu_path = cpu_dir.path().join(format!("cpu_layer_{l:02}.f32"));
        let metal_path = metal_dir.path().join(format!("metal_layer_{l:02}_h_out.f32"));
        let Some(cpu_v) = read_f32(&cpu_path) else {
            return Err(format!("[{}] L{l}: cpu dump missing at {}", case.name, cpu_path.display()));
        };
        let Some(metal_v) = read_f32(&metal_path) else {
            return Err(format!("[{}] L{l}: metal dump missing at {}", case.name, metal_path.display()));
        };
        if cpu_v.len() != metal_v.len() {
            return Err(format!(
                "[{}] L{l}: length mismatch cpu={} mtl={}",
                case.name, cpu_v.len(), metal_v.len()
            ));
        }
        let s = layer_stats(&cpu_v, &metal_v);
        let rel = if s.metal_norm > 0.0 {
            s.max_abs / s.metal_norm
        } else {
            0.0
        };
        if s.cos < COS_THRESHOLD || rel > MAX_ABS_REL_THRESHOLD {
            return Err(format!(
                "[{}] L{l}: parity broken — cos_sim={:.6} max_abs_Δ={:.3e} \
                 (= {:.3}% of mtl_norm={:.2}; thresholds: cos≥{COS_THRESHOLD}, rel≤{:.1}%)",
                case.name,
                s.cos, s.max_abs, 100.0 * rel, s.metal_norm,
                100.0 * MAX_ABS_REL_THRESHOLD
            ));
        }
        compared += 1;
    }
    eprintln!(
        "[{}] parity OK across {compared} layers (rel max_abs_Δ ≤ {:.1}%)",
        case.name,
        100.0 * MAX_ABS_REL_THRESHOLD
    );
    Ok(compared)
}

// One #[test] per architecture, mirroring `test_arch_golden`. Individual
// tests so a single regression surfaces with a specific name (not a
// buried "assertion failed at index N").

#[test]
fn parity_gemma3_4b_prefill() {
    if let Err(e) = run_parity_case(&CASES[0]) {
        panic!("{e}");
    }
}

#[test]
fn parity_gemma4_31b_dense_prefill() {
    if let Err(e) = run_parity_case(&CASES[1]) {
        panic!("{e}");
    }
}

#[test]
fn parity_llama2_7b_prefill() {
    if let Err(e) = run_parity_case(&CASES[2]) {
        panic!("{e}");
    }
}

#[test]
fn parity_mistral_7b_prefill() {
    if let Err(e) = run_parity_case(&CASES[3]) {
        panic!("{e}");
    }
}
