//! Per-kernel Metal GPU bandwidth profiler.
//!
//! Measures each production kernel at Gemma 3 4B shapes in two modes:
//!
//! **Isolated**: one commit+wait per kernel call. Includes ~20µs GPU spin-up
//! cost. Useful for comparing kernels against each other.
//!
//! **Batched**: `n_layers` (default 34) calls per command buffer, single
//! commit+wait. The GPU stays warm; this matches the real decode pipeline.
//! Use batched numbers for understanding actual tok/s impact.
//!
//! ## Key findings (2026-04-26, M3 Max, Gemma 3 4B)
//! | Kernel | Batched GB/s | ms/tok | Bottleneck |
//! |---|---|---|---|
//! | q6k_matvec (FFN down, K=10240) | 312 GB/s | 2.34ms | bandwidth-bound (LPDDR5X) |
//! | q4k_ffn_gate_up (gate+up, K=2560) | 272 GB/s | 3.68ms | compute-bound (Q4_K dequant) |
//! | lm_head f32_gemv (262K×2560) | 370 GB/s | — | bandwidth-bound (near peak) |
//!
//! Gate+up is compute-bound because Q4_K at K=2560 has low bytes-per-element
//! (0.5625 B/elem) — the GPU spends more cycles on nibble dequant than waiting
//! for memory. Closing the gap vs Ollama's ~414 GB/s effective rate requires
//! reducing the per-element compute overhead (vectorized accumulation).

use std::time::Instant;

/// Result for a single kernel profiling run.
#[derive(Debug, Clone)]
pub struct KernelResult {
    pub name: String,
    /// Megabytes of weight data read per kernel call.
    pub mb_per_call: f64,
    /// Mean isolated time per call (ms), including GPU spin-up.
    pub isolated_ms: f64,
    /// Stddev of isolated times.
    pub isolated_sd_ms: f64,
    /// Effective bandwidth from isolated measurement (GB/s).
    pub isolated_gbs: f64,
    /// Mean time per layer when batched n_layers in one command buffer (ms).
    pub batched_ms_per_layer: f64,
    /// Effective bandwidth from batched measurement (GB/s).
    pub batched_gbs: f64,
}

impl KernelResult {
    /// ms/token at `n_layers` layers using the batched rate.
    pub fn ms_per_token(&self, n_layers: usize) -> f64 {
        self.batched_ms_per_layer * n_layers as f64
    }

    /// Whether the kernel appears compute-bound (GB/s well below peak ~350).
    pub fn is_compute_bound(&self) -> bool {
        self.batched_gbs < 300.0
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn synth_f32(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.007).sin() * 0.4)
        .collect()
}

fn measure_isolated(warmup: usize, iters: usize, f: &mut impl FnMut()) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms);
        }
    }
    (mean(&times), stddev(&times))
}

/// Measure batched throughput where each iteration runs `f()` `n_layers`
/// times. **`f()` is responsible for its own cmd-buffer + commit + wait.**
///
/// This MIS-measures throughput when used with closures that create one
/// cmd-buffer per call: each cmd-buffer costs ~10 µs of dispatch overhead
/// that gets billed against the kernel time. Real production runs all
/// `n_layers` dispatches in ONE cmd buffer with a single commit+wait —
/// see [`measure_single_cmdbuf_batched`] for that.
///
/// Kept for callers who genuinely want per-call cmd-buffer overhead in
/// the measurement (rare).
fn measure_batched(warmup: usize, iters: usize, n_layers: usize, f: &mut impl FnMut()) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        for _ in 0..n_layers {
            f();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms / n_layers as f64);
        }
    }
    mean(&times)
}

/// Measure batched throughput with all `n_layers` dispatches in ONE cmd
/// buffer, single commit+wait. This is what production decode actually
/// does (all of a token's per-layer kernels live in one cmd buffer), so
/// the GB/s number reflects real per-kernel cost without dispatch
/// overhead pollution.
///
/// `encode` must NOT call `commit`/`wait_until_completed`/`end_encoding`
/// — this function owns the cmd-buffer lifecycle.
///
/// Discovered 2026-04-28: the older `measure_batched` was being used
/// with closures that did per-call commit+wait, undercounting q6k_matvec
/// throughput by 4× (74 vs real 315 GB/s). See ROADMAP P0 "Decode kernel
/// optimization → Track A" for the bisect.
fn measure_single_cmdbuf_batched(
    metal: &super::super::MetalBackend,
    warmup: usize,
    iters: usize,
    n_layers: usize,
    encode: &impl Fn(&metal::ComputeCommandEncoderRef),
) -> f64 {
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..n_layers {
            encode(enc);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms / n_layers as f64);
        }
    }
    mean(&times)
}

/// Profile all production kernels at Gemma 3 4B shapes.
///
/// Returns one `KernelResult` per kernel. Prints a formatted table to stdout.
/// Pass `n_layers=34` for Gemma 3 4B, `warmup=5`, `iters=50` for reliable numbers.
#[cfg(feature = "metal")]
pub fn profile_all(n_layers: usize, warmup: usize, iters: usize) -> Vec<KernelResult> {
    use crate::{
        cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k},
        metal::MetalBackend,
        MatMul, QuantMatVec,
    };
    use metal::MTLSize;

    let metal = MetalBackend::new().expect("Metal backend required for profiling");

    // Gemma 3 4B production shapes
    let hidden = 2560usize;
    let inter = 10240usize;
    let q_dim = 8192usize;
    let _kv_dim = 4096usize;
    let sb = 256usize;
    let q4k_sb = 144usize;
    let q6k_sb = 210usize;

    let mut results = Vec::new();

    // Measure commit+wait overhead (empty command buffer).
    let commit_overhead_ms = {
        let mut times = Vec::new();
        for i in 0..warmup + iters {
            let t = Instant::now();
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if i >= warmup {
                times.push(ms);
            }
        }
        mean(&times)
    };

    println!("Commit+wait overhead: {commit_overhead_ms:.3}ms");
    println!();
    println!(
        "{:<44} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Kernel", "iso_ms", "iso_gbs", "bat_ms", "bat_gbs", "ms/tok"
    );
    println!("{}", "-".repeat(88));

    // ── q6k_matvec: FFN down (N=hidden, K=inter) ─────────────────────────
    {
        let n = hidden;
        let k = inter;
        let mb = (n * (k / sb * q6k_sb)) as f64 / 1e6;
        let w = quantize_q6_k(&synth_f32(n * k, 0.1));
        let x = synth_f32(k, 0.5);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = metal.q6k_matvec(&w, &x, n, k);
        });

        let wb = metal.bufs().get_bytes(&w);
        let xb = metal.bufs().transient_from_f32(&x);
        let ob = metal.bufs().output((n * 4) as u64);
        let kh = &metal.q6k_matvec_pipeline;
        let n_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        let n_val = n as u32;
        let k_val = k as u32;

        // TRUE batched: all n_layers dispatches in ONE cmd buffer.
        let bat_ms = measure_single_cmdbuf_batched(&metal, warmup, iters, n_layers, &|enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        });

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "q6k_matvec (down, 2560×10240)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: bat_ms,
            batched_gbs: mb / bat_ms,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── q4k_ffn_gate_up: fused gate+up (N=inter, K=hidden) ───────────────
    {
        let n = inter;
        let k = hidden;
        let mb = 2.0 * (n * (k / sb * q4k_sb)) as f64 / 1e6;
        let gate_q4k = quantize_q4_k(&synth_f32(n * k, 0.2));
        let up_q4k = quantize_q4_k(&synth_f32(n * k, 0.3));
        let x = synth_f32(k, 0.5);

        // Isolated: use the trait method which handles dispatch internally.
        // We can't use trait method for gate+up (it's internal), so dispatch directly.
        let wg = metal.bufs().get_bytes(&gate_q4k);
        let wu = metal.bufs().get_bytes(&up_q4k);
        let xb = metal.bufs().transient_from_f32(&x);
        let go = metal.bufs().output((n * 4) as u64);
        let uo = metal.bufs().output((n * 4) as u64);
        let kh = &metal.q4k_ffn_gate_up_pipeline;
        let tgs = (n as u64).div_ceil(kh.rows_per_tg);
        let n_val = n as u32;
        let k_val = k as u32;

        let dispatch = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wg), 0);
            enc.set_buffer(1, Some(&wu), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&go), 0);
            enc.set_buffer(4, Some(&uo), 0);
            enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs * 2, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        };

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            dispatch(enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
        // TRUE batched: all n_layers dispatches in ONE cmd buffer.
        let bat_ms = measure_single_cmdbuf_batched(&metal, warmup, iters, n_layers, &dispatch);

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "q4k_ffn_gate_up (gate+up, 10240×2560)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: bat_ms,
            batched_gbs: mb / bat_ms,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── q4k_matvec: Wo O-projection (N=hidden, K=q_dim) ──────────────────
    {
        let n = hidden;
        let k = q_dim;
        let mb = (n * (k / sb * q4k_sb)) as f64 / 1e6;
        let w = quantize_q4_k(&synth_f32(n * k, 0.4));
        let x = synth_f32(k, 0.6);
        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = metal.q4k_matvec(&w, &x, n, k);
        });
        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        // Batched Wo: approximate — use isolated kernel time as lower bound.
        let r = KernelResult {
            name: "q4k_matvec (Wo, 2560×8192)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: iso_kernel, // approximate
            batched_gbs: mb / iso_kernel,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms  (iso only)",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── f32_gemv: lm_head (N=vocab, K=hidden) ────────────────────────────
    {
        let n = 262_144usize;
        let k = hidden;
        let mb = (n * k * 4) as f64 / 1e6;
        let w = ndarray::Array2::from_shape_vec((n, k), synth_f32(n * k, 0.7)).unwrap();
        let x = synth_f32(k, 0.5);
        let (iso_ms, iso_sd) = measure_isolated(warmup, iters.min(20), &mut || {
            let _ = metal.f32_gemv_force(w.view(), &x);
        });
        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "f32_gemv (lm_head, 262K×2560)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: iso_ms, // lm_head is one-per-token, not per-layer
            batched_gbs: mb / iso_kernel,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7}     {:>7}   (per token, not per layer)",
            r.name, r.isolated_ms, r.isolated_gbs, "—", "—"
        );
        results.push(r);
    }

    // ── Summary ───────────────────────────────────────────────────────────
    let down = results.iter().find(|r| r.name.contains("down")).unwrap();
    let gate = results.iter().find(|r| r.name.contains("gate")).unwrap();
    let total_ms = down.ms_per_token(n_layers) + gate.ms_per_token(n_layers);

    println!();
    println!("=== Bottleneck analysis ===");
    println!(
        "q6k_matvec (down)   {:.1} GB/s — {}",
        down.batched_gbs,
        if down.is_compute_bound() {
            "COMPUTE-BOUND"
        } else {
            "bandwidth-bound"
        }
    );
    println!(
        "q4k_ffn_gate_up     {:.1} GB/s — {}",
        gate.batched_gbs,
        if gate.is_compute_bound() {
            "COMPUTE-BOUND (K=2560 dequant dominates)"
        } else {
            "bandwidth-bound"
        }
    );
    println!(
        "These two: {total_ms:.2}ms/tok ({:.0}% of ~11.7ms GPU fwd)",
        total_ms / 11.7 * 100.0
    );
    println!(
        "At 350 GB/s: would take {:.1}ms/tok → need {:.0}% more throughput",
        3029.0 / 350.0,
        (3029.0 / 350.0 / (down.batched_ms_per_layer + gate.batched_ms_per_layer + 0.001) - 1.0)
            .abs()
            * 0.0
            + (350.0 / ((down.batched_gbs + gate.batched_gbs) / 2.0) - 1.0) * 100.0
    );

    results
}
