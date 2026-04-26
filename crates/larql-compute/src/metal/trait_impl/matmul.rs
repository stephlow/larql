//! `MatMul` impl + private encoder helpers shared by `f32_gemv` and
//! `f16_gemv` (threshold-gated and force variants).

use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;

use crate::backend::{MatMul, MatMulOp};
use crate::metal::MetalBackend;

impl MatMul for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(
            &self.queue,
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(Ordering::Relaxed),
        )
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(
            &self.queue,
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(Ordering::Relaxed),
        )
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        // Fall back below the GPU threshold — small gemvs are dominated by
        // dispatch overhead.
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (_n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        MetalBackend::f32_gemv_topk1(self, w, x)
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter()
            .map(|op| {
                if op.transpose_b {
                    self.matmul_transb(op.a.view(), op.b.view())
                } else {
                    self.matmul(op.a.view(), op.b.view())
                }
            })
            .collect()
    }
}

impl MetalBackend {
    /// Shared GPU dispatch body for `f32_gemv` (threshold-gated) and
    /// `f32_gemv_force` (direct). Kept inherent so the 30+ lines of
    /// Metal plumbing aren't duplicated.
    fn encode_f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        let w_buf = match w.as_slice() {
            Some(s) => self.bufs.get_f32(s),
            None => {
                let owned = w.as_standard_layout().into_owned();
                self.bufs.transient_from_f32(owned.as_slice().unwrap())
            }
        };
        let x_buf = self.bufs.transient_from_f32(x);
        let out_buf = self.bufs.output((n * 4) as u64);

        // Geometry travels with the f32_gemv KernelHandle.
        let kernel = &self.f32_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(kernel.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::metal::buffers::read_buffer_f32(&out_buf, n))
    }

    /// GPU gemv → GPU argmax, returning (token_id, score) without a 1MB readback.
    ///
    /// Replaces the three-step `f32_gemv` + read 262K floats + CPU argmax with:
    /// 1. f32_gemv kernel → scores buffer (stays on GPU)
    /// 2. f32_argmax_partial → 1024 (val, idx) partial results (8 KB)
    /// 3. Read back 8 KB, CPU reduces 1024 candidates (~1 µs)
    ///
    /// Saves ~0.33ms (1MB readback eliminated). Used by lm_head top-1 path.
    pub fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k || n == 0 {
            return None;
        }

        let w_buf = match w.as_slice() {
            Some(s) => self.bufs.get_f32(s),
            None => {
                let owned = w.as_standard_layout().into_owned();
                self.bufs.transient_from_f32(owned.as_slice().unwrap())
            }
        };
        let x_buf = self.bufs.transient_from_f32(x);
        let scores = self.bufs.output((n * 4) as u64);

        // Phase 1: f32_gemv
        let kh = &self.f32_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let gemv_tgs = (n as u64).div_ceil(kh.rows_per_tg);

        // Phase 2: f32_argmax_partial — TG size = 256, one TG per 256 scores.
        const ARGMAX_TG_SZ: u64 = 256;
        let argmax_tgs = (n as u64).div_ceil(ARGMAX_TG_SZ);
        let partial_vals = self.bufs.output(argmax_tgs * 4); // f32 per TG
        let partial_idxs = self.bufs.output(argmax_tgs * 4); // u32 per TG
        let argmax_tg_sz_u32 = ARGMAX_TG_SZ as u32;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // gemv dispatch
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gemv_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        // argmax partial dispatch
        enc.set_compute_pipeline_state(&self.f32_argmax_partial_pipeline);
        enc.set_buffer(0, Some(&scores), 0);
        enc.set_buffer(1, Some(&partial_vals), 0);
        enc.set_buffer(2, Some(&partial_idxs), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(argmax_tgs, 1, 1),
            metal::MTLSize::new(ARGMAX_TG_SZ, 1, 1),
        );

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // CPU final reduction over ≤1024 partial results (8 KB readback).
        let n_partials = argmax_tgs as usize;
        let vals = crate::metal::buffers::read_buffer_f32(&partial_vals, n_partials);
        let idxs_raw = {
            let ptr = partial_idxs.contents() as *const u32;
            unsafe { std::slice::from_raw_parts(ptr, n_partials) }.to_vec()
        };

        let (best_idx, best_val) = vals
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });

        if best_val == f32::NEG_INFINITY {
            return None;
        }
        Some((idxs_raw[best_idx], best_val))
    }

    /// Shared dispatch body for f16-weight gemv (behind both trait
    /// variants: threshold-gated `f16_gemv` and direct `f16_gemv_force`).
    fn encode_f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        let w_buf = self.bufs.get_bytes(w_f16);
        let x_buf = self.bufs.transient_from_f32(x);
        let out_buf = self.bufs.output((n * 4) as u64);

        // Geometry travels with the f16_gemv KernelHandle.
        let kernel = &self.f16_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(kernel.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::metal::buffers::read_buffer_f32(&out_buf, n))
    }
}
