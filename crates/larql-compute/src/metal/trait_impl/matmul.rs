//! `MatMul` impl + private encoder helpers shared by `f32_gemv` and
//! `f16_gemv` (threshold-gated and force variants).

use std::sync::atomic::Ordering;
use ndarray::{Array2, ArrayView2};

use crate::backend::{MatMul, MatMulOp};
use crate::metal::MetalBackend;

impl MatMul for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k { return None; }
        // Fall back below the GPU threshold — small gemvs are dominated by
        // dispatch overhead.
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (_n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k { return None; }
        self.encode_f32_gemv(w, x)
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k { return None; }
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) { return None; }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k { return None; }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter().map(|op| {
            if op.transpose_b { self.matmul_transb(op.a.view(), op.b.view()) }
            else { self.matmul(op.a.view(), op.b.view()) }
        }).collect()
    }
}

impl MetalBackend {
    /// Shared GPU dispatch body for `f32_gemv` (threshold-gated) and
    /// `f32_gemv_force` (direct). Kept inherent so the 30+ lines of
    /// Metal plumbing aren't duplicated.
    fn encode_f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k { return None; }
        let w_buf = match w.as_slice() {
            Some(s) => self.bufs.get_f32(s),
            None => {
                let owned = w.as_standard_layout().into_owned();
                self.bufs.transient_from_f32(owned.as_slice().unwrap())
            }
        };
        let x_buf = self.bufs.transient_from_f32(x);
        let out_buf = self.bufs.output((n * 4) as u64);

        use crate::metal::shaders::f32_gemv as sh;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(sh::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.f32_gemv_pipeline);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::metal::buffers::read_buffer_f32(&out_buf, n))
    }

    /// Shared dispatch body for f16-weight gemv (behind both trait
    /// variants: threshold-gated `f16_gemv` and direct `f16_gemv_force`).
    fn encode_f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        let w_buf = self.bufs.get_bytes(w_f16);
        let x_buf = self.bufs.transient_from_f32(x);
        let out_buf = self.bufs.output((n * 4) as u64);

        use crate::metal::shaders::f16_gemv as sh;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(sh::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.f16_gemv_pipeline);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::metal::buffers::read_buffer_f32(&out_buf, n))
    }
}
