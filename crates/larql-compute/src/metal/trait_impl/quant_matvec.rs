//! `QuantMatVec` impl for `MetalBackend`.
//!
//! Each per-format method delegates to the corresponding kernel
//! dispatcher in `metal::ops` or to a per-format dispatcher built
//! around the appropriate shader pipeline.

use crate::backend::QuantMatVec;
use crate::metal::MetalBackend;

impl QuantMatVec for MetalBackend {
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_matvec_direct(q4_data, q8_x, q8_scales, num_rows, hidden))
    }

    /// Q4 matvec → GPU argmax_partial, returning `(token_id, score)` for
    /// the top-1 element. Used by the lm_head greedy-decode path on models
    /// that have a Q4 lm_head (`lm_head_q4.bin` or synthesized from f16
    /// embeddings). Saves the 1MB readback + 262K-element CPU sort.
    fn q4_matvec_topk1(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<(u32, f32)> {
        if num_rows == 0 || q8_x.len() != hidden {
            return None;
        }
        let buf_q4 = self.bufs.get_bytes(q4_data);
        let buf_q8 = self.bufs.transient_from_i8(q8_x);
        let buf_scales = self.bufs.transient_from_f32(q8_scales);
        let scores = self.bufs.output((num_rows * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        crate::metal::ops::q4_matvec::encode(
            enc,
            &self.q4.matvec,
            &buf_q4,
            &buf_q8,
            &buf_scales,
            &scores,
            num_rows as u32,
            hidden as u32,
            num_rows,
        );
        let (partial_vals, partial_idxs, n_partials) =
            self.encode_argmax_partial(enc, &scores, num_rows);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Self::reduce_argmax_partial(&partial_vals, &partial_idxs, n_partials)
    }

    /// Q4 matvec + GPU partial top-K. Returns up to `top_k` entries
    /// (`top_k ≤ K_TOPK = 8`) sorted descending. Caller falls back to
    /// `q4_matvec` + CPU sort when this returns `None`.
    fn q4_matvec_topk(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        if top_k == 0 || top_k > crate::metal::shaders::f32_gemv::K_TOPK {
            return None;
        }
        if num_rows == 0 || q8_x.len() != hidden {
            return None;
        }
        let buf_q4 = self.bufs.get_bytes(q4_data);
        let buf_q8 = self.bufs.transient_from_i8(q8_x);
        let buf_scales = self.bufs.transient_from_f32(q8_scales);
        let scores = self.bufs.output((num_rows * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        crate::metal::ops::q4_matvec::encode(
            enc,
            &self.q4.matvec,
            &buf_q4,
            &buf_q8,
            &buf_scales,
            &scores,
            num_rows as u32,
            hidden as u32,
            num_rows,
        );
        let (partial_vals, partial_idxs, num_tgs) =
            self.encode_topk_partial(enc, &scores, num_rows);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Some(MetalBackend::reduce_topk_partial(
            &partial_vals,
            &partial_idxs,
            num_tgs,
            top_k,
        ))
    }

    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_vecmat_direct(activation, q4_data, intermediate, hidden))
    }

    fn q4_matvec_pair_batch(
        &self,
        gate_q4: &[u8],
        up_q4: &[u8],
        x_matrix: &[f32],
        seq_len: usize,
        num_rows: usize,
        hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        Some(self.q4_matvec_pair_batch_direct(gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden))
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q4k_matvec as q4k;
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::metal::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q6k_matvec as q6k;
        let buf_w = self.bufs.get_bytes(q6k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(q6k::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.q6k_matvec_pipeline.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::metal::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    fn has_q4(&self) -> bool {
        true
    }
}
