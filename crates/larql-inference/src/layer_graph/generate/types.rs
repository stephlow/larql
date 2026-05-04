/// Sum of per-stage decode times across every successful step.
///
/// Dividing each field by `GenerateResult::decode_ms.len()` gives the
/// per-token average. Populated unconditionally — the six
/// `Instant::now()` calls per step are negligible next to the GPU
/// forward pass and the LM-head gemv.
#[derive(Debug, Default, Clone, Copy)]
pub struct StageTimings {
    pub embed_ms_total: f64,
    pub gpu_ms_total: f64,
    /// Gate+up dispatch time within GPU fwd (populated when LARQL_PROFILE_SPLIT=1).
    pub gate_up_ms_total: f64,
    /// Activation+down+residual time within GPU fwd (populated when LARQL_PROFILE_SPLIT=1).
    pub down_ms_total: f64,
    pub norm_ms_total: f64,
    pub lm_head_ms_total: f64,
    pub detok_ms_total: f64,
}

/// Result of multi-token generation.
pub struct GenerateResult {
    pub tokens: Vec<(String, f64)>,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
    pub stage_timings: StageTimings,
}

impl StageTimings {
    /// Per-token average across `n` decode steps. Returns all-zero if
    /// `n == 0` (short-circuit no-decode paths safely).
    pub fn avg_per_step(&self, n: usize) -> StageTimings {
        if n == 0 {
            return Self::default();
        }
        let nf = n as f64;
        StageTimings {
            embed_ms_total: self.embed_ms_total / nf,
            gpu_ms_total: self.gpu_ms_total / nf,
            gate_up_ms_total: self.gate_up_ms_total / nf,
            down_ms_total: self.down_ms_total / nf,
            norm_ms_total: self.norm_ms_total / nf,
            lm_head_ms_total: self.lm_head_ms_total / nf,
            detok_ms_total: self.detok_ms_total / nf,
        }
    }
}

impl GenerateResult {
    pub fn avg_decode_ms(&self) -> f64 {
        if self.decode_ms.is_empty() {
            0.0
        } else {
            self.decode_ms.iter().sum::<f64>() / self.decode_ms.len() as f64
        }
    }

    pub fn decode_tok_s(&self) -> f64 {
        let avg = self.avg_decode_ms();
        if avg > 0.0 {
            1000.0 / avg
        } else {
            0.0
        }
    }

    pub fn text(&self) -> String {
        self.tokens
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("")
    }
}
