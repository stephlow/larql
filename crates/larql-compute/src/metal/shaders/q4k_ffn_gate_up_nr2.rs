//! Fused Q4_K gate+up — **NR0=2 multi-row + shared-X** variant.
//!
//! Same Q4_K (144-byte super-block) input format and output as
//! [`q4k_ffn_gate_up`], but each simdgroup computes **two output rows
//! in parallel**, with the X-vector slice loaded once into per-lane
//! registers and reused across both rows. Mirrors llama.cpp's
//! `kernel_mul_mv_q4_K_f32` shape (`N_R0_Q4_K = 2`, `N_SG_Q4_K = 2`,
//! `ggml/src/ggml-metal/ggml-metal-impl.h`).
//!
//! **Why this kernel exists**: side-by-side bench against
//! `ollama gemma3:4b` on the same prompt + num_predict (2026-05-01)
//! shows ollama at 96 tok/s vs larql at 71.5 tok/s — a 3.5 ms/tok
//! gap concentrated in GPU forward kernels. Diagnosis traced this
//! to **X-cache-traffic pressure**: our `q4k_ffn_gate_up_8sg` runs at
//! 187 GB/s = 47% of M3 Max LPDDR5X peak; the same matvec in llama.cpp
//! sits closer to ~80% peak. Difference: llama.cpp's `NR0=2` shape
//! halves the per-row X-vector reload count by reusing the per-lane
//! `xl[16]` register tile across two output rows. The G-1 cooperative-
//! dequant attempt (2026-05-01) targeted ALU instead, missed the real
//! bottleneck.
//!
//! **Pattern**:
//!
//! 1. `ROWS_PER_TG = 8` (4 simdgroups × NR0=2 rows each), same total
//!    rows-per-TG as the production 8sg variant — dispatch grid math
//!    is unchanged.
//! 2. Each simdgroup picks `row_base = mat_tg * 8 + sg_id * 2`; the
//!    two rows it owns are `row_base` and `row_base + 1` (adjacent —
//!    better L2 reuse on the per-row Q4_K weight bytes).
//! 3. Inner loop: `xl[16]` loaded once per super-block-half. For each
//!    of the two rows, the lane reads its 16-byte nibble slice from
//!    that row's super-block and accumulates into `sumf[2]`.
//! 4. Final: `simd_sum` per-row, two writes.
//!
//! **Key shared loads** (per simdgroup, per super-block):
//! - 16 X-values (`xl[16]`, register-resident) — loaded once.
//! - super-block `d` and `dmin` — decoded once (per row, but we do it
//!   per lane redundantly to avoid register pressure on per-lane scale
//!   broadcasts; the dequant ALU runs concurrently with weight loads
//!   per the G-1 finding).
//! - per-row sub-block `sc`, `mn` — each lane reads its own row's
//!   header, so 32× redundant per row × 2 rows. Keeps register
//!   footprint flat.
//!
//! **Numerics**: bit-equivalent to `q4k_ffn_gate_up` per row. Each
//! row's `scale * dot_acc - mmin * sumy` is the same expression as
//! production (only `dot_acc[row]` and per-row `scale`/`mmin` are
//! per-row; `xl[16]` and `sumy` are shared). Verified by per-row
//! parity test against the production kernel on synthetic data.
//!
//! **Register footprint risk**: from prior auto-memory:
//!     "N_DST=2 caused ~10% regression, N_DST=4 caused 24× regression
//!     (register spilling)".
//! That earlier attempt likely doubled per-thread register footprint
//! without sharing X. Here, X is loaded **once** into `xl[16]`, so
//! the additional cost is `sumf[2]` (1 extra float per lane) plus
//! per-row `dot_acc`, `scale`, `mmin` scalars (3 extra). Total +4
//! floats/lane vs production — within slack.

pub const SHADER: &str = r#"
constant uint Q4K_GUNR2_ROWS_PER_TG = 8;
constant uint Q4K_GUNR2_BLOCK_SIZE  = 144;
constant uint Q4K_GUNR2_NR0         = 2;

kernel void q4k_ffn_gate_up_nr2(
    device const uchar*  Wg    [[buffer(0)]],
    device const uchar*  Wu    [[buffer(1)]],
    device const float*  X     [[buffer(2)]],
    device float*        G_out [[buffer(3)]],
    device float*        U_out [[buffer(4)]],
    constant uint&       N     [[buffer(5)]],
    constant uint&       K     [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    // Dispatch grid: gate first half, up second half — same convention
    // as production `q4k_ffn_gate_up`.
    uint tgs_per_mat = (N + Q4K_GUNR2_ROWS_PER_TG - 1u) / Q4K_GUNR2_ROWS_PER_TG;
    bool is_up  = (tg_id >= tgs_per_mat);
    uint mat_tg = is_up ? (tg_id - tgs_per_mat) : tg_id;

    // Each simdgroup handles NR0=2 adjacent rows.
    uint row_base = mat_tg * Q4K_GUNR2_ROWS_PER_TG + sg_id * Q4K_GUNR2_NR0;
    if (row_base >= N) return;
    uint nrows = (row_base + Q4K_GUNR2_NR0 <= N) ? Q4K_GUNR2_NR0 : (N - row_base);

    device const uchar* W       = is_up ? Wu : Wg;
    device float*       out_buf = is_up ? U_out : G_out;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_GUNR2_BLOCK_SIZE;

    // Lane partition (matches production):
    //   ix  = lane & 1   → super-block parity
    //   tid = lane >> 1  → 0..15: which (sub, half) cell
    //   j   = tid >> 1   → 0..7: which sub-block (4 lanes share j)
    //   sh  = tid & 1    → 0/1: first or last 16 of the 32-elem sub-block
    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0u;
    const uint group = j >> 1u;

    // Per-row accumulators (NR0=2). Compiler keeps these in registers
    // alongside the shared `xl[16]`.
    float acc[2] = { 0.0f, 0.0f };

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        // ── Shared X-load: 16 X values into per-lane registers ──
        // This load is reused across BOTH output rows below — the
        // bandwidth saving over the production NR0=1 kernel.
        const uint x_base = sb * 256u + j * 32u + sh * 16u;
        float xl[16];
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

        // Σ X over the 16-element slice — also shared across both rows.
        float sumy = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { sumy += xl[l]; }

        // ── Per-row work: dequant + FMA chain against `xl[16]` ──
        // Manually unrolled NR0=2 (avoids array-of-pointer indirections
        // that older compilers handled poorly).
        for (uint row = 0u; row < nrows; row++) {
            device const uchar* row_w = W + (row_base + row) * bytes_per_row;
            device const uchar* block = row_w + sb * Q4K_GUNR2_BLOCK_SIZE;

            ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
            ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
            float d    = decode_f16_metal(d_bits);
            float dmin = decode_f16_metal(dmin_bits);

            device const uchar* sb_bytes = block + 4u;
            uint sc, mn;
            if (j < 4u) {
                sc = uint(sb_bytes[j])      & 0x3Fu;
                mn = uint(sb_bytes[j + 4u]) & 0x3Fu;
            } else {
                sc = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
                mn = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
            }
            float scale = d * float(sc);
            float mmin  = dmin * float(mn);

            device const uchar* qs = block + 16u + group * 32u + sh * 16u;

            float dot_acc = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                uchar byte = qs[l];
                float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
                dot_acc = fma(nib, xl[l], dot_acc);
            }
            // Q4_K deferred form: scale * Σ(nib*x) - dmin_min * Σ(x).
            acc[row] += scale * dot_acc - mmin * sumy;
        }
    }

    // Final reduction: simd_sum per row, write per row.
    for (uint row = 0u; row < nrows; row++) {
        float r = simd_sum(acc[row]);
        if (lane == 0u) out_buf[row_base + row] = r;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::metal::kernel::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_ffn_gate_up_nr2";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
