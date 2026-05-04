//! Fused Q4_K QKV projection — v2 with the (ix, j, sh) lane
//! decomposition.
//!
//! The original [`q4k_qkv_proj`] uses `for sb = lane; sb < superblocks;
//! sb += 32` — works fine when K is large (e.g. K=8192 → 32 super-blocks
//! → all 32 lanes do work) but at the production K=2560 (10
//! super-blocks) **22 of 32 lanes are idle** in every simdgroup. That
//! puts the kernel at 33% of LPDDR5X peak (131.6 GB/s on M3 Max,
//! profiler 2026-04-28) — by far the most under-utilised kernel and
//! 6.1 ms/tok of the ~12 ms GPU forward.
//!
//! This variant uses the same `(ix, j, sh)` decomposition that
//! `q4k_matvec` adopted in 2026-04-25:
//!   - `ix = lane & 1`        — 2-way inter-superblock interleave
//!   - `tid = lane >> 1`      — 0..15 within each ix-group
//!   - `j = tid >> 1`         — 0..7 sub-block within superblock
//!   - `sh = tid & 1`         — 0/1 first/last 16-elem half
//!
//! All 32 lanes are productive for any K ≥ 256 — the (j, sh) covers
//! 256 elements (= one superblock) using 16 lanes, and ix doubles
//! it across two adjacent superblocks. At K=2560 (10 superblocks)
//! ix=0 covers 5 even superblocks, ix=1 covers 5 odd. Full
//! utilisation.
//!
//! Same per-thread register footprint as the original (one float
//! accumulator + 16 X preload + scale/min decode), so no register
//! pressure regression. ROWS_PER_TG=8 / 256 threads/TG is unchanged
//! (the original is already 8sg).
//!
//! Parity contract: bit-equal output to [`q4k_qkv_proj`]. Math is
//! identical, only the lane→element mapping changes.

pub const SHADER: &str = r#"
constant uint Q4K_QKV_V2_ROWS_PER_TG = 8;
constant uint Q4K_QKV_V2_BLOCK_SIZE  = 144;

kernel void q4k_qkv_proj_v2(
    device const uchar*  Wq    [[buffer(0)]],
    device const uchar*  Wk    [[buffer(1)]],
    device const uchar*  Wv    [[buffer(2)]],
    device const float*  X     [[buffer(3)]],
    device float*        Q_out [[buffer(4)]],
    device float*        K_out [[buffer(5)]],
    device float*        V_out [[buffer(6)]],
    constant uint&       q_rows [[buffer(7)]],
    constant uint&       k_rows [[buffer(8)]],
    constant uint&       v_rows [[buffer(9)]],
    constant uint&       K      [[buffer(10)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_QKV_V2_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    device const uchar* W;
    device float* out_buf;
    uint local_row;
    if (global_row < q_rows) {
        W = Wq; out_buf = Q_out; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = Wk; out_buf = K_out; local_row = global_row - q_rows;
    } else {
        W = Wv; out_buf = V_out; local_row = global_row - q_rows - k_rows;
    }

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q4K_QKV_V2_BLOCK_SIZE;
    device const uchar* row_w = W + local_row * bytes_per_row;

    // Same lane decomposition as q4k_matvec / q4k_ffn_gate_up — uses
    // all 32 lanes per simdgroup regardless of how many superblocks
    // per row.
    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0u;
    const uint group = j >> 1u;

    float acc = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        device const uchar* block = row_w + sb * Q4K_QKV_V2_BLOCK_SIZE;
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

        const uint x_base = sb * 256u + j * 32u + sh * 16u;
        float xl[16];
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

        device const uchar* qs = block + 16u + group * 32u + sh * 16u;

        float sumy = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) { sumy += xl[l]; }

        float dot_acc = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 16u; l++) {
            uchar byte = qs[l];
            float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
            dot_acc = fma(nib, xl[l], dot_acc);
        }
        acc += scale * dot_acc - mmin * sumy;
    }

    acc = simd_sum(acc);
    if (lane == 0u) out_buf[local_row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding.
pub struct Kernel;
impl crate::metal::kernel::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_qkv_proj_v2";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
