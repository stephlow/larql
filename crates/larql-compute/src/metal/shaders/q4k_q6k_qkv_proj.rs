//! Fused mixed-quant QKV projection — Q4_K for Q/K rows, Q6_K for V rows.
//!
//! **Both branches now use the same 2-way inter-superblock interleaving
//! as `q4k_matvec` and `q6k_matvec`.**
//!
//! Previous Q/K branch used `for (sb = lane; sb < superblocks; sb += 32)` —
//! for K=2560 (10 superblocks) only lanes 0..9 were active; 22 of 32 lanes
//! sat idle (31% utilisation). New approach: `ix = lane & 1` splits 32 lanes
//! into two groups that stride alternate superblocks, keeping all 32 lanes
//! busy and letting the DRAM controller serve two banks in parallel.
//!
//! Lane decomposition (shared by Q4_K and Q6_K branches):
//!   ix  = lane & 1      — 0/1: even/odd superblock group
//!   tid = lane >> 1     — 0..15: position within the group
//!
//! Q4_K Q/K branch additionally:
//!   j  = tid >> 1       — 0..7: which sub-block (32 elements)
//!   sh = tid & 1        — 0/1: first or last 16 elements
//!   X preloaded into xl[16] before weight reads.
//!
//! Q6_K V branch additionally (matches q6k_matvec):
//!   base    = tid * 4   — 0,4,...,60
//!   sc_base = tid / 4   — scale group index
//!   4 passes × 4 elements each, xl[16] preloaded.

pub const SHADER: &str = r#"
constant uint Q4K_Q6K_ROWS_PER_TG  = 4;
constant uint Q4K_BLOCK_SIZE_MIXED  = 144;
constant uint Q6K_BLOCK_SIZE_MIXED  = 210;

kernel void q4k_q6k_qkv_proj(
    device const uchar*  Wq     [[buffer(0)]],
    device const uchar*  Wk     [[buffer(1)]],
    device const uchar*  Wv     [[buffer(2)]],
    device const float*  X      [[buffer(3)]],
    device float*        Q_out  [[buffer(4)]],
    device float*        K_out  [[buffer(5)]],
    device float*        V_out  [[buffer(6)]],
    constant uint&       q_rows [[buffer(7)]],
    constant uint&       k_rows [[buffer(8)]],
    constant uint&       v_rows [[buffer(9)]],
    constant uint&       K      [[buffer(10)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_Q6K_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    const uint superblocks = K / 256u;
    float acc = 0.0f;

    // Shared lane decomposition for both branches.
    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;   // 0..15

    if (global_row < q_rows + k_rows) {
        // ── Q/K rows: Q4_K ──
        uint local_row;
        device const uchar* W;
        device float* out_buf;
        if (global_row < q_rows) {
            W = Wq; out_buf = Q_out; local_row = global_row;
        } else {
            W = Wk; out_buf = K_out; local_row = global_row - q_rows;
        }

        const uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_MIXED;
        device const uchar* row = W + local_row * bytes_per_row;

        const uint j   = tid >> 1u;    // 0..7: sub-block
        const uint sh  = tid & 1u;     // 0/1: first/last 16 elements
        const bool hi    = (j & 1u) != 0u;
        const uint group = j >> 1u;

        for (uint sb = ix; sb < superblocks; sb += 2u) {
            device const uchar* block = row + sb * Q4K_BLOCK_SIZE_MIXED;
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
            float dot_acc = 0.0f, sum_acc = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                uchar byte = qs[l];
                float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
                dot_acc = fma(nib, xl[l], dot_acc);
                sum_acc += xl[l];
            }
            acc += scale * dot_acc - mmin * sum_acc;
        }

        acc = simd_sum(acc);
        if (lane == 0u) out_buf[local_row] = acc;

    } else {
        // ── V rows: Q6_K (matches new q6k_matvec) ──
        uint local_row = global_row - q_rows - k_rows;
        const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE_MIXED;
        device const uchar* row = Wv + local_row * bytes_per_row;

        // Exact q6k_matvec decomposition: tid=0..7 → ip=0 (elements 0..127),
        // tid=8..15 → ip=1 (elements 128..255).
        const uint ip      = tid >> 3u;
        const uint il      = tid & 7u;
        const uint l0      = il << 2u;
        const uint v_base  = (ip << 7u) + l0;   // X base: 0..28 or 128..156
        const uint q_off_l = (ip << 6u) + l0;   // lo4 base: 0..28 or 64..92
        const uint q_off_h = (ip << 5u) + l0;   // hi2 base: 0..28 or 32..60
        const uint sc_base = (ip << 3u) + (il >> 2u); // 0 or 1 (ip=0), 8 or 9 (ip=1)

        for (uint i = ix; i < superblocks; i += 2u) {
            device const uchar* block = row + i * Q6K_BLOCK_SIZE_MIXED;
            device const uchar* ql   = block;
            device const uchar* qh   = block + 128u;
            device const char*  sc   = (device const char*)(block + 192u) + sc_base;
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float  d = decode_f16_metal(d_bits);

            const uint xb = i * 256u + v_base;
            float xl[16];
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 4u; l++) {
                xl[4u*l + 0u] = X[xb + l      ];
                xl[4u*l + 1u] = X[xb + l + 32u];
                xl[4u*l + 2u] = X[xb + l + 64u];
                xl[4u*l + 3u] = X[xb + l + 96u];
            }

            float4 sums = float4(0.0f);
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 4u; l++) {
                uchar la = ql[q_off_l + l], lb = ql[q_off_l + l + 32u], hi = qh[q_off_h + l];
                sums[0] += xl[4u*l+0u] * float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32);
                sums[1] += xl[4u*l+1u] * float((char)((lb & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32);
                sums[2] += xl[4u*l+2u] * float((char)((la >> 4u)   | ((hi & 0x30u)       )) - 32);
                sums[3] += xl[4u*l+3u] * float((char)((lb >> 4u)   | ((hi & 0xC0u) >> 2u)) - 32);
            }
            acc += d * (sums[0]*float(sc[0]) + sums[1]*float(sc[2])
                      + sums[2]*float(sc[4]) + sums[3]*float(sc[6]));
        }

        acc = simd_sum(acc);
        if (lane == 0u) V_out[local_row] = acc;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::metal::kernel::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_q6k_qkv_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
