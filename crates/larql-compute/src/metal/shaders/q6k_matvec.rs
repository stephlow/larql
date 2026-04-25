//! Q6_K matrix-vector multiply — llama.cpp-compatible GGUF Q6_K kernel.
//!
//! Q6_K super-block layout (256 values = 210 bytes):
//!   [0..127]    128 bytes: ql — lower 4 bits (2 per byte, elements interleaved below)
//!   [128..191]   64 bytes: qh — upper 2 bits (4 per byte)
//!   [192..207]   16 bytes: int8 scales (one per 16-element group)
//!   [208..209]    2 bytes: f16 super-block scale d
//!
//! GGUF Q6_K element layout (per 128-element n-block, n=0 or 128):
//!   for l=0..31:  element[n+l+  0] = (ql[l]   & 0xF) | (qh[l]      & 0x03) << 4 - 32
//!                 element[n+l+ 32] = (ql[l+32] & 0xF) | (qh[l] >> 2 & 0x03) << 4 - 32
//!                 element[n+l+ 64] = (ql[l]    >> 4)  | (qh[l] >> 4 & 0x03) << 4 - 32
//!                 element[n+l+ 96] = (ql[l+32] >> 4)  | (qh[l] >> 6 & 0x03) << 4 - 32
//!
//! **Parallelism strategy — port of llama.cpp `kernel_mul_mv_q6_K_f32_impl`:**
//!
//! Why this outperforms the previous all-lanes-per-superblock approach:
//!
//! 1. **Inter-superblock interleaving**: `ix = lane & 1` splits the 32 lanes into
//!    two groups that stride over alternate superblocks. Adjacent lanes read from
//!    different 210-byte regions simultaneously, letting the DRAM controller
//!    serve two banks in parallel instead of serialising on one.
//!
//! 2. **X preloading** (`yl[16]`): all 16 X loads are issued before the weight
//!    byte reads, hiding L2 latency behind the weight fetches. With
//!    `clang loop unroll(full)` the loop index is a compile-time constant, so
//!    yl[] entries are named registers with no private-memory spill.
//!
//! 3. **Deferred scaling** (`float4 sums`): accumulates unscaled dot products
//!    for 4 scale groups, then applies `d * sc[j]` once per group — 4× fewer
//!    scale multiplications vs the previous per-element approach.
//!
//! 4. **Reduced register pressure** (ROWS_PER_TG=4, 128 threads/TG):
//!    halves the per-TG register footprint vs the previous 256-thread design,
//!    allowing 2× more concurrent TGs and better latency hiding on LPDDR5X.

pub const SHADER: &str = r#"
constant uint Q6K_ROWS_PER_TG = 4;
constant uint Q6K_BLOCK_SIZE  = 210;

kernel void q6k_matvec(
    device const uchar*  W6K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q6K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    device const uchar* row = W6K + row_idx * bytes_per_row;

    // Lane decomposition (matches llama.cpp kernel_mul_mv_q6_K_f32_impl).
    // ix=0 lanes process superblocks 0,2,4,...; ix=1 lanes process 1,3,5,...
    // Adjacent lanes read from DIFFERENT superblock regions concurrently.
    const uint ix  = lane & 1u;       // 0 or 1
    const uint tid = lane >> 1u;      // 0..15: position within the group
    const uint ip  = tid >> 3u;       // 0 or 1: upper/lower 128-element half
    const uint il  = tid & 7u;        // 0..7: stride within the half
    const uint l0  = il << 2u;        // 0,4,8,...,28

    // Byte offsets within a superblock for this tid's assigned elements.
    const uint y_off   = (ip << 7u) + l0;       // X base: 0..28 or 128..156
    const uint q_off_l = (ip << 6u) + l0;       // lo4 base in ql[]: 0..28 or 64..92
    const uint q_off_h = (ip << 5u) + l0;       // hi2 base in qh[]: 0..28 or 32..60
    // Scale base: 8*ip + l0/16 = 8*ip + il/4
    const uint sc_base = (ip << 3u) + (il >> 2u);

    float acc = 0.0f;

    for (uint i = ix; i < superblocks; i += 2u) {
        device const uchar* block = row + i * Q6K_BLOCK_SIZE;
        device const uchar* q1    = block + q_off_l;        // lo4 for elements y_off+[0..3]
        device const uchar* q2    = block + q_off_l + 32u;  // lo4 for elements y_off+[32..35]
        device const uchar* qh    = block + 128u + q_off_h; // hi2 for all four groups
        device const char*  sc    = (device const char*)(block + 192u) + sc_base;
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
        float  d = decode_f16_metal(d_bits);

        // Preload 16 X values into registers BEFORE weight byte reads.
        // With clang loop unroll(full), l is a compile-time constant so
        // yl[] indices resolve statically — all 16 slots become registers.
        const uint xb = i * 256u + y_off;
        float yl[16];
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 4u; l++) {
            yl[4u*l + 0u] = X[xb + l      ];
            yl[4u*l + 1u] = X[xb + l + 32u];
            yl[4u*l + 2u] = X[xb + l + 64u];
            yl[4u*l + 3u] = X[xb + l + 96u];
        }

        // Accumulate unscaled dot products for 4 scale groups (one per l=0..3).
        // Each group covers 4 elements at offsets l, l+32, l+64, l+96 in the
        // superblock — the four GGUF Q6_K storage bands that share one qh byte.
        // char cast gives the signed 6-bit weight in [-32, +31].
        float4 sums = float4(0.0f);
        _Pragma("clang loop unroll(full)")
        for (uint l = 0u; l < 4u; l++) {
            uchar q1b = q1[l], q2b = q2[l], qhb = qh[l];
            sums[0] += yl[4u*l+0u] * float((char)((q1b & 0x0Fu) | ((qhb & 0x03u) << 4u)) - 32);
            sums[1] += yl[4u*l+1u] * float((char)((q2b & 0x0Fu) | ((qhb & 0x0Cu) << 2u)) - 32);
            sums[2] += yl[4u*l+2u] * float((char)((q1b >> 4u)   | ((qhb & 0x30u)       )) - 32);
            sums[3] += yl[4u*l+3u] * float((char)((q2b >> 4u)   | ((qhb & 0xC0u) >> 2u)) - 32);
        }

        // One scale multiply per 32-element group — 4× fewer than per-element.
        // sc[0,2,4,6] are the four group scales, accessed via sc_base offset.
        acc += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                  + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::metal::kernel::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q6k_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
