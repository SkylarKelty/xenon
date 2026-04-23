// Tensor-core attention for prefill.
//
// ---- File-scope status ----
// Stage 1: `xk_test_mma_bf16_kernel` — sanity check that
// mma.m16n8k16.row.col.f32.bf16.bf16.f32 compiles on sm_120a and that our
// per-thread fragment gather matches PTX docs.
//
// Stage 2: `xk_attn_flash_tc_bf16_kernel` — flash-attention-2 style tiled
// kernel. BR=16 Q tile, BC=16 K/V tile, one warp per block. O_acc / m / l
// live in smem (D=256 fits default 48 KiB; D=512 uses dynamic shmem via
// cudaFuncSetAttribute). Per-block owns (q_tile × q_head) with GQA via
// kv_head = q_head / gqa_group. Causal + sliding-window mask applied after
// each QK mma. Online softmax (m, l per row) keeps P in registers; P bf16
// goes straight into the PV mma's A operand without a smem round-trip.
// K/V reuse the same smem buffer across the tile iteration. Decode stays
// on split-KV (M=1 < MMA tile min of 16).
//
// ---- MMA fragment layout reference (row-major A, col-major B, row D) ----
// mma.m16n8k16 : D[16,8] = A[16,16] @ B[16,8],  fp32 accum, bf16 operands.
// Thread T in warp (0..31) holds:
//   A frag (4 × b32 = 8 bf16):
//     rb = T / 4,  cb = (T % 4) * 2
//     reg0 : A[rb  , cb+0] | A[rb  , cb+1]
//     reg1 : A[rb+8, cb+0] | A[rb+8, cb+1]
//     reg2 : A[rb  , cb+8] | A[rb  , cb+9]
//     reg3 : A[rb+8, cb+8] | A[rb+8, cb+9]
//   B frag (2 × b32 = 4 bf16), B stored col-major [k=16, n=8]:
//     n_col = T / 4,  r_row = (T % 4) * 2
//     reg0 : B[r_row+0, n_col] | B[r_row+1, n_col]
//     reg1 : B[r_row+8, n_col] | B[r_row+9, n_col]
//   D frag (4 × f32):
//     d0 = D[rb  , cb+0]    d1 = D[rb  , cb+1]
//     d2 = D[rb+8, cb+0]    d3 = D[rb+8, cb+1]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Raw mma.m16n8k16 bf16 → fp32 wrapper. Inputs are register-packed; caller
// does the fragment gather. Kept simple — no fancy layout-aware wrapper yet.
__device__ __forceinline__ void mma_m16n8k16_bf16_bf16_f32(
    uint32_t const a0, uint32_t const a1, uint32_t const a2, uint32_t const a3,
    uint32_t const b0, uint32_t const b1,
    float& d0, float& d1, float& d2, float& d3,
    float const c0, float const c1, float const c2, float const c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// Stage-1 sanity kernel. Computes a single 16×8 tile of D = A @ B with
// A[16,16] row-major and B[16,8] col-major (so B[k, n] lives at
// offset n*16 + k in memory). One warp (32 threads) per block; grid = 1.
__global__ void xk_test_mma_bf16_kernel(
    float* __restrict__ D,           // [16, 8] row-major, fp32
    const __nv_bfloat16* __restrict__ A,   // [16, 16] row-major, bf16
    const __nv_bfloat16* __restrict__ B)   // [16, 8]  col-major, bf16
{
    const int T = threadIdx.x;
    const int rb = T >> 2;           // 0..7
    const int cb = (T & 3) << 1;     // 0, 2, 4, 6

    // ---- Load A fragment (4 b32 = 8 bf16) ----
    // reg0: (rb  , cb+0 .. cb+1)
    // reg1: (rb+8, cb+0 .. cb+1)
    // reg2: (rb  , cb+8 .. cb+9)
    // reg3: (rb+8, cb+8 .. cb+9)
    uint32_t a0 = *reinterpret_cast<const uint32_t*>(&A[(rb + 0) * 16 + cb + 0]);
    uint32_t a1 = *reinterpret_cast<const uint32_t*>(&A[(rb + 8) * 16 + cb + 0]);
    uint32_t a2 = *reinterpret_cast<const uint32_t*>(&A[(rb + 0) * 16 + cb + 8]);
    uint32_t a3 = *reinterpret_cast<const uint32_t*>(&A[(rb + 8) * 16 + cb + 8]);

    // ---- Load B fragment (2 b32 = 4 bf16) ----
    // B is col-major [k=16, n=8]; column n_col starts at memory offset n_col*16.
    // reg0: (r_row+0..r_row+1, n_col)
    // reg1: (r_row+8..r_row+9, n_col)
    const int n_col = T >> 2;         // 0..7
    const int r_row = (T & 3) << 1;   // 0, 2, 4, 6
    uint32_t b0 = *reinterpret_cast<const uint32_t*>(&B[n_col * 16 + r_row + 0]);
    uint32_t b1 = *reinterpret_cast<const uint32_t*>(&B[n_col * 16 + r_row + 8]);

    // ---- Run MMA (zero-initial accumulator) ----
    float d0, d1, d2, d3;
    mma_m16n8k16_bf16_bf16_f32(a0, a1, a2, a3, b0, b1,
                                d0, d1, d2, d3,
                                0.f, 0.f, 0.f, 0.f);

    // ---- Store D fragment (4 fp32) ----
    D[(rb + 0) * 8 + cb + 0] = d0;
    D[(rb + 0) * 8 + cb + 1] = d1;
    D[(rb + 8) * 8 + cb + 0] = d2;
    D[(rb + 8) * 8 + cb + 1] = d3;
}

extern "C" int xk_test_mma_bf16(
    void* d, const void* a, const void* b, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    xk_test_mma_bf16_kernel<<<1, 32, 0, s>>>(
        (float*)d,
        (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}

// ============================================================================
// Stage 2: flash-attention-2 style kernel on tensor cores (mma.m16n8k16)
// ============================================================================

// Pack two bf16 values into a 32-bit register in (low=lo, high=hi) order
// (the convention used by mma bf16 operand fragments).
__device__ __forceinline__ uint32_t pack_bf16_pair(__nv_bfloat16 lo, __nv_bfloat16 hi)
{
    uint32_t l = (uint32_t)__bfloat16_as_ushort(lo);
    uint32_t h = (uint32_t)__bfloat16_as_ushort(hi);
    return (h << 16) | l;
}

// Warp-level max across the 4-thread group that shares rb = tid/4.
// Used to reduce row-local scores into a shared row-max.
__device__ __forceinline__ float warp_group4_max(float v)
{
    // XOR partners within lanes {4k..4k+3}: lane XOR 1, then XOR 2.
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
    return v;
}

__device__ __forceinline__ float warp_group4_sum(float v)
{
    v += __shfl_xor_sync(0xFFFFFFFF, v, 1);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 2);
    return v;
}

// Main kernel. BR=BC=16, one warp (32 threads) per block, grid = (ceil(T_q/BR), H).
//
// Smem layout (all figures for BR=BC=16):
//     q_tile   bf16[BR, D]           BR*D*2  bytes
//     kv_tile  bf16[BC, D]           BC*D*2  bytes    (K then V, reused)
//     o_acc    fp32[BR, D]           BR*D*4  bytes
//     m_row    fp32[BR]              BR*4    bytes
//     l_row    fp32[BR]              BR*4    bytes
// Total at D=256: 8192 + 8192 + 16384 + 64 + 64 = 32 896 bytes (fits 48 KiB default).
// Total at D=512: 16384 + 16384 + 32768 + 64 + 64 = 65 664 bytes (needs
// cudaFuncSetAttribute to opt into dynamic shmem beyond 48 KiB).
__global__ void xk_attn_flash_tc_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window, int gqa_group)
{
    constexpr int BR = 16;
    constexpr int BC = 16;

    const int q_tile_idx = blockIdx.x;
    const int q_head     = blockIdx.y;
    const int kv_head    = q_head / gqa_group;
    const int q_start    = q_tile_idx * BR;
    const int q_pos_base_tile = q_pos_base + q_start;

    const int tid = threadIdx.x;              // 0..31
    const int rb  = tid >> 2;                 // 0..7
    const int cb  = (tid & 3) << 1;           // 0,2,4,6

    // ---- Smem carve-up ----
    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sKV = sQ + BR * D;
    float*         sO  = reinterpret_cast<float*>(sKV + BC * D);
    float*         sM  = sO + BR * D;
    float*         sL  = sM + BR;

    // ---- Load Q tile [BR, D] as uint32 pairs (zero-pad rows past T_q) ----
    // 2 bf16 per load halves the iteration count.
    const int Q_PAIRS = (BR * D) / 2;
    for (int i = tid; i < Q_PAIRS; i += 32) {
        int r = (i * 2) / D;
        int c = (i * 2) - r * D;
        int q_row = q_start + r;
        uint32_t val;
        if (q_row < T_q) {
            val = *reinterpret_cast<const uint32_t*>(&q[((size_t)q_row * H + q_head) * D + c]);
        } else {
            val = 0;
        }
        reinterpret_cast<uint32_t*>(sQ)[i] = val;
    }

    // ---- Init O_acc = 0, m = -inf, l = 0 ----
    for (int i = tid; i < BR * D; i += 32) {
        sO[i] = 0.0f;
    }
    for (int r = tid; r < BR; r += 32) {
        sM[r] = -INFINITY;
        sL[r] = 0.0f;
    }
    __syncthreads();

    // ---- Outer loop over K/V tiles ----
    for (int kv_start = 0; kv_start < T_kv; kv_start += BC) {
        // ---- Load K tile (reuses sKV buffer), uint32-paired ----
        const int KV_PAIRS = (BC * D) / 2;
        for (int i = tid; i < KV_PAIRS; i += 32) {
            int r = (i * 2) / D;
            int c = (i * 2) - r * D;
            int kv_row = kv_start + r;
            uint32_t val;
            if (kv_row < T_kv) {
                val = *reinterpret_cast<const uint32_t*>(&k[((size_t)kv_row * H_kv + kv_head) * D + c]);
            } else {
                val = 0;
            }
            reinterpret_cast<uint32_t*>(sKV)[i] = val;
        }
        __syncthreads();

        // ---- Compute S = Q @ K^T, [BR, BC] (per-thread 8 fp32) ----
        // S per-thread positions (matching two back-to-back mma.m16n8k16 calls,
        // N-halves {0..7} and {8..15} of BC=16):
        //   s[0] = S[rb,    cb+0]    s[1] = S[rb,    cb+1]
        //   s[2] = S[rb+8,  cb+0]    s[3] = S[rb+8,  cb+1]
        //   s[4] = S[rb,    cb+8]    s[5] = S[rb,    cb+9]
        //   s[6] = S[rb+8,  cb+8]    s[7] = S[rb+8,  cb+9]
        float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        for (int k_iter = 0; k_iter < D; k_iter += 16) {
            // A frag from sQ (row-major). Two rows {rb, rb+8}, cols {cb+0..1, cb+8..9} within the k_iter slab.
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sQ[(rb  ) * D + k_iter + cb + 0]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sQ[(rb+8) * D + k_iter + cb + 0]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sQ[(rb  ) * D + k_iter + cb + 8]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sQ[(rb+8) * D + k_iter + cb + 8]);

            // N-half 0: K rows {0..7}, K_tile col = T/4 maps to N in [0, 8).
            {
                int n_col = tid >> 2;             // 0..7
                int r_row = (tid & 3) << 1;       // 0,2,4,6
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sKV[n_col * D + k_iter + r_row + 0]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sKV[n_col * D + k_iter + r_row + 8]);
                mma_m16n8k16_bf16_bf16_f32(a0, a1, a2, a3, b0, b1,
                                           s[0], s[1], s[2], s[3],
                                           s[0], s[1], s[2], s[3]);
            }
            // N-half 1: K rows {8..15}, K_tile col = 8 + T/4 maps to N in [8, 16).
            {
                int n_col = 8 + (tid >> 2);
                int r_row = (tid & 3) << 1;
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sKV[n_col * D + k_iter + r_row + 0]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sKV[n_col * D + k_iter + r_row + 8]);
                mma_m16n8k16_bf16_bf16_f32(a0, a1, a2, a3, b0, b1,
                                           s[4], s[5], s[6], s[7],
                                           s[4], s[5], s[6], s[7]);
            }
        }

        // ---- Scale + causal + sliding-window mask ----
        // Per-s-element (q_row_in_tile, kv_col_in_tile) positions:
        //   s[0] (rb,   cb+0)   s[1] (rb,   cb+1)
        //   s[2] (rb+8, cb+0)   s[3] (rb+8, cb+1)
        //   s[4] (rb,   cb+8)   s[5] (rb,   cb+9)
        //   s[6] (rb+8, cb+8)   s[7] (rb+8, cb+9)
        int q_rows[8] = { rb, rb, rb+8, rb+8, rb, rb, rb+8, rb+8 };
        int kv_cols[8] = { cb, cb+1, cb, cb+1, cb+8, cb+9, cb+8, cb+9 };
        for (int i = 0; i < 8; ++i) {
            int q_row = q_rows[i];
            int q_pos = q_pos_base_tile + q_row;
            int kv_abs = kv_start + kv_cols[i];
            int kv_min = window > 0 ? max(0, q_pos - window + 1) : 0;
            bool in_q  = (q_start + q_row) < T_q;
            bool in_kv = kv_abs < T_kv;
            bool causal = kv_abs <= q_pos;
            bool slide  = kv_abs >= kv_min;
            if (in_q && in_kv && causal && slide) {
                s[i] = s[i] * scale;
            } else {
                s[i] = -INFINITY;
            }
        }

        // ---- Online softmax: per-row max/sum via warp shuffle within 4-thread groups ----
        // Row rb: thread owns s[0], s[1], s[4], s[5]
        // Row rb+8: thread owns s[2], s[3], s[6], s[7]
        float row_max_a = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[4], s[5]));
        float row_max_b = fmaxf(fmaxf(s[2], s[3]), fmaxf(s[6], s[7]));
        row_max_a = warp_group4_max(row_max_a);
        row_max_b = warp_group4_max(row_max_b);

        // Previous m_r, l_r (all 4 threads in group see the same row so just read once).
        float m_old_a = sM[rb];
        float m_old_b = sM[rb + 8];
        float l_old_a = sL[rb];
        float l_old_b = sL[rb + 8];

        float m_new_a = fmaxf(m_old_a, row_max_a);
        float m_new_b = fmaxf(m_old_b, row_max_b);
        float alpha_a = __expf(m_old_a - m_new_a);
        float alpha_b = __expf(m_old_b - m_new_b);

        // Compute P = exp(s - m_new) in regs (stays fp32; bf16 at pack time).
        float p[8];
        p[0] = __expf(s[0] - m_new_a);
        p[1] = __expf(s[1] - m_new_a);
        p[2] = __expf(s[2] - m_new_b);
        p[3] = __expf(s[3] - m_new_b);
        p[4] = __expf(s[4] - m_new_a);
        p[5] = __expf(s[5] - m_new_a);
        p[6] = __expf(s[6] - m_new_b);
        p[7] = __expf(s[7] - m_new_b);

        // Row sums of P.
        float psum_a = warp_group4_sum(p[0] + p[1] + p[4] + p[5]);
        float psum_b = warp_group4_sum(p[2] + p[3] + p[6] + p[7]);

        float l_new_a = alpha_a * l_old_a + psum_a;
        float l_new_b = alpha_b * l_old_b + psum_b;

        // Write m_new, l_new back (one thread per row).
        if ((tid & 3) == 0) {
            sM[rb]     = m_new_a;
            sM[rb + 8] = m_new_b;
            sL[rb]     = l_new_a;
            sL[rb + 8] = l_new_b;
        }

        // (Rescale fused into the PV loop below — each mma's C operand is
        // loaded and scaled by alpha inline, saving a separate smem pass.)

        // ---- Load V tile into sKV (overwrites K), uint32-paired ----
        for (int i = tid; i < KV_PAIRS; i += 32) {
            int r = (i * 2) / D;
            int c = (i * 2) - r * D;
            int kv_row = kv_start + r;
            uint32_t val;
            if (kv_row < T_kv) {
                val = *reinterpret_cast<const uint32_t*>(&v[((size_t)kv_row * H_kv + kv_head) * D + c]);
            } else {
                val = 0;
            }
            reinterpret_cast<uint32_t*>(sKV)[i] = val;
        }
        __syncthreads();

        // ---- Build P bf16 A-frag once (positions match mma A-frag) ----
        __nv_bfloat16 pb[8];
        for (int i = 0; i < 8; ++i) pb[i] = __float2bfloat16(p[i]);
        uint32_t ap0 = pack_bf16_pair(pb[0], pb[1]);
        uint32_t ap1 = pack_bf16_pair(pb[2], pb[3]);
        uint32_t ap2 = pack_bf16_pair(pb[4], pb[5]);
        uint32_t ap3 = pack_bf16_pair(pb[6], pb[7]);

        // ---- PV mma: O += P @ V, one 16x8 tile per n-chunk, D/8 chunks ----
        for (int n_base = 0; n_base < D; n_base += 8) {
            // B frag from V: V[k, n] = sKV[k * D + n]. Need four bf16 at
            // (r_row+0, n_abs), (r_row+1, n_abs), (r_row+8, n_abs), (r_row+9, n_abs).
            int n_abs = n_base + (tid >> 2);
            int r_row = (tid & 3) << 1;
            __nv_bfloat16 v0 = sKV[(r_row + 0) * D + n_abs];
            __nv_bfloat16 v1 = sKV[(r_row + 1) * D + n_abs];
            __nv_bfloat16 v2 = sKV[(r_row + 8) * D + n_abs];
            __nv_bfloat16 v3 = sKV[(r_row + 9) * D + n_abs];
            uint32_t bv0 = pack_bf16_pair(v0, v1);
            uint32_t bv1 = pack_bf16_pair(v2, v3);

            // C frag: current O_acc scaled by alpha (fused rescale).
            // Rows rb / rb+8 use alpha_a / alpha_b respectively.
            float c0 = sO[(rb  ) * D + n_base + cb + 0] * alpha_a;
            float c1 = sO[(rb  ) * D + n_base + cb + 1] * alpha_a;
            float c2 = sO[(rb+8) * D + n_base + cb + 0] * alpha_b;
            float c3 = sO[(rb+8) * D + n_base + cb + 1] * alpha_b;

            float d0, d1, d2, d3;
            mma_m16n8k16_bf16_bf16_f32(ap0, ap1, ap2, ap3, bv0, bv1,
                                       d0, d1, d2, d3,
                                       c0, c1, c2, c3);

            sO[(rb  ) * D + n_base + cb + 0] = d0;
            sO[(rb  ) * D + n_base + cb + 1] = d1;
            sO[(rb+8) * D + n_base + cb + 0] = d2;
            sO[(rb+8) * D + n_base + cb + 1] = d3;
        }
        __syncthreads();
    }

    // ---- Final normalize: out = O_acc / l, write bf16 ----
    for (int i = tid; i < BR * D; i += 32) {
        int r = i / D;
        int c = i - r * D;
        int q_row = q_start + r;
        if (q_row >= T_q) continue;
        float inv_l = sL[r] > 0.0f ? 1.0f / sL[r] : 0.0f;
        out[((size_t)q_row * H + q_head) * D + c] = __float2bfloat16_rn(sO[i] * inv_l);
    }
}

extern "C" int xk_attn_flash_tc_bf16(
    void* out,
    const void* q,
    const void* k,
    const void* v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window,
    void* stream)
{
    if (T_q <= 0 || T_kv <= 0 || H <= 0 || H_kv <= 0 || D <= 0) return 0;
    if (H % H_kv != 0) return -(int)cudaErrorInvalidValue;
    if (D % 16 != 0) return -(int)cudaErrorInvalidValue;
    if ((D & 7) != 0) return -(int)cudaErrorInvalidValue; // PV mma n-chunk step
    const int gqa_group = H / H_kv;
    constexpr int BR = 16, BC = 16;
    const int q_tiles = (T_q + BR - 1) / BR;

    // Dynamic shmem sizing.
    const size_t shmem_bytes =
        (size_t)(BR * D) * sizeof(__nv_bfloat16)   // sQ
      + (size_t)(BC * D) * sizeof(__nv_bfloat16)   // sKV
      + (size_t)(BR * D) * sizeof(float)           // sO
      + (size_t)(BR)     * sizeof(float)           // sM
      + (size_t)(BR)     * sizeof(float);          // sL

    cudaStream_t s = (cudaStream_t)stream;

    // Allow > 48 KiB dynamic shmem per block (needed for D=512).
    static int shmem_opt_in = 0;
    if (!shmem_opt_in) {
        // Ceiling is device-specific (sm_120a typically ~99 KiB usable).
        cudaFuncSetAttribute(
            xk_attn_flash_tc_bf16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            98304);
        shmem_opt_in = 1;
    }

    dim3 grid(q_tiles, H);
    dim3 block(32);
    xk_attn_flash_tc_bf16_kernel<<<grid, block, shmem_bytes, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v,
        T_q, T_kv, H, H_kv, D, scale, q_pos_base, window, gqa_group);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
