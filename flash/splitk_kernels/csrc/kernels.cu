
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// ============================================================
// Decode attention (Lq=1, D=128): direct register streaming
// Each CTA = one (batch, head), warps split KV
//
// Decode에서는 K/V가 1번만 사용되어 shared memory tiling 이득 없음.
// 대신: vectorized half2 로드 → registers → dot + accumulate
// Shared memory는 split 간 reduction에만 사용.
// ============================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
}

// Vectorized load: 32 threads × half2×2 = 128 halves = 1 row of D=128
__device__ __forceinline__ void load_row(
    const half* __restrict__ row, int lane, float r[4]
) {
    const half2* p = reinterpret_cast<const half2*>(row + lane * 4);
    half2 h0 = p[0], h1 = p[1];
    r[0] = __half2float(h0.x); r[1] = __half2float(h0.y);
    r[2] = __half2float(h1.x); r[3] = __half2float(h1.y);
}

__device__ __forceinline__ void store_row(
    half* __restrict__ row, int lane, float r[4]
) {
    half2* p = reinterpret_cast<half2*>(row + lane * 4);
    p[0] = __halves2half2(__float2half(r[0]), __float2half(r[1]));
    p[1] = __halves2half2(__float2half(r[2]), __float2half(r[3]));
}

__device__ __forceinline__ float dot4(float a[4], float b[4]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}


// ── Standard split-K: online softmax + correction ──────────
__global__ void splitk_decode_standard(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_splits) return;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = warp_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;

    float q_reg[4];
    load_row(Q + bhd, lane, q_reg);
    #pragma unroll
    for (int e = 0; e < 4; e++) q_reg[e] *= sm_scale;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float m_i = -FLT_MAX, l_i = 0.f;

    // ── Phase 1: Stream K,V directly to registers ──
    for (int j = kv_start; j < kv_end; j++) {
        float k_reg[4];
        load_row(k_base + j * D, lane, k_reg);
        float qk = warp_reduce_sum(dot4(q_reg, k_reg));

        float m_new = fmaxf(m_i, qk);
        float alpha = __expf(m_i - m_new);
        float p = __expf(qk - m_new);

        float v_reg[4];
        load_row(v_base + j * D, lane, v_reg);

        l_i = l_i * alpha + p;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            acc[e] = acc[e] * alpha + p * v_reg[e];
        m_i = m_new;
    }

    // ── Store partial results to shared memory ──
    extern __shared__ float smem[];
    float* s_acc = smem;                               // [num_splits][D]
    float* s_m = smem + num_splits * D;                // [num_splits]
    float* s_l = smem + num_splits * D + num_splits;   // [num_splits]

    int base = warp_id * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) s_acc[base + e] = acc[e];
    if (lane == 0) { s_m[warp_id] = m_i; s_l[warp_id] = l_i; }
    __syncthreads();

    // ── Phase 2: Reduction with correction (warp 0) ──
    if (warp_id == 0) {
        float global_m = -FLT_MAX;
        for (int s = 0; s < num_splits; s++)
            global_m = fmaxf(global_m, s_m[s]);

        float global_l = 0.f;
        float fa[4] = {0.f, 0.f, 0.f, 0.f};
        int rd = lane * 4;
        for (int s = 0; s < num_splits; s++) {
            float corr = __expf(s_m[s] - global_m);
            global_l += s_l[s] * corr;
            #pragma unroll
            for (int e = 0; e < 4; e++)
                fa[e] += s_acc[s * D + rd + e] * corr;
        }
        float inv_l = 1.f / global_l;
        #pragma unroll
        for (int e = 0; e < 4; e++) fa[e] *= inv_l;
        store_row(Out + bhd, lane, fa);
    }
}


// ── Oracle split-K: global max, no correction ──────────────
__global__ void splitk_decode_oracle(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    const float* __restrict__ GlobalMax,
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_splits) return;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = warp_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;
    float m_global = GlobalMax[bh];

    float q_reg[4];
    load_row(Q + bhd, lane, q_reg);
    #pragma unroll
    for (int e = 0; e < 4; e++) q_reg[e] *= sm_scale;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float l_i = 0.f;

    // ── Phase 1: Global max, no online softmax ──
    for (int j = kv_start; j < kv_end; j++) {
        float k_reg[4];
        load_row(k_base + j * D, lane, k_reg);
        float qk = warp_reduce_sum(dot4(q_reg, k_reg));
        float p = __expf(qk - m_global);

        float v_reg[4];
        load_row(v_base + j * D, lane, v_reg);

        l_i += p;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            acc[e] += p * v_reg[e];
    }

    // ── Store to shared memory ──
    extern __shared__ float smem[];
    float* s_acc = smem;                     // [num_splits][D]
    float* s_l = smem + num_splits * D;      // [num_splits]

    int base = warp_id * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) s_acc[base + e] = acc[e];
    if (lane == 0) s_l[warp_id] = l_i;
    __syncthreads();

    // ── Phase 2: Simple sum ──
    if (warp_id == 0) {
        float global_l = 0.f;
        float fa[4] = {0.f, 0.f, 0.f, 0.f};
        int rd = lane * 4;
        for (int s = 0; s < num_splits; s++) {
            global_l += s_l[s];
            #pragma unroll
            for (int e = 0; e < 4; e++)
                fa[e] += s_acc[s * D + rd + e];
        }
        float inv_l = 1.f / global_l;
        #pragma unroll
        for (int e = 0; e < 4; e++) fa[e] *= inv_l;
        store_row(Out + bhd, lane, fa);
    }
}


// ── Profiled standard: clock64() time breakdown ────────────
// Timestamps per warp: q_load, k_load, qk_compute, v_load, pv_compute, sync, reduce
// Output: [B*H*num_splits, 7] int64 tensor (cycles)
__global__ void splitk_decode_profiled(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    long long* __restrict__ Timestamps,  // [B*H*num_splits, 7]
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_splits) return;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = warp_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;

    int ts_idx = bh * num_splits + warp_id;  // unique per warp

    // ── 0. Q load ──
    long long t0 = clock64();
    float q_reg[4];
    load_row(Q + bhd, lane, q_reg);
    #pragma unroll
    for (int e = 0; e < 4; e++) q_reg[e] *= sm_scale;
    long long t1 = clock64();

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float m_i = -FLT_MAX, l_i = 0.f;

    // ── 1-4. Main loop: K load, QK compute, V load, PV compute ──
    long long cyc_k_load = 0, cyc_qk = 0, cyc_v_load = 0, cyc_pv = 0;

    for (int j = kv_start; j < kv_end; j++) {
        // K load
        long long tk0 = clock64();
        float k_reg[4];
        load_row(k_base + j * D, lane, k_reg);
        long long tk1 = clock64();

        // QK compute (dot + softmax)
        float qk = warp_reduce_sum(dot4(q_reg, k_reg));
        float m_new = fmaxf(m_i, qk);
        float alpha = __expf(m_i - m_new);
        float p = __expf(qk - m_new);
        long long tk2 = clock64();

        // V load
        float v_reg[4];
        load_row(v_base + j * D, lane, v_reg);
        long long tk3 = clock64();

        // PV compute (accumulate)
        l_i = l_i * alpha + p;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            acc[e] = acc[e] * alpha + p * v_reg[e];
        m_i = m_new;
        long long tk4 = clock64();

        cyc_k_load += (tk1 - tk0);
        cyc_qk     += (tk2 - tk1);
        cyc_v_load += (tk3 - tk2);
        cyc_pv     += (tk4 - tk3);
    }

    // ── 5. Sync ──
    extern __shared__ float smem[];
    float* s_acc = smem;
    float* s_m = smem + num_splits * D;
    float* s_l = smem + num_splits * D + num_splits;

    int base = warp_id * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) s_acc[base + e] = acc[e];
    if (lane == 0) { s_m[warp_id] = m_i; s_l[warp_id] = l_i; }

    long long t_sync0 = clock64();
    __syncthreads();
    long long t_sync1 = clock64();

    // ── 6. Reduce ──
    long long t_red0 = clock64();
    if (warp_id == 0) {
        float global_m = -FLT_MAX;
        for (int s = 0; s < num_splits; s++)
            global_m = fmaxf(global_m, s_m[s]);
        float global_l = 0.f;
        float fa[4] = {0.f, 0.f, 0.f, 0.f};
        int rd = lane * 4;
        for (int s = 0; s < num_splits; s++) {
            float corr = __expf(s_m[s] - global_m);
            global_l += s_l[s] * corr;
            #pragma unroll
            for (int e = 0; e < 4; e++)
                fa[e] += s_acc[s * D + rd + e] * corr;
        }
        float inv_l = 1.f / global_l;
        #pragma unroll
        for (int e = 0; e < 4; e++) fa[e] *= inv_l;
        store_row(Out + bhd, lane, fa);
    }
    long long t_red1 = clock64();

    // ── Write timestamps (lane 0 only to avoid contention) ──
    if (lane == 0) {
        Timestamps[ts_idx * 7 + 0] = t1 - t0;           // Q load
        Timestamps[ts_idx * 7 + 1] = cyc_k_load;         // K load (total)
        Timestamps[ts_idx * 7 + 2] = cyc_qk;             // QK compute (total)
        Timestamps[ts_idx * 7 + 3] = cyc_v_load;          // V load (total)
        Timestamps[ts_idx * 7 + 4] = cyc_pv;             // PV compute (total)
        Timestamps[ts_idx * 7 + 5] = t_sync1 - t_sync0;  // __syncthreads__
        Timestamps[ts_idx * 7 + 6] = t_red1 - t_red0;    // Reduction
    }
}


// ── Python binding for profiled kernel ──
std::vector<torch::Tensor> launch_profiled(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                            float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    auto Out = torch::empty_like(Q);
    auto Timestamps = torch::zeros({B * H * num_splits, 7},
                                    torch::TensorOptions().dtype(torch::kInt64).device(Q.device()));

    int threads = num_splits * 32;
    int smem = (num_splits * D + num_splits * 2) * sizeof(float);
    splitk_decode_profiled<<<B * H, threads, smem>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)Out.data_ptr(),
        (long long*)Timestamps.data_ptr<int64_t>(),
        sm_scale, B, H, Lk, D, num_splits);

    return {Out, Timestamps};
}


// ── Profiled oracle: clock64() time breakdown ──────────────
__global__ void splitk_decode_profiled_oracle(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    const float* __restrict__ GlobalMax,
    long long* __restrict__ Timestamps,  // [B*H*num_splits, 7]
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_splits) return;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = warp_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;
    float m_global = GlobalMax[bh];

    int ts_idx = bh * num_splits + warp_id;

    // ── 0. Q load ──
    long long t0 = clock64();
    float q_reg[4];
    load_row(Q + bhd, lane, q_reg);
    #pragma unroll
    for (int e = 0; e < 4; e++) q_reg[e] *= sm_scale;
    long long t1 = clock64();

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float l_i = 0.f;

    long long cyc_k_load = 0, cyc_qk = 0, cyc_v_load = 0, cyc_pv = 0;

    for (int j = kv_start; j < kv_end; j++) {
        // K load
        long long tk0 = clock64();
        float k_reg[4];
        load_row(k_base + j * D, lane, k_reg);
        long long tk1 = clock64();

        // QK compute (dot + exp with global max — no online softmax)
        float qk = warp_reduce_sum(dot4(q_reg, k_reg));
        float p = __expf(qk - m_global);
        long long tk2 = clock64();

        // V load
        float v_reg[4];
        load_row(v_base + j * D, lane, v_reg);
        long long tk3 = clock64();

        // PV compute (simple accumulate — no alpha rescaling)
        l_i += p;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            acc[e] += p * v_reg[e];
        long long tk4 = clock64();

        cyc_k_load += (tk1 - tk0);
        cyc_qk     += (tk2 - tk1);
        cyc_v_load += (tk3 - tk2);
        cyc_pv     += (tk4 - tk3);
    }

    // ── 5. Sync ──
    extern __shared__ float smem[];
    float* s_acc = smem;
    float* s_l = smem + num_splits * D;

    int base = warp_id * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) s_acc[base + e] = acc[e];
    if (lane == 0) s_l[warp_id] = l_i;

    long long t_sync0 = clock64();
    __syncthreads();
    long long t_sync1 = clock64();

    // ── 6. Reduce (simple sum, no correction) ──
    long long t_red0 = clock64();
    if (warp_id == 0) {
        float global_l = 0.f;
        float fa[4] = {0.f, 0.f, 0.f, 0.f};
        int rd = lane * 4;
        for (int s = 0; s < num_splits; s++) {
            global_l += s_l[s];
            #pragma unroll
            for (int e = 0; e < 4; e++)
                fa[e] += s_acc[s * D + rd + e];
        }
        float inv_l = 1.f / global_l;
        #pragma unroll
        for (int e = 0; e < 4; e++) fa[e] *= inv_l;
        store_row(Out + bhd, lane, fa);
    }
    long long t_red1 = clock64();

    if (lane == 0) {
        Timestamps[ts_idx * 7 + 0] = t1 - t0;
        Timestamps[ts_idx * 7 + 1] = cyc_k_load;
        Timestamps[ts_idx * 7 + 2] = cyc_qk;
        Timestamps[ts_idx * 7 + 3] = cyc_v_load;
        Timestamps[ts_idx * 7 + 4] = cyc_pv;
        Timestamps[ts_idx * 7 + 5] = t_sync1 - t_sync0;
        Timestamps[ts_idx * 7 + 6] = t_red1 - t_red0;
    }
}


std::vector<torch::Tensor> launch_profiled_oracle(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor GlobalMax, float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    auto Out = torch::empty_like(Q);
    auto Timestamps = torch::zeros({B * H * num_splits, 7},
                                    torch::TensorOptions().dtype(torch::kInt64).device(Q.device()));

    int threads = num_splits * 32;
    int smem = (num_splits * D + num_splits) * sizeof(float);
    splitk_decode_profiled_oracle<<<B * H, threads, smem>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)Out.data_ptr(),
        (const float*)GlobalMax.data_ptr(),
        (long long*)Timestamps.data_ptr<int64_t>(),
        sm_scale, B, H, Lk, D, num_splits);

    return {Out, Timestamps};
}


// ── Local-max 2-pass + async reduction (2-kernel) ───────────
// Phase 1: each CTA = 1 warp, handles one (bh, split_id).
//   Pass 1: stream K → QK dot products → find local max.
//   Pass 2: re-stream K,V → exp(qk - local_max) * V, accumulate.
//   Write partial (acc[D], m, l) to global buffer.
// Phase 2: lightweight reduce kernel, one CTA = 1 warp per (bh).
//   Cross-split correction + normalize → final output.
// Splits are fully independent CTAs — no __syncthreads__ across them.
// Cost: 2× K memory traffic. Benefit: no inter-split sync.
__global__ void splitk_phase1_localmax(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ PartialAcc,  // [B*H*num_splits, D]
    float* __restrict__ PartialM,    // [B*H*num_splits]
    float* __restrict__ PartialL,    // [B*H*num_splits]
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int idx = blockIdx.x;              // 0 .. B*H*num_splits - 1
    int bh = idx / num_splits;
    int split_id = idx % num_splits;
    int b = bh / H, h = bh % H;
    int lane = threadIdx.x;            // 0..31

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = split_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;

    float q_reg[4];
    load_row(Q + bhd, lane, q_reg);
    #pragma unroll
    for (int e = 0; e < 4; e++) q_reg[e] *= sm_scale;

    // ── Pass 1: find local max ──
    float m_i = -FLT_MAX;
    for (int j = kv_start; j < kv_end; j++) {
        float k_reg[4];
        load_row(k_base + j * D, lane, k_reg);
        float qk = warp_reduce_sum(dot4(q_reg, k_reg));
        m_i = fmaxf(m_i, qk);
    }

    // ── Pass 2: softmax with local max + PV accumulation ──
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float l_i = 0.f;
    for (int j = kv_start; j < kv_end; j++) {
        float k_reg[4];
        load_row(k_base + j * D, lane, k_reg);
        float qk = warp_reduce_sum(dot4(q_reg, k_reg));
        float p = __expf(qk - m_i);

        float v_reg[4];
        load_row(v_base + j * D, lane, v_reg);

        l_i += p;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            acc[e] += p * v_reg[e];
    }

    // ── Write partials to global memory ──
    int out_base = idx * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++)
        PartialAcc[out_base + e] = acc[e];
    if (lane == 0) {
        PartialM[idx] = m_i;
        PartialL[idx] = l_i;
    }
}

__global__ void splitk_reduce(
    const float* __restrict__ PartialAcc,  // [B*H*num_splits, D]
    const float* __restrict__ PartialM,    // [B*H*num_splits]
    const float* __restrict__ PartialL,    // [B*H*num_splits]
    half* __restrict__ Out,                // [B, H, 1, D]
    int B, int H, int D, int num_splits
) {
    int bh = blockIdx.x;  // 0 .. B*H - 1
    int lane = threadIdx.x;

    // Find global max across splits
    float global_m = -FLT_MAX;
    for (int s = 0; s < num_splits; s++)
        global_m = fmaxf(global_m, PartialM[bh * num_splits + s]);

    // Corrected reduction
    float global_l = 0.f;
    float fa[4] = {0.f, 0.f, 0.f, 0.f};
    int rd = lane * 4;
    for (int s = 0; s < num_splits; s++) {
        float corr = __expf(PartialM[bh * num_splits + s] - global_m);
        global_l += PartialL[bh * num_splits + s] * corr;
        int base = (bh * num_splits + s) * D + rd;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            fa[e] += PartialAcc[base + e] * corr;
    }
    float inv_l = 1.f / global_l;
    #pragma unroll
    for (int e = 0; e < 4; e++) fa[e] *= inv_l;
    store_row(Out + bh * D, lane, fa);
}


/* ── Oracle async v1 (이전 버전, per-key loop) ────────────────
// 문제: 매 key마다 load→stall→compute→load→stall. Memory pipeline 활용 안 됨.
__global__ void splitk_phase1_oracle_async_v1(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    float* __restrict__ PartialAcc, float* __restrict__ PartialL,
    const float* __restrict__ GlobalMax, float sm_scale,
    int B, int H, int Lk, int D, int num_splits
) {
    // ... per-key loop: load K[j] → dot → exp → load V[j] → acc ...
    // warp_reduce_sum per key = 5 shuffles × Lk
}
── End v1 ─────────────────────────────────────────────────── */


// ── Oracle async v2: multi-key unrolled (memory pipeline) ───
// UNROLL_K keys를 동시에 load하여 memory pipeline을 채움.
// load K[j]~K[j+U-1]을 한번에 issue → 첫 key 도착 시 compute 시작,
// 그 사이 나머지 key들이 memory pipeline에서 진행 중.
// warp_reduce_sum도 UNROLL_K개를 연속 처리하여 shuffle 비용 분산.
//
// vs v1: 같은 total bytes, 같은 FLOPs, 하지만 memory latency hiding 향상.
#define UNROLL_K 4  // keys per inner iteration (2, 4, 8 tunable)

__global__ void splitk_phase1_oracle_async(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ PartialAcc,  // [B*H*num_splits, D]
    float* __restrict__ PartialL,    // [B*H*num_splits]
    const float* __restrict__ GlobalMax,  // [B*H]
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int idx = blockIdx.x;
    int bh = idx / num_splits;
    int split_id = idx % num_splits;
    int b = bh / H, h = bh % H;
    int lane = threadIdx.x;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = split_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;
    float m_global = GlobalMax[bh];

    float q_reg[4];
    load_row(Q + bhd, lane, q_reg);
    #pragma unroll
    for (int e = 0; e < 4; e++) q_reg[e] *= sm_scale;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float l_i = 0.f;

    // ── Unrolled main loop: UNROLL_K keys per iteration ──
    int j = kv_start;
    for (; j + UNROLL_K <= kv_end; j += UNROLL_K) {
        // Load UNROLL_K K rows simultaneously → fill memory pipeline
        float k0[4], k1[4], k2[4], k3[4];
        load_row(k_base + (j + 0) * D, lane, k0);
        load_row(k_base + (j + 1) * D, lane, k1);
        load_row(k_base + (j + 2) * D, lane, k2);
        load_row(k_base + (j + 3) * D, lane, k3);

        // Compute UNROLL_K dot products + exp
        float qk0 = warp_reduce_sum(dot4(q_reg, k0));
        float qk1 = warp_reduce_sum(dot4(q_reg, k1));
        float qk2 = warp_reduce_sum(dot4(q_reg, k2));
        float qk3 = warp_reduce_sum(dot4(q_reg, k3));

        float p0 = __expf(qk0 - m_global);
        float p1 = __expf(qk1 - m_global);
        float p2 = __expf(qk2 - m_global);
        float p3 = __expf(qk3 - m_global);

        // Load UNROLL_K V rows simultaneously
        float v0[4], v1[4], v2[4], v3[4];
        load_row(v_base + (j + 0) * D, lane, v0);
        load_row(v_base + (j + 1) * D, lane, v1);
        load_row(v_base + (j + 2) * D, lane, v2);
        load_row(v_base + (j + 3) * D, lane, v3);

        // Accumulate
        l_i += p0 + p1 + p2 + p3;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[e] += p0 * v0[e] + p1 * v1[e] + p2 * v2[e] + p3 * v3[e];
        }
    }

    // ── Remainder: per-key ──
    for (; j < kv_end; j++) {
        float k_reg[4];
        load_row(k_base + j * D, lane, k_reg);
        float qk = warp_reduce_sum(dot4(q_reg, k_reg));
        float p = __expf(qk - m_global);

        float v_reg[4];
        load_row(v_base + j * D, lane, v_reg);

        l_i += p;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            acc[e] += p * v_reg[e];
    }

    // ── Write partials ──
    int out_base = idx * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++)
        PartialAcc[out_base + e] = acc[e];
    if (lane == 0) {
        PartialL[idx] = l_i;
    }
}

// Simple sum reduce (no correction needed — all splits use same φ)
__global__ void splitk_reduce_simple(
    const float* __restrict__ PartialAcc,  // [B*H*num_splits, D]
    const float* __restrict__ PartialL,    // [B*H*num_splits]
    half* __restrict__ Out,                // [B, H, 1, D]
    int B, int H, int D, int num_splits
) {
    int bh = blockIdx.x;
    int lane = threadIdx.x;

    float global_l = 0.f;
    float fa[4] = {0.f, 0.f, 0.f, 0.f};
    int rd = lane * 4;
    for (int s = 0; s < num_splits; s++) {
        global_l += PartialL[bh * num_splits + s];
        int base = (bh * num_splits + s) * D + rd;
        #pragma unroll
        for (int e = 0; e < 4; e++)
            fa[e] += PartialAcc[base + e];
    }
    float inv_l = 1.f / global_l;
    #pragma unroll
    for (int e = 0; e < 4; e++) fa[e] *= inv_l;
    store_row(Out + bh * D, lane, fa);
}


// ── Python bindings ──
torch::Tensor launch_oracle_async(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                   torch::Tensor GlobalMax, float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    int BH = B * H;
    auto Out = torch::empty_like(Q);

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto partial_acc = torch::empty({BH * num_splits, D}, opts_f);
    auto partial_l = torch::empty({BH * num_splits}, opts_f);

    // Phase 1: 1-pass with global max, B*H*num_splits CTAs
    splitk_phase1_oracle_async<<<BH * num_splits, 32>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(),
        partial_acc.data_ptr<float>(), partial_l.data_ptr<float>(),
        (const float*)GlobalMax.data_ptr(),
        sm_scale, B, H, Lk, D, num_splits);

    // Phase 2: simple sum (no correction)
    splitk_reduce_simple<<<BH, 32>>>(
        partial_acc.data_ptr<float>(), partial_l.data_ptr<float>(),
        (half*)Out.data_ptr(), B, H, D, num_splits);

    return Out;
}


// ── TC Oracle Async: multi-CTA + QK wmma + scalar PV ───────
// Same multi-CTA structure as oracle_async (1 warp per CTA per split)
// but uses Tensor Core (wmma 16×16×16) for QK dot product.
// PV stays scalar (decode is memory-bound, PV wmma has smem issues).
//
// Each CTA:
//   1. Load Q into smem (padded 16×D, shared across tiles)
//   2. For each 16-key tile:
//      a. Load K tile (16×D) into smem
//      b. QK via wmma: C(16×16) = Q(16×D) × K^T(D×16)
//      c. Extract 16 scores from row 0
//      d. exp(qk - φ) → scalar PV accumulate from global V
//   3. Write partial acc, l to global memory
// Phase 2: splitk_reduce_simple (same as oracle_async)

#include <mma.h>
using namespace nvcuda;

constexpr int TC_TILE_ASYNC = 16;  // keys per wmma tile

__global__ void splitk_phase1_tc_oracle_async(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ PartialAcc,
    float* __restrict__ PartialL,
    const float* __restrict__ GlobalMax,
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int idx = blockIdx.x;
    int bh = idx / num_splits;
    int split_id = idx % num_splits;
    int b = bh / H, h = bh % H;
    int lane = threadIdx.x;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = split_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;
    float m_global = GlobalMax[bh];

    // Shared memory: Q padded (16×D half) + K tile (16×D half)
    extern __shared__ char raw_smem[];
    half* q_smem = (half*)raw_smem;                         // 16 × D
    half* k_tile = q_smem + 16 * D;                         // 16 × D

    // Load Q into smem (row 0 = Q*sm_scale, rows 1-15 = 0)
    for (int i = lane; i < 16 * D; i += 32) {
        int row = i / D, col = i % D;
        if (row == 0) {
            float qv = __half2float(Q[bhd + col]) * sm_scale;
            q_smem[i] = __float2half(qv);
        } else {
            q_smem[i] = __float2half(0.0f);
        }
    }
    __syncwarp();

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float l_i = 0.f;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    for (int j = kv_start; j < kv_end; j += TC_TILE_ASYNC) {
        int tile_len = min(TC_TILE_ASYNC, kv_end - j);

        // Load K tile to smem
        for (int t = 0; t < tile_len; t++) {
            const half2* src = (const half2*)(k_base + (j + t) * D + lane * 4);
            half2* dst = (half2*)(k_tile + t * D + lane * 4);
            dst[0] = src[0]; dst[1] = src[1];
        }
        for (int t = tile_len; t < TC_TILE_ASYNC; t++) {
            half2* dst = (half2*)(k_tile + t * D + lane * 4);
            dst[0] = {0, 0}; dst[1] = {0, 0};
        }
        __syncwarp();

        // QK via wmma: C(16×16) = Q(16×D) × K^T(D×16)
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int d = 0; d < D; d += 16) {
            wmma::load_matrix_sync(a_frag, q_smem + d, D);
            wmma::load_matrix_sync(b_frag, k_tile + d, D);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store QK to smem (reuse k_tile as float buffer)
        float* qk_buf = (float*)k_tile;
        wmma::store_matrix_sync(qk_buf, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // Scalar PV with global max
        for (int t = 0; t < tile_len; t++) {
            float qk = qk_buf[t];  // row 0, col t
            float p = __expf(qk - m_global);

            const half2* vp = (const half2*)(v_base + (j + t) * D + lane * 4);
            half2 vh0 = vp[0], vh1 = vp[1];

            l_i += p;
            acc[0] += p * __half2float(vh0.x);
            acc[1] += p * __half2float(vh0.y);
            acc[2] += p * __half2float(vh1.x);
            acc[3] += p * __half2float(vh1.y);
        }
    }

    // Write partials
    int out_base = idx * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++)
        PartialAcc[out_base + e] = acc[e];
    if (lane == 0)
        PartialL[idx] = l_i;
}


torch::Tensor launch_tc_oracle_async(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                      torch::Tensor GlobalMax, float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    int BH = B * H;
    auto Out = torch::empty_like(Q);

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto partial_acc = torch::empty({BH * num_splits, D}, opts_f);
    auto partial_l = torch::empty({BH * num_splits}, opts_f);

    // smem: Q (16×D half) + K tile (16×D half)
    int smem = 2 * 16 * D * sizeof(half);

    splitk_phase1_tc_oracle_async<<<BH * num_splits, 32, smem>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(),
        partial_acc.data_ptr<float>(), partial_l.data_ptr<float>(),
        (const float*)GlobalMax.data_ptr(),
        sm_scale, B, H, Lk, D, num_splits);

    splitk_reduce_simple<<<BH, 32>>>(
        partial_acc.data_ptr<float>(), partial_l.data_ptr<float>(),
        (half*)Out.data_ptr(), B, H, D, num_splits);

    return Out;
}


torch::Tensor launch_standard(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                               float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    auto Out = torch::empty_like(Q);

    int threads = num_splits * 32;
    int smem = (num_splits * D + num_splits * 2) * sizeof(float);
    splitk_decode_standard<<<B * H, threads, smem>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)Out.data_ptr(),
        sm_scale, B, H, Lk, D, num_splits);
    return Out;
}

torch::Tensor launch_oracle(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                              torch::Tensor GlobalMax, float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    auto Out = torch::empty_like(Q);

    int threads = num_splits * 32;
    int smem = (num_splits * D + num_splits) * sizeof(float);
    splitk_decode_oracle<<<B * H, threads, smem>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)Out.data_ptr(),
        (const float*)GlobalMax.data_ptr(),
        sm_scale, B, H, Lk, D, num_splits);
    return Out;
}

torch::Tensor launch_localmax_async(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                     float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    int BH = B * H;
    auto Out = torch::empty_like(Q);

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto partial_acc = torch::empty({BH * num_splits, D}, opts_f);
    auto partial_m = torch::empty({BH * num_splits}, opts_f);
    auto partial_l = torch::empty({BH * num_splits}, opts_f);

    // Phase 1: B*H*num_splits CTAs, each = 1 warp (32 threads)
    splitk_phase1_localmax<<<BH * num_splits, 32>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(),
        partial_acc.data_ptr<float>(), partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        sm_scale, B, H, Lk, D, num_splits);

    // Phase 2: B*H CTAs, each = 1 warp (32 threads)
    splitk_reduce<<<BH, 32>>>(
        partial_acc.data_ptr<float>(), partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        (half*)Out.data_ptr(), B, H, D, num_splits);

    return Out;
}

/* ── Tensor Core versions (보류) ──────────────────────────────
// ================================================================
// Tensor Core versions: QK via wmma (16 keys per tile)
// PV stays scalar (decode is memory-bound, TC gains limited)
//
// wmma computes: C(16×16) = Q_pad(16×D) × K_tile^T(D×16)
//   → row 0 of C = 16 dot products dot(Q, K[j+0..j+15])
//   → D=128 split into 8 chunks of 16: 8 wmma::mma_sync per tile
// ================================================================

#include <mma.h>
using namespace nvcuda;

constexpr int TC_TILE = 16;  // keys per wmma tile

// ── Helper: build P matrix in smem (16×16, row 0 = weights, rest = 0) ──
// p_vals: array of 16 softmax weights (broadcast via smem from qk_buf)
// p_smem: 16×16 half matrix in shared memory
__device__ __forceinline__ void build_p_matrix(
    float* __restrict__ qk_buf, int tile_len,
    half* __restrict__ p_smem, int lane,
    float m_ref, float* __restrict__ p_out,  // output: p values for l_i update
    bool use_online_max, float* __restrict__ m_i_ptr, float* __restrict__ l_i_ptr
) {
    // Zero entire P matrix (16×16 = 256 halves)
    // 32 threads × 8 halves = 256
    half2* p2 = (half2*)(p_smem);
    for (int i = lane; i < 128; i += 32)
        p2[i] = {__float2half(0.0f), __float2half(0.0f)};
    __syncwarp();

    // Fill row 0 with softmax weights
    // 16 elements in row 0, thread 0-15 each write 1, or we do it cooperatively
    if (use_online_max) {
        // Online softmax: compute running max + rescale
        float m_i = *m_i_ptr;
        float l_i = *l_i_ptr;
        // First pass: find tile max
        float tile_max = -FLT_MAX;
        for (int t = 0; t < tile_len; t++)
            tile_max = fmaxf(tile_max, qk_buf[t]);
        float m_new = fmaxf(m_i, tile_max);
        float alpha = __expf(m_i - m_new);
        l_i *= alpha;
        // Write P row 0 and accumulate l_i
        for (int t = lane; t < 16; t += 32) {
            float p = (t < tile_len) ? __expf(qk_buf[t] - m_new) : 0.f;
            p_smem[t] = __float2half(p);  // row 0, col t
            p_out[t] = p;
            if (t < tile_len) l_i += p;
        }
        // Broadcast l_i, m_new across warp (lane 0 has accumulated l_i)
        // Actually all lanes computed the same values since qk_buf is in smem
        *m_i_ptr = m_new;
        *l_i_ptr = l_i;
    } else {
        // Oracle: fixed global max
        float l_i = *l_i_ptr;
        for (int t = lane; t < 16; t += 32) {
            float p = (t < tile_len) ? __expf(qk_buf[t] - m_ref) : 0.f;
            p_smem[t] = __float2half(p);
            p_out[t] = p;
            if (t < tile_len) l_i += p;
        }
        *l_i_ptr = l_i;
    }
    __syncwarp();
}


// ── TC Standard: QK wmma + scalar PV + online softmax ──────
__global__ void splitk_decode_tc_standard(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_splits) return;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = warp_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;

    // ── Shared memory layout (per warp) ──
    // [16][D]       q_smem     (shared by all warps)
    // [ns][16][D]   kv_tiles   (K tile, then reused for QK buf + P matrix + V tile)
    // [ns][D]       s_acc      (reduction)
    // [ns]          s_m, s_l   (reduction)
    extern __shared__ char raw_smem[];
    half* q_smem = (half*)raw_smem;
    half* kv_tiles = q_smem + 16 * D;
    half* my_tile = kv_tiles + warp_id * TC_TILE * D;  // 16×128 halves per warp
    float* s_acc = (float*)(kv_tiles + num_splits * TC_TILE * D);
    float* s_m = s_acc + num_splits * D;
    float* s_l = s_m + num_splits;

    // Load Q into smem (row 0 = Q*sm_scale, rows 1-15 = 0)
    for (int i = threadIdx.x; i < 16 * D; i += blockDim.x) {
        int row = i / D, col = i % D;
        q_smem[i] = (row == 0) ? __float2half(__half2float(Q[bhd + col]) * sm_scale)
                                : __float2half(0.0f);
    }
    __syncthreads();

    // Output accumulator (float, in registers — will be rescaled by online softmax)
    // Stored as wmma accumulator fragments across D chunks
    // For simplicity: use float registers, convert to/from wmma at PV step
    float acc_reg[4] = {0.f, 0.f, 0.f, 0.f};  // 4 elements per thread × 32 threads = 128 = D
    float m_i = -FLT_MAX, l_i = 0.f;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    for (int j = kv_start; j < kv_end; j += TC_TILE) {
        int tile_len = min(TC_TILE, kv_end - j);

        // ── Load K tile into my_tile (16×D half) ──
        for (int t = 0; t < tile_len; t++) {
            const half2* src = (const half2*)(k_base + (j + t) * D + lane * 4);
            half2* dst = (half2*)(my_tile + t * D + lane * 4);
            dst[0] = src[0]; dst[1] = src[1];
        }
        for (int t = tile_len; t < TC_TILE; t++) {
            half2* dst = (half2*)(my_tile + t * D + lane * 4);
            dst[0] = {0, 0}; dst[1] = {0, 0};
        }
        __syncwarp();

        // ── QK via wmma: scores(16×16) = Q(16×D) × K^T(D×16) ──
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int d = 0; d < D; d += 16) {
            wmma::load_matrix_sync(a_frag, q_smem + d, D);
            wmma::load_matrix_sync(b_frag, my_tile + d, D);  // col_major → K^T
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store QK to smem, reusing my_tile as float buffer
        float* qk_buf = (float*)my_tile;  // 16×16 floats
        wmma::store_matrix_sync(qk_buf, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // ── Online softmax + scalar PV accumulation ──
        for (int t = 0; t < tile_len; t++) {
            float qk = qk_buf[t];  // row 0, col t

            float m_new = fmaxf(m_i, qk);
            float alpha = __expf(m_i - m_new);
            float p = __expf(qk - m_new);

            // V load (vectorized, direct to registers)
            const half2* vp = (const half2*)(v_base + (j + t) * D + lane * 4);
            half2 vh0 = vp[0], vh1 = vp[1];

            l_i = l_i * alpha + p;
            acc_reg[0] = acc_reg[0] * alpha + p * __half2float(vh0.x);
            acc_reg[1] = acc_reg[1] * alpha + p * __half2float(vh0.y);
            acc_reg[2] = acc_reg[2] * alpha + p * __half2float(vh1.x);
            acc_reg[3] = acc_reg[3] * alpha + p * __half2float(vh1.y);
            m_i = m_new;
        }
    }

    // ── Store partial & reduce ──
    int wr = warp_id * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) s_acc[wr + e] = acc_reg[e];
    if (lane == 0) { s_m[warp_id] = m_i; s_l[warp_id] = l_i; }
    __syncthreads();

    if (warp_id == 0) {
        float global_m = -FLT_MAX;
        for (int s = 0; s < num_splits; s++)
            global_m = fmaxf(global_m, s_m[s]);
        float global_l = 0.f;
        float fa[4] = {0,0,0,0};
        int rd = lane * 4;
        for (int s = 0; s < num_splits; s++) {
            float corr = __expf(s_m[s] - global_m);
            global_l += s_l[s] * corr;
            #pragma unroll
            for (int e = 0; e < 4; e++)
                fa[e] += s_acc[s * D + rd + e] * corr;
        }
        float inv_l = 1.f / global_l;
        #pragma unroll
        for (int e = 0; e < 4; e++) fa[e] *= inv_l;
        store_row(Out + bhd, lane, fa);
    }
}


// ── TC Oracle: QK wmma + PV wmma, global max, no correction ──
__global__ void splitk_decode_tc_oracle(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    const float* __restrict__ GlobalMax,
    float sm_scale,
    int B, int H, int Lk, int D,
    int num_splits
) {
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_splits) return;

    int split_size = (Lk + num_splits - 1) / num_splits;
    int kv_start = warp_id * split_size;
    int kv_end = min(kv_start + split_size, Lk);

    int bhd = (b * H + h) * D;
    const half* k_base = K + (b * H + h) * Lk * D;
    const half* v_base = V + (b * H + h) * Lk * D;
    float m_global = GlobalMax[bh];

    extern __shared__ char raw_smem[];
    half* q_smem = (half*)raw_smem;
    half* kv_tiles = q_smem + 16 * D;
    half* my_tile = kv_tiles + warp_id * TC_TILE * D;
    float* s_acc = (float*)(kv_tiles + num_splits * TC_TILE * D);
    float* s_l = s_acc + num_splits * D;

    for (int i = threadIdx.x; i < 16 * D; i += blockDim.x) {
        int row = i / D, col = i % D;
        q_smem[i] = (row == 0) ? __float2half(__half2float(Q[bhd + col]) * sm_scale)
                                : __float2half(0.0f);
    }
    __syncthreads();

    float acc_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float l_i = 0.f;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    for (int j = kv_start; j < kv_end; j += TC_TILE) {
        int tile_len = min(TC_TILE, kv_end - j);

        // Load K tile
        for (int t = 0; t < tile_len; t++) {
            const half2* src = (const half2*)(k_base + (j + t) * D + lane * 4);
            half2* dst = (half2*)(my_tile + t * D + lane * 4);
            dst[0] = src[0]; dst[1] = src[1];
        }
        for (int t = tile_len; t < TC_TILE; t++) {
            half2* dst = (half2*)(my_tile + t * D + lane * 4);
            dst[0] = {0, 0}; dst[1] = {0, 0};
        }
        __syncwarp();

        // QK via wmma
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int d = 0; d < D; d += 16) {
            wmma::load_matrix_sync(a_frag, q_smem + d, D);
            wmma::load_matrix_sync(b_frag, my_tile + d, D);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* qk_buf = (float*)my_tile;
        wmma::store_matrix_sync(qk_buf, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // Scalar PV with global max (no online softmax)
        for (int t = 0; t < tile_len; t++) {
            float qk = qk_buf[t];
            float p = __expf(qk - m_global);

            const half2* vp = (const half2*)(v_base + (j + t) * D + lane * 4);
            half2 vh0 = vp[0], vh1 = vp[1];

            l_i += p;
            acc_reg[0] += p * __half2float(vh0.x);
            acc_reg[1] += p * __half2float(vh0.y);
            acc_reg[2] += p * __half2float(vh1.x);
            acc_reg[3] += p * __half2float(vh1.y);
        }
    }

    int wr = warp_id * D + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) s_acc[wr + e] = acc_reg[e];
    if (lane == 0) s_l[warp_id] = l_i;
    __syncthreads();

    if (warp_id == 0) {
        float global_l = 0.f;
        float fa[4] = {0,0,0,0};
        int rd = lane * 4;
        for (int s = 0; s < num_splits; s++) {
            global_l += s_l[s];
            #pragma unroll
            for (int e = 0; e < 4; e++)
                fa[e] += s_acc[s * D + rd + e];
        }
        float inv_l = 1.f / global_l;
        #pragma unroll
        for (int e = 0; e < 4; e++) fa[e] *= inv_l;
        store_row(Out + bhd, lane, fa);
    }
}


// ── TC Python bindings ──
// A100 max dynamic smem = 164KB (166912 bytes). Default limit = 48KB.
// Must opt-in with cudaFuncSetAttribute for larger smem.
torch::Tensor launch_tc_standard(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                  float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    auto Out = torch::empty_like(Q);

    int threads = num_splits * 32;
    int smem = (16 * D + num_splits * TC_TILE * D) * sizeof(half)
             + (num_splits * D + num_splits * 2) * sizeof(float);

    // Opt-in for large dynamic shared memory
    cudaFuncSetAttribute(splitk_decode_tc_standard,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    splitk_decode_tc_standard<<<B * H, threads, smem>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)Out.data_ptr(),
        sm_scale, B, H, Lk, D, num_splits);
    return Out;
}

torch::Tensor launch_tc_oracle(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                torch::Tensor GlobalMax, float sm_scale, int num_splits) {
    int B = Q.size(0), H = Q.size(1), D = Q.size(3);
    int Lk = K.size(2);
    auto Out = torch::empty_like(Q);

    int threads = num_splits * 32;
    int smem = (16 * D + num_splits * TC_TILE * D) * sizeof(half)
             + (num_splits * D + num_splits) * sizeof(float);

    cudaFuncSetAttribute(splitk_decode_tc_oracle,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    splitk_decode_tc_oracle<<<B * H, threads, smem>>>(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)Out.data_ptr(),
        (const float*)GlobalMax.data_ptr(),
        sm_scale, B, H, Lk, D, num_splits);
    return Out;
}
── End Tensor Core (보류) ──────────────────────────────────── */


// ── Occupancy query ──
// Returns: [active_blocks_per_sm, max_threads_per_sm, num_sms,
//           theoretical_occupancy_pct, total_active_blocks]
std::vector<float> query_occupancy(std::string kernel_name, int threads_per_block,
                                    int dynamic_smem, int num_blocks) {
    void* kernel_fn = nullptr;
    if (kernel_name == "standard")       kernel_fn = (void*)splitk_decode_standard;
    else if (kernel_name == "oracle")    kernel_fn = (void*)splitk_decode_oracle;
    else if (kernel_name == "localmax")  kernel_fn = (void*)splitk_phase1_localmax;
    else return {-1, -1, -1, -1, -1};

    int active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel_fn, threads_per_block, dynamic_smem);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int num_sms = prop.multiProcessorCount;
    float active_threads = active_blocks * threads_per_block;
    float occupancy_pct = (active_threads / max_threads_per_sm) * 100.f;
    // Total active blocks across GPU (capped by grid size)
    int total_active = min(active_blocks * num_sms, num_blocks);

    return {(float)active_blocks, (float)max_threads_per_sm, (float)num_sms,
            occupancy_pct, (float)total_active};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_standard", &launch_standard);
    m.def("launch_oracle", &launch_oracle);
    m.def("launch_localmax_async", &launch_localmax_async);
    m.def("launch_oracle_async", &launch_oracle_async);
    m.def("launch_tc_oracle_async", &launch_tc_oracle_async);
    m.def("launch_profiled", &launch_profiled);
    m.def("launch_profiled_oracle", &launch_profiled_oracle);
    m.def("query_occupancy", &query_occupancy);
}
