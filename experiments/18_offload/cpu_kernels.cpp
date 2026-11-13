// CPU residual-plane kernels for the offload regime (experiments/18_offload).
// One additive-codebook plane, d=8, K=256, codebook shared per linear (AQLM layout: codebook [256, 8]),
// per-output-channel scale. Codes are stored output-blocked: [out/64, G, 64] uint8 (G = in/8), i.e. for each
// block of 64 output rows the codes of group g are one 64-byte cache line. This is an offline repack of the
// AQLM [out, G] code matrix (same bytes, no extra storage).
//   y[m, o] = scale[o] * sum_g lut[m, g, codes[o, g]],   lut[m, g, c] = <x[m, 8g:8g+8], cb[c]>
// Kernels:
//   lut_build_f32  : fp32 LUT [M, G, 256]
//   gemv_gather_f32: AVX-512 vpgatherdps over the fp32 LUT (exact up to fp32 summation order)
//   lut_build_i8   : int8 LUT [M, G, 256] with one fp32 scale per (m, 32-group block)  (approximate, T-MAC style)
//   gemv_perm_i8   : AVX-512 VBMI vpermi2b table lookup (2x 128-entry halves + blend), int16 accumulation over
//                    32 groups, then fp32 FMA with the block scale
//   stream_read    : read bandwidth probe
// Build: g++-11 -O3 -march=native -fopenmp -shared -fPIC cpu_kernels.cpp -o libcpuk.so
#include <immintrin.h>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>


extern "C" void first_touch(uint8_t* p, int64_t n) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 4096) p[i] = 0;
}

extern "C" double stream_read(const float* a, int64_t n) {
    double total = 0;
#pragma omp parallel reduction(+ : total)
    {
        __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
#pragma omp for schedule(static)
        for (int64_t i = 0; i < n; i += 64) {
            s0 = _mm512_add_ps(s0, _mm512_load_ps(a + i));
            s1 = _mm512_add_ps(s1, _mm512_load_ps(a + i + 16));
            s2 = _mm512_add_ps(s2, _mm512_load_ps(a + i + 32));
            s3 = _mm512_add_ps(s3, _mm512_load_ps(a + i + 48));
        }
        total += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
    }
    return total;
}

// cbT: [8, 256] (transposed codebook). lut: [M, G, 256]
static inline void lut_group(const float* xp, const float* cbT, float* out) {
    __m512 acc[16];
    __m512 x0 = _mm512_set1_ps(xp[0]);
    for (int cc = 0; cc < 16; cc++) acc[cc] = _mm512_mul_ps(x0, _mm512_loadu_ps(cbT + 16 * cc));
    for (int j = 1; j < 8; j++) {
        __m512 xj = _mm512_set1_ps(xp[j]);
        for (int cc = 0; cc < 16; cc++) acc[cc] = _mm512_fmadd_ps(xj, _mm512_loadu_ps(cbT + j * 256 + 16 * cc), acc[cc]);
    }
    for (int cc = 0; cc < 16; cc++) _mm512_storeu_ps(out + 16 * cc, acc[cc]);
}

extern "C" void lut_build_f32(const float* x, const float* cbT, float* lut, int M, int in) {
    const int G = in / 8;
#pragma omp parallel for schedule(static) collapse(2)
    for (int m = 0; m < M; m++)
        for (int g = 0; g < G; g++) lut_group(x + (int64_t)m * in + 8 * g, cbT, lut + ((int64_t)m * G + g) * 256);
}

template <int MC>
static inline void gather_block(const uint8_t* cblk, const float* lut, int64_t lut_mstride, int g0, int g1,
                                float* yb /*[MC][64] scratch*/, bool init) {
    __m512 acc[MC][4];
    for (int m = 0; m < MC; m++)
        for (int k = 0; k < 4; k++) acc[m][k] = init ? _mm512_setzero_ps() : _mm512_loadu_ps(yb + m * 64 + 16 * k);
    for (int g = g0; g < g1; g++) {
        __m512i c = _mm512_load_si512((const void*)(cblk + (int64_t)g * 64));
        __m512i off = _mm512_set1_epi32(g * 256);
        __m512i idx[4];
        idx[0] = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(c, 0)), off);
        idx[1] = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(c, 1)), off);
        idx[2] = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(c, 2)), off);
        idx[3] = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(c, 3)), off);
        for (int m = 0; m < MC; m++) {
            const float* base = lut + m * lut_mstride;
            for (int k = 0; k < 4; k++) acc[m][k] = _mm512_add_ps(acc[m][k], _mm512_i32gather_ps(idx[k], base, 4));
        }
    }
    for (int m = 0; m < MC; m++)
        for (int k = 0; k < 4; k++) _mm512_storeu_ps(yb + m * 64 + 16 * k, acc[m][k]);
}

// codes: [out/64, G, 64]; lut [M, G, 256]; y [M, out]. gtile: groups per LUT tile (L2 blocking).
extern "C" void gemv_gather_f32(const uint8_t* codes, const float* lut, const float* scale, float* y, int M, int in, int out,
                     int gtile) {
    const int G = in / 8, OB = out / 64;
    const int64_t ms = (int64_t)G * 256;
#pragma omp parallel
    {
        int nt = omp_get_num_threads(), t = omp_get_thread_num();
        int ob0 = (int64_t)OB * t / nt, ob1 = (int64_t)OB * (t + 1) / nt;
        float scratch[4 * 64];
        for (int g0 = 0; g0 < G; g0 += gtile) {
            int g1 = std::min(G, g0 + gtile);
            bool init = g0 == 0;
            for (int ob = ob0; ob < ob1; ob++) {
                const uint8_t* cblk = codes + (int64_t)ob * G * 64;
                for (int m0 = 0; m0 < M; m0 += 4) {
                    int mc = std::min(4, M - m0);
                    float* yb = scratch;
                    // y accumulates in place: copy rows into scratch
                    if (!init)
                        for (int m = 0; m < mc; m++) memcpy(yb + m * 64, y + (int64_t)(m0 + m) * out + ob * 64, 256);
                    const float* lb = lut + m0 * ms;
                    switch (mc) {
                        case 1: gather_block<1>(cblk, lb, ms, g0, g1, yb, init); break;
                        case 2: gather_block<2>(cblk, lb, ms, g0, g1, yb, init); break;
                        case 3: gather_block<3>(cblk, lb, ms, g0, g1, yb, init); break;
                        default: gather_block<4>(cblk, lb, ms, g0, g1, yb, init); break;
                    }
                    bool last = g1 == G;
                    for (int m = 0; m < mc; m++) {
                        float* yo = y + (int64_t)(m0 + m) * out + ob * 64;
                        if (last)
                            for (int k = 0; k < 64; k += 16)
                                _mm512_storeu_ps(yo + k, _mm512_mul_ps(_mm512_loadu_ps(yb + m * 64 + k),
                                                                       _mm512_loadu_ps(scale + ob * 64 + k)));
                        else
                            memcpy(yo, yb + m * 64, 256);
                    }
                }
            }
        }
    }
}

// int8 LUT: lutq [M, G, 256] int8, lscale [M, G/32] fp32 (per 32-group block, i.e. 256 inputs)
extern "C" void lut_build_i8(const float* x, const float* cbT, int8_t* lutq, float* lscale, int M, int in) {
    const int G = in / 8, NB = G / 32;
#pragma omp parallel for schedule(static) collapse(2)
    for (int m = 0; m < M; m++)
        for (int b = 0; b < NB; b++) {
            float tmp[32 * 256] __attribute__((aligned(64)));
            for (int gg = 0; gg < 32; gg++) lut_group(x + (int64_t)m * in + 8 * (b * 32 + gg), cbT, tmp + gg * 256);
            __m512 mx = _mm512_setzero_ps();
            for (int i = 0; i < 32 * 256; i += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_load_ps(tmp + i)));
            float amax = _mm512_reduce_max_ps(mx);
            float s = amax > 0 ? amax / 127.f : 1.f;
            lscale[(int64_t)m * NB + b] = s;
            __m512 inv = _mm512_set1_ps(1.f / s);
            int8_t* o = lutq + ((int64_t)m * G + b * 32) * 256;
            for (int i = 0; i < 32 * 256; i += 16) {
                __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_load_ps(tmp + i), inv));  // round-to-nearest
                _mm_storeu_si128((__m128i*)(o + i), _mm512_cvtsepi32_epi8(q));
            }
        }
}

template <int MC, bool SH>
static inline void perm_block(const uint8_t* cblk, const int8_t* lutq, int64_t lms, const float* lscale,
                              int64_t sms, int G, float* yo /*[MC][64]*/) {
    __m512 acc[MC][4];
    for (int m = 0; m < MC; m++)
        for (int k = 0; k < 4; k++) acc[m][k] = _mm512_setzero_ps();
    for (int b = 0; b < G / 32; b++) {
        __m512i a16[MC][2];
        for (int m = 0; m < MC; m++) a16[m][0] = a16[m][1] = _mm512_setzero_si512();
        for (int gg = 0; gg < 32; gg++) {
            int g = b * 32 + gg;
            __m512i c = _mm512_load_si512((const void*)(cblk + (int64_t)g * 64));
            __mmask64 hi = _mm512_movepi8_mask(c);
            for (int m = 0; m < MC; m++) {
                const int8_t* t = lutq + m * lms + (int64_t)g * 256;
                __m512i t0 = _mm512_load_si512((const void*)t), t1 = _mm512_load_si512((const void*)(t + 64));
                __m512i t2 = _mm512_load_si512((const void*)(t + 128)), t3 = _mm512_load_si512((const void*)(t + 192));
                __m512i r0 = _mm512_permutex2var_epi8(t0, c, t1);
                __m512i r1 = _mm512_permutex2var_epi8(t2, c, t3);
                __m512i r = _mm512_mask_blend_epi8(hi, r0, r1);
                if (SH) {  // even bytes -> outputs 0,2,..,62 ; odd bytes -> 1,3,..,63 (shifts on p0 instead of p5)
                    a16[m][0] = _mm512_add_epi16(a16[m][0], _mm512_srai_epi16(_mm512_slli_epi16(r, 8), 8));
                    a16[m][1] = _mm512_add_epi16(a16[m][1], _mm512_srai_epi16(r, 8));
                } else {
                    a16[m][0] = _mm512_add_epi16(a16[m][0], _mm512_cvtepi8_epi16(_mm512_castsi512_si256(r)));
                    a16[m][1] = _mm512_add_epi16(a16[m][1], _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(r, 1)));
                }
            }
        }
        for (int m = 0; m < MC; m++) {
            __m512 s = _mm512_set1_ps(lscale[m * sms + b]);
            for (int h = 0; h < 2; h++) {
                __m512i lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a16[m][h]));
                __m512i hi2 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a16[m][h], 1));
                acc[m][2 * h] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), s, acc[m][2 * h]);
                acc[m][2 * h + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi2), s, acc[m][2 * h + 1]);
            }
        }
    }
    for (int m = 0; m < MC; m++)
        if (SH) {
            float e[64];
            for (int k = 0; k < 4; k++) _mm512_storeu_ps(e + 16 * k, acc[m][k]);  // [even 0..31 | odd 0..31]
            for (int j = 0; j < 32; j++) { yo[m * 64 + 2 * j] = e[j]; yo[m * 64 + 2 * j + 1] = e[32 + j]; }
        } else
            for (int k = 0; k < 4; k++) _mm512_storeu_ps(yo + m * 64 + 16 * k, acc[m][k]);
}

template <bool SH>
static void gemv_perm_i8_t(const uint8_t* codes, const int8_t* lutq, const float* lscale, const float* scale, float* y,
                           int M, int in, int out) {
    const int G = in / 8, OB = out / 64, NB = G / 32;
    const int64_t lms = (int64_t)G * 256;
#pragma omp parallel
    {
        int nt = omp_get_num_threads(), t = omp_get_thread_num();
        int ob0 = (int64_t)OB * t / nt, ob1 = (int64_t)OB * (t + 1) / nt;
        float yb[4 * 64];
        for (int ob = ob0; ob < ob1; ob++) {
            const uint8_t* cblk = codes + (int64_t)ob * G * 64;
            for (int m0 = 0; m0 < M; m0 += 4) {
                int mc = std::min(4, M - m0);
                const int8_t* lb = lutq + m0 * lms;
                const float* sb = lscale + (int64_t)m0 * NB;
                switch (mc) {
                    case 1: perm_block<1, SH>(cblk, lb, lms, sb, NB, G, yb); break;
                    case 2: perm_block<2, SH>(cblk, lb, lms, sb, NB, G, yb); break;
                    case 3: perm_block<3, SH>(cblk, lb, lms, sb, NB, G, yb); break;
                    default: perm_block<4, SH>(cblk, lb, lms, sb, NB, G, yb); break;
                }
                for (int m = 0; m < mc; m++) {
                    float* yo = y + (int64_t)(m0 + m) * out + ob * 64;
                    for (int k = 0; k < 64; k += 16)
                        _mm512_storeu_ps(yo + k, _mm512_mul_ps(_mm512_loadu_ps(yb + m * 64 + k),
                                                               _mm512_loadu_ps(scale + ob * 64 + k)));
                }
            }
        }
    }
}


// Run a sequence of linears (one decode step of the residual plane) entirely in C, timing each call.
// kind: 0 = fp32 gather, 1 = int8 perm (cvt widening), 2 = int8 perm (shift widening), 3 = int8 perm register-tiled (gtile arg = obt, 0 = auto). All linears read the same activation buffer x [M, max_in].
extern "C" void gemv_perm_i8(const uint8_t* codes, const int8_t* lutq, const float* lscale, const float* scale,
                             float* y, int M, int in, int out) {
    gemv_perm_i8_t<false>(codes, lutq, lscale, scale, y, M, in, out);
}
extern "C" void gemv_perm_i8s(const uint8_t* codes, const int8_t* lutq, const float* lscale, const float* scale,
                              float* y, int M, int in, int out) {
    gemv_perm_i8_t<true>(codes, lutq, lscale, scale, y, M, in, out);
}

// Register-tiled variant: OBT output blocks (64 outputs each) x MC activation rows share each loaded table, so the
// 4 table loads per (m, g) are amortized over OBT code lines. Shift widening (even/odd byte split). OBT is chosen
// per shape so that every thread gets >= ~4 tiles (load balance); MC*OBT <= 8.
template <int MC, int OBT>
static inline void perm_tile(const uint8_t* c0, int64_t obstride, const int8_t* lutq, int64_t lms,
                             const float* lscale, int64_t sms, int b0, int b1, float* yo /*[MC][OBT*64], += */) {
    __m512 acc[MC][OBT][4];
    for (int m = 0; m < MC; m++)
        for (int q = 0; q < OBT; q++)
            for (int k = 0; k < 4; k++) acc[m][q][k] = _mm512_setzero_ps();
    for (int b = b0; b < b1; b++) {
        __m512i a16[MC][OBT][2];
        for (int m = 0; m < MC; m++)
            for (int q = 0; q < OBT; q++) a16[m][q][0] = a16[m][q][1] = _mm512_setzero_si512();
        for (int gg = 0; gg < 32; gg++) {
            int g = b * 32 + gg;
            __m512i c[OBT];
            __mmask64 hi[OBT];
            for (int q = 0; q < OBT; q++) {
                c[q] = _mm512_load_si512((const void*)(c0 + q * obstride + (int64_t)g * 64));
                hi[q] = _mm512_movepi8_mask(c[q]);
            }
            for (int m = 0; m < MC; m++) {
                const int8_t* t = lutq + m * lms + (int64_t)g * 256;
                __m512i t0 = _mm512_load_si512((const void*)t), t1 = _mm512_load_si512((const void*)(t + 64));
                __m512i t2 = _mm512_load_si512((const void*)(t + 128)), t3 = _mm512_load_si512((const void*)(t + 192));
                for (int q = 0; q < OBT; q++) {
                    __m512i r0 = _mm512_permutex2var_epi8(t0, c[q], t1);
                    __m512i r1 = _mm512_permutex2var_epi8(t2, c[q], t3);
                    __m512i r = _mm512_mask_blend_epi8(hi[q], r0, r1);
                    a16[m][q][0] = _mm512_add_epi16(a16[m][q][0], _mm512_srai_epi16(_mm512_slli_epi16(r, 8), 8));
                    a16[m][q][1] = _mm512_add_epi16(a16[m][q][1], _mm512_srai_epi16(r, 8));
                }
            }
        }
        for (int m = 0; m < MC; m++) {
            __m512 s = _mm512_set1_ps(lscale[m * sms + b]);
            for (int q = 0; q < OBT; q++)
                for (int h = 0; h < 2; h++) {
                    __m512i lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a16[m][q][h]));
                    __m512i hi2 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a16[m][q][h], 1));
                    acc[m][q][2 * h] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), s, acc[m][q][2 * h]);
                    acc[m][q][2 * h + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi2), s, acc[m][q][2 * h + 1]);
                }
        }
    }
    for (int m = 0; m < MC; m++)
        for (int q = 0; q < OBT; q++) {
            float e[64];
            for (int k = 0; k < 4; k++) _mm512_storeu_ps(e + 16 * k, acc[m][q][k]);
            float* o = yo + m * OBT * 64 + q * 64;
            for (int j = 0; j < 32; j++) { o[2 * j] += e[j]; o[2 * j + 1] += e[32 + j]; }
        }
}

template <int OBT, int MCMAX>
static void gemv_tile(const uint8_t* codes, const int8_t* lutq, const float* lscale, const float* scale, float* y,
                      int M, int in, int out) {
    const int G = in / 8, OB = out / 64, NB = G / 32;
    const int64_t lms = (int64_t)G * 256, obs = (int64_t)G * 64;
    const int NTL = OB / OBT;
    // L2 blocking over the input: LUT slice of MCMAX rows x gtb blocks x 8 KiB <= ~1 MiB
    int gtb = std::max(1, (1 << 20) / (MCMAX * 8192));
    int ntile = (NB + gtb - 1) / gtb;
    gtb = (NB + ntile - 1) / ntile;
#pragma omp parallel
    {
        int nt = omp_get_num_threads(), t = omp_get_thread_num();
        int a0 = (int64_t)NTL * t / nt, a1 = (int64_t)NTL * (t + 1) / nt;
        const int TW = OBT * 64;
        float* acc = (float*)aligned_alloc(64, sizeof(float) * std::max(1, a1 - a0) * MCMAX * TW);
        for (int m0 = 0; m0 < M; m0 += MCMAX) {
            int mc = std::min(MCMAX, M - m0);
            const int8_t* lb = lutq + m0 * lms;
            const float* sb = lscale + (int64_t)m0 * NB;
            memset(acc, 0, sizeof(float) * std::max(1, a1 - a0) * MCMAX * TW);
            for (int b0 = 0; b0 < NB; b0 += gtb) {
                int b1 = std::min(NB, b0 + gtb);
                for (int a = a0; a < a1; a++) {
                    const uint8_t* c0 = codes + (int64_t)a * OBT * obs;
                    float* yb = acc + (int64_t)(a - a0) * MCMAX * TW;
                    switch (mc) {
                        case 1: perm_tile<1, OBT>(c0, obs, lb, lms, sb, NB, b0, b1, yb); break;
                        case 2: perm_tile<(MCMAX >= 2 ? 2 : 1), OBT>(c0, obs, lb, lms, sb, NB, b0, b1, yb); break;
                        case 3: perm_tile<(MCMAX >= 3 ? 3 : 1), OBT>(c0, obs, lb, lms, sb, NB, b0, b1, yb); break;
                        default: perm_tile<MCMAX, OBT>(c0, obs, lb, lms, sb, NB, b0, b1, yb); break;
                    }
                }
            }
            for (int a = a0; a < a1; a++) {
                const float* yb = acc + (int64_t)(a - a0) * MCMAX * TW;
                for (int m = 0; m < mc; m++) {
                    float* yo = y + (int64_t)(m0 + m) * out + a * TW;
                    for (int k = 0; k < TW; k += 16)
                        _mm512_storeu_ps(yo + k, _mm512_mul_ps(_mm512_loadu_ps(yb + m * TW + k),
                                                               _mm512_loadu_ps(scale + a * TW + k)));
                }
            }
        }
        free(acc);
    }
}

// obt <= 0: auto (largest of 4/2/1 that leaves >= 4 tiles per thread)
extern "C" void gemv_perm_i8t(const uint8_t* codes, const int8_t* lutq, const float* lscale, const float* scale,
                              float* y, int M, int in, int out, int obt) {
    const int OB = out / 64, nt = omp_get_max_threads();
    if (obt <= 0) obt = (OB % 4 == 0 && OB / 4 >= 4 * nt) ? 4 : (OB % 2 == 0 && OB / 2 >= 4 * nt) ? 2 : 1;
    if (obt == 4) gemv_tile<4, 2>(codes, lutq, lscale, scale, y, M, in, out);
    else if (obt == 2) gemv_tile<2, 4>(codes, lutq, lscale, scale, y, M, in, out);
    else gemv_tile<1, 4>(codes, lutq, lscale, scale, y, M, in, out);
}

extern "C" void sweep(int kind, int n, const uint8_t* const* codes, const int* ins, const int* outs, const float* x,
           const float* cbT, float* lut, int8_t* lutq, float* lscale, const float* scale, float* y, int M, int gtile,
           double* t_lut, double* t_gemv) {
    for (int i = 0; i < n; i++) {
        double t0 = omp_get_wtime();
        if (kind == 0) lut_build_f32(x, cbT, lut, M, ins[i]);
        else lut_build_i8(x, cbT, lutq, lscale, M, ins[i]);
        double t1 = omp_get_wtime();
        if (kind == 0) gemv_gather_f32(codes[i], lut, scale, y, M, ins[i], outs[i], gtile);
        else if (kind == 1) gemv_perm_i8(codes[i], lutq, lscale, scale, y, M, ins[i], outs[i]);
        else if (kind == 2) gemv_perm_i8s(codes[i], lutq, lscale, scale, y, M, ins[i], outs[i]);
        else gemv_perm_i8t(codes[i], lutq, lscale, scale, y, M, ins[i], outs[i], gtile);
        double t2 = omp_get_wtime();
        t_lut[i] = t1 - t0;
        t_gemv[i] = t2 - t1;
    }
}

