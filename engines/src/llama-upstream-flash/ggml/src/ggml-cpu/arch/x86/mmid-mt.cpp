// Multi-token row dot products for the CPU mul_mat_id (MoE experts): when several tokens of a batch are routed to
// the same expert (an MTP verify window, several slots decoding together), each weight row is decoded ONCE and
// applied to every token. ggml's own AVX2 vec_dot for these formats is single-token: every token re-does the grid
// lookups, the sign expansion and the scales.
//
// Ported from Strata: Niko1221/Strata, src/kernels/cpu/iq_avx2.cpp, read at
// 3ce2523c2823687de5372be3af58534f56cbf286 (introduced in ed14227 by pipeob0, "kernels: AVX2 multi-token i-quant
// kernels for an AVX2-only CPU"). Taken: Fmt32<> (the per-format decode of one 32-value half: grid magnitudes,
// sign vector, scales), sgn_vec, sc16/sc32, the even-signs table, hsum8 and row_dot<TY, NT>. Adapted: ggml type
// enums for Strata's numeric ids, one entry point for the rows of one expert, the fused gate/up/SiLU path left out
// (ggml runs gate, up and the GLU as separate ops).
//
//   Copyright (c) 2026 Niko1221 and the Strata contributors
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without
//   limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
//   Software, and to permit persons to whom the Software is furnished to do so, subject to the following
//   conditions:
//
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions
//   of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
//   TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
//   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.
//
// Strata's arithmetic follows ggml's (ggml-cpu/quants.c, the _generic references; the sign vector is ggml's
// bit_selector pattern from arch/x86/quants.c); only the order of the float additions differs from ggml's.
//
// Q2_0 (Q8_0 activations) is ours: the codes of a block unpacked once per row, the -1 code offset as a per-token
// correction computed once per call.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "mmid-mt.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX2__)

namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t * p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t * p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// 32 sign bits -> 32 bytes of -1 (bit set) / +1: ggml's bit_selector pattern, shared by all tokens
inline __m256i sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int) m);
    const __m128i lo = _mm_shuffle_epi8(bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// the two 16-value scales of one 32-value half (IQ2_XS, IQ2_S): int16 lanes 0-7 = a, 8-15 = b
inline __m256i sc16(int a, int b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)), _mm_set1_epi16((short) b), 1);
}
// one scale for the whole 32-value half (IQ2_XXS, IQ3_XXS, IQ3_S)
inline __m256i sc32(int s) { return _mm256_set1_epi16((short) s); }

// one u64 per 7-bit sign index: byte k = 0xFF when bit k of ksigns_iq2xs[i] is set, 0x01 otherwise
struct EvenSigns {
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) {
                r |= (uint64_t) (((ksigns_iq2xs[i] >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            }
            v[i] = r;
        }
    }
};
const EvenSigns even_signs;

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ---- per format: one 32-value half (values 64*j + 32*half .. +31) -> grid magnitudes, sign vector, scales
template <int TY> struct Fmt32;

template <> struct Fmt32<GGML_TYPE_IQ2_XXS> {   // d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        const uint8_t * q = b + 2 + 16 * j + 8 * half;
        const uint32_t w0 = u32(q), w1 = u32(q + 4);
        g = _mm256_set_epi64x((long long) iq2xxs_grid[w0 >> 24], (long long) iq2xxs_grid[(w0 >> 16) & 255],
                              (long long) iq2xxs_grid[(w0 >> 8) & 255], (long long) iq2xxs_grid[w0 & 255]);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w1 >> 21) & 127], (long long) even_signs.v[(w1 >> 14) & 127],
                                (long long) even_signs.v[(w1 >> 7) & 127], (long long) even_signs.v[w1 & 127]);
        sc = sc32(2 * (int) (w1 >> 28) + 1);
    }
};

template <> struct Fmt32<GGML_TYPE_IQ2_XS> {    // d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        uint16_t v[8];
        std::memcpy(v, b + 2 + 16 * j, 16);
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2xs_grid[v[o + 3] & 511], (long long) iq2xs_grid[v[o + 2] & 511],
                              (long long) iq2xs_grid[v[o + 1] & 511], (long long) iq2xs_grid[v[o] & 511]);
        uint32_t s = 0;
        for (int l = 0; l < 4; ++l) {
            s |= (uint32_t) ksigns_iq2xs[v[o + l] >> 9] << (8 * l);
        }
        sgn = sgn_vec(s);
        const uint8_t sb = b[66 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<GGML_TYPE_IQ2_S> {     // d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        const uint8_t * qs = b + 2 + 8 * j;
        const uint8_t h = b[66 + 2 * j + half];
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2s_grid[qs[o + 3] | ((h << 2) & 0x300)],
                              (long long) iq2s_grid[qs[o + 2] | ((h << 4) & 0x300)],
                              (long long) iq2s_grid[qs[o + 1] | ((h << 6) & 0x300)],
                              (long long) iq2s_grid[qs[o] | ((h << 8) & 0x300)]);
        const uint64_t m = u64(b + 2 + 32 + 8 * j);
        sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
        const uint8_t sb = b[74 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<GGML_TYPE_IQ3_XXS> {   // d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        const uint8_t * q = b + 2 + 16 * j + 8 * half;
        g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127],
                                (long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

template <> struct Fmt32<GGML_TYPE_IQ3_S> {     // d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        const uint8_t * q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
#define MMID_G3(k) (int) iq3s_grid[q[k] | (((h >> (k)) & 1u) << 8)]
        g = _mm256_set_epi32(MMID_G3(7), MMID_G3(6), MMID_G3(5), MMID_G3(4), MMID_G3(3), MMID_G3(2), MMID_G3(1), MMID_G3(0));
#undef MMID_G3
        const uint64_t m = u64(b + 74 + 8 * j);
        sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
        const uint8_t s = b[106 + j];
        sc = sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
    }
};

template <int TY, int NT>
inline void row_dot(const uint8_t * row, int nblocks, const block_q8_K * const * y, float * res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t * blk = row + (size_t) i * Fmt32<TY>::bytes;
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i *) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc));
                }
            }
        }
        const float dx = h2f(u16(blk)) * Fmt32<TY>::K;
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int TY, int NT>
void iq_rows(const uint8_t * w, size_t row_bytes, int nrows, int nblocks, const void * const * act, float * const * out) {
    const block_q8_K * y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K *) act[t];
    float res[NT];
    for (int r = 0; r < nrows; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, nblocks, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void iq_dispatch(int nt, const uint8_t * w, size_t row_bytes, int nrows, int nblocks, const void * const * act, float * const * out) {
    switch (nt) {
        case 1:  iq_rows<TY, 1>(w, row_bytes, nrows, nblocks, act, out); break;
        case 2:  iq_rows<TY, 2>(w, row_bytes, nrows, nblocks, act, out); break;
        case 3:  iq_rows<TY, 3>(w, row_bytes, nrows, nblocks, act, out); break;
        default: iq_rows<TY, 4>(w, row_bytes, nrows, nblocks, act, out); break;
    }
}

// ---- Q2_0 (ours): 64-value blocks, fp16 d + 16 code bytes; value v is bits 2*(v%4) of byte v/4; code c means c - 1
inline __m256i q2_0_codes32(const uint8_t * p) {
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i v  = _mm_loadl_epi64((const __m128i *) p);
    const __m128i e  = _mm_unpacklo_epi8(_mm_and_si128(v, m3), _mm_and_si128(_mm_srli_epi16(v, 2), m3));
    const __m128i o  = _mm_unpacklo_epi8(_mm_and_si128(_mm_srli_epi16(v, 4), m3), _mm_and_si128(_mm_srli_epi16(v, 6), m3));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi16(e, o)), _mm_unpackhi_epi16(e, o), 1);
}

template <int NT>
void q2_0_rows(const uint8_t * w, size_t row_bytes, int nrows, int n, const void * const * act, float * const * out) {
    const int nb = n / QK2_0;                   // Q2_0 blocks per row; two Q8_0 blocks each
    const block_q8_0 * y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0 *) act[t];
    // per token and 32-value block: d_y, and d_y * sum(q) (the -1 code offset, once per call)
    std::vector<float> dy((size_t) NT * 2 * nb), cy((size_t) NT * 2 * nb);
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    for (int t = 0; t < NT; ++t) {
        for (int b = 0; b < 2 * nb; ++b) {
            const __m256i q = _mm256_loadu_si256((const __m256i *) y[t][b].qs);
            const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q), ones16);
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
            const float d = GGML_FP16_TO_FP32(y[t][b].d);
            dy[(size_t) t * 2 * nb + b] = d;
            cy[(size_t) t * 2 * nb + b] = d * (float) _mm_cvtsi128_si32(s4);
        }
    }
    for (int r = 0; r < nrows; ++r) {
        const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) r * row_bytes);
        __m256 acc[NT];
        float corr[NT];
        for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.0f; }
        for (int i = 0; i < nb; ++i) {
            const float d0 = GGML_FP16_TO_FP32(x[i].d);
            const __m256i c0 = q2_0_codes32(x[i].qs), c1 = q2_0_codes32(x[i].qs + 8);
            for (int t = 0; t < NT; ++t) {
                const size_t k = (size_t) t * 2 * nb + 2 * i;
                const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *) y[t][2*i].qs)), ones16);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *) y[t][2*i + 1].qs)), ones16);
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d0 * dy[k]),     _mm256_cvtepi32_ps(p0), acc[t]);
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d0 * dy[k + 1]), _mm256_cvtepi32_ps(p1), acc[t]);
                corr[t] += d0 * (cy[k] + cy[k + 1]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(acc[t]) - corr[t];
    }
}

int env_mode() {
    // GGML_CPU_MMID_MT: 0 = off (ggml's single-token vec_dot everywhere), 1 = experts with >= 2 tokens (default),
    // 2 = every expert, single tokens too
    static const int mode = [] {
        const char * e = std::getenv("GGML_CPU_MMID_MT");
        return e && e[0] ? std::atoi(e) : 1;
    }();
    return mode;
}

} // namespace

extern "C" bool ggml_mmid_mt_use(enum ggml_type type, int64_t n_tokens) {
    const int mode = env_mode();
    if (mode == 0 || n_tokens < (mode >= 2 ? 1 : 2)) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_Q2_0:
            return true;
        default:
            return false;
    }
}

extern "C" void ggml_mmid_mt_rows(enum ggml_type type, int n, const void * w, size_t row_bytes, int nrows,
                                  const void * const * act, int nt, float * const * out) {
    const uint8_t * wb = (const uint8_t *) w;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        const void * const * a = act + t0;
        float * const * o = out + t0;
        switch (type) {
            case GGML_TYPE_IQ2_XXS: iq_dispatch<GGML_TYPE_IQ2_XXS>(k, wb, row_bytes, nrows, n / QK_K, a, o); break;
            case GGML_TYPE_IQ2_XS:  iq_dispatch<GGML_TYPE_IQ2_XS >(k, wb, row_bytes, nrows, n / QK_K, a, o); break;
            case GGML_TYPE_IQ2_S:   iq_dispatch<GGML_TYPE_IQ2_S  >(k, wb, row_bytes, nrows, n / QK_K, a, o); break;
            case GGML_TYPE_IQ3_XXS: iq_dispatch<GGML_TYPE_IQ3_XXS>(k, wb, row_bytes, nrows, n / QK_K, a, o); break;
            case GGML_TYPE_IQ3_S:   iq_dispatch<GGML_TYPE_IQ3_S  >(k, wb, row_bytes, nrows, n / QK_K, a, o); break;
            case GGML_TYPE_Q2_0:
                switch (k) {
                    case 1:  q2_0_rows<1>(wb, row_bytes, nrows, n, a, o); break;
                    case 2:  q2_0_rows<2>(wb, row_bytes, nrows, n, a, o); break;
                    case 3:  q2_0_rows<3>(wb, row_bytes, nrows, n, a, o); break;
                    default: q2_0_rows<4>(wb, row_bytes, nrows, n, a, o); break;
                }
                break;
            default:
                GGML_ABORT("ggml_mmid_mt_rows: unsupported type %d", (int) type);
        }
    }
}

#else  // !__AVX2__

extern "C" bool ggml_mmid_mt_use(enum ggml_type, int64_t) { return false; }
extern "C" void ggml_mmid_mt_rows(enum ggml_type, int, const void *, size_t, int, const void * const *, int, float * const *) {
    GGML_ABORT("ggml_mmid_mt_rows: built without AVX2");
}

#endif
