#pragma once

// Multi-token rows for the CPU mul_mat_id (mmid-mt.cpp; ported from Strata, MIT, see there).

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// true when the rows of an expert with n_tokens routed tokens go to ggml_mmid_mt_rows (the type has a kernel, the
// CPU has AVX2, and GGML_CPU_MMID_MT allows it: 0 off, 1 = two or more tokens (default), 2 = every expert)
bool ggml_mmid_mt_use(enum ggml_type type, int64_t n_tokens);

// out[t][r] = row r of w (rows row_bytes apart, nrows of them) . act[t], for nt tokens; act[t] is the token's
// activation in the type's vec_dot_type (Q8_K for the i-quants, Q8_0 for Q2_0); n = the row length in values
void ggml_mmid_mt_rows(enum ggml_type type, int n, const void * w, size_t row_bytes, int nrows,
                       const void * const * act, int nt, float * const * out);

#ifdef __cplusplus
}
#endif
