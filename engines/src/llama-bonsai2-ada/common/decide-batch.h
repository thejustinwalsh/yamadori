#pragma once

// DECIDE BATCH: read the next-token distribution at the end of MANY short blocks that all continue ONE shared
// prefix, in a handful of llama_decode calls, on scratch sequences.
//
//   prefix  the shared state (decoded once on its own sequence, kept between calls on request)
//   groups  each a list of blocks (token vectors) that continue the prefix; the blocks of a group are decoded
//           from their longest COMMON TOKEN PREFIX once and forked from there (a hint: sharing only ever
//           skips tokens that are identical, so it cannot change a result). A group with more blocks than forks is
//           cut into chunks of n_forks, each sharing its own head
//
// Every block runs on a sequence of its own, forked from the prefix's with llama_memory_seq_cp (attention KV:
// the cells are shared, nothing is copied; recurrent state: copy-on-write at the fork's first token), so the
// blocks never see each other and each logits row is what a lone request for prefix + block would read. The
// blocks of a wave go through llama_decode together (one ubatch group per wave stage); the scratch sequences
// are freed after every wave and the unified pool holds only the prefix between calls.
//
// Needs n_seq_max >= seq_first + n_forks (the server's --decide-seqs reserves them), and a unified KV cache
// (every sequence draws on one pool of cells).
//
// Origin: ours (2026-10-07; docs/DECIDE-BATCH.md). The mechanism -- N slots continuing one shared prompt by
// llama_memory_seq_cp -- is the one ggml-org/llama.cpp's server uses for its decision models
// (server_decision_group_tasks / copy_prompt_to, PR #29818 / #29831, MIT, as this tree).

#include "llama.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using common_decide_tokens = std::vector<llama_token>;

struct common_decide_config {
    llama_seq_id seq_prefix = 0;  // holds the prefix's state between calls
    llama_seq_id seq_first  = 1;  // first scratch (fork) sequence id; ids seq_first .. seq_first + n_forks - 1
    int32_t      n_forks    = 1;  // W: leaves decoded together in one wave (each needs its own recurrent cell)
};

// the resident prefix: persists between calls on the caller's side, never owns anything but its token list
struct common_decide_state {
    common_decide_tokens tokens;   // what sits in seq_prefix, positions [0, tokens.size())
    bool                 valid = false;
};

struct common_decide_input {
    common_decide_tokens                           prefix;
    std::vector<std::vector<common_decide_tokens>> groups;   // groups[g][b]: block b of group g
    bool                                           keep_prefix = false;
};

struct common_decide_block_info {
    int32_t n_tokens    = 0;   // the block's own length
    int32_t n_shared    = 0;   // of which decoded once for its group (a leaf's n_tokens - n_shared are its own)
    int32_t n_processed = 0;   // tokens decoded for this block (n_shared counts for the group's first block only)
};

struct common_decide_output {
    int32_t prefix_tokens    = 0;
    int32_t prefix_reused    = 0;   // tokens already resident
    int32_t prefix_processed = 0;
    bool    prefix_resident  = false;  // after the call

    int32_t n_waves          = 0;
    int32_t n_decode_calls   = 0;
    int32_t n_tokens_decoded = 0;      // blocks only (the prefix is prefix_processed)
    int32_t n_tokens_shared_saved = 0; // tokens a group did not decode again

    int64_t t_prefix_us = 0;
    int64_t t_shared_us = 0;
    int64_t t_blocks_us = 0;

    std::vector<std::vector<common_decide_block_info>> blocks;   // same nesting as the input

    // set on failure: a short code and the situation
    std::string error_code;     // "invalid_request" | "no_cells" | "decode_failed" | "memory_failed"
    std::string error;
};

// called once per block, as soon as the decode that produced its row has returned; `logits` (n_vocab floats)
// is valid only during the call
using common_decide_logits_cb = std::function<void(size_t group, size_t block, const float * logits)>;

// Runs the batch. true on success. On failure `out.error*` say why and nothing is left behind: every scratch
// sequence is freed and the resident prefix is dropped (state.valid false).
bool common_decide_run(
        llama_context              * ctx,
        common_decide_state        & state,
        const common_decide_config & cfg,
        const common_decide_input  & in,
        const common_decide_logits_cb & cb,
        common_decide_output       & out);

// frees the resident prefix (a no-op when there is none)
void common_decide_release(llama_context * ctx, common_decide_state & state, const common_decide_config & cfg);

// the n_seq_max a context must be created with so that cfg fits beside `n_slots` sequences: slots take ids
// [0, n_slots), the prefix n_slots, the forks the next n_forks
inline int32_t common_decide_n_seq_max(int32_t n_slots, int32_t n_forks) {
    return n_forks > 0 ? n_slots + 1 + n_forks : n_slots;
}
