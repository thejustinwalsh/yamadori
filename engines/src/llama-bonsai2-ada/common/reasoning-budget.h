#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
    REASONING_BUDGET_NUDGING,      // forcing the nudge mid-reasoning, then back to COUNTING
};

// Optional early "wrap up" nudge, forced INSIDE the reasoning block before the
// hard stop. Off when `tokens` is empty (the sampler then behaves exactly as
// it does without a nudge).
//
// Once COUNTING has consumed floor(at * budget) tokens, the sampler waits for a
// natural boundary -- the next token whose piece contains '\n' and completes a
// UTF-8 sequence -- then forces `tokens` one by one and returns to COUNTING.
// The forced nudge tokens COUNT against the budget. It fires at most once per
// generation (a re-armed second reasoning block does not nudge again; reset()
// re-arms it). A natural end of reasoning before the boundary means no nudge.
// The hard stop always wins: at init the nudge is dropped when the budget is
// unlimited or when budget - floor(at * budget) < tokens.size() + min_room, and
// at the boundary it is dropped when fewer than tokens.size() + min_room tokens
// of budget remain.
struct common_reasoning_budget_nudge {
    llama_tokens tokens;          // forced nudge sequence; empty = feature off
    float        at       = 0.8f; // fraction of the budget consumed before the nudge arms, in [0, 1]
    int32_t      min_room = 64;   // budget that must remain after the nudge, else no nudge

    // Boundary detection when vocab == nullptr (unit tests): only these tokens
    // count as boundaries, and when this is empty every token does. With a
    // vocab the boundary is always "the piece contains '\n'".
    llama_tokens boundary_tokens;
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> [NUDGING -> COUNTING] -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   NUDGING:      forces nudge.tokens token-by-token, then returns to COUNTING (optional, once)
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 and newline boundary detection; can be nullptr)
//   start_seqs     - token sequences, any of which activates counting
//   end_seqs       - token sequences, any of which naturally deactivates
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//   nudge          - optional early nudge (see common_reasoning_budget_nudge); default off
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state = REASONING_BUDGET_IDLE,
        const common_reasoning_budget_nudge & nudge = {});

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Allowed from COUNTING and from NUDGING (the nudge is abandoned).
// Returns true if the transition occurred.
bool common_reasoning_budget_force(struct llama_sampler * smpl);
