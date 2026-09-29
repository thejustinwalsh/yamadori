#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    // optional early nudge (see common_reasoning_budget_nudge in the header)
    llama_tokens nudge_tokens;          // forced mid-reasoning; empty = off
    llama_tokens nudge_boundary_tokens; // boundary tokens when vocab == nullptr (tests)
    int32_t      nudge_at;              // budget tokens consumed before the nudge arms; -1 = off
    int32_t      nudge_min_room;        // budget that must remain after the nudge
    bool         nudge_spent;           // fired or abandoned in this generation
    size_t       nudge_pos;             // next position in nudge_tokens to force
};

// a nudge boundary: the piece contains a newline (with a vocab), else a listed
// boundary token (or any token when none are listed)
static bool common_reasoning_budget_is_boundary(const common_reasoning_budget_ctx * ctx, llama_token token, const std::string & piece) {
    if (ctx->vocab != nullptr) {
        return piece.find('\n') != std::string::npos;
    }
    if (ctx->nudge_boundary_tokens.empty()) {
        return true;
    }
    return std::find(ctx->nudge_boundary_tokens.begin(), ctx->nudge_boundary_tokens.end(), token) != ctx->nudge_boundary_tokens.end();
}

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            std::string piece;
            if (ctx->vocab != nullptr) {
                piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                    }
                } else if (ctx->nudge_at >= 0 && !ctx->nudge_spent &&
                           ctx->budget - ctx->remaining >= ctx->nudge_at &&
                           utf8_complete && common_reasoning_budget_is_boundary(ctx, token, piece)) {
                    // at most one nudge per generation, whether it fires or is abandoned here
                    ctx->nudge_spent = true;
                    if ((int64_t) ctx->remaining >= (int64_t) ctx->nudge_tokens.size() + ctx->nudge_min_room) {
                        ctx->state = REASONING_BUDGET_NUDGING;
                        ctx->nudge_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("nudging at a boundary, remaining=%d tokens\n", ctx->remaining);
                    } else {
                        COM_TRC("nudge abandoned, only %d tokens remain\n", ctx->remaining);
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_NUDGING:
        {
            // the forced nudge tokens count against the budget
            ctx->nudge_pos++;
            ctx->remaining--;
            if (ctx->nudge_pos >= ctx->nudge_tokens.size()) {
                ctx->end_matcher.reset();
                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "nudge complete, budget exhausted, forcing end sequence\n");
                } else {
                    ctx->state = REASONING_BUDGET_COUNTING;
                    COM_TRC("nudge complete, back to counting, remaining=%d tokens\n", ctx->remaining);
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    llama_token forced;
    if (ctx->state == REASONING_BUDGET_FORCING) {
        if (ctx->force_pos >= ctx->forced_tokens.size()) {
            return;
        }
        forced = ctx->forced_tokens[ctx->force_pos];
    } else if (ctx->state == REASONING_BUDGET_NUDGING) {
        if (ctx->nudge_pos >= ctx->nudge_tokens.size()) {
            return;
        }
        forced = ctx->nudge_tokens[ctx->nudge_pos];
    } else {
        // passthrough — don't modify logits
        return;
    }

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->nudge_spent = false;
    ctx->nudge_pos = 0;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state,
        const common_reasoning_budget_nudge & nudge);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        const common_reasoning_budget_nudge & nudge) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    // the nudge arms only with a finite budget (INT_MAX is how sampling.cpp
    // spells "unlimited") that leaves room for the nudge plus min_room after it
    int32_t nudge_at = -1;
    const int32_t nudge_min_room = std::max<int32_t>(nudge.min_room, 0);
    if (!nudge.tokens.empty() && budget > 0 && budget != INT_MAX) {
        const float   at     = std::min(std::max(nudge.at, 0.0f), 1.0f);
        const int64_t at_tok = (int64_t) std::floor((double) at * (double) budget);
        if ((int64_t) budget - at_tok >= (int64_t) nudge.tokens.size() + nudge_min_room) {
            nudge_at = (int32_t) at_tok;
            COM_TRC("nudge armed at %d of %d budget tokens (%zu nudge tokens)\n", nudge_at, budget, nudge.tokens.size());
        } else {
            COM_TRC("nudge disabled: %lld tokens after %d leave no room for %zu nudge tokens + %d\n",
                    (long long) ((int64_t) budget - at_tok), (int32_t) at_tok, nudge.tokens.size(), nudge_min_room);
        }
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ token_matcher(start_seqs),
            /* .end_matcher   = */ token_matcher(end_seqs),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
            /* .nudge_tokens          = */ nudge_at >= 0 ? nudge.tokens : llama_tokens(),
            /* .nudge_boundary_tokens = */ nudge.boundary_tokens,
            /* .nudge_at              = */ nudge_at,
            /* .nudge_min_room        = */ nudge_min_room,
            /* .nudge_spent           = */ false,
            /* .nudge_pos             = */ 0,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        const common_reasoning_budget_nudge & nudge) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state, nudge);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget (or forcing the
    // nudge, which is then abandoned) may be forced; any other state (idle,
    // already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING && ctx->state != REASONING_BUDGET_NUDGING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
