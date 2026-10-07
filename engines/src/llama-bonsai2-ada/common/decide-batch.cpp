#include "decide-batch.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace {

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t common_len(const common_decide_tokens & a, const common_decide_tokens & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        i++;
    }
    return i;
}

// one run of tokens on one sequence; tag >= 0 reads the logits row of the LAST token and reports it as `tag`
struct item {
    llama_seq_id        seq;
    const llama_token * tok;
    int32_t             n;
    llama_pos           pos0;
    int64_t             tag;
};

struct failure {
    std::string code;
    std::string text;
};

// Decodes `items` (seq ids ascending, longest run first, so the finished sequences of a staircase are a suffix
// and the ubatch splitter keeps consecutive ids together) in lock step: every call carries the same slice of
// every unfinished item, at most n_batch tokens in all. Returns false and fills `fail` on a decode error.
bool run_items(llama_context * ctx, std::vector<item> items,
               const std::function<void(int64_t, const float *)> & on_row,
               int32_t & n_calls, int32_t & n_tokens, failure & fail) {
    const int32_t n_batch = std::max<int32_t>(1, (int32_t) llama_n_batch(ctx));
    std::vector<int32_t> done(items.size(), 0);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    bool ok = true;

    while (ok) {
        int32_t n_active = 0;
        for (size_t i = 0; i < items.size(); i++) {
            if (done[i] < items[i].n) {
                n_active++;
            }
        }
        if (n_active == 0) {
            break;
        }

        const int32_t slice = std::max<int32_t>(1, n_batch / n_active);

        common_batch_clear(batch);
        std::vector<std::pair<int32_t, size_t>> rows; // (batch index, item) of every finished item with an output
        for (size_t i = 0; i < items.size(); i++) {
            const int32_t left = items[i].n - done[i];
            if (left <= 0) {
                continue;
            }
            const int32_t take = std::min(slice, left);
            for (int32_t t = 0; t < take; t++) {
                const bool last = done[i] + t + 1 == items[i].n;
                const bool want = last && items[i].tag >= 0;
                if (want) {
                    rows.emplace_back(batch.n_tokens, i);
                }
                common_batch_add(batch, items[i].tok[done[i] + t], items[i].pos0 + done[i] + t, { items[i].seq }, want);
            }
            done[i] += take;
            n_tokens += take;
        }

        const int32_t ret = llama_decode(ctx, batch);
        n_calls++;
        if (ret != 0) {
            fail.code = ret == 1 ? "no_cells" : "decode_failed";
            fail.text = ret == 1 ? "no free cells in the KV pool for this batch (llama_decode returned 1)"
                                 : "llama_decode failed (" + std::to_string(ret) + ")";
            ok = false;
            break;
        }

        for (const auto & [idx, i] : rows) {
            const float * logits = llama_get_logits_ith(ctx, idx);
            if (logits == nullptr) {
                fail.code = "decode_failed";
                fail.text = "no logits for an output row";
                ok = false;
                break;
            }
            try {
                on_row(items[i].tag, logits);
            } catch (const std::exception & e) {
                fail.code = "internal";
                fail.text = std::string("reading a row failed: ") + e.what();
                ok = false;
                break;
            }
        }
    }

    llama_batch_free(batch);
    return ok;
}

struct leaf {
    size_t              g;
    size_t              b;
    const llama_token * own;
    int32_t             n_own;
    llama_seq_id        seq;
};

struct group_plan {
    size_t              g;
    std::vector<size_t> idx;  // which blocks of the group (at most n_forks of them)
    int32_t shared = 0;       // L
    int32_t n_leaves = 0;
    int32_t key_own = 0;      // own length of leaf 0 (the sort key)
};

} // namespace

void common_decide_release(llama_context * ctx, common_decide_state & state, const common_decide_config & cfg) {
    if (state.valid || !state.tokens.empty()) {
        llama_memory_seq_rm(llama_get_memory(ctx), cfg.seq_prefix, -1, -1);
    }
    state.tokens.clear();
    state.valid = false;
}

bool common_decide_run(
        llama_context              * ctx,
        common_decide_state        & state,
        const common_decide_config & cfg,
        const common_decide_input  & in,
        const common_decide_logits_cb & cb,
        common_decide_output       & out) {
    out = common_decide_output();
    llama_memory_t mem = llama_get_memory(ctx);

    auto fail_with = [&](const std::string & code, const std::string & text) {
        out.error_code = code;
        out.error      = text;
        return false;
    };

    // ---- validate (nothing is touched before this passes)
    if (cfg.n_forks < 1) {
        return fail_with("invalid_request", "n_forks must be >= 1");
    }
    if ((int64_t) cfg.seq_first + cfg.n_forks > (int64_t) llama_n_seq_max(ctx) ||
        cfg.seq_prefix < 0 || cfg.seq_prefix >= cfg.seq_first) {
        return fail_with("invalid_request", "the context has too few sequences for the prefix and forks: n_seq_max = " +
                         std::to_string(llama_n_seq_max(ctx)) + ", needs " + std::to_string(cfg.seq_first + cfg.n_forks));
    }
    if (in.prefix.empty()) {
        return fail_with("invalid_request", "the prefix has no tokens");
    }
    out.blocks.resize(in.groups.size());
    for (size_t g = 0; g < in.groups.size(); g++) {
        out.blocks[g].resize(in.groups[g].size());
        for (const auto & blk : in.groups[g]) {
            if (blk.empty()) {
                return fail_with("invalid_request", "a block has no tokens");
            }
        }
    }

    const int32_t n_prefix = (int32_t) in.prefix.size();
    out.prefix_tokens = n_prefix;

    failure fl;

    // every failure path ends here: forks freed, prefix dropped
    auto cleanup_fail = [&]() {
        for (int32_t i = 0; i < cfg.n_forks; i++) {
            llama_memory_seq_rm(mem, cfg.seq_first + i, -1, -1);
        }
        common_decide_release(ctx, state, cfg);
        out.prefix_resident = false;
        out.error_code = fl.code;
        out.error      = fl.text;
        return false;
    };

    // a clean start: no scratch sequence holds anything (a call that died half way cannot leave stale cells that a fork
    // would then inherit)
    for (int32_t i = 0; i < cfg.n_forks; i++) {
        llama_memory_seq_rm(mem, cfg.seq_first + i, -1, -1);
    }

    // ---- the prefix: reuse what is resident when it is this prefix or one this prefix extends
    const int64_t t0 = now_us();
    {
        size_t start = 0;
        if (state.valid && state.tokens.size() <= in.prefix.size() &&
            common_len(state.tokens, in.prefix) == state.tokens.size() &&
            llama_memory_seq_pos_max(mem, cfg.seq_prefix) == (llama_pos) state.tokens.size() - 1) {
            start = state.tokens.size();
        } else {
            common_decide_release(ctx, state, cfg);
        }
        out.prefix_reused = (int32_t) start;

        if (start < in.prefix.size()) {
            item it{ cfg.seq_prefix, in.prefix.data() + start, (int32_t) (in.prefix.size() - start), (llama_pos) start, -1 };
            int32_t calls = 0, toks = 0;
            if (!run_items(ctx, { it }, nullptr, calls, toks, fl)) {
                return cleanup_fail();
            }
            out.prefix_processed = toks;
            out.n_decode_calls  += calls;
        }
        state.tokens = in.prefix;
        state.valid  = true;
    }
    out.t_prefix_us = now_us() - t0;

    // ---- plan: each group's shared head (the longest common token prefix, every block keeping >= 1 token); a group
    //      with more blocks than forks is cut into chunks of n_forks, each chunk sharing its own head
    std::vector<group_plan> plans;
    for (size_t g = 0; g < in.groups.size(); g++) {
        const auto & blocks = in.groups[g];
        for (size_t c = 0; c < blocks.size(); c += (size_t) cfg.n_forks) {
            group_plan p;
            p.g = g;
            for (size_t b = c; b < std::min(blocks.size(), c + (size_t) cfg.n_forks); b++) {
                p.idx.push_back(b);
            }
            p.n_leaves = (int32_t) p.idx.size();
            if (p.idx.size() >= 2) {
                size_t l = blocks[p.idx[0]].size();
                size_t min_len = l;
                for (size_t k = 1; k < p.idx.size(); k++) {
                    l = std::min(l, common_len(blocks[p.idx[0]], blocks[p.idx[k]]));
                    min_len = std::min(min_len, blocks[p.idx[k]].size());
                }
                p.shared = (int32_t) std::min(l, min_len - 1);
            }
            p.key_own = (int32_t) blocks[p.idx[0]].size() - p.shared;

            for (size_t k = 0; k < p.idx.size(); k++) {
                auto & bi = out.blocks[g][p.idx[k]];
                bi.n_tokens    = (int32_t) blocks[p.idx[k]].size();
                bi.n_shared    = p.shared;
                bi.n_processed = (int32_t) blocks[p.idx[k]].size() - p.shared + (k == 0 ? p.shared : 0);
            }
            out.n_tokens_shared_saved += p.shared * (p.n_leaves - 1);
            plans.push_back(std::move(p));
        }
    }
    // shared heads first (their stage needs consecutive ids), longest first inside each class
    std::stable_sort(plans.begin(), plans.end(), [](const group_plan & a, const group_plan & b) {
        if ((a.shared > 0) != (b.shared > 0)) {
            return a.shared > 0;
        }
        if (a.shared != b.shared) {
            return a.shared > b.shared;
        }
        return a.key_own > b.key_own;
    });

    // ---- waves of groups whose leaves fit the forks
    size_t i_plan = 0;
    while (i_plan < plans.size()) {
        std::vector<group_plan> wave;
        int32_t n_leaves = 0;
        while (i_plan < plans.size() && n_leaves + plans[i_plan].n_leaves <= cfg.n_forks) {
            n_leaves += plans[i_plan].n_leaves;
            wave.push_back(plans[i_plan++]);
        }
        out.n_waves++;

        // ids: leaf-index-major, so the heads (leaf 0 of every group) are consecutive
        std::vector<std::vector<llama_seq_id>> ids(wave.size());
        llama_seq_id next = cfg.seq_first;
        {
            int32_t max_leaves = 0;
            for (size_t w = 0; w < wave.size(); w++) {
                max_leaves = std::max(max_leaves, wave[w].n_leaves);
                ids[w].assign(wave[w].n_leaves, -1);
            }
            for (int32_t j = 0; j < max_leaves; j++) {
                for (size_t w = 0; w < wave.size(); w++) {
                    if (j < wave[w].n_leaves) {
                        ids[w][j] = next++;
                    }
                }
            }
        }

        const llama_pos pos_block = n_prefix;

        // stage A: the shared heads, one sequence per group, forked from the prefix
        const int64_t ta = now_us();
        {
            std::vector<item> heads;
            for (size_t w = 0; w < wave.size(); w++) {
                if (wave[w].shared == 0) {
                    continue;
                }
                llama_memory_seq_cp(mem, cfg.seq_prefix, ids[w][0], -1, -1);
                heads.push_back({ ids[w][0], in.groups[wave[w].g][wave[w].idx[0]].data(), wave[w].shared, pos_block, -1 });
            }
            if (!heads.empty()) {
                int32_t calls = 0, toks = 0;
                if (!run_items(ctx, heads, nullptr, calls, toks, fl)) {
                    return cleanup_fail();
                }
                out.n_decode_calls   += calls;
                out.n_tokens_decoded += toks;
            }
        }
        out.t_shared_us += now_us() - ta;

        // fork the leaves: from the head after its shared decode, else straight from the prefix
        const int64_t tb = now_us();
        std::vector<leaf> leaves;
        for (size_t w = 0; w < wave.size(); w++) {
            const size_t g = wave[w].g;
            for (int32_t j = 0; j < wave[w].n_leaves; j++) {
                if (wave[w].shared > 0) {
                    if (j > 0) {
                        llama_memory_seq_cp(mem, ids[w][0], ids[w][j], -1, -1);
                    }
                } else {
                    llama_memory_seq_cp(mem, cfg.seq_prefix, ids[w][j], -1, -1);
                }
                const size_t b = wave[w].idx[j];
                const auto & blk = in.groups[g][b];
                leaves.push_back({ g, b, blk.data() + wave[w].shared,
                                   (int32_t) blk.size() - wave[w].shared, ids[w][j] });
            }
        }
        std::sort(leaves.begin(), leaves.end(), [](const leaf & a, const leaf & b) { return a.seq < b.seq; });

        // stage B: every leaf's own tokens, read at the last one
        {
            std::vector<item> items;
            for (size_t k = 0; k < leaves.size(); k++) {
                items.push_back({ leaves[k].seq, leaves[k].own, leaves[k].n_own,
                                  (llama_pos) (pos_block + (int32_t) (in.groups[leaves[k].g][leaves[k].b].size() - leaves[k].n_own)),
                                  (int64_t) k });
            }
            int32_t calls = 0, toks = 0;
            const bool ok = run_items(ctx, items, [&](int64_t tag, const float * logits) {
                cb(leaves[(size_t) tag].g, leaves[(size_t) tag].b, logits);
            }, calls, toks, fl);
            out.n_decode_calls   += calls;
            out.n_tokens_decoded += toks;
            if (!ok) {
                return cleanup_fail();
            }
        }
        out.t_blocks_us += now_us() - tb;

        // free the wave (only the ids it used: each seq_rm scans every cell of the pool)
        for (llama_seq_id id = cfg.seq_first; id < next; id++) {
            llama_memory_seq_rm(mem, id, -1, -1);
        }
    }

    if (!in.keep_prefix) {
        common_decide_release(ctx, state, cfg);
    }
    out.prefix_resident = state.valid;
    return true;
}
