// common/decide-batch.cpp on a tiny random HYBRID model (qwen35: linear-attention layers carrying a recurrent
// state beside full-attention layers), CPU backend. What the engine patch promises:
//
//   - a block read through the batch equals the same prompt (prefix + block) decoded alone on a fresh sequence
//   - blocks never see each other (any group order, duplicates, waves of any size give the same rows)
//   - a group's shared head (longest common token prefix) is decoded once and changes nothing
//   - the prefix is reused when it is the resident one or an extension of it, reset otherwise
//   - every scratch sequence is freed after a call; the slots' sequences are untouched
//   - a failure (too many blocks, no cells) leaves nothing behind
//
// Built in the static CPU test tree (it includes src/ internals, as test-llama-archs does).

#include "common.h"
#include "decide-batch.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_failed = 0;

#define CHECK(cond, ...) do { \
    n_checks++; \
    if (!(cond)) { \
        n_failed++; \
        fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static constexpr uint32_t N_VOCAB = 128;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    // test-llama-archs' 1e-2 everywhere (norm weights too) makes every layer's contribution vanish: a row then
    // ignores its context and a wrong fork would pass unseen. Norm weights near 1 and projections of std 0.1 let
    // the attention and recurrent layers matter; the sensitivity check in main() proves it.
    const bool is_norm = strstr(tensor->name, "norm") != nullptr;
    std::normal_distribution<float> dis(is_norm ? 1.0f : 0.0f, is_norm ? 0.05f : 0.1f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the qwen35 fixture of test-llama-archs.cpp (dense), four layers: linear, attention, linear, attention
static gguf_context_ptr get_gguf_ctx() {
    const llm_arch arch = LLM_ARCH_QWEN35;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 1024;

    const uint32_t n_vocab = N_VOCAB;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 384;
    const uint32_t n_layer = 4;
    const uint32_t n_head_kv = n_head;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,        uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,      uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,   uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head_kv);
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,     uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,     uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,          uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,     uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,   uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    // a real (if toy) SPM vocabulary, so a llama-server can load the saved model and print pieces
    // (the server smoke test, bench/decide_batch_server_smoke.py); token ids are all it is ever sent
    {
        std::vector<std::string> pieces;
        std::vector<float>       scores;
        std::vector<int32_t>     types;
        pieces.push_back("<unk>"); scores.push_back(0.0f); types.push_back(2);
        pieces.push_back("<s>");   scores.push_back(0.0f); types.push_back(3);
        pieces.push_back("</s>");  scores.push_back(0.0f); types.push_back(3);
        for (uint32_t i = 3; i < n_vocab; i++) {
            pieces.push_back("w" + std::to_string(i));
            scores.push_back(-(float) i);
            types.push_back(1);
        }
        ms.add_kv(LLM_KV_TOKENIZER_MODEL,    "llama");
        ms.add_kv(LLM_KV_TOKENIZER_LIST,     pieces);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES,   scores);
        ms.add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE, types);
        ms.add_kv(LLM_KV_TOKENIZER_UNK_ID,   uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_BOS_ID,   uint32_t(1));
        ms.add_kv(LLM_KV_TOKENIZER_EOS_ID,   uint32_t(2));
        ms.add_kv(LLM_KV_TOKENIZER_ADD_BOS,  false);
        ms.add_kv(LLM_KV_TOKENIZER_ADD_EOS,  false);
    }
    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_progress(float, void *) { return true; }

using tokens_t = std::vector<llama_token>;

static tokens_t rnd_tokens(std::mt19937 & gen, size_t n) {
    std::uniform_int_distribution<int> d(0, N_VOCAB - 1);
    tokens_t t;
    for (size_t i = 0; i < n; i++) {
        t.push_back(d(gen));
    }
    return t;
}

static tokens_t cat(const tokens_t & a, const tokens_t & b) {
    tokens_t r = a;
    r.insert(r.end(), b.begin(), b.end());
    return r;
}

struct fixture {
    gguf_context_ptr gguf;
    llama_model_ptr  model;
    llama_context_ptr ref;     // the reference: one sequence, nothing shared
    size_t seed;

    llama_context_ptr make_ctx(uint32_t n_seq_max, uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch) const {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = n_batch;
        cp.n_ubatch        = n_ubatch;
        cp.n_seq_max       = n_seq_max;
        cp.kv_unified      = true;
        cp.n_threads       = 4;
        cp.n_threads_batch = 4;
        llama_context_ptr c(llama_init_from_model(model.get(), cp));
        if (!c) {
            throw std::runtime_error("failed to create the context");
        }
        return c;
    }

    explicit fixture(size_t seed_) : seed(seed_) {
        gguf = get_gguf_ctx();
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_progress;
        std::vector<ggml_backend_dev_t> devs = { nullptr };   // CPU only
        mp.devices = devs.data();
        size_t tmp = seed;
        model.reset(llama_model_init_from_user(gguf.get(), set_tensor_data, &tmp, mp));
        if (!model) {
            throw std::runtime_error("failed to create the model");
        }
        ref = make_ctx(1, 1024, 512, 512);
    }

    // the logits row at the last token of `prompt`, decoded cold on one sequence
    std::vector<float> reference(const tokens_t & prompt) {
        llama_memory_clear(llama_get_memory(ref.get()), true);
        llama_batch batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
        for (size_t i = 0; i < prompt.size(); i++) {
            common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
        }
        const int ret = llama_decode(ref.get(), batch);
        llama_batch_free(batch);
        if (ret != 0) {
            throw std::runtime_error("reference decode failed");
        }
        const float * l = llama_get_logits_ith(ref.get(), (int32_t) prompt.size() - 1);
        return std::vector<float>(l, l + N_VOCAB);
    }
};

static std::vector<float> log_softmax(const std::vector<float> & l) {
    float m = l[0];
    for (float v : l) {
        m = std::max(m, v);
    }
    double s = 0.0;
    for (float v : l) {
        s += std::exp((double) v - m);
    }
    const double lse = m + std::log(s);
    std::vector<float> r(l.size());
    for (size_t i = 0; i < l.size(); i++) {
        r[i] = (float) (l[i] - lse);
    }
    return r;
}

static double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d;
}

static size_t argmax(const std::vector<float> & a) {
    size_t k = 0;
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i] > a[k]) {
            k = i;
        }
    }
    return k;
}

using rows_t = std::map<std::pair<size_t, size_t>, std::vector<float>>;

struct run_result {
    bool ok = false;
    common_decide_output out;
    rows_t rows;
};

static run_result run(llama_context * ctx, common_decide_state & st, const common_decide_config & cfg,
                      const tokens_t & prefix, const std::vector<std::vector<tokens_t>> & groups, bool keep) {
    run_result r;
    common_decide_input in;
    in.prefix = prefix;
    in.groups = groups;
    in.keep_prefix = keep;
    r.ok = common_decide_run(ctx, st, cfg, in, [&](size_t g, size_t b, const float * l) {
        r.rows[{g, b}] = std::vector<float>(l, l + N_VOCAB);
    }, r.out);
    return r;
}

static bool all_free(llama_context * ctx, const common_decide_config & cfg, bool prefix_free) {
    llama_memory_t mem = llama_get_memory(ctx);
    for (int32_t i = 0; i < cfg.n_forks; i++) {
        if (llama_memory_seq_pos_max(mem, cfg.seq_first + i) != -1) {
            return false;
        }
    }
    return !prefix_free || llama_memory_seq_pos_max(mem, cfg.seq_prefix) == -1;
}

static double g_worst = 0.0;   // the largest log-softmax difference against the reference over the whole run

static void compare_rows(fixture & fx, const char * what, const run_result & r, const tokens_t & prefix,
                         const std::vector<std::vector<tokens_t>> & groups, double tol) {
    double worst = 0.0;
    bool top_same = true;
    size_t n = 0;
    for (size_t g = 0; g < groups.size(); g++) {
        for (size_t b = 0; b < groups[g].size(); b++) {
            auto it = r.rows.find({g, b});
            CHECK(it != r.rows.end(), "%s: no row for block %zu/%zu", what, g, b);
            if (it == r.rows.end()) {
                continue;
            }
            const auto ref = log_softmax(fx.reference(cat(prefix, groups[g][b])));
            const auto got = log_softmax(it->second);
            worst = std::max(worst, max_abs(ref, got));
            top_same = top_same && argmax(ref) == argmax(got);
            n++;
        }
    }
    g_worst = std::max(g_worst, worst);
    CHECK(worst <= tol, "%s: max |log-softmax difference| = %.3e over %zu blocks (tolerance %.1e)", what, worst, n, tol);
    CHECK(top_same, "%s: an argmax differs from the reference", what);
    printf("  %-44s %3zu blocks, max |dlogp| vs cold single-sequence = %.3e\n", what, n, worst);
}

int main(int argc, char ** argv) {
    common_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    try {
        const size_t seed = 20261007;
        fixture fx(seed);

        // --save PATH: write this fixture model (tokenizer "no_vocab": token ids only) for the server smoke test
        // (bench/decide_batch_server_smoke.py)
        for (int i = 1; i + 1 < argc; i++) {
            if (strcmp(argv[i], "--save") == 0) {
                llama_model_save_to_file(fx.model.get(), argv[i + 1]);
                printf("saved %s\n", argv[i + 1]);
                return 0;
            }
        }
        std::mt19937 gen((uint32_t) seed);

        const int32_t n_slots = 2;
        // The reference decodes prefix + block in ONE 512-token ubatch; the batch decodes 2-16 tokens a sequence per
        // ubatch, so float32 sums run in another order: 1.3e-4 .. 2.4e-4 measured (15 of 16 comparisons, one exactly 0).
        // A wrong state moves a row by 0.76 or more (sensitivity check below), 3,000x the tolerance.
        const double  tol     = 5e-4;

        // ---- a context with W forks beside two slots; small ubatch so every stage spans several
        auto make = [&](int32_t W, uint32_t n_ctx = 1024) {
            return fx.make_ctx((uint32_t) common_decide_n_seq_max(n_slots, W), n_ctx, 64, 16);
        };
        common_decide_config cfg8;
        cfg8.seq_prefix = n_slots;
        cfg8.seq_first  = n_slots + 1;
        cfg8.n_forks    = 8;
        auto cfg_for = [&](int32_t W) {
            common_decide_config c;
            c.seq_prefix = n_slots;
            c.seq_first  = n_slots + 1;
            c.n_forks    = W;
            return c;
        };

        CHECK(common_decide_n_seq_max(2, 8) == 11 && common_decide_n_seq_max(2, 0) == 2, "n_seq_max arithmetic");

        const tokens_t prefix = rnd_tokens(gen, 57);

        // ---- 0. the test can see: the rows depend on the blocks AND on the prefix state (so a wrong fork shows)
        {
            const tokens_t b1 = rnd_tokens(gen, 12), b2 = rnd_tokens(gen, 12);
            const auto r1 = log_softmax(fx.reference(cat(prefix, b1)));
            const auto r2 = log_softmax(fx.reference(cat(prefix, b2)));
            const auto r0 = log_softmax(fx.reference(b1));              // the same block with no prefix at all
            const tokens_t prefix2 = rnd_tokens(gen, 57);
            const auto rp = log_softmax(fx.reference(cat(prefix2, b1)));
            const double d_blocks = max_abs(r1, r2), d_prefix = max_abs(r1, r0), d_prefix2 = max_abs(r1, rp);
            printf("  sensitivity: different blocks %.3e, no prefix %.3e, another prefix %.3e (tolerance %.0e)\n", d_blocks, d_prefix, d_prefix2, tol);
            CHECK(d_blocks > 100 * tol && d_prefix > 100 * tol && d_prefix2 > 100 * tol, "the fixture's rows are too flat to tell a wrong state from a right one");
        }

        // ---- 1. flat groups, blocks of different lengths, one wave
        {
            auto ctx = make(8);
            common_decide_state st;
            std::vector<std::vector<tokens_t>> groups;
            for (size_t len : { 9u, 31u, 17u, 4u, 22u, 13u }) {
                groups.push_back({ rnd_tokens(gen, len) });
            }
            auto r = run(ctx.get(), st, cfg8, prefix, groups, false);
            CHECK(r.ok, "flat run: %s", r.out.error.c_str());
            compare_rows(fx, "flat, one wave, mixed lengths", r, prefix, groups, tol);
            CHECK(r.out.n_waves == 1, "one wave, got %d", r.out.n_waves);
            CHECK(r.out.prefix_processed == 57 && r.out.prefix_reused == 0, "cold prefix: %d/%d", r.out.prefix_processed, r.out.prefix_reused);
            int total = 0;
            for (auto & g : groups) {
                total += (int) g[0].size();
            }
            CHECK(r.out.n_tokens_decoded == total, "decoded %d of %d block tokens", r.out.n_tokens_decoded, total);
            CHECK(all_free(ctx.get(), cfg8, true), "every scratch sequence and the prefix are free after the call");
            CHECK(llama_memory_cells_used_max_p1(llama_get_memory(ctx.get())) == 0, "the unified pool holds no cell after the call: %u", llama_memory_cells_used_max_p1(llama_get_memory(ctx.get())));
            CHECK(!r.out.prefix_resident && !st.valid, "prefix not resident without keep_prefix");
        }

        // ---- 2. groups with a shared head: the two "orders" of a question
        {
            auto ctx = make(8);
            common_decide_state st;
            std::vector<std::vector<tokens_t>> groups;
            for (size_t q = 0; q < 3; q++) {
                const tokens_t head = rnd_tokens(gen, 20 + 7 * q);
                groups.push_back({ cat(head, rnd_tokens(gen, 6 + q)), cat(head, rnd_tokens(gen, 8 + q)) });
            }
            auto r = run(ctx.get(), st, cfg8, prefix, groups, false);
            CHECK(r.ok, "grouped run: %s", r.out.error.c_str());
            compare_rows(fx, "groups of two sharing a head", r, prefix, groups, tol);
            // shared = the longest common token prefix of the pair (random tails can agree by chance)
            int saved = 0;
            for (size_t g = 0; g < groups.size(); g++) {
                size_t l = 0;
                const size_t m = std::min(groups[g][0].size(), groups[g][1].size()) - 1;
                while (l < m && groups[g][0][l] == groups[g][1][l]) {
                    l++;
                }
                CHECK(r.out.blocks[g][0].n_shared == (int) l && r.out.blocks[g][1].n_shared == (int) l,
                      "group %zu shares %d, expected %zu", g, r.out.blocks[g][0].n_shared, l);
                saved += (int) l;
            }
            CHECK(r.out.n_tokens_shared_saved == saved, "saved %d, expected %d", r.out.n_tokens_shared_saved, saved);
            int flat = 0;
            for (auto & g : groups) {
                flat += (int) (g[0].size() + g[1].size());
            }
            CHECK(r.out.n_tokens_decoded == flat - saved, "decoded %d, expected %d", r.out.n_tokens_decoded, flat - saved);
            CHECK(all_free(ctx.get(), cfg8, true), "free after grouped run");
        }

        // ---- 3. waves: W = 3 over groups of 1, 2 and 3 blocks (the fork budget caps a wave)
        {
            auto ctx = make(3);
            common_decide_state st;
            std::vector<std::vector<tokens_t>> groups;
            const tokens_t head = rnd_tokens(gen, 15);
            groups.push_back({ rnd_tokens(gen, 11) });
            groups.push_back({ cat(head, rnd_tokens(gen, 5)), cat(head, rnd_tokens(gen, 7)) });
            groups.push_back({ cat(head, rnd_tokens(gen, 3)), cat(head, rnd_tokens(gen, 9)), cat(head, rnd_tokens(gen, 4)) });
            groups.push_back({ rnd_tokens(gen, 19) });
            groups.push_back({ rnd_tokens(gen, 6), rnd_tokens(gen, 6) });
            auto r = run(ctx.get(), st, cfg_for(3), prefix, groups, false);
            CHECK(r.ok, "waves: %s", r.out.error.c_str());
            compare_rows(fx, "W=3, groups of 1/2/3/1/2 (several waves)", r, prefix, groups, tol);
            CHECK(r.out.n_waves >= 3, "waves = %d", r.out.n_waves);
            CHECK(all_free(ctx.get(), cfg_for(3), true), "free after waves");
        }

        // ---- 4. W = 1: each block alone (the degenerate, sequential-in-the-engine case)
        {
            auto ctx = make(1);
            common_decide_state st;
            std::vector<std::vector<tokens_t>> groups = { { rnd_tokens(gen, 8) }, { rnd_tokens(gen, 14) }, { rnd_tokens(gen, 5) } };
            auto r = run(ctx.get(), st, cfg_for(1), prefix, groups, false);
            CHECK(r.ok, "W=1: %s", r.out.error.c_str());
            compare_rows(fx, "W=1", r, prefix, groups, tol);
            CHECK(r.out.n_waves == 3, "W=1 waves = %d", r.out.n_waves);
            // the two orders of a question at W=1: one block a wave, no sharing, the same rows
            const tokens_t headq = rnd_tokens(gen, 11);
            std::vector<std::vector<tokens_t>> pairs = { { cat(headq, rnd_tokens(gen, 5)), cat(headq, rnd_tokens(gen, 7)) } };
            auto r2 = run(ctx.get(), st, cfg_for(1), prefix, pairs, false);
            CHECK(r2.ok && r2.out.n_waves == 2 && r2.out.blocks[0][0].n_shared == 0, "W=1 on a pair: waves %d", r2.out.n_waves);
            compare_rows(fx, "W=1, a group of two (no sharing)", r2, prefix, pairs, tol);
        }

        // ---- 5. isolation: any order of the groups, and a duplicated block, give the same rows
        {
            auto ctx = make(8);
            common_decide_state st;
            std::vector<std::vector<tokens_t>> groups = { { rnd_tokens(gen, 10) }, { rnd_tokens(gen, 25) },
                                                          { rnd_tokens(gen, 10) }, { rnd_tokens(gen, 3) } };
            auto a = run(ctx.get(), st, cfg8, prefix, groups, false);
            std::vector<std::vector<tokens_t>> rev(groups.rbegin(), groups.rend());
            auto b = run(ctx.get(), st, cfg8, prefix, rev, false);
            CHECK(a.ok && b.ok, "isolation runs");
            double worst = 0.0;
            for (size_t g = 0; g < groups.size(); g++) {
                worst = std::max(worst, max_abs(log_softmax(a.rows[{g, 0}]), log_softmax(b.rows[{groups.size() - 1 - g, 0}])));
            }
            CHECK(worst <= tol, "group order changed a row by %.3e", worst);
            printf("  %-44s max |dlogp| between orders = %.3e\n", "isolation: reversed group order", worst);
            g_worst = std::max(g_worst, worst);

            std::vector<std::vector<tokens_t>> dup = { { groups[0][0] }, { groups[0][0] }, { groups[1][0] } };
            auto c = run(ctx.get(), st, cfg8, prefix, dup, false);
            CHECK(c.ok, "duplicate run: %s", c.out.error.c_str());
            CHECK(max_abs(log_softmax(c.rows[{0, 0}]), log_softmax(c.rows[{1, 0}])) <= tol, "identical blocks differ");
            compare_rows(fx, "duplicated block", c, prefix, dup, tol);

            // two blocks that differ only in the LAST token share all but one token; each keeps its own last token
            tokens_t t1 = rnd_tokens(gen, 12);
            tokens_t t2 = t1;
            t2.back() = (t2.back() + 1) % N_VOCAB;
            std::vector<std::vector<tokens_t>> near = { { t1, t2 } };
            auto d = run(ctx.get(), st, cfg8, prefix, near, false);
            CHECK(d.ok, "near-identical pair: %s", d.out.error.c_str());
            CHECK(d.out.blocks[0][0].n_shared == 11, "shared head %d, expected 11", d.out.blocks[0][0].n_shared);
            compare_rows(fx, "pair differing in the last token only", d, prefix, near, tol);
            // identical blocks: every one keeps >= 1 own token
            std::vector<std::vector<tokens_t>> same = { { t1, t1 } };
            auto e = run(ctx.get(), st, cfg8, prefix, same, false);
            CHECK(e.ok && e.out.blocks[0][0].n_shared == 11, "identical pair shares all but the last token");
            compare_rows(fx, "identical pair", e, prefix, same, tol);
        }

        // ---- 6. prefix reuse: resident, extension, mismatch
        {
            auto ctx = make(4);
            common_decide_state st;
            std::vector<std::vector<tokens_t>> groups = { { rnd_tokens(gen, 7) }, { rnd_tokens(gen, 12) } };
            auto a = run(ctx.get(), st, cfg_for(4), prefix, groups, true);
            CHECK(a.ok && a.out.prefix_resident && st.valid, "kept");
            CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), n_slots) == (llama_pos) prefix.size() - 1, "the prefix sequence holds the prefix");
            CHECK(all_free(ctx.get(), cfg_for(4), false), "forks free while the prefix is kept");
            CHECK(llama_memory_cells_used_max_p1(llama_get_memory(ctx.get())) == prefix.size(), "only the prefix's cells stay in the pool: %u", llama_memory_cells_used_max_p1(llama_get_memory(ctx.get())));

            auto b = run(ctx.get(), st, cfg_for(4), prefix, groups, true);
            CHECK(b.ok && b.out.prefix_reused == 57 && b.out.prefix_processed == 0, "resident prefix reused whole: reused %d processed %d", b.out.prefix_reused, b.out.prefix_processed);
            CHECK(max_abs(log_softmax(a.rows[{0, 0}]), log_softmax(b.rows[{0, 0}])) <= tol, "a reused prefix gave another row");
            compare_rows(fx, "second call, resident prefix", b, prefix, groups, tol);

            const tokens_t ext = cat(prefix, rnd_tokens(gen, 10));
            auto c = run(ctx.get(), st, cfg_for(4), ext, groups, true);
            CHECK(c.ok && c.out.prefix_reused == 57 && c.out.prefix_processed == 10, "extension: reused %d processed %d", c.out.prefix_reused, c.out.prefix_processed);
            compare_rows(fx, "extended prefix (continues the resident state)", c, ext, groups, tol);

            tokens_t other = ext;
            other[30] = (other[30] + 1) % N_VOCAB;                       // diverges inside the resident prefix
            auto d = run(ctx.get(), st, cfg_for(4), other, groups, true);
            CHECK(d.ok && d.out.prefix_reused == 0 && d.out.prefix_processed == (int) other.size(), "mismatch resets: reused %d processed %d", d.out.prefix_reused, d.out.prefix_processed);
            compare_rows(fx, "a different prefix (reset)", d, other, groups, tol);

            tokens_t shorter(other.begin(), other.begin() + 40);         // a prefix of the resident one is not reusable (recurrent state cannot roll back)
            auto e = run(ctx.get(), st, cfg_for(4), shorter, groups, false);
            CHECK(e.ok && e.out.prefix_reused == 0, "a shorter prefix is reset, reused %d", e.out.prefix_reused);
            compare_rows(fx, "a shorter prefix (reset)", e, shorter, groups, tol);
            CHECK(all_free(ctx.get(), cfg_for(4), true) && !st.valid, "released at the end");

            // the prefix sequence is cleared out from under us (another part of the server cleared memory)
            auto f1 = run(ctx.get(), st, cfg_for(4), prefix, groups, true);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            auto f2 = run(ctx.get(), st, cfg_for(4), prefix, groups, false);
            CHECK(f1.ok && f2.ok && f2.out.prefix_reused == 0, "a cleared pool is noticed: reused %d", f2.out.prefix_reused);
            compare_rows(fx, "after the pool was cleared", f2, prefix, groups, tol);

            common_decide_state st2;
            auto g1 = run(ctx.get(), st2, cfg_for(4), prefix, groups, true);
            common_decide_release(ctx.get(), st2, cfg_for(4));
            CHECK(g1.ok && !st2.valid && all_free(ctx.get(), cfg_for(4), true), "release frees the prefix");
            CHECK(llama_memory_cells_used_max_p1(llama_get_memory(ctx.get())) == 0, "and its cells");
        }

        // ---- 7. the slots' own sequences are untouched
        {
            auto ctx = make(4);
            common_decide_state st;
            const tokens_t a_tokens = rnd_tokens(gen, 20);
            llama_memory_t mem = llama_get_memory(ctx.get());
            {
                llama_batch batch = llama_batch_init(32, 0, 1);
                for (size_t i = 0; i < a_tokens.size(); i++) {
                    common_batch_add(batch, a_tokens[i], (llama_pos) i, { 0 }, false);
                }
                CHECK(llama_decode(ctx.get(), batch) == 0, "slot 0 prompt");
                common_batch_clear(batch);
                for (size_t i = 0; i < 12; i++) {
                    common_batch_add(batch, a_tokens[i] , (llama_pos) i, { 1 }, false);
                }
                CHECK(llama_decode(ctx.get(), batch) == 0, "slot 1 prompt");
                llama_batch_free(batch);
            }
            std::vector<std::vector<tokens_t>> groups = { { rnd_tokens(gen, 9), rnd_tokens(gen, 9) }, { rnd_tokens(gen, 15) } };
            auto r = run(ctx.get(), st, cfg_for(4), prefix, groups, false);
            CHECK(r.ok, "batch beside live slots: %s", r.out.error.c_str());
            compare_rows(fx, "batch beside two live slots", r, prefix, groups, tol);
            CHECK(llama_memory_seq_pos_max(mem, 0) == 19 && llama_memory_seq_pos_max(mem, 1) == 11, "slot positions unchanged");

            // slot 0 continues: its next row equals the same prompt decoded cold
            const llama_token next = 5;
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, next, 20, { 0 }, true);
            CHECK(llama_decode(ctx.get(), batch) == 0, "slot 0 continues");
            const float * l = llama_get_logits_ith(ctx.get(), 0);
            const std::vector<float> got(l, l + N_VOCAB);
            llama_batch_free(batch);
            const auto want = fx.reference(cat(a_tokens, { next }));
            const double d = max_abs(log_softmax(got), log_softmax(want));
            CHECK(d <= tol, "slot 0's next row moved by %.3e after a batch", d);
            printf("  %-44s slot 0 next-token row vs cold = %.3e\n", "slots untouched", d);
            g_worst = std::max(g_worst, d);
        }

        // ---- 8. refusals and failures leave nothing behind
        {
            auto ctx = make(2);
            common_decide_state st;
            std::vector<std::vector<tokens_t>> three = { { rnd_tokens(gen, 4), rnd_tokens(gen, 4), rnd_tokens(gen, 4) } };
            const tokens_t head3 = rnd_tokens(gen, 9);
            three = { { cat(head3, rnd_tokens(gen, 4)), cat(head3, rnd_tokens(gen, 6)), cat(head3, rnd_tokens(gen, 5)) } };
            auto a = run(ctx.get(), st, cfg_for(2), prefix, three, false);
            CHECK(a.ok, "3 blocks in a group with 2 forks are split in chunks: %s", a.out.error.c_str());
            compare_rows(fx, "a group larger than the forks (chunks of 2 + 1)", a, prefix, three, tol);
            CHECK(a.out.n_waves == 2, "chunks of 2 and 1 are two waves, got %d", a.out.n_waves);
            CHECK(all_free(ctx.get(), cfg_for(2), true), "free after the split group");

            std::vector<std::vector<tokens_t>> empty_block = { { tokens_t() } };
            auto b = run(ctx.get(), st, cfg_for(2), prefix, empty_block, false);
            CHECK(!b.ok && b.out.error_code == "invalid_request", "an empty block is refused");
            auto c = run(ctx.get(), st, cfg_for(2), tokens_t(), { { rnd_tokens(gen, 3) } }, false);
            CHECK(!c.ok && c.out.error_code == "invalid_request", "an empty prefix is refused");
            auto d = run(ctx.get(), st, cfg_for(5), prefix, { { rnd_tokens(gen, 3) } }, false);
            CHECK(!d.ok && d.out.error_code == "invalid_request", "forks beyond n_seq_max are refused");

            // no cells: a pool of 192 cells holds a 57-token prefix and 3 small blocks but not 3 x 120 tokens
            auto small = fx.make_ctx((uint32_t) common_decide_n_seq_max(n_slots, 3), 192, 64, 16);
            std::vector<std::vector<tokens_t>> big = { { rnd_tokens(gen, 120) }, { rnd_tokens(gen, 120) }, { rnd_tokens(gen, 120) } };
            auto e = run(small.get(), st, cfg_for(3), prefix, big, true);
            CHECK(!e.ok && e.out.error_code == "no_cells", "pool full: %s %s", e.out.error_code.c_str(), e.out.error.c_str());
            CHECK(all_free(small.get(), cfg_for(3), true) && !st.valid, "everything freed after the failure");
            // and the context still works afterwards
            std::vector<std::vector<tokens_t>> ok_groups = { { rnd_tokens(gen, 10) }, { rnd_tokens(gen, 10) } };
            auto f = run(small.get(), st, cfg_for(3), prefix, ok_groups, false);
            CHECK(f.ok, "the context works after a failure: %s", f.out.error.c_str());
            compare_rows(fx, "after a failed call", f, prefix, ok_groups, tol);
        }

        printf("test-decide-batch: worst |dlogp| against a cold single sequence over the run = %.3e\n", g_worst);
        printf("test-decide-batch: %d checks, %d failed\n", n_checks, n_failed);
        return n_failed == 0 ? 0 : 1;
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
