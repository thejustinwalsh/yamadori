// The clef decision head (src/models/clef.cpp) on a tiny random model, CPU backend.
//
//   test-clef                       the self-contained checks below (ctest)
//   test-clef --save PATH           write the fixture model (random weights, the clef layout the converter writes)
//   test-clef MODEL.gguf IN OUT     dump mode: IN has one "token order" pair per line, OUT gets one score per line
//                                   (the first n_options rows of the embeddings output), for the comparison with a
//                                   PyTorch reference of the head (bench/clef_smoke.py in the Yamadori repository)
//
// What the port promises (ggml-org/llama.cpp PRs 29818 + 29831, on a tree that has no llama_batch_ext: the per-token
// order rides in llama_batch.decision_order):
//   - a batch with spans yields one finite score per option, in the first rows of the embeddings output
//   - the scores are reproducible, and move when an option's tokens or a question's tokens change
//   - a batch whose order cannot be used (an option before any question, a question without an option) yields NaN,
//     which the server reports as an error rather than an answer
//   - the order is split with the batch: the same prompt decoded as ONE ubatch gives the same rows whatever n_batch is
//     (a clef batch is a single ubatch, so n_ubatch must hold the whole prompt)

#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
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

// the sizes of the fixture, the head too
static constexpr uint32_t N_EMBD     = 64;
static constexpr uint32_t N_EMBD_H   = 32; // width of the head
static constexpr uint32_t N_FF_H     = 64;
static constexpr uint32_t N_HEAD_DEC = 2;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    // norm weights near 1 and projections of std 0.1, as test-decide-batch: every layer matters to the output
    const bool is_norm = strstr(tensor->name, "norm") != nullptr;
    const bool is_bias = strstr(tensor->name, "bias") != nullptr;
    std::normal_distribution<float> dis(is_norm && !is_bias ? 1.0f : 0.0f, is_norm ? 0.05f : 0.1f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        // decision.scales: prior scale, joint scale, residual gate (the values used at inference)
        if (strcmp(tensor->name, "decision.scales") == 0) {
            tmp[0] = 3.0f; tmp[1] = 5.0f; tmp[2] = 0.5f;
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

// the KVs of a four layer hybrid qwen35 trunk (linear, attention, linear, attention) under the arch's own names, and for
// clef the head of one routing and one joint block
static gguf_context_ptr get_gguf_ctx(const llm_arch arch) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 512;

    const uint32_t n_vocab = N_VOCAB;
    const uint32_t n_embd  = N_EMBD;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 128;
    const uint32_t n_layer = 4;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,   uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-6f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/8, n_embd_head/8, n_embd_head/8, n_embd_head/8}));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            uint32_t(n_embd));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(n_embd / n_head));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           n_head);

    if (arch == LLM_ARCH_CLEF) {
        ms.add_kv(LLM_KV_DECISION_BLOCK_COUNT,         uint32_t(1));
        ms.add_kv(LLM_KV_DECISION_ROUTING_BLOCK_COUNT, uint32_t(1));
        ms.add_kv(LLM_KV_DECISION_HEAD_COUNT,          N_HEAD_DEC);
    }

    // a toy SPM vocabulary, as test-decide-batch: token ids are all the test sends
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
    return ret;
}

// A model made from user data has no tensor listing, and llama_model_clef reads the width of its head from the
// listing (as the converter wrote it), so the fixture is a file: the trunk is a random qwen35 saved by the model saver
// (every tensor name and shape the loader wants, with data), the head is added here, the KVs are clef's.
static bool silent_progress(float, void *) { return true; }

static void fill_random(ggml_tensor * t, std::mt19937 & gen, bool is_norm_weight) {
    std::normal_distribution<float> dis(is_norm_weight ? 1.0f : 0.0f, is_norm_weight ? 0.05f : 0.1f);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        d[i] = dis(gen);
    }
}


static void build_clef_file(const std::string & path, size_t seed) {
    // the trunk
    const std::string path_q = path + ".qwen35";
    {
        gguf_context_ptr gq = get_gguf_ctx(LLM_ARCH_QWEN35);
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_progress;
        std::vector<ggml_backend_dev_t> devs = { nullptr };
        mp.devices = devs.data();
        size_t tmp = seed;
        llama_model_ptr mq(llama_model_init_from_user(gq.get(), set_tensor_data, &tmp, mp));
        if (!mq) {
            throw std::runtime_error("failed to create the qwen35 trunk");
        }
        llama_model_save_to_file(mq.get(), path_q.c_str());
    }
    ggml_context * ctx_trunk = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_trunk };
    gguf_context_ptr src(gguf_init_from_file(path_q.c_str(), ip));
    ggml_context_ptr ctx_trunk_ptr(ctx_trunk);
    if (!src) {
        throw std::runtime_error("failed to read the saved trunk");
    }

    gguf_context_ptr dst = get_gguf_ctx(LLM_ARCH_CLEF);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_trunk); t; t = ggml_get_next_tensor(ctx_trunk, t)) {
        // the saver also wrote the optional tensors the loader made up for the user data (cls.output, the per-tensor
        // scales, the attention biases): a converted checkpoint has none of them, and the fixture should not either
        const std::string name = t->name;
        const auto ends_with = [&](const char * suf) { return name.size() >= strlen(suf) && name.compare(name.size() - strlen(suf), strlen(suf), suf) == 0; };
        if (name == "GGUF tensor data binary blob" || name.rfind("cls.output", 0) == 0 || ends_with(".scale") || ends_with(".input_scale") || (ends_with(".bias") && !ends_with("ssm_dt.bias"))) {
            continue;
        }
        gguf_add_tensor(dst.get(), t);
        gguf_set_tensor_data(dst.get(), t->name, t->data);
    }

    // the head
    ggml_init_params gp = { /*.mem_size =*/ 64u*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context_ptr ctx_head(ggml_init(gp));
    std::mt19937 gen((uint32_t) seed + 1);
    LLM_TN tn(LLM_ARCH_CLEF);
    auto add = [&](const std::string & name, int64_t ne0, int64_t ne1, bool norm_weight = false) {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx_head.get(), GGML_TYPE_F32, ne0, ne1) : ggml_new_tensor_1d(ctx_head.get(), GGML_TYPE_F32, ne0);
        ggml_set_name(t, name.c_str());
        fill_random(t, gen, norm_weight);
        gguf_add_tensor(dst.get(), t);
        gguf_set_tensor_data(dst.get(), t->name, t->data);
        return t;
    };
    auto add_norm = [&](llm_tensor type, int64_t n, int bid) {
        add((bid < 0 ? tn(type, "weight") : tn(type, "weight", bid)).str(), n, 0, true);
        add((bid < 0 ? tn(type, "bias")   : tn(type, "bias",   bid)).str(), n, 0);
    };
    auto add_linear = [&](llm_tensor type, int64_t n_in, int64_t n_out, int bid, bool bias = true) {
        add((bid < 0 ? tn(type, "weight") : tn(type, "weight", bid)).str(), n_in, n_out);
        if (bias) {
            add((bid < 0 ? tn(type, "bias") : tn(type, "bias", bid)).str(), n_out, 0);
        }
    };
    auto add_attn = [&](llm_tensor q, llm_tensor k, llm_tensor v, llm_tensor o, int bid) {
        for (llm_tensor type : { q, k, v, o }) {
            add_linear(type, N_EMBD_H, N_EMBD_H, bid);
        }
    };
    for (int il = 0; il < 2; il++) { // block 0 routes, block 1 is joint
        if (il == 0) {
            add_norm(LLM_TENSOR_DEC_CROSS_ATTN_NORM_KV, N_EMBD_H, il);
        } else {
            add_norm(LLM_TENSOR_DEC_ATTN_NORM, N_EMBD_H, il);
            add_attn(LLM_TENSOR_DEC_ATTN_Q, LLM_TENSOR_DEC_ATTN_K, LLM_TENSOR_DEC_ATTN_V, LLM_TENSOR_DEC_ATTN_OUT, il);
        }
        add_norm(LLM_TENSOR_DEC_CROSS_ATTN_NORM, N_EMBD_H, il);
        add_attn(LLM_TENSOR_DEC_CROSS_ATTN_Q, LLM_TENSOR_DEC_CROSS_ATTN_K, LLM_TENSOR_DEC_CROSS_ATTN_V, LLM_TENSOR_DEC_CROSS_ATTN_OUT, il);
        add_norm(LLM_TENSOR_DEC_FFN_NORM, N_EMBD_H, il);
        add_linear(LLM_TENSOR_DEC_FFN_UP,   N_EMBD_H, N_FF_H,   il);
        add_linear(LLM_TENSOR_DEC_FFN_DOWN, N_FF_H,   N_EMBD_H, il);
    }
    add_norm(LLM_TENSOR_DECISION_HIDDEN_NORM,         N_EMBD,   -1);
    add_norm(LLM_TENSOR_DECISION_OPTION_SUMMARY_NORM, N_EMBD_H, -1);
    add_norm(LLM_TENSOR_DECISION_FIELD_NORM,          N_EMBD_H, -1);
    add_norm(LLM_TENSOR_DECISION_OPTION_NORM,         N_EMBD_H, -1);
    for (llm_tensor type : { LLM_TENSOR_DECISION_PROJ_MEMORY, LLM_TENSOR_DECISION_PROJ_QUESTION, LLM_TENSOR_DECISION_PROJ_OPTION_QUESTION,
                             LLM_TENSOR_DECISION_PROJ_GLOBAL, LLM_TENSOR_DECISION_PROJ_OPTION_CONTEXT, LLM_TENSOR_DECISION_PROJ_OPTION_LEXICAL }) {
        add_linear(type, N_EMBD, N_EMBD_H, -1, false);
    }
    {
        // prior scale, joint scale, residual gate: the values used at inference
        ggml_tensor * t = add(tn(LLM_TENSOR_DECISION_SCALES).str(), 3, 0);
        float * d = (float *) t->data;
        d[0] = 3.0f; d[1] = 5.0f; d[2] = 0.5f;
        gguf_set_tensor_data(dst.get(), t->name, t->data);
    }
    add(tn(LLM_TENSOR_TOKEN_TYPES, "weight").str(), N_EMBD_H, 3);
    add_linear(LLM_TENSOR_DECISION_SCORER,     4 * N_EMBD_H, N_EMBD_H, -1);
    add_linear(LLM_TENSOR_DECISION_SCORER_OUT, N_EMBD_H,     1,        -1);

    if (!gguf_write_to_file(dst.get(), path.c_str(), false)) {
        throw std::runtime_error("failed to write " + path);
    }
    std::remove(path_q.c_str());
}


struct entry {
    llama_token token;
    int32_t     order;
};
using prompt_t = std::vector<entry>;

// the prompt shape of a clef request: text, [question] text, [option] text, [option] text, ... (two questions)
static prompt_t make_prompt(std::mt19937 & gen) {
    std::uniform_int_distribution<int> d(3, N_VOCAB - 1);
    prompt_t p;
    auto run = [&](int32_t order, size_t n) {
        for (size_t i = 0; i < n; i++) {
            p.push_back({ (llama_token) d(gen), order });
        }
    };
    run(LLAMA_DECISION_ORDER_NONE, 9);
    run(LLAMA_DECISION_ORDER_QUESTION_NOUL, 5);
    run(LLAMA_DECISION_ORDER_NONE, 2);
    run(LLAMA_DECISION_ORDER_OPTION, 3);
    run(LLAMA_DECISION_ORDER_NONE, 1);
    run(LLAMA_DECISION_ORDER_OPTION, 2);
    run(LLAMA_DECISION_ORDER_NONE, 3);
    run(LLAMA_DECISION_ORDER_QUESTION_CHOICE, 4);
    run(LLAMA_DECISION_ORDER_NONE, 2);
    run(LLAMA_DECISION_ORDER_OPTION, 2);
    run(LLAMA_DECISION_ORDER_NONE, 1);
    run(LLAMA_DECISION_ORDER_OPTION, 4);
    run(LLAMA_DECISION_ORDER_NONE, 1);
    run(LLAMA_DECISION_ORDER_OPTION, 1);
    run(LLAMA_DECISION_ORDER_NONE, 6);
    return p;
}

static size_t n_options_of(const prompt_t & p) {
    size_t n = 0;
    for (size_t i = 0; i < p.size(); i++) {
        if (p[i].order == LLAMA_DECISION_ORDER_OPTION && (i == 0 || p[i - 1].order != LLAMA_DECISION_ORDER_OPTION)) {
            n++;
        }
    }
    return n;
}

struct runner {
    llama_model * model;

    llama_context_ptr make_ctx(uint32_t n_batch, uint32_t n_ubatch) const {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 512;
        cp.n_batch         = n_batch;
        cp.n_ubatch        = n_ubatch;
        cp.n_seq_max       = 1;
        cp.embeddings      = true;
        cp.pooling_type    = LLAMA_POOLING_TYPE_NONE;
        cp.n_threads       = 4;
        cp.n_threads_batch = 4;
        llama_context_ptr c(llama_init_from_model(model, cp));
        if (!c) {
            throw std::runtime_error("failed to create the context");
        }
        return c;
    }

    // the first n_scores rows of the embeddings output, with the order passed (or not)
    std::vector<float> scores(llama_context * ctx, const prompt_t & p, size_t n_scores, bool with_order = true) const {
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
        llama_batch batch = llama_batch_init((int32_t) p.size(), 0, 1);
        std::vector<int32_t> order(p.size());
        for (size_t i = 0; i < p.size(); i++) {
            common_batch_add(batch, p[i].token, (llama_pos) i, { 0 }, true);
            order[i] = p[i].order;
        }
        batch.decision_order = with_order ? order.data() : nullptr;
        const int ret = llama_decode(ctx, batch);
        batch.decision_order = nullptr;
        llama_batch_free(batch);
        if (ret != 0) {
            throw std::runtime_error("decode failed: " + std::to_string(ret));
        }
        std::vector<float> out;
        for (size_t i = 0; i < n_scores; i++) {
            const float * e = llama_get_embeddings_ith(ctx, (int32_t) i);
            out.push_back(e ? e[0] : NAN);
        }
        return out;
    }
};

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return !v.empty();
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d;
}

static llama_model_ptr load_model(const std::string & path) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    std::vector<ggml_backend_dev_t> devs = { nullptr };   // CPU only
    mp.devices = devs.data();
    mp.n_gpu_layers = 0;
    return llama_model_ptr(llama_model_load_from_file(path.c_str(), mp));
}

int main(int argc, char ** argv) {
    common_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    try {
        // --save PATH: write the fixture model, for the comparison with a PyTorch reference (bench/clef_smoke.py)
        if (argc == 3 && strcmp(argv[1], "--save") == 0) {
            build_clef_file(argv[2], 20261008);
            printf("saved %s\n", argv[2]);
            return 0;
        }

        // ---- dump mode
        if (argc == 4) {
            auto model = load_model(argv[1]);
            if (!model) {
                fprintf(stderr, "failed to load %s\n", argv[1]);
                return 1;
            }
            prompt_t p;
            {
                std::ifstream in(argv[2]);
                long t, o;
                while (in >> t >> o) {
                    p.push_back({ (llama_token) t, (int32_t) o });
                }
            }
            runner r{ model.get() };
            auto ctx = r.make_ctx((uint32_t) p.size(), (uint32_t) p.size());
            const auto s = r.scores(ctx.get(), p, n_options_of(p));
            std::ofstream out(argv[3]);
            out.precision(9);
            for (float x : s) {
                out << x << "\n";
            }
            printf("wrote %zu scores to %s\n", s.size(), argv[3]);
            return 0;
        }

        const std::string path_model = (std::filesystem::temp_directory_path() / "test-clef-tiny.gguf").string();
        build_clef_file(path_model, 20261008);
        auto model = load_model(path_model);
        std::remove(path_model.c_str());
        if (!model) {
            fprintf(stderr, "failed to load the clef fixture\n");
            return 1;
        }
        runner r{ model.get() };
        std::mt19937 gen(20261008);

        const prompt_t p = make_prompt(gen);
        const size_t   n_opt = n_options_of(p);
        CHECK(n_opt == 5, "the fixture prompt has %zu options", n_opt);

        auto ctx = r.make_ctx((uint32_t) p.size(), (uint32_t) p.size());

        // ---- 1. one finite score per option, reproducible
        const auto s1 = r.scores(ctx.get(), p, n_opt);
        const auto s2 = r.scores(ctx.get(), p, n_opt);
        CHECK(all_finite(s1) && s1.size() == n_opt, "scores are not finite");
        CHECK(max_abs_diff(s1, s2) == 0.0, "a repeat differs by %.3e", max_abs_diff(s1, s2));
        double spread = 0.0;
        for (size_t i = 1; i < s1.size(); i++) {
            spread = std::max(spread, (double) std::fabs(s1[i] - s1[0]));
        }
        printf("  scores: ");
        for (float x : s1) printf("%+.4f ", x);
        printf("\n");
        CHECK(spread > 1e-4, "the options all score alike (%.3e): the head does not see them", spread);

        // ---- 2. the scores move with the text of an option, and with the text of a question
        {
            prompt_t q = p;
            for (auto & e : q) {
                if (e.order == LLAMA_DECISION_ORDER_OPTION) {
                    e.token = 3 + (e.token + 17) % (N_VOCAB - 3);
                    break; // the first option only
                }
            }
            const auto s = r.scores(ctx.get(), q, n_opt);
            CHECK(all_finite(s) && max_abs_diff(s, s1) > 1e-4, "another first option leaves the scores as they were (%.3e)", max_abs_diff(s, s1));

            prompt_t u = p;
            for (auto & e : u) {
                if (e.order == LLAMA_DECISION_ORDER_QUESTION_CHOICE) {
                    e.token = 3 + (e.token + 29) % (N_VOCAB - 3);
                }
            }
            const auto su = r.scores(ctx.get(), u, n_opt);
            CHECK(all_finite(su) && max_abs_diff(su, s1) > 1e-4, "another question leaves the scores as they were (%.3e)", max_abs_diff(su, s1));
        }

        // ---- 3. the order is what the head reads: the same tokens with the spans moved score differently
        {
            prompt_t q = p;
            // the second question becomes ordinary text
            for (auto & e : q) {
                if (e.order == LLAMA_DECISION_ORDER_QUESTION_CHOICE) {
                    e.order = LLAMA_DECISION_ORDER_NONE;
                }
            }
            // its three options now belong to the first question
            const auto s = r.scores(ctx.get(), q, n_opt);
            CHECK(all_finite(s) && max_abs_diff(s, s1) > 1e-4, "the question order does not matter (%.3e)", max_abs_diff(s, s1));
        }

        // ---- 4. an order that cannot be used is NaN, not a score
        {
            prompt_t q = p;
            for (auto & e : q) {
                e.order = e.order == LLAMA_DECISION_ORDER_OPTION ? LLAMA_DECISION_ORDER_OPTION : LLAMA_DECISION_ORDER_NONE;
            }
            const auto s = r.scores(ctx.get(), q, 1); // options with no question
            CHECK(s.size() == 1 && std::isnan(s[0]), "options without a question gave %f", s.empty() ? 0.0 : (double) s[0]);

            prompt_t w = p;
            for (auto & e : w) {
                if (e.order == LLAMA_DECISION_ORDER_OPTION) {
                    e.order = LLAMA_DECISION_ORDER_NONE;
                }
            }
            const auto sw = r.scores(ctx.get(), w, 1); // questions with no option
            CHECK(sw.size() == 1 && std::isnan(sw[0]), "questions without an option gave %f", sw.empty() ? 0.0 : (double) sw[0]);
        }

        // ---- 5. no order at all: the head still runs (a batch that is not a decision), one score, finite
        {
            const auto s = r.scores(ctx.get(), p, 1, /*with_order=*/false);
            CHECK(s.size() == 1 && std::isfinite(s[0]), "no order: %f", s.empty() ? 0.0 : (double) s[0]);
        }

        // ---- 6. a larger n_batch and n_ubatch (the same single ubatch) gives the same rows
        {
            auto ctx2 = r.make_ctx(256, 256);
            const auto s = r.scores(ctx2.get(), p, n_opt);
            CHECK(all_finite(s) && max_abs_diff(s, s1) < 5e-4, "n_batch 256: rows differ by %.3e", max_abs_diff(s, s1));
        }

        printf("test-clef: %d checks, %d failed\n", n_checks, n_failed);
        return n_failed == 0 ? 0 : 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "test-clef: %s\n", e.what());
        return 1;
    }
}
