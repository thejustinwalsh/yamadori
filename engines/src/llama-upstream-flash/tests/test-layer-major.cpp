// 0029: layer-major prefill (LLAMA_LAYER_MAJOR) against the ubatch-by-ubatch path, on a small generated qwen4exp
// (Gated DeltaNet + QSA attention layers, hyper-connection residual, PLE, routed experts): the same batches through
// one model in two fresh contexts, the second with the switch on, and the logits of every output, the serialized
// state of the sequence (KV cells, indexer cells, recurrent state) and a few generated tokens must agree.
//
//   test-layer-major                                  CPU only (mode 2): exact equality is the pass condition
//   test-layer-major --device CUDA0 --experts-cpu     the experts in host memory, uploaded by the scheduler
//                                                     (mode 1); the logits' NMSE against the ubatch path must stay
//                                                     below test-llama-archs' 1e-4 (it is 0 when the kernels are the same)
//
// The model's tensors are synthesised from metadata (llama_model_init_from_user), like test-llama-archs does.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "../src/llama-ext.h"   // llama_set_embeddings_nextn: what the MTP draft reads of the target

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <stdlib.h>
static void set_env(const char * k, const char * v) { _putenv_s(k, v ? v : ""); }
#else
static void set_env(const char * k, const char * v) { if (v) setenv(k, v, 1); else unsetenv(k); }
#endif

struct test_args {
    std::string device;          // "" = the CPU
    bool        experts_cpu = false;
    bool        quant       = false;   // Q4_0 experts
    int         mode        = 0;       // layer-major mode of the second run (0: 2 on the CPU, 1 with a device)
    uint32_t    seed        = 1234;
    float       stdev       = 0.1f;
    uint32_t    n_layer     = 8;
    uint32_t    interval    = 4;
    uint32_t    n_expert    = 16;
    uint32_t    n_used      = 4;
    uint32_t    top_k       = 131072;
    uint32_t    n_ubatch    = 64;
    uint32_t    n_batch     = 512;
    uint32_t    n_prompt    = 401;   // first decode: every token an output
    uint32_t    n_prompt2   = 300;   // second decode: the last token only (a server's prompt batch)
    uint32_t    n_gen       = 6;
    uint32_t    rs_seq      = 0;      // recurrent rollback snapshots (what an MTP server runs with): n_rs_seq
    uint32_t    rollback    = 0;      // after the second batch: seq_rm of its last K positions (a rejected draft), then a third batch
    uint32_t    n_prompt3   = 200;    // the batch after the rollback
    int         abort_at    = 0;      // CPU: the abort callback fires from its Nth call (a cancelled batch); 0 = no such run
    bool        nextn       = false;  // llama_set_embeddings_nextn on: the rows an MTP draft takes of the target
    uint32_t    n_seqs      = 1;     // sequences in every batch (equal-length ubatches across them)
    int         threads     = 4;
    std::string kv_type     = "f16";
    std::string save;            // write the synthesised model to this GGUF (see split_gate_up.py)
    std::string load;            // use this GGUF instead of synthesising one
    bool        verbose     = false;
};

static std::vector<std::string> g_log_lines;
static bool g_capture = false;
static bool g_verbose = false;

static void log_cb(enum ggml_log_level level, const char * text, void * /*ud*/) {
    if (g_capture && (strstr(text, "decode_layer_major") || strstr(text, "decode_normal"))) {
        g_log_lines.push_back(text);
    }
    if (g_verbose || level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

struct data_params { uint32_t seed; float stdev; };

static void set_tensor_data(ggml_tensor * t, void * ud) {
    const data_params & p = *(const data_params *) ud;
    size_t seed = p.seed;
    seed ^= std::hash<std::string>()(t->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, p.stdev);

    const bool is_ssm_a = strstr(t->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(t);
    std::vector<float> f(ne);
    for (int64_t i = 0; i < ne; i++) {
        const float v = dis(gen);
        f[i] = is_ssm_a ? -fabsf(v) : v;
    }

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(ne);
        for (int64_t i = 0; i < ne; i++) {
            h[i] = ggml_fp32_to_fp16(f[i]);
        }
        ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
    } else if (ggml_is_quantized(t->type)) {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    } else {
        fprintf(stderr, "unsupported tensor type %s for %s\n", ggml_type_name(t->type), t->name);
        abort();
    }
}

static void kv_u32(gguf_context * c, const char * k, uint32_t v) { gguf_set_val_u32(c, k, v); }
static void kv_arr_u32(gguf_context * c, const char * k, const std::vector<uint32_t> & v) { gguf_set_arr_data(c, k, GGUF_TYPE_UINT32, v.data(), v.size()); }
static void kv_arr_u64(gguf_context * c, const char * k, const std::vector<uint64_t> & v) { gguf_set_arr_data(c, k, GGUF_TYPE_UINT64, v.data(), v.size()); }

// the metadata of a small qwen4exp (the layout of test-llama-archs' fixture, with layers of both kinds and more experts)
static gguf_context * make_metadata(const test_args & a) {
    gguf_context * c = gguf_init_empty();

    const uint32_t n_vocab = 128, n_embd = 256, n_head = 4, n_head_kv = 2, head = 64, n_ff = 384;
    const uint32_t hc = 4;

    gguf_set_val_str(c, "general.architecture", "qwen4exp");
    kv_u32(c, "qwen4exp.vocab_size", n_vocab);
    kv_u32(c, "qwen4exp.context_length", 8192);
    kv_u32(c, "qwen4exp.embedding_length", n_embd);
    kv_u32(c, "qwen4exp.block_count", a.n_layer);
    kv_u32(c, "qwen4exp.feed_forward_length", n_ff);
    kv_u32(c, "qwen4exp.attention.head_count", n_head);
    kv_u32(c, "qwen4exp.attention.head_count_kv", n_head_kv);
    kv_u32(c, "qwen4exp.attention.key_length", head);
    kv_u32(c, "qwen4exp.attention.value_length", head);
    gguf_set_val_f32(c, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-5f);
    kv_u32(c, "qwen4exp.rope.dimension_count", head);
    kv_arr_u32(c, "qwen4exp.rope.dimension_sections", { head/4, head/4, head/4, head/4 });
    gguf_set_val_f32(c, "qwen4exp.rope.freq_base", 10000.0f);

    kv_u32(c, "qwen4exp.expert_count", a.n_expert);
    kv_u32(c, "qwen4exp.expert_used_count", a.n_used);
    kv_u32(c, "qwen4exp.expert_feed_forward_length", 128);
    kv_u32(c, "qwen4exp.expert_shared_feed_forward_length", 64);
    gguf_set_val_f32(c, "qwen4exp.expert_weights_scale", 1.0f);

    kv_u32(c, "qwen4exp.ssm.conv_kernel", 4);
    kv_u32(c, "qwen4exp.ssm.state_size", 32);
    kv_u32(c, "qwen4exp.ssm.group_count", 2);
    kv_u32(c, "qwen4exp.ssm.time_step_rank", 4);
    kv_u32(c, "qwen4exp.ssm.inner_size", 128);
    kv_u32(c, "qwen4exp.full_attention_interval", a.interval);

    kv_u32(c, "qwen4exp.hyper_connection.count", hc);
    kv_u32(c, "qwen4exp.hyper_connection.low_rank", 8);

    kv_u32(c, "qwen4exp.attention.indexer.head_count", 4);
    kv_u32(c, "qwen4exp.attention.indexer.key_length", head);
    kv_u32(c, "qwen4exp.attention.indexer.top_k", a.top_k);
    kv_u32(c, "qwen4exp.attention.indexer.block_size", 4);
    kv_u32(c, "qwen4exp.attention.indexer.local_blocks", 1);
    kv_arr_u32(c, "qwen4exp.attention.compress_ratios", std::vector<uint32_t>(a.n_layer, 4));

    // PLE on a linear-attention layer (the n-gram history is a row of the recurrent cache)
    const uint32_t ple_ngram = 3, ple_hpn = 2, ple_heads = (ple_ngram - 1)*ple_hpn;
    std::vector<uint64_t> off(ple_heads), voc(ple_heads, n_vocab);
    for (uint32_t h = 0; h < ple_heads; h++) { off[h] = (uint64_t) h*n_vocab; }
    kv_arr_u32(c, "qwen4exp.ple.layers", { 1 });
    kv_u32(c, "qwen4exp.ple.ngram_size", ple_ngram);
    kv_u32(c, "qwen4exp.ple.heads_per_ngram", ple_hpn);
    kv_u32(c, "qwen4exp.ple.conv_kernel", 4);
    kv_u32(c, "qwen4exp.ple.eos_token_id", 0);
    kv_u32(c, "qwen4exp.embedding_length_per_layer_input", n_embd/ple_heads);
    kv_arr_u64(c, "qwen4exp.ple.layer_multipliers", { 1, 3, 5 });
    kv_arr_u64(c, "qwen4exp.ple.head_offsets", off);
    kv_arr_u64(c, "qwen4exp.ple.head_vocab_sizes", voc);

    // a dummy tokenizer
    {
        std::vector<std::string> toks(n_vocab);
        std::vector<const char *> ptrs(n_vocab);
        std::vector<float> scores(n_vocab, 0.0f);
        for (uint32_t i = 0; i < n_vocab; i++) {
            toks[i] = "tok_" + std::to_string(i);
            ptrs[i] = toks[i].c_str();
        }
        gguf_set_val_str(c, "tokenizer.ggml.model", "test");
        gguf_set_arr_str(c, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
        gguf_set_arr_data(c, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    }

    if (a.quant) {
        // a tensor of the same name and another type makes the loader create that type
        for (uint32_t il = 0; il < a.n_layer; il++) {
            for (const char * w : { "gate_up", "gate", "up", "down" }) {
                ggml_tensor t;
                memset(&t, 0, sizeof(t));
                t.type = GGML_TYPE_Q4_0;
                ggml_format_name(&t, "blk.%u.ffn_%s_exps.weight", il, w);
                gguf_add_tensor(c, &t);
            }
        }
    }

    return c;
}

struct run_result {
    std::vector<std::vector<float>> logits;   // one entry per decode call (all its output rows)
    std::vector<uint8_t>            state1, state2, state3;
    std::vector<float>              nextn;    // the hidden state of every token of the two batches (--nextn)
    std::vector<std::string>        lines;    // the layer-major debug lines
    bool                            ok = false;
};

// one batch of n_seq sequences, each with the same number of tokens, sequence by sequence (llama_decode splits it into
// ubatches of equal length per sequence); out[i] says whether token i of every sequence has an output; the output
// rows come back in batch order
static bool decode_seqs(llama_context * ctx, const std::vector<std::vector<llama_token>> & toks, int32_t pos0,
        const std::vector<bool> & out, std::vector<float> & logits_out, uint32_t n_vocab) {
    const size_t n_seq = toks.size();
    const size_t n_per = toks[0].size();
    llama_batch b = llama_batch_init((int32_t) (n_seq*n_per), 0, 1);
    size_t k = 0;
    for (size_t s = 0; s < n_seq; s++) {
        for (size_t i = 0; i < n_per; i++, k++) {
            b.token[k]     = toks[s][i];
            b.pos[k]       = pos0 + (int32_t) i;
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = (llama_seq_id) s;
            b.logits[k]    = out[i];
        }
    }
    b.n_tokens = (int32_t) k;

    const int rc = llama_decode(ctx, b);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        llama_batch_free(b);
        return false;
    }

    logits_out.clear();
    k = 0;
    for (size_t s = 0; s < n_seq; s++) {
        for (size_t i = 0; i < n_per; i++, k++) {
            if (!out[i]) {
                continue;
            }
            const float * l = llama_get_logits_ith(ctx, (int32_t) k);
            logits_out.insert(logits_out.end(), l, l + n_vocab);
        }
    }
    llama_batch_free(b);
    return true;
}

static std::vector<uint8_t> get_state(llama_context * ctx, int n_seq) {
    std::vector<uint8_t> all;
    for (int s = 0; s < n_seq; s++) {
        std::vector<uint8_t> v(llama_state_seq_get_size(ctx, s));
        const size_t n = llama_state_seq_get_data(ctx, v.data(), v.size(), s);
        v.resize(n);
        all.insert(all.end(), v.begin(), v.end());
    }
    return all;
}

static run_result run(llama_model * model, const test_args & a, const char * layer_major, const std::vector<llama_token> & prompt) {
    run_result r;

    set_env("LLAMA_LAYER_MAJOR", layer_major);
    set_env("LLAMA_LAYER_MAJOR_DEBUG", "1");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx            = 4096;
    cp.n_batch          = a.n_batch;
    cp.n_ubatch         = a.n_ubatch;
    cp.n_seq_max        = a.n_seqs;
    cp.n_rs_seq         = a.rs_seq;
    cp.n_threads        = a.threads;
    cp.n_threads_batch  = a.threads;
    cp.kv_unified       = true;
    if (a.kv_type == "q8_0") {
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q8_0;
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return r;
    }

    if (a.nextn) {
        llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
    }
    const uint32_t n_embd_out = llama_model_n_embd_out(model);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_seq   = a.n_seqs;
    const uint32_t stride  = a.n_prompt + a.n_prompt2 + a.n_prompt3 + 8;

    // sequence s reads its own stretch of the prompt array
    auto slice = [&](uint32_t s, uint32_t from, uint32_t n) {
        return std::vector<llama_token>(prompt.begin() + s*stride + from, prompt.begin() + s*stride + from + n);
    };

    g_capture = true;
    g_log_lines.clear();

    std::vector<float> lg;
    bool ok = true;

    // 1. a prompt batch with an output on every token
    {
        std::vector<std::vector<llama_token>> t;
        for (uint32_t s = 0; s < n_seq; s++) { t.push_back(slice(s, 0, a.n_prompt)); }
        ok = ok && decode_seqs(ctx, t, 0, std::vector<bool>(a.n_prompt, true), lg, n_vocab);
        r.logits.push_back(lg);
        r.state1 = get_state(ctx, (int) n_seq);
        for (uint32_t i = 0; a.nextn && i < n_seq*a.n_prompt; i++) {
            const float * e = llama_get_embeddings_nextn_ith(ctx, (int32_t) i);
            r.nextn.insert(r.nextn.end(), e, e + n_embd_out);
        }
    }

    // 2. the next batch, a server's: only the last token has an output
    if (ok) {
        std::vector<std::vector<llama_token>> t;
        for (uint32_t s = 0; s < n_seq; s++) { t.push_back(slice(s, a.n_prompt, a.n_prompt2)); }
        std::vector<bool> out(a.n_prompt2, false);
        out.back() = true;
        ok = ok && decode_seqs(ctx, t, (int32_t) a.n_prompt, out, lg, n_vocab);
        r.logits.push_back(lg);
        r.state2 = get_state(ctx, (int) n_seq);
        for (uint32_t i = 0; a.nextn && i < n_seq*a.n_prompt2; i++) {
            const float * e = llama_get_embeddings_nextn_ith(ctx, (int32_t) i);
            r.nextn.insert(r.nextn.end(), e, e + n_embd_out);
        }
    }

    // 2b. a rejected draft: the last K positions leave the memory (the recurrent state goes back to a snapshot, used once
    // by the next batch), and a batch of several ubatches continues from there
    int32_t pos_end = (int32_t) (a.n_prompt + a.n_prompt2);
    if (ok && a.rollback > 0) {
        bool all_rm = true;
        for (uint32_t s = 0; s < n_seq; s++) {
            all_rm = llama_memory_seq_rm(llama_get_memory(ctx), (llama_seq_id) s, pos_end - (int32_t) a.rollback, -1) && all_rm;
        }
        if (!all_rm) {
            fprintf(stderr, "seq_rm of the last %u positions failed (n_rs_seq %u)\n", a.rollback, a.rs_seq);
            ok = false;
        }
        pos_end -= (int32_t) a.rollback;

        std::vector<std::vector<llama_token>> t;
        for (uint32_t s = 0; s < n_seq; s++) { t.push_back(slice(s, a.n_prompt + a.n_prompt2, a.n_prompt3)); }
        std::vector<bool> out(a.n_prompt3, false);
        out.back() = true;
        ok = ok && decode_seqs(ctx, t, pos_end, out, lg, n_vocab);
        r.logits.push_back(lg);
        pos_end += (int32_t) a.n_prompt3;
    }

    // 3. generated tokens, one at a time per sequence (the ordinary path, on the state the batches left)
    if (ok) {
        int32_t pos = pos_end;
        for (uint32_t g = 0; ok && g < a.n_gen; g++) {
            const std::vector<float> & last = r.logits.back();   // one row per sequence
            std::vector<std::vector<llama_token>> t;
            for (uint32_t s = 0; s < n_seq; s++) {
                const float * row = last.data() + (size_t) s*n_vocab;
                t.push_back({ (llama_token) (std::max_element(row, row + n_vocab) - row) });
            }
            ok = ok && decode_seqs(ctx, t, pos++, { true }, lg, n_vocab);
            r.logits.push_back(lg);
        }
        r.state3 = get_state(ctx, (int) n_seq);
    }

    r.lines = g_log_lines;
    g_capture = false;
    llama_free(ctx);
    r.ok = ok;
    return r;
}

struct abort_state { int calls = 0; int from = 0; };

static bool abort_cb(void * ud) {
    abort_state * st = (abort_state *) ud;
    return st->from > 0 && ++st->calls >= st->from;
}

// a prompt batch cancelled in the middle of a layer-major chunk: decode answers 2, the sequence leaves the memory (the
// batch started at position 0, so nothing of it may stay: the recurrent state and the K/V are cleared), and the same
// batch decoded again gives exactly the reference's rows and state
static bool run_abort(llama_model * model, const test_args & a, const char * layer_major, const std::vector<llama_token> & prompt,
        const run_result & ref) {
    set_env("LLAMA_LAYER_MAJOR", layer_major);
    set_env("LLAMA_LAYER_MAJOR_DEBUG", "1");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = a.n_batch; cp.n_ubatch = a.n_ubatch; cp.n_seq_max = 1;
    cp.n_threads = a.threads; cp.n_threads_batch = a.threads; cp.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return false;
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    abort_state st;
    st.from = a.abort_at;
    llama_set_abort_callback(ctx, abort_cb, &st);

    std::vector<std::vector<llama_token>> t(1, std::vector<llama_token>(prompt.begin(), prompt.begin() + a.n_prompt));
    std::vector<float> lg;
    g_capture = false;
    const bool first = decode_seqs(ctx, t, 0, std::vector<bool>(a.n_prompt, true), lg, n_vocab);   // false: it failed
    const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    printf("abort: the batch %s after %d abort-callback calls, the sequence's last position is %d\n",
            first ? "was not cancelled" : "was cancelled", st.calls, (int) pmax);

    st.from = 0;   // let it run
    const bool second = decode_seqs(ctx, t, 0, std::vector<bool>(a.n_prompt, true), lg, n_vocab);
    const bool same = second && lg.size() == ref.logits[0].size() &&
        memcmp(lg.data(), ref.logits[0].data(), lg.size()*sizeof(float)) == 0;
    const bool same_state = get_state(ctx, 1) == ref.state1;
    llama_free(ctx);

    int fails = 0;
    auto check = [&](bool ok, const char * what) { printf("%s: %s\n", ok ? "PASS" : "FAIL", what); fails += ok ? 0 : 1; };
    check(!first, "a cancelled layer-major batch is reported (llama_decode != 0)");
    check(pmax == -1, "and leaves nothing of the sequence in the memory");
    check(same && same_state, "the batch decoded again equals the reference (logits and state)");
    return fails == 0;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double e = 0.0, n = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        e += (double) (a[i] - b[i])*(a[i] - b[i]);
        n += (double) a[i]*a[i];
    }
    return n > 0 ? e/n : e;
}

int main(int argc, char ** argv) {
    test_args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if      (s == "--device")      { a.device = next(); }
        else if (s == "--experts-cpu") { a.experts_cpu = true; }
        else if (s == "--quant")       { a.quant = true; }
        else if (s == "--mode")        { a.mode = atoi(next()); }
        else if (s == "--seed")        { a.seed = (uint32_t) atoi(next()); }
        else if (s == "--stdev")       { a.stdev = (float) atof(next()); }
        else if (s == "--layers")      { a.n_layer = (uint32_t) atoi(next()); }
        else if (s == "--interval")    { a.interval = (uint32_t) atoi(next()); }
        else if (s == "--experts")     { a.n_expert = (uint32_t) atoi(next()); }
        else if (s == "--used")        { a.n_used = (uint32_t) atoi(next()); }
        else if (s == "--top-k")       { a.top_k = (uint32_t) atoi(next()); }
        else if (s == "--ub")          { a.n_ubatch = (uint32_t) atoi(next()); }
        else if (s == "--batch")       { a.n_batch = (uint32_t) atoi(next()); }
        else if (s == "--prompt")      { a.n_prompt = (uint32_t) atoi(next()); }
        else if (s == "--prompt2")     { a.n_prompt2 = (uint32_t) atoi(next()); }
        else if (s == "--gen")         { a.n_gen = (uint32_t) atoi(next()); }
        else if (s == "--seqs")        { a.n_seqs = (uint32_t) atoi(next()); }
        else if (s == "--nextn")       { a.nextn = true; }
        else if (s == "--abort-at")    { a.abort_at = atoi(next()); }
        else if (s == "--rs-seq")      { a.rs_seq = (uint32_t) atoi(next()); }
        else if (s == "--rollback")    { a.rollback = (uint32_t) atoi(next()); }
        else if (s == "--prompt3")     { a.n_prompt3 = (uint32_t) atoi(next()); }
        else if (s == "--threads")     { a.threads = atoi(next()); }
        else if (s == "--kv")          { a.kv_type = next(); }
        else if (s == "--save")        { a.save = next(); }
        else if (s == "--model")       { a.load = next(); }
        else if (s == "-v")            { a.verbose = true; g_verbose = true; }
        else { fprintf(stderr, "unknown argument %s\n", s.c_str()); return 2; }
    }
    if (a.mode == 0) {
        a.mode = a.device.empty() ? 2 : 1;
    }

    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    gguf_context * meta = make_metadata(a);

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = [](float, void *) { return true; };
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (!a.device.empty()) {
        devs[0] = ggml_backend_dev_by_name(a.device.c_str());
        if (!devs[0]) {
            fprintf(stderr, "no device %s\n", a.device.c_str());
            return 2;
        }
        mp.devices = devs;
        mp.n_gpu_layers = 999;
    } else {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        devs[0] = cpu;
        mp.devices = devs;
        mp.n_gpu_layers = 0;
    }
    llama_model_tensor_buft_override ov[2] = { { "ffn_(gate_up|up|down|gate)_exps", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };
    if (a.experts_cpu) {
        mp.tensor_buft_overrides = ov;
    }

    data_params dp = { a.seed, a.stdev };
    llama_model * model = a.load.empty() ? llama_model_init_from_user(meta, set_tensor_data, &dp, mp)
                                         : llama_model_load_from_file(a.load.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to create the model\n");
        return 2;
    }
    if (!a.save.empty()) {
        llama_model_save_to_file(model, a.save.c_str());
        printf("saved %s\n", a.save.c_str());
        llama_model_free(model);
        gguf_free(meta);
        return 0;
    }

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 gen(a.seed);
    std::uniform_int_distribution<int> dis(0, (int) n_vocab - 1);
    std::vector<llama_token> prompt((size_t) (a.n_prompt + a.n_prompt2 + a.n_prompt3 + 8)*a.n_seqs);
    for (auto & t : prompt) { t = dis(gen); }

    printf("model: %u layers (attention every %u), %u experts (%u used), top_k %u, experts %s%s, ubatch %u, batch %u, "
           "prompt %u + %u (x%u sequences), device %s\n", a.n_layer, a.interval, a.n_expert, a.n_used, a.top_k,
           a.experts_cpu ? "in host memory" : "with the model", a.quant ? " (Q4_0)" : "", a.n_ubatch, a.n_batch,
           a.n_prompt, a.n_prompt2, a.n_seqs, a.device.empty() ? "CPU" : a.device.c_str());

    run_result ref = run(model, a, nullptr, prompt);
    run_result lmr = run(model, a, std::to_string(a.mode).c_str(), prompt);

    // a control the comparison must be able to fail: the same layer-major run with the ubatches of every layer past the
    // first taken back to front (the recurrent state and the K/V of a layer would see the batch in the wrong order)
    run_result brk;
    // mode 1 is for experts in host memory: with the experts on the device the switch must leave the batch alone
    const bool expect_lm = a.n_prompt > a.n_ubatch && !(a.mode == 1 && !(a.experts_cpu && !a.device.empty()));
    const bool multi = expect_lm;
    if (multi) {
        set_env("LLAMA_LAYER_MAJOR_BREAK", "1");
        brk = run(model, a, std::to_string(a.mode).c_str(), prompt);
        set_env("LLAMA_LAYER_MAJOR_BREAK", nullptr);
    }
    if (!ref.ok || !lmr.ok) {
        printf("FAIL: a run did not finish\n");
        return 1;
    }

    // the layer-major path must have run (not silently fallen back)
    printf("layer-major lines: %zu\n", lmr.lines.size());
    for (const auto & l : lmr.lines) { printf("  %s", l.c_str()); if (l.back() != '\n') printf("\n"); }

    int fails = 0;
    auto check = [&](bool ok, const char * what) { printf("%s: %s\n", ok ? "PASS" : "FAIL", what); fails += ok ? 0 : 1; };

    {
        size_t n_lm = 0;
        for (const auto & l : lmr.lines) { n_lm += l.find("decode_layer_major") != std::string::npos; }
        if (expect_lm) {
            check(n_lm > 0, "layer-major prefill ran");
        } else {
            check(n_lm == 0, "layer-major prefill stayed off (one ubatch, or mode 1 without experts in host memory)");
        }
    }
    printf("ubatch by ubatch (the reference run):\n");
    for (const auto & l : ref.lines) { printf("  %s", l.c_str()); if (l.back() != '\n') printf("\n"); }

    bool finite = true, argmax_same = true, exact = true;
    double worst_nmse = 0.0, worst_abs = 0.0;
    for (size_t k = 0; k < ref.logits.size(); k++) {
        const auto & x = ref.logits[k];
        const auto & y = lmr.logits[k];
        if (x.size() != y.size() || x.empty()) { finite = false; continue; }
        for (float v : x) { finite = finite && std::isfinite(v); }
        for (float v : y) { finite = finite && std::isfinite(v); }
        const size_t rows = x.size()/n_vocab;
        for (size_t r = 0; r < rows; r++) {
            const float * px = x.data() + r*n_vocab;
            const float * py = y.data() + r*n_vocab;
            argmax_same = argmax_same && (std::max_element(px, px + n_vocab) - px) == (std::max_element(py, py + n_vocab) - py);
        }
        exact = exact && memcmp(x.data(), y.data(), x.size()*sizeof(float)) == 0;
        worst_nmse = std::max(worst_nmse, nmse(x, y));
        for (size_t i = 0; i < x.size(); i++) { worst_abs = std::max(worst_abs, (double) fabsf(x[i] - y[i])); }
    }
    printf("logits: %zu decode calls, worst NMSE %.3e, worst |diff| %.3e, bit-exact %s\n", ref.logits.size(), worst_nmse, worst_abs, exact ? "yes" : "no");
    if (a.nextn) {
        const bool same = ref.nextn.size() == lmr.nextn.size() && !ref.nextn.empty() &&
            memcmp(ref.nextn.data(), lmr.nextn.data(), ref.nextn.size()*sizeof(float)) == 0;
        printf("nextn (MTP) hidden states: %zu floats, %s\n", ref.nextn.size(), same ? "identical" : "DIFFER");
        check(same || !a.device.empty(), "the hidden state of every token (what the MTP draft reads) is identical");
        check(nmse(ref.nextn, lmr.nextn) <= 1e-4, "hidden states' NMSE <= 1e-4");
    }
    check(finite, "every logit is finite");
    check(argmax_same, "the argmax of every output row agrees");
    check(worst_nmse <= 1e-4, "logits' NMSE <= 1e-4 (test-llama-archs' limit)");
    if (a.device.empty()) {
        check(exact, "CPU: logits bit-exact");
    }

    auto state_cmp = [&](const std::vector<uint8_t> & x, const std::vector<uint8_t> & y, const char * what) {
        size_t diff = 0, first = (size_t) -1;
        const size_t n = std::min(x.size(), y.size());
        for (size_t i = 0; i < n; i++) {
            if (x[i] != y[i]) { diff++; if (first == (size_t) -1) first = i; }
        }
        printf("state %s: %zu vs %zu bytes, %zu differ%s\n", what, x.size(), y.size(), diff + (x.size() != y.size() ? 1 : 0),
               first == (size_t) -1 ? "" : (" (first at " + std::to_string(first) + ")").c_str());
        return x.size() == y.size() && diff == 0;
    };
    const bool s1 = state_cmp(ref.state1, lmr.state1, "after the prompt batch");
    const bool s2 = state_cmp(ref.state2, lmr.state2, "after the server-style batch");
    const bool s3 = state_cmp(ref.state3, lmr.state3, "after the generated tokens");
    if (a.device.empty()) {
        check(s1 && s2 && s3, "CPU: sequence state (KV, indexer, recurrent) byte-identical");
    } else {
        printf("%s: sequence state byte-identical on the device (informational)\n", s1 && s2 && s3 ? "PASS" : "NOTE");
    }

    if (a.abort_at > 0 && multi) {
        check(run_abort(model, a, std::to_string(a.mode).c_str(), prompt, ref), "a cancelled batch: rolled back and repeatable");
    }

    if (multi) {
        bool logits_differ = false;
        for (size_t k = 0; k < ref.logits.size() && k < brk.logits.size(); k++) {
            logits_differ = logits_differ || ref.logits[k].size() != brk.logits[k].size() ||
                memcmp(ref.logits[k].data(), brk.logits[k].data(), ref.logits[k].size()*sizeof(float)) != 0;
        }
        const bool state_differs = ref.state3 != brk.state3;
        printf("control (layers past the first take the ubatches back to front): logits differ %s, state differs %s\n",
                logits_differ ? "yes" : "no", state_differs ? "yes" : "no");
        check(brk.ok && (logits_differ || state_differs), "the control is detected (the comparison can fail)");
    }

    // the activations are not degenerate: the rows differ from one another and the argmax moves
    {
        std::vector<llama_token> am;
        const auto & x = ref.logits[0];
        for (size_t r = 0; r < x.size()/n_vocab; r++) {
            am.push_back((llama_token) (std::max_element(x.data() + r*n_vocab, x.data() + (r + 1)*n_vocab) - (x.data() + r*n_vocab)));
        }
        std::sort(am.begin(), am.end());
        const size_t distinct = std::unique(am.begin(), am.end()) - am.begin();
        double mean = 0, var = 0;
        for (float v : x) { mean += v; }
        mean /= x.size();
        for (float v : x) { var += (v - mean)*(v - mean); }
        var /= x.size();
        printf("activations: logit std %.3g over %zu rows, %zu distinct argmax tokens\n", sqrt(var), x.size()/n_vocab, distinct);
        check(var > 1e-8 && distinct > 1, "the logits are not degenerate");
    }

    llama_model_free(model);
    gguf_free(meta);

    printf(fails == 0 ? "all passed\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
