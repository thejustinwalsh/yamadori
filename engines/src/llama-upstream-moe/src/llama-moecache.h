#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router;
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[512] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0
//    through the cache chain (zero slot) and cached ids contribute 0 through
//    the CPU chain (skip), so the result is exact.
//  - llama_moe_cache_step(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).
//
// Yamadori 0004, ported from Strata (github.com/Niko1221/Strata @ d551edf4, no licence file; used with the
// operator's authorisation, 2026-09-28):
//  - LLAMA_MOE_CACHE_PROFILE=<file>: the resident experts are chosen from an expert profile (Strata's STRP
//    format: src/core/expert_cache.cpp:12-72 read_expert_profile; data/expert-profile.bin;
//    tools/make_profile.py). The profile ranks (layer, expert) pairs by routing frequency; the cache takes
//    n_slots x (host-resident expert layers) pairs in rank order, so a layer holds as many slots as the ranking
//    gives it, filled before the first decode (Strata's blocking startup fill,
//    include/strata/core/expert_cache.hpp:112-118). Without a profile every layer gets n_slots, empty, as in
//    #27861.
//  - LLAMA_MOE_CACHE_POLICY=strata: promotions at deliberate moments instead of per-step LRU inserts --
//    Strata's adaptive tier (src/program/generate.cpp:2418-2457, the `adapt` lambda; usage counted per routed
//    id at src/core/expert_source.cpp:275-277): every LLAMA_MOE_CACHE_ADAPT_EVERY decode steps (Strata's
//    --adapt-every, default 4) and only when the previous swaps have landed, per layer the missing experts with
//    usage >= 2 replace the resident ones with the least usage while the gain is >= 1.5, the best
//    LLAMA_MOE_CACHE_ADAPT_SWAPS (Strata's --adapt-swaps, default 96) swaps overall; the victim is evicted at
//    once (the CPU computes it meanwhile) and the newcomer admitted when its upload has landed; then usage *=
//    0.7. Every constant is Strata's (generate.cpp:246, 258, 2430, 2439, 2456).
//  - the cache serves every batch the CPU computes: n_tokens < ggml-cuda's op-offload batch
//    (GGML_OP_OFFLOAD_MIN_BATCH, default 32, ggml-cuda.cu), above which the op runs on the GPU with the
//    experts copied in (and would ignore the skip table). #27861 served n_tokens == 1 only.

#include <cstdint>

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// apply throttled LRU updates; call between graph executions only
void llama_moe_cache_step();

// the largest batch the cache chain is built for (0004): one less than ggml-cuda's op-offload batch
int64_t llama_moe_cache_max_tokens();
