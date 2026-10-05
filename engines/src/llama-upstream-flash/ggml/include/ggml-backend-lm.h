#pragma once

// 0029: layer-major prefill (LLAMA_LAYER_MAJOR): weight-stationary expert slots in the backend scheduler.
//
// An offloaded MUL_MAT_ID (a MoE layer whose experts live in host memory) normally copies the experts its ubatch uses
// to the device in front of every graph: with 512-token ubatches that is nearly every expert of the layer, once per
// ubatch. Layer-major prefill runs the graph of ONE layer for each ubatch of a chunk in a row; with the slots on, the
// scheduler uploads that layer's expert tensors whole into a slot on first use and every later graph finds them there.
// The caller marks layers (an epoch) and names the next layer's weights so that their upload overlaps this layer's
// compute. The slots are device memory: n_slots x slot_bytes.

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct ggml_backend_sched_upload_stats {
        uint64_t resident_bytes;      // uploaded into slots (demand + prefetch)
        uint64_t resident_uploads;
        uint64_t resident_hits;       // splits that found their weight in a slot
        uint64_t resident_fallbacks;  // splits that could not use a slot and took the used-experts copy
        uint64_t resident_prefetched; // uploads queued by ggml_backend_sched_resident_prefetch
        uint64_t sparse_bytes;        // bytes of the used-experts copy path (always counted)
        uint64_t sparse_copies;
        uint64_t full_bytes;          // bytes of host weights copied WHOLE and synchronously (always counted)
        uint64_t full_copies;
    };

    // n_slots < 3 or slot_bytes == 0: off (the slots are freed). Slots are allocated on the first split that uses them.
    GGML_API void ggml_backend_sched_resident_set(ggml_backend_sched_t sched, int n_slots, size_t slot_bytes);
    GGML_API bool ggml_backend_sched_resident_is_on(ggml_backend_sched_t sched);

    // a new layer begins: the slots stamped with an earlier epoch may be reused
    GGML_API void ggml_backend_sched_resident_epoch(ggml_backend_sched_t sched);

    // queue the uploads of the next layer's host weights now (async, on the upload stream)
    GGML_API void ggml_backend_sched_resident_prefetch(ggml_backend_sched_t sched, struct ggml_tensor * const * weights, int n);

    GGML_API void ggml_backend_sched_upload_stats(ggml_backend_sched_t sched, struct ggml_backend_sched_upload_stats * stats);

#ifdef __cplusplus
}
#endif
