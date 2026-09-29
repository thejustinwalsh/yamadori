#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending

    std::vector<float>    usage;          // 0004 (Strata's adaptive tier): decayed routed count per expert

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;

    // 0004: Strata's adaptive tier (generate.cpp:246 adapt_every = 4, :258 adapt_swaps = 96)
    bool    policy_strata = false;
    int32_t adapt_every   = 4;
    int32_t adapt_swaps   = 96;
    int64_t max_tokens    = 31;
    size_t  in_flight     = 0;   // swaps queued and not yet published

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > mc->max_tokens) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            if (!ls->usage.empty()) {
                ls->usage[id] += 1.0f; // Strata expert_source.cpp:277: one per routed id
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

// Strata's STRP expert profile (src/core/expert_cache.cpp:12-72, read_expert_profile): "STRP", then uint32
// version, n_layers, n_expert, slots, n_ranked, then n_ranked (uint16 layer, uint16 expert) pairs ranked by
// descending routing frequency, then (unread) the frequency table. Checked against the model's shape.
bool read_expert_profile(const char * path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>> & ranked, std::string & err) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        err = std::string("cannot open ") + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (fread(magic, 1, 4, f) != 4 || fread(hdr, 4, 5, f) != 5 || memcmp(magic, "STRP", 4) != 0) {
        fclose(f);
        err = std::string(path) + " is not an STRP expert profile";
        return false;
    }
    const uint32_t nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        fclose(f);
        err = std::string(path) + " is " + std::to_string(nl) + "x" + std::to_string(ne) + " but this model is " +
              std::to_string(n_layers) + "x" + std::to_string(n_expert) + ": a profile for a different model";
        return false;
    }
    if (n_ranked > want) {
        fclose(f);
        err = "the header claims more ranked pairs than slots";
        return false;
    }
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && fread(raw.data(), 2, raw.size(), f) != raw.size()) {
        fclose(f);
        err = "the ranked list is truncated";
        return false;
    }
    fclose(f);
    ranked.clear();
    ranked.reserve(n_ranked);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = raw[(size_t) i*2], e = raw[(size_t) i*2 + 1];
        if (l >= n_layers || e >= n_expert) {
            err = "pair " + std::to_string(i) + " is out of range";
            return false;
        }
        ranked.emplace_back(l, e);
    }
    return true;
}

int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && v[0] ? atoi(v) : def;
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // 0004: the slots each layer gets, and the experts that start resident (from the profile)
        std::map<int, int32_t> slots_of;
        std::map<int, std::vector<int32_t>> seed_of;
        {
            int64_t n_cached = 0;
            for (auto & g : groups) {
                for (auto & c : g.second) {
                    slots_of[c.il] = n_slots;
                    n_cached++;
                }
            }
            const char * prof = getenv("LLAMA_MOE_CACHE_PROFILE");
            if (prof && prof[0]) {
                std::vector<std::pair<int32_t, int32_t>> ranked;
                std::string err;
                const int64_t n_expert = model.layers[slots_of.begin()->first].ffn_up_exps->ne[2];
                if (!read_expert_profile(prof, (int64_t) model.hparams.n_layer(), n_expert, ranked, err)) {
                    LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_PROFILE: %s - uniform %d slots per layer, empty\n",
                            __func__, err.c_str(), n_slots);
                } else {
                    int64_t budget = (int64_t) n_slots * n_cached;
                    for (auto & kv : slots_of) {
                        kv.second = 0;
                    }
                    for (const auto & p : ranked) {
                        if (budget == 0) {
                            break;
                        }
                        auto it = slots_of.find(p.first);
                        if (it == slots_of.end()) {
                            continue; // that layer's experts are on a device already
                        }
                        it->second++;
                        seed_of[p.first].push_back(p.second);
                        budget--;
                    }
                    int32_t lo = INT32_MAX, hi = 0;
                    for (auto & kv : slots_of) {
                        lo = std::min(lo, kv.second);
                        hi = std::max(hi, kv.second);
                    }
                    LLAMA_LOG_INFO("%s: expert profile %s: %zu ranked pairs, %" PRId64 " slots over %" PRId64
                            " layers (%d-%d per layer)\n", __func__, prof, ranked.size(),
                            (int64_t) n_slots * n_cached - budget, n_cached, lo, hi);
                }
            }
            // a layer the profile gives no slot is not cached at all
            for (auto & g : groups) {
                g.second.erase(std::remove_if(g.second.begin(), g.second.end(),
                        [&](const cand & c) { return slots_of[c.il] <= 0; }), g.second.end());
            }
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = slots_of[c.il];
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    const int32_t ns = ls->pub.n_slots;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], ns + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], ns + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], ns + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // 0004: Strata's adaptive tier instead of per-step LRU inserts
        {
            const char * pol = getenv("LLAMA_MOE_CACHE_POLICY");
            mc->policy_strata = pol && strcmp(pol, "strata") == 0;
            mc->adapt_every   = std::max(1, env_int("LLAMA_MOE_CACHE_ADAPT_EVERY", mc->adapt_every));
            mc->adapt_swaps   = std::max(0, env_int("LLAMA_MOE_CACHE_ADAPT_SWAPS", mc->adapt_swaps));
            // ggml-cuda offloads MUL_MAT_ID to the GPU from this batch up (ggml-cuda.cu, same variable and default)
            mc->max_tokens    = std::max(1, env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32)) - 1;
        }

        // init state + tables (everything uncached -> the layer's dummy slot, n_slots)
        size_t vram = 0;
        size_t n_seeded = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            const int32_t ns = ls.pub.n_slots;
            ls.slot_expert.assign(ns, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(ns, 0);
            ls.slot_in_flight.assign(ns, false);
            if (mc->policy_strata) {
                ls.usage.assign(n_expert, 0.0f);
            }

            std::vector<int32_t> table(n_expert, ns);
            // 0004: the profile's experts start resident, in rank order (the highest rank is the last evicted
            // by LRU); blocking copies before the first decode, as Strata's startup fill
            const auto seeds = seed_of.find(ls.pub.il);
            if (seeds != seed_of.end()) {
                const int32_t n_seed = (int32_t) std::min<size_t>(seeds->second.size(), (size_t) ns);
                for (int32_t s = 0; s < n_seed; ++s) {
                    const int32_t e = seeds->second[s];
                    upload_slice(ls.pub.up_c,   ls.pub.up_src,   e, s);
                    upload_slice(ls.pub.gate_c, ls.pub.gate_src, e, s);
                    upload_slice(ls.pub.down_c, ls.pub.down_src, e, s);
                    ls.slot_expert[s]   = e;
                    ls.expert_slot[e]   = s;
                    ls.slot_last_use[s] = (uint64_t) (n_seed - s);
                    table[e] = s;
                    n_seeded++;
                }
            }
            ggml_backend_tensor_set(ls.pub.dev_table,  table.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, table.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        mc->worker = std::thread([mc]() {
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                }
                auto & ls = mc->layers[j.layer_idx];
                upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    j.done = true;
                    mc->done.push_back(j);
                }
            }
        });

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers, %d slots/layer requested, %zu seeded from a profile, "
                "policy %s, batches up to %" PRId64 " tokens, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, n_seeded,
                mc->policy_strata ? "strata (adaptive tier)" : "lru", mc->max_tokens, vram/1024.0/1024.0);
        if (mc->policy_strata) {
            LLAMA_LOG_INFO("%s: adaptive tier: every %d steps, up to %d swaps\n", __func__, mc->adapt_every, mc->adapt_swaps);
        }
    }();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            ls.slot_in_flight[j.slot]  = false;
            set_table_entry(ls.pub, j.expert, j.slot);
            if (mc->in_flight > 0) {
                mc->in_flight--;
            }
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 0004: Strata's adaptive tier (generate.cpp:2418-2457), at deliberate moments only
    if (mc->policy_strata) {
        for (auto & ls : mc->layers) {
            ls.pending.clear();
        }
        if (mc->n_steps % (uint64_t) mc->adapt_every != 0 || mc->in_flight > 0) {
            return; // not this step, or the previous swaps are still in flight
        }
        struct swap { float gain; size_t li; int32_t in, out; };
        std::vector<swap> swaps;
        std::vector<std::pair<float, int32_t>> cand, vict;
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            auto & ls = mc->layers[li];
            cand.clear();
            vict.clear();
            for (int32_t e = 0; e < (int32_t) ls.usage.size(); ++e) {
                const int32_t slot = ls.expert_slot[e];
                if (slot < 0) {
                    if (ls.usage[e] >= 2.0f) {
                        cand.emplace_back(ls.usage[e], e);
                    }
                } else if (!ls.slot_in_flight[slot]) {
                    vict.emplace_back(ls.usage[e], e);
                }
            }
            if (cand.empty() || vict.empty()) {
                continue;
            }
            std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            const size_t nc = std::min(cand.size(), vict.size());
            std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                    [](const auto & a, const auto & b) { return a.first < b.first; });
            for (size_t i = 0; i < nc; ++i) {
                if (cand[i].first < vict[i].first + 1.5f) {
                    break;
                }
                swaps.push_back({cand[i].first - vict[i].first, li, cand[i].second, vict[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](const swap & a, const swap & b) { return a.gain > b.gain; });
        if ((int32_t) swaps.size() > mc->adapt_swaps) {
            swaps.resize((size_t) mc->adapt_swaps);
        }
        {
            std::lock_guard<std::mutex> wlk(mc->wmtx);
            for (const swap & s : swaps) {
                auto & ls = mc->layers[s.li];
                const int32_t slot = ls.expert_slot[s.out];
                // evicted now: the CPU computes it meanwhile; the newcomer is resident once its copy has landed
                ls.expert_slot[s.out] = -1;
                ls.slot_expert[slot]  = -1;
                set_table_entry(ls.pub, s.out, ls.pub.n_slots);
                ls.slot_in_flight[slot] = true;
                mc->todo.push_back({s.li, s.in, slot});
                mc->in_flight++;
            }
        }
        mc->wcv.notify_one();
        for (auto & ls : mc->layers) {
            for (float & v : ls.usage) {
                v *= 0.7f;
            }
        }
        return;
    }

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, ls.pub.n_slots);
            }
            ls.slot_in_flight[slot] = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
            mc->in_flight++;
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

int64_t llama_moe_cache_max_tokens() {
    return g_cache ? g_cache->max_tokens : 0;
}
