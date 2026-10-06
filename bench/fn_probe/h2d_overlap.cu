// h2d_overlap: does a host-to-device copy on a SECOND stream overlap a compute kernel on the main stream, on this card,
// and does it from (a) pinned memory, (b) pageable memory, (c) a pinned staging ring fed by memcpy threads from pageable
// memory? docs/FLASH-NEXT.md 13.4. One GPU consumer: run it with the stack's model unloaded (bench/fn_probe/run_probe.py).
//
// The experiment copies the structure of 0029's layer-major prefill, the thing it has to explain:
//   per "layer":  `graphs` compute kernels on the main stream (each ~`kernel_ms`; the host synchronises after each, like the
//                 engine's graph compute call), and right after the first one the host asks for the NEXT layer's `tensors`
//                 weight tensors of `mib` MiB each to be uploaded into a second slot set; the next layer's first kernel
//                 waits for that upload's event.
// Per mode it measures T_c (the same loop with no upload), T_u (the uploads alone), T_o (both) per layer, and the host
// thread's time inside the upload call. Overlap efficiency E = (T_c + T_u - T_o) / (T_c + T_u - max(T_c, T_u)): 1 = the
// upload is hidden behind the compute, 0 = they add. The shipped engine's 8,192-token batch is the E ~ 0 case (433-443 tok/s
// with 6 slots and prefetch, 433-439 without, n=3: docs/FLASH-NEXT.md 13.3).
//
// Modes (a source and a method):
//   none         compute only (T_c)
//   pin          pinned host memory, cudaMemcpyAsync on the copy stream                         (a)
//   pageable     malloc'd pageable memory, cudaMemcpyAsync                                      (b)
//   mmap         an mmapped file's pages (the experts' situation), cudaMemcpyAsync              (b, the real source)
//   stage-heap   the staging ring (ggml-stager.cpp, the engine's own source), source malloc'd   (c)
//   stage-mmap   the staging ring, source mmapped                                               (c, the real source)
//   register     cudaHostRegister on the mapped range, cudaMemcpyAsync, cudaHostUnregister      (e, an alternative to the ring)
//   d2d          device-to-device copies on the second stream (docs/FLASH-NEXT.md 10.2's observation: on WDDM the compute
//                stream waited for work queued on a second stream)                              (d)
//
//   h2d_overlap --sim      the whole harness on a simulated device (threads and spin loops): the harness's own correctness
//                          test, no GPU and no CUDA call
//   h2d_overlap            the card (CUDA device 0 of CUDA_VISIBLE_DEVICES)
//
// Build: bench/fn_probe/build.bat. Standard run: bench/fn_probe/run_probe.py. Expected run time on the 5060 Ti: ~3 minutes.
//
// The staging ring's source is the engine's ggml/src/ggml-stager.cpp (0030) compiled with GGML_STAGER_NO_BACKEND, so the
// probe measures the code that ships, with CUDA hooks written here.

#include "ggml-stager.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#ifndef PROBE_NO_CUDA
#  include <cuda_runtime.h>
#endif

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

static void spin_ms(double ms) {
    const auto t = clk::now() + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double, std::milli>(ms));
    while (clk::now() < t) {
        // busy: a sleep would not model a kernel that holds the SMs
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the device

struct Dev {
    virtual ~Dev() {}
    virtual std::string describe() = 0;
    virtual void * alloc_dev(size_t n) = 0;
    virtual void   free_dev(void * p) = 0;
    virtual void * alloc_pinned(size_t n) = 0;
    virtual void   free_pinned(void * p) = 0;
    virtual int    new_stream() = 0;
    virtual int    new_event(bool timing) = 0;
    virtual void   kernel(int s, double ms) = 0;                                  // all SMs busy for about ms
    virtual void   h2d(int s, void * dst, const void * src, size_t n) = 0;        // cudaMemcpyAsync(H2D)
    virtual void   d2d(int s, void * dst, const void * src, size_t n) = 0;
    virtual void   record(int e, int s) = 0;
    virtual void   wait(int s, int e) = 0;                                        // the stream waits for the event
    virtual void   esync(int e) = 0;                                              // the host waits for the event
    virtual void   ssync(int s) = 0;
    virtual double ems(int e0, int e1) = 0;
    virtual bool   host_register(void * p, size_t n) = 0;
    virtual void   host_unregister(void * p) = 0;
    virtual void   bind() {}
    virtual size_t free_vram() { return 0; }
    virtual void   dev_sync() = 0;
};

// ---- the simulated device: a stream is a thread draining a queue; a kernel spins; a pinned H2D occupies the stream for
// n / pcie; a pageable one blocks the CALLER for n / pageable (the driver's staging copy), which is the semantics the probe
// is about; events are counters (the same as the stager test's fake engine)
struct SimDev : Dev {
    double pcie_gbs = 14.0, pageable_gbs = 11.0, d2d_gbs = 400.0;
    struct Ev { std::mutex mu; std::condition_variable cv; int64_t recorded = 0, completed = 0; clk::time_point t; };
    struct Op { int kind; double ms; Ev * ev; int64_t target; };   // 0 spin, 1 record, 2 wait
    struct Stream {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Op> q;
        bool stop = false, busy = false;
        std::thread th;
    };
    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<std::unique_ptr<Ev>> events;
    std::mutex mu;
    std::map<uintptr_t, size_t> pinned;     // start -> size
    std::vector<void *> blocks;

    ~SimDev() override {
        for (auto & s : streams) {
            { std::lock_guard<std::mutex> l(s->mu); s->stop = true; }
            s->cv.notify_all();
            s->th.join();
        }
        for (void * p : blocks) { free(p); }
    }
    std::string describe() override { return "SIMULATED device (threads and spin loops; no CUDA)"; }
    void * alloc_dev(size_t n) override { void * p = malloc(n); std::lock_guard<std::mutex> l(mu); blocks.push_back(p); return p; }
    void   free_dev(void *) override {}
    void * alloc_pinned(size_t n) override {
        void * p = malloc(n);
        std::lock_guard<std::mutex> l(mu);
        blocks.push_back(p);
        pinned[(uintptr_t) p] = n;
        return p;
    }
    void   free_pinned(void *) override {}
    bool is_pinned(const void * p) {
        std::lock_guard<std::mutex> l(mu);
        auto it = pinned.upper_bound((uintptr_t) p);
        if (it == pinned.begin()) return false;
        --it;
        return (uintptr_t) p < it->first + it->second;
    }
    void run(Stream * s) {
        for (;;) {
            Op o;
            {
                std::unique_lock<std::mutex> l(s->mu);
                s->cv.wait(l, [&] { return s->stop || !s->q.empty(); });
                if (s->q.empty()) return;
                o = s->q.front();
                s->q.pop_front();
                s->busy = true;
            }
            if (o.kind == 0) {
                spin_ms(o.ms);
            } else if (o.kind == 1) {
                std::lock_guard<std::mutex> l(o.ev->mu);
                o.ev->completed++;
                o.ev->t = clk::now();
                o.ev->cv.notify_all();
            } else {
                std::unique_lock<std::mutex> l(o.ev->mu);
                o.ev->cv.wait(l, [&] { return o.ev->completed >= o.target; });
            }
            {
                std::lock_guard<std::mutex> l(s->mu);
                s->busy = false;
            }
            s->cv.notify_all();
        }
    }
    void push(int s, const Op & o) {
        Stream * st = streams[(size_t) s].get();
        { std::lock_guard<std::mutex> l(st->mu); st->q.push_back(o); }
        st->cv.notify_all();
    }
    int new_stream() override {
        streams.emplace_back(new Stream());
        Stream * s = streams.back().get();
        s->th = std::thread([this, s] { run(s); });
        return (int) streams.size() - 1;
    }
    int new_event(bool) override { events.emplace_back(new Ev()); return (int) events.size() - 1; }
    void kernel(int s, double ms) override { push(s, {0, ms, nullptr, 0}); }
    void h2d(int s, void * dst, const void * src, size_t n) override {
        (void) dst;   // data is not moved in the simulation
        if (is_pinned(src)) {
            push(s, {0, (double) n / (pcie_gbs * 1e6), nullptr, 0});
        } else {
            spin_ms((double) n / (pageable_gbs * 1e6));   // the calling thread is blocked for the driver's staging copy
            push(s, {0, 0.0, nullptr, 0});
        }
    }
    void d2d(int s, void *, const void *, size_t n) override { push(s, {0, (double) n / (d2d_gbs * 1e6), nullptr, 0}); }
    void record(int e, int s) override {
        Ev * ev = events[(size_t) e].get();
        { std::lock_guard<std::mutex> l(ev->mu); ev->recorded++; }
        push(s, {1, 0, ev, 0});
    }
    void wait(int s, int e) override {
        Ev * ev = events[(size_t) e].get();
        int64_t t;
        { std::lock_guard<std::mutex> l(ev->mu); t = ev->recorded; }
        push(s, {2, 0, ev, t});
    }
    void esync(int e) override {
        Ev * ev = events[(size_t) e].get();
        std::unique_lock<std::mutex> l(ev->mu);
        const int64_t t = ev->recorded;
        ev->cv.wait(l, [&] { return ev->completed >= t; });
    }
    void ssync(int s) override {
        Stream * st = streams[(size_t) s].get();
        std::unique_lock<std::mutex> l(st->mu);
        st->cv.wait(l, [&] { return st->q.empty() && !st->busy; });
    }
    double ems(int e0, int e1) override {
        return std::chrono::duration<double, std::milli>(events[(size_t) e1]->t - events[(size_t) e0]->t).count();
    }
    bool host_register(void *, size_t) override { return true; }
    void host_unregister(void *) override {}
    void dev_sync() override { for (size_t i = 0; i < streams.size(); ++i) ssync((int) i); }
};

#ifndef PROBE_NO_CUDA
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

__global__ static void alu_kernel(float * sink, long iters) {
    float a = (float) threadIdx.x, b = 1.0001f;
    for (long i = 0; i < iters; ++i) {
        a = fmaf(a, b, 0.5f);
        b = fmaf(b, a, 1e-7f);
        if (b > 1e30f) { b = 1.0001f; }
    }
    if (a == 123.456f) { sink[0] = a + b; }
}

struct CudaDev : Dev {
    int device = 0;
    cudaDeviceProp prop;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t>  events;
    float * sink = nullptr;
    double iters_per_ms = 0;
    int blocks = 0;

    explicit CudaDev(int dev) : device(dev) {
        CK(cudaSetDevice(device));
        CK(cudaGetDeviceProperties(&prop, device));
        blocks = prop.multiProcessorCount * 8;
        CK(cudaMalloc((void **) &sink, 64));
        // calibrate: one all-SM kernel of a known number of iterations, timed with events (the first launch is a warm-up)
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        long iters = 200000;
        for (int k = 0; k < 4; ++k) {
            CK(cudaEventRecord(a, 0));
            alu_kernel<<<blocks, 256>>>(sink, iters);
            CK(cudaEventRecord(b, 0));
            CK(cudaEventSynchronize(b));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, a, b));
            if (ms > 0.01f) {
                iters_per_ms = (double) iters / ms;
                iters = (long) (iters_per_ms * 16.0);
            }
        }
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }
    ~CudaDev() override {
        cudaDeviceSynchronize();
        for (auto s : streams) cudaStreamDestroy(s);
        for (auto e : events) cudaEventDestroy(e);
        cudaFree(sink);
    }
    std::string describe() override {
        char b[512];
        snprintf(b, sizeof(b), "%s (sm_%d%d, %d SMs, %.1f GiB), device %d, kernel rate %.3g iters/ms", prop.name, prop.major, prop.minor,
                 prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0, device, iters_per_ms);
        return b;
    }
    void * alloc_dev(size_t n) override { void * p = nullptr; CK(cudaMalloc(&p, n)); return p; }
    void   free_dev(void * p) override { cudaFree(p); }
    void * alloc_pinned(size_t n) override { void * p = nullptr; CK(cudaHostAlloc(&p, n, cudaHostAllocDefault)); return p; }
    void   free_pinned(void * p) override { cudaFreeHost(p); }
    int new_stream() override {
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));   // as ggml-cuda creates them
        streams.push_back(s);
        return (int) streams.size() - 1;
    }
    int new_event(bool timing) override {
        cudaEvent_t e;
        CK(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming));
        events.push_back(e);
        return (int) events.size() - 1;
    }
    void kernel(int s, double ms) override {
        alu_kernel<<<blocks, 256, 0, streams[(size_t) s]>>>(sink, (long) (iters_per_ms * ms));
    }
    void h2d(int s, void * dst, const void * src, size_t n) override {
        CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, streams[(size_t) s]));
    }
    void d2d(int s, void * dst, const void * src, size_t n) override {
        CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, streams[(size_t) s]));
    }
    void record(int e, int s) override { CK(cudaEventRecord(events[(size_t) e], streams[(size_t) s])); }
    void wait(int s, int e) override { CK(cudaStreamWaitEvent(streams[(size_t) s], events[(size_t) e], 0)); }
    void esync(int e) override { CK(cudaEventSynchronize(events[(size_t) e])); }
    void ssync(int s) override { CK(cudaStreamSynchronize(streams[(size_t) s])); }
    double ems(int e0, int e1) override {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, events[(size_t) e0], events[(size_t) e1]));
        return ms;
    }
    bool host_register(void * p, size_t n) override {
        const cudaError_t e = cudaHostRegister(p, n, cudaHostRegisterPortable);
        if (e != cudaSuccess) { cudaGetLastError(); return false; }
        return true;
    }
    void host_unregister(void * p) override { cudaHostUnregister(p); }
    void bind() override { cudaSetDevice(device); }
    size_t free_vram() override { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return f; }
    void dev_sync() override { CK(cudaDeviceSynchronize()); }
};
#endif

// ---------------------------------------------------------------------------------------------------------------------
// host sources

struct MappedFile {
    void * ptr = nullptr;
    size_t size = 0;
    std::string path;
#ifdef _WIN32
    HANDLE hf = INVALID_HANDLE_VALUE, hm = nullptr;
#else
    int fd = -1;
#endif
    bool open_new(const std::string & p, size_t n, std::mt19937_64 & rng) {
        path = p;
        size = n;
#ifdef _WIN32
        hf = CreateFileA(p.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        if (hf == INVALID_HANDLE_VALUE) return false;
        hm = CreateFileMappingA(hf, nullptr, PAGE_READWRITE, (DWORD) (n >> 32), (DWORD) (n & 0xffffffffu), nullptr);
        if (!hm) return false;
        ptr = MapViewOfFile(hm, FILE_MAP_ALL_ACCESS, 0, 0, n);
#else
        fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0 || ftruncate(fd, (off_t) n) != 0) return false;
        ptr = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) ptr = nullptr;
#endif
        if (!ptr) return false;
        uint64_t * w = (uint64_t *) ptr;
        for (size_t i = 0; i < n / 8; ++i) w[i] = rng();
        return true;
    }
    ~MappedFile() {
#ifdef _WIN32
        if (ptr) UnmapViewOfFile(ptr);
        if (hm) CloseHandle(hm);
        if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
        if (!path.empty()) DeleteFileA(path.c_str());
#else
        if (ptr) munmap(ptr, size);
        if (fd >= 0) ::close(fd);
        if (!path.empty()) ::unlink(path.c_str());
#endif
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// the experiment

struct Cfg {
    size_t mib        = 256;     // one weight tensor (the largest expert tensor of the model: 256.25 MiB)
    int    tensors    = 3;       // gate, up, down
    int    layers     = 10;      // timed layers per repetition (the first only primes the slots)
    int    graphs     = 16;      // kernels per layer (a 8,192-token batch is 16 ubatches)
    double kernel_ms  = 16.2;    // one graph's GPU time (776 ms of compute per 512-token ubatch / 48 layers)
    bool   sync_graph = true;    // the host waits for each graph, as the engine's compute call does
    int    reps       = 5;
    int    threads    = 4;
    size_t chunk_mib  = 16;
    int    ring       = 16;
    int    d2d_copies = 24;      // device-to-device copies of one tensor per layer in mode d2d
    bool   sweep      = true;
    bool   json_only  = false;
    std::string modes = "none,pin,pageable,mmap,stage-heap,stage-mmap,register,d2d";
    std::string tmp   = ".";
    std::string json;
    uint64_t seed     = 20261006;
};

struct Slots {
    std::vector<void *> d[2];     // [set][tensor] device slots
    int free_ev[2];               // recorded on the compute stream when a set's last reader has been launched
    int ready_ev[2];              // recorded on the copy stream when a set's upload has landed
};

struct Stats {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double med() const { if (v.empty()) return 0; auto w = v; std::sort(w.begin(), w.end()); return w[w.size() / 2]; }
    double lo() const { return v.empty() ? 0 : *std::min_element(v.begin(), v.end()); }
    double hi() const { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }
};

struct Sources {
    std::vector<void *> pin, heap;
    std::vector<void *> map;      // pointers into the mapped file
    std::unique_ptr<MappedFile> file;
    std::vector<void *> dev;      // d2d source
};

struct StageCtx {
    Dev * dev;
    int stream;
    std::vector<int> evs;         // event ids handed out as void * (id + 1)
};

static void h_bind(void * c) { ((StageCtx *) c)->dev->bind(); }
static void * h_alloc(void * c, size_t n, bool * pinned) { *pinned = true; return ((StageCtx *) c)->dev->alloc_pinned(n); }
static void h_free(void * c, void * p) { ((StageCtx *) c)->dev->free_pinned(p); }
static void h_copy(void * c, void * dst, size_t off, const void * src, size_t n) {
    auto * x = (StageCtx *) c;
    x->dev->h2d(x->stream, (char *) dst + off, src, n);
}
static void * h_evnew(void * c) {
    auto * x = (StageCtx *) c;
    const int e = x->dev->new_event(false);
    return (void *) (intptr_t) (e + 1);
}
static void h_evfree(void *, void *) {}
static void h_evrec(void * c, void * e) { auto * x = (StageCtx *) c; x->dev->record((int) (intptr_t) e - 1, x->stream); }
static void h_evwait(void * c, void * e) { auto * x = (StageCtx *) c; x->dev->wait(x->stream, (int) (intptr_t) e - 1); }
static void h_evsync(void * c, void * e) { ((StageCtx *) c)->dev->esync((int) (intptr_t) e - 1); }
static void h_sync(void * c) { auto * x = (StageCtx *) c; x->dev->ssync(x->stream); }

struct Mode {
    std::string name;
    int kind;      // 0 none, 1 pin, 2 pageable, 3 mmap, 4 stage-heap, 5 stage-mmap, 6 register, 7 d2d
};

struct Result {
    std::string mode;
    bool ok = true;
    std::string note;
    Stats Tc, Tu, To, blocked, copy_gbs;
};

struct Probe {
    Dev * dev;
    Cfg cfg;
    int s_comp, s_copy;
    Sources src;
    Slots sl;
    size_t tbytes;
    std::unique_ptr<StageCtx> sctx;
    ggml_stager_t stager = nullptr;
    int ev_t0, ev_t1;

    void setup(std::mt19937_64 & rng) {
        tbytes = cfg.mib << 20;
        s_comp = dev->new_stream();
        s_copy = dev->new_stream();
        for (int s = 0; s < 2; ++s) {
            for (int t = 0; t < cfg.tensors; ++t) {
                sl.d[s].push_back(dev->alloc_dev(tbytes));
            }
            sl.free_ev[s]  = dev->new_event(false);
            sl.ready_ev[s] = dev->new_event(false);
        }
        ev_t0 = dev->new_event(true);
        ev_t1 = dev->new_event(true);
        for (int t = 0; t < cfg.tensors; ++t) {
            void * p = dev->alloc_pinned(tbytes);
            fill(p, rng);
            src.pin.push_back(p);
            void * h = malloc(tbytes);
            fill(h, rng);
            src.heap.push_back(h);
            src.dev.push_back(dev->alloc_dev(tbytes));
        }
        // the mapped file: tensors x tbytes of random data, written through the mapping and touched
        src.file.reset(new MappedFile());
        const std::string path = cfg.tmp + "/fn_probe_mapped.bin";
        if (src.file->open_new(path, tbytes * (size_t) cfg.tensors, rng)) {
            for (int t = 0; t < cfg.tensors; ++t) {
                src.map.push_back((char *) src.file->ptr + tbytes * (size_t) t);
            }
        } else {
            fprintf(stderr, "could not create the mapped file %s: the mmap modes are skipped\n", path.c_str());
            src.file.reset();
        }
    }
    void fill(void * p, std::mt19937_64 & rng) {
        // touch every page (so a pageable source is committed and resident before anything is timed) with a distinct word
        uint64_t * w = (uint64_t *) p;
        for (size_t i = 0; i < tbytes / 8; i += 512) {
            w[i] = rng();
        }
    }

    // upload one set's `tensors` tensors; returns the host's time inside the call (ms). `kind` as Mode.
    // The set's event `ready` is recorded on the copy stream behind the last copy (ticketed for the stager).
    struct Pending { uint64_t ticket = 0; bool registered = false; std::vector<void *> reg; };

    double issue(int kind, int set, Pending & pd) {
        const auto t0 = clk::now();
        // the last reader of the set has been launched: the copy stream waits for it
        if (kind == 7) {
            dev->wait(s_copy, sl.free_ev[set]);
            for (int c = 0; c < cfg.d2d_copies; ++c) {
                dev->d2d(s_copy, sl.d[set][(size_t) c % sl.d[set].size()], src.dev[(size_t) c % src.dev.size()], tbytes);
            }
            dev->record(sl.ready_ev[set], s_copy);
            return ms_since(t0);
        }
        if (kind >= 4 && kind <= 5) {
            // the ring: one request per tensor; the free event only on the first, the ready event on the last
            for (int t = 0; t < cfg.tensors; ++t) {
                const void * s = kind == 4 ? src.heap[(size_t) t] : src.map[(size_t) t];
                void * wait_ev = t == 0 ? (void *) (intptr_t) (sl.free_ev[set] + 1) : nullptr;
                void * done_ev = t + 1 == cfg.tensors ? (void *) (intptr_t) (sl.ready_ev[set] + 1) : nullptr;
                pd.ticket = ggml_stager_submit(stager, sl.d[set][(size_t) t], s, tbytes, wait_ev, done_ev);
            }
            return ms_since(t0);
        }
        dev->wait(s_copy, sl.free_ev[set]);
        for (int t = 0; t < cfg.tensors; ++t) {
            const void * s = kind == 1 ? src.pin[(size_t) t] : kind == 2 ? src.heap[(size_t) t] : src.map[(size_t) t];
            if (kind == 6) {
                pd.registered = dev->host_register(src.map[(size_t) t], tbytes);
                if (!pd.registered) { return -1; }
                pd.reg.push_back(src.map[(size_t) t]);
            }
            dev->h2d(s_copy, sl.d[set][(size_t) t], s, tbytes);
        }
        dev->record(sl.ready_ev[set], s_copy);
        return ms_since(t0);
    }

    void finish_pending(Pending & pd) {
        for (void * p : pd.reg) {
            dev->host_unregister(p);
        }
        pd.reg.clear();
    }

    // the layer loop. kind 0 = no upload. Returns per-layer wall ms (layers 1..n-1), the host time inside the upload calls
    bool run_layers(int kind, std::vector<double> & wall, std::vector<double> & blocked) {
        // prime set 0 (the first layer's weights), untimed
        Pending pd0;
        if (kind != 0) {
            if (issue(kind, 1, pd0) < 0) return false;       // layer 1 reads set 1
            if (pd0.ticket) ggml_stager_wait_enqueued(stager, pd0.ticket);
            dev->wait(s_comp, sl.ready_ev[1]);
            dev->ssync(s_comp);
            finish_pending(pd0);
        }
        dev->kernel(s_comp, 1.0);
        dev->ssync(s_comp);
        for (int L = 1; L < cfg.layers; ++L) {
            const int nxt = (L + 1) % 2;   // layer L reads set L % 2; layer L + 1's set is uploaded during layer L
            const auto t0 = clk::now();
            double blk = 0;
            Pending pd;
            for (int g = 0; g < cfg.graphs; ++g) {
                dev->kernel(s_comp, cfg.kernel_ms);
                if (g == 0) {
                    dev->record(sl.free_ev[(L - 1) % 2], s_comp);   // set (L-1)%2's last reader (layer L-1) is done once layer L starts
                    // upload the NEXT layer's set during this layer; it overwrites the set layer L-1 used
                    if (kind != 0) {
                        const double b = issue(kind, nxt, pd);
                        if (b < 0) return false;
                        blk += b;
                    }
                }
                if (cfg.sync_graph) dev->ssync(s_comp);
            }
            // the next layer's first kernel waits for its set
            if (kind != 0) {
                const auto tb = clk::now();
                if (pd.ticket) ggml_stager_wait_enqueued(stager, pd.ticket);
                blk += ms_since(tb);
                dev->wait(s_comp, sl.ready_ev[nxt]);
            }
            dev->ssync(s_comp);
            finish_pending(pd);
            wall.push_back(ms_since(t0));
            blocked.push_back(blk);
        }
        return true;
    }

    // the uploads alone: per layer, issue then wait for the event
    bool run_copy_only(int kind, std::vector<double> & wall) {
        for (int L = 0; L < cfg.layers; ++L) {
            Pending pd;
            const auto t0 = clk::now();
            if (issue(kind, L % 2, pd) < 0) return false;
            if (pd.ticket) ggml_stager_wait_enqueued(stager, pd.ticket);
            dev->esync(sl.ready_ev[L % 2]);
            wall.push_back(ms_since(t0));
            finish_pending(pd);
            // the next upload into this set must wait for "its last reader": none, so make the event complete
            dev->record(sl.free_ev[L % 2], s_comp);
        }
        return true;
    }

    void make_stager(int threads, size_t chunk_mib, int ring) {
        destroy_stager();
        sctx.reset(new StageCtx{dev, s_copy, {}});
        ggml_stager_ops ops = {};
        ops.ctx = sctx.get();
        ops.bind_thread = h_bind;
        ops.host_alloc = h_alloc;
        ops.host_free = h_free;
        ops.copy_async = h_copy;
        ops.event_new = h_evnew;
        ops.event_free = h_evfree;
        ops.event_record = h_evrec;
        ops.stream_wait = h_evwait;
        ops.event_sync = h_evsync;
        ops.sync = h_sync;
        ggml_stager_params p = ggml_stager_params_default();
        p.n_workers = threads;
        p.chunk_bytes = chunk_mib << 20;
        p.n_chunks = ring;
        stager = ggml_stager_new(&ops, &p);
    }
    void destroy_stager() {
        if (stager) { ggml_stager_free(stager); stager = nullptr; }
    }
};

static int kind_of(const std::string & m) {
    if (m == "none") return 0;
    if (m == "pin") return 1;
    if (m == "pageable") return 2;
    if (m == "mmap") return 3;
    if (m == "stage-heap") return 4;
    if (m == "stage-mmap") return 5;
    if (m == "register") return 6;
    if (m == "d2d") return 7;
    return -1;
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> r;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) r.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return r;
}

static void jstr(FILE * f, const char * k, const std::string & v, bool last = false) {
    fprintf(f, "\"%s\": \"%s\"%s", k, v.c_str(), last ? "" : ", ");
}
static void jnum(FILE * f, const char * k, double v, bool last = false) {
    fprintf(f, "\"%s\": %.4f%s", k, v, last ? "" : ", ");
}
static void jstat(FILE * f, const char * k, const Stats & s, bool last = false) {
    fprintf(f, "\"%s\": {\"median\": %.4f, \"min\": %.4f, \"max\": %.4f, \"n\": %zu}%s", k, s.med(), s.lo(), s.hi(), s.v.size(), last ? "" : ", ");
}

int main(int argc, char ** argv) {
    Cfg cfg;
    bool sim = false;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--sim") sim = true;
        else if (a == "--device") device = atoi(next().c_str());
        else if (a == "--mib") cfg.mib = (size_t) atoll(next().c_str());
        else if (a == "--tensors") cfg.tensors = atoi(next().c_str());
        else if (a == "--layers") cfg.layers = atoi(next().c_str());
        else if (a == "--graphs") cfg.graphs = atoi(next().c_str());
        else if (a == "--kernel-ms") cfg.kernel_ms = atof(next().c_str());
        else if (a == "--no-sync-graph") cfg.sync_graph = false;
        else if (a == "--reps") cfg.reps = atoi(next().c_str());
        else if (a == "--threads") cfg.threads = atoi(next().c_str());
        else if (a == "--chunk-mib") cfg.chunk_mib = (size_t) atoll(next().c_str());
        else if (a == "--ring") cfg.ring = atoi(next().c_str());
        else if (a == "--d2d-copies") cfg.d2d_copies = atoi(next().c_str());
        else if (a == "--modes") cfg.modes = next();
        else if (a == "--no-sweep") cfg.sweep = false;
        else if (a == "--tmp") cfg.tmp = next();
        else if (a == "--json") cfg.json = next();
        else if (a == "--seed") cfg.seed = (uint64_t) atoll(next().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    std::unique_ptr<Dev> dev;
    if (sim) {
        dev.reset(new SimDev());
    } else {
#ifdef PROBE_NO_CUDA
        fprintf(stderr, "built without CUDA: use --sim\n");
        return 2;
#else
        dev.reset(new CudaDev(device));
#endif
    }
    std::mt19937_64 rng(cfg.seed);
    printf("device: %s\n", dev->describe().c_str());
    printf("config: %d x %zu MiB tensors per layer, %d graphs of %.1f ms per layer, %d timed layers x %d reps, host syncs per graph: %s, ring %d x %zu MiB, %d threads\n",
           cfg.tensors, cfg.mib, cfg.graphs, cfg.kernel_ms, cfg.layers - 1, cfg.reps, cfg.sync_graph ? "yes" : "no", cfg.ring, cfg.chunk_mib, cfg.threads);
    if (!sim) printf("free VRAM before: %.0f MiB\n", dev->free_vram() / 1048576.0);

    Probe pr;
    pr.dev = dev.get();
    pr.cfg = cfg;
    pr.setup(rng);

    std::vector<Result> results;
    Stats Tc_all;
    std::vector<std::string> mode_list = split(cfg.modes);
    if (std::find(mode_list.begin(), mode_list.end(), "none") == mode_list.end()) {
        mode_list.insert(mode_list.begin(), "none");   // T_c is every other mode's baseline
    } else {
        mode_list.erase(std::find(mode_list.begin(), mode_list.end(), "none"));
        mode_list.insert(mode_list.begin(), "none");
    }
    for (const std::string & name : mode_list) {
        const int kind = kind_of(name);
        Result r;
        r.mode = name;
        if (kind < 0) { fprintf(stderr, "unknown mode %s\n", name.c_str()); return 2; }
        if ((kind == 3 || kind == 5 || kind == 6) && pr.src.map.empty()) { r.ok = false; r.note = "no mapped file"; results.push_back(r); continue; }
        if (kind == 4 || kind == 5) pr.make_stager(cfg.threads, cfg.chunk_mib, cfg.ring);
        bool ok = true;
        for (int rep = 0; rep < cfg.reps && ok; ++rep) {
            std::vector<double> wall, blk;
            ok = pr.run_layers(kind, wall, blk);
            if (!ok) break;
            for (double w : wall) r.To.add(w);
            for (double b : blk) r.blocked.add(b);
        }
        if (ok && kind != 0) {
            for (int rep = 0; rep < cfg.reps && ok; ++rep) {
                std::vector<double> w;
                ok = pr.run_copy_only(kind, w);
                for (double x : w) {
                    r.Tu.add(x);
                    r.copy_gbs.add(kind == 7 ? (double) cfg.d2d_copies * pr.tbytes / 1e6 / x : (double) cfg.tensors * pr.tbytes / 1e6 / x);
                }
            }
        }
        if (!ok) { r.ok = false; r.note = kind == 6 ? "cudaHostRegister failed on the mapped file" : "failed"; }
        pr.destroy_stager();
        results.push_back(r);
        if (kind == 0) Tc_all = results.back().To;
    }

    // ---- the table
    const double Tc = Tc_all.med();
    printf("\nper layer, median of %d x %d samples (n=%d reps); ms. T_c = compute alone, T_u = the upload alone, T_o = both\n", cfg.reps, cfg.layers - 1, cfg.reps);
    printf("%-11s %9s %9s %9s %9s %9s %7s %12s %9s  %s\n", "mode", "T_c", "T_u", "T_o", "T_c+T_u", "max", "E", "host-blocked", "copy GB/s", "reading");
    for (auto & r : results) {
        if (r.mode == "none") continue;
        if (!r.ok) { printf("%-11s  not run: %s\n", r.mode.c_str(), r.note.c_str()); continue; }
        const double Tu = r.Tu.med(), To = r.To.med();
        const double serial = Tc + Tu, ideal = std::max(Tc, Tu);
        const double denom = serial - ideal;
        const double E = denom > 1e-6 ? (serial - To) / denom : 0;
        const char * reading = E >= 0.8 ? "OVERLAPS" : E < 0.5 ? "mostly serial" : "partial";   // the first graph always runs while the host is blocked: a blocking copy still scores ~ graph / upload
        printf("%-11s %9.1f %9.1f %9.1f %9.1f %9.1f %7.2f %12.1f %9.2f  %s\n", r.mode.c_str(), Tc, Tu, To, serial, ideal, E, r.blocked.med(), r.copy_gbs.med(), reading);
    }
    printf("compute alone: median %.1f ms a layer (%d graphs x %.1f ms = %.1f ms ideal), min %.1f max %.1f\n", Tc, cfg.graphs, cfg.kernel_ms, cfg.graphs * cfg.kernel_ms, Tc_all.lo(), Tc_all.hi());

    // ---- the ring's sweep (copy-only throughput of the mapped source through the stager)
    struct Sw { int th; size_t chunk; int ring; double gbs; };
    std::vector<Sw> sweep;
    if (cfg.sweep) {
        const int kind = !pr.src.map.empty() ? 5 : 4;
        printf("\nring sweep (copy-only throughput through the stager, source %s, n=3 layers each):\n", kind == 5 ? "mmap" : "heap");
        const int threads[] = { 1, 2, 3, 4, 6 };
        const size_t chunks[] = { 4, 16, 64 };
        const int rings[] = { 8, 16, 32 };
        for (int th : threads) {
            for (size_t ch : chunks) {
                for (int rg : rings) {
                    pr.make_stager(th, ch, rg);
                    Probe & p = pr;
                    Cfg save = p.cfg;
                    p.cfg.layers = 3;
                    Stats g;
                    for (int rep = 0; rep < 3; ++rep) {
                        std::vector<double> w;
                        if (!p.run_copy_only(kind, w)) break;
                        for (double x : w) g.add((double) cfg.tensors * p.tbytes / 1e6 / x);
                    }
                    p.cfg = save;
                    p.destroy_stager();
                    sweep.push_back({th, ch, rg, g.med()});
                }
            }
        }
        std::sort(sweep.begin(), sweep.end(), [](const Sw & a, const Sw & b) { return a.gbs > b.gbs; });
        for (size_t i = 0; i < std::min<size_t>(8, sweep.size()); ++i) {
            printf("  %2d threads, chunk %2zu MiB, ring %2d: %.2f GB/s\n", sweep[i].th, sweep[i].chunk, sweep[i].ring, sweep[i].gbs);
        }
        printf("  ... worst: %d threads, chunk %zu MiB, ring %d: %.2f GB/s (%zu configurations)\n", sweep.back().th, sweep.back().chunk, sweep.back().ring, sweep.back().gbs, sweep.size());
    }

    if (!cfg.json.empty()) {
        FILE * f = fopen(cfg.json.c_str(), "w");
        if (f) {
            fprintf(f, "{");
            jstr(f, "device", dev->describe());
            jstr(f, "simulated", sim ? "yes" : "no");
            jnum(f, "tensor_mib", (double) cfg.mib);
            jnum(f, "tensors", cfg.tensors);
            jnum(f, "graphs", cfg.graphs);
            jnum(f, "kernel_ms", cfg.kernel_ms);
            jnum(f, "layers_timed", cfg.layers - 1);
            jnum(f, "reps", cfg.reps);
            jnum(f, "stager_threads", cfg.threads);
            jnum(f, "stager_chunk_mib", (double) cfg.chunk_mib);
            jnum(f, "stager_ring", cfg.ring);
            jstat(f, "T_c_ms", Tc_all);
            fprintf(f, "\"modes\": [");
            bool first = true;
            for (auto & r : results) {
                if (r.mode == "none") continue;
                fprintf(f, "%s{", first ? "" : ", ");
                first = false;
                jstr(f, "mode", r.mode);
                if (!r.ok) { jstr(f, "not_run", r.note, true); fprintf(f, "}"); continue; }
                const double Tu = r.Tu.med(), To = r.To.med(), serial = Tc + Tu, ideal = std::max(Tc, Tu);
                const double denom = serial - ideal;
                jnum(f, "E", denom > 1e-6 ? (serial - To) / denom : 0);
                jstat(f, "T_u_ms", r.Tu);
                jstat(f, "T_o_ms", r.To);
                jstat(f, "host_blocked_ms", r.blocked);
                jstat(f, "copy_gb_s", r.copy_gbs, true);
                fprintf(f, "}");
            }
            fprintf(f, "], \"sweep_top\": [");
            for (size_t i = 0; i < std::min<size_t>(8, sweep.size()); ++i) {
                fprintf(f, "%s{\"threads\": %d, \"chunk_mib\": %zu, \"ring\": %d, \"gb_s\": %.3f}", i ? ", " : "", sweep[i].th, sweep[i].chunk, sweep[i].ring, sweep[i].gbs);
            }
            fprintf(f, "]}\n");
            fclose(f);
        }
    }
    dev->dev_sync();
    return 0;
}
