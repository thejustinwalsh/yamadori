#!/usr/bin/env python
"""The A4000 coordinator (mcp/gpu_room.py), against a fake stack. No GPU.

    python mcp/test_gpu_room.py      -> "N/M checks passed"

WHAT THIS GATES (docs/SELF-IMPROVEMENT-LOG.md #16). The live gate of
2026-09-24 ended "look, draw" with the A4000 at 16,068 of 16,376 MiB (308
free): llama-swap's groups act only when an exclusive member loads, so the
image server loaded beside the vision copy. The operator's rule: if it fits
with headroom, fine; if not, drop what must go and load what is needed.

THE FAKE. One HTTP server plays llama-swap: `GET /running`,
`POST /api/models/unload/<id>`, and the four endpoints that make it load an
A4000 model (/v1/chat/completions for the vision copy, /sdapi/v1/txt2img,
/v1/embeddings, /v1/rerank), with llama-swap's own group behaviour (vision is
exclusive; the two image models swap) kept as the backstop. A simulated card
holds what is loaded (sizes from gpu_room.SIZES), Laya as a fixed resident,
and optionally a foreign consumer; nvidia-smi is replaced by a read of that
card. Every allocation updates the card's minimum free memory, and one past
the card's total is recorded as an out-of-memory.

The sequences run through the REAL callers -- vision.describe (via
model.post), images.generate, code_search.embed / rerank -- so the test
covers the wiring, not only the module. A control arm runs the row-16
sequence with the coordinator off and must reproduce the failure, or the
check that the coordinator prevents it could not fail.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_gpu_room_")


def _dead_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


DEAD = f"http://127.0.0.1:{_dead_port()}"

# --------------------------------------------------------------------------
# The simulated A4000 and the fake llama-swap in front of it
# --------------------------------------------------------------------------
TOTAL = 16376                      # the A4000's total, nvidia-smi (#16)
LAYA = 2465                        # 7,565 - (2,100 + 3,000): see SIZES
RETRIEVAL = {"embeddings", "reranker"}
IMAGE = {"imagegen", "imagegen-turbo"}


def _png() -> bytes:
    raw = b"\x00" + b"\xff\x00\x00" * 2 + b"\x00" + b"\x00\x00\xff" * 2

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Card:
    """What llama-swap has loaded on the A4000, in MiB, and what that
    leaves free."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self, held: dict | None = None, foreign: int = 0,
              backstop: bool = True, hold_s: float = 0.0):
        import gpu_room
        with self.lock:
            self.sizes = {m: (s.peak_mib, s.resident_mib)
                          for m, s in gpu_room.SIZES.items()}
            self.held = {m: self.sizes[m][1] for m in (held if held is not None
                                                       else RETRIEVAL)}
            self.foreign = foreign
            self.backstop = backstop
            self.hold_s = hold_s
            self.min_free = self._free()
            self.oom: list[tuple] = []
            self.unloads: list[tuple[float, str]] = []
            self.loads: list[tuple[float, str]] = []
            self.served: list[str] = []
            self.smi_reads = 0
            # On the OTHER card: listed by /running, never on this one.
            self.elsewhere = ["bonsai"]

    def _free(self) -> int:
        return TOTAL - LAYA - self.foreign - sum(self.held.values())

    def _note(self, what: str):
        f = self._free()
        self.min_free = min(self.min_free, f)
        if f < 0:
            self.oom.append((what, f))

    def free(self) -> int:
        with self.lock:
            return self._free()

    def smi(self) -> dict:
        with self.lock:
            self.smi_reads += 1
            f = self._free()
            return {"free_mib": f, "used_mib": TOTAL - f, "total_mib": TOTAL}

    def running(self) -> dict:
        with self.lock:
            rows = [{"model": m, "state": "ready", "ttl": 0 if m in RETRIEVAL
                     else 300} for m in self.held]
            rows += [{"model": m, "state": "ready", "ttl": 0}
                     for m in self.elsewhere]
            return {"running": rows}

    def unload(self, m: str):
        with self.lock:
            self.unloads.append((time.time(), m))
            self.held.pop(m, None)
            if m in self.elsewhere:
                self.elsewhere.remove(m)

    def serve(self, m: str):
        """A request for `m`: llama-swap loads it if needed (with its group
        rules as the backstop), a growing server goes to peak for the
        request, then back to idle."""
        with self.lock:
            self.served.append(m)
            if m not in self.held:
                if self.backstop:
                    if m == "bonsai-vision":           # `ondemand`, exclusive
                        for x in list(self.held):
                            if x in RETRIEVAL | IMAGE:
                                del self.held[x]
                    if m in IMAGE:                     # `imagegen`, swap
                        for x in IMAGE - {m}:
                            self.held.pop(x, None)
                self.loads.append((time.time(), m))
            self.held[m] = self.sizes[m][0]
            self._note(f"serve {m}")
        if self.hold_s:
            time.sleep(self.hold_s)
        with self.lock:
            if m in self.held:
                self.held[m] = self.sizes[m][1]


CARD = Card.__new__(Card)          # filled once gpu_room is importable


class _Swap(BaseHTTPRequestHandler):
    def _send(self, code: int, payload) -> None:
        data = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass

    def do_GET(self):                                            # noqa: N802
        if self.path == "/running":
            return self._send(200, CARD.running())
        return self._send(404, {"error": "not found"})

    def do_POST(self):                                           # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if self.path.startswith("/api/models/unload/"):
            CARD.unload(self.path.rsplit("/", 1)[1])
            return self._send(200, {"ok": True})
        if self.path == "/v1/embeddings":
            CARD.serve(body.get("model"))
            return self._send(200, {"data": [{"embedding": [1.0, 0.5, 0.25]}
                                             for _ in body.get("input") or []]})
        if self.path == "/v1/rerank":
            CARD.serve(body.get("model"))
            return self._send(200, {"results": [
                {"index": i, "relevance_score": 1.0 / (i + 1)}
                for i in range(len(body.get("documents") or []))]})
        if self.path == "/sdapi/v1/txt2img":
            CARD.serve(body.get("model"))
            return self._send(200, {"images": [
                base64.b64encode(_png()).decode()]})
        if self.path == "/v1/chat/completions":
            CARD.serve(body.get("model"))
            return self._send(200, {
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "left red, right blue"}}],
                "usage": {"prompt_tokens": 170, "completion_tokens": 9}})
        return self._send(404, {"error": "not found"})

    def log_message(self, *a):                                   # noqa: D102
        pass


_server = ThreadingHTTPServer(("127.0.0.1", 0), _Swap)
_server.daemon_threads = True
threading.Thread(target=_server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{_server.server_address[1]}"

os.environ.update({
    "YAMADORI_GPU_ROOM": "1",
    "YAMADORI_GPU_ROOM_DIR": os.path.join(_TMP, "room"),
    "YAMADORI_GPU_ROOM_SETTLE": "2",
    "LLAMA_STACK_URL": URL,
    "YAMADORI_IMAGEGEN_URL": URL,
    "YAMADORI_ACCOUNTS_DIR": os.path.join(_TMP, "accounts"),
    "YAMADORI_CORPUS_DB": os.path.join(_TMP, "corpus.sqlite3"),
    "YAMADORI_NEBARI_DB": os.path.join(_TMP, "nebari.sqlite3"),
    "RINGS_DB": os.path.join(_TMP, "rings.sqlite3"),
    "YAMADORI_PKG_DIR": os.path.join(_TMP, "pkgs"),
    "CODE_INDEX_DB": os.path.join(_TMP, "code.sqlite3"),
    "YAMADORI_INDEX_DIR": os.path.join(_TMP, "repos"),
    "YAMADORI_MEDIA_DIR": os.path.join(_TMP, "media"),
    "YAMADORI_MEDIA_SECRET_FILE": os.path.join(_TMP, "media_url.key"),
    "YAMADORI_MODEL_SERVER": DEAD, "LAYA_URL": DEAD,
})
for k in ("YAMADORI_VISION", "YAMADORI_VISION_MODEL", "YAMADORI_IMAGEGEN_MODEL",
          "YAMADORI_IMAGEGEN_TURBO_MODEL", "YAMADORI_A4000_HEADROOM_MIB",
          "YAMADORI_GPU_ROOM_WAIT", "EMBED_MODEL", "RERANK_MODEL"):
    os.environ.pop(k, None)

import gpu_room  # noqa: E402
import budget  # noqa: E402
import tiers  # noqa: E402
import images  # noqa: E402
import vision  # noqa: E402
import code_search as cs  # noqa: E402

tiers._accepted = tiers.FALLBACK_EFFORTS     # no network for the template
_real_budgets = budget.budgets
budget.budgets = lambda pool=None: _real_budgets(pool or 163840)

CARD.__init__()
gpu_room.CARD_READER = CARD.smi
H = gpu_room.HEADROOM_MIB

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def fresh(held=None, foreign: int = 0, backstop: bool = True,
          hold_s: float = 0.0) -> None:
    """A new card and an empty coordinator state (no leases, no history)."""
    CARD.reset(held, foreign, backstop, hold_s)
    try:
        os.remove(os.path.join(os.environ["YAMADORI_GPU_ROOM_DIR"],
                               "gpu_room.json"))
    except OSError:
        pass
    gpu_room.ROOM_WAIT_S = 30.0
    os.environ["YAMADORI_GPU_ROOM"] = "1"


def look() -> dict:
    return vision.describe(_png(), "png", "What colour is each half?")


def draw() -> list:
    return images.generate("a red apple", size="512x512", model="turbo")


def search() -> None:
    cs.embed(["how is a tool call parsed"], is_query=True)
    cs.rerank("how is a tool call parsed", ["a", "b"], 2)


def _unloaded() -> list[str]:
    return [m for _t, m in CARD.unloads]


def _actions(sink: list) -> list[tuple]:
    return [(d["model"], d["action"]) for d in sink]


# --------------------------------------------------------------------------
def test_the_table_matches_the_config():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"),
                              encoding="utf-8"))
    on, off = set(), set()
    for mid, m in (cfg.get("models") or {}).items():
        env = " ".join(m.get("env") or [])
        if gpu_room.CARD_UUID in env:
            on.add(mid)
        elif "CUDA_VISIBLE_DEVICES=" in env:
            off.add(mid)
    check(set(gpu_room.SIZES) == on,
          "SIZES has a row for exactly the models config.yaml pins to the "
          "A4000's UUID", f"table={sorted(gpu_room.SIZES)} config={sorted(on)}")
    check(gpu_room.NEVER <= (off | {"bonsai-agent"}) and "bonsai" in off,
          "the models the coordinator never touches are pinned to the OTHER "
          "card", f"never={sorted(gpu_room.NEVER)} other card={sorted(off)}")
    check(all(s.source and s.peak_mib >= s.resident_mib > 0
              for s in gpu_room.SIZES.values()),
          "every row names its source; peak >= resident > 0")
    unmeasured = sorted(m for m, s in gpu_room.SIZES.items() if not s.measured)
    check(gpu_room.SIZES["imagegen"].measured and all(
        "ESTIMATE" in gpu_room.SIZES[m].source
        or "ARITHMETIC" in gpu_room.SIZES[m].source
        or "measured" in gpu_room.SIZES[m].source
        for m in unmeasured),
          "only imagegen's row is a reading of that model; every other row "
          "says it is an estimate or borrowed", str(unmeasured))
    check(H == 1331, "headroom is the operator's 1.3 GB (1,331 MiB) unless "
          "overridden", str(H))


def test_the_control_arm_reproduces_row_16():
    # The coordinator OFF: exactly what the live gate had, llama-swap's
    # groups alone. If this arm did not fail, the next test could not either.
    fresh()
    os.environ["YAMADORI_GPU_ROOM"] = "0"
    try:
        look()
        draw()
        search()
    finally:
        os.environ["YAMADORI_GPU_ROOM"] = "1"
    check(CARD.min_free < H,
          f"CONTROL: without the coordinator, look -> draw -> search takes the "
          f"card below the headroom (min free {CARD.min_free:,} MiB, "
          f"{len(CARD.oom)} allocations past the total)",
          json.dumps({"min_free": CARD.min_free, "oom": CARD.oom}))


def test_look_draw_search():
    fresh()
    sink: list = []
    with gpu_room.recording(sink):
        r1 = look()
        after_look = dict(CARD.held)
        r2 = draw()
        after_draw = dict(CARD.held)
        search()
    ev = json.dumps({"decisions": sink, "unloads": _unloaded(),
                     "min_free": CARD.min_free})[:1500]
    check(r1.get("answer") and r2 and r2[0].get("id"),
          "look -> draw -> search: all three calls succeed through the real "
          "callers (vision.describe, images.generate, code_search)", ev)
    check(CARD.min_free >= H and not CARD.oom,
          f"the A4000 never goes below {H:,} MiB free (min "
          f"{CARD.min_free:,}); no allocation past the total", ev)
    look_d = sink[0]
    check(look_d["model"] == "bonsai-vision" and look_d["action"] == "evicted"
          and [e["model"] for e in look_d["evicted"]] == ["reranker"],
          "look: vision (9,449 est.) does not fit beside search, so the "
          "coordinator unloads the reranker -- the bigger of two never-used "
          "models -- and stops as soon as it fits", ev)
    check(look_d["free_after_mib"] - look_d["need_mib"] >= H
          and look_d["free_before_mib"] - look_d["need_mib"] < H,
          "look: free before < need + headroom <= free after (the "
          "measurement, re-read after each unload, decides)",
          json.dumps(look_d))
    check("bonsai-vision" in after_look and not (set(after_look) & RETRIEVAL),
          "the backstop still acts: loading vision (exclusive) took "
          "embeddings out too", json.dumps(after_look))
    draw_d = next(d for d in sink if d["model"] == "imagegen-turbo")
    check(draw_d["action"] == "evicted"
          and [e["model"] for e in draw_d["evicted"]] == ["bonsai-vision"]
          and "bonsai-vision" not in after_draw,
          "draw: the image server's peak (6,389) does not fit beside vision, "
          "so vision leaves before the draw", json.dumps(draw_d))
    later = [d for d in sink if d["model"] in RETRIEVAL]
    check([d["action"] for d in later] == ["fit", "fit"]
          and not any(d.get("evicted") for d in later),
          "search: embeddings and the reranker load beside the idle image "
          "server with room to spare, and unload nothing (search + image "
          "generator fits)", json.dumps(later)[:600])
    check(set(CARD.held) == {"imagegen-turbo", "embeddings", "reranker"},
          "afterwards: search and the idle image server are resident",
          json.dumps(CARD.held))


def test_draw_then_look():
    fresh()
    sink: list = []
    with gpu_room.recording(sink):
        draw()
        after_draw = list(_unloaded())
        look()
    ev = json.dumps({"decisions": sink, "unloads": _unloaded()})[:1500]
    check(after_draw == [] and sink[0]["action"] == "fit",
          "draw first: the image server fits beside search (search + image "
          "generator), nothing is unloaded", ev)
    look_d = sink[-1]
    check(look_d["model"] == "bonsai-vision"
          and [e["model"] for e in look_d["evicted"]] == ["reranker"],
          "then look: the least recently used goes first -- the never-used "
          "reranker, not the image server that just drew", ev)
    check(CARD.min_free >= H and not CARD.oom,
          f"draw -> look never goes below {H:,} MiB free (min "
          f"{CARD.min_free:,})", ev)


def test_a_model_already_loaded():
    fresh()
    reads = CARD.smi_reads
    sink: list = []
    with gpu_room.recording(sink):
        cs.embed(["q"], is_query=True)
    check(_actions(sink) == [("embeddings", "loaded")]
          and CARD.smi_reads == reads and not CARD.unloads,
          "a loaded search model takes the fast path: decision `loaded`, no "
          "nvidia-smi read, nothing unloaded", json.dumps(sink))
    # A LOADED image server still grows from idle to peak on each draw: it is
    # not "nothing to do" when vision sits beside it.
    fresh(held={"bonsai-vision", "imagegen-turbo"})
    sink = []
    with gpu_room.recording(sink):
        draw()
    d = sink[0]
    check(d["action"] == "evicted" and d["loaded"] is True
          and d["need_mib"] == 6389 - 319
          and [e["model"] for e in d["evicted"]] == ["bonsai-vision"]
          and CARD.min_free >= H,
          "a loaded but idle image server counts its growth (6,389 - 319 = "
          "6,070) and still makes room", json.dumps(d))


def test_concurrent_callers():
    # Three callers at once from the preloaded card; each request holds its
    # model for 0.4 s. Alone each fits or makes room; together they must not
    # race into the same free megabytes.
    fresh(hold_s=0.4)
    errors: list = []
    sink_lock = threading.Lock()
    sink: list = []

    def run(fn):
        mine: list = []
        try:
            with gpu_room.recording(mine):
                fn()
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"{fn.__name__}: {type(e).__name__}: {e}")
        with sink_lock:
            sink.extend(mine)

    ts = [threading.Thread(target=run, args=(f,)) for f in (look, draw, search)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    ev = json.dumps({"errors": errors, "decisions": sink,
                     "min_free": CARD.min_free, "oom": CARD.oom})[:1500]
    check(not errors and CARD.min_free >= H and not CARD.oom,
          f"three concurrent callers (look, draw, search): all succeed and "
          f"the card never goes below {H:,} MiB free (min {CARD.min_free:,})",
          ev)

    # A model in use is never unloaded under its request: the caller waits.
    # A 1,500 MiB foreign consumer makes vision need BOTH search models gone.
    fresh(foreign=1500)
    released: dict = {}
    started = threading.Event()

    def hold_embeddings():
        with gpu_room.use("embeddings", upstream=URL):
            started.set()
            time.sleep(0.8)
            released["t"] = time.time()

    t = threading.Thread(target=hold_embeddings)
    t.start()
    started.wait(5)
    d = gpu_room.ensure_room("bonsai-vision", upstream=URL)
    t.join(5)
    t_unload = next((tt for tt, m in CARD.unloads if m == "embeddings"), None)
    check(t_unload is not None and "t" in released and t_unload >= released["t"]
          and d["waited_s"] >= 0.5 and "embeddings" in d["in_use"],
          "a search model another request is using is unloaded only after "
          "that request releases it (vision waited)",
          json.dumps({"decision": d, "released": released.get("t"),
                      "unloads": CARD.unloads}))

    # ACROSS PROCESSES: another process's lease pins the model too (the
    # worker and the tools API load embeddings from their own processes).
    fresh(foreign=1500)
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import os,sys,time;sys.path.insert(0,%r);import gpu_room;"
         "u=gpu_room.use('embeddings',upstream=%r);d=u.__enter__();"
         "print('pinned',d and d.get('action'),flush=True);time.sleep(1.5);"
         "u.__exit__(None,None,None);print('released',time.time(),flush=True)"
         % (HERE, URL)],
        stdout=subprocess.PIPE, text=True, env=dict(os.environ))
    first = ""
    while not first.startswith("pinned"):       # skip the child's own log line
        first = child.stdout.readline().strip()
        if not first and child.poll() is not None:
            break
    d = gpu_room.ensure_room("bonsai-vision", upstream=URL)
    rest = child.stdout.read()
    child.wait(10)
    rel = float(rest.split("released", 1)[1].split()[0]) if "released" in rest else None
    t_unload = next((tt for tt, m in CARD.unloads if m == "embeddings"), None)
    check(first == "pinned loaded" and rel is not None and t_unload is not None
          and t_unload >= rel - 0.05 and CARD.min_free >= H,
          "a lease taken in ANOTHER process pins the model: vision waits for "
          "the other process to release embeddings before unloading it",
          json.dumps({"child": [first, rest.strip()], "decision": d,
                      "unloads": CARD.unloads}))

    # And the room lock itself excludes another process.
    fresh(held=set())
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys,time;sys.path.insert(0,%r);import gpu_room;"
         "lk=gpu_room.room_lock();ok=lk.acquire(5);print('held',ok,flush=True);"
         "time.sleep(1.5);lk.release()" % HERE],
        stdout=subprocess.PIPE, text=True, env=dict(os.environ))
    line = holder.stdout.readline().strip()     # acquire() logs nothing
    gpu_room.ROOM_WAIT_S = 0.4
    try:
        gpu_room.ensure_room("bonsai-vision", upstream=URL)
        busy = None
    except gpu_room.NoRoom as e:
        busy = e
    gpu_room.ROOM_WAIT_S = 10.0
    d = gpu_room.ensure_room("bonsai-vision", upstream=URL)
    holder.wait(10)
    check(line == "held True" and busy is not None
          and busy.code == "A4000_BUSY" and busy.retryable is True
          and d["action"] == "fit",
          "the card's room lock is held across processes: a second process "
          "waiting 0.4 s gets A4000_BUSY (retryable), then gets the card once "
          "the first releases it",
          json.dumps({"holder": line, "busy": busy and busy.envelope("t"),
                      "then": d})[:900])


def test_nothing_evictable():
    # A foreign consumer holds 6,000 MiB and nothing the coordinator may
    # unload is loaded: vision cannot fit, and nothing is touched.
    fresh(held=set(), foreign=6000)
    sink: list = []
    err = None
    with gpu_room.recording(sink):
        try:
            look()
        except images.ImageError as e:
            err = e
    env = err.envelope("describe_image") if err else {}
    check(err is not None and err.code == "A4000_NO_ROOM"
          and err.retryable is False and err.status == 507
          and not CARD.unloads and "bonsai-vision" not in CARD.served,
          "vision with nothing evictable: A4000_NO_ROOM, not retryable, "
          "nothing unloaded, and the request never reached llama-swap (no "
          "load into an out-of-memory)", json.dumps(env)[:900])
    owners = {r.get("fixable_by") for r in env.get("remedies") or []}
    check({"operator", "agent"} <= owners and env.get("need_mib") == 9449
          and env.get("headroom_mib") == H and env.get("free_mib") is not None,
          "the failure carries the numbers and a remedy for each owner",
          json.dumps(env)[:900])
    check(sink and sink[-1]["action"] == "no_room",
          "the refusal is recorded (x_yamadori.gpu_room)", json.dumps(sink))

    # A model that cannot fit even on an empty card: nothing is unloaded to
    # find that out.
    fresh(held={"embeddings", "reranker", "imagegen-turbo"})
    try:
        gpu_room.ensure_room("critic-disabled", upstream=URL)
        e2 = None
    except gpu_room.NoRoom as e:
        e2 = e
    check(e2 is not None and e2.code == "A4000_NO_ROOM" and not CARD.unloads,
          "a model bigger than everything evictable is refused WITHOUT "
          "unloading anything first", str(e2))

    # Through the image tool: the envelope the model reads.
    fresh(held=set(), foreign=9000)
    out = json.loads(images.run_tool({"prompt": "a fox"}, "http://example"))
    check(out.get("ok") is False and out.get("error") == "A4000_NO_ROOM"
          and out.get("remedies"),
          "generate_image returns the A4000_NO_ROOM envelope to the model",
          json.dumps(out)[:600])
    # And through search: the error propagates as NoRoom, not an empty list.
    fresh(held=set(), foreign=13000)
    try:
        cs.embed(["q"], is_query=True)
        e3 = None
    except gpu_room.NoRoom as e:
        e3 = e
    check(e3 is not None and e3.code == "A4000_NO_ROOM",
          "search on a card with no room raises NoRoom (never a silent "
          "empty result)", str(e3))


def test_the_main_model_is_never_touched():
    fresh(held={"embeddings", "reranker"})
    CARD.elsewhere += ["bonsai-agent", "bonsai-q4kv"]
    err = gpu_room.unload(URL, "bonsai")
    err2 = gpu_room.unload(URL, "bonsai-q4kv")
    check(err and err.startswith("refused") and err2
          and err2.startswith("refused") and not CARD.unloads,
          "unload() refuses the main model and anything not in the table, "
          "and sends nothing", f"{err} | {err2} | {CARD.unloads}")
    # The biggest demand there is, with the main model listed: everything
    # evictable on the A4000 may go, the main model never.
    try:
        gpu_room.ensure_room("critic-disabled", upstream=URL)
    except gpu_room.NoRoom:
        pass
    fresh(held={"embeddings", "reranker", "imagegen"}, foreign=3000)
    try:
        gpu_room.ensure_room("bonsai-vision", upstream=URL)
    except gpu_room.NoRoom:
        pass
    check(not any(m in gpu_room.NEVER or m not in gpu_room.SIZES
                  for m in _unloaded()),
          "under maximum pressure only A4000 table models are unloaded",
          str(CARD.unloads))


def test_no_llama_swap_means_uncoordinated_not_blocked():
    # Pre-deploy review, 2026-09-24 (FIX SOON #6): an unreadable card used to
    # let ANY load through as `uncoordinated`. Operator rule: never load into
    # an out-of-memory. An on-demand model is refused; a resident search
    # model (presumed loaded, allocating nothing) goes ahead.
    fresh()
    sink: list = []
    with gpu_room.recording(sink):
        try:
            with gpu_room.use("bonsai-vision", upstream=DEAD):
                pass
            err = None
        except gpu_room.NoRoom as e:
            err = e
    env = err.envelope("describe_image") if err else {}
    check(err is not None and err.code == "A4000_UNREADABLE"
          and err.retryable is True
          and {r.get("fixable_by") for r in err.remedies} >= {"agent",
                                                               "operator"}
          and "NOT loaded" in err.reason and sink
          and sink[-1]["action"] == "unreadable",
          "llama-swap unreachable: an ON-DEMAND load (vision) is refused with "
          "the situation, retryable as a fact, and a remedy per owner; "
          "recorded as `unreadable`", json.dumps(env)[:600])
    for m in ("imagegen-turbo", "imagegen"):
        try:
            with gpu_room.use(m, upstream=DEAD):
                pass
            code = None
        except gpu_room.NoRoom as e:
            code = e.code
        check(code == "A4000_UNREADABLE",
              f"llama-swap unreachable: `{m}` is refused too", str(code))
    sink = []
    with gpu_room.recording(sink):
        with gpu_room.use("embeddings", upstream=DEAD) as d:
            pass
    check(d and d["action"] == "uncoordinated" and "did not answer" in d["why"]
          and "resident" in d["why"] and sink == [d],
          "llama-swap unreachable: a RESIDENT search model goes ahead, "
          "recorded as `uncoordinated` with the reason", json.dumps(d))
    # /running answers but nvidia-smi does not: the load (vision is not
    # loaded) is refused; an already-loaded search model takes its lease.
    fresh(held={"embeddings", "reranker"})
    saved = gpu_room.CARD_READER
    gpu_room.CARD_READER = lambda: None
    try:
        try:
            gpu_room.ensure_room("bonsai-vision", upstream=URL)
            code = None
        except gpu_room.NoRoom as e:
            code = e.code
        with gpu_room.use("embeddings", upstream=URL) as d_emb:
            pass
    finally:
        gpu_room.CARD_READER = saved
    check(code == "A4000_UNREADABLE" and not CARD.unloads
          and "bonsai-vision" not in CARD.served,
          "nvidia-smi unreadable: vision is refused, nothing unloaded, "
          "nothing loaded", str(code))
    check(d_emb and d_emb["action"] == "loaded",
          "nvidia-smi unreadable: a loaded, resident search model is still "
          "served (fast path, no card read)", json.dumps(d_emb))
    os.environ["YAMADORI_GPU_ROOM"] = "0"
    with gpu_room.use("bonsai-vision", upstream=URL) as d2:
        pass
    os.environ["YAMADORI_GPU_ROOM"] = "1"
    check(d2 is None and not CARD.unloads,
          "YAMADORI_GPU_ROOM=0 turns it off (the offline suites' setting)")
    with gpu_room.use("bonsai", upstream=URL) as d3:
        pass
    check(d3 is None, "the main model is not the coordinator's: no decision")


def test_the_chat_path_never_waits_for_a_draw():
    """Pre-deploy review, 2026-09-24 (FIX SOON #5): a chat request's skill /
    hint selection embeds through use(), and with the search model not
    loaded it waited up to ROOM_WAIT_S (300 s) for the room lock an image
    draw holds for its whole run. Inside fail_fast() it waits at most
    FAIL_FAST_WAIT_S (2 s) in all, however many embeddings it makes."""
    import skill_select
    fresh(held=set())                  # embeddings evicted (vision ran)
    gpu_room.ROOM_WAIT_S = 30.0
    held_room = threading.Event()
    let_go = threading.Event()

    def a_draw():                      # holds the room as images.py does
        lk = gpu_room.room_lock()
        lk.acquire(5)
        held_room.set()
        let_go.wait(20)
        lk.release()
    t = threading.Thread(target=a_draw)
    t.start()
    held_room.wait(5)
    sink: list = []
    t0 = time.time()
    try:
        with gpu_room.recording(sink), gpu_room.fail_fast("a chat turn"):
            try:
                cs.embed(["q"], is_query=True)
                err = None
            except gpu_room.NoRoom as e:
                err = e
            # The real selection path (its trigger store stubbed: no real
            # skills database is read): an embedding failure is reported.
            saved = (skill_select.trigger_rows, skill_select.TRIGGER_CACHE)
            skill_select.trigger_rows = lambda pool: [("s1", "bind a buffer")]
            skill_select.TRIGGER_CACHE = os.path.join(_TMP, "triggers.npz")
            try:
                best, why = skill_select.best_cosines(
                    "how do I bind a buffer", [{"id": "s1"}])
            finally:
                skill_select.trigger_rows, skill_select.TRIGGER_CACHE = saved
        took = time.time() - t0
    finally:
        let_go.set()
        t.join(10)
    check(err is not None and err.code == "A4000_BUSY" and took < 3.0
          and sink and sink[0]["action"] == "busy"
          and sink[0].get("fail_fast") is True
          and "embeddings" not in CARD.served,
          f"embeddings NOT loaded, a draw holding the room: the chat path "
          f"fails fast ({took:.2f}s, not {gpu_room.ROOM_WAIT_S:.0f}s), and "
          f"the decision is recorded", json.dumps({"err": str(err),
                                                   "sink": sink})[:600])
    check(not best and why.get("ok") is False
          and "NoRoom" in (why.get("why") or "")
          and took < 3.0,
          "skill selection records the skip (embedding failed: NoRoom) and "
          "goes on, at once", json.dumps(why))
    # Loaded: the lease needs no room lock, so a draw does not stall it.
    fresh()
    held_room.clear()
    let_go.clear()
    t = threading.Thread(target=a_draw)
    t.start()
    held_room.wait(5)
    t0 = time.time()
    try:
        with gpu_room.fail_fast("a chat turn"):
            cs.embed(["q"], is_query=True)
        took = time.time() - t0
    finally:
        let_go.set()
        t.join(10)
    check(took < 3.0 and "embeddings" in CARD.served,
          f"embeddings loaded, a draw holding the room: served at once on "
          f"its lease ({took:.2f}s)")
    # Outside fail_fast (a tool, the worker) the wait is unchanged.
    check(gpu_room._wait_budget() == (gpu_room.ROOM_WAIT_S, None),
          "outside fail_fast() a caller still waits ROOM_WAIT_S")
    # The proxy wraps prepare() in it.
    import inspect
    import proxy
    src = inspect.getsource(proxy._run_turn)
    check("gpu_room.fail_fast(" in src.split("payload = prepare(body)")[0],
          "proxy._run_turn runs prepare() inside gpu_room.fail_fast()")


def test_the_chat_path_loads_embeddings_when_the_room_is_free():
    """Live gate 2026-09-24: after a restart the A4000 is empty, and with a
    ZERO wait the chat path skipped its embedding (hints, library help, E1)
    whenever the room lock was held even for a moment -- another process's
    load decision, a look's lease-free check -- and nothing else loads the
    embedder for it. Now the chat path waits up to FAIL_FAST_WAIT_S (2 s)
    for the room: a lock that comes free in that time is taken, and the
    chat path makes room and LOADS the embedder. A draw (held far longer)
    still fails fast."""
    fresh(held=set())                  # the A4000 empty, as after a restart
    gpu_room.ROOM_WAIT_S = 30.0
    held_room = threading.Event()

    def a_brief_load():                # another caller's load decision
        lk = gpu_room.room_lock()
        lk.acquire(5)
        held_room.set()
        time.sleep(0.6)
        lk.release()
    t = threading.Thread(target=a_brief_load)
    t.start()
    held_room.wait(5)
    sink: list = []
    t0 = time.time()
    try:
        with gpu_room.recording(sink), gpu_room.fail_fast("a chat turn"):
            try:
                cs.embed(["q"], is_query=True)
                err = None
            except gpu_room.NoRoom as e:
                err = e
        took = time.time() - t0
    finally:
        t.join(10)
    acts = _actions(sink)
    check(err is None and "embeddings" in CARD.served
          and ("embeddings", "fit") in acts and 0.4 <= took < 3.0,
          f"the room held for 0.6 s: the chat path waited ({took:.2f}s) and "
          f"LOADED the embedder (action fit), instead of skipping",
          json.dumps({"err": str(err), "sink": sink})[:600])
    with gpu_room.fail_fast("a chat turn"):
        inside = gpu_room._wait_budget()
    check(gpu_room.FAIL_FAST_WAIT_S == 2.0 and 1.5 < inside[0] <= 2.0
          and gpu_room._wait_budget() == (gpu_room.ROOM_WAIT_S, None),
          "the chat path's wait is 2 s in all; outside fail_fast() it is "
          "unchanged", json.dumps(inside))


def test_a_vision_turn_with_no_room_is_a_structured_error():
    """Pre-deploy review, 2026-09-24 (MINOR): a client naming
    `yamadori-vision` when the A4000 has no room got gpu_room.NoRoom as a
    bare 502 (blocking) or a broken stream. Now: the refusal's status and a
    structured error -- situation, retryable as a fact, remedies."""
    import accounts
    import proxy
    from starlette.testclient import TestClient
    import server
    fresh(held=set(), foreign=13000)      # vision cannot fit, nothing to evict
    body = {"model": "yamadori-vision", "_client_ip": "127.0.0.1",
            "_features": json.dumps({"hints": False, "retrieval": False,
                                     "investigate": False, "fanout": 1}),
            "messages": [{"role": "user", "content": "Describe a cat."}]}
    try:
        proxy.complete(dict(body))
        err = None
    except proxy.TurnRefused as e:
        err = e
    b = err.body()["error"] if err else {}
    check(err is not None and err.status == 507
          and b.get("code") == "A4000_NO_ROOM" and b.get("retryable") is False
          and b.get("remedies") and "Nothing was generated" in b["message"]
          and "bonsai-vision" not in CARD.served,
          "blocking: a structured refusal (507, A4000_NO_ROOM, remedies), "
          "and nothing reached the card", json.dumps(b)[:500])
    fresh(held=set(), foreign=13000)
    raw = b"".join(proxy.stream_body(dict(body, stream=True),
                                     "yamadori-vision"))
    events = [ln[6:] for ln in raw.decode().split("\n\n") if ln.startswith(
        "data: ")]
    first = json.loads(events[0]) if events and events[0] != "[DONE]" else {}
    check(len(events) == 2 and events[-1] == "[DONE]"
          and (first.get("error") or {}).get("code") == "A4000_NO_ROOM",
          "streamed: one SSE error event in OpenAI's shape, then [DONE] -- "
          "not a broken stream", raw.decode()[:400])
    fresh(held=set(), foreign=13000)
    key = accounts.create("gpu-room-tests")
    r = TestClient(server.app).post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
        json={k: v for k, v in body.items() if not k.startswith("_")})
    got = r.json() if r.headers.get("content-type", "").startswith(
        "application/json") else {}
    check(r.status_code == 507
          and (got.get("error") or {}).get("code") == "A4000_NO_ROOM",
          "through the server: HTTP 507 with the structured error, not a "
          "bare 502", f"{r.status_code} {r.text[:300]}")


def test_the_proxy_records_decisions():
    import proxy
    fresh(held={"bonsai-vision"})
    state = {"_gpu_room": [], "_public_base": "http://example",
             "_account": ""}
    out = json.loads(proxy.run_our_tool(images.TOOL_NAME,
                                        {"prompt": "a fox"}, None,
                                        state=state))
    rec = state["_gpu_room"]
    check(out.get("ok") is True and rec and rec[0]["model"] in IMAGE
          and rec[0]["action"] == "evicted"
          and [e["model"] for e in rec[0]["evicted"]] == ["bonsai-vision"],
          "a tool the proxy runs records its A4000 decision in the request's "
          "x_yamadori.gpu_room list", json.dumps({"out": out, "rec": rec})[:900])
    x = proxy._x_yamadori({"_gpu_room": rec}, hops=0, fan=None, think=None)
    check(x.get("gpu_room") == rec,
          "x_yamadori carries the request's gpu_room decisions",
          json.dumps(x.get("gpu_room"))[:300])


TESTS = [test_the_table_matches_the_config,
         test_the_control_arm_reproduces_row_16,
         test_look_draw_search,
         test_draw_then_look,
         test_a_model_already_loaded,
         test_concurrent_callers,
         test_nothing_evictable,
         test_the_main_model_is_never_touched,
         test_no_llama_swap_means_uncoordinated_not_blocked,
         test_the_chat_path_never_waits_for_a_draw,
         test_the_chat_path_loads_embeddings_when_the_room_is_free,
         test_a_vision_turn_with_no_room_is_a_structured_error,
         test_the_proxy_records_decisions]


def main() -> int:
    for fn in TESTS:
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} raised", traceback.format_exc())
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + ("" if ok else f"\n        <- {detail[:1500]}"))
    # Across every test above: the main model was never asked to unload.
    passed = sum(ok for ok, _, _ in _results)
    print(f"\n{passed}/{len(_results)} checks passed")
    _server.shutdown()
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
