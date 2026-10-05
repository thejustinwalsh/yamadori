#!/usr/bin/env python
"""THE DASHBOARD NEVER LOADS A MODEL, asserted through the real server.app.

Operator, 2026-09-30: "right now the dashboard loading loads bonsai, so we
need to make sure we have a dashboard api that doesn't require that". Any
request to llama-swap's /upstream/<model>/... makes llama-swap LOAD that
model; a stray GET /upstream/bonsai/slots loaded bonsai in the middle of a
GPU window (03:29:50 that day).

THE FAKE LLAMA-SWAP is a real HTTP server (LLAMA_STACK_URL points at it) that
records every request. It answers GET /running (what is loaded; it never
loads) and GET /v1/models (the liveness probe); ANY /upstream/ request is a
violation and is answered 500; any other path is recorded as unexpected. The
main model's own server (YAMADORI_MODEL_SERVER, a direct llama-server port
that llama-swap does not route) is a second fake.

Every dashboard GET is sent through server.app (TestClient): the SPA's page,
every /dash/api/* route the SPA and the classic pages read, the new JJAVA and
PERFORMANCE routes -- in three states:

  off        llama-swap lists only the always-resident embedder; the main
             model's port does not answer (a GPU window holds the card)
  unreadable llama-swap's /running answers 500 (busy or restarting)
  loaded     /running lists the main model; its own server answers -- the
             positive control: the pool is re-read DIRECT, still no /upstream

AND EVERY READER OF A MODEL'S /slots (test_slot_readers_never_load): the
stray GET /upstream/bonsai/slots of that morning came from a slot reader
guarded only by max_mode.blocks(), which is False when NO main model is
loaded. model.release_slot, skill_questions_bank.slots_busy, the decider's
prime (decide_turn._idle), decider_bonsai.slot_cells, idle.slots_processing
and the two bench runners (bench/octopus/run.py slots_idle,
bench/decider/bonsai_decider.py preflight) each read /upstream/<model>/slots
only while /running lists the model ready, and otherwise skip and say why.

PASS: zero /upstream requests with the model off the card or /running
unreadable; with it loaded, /upstream only for that model (the rule:
llama-swap's GET /running first, an /upstream read ONLY when it lists the
model loaded) and only its /props; zero unexpected llama-swap paths; every
route answers (no 5xx); in `off` the KV pool says why it was not read.
Every store is a temp path; nothing reaches a stack port.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_dash_no_load_")
MAIN = os.environ.get("YAMADORI_MODEL", "bonsai")

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, str(detail)[:600]))


# ------------------------------------------------------------------ fakes ----
class Fake:
    """One fake server: its mode, and every (method, path) it was asked."""

    def __init__(self, name: str):
        self.name = name
        self.mode = "off"
        self.seen: list[tuple[str, str]] = []
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, obj) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                with fake.lock:
                    fake.seen.append(("GET", self.path))
                code, obj = fake.answer("GET", self.path)
                self._send(code, obj)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                with fake.lock:
                    fake.seen.append(("POST", self.path))
                code, obj = fake.answer("POST", self.path)
                self._send(code, obj)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def reset(self, mode: str) -> None:
        with self.lock:
            self.mode = mode
            self.seen = []

    def answer(self, method: str, path: str):
        raise NotImplementedError


class Swap(Fake):
    def loaded(self) -> set:
        return {MAIN} if self.mode == "loaded" else set()

    def answer(self, method, path):
        p = path.split("?", 1)[0]
        if p.startswith("/upstream/"):
            model = p.split("/")[2]
            if model in self.loaded() and p.endswith("/props"):
                # the rule's one allowance: a model /running lists as loaded
                return 200, {"chat_template": "{%- if reasoning_effort not in "
                             "('xhigh', 'medium', 'low') %}"}
            if model in self.loaded() and p.endswith("/slots"):
                return 200, [{"id": i, "is_processing": False,
                              "n_prompt_tokens": 0} for i in range(3)]
            return 500, {"error": "the test's llama-swap: /upstream would load a model"}
        if p == "/running":
            if self.mode == "unreadable":
                return 500, {"error": "busy"}
            rows = [{"model": "embeddings", "state": "ready",
                     "proxy": "http://127.0.0.1:9", "cmd": "llama-server -m e.gguf"}]
            if self.mode == "flash":
                # flash-next holds the card, as llama-swap reports it: `localhost`, its own port
                rows.append({"model": "flash-next", "state": "ready",
                             "proxy": FLASH.url.replace("127.0.0.1", "localhost"),
                             "cmd": f"llama-server -m flash-next.gguf --port {FLASH.port}"})
            if self.mode == "loaded":
                rows.append({"model": MAIN, "state": "ready", "proxy": DIRECT.url,
                             "cmd": f"llama-server -m {MAIN}.gguf --port {DIRECT.port}"})
            return 200, {"running": rows}
        if p in ("/v1/models", "/health"):
            return 200, {"data": [{"id": MAIN}, {"id": "embeddings"}]}
        return 404, {"error": "not a route of the test's llama-swap"}

    def upstream(self) -> list:
        """/upstream requests for a model /running did NOT list as loaded:
        each one would have made llama-swap load it."""
        return [x for x in self.seen if x[1].startswith("/upstream/")
                and x[1].split("/")[2] not in self.loaded()]

    def upstream_all(self) -> list:
        return [x for x in self.seen if x[1].startswith("/upstream/")]

    def unexpected(self) -> list:
        return [x for x in self.seen if x[1].split("?", 1)[0]
                not in ("/running", "/v1/models", "/health")
                and not x[1].startswith("/upstream/")]


class Direct(Fake):
    """The main model's own llama-server (config.yaml startPort)."""

    def answer(self, method, path):
        if self.mode != "loaded":
            return 503, {"error": {"message": "Loading model", "code": 503}}
        p = path.split("?", 1)[0]
        if p == "/props":
            return 200, {"default_generation_settings": {"n_ctx": 147456},
                         "total_slots": 3, "kv_vram_cells": 0}
        if p == "/slots":
            return 200, [{"id": i, "is_processing": False, "n_ctx": 49152,
                          "n_prompt_tokens": 0, "next_token": [{"n_decoded": 0}]}
                         for i in range(3)]
        if p == "/health":
            return 200, {"status": "ok"}
        return 404, {"error": "not served"}


class FlashServer(Fake):
    """flash-next's own llama-server: -np 1, its native window."""

    def answer(self, method, path):
        p = path.split("?", 1)[0]
        if p == "/slots":
            return 200, [{"id": 0, "is_processing": True, "id_task": 5, "n_ctx": 262144,
                          "n_prompt_tokens": 26000, "n_prompt_tokens_processed": 40,
                          "next_token": [{"n_decoded": 700, "n_remain": 1000}]}]
        if p == "/props":
            return 200, {"default_generation_settings": {"n_ctx": 262144}, "total_slots": 1}
        return 404, {"error": "not served"}


def _dead_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


SWAP = Swap("llama-swap")
DIRECT = Direct("main model server")
FLASH = FlashServer("flash-next's server")
DEAD = _dead_port()
os.environ["LLAMA_STACK_URL"] = SWAP.url
os.environ["YAMADORI_MODEL_SERVER"] = DIRECT.url
os.environ["TOOLS_API_PORT"] = str(DEAD)
os.environ["YAMADORI_SEARCH_URL"] = f"http://127.0.0.1:{DEAD}"
os.environ["YAMADORI_ACCOUNTS_DIR"] = tempfile.mkdtemp(prefix="acct_", dir=_TMP)
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ["YAMADORI_DECIDER_MODELS_DIR"] = tempfile.mkdtemp(prefix="dm_", dir=_TMP)
os.environ["YAMADORI_OCTO_DIR"] = tempfile.mkdtemp(prefix="octo_", dir=_TMP)

# Every URL this process opens is recorded too: the fakes see the network,
# this sees anything that went elsewhere (a real stack port, the internet).
import urllib.request  # noqa: E402

OPENED: list[str] = []
_real_urlopen = urllib.request.urlopen


def _recording_urlopen(url, *a, **kw):
    OPENED.append(url.full_url if hasattr(url, "full_url") else str(url))
    return _real_urlopen(url, *a, **kw)


urllib.request.urlopen = _recording_urlopen

import vitals  # noqa: E402

# nvidia-smi, PowerShell and netstat are the machine's, not the network's:
# answered empty so the suite is hermetic and fast.
vitals._sh = lambda cmd, timeout=25: ""

import server  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

CLIENT = TestClient(server.app)
HTML = {"accept": "text/html"}

ROUTES = ["/dash/api/vitals", "/dash/api/vitals/pulse", "/dash/api/power",
          "/dash/api/power/series", "/dash/api/tiers", "/dash/api/tokens",
          "/dash/api/mcp", "/dash/api/skill-factory/library",
          "/dash/api/datasets", "/dash/api/skills",
          "/dash/api/skill-factory/prompts",
          "/dash/api/skill-factory/selections",
          "/dash/api/skill-factory/recent",
          "/dash/api/skill-factory/onboarding",
          "/dash/api/harness-kit", "/dash/api/harness-kit/export/hermes",
          "/dash/api/settings/image",
          "/dash/api/jjava", "/dash/api/jjava/1h", "/dash/api/jjava/7d",
          "/dash/api/perf", "/dash/api/perf/1h", "/dash/api/perf/30d"]
PAGES = ["/", "/jjava", "/performance", "/skills", "/nebari", "/dash/classic/vitals",
         "/dash/classic/data"]


def _reset_caches() -> None:
    import budget
    import dash_jjava
    import dash_skills
    import dash_perf
    import max_mode
    import tiers
    vitals._running_cache = (-1e9, None)
    max_mode._running_cache = (0.0, set())
    tiers._accepted = None
    tiers._accepted_of.clear()
    budget._POOL = None
    dash_jjava._cache.clear()
    dash_perf._cache.clear()
    dash_skills._library_cache.clear()


def _run(mode: str) -> dict:
    _reset_caches()
    SWAP.reset(mode)
    DIRECT.reset("loaded" if mode == "loaded" else "off")
    OPENED.clear()
    codes: dict[str, int] = {}
    bodies: dict[str, object] = {}
    for path in ROUTES:
        r = CLIENT.get(path)
        codes[path] = r.status_code
        try:
            bodies[path] = r.json()
        except ValueError:
            bodies[path] = None
    for path in PAGES:
        r = CLIENT.get(path, headers=HTML)
        codes[path] = r.status_code
    return {"codes": codes, "bodies": bodies,
            "upstream": list(SWAP.upstream()),
            "upstream_all": list(SWAP.upstream_all()),
            "unexpected": list(SWAP.unexpected()),
            "direct": list(DIRECT.seen), "opened": list(OPENED)}


def _stack_ports(urls: list[str]) -> list[str]:
    real = (":11434", ":1234/", ":1235", ":10001", ":8888")
    return [u for u in urls if any(p in u for p in real)
            and not u.startswith((SWAP.url, DIRECT.url))]


def test_off_the_card():
    out = _run("off")
    check(not out["upstream_all"], "main model off the card: ZERO /upstream "
          "requests", json.dumps(out["upstream_all"]))
    check(not out["unexpected"], "and no other llama-swap path than /running, "
          "/v1/models, /health", json.dumps(out["unexpected"]))
    bad = {p: c for p, c in out["codes"].items() if c >= 500}
    check(not bad, "every dashboard route answers (no 5xx)", json.dumps(bad))
    check(not _stack_ports(out["opened"]), "nothing reached a real stack port",
          json.dumps(_stack_ports(out["opened"])))
    ctx = (out["bodies"].get("/dash/api/vitals") or {}).get("context") or {}
    check("not loaded" in str(ctx.get("pool_read") or ctx.get("error") or ""),
          "the KV pool says it was not read because the model is not loaded",
          json.dumps(ctx)[:300])
    t = out["bodies"].get("/dash/api/tiers") or {}
    check(bool(t.get("tiers")), "the tier ladder still answers (the template's "
          "fallback efforts)", json.dumps(t)[:200])
    jj = out["bodies"].get("/dash/api/jjava") or {}
    check(all(k in jj for k in ("usage", "latency", "question_sets", "jev",
                                "lane", "priors", "injector")),
          "/dash/api/jjava answers every section", json.dumps(sorted(jj))[:300])
    pf = out["bodies"].get("/dash/api/perf") or {}
    check(all(k in pf for k in ("models", "gpus", "swaps", "gates", "left_out")),
          "/dash/api/perf answers every section", json.dumps(sorted(pf))[:300])
    check(out["codes"].get("/dash/api/jjava/2d") is None
          and CLIENT.get("/dash/api/jjava/2d").status_code == 404,
          "an unknown window is a 404, not a guess")


def test_running_unreadable():
    out = _run("unreadable")
    check(not out["upstream_all"], "llama-swap's /running unreadable: still "
          "ZERO /upstream requests (unreadable is not a licence to ask)",
          json.dumps(out["upstream_all"]))
    check(not out["unexpected"], "and no other llama-swap path",
          json.dumps(out["unexpected"]))
    bad = {p: c for p, c in out["codes"].items() if c >= 500}
    check(not bad, "every dashboard route answers (no 5xx)", json.dumps(bad))


def test_loaded_reads_go_direct():
    out = _run("loaded")
    check(not out["upstream"], "main model loaded: no /upstream request for a "
          "model /running does not list (only the loaded model's /props, the "
          "tier ladder's template read, may go through llama-swap)",
          json.dumps(out["upstream_all"]))
    check(all(p.endswith("/props") for _, p in out["upstream_all"]),
          "and nothing but /props goes through /upstream (the pool and the "
          "slots are read from the model's own port)",
          json.dumps(out["upstream_all"]))
    check(any(p.startswith("/props") for _, p in out["direct"]),
          "the positive control: the pool is re-read DIRECT from the main "
          "model's own server", json.dumps(out["direct"])[:300])
    ctx = (out["bodies"].get("/dash/api/vitals") or {}).get("context") or {}
    check(ctx.get("pool") == 147456, "and the KV panel shows the pool it read",
          json.dumps(ctx)[:300])
    bad = {p: c for p, c in out["codes"].items() if c >= 500}
    check(not bad, "every dashboard route answers (no 5xx)", json.dumps(bad))


def test_flash_next_holds_the_card():
    """FLASH-NEXT IS NOT BONSAI (2026-10-05, the operator: "Flash-Next is not showing anything when running in the
    dashboard"): with the max model on the card the dashboard reads ITS port, as /running reports it, shows its
    slots and its window, and still never asks /upstream."""
    import importlib
    import max_mode
    import tier_models
    real_env = os.environ.get("YAMADORI_TIER_MODELS")
    os.environ["YAMADORI_TIER_MODELS"] = os.path.join(HERE, "tier_models.yaml")
    importlib.reload(max_mode)
    try:
        _reset_caches()
        SWAP.reset("flash")
        DIRECT.reset("off")
        FLASH.reset("on")
        OPENED.clear()
        codes, bodies = {}, {}
        for path in ("/dash/api/vitals", "/dash/api/vitals/pulse", "/dash/api/tiers"):
            r = CLIENT.get(path)
            codes[path] = r.status_code
            bodies[path] = r.json() if r.status_code == 200 else None
        check(not SWAP.upstream_all(), "flash-next on the card: ZERO /upstream requests",
              json.dumps(SWAP.upstream_all()))
        check(not any(c >= 500 for c in codes.values()), "every route answers (no 5xx)", json.dumps(codes))
        pulse = bodies["/dash/api/vitals/pulse"] or {}
        sl = pulse.get("slots") or {}
        check(sl.get("ok") and sl.get("model") == "flash-next" and sl["slots"][0]["ctx"] == 26000
              and sl["slots"][0]["n_ctx"] == 262144,
              "/slots is read from flash-next's own server (not bonsai's port)", json.dumps(sl)[:300])
        check(any(p.startswith("/slots") for _, p in FLASH.seen) and not DIRECT.seen,
              "the positive control: flash-next's server was asked, bonsai's was not",
              json.dumps({"flash": FLASH.seen, "bonsai": DIRECT.seen}))
        check(not any("localhost" in u for u in OPENED if u.endswith("/slots")),
              "and as 127.0.0.1, never the IPv6-first `localhost` llama-swap reports", json.dumps(OPENED)[:300])
        cards = (pulse.get("cards") or {}).get("cards") or []
        # nvidia-smi is answered empty here: the cards are unplaced, the models are still all listed
        names = {m["model"]: m for c in cards for m in c.get("models", [])}
        check("flash-next" in names and names["flash-next"]["slots"]["ok"] and names["flash-next"]["card"] == "main"
              and names["embeddings"]["card"] == "a4000",
              "the lane view lists flash-next on the main card with its slots and the embedder on the A4000",
              json.dumps({k: (v["card"], v.get("state")) for k, v in names.items()}))
        ctx = pulse.get("context") or {}
        check(ctx.get("pool") == 262144 and ctx.get("model") == "flash-next",
              "the KV pool is flash-next's window (262,144), not bonsai's", json.dumps(ctx)[:200])
        sv = (bodies["/dash/api/vitals"] or {}).get("serving") or {}
        check(sv.get("on_card") == "flash-next", "serving.on_card is flash-next", json.dumps(sv)[:200])
    finally:
        if real_env is None:
            os.environ.pop("YAMADORI_TIER_MODELS", None)
        else:
            os.environ["YAMADORI_TIER_MODELS"] = real_env
        tier_models.reload()
        importlib.reload(max_mode)
        _reset_caches()


def _slot_readers() -> dict:
    """Every reader of a main model's /slots in the proxy's process (and the
    two bench runners), called once each: what it returned."""
    import decide_turn
    import decider_bonsai
    import idle
    import model
    import skill_questions_bank as SQB
    out = {"model.release_slot": model.release_slot(2),
           "skill_questions_bank.slots_busy": SQB.slots_busy(),
           "skill_questions_bank.why": SQB.SLOTS_WHY.get("why"),
           "decide_turn._idle": decide_turn._idle(),
           "decider_bonsai.slot_cells": decider_bonsai.slot_cells(2),
           "idle.slots_processing": idle.slots_processing()}
    # the bench runners, their llama-swap pointed at the fake
    bench = os.path.join(os.path.dirname(HERE), "bench")
    for sub in ("octopus", "decider"):
        p = os.path.join(bench, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import run as octo
        octo.SLOTS = f"{SWAP.url}/upstream/{MAIN}/slots"
        out["bench/octopus/run.slots_idle"] = octo.slots_idle()
    except Exception as e:                                       # noqa: BLE001
        out["bench/octopus/run.slots_idle"] = f"raised {type(e).__name__}: {e}"
    try:
        import bonsai_decider as BD
        BD.SLOTS = f"{SWAP.url}/upstream/{MAIN}/slots"
        BD.TARGET = None
        out["bench/decider/bonsai_decider.preflight"] = BD.preflight()
    except Exception as e:                                       # noqa: BLE001
        out["bench/decider/bonsai_decider.preflight"] = f"raised {type(e).__name__}: {e}"
    return out


def test_slot_readers_never_load():
    """The stray GET /upstream/bonsai/slots (2026-09-30 03:29:50): no reader
    of a model's /slots may ask for a model /running does not list ready.
    max_mode.blocks() alone was False when NO main model was loaded."""
    for mode in ("off", "unreadable"):
        _reset_caches()
        SWAP.reset(mode)
        DIRECT.reset("off")
        got = _slot_readers()
        check(not SWAP.upstream_all(), f"{mode}: the slot readers make ZERO "
              "/upstream requests", json.dumps(SWAP.upstream_all()))
        check(got["model.release_slot"].get("skipped")
              and not got["model.release_slot"].get("ok"),
              f"{mode}: model.release_slot skips and says why",
              json.dumps(got["model.release_slot"]))
        want = [] if mode == "off" else None
        check(got["skill_questions_bank.slots_busy"] == want
              and got["skill_questions_bank.why"],
              f"{mode}: slots_busy is {want} (nothing generates on an unloaded "
              "model / cannot tell) and records why",
              json.dumps({k: got[k] for k in ("skill_questions_bank.slots_busy",
                                              "skill_questions_bank.why")}))
        check(got["decide_turn._idle"] is False and got["decider_bonsai.slot_cells"] is None,
              f"{mode}: the decider's prime waits and slot_cells reads nothing",
              json.dumps({k: got[k] for k in ("decide_turn._idle", "decider_bonsai.slot_cells")}))
        oct_ = got["bench/octopus/run.slots_idle"]
        check(isinstance(oct_, tuple) and oct_[0] is (mode == "off")
              and "slots not read" in json.dumps(oct_),
              f"{mode}: bench/octopus/run.slots_idle skips the read and says why",
              json.dumps(oct_, default=str))
        bd = got["bench/decider/bonsai_decider.preflight"]
        if mode == "off":
            check(isinstance(bd, dict) and bd.get("slots_busy") == []
                  and "not loaded" in str(bd.get("slots_read")),
                  "off: bench/decider/bonsai_decider.preflight skips the read, says why",
                  json.dumps(bd, default=str))
        else:
            check(isinstance(bd, str) and "Busy" in bd,
                  "unreadable: bench/decider/bonsai_decider.preflight refuses (Busy)",
                  json.dumps(bd, default=str))
    _reset_caches()
    SWAP.reset("loaded")
    DIRECT.reset("loaded")
    got = _slot_readers()
    ups = SWAP.upstream_all()
    check(not SWAP.upstream() and ups and all(p.endswith("/slots") for _, p in ups),
          "loaded: the readers read /upstream/<model>/slots of the loaded model "
          "only", json.dumps(ups))
    check(got["skill_questions_bank.slots_busy"] == [] and got["decide_turn._idle"] is True
          and got["bench/octopus/run.slots_idle"][0] is True,
          "loaded: and read it", json.dumps({k: str(v)[:80] for k, v in got.items()}))


def main() -> int:
    for fn in (test_off_the_card, test_running_unreadable,
               test_loaded_reads_go_direct, test_slot_readers_never_load,
               test_flash_next_holds_the_card):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
