#!/usr/bin/env python
"""WHICH ENGINE A DECIDER BENCH READS: one model, by name or by base URL.

    import decider_target as DT
    t = DT.resolve(model="flash-next", base_url="http://127.0.0.1:18095")
    DT.install(t)              # every decider call of this process -> t
    DT.preflight(t)            # one GPU consumer, on THAT server

Three modes (2026-09-29; jjava per model, docs/JJAVA.md "Per model"):

  one door    no --base-url and the model is the stack's own main model
              (model.MODEL, `bonsai`; its alias `bonsai-agent`): NOTHING is
              patched -- the benches run exactly as before (mcp/model.py
              post(), the slots module's lane, llama-swap's /upstream/
              bonsai), and the preflight is bonsai_decider's.
  llama-swap  no --base-url, another model name: that model's llama-swap
              passthrough, <LLAMA_STACK_URL>/upstream/<model> -- chat at
              /v1/chat/completions, /tokenize, /slots under it. REFUSED
              (NotRun) unless llama-swap's GET /running already lists the
              model: /running never loads anything, and any /upstream/<m>
              request WOULD load it (and swap the main card). The engine
              owner loads it in their GPU window, or uses a bare server.
  base-url    --base-url http://127.0.0.1:18095 (a bare llama-server on a
              test port): chat, /tokenize, /slots and /props at that base.
              The name given with --model is the profile's key (what the
              record is filed under) -- the server's own model is recorded
              beside it (/props model_path), never trusted as the name.

In the last two modes install() points the decider's two doors at the
target (decider_bonsai._upstream and _post_default, mcp/model.py post and
MODEL -- so anything that falls back to the one door reaches the target,
never llama-swap's `bonsai`), makes the slots module hand every decider
batch ONE slot of the target (--slot, else the highest id its /slots lists,
else the server's choice) and never release it (the bench's own server:
nothing to give back), and names the model (model.MODEL) so every record,
cache key and profile lookup is the target's. Nothing here reaches a port
until a bench calls it; the offline suites drive it with fakes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

MODES = ("one door", "llama-swap", "base-url")


class Busy(RuntimeError):
    """Another GPU consumer is at work: the run stops (NOT RUN, exit 2)."""


class NotRun(RuntimeError):
    """The target cannot be read without breaking a rule (exit 3)."""


def _http(url: str, payload: dict | None = None, timeout: float = 30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "null")


class Target:
    """One engine the decider reads: post(body, timeout) -> the chat answer;
    upstream(path, payload=None, timeout=30) -> a model-server endpoint."""

    def __init__(self, model: str, mode: str, base: str,
                 slot: int | None = None, http=_http):
        assert mode in MODES, mode
        self.model, self.mode, self.base, self.slot = model, mode, base, slot
        self._http = http

    def post(self, b: dict, timeout: float = 120) -> dict:
        send = {k: v for k, v in b.items() if not k.startswith("_")}
        return self._http(f"{self.base}/v1/chat/completions", send,
                          timeout=timeout)

    def upstream(self, path: str, payload: dict | None = None,
                 timeout: float = 30):
        return self._http(f"{self.base}{path}", payload, timeout=timeout)

    def describe(self) -> dict:
        """What the result records about the engine (no secrets: a local
        URL, the server's model file name and build)."""
        out = {"model": self.model, "mode": self.mode, "base": self.base,
               "slot": self.slot}
        if self.mode == "one door":
            return out
        try:
            p = self.upstream("/props", timeout=10) or {}
            out["model_path"] = os.path.basename(str(p.get("model_path")
                                                     or "")) or None
            out["build"] = p.get("build_info")
            out["total_slots"] = p.get("total_slots")
            out["n_ctx"] = ((p.get("default_generation_settings") or {})
                            .get("n_ctx"))
        except Exception as e:                                   # noqa: BLE001
            out["props"] = f"unreadable: {type(e).__name__}"
        return out


def _default_model() -> str:
    import model
    return model.MODEL


def resolve(model: str | None = None, base_url: str | None = None,
            slot: int | None = None, *, running=None, http=_http) -> Target:
    """The Target for --model / --base-url / --slot. `running()` -> the
    models llama-swap has loaded (default max_mode.running(): GET /running,
    which never loads). Raises NotRun for a llama-swap model that is not
    loaded, ValueError for a malformed base URL."""
    import decider_bonsai as D
    default = _default_model()
    name = model or default
    if base_url is None and D.canonical_model(name) == \
            D.canonical_model(default):
        import model as M
        return Target(name, "one door", f"{M.UPSTREAM}/upstream/{default}",
                      None, http)
    if base_url is None:
        import model as M
        if running is None:
            import max_mode
            running = max_mode.running
        if name not in (running() or set()):
            raise NotRun(
                f"{name} is not loaded in llama-swap (GET /running); "
                f"reading /upstream/{name} would load it and swap the main "
                "card. The engine owner loads it in the GPU window, or "
                "passes --base-url for a bare llama-server")
        t = Target(name, "llama-swap", f"{M.UPSTREAM}/upstream/{name}",
                   slot, http)
    else:
        b = base_url.strip().rstrip("/")
        if b.endswith("/v1"):
            b = b[:-3]
        if not b.startswith(("http://", "https://")):
            raise ValueError(f"--base-url must be http(s)://host:port, got "
                             f"{base_url!r}")
        t = Target(name, "base-url", b, slot, http)
    if t.slot is None:
        try:
            table = t.upstream("/slots", timeout=10)
            ids = [s.get("id") for s in table or []
                   if isinstance(s, dict) and isinstance(s.get("id"), int)]
            t.slot = max(ids) if ids else None
        except Exception:                                        # noqa: BLE001
            t.slot = None
    return t


def install(t: Target, *, decisions_dir: str | None = None) -> dict:
    """Point this process's decider at `t` (a no-op in the one-door mode).
    `decisions_dir`: where decide_turn's decision and disagreement logs go
    for this run (<model>.decisions.jsonl, <model>.disagreements.jsonl),
    instead of the live logs. Returns what was patched, and `restore` (a
    callable that puts it back: the offline suites)."""
    import decide_turn as T
    import decider_bonsai as D
    import model as M
    import slots
    saved = {(T, "DECISIONS"): T.DECISIONS, (T, "LOG"): T.LOG}
    if t.mode != "one door":
        saved.update({(D, "_upstream"): D._upstream,
                      (D, "_post_default"): D._post_default,
                      (M, "post"): M.post,
                      (M, "release_slot"): M.release_slot,
                      (M, "MODEL"): M.MODEL,
                      (slots, "acquire"): slots.acquire,
                      (slots, "lane_kept"): slots.lane_kept})

    def restore():
        for (mod, name), v in saved.items():
            setattr(mod, name, v)
        D._SPELL_IDS.clear()
        T.TEMPLATE_CF.clear()
    done: dict = {"mode": t.mode, "model": t.model, "restore": restore}
    if decisions_dir:
        os.makedirs(decisions_dir, exist_ok=True)
        T.DECISIONS = os.path.join(decisions_dir,
                                   f"{t.model}.decisions.jsonl")
        T.LOG = os.path.join(decisions_dir, f"{t.model}.disagreements.jsonl")
        done["decisions"] = T.DECISIONS
    if t.mode == "one door":
        return done
    D._SPELL_IDS.clear()
    T.TEMPLATE_CF.clear()
    M.MODEL = t.model
    D._upstream = t.upstream
    D._post_default = lambda b, timeout: t.post(b, timeout)
    M.post = lambda body, timeout=M.TIMEOUT: t.post(body, timeout)
    M.release_slot = lambda slot, model=None, timeout=None: {
        "ok": False, "method": None,
        "skipped": "bench target: its own server's slot is left as is"}

    def acquire(key=None, transient=False, **kw):
        return {"slot": t.slot, "mode": "bench", "how":
                f"bench target {t.mode} {t.model}", "evicted": None}
    slots.acquire = acquire
    slots.lane_kept = lambda: True
    done.update(patched=["decider_bonsai._upstream",
                         "decider_bonsai._post_default", "model.post",
                         "model.release_slot", "model.MODEL",
                         "slots.acquire", "slots.lane_kept"],
                slot=t.slot)
    return done


def preflight(t: Target, mine: tuple = (), *, tasklist=None) -> dict:
    """ONE GPU CONSUMER on the target: no slot of ITS server processing
    (other than `mine`) and no hermes.exe; else Busy. The one-door mode is
    bonsai_decider.preflight, unchanged."""
    if t.mode == "one door":
        from bonsai_decider import preflight as pf
        return pf(mine)
    out: dict = {"at": time.time(), "target": t.mode}
    try:
        table = t.upstream("/slots", timeout=10)
        busy = [s.get("id") for s in table or [] if isinstance(s, dict)
                and s.get("is_processing") and s.get("id") not in mine]
        out["slots_busy"] = busy
    except urllib.error.URLError as e:
        raise NotRun(f"the target {t.base} does not answer: {e}") from e
    except Exception as e:                                       # noqa: BLE001
        busy = []
        out["slots_busy"] = f"unreadable ({type(e).__name__})"
    tl = (tasklist or (lambda: subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq hermes.exe"],
        capture_output=True, text=True).stdout))()
    hermes = "hermes.exe" in (tl or "").lower()
    out["hermes"] = hermes
    if busy or hermes:
        raise Busy(f"{t.model} slots processing {busy}; hermes.exe running: "
                   f"{hermes}")
    return out


def add_args(ap) -> None:
    """--model / --base-url / --slot, the same on every decider bench."""
    ap.add_argument("--model", help="the model to measure (its profile's "
                    "name; default the stack's main model, "
                    "YAMADORI_MODEL / bonsai)")
    ap.add_argument("--base-url", help="a bare llama-server's base, e.g. "
                    "http://127.0.0.1:18095 (default: llama-swap's "
                    "/upstream/<model>, which must already be loaded)")
    ap.add_argument("--slot", type=int, help="the target server's slot for "
                    "the decider's reads (default: its highest slot id)")


def from_args(a) -> Target:
    return resolve(model=a.model, base_url=a.base_url, slot=a.slot)
