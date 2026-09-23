#!/usr/bin/env python
"""Long-context characterisation of ONE endpoint: speed and accuracy vs L.

    # the real sweep: a dedicated llama-server, -np 1 (see README.md)
    python bench/longctx/run.py --url http://127.0.0.1:18080 --label A \\
        --run-id A-2026-09-23

    # through the live proxy (per-request cap = the main share, wall-clock speed)
    python bench/longctx/run.py --url http://127.0.0.1:1234 --key-file K \\
        --ladder 2k,8k,16k --items 3 --speed-reps 2 --run-id validate

    python bench/longctx/run.py --plan --url ...    # print the plan, send nothing

WHAT A ROW IS

One request. Rows append to bench/longctx/results/<run-id>/rows.jsonl, keyed
(L, task, item, seed); a rerun of the same run-id skips every key whose LAST
row is scored, and retries a key whose last row is a `stack_error` (an error
is not a completion: livecodebench.load_done). Outcomes:

    correct | wrong      scored
    budget               finish_reason `length` with no ANSWER line: a budget
                         event, never an answer (AGENTS.md); not scored
    stack_error          HTTP error, timeout, 429, unparseable reply, context
                         overflow. Never scored as wrong (PROTOCOL rule 3).
    measured             a speed row (task `speed_<content>`), no answer

SPEED -- two sources, never mixed, always labelled (`speed_source`):

  server_timings    llama-server's own `timings` object on the reply:
                    prompt_n / prompt_per_second (prefill), predicted_n /
                    predicted_per_second (decode), draft_n / draft_n_accepted
                    when speculative decoding (MTP) is on -- else the delta of
                    /metrics spec_decode_* counters around the request (needs
                    --metrics), else null. Null when speculation is off.
                    Each speed rep decodes three CONTENT TYPES (prose, code,
                    json: MTP acceptance depends on the text) from one
                    haystack. The first carries a fresh nonce at the head of
                    the prompt, so nothing is reused and its prefill is real
                    (`cold`: true); the other two reuse the cached haystack
                    and measure decode only. Fixed decode length
                    (--decode-tokens, ignore_eos), temperature 0.
  wallclock_stream  the proxy on :1234 drops `timings` (it re-assembles the
                    upstream stream, mcp/proxy.py `_post`). So via the proxy:
                    prefill = prompt_tokens / time-to-first-streamed-token,
                    decode = (completion_tokens - 1) / (last - first token).
                    It includes the proxy's own overhead, and decode length is
                    the model's (ignore_eos through a thinking budget of tens of
                    thousands of tokens would be a runaway). A smoke number,
                    not a model characterisation.

ACCURACY -- bench/longctx/haystack.py. Per (L, item) one haystack of real
package source with 4 needles + 4 near-miss decoys; three questions (single,
multi, reason) are asked of the SAME haystack with cache_prompt:true, so the
haystack is prefilled once per item, not three times. Items are paired across
L (same needles, same depths, same questions).

THINKING is explicit and recorded on every row (`thinking`):
  --thinking off         chat_template_kwargs.enable_thinking=false (default
                         against llama-server; greedy, measures raw recall)
  --thinking budget:N    thinking on, reasoning_budget_tokens=N, with the
                         production reasoning_budget_message
  --thinking server      whatever the server's own defaults are
  via the proxy          always `proxy:<features>`: the tier decides; the
                         default header turns every augmentation off at effort
                         low.

LENGTH. The target L is the whole prompt. The haystack size in characters comes
from a calibration done once per run (two requests; calibration.json), so a
resumed run rebuilds byte-identical prompts. The server's own count is recorded
(`prompt_tokens`, from usage) and is what analyse.py reports as L_actual. A
rung is run only if L * 1.03 + the request's generation allowance fits the
server's limit -- n_ctx against llama-server, the main share
(mcp/budget.py, 5/8 of the pool) minus the proxy's minimum answer+thinking
allowance against the proxy. The first rung that does not fit ends the ladder
and the manifest says why.

GPU MANNERS. Before sending anything this pauses the worker's gpu lane
(mcp/jobs.py; refreshed as it runs, released at the end, expires by itself if
this dies) and refuses to start if the lane is already paused by someone else,
if a known benchmark process is running, or if the card is busy (utilisation
sampled for 10 s). --wait-idle MIN waits instead of refusing. It never starts,
stops or loads anything.

The key is read from --key-file and sent only as a header. It is never
printed or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MCP = os.path.join(ROOT, "mcp")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import haystack as hs  # noqa: E402

# 240k is the largest rung that fits a -c 262144 server (L*1.03 + answer).
DEFAULT_LADDER = "2k,8k,16k,32k,64k,96k,128k,160k,192k,224k,240k,256k"
DEFAULT_FEATURES = {"retrieval": False, "hints": False, "investigate": False,
                    "fanout": 1, "effort": "low"}
# config.yaml's --reasoning-budget-message, so a thinking-on run closes the
# same way production does.
BUDGET_MESSAGE = ("Thinking budget reached. I will stop deliberating and "
                  "write the final answer now.")
FIT_MARGIN = 1.03             # calibration error allowance on L
PROXY_MIN_GEN = 2048 + 1024   # tiers.A_MIN + tiers.MIN_THINKING
ANSWER_TOKENS = 768           # thinking off: an answer, never a thought
CAL_CHARS = 48_000
MAX_CONSECUTIVE_ERRORS = 3    # PROTOCOL rule 2: the rate where a fallback is an outage
PAUSE_TTL = 1800
BUSY_PATTERNS = ("bench/domain/run.py", "bench\\domain\\run.py",
                 "livecodebench.py", "swe_bench", "swebench", "queue_runner.py",
                 "bench/longctx/run.py", "bench\\longctx\\run.py")
# Decode speed -- and above all MTP draft acceptance -- depends on WHAT is being
# generated, so each speed rep decodes three kinds of text from the same
# haystack. The first is sent with a fresh nonce (a cold prefill: the prefill
# measurement); the other two reuse the cached haystack and only decode.
SPEED_CONTENT = {
    "prose": ("Summarise, file by file, what the code above defines and how the "
              "pieces fit together. Plain prose."),
    "code": ("Write a new TypeScript module that uses the APIs defined in the "
             "code above: one class with three methods and full type "
             "annotations. Output only code."),
    "json": ("List every exported function in the code above as a JSON array of "
             'objects {"file": ..., "name": ..., "params": [...]}. Output only '
             "JSON."),
}
PROXY_SPEED_SUFFIX = " Keep the answer under 150 words."
METRIC_DRAFT = "llamacpp:spec_decode_num_draft_tokens_total"
METRIC_ACCEPT = "llamacpp:spec_decode_num_accepted_tokens_total"


# ------------------------------------------------------------- plumbing ----
def parse_ladder(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(int(float(tok[:-1]) * 1024) if tok.endswith("k") else int(tok))
    return sorted(set(out))


def row_key(L: int, task: str, item: int, seed: int) -> str:
    return f"{L}|{task}|{item}|{seed}"


def load_rows(path: str) -> tuple[list[dict], int]:
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def done_keys(rows: list[dict]) -> set[str]:
    """Keys whose LAST row is a completion. A stack_error is not one, so it is
    retried on resume; a budget row is (the model was asked and answered
    with a budget event, which is itself the measurement)."""
    last: dict[str, dict] = {}
    for r in rows:
        if r.get("key"):
            last[r["key"]] = r
    return {k for k, r in last.items() if r.get("outcome") != "stack_error"}


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------- gpu ---------
def nvsmi(idx: int, fields: str = "memory.used,memory.total,utilization.gpu,name"
          ) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
             "-i", str(idx)], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return [x.strip() for x in out.stdout.strip().split(",")]
    except (OSError, subprocess.SubprocessError):
        return None


def vram_used(idx: int) -> int | None:
    v = nvsmi(idx, "memory.used")
    try:
        return int(v[0]) if v else None
    except ValueError:
        return None


class VramPeak(threading.Thread):
    """Polls the card while one request runs and keeps the maximum."""

    def __init__(self, idx: int, every: float = 1.0):
        super().__init__(daemon=True)
        self.idx, self.every = idx, every
        self.peak: int | None = None
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = vram_used(self.idx)
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v
            self._stop.wait(self.every)

    def stop(self) -> int | None:
        self._stop.set()
        self.join(timeout=15)
        return self.peak


def gpu_busy(idx: int, seconds: float = 10.0) -> float | None:
    """Mean utilisation of the card over `seconds`, or None if unreadable."""
    xs = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        v = nvsmi(idx, "utilization.gpu")
        if v:
            try:
                xs.append(float(v[0]))
            except ValueError:
                pass
        time.sleep(1.0)
    return statistics.mean(xs) if xs else None


def busy_processes() -> list[str]:
    me = os.getpid()
    out = []
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            # Python interpreters only: a shell wrapper whose command line
            # merely mentions a benchmark is not one. Anything else on the
            # card (WSL, another agent's server) shows up in the
            # utilisation check instead.
            if p.info["pid"] == me or not (p.info.get("name") or "").lower().startswith("python"):
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            if any(b in cmd for b in BUSY_PATTERNS) and "test_" not in cmd:
                out.append(f"pid {p.info['pid']}: {cmd[:160]}")
    except Exception:                                            # noqa: BLE001
        pass
    return out


def server_cmdline(model_path: str | None, port: int | None) -> str | None:
    """The llama-server command line serving `model_path` (best effort)."""
    try:
        import psutil
    except ImportError:
        return None
    base = os.path.basename(model_path or "") or None
    hits = []
    for p in psutil.process_iter(["name", "cmdline"]):
        name = (p.info.get("name") or "").lower()
        if "llama-server" not in name:
            continue
        cmd = p.info.get("cmdline") or []
        s = " ".join(cmd)
        if port and (f"--port {port}" in s or f"--port={port}" in s):
            return s
        if base and base in s:
            hits.append(s)
    return hits[0] if len(hits) == 1 else None


# ------------------------------------------------------------- jobs lane ---
def _jobs():
    if MCP not in sys.path:
        sys.path.insert(0, MCP)
    import jobs
    return jobs


def other_pause(me: str) -> dict | None:
    try:
        rec = _jobs().paused("gpu")
    except Exception:                                            # noqa: BLE001
        return None
    if rec and not str(rec.get("by", "")).startswith(me):
        return rec
    return None


def pause_lane(me: str, why: str) -> None:
    try:
        _jobs().pause("gpu", by=me, why=why, ttl_seconds=PAUSE_TTL)
    except Exception as e:                                       # noqa: BLE001
        print(f"  WARNING: could not pause the gpu lane: {type(e).__name__}: {e}")


def resume_lane(me: str) -> None:
    try:
        rec = _jobs().paused("gpu")
        if rec is None or str(rec.get("by", "")).startswith(me):
            _jobs().resume("gpu")
    except Exception:                                            # noqa: BLE001
        pass


# ------------------------------------------------------------- client ------
class Client:
    def __init__(self, url: str, key: str | None, features: dict | None,
                 timeout: float = 1800.0):
        self.base = url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.key = key
        self.features = features
        self.timeout = timeout
        self.port = urllib.parse.urlparse(self.base).port

    def _headers(self, stream: bool = False) -> dict:
        h = {"Content-Type": "application/json", "Connection": "close"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        if self.features is not None:
            h["X-Yamadori-Features"] = json.dumps(self.features, sort_keys=True)
        if stream:
            h["Accept"] = "text/event-stream"
        return h

    def get_json(self, url: str, timeout: float = 15) -> dict | None:
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:                                        # noqa: BLE001
            return None

    def props(self) -> dict | None:
        d = self.get_json(f"{self.base}/props")
        if isinstance(d, dict) and ("default_generation_settings" in d or "n_ctx" in d):
            return d
        return None

    def draft_counters(self) -> tuple[float, float] | None:
        """(draft tokens, accepted tokens) from llama-server's Prometheus
        /metrics (needs --metrics), or None if not exposed."""
        try:
            req = urllib.request.Request(f"{self.base}/metrics", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:                                        # noqa: BLE001
            return None
        return parse_draft_metrics(text)

    def chat(self, body: dict, stream: bool) -> dict:
        """One request. Returns a dict with `ok`, and either the reply fields
        or `error_kind` / `error`. Never raises."""
        url = f"{self.base}/v1/chat/completions"
        body = dict(body, stream=stream)
        if stream:
            body["stream_options"] = {"include_usage": True}
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(stream))
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                if not stream:
                    raw = r.read().decode("utf-8", "replace")
                    t_end = time.time()
                    try:
                        d = json.loads(raw)
                    except json.JSONDecodeError:
                        return {"ok": False, "error_kind": "unparseable",
                                "error": raw[:300], "seconds": t_end - t0}
                    return _from_json(d, t0, t_end)
                return _from_stream(r, t0)
        except urllib.error.HTTPError as e:
            try:
                txt = e.read().decode("utf-8", "replace")[:600]
            except Exception:                                    # noqa: BLE001
                txt = ""
            kind = f"http_{e.code}"
            low = txt.lower()
            if e.code == 429:
                kind = "busy_429"
            elif ("context" in low and ("exceed" in low or "too long" in low
                                        or "size" in low)) or "n_ctx" in low:
                kind = "context_exceeded"
            return {"ok": False, "error_kind": kind, "error": txt,
                    "seconds": time.time() - t0}
        except Exception as e:                                   # noqa: BLE001
            kind = "timeout" if "timed out" in str(e).lower() else "transport"
            return {"ok": False, "error_kind": kind,
                    "error": f"{type(e).__name__}: {e}"[:300],
                    "seconds": time.time() - t0}


def _from_json(d: dict, t0: float, t_end: float) -> dict:
    ch = (d.get("choices") or [None])[0]
    if not isinstance(ch, dict):
        return {"ok": False, "error_kind": "no_choices",
                "error": json.dumps(d)[:300], "seconds": t_end - t0}
    msg = ch.get("message") or {}
    return {"ok": True, "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"),
            "usage": d.get("usage") or {}, "timings": d.get("timings"),
            "x_yamadori": d.get("x_yamadori"),
            "t_first": None, "seconds": t_end - t0}


def _from_stream(r, t0: float) -> dict:
    content, reasoning = [], []
    finish, usage, timings, xy = None, {}, None, None
    t_first = None
    n_delta = 0
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if d.get("error"):
            return {"ok": False, "error_kind": "stream_error",
                    "error": json.dumps(d["error"])[:300],
                    "seconds": time.time() - t0}
        if d.get("usage"):
            usage = d["usage"]
        if d.get("timings"):
            timings = d["timings"]
        if d.get("x_yamadori"):
            xy = d["x_yamadori"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            c = delta.get("content")
            rc = delta.get("reasoning_content")
            if c or rc:
                if t_first is None:
                    t_first = time.time()
                n_delta += 1
                if c:
                    content.append(c)
                if rc:
                    reasoning.append(rc)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.time()
    return {"ok": True, "content": "".join(content), "reasoning": "".join(reasoning),
            "finish_reason": finish, "usage": usage, "timings": timings,
            "x_yamadori": xy, "t_first": t_first, "t_last": t_end,
            "n_delta": n_delta, "seconds": t_end - t0, "t0": t0}


def parse_draft_metrics(text: str) -> tuple[float, float] | None:
    vals = {}
    for ln in text.splitlines():
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 2 and parts[0] in (METRIC_DRAFT, METRIC_ACCEPT):
            try:
                vals[parts[0]] = float(parts[1])
            except ValueError:
                pass
    if METRIC_DRAFT in vals and METRIC_ACCEPT in vals:
        return vals[METRIC_DRAFT], vals[METRIC_ACCEPT]
    return None


def draft_of(timings: dict | None, before, after) -> dict:
    """Draft acceptance for one request: the reply's own timings when it
    carries draft_n (per request, exact), else the /metrics counter delta
    around it (exact on a -np 1 server with nothing else running), else null.
    Null also when no tokens were drafted -- speculation off."""
    if isinstance(timings, dict) and "draft_n" in timings:
        dn, da = timings.get("draft_n") or 0, timings.get("draft_n_accepted") or 0
        src = "timings"
    elif before is not None and after is not None:
        dn, da = after[0] - before[0], after[1] - before[1]
        src = "metrics_delta"
    else:
        return {"draft_n": None, "draft_n_accepted": None,
                "draft_accept_rate": None, "draft_source": None}
    if not dn:
        return {"draft_n": None, "draft_n_accepted": None,
                "draft_accept_rate": None, "draft_source": None}
    return {"draft_n": int(dn), "draft_n_accepted": int(da),
            "draft_accept_rate": da / dn, "draft_source": src}


def speed_of(res: dict) -> dict:
    """Prefill/decode tok/s from the server's timings if present, else from
    the stream's wall clock -- labelled, never mixed."""
    t = res.get("timings")
    if isinstance(t, dict) and t.get("predicted_per_second") is not None:
        out = {"speed_source": "server_timings",
               "prefill_tps": t.get("prompt_per_second"),
               "decode_tps": t.get("predicted_per_second"),
               "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
               "predicted_n": t.get("predicted_n"),
               "prompt_ms": t.get("prompt_ms"), "predicted_ms": t.get("predicted_ms")}
        out.update(draft_of(t, res.get("metrics_before"), res.get("metrics_after")))
        return out
    u = res.get("usage") or {}
    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
    tf, tl, t0 = res.get("t_first"), res.get("t_last"), res.get("t0")
    out = {"speed_source": "wallclock_stream" if tf is not None else "none",
           "prefill_tps": None, "decode_tps": None,
           "prompt_n": pt, "cache_n": None, "predicted_n": ct}
    out.update(draft_of(None, res.get("metrics_before"), res.get("metrics_after")))
    if tf is not None and t0 is not None and pt:
        out["ttft_s"] = round(tf - t0, 3)
        out["prefill_tps"] = pt / max(tf - t0, 1e-6)
    if tf is not None and tl is not None and ct and ct > 1 and tl > tf:
        out["decode_tps"] = (ct - 1) / (tl - tf)
    return out


# ------------------------------------------------------------- bodies ------
def thinking_fields(thinking: str, answer_tokens: int) -> tuple[dict, int]:
    """(extra body fields, max_tokens) for a --thinking setting on a direct
    llama-server."""
    if thinking == "off":
        return {"chat_template_kwargs": {"enable_thinking": False}}, answer_tokens
    if thinking.startswith("budget:"):
        n = int(thinking.split(":", 1)[1])
        return ({"chat_template_kwargs": {"enable_thinking": True},
                 "reasoning_budget_tokens": n,
                 "reasoning_budget_message": "\n\n" + BUDGET_MESSAGE + "\n"},
                n + answer_tokens)
    if thinking == "server":
        return {}, 32768 + answer_tokens
    raise ValueError(f"--thinking {thinking!r}: off | budget:N | server")


def gen_allowance(cfg: dict, kind: str) -> int:
    if cfg["mode"] == "proxy":
        return PROXY_MIN_GEN
    if kind == "speed":
        return cfg["decode_tokens"] + 16
    return thinking_fields(cfg["thinking"], ANSWER_TOKENS)[1] + 16


def body_for(cfg: dict, msgs: list[dict], kind: str) -> dict:
    # cache_prompt is ON everywhere: a speed rep's first request carries a
    # fresh nonce at the head of the prompt, so nothing is reused and its
    # prefill is real; the rep's other content types then reuse that
    # haystack and measure decode only. Accuracy questions share their item's
    # haystack the same way.
    b = {"model": cfg["model"], "messages": msgs, "temperature": 0,
         "top_k": 1, "seed": 1234, "cache_prompt": True}
    if cfg["mode"] == "proxy":
        b["max_tokens"] = 1024 if kind != "speed" else 512
        return b
    if kind == "speed":
        extra, _ = thinking_fields(cfg["thinking"], 0)
        b.update(extra)
        b["max_tokens"] = cfg["decode_tokens"]
        b["ignore_eos"] = True
        return b
    extra, mt = thinking_fields(cfg["thinking"], ANSWER_TOKENS)
    b.update(extra)
    b["max_tokens"] = mt
    return b


def thinking_label(cfg: dict) -> str:
    if cfg["mode"] == "proxy":
        return "proxy:" + json.dumps(cfg["features"], sort_keys=True)
    return cfg["thinking"]


# ------------------------------------------------------------- calibrate ---
def calibrate(client: Client, cfg: dict, chunks) -> dict:
    """Prompt tokens of the wrapper alone (overhead) and chars/token of the
    filler, from the server's own count on two requests."""
    it = hs.make_item(cfg["seed"], 0)
    q = hs.question(it, "multi")
    out = {}
    n_cal = min(CAL_CHARS, int(cfg["limit"] * 1.5))
    for n in (0, n_cal):
        text = hs.filler(cfg["seed"], 0, n, chunks) if n else ""
        msgs = hs.messages(text, q, nonce=f"cal{n}-{time.time_ns()}")
        b = body_for(cfg, msgs, "accuracy")
        if cfg["mode"] != "proxy":
            b["max_tokens"] = 1
        b["cache_prompt"] = False
        res = client.chat(b, stream=cfg["mode"] == "proxy")
        if not res.get("ok"):
            raise RuntimeError(f"calibration request failed: {res.get('error_kind')}: "
                               f"{res.get('error')}")
        pt = (res.get("usage") or {}).get("prompt_tokens")
        if not pt:
            raise RuntimeError("calibration reply carried no usage.prompt_tokens")
        out[n] = (len(text), pt)
    (c0, t0), (c1, t1) = out[0], out[n_cal]
    return {"overhead_tokens": t0, "chars_per_token": (c1 - c0) / max(t1 - t0, 1),
            "method": "two requests, server usage.prompt_tokens",
            "at": time.time()}


def chars_for(L: int, cal: dict) -> int:
    return max(int((L - cal["overhead_tokens"]) * cal["chars_per_token"]), 1000)


# ------------------------------------------------------------- the run -----
def fits(L: int, cfg: dict, kind: str) -> bool:
    return L * FIT_MARGIN + gen_allowance(cfg, kind) <= cfg["limit"]


def plan(cfg: dict, done: set[str]) -> list[tuple[int, str, int]]:
    """Every (L, task, item) still to run, in execution order."""
    out = []
    for L in cfg["ladder"]:
        if not fits(L, cfg, "accuracy") or not fits(L, cfg, "speed"):
            break
        for rep in range(cfg["speed_reps"]):
            for c in cfg["speed_content"]:
                if row_key(L, f"speed_{c}", rep, cfg["seed"]) not in done:
                    out.append((L, f"speed_{c}", rep))
        for i in range(cfg["items"]):
            for task in cfg["tasks"]:
                if row_key(L, task, i, cfg["seed"]) not in done:
                    out.append((L, task, i))
    return out


def base_row(cfg: dict, L: int, task: str, item: int) -> dict:
    return {"key": row_key(L, task, item, cfg["seed"]), "run_id": cfg["run_id"],
            "label": cfg["label"], "model": cfg["model_id"], "build": cfg["build"],
            "mode": cfg["mode"], "thinking": thinking_label(cfg),
            "L": L, "task": task, "item": item, "seed": cfg["seed"],
            "t": time.time()}


def execute(client: Client, cfg: dict, L: int, task: str, item: int,
            chunks, cache: dict) -> dict:
    row = base_row(cfg, L, task, item)
    speed = task.startswith("speed_")
    if speed:
        content_type = task[len("speed_"):]
        sk = ("speed", L, item)
        cold = sk not in cache
        if cold:
            cache.clear()
            nonce = hashlib.sha1(f"{cfg['run_id']}|{L}|{item}|{time.time_ns()}"
                                 .encode()).hexdigest()[:16]
            cache[sk] = (nonce, hs.filler(cfg["seed"], 10_000 + item,
                                          chars_for(L, cfg["cal"]), chunks))
        nonce, text = cache[sk]
        q = SPEED_CONTENT[content_type] + (PROXY_SPEED_SUFFIX if cfg["mode"] == "proxy" else "")
        msgs = hs.messages(text, q, nonce)
        row.update(haystack_chars=len(text), content_type=content_type, cold=cold)
    else:
        row["content_type"] = f"answer:{task}"
        ck = (L, item)
        if ck not in cache:
            cache.clear()
            it = hs.make_item(cfg["seed"], item)
            cache[ck] = (it, hs.build_haystack(it, chars_for(L, cfg["cal"]), chunks))
        it, h = cache[ck]
        nonce = hashlib.sha1(f"{cfg['seed']}|{item}|{L}".encode()).hexdigest()[:16]
        msgs = hs.messages(h["text"], hs.question(it, task), nonce)
        row["haystack_chars"] = h["chars"]
        if task == "single":
            row["depths"] = [it["single"]["depth"]]
        elif task == "reason":
            row["depths"] = it["reason"]["depths"]
        else:
            row["depths"] = [n["depth"] for n in it["needles"]]
        row["landed"] = {k: v["actual"] for k, v in h["landed"].items()
                         if v["kind"] == "needle"}
    body = body_for(cfg, msgs, "speed" if speed else "accuracy")
    row["max_tokens_sent"] = body.get("max_tokens")
    idx = cfg["gpu_index"]
    row["vram_before_mib"] = vram_used(idx)
    peak = VramPeak(idx)
    peak.start()
    mb = client.draft_counters() if cfg.get("metrics") else None
    res = client.chat(body, stream=cfg["mode"] == "proxy")
    if mb is not None:
        res["metrics_before"], res["metrics_after"] = mb, client.draft_counters()
    row["vram_peak_mib"] = peak.stop()
    row["vram_after_mib"] = vram_used(idx)
    row["seconds"] = round(res.get("seconds") or 0.0, 3)
    if not res.get("ok"):
        row.update(outcome="stack_error", stack_error_kind=res.get("error_kind"),
                   error=res.get("error"))
        return row
    u = res.get("usage") or {}
    row["prompt_tokens"] = u.get("prompt_tokens")
    row["completion_tokens"] = u.get("completion_tokens")
    row["finish_reason"] = res.get("finish_reason")
    row.update(speed_of(res))
    if res.get("timings"):
        row["timings"] = res["timings"]
    if res.get("x_yamadori") is not None:
        xs = json.dumps(res["x_yamadori"])
        row["x_yamadori"] = res["x_yamadori"] if len(xs) < 4000 else xs[:4000]
    row["reasoning_chars"] = len(res.get("reasoning") or "")
    content = res.get("content") or ""
    row["content"] = content[-1500:]
    if speed:
        row["outcome"] = "measured"
        return row
    it, _h = cache[(L, item)]
    g = hs.grade(it, task, content, res.get("finish_reason"))
    row.update(outcome=g["outcome"], correct=g["correct"], parse=g["parse"],
               got=g["got"], expected=hs.expected(it, task))
    if "per_needle" in g:
        by_name = {n["name"]: n["depth"] for n in it["needles"]}
        row["per_needle"] = [{"depth": by_name[k], "ok": v}
                             for k, v in g["per_needle"].items()]
    return row


def run(client: Client, cfg: dict, run_dir: str, chunks, log=print,
        deadline: float | None = None, refresh=None) -> dict:
    """The loop. Returns a summary dict. `refresh` is called between rows
    (the lane pause); `deadline` is a wall-clock stop."""
    path = os.path.join(run_dir, "rows.jsonl")
    rows, _bad = load_rows(path)
    done = done_keys(rows)
    todo = plan(cfg, done)
    log(f"  {len(done)} rows done, {len(todo)} to run")
    cache: dict = {}
    consecutive = 0
    stopped = None
    blocked_L = None
    n = 0
    for L, task, item in todo:
        if blocked_L is not None and L >= blocked_L:
            continue
        if deadline and time.time() > deadline:
            stopped = "deadline"
            break
        if refresh:
            refresh()
        row = execute(client, cfg, L, task, item, chunks, cache)
        append_row(path, row)
        n += 1
        spd = ""
        if row.get("prefill_tps") is not None:
            spd = (f" prefill {row['prefill_tps']:.0f} t/s"
                   f" decode {row.get('decode_tps') or 0:.1f} t/s"
                   f" [{row.get('speed_source')}]")
        log(f"  L={L:>6} {task:<12} #{item:<3} -> {row['outcome']:<11}"
            f" pt={row.get('prompt_tokens')} {row['seconds']:.1f}s{spd}"
            + (f" ({row.get('stack_error_kind')})" if row["outcome"] == "stack_error" else ""))
        if row["outcome"] == "stack_error":
            consecutive += 1
            if row.get("stack_error_kind") == "context_exceeded":
                blocked_L = L
                log(f"  context exceeded at L={L}: no larger rung will run")
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                stopped = f"{consecutive} consecutive stack errors -- an outage, not data"
                break
        else:
            consecutive = 0
    return {"ran": n, "stopped": stopped, "blocked_L": blocked_L}


# ------------------------------------------------------------- main --------
def build_cfg(args, client: Client, props: dict | None) -> dict:
    mode = args.mode
    if mode == "auto":
        mode = "server" if props else "proxy"
    feats = None
    if mode == "proxy":
        feats = json.loads(args.features) if args.features else dict(DEFAULT_FEATURES)
    client.features = feats
    n_ctx = None
    if props:
        n_ctx = ((props.get("default_generation_settings") or {}).get("n_ctx")
                 or props.get("n_ctx"))
    limit, limit_why = None, None
    if args.max_ctx:
        limit, limit_why = args.max_ctx, "--max-ctx"
    elif mode == "server" and n_ctx:
        limit, limit_why = int(n_ctx), "server n_ctx (/props)"
    elif mode == "proxy":
        try:
            if MCP not in sys.path:
                sys.path.insert(0, MCP)
            import budget
            b = budget.budgets()
            limit = int(b["main"])
            limit_why = (f"proxy main share {b['main']} of pool {b['pool']} "
                         f"(mcp/budget.py)")
            n_ctx = n_ctx or b["pool"]
            # Identity of the model behind the proxy: a read-only GET of the
            # llama-server's /props, the same one mcp/budget.py reads. No key
            # is sent there.
            if not props:
                try:
                    with urllib.request.urlopen(f"{budget.DIRECT}/props", timeout=10) as r:
                        up = json.loads(r.read().decode("utf-8", "replace"))
                    props = {k: up.get(k) for k in ("model_path", "build_info",
                                                    "total_slots", "model_alias")}
                    props["_source"] = f"{budget.DIRECT}/props (behind the proxy)"
                except Exception:                                # noqa: BLE001
                    props = None
        except Exception as e:                                   # noqa: BLE001
            raise SystemExit(f"  cannot discover the proxy's per-request cap "
                             f"({e}); pass --max-ctx")
    if not limit:
        raise SystemExit("  no context limit: /props gave no n_ctx; pass --max-ctx")
    model_path = (props or {}).get("model_path")
    build = (props or {}).get("build_info")
    return {"mode": mode, "features": feats, "model": args.model,
            "model_path": model_path,
            "model_id": os.path.basename(model_path) if model_path else args.model,
            "build": build, "n_ctx": n_ctx, "limit": limit, "limit_why": limit_why,
            "total_slots": (props or {}).get("total_slots"),
            "thinking": args.thinking, "decode_tokens": args.decode_tokens,
            "ladder": parse_ladder(args.ladder), "items": args.items,
            "speed_reps": args.speed_reps, "seed": args.seed,
            "speed_content": [c for c in (args.speed_content
                                          or ",".join(SPEED_CONTENT)).split(",")
                              if c in SPEED_CONTENT],
            "metrics": bool(mode == "server" and client.draft_counters() is not None),
            "spec_types": ((props or {}).get("default_generation_settings") or {})
            .get("params", {}).get("speculative.types"),
            "tasks": [t for t in args.tasks.split(",") if t],
            "sources": args.sources.split(",") if args.sources else hs.DEFAULT_SOURCES,
            "gpu_index": args.gpu_index, "label": args.label,
            "run_id": args.run_id,
            "props": {k: (props or {}).get(k) for k in (
                "model_path", "build_info", "total_slots", "model_alias", "_source")}}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True,
                    help="llama-server base (http://host:port) or the proxy")
    ap.add_argument("--key-file", default=None, help="file holding an API key")
    ap.add_argument("--mode", default="auto", choices=("auto", "server", "proxy"),
                    help="auto: /props answers -> server, else proxy")
    ap.add_argument("--features", default=None,
                    help="X-Yamadori-Features JSON (proxy mode); default: all off, effort low")
    ap.add_argument("--model", default="yamadori", help="model field sent")
    ap.add_argument("--label", default=None,
                    help="config label for analysis, e.g. A, B-mtp-on (default: run id)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ladder", default=DEFAULT_LADDER)
    ap.add_argument("--max-ctx", type=int, default=None,
                    help="override the context limit the ladder stops at")
    ap.add_argument("--items", type=int, default=20, help="haystacks per length")
    ap.add_argument("--speed-reps", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--speed-content", default=None,
                    help="decode content types per speed rep, from "
                         + ",".join(SPEED_CONTENT) + " (default: all)")
    ap.add_argument("--thinking", default="off", help="off | budget:N | server")
    ap.add_argument("--tasks", default=",".join(hs.TASKS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sources", default=None, help="comma list of index/packages/_src dirs")
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="nvidia-smi index of the card serving the model")
    ap.add_argument("--flags", default=None,
                    help="the server's launch flags, recorded verbatim (else detected)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per request")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop after this long")
    ap.add_argument("--wait-idle", type=float, default=0.0,
                    help="minutes to wait for a busy card / another pause before giving up")
    ap.add_argument("--no-pause-lane", action="store_true",
                    help="do not pause the worker's gpu lane (a dedicated instance)")
    ap.add_argument("--busy-util", type=float, default=30.0,
                    help="mean %% utilisation over 10 s above which the card is busy")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    key = None
    if args.key_file:
        with open(args.key_file, encoding="utf-8") as f:
            key = f.read().strip() or None
        if not key:
            print("  the key file is empty")
            return 2
    args.run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
    args.label = args.label or args.run_id
    run_dir = os.path.join(RESULTS, args.run_id)
    client = Client(args.url, key, None, timeout=args.timeout)
    props = client.props() if args.mode in ("auto", "server") else None
    if args.mode == "server" and not props:
        print(f"  {args.url}/props did not answer like llama-server")
        return 2
    cfg = build_cfg(args, client, props)
    if cfg["mode"] == "proxy" and not key:
        print("  the proxy needs --key-file")
        return 2
    cfg["flags"] = args.flags or server_cmdline(cfg["model_path"], client.port)
    cfg["flags_source"] = "--flags" if args.flags else ("detected" if cfg["flags"] else None)
    gpu = nvsmi(args.gpu_index)
    cfg["gpu"] = {"index": args.gpu_index, "name": gpu[3] if gpu else None,
                  "total_mib": int(gpu[1]) if gpu else None}

    chunks = hs.load_chunks(cfg["sources"])
    print(f"  endpoint   {client.base}  mode={cfg['mode']}")
    print(f"  model      {cfg['model_id']}  build={cfg['build']}  n_ctx={cfg['n_ctx']}")
    print(f"  limit      {cfg['limit']}  ({cfg['limit_why']})")
    print(f"  thinking   {thinking_label(cfg)}")
    print(f"  sources    {len(chunks)} chunks, {hs.corpus_chars(chunks):,} chars")
    fit = [L for L in cfg["ladder"] if fits(L, cfg, "accuracy") and fits(L, cfg, "speed")]
    cut = [L for L in cfg["ladder"] if L not in fit]
    print(f"  ladder     {fit}" + (f"   (stops before {cut} at the limit)" if cut else ""))
    if not chunks:
        print("  no source chunks: index/packages/_src is missing")
        return 2
    if args.plan:
        rows, _ = load_rows(os.path.join(run_dir, "rows.jsonl"))
        todo = plan(cfg, done_keys(rows))
        print(f"  plan       {len(todo)} requests "
              f"({sum(1 for x in todo if x[1].startswith('speed_'))} speed)")
        return 0

    me = f"bench/longctx/run.py {args.run_id}"
    # --- is anyone else on the card? ---------------------------------------
    t_wait = time.time() + 60 * args.wait_idle
    while True:
        why = []
        rec = other_pause(me)
        if rec:
            why.append(f"gpu lane paused by {rec.get('by')!r} until "
                       f"{time.strftime('%H:%M:%S', time.localtime(rec.get('until', 0)))}")
        why += [f"benchmark process running: {p}" for p in busy_processes()]
        util = gpu_busy(args.gpu_index)
        if util is not None and util > args.busy_util:
            why.append(f"card {args.gpu_index} busy: {util:.0f}% mean utilisation over 10 s")
        if not why:
            break
        for w in why:
            print(f"  BUSY: {w}")
        if time.time() > t_wait:
            print("  not starting: one GPU consumer at a time (AGENTS.md). "
                  "Rerun later or pass --wait-idle MINUTES.")
            return 3
        time.sleep(60)

    os.makedirs(run_dir, exist_ok=True)
    man_path = os.path.join(run_dir, "manifest.json")
    man = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
        for k in ("model_path", "seed", "mode", "thinking"):
            if man.get(k) is not None and man.get(k) != cfg.get(k):
                print(f"  refusing to resume: manifest {k}={man.get(k)!r}, "
                      f"this run {cfg.get(k)!r}. Use a new --run-id.")
                return 2
    pause = not args.no_pause_lane
    if pause:
        pause_lane(me, "long-context benchmark measuring tokens/s")
    try:
        if man.get("calibration"):
            cfg["cal"] = man["calibration"]
            print(f"  calibration (reused) {cfg['cal']['chars_per_token']:.3f} chars/token, "
                  f"overhead {cfg['cal']['overhead_tokens']} tokens")
        else:
            cfg["cal"] = calibrate(client, cfg, chunks)
            print(f"  calibration {cfg['cal']['chars_per_token']:.3f} chars/token, "
                  f"overhead {cfg['cal']['overhead_tokens']} tokens")
        man.update({k: v for k, v in cfg.items() if k not in ("features",)},
                   features=cfg["features"], calibration=cfg["cal"],
                   url=client.base,
                   skipped_lengths=cut,
                   started=man.get("started") or time.time(), updated=time.time())
        man.pop("cal", None)
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)
        deadline = time.time() + 60 * args.max_minutes if args.max_minutes else None
        refresh = (lambda: pause_lane(me, "long-context benchmark measuring tokens/s")
                   ) if pause else None
        t0 = time.time()
        res = run(client, cfg, run_dir, chunks, deadline=deadline, refresh=refresh)
        man["updated"] = time.time()
        man.setdefault("sessions", []).append(
            {"start": t0, "end": time.time(), "ran": res["ran"],
             "stopped": res["stopped"], "blocked_L": res["blocked_L"]})
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)
        print(f"  ran {res['ran']} requests in {time.time() - t0:.0f}s"
              + (f"; STOPPED: {res['stopped']}" if res["stopped"] else ""))
        print(f"  rows: {os.path.relpath(os.path.join(run_dir, 'rows.jsonl'), ROOT)}")
        return 1 if res["stopped"] and "outage" in res["stopped"] else 0
    finally:
        if pause:
            resume_lane(me)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
