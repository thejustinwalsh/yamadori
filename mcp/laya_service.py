#!/usr/bin/env python
"""Laya decision engine: HTTP + WebSocket in one process.

Laya is a non-autoregressive "System 1" model (421M, ModernBERT-large) that
answers TYPED questions about a state in a single forward pass -- no token
generation. Measured here: 3 questions in 69 ms, ~23 ms/question.

    choice  pick one of N labelled options, with probabilities
    score   ordinal level against a rubric
    noul    boolean, returned as a probability

WHY IT IS HERE
--------------
An agent loop makes hundreds of small judgements -- "is this patch worth
benchmarking?", "did that output indicate a regression?", "is this retrieved
snippet actually relevant?". Asking a 27B to generate an answer for each costs
seconds and a slot. Laya answers in tens of milliseconds with a calibrated
probability, which is also a better signal than prose you then have to parse.

It is NOT a replacement for the chat model. It cannot write, explain or plan.
It classifies.

TRANSPORTS
----------
    POST /decide          one-shot HTTP, for scripts and CI
    WS   /ws              persistent socket, for hot loops
    GET  /health, /schema, /openapi.json

The WebSocket matters more than it looks: at 23 ms of compute, HTTP connection
setup and JSON headers are a large fraction of wall time. A held-open socket
removes that per-call overhead, which is the difference between "fast enough to
call in a loop" and "fast enough to call on every step".

Runs in its own venv (.venv-laya) because laya pulls a newer transformers than
the rest of the stack uses.
"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json
import os
import threading
import time

# Pin to the A4000 BEFORE torch is imported -- once CUDA initialises this has
# no effect. laya defaults to cuda:0, which is the 5060 Ti running the primary
# model at 208k context; adding ~1 GiB there took that card to 98% and it is
# the card that dies first under a long prefill. The A4000 has ~5 GiB spare.
# PCI_BUS_ID matches the ordering the rest of the stack assumes.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("LAYA_GPU", "1"))

import laya  # noqa: E402

import laya_head  # noqa: E402  -- same directory as this file

MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
HOST = os.environ.get("LAYA_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("LAYA_HTTP_PORT", "1237"))
# No WS_PORT. WebSocket is an HTTP Upgrade on the SAME connection; the second
# port only ever existed because ThreadingHTTPServer cannot perform the
# handshake, which forced a separate library onto its own socket.

_agent = None
_lock = threading.Lock()          # the model is not thread-safe; serialise it


def agent():
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                t0 = time.time()
                _agent = laya.load(MODEL)
                print(f"laya loaded in {time.time()-t0:.1f}s", flush=True)
    return _agent


SCHEMA = {
    "state": "string | object | list  -- the thing being judged",
    "questions": {
        "<id>": {
            "choice": {"type": "choice", "instructions": "...",
                       "criteria": {"option_a": "what it means",
                                    "option_b": "what it means"}},
            "score": {"type": "score", "instructions": "...",
                      "criteria": ["level 0", "level 1", "level 2"]},
            "noul": {"type": "noul", "instructions": "a statement to judge true/false"},
        }
    },
    "note": "criteria are REQUIRED for choice and score; optional for noul.",
}


def decide(payload: dict) -> dict:
    state = payload.get("state")
    questions = payload.get("questions")
    if state is None:
        raise ValueError("'state' is required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("'questions' must be a non-empty object")

    t0 = time.time()
    with _lock:                    # single GPU model, one caller at a time
        res = agent().predict(state, questions)
    ms = (time.time() - t0) * 1000
    return {"answers": res.get("answers", res),
            "elapsed_ms": round(ms, 1),
            "questions": len(questions)}


# ------------------------------------------------------- trained heads -----
#
# Zero-shot prompting of a 421M encoder decides our routing questions barely
# above chance and abstains on almost all of them -- measured on this repo's
# own labels, accuracy 0.546 at a 0.96 abstention rate. A linear head fitted on
# frozen features from the SAME forward pass reaches 0.746 at 0.03 abstention.
# So the service serves both and defaults to whichever the artefact's own
# recorded held-out numbers say is better. See docs/LAYA.md.

_heads: dict = {}
_heads_loaded = False
_HEAD_DIR = os.environ.get("LAYA_INDEX_DIR", laya_head.INDEX_DIR)


def heads() -> dict:
    """Load every artefact in the index dir once, tolerating bad ones.

    A head that fails to load must not take the service down: zero-shot still
    works, and /heads reports the error so it is visible rather than silent.
    """
    global _heads_loaded
    if not _heads_loaded:
        with _lock:
            if not _heads_loaded:
                for task in laya_head.TASKS:
                    try:
                        h = laya_head.TrainedHead.load_task(task, _HEAD_DIR)
                        if h is not None:
                            _heads[task] = h
                            print(f"laya head loaded: {task} "
                                  f"({h.kind}, gate {h.gate})", flush=True)
                    except Exception as e:                       # noqa: BLE001
                        _heads[task] = {"error": f"{type(e).__name__}: {e}"}
                        print(f"laya head FAILED: {task}: {e}", flush=True)
                _heads_loaded = True
    return _heads


def _head(task: str):
    h = heads().get(task)
    return None if h is None or isinstance(h, dict) else h


def prefers_trained(task: str) -> bool:
    """Default engine for a task, from the artefact's own recorded metrics.

    The artefact records what it scored on held-out data against the zero-shot
    baseline, so 'which is better' is answered by measurement rather than by
    whoever last edited this file.
    """
    h = _head(task)
    if h is None:
        return False
    m = h.metrics or {}
    try:
        before = m["held_out_before"]["accuracy"]["mean"]
        after = m["held_out_after"]["accuracy"]["mean"]
    except (KeyError, TypeError):
        return False
    return bool(after is not None and before is not None and after > before)


def beats_rule_baseline(task: str):
    """Does this head beat the six-line regex on the same held-out splits?

    For route_in it does NOT: the regex scores 0.841 against the head's 0.726.
    That comparison is recorded in the artefact by the trainer, and surfaced
    here so a caller choosing a router sees it. A head can be the best of the
    two engines THIS service offers and still be the wrong thing to route
    with -- those are different questions and conflating them is how a worse
    component ships because it was the only one measured.
    """
    h = _head(task)
    if h is None:
        return None
    rb = (h.metrics or {}).get("rule_baseline")
    after = ((h.metrics or {}).get("held_out_after") or {}).get("accuracy")
    if not rb or not after or rb.get("mean") is None:
        return None
    return bool(after["mean"] > rb["mean"])


def zero_shot_route(task: str, rec: dict) -> dict:
    """The untrained baseline, kept serveable so before/after stays checkable."""
    spec = laya_head.TASKS[task]
    state = laya_head.render_state(task, rec)
    out = decide({"state": state, "questions": {task: spec["question"]}})
    ans = out["answers"][task]
    probs = ans.get("probabilities") or {}
    order = sorted(probs.items(), key=lambda kv: -kv[1])
    margin = (order[0][1] - order[1][1]) if len(order) > 1 else 1.0
    gate = float(os.environ.get("LAYA_ZS_GATE", "0.3"))
    return {"type": "choice", "choice": ans.get("choice"),
            "probabilities": probs, "margin": round(margin, 6),
            "abstain": bool(margin < gate), "gate": gate,
            "engine": "zero_shot",
            "act_probability": (ans.get("action") or {}).get("act_probability"),
            "elapsed_ms": out["elapsed_ms"]}


def trained_route(task: str, rec: dict) -> dict:
    h = _head(task)
    if h is None:
        raise ValueError(
            f"no trained artefact for task {task!r} in {_HEAD_DIR}. "
            f"Run: .venv-laya/Scripts/python.exe scripts/train_laya.py "
            f"--task {task}"
        )
    t0 = time.time()
    with _lock:
        fx = laya_head.FeatureExtractor(agent())
        # Skip the contrast pass unless this head's features need it.
        need = h.feature_spec in laya_head.CONTRAST_SPECS
        f = fx.record(task, rec, contrast=need)
    out = h.decide(f)
    out["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return out


def route(payload: dict) -> dict:
    task = payload.get("task")
    if task not in laya_head.TASKS:
        raise ValueError(
            f"'task' must be one of {sorted(laya_head.TASKS)}, got {task!r}")
    engine = (payload.get("engine") or "auto").lower()
    if engine not in ("auto", "trained", "zero_shot", "both"):
        raise ValueError("'engine' must be auto, trained, zero_shot or both")

    rec = {kk: vv for kk, vv in payload.items()
           if kk not in ("task", "engine")}
    if engine == "auto":
        engine = "trained" if prefers_trained(task) else "zero_shot"

    if engine == "both":
        return {"task": task, "engine": "both",
                "zero_shot": zero_shot_route(task, rec),
                "trained": (trained_route(task, rec)
                            if _head(task) is not None else None)}
    res = (trained_route if engine == "trained" else zero_shot_route)(task, rec)
    res["task"] = task
    return res


# ---------------------------------------------------------------- HTTP ------

# ------------------------------------------------------- serving layer -----
#
# ONE PORT. WebSocket is an HTTP Upgrade on the same connection, so a second
# port was never needed -- it existed because ThreadingHTTPServer cannot
# perform the upgrade handshake, which forced a second library onto 1238.
# That is the same root cause as the proxy's: a stdlib demo server standing in
# for a real one. An ASGI app serves POST /decide and the /ws upgrade from the
# same socket, and 1238 is gone.

app = FastAPI(title="Laya decision engine", version="1.0.0",
              description="Typed decisions (choice/score/noul) in one forward "
                          "pass. A classifier, not a generator.")

# Bounded on purpose. An unbounded queue turns a client that outruns the GPU
# into a memory leak with no symptom until the process dies; a bounded one
# pushes back through TCP, which is what backpressure is for.
WS_QUEUE_DEPTH = int(os.environ.get("LAYA_WS_QUEUE", "64"))


@app.get("/health")
@app.head("/health")
async def health():
    return {"ok": True, "model": MODEL, "loaded": _agent is not None,
            "heads": sorted(t for t, h in heads().items()
                            if not isinstance(h, dict)),
            "ws": f"ws://<host>:{HTTP_PORT}/ws"}


@app.get("/schema")
async def schema():
    return SCHEMA


@app.get("/heads")
async def heads_route():
    """What is loaded, what it scored, and which engine /route defaults to."""
    out = {}
    for task in laya_head.TASKS:
        h = heads().get(task)
        if h is None:
            out[task] = {"loaded": False, "default_engine": "zero_shot"}
        elif isinstance(h, dict):
            out[task] = {"loaded": False, "error": h["error"],
                         "default_engine": "zero_shot"}
        else:
            out[task] = {
                "loaded": True, "kind": h.kind,
                "feature_spec": h.feature_spec, "gate": h.gate,
                "artefact_version": h.version, "prompt_rev": h.prompt_rev,
                "metrics": h.metrics,
                "default_engine": ("trained" if prefers_trained(task)
                                   else "zero_shot"),
                "beats_rule_baseline": beats_rule_baseline(task),
            }
            if beats_rule_baseline(task) is False:
                rb = h.metrics["rule_baseline"]["mean"]
                acc = h.metrics["held_out_after"]["accuracy"]["mean"]
                out[task]["note"] = (
                    f"this head ({acc:.3f}) loses to the regex baseline "
                    f"({rb:.3f}) on the same held-out splits. It is the better "
                    f"of the two engines here, but it is NOT the best router "
                    f"available -- see docs/LAYA.md."
                )
    return {"index_dir": os.path.abspath(_HEAD_DIR), "tasks": out}


@app.post("/route")
async def route_endpoint(payload: dict):
    """Serve a routing decision, trained or zero-shot, selectable per request.

        {"task": "route_in", "question": "...", "context": "...",
         "engine": "auto" | "trained" | "zero_shot" | "both"}

    'auto' picks by the artefact's own recorded held-out numbers, so a head
    that did not beat zero-shot never becomes the default by accident.
    """
    try:
        return await asyncio.to_thread(route, payload)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:                                       # noqa: BLE001
        return JSONResponse(status_code=500,
                            content={"error": f"{type(e).__name__}: {e}"})


@app.post("/decide")
async def decide_route(payload: dict):
    try:
        # decide() blocks on the GPU; keep the event loop free.
        return await asyncio.to_thread(decide, payload)
    except ValueError as e:
        return JSONResponse(status_code=400,
                            content={"error": str(e), "schema": SCHEMA})
    except Exception as e:                                       # noqa: BLE001
        return JSONResponse(status_code=500,
                            content={"error": f"{type(e).__name__}: {e}"})


@app.websocket("/ws")
async def ws_route(ws: WebSocket):
    """A sideband channel: upgrade once, then pipeline requests over it.

    THE SHAPE, AND WHY

    The first version read a frame, processed it to completion, and only then
    read the next. That is serial, not pipelined: the socket is not drained
    while the GPU works, so a client that sends five requests gets no overlap
    and TCP buffers fill behind a decision that takes milliseconds.

    So there are two tasks over one queue. The reader does nothing but parse
    and enqueue, so the socket is always being drained. A single worker pops
    and answers, which makes ordering FIFO by construction rather than by
    hope.

    ONE worker, deliberately. The model is a single GPU-resident instance and
    concurrent calls into it would serialise anyway, lower down and less
    visibly. An explicit queue makes the depth a number someone can see rather
    than an accident of how the runtime schedules threads.

    IDS ARE OPTIONAL, AND ORDER IS THE GUARANTEE

    One worker draining a FIFO queue means responses come back in request
    order, so a pipelining client can correlate positionally and needs no id
    at all. An id is echoed when supplied, for clients that would rather be
    explicit than count.

    That guarantee is load-bearing and narrow: it holds BECAUSE there is
    exactly one worker. Anything that adds a second -- batching, a worker
    pool, per-question parallelism -- breaks ordering, and at that point ids
    stop being a convenience and become mandatory. Whoever makes that change
    has to require them in the same commit.
    """
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=WS_QUEUE_DEPTH)
    done = object()

    async def reader():
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    req = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps(
                        {"id": None, "error": "invalid JSON"}))
                    continue
                if not isinstance(req, dict):
                    await ws.send_text(json.dumps(
                        {"id": None, "error": "frame must be a JSON object"}))
                    continue
                rid = req.get("id")
                # Blocks once the queue is full, which stops reading, which
                # stops the client. That is the intended backpressure path.
                await queue.put((rid, req))
        except WebSocketDisconnect:
            pass
        finally:
            await queue.put(done)

    async def worker():
        while True:
            item = await queue.get()
            if item is done:
                return
            rid, req = item
            try:
                out = await asyncio.to_thread(decide, req)
                if rid is not None:
                    out["id"] = rid
                await ws.send_text(json.dumps(out, default=str))
            except ValueError as e:
                await ws.send_text(json.dumps({"id": rid, "error": str(e)}))
            except Exception as e:                               # noqa: BLE001
                await ws.send_text(json.dumps(
                    {"id": rid, "error": f"{type(e).__name__}: {e}"}))
            finally:
                queue.task_done()

    r = asyncio.create_task(reader())
    w = asyncio.create_task(worker())
    try:
        await asyncio.gather(r, w)
    except Exception:                                            # noqa: BLE001
        # A dropped connection must not leave either task running against a
        # socket that is gone.
        for t in (r, w):
            if not t.done():
                t.cancel()


def main():
    import uvicorn
    agent()          # load before binding, so a health check means ready
    heads()          # and surface a broken artefact at startup, not mid-request
    print(f"laya on {HOST}:{HTTP_PORT}  POST /decide  POST /route  WS /ws  "
          f"(one port)", flush=True)
    # Ping/pong is handled by the protocol layer, BELOW this handler, so an
    # idle connection survives without the application sending anything. The
    # values are set explicitly because an implicit timeout is one nobody can
    # reason about until it fires at 3am.
    #
    # 20s ping with a 20s pong deadline: a dead peer is noticed within 40s,
    # and a live-but-slow one is not mistaken for dead, since pongs are
    # answered by the protocol even while the application is busy.
    uvicorn.run(app, host=HOST, port=HTTP_PORT, log_level="warning",
                ws_ping_interval=20, ws_ping_timeout=20,
                timeout_graceful_shutdown=30)


if __name__ == "__main__":
    main()
