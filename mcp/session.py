#!/usr/bin/env python
"""A held-open decision session: ask, read the answer, ask the next thing.

WHAT THIS IS AND IS NOT FOR

Not a latency optimisation. Measured, the cost of a decision does NOT scale
with the size of the state -- 410 characters takes 53.3ms and 4,920 takes
58.4ms -- so holding the state server-side saves nothing worth having.
Pipelining does not help either: frames are processed serially, so firing many
at once measured 0.9x, marginally worse than waiting for each.

The cost is roughly 35ms fixed per call plus 15-19ms per question. Eight
questions in one call cost 19.3ms each against 49.8ms for a single question
alone, so batching what you already know you want to ask is worth it.

What a session is actually for is the thing batching cannot do: choosing the
NEXT question based on the last answer. A flat batch asks everything at once
and learns nothing from itself. A session branches -- and at a 39ms round trip
a ten-step adaptive chain finishes in under half a second, which is fast
enough to sit inside a single tool call without anyone noticing.

That is the dialectic shape: hold several readings at once, follow the one
that disagrees.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

LAYA_WS = os.environ.get("LAYA_WS", "ws://127.0.0.1:1238/ws")


class Session:
    """One socket, many decisions, state sent per frame because it is free."""

    def __init__(self, state: str, url: str = LAYA_WS):
        self.state = state
        self.url = url
        self._ws = None
        self.asked: list[dict] = []

    async def __aenter__(self):
        import websockets
        self._ws = await websockets.connect(self.url, max_size=8 * 1024 * 1024)
        return self

    async def __aexit__(self, *exc):
        if self._ws is not None:
            await self._ws.close()

    async def ask(self, name: str, instructions: str,
                  options: dict[str, str]) -> dict:
        """One closed choice. Returns choice, margin and the full distribution.

        Always a choice over named options, never a score: the model's scores
        are not comparable between different inputs, and every use of it as a
        scalar in this repo failed.
        """
        t0 = time.time()
        await self._ws.send(json.dumps({
            "state": self.state,
            "questions": {name: {"type": "choice",
                                 "instructions": instructions,
                                 "criteria": options}}}))
        d = json.loads(await self._ws.recv())
        a = (d.get("answers") or {}).get(name) or {}
        probs = {k: round(float(v), 3)
                 for k, v in (a.get("probabilities") or {}).items()}
        ranked = sorted(probs.values(), reverse=True)
        margin = round(ranked[0] - ranked[1], 3) if len(ranked) > 1 else 0.0
        out = {"name": name, "choice": a.get("choice"), "margin": margin,
               "probabilities": probs, "ms": round((time.time() - t0) * 1000, 1),
               # Below this the model did not separate the options. Acting on
               # it would be inventing a decision it did not make.
               "decided": margin >= 0.15}
        self.asked.append(out)
        return out

    async def ask_many(self, questions: dict) -> dict:
        """Everything you already know you want to ask, in one frame.

        Use this for questions that do not depend on each other; use ask() when
        the next question depends on the last answer.
        """
        t0 = time.time()
        await self._ws.send(json.dumps({"state": self.state,
                                        "questions": questions}))
        d = json.loads(await self._ws.recv())
        el = round((time.time() - t0) * 1000, 1)
        out = {}
        for k, a in (d.get("answers") or {}).items():
            probs = {kk: round(float(vv), 3)
                     for kk, vv in (a.get("probabilities") or {}).items()}
            ranked = sorted(probs.values(), reverse=True)
            margin = round(ranked[0] - ranked[1], 3) if len(ranked) > 1 else 0.0
            out[k] = {"choice": a.get("choice"), "margin": margin,
                      "probabilities": probs, "decided": margin >= 0.15}
        out["_ms"] = el
        return out


async def triage(code: str) -> dict:
    """An adaptive chain: each question is chosen by the previous answer.

    This is what a session buys. A batch would ask all of these regardless and
    spend most of them on branches that do not apply.
    """
    async with Session(code) as s:
        trail = []
        kind = await s.ask(
            "kind", "What kind of code is this?",
            {"pure": "a pure computation with no side effects",
             "stateful": "it mutates something or holds state",
             "io": "it performs input or output"})
        trail.append(kind)

        if not kind["decided"]:
            return {"trail": trail, "total_ms": sum(x["ms"] for x in trail),
                    "conclusion": "could not classify the code"}

        if kind["choice"] == "pure":
            nxt = await s.ask(
                "purity_risk", "What is the risk in this computation?",
                {"none": "it looks sound",
                 "boundary": "an off-by-one or boundary error",
                 "precision": "a numeric precision problem"})
        elif kind["choice"] == "stateful":
            nxt = await s.ask(
                "mutation_risk", "What is the risk in this mutation?",
                {"none": "it looks sound",
                 "shared": "it mutates something shared or aliased",
                 "ordering": "it depends on the order of operations"})
        else:
            nxt = await s.ask(
                "io_risk", "What is the risk in this IO?",
                {"none": "it looks sound",
                 "unawaited": "an asynchronous call is not awaited",
                 "unhandled": "a failure path is not handled"})
        trail.append(nxt)

        conclusion = f"{kind['choice']}"
        if nxt["decided"] and nxt["choice"] != "none":
            conclusion += f", risk: {nxt['choice']}"
        elif nxt["decided"]:
            conclusion += ", no risk identified"
        else:
            conclusion += ", risk undetermined"
        total = sum(x["ms"] for x in trail)
        return {"trail": trail, "conclusion": conclusion, "total_ms": total}


if __name__ == "__main__":
    SAMPLES = {
        "unawaited io": "async function save(d){ writeFile('/tmp/x', d); }",
        "shared mutation": "function addAll(dst, src){ src.forEach(x => dst.push(x)); return dst; }",
        "boundary": "function last(xs){ return xs[xs.length]; }",
    }
    for label, code in SAMPLES.items():
        r = asyncio.run(triage(code))
        steps = " -> ".join(f"{x['name']}={x['choice']}({x['margin']})"
                            for x in r["trail"])
        print(f"  {label:<18} {steps}")
        print(f"  {'':<18} => {r['conclusion']}   [{r.get('total_ms')} ms]")
