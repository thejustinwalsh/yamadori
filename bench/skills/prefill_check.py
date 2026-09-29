#!/usr/bin/env python
"""SETTLE THE PREFILL ASSUMPTION ON THE LIVE ENGINE (GPU, the model loaded):
after an assistant turn with reasoning_content and NO content, does the
served template leave the think block OPEN, and does the generation continue
the reasoning?

    python bench/skills/prefill_check.py --model bonsai

AGENTS.md "Heavy-handed, at the right time": "That llama-server leaves the
block open for a reasoning-only prefill is INFERRED from STEP 0's
/apply-template probe ... not measured: confirm on the live /apply-template."
The coordinator (2026-09-29): no prefill result counts until this is
settled, with the evidence recorded.

TWO CHECKS, each recorded with its evidence (never conversation text: the
probe is ours):

  1  TEMPLATE  POST /upstream/<model>/apply-template with [system, user,
               assistant {content: "", reasoning_content: LINE}]. Confirmed
               when the rendered prompt ENDS with LINE inside an open
               <think> block: the last "<think>" after the last user turn,
               LINE after it, and no "</think>" after LINE.
  2  GENERATION  POST /v1/chat/completions (through mcp/model.py, the one
               door) with the same messages, thinking on, max_tokens
               GEN_TOKENS. Confirmed when the response carries
               reasoning_content that is not empty (the model went on
               thinking) -- recorded with whether it re-sends LINE first
               (AGENTS.md: "llama-server re-sends a prefill first") and the
               finish reason.

OFFLINE, the served template ALONE closes the block: mcp/fixtures/
bonsai_chat_template.jinja (byte-identical to the GGUF's) rendered with
these messages ends "...That is how I'll write it\\n</think>\\n\\n<|im_end|>"
(2026-09-29). So an open block, if the live check finds one, is llama-
server's assistant-prefill handling (it renders the trailing assistant
turn and cuts after its prefix), not the template's -- which is why this
must be read on the live /apply-template.

GEN_TOKENS = 64: enough tokens to see whether the next ones are reasoning or
answer; the check reads their channel, not their quality.

Refuses unless --model is loaded (GET /running: nothing here loads a model;
/apply-template of an unloaded model WOULD load it). Writes
bench/skills/inject/results/prefill_check_<model>.json, which
render_probe.py reads before a prefill result may count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "inject", "results")
sys.path.insert(0, os.path.join(ROOT, "mcp"))

GEN_TOKENS = 64
LINE = ("While I write this: I'll keep each system to one concern; I won't "
        "hold entity references across frames. That is how I'll write it")
MESSAGES = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Write a small TypeScript function that "
     "moves every entity with a Position and a Velocity by dt."},
    {"role": "assistant", "content": "", "reasoning_content": LINE}]


def _post(url: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def running(upstream: str) -> list[str]:
    with urllib.request.urlopen(f"{upstream}/running", timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = d.get("running") if isinstance(d, dict) else d
    return [str((x or {}).get("model")) for x in rows or []
            if isinstance(x, dict)]


def template_verdict(prompt: str) -> dict:
    """Check 1 on a rendered prompt."""
    i = prompt.rfind(LINE)
    think = prompt.rfind("<think>", 0, i) if i >= 0 else -1
    closed = "</think>" in prompt[i:] if i >= 0 else None
    return {"line_found": i >= 0, "think_open_before_line": think >= 0,
            "closed_after_line": closed,
            "tail": prompt[-(len(LINE) + 60):],
            "confirmed": bool(i >= 0 and think >= 0 and closed is False)}


def generation_verdict(d: dict) -> dict:
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    rc = msg.get("reasoning_content") or ""
    return {"finish_reason": ch.get("finish_reason"),
            "reasoning_chars": len(rc), "content_chars": len(
                msg.get("content") or ""),
            "resent_line_first": rc.startswith(LINE),
            "reasoning_head": rc[:160],
            "confirmed": bool(rc.strip()) and rc.strip() != LINE}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    import model as M
    have = running(M.UPSTREAM)
    if a.model not in have:
        print(f"REFUSED: {a.model} is not loaded (GET /running: {have})")
        return 2
    t0 = time.time()
    out: dict = {"model": a.model, "at": t0, "gen_tokens": GEN_TOKENS}
    tpl = _post(f"{M.UPSTREAM}/upstream/{a.model}/apply-template",
                {"messages": MESSAGES})
    out["template"] = template_verdict(str(tpl.get("prompt") or ""))
    import cancel
    import max_mode
    with cancel.bound(cancel.Token()):
        max_mode.set_current(a.model)
        body = M.shape({"messages": MESSAGES, "max_tokens": GEN_TOKENS},
                       effort="medium")
        body.pop("_share", None)
        body["model"] = a.model
        body["max_tokens"] = GEN_TOKENS      # tiers.apply added A_MIN
        body.pop("reasoning_budget_tokens", None)
        d = M.post(body)
    out["generation"] = generation_verdict(d)
    out["confirmed"] = bool(out["template"]["confirmed"]
                            and out["generation"]["confirmed"])
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"prefill_check_{a.model}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out["confirmed"] else 1


if __name__ == "__main__":
    sys.exit(main())
