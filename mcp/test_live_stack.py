#!/usr/bin/env python
"""The real stack, on the real card, judged on the model's actual output. OPT-IN.

    YAMADORI_TEST_KEY=ym-... python mcp/test_live_stack.py --live
    python mcp/test_live_stack.py --live --key-file PATH [--only tiers,tools]

WHY THIS EXISTS

Earlier sessions tested around the stack -- stubs, direct calls to llama-swap,
fixtures -- and then declared whole systems dead on the evidence of a failure
that was really a setting: a token budget smaller than the model's own
reasoning, a hop cap, a timeout. The cheapest way to stop that is to ask the
real thing, through the one door users use (the proxy on :1234), and to judge
the answer the model actually wrote.

WHAT IS ASSERTED

Contract, never wording:

    complete      a task with a checkable answer comes back WHOLE, at every
                  tier, even when the caller asks for a small max_tokens -- the
                  exact failure that produced empty replies (fdc9067, and
                  summarize_text at max_tokens=900 this session)
    streamed      the same, over SSE
    tools         a question only the index can answer makes the model call a
                  tool and name the right file
    summarize     the summarize_text tool returns a summary SHORTER than its
                  input, through the tools API on :1235
    seeds         a fan-out tier records the concept seed it injected
    hints         at `medium`, the range-sum prompt gets the prefix-sums hint
                  and never the Fenwick sibling (bucket collapse)

Every check prints the model's own words on failure, because "it failed" with
no output is how a setting gets mistaken for a dead system.

IT USES THE CARD. Run it alone. Two consumers on one GPU degrade each other
into 429s and 502s, and the loser looks like the one with the bug.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import traceback
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

PROXY = os.environ.get("YAMADORI_PROXY", "http://127.0.0.1:1234")
TOOLS = os.environ.get("YAMADORI_TOOLS", "http://127.0.0.1:1235")
TIMEOUT = 1800

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73, 79, 83, 89, 97]

_results: list[tuple[bool, str, str]] = []
KEY = ""


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def _post(url: str, body: dict, *, auth: bool = True,
          timeout: int = TIMEOUT,
          features: dict | None = None) -> tuple[int, dict | str, float]:
    headers = {"Content-Type": "application/json"}
    if auth and KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    if features:
        # Forces those systems on or off for this request (experiments only);
        # everything else stays the selection engine's decision.
        headers["X-Yamadori-Features"] = json.dumps(features)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        status = e.code
    try:
        return status, json.loads(raw), time.time() - t0
    except ValueError:
        return status, raw, time.time() - t0


def chat(messages: list[dict], features: dict | None = None,
         **kw) -> tuple[int, dict | str, float]:
    body = {"model": "yamadori", "messages": messages, "temperature": 0}
    body.update(kw)
    return _post(f"{PROXY}/v1/chat/completions", body, features=features)


def _x(d) -> dict:
    """The proxy's `x_yamadori` decision record, or {}."""
    return (d.get("x_yamadori") or {}) if isinstance(d, dict) else {}


def _stream(messages: list[dict], **kw) -> tuple[str, str, dict, float]:
    """(content, finish, x_yamadori from the final chunk, seconds)."""
    body = {"model": "yamadori", "stream": True, "temperature": 0,
            "messages": messages}
    body.update(kw)
    req = urllib.request.Request(
        f"{PROXY}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    parts, fin, x = [], "", {}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
                ch = ev["choices"][0]
            except (ValueError, KeyError, IndexError):
                continue
            parts.append((ch.get("delta") or {}).get("content") or "")
            if ch.get("finish_reason"):
                fin = ch["finish_reason"]
                x = ev.get("x_yamadori") or x
    return "".join(parts), fin, x, time.time() - t0


def _content(d) -> tuple[str, str]:
    if not isinstance(d, dict) or not d.get("choices"):
        return "", ""
    c = d["choices"][0]
    return (c.get("message") or {}).get("content") or "", c.get("finish_reason") or ""


def _primes_in(text: str) -> list[int]:
    import re
    return [int(x) for x in re.findall(r"\b\d+\b", text)]


# ---------------------------------------------------------------------------
def test_the_proxy_answers():
    try:
        with urllib.request.urlopen(f"{PROXY}/health", timeout=20) as r:
            ok = r.status == 200
    except Exception as e:                                       # noqa: BLE001
        ok = False
        check(False, "the proxy /health answers", str(e))
        return
    check(ok, "the proxy /health answers")
    status, d, _ = chat([{"role": "user", "content": "Reply with exactly: ok"}],
                        max_tokens=24)
    text, fin = _content(d)
    check(status == 200 and text.strip().lower().rstrip(".") == "ok",
          "a trivial request through the proxy returns the model's answer",
          f"HTTP {status} finish={fin} content={text!r} body={str(d)[:300]}")


def test_every_tier_returns_a_complete_answer_on_a_small_budget():
    """The empty-reply class of bug, asked of every tier at once.

    The caller asks for max_tokens=64 -- a common client default range -- for
    an answer that needs ~75 tokens after the model's own reasoning. A stack
    that passes the caller's number through, or floors it below the reasoning
    budget, returns nothing or half a list.
    """
    prompt = ("List the first 25 prime numbers in ascending order, one per "
              "line, digits only, nothing else.")
    took: dict[str, float] = {}
    for tier in ("minimal", "low", "medium", "high", "max"):
        status, d, dt = chat([{"role": "user", "content": prompt}],
                             max_tokens=64, reasoning_effort=tier)
        took[tier] = dt
        text, fin = _content(d)
        got = _primes_in(text)
        check(status == 200 and got[:25] == PRIMES and fin == "stop",
              f"tier {tier}: all 25 primes, finish=stop ({dt:.0f}s)",
              f"HTTP {status} finish={fin} got={got[:30]} "
              f"content={text[:200]!r}")
        check("unsettled" not in text and "agreed on the same file" not in text,
              f"tier {tier}: no fan-out dissent note on an answer naming no file",
              f"content={text[-300:]!r}")
        x = _x(d)
        sel = x.get("selection") or {}
        check(bool(x) and sel.get("investigate") is False
              and x.get("investigate") is None and sel.get("fanout_n") == 1,
              f"tier {tier}: x_yamadori says no deep thinking, no fan-out",
              json.dumps({"selection": {k: sel.get(k) for k in
                                        ("investigate", "fanout_n", "because")},
                          "investigate": x.get("investigate")})[:400])
    if "max" in took and "minimal" in took:
        check(took["max"] <= 1.5 * took["minimal"] + 5,
              f"max wall clock within 1.5x of minimal "
              f"({took['max']:.0f}s vs {took['minimal']:.0f}s)",
              json.dumps({k: round(v) for k, v in took.items()}))


def test_streaming_returns_the_whole_answer():
    body = {"model": "yamadori", "stream": True, "temperature": 0,
            "max_tokens": 64, "reasoning_effort": "medium",
            "messages": [{"role": "user", "content":
                          "List the first 25 prime numbers in ascending "
                          "order, one per line, digits only, nothing else."}]}
    req = urllib.request.Request(
        f"{PROXY}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    parts, fin, n = [], "", 0
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                n += 1
                try:
                    ch = json.loads(data)["choices"][0]
                except (ValueError, KeyError, IndexError):
                    continue
                parts.append((ch.get("delta") or {}).get("content") or "")
                fin = ch.get("finish_reason") or fin
    except Exception as e:                                       # noqa: BLE001
        check(False, "streaming completes", f"{type(e).__name__}: {e}")
        return
    text = "".join(parts)
    check(_primes_in(text)[:25] == PRIMES and fin == "stop",
          f"streamed: all 25 primes over {n} events, finish=stop",
          f"finish={fin} content={text[:200]!r}")


def _real_symbol() -> tuple[str, str] | None:
    """A class defined in exactly ONE file of the package index the proxy
    actually searches when no repository is bound.

    This used to read index/code.sqlite3 -- the server's OWN index, which a
    remote caller never reaches -- and it picked AnalyticLightNode, which the
    symbol table lists in 8 files (every subclass's `extends` clause is
    recorded as a definition). The model answered the real file and the test
    failed it: a ground-truth bug scored as a model failure.
    """
    try:
        import domains
        held = domains.held_sources()
        if "three" not in held:
            return None
        db = held["three"][0][1]
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = con.execute(
            "SELECT name, MIN(path) FROM defs WHERE kind='class' "
            "AND length(name) > 12 GROUP BY name HAVING COUNT(*) = 1 "
            "ORDER BY name LIMIT 1 OFFSET 40").fetchone()
        con.close()
        return (row[0], row[1]) if row else None
    except Exception:                                            # noqa: BLE001
        return None


def test_a_question_only_the_index_can_answer_uses_a_tool():
    sym = _real_symbol()
    if not check(sym is not None, "the code index has a class to ask about",
                 "index/code.sqlite3 unreadable or empty"):
        return
    name, path = sym
    status, d, dt = chat([{"role": "user", "content":
                           f"In the indexed codebase, which file defines the "
                           f"class `{name}`? Answer with the file path."}],
                         reasoning_effort="low")
    text, fin = _content(d)
    hops = (d.get("usage") or {}).get("hops") if isinstance(d, dict) else None
    check(status == 200 and hops is not None and hops > 1,
          f"the model called a tool ({hops} hops, {dt:.0f}s)",
          f"HTTP {status} hops={hops} content={text[:300]!r}")
    check(os.path.basename(path) in text,
          f"and named the defining file {os.path.basename(path)}",
          f"content={text[:300]!r}")


def test_summarize_text_through_the_tools_api():
    text = open(os.path.join(HERE, "jobs.py"), encoding="utf-8").read()[:6000]
    status, d, dt = _post(f"{TOOLS}/mcp", {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "summarize_text",
                   "arguments": {"text": text, "max_words": 300}}},
        auth=False)
    out = ""
    if isinstance(d, dict):
        for c in ((d.get("result") or {}).get("content") or []):
            out += c.get("text") or ""
    check(status == 200 and out and "MODEL_RETURNED_NOTHING" not in out
          and len(out) < len(text),
          f"summarize_text returns a summary shorter than its input ({dt:.0f}s)",
          f"HTTP {status} out={out[:300]!r}")
    check("STALE_SECONDS" in out or "reclaim" in out,
          "and keeps an identifier from the source verbatim", out[:300])


def test_a_fanout_tier_records_its_seed():
    import concept_seed
    before = (concept_seed.last() or {}).get("at", 0)
    status, d, dt = chat([{"role": "user", "content":
                           "Name one data structure for fast prefix lookups "
                           "on strings, in one word."}],
                         reasoning_effort="high", features={"fanout": 3})
    after = concept_seed.last() or {}
    fx = _x(d).get("fanout") or {}
    check(status == 200 and fx.get("n", 0) >= 2 and len(fx.get("seeds") or []) >= 2,
          "x_yamadori.fanout reports the variants and their seed words",
          json.dumps(fx)[:300])
    check(status == 200 and after.get("at", 0) > before
          and after.get("where") == "fanout",
          f"a high-tier request injected and recorded a seed ({dt:.0f}s)",
          f"HTTP {status} last={json.dumps(after)} "
          f"content={_content(d)[0][:160]!r}")


def test_hints_reach_the_model_collapsed():
    status, d, dt = chat([
        {"role": "system", "content":
         "Before answering, quote verbatim the first sentence of every "
         "engineering note appended to my message, each on its own line "
         "prefixed NOTE:."},
        {"role": "user", "content": "static array, many range-sum queries"}],
        reasoning_effort="medium")
    text, _ = _content(d)
    notes = "\n".join(line for line in text.splitlines()
                      if line.strip().upper().startswith("NOTE"))
    # Whether the model CHOOSES to quote its notes is its instruction-
    # following, not our delivery: it quoted them in one run and not the
    # next. Delivery is asserted deterministically below from x_yamadori; the
    # quote is printed for the record and never fails the suite.
    check(status == 200, f"the hints request completed ({dt:.0f}s)",
          f"HTTP {status}")
    print(f"  info  model quoted {'the prefix-sums note' if 'prefix' in notes.lower() else 'no note'}"
          f" (notes={notes[:160]!r})")
    check("Point updates interleaved with range sums" not in notes,
          "and its mutually exclusive Fenwick sibling did not", notes[:400])
    x = _x(d)
    shown = [h.get("recipe") or "" for h in x.get("hints") or []]
    check(any("prefix" in r.lower() for r in shown),
          "x_yamadori.hints: the prefix-sums hint, read off the response",
          json.dumps(shown)[:400])
    check(not any(r.startswith("Point updates interleaved") for r in shown),
          "x_yamadori.hints: the Fenwick sibling is not among them",
          json.dumps(shown)[:400])
    check(all(len(r) <= 120 for r in shown)
          and isinstance(x.get("suppressed_hints"), list),
          "recipes cut at 120 chars; suppressed siblings listed separately",
          json.dumps(x.get("suppressed_hints"))[:300])


def test_selection_decides_deep_thinking():
    """docs/SELECTION-BUILD.md steps 4 and 5, Live lines."""
    q = ("In three r185 TSL, the node method `label()` is deprecated. What "
         "replaces it, in which release was it deprecated, and which file "
         "emits the warning?")
    status, d, dt = chat([{"role": "user", "content": q}],
                         reasoning_effort="max")
    x = _x(d)
    sel = x.get("selection") or {}
    sig = sel.get("signals") or {}
    inv = x.get("investigate") or {}
    text, _fin = _content(d)
    check(status == 200 and sel.get("investigate") is True,
          f"a TSL deprecation question at max investigates ({dt:.0f}s)",
          json.dumps(sel.get("because"))[:400])
    check(sig.get("rule") is not None and sig.get("laya") is not None,
          "both signals recorded: the rule AND Laya's trained head",
          json.dumps({"rule": sig.get("rule"), "laya": sig.get("laya"),
                      "status": sig.get("laya_status")})[:400])
    check(inv.get("ran") and inv.get("hops", 0) > 0 and inv.get("injected")
          and inv.get("cited", 0) > 0,
          "the investigation searched, cited a retrieved path, and crossed",
          json.dumps(inv))
    check("nodes/" in text or "Node.js" in text,
          "the answer names a real three@0.185.1 source path",
          f"content={text[:400]!r}")

    p = ("List the first 25 prime numbers in ascending order, one per "
         "line, digits only, nothing else.")
    status, d, dt = chat([{"role": "user", "content": p}], max_tokens=64,
                         reasoning_effort="max")
    x = _x(d)
    check(status == 200 and (x.get("selection") or {}).get("investigate") is False
          and x.get("investigate") is None,
          f"the primes prompt at max does not investigate ({dt:.0f}s)",
          json.dumps((x.get("selection") or {}).get("because"))[:300])


def test_streamed_and_blocking_are_one_system():
    """docs/SELECTION-BUILD.md step 2, Live line: same prompt, same hops."""
    sym = _real_symbol()
    q = (f"In the indexed codebase, which file defines the class "
         f"`{sym[0]}`? Answer with the file path." if sym else
         "Which file in three.js defines Object3D? Answer with the path.")
    msgs = [{"role": "user", "content": q}]
    status, d, _dt = chat(msgs, reasoning_effort="low")
    xb = _x(d)
    try:
        _text, fin, xs, _ds = _stream(msgs, reasoning_effort="low")
    except Exception as e:                                       # noqa: BLE001
        check(False, "the streamed request completes", f"{type(e).__name__}: {e}")
        return
    check(bool(xs) and fin, "x_yamadori arrives on the streamed final chunk",
          f"finish={fin} keys={sorted(xs)}")
    check(set(xs) == set(xb), "streamed and blocking carry the same keys",
          str(sorted(set(xs) ^ set(xb))))
    check(xs.get("hops") == xb.get("hops"),
          f"the same prompt: {xb.get('hops')} hops blocking, "
          f"{xs.get('hops')} streamed",
          json.dumps({"blocking": xb.get("hops"), "streamed": xs.get("hops")}))


TESTS = {
    "health": test_the_proxy_answers,
    "tiers": test_every_tier_returns_a_complete_answer_on_a_small_budget,
    "stream": test_streaming_returns_the_whole_answer,
    "tools": test_a_question_only_the_index_can_answer_uses_a_tool,
    "summarize": test_summarize_text_through_the_tools_api,
    "seeds": test_a_fanout_tier_records_its_seed,
    "hints": test_hints_reach_the_model_collapsed,
    "selection": test_selection_decides_deep_thinking,
    "parity": test_streamed_and_blocking_are_one_system,
}


def main(argv: list[str]) -> int:
    global KEY
    if "--live" not in argv and os.environ.get("YAMADORI_LIVE_TESTS") != "1":
        print("not run: this suite uses the GPU. Pass --live.")
        return 0
    KEY = os.environ.get("YAMADORI_TEST_KEY", "")
    if "--key-file" in argv:
        KEY = open(argv[argv.index("--key-file") + 1]).read().strip()
    only = None
    if "--only" in argv:
        only = set(argv[argv.index("--only") + 1].split(","))
    t_all = time.time()
    for name, fn in TESTS.items():
        if only and name not in only:
            continue
        print(f"\n--- {fn.__name__} ---", flush=True)
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, nm, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + nm
                  + (f"\n        <- {detail}" if not ok and detail else ""),
                  flush=True)
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} live checks passed "
          f"in {time.time() - t_all:.0f}s")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
