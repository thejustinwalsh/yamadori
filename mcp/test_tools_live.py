#!/usr/bin/env python
"""The two tools that need the card, tested against the real one. OPT-IN.

WHY THIS IS A SEPARATE FILE

`test_tools.py` is the gate: it runs in about two seconds, needs no GPU, and
therefore runs every time. The moment a suite needs a model it acquires a
reason not to be run -- the card is busy, the stack is restarting, somebody is
benchmarking -- and a gate that is skipped is not a gate. So the tools that
call the model live here instead, behind an explicit flag.

    python mcp/test_tools_live.py --live
    YAMADORI_LIVE_TESTS=1 python mcp/test_tools_live.py

Without one of those it prints why it did nothing and exits 0. That is
deliberate: this file is not a gate and must never fail a pipeline because a
GPU was unavailable.

BEFORE YOU RUN IT

This stack holds ONE KV pool and ONE helper lane. `delegate_investigation`
starts a second context on the same card. Running this while a benchmark or
another agent is using the stack does not produce two results more slowly -- it
produces two degraded ones and a 429, and the arm that happens to lose most
looks like the arm with a bug. Check that the card is yours first.

WHAT IS ASSERTED, AND WHAT IS NOT

Shape and contract only. Never wording, never a specific sentence, never a
ranking. A generated summary is not reproducible, so a test that pins its
prose is a test that fails on a model upgrade and teaches nobody anything. The
questions asked here are the ones with stable answers:

    did it run                     a result, not an empty string
    did it do the job              a summary is SHORTER than its input
    did it keep what it promised   identifiers, paths, numbers and error
                                   strings survive verbatim, because that is
                                   what the tool's own description guarantees
    did it cost what it claims     the investigation reports hops, tokens and
                                   seconds, and the receipts exist
    did it keep its searching out  the whole point of a second context is that
      of this conversation         its tool traffic does not arrive here

The fixture index and the repository under it are inherited from
`test_tools.py`, so these run against three known files and never against
index/code.sqlite3.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Importing the gate suite builds the fixture, points CODE_INDEX_DB, RINGS_DB
# and YAMADORI_CORPUS_DB at temporary files, and hands us its check harness.
# Nothing in that import touches the model.
import test_tools as T  # noqa: E402

import code_search as cs  # noqa: E402
import shomen  # noqa: E402

check, indexed, as_json = T.check, T.indexed, T.as_json


def enabled(argv: list[str]) -> bool:
    return "--live" in argv or os.environ.get("YAMADORI_LIVE_TESTS") == "1"


# The text handed to summarize_text. Every item in KEEP appears in it exactly
# once and is the kind of token the tool's description promises to preserve:
# an identifier, a file path with a line number, a measurement, and an error
# string. If compression drops these it has thrown away the only parts that
# could not be reconstructed from memory.
#
# The error string is ENOSPC and not, say, OperationalError, because the leak
# detector in test_tools.py treats Python exception names as contract-2
# violations. A summary that correctly preserved "OperationalError" because it
# was asked to would be indistinguishable from one that leaked it -- the first
# dry run of this file failed exactly that way. Picking a non-Python error
# string keeps both checks meaningful instead of making one of them lie.
KEEP = ("sizeKvPool", "src/pool.ts:1-4", "147456", "ENOSPC")

LONG_TEXT = """
Notes from a debugging session, written badly on purpose.

So I was looking at the KV pool sizing and honestly I went round in circles for
a while, you know how it is. Anyway. The function is sizeKvPool and it lives in
src/pool.ts:1-4, which took me far too long to find because I kept grepping for
the wrong thing and then getting distracted.

The measured pool is 147456 tokens, not the 131072 per slot that the server
reports, because four slots reporting 131072 each would be about 22 GB of KV
against a measured 5.8 GB, which is obviously not what is happening.

Then the whole thing fell over with ENOSPC while writing the index and I spent
another twenty minutes on that before realising the disk had filled. Not the
code's fault, or well, sort of the code's fault.

Anyway that is where I got to. There is more but it is mostly me complaining
about the reranker, which as far as I can tell does nothing at the shipped
cutoff and costs about a second a query. I will write that up separately.
""" * 3


def test_summarize_text_live():
    out = indexed("summarize_text",
                  {"text": LONG_TEXT, "focus": "the KV pool numbers",
                   "max_words": 120})

    if not check(bool((out or "").strip()), "summarize_text returns something",
                 repr(out)[:80]):
        return
    check(T.leaked_exception(out) == [],
          "summarize_text leaks no raw exception", out[:160])

    d = as_json(out)
    if d is not None and d.get("ok") is False:
        # A structured failure is a legitimate outcome -- the stack may be
        # down -- but it is not a summary, so nothing below applies. It must
        # still obey the contract.
        check(d.get("error") in ("MODEL_UNAVAILABLE", "MODEL_RETURNED_NOTHING"),
              "a live failure names a known situation", str(d.get("error")))
        check(d.get("retryable") is True,
              "a model that could not be reached is retryable",
              str(d.get("retryable")))
        check(bool(d.get("remedies")), "a live failure offers a remedy")
        check(False, "THE MODEL DID NOT ANSWER -- the checks below did not run",
              json.dumps(d)[:200])
        return

    check(d is None, "a successful summary is prose, not an envelope", out[:80])

    # COMPRESSION IS THE JOB. A "summary" at or above the size of its input has
    # done nothing, and the caller has paid a model round trip plus the tokens
    # to read it back. This is the one quantitative claim the tool makes.
    check(len(out) < len(LONG_TEXT),
          "the summary is shorter than its input",
          f"{len(out)} chars from {len(LONG_TEXT)}")
    check(len(out) < len(LONG_TEXT) * 0.6,
          "the summary is MEANINGFULLY shorter, not trimmed",
          f"{len(out)} chars from {len(LONG_TEXT)}")

    # The length target is a target, not a limit -- the prompt says "at most N
    # words" and models overshoot. A generous ceiling still catches the real
    # failure, which is the tool ignoring max_words entirely and echoing the
    # input back.
    words = len(out.split())
    check(words <= 120 * 3,
          "max_words is treated as a target rather than ignored",
          f"{words} words against a target of 120")

    # WHAT THE DESCRIPTION PROMISES: "keeping identifiers, numbers, file paths
    # and errors verbatim". These four tokens cannot be recovered from a
    # paraphrase, so losing them silently destroys the only information the
    # summary was holding.
    for token in KEEP:
        check(token in out, f"summarize_text preserves {token!r} verbatim",
              out[:400])

    # Pleasantries and repetition are what it is told to drop, and the input
    # is three identical copies. A summary repeating the body three times has
    # compressed nothing.
    check(out.count("sizeKvPool") < LONG_TEXT.count("sizeKvPool"),
          "the threefold repetition in the input is not reproduced",
          f"{out.count('sizeKvPool')} against {LONG_TEXT.count('sizeKvPool')}")


def test_summarize_text_live_handles_a_trivial_input():
    """Short text is not an error, and must not come back empty.

    A model given almost nothing to compress can return almost nothing. That
    is contract 1 arriving from the other side of the HTTP call, which is why
    the empty-completion branch exists at all.
    """
    out = indexed("summarize_text", {"text": "The build passed.",
                                     "max_words": 50})
    check(bool((out or "").strip()), "a trivial input still gets an answer",
          repr(out)[:80])
    d = as_json(out)
    if d is not None and d.get("ok") is False:
        check(d.get("error") == "MODEL_RETURNED_NOTHING",
              "an empty completion is named rather than passed through",
              str(d.get("error")))
        check(d.get("retryable") is True, "and is retryable",
              str(d.get("retryable")))


def test_delegate_investigation_live():
    """A real second context, over the fixture index.

    The question has a known answer in the fixture -- sizeKvPool is declared in
    src/pool.ts -- so the investigation has something to find, and a finding
    that cites nothing is a real signal rather than an artefact of an empty
    index.
    """
    out = indexed("delegate_investigation",
                  {"question": "Where is sizeKvPool defined and what does it "
                               "divide?",
                   "context": "The repository is a small TypeScript project."})

    if not check(bool((out or "").strip()),
                 "delegate_investigation returns something", repr(out)[:80]):
        return
    check(T.leaked_exception(out) == [],
          "delegate_investigation leaks no raw exception", out[:200])

    d = as_json(out)
    if d is not None and d.get("ok") is False:
        # HELPER_BUSY is the expected shape when something else holds the lane.
        # It is a correct answer, and it means nothing below can be checked.
        check(d.get("error") in ("HELPER_BUSY",),
              "a refusal names a known situation", str(d.get("error")))
        check(False, "THE LANE WAS BUSY -- the checks below did not run. "
                     "Stop the other consumer and run again",
              json.dumps(d)[:200])
        return

    # THE COST LINE IS THE PRODUCT. Context economy is the entire argument for
    # this tool; an unreported saving is a claim rather than a result.
    # The wording is "thought about deeply" on purpose: the standing rule is
    # that the second brain is "thinking" / "deep thinking" in anything a
    # user sees, never "separate context". This check used to pin the old,
    # forbidden phrase and so failed against a correct rename.
    check("[thought about deeply:" in out,
          "the cost of the investigation is reported", out[-300:])
    handle = ""
    if "Trace handle " in out:
        handle = out.split("Trace handle ")[-1].strip().rstrip("].")
    check(bool(handle), "a trace handle is returned", out[-120:])

    trace = shomen.get_trace(handle) if handle else None
    if check(trace is not None, "the trace handle resolves to receipts", handle):
        steps = trace.get("trace") or []
        check(isinstance(steps, list), "the trace is a list of steps",
              type(steps).__name__)
        unknown = sorted({s.get("tool") for s in steps}
                         - set(T.proxy.OUR_NAMES))
        check(not unknown, "the investigator used only tools we offer",
              ", ".join(str(u) for u in unknown))
        check("delegate_investigation" not in {s.get("tool") for s in steps},
              "the investigator did not delegate again -- nothing bounds "
              "that depth")

    # THE SEARCHING MUST NOT ARRIVE HERE. The saving is the whole point: a
    # six-search investigation should cost the caller a few hundred tokens.
    # If the raw tool output crossed back, this string would carry the fenced
    # source of the files it read.
    check(len(out) < 4000,
          "the investigation's own traffic did not enter this conversation",
          f"{len(out)} chars returned")
    check("== TAPROOT" not in out and "CALL SITES:" not in out,
          "no raw search result crossed back", out[:200])

    # It really ran, rather than returning a canned line.
    for field in ("tool call", "tokens spent there"):
        check(field in out, f"the cost line reports {field}", out[-300:])


def main(argv: list[str]) -> int:
    if not enabled(argv):
        print(__doc__.strip().split("\n\n")[0])
        print("\nSKIPPED. These tests need the GPU and are opt-in.\n"
              "  run with:  python mcp/test_tools_live.py --live\n"
              "  or set:    YAMADORI_LIVE_TESTS=1\n"
              "\nThe gate that must always pass is mcp/test_tools.py, which "
              "needs no GPU.")
        return 0

    print(f"LIVE. stack={cs.STACK}  index={cs.INDEX_DB}")
    print("This uses the card. Nothing else should be using it.\n")

    for fn in (test_summarize_text_live,
               test_summarize_text_live_handles_a_trivial_input,
               test_delegate_investigation_live):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(T._results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            import traceback
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in T._results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in T._results if ok)
    total = len(T._results)
    print(f"\n{'=' * 70}\n  {passed}/{total} live checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
