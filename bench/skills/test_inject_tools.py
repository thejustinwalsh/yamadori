#!/usr/bin/env python
"""The injector's label and tuning tools (bench/skills/inject_labels.py,
inject_tune.py): offline, synthetic data.

    python bench/skills/test_inject_tools.py   -> "N/M checks passed"

GATES: H1's exact test (a code token used in the next code written and
absent from the material is NEEDED; present in both is DONE; in the
material only is AREA; neither is OFF; no code token -> not settled);
the three horizons stop where they say; Cohen's kappa; the tuning rows
take their truth from the labels; leave-one-run-out never scores a run on
thresholds selected with it.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_inject_tools_")
os.environ["YAMADORI_INJECT_LABELS"] = os.path.join(_TMP, "labels.jsonl")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
sys.path.insert(0, HERE)

import inject_labels as IL  # noqa: E402
import inject_tune as IT  # noqa: E402

CHECKS: list = []


def check(name, ok, detail=None):
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:300])


def call(name, args):
    return {"id": "c", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}


def main() -> int:
    fact = "Query entities with world.query(Position) each frame."
    check("code tokens are the item's code-shaped words",
          "world.query(Position)" in IL.code_tokens(fact)
          or "world.query" in IL.code_tokens(fact), IL.code_tokens(fact))
    check("an item with no code token is not settled by H1",
          IL.h1_level("Keep systems small and focused.", "", "x") is None)
    check("H1: used next, absent from the material -> 3",
          IL.h1_level(fact, "ls src", "const a = world.query(Position)")
          == 3)
    check("H1: used next and in the material -> 2",
          IL.h1_level(fact, "world.query(Position) failed",
                      "world.query(Position)") == 2)
    check("H1: in the material only -> 1",
          IL.h1_level(fact, "world.query(Position)", "npm run build") == 1)
    check("H1: neither -> 0", IL.h1_level(fact, "ls", "npm run build") == 0)
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                call("terminal", {"command": "ls src"})]},
            {"role": "tool", "content": "a.ts"},
            {"role": "assistant", "content": "", "tool_calls": [
                call("terminal", {"command": "cat src/a.ts"})]},
            {"role": "tool", "content": "..."},
            {"role": "assistant", "content": "", "tool_calls": [
                call("write_file", {"path": "src/b.ts",
                                    "content": "world.query(Position)"})]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "", "tool_calls": [
                call("terminal", {"command": "npm run build"})]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "", "tool_calls": [
                call("terminal", {"command": "echo later"})]}]
    one = IL.next_calls(msgs, 1, 1)
    wr = IL.next_calls(msgs, 1, "write")
    ep = IL.next_calls(msgs, 1, None)
    check("K=1 is the next assistant turn only",
          "ls src" in one and "cat src" not in one)
    check("K=write runs through the first turn that writes a file",
          "world.query" in wr and "npm run build" not in wr, wr)
    check("K=episode stops at the next user message",
          "npm run build" in ep and "echo later" not in ep)
    check("kappa: perfect agreement is 1, chance-level is ~0",
          IL.kappa([(1, 1), (0, 0), (1, 1), (0, 0)], (0, 1)) == 1.0
          and abs(IL.kappa([(1, 0), (0, 1), (1, 1), (0, 0)], (0, 1)))
          < 1e-9)
    # tuning rows and leave-one-run-out on synthetic runs
    runs = []
    truth = {}
    for g in ("runA", "runB", "runC"):
        for i in range(12):
            case = f"{g}{i}"
            need = i % 3 == 0
            truth[(case, "s#0")] = 3 if need else 0
            runs.append({"variant": "a", "model": "m", "case": case,
                         "source": g, "group": "hermes", "kind": "step",
                         "items": [{"key": "s#0", "belief": 0.9 if need
                                    else 0.1, "pass": need,
                                    "decision_id": f"d{case}"}],
                         "stage3": {"noul": 0.8 if need else 0.2,
                                    "tie": False, "listed": ["s#0"],
                                    "how": "passed",
                                    "decision_id": f"s{case}"}})
    s2, s3 = IT.rows_for(runs, "a", truth)
    check("tuning rows take their truth from the labels (level 3 -> true)",
          len(s2) == 36 and sum(r["truth"] == "true" for r in s2) == 12
          and len(s3) == 36)
    c = IT.confusion(s2, 0.5)
    check("confusion at a threshold", c["tp"] == 12 and c["fp"] == 0
          and c["precision"] == 1.0)
    IT.load = lambda model: runs
    IL.truth = lambda kind="hindsight": truth
    lo = IT.loro("m", 0.409, 0.266)
    check("leave-one-run-out: one fold per run, every row scored held out",
          lo["folds"] == 3 and lo["skill_item"]["untuned"]["tp"]
          + lo["skill_item"]["untuned"]["fn"] == 12, lo)
    check("a separable synthetic set ships at held-out precision's lower "
          "bound", lo["skill_item"]["act_yes"]["ships"] is True,
          lo["skill_item"])
    # inject_decide's case run, on a fake Turn (no GPU)
    import inject_decide as ID
    import decide_turn as T
    seen_states = []

    class FakeTurn:
        def __init__(self, msgs, state=None, **kw):
            seen_states.append(state)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def decide(self, qs):
            out = []
            for q in qs:
                if q["type"] == "noul":
                    out.append({"type": "noul", "name": q["name"],
                                "noul": 0.7, "decision_id": "x",
                                "diagnostics": {}})
                else:
                    k = len(q["options"])
                    p = {str(i): (0.7 if i == k - 1 else 0.3 / (k - 1))
                         for i in range(k)}
                    out.append({"type": "score", "name": q["name"],
                                "probabilities": p, "score": 1.0,
                                "confidence": 0.5, "decision_id": "y",
                                "diagnostics": {"argmax": str(k - 1)}})
            return out
    real = T.Turn
    T.Turn = FakeTurn
    try:
        case = {"case": "c1", "source": "runA", "group": "hermes",
                "kind": "step", "state": "Assistant called terminal: ls",
                "state_info": {"cut": False}, "state_sha": "s",
                "task": "build a game", "items": [
                    {"key": "k#0", "sha": "h", "fact": "Use `world.query`."},
                    {"key": "k#1", "sha": "h2", "fact": "Keep it small."}]}
        rows = ID.run_case(case, ["a", "b", "c", "d"], "bonsai")
    finally:
        T.Turn = real
    check("inject_decide: one row per variant, every item read, stage 3 "
          "over the passed items", [r["variant"] for r in rows]
          == ["a", "b", "c", "d"] and all(len(r["items"]) == 2
                                          and r["stage3"]["how"] == "passed"
                                          for r in rows), rows[:1])
    check("inject_decide: variant d reads a second state that names the "
          "goal first", len(seen_states) == 2
          and seen_states[1].startswith("GOAL (the session's opening "
                                        "request): build a game")
          and seen_states[1].endswith("Assistant called terminal: ls"),
          seen_states)
    # the Codex pass-C kit (bench/skills/codex_label/label_codex.py)
    sys.path.insert(0, os.path.join(HERE, "codex_label"))
    import label_codex as LC
    batch = [{"case": "c1", "kind": "step", "task": "t", "state": "s",
              "items": [{"key": "k#0", "skill": "x", "fact": "f0"},
                        {"key": "k#1", "skill": "x", "fact": "f1"}]}]
    got, why = LC.parse('{"labels": [{"case": "c1", "key": "k#0", '
                        '"level": 3}, {"case": "c1", "key": "k#1", '
                        '"level": 1}]}', batch)
    check("codex kit: a full reply parses into pass B's schema",
          got == {"c1": {"k#0": 3, "k#1": 1}} and why is None, (got, why))
    got, why = LC.parse('{"labels": [{"case": "c1", "key": "k#0", '
                        '"level": 7}]}', batch)
    check("codex kit: an out-of-range level or a missing fact is refused "
          "with why", got == {} and "2 of 2" in why, (got, why))
    check("codex kit: the call is read-only, ephemeral, schema-bound, and "
          "carries no dangerous flag",
          all(x in open(LC.__file__, encoding="utf-8").read() for x in (
              '"read-only"', '"--ephemeral"', '"--output-schema"'))
          and "dangerously" not in open(
              LC.__file__, encoding="utf-8").read().replace(
                  "no dangerous flag", "").split('"""', 2)[2]
          and "full-auto" not in open(LC.__file__, encoding="utf-8").read(
          ).split('"""', 2)[2])
    import prefill_check as PC
    open_ = ("<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n"
             "<think>\n" + PC.LINE)
    closed = open_ + "\n</think>\n\n<|im_end|>\n"
    check("prefill_check: an open think block after the line confirms, a "
          "closed one does not", PC.template_verdict(open_)["confirmed"]
          and not PC.template_verdict(closed)["confirmed"])
    g_yes = PC.generation_verdict({"choices": [{"finish_reason": "length",
                                                "message": {
                                                    "reasoning_content":
                                                    PC.LINE + " so next"}}]})
    g_no = PC.generation_verdict({"choices": [{"finish_reason": "stop",
                                               "message": {"content": "x"}}]})
    check("prefill_check: generation confirmed only when it goes on "
          "reasoning", g_yes["confirmed"] and g_yes["resent_line_first"]
          and not g_no["confirmed"])
    import render_probe as RP
    rows = [{"model": "m", "area": a_, "variants": {v: {
        "answered": True, "gen": {}, "pairs": [{"pair": (
            "better" if v == "table" else "same")}]} for v in
        ("list", "table", "first_person_prefill")}} for a_ in ("x", "y",
                                                               "z")]
    for r in rows:
        r["variants"]["first_person_prefill"]["pairs"] = [
            {"pair": "better"}, {"pair": "better"}]
    d = RP.decide(rows, "m")
    check("render_probe: the prefill variant is not eligible until the "
          "engine check confirms it", d["pick"] == "table"
          and not d["prefill_confirmed"], d)
    ho = RP.held_out(rows, "m")
    check("render_probe: leave-one-area-out scores each area on a pick made "
          "without it", ho["areas"] == 3 and ho["better"] == 3
          and ho["picks"] == {"table": 3}, ho)
    ok = sum(1 for _n, v in CHECKS if v)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
