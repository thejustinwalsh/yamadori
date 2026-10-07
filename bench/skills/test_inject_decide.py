#!/usr/bin/env python
"""The injector's window runner (bench/skills/inject_decide.py): offline, a
fake model server behind the REAL typed readout, synthetic cases.

    python bench/skills/test_inject_decide.py   -> "N/M checks passed"

GATES: the served variants a-e ask what they asked (the default of
`--variants` is still those five); the new variants f, g and h read ONE
framed state per case (the goal first, the parts named) and write rows in the
shape inject_tune.py reads (one belief an item, `pass`, a decision id of the
case on every row, a stage-3 noul, h's `pick` record); `--dry-run` counts
their questions; `--skip-done` resumes per requested variant; `--only-
labelled` follows the rubric truth (pass A and the blind pass B); `--estimate`
fits the run's own rate on synthetic decisions and predicts the held-out half
within a few percent.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_inject_decide_")
os.environ["YAMADORI_INJECT_LABELS"] = os.path.join(_TMP, "labels.jsonl")
os.environ["YAMADORI_SLOTS"] = "4"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
sys.path.insert(0, HERE)

import decider_bonsai as D  # noqa: E402
import inject_decide as ID  # noqa: E402
import inject_tune as IT  # noqa: E402
import skill_inject as I  # noqa: E402

CHECKS: list = []


def check(name, ok, detail=None):
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


IDS = {}
for _i, _L in enumerate(D.LETTERS):
    IDS[_L] = 600 + _i
    IDS[" " + _L] = 700 + _i


def upstream(path, payload=None, timeout=30):
    if path == "/tokenize":
        s = payload["content"]
        return {"tokens": [IDS[s]] if s in IDS else [1, 2]}
    raise AssertionError(path)


class Fake:
    """A model server answering the new questions by their fact: every noul
    but DONE is yes for a fact with `yes_word` (DONE is always no), the pick
    favours that fact; the served Score answers its top level for it."""

    def __init__(self, yes_word="world.query", p=0.9):
        self.yes, self.p, self.bodies = yes_word, p, []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        m = body["messages"]
        q = m[2]["content"]
        opts = []
        for ln in q.split("OPTIONS:\n")[1].split("\n\n")[0].splitlines():
            lab, text = ln.split(". ", 1)
            opts.append((lab, text))
        fav = None
        if "FACT: " in q:
            fact = q.split("FACT: ", 1)[1].split("\n")[0]
            hit = self.yes in fact and not q.startswith(
                "QUESTION: " + I.DONE_Q)          # nothing is done yet
            fav = ("yes" if hit else "no") if any(
                t in ("yes", "no") for _l, t in opts) else (
                I.ITEM_LEVELS[3] if hit else I.ITEM_LEVELS[0])
        elif q.startswith("QUESTION: " + I.PICK_Q):
            fav = next((t for _l, t in opts if self.yes in t), None)
        elif "more correct with these facts" in q:
            fav = "yes"
        n = len(opts)
        top = []
        for lab, text in opts:
            pr = (self.p if text == fav else (1 - self.p) / max(n - 1, 1)) \
                if fav is not None else 1.0 / n
            top.append({"id": IDS[" " + lab], "logprob": math.log(pr)})
        top.sort(key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 60},
            "timings": {"prompt_n": 15, "cache_n": 45, "prompt_ms": 5.0}}


def case(cid, kind, items, state, task="Build the thing with koota.",
         info=None):
    return {"case": cid, "source": "src-" + cid, "group": "hermes",
            "index": 3, "kind": kind, "state_sha": "s" + cid,
            "state": state, "task": task,
            "state_info": info or {"kind": kind, "tokens": len(state) // 3,
                                   "cut": False, "previous_assistant": None},
            "items": [{"key": f"{cid}#{i}", "sha": f"sha{cid}{i}",
                       "skill": nm, "fact": fact}
                      for i, (nm, fact) in enumerate(items)]}


ITEMS = [("koota-queries", "Query entities with `world.query(Position)`."),
         ("koota-queries", "Do not: Store entity references across frames."),
         ("r3f-frame-loop", "Mutate refs inside `useFrame`."),
         ("math-noise", "Seed noise with `createNoise2D(seed)`.")]
STEP = case("aa11", "step", ITEMS,
            "Assistant called terminal: ls\nResult of terminal: a.ts")
USER = case("bb22", "user", ITEMS[:2], "Build the thing with koota.",
            info={"kind": "user", "tokens": 9, "cut": False,
                  "previous_assistant": None})


def main() -> int:
    D.release = lambda slot, why="": {"released": True, "slot": slot}
    # ---- the served variants are what they were
    qa = ID.questions("a", STEP["items"])
    check("variant a is still the served Score over the bare fact",
          qa[0]["type"] == "score" and qa[0]["options"] == I.ITEM_LEVELS
          and qa[0]["text"] == I.ITEM_Q.format(fact=STEP["items"][0]["fact"])
          and qa[0]["name"] == "skill_item:aa11#0")
    check("the default --variants is the served five, not f g h",
          ID.SERVED_VARIANTS == ("a", "b", "c", "d", "e")
          and ID.NEW_VARIANTS == ("f", "g", "h")
          and ID.VARIANTS == ID.SERVED_VARIANTS + ID.NEW_VARIANTS
          and ID.NEW_VARIANTS == I.NEW_VARIANTS)
    check("new variants' questions come from skill_inject, with the craft "
          "name the labellers saw", all(
              "CRAFT: koota-queries" in q["text"] for q in
              ID.questions("g", STEP["items"])[:2])
          and len(ID.questions("f", STEP["items"])) == 8
          and len(ID.questions("h", STEP["items"])) == 4)
    # ---- states
    fs = ID.frame_state(STEP)
    fu = ID.frame_state(USER)
    check("frame_state: a step is the goal, then the latest step",
          fs.startswith(I.GOAL_HEAD + "\nBuild the thing with koota.")
          and I.STEP_HEAD in fs and fs.endswith("Result of terminal: a.ts"),
          fs)
    check("frame_state: a first user turn says the request once",
          fu.startswith(I.OPENING_HEAD) and fu.count("Build the thing") == 1,
          fu)
    check("goal_state (variant d) is unchanged",
          ID.goal_state(STEP).startswith(
              "GOAL (the session's opening request): Build the thing")
          and "\n\nNOW:\n" in ID.goal_state(STEP))
    # ---- run one case, new variants, one Turn
    fake = Fake()
    rows = ID.run_case(STEP, ["f", "g", "h"], "m", post=fake,
                       upstream=upstream)
    check("run_case: one row per variant, in the shape inject_tune reads",
          [r["variant"] for r in rows] == ["f", "g", "h"]
          and all(set(r) >= {"v", "model", "variant", "case", "source",
                             "group", "kind", "state_sha", "items", "stage3",
                             "ms"} for r in rows)
          and all(len(r["items"]) == 4 for r in rows), rows[0].keys())
    mats = {b["messages"][1]["content"] for b in fake.bodies}
    check("f, g and h read ONE framed state (placed once on the lane)",
          len(mats) == 1 and next(iter(mats)).startswith(
              D.STATE_HEAD + I.GOAL_HEAD), list(mats)[:1])
    r = {x["variant"]: x for x in rows}
    check("item rows carry belief, pass, tie, a sha and a decision id; the "
          "stage-3 noul and what it listed",
          all(it["decision_id"] and it["sha"] and 0.0 <= it["belief"] <= 1.0
              and isinstance(it["pass"], bool) for x in rows
              for it in x["items"])
          and all(x["stage3"]["decision_id"] and x["stage3"]["listed"]
                  for x in rows), rows[1]["stage3"])
    check("f/g: the item the fake says yes to passes; the others do not",
          [it["pass"] for it in r["g"]["items"]] == [True, False, False,
                                                     False]
          and [it["pass"] for it in r["f"]["items"]] == [True, False, False,
                                                         False]
          and r["g"]["stage3"]["how"] == "passed", r["g"]["items"])
    check("h: the row carries the pick (none mass, the top, the decision) "
          "and an item outside the top has belief 0",
          r["h"]["pick"]["top"][0] == "aa11#0" and len(
              r["h"]["pick"]["top"]) == 3
          and sorted(it["belief"] for it in r["h"]["items"])[0] == 0.0
          and "pick" not in r["f"] and "pick" not in r["g"], r["h"]["pick"])
    # a, d unchanged next to them on a second fake
    fake2 = Fake()
    ra = ID.run_case(STEP, ["a", "d"], "m", post=fake2, upstream=upstream)
    mats2 = [b["messages"][1]["content"] for b in fake2.bodies]
    check("a reads the plain state, d the goal state, each its own Turn",
          [x["variant"] for x in ra] == ["a", "d"]
          and mats2[0] == D.STATE_HEAD + STEP["state"]
          and any(m.startswith(D.STATE_HEAD + "GOAL (the session's opening")
                  for m in mats2), mats2[:1])
    # ---- inject_tune reads the new rows
    truth = {("aa11", "aa11#0"): 3, ("aa11", "aa11#1"): 1,
             ("aa11", "aa11#2"): 0, ("aa11", "aa11#3"): 2}
    for v in ("f", "g", "h"):
        s2, s3 = IT.rows_for(rows, v, truth)
        check(f"inject_tune.rows_for reads variant {v}: 4 stage-2 rows, 1 "
              "stage-3 row, every row with a decision id",
              len(s2) == 4 and len(s3) == 1
              and all(x["id"] for x in s2 + s3)
              and [x["truth"] for x in s2] == ["true", "false", "false",
                                               "false"], (s2, s3))
    # ---- the CLI: dry run, skip-done, only-labelled
    cdir = tempfile.mkdtemp(prefix="inject_decide_cli_", dir=_TMP)
    cases_path = os.path.join(cdir, "cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as f:
        for c in (STEP, USER):
            f.write(json.dumps(c) + "\n")
    ID.RESULTS = os.path.join(cdir, "results")
    saved = {k: os.environ.get(k) for k in ("YAMADORI_DECIDER_DECISIONS",
                                            "YAMADORI_DECIDER_LOG")}

    def cli(*args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = ID.main(["--cases", cases_path, "--model", "m", *args])
        return code, json.loads(buf.getvalue().strip().splitlines()[-1])
    try:
        code, out = cli("--dry-run")
        check("--dry-run with no --variants counts the served five only",
              code == 0 and out["variants"] == ["a", "b", "c", "d", "e"]
              and out["cases"] == 2, out)
        code, out = cli("--dry-run", "--variants", "f,g,h")
        n = sum(len(ID.questions(v, c["items"])) + 1 for c in (STEP, USER)
                for v in ("f", "g", "h"))
        check("--dry-run --variants f,g,h counts every question of every "
              "case plus a stage-3 each", out["variants"] == ["f", "g", "h"]
              and out["questions"] == n and out["reads"] == 2 * n, (out, n))
        os.makedirs(ID.RESULTS, exist_ok=True)
        with open(os.path.join(ID.RESULTS, "run_m.jsonl"), "w",
                  encoding="utf-8") as f:
            for v in ("a", "b", "c", "d", "e"):
                f.write(json.dumps({"case": "aa11", "variant": v}) + "\n")
        _c, out = cli("--dry-run", "--variants", "a,b", "--skip-done")
        check("--skip-done skips a case with a row for every requested "
              "variant", out["cases"] == 1, out)
        _c, out = cli("--dry-run", "--variants", "f,g,h", "--skip-done")
        check("--skip-done: a case done for a-e is NOT done for f, g, h",
              out["cases"] == 2, out)
        # --only-labelled follows the rubric truth: pass A and pass B
        with open(os.environ["YAMADORI_INJECT_LABELS"], "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"row": "label", "case": "aa11",
                                "item": "aa11#0", "level": 3,
                                "labeller": "A"}) + "\n")
            f.write(json.dumps({"row": "label", "case": "bb22",
                                "item": "bb22#0", "level": 1,
                                "labeller": "B"}) + "\n")
        _c, out = cli("--dry-run", "--only-labelled")
        check("--only-labelled keeps the cases only the blind pass B read "
              "(the 2026-10-06 run left out 8 of them)", out["cases"] == 2,
              out)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # ---- the estimate: a synthetic run on a known line, held-out halves
    rdir = os.path.join(cdir, "est")
    os.makedirs(rdir, exist_ok=True)
    cases, run_rows, dec_rows = [], [], []
    for k in range(12):
        items = [(f"craft-{k % 3}", "fact " + "word " * (3 + (k * i) % 11))
                 for i in range(2 + k % 5)]
        c = case(f"{k + 1:04x}", "step" if k % 2 else "user", items,
                 "state " * (10 + 7 * k))
        cases.append(c)
        for v in ID.SERVED_VARIANTS:
            its = []
            for qi, (q, it) in enumerate(zip(ID.questions(v, c["items"]),
                                             c["items"])):
                chars = ID._chars(q)
                tok = 40 + 0.23 * chars
                ms = 500 + 1.4 * tok + (900 + 0.6 * c["state_info"][
                    "tokens"] if qi == 0 and v in ("a", "d") else 0)
                did = f"{c['case']}{v}{qi}"
                dec_rows.append({"row": "decision", "id": did,
                                 "processed_tokens": tok, "ms": ms})
                its.append({"key": it["key"], "decision_id": did})
            d3 = f"{c['case']}{v}s"
            dec_rows.append({"row": "decision", "id": d3,
                             "processed_tokens": 400, "ms": 1100})
            run_rows.append({"case": c["case"], "variant": v, "items": its,
                             "stage3": {"decision_id": d3},
                             "ms": sum(x["ms"] for x in dec_rows
                                       if x["id"].startswith(c["case"] + v))})
    with open(os.path.join(rdir, "run_m.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in run_rows) + "\n")
    with open(os.path.join(rdir, "decisions_m.jsonl"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in dec_rows) + "\n")
    est = ID.estimate(cases, cases, ["f", "g", "h"], "m", rdir)
    cal = est["calibration"]
    check("estimate: the fit recovers the synthetic lines (tokens = 40 + "
          "0.23 chars; ms = 500 + 1.4 tokens)",
          abs(cal["tokens_vs_chars"]["slope"] - 0.23) < 0.01
          and abs(cal["ms_vs_tokens"]["slope"] - 1.4) < 0.1
          and abs(cal["stage3_ms"] - 1100) < 1, cal)
    check("estimate: minutes for each new variant, g about half of f, "
          "every variant with its question and row counts",
          set(est["variants"]) == {"f", "g", "h"}
          and est["variants"]["f"]["stage2_questions"] == 2 * sum(
              len(c["items"]) for c in cases)
          and est["variants"]["g"]["stage2_questions"] == sum(
              len(c["items"]) for c in cases)
          and all(v["minutes"] > 0 for v in est["variants"].values())
          and est["variants"]["g"]["minutes"] < est["variants"]["f"][
              "minutes"], est["variants"])
    check("estimate: fitted on one half, it predicts the other half's "
          "recorded minutes within 5%, both ways round",
          len(est["check_held_out_halves"]) == 2
          and all(abs(h["error"]) < 0.05
                  for h in est["check_held_out_halves"]),
          est["check_held_out_halves"])
    check("estimate with no run to fit from says so", "error" in
          ID.estimate(cases, cases, ["g"], "none", rdir))
    ok = sum(1 for _n, v in CHECKS if v)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
