#!/usr/bin/env python
"""The JevBench adapter (bench/decider/jevbench_run.py). No GPU, no network:
the model server is a fake passed in (post=, upstream=); runs write to a
temp directory.

    python mcp/test_jevbench_run.py      -> "N/M checks passed"

GATES: JevBench's item form maps onto jjava's typed questions (a noul with
its {true, false} criteria, a choice's {label: description} in the label
order, a score's levels 0..k; a JSON state as JSON text; more than 26
options refused before anything is sent); each item is one decide() batch
of two reads; our answer maps back to JevBench's exact-label distribution
(a noul as {yes, no}); a run writes ids and distributions, never item text,
resumes, and stops on three retryable failures without scoring the rest.

WITH THE CLONE (YAMADORI_JEVBENCH_DIR, else the default the adapter
records): every public item maps; real items are scored by JevBench's own
scoring.score_task and summarised with its composite_v13 beside the three
reference rows on the same items. Without the clone those checks are not
added (said in the output) and the rest use a stand-in scorer.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench", "decider"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_jevbench_run_")
os.environ["YAMADORI_SLOTS"] = "4"

import decider_bonsai as D  # noqa: E402
import jevbench_run as J  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail=None) -> None:
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


IDS = {L_: 600 + i for i, L_ in enumerate(D.LETTERS)}
IDS.update({" " + L_: 700 + i for i, L_ in enumerate(D.LETTERS)})


def upstream(path, payload=None, timeout=30):
    if path == "/tokenize":
        s = payload["content"]
        return {"tokens": [IDS[s]] if s in IDS else [1, 2]}
    raise AssertionError(path)


class Fake:
    """Favours the printed option whose text `want(texts)` returns (0.85),
    else uniform; records every body."""

    def __init__(self, want=None):
        self.want, self.bodies = want, []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        q = body["messages"][2]["content"]
        opts = [ln.split(". ", 1) for ln in
                q.split("OPTIONS:\n")[1].split("\n\n")[0].splitlines()]
        fav = self.want([t for _l, t in opts]) if self.want else None
        n = len(opts)
        top = sorted(({"id": IDS[" " + lab], "logprob": math.log(
            (0.85 if t == fav else 0.15 / (n - 1)) if fav else 1 / n)}
            for lab, t in opts), key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 70},
            "timings": {"prompt_n": 20, "cache_n": 50, "prompt_ms": 3.0}}


def task(**kw):
    """A JevBench-form item (our own wording, not the dataset's)."""
    base = {"id": "x", "family": "intent", "state": "s", "labels": [],
            "expected": None, "split": "public", "group": None,
            "provenance": {}}
    base.update(kw)
    return types.SimpleNamespace(**base)


def expected_text(t):
    """The printed option text of the item's expected answer."""
    q = t.question
    if q["type"] == "noul":
        c = q.get("criteria") or {}
        k = "true" if t.expected == "yes" else "false"
        head = "yes" if k == "true" else "no"
        return f"{head}: {c[k]}" if c.get(k) else head
    if q["type"] == "choice":
        return (q.get("criteria") or {}).get(t.expected, t.expected)
    return q["criteria"][int(t.expected)]


def stand_in():
    """A stand-in for JevBench's scoring when the clone is absent (argmax
    over the exact labels); the real one is used whenever it is there."""
    def score_task(probs, t):
        pred = max(sorted(probs), key=lambda k: probs[k])
        exp = str(t.expected)
        return {"valid": True, "probs": probs, "predicted": pred,
                "correct": pred == exp}
    return {"scoring": types.SimpleNamespace(score_task=score_task)}


def main() -> int:
    D.release = lambda slot, why="": {"released": True, "slot": slot}  # noqa: E731
    # --- the mapping
    tn = task(id="n1", question={"type": "noul", "instructions":
                                 "Does the customer want a refund?",
                                 "criteria": {"true": "money back asked",
                                              "false": "no money back"}},
              labels=["no", "yes"], expected="yes",
              state={"ticket": "I want my money back.", "tier": "gold"})
    tc = task(id="c1", question={"type": "choice", "instructions":
                                 "Which team owns it?", "criteria": {
                                     "billing": "charges and refunds",
                                     "shipping": "parcels in transit",
                                     "returns": "items sent back"}},
              labels=["shipping", "billing", "returns"], expected="billing")
    ts = task(id="s1", question={"type": "score", "instructions":
                                 "How severe?", "criteria": [
                                     "cosmetic", "degraded, workaround",
                                     "blocking"]},
              labels=["0", "1", "2"], expected=2)
    qn, qc, qs = J.to_question(tn), J.to_question(tc), J.to_question(ts)
    check("noul: the item's {true, false} criteria print on the lettered "
          "pair", qn["type"] == "noul" and qn["options"] ==
          ["yes: money back asked", "no: no money back"]
          and qn["name"] == J.QNAME)
    check("choice: options in the LABEL order, each the label's criterion, "
          "keyed by the label", qc["keys"] == ["shipping", "billing",
                                               "returns"]
          and qc["options"] == ["parcels in transit", "charges and refunds",
                                "items sent back"])
    check("score: levels 0..k as JevBench's labels are",
          qs["type"] == "score" and qs["keys"] == ["0", "1", "2"]
          and qs["options"][2] == "blocking")
    check("a JSON state is read as JSON text",
          J.state_text(tn.state).startswith("{\n")
          and json.loads(J.state_text(tn.state)) == tn.state
          and J.state_text("plain") == "plain")
    big = task(id="b1", question={"type": "choice", "instructions": "?",
                                  "criteria": {f"o{i}": f"option {i}"
                                               for i in range(30)}},
               labels=[f"o{i}" for i in range(30)], expected="o3")
    fk = Fake()
    rec = J.run_item("hard", big, stand_in(), post=fk, upstream=upstream)
    check("more than 26 options: refused before anything is sent, scored "
          "as a failure", not rec["ok"] and not rec["correct"]
          and rec["error"]["code"] == "TOO_MANY_OPTIONS"
          and fk.bodies == [], rec)

    # --- one item, one batch, two reads; our answer -> JevBench's labels
    jb = stand_in()
    for t in (tn, tc, ts):
        fk = Fake(lambda texts, t=t: expected_text(t))
        r = J.run_item("easy", t, jb, post=fk, upstream=upstream)
        check(f"[{t.question['type']}] one decide() batch of two reads; the "
              "distribution is over JevBench's exact labels and sums to 1; "
              "the expected answer wins", r["ok"] and len(fk.bodies) == 2
              and set(r["probs"]) == set(t.labels)
              and abs(sum(r["probs"].values()) - 1) < 1e-6
              and r["correct"] and r["usage"]["output_tokens"] == 2, r)
    r = J.run_item("easy", tn, jb, post=Fake(lambda x: expected_text(tn)),
                   upstream=upstream)
    check("a noul maps as JevBench's TypeSafe adapter maps Jev's: yes = "
          "noul, no = 1 - noul", abs(r["probs"]["yes"]
                                     - r["answer"]["noul"]) < 1e-9
          and abs(r["probs"]["no"] - (1 - r["answer"]["noul"])) < 1e-9)

    # --- a run: ids and distributions only; resume; stop on failures
    items = [("easy", tn), ("standard", tc), ("hard", ts)]
    run_dir = os.path.join(_TMP, "run1")
    fk = Fake(lambda texts: texts[0])
    recs = J.run(None, run_dir, items, jb, post=fk, upstream=upstream,
                 log=lambda *a: None)
    lines = open(os.path.join(run_dir, "items.jsonl"),
                 encoding="utf-8").read()
    check("a run writes one row per item: ids and distributions, never the "
          "item text", len(recs) == 3 and len(lines.splitlines()) == 3
          and "money back" not in lines and "Which team" not in lines)
    n_before = len(fk.bodies)
    recs = J.run(None, run_dir, items, jb, resume=True, post=fk,
                 upstream=upstream, log=lambda *a: None)
    check("--resume skips the items already run", len(recs) == 3
          and len(fk.bodies) == n_before)

    def down(body, timeout):
        raise ConnectionRefusedError("refused")
    said = []
    recs = J.run(None, os.path.join(_TMP, "run2"), items * 2, jb,
                 post=down, upstream=upstream, log=said.append)
    check("three retryable failures in a row stop the run; nothing is "
          "scored as wrong for them (NOT RUN, resumable)", recs == []
          and any("STOP" in s for s in said), said)
    check("McNemar's exact p: none without discordant pairs; small for "
          "10 vs 0", J.mcnemar_p(0, 0) is None
          and J.mcnemar_p(10, 0) < 0.01 and J.mcnemar_p(3, 3) == 1.0)

    # --- WITH THE CLONE: JevBench's own items, scoring and composite
    clone = J.DEFAULT_CLONE
    have = os.path.isdir(os.path.join(clone, "jevbench"))
    if not have:
        print(f"  (no JevBench clone at {clone}: the clone checks are not "
              "added)")
    else:
        st = J.clone_state(clone)
        check("[clone] the commit is recorded and pinned; the public files' "
              "canonical hashes match the clone's manifest",
              st["commit_ok"] and all(f["matches"]
                                      for f in st["files"].values()), st)
        jbr = J.jevbench(clone)
        items = J.load_items(clone, jbr)
        bad, most = [], 0
        for _tier, t in items:
            try:
                q = J.to_question(t)
                most = max(most, len(q["keys"]))
                want = (["true", "false"] if q["type"] == "noul"
                        else [str(x) for x in t.labels])
                if q["keys"] != want or (q["type"] == "noul"
                                         and sorted(t.labels)
                                         != ["no", "yes"]):
                    bad.append((t.id, "keys"))
            except D.DeciderUnavailable as e:
                bad.append((t.id, e.code))
        check("[clone] all 231 public items map onto typed questions (a "
              "noul keyed true/false over labels no/yes), keys "
              "= JevBench's labels, at most 26 options", len(items) == 231
              and not bad and most <= 26, (len(items), bad[:5], most))
        pick = []
        for tier in ("easy", "standard", "hard"):
            pick += [x for x in items if x[0] == tier][:2]
        pick += [x for x in items if x[1].question["type"] == "score"][:1]
        pick += [x for x in items if (x[1].provenance or {}).get(
            "gold_probs")][:1]
        recs = []
        for tier, t in pick:
            fk = Fake(lambda texts, t=t: expected_text(t)
                      if t.expected is not None else None)
            recs.append(J.run_item(tier, t, jbr, post=fk,
                                   upstream=upstream))
        check("[clone] real items are scored by JevBench's score_task: "
              "valid, and right when the model favours the gold answer",
              all(r["ok"] and r["valid"] and r["correct"] for r in recs),
              [(r["task_id"], r.get("valid"), r.get("correct"),
                r.get("error")) for r in recs])
        s = J.summarize(recs, [x for x in items if x in pick], jbr, clone)
        check("[clone] the summary: tier accuracies, intelligence (public "
              "chances and the release's), the public-hard calibration",
              s["tiers"] == {"easy": 1.0, "standard": 1.0, "hard": 1.0}
              and 0 <= s["intelligence_public"] <= 100
              and s["intelligence_release_chances"] is not None
              and s["calibration_public_hard"] is not None
              and s["n_tvd"] >= 1, s)
        refs = s["references"]
        check("[clone] the three reference rows of the same release, on the "
              "SAME items, with a paired count each",
              set(refs) == set(J.REFERENCES) and all(
                  set(r["tiers_public"]) == {"easy", "standard", "hard"}
                  and r["paired_on_items_run"]["n"] == len(pick)
                  and r["published_all_items"]["intelligence"]
                  for r in refs.values()), refs)
        full = J.summarize([], items, jbr, clone)
        ref = full["references"]["jev-1.13.0"]["tiers_public"]
        check("[clone] Jev 1.13.0's public-item accuracies are read from "
              "its per-item outcomes (easy all right, as published)",
              ref["easy"] == 1.0 and 0.5 < ref["hard"] < 1.0, ref)
    ok = sum(1 for _, o in CHECKS if o)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
