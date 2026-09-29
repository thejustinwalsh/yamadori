#!/usr/bin/env python
"""The decider's decision log and the labelling tool (mcp/decide_turn.py
DECISIONS + join_corpus; bench/decider/label.py). No GPU, no network: the
model server is a fake, the corpus and every log are temp files.

    python mcp/test_decider_label.py      -> "N/M checks passed"

GATES: a typed decision made through the real Turn and joined through
join_corpus is found by the labeller with its corpus turn; legacy
disagreement rows are read (phase letters -> phase names, booleans ->
true/false) and joined by time to the same account's next corpus turn,
the gap and the ends_on consistency shown; the h4 replay rows are rebuilt
from a session export and checked against the logged state hash; the
strata put disagreements first and split the rest at the family's median
margin; the research's first batch (phase: decider verify, rule implement)
is a pattern; labels hold ids and labels, never text, and are skipped next
time; the corpus is opened read-only.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench", "decider"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_decider_label_")
os.environ["YAMADORI_SLOTS"] = "4"

import corpus  # noqa: E402
import decider_bonsai as D  # noqa: E402
import decide_turn as T  # noqa: E402
import label as L  # noqa: E402

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


def fake(want):
    def post(body, timeout):
        m = body["messages"]
        q = next(x["content"] for x in m if x["role"] == "user"
                 and "OPTIONS:\n" in x["content"])
        opts = [ln.split(". ", 1) for ln in
                q.split("OPTIONS:\n")[1].split("\n\n")[0].splitlines()]
        fav = want([t for _l, t in opts])
        top = sorted(({"id": IDS[" " + lab], "logprob": math.log(
            (0.8 if t == fav else 0.2 / (len(opts) - 1)) if fav
            else 1 / len(opts))} for lab, t in opts),
            key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 40},
            "timings": {"prompt_n": 10, "cache_n": 30, "prompt_ms": 2.0}}
    return post


def main() -> int:
    D.release = lambda slot, why="": {"released": True, "slot": slot}  # noqa: E731
    check("every store is a temp file", all(
        os.path.abspath(p).startswith(os.path.abspath(_TMP))
        for p in (T.DECISIONS, T.LOG, L.DECISIONS, L.LEGACY, L.CORPUS,
                  L.LABELS)))
    t0 = time.time()
    # --- LEGACY rows (the disagreement log's shape), written before the
    # corpus turns they belong to
    legacy = [
        {"ts": t0 - 5, "account": "acct1", "conversation": "c1",
         "request": "u:aaa", "kind": "step", "question": "phase",
         "decider": "verify", "rule": "implement",
         "p": {"A": 0.05, "B": 0.25, "C": 0.1, "D": 0.6}},
        {"ts": t0 - 4, "account": "acct1", "conversation": "c1",
         "request": "u:bbb", "kind": "user", "question": "build_intent",
         "decider": True, "rule": False, "p": {"yes": 0.9, "no": 0.1}},
        {"ts": t0 - 3, "account": "", "conversation": "h4",
         "request": "req1", "kind": "step", "question": "phase",
         "decider": "verify", "rule": "implement",
         "p": {"A": 0.0, "B": 0.4, "C": 0.1, "D": 0.5}},
    ]
    with open(T.LOG, "w", encoding="utf-8") as f:
        for r in legacy:
            f.write(json.dumps(r) + "\n")
    # --- the corpus: two turns of acct1 (the real writer), after them
    corpus.log_turn("turnStep01", None, [
        {"role": "user", "content": "Build the voxel pagoda."},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "ok"}], [], False,
        route={"class": "agent_step", "signals": {"ends_on": "tool"}},
        account="acct1")
    time.sleep(0.01)
    corpus.log_turn("turnUser02", None, [
        {"role": "user", "content": "Add a pause menu please."}], [], False,
        route={"class": "code_generation", "signals": {"ends_on": "user"}},
        account="acct1")
    # --- a TYPED decision through the real Turn, joined by join_corpus
    with T.Turn([{"role": "user", "content": "Add a pause menu please."}],
                post=fake(lambda o: "yes" if "yes" in o else
                          "implementing the work"), upstream=upstream,
                count=len, on=True, key="c2", account="acct2",
                request="u:ccc") as t:
        facts = t.facts()
    j = T.join_corpus("u:ccc", "turnTyped03", account="acct2")
    check("join_corpus names the request's decisions (build intent, phase)",
          j["joined"] == 2, j)
    # --- the labeller
    ds, joins = L.load_decisions()
    by = {d["id"]: d for d in ds}
    L.join(ds, joins, L.corpus_turns())
    L.strata(ds)
    typed = [d for d in ds if d["source"] == "decisions"]
    check("the typed decisions are read with their join row",
          len(typed) == 2 and all(d["join"] == "join row"
                                  and d["corpus_turn"] == "turnTyped03"
                                  for d in typed), typed)
    bi = next(d for d in typed if d["question"] == "build_intent")
    check("a typed noul: keys true/false, its model, its orders' TV",
          bi["keys"] == ["true", "false"] and bi["pick"] == "true"
          and bi["rule"] == "true" and bi["disagreement"] == 0
          and bi["model"] is not None and bi["state_sha"]
          and len(bi["orders"]) == 2, bi)
    check("the typed decision ids are the ones the facts carry",
          {d["id"] for d in typed} == {facts["build_intent"]["decision_id"],
                                       facts["phase"]["decision_id"]})
    ph = next(d for d in typed if d["question"] == "phase")
    check("a typed choice carries Jev's confidence as logged, a noul its "
          "noul and no confidence; both untuned",
          abs(ph["confidence"] - D.confidence(ph["p"])) < 1e-5
          and ph["confidence"] > 0.5 and bi["confidence"] is None
          and abs(bi["noul"] - bi["p"]["true"]) < 1e-9
          and ph["tier"] == bi["tier"] == "untuned", (ph, bi))
    l1, l2, l3 = by["legacy:1"], by["legacy:2"], by["legacy:3"]
    check("a legacy row's confidence is computed from its distribution "
          "(Jev's form)", abs(l1["confidence"] - D.confidence(l1["p"]))
          < 1e-6 and l2["confidence"] is None and l2["noul"] == 0.9, l1)
    v1 = os.path.join(_TMP, "v1.jsonl")
    with open(v1, "w", encoding="utf-8") as f:
        f.write(json.dumps({"row": "decision", "v": 1, "id": "old1",
                            "question": {"type": "noul", "name": "x",
                                         "keys": ["true", "false"]},
                            "p": {"true": 0.5, "false": 0.5},
                            "pick": None, "argmax": "true",
                            "abstain": True, "tie": False}) + "\n")
    old, _j = L.load_decisions(v1, os.path.join(_TMP, "none.jsonl"))
    L.strata(old)
    check("a v1 row that abstained (the removed built-in abstain) reads as "
          "no pick, stratum tie", old[0]["pick"] is None
          and old[0]["stratum"] == "tie" and "abstain" not in old[0], old)
    check("legacy rows: ids by line, phase letters -> names, booleans -> "
          "true/false", l1["p"]["verify"] == 0.6 and l1["pick"] == "verify"
          and l1["rule"] == "implement" and l2["pick"] == "true"
          and l2["rule"] == "false" and l2["keys"] == ["true", "false"])
    check("legacy join: the same account's next corpus turn, the gap and "
          "the ends_on consistency (step -> tool, user -> user)",
          l1["join"] == "time (legacy)" and l1["corpus_turn"] == "turnStep01"
          and l1["join_consistent"] is True and l1["join_gap_s"] >= 0
          and l2["corpus_turn"] == "turnStep01"
          and l2["join_consistent"] is False, (l1, l2))
    check("a replay row (conversation h4, request req<k>) is marked replay",
          l3["join"] == "replay" and l3["corpus_turn"] is None)
    check("strata: the three legacy rows disagree; the typed agree with "
          "their rules", {l1["stratum"], l2["stratum"], l3["stratum"]} ==
          {"disagree"} and {d["stratum"] for d in typed} <=
          {"agree-low", "agree-high", "no-rule-low", "no-rule-high"},
          [(d["question"], d["pick"], d["rule"], d["stratum"]) for d in ds])
    pat = L.sample(ds, 10, pattern=L.FIRST_PATTERN)
    check("the research's first batch is a pattern: phase, decider verify, "
          "rule implement", {d["id"] for d in pat} == {"legacy:1",
                                                        "legacy:3"}, pat)
    allp = L.sample(ds, 10)
    check("the sample leads with disagreements", allp and all(
        d["stratum"] == "disagree" for d in allp[:3]),
        [(d["id"], d["stratum"]) for d in allp])
    # --- the text: corpus, with the step note; replay from a session
    turns = {t["turn"]: t for t in L.corpus_turns()}
    text, where = L.state_text(l1, turns, None)
    check("a step decision shows the corpus turn's user text and says the "
          "step itself is not in the corpus", text ==
          "Build the voxel pagoda." and "NOTE" in where
          and "time (legacy)" in where, where)
    session = os.path.join(_TMP, "session.jsonl")
    with open(session, "w", encoding="utf-8") as f:
        f.write(json.dumps({"system_prompt": "sys", "messages": [
            {"role": "user", "content": "make a game"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "terminal", "arguments": "{\"command\": \"ls\"}"
                }}]},
            {"role": "tool", "content": "index.html"},
            {"role": "assistant", "content": "done"}]}) + "\n")
    msgs = L._hermes_messages(session)
    text, where = L.state_text(l3, turns, msgs)
    check("a replay row's state is rebuilt from the session (decide_turn."
          "state_of at the k-th assistant point)", text is not None
          and "terminal" in text and "index.html" in text
          and "replay" in where, (text, where))
    # A typed decision of the replay (the h4turns shape: key "h4", request
    # "req<k>") logs its state hash; the rebuilt state matches it.
    clean = [m for m in msgs[:[i for i, m in enumerate(msgs)
                               if m["role"] == "assistant" and i > 0][1]]]
    with T.Turn(clean, post=fake(lambda o: "verifying or testing the work"),
                upstream=upstream, count=lambda s: len(s) // 3, on=True,
                key="h4", request="req1") as t:
        t.facts()
    ds2, joins2 = L.load_decisions()
    L.join(ds2, joins2, L.corpus_turns())
    L.strata(ds2)
    rp = next(d for d in ds2 if d["source"] == "decisions"
              and d["conversation"] == "h4")
    text, where = L.state_text(rp, turns, msgs)
    check("a replay decision's rebuilt state matches its logged hash",
          rp["join"] == "replay" and "matches the log" in where, where)
    # --- labelling: scripted answers
    out_lines = []
    answers = iter(["impl", "y", "q"])
    chosen = [l1, l2, l3]
    res = L.label_loop(chosen, turns, msgs, ask=lambda p: next(answers),
                       out=out_lines.append)
    rows = [json.loads(x) for x in open(L.LABELS, encoding="utf-8")]
    check("labels: a prefix picks the phase, y a noul's true; quit stops",
          res == {"labelled": 2, "skipped": 0, "unsure": 0}
          and [r["truth"] for r in rows] == ["implement", "true"], rows)
    check("a label row holds ids and labels (decision id, model, question, "
          "state hash, corpus turn), never the text",
          all("voxel" not in json.dumps(r) and "pause" not in json.dumps(r)
              for r in rows) and rows[0]["decision_id"] == "legacy:1"
          and rows[0]["corpus_turn"] == "turnStep01"
          and rows[0]["confidence"] == l1["confidence"]
          and rows[1]["noul"] == 0.9)
    check("labelled decisions are skipped next time",
          {"legacy:1", "legacy:2"} <= L.labelled_ids()
          and not any(d["id"] in ("legacy:1", "legacy:2") for d in
                      L.sample(ds, 10, skip=L.labelled_ids())))
    check("parse_truth: an option id names its key; u is unsure; nonsense "
          "is not understood",
          L.parse_truth({"type": "choice", "keys": ["A", "B"],
                         "option_ids": ["bash", "none"]}, "bash")
          == ("A", False)
          and L.parse_truth(l2, "u") == (None, True)
          and L.parse_truth(l2, "maybe") is None
          and L.parse_truth(l1, "") == "skip")
    before = os.path.getmtime(L.CORPUS)
    rc = L.main(["--list", "--n", "3"])
    rc2 = L.main(["--stats"])
    check("--list and --stats run, the corpus untouched (read-only)",
          rc == 0 and rc2 == 0 and os.path.getmtime(L.CORPUS) == before)
    ok = sum(1 for _, o in CHECKS if o)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
