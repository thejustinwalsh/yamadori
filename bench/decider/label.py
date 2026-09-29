#!/usr/bin/env python
"""LABEL THE DECIDER'S DECISIONS: a stratified sample of logged decisions,
the state text shown from where it is kept (read-only), a truth label typed
by the operator and appended to the labels file.

    python bench/decider/label.py --stats                 # counts, no text
    python bench/decider/label.py --pattern phase:verify/implement --n 20
    python bench/decider/label.py --question build_intent --n 30
    python bench/decider/label.py --list --n 40            # the sample, no
                                                           #   text, no input
    ... --session C:/Users/jwals/octo/logs/pagoda-h4-pagoda-xhigh-1/session.jsonl
                                  # the replay rows (conversation "h4")

docs/research/DECIDER-RESEARCH.md 4.2 item 1 (the highest-EV change): every
decider number we have is in-sample, and the disagreement log could not be
joined to the corpus (0 of 222 request ids matched), so nothing could be
labelled. This is the labelling half; decide_turn logs every decision with
its join keys (logs/decider_decisions.jsonl).

WHAT IS READ (never written):
  logs/decider_decisions.jsonl   decision rows and join rows (decide_turn
                                 DECISIONS; YAMADORI_DECIDER_DECISIONS)
  logs/decider_disagreements.jsonl  the LEGACY rows (before 2026-09-29): the
                                 research's 115 "decider verify vs rule
                                 implement" rows are here (--pattern
                                 phase:verify/implement)
  index/corpus.sqlite3           opened read-only (mode=ro): each corpus
                                 turn's head of the last user message,
                                 route and account
  --session PATH                 a Hermes session export, for the bench
                                 replay rows (conversation "h4": the
                                 decisions of bench/decider/bonsai_decider.py
                                 h4turns, request "req<k>" = the k-th
                                 assistant point)

WHAT IS WRITTEN: one row per label to index/decider/labels.jsonl
(YAMADORI_DECIDER_LABELS): the decision id, its MODEL (thresholds are tuned
per question set per model: bench/decider/tune.py reads these labels and
proposes decide_turn.THRESHOLDS rows for the owner to accept), the
question, the state hash and the corpus turn -- ids and labels, never
text. The truth of a state does not
depend on the model; the state hash lets a label be reused for another
model's decision on the same state and question.

THE JOIN, per decision (`join` in the sample):
  direct         the decision row carries corpus_turn
  join row       a "join" row (decide_turn.join_corpus) names it
  time (legacy)  a LEGACY row: the first corpus turn of the SAME ACCOUNT at
                 or after the decision's time (the decider runs in prepare,
                 before _run_turn logs the turn). A HEURISTIC, shown with
                 its gap in seconds and whether the corpus turn's ends_on
                 agrees with the decision's kind; judge it before trusting
                 the text
  replay         conversation "h4" with --session: the state rebuilt with
                 decide_turn.state_of (chars/3 as the token count: the
                 head+tail cut can differ from the live one)
  none           no text source; the row is listed, not shown

WHAT A STEP SHOWS: the corpus keeps only the LAST USER MESSAGE (corpus.
log_turn: "the full history is the user's code and belongs to them"), so an
agent step's own call and result are NOT in it -- the step decisions of
client traffic show the user's task only, and say so. Their state hash is
logged for when a step can be rebuilt from a source that holds it.

THE STRATA (sampled round-robin, in this priority): disagree (the rule's
answer differs), tie (no pick: a TIE_BAND tie, or a question set's low
tier), agree-low, no-rule-low, agree-high, no-rule-high. "low"/"high" split
at the MEDIAN margin of that question family's decisions in the log being
sampled (derived from the data, not a chosen number). The decider's
built-in abstain is gone (2026-09-29); a v1 row that says abstain is read
as no pick.

EACH DECISION carries Jev's `confidence` (Choice / Score; the logged value,
or decider_bonsai.confidence of its distribution for a row that predates
it) or its `noul` -- what tune.py bins accuracy against.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

DECISIONS = os.environ.get("YAMADORI_DECIDER_DECISIONS") or os.path.join(
    ROOT, "logs", "decider_decisions.jsonl")
LEGACY = os.environ.get("YAMADORI_DECIDER_LOG") or os.path.join(
    ROOT, "logs", "decider_disagreements.jsonl")
CORPUS = os.environ.get("YAMADORI_CORPUS_DB") or os.path.join(
    ROOT, "index", "corpus.sqlite3")
LABELS = os.environ.get("YAMADORI_DECIDER_LABELS") or os.path.join(
    ROOT, "index", "decider", "labels.jsonl")
LABEL_VERSION = 1

PHASES = ["plan", "implement", "debug", "verify"]
STRATA = ("disagree", "tie", "agree-low", "no-rule-low", "agree-high",
          "no-rule-high")
# The research's first batch (DECIDER-RESEARCH.md 4.2 item 1): the decider
# says verify, the rule says implement (115 of the 260 legacy rows).
FIRST_PATTERN = "phase:verify/implement"


# ------------------------------------------------------------- reading -----
def _jsonl(path: str) -> list[tuple[int, dict]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            try:
                out.append((i, json.loads(ln)))
            except ValueError:
                continue
    return out


def _margin(p: dict | None) -> float | None:
    vals = sorted((float(v) for v in (p or {}).values()), reverse=True)
    if not vals:
        return None
    return round(vals[0] - (vals[1] if len(vals) > 1 else 0.0), 6)


def _confidence(qtype, p: dict | None, logged):
    """Jev's confidence of a Choice / Score decision (the logged value, else
    decider_bonsai.confidence of its distribution); None for a noul, which
    has none (Jev /primitives/noul)."""
    if qtype == "noul":
        return None
    if logged is not None:
        return logged
    if not p:
        return None
    import decider_bonsai
    return round(decider_bonsai.confidence(p), 6)


def _family(name: str) -> str:
    return str(name or "").split(":", 1)[0]


def _norm_value(v):
    """A pick or a rule as a key: True/False -> "true"/"false"."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    return v


def _disagree(pick, rule) -> bool | None:
    if rule is None or pick is None:
        return None
    rule = _norm_value(rule)
    pick = _norm_value(pick)
    if isinstance(rule, list):
        return pick not in rule
    return pick != rule


def load_decisions(decisions: str = DECISIONS, legacy: str = LEGACY
                   ) -> tuple[list[dict], list[dict]]:
    """(decisions, join rows): every decision normalised; legacy rows get the
    id legacy:<line> (the file is append-only, so the line is stable)."""
    out, joins = [], []
    for _i, r in _jsonl(decisions):
        if r.get("row") == "join":
            joins.append(r)
            continue
        if r.get("row") != "decision":
            continue
        q = r.get("question") or {}
        pick = r.get("pick")
        if r.get("abstain") or r.get("tie"):       # abstain: v1 rows only
            pick = None
        p = r.get("p") or {}
        out.append({
            "id": r.get("id"), "source": "decisions", "ts": r.get("ts"),
            "model": r.get("model"), "readout": r.get("readout"),
            "account": r.get("account") or "",
            "conversation": r.get("conversation") or "",
            "request": r.get("request") or "", "kind": r.get("kind"),
            "question": q.get("name"), "family": _family(q.get("name")),
            "type": q.get("type"), "keys": q.get("keys") or [],
            "option_ids": r.get("option_ids"),
            "p": r.get("p") or {}, "pick": pick, "argmax": r.get("argmax"),
            "rule": _norm_value(r.get("rule")),
            "margin": r.get("margin") if r.get("margin") is not None
            else _margin(r.get("p")),
            "disagreement": r.get("disagreement"),
            "label_mass_min": r.get("label_mass_min"),
            "tie": bool(r.get("tie") or r.get("abstain")),
            "tier": r.get("tier") or "untuned",
            "confidence": _confidence(q.get("type"), p, r.get("confidence")),
            "noul": (r.get("noul") if r.get("noul") is not None
                     else p.get("true")) if q.get("type") == "noul"
            else None,
            "orders": r.get("orders"),
            "state_sha": (r.get("state") or {}).get("sha"),
            "corpus_turn": r.get("corpus_turn"),
            "join": "direct" if r.get("corpus_turn") else None})
    for i, r in _jsonl(legacy):
        name = r.get("question") or ""
        fam = _family(name)
        p = dict(r.get("p") or {})
        if fam == "phase" and set(p) <= set("ABCD"):
            p = {PHASES["ABCD".index(k)]: v for k, v in p.items()}
            keys = list(PHASES)
        elif set(p) <= {"yes", "no"} and p:
            p = {("true" if k == "yes" else "false"): v for k, v in p.items()}
            keys = ["true", "false"]
        else:
            keys = sorted(p)
        out.append({
            "id": f"legacy:{i}", "source": "legacy", "ts": r.get("ts"),
            "model": None, "readout": "legacy (disagreements only)",
            "account": r.get("account") or "",
            "conversation": r.get("conversation") or "",
            "request": r.get("request") or "", "kind": r.get("kind"),
            "question": name, "family": fam,
            "type": "choice" if fam == "phase" else "noul", "keys": keys,
            "option_ids": None, "p": p, "pick": _norm_value(r.get("decider")),
            "argmax": _norm_value(r.get("decider")),
            "rule": _norm_value(r.get("rule")), "margin": _margin(p),
            "disagreement": None, "label_mass_min": None, "tie": False,
            "tier": "untuned",
            "confidence": _confidence("choice" if fam == "phase" else "noul",
                                      p, None),
            "noul": p.get("true") if fam != "phase" else None,
            "orders": None, "state_sha": None,
            "corpus_turn": None, "join": None})
    return out, joins


def corpus_turns(path: str = CORPUS) -> list[dict]:
    """Every corpus turn: {turn, ts, account, request, route, ends_on,
    traffic}. Opened READ-ONLY."""
    if not os.path.exists(path):
        return []
    uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        rows = con.execute("SELECT turn, ts, payload FROM events WHERE "
                           "kind='turn' ORDER BY ts").fetchall()
    finally:
        con.close()
    out = []
    for turn, ts, payload in rows:
        try:
            pl = json.loads(payload or "{}")
        except ValueError:
            pl = {}
        out.append({"turn": turn, "ts": ts,
                    "account": pl.get("account") or "",
                    "request": pl.get("request") or "",
                    "route": pl.get("route"), "ends_on": pl.get("ends_on"),
                    "traffic": pl.get("traffic")})
    return out


def join(decisions: list[dict], joins: list[dict], turns: list[dict]
         ) -> None:
    """Fill each decision's corpus_turn and `join` (in place)."""
    by_id = {}
    for j in joins:
        for did in j.get("decisions") or []:
            by_id[did] = j.get("corpus_turn")
    by_acct: dict = collections.defaultdict(list)
    for t in turns:
        if t["account"]:
            by_acct[t["account"]].append(t)
    for d in decisions:
        if d["corpus_turn"]:
            continue
        if d["id"] in by_id:
            d.update(corpus_turn=by_id[d["id"]], join="join row")
            continue
        if d["source"] == "legacy" and d["account"] and d["ts"]:
            nxt = next((t for t in by_acct.get(d["account"], [])
                        if t["ts"] >= d["ts"]), None)
            if nxt:
                want = {"user": "user", "step": "tool"}.get(d["kind"])
                d.update(corpus_turn=nxt["turn"], join="time (legacy)",
                         join_gap_s=round(nxt["ts"] - d["ts"], 3),
                         join_consistent=(nxt["ends_on"] == want
                                          if nxt["ends_on"] and want
                                          else None))
                continue
        if d["conversation"] == "h4" and d["request"].startswith("req"):
            d["join"] = "replay"


# ------------------------------------------------------------ sampling -----
def strata(decisions: list[dict]) -> None:
    """Each decision's `stratum` (in place). The low/high split is the
    MEDIAN margin of the family's decisions."""
    med = {}
    for fam in {d["family"] for d in decisions}:
        ms = sorted(d["margin"] for d in decisions
                    if d["family"] == fam and d["margin"] is not None)
        med[fam] = ms[len(ms) // 2] if ms else 0.0
    for d in decisions:
        dis = _disagree(d["pick"], d["rule"])
        if dis:
            s = "disagree"
        elif d["pick"] is None:
            s = "tie"
        else:
            hi = (d["margin"] or 0.0) >= med[d["family"]]
            s = ("agree" if dis is False else "no-rule") + (
                "-high" if hi else "-low")
        d["stratum"] = s
        d["median_margin"] = med[d["family"]]


def matches(d: dict, pattern: str | None, question: str | None) -> bool:
    if question and d["question"] != question and d["family"] != question:
        return False
    if not pattern:
        return True
    fam, _, pr = pattern.partition(":")
    pick, _, rule = pr.partition("/")
    return (d["family"] == fam and (not pick or str(d["pick"]) == pick)
            and (not rule or str(d["rule"]) == rule))


def labelled_ids(path: str = LABELS) -> set:
    return {r.get("decision_id") for _i, r in _jsonl(path)
            if r.get("row") == "label"}


def sample(decisions: list[dict], n: int, *, seed: int = 0,
           pattern: str | None = None, question: str | None = None,
           skip: set | None = None) -> list[dict]:
    """The strata in STRATA's priority (disagreements first, then the low
    margins, then the high: DECIDER-RESEARCH.md 4.2 item 1), each drained
    round-robin across question families, seeded shuffle within a family;
    already labelled ids skipped."""
    rnd = random.Random(seed)
    pool = [d for d in decisions if matches(d, pattern, question)
            and d["id"] not in (skip or set())]
    out = []
    for s in STRATA:
        groups = []
        for fam in sorted({d["family"] for d in pool}):
            g = [d for d in pool if d["stratum"] == s and d["family"] == fam]
            if g:
                rnd.shuffle(g)
                groups.append(g)
        while len(out) < n and any(groups):
            for g in groups:
                if g and len(out) < n:
                    out.append(g.pop())
    return out


# ------------------------------------------------------------- the text ----
def _hermes_messages(path: str) -> list[dict]:
    """A Hermes session export -> chat messages (bench/skills/
    replay_selection._hermes_messages; the first line is the session)."""
    with open(path, encoding="utf-8") as f:
        s = json.loads(f.readline())
    msgs = [{"role": "system", "content": s.get("system_prompt") or ""}]
    for m in s.get("messages") or []:
        mm = {"role": m.get("role"), "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = m["tool_calls"]
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except ValueError:
                    calls = []
            mm["tool_calls"] = calls
        msgs.append(mm)
    return msgs


def state_text(d: dict, turns_by_id: dict, session: list[dict] | None
               ) -> tuple[str | None, str]:
    """(text, where it came from) for one decision; (None, why) when no
    source holds it."""
    if d["join"] == "replay" and session is not None:
        import decide_turn
        points = [i for i, m in enumerate(session)
                  if m["role"] == "assistant" and i > 0]
        k = int(d["request"][3:]) if d["request"][3:].isdigit() else -1
        if 0 <= k < len(points):
            clean = [m for m in session[:points[k]] if m.get("role") in (
                "system", "user", "assistant", "tool")]
            text, _info = decide_turn.state_of(clean)
            sha = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            ok = ("" if not d["state_sha"] else
                  " (state hash matches the log)" if sha == d["state_sha"]
                  else " (state hash DIFFERS from the log: the cut may "
                  "differ)")
            return text, f"replay: the session's point {k}" + ok
        return None, "replay: no such point in the session"
    t = turns_by_id.get(d["corpus_turn"]) if d["corpus_turn"] else None
    if t is None:
        return None, "no text source for this decision (join: " \
            f"{d['join'] or 'none'})"
    head = f"corpus turn {t['turn']} (route {t['route']}, ends on " \
        f"{t['ends_on']}, traffic {t['traffic']}; join: {d['join']}"
    if d.get("join_gap_s") is not None:
        head += f", {d['join_gap_s']} s after the decision" + (
            "" if d.get("join_consistent") is not False
            else ", ENDS_ON DISAGREES WITH THE DECISION'S KIND")
    head += ")"
    if d["kind"] == "step":
        head += ("\n  NOTE: an agent step's call and result are not in the "
                 "corpus (it keeps the last user message only); the text "
                 "below is the user's task, not the step judged")
    return t["request"], head


def show(d: dict, text: str | None, where: str, out=print) -> None:
    out("=" * 72)
    out(f"decision {d['id']}  [{d['stratum']}]  {d['question']} "
        f"({d['type']})  model {d['model'] or 'not recorded'}")
    out(f"  account {d['account'] or '-'}  conversation "
        f"{d['conversation'] or '-'}  kind {d['kind']}")
    out(f"  decider: {d['pick']}   rule: {d['rule']}   margin "
        f"{d['margin']} (family median {d['median_margin']})")
    if d.get("confidence") is not None or d.get("noul") is not None:
        out(f"  confidence {d.get('confidence')}   noul {d.get('noul')}   "
            f"tier {d.get('tier')}")
    if d.get("disagreement") is not None:
        out(f"  orders' disagreement (TV) {d['disagreement']}, label mass "
            f"{d['label_mass_min']}")
    out(f"  p: {json.dumps(d['p'])}")
    if d.get("option_ids"):
        out(f"  options: {d['keys']} = {d['option_ids']}")
    out(f"  -- {where}")
    out(text if text is not None else "  (no text to show)")


def record(d: dict, truth, *, unsure: bool = False, path: str = LABELS,
           labeller: str = "operator") -> dict:
    row = {"row": "label", "v": LABEL_VERSION, "ts": round(time.time(), 3),
           "decision_id": d["id"], "model": d["model"],
           "question": d["question"], "family": d["family"],
           "type": d["type"], "state_sha": d["state_sha"],
           "corpus_turn": d["corpus_turn"], "join": d["join"],
           "truth": truth, "unsure": bool(unsure), "labeller": labeller,
           "decider": d["pick"], "argmax": d.get("argmax"),
           "confidence": d.get("confidence"), "noul": d.get("noul"),
           "rule": d["rule"]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def parse_truth(d: dict, answer: str):
    """The operator's answer -> (truth key, unsure) | "skip" | "quit" |
    None (not understood)."""
    a = (answer or "").strip()
    low = a.lower()
    if low in ("s", "skip", ""):
        return "skip"
    if low in ("q", "quit"):
        return "quit"
    if low in ("u", "?", "unsure"):
        return (None, True)
    if d["type"] == "noul" or d["keys"] == ["true", "false"]:
        if low in ("y", "yes", "t", "true"):
            return ("true", False)
        if low in ("n", "no", "f", "false"):
            return ("false", False)
        return None
    keys = [str(k) for k in d["keys"]]
    if a in keys:
        return (a, False)
    ids = [str(x) for x in d.get("option_ids") or []]
    if a in ids:
        return (keys[ids.index(a)], False)
    hits = [k for k in keys if k.lower().startswith(low)]
    return (hits[0], False) if len(hits) == 1 else None


def label_loop(chosen: list[dict], turns_by_id: dict,
               session: list[dict] | None, *, ask=input, out=print,
               path: str = LABELS) -> dict:
    done = {"labelled": 0, "skipped": 0, "unsure": 0}
    for d in chosen:
        text, where = state_text(d, turns_by_id, session)
        show(d, text, where, out)
        hint = ("y / n" if d["type"] == "noul" else " / ".join(
            str(k) for k in d["keys"]))
        while True:
            got = parse_truth(d, ask(f"truth [{hint}; u unsure, s skip, "
                                     "q quit]: "))
            if got is not None:
                break
            out("  not understood")
        if got == "quit":
            break
        if got == "skip":
            done["skipped"] += 1
            continue
        truth, unsure = got
        record(d, truth, unsure=unsure, path=path)
        done["unsure" if unsure else "labelled"] += 1
    return done


def stats(decisions: list[dict]) -> dict:
    c = collections.Counter((d["source"], d["family"], d["stratum"],
                             d["join"] or "none") for d in decisions)
    return {"n": len(decisions),
            "by": [{"source": s, "family": f, "stratum": st, "join": j,
                    "n": n} for (s, f, st, j), n in sorted(c.items())]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--decisions", default=DECISIONS)
    ap.add_argument("--legacy", default=LEGACY)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--session", help="a Hermes session export for the "
                    "replay rows (conversation h4)")
    ap.add_argument("--pattern", help="family:pick/rule, e.g. "
                    f"{FIRST_PATTERN} (the research's first batch)")
    ap.add_argument("--question", help="a question name or family")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true",
                    help="print the sample (ids, strata, joins), no text")
    ap.add_argument("--stats", action="store_true",
                    help="counts by source, family, stratum and join")
    a = ap.parse_args(argv)
    decisions, joins = load_decisions(a.decisions, a.legacy)
    turns = corpus_turns(a.corpus)
    join(decisions, joins, turns)
    strata(decisions)
    if a.stats:
        print(json.dumps(stats(decisions), indent=1))
        return 0
    chosen = sample(decisions, a.n, seed=a.seed, pattern=a.pattern,
                    question=a.question, skip=labelled_ids(a.labels))
    if a.list:
        for d in chosen:
            print(json.dumps({k: d.get(k) for k in (
                "id", "stratum", "question", "pick", "rule", "margin",
                "join", "corpus_turn", "join_gap_s", "join_consistent")}))
        return 0
    session = _hermes_messages(a.session) if a.session else None
    res = label_loop(chosen, {t["turn"]: t for t in turns}, session,
                     path=a.labels)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
