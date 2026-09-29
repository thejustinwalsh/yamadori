#!/usr/bin/env python
"""The idle-time half of deep thinking's triggers: learn from the labels.

Phase 0.6 (docs/SELF-IMPROVEMENT-PLAN.md; operator, 2026-09-24). mcp/deep.py
records every request's decision -- a trigger, or none -- and labels it from
what the conversation did next (helped / not_helped / wasted for a run;
fine / missed / escalated_later / skipped otherwise). This job, on the
existing queue like skill_learn, turns labelled rows into:

  (a) THRESHOLD ADJUSTMENTS, armed automatically within deep.BOUNDS:
      struggle_threshold  missed struggles (MISSED) pull it down by 1;
                          wasted struggle runs (WASTED) push it up by 1
      (kickoff_tokens was retired 2026-09-27: every new task is planned,
      whatever its size. Its old deep_adjustments rows stay listed by
      adjustments() and set nothing; revert() refuses them.)
      A struggle run's `helped` means HELPED_WINDOW (20) later requests
      without the pattern it ran on (#45: with 5 and a reset episode it was
      automatic, so `wasted > helped + missed` could never hold). The
      same-pattern rule (label `missed_same_pattern`), the cooldown (label
      `in_cooldown`) and the ENVIRONMENT table were removed 2026-09-27
      (docs/CONSTANTS-AUDIT.md); old rows keep those labels, none are made.
      Each adjustment counts only the rows labelled since that parameter's
      last change, needs LEARN_MIN_N of the deciding label, and records its
      old and new value, its n and the counts -- and is reversible
      (revert(); POST /dash/api/deep/revert). An environment variable pins a
      parameter; the learner then records nothing for it.
  (b) yama_think_deeply DESCRIPTION VARIANTS, PROPOSED, NEVER ARMED: the user's
      "still broken" phrasings from missed rows where the model did not call
      the tool, added to the description's phrasing list. A description is
      a prompt (AGENTS.md); a variant is armed only after a paired, repeated
      measurement (docs/PROTOCOL.md), by a person. INERT for new rows since
      2026-09-27: the phrasings came from deep.STILL_BROKEN's
      user_still_broken events, which were removed (docs/CONSTANTS-AUDIT.md);
      only rows recorded before then carry a phrase.
  (c) E1 HEADS (mcp/e1.py, e1.learn): a head with enough labels its current
      version never saw is refitted; the candidate is promoted only if it
      beats the current version (or, untrained, the rule) on a held-out
      split of those new rows with an exact McNemar p < 0.05. Every
      candidate is kept as a version; e1.revert serves an older one
      (POST /dash/api/deep/e1/revert).

Every rule and number here is a CHOICE, none measured: LEARN_MIN_N 5, the
step size. Only CLIENT traffic is learned from (test traffic is
recorded and labelled, never counted: corpus.account_traffic).

WHEN IT RUNS: schedule() (the worker's loop, once a minute) enqueues one
`deep.learn` job on the cpu lane -- it reads sqlite and runs no model --
when there are labelled, unlearned client rows, no learn job is queued or
running, and no client request has reached the proxy for IDLE_MINUTES
(skill_learn's own reading of the corpus).
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deep  # noqa: E402
import jobs  # noqa: E402

LEARN = ("deep.learn", "cpu")
LEARN_MIN_N = int(os.environ.get("YAMADORI_DEEP_LEARN_MIN_N", "5"))
IDLE_MINUTES = float(os.environ.get("YAMADORI_DEEP_LEARN_IDLE_MINUTES", "15"))
# Labelled rows older than this that were never closed (the conversation
# stopped coming back) are closed as `unobserved` and not learned from.
STALE_SECONDS = 24 * 3600


def pending() -> int:
    con = deep._db()
    try:
        return con.execute(
            "SELECT COUNT(*) FROM deep_decisions WHERE label IS NOT NULL AND "
            "learned_at IS NULL AND traffic='client'").fetchone()[0]
    finally:
        con.close()


def idle_state(*, running: str | None = None) -> dict:
    n = pending()
    if not n:
        return {"idle": False, "why": "no labelled, unlearned client rows",
                "pending": 0}
    con = jobs._db()
    try:
        live = con.execute(
            "SELECT COUNT(*) FROM jobs WHERE queue=? AND state IN "
            "('queued','running') AND id != ?",
            (LEARN[0], running or "")).fetchone()[0]
    finally:
        con.close()
    if live and not running:
        return {"idle": False, "why": "a deep learn job is already queued or "
                                      "running", "pending": n}
    import skill_learn
    at, how = skill_learn._last_request_at()
    if at is None:
        return {"idle": False, "why": how, "pending": n}
    quiet = (time.time() - at) / 60 if at else float("inf")
    if quiet < IDLE_MINUTES:
        return {"idle": False, "why": f"the last client request was "
                                      f"{quiet:.1f} min ago (idle means "
                                      f"{IDLE_MINUTES:g})", "pending": n}
    return {"idle": True, "why": f"{n} labelled row(s) pending; no request "
                                 f"for {quiet:.0f} min", "pending": n}


def schedule() -> str | None:
    """Enqueue one learn job when the stack is idle. The worker calls this."""
    st = idle_state()
    if not st["idle"]:
        return None
    return jobs.add(LEARN[0], {"pending": st["pending"]}, lane=LEARN[1],
                    stage="learn")


def _last_change(con, param: str) -> float:
    row = con.execute("SELECT MAX(created) FROM deep_adjustments WHERE "
                      "param=?", (param,)).fetchone()
    return float(row[0] or 0.0)


# RUN ROWS LABELLED UNDER THE OLD RULE ARE NOT LEARNED FROM (#52,
# deep.LABEL_RULE): before rule 2 a run was `helped` whether or not anything
# changed, so 10 of 11 Octopus client runs were. A run row counts only when
# its label_rule is at least deep.LABEL_RULE; non-run rows are unchanged by
# the rule and always count. `no_effect` (a run after which no project file
# changed) counts with `wasted`: both say the run did not help (a CHOICE).
# Checked 2026-09-26 on index/corpus.sqlite3 (read-only): deep_adjustments
# held no rows -- the learner had adjusted nothing -- and E1's escalate and
# kickoff heads had no promoted version, so nothing was learned from the
# old-rule labels (8 struggle, 3 kickoff and 3 forced client runs `helped`).
_RULE_OK = ("(ran=0 OR COALESCE(label_rule, 0) >= %d)" % deep.LABEL_RULE)


def _counts(con, since: float) -> dict:
    """Labelled client rows since `since`, by (trigger, label), plus the
    missed rows where deep thinking was allowed."""
    out: dict = {}
    for r in con.execute(
            "SELECT trigger, label, allowed, signals FROM deep_decisions "
            "WHERE traffic='client' AND label IS NOT NULL AND labelled_at > ?"
            " AND " + _RULE_OK, (since,)):
        key = f"{r['trigger']}:{r['label']}"
        out[key] = out.get(key, 0) + 1
        if r["label"] == "missed" and r["allowed"]:
            out["missed_allowed"] = out.get("missed_allowed", 0) + 1
    return out


def _adjust(con, param: str, old: int, new: int, n: int, evidence: dict,
            why: str, author: str = "learner") -> str:
    aid = uuid.uuid4().hex[:12]
    con.execute("INSERT INTO deep_adjustments(id,created,param,old,new,n,"
                "evidence,why,author) VALUES(?,?,?,?,?,?,?,?,?)",
                (aid, time.time(), param, old, new, n, json.dumps(evidence),
                 why, author))
    return aid


def learn_thresholds(con) -> list[dict]:
    """(a): at most one step per parameter per run. Returns the changes."""
    thr = deep.thresholds(fresh=True)
    made = []
    for param in deep.DEFAULTS:
        cur = thr[param]
        if cur["source"] == "env":
            continue
        lo, hi = deep.BOUNDS[param]
        c = _counts(con, _last_change(con, param))
        old = int(cur["value"])
        new, why = old, None
        if param == "struggle_threshold":
            missed = c.get("missed_allowed", 0)
            wasted = c.get("struggle:wasted", 0) + c.get(
                "struggle:no_effect", 0)
            helped = c.get("struggle:helped", 0)
            n = missed + wasted + helped
            ev = {"missed": missed, "wasted": wasted, "helped": helped}
            if missed >= LEARN_MIN_N and missed > wasted:
                new = max(lo, old - 1)
                why = (f"{missed} missed struggles against {wasted} wasted "
                       f"struggle runs since the last change: fire earlier")
            elif wasted >= LEARN_MIN_N and wasted > helped + missed:
                new = min(hi, old + 1)
                why = (f"{wasted} wasted struggle runs against {helped} "
                       f"helped and {missed} missed: fire later")
        if why and new != old:
            aid = _adjust(con, param, old, new, n, ev, why)
            made.append({"id": aid, "param": param, "old": old, "new": new,
                         "n": n, "evidence": ev, "why": why})
    return made


def propose_descriptions(con, since: float) -> dict | None:
    """(b): one proposed variant of the yama_think_deeply description from the
    phrasings of missed rows where the model did not call it."""
    phrases: dict[str, int] = {}
    n = 0
    for r in con.execute(
            "SELECT signals FROM deep_decisions WHERE traffic='client' AND "
            "label='missed' AND model_calls=0 AND labelled_at > ?", (since,)):
        n += 1
        try:
            evs = (json.loads(r["signals"] or "{}").get("struggle") or {}
                   ).get("events") or []
        except ValueError:
            evs = []
        for e in evs:
            p = " ".join(str(e.get("phrase") or "").lower().split())
            if p:
                phrases[p] = phrases.get(p, 0) + 1
    known = deep.THINK_DESCRIPTION.lower()
    new = [p for p, _k in sorted(phrases.items(), key=lambda x: -x[1])
           if p not in known][:5]
    if n < LEARN_MIN_N or not new:
        return None
    text = deep.THINK_DESCRIPTION.replace(
        "or shows the same error.",
        "or shows the same error; or says " + ", ".join(
            f"'{p}'" for p in new) + ".")
    pid = uuid.uuid4().hex[:12]
    con.execute("INSERT INTO deep_proposals(id,created,kind,text,n,evidence,"
                "status) VALUES(?,?,?,?,?,?,?)",
                (pid, time.time(), "think_deeply_description", text, n,
                 json.dumps({"phrases": new, "counts": phrases}), "proposed"))
    return {"id": pid, "n": n, "phrases": new}


def handle_learn(job: dict, ctx) -> dict:
    st = idle_state(running=job["id"])
    if not st["idle"] and st.get("pending"):
        return {"deferred": st["why"]}
    con = deep._db()
    try:
        # Rows whose conversation never came back: closed, not learned from.
        stale = con.execute(
            "UPDATE deep_decisions SET label='unobserved', labelled_at=? "
            "WHERE label IS NULL AND created < ?",
            (time.time(), time.time() - STALE_SECONDS)).rowcount
        since = _last_change(con, "struggle_threshold")
        made = learn_thresholds(con)
        prop = propose_descriptions(con, since)
        n = con.execute(
            "UPDATE deep_decisions SET learned_at=? WHERE label IS NOT NULL "
            "AND learned_at IS NULL AND traffic='client'",
            (time.time(),)).rowcount
    finally:
        con.close()
    deep._THR_CACHE["val"] = None
    # (c) E1's heads (mcp/e1.py): retrain a head that has LEARN_MIN_NEW
    # labels its current version never saw; promote only on a paired win on
    # rows neither trained on. Reads vectors E1 stored when it served; embeds
    # nothing (this is the cpu lane). Every candidate is kept as a version.
    try:
        import e1
        e1_out = e1.learn(author=f"deep.learn:{job.get('id', '')}")
    except Exception as e:                                       # noqa: BLE001
        e1_out = [{"skipped": f"e1.learn raised {type(e).__name__}: {e}"}]
    if ctx is not None:
        ctx.beat(f"learned from {n} row(s)")
    return {"rows": n, "stale_closed": stale, "adjustments": made,
            "proposal": prop, "e1": e1_out}


def revert(adjustment_id: str, author: str = "operator") -> dict:
    """Undo one adjustment: its parameter goes back to the value before it,
    recorded as a new adjustment that names the one it reverts."""
    con = deep._db()
    try:
        row = con.execute("SELECT * FROM deep_adjustments WHERE id=?",
                          (adjustment_id,)).fetchone()
        if row is None:
            return {"ok": False, "error": f"no adjustment {adjustment_id}"}
        if row["reverted_at"]:
            return {"ok": False, "error": f"{adjustment_id} is already "
                                          f"reverted"}
        thr = deep.thresholds(fresh=True)
        if row["param"] not in thr:
            return {"ok": False, "error": f"{row['param']} is retired; its "
                                          f"adjustment {adjustment_id} sets "
                                          f"nothing"}
        cur = thr[row["param"]]["value"]
        con.execute("UPDATE deep_adjustments SET reverted_at=? WHERE id=?",
                    (time.time(), adjustment_id))
        aid = uuid.uuid4().hex[:12]
        con.execute("INSERT INTO deep_adjustments(id,created,param,old,new,n,"
                    "evidence,why,author,reverts) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (aid, time.time(), row["param"], cur, row["old"], 0, "{}",
                     f"revert of {adjustment_id}", author, adjustment_id))
    finally:
        con.close()
    deep._THR_CACHE["val"] = None
    return {"ok": True, "id": aid, "param": row["param"], "value": row["old"]}


def adjustments(limit: int = 50) -> list[dict]:
    con = deep._db()
    try:
        out = []
        for r in con.execute("SELECT * FROM deep_adjustments ORDER BY "
                             "created DESC LIMIT ?", (limit,)):
            d = dict(r)
            try:
                d["evidence"] = json.loads(d.get("evidence") or "{}")
            except ValueError:
                pass
            out.append(d)
        return out
    finally:
        con.close()


def proposals(limit: int = 20) -> list[dict]:
    con = deep._db()
    try:
        return [dict(r) for r in con.execute(
            "SELECT * FROM deep_proposals ORDER BY created DESC LIMIT ?",
            (limit,))]
    finally:
        con.close()


def overview() -> dict:
    """GET /dash/api/deep: thresholds in force, adjustments (reversible),
    proposals, per-day runs by trigger and labels, recent rows, idleness."""
    return {"thresholds": deep.thresholds(fresh=True),
            "bounds": {k: list(v) for k, v in deep.BOUNDS.items()},
            "adjustments": adjustments(),
            "proposals": proposals(),
            "per_day": deep.per_day(),
            "recent": [{k: r.get(k) for k in (
                "id", "created", "tier", "route", "allowed", "forced",
                "trigger", "fired", "ran", "model_calls", "label", "outcome",
                "traffic")} for r in deep.rows(30)],
            "learning": idle_state(),
            # missed_same_pattern and in_cooldown are no longer made
            # (2026-09-27); old rows still carry them.
            "labels": ["helped", "not_helped", "wasted", "no_effect",
                       "fine", "missed",
                       "missed_same_pattern", "escalated_later", "skipped",
                       "in_cooldown", "same_episode", "unobserved"],
            "e1": _e1_overview(),
            "note": "every threshold, rule and bound is a choice, unmeasured"}


def _e1_overview() -> dict:
    """E1's heads for the dashboard: every version with n, CV accuracy,
    status and the paired evaluation that promoted or rejected it."""
    try:
        import e1
        return e1.overview()
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print(json.dumps(overview(), indent=1, default=str))
