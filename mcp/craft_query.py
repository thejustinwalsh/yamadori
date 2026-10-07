#!/usr/bin/env python
"""THE MODEL ASKS THE CRAFT LIBRARY A QUESTION, and jjava weighs it.

    text, rec = craft_query.answer(question, pool, ctx)

Operator, 2026-10-06 (verbatim): "a natural language query the model asks
that is weighed against the skills ... maybe the model asks some good
questions against our skills router?" Folded into yama_recall_craft (THE
TOOL RECIPE rule 1: no new tool; its argument is `name_or_topic`):

  * an exact or near-exact craft NAME is that craft, as before
    (skill_select.read_craft; `is_name`);
  * anything else is a QUESTION, answered in three steps:
      1. NARROW. The resident embedder ranks the armed, proven crafts against
         the question (skill_match.rank_all: the skills' own retrieval, no
         new index; lexical relevance when the embedder cannot answer) and the
         top skill_match.QUESTION_CANDIDATES go on, the crafts of the
         conversation's latest package-tool package first when there is one
         (package_skills' section state), then those of the packages the USER
         named in their own words (package_skills.user_named: the model's
         first move is often a question, before any lookup);
      2. DECIDE. ONE typed choice over that shortlist, WITHOUT a "none of
         these" option, read in both printed orders and averaged
         (decider_bonsai.read: the mc_avg form) -- the shortlist is already
         relevant, so the choice only ranks it (operator, 2026-10-06: "none is
         very tempting to call if your confidence is low"; SKILLS-RESEARCH.md
         Part 4 item 2) -- then ONE noul relevance gate on the best craft ("The
         craft <name> would make the answer more correct": skill_inject.FIT_Q, the
         measured variant g, both orders). The
         state is framed like skill_inject.framed_state: the model's question
         and the step's evidence -- WITHOUT the session goal (coordinator,
         2026-10-07: the probe pkgskills2's magnet craft won 15 of 25
         questions, generic ones included, and the likely cause is a goal that
         names r3f, Koota and a voxel scene in every read; the goal is out of
         these two reads only, recorded in `state`); each craft shown under its
         name (the 2026-10-06 diagnosis, docs/JJAVA.md 9: jjava separates
         on-topic from off-topic at AUROC 0.92 -- the model's question
         already asserts the need);
      3. ANSWER. The best craft in full, as the name path returns it
         (assured voice and caps are the skill's own), unless the gate is
         confidently "no" (tuned only) or the shortlist is empty: then an
         honest NO_SUCH_CRAFT answer that names the nearest crafts and says
         the package README (yama_read_package_readme) may cover it (Failure
         returns carry the next step). jjava unavailable: the same honest
         answer, naming the nearest.

THE CUT. skill_inject.tier(P(no), "craft_query_gate", model) reads the tuned
rows (skill_inject.THRESHOLDS); with none -- today -- the best craft is ALWAYS
returned and the record says `untuned`. No threshold is invented here. Both
reads (the choice's probabilities, the gate's p, ties and the orders'
disagreement) are recorded, so the gate can be tuned on labelled questions
(label: does the returned craft answer it).

THE RECORD, x_yamadori.craft.query: {question (first 200 characters),
shortlist ids, chosen, p, gate_p, untuned, choice, gate, ms, ...}; and a durable row per
query (skill_learn.record_craft_query: the account's traffic class, never the
key) so what the model asked becomes labels for jjava and the evidence for
gap-fill (what it asked that no craft answered). Only `client` traffic is
ever learned from (skill_learn.traffic_of).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skill_limits as L  # noqa: E402
import skill_prompts as P  # noqa: E402

QSET = "craft_query"
QSET_GATE = "craft_query_gate"
QUESTION_CHARS = 200            # what the record keeps of the question
LOG_QUESTION_CHARS = 1000       # what the durable log keeps (a model's own words)
# UNMEASURED WORDING until bench/skills/inject_decide.py (or the lookup probe's
# question arms) has read it on each model.
QUESTION_Q = ("Which craft answers the assistant's question about how to do "
              "this well?")
# THE GATE'S FORM (coordinator, 2026-10-07): the rework agent's variant g noul,
# skill_inject.FIT_Q -- "The assistant's next code or command would be more
# correct with this fact in front of it." -- the item shown under its craft
# name (skill_inject.q_fit / item_block: "CRAFT: <name>\nFACT: <the craft's
# trigger text>"); relevance AUROC 0.915 [0.888, 0.944] on 201 labelled cases
# (docs/JJAVA.md 9, bench/skills/inject/results/diagnose_20261006.json). It
# replaced "The craft <name> answers the question: <question>" (never
# measured). The question is in the state (state_text), so the read is
# "would this craft make the answer to THIS question more correct". Both
# orders, untuned, recorded.
GATE_VARIANT = "craft"
QUESTION_HEAD = "THE ASSISTANT'S QUESTION TO THE CRAFT LIBRARY:"


def _norm(x: str) -> str:
    return re.sub(r"[\s_]+", "-", str(x or "").strip().lower())


def is_name(q: str, pool: list[dict]) -> dict | None:
    """The craft the query NAMES: its name normalised, or that name less its
    trailing numeric id (`r3f-useframe` for `r3f-useframe-2`) when exactly
    one craft matches. A structural rule: no similarity score."""
    nq = _norm(q)
    if not nq:
        return None
    by = {_norm(s.get("name")): s for s in pool}
    if nq in by:
        return by[nq]
    hits = [s for n, s in by.items() if re.sub(r"-\d+$", "", n) == nq]
    return hits[0] if len(hits) == 1 else None


def proven(pool: list[dict]) -> list[dict]:
    """The armed crafts a question may be answered with: not `worse`,
    `unproven` or `not_run` at PROVE, at least one item that passes the
    assured voice (package_skills.PROVE_OF / eligible's rules)."""
    import package_skills as PS
    out = []
    for s in pool or []:
        if PS.PROVE_OF(s) in PS.PROVE_EXCLUDED:
            continue
        if not PS._items_of(s):
            continue
        out.append(s)
    return out


def shortlist(question: str, pool: list[dict], ctx: dict
              ) -> tuple[list[dict], dict]:
    """(the crafts to ask jjava about, the narrowing's record): the
    embedder's ranking (lexical relevance when it cannot answer), the
    package's crafts first, at most skill_match.QUESTION_CANDIDATES."""
    import skill_match
    import skill_select
    cos = None
    try:
        cos, _info = skill_match.rank_all(question, pool, ctx.get("embed"))
    except Exception:                                            # noqa: BLE001
        cos = None
    if cos is None:
        qs = skill_select.stems(question)
        score = {s["id"]: skill_select.relevance(qs, s, pool)[0]
                 for s in pool}
        how = "lexical"
    else:
        score, how = cos, "embedding"
    # THE PACKAGE BOOST, in two tiers (2026-10-07; the probe pkgskills2: the
    # model's first move was a craft question, before any package lookup, so
    # the boost never applied to 21 of 25): the package of the conversation's
    # latest package-tool section first, then the packages the USER named in
    # their own words (ctx["named"], package_skills.user_named: the registry's
    # terms, never a pasted manifest), then the rest by the embedder.
    import package_skills as PS
    tiers_: list[tuple[str, str, int | None]] = []
    pkg = ctx.get("package")
    if pkg:
        tiers_.append(("section", pkg, ctx.get("major")))
    for np_, nm in ctx.get("named") or []:
        if np_ != pkg:
            tiers_.append(("named", np_, nm))
    rank_of: dict[str, int] = {}
    for k, (_kind, p_, m_) in enumerate(tiers_):
        el = PS.eligible(p_, m_, pool)
        for s in ([el["lead"]] if el["lead"] else []) + el["others"]:
            rank_of.setdefault(s["id"], k)
    ranked = sorted(pool, key=lambda s: (rank_of.get(s["id"], len(tiers_)),
                                         -float(score.get(s["id"], -1.0)),
                                         s["name"]))
    top = ranked[:skill_match.QUESTION_CANDIDATES]
    in_top = {s["id"] for s in top}
    return top, {"retrieval": how, "package": pkg,
                 "named": [[p_, m_] for _k, p_, m_ in tiers_
                           if _k == "named"],
                 "package_boosted": sorted(
                     i for i in rank_of if i in in_top)[:8],
                 "embedding_ok": cos is not None}


def state_text(question: str, ctx: dict) -> tuple[str, dict, dict]:
    """(the state both of jjava's reads see, decide_turn's state info, the
    state's record): the model's QUESTION and the current step's evidence,
    framed as skill_inject.framed_state frames it -- WITHOUT the session goal
    (coordinator, 2026-10-07: the probe pkgskills2's magnet craft won 15 of 25
    questions, the generic ones included, and the likely cause is a goal that
    names r3f, Koota and a voxel scene in every read; the goal stays out of
    these two reads only). When the request is the user's opening turn (the
    model asked before any step) that turn IS the goal: nothing but the
    question is read. The record says the goal was kept out, how long it is
    and its sha1; the goal itself (cut to skill_inject.GOAL_CHARS) goes only
    to the durable log (`goal_full`), where a later read can be labelled with
    it."""
    import decide_turn
    import hashlib
    import skill_inject
    msgs = [m for m in ctx.get("messages") or [] if isinstance(m, dict)]
    try:
        st, info = decide_turn.state_of(msgs)
    except Exception:                                            # noqa: BLE001
        st, info = "", {"kind": "step"}
    goal = skill_inject.goal_of(msgs)
    first = info.get("kind") == "user" and not info.get("previous_assistant")
    body = "" if first else skill_inject.framed_state("", st, info)
    text = f"{QUESTION_HEAD}\n{question.strip()}" + (
        f"\n\n{body}" if body else "")
    return text, info, {
        "goal_in_state": False, "goal_chars": len(goal),
        "goal_sha1": hashlib.sha1(goal.encode("utf-8")).hexdigest()[:16]
        if goal else None,
        "evidence": "none (the opening turn is the goal)" if first else
        ("the newest user message" if info.get("kind") == "user"
         else "the step"),
        "goal_full": goal}


def _option(s: dict) -> str:
    import package_skills as PS
    return f"[{s['name']}] {PS.trigger_text(s)}"


def make_turn(msgs: list[dict], state: str, info: dict, ctx: dict):
    import decide_turn
    return decide_turn.Turn(msgs, key=ctx.get("lineage") or None,
                            account=ctx.get("account") or "", state=state,
                            state_info=info)


def _diag(a: dict) -> dict:
    d = a.get("diagnostics") or {}
    return {"tie": bool(d.get("tie")), "disagreement": d.get("disagreement"),
            "argmax_agree": d.get("argmax_agree")}


def ask(question: str, top: list[dict], ctx: dict) -> dict:
    """jjava's TWO typed reads, each in both option orders (decider_bonsai.
    read; the printed orders are averaged). Never raises.

      1. ONE choice over the shortlist, WITHOUT a "none" option (the
         operator, 2026-10-06: "none is very tempting to call if your
         confidence is low ... I thought the power pattern was ensuring the
         answer set was relevant"; SKILLS-RESEARCH.md Part 4 item 2: a "None
         of these" printed last sits in the last-option-favoured position,
         2607.05552, and low-confidence mass pools into a catch-all): the
         shortlist is already relevant, so the choice only ranks it.
      2. ONE noul relevance gate on the best craft: "The craft <name> answers
         the question: <question>" -- jjava's measured strength is relevance
         (AUROC 0.92 on-topic against off, 2026-10-06).

    {chosen: skill | None, best, p (the choice's), gate_p (P(yes)), untuned,
    tier, choice {probs, tie, ...}, gate {noul, ...}, failure?}. The craft is
    returned unless the gate's P(no) is high; with no tuned row
    (skill_inject.THRESHOLDS) it is ALWAYS returned and the record says
    `untuned` -- no cut is invented."""
    import decider_bonsai as D
    import skill_inject
    out: dict = {"chosen": None, "best": None, "p": None, "gate_p": None,
                 "untuned": True, "tier": "untuned"}
    state, info, srec = state_text(question, ctx)
    out["state"] = {k: v for k, v in srec.items() if k != "goal_full"}
    out["_goal_full"] = srec.get("goal_full")
    opts = [_option(s) for s in top]
    keys = [s["id"] for s in top]
    try:
        q = D.q_choice(QSET, QUESTION_Q, opts, keys=keys)
        factory = ctx.get("turn_factory") or make_turn
        turn = factory(ctx.get("messages") or [], state, info, ctx)
        if getattr(turn, "on", True) is False:
            raise D.DeciderUnavailable(
                "DECIDER_OFF", "the decider is off in this process", False,
                "the operator: the serving process enables the decider")
        with turn:
            a = turn.decide([q])[0]
            probs = {k: float(v) for k, v in
                     (a.get("probabilities") or {}).items() if k in keys}
            best = max(probs, key=lambda k: probs[k], default=None)                 if probs else keys[0]
            s = next((x for x in top if x["id"] == best), top[0])
            out["choice"] = dict(_diag(a), probs={k: round(v, 6) for k, v in
                                                  sorted(probs.items(),
                                                         key=lambda kv: -kv[1]
                                                         )[:6]},
                                 decision_id=a.get("decision_id"))
            out["best"], out["p"] = s, round(probs.get(best, 0.0), 6)
            import package_skills as PS
            g = turn.decide([skill_inject.q_fit(
                {"key": s["id"], "name": s["name"],
                 "fact": PS.trigger_text(s)}, GATE_VARIANT)])[0]
    except D.DeciderUnavailable as e:
        out["failure"] = e.facts()
        return out
    except Exception as e:                                       # noqa: BLE001
        out["failure"] = {"code": type(e).__name__,
                          "situation": str(e)[:160], "retryable": False}
        return out
    p_yes = float(g.get("noul") or 0.0)
    out["gate_p"] = round(p_yes, 6)
    out["gate"] = dict(_diag(g), noul=round(p_yes, 6),
                       decision_id=g.get("decision_id"))
    model = ctx.get("model") or D.model_name()
    t = skill_inject.tier(1.0 - p_yes, QSET_GATE, model)
    out["tier"], out["untuned"] = t, t == "untuned"
    # Untuned: the best craft is returned whatever the gate says. Tuned: it
    # is withheld only where the gate is confidently "no" (a high tier on
    # P(no) >= 0.5).
    if t == "untuned" or not (t == "high" and p_yes < 0.5):
        out["chosen"] = s
    return out


def no_answer(question: str, top: list[dict], rec: dict, ctx: dict) -> str:
    """The honest answer: no craft fits (or jjava could not be asked), the
    nearest crafts, and where else to look."""
    near = [s["name"] for s in top[:5]]
    pkg = ctx.get("package")
    why = ("jjava could not be asked (" + str(rec["failure"].get("code"))
           + "): these are the nearest by meaning." if rec.get("failure")
           else "No craft in this service's library answers that question.")
    remedies = [{"fixable_by": "agent",
                 "action": ("call it again with one of `near` by name, or "
                            "ask again naming the library and the API"
                            if near else
                            "carry on without it: nothing in this service's "
                            "library covers this"),
                 "effect": "the craft in full"}]
    if pkg:
        remedies.append({"fixable_by": "agent",
                         "action": f"call yama_read_package_readme for "
                                   f"{pkg}: its README may cover it",
                         "effect": "the package's own usage text"})
    return json.dumps({
        "tool": P.CRAFT_TOOL_NAME, "ok": False, "error": "NO_SUCH_CRAFT",
        "reason": why,
        "retryable": True, "near": near, "remedies": remedies})


def answer(question: str, pool: list[dict], ctx: dict
           ) -> tuple[str, dict]:
    """(the tool result, the query's record). Never raises: a failure is an
    honest no_answer with the next step."""
    import skill_select
    t0 = time.time()
    q = " ".join(str(question or "").split())
    rec: dict = {"question": q[:QUESTION_CHARS], "shortlist": [],
                 "chosen": None, "p": None, "gate_p": None, "untuned": True,
                 # for the durable log (the proxy pops it from the record)
                 "question_full": q[:LOG_QUESTION_CHARS]}
    try:
        cand = proven(pool)
        top, nrec = shortlist(q, cand, ctx) if cand else ([], {
            "retrieval": "none"})
        rec.update(nrec, shortlist=[s["id"] for s in top])
        if not top:
            rec["answered"] = False
            rec["ms"] = round((time.time() - t0) * 1000, 1)
            return no_answer(q, top, rec, ctx), rec
        got = ask(q, top, ctx)
        for k in ("p", "gate_p", "untuned", "tier", "choice", "gate",
                  "failure", "state"):
            if k in got:
                rec[k] = got[k]
        if got.get("_goal_full") is not None:
            rec["goal_full"] = got["_goal_full"]
        s = got.get("chosen")
        if got.get("best") is not None:
            rec["best"] = got["best"]["id"]
        if s is None:
            rec["answered"] = False
            rec["ms"] = round((time.time() - t0) * 1000, 1)
            return no_answer(q, top, rec, ctx), rec
        rec.update(chosen=s["id"], name=s.get("name"),
                   version=s.get("version"), answered=True,
                   tokens=L.tokens(skill_select.injected_text(s)))
        rec["ms"] = round((time.time() - t0) * 1000, 1)
        return (P.CRAFT_RESULT_HEAD.format(name=s.get("name")) + "\n\n"
                + skill_select.injected_text(s)), rec
    except Exception as e:                                       # noqa: BLE001
        rec.update(answered=False, failure={
            "code": type(e).__name__, "situation": str(e)[:160],
            "retryable": False})
        rec["ms"] = round((time.time() - t0) * 1000, 1)
        return no_answer(q, [], rec, ctx), rec
