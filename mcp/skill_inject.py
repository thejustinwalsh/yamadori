#!/usr/bin/env python
"""THE SKILLS INJECTOR: one source, a jjava gate at every turn, a composer,
and a renderer per model.

    text, rec = skill_inject.inject(skills, kind, turn=t, model=m,
                                    given=st_items, events=triggers)

Operator, 2026-09-29 (verbatim): "Our skills engine also needs a skills
translator or some mechanism to be able to hand skills based on the same
source evidence over to each model with the best results, we know what works
and does not work from the skill research, but the wording and format may
change model to model, I don't want to re-write skills over and over, ideally
we have a skills engine that knows how to translate or compose skills
injections, with one reasoning engine hooked in at every turn weighing when
to use skills, this is aligned with the Jev research I posted from X, jjava
engine needs to be dialed in here, no easy bails, too much research showing
Jev works, fit the model, figure out the tune, send it home with a pagoda
that works on every tier." And, still standing: "All of our skills increase
confidence and improve correctness; if it can't, then the line doesn't need
to exist." (2026-09-28); skills reinforce what is asked, never blindly
(2026-09-27).

THE PIPELINE (docs/JJAVA.md 5.1, "two stages for a large candidate set":
filter in code, Score the survivors, a noul as the gate; Jev's skill
cookbook: wrong loads 16.8% -> 7.3%, needless 9.8% -> 4.0%)

  SOURCE    a skill's ITEMS are the one source of truth: each DO / WHEN / DO
            NOT line with its verbatim source quote (skill_md; the pipeline
            wrote and proved them). Nothing here rewrites an item's words: a
            model gets the same item in its own FORMAT and VOICE.
  STAGE 1   the mechanical filter, skill_select.decide as it stands: the
            filter rounds (language, framework, phase, situation, legality
            under the statechart), the package questions and the per-area
            choice. Its candidates -- at most MAX_SKILLS of them, in its
            order -- come here as skill rows; their items are the candidates.
  STAGE 2   jjava SCORE, one question per candidate item over THIS turn's
            state (decide_turn.Turn: the user turn or the agent step, cut to
            STATE_TOKENS; the state is placed once on the lane and each
            question is a short suffix after it). ITEM_Q / ITEM_LEVELS: the
            levels describe situations, lowest first (docs/JJAVA.md 4.1).
            Its gate is the belief that the item is at the TOP level
            (NEED_LEVEL: the turn works on what the item is about and does
            not yet show it done that way): `need` = P(top level).
  STAGE 3   jjava NOUL over the shortlist the composer built from the items
            stage 2 passed: "inject now or not" (INJECT_Q, a statement to
            judge, docs/JJAVA.md 4.1).
  COMPOSE   the passing items across skills, never whole-skill dumps:
            de-duplicated (the same text, or the same code spans), ordered
            by the belief (stage 2's tier, then its expected level), at most
            MAX_SKILLS skills and the profile's max_items.
  RENDER    the model's PROFILE (PROFILES, keyed by model family): format
            (list | table), voice (note: under the craft header |
            first_person: the model's own voice, "I'll ...", as text |
            first_person_prefill: one line of the model's own reasoning,
            which the proxy prefills -- the operator kept that channel,
            2026-09-29, to be measured against injected text), size
            (max_items). Text goes at the end of what the model reads next
            -- the user turn's tail or the tool result's, replayed by the
            ledger; a prefill line rides in rec["prefill"]. Every profile
            choice names its
            evidence: a measurement on that model (bench/skills/
            pitfall_harness.py: docs pitfall cases, WITHOUT and WITH each rendering, like
            skill_prove), else the research default it starts from.

THRESHOLDS belong to the question set (docs/JJAVA.md 3; operator 2026-09-29,
"callers own thresholds"): THRESHOLDS below, keyed by MODEL then question
set, three tiers -- high acts, medium proceeds with caution, low falls back
to the SAFE DEFAULT, which is NO INJECTION ("a safe default destination, and
confidence required to leave it", [AL]). What each tier does here:

  skill_item (stage 2, on `need`)  high: into the shortlist; medium: into
                                   the shortlist as "caution"; low: out.
  skill_inject (stage 3, noul)     high yes: every shortlisted item goes in;
                                   medium yes: only the stage-2 HIGH items;
                                   anything else: nothing goes in.

Tuned per model on labelled turns (bench/skills/inject_labels.py builds and
labels the set; bench/skills/inject_tune.py proposes rows through bench/
decider/tune.py's own code). EMPTY until the owner pastes a tuned row: an
untuned question set acts on the argmax (stage 2: the argmax level is the
top one; stage 3: noul >= 0.5), as every other question set does untuned
(decide_turn THRESHOLDS). The tier flag `skills` stays OFF at every tier
until the operator turns it on (2026-09-29, "Stop skills until we have a
good skill injector"); nothing here turns it on.

WHAT GOES TO RULES, AND WHY (coordinator, 2026-09-29: "Where a rule
already answers exactly, use the rule and keep jjava for what rules can't
see. Record which questions went to rules and why."):

  rule   the language / framework / phase / situation of a skill against
         the turn (stage 1's rounds): facts in the text -- fences, imports,
         pins, paths, the user's own words (skill_classify)
  rule   a package the conversation uses by name, import or pin (a fact;
         skill_packages.detect); a read inside node_modules/<pkg>/ is a
         path fact (skill_select.probed_packages: 0.948 vs the decider's
         best 0.888 on pagoda-h4, bench/decider/results/bonsai.json)
  rule   a craft about a framework the conversation does not use
         (skill_select.subject_unused) and a doubt-bearing line
         (skill_limits.doubt): never asked about
  rule   an item given before comes back only with an event (the chart)
  rule   a repeat (the same text or code spans) is asked once
  jjava  whether an item bears on what the assistant writes or runs NEXT,
         and whether it is still missing (stage 2) -- no rule sees it: on
         201 real turns, 209 of stage 1's 1,372 candidate items were
         NEEDED by the rubric (15.2%; bench/skills/inject_labels.py), and
         the mechanical hindsight rule H1 (the item's code used in the next
         code written) did not agree with the rubric (kappa 0.08, n=470)
  jjava  inject now or not (stage 3)

FAILURES (operator, 2026-09-29: "Why would jjava not be available? This
sounds like a failure to build our platform."): jjava is part of the
platform, always in scope where skills serve; there is NO path that picks
skills without it. When it truly cannot answer on a request -- no decider
Turn in scope, a decider switched off in the process, a DeciderUnavailable
(the model server down, the lane refused) -- NOTHING is injected and
x_yamadori.skills.inject says why. Offline suites answer through a stubbed
decider (mcp/decider_stub.py). Never raises out of inject().

THE RECORD (x_yamadori.skills.inject): ids, hashes and numbers -- never the
conversation's text; an item is named by its key "<skill id>#<index>".
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skill_limits as L  # noqa: E402

INJECTOR_VERSION = "inject/1"
QUESTION_VERSION = "inject-q/1"
# On unless switched off (the tier's `skills` flag is what allows skills at
# all; this picks the injector over the old per-skill bodies and recall
# lines when skills run).
ON = os.environ.get("YAMADORI_SKILL_INJECTOR", "1") != "0"

# ---------------------------------------------------------------- sizes ----
# At most this many SKILLS feed one injection. SkillsBench 2602.12670v4
# (87 tasks, 18 configurations, 3 trials): 1 skill +18.0 pp, 2-3 skills
# +19.0, 4 or more +10.1 (the buckets are confounded); docs/research/
# SKILLS-RESEARCH.md Part 4 item 1 recommends 3 on this evidence. Class C.
MAX_SKILLS = 3
# A profile's default max_items: one compact skill's worth, the pipeline's
# own per-skill item cap (skill_limits.MAX_ITEMS). SkillsBench: compact
# skills +19.0, standard +21.5, detailed +14.5, comprehensive +0.7.
DEFAULT_MAX_ITEMS = L.MAX_ITEMS

# ------------------------------------------------------------ questions ----
# THE STAGE-2 QUESTION (a Score, docs/JJAVA.md 4.1: one dimension; levels
# describe situations, not degrees; the fact is named in the question, the
# turn is the state). UNMEASURED WORDING until bench/skills/inject_decide.py
# has read it on each model against the labels.
ITEM_Q = ("How does this fact bear on what the assistant writes or runs "
          "next in the material?\n\nFACT: {fact}")
ITEM_LEVELS = [
    "It is about a library, API or task that the material does not involve.",
    "It is about a library the material involves, but not about the code, "
    "command or error the assistant is working on now.",
    "It is about the code, command or error the assistant is working on "
    "now, and the material already shows it done the way the fact says.",
    "It is about the code, command or error the assistant is working on "
    "now, and the material does not yet show it done the way the fact "
    "says.",
]
NEED_LEVEL = str(len(ITEM_LEVELS) - 1)
# THE STAGE-3 QUESTION (a Noul, a statement to judge). UNMEASURED WORDING.
INJECT_Q = ("The assistant's next code or command would be more correct "
            "with these facts in front of it:\n{facts}")
QSET_ITEM = "skill_item"
QSET_INJECT = "skill_inject"

# THE LANE'S ROOM FOR A QUESTION, DERIVED (mcp/budget.py "THE LANE'S SIZE"):
# the lane holds STATE_TOKENS of state plus LANE_TOKENS - STATE_TOKENS
# (1,024) cells for everything else of one read -- system, the state's head
# line, the question, its options, the answer lead. A stage-3 question whose
# estimate (skill_limits.tokens: chars / 3, a high estimate for English and
# code) would not fit drops its lowest-ranked facts until it does.
def question_room() -> int:
    import budget
    import decide_turn
    import decider_bonsai as D
    fixed = (D.SYSTEM + D.STATE_HEAD + D.CHOICE.format(
        question="", options="A. yes\nB. no") + D.ANSWER_LEAD)
    return max(budget.LANE_TOKENS - decide_turn.STATE_TOKENS
               - L.tokens(fixed), 0)


# ----------------------------------------------------------- thresholds ----
# {model: {question set: row}}: the rows decide_turn.tier reads, same shape
# (a Noul's act_yes / act_no / caution_yes / caution_no on the belief).
# skill_item's belief is `need` (P(top level)), read as a noul; skill_inject
# is a noul. EMPTY: NONE SHIP UNTUNED (operator). bench/skills/
# inject_tune.py proposes rows at the OPERATOR-ACCEPTED target precisions
# (2026-09-29, "yes"): high 0.409 = 11.5 / (16.6 + 11.5), SkillsBench
# 2602.12670v4's curated gain against its worst self-generated loss;
# medium 0.266 = 7.5 / (20.7 + 7.5), Skills in the Wild 2604.04323's curated
# gain against its distractor cost on Qwen -- a tier ships only where the
# LOWER Wilson bound of its held-out precision clears it. The owner pastes
# an accepted row here with its n and date. Nothing writes this table.
THRESHOLDS: dict = {}


def tier(belief: float, qset: str, model: str) -> str:
    """untuned | high | medium | low for a noul-like belief (decide_turn.tier's
    Noul bands, this module's THRESHOLDS)."""
    row = (THRESHOLDS.get(model) or {}).get(qset)
    if row is None:
        return "untuned"
    inf = float("inf")
    p = float(belief)
    if p >= row.get("act_yes", inf) or p <= row.get("act_no", -inf):
        return "high"
    if p >= row.get("caution_yes", inf) or p <= row.get("caution_no", -inf):
        return "medium"
    return "low"


# -------------------------------------------------------------- profiles ---
# ONE PROFILE PER MODEL FAMILY. Each choice carries its evidence: `measured`
# (bench/skills/pitfall_harness.py on that model: script, n, the paired
# better/worse counts) or `default` (the research finding it starts from).
# Mirai S is a Qwen3.8-27B 2.4-bit quant and Flash-Next is Qwen3.8-Flash-
# Next: the same family's defaults, until each is measured on its own model.
RESEARCH_DEFAULTS = {
    "format": ("list", "SkillsBench 2602.12670v4: curated compact skills "
               "(markdown lists) +19.0 pp; AWM 2409.07429: text vs code "
               "format within 0.3-0.6; the table is the measured "
               "alternative (AGENTS.md 'Prompting this model': a decision "
               "table beat prose 10.7 vs 10.0 on Bonsai, at risk #31)"),
    "voice": ("note", "the end of what the model reads next: Lost in the "
              "Middle 2307.03172 (U-curve), Laban 2505.06120 (Recap 50.4 -> "
              "66.5), Manus [V] (recitation at the end); first_person (the "
              "model's own reasoning line) is the measured alternative"),
    "max_items": (DEFAULT_MAX_ITEMS, "one compact skill (skill_limits."
                  "MAX_ITEMS); ReasoningBank 2509.25140 (1 item 49.7 -> 4 "
                  "items 44.4) is the measured alternative, max_items 1"),
}


def _default_profile(family: str) -> dict:
    return {"family": family,
            "format": RESEARCH_DEFAULTS["format"][0],
            "voice": RESEARCH_DEFAULTS["voice"][0],
            "max_items": RESEARCH_DEFAULTS["max_items"][0],
            "evidence": {k: {"kind": "default", "source": v[1]}
                         for k, v in RESEARCH_DEFAULTS.items()}}


PROFILES: dict[str, dict] = {f: _default_profile(f) for f in
                             ("bonsai", "mirai-s", "flash-next")}
FAMILIES = (("flash-next", ("flash-next", "flash_next", "flashnext")),
            ("mirai-s", ("mirai",)),
            ("bonsai", ("bonsai",)))


def family_of(model: str | None) -> str:
    m = str(model or "").lower()
    for fam, keys in FAMILIES:
        if any(k in m for k in keys):
            return fam
    return "default"


def profile_for(model: str | None) -> dict:
    fam = family_of(model)
    return PROFILES.get(fam) or _default_profile(fam)


# ----------------------------------------------------------------- items ---
_SPAN = re.compile(r"`([^`\n]{2,80})`")


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _form(it: dict) -> str:
    f = str(it.get("form") or "DO").upper().replace("_", " ").strip()
    return "DO NOT" if f == "DO NOT" else ("WHEN" if f == "WHEN" else "DO")


def skill_items(s: dict) -> list[dict]:
    """The skill's items as the injector reads them: {key, skill, name, i,
    form, situation, text, quote, spans, doubt}. `doubt` is skill_limits.
    doubt's reason (the ASSURED VOICE rule): such an item never goes in."""
    import skill_select
    out = []
    for i, it in enumerate(skill_select._items(s)):
        text = " ".join(str(it.get("text") or "").split())
        if not text:
            continue
        sit = " ".join(str(it.get("situation") or "").split())
        out.append({"key": f"{s['id']}#{i}", "skill": s["id"],
                    "name": s.get("name") or s["id"], "i": i,
                    "form": _form(it), "situation": sit, "text": text,
                    "quote": str(it.get("quote") or ""),
                    "spans": sorted({m.group(1).strip() for m in
                                     _SPAN.finditer(text)}),
                    "doubt": L.doubt(text) or (L.doubt(sit) if sit else None)})
    return out


def fact_text(it: dict) -> str:
    """One item as the decider reads it (its form, as the skill says it)."""
    if it["form"] == "WHEN" and it["situation"]:
        return f"When {it['situation']}: {it['text']}"
    if it["form"] == "DO NOT":
        return f"Do not: {it['text']}"
    return it["text"]


def candidates(skills: list[dict], given: dict | None = None,
               events: dict | None = None, last: list | None = None
               ) -> tuple[list[dict], list[dict]]:
    """(items to ask about, items left out with why): the first MAX_SKILLS
    skills' items in stage 1's order, the doubt-bearing ones out, a repeat
    of an earlier item's text or code spans out (the first kept), an item
    this conversation was already given out unless its skill comes with an
    event (`events` {skill id: trigger}: error / phase / asked), and an item
    of the conversation's LAST injection (`last`, {key: the trigger it went
    in on}) out -- never the identical injection twice in a row (the
    selector's standing rule, 2026-09-27: "never the identical line twice
    in a row", carried to items) -- except that an ERROR brings back an
    item last given for another reason (a body, then the error it
    prevents: the old recall), once."""
    given = given or {}
    events = events or {}
    last = dict(last or {}) if isinstance(last, dict) else {
        k: None for k in (last or ())}
    keep, out = [], []
    seen_text, seen_spans = set(), set()
    for n, s in enumerate(skills or []):
        if n >= MAX_SKILLS:
            out.append({"skill": s["id"], "why": f"past MAX_SKILLS "
                        f"({MAX_SKILLS}, SkillsBench)"})
            continue
        for it in skill_items(s):
            if it["doubt"]:
                out.append({"key": it["key"], "why": f"doubt: {it['doubt']}"})
                continue
            nt = _norm(fact_text(it))
            sp = frozenset(x.lower() for x in it["spans"])
            if nt in seen_text or (sp and sp in seen_spans):
                out.append({"key": it["key"], "why": "a repeat of an item "
                            "already asked"})
                continue
            if it["key"] in given and events.get(it["skill"]) not in (
                    "error", "phase", "asked"):
                out.append({"key": it["key"], "why": "given before; no "
                            "event brings it back"})
                continue
            if it["key"] in last and not (
                    events.get(it["skill"]) == "error"
                    and last[it["key"]] != "error"):
                out.append({"key": it["key"], "why": "the same item as the "
                            "last injection"})
                continue
            seen_text.add(nt)
            if sp:
                seen_spans.add(sp)
            keep.append(it)
    return keep, out


# ------------------------------------------------------------------ gate ---
def item_question(it: dict) -> dict:
    import decider_bonsai as D
    return D.q_score(f"{QSET_ITEM}:{it['key']}",
                     ITEM_Q.format(fact=fact_text(it)), ITEM_LEVELS)


def inject_question(items: list[dict], room: int | None = None
                    ) -> tuple[dict, list[dict]]:
    """(the stage-3 noul, the items it lists): the shortlist in its order,
    the lowest-ranked facts dropped until the question fits the lane's
    room (question_room)."""
    import decider_bonsai as D
    room = question_room() if room is None else room
    use = list(items)
    while True:
        facts = "\n".join(f"- {fact_text(it)}" for it in use)
        text = INJECT_Q.format(facts=facts)
        if L.tokens(text) <= room or len(use) <= 1:
            return D.q_noul(QSET_INJECT, text), use
        use = use[:-1]


def _need(a: dict) -> float:
    return float((a.get("probabilities") or {}).get(NEED_LEVEL, 0.0))


def _argmax_level(a: dict) -> str | None:
    p = a.get("probabilities") or {}
    return max(p, key=p.get) if p else None


def gate(items: list[dict], turn, model: str) -> dict:
    """Stages 2 and 3 on `turn` (decide_turn.Turn, or anything with its
    decide()). Returns {items: [{key, need, score, confidence, level, tier,
    pass, decision_id}], shortlist, inject: {noul, tier, act, keys,
    decision_id} | None, chosen: [keys], ms, failure?}. Raises nothing: a
    DeciderUnavailable leaves `failure` and chooses nothing."""
    import decider_bonsai as D
    t0 = time.time()
    rec: dict = {"items": [], "shortlist": [], "inject": None, "chosen": []}
    by = {it["key"]: it for it in items}
    try:
        answers = turn.decide([item_question(it) for it in items]) \
            if items else []
        for it, a in zip(items, answers):
            need = _need(a)
            lvl = _argmax_level(a)
            t = tier(need, QSET_ITEM, model)
            ok = (lvl == NEED_LEVEL and not (a.get("diagnostics") or {})
                  .get("tie")) if t == "untuned" else t in ("high", "medium")
            rec["items"].append({
                "key": it["key"], "need": round(need, 6),
                "score": a.get("score"), "confidence": a.get("confidence"),
                "level": lvl, "tier": t, "pass": bool(ok),
                "caution": t == "medium",
                "decision_id": a.get("decision_id")})
        passed = [r for r in rec["items"] if r["pass"]]
        passed.sort(key=lambda r: (r["caution"], -float(r["score"] or 0.0),
                                   -r["need"]))
        rec["shortlist"] = [r["key"] for r in passed]
        if passed:
            q, listed = inject_question([by[r["key"]] for r in passed])
            a = turn.decide([q])[0]
            p = float(a.get("noul") or 0.0)
            t = tier(p, QSET_INJECT, model)
            tie = bool((a.get("diagnostics") or {}).get("tie"))
            if t == "untuned":
                act = "all" if p >= 0.5 and not tie else "none"
            elif t == "high" and p >= 0.5:
                act = "all"
            elif t == "medium" and p >= 0.5:
                act = "sure_only"
            else:
                act = "none"
            keys = [it["key"] for it in listed]
            rec["inject"] = {"noul": round(p, 6), "tier": t, "act": act,
                             "tie": tie, "keys": keys,
                             "decision_id": a.get("decision_id")}
            if act == "all":
                rec["chosen"] = keys
            elif act == "sure_only":
                sure = {r["key"] for r in passed if not r["caution"]}
                rec["chosen"] = [k for k in keys if k in sure]
    except D.DeciderUnavailable as e:
        rec["failure"] = e.facts()
        rec["chosen"] = []
    rec["ms"] = round((time.time() - t0) * 1000, 1)
    return rec


# ================================================ QUESTION-SET VARIANTS ====
# THE REWORK OF 2026-10-06 (operator: "I want skills and the decision maker
# doing their damn jobs"; standing rule 2026-09-29: "no easy bails" -- a
# question that will not separate is rewritten and re-measured, never
# replaced by a default). WHAT THE MEASUREMENT SAID, and what each variant
# is built against (bench/skills/inject/results, reader bonsai-a4000, 193
# labelled cases, 1,338 labelled items, rubric truth, variant a as served;
# every number below is that run's, n=1 run, the reads repeat to SD 0.002;
# bench/skills/inject_diagnose.py reproduces each one, docs/JJAVA.md 9
# writes it up):
#
#   D1  The reader sees RELEVANCE and not NEED. AUROC for NEEDED against OFF
#       is 0.924, against AREA (the rubric's level 1, 64% of all items) 0.631,
#       against DONE (level 2) 0.581; mean P(top level) by true level 0 / 1 /
#       2 / 3 is 0.24 / 0.62 / 0.67 / 0.73. No variant a-e separates NEEDED
#       from AREA above 0.63. On steps recall at P >= 0.5 is 0.99 and
#       precision 0.18 (base rate 0.12): "the material involves this
#       library" and "the next action needs this fact" read as the same thing.
#   D2  The Score's levels are two judgments in one (is it about the code at
#       hand / is it already done) in long multi-clause texts, and the reader
#       cannot use them: argmax lands on level 1 or 2 in 6.4% of reads (order
#       1) and 4.4% (order 2) though 68% of items ARE level 1 or 2; the two
#       orders disagree by TV 0.34 on average (argmax agrees 64%) and level 3
#       wins 83% of order-2 reads, where it is printed first, against 49% of
#       order-1 reads, where it is printed last. A Noul (a lettered yes /
#       no) is stable by comparison (variant c: TV 0.055, argmax agrees 84%)
#       -- but c's statement joined two conditions with a negation ("... and
#       the material does NOT yet show it ...") and separated worst (AUROC
#       0.554). TS jaggedness 1, 4 and "Keep each Score question to one
#       dimension": split, combine in code (docs/JJAVA.md 4.1).
#   D3  The generic process skill dominates the false yeses: on steps 215 of
#       the 343 non-needed items scoring above 0.7 (63%) belong to
#       fix-located-defect-first, which is 315 of the 978 step rows (33
#       needed, AUROC 0.562, mean belief of its AREA items 0.82). The rubric
#       calls a generic good-practice fact at most AREA unless the material
#       shows the specific mistake; the reader was not asked about the
#       mistake.
#   D4  Within a case the reader does not rank: in the 80 cases with a needed
#       item the highest-belief item is needed 30% of the time, a random item
#       31%. Absolute reads of seven items each, case by case, never compare
#       the items -- a CHOICE does (TS skill_suggestion: a choice over every
#       candidate with an explicit none, then a "fits" noul on the top 3:
#       wrong loads 16.8% -> 7.3%, docs/JJAVA.md 5.1).
#   D5  The labellers saw each fact under its craft's name ("[koota-traits-
#       and-entities] ..."), the reader the bare fact; a DO item such as
#       "pass the out argument first" does not say which library it is about.
#   D6  The session's goal helps the gate (stage 3, goal first: AUROC 0.771
#       against 0.667, paired by-case bootstrap +0.104, 95% CI 0.048-0.167)
#       and not stage 2 (variant d 0.675 against a 0.687, CI -0.022 to
#       -0.003 on the same bootstrap). Variant d repeats the opening request
#       inside itself on a first user turn; framed_state names the parts
#       once.
#   D7  Variant e's readout is invalid, not weak: the labels hold 1.4% of the
#       next-token distribution (letters: 94%; level 2 reads exactly 0.0 on 83
#       of 1,338 items), so the answer is a renormalised remainder -- the
#       digit labels do not land on the answer token in this template. Not a
#       finding about the question.
#   D8  What the label set can and cannot say: the 29 user-turn cases are 11
#       distinct states; the pagoda request alone is 16 cases (3 states),
#       287 of the 360 user items and 84 of the 93 needed ones, so every
#       user-turn number is about a handful of prompts and leave-one-run-out
#       puts the same prompt in train and test. The steps are 164 distinct
#       states (978 items). User turns: AUROC 0.58, steps 0.80.
#
# THE VARIANTS (each on the FRAMED STATE below; typed questions only: noul /
# choice, no generation; every one answers in two orders like the served
# set):
#   f  SPLIT. Two single-judgment nouls per item, combined in CODE:
#        NEXT  "The assistant is about to write or run something that this
#               fact covers."          (separates NEEDED from AREA: D1)
#        DONE  "The material already shows the assistant doing what this
#               fact says."            (separates NEEDED from DONE: D1)
#      need = P(NEXT) x (1 - P(DONE)); passes when P(NEXT) >= 0.5 and
#      P(DONE) < 0.5. 2 questions an item. [TS jaggedness 1, 4; TS score;
#      docs/JJAVA.md 4.1; DECIDER-RESEARCH 2.2: a lettered yes/no is
#      the stable readout, reflex ECE 0.104 -> 0.046 with it]
#   g  FIT. One noul per item, the rubric's own closing test as a statement
#      about THIS fact: "The assistant's next code or command would be more
#      correct with this fact in front of it." (stage 3's statement, singular
#      -- the wording whose goal-first read separated best, D6). 1 question an
#      item. It asks about the consequence (a mistake avoided), not the
#      topic: D3.
#   h  PICK THEN FIT. The two stages of TS skill_suggestion: ONE choice per
#      case over all its candidate facts with a "none" option in the middle
#      of both orders (PICK_Q), then g's noul on the PICK_TOP = 3 facts the
#      choice ranked first (the cookbook re-reads "the top 3"); a fact the
#      choice did not rank in the top 3 never goes in (belief 0). need =
#      P(FIT) x (1 - P(none)). Relative reads compare the items (D4) and the
#      none mass is the case-level "nothing here is needed". About 1 + 3
#      questions a case instead of one an item.
# THE ITEM IS SHOWN WITH ITS CRAFT'S NAME (D5), exactly what the labellers
# saw. UNMEASURED WORDING for all of them until bench/skills/inject_decide.py
# --variants f,g,h has read them; nothing here changes the served set
# (SERVED_VARIANT "a": ITEM_Q / ITEM_LEVELS above, byte for byte), and
# gate() does not call these -- adopting a measured variant also needs the
# Turn's state built with framed_state, which is skill_select's Turn
# construction, left alone until the measurement says which one.
SERVED_VARIANT = "a"
NEW_VARIANTS = ("f", "g", "h")
NEXT_Q = ("The assistant is about to write or run something that this fact "
          "covers.")
DONE_Q = ("The material already shows the assistant doing what this fact "
          "says.")
FIT_Q = ("The assistant's next code or command would be more correct with "
         "this fact in front of it.")
PICK_Q = ("Which of these facts would most change the assistant's next code "
          "or command?")
PICK_NONE = "None of these facts would change it."
NONE_KEY = "none"
PICK_TOP = 3      # TS skill_suggestion: the top 3 of the choice are re-read
PICK_MAX = 25     # a single-token letter each, A..Z, less the "none" option
QSET_PICK = "skill_pick"
GOAL_CHARS = 600  # the label set's TASK_HEAD (bench/skills/inject_labels.py)
GOAL_HEAD = "SESSION GOAL (the user's opening request):"
OPENING_HEAD = "THE USER'S OPENING REQUEST (the session goal):"
USER_HEAD = "THE USER'S NEWEST MESSAGE:"
STEP_HEAD = "THE ASSISTANT'S LATEST STEP (what it did and what came back):"


def goal_of(messages: list[dict]) -> str:
    """The session's goal: the first user message, whitespace folded, cut to
    GOAL_CHARS (the text the label set's `task` holds)."""
    import decide_turn
    um = next((m for m in messages or [] if isinstance(m, dict)
               and m.get("role") == "user"), None)
    return " ".join(decide_turn._text(um).split())[:GOAL_CHARS] if um else ""


def framed_state(goal: str, state: str, info: dict | None = None,
                 max_chars: int | None = None) -> str:
    """The state with its parts NAMED (TS state: "each part of the state has
    a descriptive name"; docs/JJAVA.md 4.2) and the session's goal first:
    the goal, then the user's newest message or the assistant's latest step.
    On a first user turn the message IS the goal and is said once (variant d
    said it twice). `info` is decide_turn.state_of's: kind, cut,
    previous_assistant. The frame never makes a state longer than the lane
    budgeted: when the state was already cut (info.cut) or `max_chars` is
    given, the excess comes off the HEAD of the state, the way the head+tail
    cut already shortens it."""
    info = info or {}
    kind = info.get("kind") or "step"
    goal = " ".join(str(goal or "").split())[:GOAL_CHARS]
    first = kind == "user" and not info.get("previous_assistant")
    head = (OPENING_HEAD if first else USER_HEAD) if kind == "user" \
        else STEP_HEAD
    pre = (f"{GOAL_HEAD}\n{goal}\n\n" if goal and not first else "") \
        + head + "\n"
    limit = max_chars if max_chars is not None else (
        len(state) if info.get("cut") else None)
    body = str(state or "")
    if limit is not None and len(pre) + len(body) > limit:
        body = body[min(len(pre) + len(body) - limit, len(body)):]
    return pre + body


def fact_of(it: dict) -> str:
    """The item's fact as the questions state it: the case's own `fact` text
    (the label set), else fact_text of a skill item."""
    return it["fact"] if it.get("fact") else fact_text(it)


def craft_of(it: dict) -> str:
    """The craft the item belongs to, by the name the labellers saw."""
    return str(it.get("name") or it.get("craft") or "").strip()


def item_block(it: dict) -> str:
    c = craft_of(it)
    return (f"CRAFT: {c}\n" if c else "") + f"FACT: {fact_of(it)}"


def q_next(it: dict) -> dict:
    import decider_bonsai as D
    return D.q_noul(f"{QSET_ITEM}_f_next:{it['key']}",
                    f"{NEXT_Q}\n\n{item_block(it)}")


def q_done(it: dict) -> dict:
    import decider_bonsai as D
    return D.q_noul(f"{QSET_ITEM}_f_done:{it['key']}",
                    f"{DONE_Q}\n\n{item_block(it)}")


def q_fit(it: dict, variant: str = "g") -> dict:
    import decider_bonsai as D
    return D.q_noul(f"{QSET_ITEM}_{variant}:{it['key']}",
                    f"{FIT_Q}\n\n{item_block(it)}")


def pick_option(it: dict) -> str:
    c = craft_of(it)
    return f"[{c}] {fact_of(it)}" if c else fact_of(it)


def q_pick(items: list[dict]) -> tuple[dict, list[dict]]:
    """(the PICK choice, the items it lists): the facts as options keyed by
    their item keys, "none" last in the list and moved to the middle of both
    printed orders (decider_bonsai.two_orders). At most PICK_MAX facts (a
    single-token letter each; the candidates are at most MAX_SKILLS x
    skill_limits.MAX_ITEMS = 18 today)."""
    import decider_bonsai as D
    use = list(items)[:PICK_MAX]
    opts = [pick_option(it) for it in use] + [PICK_NONE]
    return D.q_choice(f"{QSET_PICK}_h", PICK_Q, opts,
                      keys=[it["key"] for it in use] + [NONE_KEY],
                      none=len(opts) - 1), use


def planned_questions(variant: str, items: list[dict]) -> list[dict]:
    """The typed questions a variant asks over `items` (h: the choice and a
    fit for the first PICK_TOP items, standing in for the ones it will rank
    first -- the real count is the same, the texts differ). For a dry run
    and the cost estimate; sends nothing."""
    if variant == "f":
        return [q_next(it) for it in items] + [q_done(it) for it in items]
    if variant == "g":
        return [q_fit(it) for it in items]
    if variant == "h":
        return [q_pick(items)[0]] + [q_fit(it, "h")
                                     for it in items[:PICK_TOP]]
    raise ValueError(f"variant {variant!r} is not one of {NEW_VARIANTS}")


def _noul(a: dict) -> float:
    return float(a.get("noul") or 0.0)


def _tie(a: dict) -> bool:
    return bool((a.get("diagnostics") or {}).get("tie"))


def _part(a: dict) -> dict:
    d = a.get("diagnostics") or {}
    return {"noul": round(_noul(a), 6), "tie": _tie(a),
            "disagreement": d.get("disagreement"),
            "decision_id": a.get("decision_id")}


def item_beliefs(variant: str, items: list[dict], turn) -> tuple[list[dict],
                                                                  dict]:
    """Stage 2 for a new variant: ([{key, belief, pass, tie, disagreement,
    decision_id, probs, parts}] in `items`' order, meta). `belief` is what
    inject_tune.py tunes on; `pass` is the untuned argmax rule (every noul
    part on its side of 0.5, none tied). Raises DeciderUnavailable like
    turn.decide."""
    n = len(items)
    rows: list[dict] = []
    meta: dict = {}
    if variant == "f":
        ans = turn.decide([q_next(it) for it in items]
                          + [q_done(it) for it in items]) if items else []
        for it, a, b in zip(items, ans[:n], ans[n:]):
            nx, dn = _noul(a), _noul(b)
            tie = _tie(a) or _tie(b)
            rows.append({
                "key": it["key"], "belief": round(nx * (1.0 - dn), 6),
                "pass": bool(nx >= 0.5 and dn < 0.5 and not tie),
                "tie": tie, "probs": {"next": round(nx, 6),
                                      "done": round(dn, 6)},
                "disagreement": max(
                    (x["diagnostics"] or {}).get("disagreement") or 0.0
                    for x in (a, b)),
                "decision_id": a.get("decision_id"),
                "parts": {"next": _part(a), "done": _part(b)}})
        return rows, meta
    if variant == "g":
        ans = turn.decide([q_fit(it) for it in items]) if items else []
        for it, a in zip(items, ans):
            p = _noul(a)
            rows.append({
                "key": it["key"], "belief": round(p, 6),
                "pass": bool(p >= 0.5 and not _tie(a)), "tie": _tie(a),
                "probs": {"true": round(p, 6)},
                "disagreement": (a.get("diagnostics") or {}).get(
                    "disagreement"),
                "decision_id": a.get("decision_id"),
                "parts": {"fit": _part(a)}})
        return rows, meta
    if variant != "h":
        raise ValueError(f"variant {variant!r} is not one of {NEW_VARIANTS}")
    if not items:
        return rows, meta
    q, listed = q_pick(items)
    pk = turn.decide([q])[0]
    probs = {k: float(v) for k, v in (pk.get("probabilities") or {}).items()}
    p_none = probs.get(NONE_KEY, 0.0)
    ranked = sorted((it["key"] for it in listed),
                    key=lambda k: (-probs.get(k, 0.0), k))
    top = ranked[:PICK_TOP]
    by = {it["key"]: it for it in items}
    fits = turn.decide([q_fit(by[k], "h") for k in top]) if top else []
    fit_of = dict(zip(top, fits))
    meta = {"pick": {"none": round(p_none, 6), "top": top,
                     "probs": {k: round(v, 6) for k, v in probs.items()},
                     "tie": _tie(pk),
                     "disagreement": (pk.get("diagnostics") or {}).get(
                         "disagreement"),
                     "decision_id": pk.get("decision_id"),
                     "listed": len(listed), "of": n}}
    for it in items:
        k = it["key"]
        a = fit_of.get(k)
        if a is None:
            rows.append({"key": k, "belief": 0.0, "pass": False,
                         "tie": False, "disagreement": None,
                         "probs": {"pick": round(probs.get(k, 0.0), 6),
                                   "none": round(p_none, 6)},
                         # the case's pick: every row of a case names a
                         # decision of that case (inject_tune.loro groups
                         # the rows by it)
                         "decision_id": pk.get("decision_id"),
                         "parts": {"pick": {"p": round(probs.get(k, 0.0), 6),
                                            "in_top": False}}})
            continue
        p = _noul(a)
        rows.append({
            "key": k, "belief": round(p * (1.0 - p_none), 6),
            "pass": bool(p >= 0.5 and not _tie(a)), "tie": _tie(a),
            "probs": {"true": round(p, 6),
                      "pick": round(probs.get(k, 0.0), 6),
                      "none": round(p_none, 6)},
            "disagreement": (a.get("diagnostics") or {}).get(
                "disagreement"),
            "decision_id": a.get("decision_id"),
            "parts": {"fit": _part(a),
                      "pick": {"p": round(probs.get(k, 0.0), 6),
                               "in_top": True}}})
    return rows, meta


# --------------------------------------------------------------- compose ---
def compose(items: list[dict], chosen: list[str], profile: dict
            ) -> list[dict]:
    """The chosen items in the gate's order (it ranks them), at most the
    profile's max_items. De-duplication was done before the questions
    (candidates): nothing is asked about twice."""
    by = {it["key"]: it for it in items}
    out = [by[k] for k in chosen if k in by]
    return out[:max(int(profile.get("max_items") or DEFAULT_MAX_ITEMS), 1)]


# ---------------------------------------------------------------- render ---
# The block's head in the NOTE voice: the craft header the selector has
# always used (skill_prompts.CRAFT_HEADER), so a model reads the same frame.
# The FIRST-PERSON voice is INJECTED TEXT too, at the same place -- the
# user turn's or the tool result's tail, replayed by the ledger -- written
# in the model's own voice (the voice of the operator's heavy-handed lines,
# AGENTS.md "Heavy-handed, at the right time"); never a reasoning prefill
# (coordinator, 2026-09-29: the directive prefills are on the removal
# list). Each item keeps its words after "I'll" / "I won't" / "When ...,
# I'll". UNMEASURED WORDING until pitfall_harness.py has measured it.
FIRST_PERSON_HEAD = "What I'll hold to while I write this:"
# The prefill channel's line (voice first_person_prefill): the same items in
# the model's voice as one reasoning line ending on a letter. UNMEASURED
# WORDING; it counts only once bench/skills/prefill_check.py has shown the
# engine keeps the think block open after it.
PREFILL_HEAD = "While I write this:"
PREFILL_TAIL = "That is how I'll write it"
TABLE_HEAD = "| when | do |\n|---|---|"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").strip()


def _item_line(it: dict) -> str:
    import skill_md
    return skill_md.item_line({"form": it["form"],
                               "situation": it["situation"],
                               "text": it["text"]})


def _lower_first(t: str) -> str:
    """The first letter lowered unless the first word is a name or code
    (a capital inside it, a dot, a backtick, all caps)."""
    w = t.split(" ", 1)[0]
    if not w or "`" in w or "." in w or any(c.isupper() for c in w[1:]):
        return t
    return t[:1].lower() + t[1:]


def _first_person(it: dict) -> str:
    t = it["text"].strip()
    if it["form"] == "WHEN" and it["situation"]:
        return f"- When {it['situation']}, I'll {_lower_first(t)}"
    if it["form"] == "DO NOT":
        return f"- I won't {_lower_first(t)}"
    return f"- I'll {_lower_first(t)}"


def render(items: list[dict], profile: dict) -> dict:
    """{text, format, voice, placement, chars, tokens}: the injection in the
    profile's format and voice, appended to the end of what the model reads
    next (the user turn or the tool result). Only the items' own words
    (the first-person voice adds the pronoun before them)."""
    import skill_prompts as P
    fmt = profile.get("format") or "list"
    voice = profile.get("voice") or "note"
    if not items:
        return {"text": "", "format": fmt, "voice": voice,
                "placement": None, "chars": 0, "tokens": 0}
    if voice == "first_person":
        text = "\n---\n" + FIRST_PERSON_HEAD + "\n" + "\n".join(
            _first_person(it) for it in items)
        return {"text": text, "format": "list", "voice": voice,
                "placement": "tail", "chars": len(text),
                "tokens": L.tokens(text)}
    if voice == "first_person_prefill":
        # THE PREFILL CHANNEL (operator, 2026-09-29: keep proxy.
        # directive_prefill as a delivery channel, measured against injected
        # text): one line in the model's own voice, prefilled as the last
        # line of main's reasoning with the think block left open; it ends
        # on a letter (the prefill rule). The proxy delivers `prefill`
        # (rec["inject"]["prefill"]); nothing is appended as text.
        parts = [_first_person(it)[2:].rstrip(" .;") for it in items]
        parts = [("w" + p[1:]) if p.startswith("When ") else p
                 for p in parts]
        line = (PREFILL_HEAD + " " + "; ".join(parts) + ". "
                + PREFILL_TAIL)
        return {"text": "", "prefill": line, "format": "line",
                "voice": voice, "placement": "reasoning_prefill",
                "chars": len(line), "tokens": L.tokens(line)}
    if fmt == "table":
        rows = []
        for it in items:
            when = it["situation"] if it["form"] == "WHEN" else (
                "never" if it["form"] == "DO NOT" else "always")
            rows.append(f"| {_cell(when)} | {_cell(it['text'])} |")
        body = TABLE_HEAD + "\n" + "\n".join(rows)
    else:
        body = "\n".join(_item_line(it) for it in items)
    text = "\n---\n" + P.CRAFT_HEADER + "\n\n" + body
    return {"text": text, "format": fmt, "voice": voice,
            "placement": "tail", "chars": len(text), "tokens": L.tokens(text)}


# ----------------------------------------------------------------- entry ---
def inject(skills: list[dict], kind: str, *, turn=None, model: str | None
           = None, given: dict | None = None, events: dict | None = None,
           profile: dict | None = None, last: list | None = None,
           serving: str | None = None) -> tuple[str, dict]:
    """(text for the end of what the model reads next, the record). `skills` are stage 1's candidates in its
    order; `turn` the request's decide_turn.Turn (None: no decider -> the
    safe default, nothing); `given` the conversation's items given before
    ({key: req}); `events` {skill id: stage 1's trigger}; `last` the keys of
    the conversation's last injection. `model` is the model that READS (the
    decider's: `bonsai-a4000` while a LOCKED model holds the card,
    max_mode.decider_model) and keys THRESHOLDS; `serving` is the model that
    RECEIVES the text (the request's own: max_mode.bound_model) and keys the
    PROFILE (format, voice, size), which is a property of the reader of the
    injected text, not of the model that judged it. Where the two are not
    told apart, `serving` is the bound model of the request, else `model`."""
    t0 = time.time()
    if model is None:
        try:
            import decider_bonsai as D
            model = D.model_name()
        except Exception:                                        # noqa: BLE001
            model = ""
    if serving is None:
        try:
            import max_mode
            serving = max_mode.bound_model()
        except Exception:                                        # noqa: BLE001
            serving = None
    prof = profile or profile_for(serving or model)
    rec: dict = {"version": INJECTOR_VERSION, "questions": QUESTION_VERSION,
                 "model": model, "serving": serving or model,
                 "profile": prof.get("family"),
                 "kind": kind, "stage1": {"skills": [s["id"] for s in
                                                     skills or []]}}
    try:
        items, left = candidates(skills, given, events, last)
        rec["stage1"].update(items=len(items), left_out=left)
        if not items:
            rec.update(why="stage 1 left no item to ask about", chosen=[],
                       rendered=render([], prof),
                       ms=round((time.time() - t0) * 1000, 1))
            return "", rec
        if turn is None or not getattr(turn, "on", True):
            rec.update(why=("no decider Turn in scope" if turn is None else
                            "the decider is off in this process")
                       + ": nothing goes in (jjava decides every injection)",
                       failure={"code": "NO_DECIDER" if turn is None
                                else "DECIDER_OFF", "retryable": False,
                                "remedy": "the operator: the serving "
                                "process enables the decider (decide_turn."
                                "enable in server.py); YAMADORI_DECIDER=0 "
                                "turns skills off with it"},
                       chosen=[],
                       rendered=render([], prof),
                       ms=round((time.time() - t0) * 1000, 1))
            return "", rec
        g = gate(items, turn, model)
        rec["gate"] = g
        chosen = compose(items, g["chosen"], prof)
        out = render(chosen, prof)
        if out.get("prefill"):
            rec["prefill"] = out["prefill"]
        rec.update(chosen=[it["key"] for it in chosen], rendered={
            k: v for k, v in out.items() if k not in ("text", "prefill")},
            text_sha=hashlib.sha1((out["text"] or out.get("prefill") or "")
                                  .encode("utf-8")).hexdigest()[:16],
            why=("decider unavailable: nothing goes in (the safe default)"
                 if g.get("failure") else
                 f"{len(chosen)} item(s) from {len({it['skill'] for it in chosen})} "
                 f"skill(s)" if chosen else "the gate chose nothing"),
            ms=round((time.time() - t0) * 1000, 1))
        return out["text"], rec
    except Exception as e:                                       # noqa: BLE001
        rec.update(why=f"the injector raised {type(e).__name__}: {e}"[:300],
                   chosen=[], ms=round((time.time() - t0) * 1000, 1))
        return "", rec


def summary() -> dict:
    """For the dashboard and the record: the question sets, the profiles,
    the thresholds."""
    return {"version": INJECTOR_VERSION, "questions": QUESTION_VERSION,
            "item_q": ITEM_Q, "levels": ITEM_LEVELS, "inject_q": INJECT_Q,
            "max_skills": MAX_SKILLS, "profiles": PROFILES,
            "thresholds": THRESHOLDS or "untuned"}
