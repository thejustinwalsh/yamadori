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
           profile: dict | None = None, last: list | None = None
           ) -> tuple[str, dict]:
    """(text for the end of what the model reads next, the record). `skills` are stage 1's candidates in its
    order; `turn` the request's decide_turn.Turn (None: no decider -> the
    safe default, nothing); `given` the conversation's items given before
    ({key: req}); `events` {skill id: stage 1's trigger}; `last` the keys of
    the conversation's last injection."""
    t0 = time.time()
    if model is None:
        try:
            import decider_bonsai as D
            model = D.model_name()
        except Exception:                                        # noqa: BLE001
            model = ""
    prof = profile or profile_for(model)
    rec: dict = {"version": INJECTOR_VERSION, "questions": QUESTION_VERSION,
                 "model": model, "profile": prof.get("family"),
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
