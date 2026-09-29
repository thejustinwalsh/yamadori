#!/usr/bin/env python
"""MATCHING non-package skills: embeddings retrieve, a decider chooses.

Operator, 2026-09-27: "use embeddings for matching documents and skills from
prompts, use bonsai decider for the other things" -- one tool per job. And:
the decider "is better at choosing among options than at per-item yes/no",
so its question is MULTIPLE CHOICE per legal area.

  THE TURN PLAN (plan_turn; docs/research/SKILLS-RESEARCH.md, the
  coordinator's fixes of 2026-09-27):

  PACKAGES   a package skill_packages.detect() puts in play opens ITS WHOLE
             SKILL SET (the skills filed under its area, or about it) as one
             question per package area. HARD when the package is named,
             imported, pinned, seen in code or in an error, on its first
             appearance or on an event (ASKED, ERROR, PHASE): the first
             round has no "None of these". A prose-only detection is WEAK:
             the decider confirms it. On a package's first appearance its
             LEAD skill (metadata.yamadori.lead_for, else the canonical
             mapping) goes first; a lead is never a candidate otherwise.
             After that the turn's evidence picks among the package's
             skills. A given skill is a candidate again only on an event.
  MATCHING   every armed NON-package skill is ranked by the resident
             Qwen3-Embedding (its documented query instruction,
             `Instruct: {task}
Query:{query}`; documents plain: each
             skill's trigger texts AND its body -- SkillRet App. E); rank
             only, no cosine cut; the top QUESTION_CANDIDATES (SRA
             2604.24594v3's top 50, bounded to 25 by the single-letter
             labels the decider answers reliably) across all areas
             are ONE selection question per turn (SRA's own shape; each
             decider call costs time before main). The chart removes only
             what is illegal (given). Cosines are recorded as information.
  CHOOSING   each question is multiple choice: "Which of these fits what
             the agent is doing now?", numbered options = the candidates'
             descriptions, plus "None of these" (not in a HARD question's
             first round); a pick is removed and the question asked again
             until "None of these" wins, nothing remains, or (the selection
             question) the decision holds BODIES_PER_DECISION bodies. The decider is
             mcp/decide_turn.py's `choose` when that module exists; until
             then the STUB answers with the pattern path's own picks (and
             a package's lead), so the non-package delivery is unchanged.
  DELIVERY   at most BODIES_PER_DECISION (3; SkillsBench 2602.12670v4)
             bodies per decision, every package question's first pick
             first; the rest go as index lines the model can recall.

The texts the decider reads are versioned here (MATCH_VERSION); the task
sentence is ours, a choice.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import package_registry  # noqa: E402

MATCH_VERSION = "skillmatch/1"
RETRIEVE_TASK = ("Given a developer's request or an agent's latest step, "
                 "retrieve the skills that apply")
CHOICE_QUESTION = "Which of these fits what the agent is doing now?"
NONE_OPTION = "None of these"
# Evidence that makes an area HARD: the user asked for it by name, code
# imports or pins it, an error line names it (skill_select.open_areas'
# reasons).
HARD_WHYS = frozenset({"asked", "import", "pinned", "error"})


def query_text(state: str) -> str:
    """Qwen3-Embedding's get_detailed_instruct, verbatim."""
    return f"Instruct: {RETRIEVE_TASK}\nQuery:{state}"


def stub_decider(prefer: list[str]):
    """The fallback decider: in each round, the first ALLOWED option (list
    order) whose skill the current pattern path chose, else "None of these"
    when it is allowed, else nothing. One-hot distribution."""
    want = set(prefer)

    def ask(area, options, turn):
        ex = set(turn.get("exclude") or ())
        ok = [o for o in options if o["letter"] not in ex]
        pick = next((o for o in ok if o["id"] in want), None)
        if pick is None:
            pick = next((o for o in ok if o["id"] is None), None)
        dist = {o["letter"]: (1.0 if o is pick else 0.0) for o in options}
        return (pick["letter"] if pick else None), dist
    return ask


def decider(prefer: list[str]):
    """(name, ask) -- decide_turn.choose when the module exists, else the
    stub (reproducing the pattern path's picks)."""
    try:
        import decide_turn
        fn = getattr(decide_turn, "choose", None)
        if callable(fn):
            return "bonsai", fn
    except ImportError:
        pass
    return "stub", stub_decider(prefer)


# ===========================================================================
# THE TURN PLAN (2026-09-27, docs/research/SKILLS-RESEARCH.md): packages open
# their skill sets as questions, retrieval ranks every non-package skill, a
# decider picks, and at most BODIES_PER_DECISION bodies are delivered.
# ===========================================================================
# Candidates per question: SRA (Su et al., 2604.24594v3, Table 2) gives the
# LLM its skills to select from as "the top 50" of the retriever; that
# setting beat top-1 full injection and progressive disclosure on every
# model it ran (Qwen3-32B: 62.4 vs 54.3 and 55.3).
CANDIDATES_PER_QUESTION = 50
# Bodies delivered together: SkillsBench (Li et al., 2602.12670v4): 1 skill
# +18.0, 2-3 skills +19.0, 4+ only +10.1 points (compact skills far ahead of
# comprehensive ones). 3 is its bucket boundary, not a tuned number, and
# its buckets are confounded (docs/research/SKILLS-RESEARCH.md, Part 4 #1).
# The rest go as index lines the model can recall (yama_recall_craft).
BODIES_PER_DECISION = 3
# The package area a held package's skills are filed or written under
# (skill_classify's vocabulary: skill_packages.TERM_PACKAGE, inverted, with
# the packages that share an area). Read live from the package registry
# (package_registry: its SEED is the hand-written map; an onboarded package
# adds its own area), so plan_turn sees a promoted package without a
# restart.
PACKAGE_AREA = package_registry.live("package_area")
AREA_PACKAGE = package_registry.live("area_package")
MATCH_CACHE = os.environ.get("YAMADORI_SKILL_MATCH_CACHE")


def _cache_path() -> str:
    if MATCH_CACHE:
        return MATCH_CACHE
    import skill_select
    return os.path.join(os.path.dirname(os.path.abspath(
        skill_select.TRIGGER_CACHE)), "skill_match_docs.npz")


def doc_rows(pool: list[dict]) -> list[tuple[str, str]]:
    """(skill id, text) for every document of every armed skill: its
    trigger texts (skill_select.trigger_rows: description sentences,
    when-to-use lines, title) AND its body (SkillRet App. E: indexing the
    body beside name + description adds 1.5-11.4 nDCG@10; research doc
    Part 4 #3). No model-written example queries (Tool-DE)."""
    import skill_select
    rows = list(skill_select.trigger_rows(pool))
    for s in pool:
        body = " ".join(str(s.get("body") or "").split())
        if body:
            rows.append((s["id"], body))
    return rows


_DOC: dict = {"sig": None, "rows": [], "mat": None}


def doc_index(pool: list[dict], *, build: bool = False):
    """(rows, matrix) from memory or the cache file; built (embedded plain,
    in batches) only when `build` -- the request path never embeds the
    whole store (skill_select.REQUEST_BUILD_MAX's reason). Raises when the
    index is not built for this armed set."""
    import numpy as np
    import skill_select
    rows = doc_rows(pool)
    sig = skill_select._sig(rows)
    if _DOC["sig"] == sig and _DOC["mat"] is not None:
        return _DOC["rows"], _DOC["mat"]
    path = _cache_path()
    mat = None
    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["sig"]) == sig and int(z["n"]) == len(rows):
                mat = z["mat"]
        except Exception:                                        # noqa: BLE001
            mat = None
    if mat is None:
        if not build:
            raise RuntimeError("the skill document index is not built for "
                               "this armed set (python mcp/skill_match.py "
                               "--build)")
        import code_search
        texts = [t for _sid, t in rows]
        mat = np.vstack([code_search.embed(texts[i:i + 32], is_query=False)
                         for i in range(0, len(texts), 32)])
        if float(np.linalg.norm(mat, axis=1).min()) < 0.5:
            raise RuntimeError("the embedder returned zero vectors")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, mat=mat, n=len(rows), sig=sig)
    _DOC.update(sig=sig, rows=rows, mat=mat)
    return rows, mat


def rank_all(state: str, pool: list[dict], embed=None
             ) -> tuple[dict | None, dict]:
    """({skill id: best cosine over its documents}, info) for every armed
    skill, or (None, info with why) when the stage cannot run. For ranking
    only, never a gate."""
    import numpy as np
    info = {"version": MATCH_VERSION, "task": RETRIEVE_TASK,
            "documents": "trigger texts + body"}
    if not state.strip():
        return None, dict(info, ok=False, why="no state to embed")
    try:
        rows, mat = doc_index(pool)
        if embed is None:
            import code_search
            embed = (lambda t: code_search.embed(t, is_query=False))
        q = embed([query_text(state[:4000])])[0]
        if float(np.linalg.norm(q)) < 0.5:
            return None, dict(info, ok=False, why="the query vector is zero")
        sims = mat @ q
    except Exception as e:                                       # noqa: BLE001
        return None, dict(info, ok=False,
                          why=f"{type(e).__name__}: {e}"[:200])
    best: dict = {}
    for (sid, _t), c in zip(rows, sims):
        best[sid] = max(best.get(sid, -1.0), float(c))
    return best, dict(info, ok=True, documents_n=len(rows))


# THE LABELS: each ONE token of the decider's tokenizer, both where the
# option is listed ("\nA) ...") and where it is answered ("Answer: A").
# Checked on the served Bonsai 2 (llama-swap `bonsai`, /tokenize,
# 2026-09-27; `python mcp/skill_match.py --check-labels` re-checks): numbers
# split per digit ("17" is "1","7"); A-Z are one token each.
#
# SINGLE LETTERS ONLY (deploy check 2026-09-27, h5). The doubles AA..ZZ are
# one token each too, but the model does not ANSWER them: it writes the
# double's single letter. Measured on the served model (the decider's own
# rendering, one prompt each, top_logprobs at the answer position; replay
# of the live records in x_yamadori.skills.match): "Reply with exactly: ok"
# against 50 retrieved skills, "None of these" labelled ZZ -> " Z" 0.371
# (an unrelated skill), " ZZ" 0.037; the same prompt and the first 25 of
# those skills, "None of these" labelled Z -> " Z" 0.624 (none). With 40
# skills ("None" = OO) -> " O" 0.217, OO not in the top 10; with 49
# ("None" = YY) -> " YY" 0.259 and " Y" 0.130. Live, every request at medium
# and up injected three unrelated bodies this way (the trivial "ok" answer
# went on to write one of them out; every cache check's tail grew by them).
# So a question holds at most len(LABELS) - 1 candidates plus "None of
# these" -- 25, the bound the labels set (QUESTION_CANDIDATES), under SRA's
# 50 (CANDIDATES_PER_QUESTION).
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# The candidates one question can carry: SRA's 50, bounded by the labels
# the decider answers reliably (one is "None of these").
QUESTION_CANDIDATES = min(CANDIDATES_PER_QUESTION, len(LABELS) - 1)


def _letter(i: int) -> str:
    return LABELS[i]


def _round_options(cands: list[dict], none: bool) -> list[dict]:
    opts = [{"letter": _letter(i), "id": c["id"], "name": c["name"],
             "text": c["description"]} for i, c in enumerate(cands)]
    if none:
        opts.append({"letter": _letter(len(opts)), "id": None, "name": None,
                     "text": NONE_OPTION})
    return opts


def render_options(opts: list[dict]) -> str:
    """The shared PREFIX of every round of a question: the options, one per
    line, label then description. Identical across the turn's rounds, so
    the decider's cached prefix is reused."""
    return "Options:\n" + "\n".join(f"{o['letter']}) {o['text']}"
                                     for o in opts)


def render_question(opts: list[dict], chosen: list[str], hard_first: bool
                    ) -> str:
    """The round's QUESTION line, after the options: the question, then the
    round's suffix (what is already chosen, or that "none" is not an answer
    under hard evidence's first round)."""
    none = next((o["letter"] for o in opts if o["id"] is None), None)
    q = CHOICE_QUESTION
    if chosen:
        q += (f" Already chosen: {', '.join(chosen)}; choose another, or "
              f"{none} (none of these).")
    elif hard_first and none:
        q += f" Choose one of the skills; {none} is not an answer this time."
    return q


def _tie_band() -> float:
    """The decider's own tie band (decider_bonsai.TIE_BAND, measured by the
    decider agent), else 0: only an exact tie."""
    try:
        import decider_bonsai
        return float(getattr(decider_bonsai, "TIE_BAND", 0.0))
    except Exception:                                            # noqa: BLE001
        return 0.0


def _tokens(text: str) -> int:
    import skill_limits as L
    return L.tokens(text)


def ask_rounds(area: str, cands: list[dict], turn: dict, hard: bool,
               ask, max_picks: int | None = None) -> dict:
    """The multiple-choice rounds of one question over ONE option list,
    identical in every round (the candidates, then "None of these"): the
    options come first and the question last, so every round after the
    first reuses the decider's cached prefix and processes only its
    question line. A pick is NOT removed: the next round's question says
    what is already chosen, and the answer is read with the chosen labels
    (and, under HARD evidence's first round, "None of these") EXCLUDED and
    the rest renormalised. Rounds stop when "None of these" wins, the
    decider picks nothing, every candidate is chosen, or `max_picks` is
    reached."""
    opts = _round_options(cands, none=True)
    prefix = render_options(opts)
    p_tok = _tokens(prefix + "\n\n")
    picks: list[str] = []
    chosen: list[str] = []
    rounds: list[dict] = []
    while len(picks) < len(cands) and (max_picks is None
                                       or len(picks) < max_picks):
        first = not rounds
        exclude = list(chosen)
        none = opts[-1]["letter"]
        if hard and first:
            exclude.append(none)
        question = render_question(opts, chosen, hard and first)
        letter, dist = ask(area, [dict(o) for o in opts],
                           dict(turn, question=question, hard=hard,
                                exclude=exclude,
                                prompt={"prefix": prefix,
                                        "question": question}))
        allowed = [o for o in opts if o["letter"] not in exclude]
        dist = dict(dist or {})
        mass = sum(float(dist.get(o["letter"], 0.0)) for o in allowed)
        norm = ({o["letter"]: round(float(dist.get(o["letter"], 0.0)) / mass,
                                    4) for o in allowed} if mass > 0 else {})
        tie = None
        if norm and letter is None:
            # The decider answered no label: a TIE within its own measured
            # band (decide_turn / decider_bonsai.TIE_BAND). The embedder's
            # retrieval rank breaks it: the option list is in retrieval
            # order, so the first tied option wins (structural).
            band = _tie_band()
            top = max(norm.values())
            tied = [o["letter"] for o in allowed
                    if norm[o["letter"]] >= top - band]
            letter = tied[0] if tied else None
            tie = {"tied": tied, "band": band,
                   "broken_by": "retrieval rank (option order)"}
        elif norm and letter not in norm:
            # A label that is excluded (already chosen): the rest,
            # renormalised, decide.
            letter = max(norm, key=norm.get)
        pick = next((o for o in allowed if o["letter"] == letter), None)
        q_tok = _tokens(question)
        rounds.append({"question": question,
                       "excluded": exclude, "tie": tie,
                       "distribution": norm or dist,
                       "pick": (pick or {}).get("name") or (
                           NONE_OPTION if pick else None),
                       "tokens": {"processed": (p_tok if first else 0)
                                  + q_tok,
                                  "reused": 0 if first else p_tok}})
        if pick is None or pick["id"] is None:
            break
        picks.append(pick["id"])
        chosen.append(pick["letter"])
    return {"picks": picks, "rounds": rounds,
            "options": [f"{o['letter']}) {o['name'] or NONE_OPTION}"
                        for o in opts],
            "tokens": {"prefix": p_tok, "estimate": "skill_limits.tokens "
                       "(3 characters a token, high on purpose)"}}


def _cand(s: dict, cos) -> dict:
    return {"id": s["id"], "name": s["name"],
            "cos": None if cos is None or s["id"] not in cos
            else round(cos[s["id"]], 4),
            "description": s.get("description") or s.get("title") or ""}


def knn_skill_ids(knn: dict | None, pool: list[dict], area_of,
                  subject_areas) -> list[str]:
    """The armed skills the kNN neighbours' chunks LABEL (docs/PACKAGE-
    ONBOARDING.md 6.2: a chunk's skills are computed at load, never stored):
    a skill of a voted package's area whose verified code-shaped topics are
    among the neighbours' identifiers. Ordered by the neighbours' rank."""
    if not knn or not knn.get("ok"):
        return []
    import example_knn
    out: list[str] = []
    for g in knn.get("groups") or []:
        # example_knn.derived_skills: the one labelling rule, shared with
        # the evaluation (a PROXY for "this skill would have helped").
        for sid in example_knn.derived_skills(g.get("packages") or [],
                                              g.get("identifiers") or [],
                                              pool):
            if sid not in out:
                out.append(sid)
    return out


def plan_turn(*, pool: list[dict], old: list[dict], pkg_play: dict,
              pkg_state, given: set, cos, legal_whys: dict,
              phase_changed: bool, leads_for, area_of, subject_areas,
              knn: dict | None = None):
    """(the turn's ordered picks [{skill, form, source, area, ...}], the
    record, the new package state).

    `old`: the pattern path's candidates this turn ([{skill, form, trigger,
    ...}], the chart already applied) -- the STUB decider's answers and,
    for non-package skills, what is delivered until the Bonsai decider
    lands. `pkg_play`: skill_packages.detect(). `cos`: rank_all's cosines,
    or None when the embedding stage is off. `leads_for(pkg)`: the
    package's lead skill name (its lead_for skill, else the canonical
    mapping)."""
    pst = dict(pkg_state or {})
    pst["seen"] = dict(pst.get("seen") or {})
    by_name = {s["name"]: s for s in pool}
    old_ids = [c["skill"]["id"] for c in old]
    old_by = {c["skill"]["id"]: c for c in old}
    old_recalls = {c["skill"]["id"] for c in old if c.get("form") == "recall"}
    dname, dask = decider(old_ids)
    rec: dict = {"decider": dname, "questions": [],
                 "bounds": {"candidates_per_question":
                            CANDIDATES_PER_QUESTION,
                            "source": "SRA 2604.24594v3 (LLM selection "
                                      "from the top 50)",
                            "question_candidates": QUESTION_CANDIDATES,
                            "labels": "A-Z only: the decider answers a "
                                      "double (ZZ) with its single letter "
                                      "(deploy check 2026-09-27)"}}

    # A LEAD skill (metadata.yamadori.lead_for) is pushed ONLY by its
    # package's question, on the package's first appearance: never a
    # candidate of the general matcher (a broad overview skill captures
    # everything -- Scaling Laws of Skills 2605.16508's "black-hole"
    # skills), nor of its package's later questions.
    def is_lead(s):
        return bool(s.get("lead_for"))

    def is_package_skill(s):
        return area_of(s.get("rule") or {}) in AREA_PACKAGE or bool(
            set(subject_areas(s)) & set(AREA_PACKAGE))

    # --- the packages in play, by area -----------------------------------
    areas: dict = {}
    for pkg, e in pkg_play.items():
        a = PACKAGE_AREA.get(pkg)
        if not a:
            continue
        g = areas.setdefault(a, {"packages": [], "strong": False,
                                 "events": set(), "why": []})
        g["packages"].append(pkg)
        g["strong"] = g["strong"] or e.get("strength", "strong") == "strong"
        g["events"] |= set(e.get("events") or [])
        g["why"] += [f"{pkg}: {w['how']} {w['what']}"
                     for w in e["why"]][:3]
    for c in old:
        # A pattern-path pick of a package skill with no detection this
        # turn (a recall on an error, a "still broken" turn): WEAK evidence
        # of its package, for the decider to confirm.
        a = area_of(c["skill"].get("rule") or {})
        if a in AREA_PACKAGE and a not in areas:
            areas[a] = {"packages": [AREA_PACKAGE[a]], "strong": False,
                        "events": set(),
                        "why": [f"pattern: {c['skill']['name']}"]}
    # THE EXAMPLE kNN VOTE (docs/PACKAGE-ONBOARDING.md 6.5): the package at
    # the TOP of the neighbours' tally (all of them on a tie) that detect()
    # did not put in play opens as a WEAK area, for the decider to confirm
    # ("None of these" offered). Rank only, no cosine cut. With the STUB
    # decider a kNN-opened area delivers nothing -- no lead is pushed on a
    # vote alone -- so live delivery is unchanged until a model decider
    # replaces the stub; the record shows what the vote proposed.
    knn_rec = None
    knn_areas: set = set()
    if knn is not None:
        knn_rec = {k: knn.get(k) for k in ("index", "k", "tally", "ok",
                                            "why")}
        knn_rec["groups"] = [{"rank": g.get("rank"),
                              "packages": g.get("packages")}
                             for g in knn.get("groups") or []]
        knn_rec["opened_weak"] = []
        if knn.get("ok"):
            k_ = knn.get("k")
            for pkg in knn.get("top") or []:
                if pkg in pkg_play:
                    continue
                a = PACKAGE_AREA.get(pkg)
                if not a or a in areas:
                    continue
                m = (knn.get("tally") or {}).get(pkg)
                areas[a] = {"packages": [pkg], "strong": False,
                            "events": set(), "knn": True,
                            "why": [f"knn: {m} of {k_} neighbours"]}
                knn_areas.add(a)
                knn_rec["opened_weak"].append(pkg)
    knn_ids = knn_skill_ids(knn, pool, area_of, subject_areas)
    if knn_rec is not None:
        knn_rec["labelled"] = [s["name"] for s in pool if s["id"] in knn_ids]
    asked_first = sorted(areas, key=lambda x: (not any(
        "ASKED" in ((pkg_play.get(p) or {}).get("events") or [])
        for p in areas[x]["packages"]), x))
    picks: list[dict] = []
    taken: set = set()
    for a in asked_first:
        g = areas[a]
        first = [p for p in g["packages"] if p not in pst["seen"]]
        eventful = bool(g["events"]) or phase_changed
        hard = g["strong"] and (bool(first) or eventful)
        # A vote alone pushes no lead: its package's first appearance waits
        # for evidence detect() reads.
        leads = [] if g.get("knn") else [
            n for n in (leads_for(p) for p in first) if n]
        lead_ids = [by_name[n]["id"] for n in leads if n in by_name]
        pool_a = [s for s in pool if area_of(s.get("rule") or {}) == a
                  or a in subject_areas(s)]
        # A skill about two packages (koota-with-react-three-fiber) is a
        # candidate of the first question that reaches it, once per turn.
        # A given skill is a candidate again only on an event: its
        # package's (ASKED, ERROR, PHASE), or the chart's own judgement that
        # made the pattern path offer it as a RECALL this turn.
        cands = [s for s in pool_a if (s["id"] not in given or eventful
                                       or s["id"] in old_recalls)
                 and s["id"] not in taken
                 and (not is_lead(s) or s["id"] in lead_ids)]

        def rank_key(s, lead_ids=lead_ids):
            if s["id"] in lead_ids:
                return (0, lead_ids.index(s["id"]), 0.0, s["name"])
            # A skill the kNN neighbours' chunks label ranks after the lead
            # and before the cosine order (docs/PACKAGE-ONBOARDING.md 6.5).
            if s["id"] in knn_ids:
                return (1, knn_ids.index(s["id"]), 0.0, s["name"])
            if cos is not None:
                return (2, 0, -cos.get(s["id"], -1.0), s["name"])
            return (2, old_ids.index(s["id"]) if s["id"] in old_ids
                    else len(old_ids), 0.0, s["name"])
        cands.sort(key=rank_key)
        cands = cands[:QUESTION_CANDIDATES]
        in_c = {s["id"] for s in cands}
        # The stub's answers: the lead, then the pattern path's picks in
        # the embedder's order when it ran (the turn's evidence decides
        # which of the package's skills comes first), else the pattern
        # path's own order.
        mine = [i for i in old_ids if i in in_c and i not in lead_ids]
        if cos is not None:
            mine.sort(key=lambda i: -cos.get(i, -1.0))
        prefer = lead_ids + mine
        q_ask = dask if dname != "stub" else stub_decider(prefer)
        if dname == "stub":
            order = {i: k for k, i in enumerate(prefer)}
            shown = sorted(cands, key=lambda s: (order.get(s["id"],
                                                           len(order)),
                                                 rank_key(s)))
        else:
            shown = cands
        room = max(0, BODIES_PER_DECISION - sum(
            1 for p in picks if p["form"] == "body"))
        if room:
            res = ask_rounds(a, [_cand(s, cos) for s in shown],
                             {"kind": "package"}, hard, q_ask,
                             max_picks=room)
        else:
            # No room: nothing is asked (the cap is SkillsBench's, for the
            # whole decision); the package's lead on its first appearance
            # still goes out, as an index line.
            res = {"picks": [], "rounds": [], "options": [],
                   "tokens": {"prefix": 0}}
            for i in lead_ids:
                if i in {s["id"] for s in cands}:
                    picks.append({"skill": by_name[next(
                        n for n in leads if by_name.get(n, {}).get("id")
                        == i)], "form": "index", "round": 0,
                        "source": "package", "package": AREA_PACKAGE[a],
                        "events": sorted(g["events"]), "area": a,
                        "trigger": "package", "why": g["why"][:4]})
        rec["questions"].append({
            "kind": "package", "area": a, "packages": g["packages"],
            "hard": hard, "evidence": g["why"][:6],
            "strength": "strong" if g["strong"] else "weak",
            "first_appearance": first, "leads": leads, "room": room,
            "candidates": [{"name": s["name"], "rank": k + 1,
                            "cos": _cand(s, cos)["cos"]}
                           for k, s in enumerate(cands)], **res})
        for p in g["packages"]:
            if not g.get("knn"):
                # A vote is not the package's first appearance.
                pst["seen"].setdefault(p, True)
        by_id = {s["id"]: s for s in cands}
        taken |= {s["id"] for s in cands}
        for rnd, i in enumerate(res["picks"]):
            oc = old_by.get(i)
            form = (oc or {}).get("form") or ("recall" if i in given
                                               else "body")
            picks.append({"skill": by_id[i], "form": form, "round": rnd,
                          "source": "package", "package": AREA_PACKAGE[a],
                          "events": sorted(g["events"]), "area": a,
                          "trigger": (oc or {}).get("trigger") or "package",
                          "why": g["why"][:4]})

    # Delivery order across packages: every question's FIRST pick (the
    # asked areas first, then the others: a lead on its first appearance),
    # then the second picks, and so on (research doc Part 4 #1: the asked
    # areas first, then the implied).
    picks.sort(key=lambda p: p["round"])

    # --- non-package skills: retrieval over ALL of them -------------------
    rest = [s for s in pool if not is_package_skill(s) and not is_lead(s)]
    retrieved: set = set()
    if cos is not None:
        ranked = sorted((s for s in rest if s["id"] not in given
                         and s["id"] in cos), key=lambda s: -cos[s["id"]])
        top = ranked[:QUESTION_CANDIDATES]
        retrieved = {s["id"] for s in top}
        # ONE selection question per turn over the retrieved top
        # candidates across all areas -- SRA 2604.24594v3's shape (the LLM
        # selects from the retriever's top 50) -- soft ("None of these"
        # always offered), repeated with the pick removed until "None of
        # these" wins or the decision holds BODIES_PER_DECISION bodies
        # (SkillsBench 2602.12670v4). Package questions stay per package.
        cands_all = [dict(_cand(s, cos), rank=k + 1,
                          area=area_of(s.get("rule") or {}))
                     for k, s in enumerate(top)]
        room = max(0, BODIES_PER_DECISION - sum(
            1 for p in picks if p["form"] == "body"))
        old_rest = [i for i in old_ids if i in retrieved]
        q_ask = dask if dname != "stub" else stub_decider(old_rest)
        res = ask_rounds("any", cands_all, {"kind": "match"}, False, q_ask,
                         max_picks=room)
        rec["questions"].append({
            "kind": "match", "area": "any", "hard": False,
            "room": room,
            "candidates": [{k2: c[k2] for k2 in ("name", "rank", "cos",
                                                  "area")}
                           for c in cands_all], **res})
        if dname != "stub":
            by_id = {s["id"]: s for s in rest}
            for i in res["picks"]:
                s_ = by_id[i]
                picks.append({"skill": s_, "form": "body",
                              "source": "match",
                              "area": area_of(s_.get("rule") or {}),
                              "trigger": "match",
                              "why": [f"chosen by the {dname} decider"]})
    if dname == "stub":
        # The stub decider: the pattern path's own non-package picks.
        for c in old:
            s = c["skill"]
            if is_package_skill(s):
                continue
            picks.append({"skill": s, "form": c["form"], "source": "pattern",
                          "area": area_of(s.get("rule") or {}),
                          "trigger": c["trigger"], "slot": c.get("slot"),
                          "question": c.get("question"),
                          "why": c.get("why") or [],
                          "retrieved": (s["id"] in retrieved)
                          if cos is not None else None})
    # --- delivery: at most BODIES_PER_DECISION bodies, the rest indexed ---
    out: list[dict] = []
    nb = 0
    for p in picks:
        if p["form"] == "body":
            if nb >= BODIES_PER_DECISION:
                p = dict(p, form="index")
            else:
                nb += 1
        out.append(p)
    rec["delivery"] = {"bodies_max": BODIES_PER_DECISION,
                       "source": "SkillsBench 2602.12670v4",
                       "bodies": [p["skill"]["name"] for p in out
                                  if p["form"] == "body"],
                       "indexed": [p["skill"]["name"] for p in out
                                   if p["form"] == "index"],
                       "recalls": [p["skill"]["name"] for p in out
                                   if p["form"] == "recall"]}
    if cos is not None:
        rec["pattern_not_retrieved"] = [
            c["skill"]["name"] for c in old
            if not is_package_skill(c["skill"])
            and c["skill"]["id"] not in retrieved]
    if knn_rec is not None:
        # x_yamadori.skills.knn: group ranks and packages, the tally, what
        # opened -- never example text.
        rec["knn"] = knn_rec
    return out, rec, pst


# ---------------------------------------------------------------------------
# UPKEEP: the index is never missing live. The worker checks it every minute
# (worker.py's loop, beside the watch and learn schedules) and, when it does
# not match the armed set -- a skill armed, archived, disabled, superseded,
# edited, or a trigger learned -- enqueues ONE rebuild on the gpu lane (the
# embedder is on the A4000; the lane runs one job at a time), IDLE-GATED
# since 2026-09-27 (operator decision 5, docs/PACKAGE-ONBOARDING.md 3.2):
# it runs only when mcp/idle.py says the stack is idle, so a newly armed
# skill is out of retrieval until then. The request
# path keeps its behaviour when the index is stale or missing (rank_all
# says so; the stub keeps the delivery). scripts/deploy_check.py rebuilds
# it on the idle card after a restart and fails the deploy if it is still
# stale.
# ---------------------------------------------------------------------------
QUEUE = "skill.match_index"
LANE = "gpu"


def index_state(pool: list[dict] | None = None) -> dict:
    """{fresh, documents, path, why} for the armed set now."""
    import numpy as np
    import skill_select
    import skills
    pool = skills.armed() if pool is None else pool
    rows = doc_rows(pool)
    sig = skill_select._sig(rows)
    path = _cache_path()
    out = {"fresh": False, "documents": len(rows), "path": path,
           "skills": len(pool)}
    if not os.path.exists(path):
        return dict(out, why="no index file")
    try:
        z = np.load(path, allow_pickle=False)
        have, n = str(z["sig"]), int(z["n"])
    except Exception as e:                                       # noqa: BLE001
        return dict(out, why=f"unreadable: {type(e).__name__}")
    if have != sig or n != len(rows):
        return dict(out, why=f"stale: built for {n} documents, the armed "
                             f"set has {len(rows)} (or the texts changed)")
    return dict(out, fresh=True, why="matches the armed set")


def schedule() -> str | None:
    """Enqueue one rebuild when the index is stale and none is queued or
    running. The worker calls this every minute."""
    import jobs
    if index_state()["fresh"]:
        return None
    for st in ("queued", "running"):
        if any(j.get("queue") == QUEUE for j in jobs.listing(state=st,
                                                              limit=500)):
            return None
    # Idle-gated (operator, 2026-09-27, decision 5: "The skill doc-index
    # rebuild is idle-gated (one GPU consumer at a time)"): worker.run_one
    # defers it while the stack is busy (mcp/idle.py), without an attempt.
    return jobs.add(QUEUE, {"idle": True}, lane=LANE, stage="index")


def handle_build(job: dict, ctx=None) -> dict:
    """The gpu-lane job: embed the armed set's documents (trigger texts +
    body) and write the cache."""
    import skills
    _DOC.update(sig=None, rows=[], mat=None)
    rows, mat = doc_index(skills.armed(), build=True)
    return {"documents": len(rows), "dim": int(mat.shape[1]),
            "path": _cache_path()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="embed every armed skill's documents (trigger "
                         "texts + body) into the cache: the A4000 embedder")
    ap.add_argument("--check", action="store_true",
                    help="is the index fresh for the armed set? (exit 1 if "
                         "not)")
    ap.add_argument("--check-labels", action="store_true",
                    help="each label one token on the served decider "
                         "(llama-swap <--model> /tokenize)")
    ap.add_argument("--model", default=os.environ.get("YAMADORI_MODEL", "bonsai"),
                    help="the main model whose tokenizer is checked (max mode: run it for flash-next too)")
    a = ap.parse_args()
    if a.check_labels:
        import json
        import urllib.request
        import max_mode
        if max_mode.blocks(a.model):
            print(f"not run: {max_mode.blocked_reason(a.model)}")
            sys.exit(3)

        def tok(text):
            req = urllib.request.Request(
                f"http://127.0.0.1:11434/upstream/{a.model}/tokenize",
                data=json.dumps({"content": text,
                                 "with_pieces": True}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return [t["piece"] for t in json.loads(r.read())["tokens"]]
        bad = []
        for lab in LABELS:
            listed = [x for x in tok(f"\n{lab}) x")
                      if x not in ("\n", ")", " x")]
            answered = tok(f"Answer: {lab}")[2:]
            if len(listed) != 1 or len(answered) != 1:
                bad.append((lab, listed, answered))
        print(f"{len(LABELS)} labels; not one token: {bad or 'none'}")
        sys.exit(1 if bad else 0)
    if a.check:
        st = index_state()
        print(st)
        sys.exit(0 if st["fresh"] else 1)
    if a.build:
        import skills
        rows, mat = doc_index(skills.armed(), build=True)
        print(f"{len(rows)} documents, dim {mat.shape[1]} -> {_cache_path()}")
