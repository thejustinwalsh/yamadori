#!/usr/bin/env python
"""PROVE: a paired A/B, per skill, that the skill does not make a task worse.

Operator, 2026-09-28: "The research told us each skill needs proof that it
works, right? We skipped that in our injection loop; we need to derive a mini
A/B test to ensure it doesn't make a task worse, generated in the skill
pipeline."

WHY (docs/research/SKILLS-RESEARCH.md, Part 1 finding 2): SkillsBench
2602.12670 -- self-generated skills cost -8.1 to -11.5 pp unless verified;
ASI 2504.06821 -- a skill admitted only after RE-EXECUTION passes, and
verification alone added +4.2. The faithfulness check in validate is a model
reading items beside quotes (skill_pipeline.FAITHFUL_JUDGE says it is weaker
than an execution-based check); this stage is the execution-based half.

WHERE IT RUNS

    ... -> tests -> validate -> PROVE -> arm        (the SKILL.md text is final)

  pipeline   a version running at stage `prove` (skill_pipeline._target). A
             pass returns and the worker's advance enqueues `arm`; WORSE
             quarantines the version (skills.quarantine, the evidence in its
             reason and its record).
  inline     a migration or authored install (skill_pipeline._INLINE, no
             model): recorded "not run: inline install"; it does not block
             arming. The backlog proves it later.
  backlog    an ARMED library skill not yet proved at its served version
             (`enqueue_backlog`, `schedule_backlog`): idle-gated gpu-lane jobs
             (payload `idle`, worker.run_one defers them while the stack is
             busy, mcp/idle.py). WORSE quarantines it -- which disarms it --
             anything else only records.

WHAT A PROOF IS

  1. PROBES. Up to PROBES (3) tasks from the skill's OWN should-cases
     (tests.activation.should) -- the operator's "2-3". A case that shares
     an 8-word run with an item is skipped (it would leak the answer; 8 is
     the run test_skill_factory already gates authored skills on). The
     `probe` template (PROBE_VERSION) may PROPOSE a concrete small
     code-writing task per case and checks; CODE verifies every proposal
     (`verify_proposals`): a regex compiles, does not match the empty text,
     matches the skill's own items or topics and carries a literal of at
     least 3 characters from them; a task must not share an 8-word run with
     an item or its quote. A refused proposal is recorded with why.
  2. CHECKS, per probe, by code:
       parse   every fenced code block of the answer parses
               (code_check.syntax_errors -- tool_code's parsers)
       types   the injectable TYPE_CHECKER against the held package's types;
               absent (offline, or unconfigured) it is "not run", never a pass
       present a DO / WHEN item's code span or topic is in the answer's code
       absent  a DO NOT item's code span or topic is not
     A present-check whose literal the probe itself names is dropped (it
     would measure echo, not the skill); a literal both a DO and a DO NOT
     item name is dropped (the checks would contradict). When NO check can
     decide a probe on both sides, one model JUDGE check (JUDGE_VERSION)
     reads each answer blind -- the last resort, labelled `judge: model`.
  3. THE PAIR. The served model (mcp/model.py, the one door; the shape
     skill_pipeline.shaped_body gives a pipeline job: role helper, effort
     medium, the job's thinking cap, job `skill.prove`) answers each probe
     WITHOUT and WITH the skill, the skill injected exactly as the selector
     injects it (skill_select._render_turn: the body under the craft header,
     appended to the user turn by message_text.append_text), with the same
     `seed` and the same sampling and budget fields on both bodies.
  4. THE DECISION, paired per probe and per check: a check that passed
     WITHOUT and fails WITH is FLAGGED. A flag is one sample, so the probe
     is run again (THE REPEAT RULE, below) and the check is WORSE only if
     the worse result repeats. Any worse -> quarantine. Otherwise arm: a
     check that failed without and passed with is `better`; none either way
     is a tie, recorded "no measurable gain". No threshold (operator: "do
     not invent a threshold"). A side that ended on the token limit is a
     budget event, never an answer (model.BudgetEvent): that probe is
     `undecided`.

  reprove    a version PROVE quarantined as worse under an earlier rule (a
             record without `rule`): `reprove_quarantined` enqueues it as
             an idle-gated gpu-lane job (payload `reprove`); the proof runs
             again under the repeat rule, and only a CONFIRMED worse keeps
             it quarantined -- anything else serves it again
             (skills.rearm), `unproven` leaves it quarantined.

THE RECORD: `validate["prove"]` until skill_versions has its own `prove`
column (skills._ADDED / _JSON_V); `record_of` reads either. Per probe: the
probe text's sha256, where it came from, the language, the seed, each check
with its pass/fail without and with, and the seconds of each side.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402
import skill_prompts  # noqa: E402
import skills  # noqa: E402

# The queue this stage's jobs take. skills.JOBS["prove"] is the wiring; this
# is the same pair, for the backlog before the wiring lands.
JOB = ("skill.prove", "gpu")
# The operator's figure (2026-09-28, "derive 2-3 probe tasks").
PROBES = 3
MIN_PROBES = 2
# A probe that shares a run of this many words with one of the skill's items
# quotes the answer. The same run length test_skill_factory already gates on
# ("no authored skill shares an 8-word run with the Octopus task spec").
LEAK_WORDS = 8
# The probe's answer allowance: tiers.A_MIN, the stack's answer floor (2,048;
# thinking is added by tiers.budget) -- the same allowance the distil stage
# uses (skill_pipeline.DISTIL_MAX_TOKENS).
ANSWER_TOKENS = None      # None: tiers.A_MIN, read at call time
# The model half (the probe proposals), like the pipeline's other model
# halves: on unless switched off; the skip is recorded.
MODEL_PROBE = os.environ.get("YAMADORI_SKILL_MODEL_PROBE", "1") != "0"
# The library backlog: on unless switched off (the worker's schedule).
BACKLOG = os.environ.get("YAMADORI_SKILL_PROVE_BACKLOG", "1") != "0"

# Verdicts that count as PROVED at a version: the backlog skips them.
# `not_run` (an inline install, no probe) is not proved.
# `unproven` (operator, 2026-09-28): no check decided on both sides even
# after the one retry -- the version does NOT arm. `undecided` is the same
# outcome in records written before that rule.
PROVED = ("worse", "better", "tie", "unproven", "undecided")
PAIR_WORSE, PAIR_BETTER, PAIR_SAME, PAIR_NOT_RUN = (
    "worse", "better", "same", "not_run")

# THE REPEAT RULE (operator, 2026-10-06: "fix the quarantine rule"; the repo's
# own rule, docs/PROTOCOL.md: "If a measurement does not survive a repeat, it
# is not a result"). Until 2026-10-06 ONE pair per probe decided. Measured on
# the local library (index/jobs.sqlite3, read 2026-10-06, n=114 quarantined
# skills): 107 were quarantined by a proof of that kind, all backlog proofs,
# 100 of them on 2 probes, 67 of the worse checks a `parse` check -- a
# syntax error in one WITH sample.
#
# THE RULE. A check that passed WITHOUT and failed WITH (a FLAG) is run again:
# its probe's pair is generated REPEATS more times, each time on a NEW seed
# shared by both sides (the same seed on the same prompt gives the same text,
# so a repeat on the old seed repeats nothing). The check is WORSE only if
# WITH is worse than WITHOUT on it in a strict majority (more than half) of
# its runs: the flagging run plus the repeats. A repeat that ends on the
# token limit, or that no check decided, is not a worse result. Only a probe
# with a flag is repeated.
#
# WHY REPEATS = 1 (derived; not measured). A generation fails a check with
# some probability q even when the skill changes nothing, and the two sides
# are independent draws (the WITH prompt differs). So a skill that changes
# nothing is FLAGGED on a check with probability w = q(1-q) <= 1/4 (q = 1/2),
# and a skill has probes x checks of them. REPEATS = 0 is the old rule: it
# quarantines such a skill with probability w. With REPEATS = 1 there are two
# runs and a majority of two is both, so it is quarantined with probability
# w^2 (<= 1/16: at least 4x fewer false quarantines, and more at smaller w);
# a tie, one worse and one not, quarantines nothing. REPEATS = 1 is the
# smallest count that is a repeat at all. Two repeats (3 runs, worse in 2 of
# 3) would be WEAKER: the flagging run is selected for being worse, so it
# counts as a free vote and either repeat confirms: 2w^2 - w^3 > w^2. Three
# repeats (4 runs, worse in 3) is stricter, 3w^3 - 2w^4, at three times the
# generations; the operator may raise REPEATS. The price of any repeat is
# recall: a skill that is really worse is confirmed with probability about
# (1 - q0) q1 per repeat, where q0 and q1 are its failure rates without and
# with it, so one that is worse only some of the time may arm. The repeat's
# own sample count (n=1 per flag) is the evidence; no threshold is added.
REPEATS = 1
# Stamped on every record this rule decided: the re-prove path
# (`reprove_backlog`) selects quarantines whose record lacks it.
RULE = "repeat/1"

# ---------------------------------------------------------------------------
# Injectables. Tests replace them; the defaults are the stack's.
# ---------------------------------------------------------------------------
# CHAT(body) -> the model server's response dict. Default: model.post.
CHAT = None
# ASK(system, user, *, max_tokens, purpose) -> the reply text. Default:
# skill_pipeline.ask_model (the one door, the helper job skill.<purpose>).
ASK = None
# TYPE_CHECKER(code, lang, packages) -> {"ran": bool, "ok": bool,
# "errors": [...], "why": str}. None: the stack's own checker
# (mcp/typecheck.py: tsc / pyright from the harness box's tools volume, in a
# throwaway container, against the HELD package versions), which records
# "not run" only when the package has no types or the checker cannot run
# here (said why). Tests inject their own.
# (Before 2026-09-28, None meant no checker at all.)
# The fields below this line are the old comment's:
# TYPE_CHECKER(code, lang, packages) -> {"ran": bool, "ok": bool,
# "errors": [...], "why": str}. None: no type checker is configured for the
# held packages' types, and the record says "not run" -- never a pass.
TYPE_CHECKER = None
TYPED_LANGS = ("typescript", "tsx", "python")


def _chat(body: dict) -> dict:
    if CHAT is not None:
        return CHAT(body)
    import model
    import skill_pipeline
    return model.post(body, timeout=skill_pipeline.MODEL_TIMEOUT)


def _ask(system: str, user: str, *, max_tokens: int, purpose: str) -> str:
    if ASK is not None:
        return ASK(system, user, max_tokens=max_tokens, purpose=purpose)
    import skill_pipeline
    return skill_pipeline.ask_model(system, user, max_tokens=max_tokens,
                                    purpose=purpose)


def _describe(situation: str, retryable: bool, remedy: str,
              owner: str) -> str:
    """skill_pipeline.describe's shape (worker.describe's), one format."""
    return (f"{situation} | retryable: {'yes' if retryable else 'no'} | "
            f"remedy ({owner}): {remedy}")


# ---------------------------------------------------------------------------
# THE TEMPLATES (register them in skill_prompts.registry: `templates()`).
# Every one is a CHOICE, unmeasured, shaped by AGENTS.md "Prompting this
# model" and carrying skill_prompts' data-not-instructions paragraph.
# ---------------------------------------------------------------------------
PROBE_VERSION = "probe/1"
PROBE_SYSTEM = f"""You turn a coding skill's test requests into small PROBE \
TASKS whose answers code can check, and propose the checks.

{skill_prompts._data("skill", "write probes for", "write tasks and checks "
                     "about it", "skill")}

| field | what to write |
|---|---|
| case | the number of the request below that the task rewrites |
| task | a small, self-contained request for code a developer would send, in the request's language and library: what to build, its inputs and its outputs. It does NOT copy the skill's items or name the practice they teach: the task must be solvable without the skill, so the checks can tell whether the skill changed the answer |
| language | the answer's code language: typescript, tsx, javascript, jsx, python, rust, c or cpp |
| checks | regular expressions over the answer's code: "present" for code an answer that follows a DO or WHEN item contains, "absent" for code an answer that follows a DO NOT item avoids; each names the number of the item it comes from and uses only API names and code copied from that item |

Reply with one JSON object and nothing else:
{{"probes": [{{"case": 1, "task": "...", "language": "typescript", \
"checks": [{{"kind": "present", "pattern": "...", "item": 1}}]}}]}}"""


def probe_user(items: list[dict], topics: list[str],
               cases: list[str]) -> str:
    import skill_md
    lines = ["ITEMS:"]
    lines += [f"{n}. {skill_md.item_line(it)}" for n, it in
              enumerate(items, 1)]
    if topics:
        lines.append("TOPICS: " + ", ".join(topics))
    lines.append("REQUESTS:")
    lines += [f"{n}. {c}" for n, c in enumerate(cases, 1)]
    return "<skill>\n" + "\n".join(lines) + "\n</skill>"


# The system text both sides of a pair run under. The user turn is the probe
# (WITHOUT) or the probe plus the craft block (WITH); nothing else differs.
ANSWER_VERSION = "prove_answer/1"
ANSWER_SYSTEM = ("You are a coding assistant. Answer the request with the "
                 "code in one fenced code block tagged with its language, "
                 "then at most a few sentences.")

# The last resort: one answer, read blind (the judge is not told which side
# wrote it). Labelled `judge: model` wherever it decides.
JUDGE_VERSION = "prove_judge/1"
JUDGE_SYSTEM = f"""You check whether an answer does what a coding request \
asks.

{skill_prompts._data("request and answer", "check", "judge the answer",
                     "task")}

| the answer | pass |
|---|---|
| does what the request asks, and nothing in it is wrong for the request | true |
| misses part of the request, is wrong, or gives no answer | false |

Reply with one JSON object and nothing else:
{{"pass": true, "why": "..."}}"""
JUDGE_CAVEAT = ("the served model reading one answer blind (JUDGE_SYSTEM); "
                "the last resort when no code check could decide the probe, "
                "weaker than an execution-based check (ASI 2504.06821)")


def judge_user(task: str, answer: str) -> str:
    return (f"<task>\nREQUEST:\n{task}\n\nANSWER:\n{answer}\n</task>")


def templates() -> list[dict]:
    """(name, version, stage, text) rows for skill_prompts.registry."""
    rows = [("probe", PROBE_VERSION, PROBE_SYSTEM),
            ("prove_answer", ANSWER_VERSION, ANSWER_SYSTEM),
            ("prove_judge", JUDGE_VERSION, JUDGE_SYSTEM)]
    return [{"name": n, "version": v, "stage": "prove", "chars": len(s),
             "sha256": hashlib.sha256(s.encode("utf-8")).hexdigest()[:16],
             "text": s} for n, v, s in rows]


# ---------------------------------------------------------------------------
# The skill as a proof reads it.
# ---------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _items(ver: dict) -> list[dict]:
    val = ver.get("validate") or {}
    items = [it for it in val.get("items") or [] if isinstance(it, dict)]
    if not items and (ver.get("text") or "").startswith("---"):
        import skill_md
        items = list(skill_md.parse(ver["text"]).get("items") or [])
    return items


def _form(it: dict) -> str:
    f = str(it.get("form") or "DO").upper().strip()
    return "DO NOT" if f.replace("_", " ") == "DO NOT" else (
        "WHEN" if f == "WHEN" else "DO")


def _topics(ver: dict) -> list[str]:
    import skill_classify as C
    return list(C.gates(ver.get("classify") or {})["topics"])


def row_for(sid: str, ver: dict) -> dict:
    """The skill as skill_select sees it (skills.row_of): what the WITH side
    injects is skill_select.injected_text of this row."""
    s = skills.get(sid) or {}
    meta = s.get("meta") or {}
    return skills.row_of(sid, s.get("name") or sid, s.get("source_kind"),
                         s.get("source_url"),
                         {"version": ver.get("version"),
                          "text": ver.get("text") or "",
                          "classify": ver.get("classify") or {},
                          "validate": ver.get("validate") or {},
                          "package": meta.get("package"),
                          "package_version": meta.get("package_version")})


def craft_block(row: dict) -> str:
    """Exactly what the selector appends to the user turn when this skill
    is given as a body (skill_select._render_turn, no recall lines)."""
    import skill_select
    return skill_select._render_turn([row], [])


def with_message(probe: str, row: dict) -> dict:
    import message_text
    return message_text.append_text({"role": "user", "content": probe},
                                    craft_block(row))


# ---------------------------------------------------------------------------
# Languages.
# ---------------------------------------------------------------------------
def _norm_lang(name) -> str | None:
    import code_check
    lang = code_check.normalize_language(name) if name else None
    return lang if lang in code_check.CODE_LANGS else None


def _case_lang(case: dict, rule: dict) -> str | None:
    """A should-case's code language: its fence, a file it names, else the
    skill's own language or framework (skill_tests._FRAMEWORK_LANG)."""
    import skill_classify as C
    import skill_tests
    import tool_code
    for m in re.finditer(r"(?m)^\s*```([A-Za-z0-9+#-]+)", str(
            case.get("text") or "")):
        lang = _norm_lang(m.group(1))
        if lang:
            return lang
    for f in case.get("files") or []:
        lang = _norm_lang(tool_code.language_of(str(f)))
        if lang:
            return lang
    a = C.applies_to(rule)
    for x in a["languages"]:
        t = C.BY_ID.get(x)
        for cand in (x,) + tuple((t.fences if t else ()) or ()):
            lang = _norm_lang(cand)
            if lang:
                return lang
    for x in a["frameworks"]:
        lang = _norm_lang(skill_tests._FRAMEWORK_LANG.get(x))
        if lang:
            return lang
    return None


def _packages(sid: str, rule: dict) -> list[str]:
    import skill_classify as C
    out = []
    for x in C.applies_to(rule)["frameworks"]:
        t = C.BY_ID.get(x)
        out += list((t.packages if t else ()) or ())
    pkg = ((skills.get(sid) or {}).get("meta") or {}).get("package")
    if pkg:
        out.append(str(pkg))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Leaks and literals.
# ---------------------------------------------------------------------------
def _shingles(text: str) -> set[tuple]:
    w = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {tuple(w[i:i + LEAK_WORDS]) for i in range(len(w) - LEAK_WORDS
                                                        + 1)}


def leaks(task: str, items: list[dict]) -> str | None:
    """The item a task quotes (a shared LEAK_WORDS-word run with its text or
    its source quote), or None."""
    sh = _shingles(task)
    if not sh:
        return None
    for n, it in enumerate(items, 1):
        for field in ("text", "quote"):
            if sh & _shingles(str(it.get(field) or "")):
                return f"item {n}'s {field}"
    return None


_SPAN = re.compile(r"`([^`\n]{2,80})`")
_TOKEN = re.compile(r"[A-Za-z_$][\w$.]{2,}")


def _code_literal(span: str) -> bool:
    """A backticked span that is code: code-shaped (skill_classify), or
    letters with code punctuation or a keyword pair (`as any`)."""
    import skill_classify as C
    s = span.strip()
    if len(s.replace(" ", "")) < 3 or not re.search(r"[A-Za-z_$]", s):
        return False
    return C.code_shaped(s) or bool(re.search(r"[().<>\[\]{}=:;,*+/!&|]|\s",
                                              s))


def literal_pattern(lit: str) -> str:
    """The regex for a literal: skill_classify._topic_rx's for a code-shaped
    name (a call stays a call), else the escaped text with any run of
    whitespace as \\s+ and identifier boundaries at word ends."""
    import skill_classify as C
    s = lit.strip()
    if C.code_shaped(s):
        return C._topic_rx(s).pattern
    core = r"\s+".join(re.escape(p) for p in s.split())
    head = r"(?<![\w$])" if re.match(r"[\w$]", s) else ""
    tail = r"(?![\w$])" if re.search(r"[\w$]$", s) else ""
    return head + core + tail


def floor_checks(items: list[dict], topics: list[str]
                 ) -> tuple[list[dict], list[dict]]:
    """(checks, dropped) from the items alone: each DO / WHEN item's code
    spans and the topics it names -> present; each DO NOT item's -> absent.
    A literal on both sides is dropped (the checks would contradict)."""
    import skill_classify as C
    want: dict[str, dict] = {}
    for n, it in enumerate(items, 1):
        text = str(it.get("text") or "")
        kind = "absent" if _form(it) == "DO NOT" else "present"
        lits = [m.group(1).strip() for m in _SPAN.finditer(text)
                if _code_literal(m.group(1))]
        lits += [t for t in topics if C.code_shaped(t)
                 and C._topic_rx(t).search(text)]
        for lit in dict.fromkeys(lits):
            key = lit.strip("`").strip()
            row = want.setdefault(key, {"kinds": set(), "items": []})
            row["kinds"].add(kind)
            row["items"].append(n)
    checks, dropped = [], []
    for lit, row in want.items():
        if len(row["kinds"]) > 1:
            dropped.append({"literal": lit, "why": "a DO and a DO NOT item "
                            "both name it; the checks would contradict"})
            continue
        kind = next(iter(row["kinds"]))
        checks.append({"kind": kind, "pattern": literal_pattern(lit),
                       "literal": lit, "item": row["items"][0],
                       "from": "items", "judge": "code"})
    return checks, dropped


def verify_pattern(pattern, skill_blob: str) -> str | None:
    """Why a proposed pattern is refused, or None: it must compile, not
    match the empty text, match the skill's own items or topics, and carry a
    literal of 3+ characters that the skill names."""
    if not isinstance(pattern, str) or not pattern.strip():
        return "no pattern"
    if len(pattern) > 200:
        return "longer than 200 characters"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"the pattern does not compile ({e})"
    if rx.search(""):
        return "it matches the empty text"
    if not rx.search(skill_blob):
        return "the skill's items and topics do not contain it"
    bare = re.sub(r"\\[a-zA-Z]", " ", pattern)
    bare = re.sub(r"\\(.)", r"\1", bare)
    toks = [t.strip(".") for t in _TOKEN.findall(bare)]
    if not any(len(t) >= 3 and t in skill_blob for t in toks):
        return ("no literal of 3 or more characters from the skill's items "
                "or topics")
    return None


def _skill_blob(items: list[dict], topics: list[str]) -> str:
    return "\n".join([str(it.get("text") or "") for it in items]
                     + list(topics))


# ---------------------------------------------------------------------------
# 1. Probes.
# ---------------------------------------------------------------------------
def _case_task(case: dict) -> str:
    """A should-case as one request: its text, then the files it names and
    the output it carries (the context skill_tests.messages_of gives the
    selector, said in the user turn)."""
    parts = [str(case.get("text") or "").strip()]
    files = [str(f) for f in case.get("files") or []]
    if files:
        parts.append("The project has these files: " + ", ".join(files)
                     + ".")
    if case.get("tool_output"):
        parts.append("The last run printed:\n```\n"
                     + str(case["tool_output"]).strip() + "\n```")
    return "\n\n".join(p for p in parts if p)


def floor_probes(ver: dict) -> tuple[list[dict], list[dict]]:
    """(probes, skipped): up to PROBES of the skill's own should-cases,
    those whose code language is known first, in their order; a case that
    quotes an item is skipped."""
    rule = ver.get("classify") or {}
    items = _items(ver)
    should = [c for c in (((ver.get("tests") or {}).get("activation") or {})
                          .get("should") or []) if isinstance(c, dict)
              and str(c.get("text") or "").strip()]
    rows, skipped = [], []
    for i, c in enumerate(should):
        task = _case_task(c)
        why = leaks(task, items)
        if why:
            skipped.append({"case": i + 1, "why": f"it quotes {why}"})
            continue
        rows.append({"case": i + 1, "task": task,
                     "language": _case_lang(c, rule),
                     "source": f"should[{i + 1}]"})
    rows.sort(key=lambda r: (r["language"] is None, r["case"]))
    for r in rows[PROBES:]:
        skipped.append({"case": r["case"], "why": f"past the operator's "
                        f"{PROBES} probes"})
    return rows[:PROBES], skipped


def verify_proposals(got: dict, probes: list[dict], items: list[dict],
                     topics: list[str]) -> tuple[dict, list[dict]]:
    """({probe index: {task?, language?, checks}}, refused): each proposal
    verified by code. A task replaces its case's text only when it names
    no item's words (leaks) and keeps the case's language; a check is kept
    only when verify_pattern passes and its kind is present or absent."""
    blob = _skill_blob(items, topics)
    out: dict[int, dict] = {}
    refused: list[dict] = []
    for p in got.get("probes") or []:
        if not isinstance(p, dict):
            continue
        k = p.get("case")
        if not (isinstance(k, int) or str(k).isdigit()) or not (
                1 <= int(k) <= len(probes)):
            refused.append({"case": k, "why": "no such case"})
            continue
        idx = int(k) - 1
        rec = out.setdefault(idx, {"checks": []})
        task = str(p.get("task") or "").strip()
        lang = _norm_lang(p.get("language"))
        if task:
            why = None
            if len(task) > 1200:
                why = "a task of 1 to 1,200 characters"
            elif leaks(task, items):
                why = f"it quotes {leaks(task, items)} (leaks the answer)"
            elif probes[idx]["language"] and lang and \
                    lang != probes[idx]["language"]:
                why = (f"language {lang} is not the case's "
                       f"{probes[idx]['language']}")
            if why:
                refused.append({"case": idx + 1, "task": task[:120],
                                "why": why})
            else:
                rec["task"] = task
                if lang and not probes[idx]["language"]:
                    rec["language"] = lang
        for c in p.get("checks") or []:
            if not isinstance(c, dict):
                continue
            kind = c.get("kind")
            pat = c.get("pattern")
            why = ("kind must be present or absent"
                   if kind not in ("present", "absent")
                   else verify_pattern(pat, blob))
            if why:
                refused.append({"case": idx + 1, "pattern": str(pat)[:120],
                                "why": why})
                continue
            item = c.get("item")
            rec["checks"].append({"kind": kind, "pattern": pat,
                                  "item": int(item) if str(item).isdigit()
                                  else None, "from": "model",
                                  "judge": "code"})
    return out, refused


def derive(sid: str, ver: dict, *, use_model: bool | None = None
           ) -> dict:
    """{probes: [{task, language, source, checks}], dropped, model}. The
    deterministic floor, then the model's verified proposals."""
    import skill_pipeline
    items = _items(ver)
    topics = _topics(ver)
    probes, skipped = floor_probes(ver)
    base, conflicts = floor_checks(items, topics)
    dropped = [dict(x, stage="case") for x in skipped] + \
        [dict(x, stage="check") for x in conflicts]
    model_rec: dict = {"prompt": PROBE_VERSION}
    use_model = MODEL_PROBE if use_model is None else use_model
    proposals: dict = {}
    if not probes:
        model_rec["skipped"] = "no probe task to rewrite"
    elif not use_model:
        model_rec["skipped"] = "YAMADORI_SKILL_MODEL_PROBE=0"
    else:
        try:
            got = skill_prompts.parse_object(_ask(
                PROBE_SYSTEM, probe_user(items, topics,
                                         [p["task"] for p in probes]),
                max_tokens=skill_pipeline.SCREEN_MAX_TOKENS,
                purpose="prove"))
            proposals, refused = verify_proposals(got, probes, items, topics)
            model_rec.update(
                proposed=len(got.get("probes") or []),
                tasks_kept=sum(1 for v in proposals.values() if v.get("task")),
                checks_kept=sum(len(v["checks"]) for v in proposals.values()),
                refused=refused)
        except (ValueError, RuntimeError) as e:
            model_rec["error"] = f"{type(e).__name__}: {e}"[:200]
    out = []
    for i, p in enumerate(probes):
        prop = proposals.get(i) or {}
        task = prop.get("task") or p["task"]
        lang = p["language"] or prop.get("language")
        checks = []
        if lang:
            checks.append({"kind": "parse", "judge": "code"})
            if lang in TYPED_LANGS:
                checks.append({"kind": "types", "judge": "code"})
        seen = set()
        for c in base + prop.get("checks", []):
            if c["pattern"] in seen:
                continue
            seen.add(c["pattern"])
            if c["kind"] == "present" and re.search(c["pattern"], task):
                dropped.append({"stage": "check", "case": p["case"],
                                "pattern": c["pattern"], "why":
                                "the probe names it: a present check would "
                                "measure echo, not the skill"})
                continue
            checks.append(dict(c))
        for n, c in enumerate(checks, 1):
            c["id"] = f"{c['kind']}{n}"
        out.append({"task": task, "language": lang, "case": p["case"],
                    "source": p["source"] + ("+model" if prop.get("task")
                                             else ""),
                    "checks": checks})
    return {"probes": out, "dropped": dropped, "model": model_rec}


# ---------------------------------------------------------------------------
# 2. Checks, by code.
# ---------------------------------------------------------------------------
def code_blocks(answer: str, lang: str | None) -> list[dict]:
    """The answer's fenced code blocks, each with the language it parses
    as: its own tag when that is a code language, the probe's when it is
    untagged; other tags (bash, json, text) are not code of the answer."""
    import code_check
    out = []
    for b in code_check.fenced_blocks(answer or ""):
        tag = (b.get("lang") or "").strip()
        bl = _norm_lang(tag) if tag else lang
        if bl:
            out.append(dict(b, parse_as=bl))
    return out


def run_check(chk: dict, answer: str, lang: str | None,
              packages: list[str]) -> tuple[bool | None, str]:
    """(pass, why) for one check on one answer; pass None = not run."""
    blocks = code_blocks(answer, lang)
    code = "\n".join(b["code"] for b in blocks) if blocks else (
        "" if lang else (answer or ""))
    kind = chk["kind"]
    if kind == "parse":
        import code_check
        if not blocks:
            return False, "no fenced code block"
        for b in blocks:
            if not b.get("closed"):
                return False, (f"the {b['parse_as']} block at line "
                               f"{b['line'] - 1} is not closed")
            errs = code_check.syntax_errors(b["code"], b["parse_as"])
            if errs:
                e = errs[0]
                return False, (f"{b['parse_as']} syntax error, block line "
                               f"{e.get('line')}: {e.get('message')}")[:200]
        return True, f"{len(blocks)} block(s) parse"
    if kind == "types":
        if not blocks:
            return None, "not run: no code block to type-check"
        checker = TYPE_CHECKER or _stack_type_checker
        try:
            res = checker(code, lang, packages) or {}
        except Exception as e:                                   # noqa: BLE001
            return None, f"not run: the type checker raised {e}"[:200]
        if not res.get("ran"):
            return None, "not run: " + str(res.get("why") or "")[:180]
        errs = res.get("errors") or []
        return bool(res.get("ok")), ("types check" if res.get("ok") else
                                     str(errs[0] if errs else "type errors")
                                     [:200])
    if kind in ("present", "absent"):
        m = re.search(chk["pattern"], code)
        if kind == "present":
            return bool(m), (f"found {m.group(0)[:60]!r}" if m else
                             "not in the answer's code")
        return (not m), ("not in the answer's code" if not m else
                         f"found {m.group(0)[:60]!r}")
    return None, f"not run: unknown check kind {kind!r}"


# The pin the stack's checker installs (set by prove() for the version it
# proves): the skill's package_version, and the major its name names.
_PIN: dict = {"version": None, "majors": {}}


def _stack_type_checker(code: str, lang: str | None,
                        packages: list[str]) -> dict:
    import typecheck
    return typecheck.check(code, lang, packages,
                           version=_PIN.get("version"),
                           majors=_PIN.get("majors") or {})


def pin_of(sid: str, ver: dict, packages: list[str]) -> dict:
    """{version, majors}: the skill's own package_version, and the major its
    name or description names ("v10") for each package (typecheck.pinned
    uses a major only where a held version has it)."""
    s = skills.get(sid) or {}
    meta = s.get("meta") or {}
    text = " ".join([str(s.get("name") or ""), str(ver.get("text") or "")
                     [:600]])
    m = re.search(r"(?<![\w.])v(\d{1,3})(?![\w.])", text)
    majors = {p: int(m.group(1)) for p in packages} if m else {}
    return {"version": meta.get("package_version"), "majors": majors}


def judge(task: str, answer: str) -> tuple[bool | None, str]:
    """The model judge for one answer (blind). (None, why) when its reply
    is not a verdict."""
    import skill_pipeline
    try:
        got = skill_prompts.parse_object(_ask(
            JUDGE_SYSTEM, judge_user(task, answer),
            max_tokens=skill_pipeline.SCREEN_MAX_TOKENS, purpose="prove"))
    except (ValueError, RuntimeError) as e:
        return None, f"not run: the judge gave no verdict ({e})"[:200]
    if not isinstance(got.get("pass"), bool):
        return None, "not run: the judge's reply has no boolean pass"
    return got["pass"], str(got.get("why") or "")[:200]


def pair(without: bool | None, with_: bool | None) -> str:
    if without is None or with_ is None:
        return PAIR_NOT_RUN
    if without and not with_:
        return PAIR_WORSE
    if with_ and not without:
        return PAIR_BETTER
    return PAIR_SAME


# ---------------------------------------------------------------------------
# 3. The pair.
# ---------------------------------------------------------------------------
# Every field that sets how a generation samples or how long it may run: the
# WITH body takes the WITHOUT body's values, so the only difference between
# the two sides is the craft block in the user turn.
_SAME_FIELDS = ("max_tokens", "reasoning_budget_tokens",
                "reasoning_budget_message", "reasoning_budget_nudge",
                "reasoning_budget_nudge_at", "reasoning_effort",
                "enable_thinking", "chat_template_kwargs", "temperature",
                "top_p", "top_k", "min_p", "presence_penalty",
                "repeat_penalty", "seed")


def seed_of(sid: str, v: int, task: str) -> int:
    return int(_sha(f"{sid}:{v}:{_sha(task)}")[:8], 16) & 0x7FFFFFFF


def full_answer_room(task: str, row: dict) -> int:
    """The job's FULL answer room: the helper share less the longer (WITH)
    prompt and the helper's thinking cap (tiers.budget's terms), never below
    tiers.A_MIN. The retry of an undecided probe asks for this as its
    answer allowance (operator, 2026-09-28)."""
    import tiers
    s = tiers._shares()
    prompt = tiers.estimate_prompt_tokens({"messages": [
        {"role": "system", "content": ANSWER_SYSTEM},
        with_message(task, row)]})
    think = max(tiers.HELPER_THINKING, tiers.MIN_THINKING)
    return max(int(s["helper"]) - prompt - think, tiers.A_MIN)


def bodies(task: str, row: dict, seed: int, *, answer: int | None = None
           ) -> tuple[dict, dict, dict]:
    """(without, with, equalised): the two upstream bodies of one pair,
    shaped as a pipeline job is (skill_pipeline.shaped_body, job
    skill.prove), the same seed on both, and every sampling and budget
    field of WITH set to WITHOUT's (`equalised` lists any that differed).
    `answer`: the answer allowance (default tiers.A_MIN)."""
    import skill_pipeline
    import tiers
    answer = answer or ANSWER_TOKENS or tiers.A_MIN
    a = skill_pipeline.shaped_body(ANSWER_SYSTEM, task, max_tokens=answer,
                                   purpose="prove")
    # WITH is shaped from its own prompt (so `equalised` shows what the
    # longer prompt would have changed), then set to WITHOUT's fields.
    b = skill_pipeline.shaped_body(ANSWER_SYSTEM,
                                   with_message(task, row)["content"],
                                   max_tokens=answer, purpose="prove")
    a["seed"] = seed
    b["seed"] = seed
    equalised = {}
    for k in _SAME_FIELDS:
        if k in a and b.get(k) != a[k]:
            equalised[k] = [a.get(k), b.get(k)]
            b[k] = a[k]
        elif k not in a and k in b:
            equalised[k] = [None, b[k]]
            b.pop(k)
    return a, b, equalised


def _generate(body: dict) -> tuple[str | None, str, float]:
    """(answer or None, why, seconds). A transport or server failure RAISES
    a retryable error: a proof is never recorded from half its pairs."""
    import model
    import skill_pipeline
    prev = getattr(skill_pipeline._CURRENT, "job", None)
    skill_pipeline._CURRENT.job = skill_pipeline.job_name("prove")
    t0 = time.time()
    try:
        d = _chat(body)
    except Exception as e:                                       # noqa: BLE001
        raise RuntimeError(_describe(
            f"a prove generation failed ({type(e).__name__}: "
            f"{str(e)[:160]})", True, "retried automatically; if it repeats, "
            "check the model on llama-swap", "worker")) from e
    finally:
        skill_pipeline._CURRENT.job = prev
    secs = round(time.time() - t0, 3)
    try:
        return model.answer(d), "answered", secs
    except model.BudgetEvent as e:
        return None, f"budget event: {e}"[:200], secs


def run_probe(sid: str, v: int, probe: dict, row: dict,
              packages: list[str], beat=None, *, attempt: int = 1,
              repeat: int = 0) -> dict:
    """One pair. `attempt` 2 is the retry of an undecided probe: the job's
    full answer room (full_answer_room) and a second seed, the same on both
    sides (the first pair's seed would give the first pair back). `repeat`
    k >= 1 is the k-th REPEAT of a flagged probe (THE REPEAT RULE): the same
    task and answer room as the run it repeats, a seed of its own, the same
    on both sides."""
    task = probe["task"]
    base = task if attempt == 1 else f"{task}#{attempt}"
    seed = seed_of(sid, v, base if not repeat else f"{base}#r{repeat}")
    answer = full_answer_room(task, row) if attempt > 1 else None
    a, b, equalised = bodies(task, row, seed, answer=answer)
    res: dict = {"probe_sha256": _sha(task), "excerpt": task[:160],
                 "case": probe["case"], "source": probe["source"],
                 "language": probe["language"], "seed": seed,
                 "attempt": attempt, "max_tokens": a.get("max_tokens"),
                 "seconds": {}, "checks": []}
    if repeat:
        res["repeat"] = repeat
    if equalised:
        res["equalised"] = equalised
    answers = {}
    for side, body in (("without", a), ("with", b)):
        if beat:
            beat(f"probe {probe['case']} {side}")
        ans, why, secs = _generate(body)
        res["seconds"][side] = secs
        answers[side] = ans
        res[f"{side}_answer"] = ({"sha256": _sha(ans), "chars": len(ans)}
                                 if ans is not None else {"why": why})
    if answers["without"] is None or answers["with"] is None:
        res["decided"] = False
        res["why"] = "; ".join(f"{s}: {res[f'{s}_answer']['why']}"
                               for s in ("without", "with")
                               if answers[s] is None)
        for c in probe["checks"]:
            res["checks"].append(dict(c, without=None, with_=None,
                                      pair=PAIR_NOT_RUN))
        return _rename(res)
    for c in probe["checks"]:
        w0, why0 = run_check(c, answers["without"], probe["language"],
                             packages)
        w1, why1 = run_check(c, answers["with"], probe["language"],
                             packages)
        res["checks"].append(dict(c, without=w0, with_=w1,
                                  why_without=why0, why_with=why1,
                                  pair=pair(w0, w1)))
    if not any(c["pair"] != PAIR_NOT_RUN for c in res["checks"]):
        # THE LAST RESORT (operator: "a model judge only as a last resort,
        # labelled judge: model").
        w0, why0 = judge(task, answers["without"])
        w1, why1 = judge(task, answers["with"])
        res["checks"].append({"id": "judge", "kind": "judge",
                              "judge": "model", "caveat": JUDGE_CAVEAT,
                              "prompt": JUDGE_VERSION, "without": w0,
                              "with_": w1, "why_without": why0,
                              "why_with": why1, "pair": pair(w0, w1)})
    res["decided"] = any(c["pair"] != PAIR_NOT_RUN for c in res["checks"])
    if not res["decided"]:
        res["why"] = "no check ran on both sides"
    return _rename(res)


def _rename(res: dict) -> dict:
    """`with` is a keyword in the code above; the record says `with`."""
    for c in res["checks"]:
        if "with_" in c:
            c["with"] = c.pop("with_")
    return res


# ---------------------------------------------------------------------------
# 4. The decision.
# ---------------------------------------------------------------------------
def confirm(sid: str, v: int, derived: list[dict], probes: list[dict],
            row: dict, packages: list[str], beat=None) -> list[dict]:
    """THE REPEAT RULE. `probes` are the run results of `derived` (the same
    order). Every probe with a worse check is run REPEATS more times on fresh
    seeds (both sides, one seed per run); each flagged check keeps `runs`
    {pairs, worse, of} and stays worse only when worse in a strict majority
    of its runs. A flag that does not repeat becomes `same` with
    `unconfirmed` set, its first result kept as `flagged`. Returns the
    probes, changed in place; each repeated probe carries `repeats` (the
    compact records of its repeat runs)."""
    if REPEATS < 1:
        return probes
    for d, p in zip(derived, probes):
        flagged = [c for c in p["checks"] if c["pair"] == PAIR_WORSE]
        if not flagged:
            continue
        runs = [run_probe(sid, v, d, row, packages, beat,
                          attempt=int(p.get("attempt") or 1), repeat=k)
                for k in range(1, REPEATS + 1)]
        p["repeats"] = [{
            "repeat": r["repeat"], "seed": r["seed"],
            "seconds": r["seconds"], "decided": r.get("decided"),
            "checks": [{k: c.get(k) for k in ("id", "kind", "pair", "without",
                                              "with", "why_without",
                                              "why_with")}
                       for c in r["checks"]]} for r in runs]
        for c in flagged:
            pairs = [PAIR_WORSE]
            for r in runs:
                m = next((x for x in r["checks"] if x.get("id") == c.get("id")
                          and x["kind"] == c["kind"]), None)
                pairs.append(m["pair"] if m else PAIR_NOT_RUN)
            n_worse = pairs.count(PAIR_WORSE)
            c["runs"] = {"pairs": pairs, "worse": n_worse, "of": len(pairs)}
            if n_worse * 2 > len(pairs):
                c["confirmed"] = True
            else:
                c["flagged"] = {k: c.get(k) for k in ("without", "with",
                                                      "why_without",
                                                      "why_with")}
                c["unconfirmed"] = True
                c["pair"] = PAIR_SAME
    return probes


def decide(probes: list[dict]) -> dict:
    """{verdict, worse, better, why}: paired, per probe and per check. A
    check `confirm` did not confirm is listed in `unconfirmed`, and counts as
    the same on both sides."""
    worse, better, same, unconfirmed = [], [], 0, []
    for n, p in enumerate(probes, 1):
        for c in p["checks"]:
            ref = {"probe": n, "check": c.get("id"), "kind": c["kind"],
                   "judge": c.get("judge"), "pattern": c.get("pattern"),
                   "without": c.get("why_without"),
                   "with": c.get("why_with")}
            if c.get("runs"):
                ref["runs"] = c["runs"]
            if c.get("unconfirmed"):
                fl = c.get("flagged") or {}
                unconfirmed.append(dict(ref, without=fl.get("why_without"),
                                        **{"with": fl.get("why_with")}))
            if c["pair"] == PAIR_WORSE:
                worse.append(ref)
            elif c["pair"] == PAIR_BETTER:
                better.append(ref)
            elif c["pair"] == PAIR_SAME:
                same += 1
    extra = {"unconfirmed": unconfirmed} if unconfirmed else {}
    note = (f"; {len(unconfirmed)} flagged check(s) did not repeat "
            "(unconfirmed)" if unconfirmed else "")
    if worse:
        return dict({"verdict": "worse", "worse": worse, "better": better,
                     "why": f"{len(worse)} check(s) that passed without the "
                            "skill failed with it, in a majority of their "
                            "runs" + note}, **extra)
    if better:
        return dict({"verdict": "better", "worse": [], "better": better,
                     "why": f"{len(better)} check(s) that failed without the "
                            "skill passed with it; none got worse" + note},
                    **extra)
    if same:
        return dict({"verdict": "tie", "worse": [], "better": [],
                     "gain": "no measurable gain",
                     "why": f"{same} paired check(s), each the same on both "
                            "sides" + note}, **extra)
    return {"verdict": "unproven", "worse": [], "better": [],
            "why": "no check decided on both sides (a budget event, or "
                   "nothing could be checked)",
            "retryable": False,
            "remedy": "(operator) read the probes' `why`: add should-cases "
                      "that ask for code (tests.json), or fix what cut the "
                      "answers off, then re-run the pipeline from tests"}


def prove(sid: str, v: int, ver: dict, *, mode: str = "pipeline",
          beat=None, use_model: bool | None = None) -> dict:
    """Run the whole proof for one version. Returns the record."""
    t0 = time.time()
    rec: dict = {"skill": sid, "skill_version": int(v), "mode": mode,
                 "at": t0, "templates": {"probe": PROBE_VERSION,
                                         "answer": ANSWER_VERSION,
                                         "judge": JUDGE_VERSION},
                 "type_checker": ("injected" if TYPE_CHECKER is not None
                                  else "the stack's (mcp/typecheck.py)")}
    if not (ver.get("text") or "").strip():
        rec.update(verdict="not_run", why="the version has no validated "
                   "SKILL.md text to inject", retryable=False,
                   remedy="(pipeline) re-run validate", seconds=0.0)
        return rec
    d = derive(sid, ver, use_model=use_model)
    rec.update(dropped=d["dropped"], model=d["model"])
    if not d["probes"]:
        # Nothing to prove it with: UNPROVEN (operator, 2026-09-28: a proof
        # that is otherwise undecided does not arm).
        rec.update(verdict="unproven", why="no probe task could be derived "
                   "from the skill's should-cases", retryable=False,
                   remedy="(operator) add should-cases that ask for code "
                   "(tests.json), then re-run tests", seconds=round(
                       time.time() - t0, 3))
        return rec
    row = row_for(sid, ver)
    pk = _packages(sid, ver.get("classify") or {})
    _PIN.update(pin_of(sid, ver, pk))
    rec["type_pin"] = dict(_PIN)
    rec.update(rule=RULE if REPEATS >= 1 else "one-sample",
               repeats=REPEATS)
    probes = [run_probe(sid, v, p, row, pk, beat) for p in d["probes"]]
    # THE REPEAT RULE: a flagged check is run again before it counts.
    confirm(sid, v, d["probes"], probes, row, pk, beat)
    rec["probes"] = probes
    if len(probes) < MIN_PROBES:
        rec["probes_short"] = (f"{len(probes)} probe(s), under the operator's "
                               f"{MIN_PROBES}: the skill has too few usable "
                               "should-cases")
    dec = decide(probes)
    if dec["verdict"] == "unproven":
        # ONE RETRY (operator, 2026-09-28): every probe undecided -- a token
        # limit, or nothing checkable -- is run once more with the job's
        # full answer room; still undecided, the version is UNPROVEN.
        again = [run_probe(sid, v, p, row, pk, beat, attempt=2)
                 for p in d["probes"]]
        rec["retry"] = {"why": dec["why"], "probes": len(again),
                        "max_tokens": [p.get("max_tokens") for p in again]}
        # A probe the retry decided counts; one it did not either is
        # recorded as the retry (the latest attempt).
        probes = [a if a.get("decided") else b
                  for a, b in zip(probes, again)]
        confirm(sid, v, d["probes"], probes, row, pk, beat)
        rec["probes"] = probes
        dec = decide(probes)
    rec.update(dec)
    rec["seconds"] = round(time.time() - t0, 3)
    rec["generation_seconds"] = round(sum(
        sum(p["seconds"].values()) + sum(
            sum(r["seconds"].values()) for r in p.get("repeats") or [])
        for p in probes), 3)
    return rec


# ---------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------
def record_of(ver: dict | None) -> dict:
    """The version's prove record: its own column when skills has one, else
    validate["prove"]."""
    ver = ver or {}
    got = ver.get("prove")
    if isinstance(got, dict) and got:
        return got
    return dict((ver.get("validate") or {}).get("prove") or {})


def _fields(sid: str, v: int, rec: dict) -> dict:
    """The update_version / quarantine fields that store `rec`."""
    if "prove" in skills._JSON_V:
        return {"prove": rec}
    val = dict((skills.version(sid, v) or {}).get("validate") or {})
    val["prove"] = rec
    return {"validate": val}


def store(sid: str, v: int, rec: dict) -> None:
    skills.update_version(sid, v, **_fields(sid, v, rec))


def _x(rec: dict) -> dict:
    """The x-record a handler returns (the job result)."""
    out = {"prove": rec.get("verdict"), "mode": rec.get("mode"),
           "probes": len(rec.get("probes") or []),
           "worse": len(rec.get("worse") or []),
           "better": len(rec.get("better") or []),
           "seconds": rec.get("seconds")}
    for k in ("gain", "why", "probes_short"):
        if rec.get(k):
            out[k] = rec[k]
    return out


def unproven_reason(rec: dict) -> str:
    whys = sorted({str(p.get("why") or "")[:80] for p in
                   rec.get("probes") or [] if not p.get("decided")} - {""})
    return ("prove: unproven -- " + str(rec.get("why") or "")
            + (" (" + "; ".join(whys[:3]) + ")" if whys else "")
            + (", after one retry with the full answer room"
               if rec.get("retry") else "")
            + " | retryable: no | remedy (operator): " + str(
                rec.get("remedy") or "re-run the pipeline from tests"))


def quarantine_reason(rec: dict) -> str:
    parts = []
    for w in (rec.get("worse") or [])[:4]:
        what = w["kind"] + (f" {w['pattern']}" if w.get("pattern") else "")
        runs = w.get("runs") or {}
        parts.append(f"probe {w['probe']} {what}: without {w['without']!r}; "
                     f"with {w['with']!r}"
                     + (f"; worse in {runs['worse']} of {runs['of']} runs"
                        if runs else ""))
    return ("prove: WITH the skill a check that passed without it failed ("
            + "; ".join(parts) + ") | retryable: no | remedy (operator): "
            "read validate.prove (or the prove record) for the probes, edit "
            "the item the failing check comes from, and re-run the pipeline")


# ---------------------------------------------------------------------------
# The handler (skill_pipeline's contract: returns a dict; a permanent
# outcome is recorded on the version, never raised; a retryable failure
# RAISES so the worker retries it).
# ---------------------------------------------------------------------------
def _backlog_target(p: dict) -> tuple[str, int, dict] | None:
    sid, v = p.get("skill"), p.get("version")
    s = skills.get(sid) if sid else None
    ver = skills.version(sid, v) if s and v else None
    if ver is None or ver["state"] != "armed" or s.get("served_version") \
            != int(v):
        return None
    return sid, int(v), ver


def handle_prove(job: dict, ctx, *, inline: bool = False) -> dict:
    import skill_pipeline
    p = job.get("payload") or {}
    backlog = bool(p.get("backlog"))
    if p.get("reprove"):
        return _handle_reprove(job, ctx)
    if backlog:
        t = _backlog_target(p)
        if t is None:
            ver = skills.version(p.get("skill"), p.get("version")) \
                if p.get("skill") else None
            return {"skipped": (f"v{p.get('version')} of {p.get('skill')} "
                                f"is {ver['state']}, not the armed served "
                                "version" if ver else "no such skill version")}
    else:
        t = skill_pipeline._target(job, "prove")
        if t is None:
            return skill_pipeline._skipped(job, "prove")
    sid, v, ver = t
    beat = getattr(ctx, "beat", None)
    no_model = skill_pipeline.inline_without_model(ver) if not inline \
        else "inline install (no model)"
    if not backlog and no_model:
        # A migration or authored install runs with no model: recorded, and
        # it does not block arming (the operator's ask). The backlog proves
        # it later.
        rec = {"skill": sid, "skill_version": v, "mode": "inline",
               "at": time.time(), "verdict": "not_run",
               "why": f"not run: {no_model}", "retryable": True,
               "remedy": "(worker) the prove backlog (enqueue_backlog) "
                         "proves it once armed, on an idle stack",
               "seconds": 0.0}
        store(sid, v, rec)
        return _x(rec)
    rec = prove(sid, v, ver, mode="backlog" if backlog else "pipeline",
                beat=beat)
    if rec["verdict"] == "unproven" and not backlog:
        # Not armed (operator, 2026-09-28). A backlog proof of a skill that
        # is ALREADY serving records it and leaves the skill serving (the
        # operator decides; skill_prove.backlog lists it as proved).
        skills.quarantine(sid, v, unproven_reason(rec),
                          **_fields(sid, v, rec))
        return dict(_x(rec), quarantined=True)
    if rec["verdict"] == "worse":
        skills.quarantine(sid, v, quarantine_reason(rec),
                          **_fields(sid, v, rec))
        return dict(_x(rec), quarantined=True)
    store(sid, v, rec)
    return _x(rec)


# ---------------------------------------------------------------------------
# The re-prove: versions quarantined as worse by the one-sample rule.
# ---------------------------------------------------------------------------
def _reprove_ok(s: dict | None, ver: dict | None, v: int,
                include_unproven: bool = False) -> str | None:
    """Why a version is NOT a re-prove target (None: it is). A target is the
    skill's latest version, quarantined by PROVE (its reason says so), whose
    record has a verdict `worse` (or, with include_unproven, `unproven`)
    and was not decided by this rule (no `rule`), of a skill that is enabled
    and not archived, and that has text to inject."""
    if s is None or ver is None:
        return "no such skill version"
    if int(s.get("latest_version") or 0) != int(v):
        return "not the skill's latest version"
    if ver.get("state") != "quarantined":
        return f"the version is {ver.get('state')}, not quarantined"
    if not str(ver.get("reason") or "").startswith("prove:"):
        return "not quarantined by prove"
    if not (ver.get("text") or "").strip():
        return "the version has no text"
    if not s.get("enabled", True) or s.get("status") == "archived":
        return "the skill is disabled or archived"
    rec = record_of(ver)
    if rec.get("rule") == RULE:
        return f"already decided by {RULE}"
    ok = ("worse", "unproven") if include_unproven else ("worse",)
    if rec.get("verdict") not in ok:
        return f"the prove verdict is {rec.get('verdict')!r}"
    return None


def reprove_targets(include_unproven: bool = False
                    ) -> tuple[list[dict], dict]:
    """(rows, skipped): every quarantined skill whose latest version PROVE
    quarantined under an earlier rule (see `_reprove_ok`) and that has no
    live prove job. Read-only. Sorted by skill id for a stable order."""
    live = _live_jobs()
    rows, skipped = [], {"live_job": 0, "not_target": 0}
    con = skills._db()
    try:
        cands = [r["id"] for r in con.execute(
            "SELECT id FROM skills WHERE status='quarantined' ORDER BY id")]
    finally:
        con.close()
    for sid in cands:
        s = skills.get(sid)
        v = int((s or {}).get("latest_version") or 0)
        ver = skills.version(sid, v)
        if _reprove_ok(s, ver, v, include_unproven) is not None:
            skipped["not_target"] += 1
            continue
        if live.get(sid):
            skipped["live_job"] += 1
            continue
        rec = record_of(ver)
        probes = rec.get("probes") or []
        rows.append({"skill": sid, "version": v, "name": s.get("name"),
                     "verdict": rec.get("verdict"), "probes": len(probes),
                     "seconds_before": rec.get("seconds"),
                     # what each flagged probe's pair took: its repeat costs
                     # about the same again
                     "flagged_pair_seconds": round(sum(
                         sum((p.get("seconds") or {}).values())
                         for p in probes if any(
                             c.get("pair") == PAIR_WORSE
                             for c in p.get("checks") or [])), 3)})
    return rows, skipped


def reprove_quarantined(limit: int | None = None, *,
                        include_unproven: bool = False,
                        dry_run: bool = False) -> dict:
    """THE re-prove command: enqueue one idle-gated gpu-lane prove job
    (payload `reprove`) for every skill `reprove_targets` lists. Each runs
    the proof again under the repeat rule when the stack is idle
    (worker.run_one defers it while busy, mcp/idle.py); `_handle_reprove`
    then serves the skill again or keeps it quarantined. `dry_run` lists and
    enqueues nothing. Idempotent: a skill with a live job, or already
    decided under this rule, is not enqueued again."""
    rows, skipped = reprove_targets(include_unproven)
    if limit is not None:
        rows = rows[:max(0, int(limit))]
    q, lane = _queue()
    out = []
    for r in rows:
        if dry_run:
            out.append(dict(r))
            continue
        jid = jobs.add(q, {"skill": r["skill"], "version": r["version"],
                           "stage": "prove", "reprove": True, "idle": True},
                       lane=lane, dataset=f"skill:{r['skill']}",
                       stage="prove")
        out.append(dict(r, job=jid))
    return {"dry_run": dry_run, "rule": RULE, "repeats": REPEATS,
            ("would_enqueue" if dry_run else "enqueued"): out,
            "skipped": skipped, "estimate": reprove_estimate(rows)}


def reprove_estimate(rows: list[dict]) -> dict:
    """What a re-prove costs, from the targets' own earlier records (the
    seconds those proofs MEASURED): a fresh proof of the same skills takes
    about what the old ones took (`lower_bound_seconds`: no flag repeats),
    and at most that plus one more pair for each probe that was flagged
    before (`upper_bound_seconds`: every old flag flags again, which the
    repeat rule makes less likely). Generations: 2 per probe, plus 2 x
    REPEATS per flagged probe."""
    first = sum(float(r.get("seconds_before") or 0) for r in rows)
    flagged = sum(float(r.get("flagged_pair_seconds") or 0) for r in rows)
    out: dict = {"skills": len(rows), "generations": (
        f"2 x probes, plus 2 x {REPEATS} per probe with a flagged check")}
    if first:
        out.update(lower_bound_seconds=round(first, 1),
                   upper_bound_seconds=round(first + REPEATS * flagged, 1),
                   basis=f"the {len(rows)} targets' own earlier proof "
                         "seconds (flagged probes' pair seconds for the "
                         "upper bound); the model and card that run the "
                         "job set the real figure")
    else:
        out["basis"] = "no earlier proof seconds recorded: unknown"
    return out


# The order a re-prove runs in when released (operator via coordinator,
# 2026-10-06: "prioritise the skills the pagoda stack needs first (r3f v10 /
# drei, koota, pmndrs math, TSL/three, React, TypeScript)"): higher first
# (jobs.claim orders by priority DESC). Areas are skill_select.area_of's.
REPROVE_PRIORITY = {"r3f": 60, "koota": 60, "pmndrs_math": 60, "threejs": 50,
                    "react": 40, "typescript": 30}


def release_reprove(priority: dict | None = None) -> dict:
    """Take the idle gate off the QUEUED re-prove jobs (payload `reprove`)
    and order them by area (REPROVE_PRIORITY). Touches only those rows'
    `idle` flag and `priority`; mcp/idle.py's definition of an idle stack is
    unchanged, and every other idle-gated job keeps its gate. The operator's
    one-off: the stack was never idle for 15 minutes, and the card is theirs
    to use ("GPU go", 2026-10-06)."""
    import skill_select
    prio = REPROVE_PRIORITY if priority is None else priority
    q = _queue()[0]
    con = jobs._db()
    done: dict[str, int] = {}
    try:
        rows = con.execute("SELECT id, payload FROM jobs WHERE queue=? AND "
                           "state='queued'", (q,)).fetchall()
        for jid, payload in rows:
            pl = json.loads(payload or "{}")
            if not pl.get("reprove"):
                continue
            ver = skills.version(pl.get("skill"), pl.get("version")) or {}
            area = skill_select.area_of(ver.get("classify") or {})
            pl["idle"] = False
            con.execute("UPDATE jobs SET payload=?, priority=? WHERE id=? "
                        "AND state='queued'",
                        (json.dumps(pl), int(prio.get(area, 0)), jid))
            done[area] = done.get(area, 0) + 1
    finally:
        con.close()
    return {"released": sum(done.values()), "by_area": done,
            "priority": {a: prio.get(a, 0) for a in done}}


def run_reprove_here(limit: int | None = None, worker: str | None = None,
                     stop_after_s: float | None = None) -> dict:
    """Run queued re-prove jobs IN THIS PROCESS, highest priority first,
    claiming each row atomically (so a worker and any number of these never
    take the same one) and finishing it like the worker does. The process has
    no tier table, so its generations go to the model named in model.py
    (`bonsai`, the MAIN card) instead of the locked model's helper on the
    A4000: the main card decodes about 74 tok/s (AGENTS.md; kv_rank lane
    arms), the A4000 helper about a third of that, and a decider read beside
    it on the same A4000 cut its decode to ~6 tok/s (measured 2026-10-06,
    slot 0 n_decoded 2587 -> 2644 in 10 s while slot 1 read). The coordinator's
    GPU window ("GPU go", 2026-10-06): the card is idle."""
    import socket
    import threading
    worker = worker or f"{socket.gethostname()}:{os.getpid()}:reprove-here"
    q = _queue()[0]
    done, t0 = [], time.time()

    def claim():
        con = jobs._db()
        try:
            con.execute("BEGIN IMMEDIATE")
            for r in con.execute(
                    "SELECT id, payload, attempts FROM jobs WHERE queue=? "
                    "AND state='queued' AND (not_before IS NULL OR "
                    "not_before <= ?) ORDER BY priority DESC, created ASC",
                    (q, time.time())).fetchall():
                pl = json.loads(r[1] or "{}")
                if not pl.get("reprove"):
                    continue
                now = time.time()
                con.execute("UPDATE jobs SET state='running', started=?, "
                            "heartbeat=?, worker=?, card='gpu', "
                            "attempts=attempts+1 WHERE id=? AND "
                            "state='queued'", (now, now, worker, r[0]))
                con.execute("COMMIT")
                return {"id": r[0], "payload": pl}
            con.execute("COMMIT")
            return None
        except Exception:                                        # noqa: BLE001
            try:
                con.execute("ROLLBACK")
            except Exception:                                    # noqa: BLE001
                pass
            raise
        finally:
            con.close()

    class Ctx:
        def __init__(self, jid):
            self.jid = jid

        def beat(self, progress=None):
            jobs.beat(self.jid, progress)

    while limit is None or len(done) < limit:
        if stop_after_s and time.time() - t0 > stop_after_s:
            break
        job = claim()
        if job is None:
            break
        stop = threading.Event()

        def hb(jid=job["id"], ev=stop):
            while not ev.wait(60):
                try:
                    jobs.beat(jid)
                except Exception:                                # noqa: BLE001
                    pass
        threading.Thread(target=hb, daemon=True).start()
        try:
            res = _handle_reprove(job, Ctx(job["id"]))
            jobs.finish(job["id"], res)
            done.append({"job": job["id"], "skill": job["payload"]["skill"],
                         "verdict": res.get("prove"),
                         "rearmed": bool(res.get("rearmed")),
                         "seconds": res.get("seconds")})
            print(f"  reprove {job['payload']['skill']}: "
                  f"{res.get('prove')}"
                  f"{' REARMED' if res.get('rearmed') else ''} "
                  f"{res.get('seconds')} s", flush=True)
        except Exception as e:                                   # noqa: BLE001
            jobs.fail(job["id"], f"{type(e).__name__}: {e}")
            print(f"  reprove {job['payload'].get('skill')}: failed "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            stop.set()
    return {"done": len(done), "rows": done,
            "seconds": round(time.time() - t0, 1)}


def _handle_reprove(job: dict, ctx) -> dict:
    p = job.get("payload") or {}
    sid, v = p.get("skill"), p.get("version")
    s = skills.get(sid) if sid else None
    ver = skills.version(sid, v) if s and v else None
    why = _reprove_ok(s, ver, int(v or 0), include_unproven=True)
    if why:
        return {"skipped": f"reprove of {sid} v{v}: {why}"}
    v = int(v)
    rec = prove(sid, v, ver, mode="reprove", beat=getattr(ctx, "beat", None))
    old = record_of(ver)
    rec["reproved"] = {"from": {k: old.get(k) for k in
                                ("verdict", "mode", "at", "seconds")},
                       "reason": str(ver.get("reason") or "")[:300]}
    if rec["verdict"] == "worse":
        skills.quarantine(sid, v, quarantine_reason(rec),
                          **_fields(sid, v, rec))
        return dict(_x(rec), quarantined=True)
    if rec["verdict"] in ("unproven", "not_run"):
        # No decided proof: it stays quarantined, its record updated.
        reason = (unproven_reason(rec) if rec["verdict"] == "unproven"
                  else str(ver.get("reason")))
        skills.quarantine(sid, v, reason, **_fields(sid, v, rec))
        return dict(_x(rec), quarantined=True)
    skills.rearm(sid, v, **_fields(sid, v, rec))
    try:
        import skill_select
        skill_select.refresh_triggers()
    except Exception as e:                                       # noqa: BLE001
        print(f"  prove: trigger index not rebuilt after rearm: "
              f"{type(e).__name__}: {e}"[:200], file=sys.stderr, flush=True)
    return dict(_x(rec), rearmed=True)


# ---------------------------------------------------------------------------
# The backlog: the armed library, proved on an idle stack.
# ---------------------------------------------------------------------------
def _queue() -> tuple[str, str]:
    return tuple(skills.JOBS.get("prove") or JOB)


def _live_jobs(backlog_only: bool = False) -> dict[str, int]:
    """{skill id: live prove jobs}. A queued or running job of this queue."""
    q = _queue()[0]
    con = jobs._db()
    try:
        rows = con.execute(
            "SELECT dataset, payload FROM jobs WHERE queue=? AND state IN "
            "('queued','running')", (q,)).fetchall()
    finally:
        con.close()
    out: dict[str, int] = {}
    for ds, payload in rows:
        try:
            pl = json.loads(payload or "{}")
        except ValueError:
            pl = {}
        if backlog_only and not pl.get("backlog"):
            continue
        sid = pl.get("skill") or str(ds or "").replace("skill:", "", 1)
        out[sid] = out.get(sid, 0) + 1
    return out


def backlog(limit: int | None = None) -> tuple[list[dict], dict]:
    """(rows to prove, skipped counts): every armed skill whose SERVED
    version has no decided proof, the most recently updated version first
    (new and changed skills first: the operator's order)."""
    live = _live_jobs()
    rows, skipped = [], {"proved": 0, "live_job": 0}
    for s in skills.armed(include_package_only=True):
        ver = skills.version(s["id"], s["version"]) or {}
        r = record_of(ver)
        if r.get("verdict") in PROVED and \
                int(r.get("skill_version") or 0) == int(s["version"]):
            skipped["proved"] += 1
            continue
        if live.get(s["id"]):
            skipped["live_job"] += 1
            continue
        rows.append({"skill": s["id"], "version": int(s["version"]),
                     "name": s.get("name"),
                     "updated": float(ver.get("updated") or 0),
                     "armed_at": float(ver.get("armed_at") or 0)})
    rows.sort(key=lambda r: (-r["updated"], -r["armed_at"], r["skill"]))
    return (rows if limit is None else rows[:max(0, int(limit))]), skipped


def enqueue_backlog(limit: int | None = None) -> dict:
    """Enqueue idle-gated gpu-lane prove jobs for the armed library (see
    `backlog`). Each job's payload carries `idle`: worker.run_one defers it
    while the stack is busy (mcp/idle.py), without an attempt."""
    rows, skipped = backlog(limit)
    q, lane = _queue()
    out = []
    for r in rows:
        jid = jobs.add(q, {"skill": r["skill"], "version": r["version"],
                           "stage": "prove", "backlog": True, "idle": True},
                       lane=lane, dataset=f"skill:{r['skill']}",
                       stage="prove")
        out.append({"skill": r["skill"], "version": r["version"],
                    "job": jid})
    return {"enqueued": out, "skipped": skipped, "estimate": estimate()}


def schedule_backlog() -> str | None:
    """The worker's once-a-minute hook: ONE backlog job at a time, when none
    is live, so a proof never floods the gpu lane ahead of the pipeline's
    own stages. Off with YAMADORI_SKILL_PROVE_BACKLOG=0."""
    if not BACKLOG or _live_jobs(backlog_only=True):
        return None
    got = enqueue_backlog(limit=1)["enqueued"]
    return got[0]["job"] if got else None


def estimate() -> dict:
    """What the backlog costs, from the proofs already MEASURED: their
    seconds per skill (n, mean, max) and the armed skills still unproved.
    No figure of ours: with no measured proof the estimate says so."""
    con = skills._db()
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(skill_versions)")}
        sel = "validate" + (", prove" if "prove" in cols else "")
        raw = con.execute(f"SELECT {sel} FROM skill_versions WHERE "
                          "validate LIKE '%\"prove\"%'"
                          + (" OR prove LIKE '%\"verdict\"%'"
                             if "prove" in cols else "")).fetchall()
    finally:
        con.close()
    secs = []
    for r in raw:
        ver = {"validate": json.loads(r[0] or "{}")}
        if len(r) > 1:
            ver["prove"] = json.loads(r[1] or "{}")
        rec = record_of(ver)
        if rec.get("verdict") in PROVED and rec.get("seconds"):
            secs.append(float(rec["seconds"]))
    remaining = len(backlog()[0])
    out: dict = {"measured": len(secs), "remaining": remaining}
    if secs:
        mean = sum(secs) / len(secs)
        out.update(mean_seconds=round(mean, 1), max_seconds=round(max(secs),
                                                                  1),
                   remaining_seconds=round(mean * remaining, 1),
                   basis=f"the mean of {len(secs)} measured proof(s)")
    else:
        out["remaining_seconds"] = None
        out["basis"] = "no proof measured yet: unknown until the first runs"
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="the skill PROVE backlog")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--enqueue", type=int, metavar="N",
                    help="enqueue N idle-gated backlog proofs")
    ap.add_argument("--reprove", action="store_true",
                    help="enqueue idle-gated re-proofs of every skill PROVE "
                    "quarantined under the one-sample rule (repeat rule)")
    ap.add_argument("--limit", type=int, default=None,
                    help="with --reprove: at most N skills")
    ap.add_argument("--include-unproven", action="store_true",
                    help="with --reprove: also the `unproven` quarantines")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --reprove: list, enqueue nothing")
    ap.add_argument("--release-reprove", action="store_true",
                    help="take the idle gate off the queued re-proofs and "
                    "order them by area (REPROVE_PRIORITY)")
    ap.add_argument("--promote-package-only", action="store_true",
                    help="run the activation tests again for every "
                    "package-only skill and make each one that passes a "
                    "full skill")
    ap.add_argument("--run-reprove-here", action="store_true",
                    help="run the queued re-proofs in THIS process on the "
                    "main card's model (run_reprove_here), highest priority "
                    "first")
    a = ap.parse_args()
    if a.promote_package_only:
        import skill_pipeline
        rows = [skill_pipeline.promote_package_only(sid)
                for sid in skills.package_only_ids()]
        print(json.dumps(rows, indent=1))
    elif a.run_reprove_here:
        print(json.dumps(run_reprove_here(a.limit), indent=1))
    elif a.release_reprove:
        print(json.dumps(release_reprove(), indent=1))
    elif a.reprove:
        print(json.dumps(reprove_quarantined(
            a.limit, include_unproven=a.include_unproven,
            dry_run=a.dry_run), indent=1))
    elif a.enqueue:
        print(json.dumps(enqueue_backlog(limit=a.enqueue), indent=1))
    else:
        print(json.dumps(estimate(), indent=1))
