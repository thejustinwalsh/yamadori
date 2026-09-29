#!/usr/bin/env python
"""Tests shaped per skill, stored with it: does selection pick it for the
requests it is for, and only those?

Operator, 2026-09-26: "a pipeline ... that shapes tests and distills the
skill for activation and selection."

TWO KINDS (tests.json in the skill's folder; `validate.tests` on the store)

  activation   request shapes that SHOULD select the skill and near misses
               that should NOT. Each is a CASE:

                 {"text": "the user turn",
                  "files": ["js/game.js", "index.html"],  # tool-call paths
                  "tool_output": "TypeError: x is not a function",
                  "route_class": "code_edit",              # all optional
                  "tools": ["terminal", "vision_analyze"]} # the client's

               A case becomes a small conversation (`messages_of`): when it
               names files or tool output, an earlier agent step that wrote
               those files and got that output, then the user turn. It is
               run through skill_classify.request_signals + match and
               skill_select.verdict -- the DETERMINISTIC stages of the real
               selector. The embedding stage is not run here (it needs the
               live embedder); what it would add is recorded as
               `embedding: not run`.

  behaviour    a short prompt and a check an answer that followed the skill
               would pass ({"kind": "regex" | "not_regex", "pattern"}).
               STORED, NOT RUN: they are the A/B for a live run (with vs
               without the skill) and are only validated here (the pattern
               compiles, the prompt is short).

THE ARMING RULE (a CHOICE, unmeasured)

  every `should` case      the skill is a candidate (verdict inject or ask)
  every `should_not` case  the skill is not a candidate (verdict none)

A skill that fails its activation tests does not arm: it is QUARANTINED
with the failing cases in its reason (operator, 2026-09-26). The score
(cases passed / cases run) is recorded either way. Against a POOL (the armed
skills) each `should` case also records the skill's rank among the
candidates -- whether it would survive the per-turn cap -- which is reported,
not gated: a pool changes as skills arm, a skill's own tests do not.

GENERATED TESTS (`generate`)

Every skill gets a deterministic floor from its own rule: a should case per
primary cue (a fence or file of its language, an import of its framework,
its topics in the text, its phase's words, its situations' facts) and near
misses that break exactly one gate (another language, its topics absent, the
wrong phase). A model-proposed set (skill_prompts TESTS) is added on top for
sources that were distilled; authored skills ship their own.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skill_limits as L  # noqa: E402

# A near miss's language: one the rule does not use, far enough that its
# fence and extension name nothing the rule holds.
_OTHER = {"typescript": "python", "javascript": "python", "html": "python",
          "css": "python", "rust": "python", "c": "python", "cpp": "python",
          "zig": "python", "wgsl": "python", "glsl": "python",
          "python": "rust", "sql": "rust"}
_FRAMEWORK_LANG = {"react": "tsx", "r3f": "tsx", "threejs": "ts",
                   "typegpu": "ts", "webgpu": "ts", "wasm_bindgen": "rust",
                   "emscripten": "cpp", "webassembly": "rust",
                   "cbindgen": "rust", "tailwind": "tsx", "vitest": "ts",
                   "jest": "ts", "pytest": "python"}
_ARTIFACT_PHRASE = {"tests": "Write unit tests for the parser module",
                    "docs": "Write a README for this project",
                    "slides": "Make a slide deck for the launch",
                    "ui_design": "Design the landing page layout",
                    "data": "Write the SQL schema migration",
                    "config": "Write the Dockerfile for the service",
                    "prose": "Write a blog post about the release"}
_PHASE_TEXT = {"plan": "Plan how to structure this before writing it.",
               "implement": "Implement it.",
               "debug": "It fails with an error and nothing works; fix it.",
               "verify": "Verify that it works and add a test.",
               "refactor": "Refactor this module and clean it up.",
               "review": "Review this code."}
_SITUATION_CASE = {
    "error_output": {"tool_output": "Uncaught TypeError: update failed "
                                    "(exit code 1)"},
    "call_mismatch": {"tool_output": "Uncaught TypeError: STATE.step is not "
                                     "a function"},
    "file_line": {"text_add": "The failure is at src/main.js:42."},
    "multi_file": {"files": ["src/main.js", "src/util.js"]},
    "screenshot_made": {"tool_output": "screenshot saved to shots/view.png"},
}


def messages_of(case: dict) -> list[dict]:
    """A case as the conversation a selector would see."""
    text = str(case.get("text") or "")
    files = [str(f) for f in case.get("files") or []]
    out_text = str(case.get("tool_output") or "")
    if not files and not out_text:
        return [{"role": "user", "content": text}]
    calls = [{"id": f"call_{i}", "type": "function",
              "function": {"name": "write_file", "arguments": json.dumps(
                  {"path": f, "content": "// ..."})}}
             for i, f in enumerate(files or ["notes.txt"])]
    msgs = [{"role": "user", "content": "Continue the task."},
            {"role": "assistant", "content": "", "tool_calls": calls}]
    for i, c in enumerate(calls):
        msgs.append({"role": "tool", "tool_call_id": c["id"],
                     "content": out_text if i == len(calls) - 1 and out_text
                     else "ok"})
    msgs.append({"role": "user", "content": text})
    return msgs


def run_case(case: dict, rule: dict) -> dict:
    """{verdict, strength, score, why, gate} for one case, deterministic
    stages only."""
    import skill_classify
    import skill_select
    sig = skill_classify.request_signals(messages_of(case),
                                         case.get("route_class"),
                                         case.get("tools"))
    det = skill_classify.match(rule or {}, sig)
    v = skill_select.verdict(det["strength"], None)
    return {"verdict": v, "strength": det["strength"],
            "score": round(det["score"], 3), "why": det["why"][:4],
            "gate": det.get("gate")}


def run(tests: dict, rule: dict, *, skill_id: str = "",
        pool: list[dict] | None = None) -> dict:
    """Run a skill's activation tests. {passed, score, n, failures,
    cases: [...], embedding, pool}."""
    act = (tests or {}).get("activation") or {}
    should = [c for c in act.get("should") or [] if isinstance(c, dict)]
    should_not = [c for c in act.get("should_not") or []
                  if isinstance(c, dict)]
    cases, failures = [], []
    ok_n = 0
    for kind, group in (("should", should), ("should_not", should_not)):
        for c in group:
            r = run_case(c, rule)
            good = (r["verdict"] != "none") if kind == "should" else \
                (r["verdict"] == "none")
            ok_n += good
            row = {"kind": kind, "text": str(c.get("text") or "")[:160],
                   "ok": good, **r}
            if kind == "should" and pool is not None and skill_id:
                row["pool_rank"] = pool_rank(c, skill_id, pool)
            cases.append(row)
            if not good:
                failures.append(
                    f"{kind}: {str(c.get('text') or '')[:80]!r} -> "
                    f"{r['verdict']}" + (f" ({r['gate']})" if r.get("gate")
                                         else ""))
    n = len(should) + len(should_not)
    enough = len(should) >= L.MIN_SHOULD and len(should_not) >= \
        L.MIN_SHOULD_NOT
    if not enough:
        failures.append(f"too few tests: {len(should)} should / "
                        f"{len(should_not)} should_not (at least "
                        f"{L.MIN_SHOULD} / {L.MIN_SHOULD_NOT})")
    out = {"passed": not failures, "score": round(ok_n / n, 3) if n else 0.0,
           "n": n, "failures": failures[:8], "cases": cases,
           "embedding": "not run (deterministic stages only)"}
    if pool is not None and skill_id:
        ranks = [c.get("pool_rank") for c in cases if c["kind"] == "should"]
        within = sum(1 for r in ranks if r and r <= L.MAX_SKILLS_PER_TURN)
        out["pool"] = {"size": len(pool), "should_within_cap": within,
                       "should": len(ranks)}
    return out


def pool_rank(case: dict, skill_id: str, pool: list[dict]) -> int | None:
    """Where the skill ranks by confidence among the pool's candidates for
    this case (1 = first), or None when it is not a candidate."""
    import skill_classify
    import skill_select
    sig = skill_classify.request_signals(messages_of(case),
                                         case.get("route_class"),
                                         case.get("tools"))
    rows = []
    for s in pool:
        det = skill_classify.match(s.get("rule") or {}, sig)
        if skill_select.verdict(det["strength"], None) != "none":
            rows.append((-skill_select.confidence(det, None),
                         s["id"]))
    rows.sort()
    for i, (_c, sid) in enumerate(rows, 1):
        if sid == skill_id:
            return i
    return None


# ---------------------------------------------------------------------------
# Behaviour checks: stored, validated, never run here.
# ---------------------------------------------------------------------------
def clean_behaviour(checks) -> tuple[list[dict], list[str]]:
    """(the checks that are well formed, why each other one was dropped)."""
    out, dropped = [], []
    for c in checks or []:
        if not isinstance(c, dict):
            continue
        prompt = str(c.get("prompt") or "").strip()
        chk = c.get("check") if isinstance(c.get("check"), dict) else {}
        kind, pat = chk.get("kind"), str(chk.get("pattern") or "")
        why = None
        if not prompt or len(prompt) > 1200:
            why = "a prompt of 1 to 1,200 characters"
        elif kind not in ("regex", "not_regex") or not pat:
            why = "check.kind regex or not_regex, with a pattern"
        else:
            try:
                re.compile(pat)
            except re.error as e:
                why = f"the pattern does not compile ({e})"
        if why:
            dropped.append(f"behaviour check {prompt[:40]!r}: needs {why}")
            continue
        out.append({"prompt": prompt, "check": {"kind": kind,
                                                "pattern": pat},
                    "why": str(c.get("why") or "")[:200]})
    return out, dropped


def grade_behaviour(check: dict, answer: str) -> bool:
    """For the live A/B (not run in this pass): does an answer pass?"""
    rx = re.compile(check["check"]["pattern"], re.I | re.S)
    hit = bool(rx.search(answer or ""))
    return hit if check["check"]["kind"] == "regex" else not hit


# ---------------------------------------------------------------------------
# Generation: the deterministic floor.
# ---------------------------------------------------------------------------
def _cue(rule: dict) -> dict:
    """The primary cue of a rule as case fields: a fence, a file, an
    import. {} when the rule's primary is a non-code artifact or domain."""
    import skill_classify as C
    a = C.applies_to(rule)
    if a["frameworks"]:
        fw = C.BY_ID[a["frameworks"][0]]
        lang = _FRAMEWORK_LANG.get(fw.id, "ts")
        pkg = fw.packages[0] if fw.packages else fw.name.lower()
        if lang == "rust":
            code = f"use {pkg.replace('-', '_')}::prelude::*;"
        elif lang == "python":
            code = f"import {pkg}"
        elif lang == "cpp":
            code = f"#include <{pkg}.h>  // {fw.name}"
        else:
            code = f"import {{ x }} from \"{pkg}\";"
        return {"fence": lang, "code": code}
    if a["languages"]:
        t = C.BY_ID[a["languages"][0]]
        return {"fence": (t.fences or t.exts or (t.id,))[0],
                "file": f"src/main.{(t.exts or ('txt',))[0]}"}
    if a["artifacts"] == ["code"]:
        # Code in any language: the router's code class is the signal.
        return {"route_class": "code_generation"}
    return {}


def _with_cue(text: str, cue: dict, code: str = "") -> str:
    if cue.get("fence"):
        body = "\n".join(x for x in (cue.get("code") or "", code) if x)
        return f"{text}\n\n```{cue['fence']}\n{body or '// ...'}\n```"
    return text


def generate(rule: dict, *, description: str = "",
             title: str = "") -> dict:
    """The deterministic activation tests for a rule."""
    import skill_classify as C
    a = C.applies_to(rule)
    g = C.gates(rule)
    cue = _cue(rule)
    noncode = [x for x in a["artifacts"] if x != "code"]
    topics = g["topics"]
    phase = g["phases"][0] if g["phases"] else None
    base = (description or title or C.condition_text(rule) or "this")
    base = re.sub(r"^use (?:it |this )?when\s+", "", base.strip(),
                  flags=re.I).rstrip(".")
    extra: dict = {}
    for s in g["situations"][:1]:
        extra = dict(_SITUATION_CASE[s])
    files = list(extra.get("files") or [])
    for t in g["all_of"]:
        term = C.BY_ID.get(t)
        if term and term.exts:
            files.append(f"index.{term.exts[0]}" if t == "html"
                         else f"src/extra.{term.exts[0]}")
    if a["languages"] and (files or extra.get("tool_output")):
        t0 = C.BY_ID[a["languages"][0]]
        if t0.exts:
            files.append(f"src/app.{t0.exts[0]}")
    tail = " ".join(x for x in (_PHASE_TEXT.get(phase, "") if phase else "",
                                extra.get("text_add", "")) if x)
    topic_line = (" It uses " + ", ".join(topics[:2]) + ".") if topics else ""

    # A rule keyed on the harness: the should cases carry its tools.
    tools = list(dict.fromkeys(g["tools_all"] + g["tools_any"][:1]))

    def case(text, **kw):
        c = {"text": text}
        if tools:
            c["tools"] = tools
        if cue.get("route_class"):
            c["route_class"] = cue["route_class"]
        if files:
            c["files"] = list(dict.fromkeys(files))
        if extra.get("tool_output"):
            c["tool_output"] = extra["tool_output"]
        c.update(kw)
        return c

    should, should_not = [], []
    if noncode and not cue:
        art = noncode[0]
        phrase = _ARTIFACT_PHRASE.get(art, "Write it")
        should.append(case(f"{phrase}: {base}.{topic_line} {tail}".strip()))
        should.append(case(f"{phrase}.{topic_line} {tail}".strip()))
    else:
        code = "\n".join(f"{t}" + ("" if t.endswith(")") else "")
                         for t in topics[:2])
        # An all_of FRAMEWORK has no file extension to cue it (react, r3f):
        # its import does (2026-09-26: a koota skill gated on React Three
        # Fiber failed its own generated cases).
        for t in g["all_of"]:
            term = C.BY_ID.get(t)
            if term and not term.exts and term.packages:
                code += f"\nimport {{ y }} from \"{term.packages[0]}\";"
        should.append(case(_with_cue(
            f"{base[0].upper()}{base[1:]}.{topic_line} {tail}".strip(),
            cue, code)))
        if cue.get("file") and not files:
            should.append(case(f"Update {cue['file']}.{topic_line} "
                               f"{tail}".strip(), files=[cue["file"]]))
        else:
            should.append(case(_with_cue(
                f"Help me with this.{topic_line} {tail}".strip(), cue,
                code)))
    # Near misses: break exactly one gate each.
    langs = a["languages"] + [f for f in a["frameworks"]]
    other = next((_OTHER[x] for x in a["languages"] if x in _OTHER), None)
    if other is None and a["frameworks"]:
        other = "python" if _FRAMEWORK_LANG.get(a["frameworks"][0]) != \
            "python" else "rust"
    if other and not any(C.code_shaped(t) for t in topics):
        # Another language, the same words -- less the names of the rule's
        # own languages and frameworks, which would make it no near miss.
        ot = C.BY_ID[other]
        near = f"{base[0].upper()}{base[1:]}.{topic_line} {tail}"
        for x in langs:
            near = C._WORDS[x].sub("it", near)
        should_not.append({"text": f"{near.strip()}\n\n```{ot.fences[0]}\n"
                                   f"# ...\n```"})
    elif other:
        should_not.append({"text": f"Write a small {C.BY_ID[other].name} "
                                   f"command-line tool that reads a file."
                                   f"\n\n```{C.BY_ID[other].fences[0]}\n"
                                   f"# ...\n```"})
    if topics:
        # The right language, none of its topics.
        should_not.append({"text": _with_cue(
            "Write a function that sorts a list of numbers.", cue,
            "// sort")})
    if phase and set(g["phases"]) != set(C.PHASES):
        wrong = next((p for p in ("review", "plan", "refactor", "implement")
                      if p not in g["phases"]), None)
        if wrong:
            should_not.append({"text": _with_cue(
                f"{_PHASE_TEXT[wrong].replace('it', 'the module')}",
                cue)})
    if noncode and len(should_not) < L.MIN_SHOULD_NOT:
        should_not.append({"text": "Write a small Python command-line tool "
                                   "that reads a file.\n\n```python\n"
                                   "# ...\n```"})
    if tools and should:
        # The same request from a harness without those tools.
        miss = {k: v for k, v in should[0].items() if k != "tools"}
        miss["tools"] = ["bash", "read", "edit"]
        should_not.insert(0, miss)
    if len(should_not) < L.MIN_SHOULD_NOT:
        should_not.append({"text": "Write a short essay about the history "
                                   "of the printing press."})
    if len(should_not) < L.MIN_SHOULD_NOT:
        should_not.append({"text": "What is the capital of Australia?"})
    _ = langs
    # Every case written (no case caps since 2026-09-27,
    # docs/CONSTANTS-AUDIT.md "case caps").
    return {"activation": {"should": should,
                           "should_not": should_not,
                           "generated": "skill_tests.generate"},
            "behaviour": []}


def merge(base: dict, extra: dict) -> dict:
    """Two test sets as one: cases de-duplicated by text (no caps since
    2026-09-27, docs/CONSTANTS-AUDIT.md)."""
    out = {"activation": {"should": [], "should_not": []}, "behaviour": []}
    for src in (base or {}, extra or {}):
        act = src.get("activation") or {}
        for k in ("should", "should_not"):
            for c in act.get(k) or []:
                if isinstance(c, str):
                    c = {"text": c}
                if isinstance(c, dict) and c.get("text") and c["text"] not in [
                        x["text"] for x in out["activation"][k]]:
                    out["activation"][k].append(c)
        out["behaviour"] += list(src.get("behaviour") or [])
    return out


def from_model(reply_obj: dict) -> dict:
    """A tests-template reply as a test set (strings become cases)."""
    def cases(xs):
        return [{"text": str(x)[:1200]} if isinstance(x, str) else x
                for x in (xs or []) if (isinstance(x, str) and x.strip())
                or (isinstance(x, dict) and x.get("text"))][:4]
    beh, _dropped = clean_behaviour(reply_obj.get("behaviour"))
    return {"activation": {"should": cases(reply_obj.get("should")),
                           "should_not": cases(reply_obj.get("should_not"))},
            "behaviour": beh}
