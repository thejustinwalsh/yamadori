#!/usr/bin/env python
"""Fan-out: one main brain, one second brain, candidates graded in sequence.

THE DESIGN (operator decision, 2026-09-23)

"One main brain, one second brain; sequential; grade after two; tie-breaker
with both candidates' details." At most TWO contexts are live at once: the
conversation (main role, 5/8 of the KV pool) and ONE helper (3/8, the
second brain deep thinking also uses -- mcp/budget.py). Fan-out work is
handed to the second brain one step at a time:

  A  the original answer. Main role, full 5/8. It already exists when
     fan-out runs (proxy._fan_out), and it is always candidate 0.
  B  one more answer, generated in the HELPER role with the helper's whole
     3/8 (tiers.rebudget(role="helper", share_n=1)), as the one
     second-brain runner's `alternative` job (shomen.run, on the helper
     lane). The "direct" variant plus a concept seed.
     Vendor sampling is what the shaped payload carries (tiers.apply).
  GRADE after two, with the code check (never execution):
     - exactly one parses                      -> it wins, stop
       ("clear: only one parses")
     - both parse, similarity >= AGREE          -> A wins, stop
       ("clear: agreement")
     - otherwise (both parse and disagree, or neither parses) -> C
  C  the TIE-BREAKER, helper role, the runner's `tiebreak` job. Its
     prompt is the task plus BOTH candidates' code and each one's check
     result (parses? which syntax errors, on which lines), and it asks for
     the single best final solution, fixing what is wrong. Then consensus
     over [A, B, C]: the code medoid among the parsing candidates, C
     preferred on a tie (see "HOW THE WINNER IS PICKED" below).

A prose answer (no fenced block in a code language in A) keeps today's
behaviour: B runs, sequentially in the helper role, the path/symbol vote over
[A, B] is RECORDED, and the original is delivered. No tie-breaker for prose.

NOTHING THE SECOND BRAIN WROTE IS THROWN AWAY (operator decision,
2026-09-23). `handback()` returns what the other candidates say that the
delivered answer does not: for prose, B's differing points (an "alternative
view"); for code, one line naming the APIs each loser used that the winner
does not. proxy._fan_out decides where that crosses -- see its docstring.

The tier's `fanout` value (3 at high and max) is the MOST candidates, A
included: A + B + an optional C. fanout:1 is off; a forced 3 always runs the
adaptive procedure; 2 allows A + B and no tie-breaker.

WHY (what it replaced)

Until 2026-09-23 fan-out ran n variants CONCURRENTLY in a thread pool on the
MAIN role, with main's 5/8 split n ways (tiers.rebudget(share_n=n)). So each
variant thought with about a third of the room the original had, and up to n
extra contexts were live beside the conversation and any deep thinking --
the "two consumers on one card" condition AGENTS.md warns about, inside one
request. The batching throughput that justified it (N=4 at 2.7x the wall
clock of one, measured) bought candidates that each had less room to think.
The sequential design gives every extra candidate the helper's whole 3/8 and
stops after two when the check already separates them.

AGREE is 0.80, a CHOICE, not a measurement: nothing in this repo has graded
how often two parsing candidates at >= 0.80 similarity are both right. Measure
it on benchmark data (bench/domain grades every candidate) before citing it,
PROTOCOL rule 10.

WHY AGREEMENT AND NOT A SCORER

Published results say a trained picker does not beat agreement yet: on
MATH500 at 16 generations, plain self-consistency scored 86.00 against 85.00
for best-of-N with an external reward model. So the grade is a code check and
a similarity, and a learned selector is an upgrade, not a prerequisite.

MEASURED -- WHEN NOT TO USE THIS

On eight file-location questions against koota, four-way consensus scored 7/8
against 7/8 for a single answer, with ZERO discordant pairs, at 3.2x the wall
clock (the old concurrent design). "Where is X defined" has one right answer,
so there is no quality variance for a second candidate to exploit. The gate
is in mcp/selection.py (`fanout_n`): lookups answer once; design questions and
code-writing tasks may fan out at high and max.

HOW THE WINNER IS PICKED AMONG CODE CANDIDATES (2026-09-23)

The old vote was over file paths and backticked symbols with the SHORTEST
answer winning a tie. A coding answer names no path, so every candidate scored
0 and the tersest won (5 of 6 fan-out records in bench/**/*.jsonl on
2026-09-23 were decided by length alone -- a count, not a graded cost). So a
code answer is picked by `consensus()` in three steps, and only when most
candidates carry a fenced block in a code language (code_check.CODE_LANGS):

  1. EXTRACT each candidate's code: the largest fenced block in the dominant
     language (ties to the later block), as bench/domain/grade.py reads a
     reply. An untagged block counts only when some candidate tagged the
     language; it is never guessed.
  2. VALIDITY: keep candidates whose code parses -- code_check.check(...,
     run_format=False) (Python's compile(), which builds a code object and
     runs nothing, plus tree-sitter), else tree-sitter directly. A truncated
     candidate (unclosed fence or finish_reason "length") is set aside
     whenever a complete one parses. Nothing here ever EXECUTES candidate
     code. When the question carries fenced code (code_check.prompt_code),
     a candidate PARSES if it parses on its own OR appended directly after
     one of those blocks (code_check.check_against_prompt, the general
     rule; operator decision 2026-09-23). Checked on its own only, every
     continuation of starter code fails, and the grade could never separate
     two (lb-20260923-minp0, yamadori-xhigh: 10 of 10 completion rows graded
     A and B "neither parses"). No phrasing is read. Per candidate,
     check_mode records which form parsed: "standalone", "appended",
     "neither", or None when the question carries no code.
  3. CONSENSUS: the medoid of the valid pool -- highest mean similarity to
     the others. Similarity is the multiset Jaccard of token trigrams over
     tree-sitter leaves, comments and whitespace dropped, string literals
     collapsed, and every name the candidate binds collapsed to one token, so
     a renamed variable still agrees. Ties (within SIM_TIE) go to the
     preferred candidate when there is one (the tie-breaker C), then fewer
     checker warnings, a complete answer, and the length nearest the median
     -- never the shortest.

If nothing parses, the path vote runs with ties to the LONGEST answer and the
result says `selection: "fallback"`; a fallback is recorded, never delivered.
SIM_TIE, NGRAM and AGREE are choices, not measurements.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import textwrap
from collections import Counter

# The MODEL SERVER, not the proxy. Port 1234 is the proxy now, and fan-out
# running through it would re-enter the tool loop once per variant: N times the
# work, N nested tool loops, and a request that may never return. This is the
# same stale-default class of bug that left the indexer writing 18,750 zero
# vectors and the benchmark querying a dead endpoint. It was never hit here
# only because nothing calls fan-out yet.
UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")

# Each variant is a different way of approaching the same question, not a
# different random seed. `nudge` is appended to the system message.
#
# NO PER-VARIANT TEMPERATURE (2026-09-23). These carried 0.3/0.3/0.7/0.2,
# off-spec for a thinking model whose vendor sampling is temperature 1.0
# (tiers.VENDOR_SAMPLING, which tiers.apply enforces on every request). Every
# variant now samples at the vendor settings the shaped payload already
# carries; diversity comes from the nudge and the concept seed, which is the
# stated intent ("diverse variants and not just temperature").
#
# SEQUENTIAL (2026-09-23): only VARIANTS[0], "direct", is generated now -- it
# is candidate B, and its diversity from the original comes from the concept
# seed. The others are kept, named, for a benchmark arm that wants them.
VARIANTS = [
    {"name": "direct", "nudge": ""},
    {"name": "evidence",
     "nudge": "\n\nGround every claim in a file you have actually read. "
              "Name the path and line."},
    {"name": "skeptical",
     "nudge": "\n\nConsider the obvious answer, then check whether a second "
              "place in the codebase is a better fit before committing."},
    {"name": "terse",
     "nudge": "\n\nAnswer in as few words as the question allows. No preamble."},
]

# What a consensus is taken OVER. Free prose cannot be voted on, but the
# artefacts that matter in these answers can.
_PATH = re.compile(r"[\w./\\-]+\.(?:ts|tsx|js|jsx|mjs|cjs|rs|c|h|cpp|hpp|py|go|zig|wgsl|glsl|lua|json|toml)")
_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})`")


def claims(text: str) -> tuple[set, set]:
    """The checkable assertions in an answer: which files, which symbols."""
    paths = {p.replace("\\", "/").lstrip("./") for p in _PATH.findall(text or "")}
    syms = set(_SYMBOL.findall(text or ""))
    return paths, syms


# Seed concepts come from concept_seed, which draws a random direction in
# embedding space and takes the nearest real word. A curated list was tried
# first and is the author's taste in shuffled order -- the same narrow, lumpy
# slice of concept space every time. Random directions in high dimensions are
# near-orthogonal, so consecutive draws genuinely differ.
import concept_seed  # noqa: E402
import repeats  # noqa: E402


def _one(payload: dict, variant: dict, timeout: int) -> dict:
    body = json.loads(json.dumps(payload))     # deep copy
    msgs = body.get("messages") or []
    if variant["nudge"]:
        for m in msgs:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] += variant["nudge"]
                break
        else:
            msgs.insert(0, {"role": "system", "content": variant["nudge"].strip()})
    body.pop("stream", None)
    seed = variant.get("seed")
    word = seed.get("word") if isinstance(seed, dict) else seed
    if word:
        # USER message, not system. The originating project tested both and
        # found the model would ignore a system-message seed and fall back to
        # its default approach; in the user turn it cannot. phrase() also
        # records it (concept_seed.record), which the dashboard's last-seed
        # panel reads.
        import shomen
        for m in reversed(body["messages"]):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] += shomen.seed_phrase(
                    word, where=f"fanout:{variant['name']}")
                break

    # Each variant must run the WHOLE tool loop. Doing a single completion
    # returns an empty answer and a pending tool call, which then scores zero
    # on every path-based grader -- measured, 0/8, and it was this bug rather
    # than anything about consensus.
    d = _tool_loop(body, timeout)
    msg = d["choices"][0]["message"]
    content, echoed = strip_seed_echo(msg.get("content") or "", word)
    if echoed:
        print(f"  fan-out {variant['name']}: removed {echoed} line(s) that "
              f"only repeated the seed word {word!r}", flush=True)
    return {"variant": variant["name"],
            "seed": word or None,
            "seed_info": (concept_seed.summary(seed) if isinstance(seed, dict)
                          else ({"word": word, "token_id": None,
                                 "u32": concept_seed.encode(word)}
                                if word else None)),
            "content": content,
            "seed_echo_stripped": echoed,
            "raw": d}


# THE SEED ECHO (#13, docs/SELF-IMPROVEMENT-LOG.md). Live gate 2026-09-24:
# the tie-breaker C's seed was `humanidad`, and the delivered `is_balanced`
# opened with the line `# humanidad` -- the model copied the seed into its
# code as a comment. The seed's wording now says what it is for
# (shomen.seed_phrase); this removes what still leaks: a LEADING line --
# the first line of the answer, or the first line of a fenced block -- that
# is nothing but a comment (or a markdown heading) whose text is the seed
# word. Only that exact shape: a comment that says anything more is the
# model's, and stays. Counted per candidate (x_yamadori.fanout candidates
# `seed_echo_stripped`) and per fix-up unit.
_ECHO_LINE = re.compile(r"^\s*(?:#+|//+|--|;+|/\*+|\*+|<!--)\s*(.*?)\s*"
                        r"(?:\*+/|-->)?\s*$")


def _is_seed_echo(line: str, word: str) -> bool:
    m = _ECHO_LINE.match(line)
    if not m:
        return False
    body = re.sub(r"[\W_]+", "", m.group(1)).lower()
    return bool(body) and body == re.sub(r"[\W_]+", "", word).lower()


def strip_seed_echo(text: str, word: str | None) -> tuple[str, int]:
    """(`text` without leading lines that only repeat the seed word, how
    many were removed). See THE SEED ECHO."""
    if not text or not word:
        return text or "", 0
    lines = text.split("\n")
    drop: set[int] = set()
    first = next((i for i, x in enumerate(lines) if x.strip()), None)
    if first is not None and _is_seed_echo(lines[first], word):
        drop.add(first)
    in_block = False
    for i, line in enumerate(lines):
        if _FENCE.match(line) and not in_block:
            in_block = True
            j = next((k for k in range(i + 1, len(lines))
                      if lines[k].strip()), None)
            if j is not None and not _FENCE.match(lines[j]) \
                    and _is_seed_echo(lines[j], word):
                drop.add(j)
            continue
        if in_block and line.strip() and set(line.strip()) <= {"`", "~"}:
            in_block = False
    if not drop:
        return text, 0
    return "\n".join(x for i, x in enumerate(lines) if i not in drop), \
        len(drop)


def _tool_loop(body: dict, timeout: int) -> dict:
    """Same loop the proxy runs, so a variant sees the same tool results."""
    import proxy as _p
    injected = {t["function"]["name"] for t in (body.get("tools") or [])
                if t.get("function", {}).get("name") in _p.OUR_NAMES}
    convo = body["messages"]
    # No hop count (removed 2026-09-22). The loop ends when the variant stops
    # calling tools, or when its conversation no longer fits its share -- the
    # helper's 3/8 for a second-brain candidate -- then the tools are
    # withdrawn and it answers.
    share_n = body.get("_share_n") or 1
    role = body.get("_role") or "main"
    while True:
        if _p.context_full(body, convo, role=role, share_n=share_n):
            body = _p._land(body, convo)
        # Through the proxy's own reader, not a bare urlopen: it streams, so
        # `timeout` bounds the gap between tokens rather than the whole
        # generation (docs/CONSTRAINTS.md item 13), and it strips the
        # proxy's `_underscored` bookkeeping, which prepare() leaves in the
        # payload and which this loop was shipping to llama-server.
        d = _p._post("/v1/chat/completions", body, timeout=timeout)
        msg = d["choices"][0]["message"]
        calls = [c for c in (msg.get("tool_calls") or [])
                 if c.get("function", {}).get("name") in injected]
        if not calls or not body.get("tools"):
            return d
        convo.append({"role": "assistant", "content": msg.get("content") or "",
                      "tool_calls": msg.get("tool_calls") or []})
        for c in calls:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            out = _p.run_our_tool(c["function"]["name"], args, None)
            convo.append({"role": "tool", "tool_call_id": c["id"],
                          "content": repeats.cap_tool_result(
                              out, c["function"]["name"], args)})
        body["messages"] = convo


def _seeds(payload: dict, n: int) -> list[dict]:
    """n concept seeds ({word, token_id, u32, hex}), mutually orthogonal and
    far from the prompt in the model's own space. Empty -- never an exception
    -- when the matrix is not extracted: a missing seed must not cost the
    request."""
    prompt = _task_text(payload.get("messages") or [])
    return concept_seed.seed_for(prompt or None, n)


# Candidate names. A is the answer already on its way to the client; B is the
# second brain's answer (VARIANTS[0]); C is the second brain's tie-breaker.
ORIGINAL = "original"
TIEBREAK = "tiebreak"

# Two candidates whose code parses and whose similarity (code_similarity, the
# same measure the medoid uses) is at least this count as AGREEING, and the
# original is kept without a tie-breaker. 0.80 is an UNMEASURED choice
# (operator, 2026-09-23): nothing here has graded how often two parsing
# candidates this similar are both right. bench/domain grades every candidate
# and can measure it; until then do not cite it (PROTOCOL rule 10).
AGREE = 0.80

# The most steps the procedure runs: A, B and the tie-breaker C. A tier's
# fanout above 3 allows nothing more.
MAX_STEPS = 3

# How long fan-out waits for the helper lane (admission.helper_lane) before it
# is recorded as skipped: "helper busy". Generous, because the lane is held by
# another request's deep thinking or fan-out for minutes, and the answer is
# already written -- waiting costs latency, not correctness. A choice.
LANE_WAIT = float(os.environ.get("YAMADORI_FANOUT_LANE_WAIT", "1200"))

# What the tie-breaker is asked. The task comes first (Laya is not reading
# this, but the model reads a long prompt better with the question up front),
# then both candidates with their check results, then the one request.
TIEBREAK_PROMPT = (
    "{task}\n\n---\n\nTwo candidate solutions to the task above were written "
    "independently. {verdict}\n\n"
    "Candidate 1:\n{code_a}\nCheck: {check_a}\n\n"
    "Candidate 2:\n{code_b}\nCheck: {check_b}\n\n"
    "Write the single best final solution to the task. Start from whichever "
    "candidate is closer to correct, fix what is wrong in it (the syntax "
    "errors listed above, and any logic error you find), and give the "
    "complete {lang} code in one fenced block.")


def _helper_body(payload: dict, messages: list | None = None) -> dict:
    """A payload for ONE second-brain generation: no tools, the helper's whole
    3/8 of the pool (share_n=1), re-derived from its own prompt. The shaped
    payload's effort and vendor sampling (tiers.apply) are kept."""
    import tiers
    body = dict(payload, tools=[])
    body.pop("tool_choice", None)
    if messages is not None:
        body["messages"] = messages
    body = tiers.rebudget(body, role="helper", share_n=1)
    body["_role"] = "helper"
    body["_share_n"] = 1
    return body


def _generate(body: dict, variant: dict, timeout: int) -> dict:
    """One candidate; a failure is a candidate with no content and the error
    recorded, never an exception out of fan-out."""
    try:
        r = _one(body, variant, timeout)
    except Exception as e:                                       # noqa: BLE001
        r = {"variant": variant["name"], "seed": variant.get("seed"),
             "content": "", "error": f"{type(e).__name__}: {e}"}
    r["role"] = body.get("_role") or "main"
    return r


def _task_text(messages: list) -> str:
    """The task as the user wrote it: the last user message's text."""
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(p.get("text") or "" for p in c
                             if isinstance(p, dict) and p.get("type") == "text")
    return ""


# THE LANGUAGE THE REQUEST ASKED FOR (live gate 2026-09-24, second run: the
# cache test's "Write a Python function fib(n) ... Just the code." routed
# code_generation, B ran, and the fan-out stopped as "prose: recorded only"
# -- the answer, 152 characters, carried its code in an UNTAGGED fence or
# none at all, and selection reads only language-tagged blocks). A request
# that names its language is read as that language's code.
_ASKED_LANG = re.compile(
    r"\b(python|typescript|javascript|rust|tsx|jsx|c\+\+|golang|c)\b"
    r"(?=\s+(?:function|method|class|script|program|module|code|snippet|"
    r"implementation|file|struct|trait|component|hook)\b)", re.IGNORECASE)
_CODE_WORD = re.compile(r"^\s*(?:def|class|async\s+def|function|export|"
                        r"import|from|const|let|fn|pub|struct|impl|"
                        r"interface|type|#include)\b", re.MULTILINE)


def asked_language(text: str) -> str | None:
    """The code language a request names ("Write a Python function"), in
    selection's canonical form, or None."""
    m = _ASKED_LANG.search(text or "")
    return _language({"c++": "cpp", "golang": "go"}.get(
        m.group(1).lower(), m.group(1).lower())) if m else None


def with_language(cand: dict, lang: str | None) -> dict:
    """The candidate as selection reads it, when the request named `lang`:
    an untagged fence is tagged `lang`, and an answer with no fence at all
    that PARSES as `lang` (and opens a definition) is fenced whole. The
    candidate's own content is kept as `raw_content`; what the client is
    delivered for the original is the proxy's own copy, untouched. A block
    already tagged, or content that does not parse, is left as it is."""
    content = cand.get("content") or ""
    if not lang or not content.strip():
        return cand
    blocks = _blocks(content)
    if any(_language(b["tag"]) for b in blocks if b["tag"]):
        return cand
    if blocks:
        out, n = [], 0
        for line in content.split("\n"):
            m = _FENCE.match(line)
            if m and not m.group(2) and n % 2 == 0:
                line = line.rstrip() + lang
            if m:
                n += 1
            out.append(line)
        new = "\n".join(out)
    else:
        cc = _code_check()
        code = content.strip()
        if cc is None or not _CODE_WORD.search(code) \
                or code.count("\n") < 1:
            return cand
        try:
            if cc.check(code, lang, run_format=False).get("syntax_errors"):
                return cand
        except Exception:                                        # noqa: BLE001
            return cand
        new = f"```{lang}\n{code}\n```"
    return dict(cand, content=new, raw_content=content,
                language_from="request")


def _prompt_code(messages: list) -> list[dict]:
    """The fenced code blocks of the question's latest user message
    (code_check.prompt_code); [] when it has none or the checker is absent,
    and then every candidate is checked on its own, exactly as before."""
    cc = _code_check()
    if cc is None or not hasattr(cc, "prompt_code"):
        return []
    try:
        return cc.prompt_code(messages or [])
    except Exception:                                            # noqa: BLE001
        return []


def analyse(r: dict, lang: str, prompt: list | None = None) -> dict:
    """One candidate's code in `lang`, graded by the checker. Never executed.

    {code, parses, errors, error_lines, warnings, truncated, ok, check_mode}:
    `ok` is what the grade uses -- it parses AND it was not cut off (a
    truncated answer that happens to parse is still missing its tail).
    `prompt` is the question's code blocks (_prompt_code); see check_code."""
    blocks = [dict(b, lang=_language(b["tag"])) for b in
              _blocks(r.get("content") or "") if b["code"].strip()]
    for b in blocks:
        if b["lang"] is None and not b["tag"]:
            b["lang"] = lang
    mine = [b for b in blocks if b["lang"] == lang]
    if not mine:
        other = sorted({b["lang"] for b in blocks if b["lang"]})
        why = (f"no {lang} code block"
               + (f" (answered in {', '.join(other)})" if other else ""))
        return {"code": "", "parses": False, "errors": None,
                "error_lines": [why], "warnings": 0, "truncated": False,
                "ok": False, "check_mode": None}
    _, b = max(enumerate(mine), key=lambda kb: (len(kb[1]["code"].strip()),
                                                 kb[0]))
    chk = check_code(b["code"], lang, prompt)
    truncated = not b["closed"] or _finish_reason(r) == "length"
    return {"code": b["code"], "parses": chk["parses"],
            "errors": chk["errors"], "error_lines": chk.get("error_lines") or [],
            "warnings": chk["warnings"], "truncated": truncated,
            "ok": bool(chk["parses"]) and not truncated,
            "check_mode": chk.get("check_mode")}


def _check_text(g: dict) -> str:
    """A candidate's check result, in words, for the tie-breaker prompt."""
    if g["parses"] is None:
        return "not checked: no syntax checker could run."
    if g["parses"] and g["truncated"]:
        return "parses, but the answer was cut off before it finished."
    if g["parses"]:
        return "parses; no syntax errors."
    lines = "; ".join(g["error_lines"][:5]) or "a syntax error"
    return f"does NOT parse: {lines}."


def _fenced(code: str, lang: str) -> str:
    if not code.strip():
        return "(no code)"
    fence = "````" if "```" in code else "```"
    return f"{fence}{lang}\n{code.rstrip()}\n{fence}"


def tiebreak_messages(messages: list, lang: str, a: dict, b: dict,
                      verdict: str) -> list:
    """C's conversation: the original one, with the last user turn replaced by
    the task plus both candidates and their check results (appended as a new
    user turn when the last message is not a plain-text user message)."""
    prompt = TIEBREAK_PROMPT.format(
        task=_task_text(messages).strip(), verdict=verdict, lang=lang,
        code_a=_fenced(a["code"], lang), check_a=_check_text(a),
        code_b=_fenced(b["code"], lang), check_b=_check_text(b))
    msgs = json.loads(json.dumps(messages or []))
    if msgs and msgs[-1].get("role") == "user" \
            and isinstance(msgs[-1].get("content"), str):
        msgs[-1] = dict(msgs[-1], content=prompt)
    else:
        msgs.append({"role": "user", "content": prompt})
    return msgs


def _row(i: int, r: dict, g: dict | None, similarity) -> dict:
    row = {"index": i, "variant": r.get("variant"), "role": r.get("role"),
           "seed": r.get("seed_info"), "length": len(r.get("content") or ""),
           "seed_echo_stripped": int(r.get("seed_echo_stripped") or 0)}
    if r.get("error") or not r.get("content"):
        row["error"] = bool(r.get("error")) or "empty"
    if g is None:
        row.update(parses=None, errors=None, similarity=None,
                   check_mode=None)
    else:
        row.update(parses=g["parses"], errors=g["errors"],
                   truncated=g["truncated"], warnings=g["warnings"],
                   similarity=similarity, check_mode=g.get("check_mode"))
    return row


def _decided(results: list, winner: int, selection: str, why: str,
             rows: list, lang: str | None) -> dict:
    return {"n": sum(1 for r in results if r.get("content")),
            "agreement": None, "votes": 0, "consensus_path": None,
            "path_votes": {}, "winner": results[winner], "results": results,
            "selection": selection, "fallback": False, "code_language": lang,
            "winner_index": winner, "winner_reason": why,
            "candidates": rows}


def _out(base: dict, v: dict, **kw) -> dict:
    """A run() result: the selection `v`, the procedure's `base` (mode,
    steps, similarity_ab ...) over it, and `kw` over both."""
    return {**v, **base, **kw}


def run(payload: dict, original: dict | None = None, n: int = MAX_STEPS,
        timeout: int = 3600, lane_timeout: float | None = None,
        seed_for=None) -> dict:
    """The sequential fan-out: A (given), B, grade, and C only when needed.

    `original` is candidate A, {"content", "raw"?}; when None (a script
    calling this directly) it is generated first, in the main role, from
    `payload` as shaped. `n` is the most candidates including A. B and C run
    one after the other in the HELPER role, each a job of shomen.run on the
    helper lane, so at most one second-brain context is live. Returns the consensus()
    shape plus mode, steps, stop_reason, similarity_ab and, when the lane
    never came free, skipped: "helper busy".

    `timeout` is 3600 s and bounds the gap between streamed tokens (see
    `_tool_loop`), matching every other generation timeout in the stack and
    llama-server's own `-to`; decode runs at 16-40 tok/s on this card.

    B and C are jobs of the ONE second-brain runner (shomen.run, jobs
    `alternative` and `tiebreak`), each on the helper lane. `seed_for(job,
    prompt)` supplies each job's concept seed -- the proxy's ledger, so a
    replay of the request draws the same word; without it a fresh one is
    drawn per job (a script calling this directly).
    """
    import admission
    import shomen
    n_max = max(1, min(int(n or 1), MAX_STEPS))
    if original is None:
        a = _generate(dict(payload, _role="main"),
                      {"name": ORIGINAL, "nudge": ""}, timeout)
    else:
        a = dict(original, variant=ORIGINAL, role="main")
    a.setdefault("seed", None)
    # The language the request named reads an untagged or bare-code answer
    # as code (with_language); without it, only tagged blocks count.
    asked = asked_language(_task_text(payload.get("messages") or []))
    a = with_language(a, asked)
    results = [a]
    base = {"mode": "sequential", "asked": n_max, "steps": 1,
            "stop_reason": None, "similarity_ab": None}
    if n_max < 2:
        return _out(base, _decided(results, 0, None, "fan-out of 1: "
                                     "nothing to compare", [], None),
                    stop_reason="n=1")
    code = _code_selection([a]) if a.get("content") else None
    lang = code["language"] if code else None
    # The question's own code, if it carries any: a candidate parses on its
    # own OR appended after it (code_check.check_against_prompt). Checked on
    # its own only, every continuation of starter code "does not parse" and
    # the grade can never separate two of them.
    pre = _prompt_code(payload.get("messages") or []) if lang else []
    wait = LANE_WAIT if lane_timeout is None else lane_timeout

    def seed(job: str):
        prompt = _task_text(payload.get("messages") or [])
        if seed_for is not None:
            return seed_for(job, prompt)
        return (concept_seed.seed_for(prompt or None, 1) or [None])[0]

    # ONE LANE HOLD FOR B AND C: one fan-out's candidates are never split by
    # another request's second-brain job. Each is still a job of the one
    # runner (shomen.run, held=True).
    with admission.helper_lane(timeout=wait, what="fan-out") as got:
        if not got:
            out = _decided(results, 0, None, "the helper lane stayed busy; "
                           "the original is kept", [], lang)
            return _out(base, out, skipped="helper busy",
                        stop_reason="skipped: helper busy")
        # A fresh concept seed for EVERY second-brain run (operator,
        # 2026-09-23): one for B and, drawn separately so it differs, one for the
        # tie-breaker C.
        b = shomen.run("alternative", body=_helper_body(payload),
                       variant=VARIANTS[0], seed=seed("alternative"),
                       timeout=timeout, held=True)
        b = with_language(b, asked)
        results.append(b)
        base["steps"] = 2

        if lang is None:
            # PROSE: the path/symbol vote over [A, B], recorded only.
            v = consensus(results)
            return _out(base, v, stop_reason="prose: recorded only; the "
                        "original is delivered")

        ga, gb = analyse(a, lang, pre), analyse(b, lang, pre)
        sim = (code_similarity(ga["code"], gb["code"], lang)
               if ga["ok"] and gb["ok"] else None)
        base["similarity_ab"] = sim
        rows = [_row(0, a, ga, sim), _row(1, b, gb, sim)]
        if not b.get("content"):
            return _out(base, _decided(
                results, 0, "code_grade", "the second candidate returned "
                "nothing; the original is kept", rows, lang),
                stop_reason="second candidate failed")
        if ga["parses"] is None and gb["parses"] is None:
            return _out(base, _decided(
                results, 0, "code_grade", "no syntax checker could run; the "
                "original is kept", rows, lang),
                stop_reason="unchecked: no syntax checker")
        if ga["ok"] != gb["ok"]:
            w = 0 if ga["ok"] else 1
            return _out(base, _decided(
                results, w, "code_grade", f"clear: only candidate {w + 1}'s "
                f"{lang} code parses (and is complete)", rows, lang),
                stop_reason="clear: only one parses")
        if ga["ok"] and sim is not None and sim >= AGREE:
            return _out(base, _decided(
                results, 0, "code_grade", f"clear: both parse and agree "
                f"(similarity {sim:.2f} >= {AGREE}); the main brain's answer "
                f"is kept", rows, lang),
                stop_reason="clear: agreement")

        verdict = (("Both parse but they disagree"
                    + (f" (similarity {sim:.2f})." if sim is not None
                       else "."))
                   if ga["ok"] else "Neither one is a complete solution that "
                   "parses as written.")
        why_c = ("both parse but disagree" if ga["ok"]
                 else "neither parses")
        if n_max < 3:
            return _out(base, _decided(
                results, 0, "code_grade", f"unresolved ({why_c}) and fan-out "
                f"{n_max} leaves no room for a tie-breaker; the original is "
                f"kept", rows, lang),
                stop_reason=f"unresolved: {why_c}, no tie-breaker allowed")

        msgs = tiebreak_messages(payload.get("messages") or [], lang, ga, gb,
                                 verdict)
        c = shomen.run("tiebreak", body=_helper_body(payload, msgs),
                       variant={"name": TIEBREAK, "nudge": ""},
                       seed=seed("tiebreak"), timeout=timeout, held=True)
        c = with_language(c, asked)
        results.append(c)
        base["steps"] = 3

    v = consensus(results, prefer=2, prompt=pre)
    return _out(base, v, stop_reason=f"tie-breaker: {why_c}")


def consensus(results: list[dict], prefer: int | None = None,
              prompt: list | None = None) -> dict:
    """Pick the winner among the candidates' answers and report the vote.

    `prefer` is the index that wins a code-medoid tie (the tie-breaker C in
    the sequential fan-out). `prompt` is the question's code blocks
    (_prompt_code): a candidate then parses on its own or appended after
    one of them (check_code).

    Code answers go to the code medoid (see the module docstring); answers
    without code keep the path/symbol vote. Every return names how the
    winner was chosen (`selection`), the winner's index into `results`, why
    it won, and per candidate whether its code parses, its similarity to the
    others and its length.
    """
    good = [r for r in results if r.get("content")]
    if not good:
        return {"results": results, "agreement": None, "winner": None,
                "selection": None, "winner_index": None,
                "winner_reason": "no variant returned an answer",
                "candidates": _candidate_rows(results, {})}

    path_votes: Counter = Counter()
    sym_votes: Counter = Counter()
    for r in good:
        p, s = claims(r["content"])
        # One vote per answer per claim, so a variant repeating a path five
        # times does not outvote three variants naming it once.
        for x in p:
            path_votes[x] += 1
        for x in s:
            sym_votes[x] += 1

    top_path, top_n = (path_votes.most_common(1) or [(None, 0)])[0]
    # NO VOTES IS NOT DISAGREEMENT. An answer that names no file -- a list of
    # primes, a paragraph of design advice -- casts no path vote, and this
    # used to report agreement 0.0 for it, which `dissent_note` then turned
    # into "only 0% agreed on the same file. Treat this as unsettled." on
    # every such answer at `high` and `max`. Agreement over nothing is
    # undefined, so it is None, and None says nothing.
    votes = sum(path_votes.values())
    agreement = (top_n / len(good)) if votes else None

    # The winning ANSWER is the one that best represents the consensus, not the
    # first or the longest: the answer that names the most agreed-upon claims.
    def claim_score(r):
        p, s = claims(r["content"])
        return sum(path_votes[x] for x in p) + sum(sym_votes[x] for x in s)

    code = _code_selection(results, prefer=prefer, prompt=prompt)
    if code is None:
        # No code to judge: the path/symbol vote, exactly as before. Its
        # shortest-on-tie is kept for prose; it is what the file-location
        # measurement above ran with.
        winner = max(good, key=lambda r: (claim_score(r), -len(r["content"])))
        selection = "path_consensus"
        info: dict = {}
        top = claim_score(winner)
        why = (f"names the most agreed-upon paths/symbols ({top} votes)" if top
               else "no path or symbol votes were cast; the shortest answer "
                    "(the path vote's tie-break)")
    elif code["winner"] is not None:
        winner = results[code["winner"]]
        selection = "code_medoid"
        info = code["info"]
        why = code["why"]
    else:
        # Nothing parses. The vote, but a tie goes to the LONGEST answer: the
        # shortest broken answer is the likeliest to be a truncated one.
        winner = max(good, key=lambda r: (claim_score(r), len(r["content"])))
        selection = "fallback"
        info = code["info"]
        why = (f"no candidate's {code['language']} code parses "
               f"({code['why']}); fell back to the path/symbol vote with "
               f"ties to the longest answer")
        print(f"  fan-out: FALLBACK selection -- {why}", flush=True)

    return {
        "n": len(good),
        "agreement": round(agreement, 2) if agreement is not None else None,
        "votes": votes,
        "consensus_path": top_path,
        "path_votes": dict(path_votes.most_common(5)),
        "winner": winner,
        "results": results,
        "selection": selection,
        "fallback": selection == "fallback",
        "code_language": code["language"] if code else None,
        "winner_index": next(i for i, r in enumerate(results) if r is winner),
        "winner_reason": why,
        "candidates": _candidate_rows(results, info),
    }


def _candidate_rows(results: list[dict], info: dict) -> list[dict]:
    rows = []
    for i, r in enumerate(results):
        row = {"index": i, "variant": r.get("variant"), "role": r.get("role"),
               "seed": r.get("seed_info"),
               "length": len(r.get("content") or ""),
               "seed_echo_stripped": int(r.get("seed_echo_stripped") or 0)}
        if r.get("error") or not r.get("content"):
            row["error"] = bool(r.get("error")) or "empty"
        c = info.get(i)
        if c is not None:
            row.update(parses=c["parses"], errors=c.get("errors"),
                       similarity=c["similarity"],
                       code_length=len(c["code"]), truncated=c["truncated"],
                       warnings=c["warnings"],
                       check_mode=c.get("check_mode"))
        else:
            row.update(parses=None, errors=None, similarity=None,
                       check_mode=None)
        rows.append(row)
    return rows


def selection_record(v: dict) -> dict:
    """The compact part of a `run()` result for x_yamadori.fanout: the mode,
    the steps run and why it stopped, the A-B similarity, how the winner was
    chosen, its index and why, and per candidate its role, seed, parses,
    error count, similarity and length. Carries no answer text. The keys the
    benchmark harnesses read (selection, candidates[].parses) are kept.

    Per candidate, `check_mode` says which form made `parses` true
    (code_check.check_against_prompt): "standalone" (on its own),
    "appended" (only after one of the question's code blocks), "neither";
    None when the question carries no code, or the answer is prose."""
    if not v:
        return {}
    out = {"mode": v.get("mode"), "steps": v.get("steps"),
           "stop_reason": v.get("stop_reason"),
           "similarity_ab": v.get("similarity_ab"),
           "selection": v.get("selection"),
           "winner_index": v.get("winner_index"),
           "why": v.get("winner_reason"),
           "candidates": [{"variant": c.get("variant"),
                           "role": c.get("role"),
                           "seed": c.get("seed"),
                           "parses": c.get("parses"),
                           "check_mode": c.get("check_mode"),
                           "errors": c.get("errors"),
                           "similarity": c.get("similarity"),
                           "length": c.get("length"),
                           "seed_echo_stripped": c.get("seed_echo_stripped")}
                          for c in v.get("candidates") or []]}
    if v.get("skipped"):
        out["skipped"] = v["skipped"]
    return out


# ---------------------------------------------------------------------------
# Code-aware selection. See "HOW THE WINNER IS PICKED FOR A CODE ANSWER" above.
# ---------------------------------------------------------------------------

# Mean similarities this close count as a tie and go to the tie-breaks. A
# choice, not a measurement (module docstring, last paragraph).
SIM_TIE = 0.01
# Token n-gram width for the similarity. Trigrams keep local order, so the
# same tokens in a different structure do not agree. Also a choice.
NGRAM = 3
# Bounded work per candidate: a pathological answer must not stall selection.
MAX_TOKENS = 20000

# Used only when mcp/code_check.py cannot be imported.
_CODE_LANGS = {"python", "typescript", "tsx", "javascript", "jsx", "rust",
               "c", "cpp"}
_GRAMMAR = {"python": "python", "typescript": "typescript", "tsx": "tsx",
            "javascript": "javascript", "jsx": "javascript", "rust": "rust",
            "c": "c", "cpp": "cpp"}
_ALIASES = {"py": "python", "python3": "python", "py3": "python",
            "ts": "typescript", "mts": "typescript", "cts": "typescript",
            "js": "javascript", "mjs": "javascript", "cjs": "javascript",
            "node": "javascript", "rs": "rust", "h": "c", "c++": "cpp",
            "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp", "hxx": "cpp"}
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([^\s`{]*)")


def _code_check():
    """mcp/code_check.py when it imports, else None. It is being built
    concurrently; selection must survive it being absent or mid-edit."""
    try:
        import code_check
        if all(hasattr(code_check, a) for a in ("check", "fenced_blocks",
                                                 "normalize_language")):
            return code_check
    except Exception:                                            # noqa: BLE001
        pass
    return None


def _language(tag: str) -> str | None:
    cc = _code_check()
    if cc is not None:
        lang = cc.normalize_language(tag)
        return lang if lang in cc.CODE_LANGS else None
    k = (tag or "").strip().lower().lstrip(".")
    k = _ALIASES.get(k, k)
    return k if k in _CODE_LANGS else None


def _blocks(text: str) -> list[dict]:
    """[{tag, code, closed}] for every fenced block, in order."""
    cc = _code_check()
    if cc is not None:
        return [{"tag": b["lang"], "code": b["code"], "closed": b["closed"]}
                for b in cc.fenced_blocks(text or "")]
    # The same reading as bench/domain/grade.py code_blocks(): markdown fences
    # are flat text, so this is a line scanner, not a parser (PROTOCOL rule 8).
    out, cur = [], None
    for line in (text or "").splitlines():
        if cur is None:
            m = _FENCE.match(line)
            if m:
                cur = {"fence": m.group(1),
                       "tag": m.group(2).strip().lower().lstrip("."),
                       "lines": []}
            continue
        s = line.strip()
        if s and set(s) == {cur["fence"][0]} and len(s) >= len(cur["fence"]):
            out.append({"tag": cur["tag"], "code": "\n".join(cur["lines"]),
                        "closed": True})
            cur = None
        else:
            cur["lines"].append(line)
    if cur is not None and cur["lines"]:
        out.append({"tag": cur["tag"], "code": "\n".join(cur["lines"]),
                    "closed": False})
    return out


_ts_parsers: dict = {}


def _tree(code: str, lang: str):
    """Tree-sitter root node, or None when no parser is available."""
    cc = _code_check()
    try:
        if cc is not None and hasattr(cc, "parse_tree"):
            return cc.parse_tree(code, lang)
        g = _GRAMMAR[lang]
        if g not in _ts_parsers:
            from tree_sitter_language_pack import get_parser
            _ts_parsers[g] = get_parser(g)
        return _ts_parsers[g].parse(code.encode("utf-8")).root_node
    except Exception:                                            # noqa: BLE001
        return None


def _prepared(code: str, lang: str) -> str:
    # A method excerpt is valid Python once its common indent is removed;
    # code_check.check() does the same before it parses.
    return textwrap.dedent(code) if lang == "python" else code


def check_code(code: str, lang: str, prompt: list | None = None) -> dict:
    """{parses, errors, error_lines, warnings, checker, check_mode}. `parses`
    is None when no checker could run -- unknown is not the same as broken.
    Never executes `code`: code_check uses compile() and tree-sitter, and
    run_format=False keeps its formatter subprocess off.

    `prompt` is the question's code blocks (_prompt_code). The general rule
    (code_check.check_against_prompt): the code parses if it parses on its
    own OR appended directly after one of them. `check_mode` records which
    ("standalone", "appended", "neither"); None when `prompt` is empty, and
    then this is exactly the standalone check it always was. The errors and
    warnings are those of the form that parsed; for "neither", the
    standalone form's, as before."""
    cc = _code_check()
    if cc is not None:
        try:
            if hasattr(cc, "check_against_prompt"):
                g = cc.check_against_prompt(code, lang, prompt or [])
                res, mode = g["result"], g["check_mode"]
            else:
                res, mode = cc.check(code, lang, run_format=False), None
            se = res.get("syntax_errors") or []
            errs = len(se)
            return {"parses": errs == 0, "errors": errs,
                    "error_lines": [f"line {e.get('line')}: "
                                    f"{str(e.get('message') or '')[:160]}"
                                    for e in se[:5]],
                    "warnings": len(res.get("defects") or []),
                    "checker": "code_check", "check_mode": mode}
        except Exception:                                        # noqa: BLE001
            pass
    root = _tree(_prepared(code, lang), lang)
    if root is None:
        return {"parses": None, "errors": None, "warnings": 0,
                "checker": None, "check_mode": None}
    return {"parses": not root.has_error, "errors": int(root.has_error),
            "error_lines": (["tree-sitter reports a syntax error"]
                            if root.has_error else []),
            "warnings": 0, "checker": "tree-sitter", "check_mode": None}


_ID_TYPES = {"identifier", "type_identifier", "property_identifier",
             "field_identifier", "shorthand_property_identifier",
             "shorthand_property_identifier_pattern",
             "private_property_identifier", "statement_identifier"}
_STRING_TYPES = {"string", "template_string", "string_literal",
                 "raw_string_literal", "char_literal", "concatenated_string",
                 "character_literal"}
# Fields whose subtree binds a name: definitions, parameters, assignment
# targets, patterns and declarators across the tree-sitter grammars in use.
_BIND_FIELDS = ("name", "left", "parameters", "parameter", "pattern",
                "declarator", "alias")


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _bound_names(root) -> set[str]:
    names: set[str] = set()
    for n in _walk(root):
        for f in _BIND_FIELDS:
            c = n.child_by_field_name(f)
            if c is None:
                continue
            for x in _walk(c):
                if x.type in _ID_TYPES and x.text:
                    names.add(x.text.decode("utf-8", "replace"))
    return names


def code_tokens(code: str, lang: str) -> list[str] | None:
    """The normalized token sequence the similarity is taken over.

    Tree-sitter leaves in order; comments and whitespace dropped; a string
    literal is one token; a name the code binds itself becomes `$id`, so
    renaming a variable changes nothing, while a name it only USES (a
    builtin, an imported API, a method) keeps its spelling, so `sorted` and
    `min` still disagree. Python blocks get open/close tokens, because there
    the indentation IS the structure.
    """
    root = _tree(_prepared(code, lang), lang)
    if root is None:
        return None
    bound = _bound_names(root)
    out: list[str] = []
    stack: list = [root]
    while stack and len(out) < MAX_TOKENS:
        n = stack.pop()
        if isinstance(n, str):
            out.append(n)
            continue
        t = n.type
        if "comment" in t:
            continue
        if t in _STRING_TYPES:
            out.append("$str")
            continue
        if n.child_count == 0:
            text = (n.text or b"").decode("utf-8", "replace")
            if t in _ID_TYPES and text in bound:
                out.append("$id")
            elif text.strip():
                out.append(text)
            continue
        if t == "block":
            stack.append("}")
        stack.extend(reversed(n.children))
        if t == "block":
            stack.append("{")
    return out


def _ngrams(tokens: list[str]) -> Counter:
    if len(tokens) < NGRAM:
        return Counter([tuple(tokens)]) if tokens else Counter()
    return Counter(tuple(tokens[i:i + NGRAM])
                   for i in range(len(tokens) - NGRAM + 1))


def _jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def code_similarity(a: str, b: str, lang: str) -> float | None:
    """Similarity in [0, 1] of two pieces of code in `lang`; None when either
    cannot be tokenized."""
    ta, tb = code_tokens(a, lang), code_tokens(b, lang)
    if ta is None or tb is None:
        return None
    return round(_jaccard(_ngrams(ta), _ngrams(tb)), 4)


def _finish_reason(r: dict) -> str | None:
    try:
        return ((r.get("raw") or {}).get("choices") or [{}])[0].get(
            "finish_reason")
    except (AttributeError, IndexError, TypeError):
        return None


def _code_selection(results: list[dict],
                    prefer: int | None = None,
                    prompt: list | None = None) -> dict | None:
    """The code medoid among `results`, or None when this is not a code
    answer (fewer than half the answers carry a code-language block).

    Returns {language, winner (index into results, or None when nothing
    parses), why, info: {index: {code, parses, similarity, truncated,
    warnings, length, check_mode}}}. `prompt`: see check_code. The
    similarity is always taken over each candidate's own code.
    """
    good = [(i, r) for i, r in enumerate(results) if r.get("content")]
    blocks = {i: [dict(b, lang=_language(b["tag"]))
                  for b in _blocks(r["content"]) if b["code"].strip()]
              for i, r in good}
    tagged = Counter(b["lang"] for bs in blocks.values() for b in bs
                     if b["lang"])
    if not tagged:
        return None
    lang = tagged.most_common(1)[0][0]
    # An untagged block is read as the dominant language -- someone tagged
    # it, so that is what the question was answered in. Never guessed alone.
    for bs in blocks.values():
        for b in bs:
            if b["lang"] is None and not b["tag"]:
                b["lang"] = lang
    with_code = [i for i, _ in good if any(b["lang"] for b in blocks[i])]
    if len(with_code) * 2 <= len(good):
        return None

    info: dict = {}
    for i in with_code:
        r = results[i]
        mine = [b for b in blocks[i] if b["lang"] == lang]
        cand_lang = lang
        if not mine:                      # answered in another language
            mine = [b for b in blocks[i] if b["lang"]]
            cand_lang = mine[-1]["lang"]
        # The largest block; a tie goes to the later one (a corrected version
        # usually follows the first attempt).
        k, b = max(enumerate(mine),
                   key=lambda kb: (len(kb[1]["code"].strip()), kb[0]))
        chk = (check_code(b["code"], cand_lang, prompt) if cand_lang == lang
               else {"parses": False, "warnings": 0, "errors": None,
                     "check_mode": None})
        info[i] = {"code": b["code"], "language": cand_lang,
                   "parses": chk["parses"], "warnings": chk["warnings"],
                   "errors": chk.get("errors"),
                   "check_mode": chk.get("check_mode"),
                   "truncated": (not b["closed"]
                                 or _finish_reason(r) == "length"),
                   "length": len(r["content"]), "similarity": None}

    valid = [i for i in info if info[i]["parses"]]
    if not valid:
        unchecked = all(info[i]["parses"] is None for i in info)
        return {"language": lang, "winner": None, "info": info,
                "why": ("no syntax checker could run: tree-sitter is "
                        "unavailable" if unchecked
                        else f"{len(info)} of {len(info)} have syntax errors "
                             f"or are in another language")}
    complete = [i for i in valid if not info[i]["truncated"]]
    set_aside = [i for i in valid if i not in complete] if complete else []
    pool = complete or valid

    toks = {i: _ngrams(code_tokens(info[i]["code"], lang) or []) for i in pool}
    for i in pool:
        others = [_jaccard(toks[i], toks[j]) for j in pool if j != i]
        info[i]["similarity"] = (round(sum(others) / len(others), 4)
                                 if others else None)

    excluded = len(info) - len(valid)
    notes = []
    if excluded:
        notes.append(f"{excluded} excluded for syntax errors or another "
                     f"language")
    if set_aside:
        notes.append(f"{len(set_aside)} set aside as truncated")
    tail = f" ({'; '.join(notes)})" if notes else ""

    if len(pool) == 1:
        w = pool[0]
        return {"language": lang, "winner": w, "info": info,
                "why": f"the only candidate left whose {lang} code "
                       f"parses{tail}"}

    best = max(info[i]["similarity"] for i in pool)
    tied = [i for i in pool if info[i]["similarity"] >= best - SIM_TIE]
    median = statistics.median(len(info[i]["code"]) for i in pool)

    def tie_key(i):
        c = info[i]
        return (c["warnings"], c["truncated"],
                abs(len(c["code"]) - median), -len(c["code"]), i)

    w = (prefer if prefer in tied and len(tied) > 1
         else min(tied, key=tie_key))
    why = (f"medoid: highest mean similarity "
           f"({info[w]['similarity']:.2f}) to the other {len(pool) - 1} "
           f"candidates whose {lang} code parses{tail}")
    if len(tied) > 1:
        rest = [info[i] for i in tied if i != w]
        if w == prefer:
            by = "preferring the tie-breaker's answer"
        elif any(c["warnings"] > info[w]["warnings"] for c in rest):
            by = "fewer checker warnings"
        elif any(c["truncated"] and not info[w]["truncated"] for c in rest):
            by = "a complete answer"
        else:
            by = "code length nearest the median"
        why += (f"; tied within {SIM_TIE} with {len(tied) - 1} other(s), "
                f"broken by {by}")
    return {"language": lang, "winner": w, "info": info, "why": why}


def _choice_averaged(state: str, instructions: str, criteria: dict,
                     timeout: int = 30) -> dict | None:
    """A choice, averaged over option orderings.

    The model is NOT permutation invariant. Reported upstream at a 27-33%
    semantic flip rate on two options, and reproduced here: asking which file
    declares a type was stable, but "which loop visits every element once"
    flipped its answer purely on option order at a margin of 0.441, well above
    any gate worth setting.

    Averaging the distributions over forward and reversed order cancels the
    positional component. It also makes the MARGIN honest, which matters more:
    measured, cases that flip average down to margins of 0.012-0.184 while a
    genuinely decided case stays at 0.800. So a reordering-sensitive answer
    stops being a confident answer and starts being an undecided one, which is
    the correct outcome and what the margin gate is then able to catch.

    Costs two calls instead of one, about 100ms at this model's latency.
    None without a call when YAMADORI_E1=1 (Laya is off the request path).
    """
    import code_search as cs
    import e1
    if not e1.laya_allowed():
        return None
    keys = list(criteria)
    if len(keys) < 2:
        return None
    orders = [tuple(keys), tuple(reversed(keys))]
    acc = {k: 0.0 for k in keys}
    for order in orders:
        try:
            d = cs._post_json(LAYA_URL_ + "/decide", {
                "state": state[:2000],
                "questions": {"pick": {
                    "type": "choice", "instructions": instructions,
                    "criteria": {k: criteria[k] for k in order}}}},
                timeout=timeout)
            probs = (d["answers"]["pick"].get("probabilities") or {})
        except Exception:                                        # noqa: BLE001
            return None
        for k in keys:
            acc[k] += float(probs.get(k, 0.0)) / len(orders)
    ranked = sorted(acc.items(), key=lambda x: -x[1])
    return {"choice": ranked[0][0],
            "margin": round(ranked[0][1] - ranked[1][1], 3),
            "probabilities": {k: round(v, 3) for k, v in acc.items()}}


def break_tie(question: str, results: list[dict], timeout: int = 30) -> dict | None:
    """Ask the decision model to pick, as ONE choice whose options are the
    candidates.

    This is the primitive it is actually good at. Scoring candidates
    independently does not work -- its scores are not comparable across
    different inputs -- and handing it a LIST of states does not batch, it
    concatenates them and returns a single verdict for the blob.

    Used only when consensus fails. When the variants agree, the agreement is
    free and better evidenced than any classifier.
    """
    good = [r for r in results if r.get("content")]
    if len(good) < 2:
        return None
    criteria = {}
    for i, r in enumerate(good):
        paths, _ = claims(r["content"])
        label = chr(ord("a") + i)
        criteria[label] = (", ".join(sorted(paths)[:3]) or
                           r["content"][:120].replace("\n", " "))
    v = _choice_averaged(question, "Which answer best answers the question?",
                         criteria, timeout)
    # Below the floor the options did not separate once position bias is
    # removed. Reporting a pick it did not really make is inventing a decision.
    if not v or v["margin"] < 0.15:
        return None
    idx = ord(v["choice"]) - ord("a")
    if not 0 <= idx < len(good):
        return None
    return {"winner": good[idx], "margin": v["margin"],
            "ranking": v["probabilities"]}


LAYA_URL_ = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")


def dissent_note(v: dict) -> str:
    """What to tell the user when the variants did NOT agree.

    Disagreement is information and hiding it is the failure mode that makes a
    confident wrong answer indistinguishable from a confident right one.

    But only disagreement that was OBSERVED. With no path votes cast there is
    nothing the variants could have agreed on, and saying "0% agreed" about
    that is a false claim appended to a correct answer
    (docs/SELECTION-BUILD.md harm 2; the live suite's primes answer at `high`).
    """
    if not v or v.get("agreement") is None or not v.get("path_votes"):
        return ""
    if v["agreement"] >= 0.75:
        return ""
    alts = [p for p in (v.get("path_votes") or {}) if p != v.get("consensus_path")]
    note = (f"\n\n> Answered {v['n']} ways; only {int(v['agreement'] * 100)}% "
            f"agreed on the same file. Treat this as unsettled.")
    if alts:
        note += " Other candidates: " + ", ".join(alts[:3]) + "."
    return note


# ---------------------------------------------------------------------------
# THE HAND-BACK (operator decision, 2026-09-23): the second brain's work is
# never thrown away. Until then a prose B was recorded as a vote and dropped,
# and a code loser's different approach was dropped with it.
#
#   PROSE  B's (or C's) points that the original does not make -- a named
#          path or `symbol` the original lacks, or a sentence whose content
#          words are mostly absent from the original -- as an "alternative
#          view". proxy._fan_out decides where it crosses.
#   CODE   the winner is still what is delivered; one line names what each
#          loser did differently: the APIs it USES that the delivered code
#          does not (names the code binds itself are ignored, so a renamed
#          variable is not a difference). Material = at least one such name,
#          in a loser whose code parses (a broken tree cannot say which
#          names it binds).
#
# Every threshold here is a CHOICE, not a measurement (PROTOCOL rule 10):
# nothing in the repo has graded whether a handed-back point improves an
# answer. bench/domain can.
# ---------------------------------------------------------------------------
ALT_MIN_WORDS = 4        # a sentence shorter than this is never "a point"
ALT_NOVELTY = 0.5        # share of its content words absent from the original
ALT_MAX_POINTS = 6
ALT_MAX_CHARS = 1200
CODE_NOTE_NAMES = 5      # names listed per loser in the one-line code note

_LETTERS = "ABCDEFGH"
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "are", "was", "were", "not",
    "but", "you", "your", "can", "will", "would", "should", "could", "from",
    "have", "has", "had", "its", "it's", "they", "them", "their", "there",
    "then", "than", "which", "what", "when", "where", "who", "how", "why",
    "into", "onto", "also", "just", "use", "used", "uses", "using", "one",
    "all", "any", "each", "other", "some", "such", "these", "those", "only",
    "more", "most", "very", "here", "does", "did", "done", "been", "being",
    "may", "might", "must", "about", "over", "under", "out", "because"}
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(\[])")


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")} - _STOPWORDS


def _prose_points(text: str) -> list[str]:
    """The answer as points: each bullet or line, each sentence of a
    paragraph. Fenced code is left out -- a prose hand-back carries prose."""
    out, fence = [], False
    for raw in (text or "").splitlines():
        s = raw.strip()
        if s.startswith(("```", "~~~")):
            fence = not fence
            continue
        if fence or not s:
            continue
        s = re.sub(r"^(?:[-*•]|\d+[.)]|#+)\s+", "", s)
        out += [p.strip() for p in _SENTENCE.split(s) if p.strip()]
    return out


def differing_points(original: str, other: str) -> list[str]:
    """The points in `other` that `original` does not make."""
    a_words = _content_words(original)
    a_paths, a_syms = claims(original)
    out = []
    for p in _prose_points(other):
        paths, syms = claims(p)
        words = _content_words(p)
        new_claim = (paths - a_paths) or (syms - a_syms)
        novel = (len(words) >= ALT_MIN_WORDS
                 and len(words - a_words) / len(words) >= ALT_NOVELTY)
        if new_claim or novel:
            out.append(p)
    return out


def _used_names(code: str, lang: str) -> tuple[set[str], set[str]] | None:
    """(identifiers the code uses, identifiers it binds), from tree-sitter;
    None when no parser is available."""
    root = _tree(_prepared(code, lang), lang)
    if root is None:
        return None
    used = set()
    for n in _walk(root):
        if n.child_count == 0 and n.type in _ID_TYPES and n.text:
            used.add(n.text.decode("utf-8", "replace"))
    return used, _bound_names(root)


def handback(v: dict) -> dict | None:
    """What the second brain's candidates hand back beyond the winner.

    {"kind": "prose"|"code", "text", "from": [letters], "points": n}, or
    None when nothing differs materially. `text` is bare (points as "- "
    lines, or the code note's one sentence); proxy._fan_out frames it.
    """
    results = list((v or {}).get("results") or [])
    if len(results) < 2 or not (results[0].get("content") or "").strip():
        return None
    lang = v.get("code_language")
    if lang is None:
        return _prose_handback(results)
    return _code_handback(results, v.get("winner_index"), lang)


def _prose_handback(results: list[dict]) -> dict | None:
    original = results[0].get("content") or ""
    labelled = len([r for r in results[1:] if r.get("content")]) > 1
    points, sources = [], []
    for i, r in enumerate(results[1:], start=1):
        pts = differing_points(original, r.get("content") or "")
        if pts:
            sources.append(_LETTERS[i])
            points += [(f"({_LETTERS[i]}) " if labelled else "") + p
                       for p in pts]
    if not points:
        return None
    lines, used = [], 0
    for p in points[:ALT_MAX_POINTS]:
        line = f"- {p}"
        if used + len(line) > ALT_MAX_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(points):
        lines.append(f"- (+{len(points) - len(lines)} more points not shown; "
                     f"the limit is {ALT_MAX_POINTS} points or "
                     f"{ALT_MAX_CHARS} characters)")
    return {"kind": "prose", "text": "\n".join(lines), "from": sources,
            "points": len(points)}


def _code_handback(results: list[dict], winner: int | None,
                   lang: str) -> dict | None:
    if winner is None or not 0 <= winner < len(results):
        return None
    w_code = analyse(results[winner], lang)["code"]
    w_names = _used_names(w_code, lang) if w_code.strip() else None
    if w_names is None:
        return None
    parts, sources = [], []
    for i, r in enumerate(results):
        if i == winner or not (r.get("content") or "").strip():
            continue
        g = analyse(r, lang)
        # Only a loser whose code PARSES. In a broken tree the binding sites
        # are error nodes, so its own parameters read as "used" names -- the
        # first version reported `x` as an API the original used, for
        # `def add_one(x:` (mcp/test_fanout_delivery.py).
        if g["parses"] is not True:
            continue
        names = _used_names(g["code"], lang) if g["code"].strip() else None
        if names is None:
            continue
        diff = sorted(names[0] - w_names[0] - w_names[1] - names[1])
        if not diff:
            continue
        who = "the original" if i == 0 else f"candidate {_LETTERS[i]}"
        shown = ", ".join(f"`{d}`" for d in diff[:CODE_NOTE_NAMES])
        more = (f" and {len(diff) - CODE_NOTE_NAMES} more"
                if len(diff) > CODE_NOTE_NAMES else "")
        state = " (cut off before it finished)" if g["truncated"] else ""
        parts.append(f"{who} used {shown}{more}{state}")
        sources.append(_LETTERS[i])
    if not parts:
        return None
    text = ("The other candidates differed: " + "; ".join(parts)
            + ". The delivered code uses none of these.")
    return {"kind": "code", "text": text, "from": sources,
            "points": len(parts)}
