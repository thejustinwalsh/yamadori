#!/usr/bin/env python
"""THE CODE-WORK ROUTER: one class per request, decided once, in prepare.

WHY IT EXISTS (operator, 2026-09-24). The biggest proven win is repairing the
code the agent writes, automatically. That only pays if the code pipelines --
fan-out, the repair pass, skills later -- run on code work and nothing else:
a client's tool-loop step or a random chat turn sent through them costs
minutes of GPU and gains nothing. Each feature used to re-decide "is this
code?" with its own words (`selection._CODE_TASK` for fan-out, the tier flag
alone for repair). They now read ONE class from here.

THE CLASSES, in the order they are decided -- the first rule that fits wins:

  utility           a client's own side call: an approval classifier, a
                    session title, a compaction (selection.utility_call,
                    measured there). The bare model, nothing of ours.
  agent_step        the turn belongs to the CLIENT's agent loop. The
                    conversation ends on a client tool result; or the last
                    user turn is a harness notice / a bare "continue" with the
                    client's tools present; or the instruction asks to act on
                    the user's machine (selection.acts_locally).
  code_edit         the prompt CARRIES code and the instruction asks for work
                    on it (fix, change, port, complete, continue ...), or is
                    not a question at all.
  code_generation   the instruction asks for new code: a write/implement/
                    create... verb whose object is a code noun (function,
                    program, module, type, struct, tests ...).
  library_question  a question or lookup, not a code write, and the tool gate
                    offered our tools for a reason that is ABOUT this request
                    (selection.READABLE_GATE: a bound repository, or a held
                    package it imports, names, or whose symbol it uses).
  prose             everything else.

WHAT READS THE CLASS

  fan-out           code_generation / code_edit only (selection.decide)
  repair pass       code_generation / code_edit only (proxy.prepare)
  library help      the definitions injection: library_question only
                    (proxy._library_definitions)
  deep thinking     NOT the class since Phase 0.6 (operator, 2026-09-24):
                    four triggers on any class (mcp/deep.py); the class is
                    recorded beside the trigger
  tool-call check   NOT the request class: it is keyed on the RESPONSE. A
                    generation that writes or patches a file through a client
                    tool call IS code_edit work, whatever turn it happens on
                    -- an agent writes most of its files on agent_step turns.
                    See mcp/tool_code.py. A utility call has no client tools,
                    so it can never reach it.

A header that forces an augmentation (X-Yamadori-Features) still forces it:
a benchmark arm that says "repair on" means it, whatever the class.

THE SIGNALS ARE DETERMINISTIC, AND EACH ONE IS JUSTIFIED

  ends on a tool result   STRUCTURE: the role of the last non-system message.
                          A client tool result means the model is mid-way
                          through the client's loop (selection.question_of
                          uses the same fact to keep deep thinking out).
  harness notice          THE REAL PRODUCER'S WORDS (PROTOCOL rule 7). Hermes
                          resumes with "[System: The previous response was cut
                          off ...]", "[Context from the interrupted assistant
                          response]" and "[CONTEXT COMPACTION ...]" (corpus
                          turns 3409-3535, 3542-3636, 3642-3699), and users
                          type a bare "continue" (3539). Each carries no new
                          task; the task is the in-flight one.
  acts locally            selection.acts_locally, unchanged in meaning; the
                          verbs push / commit / deploy / publish / upload were
                          added on 2026-09-24 for "lets push this to a github
                          repo" (corpus 3542, 3653).
  code in the prompt      PARSED, not pattern-matched (PROTOCOL rule 8): a
                          fenced block tagged with a programming language, or
                          an untagged one tree-sitter reads as code (most of
                          its lines outside ERROR nodes in Python, TypeScript
                          or Rust). Hermes wraps every attachment in ``` and
                          a pasted game spec is prose: it does not parse.
  a code request          selection._CODE_TASK's shape (a write verb, then a
                          code noun within 80 characters), widened by the
                          verbs the benchmark prompts actually use ("define a
                          struct", "expose ... to C").
  a question              a "?" or an interrogative / lookup lead ("where",
                          "which", "explain", "find", "search") in the
                          instruction, code removed first.
  the gate's situation    domains.tool_admission, the fact about what is
                          indexed; only the situations that are evidence about
                          THIS request count (selection.READABLE_GATE). The
                          gate also offers on NO_DOMAIN_EVIDENCE and
                          HELD_SOURCE_UNMAPPED, which on 2026-09-22 opened it
                          for all 342 LiveCodeBench prompts.

WHAT THE SIGNALS READ. The verbs and the question read the user's
INSTRUCTION, not an attachment Hermes inlined after it
(selection.instruction_of): an attachment is material, the instruction says
what to do with it. The code-in-prompt check reads the whole last user turn,
because an attached source file IS code the instruction may ask to change.

MEASURED on the labelled corpus set (bench/route/labels.jsonl, rubric and
replay in bench/route/eval_route.py; the numbers are in AGENTS.md "The
code-work router").
"""
from __future__ import annotations

import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import selection  # noqa: E402

CLASSES = ("utility", "agent_step", "code_edit", "code_generation",
           "library_question", "prose")
CODE_CLASSES = frozenset({"code_generation", "code_edit"})

# ------------------------------------------------------------ structure ----


def ends_on(messages: list[dict]) -> str | None:
    """The role of the last non-system message: "user", "tool" (a client
    tool result; "function" is the legacy name), "assistant", or None."""
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            continue
        return "tool" if role == "function" else role
    return None


def _has_answers(messages: list[dict]) -> bool:
    return any(isinstance(m, dict) and m.get("role") == "assistant"
               for m in messages or [])


# Hermes' resume notices, verbatim heads (corpus, see the module docstring),
# and a bare "continue": a short turn whose whole instruction is to go on.
_NOTICE = re.compile(r"^\s*\[(?:system:|context from the interrupted|"
                     r"context compaction)", re.I)
_CONTINUE = re.compile(r"^\W*(?:please\s+)?(?:continue|resume|keep going|go on|"
                       r"carry on|proceed|go ahead)\b[^?]{0,60}$", re.I)


def harness_notice(instruction: str) -> str | None:
    """"notice" / "continue" when the turn carries no new task, else None."""
    text = (instruction or "").strip()
    if _NOTICE.match(text):
        return "notice"
    if _CONTINUE.match(text):
        return "continue"
    return None

# ----------------------------------------------------- code in the prompt --


# Fence tags that name a programming or configuration language. Anything
# code_check can parse, plus languages it cannot (the class is about whether
# the prompt carries code, not whether we can check it).
_CODE_TAGS = frozenset("""
python py python3 py3 typescript ts tsx mts cts javascript js jsx mjs cjs node
rust rs c h cpp c++ cc cxx hpp hh hxx json toml yaml yml wgsl glsl hlsl metal
go golang java kotlin kt swift cs csharp fsharp sql sh bash zsh shell fish
powershell ps1 bat cmd lua ruby rb php zig html css scss sass less vue svelte
dart scala haskell hs ocaml ml elixir ex erlang clojure clj r julia jl nim
cuda cu asm nasm makefile make cmake dockerfile proto graphql gql diff patch
solidity sol verilog vhdl lisp scheme elm purescript astro
""".split())
# Tags that say "this is not code".
_TEXT_TAGS = frozenset("text txt plaintext plain markdown md log logs console "
                       "output stdout stderr csv tsv prompt".split())
# Share of an untagged block's lines that may sit in tree-sitter ERROR nodes
# for it still to read as code. Prose in a fence fails every grammar on most
# lines; code with a typo fails on a few. 0.25 is a choice, checked on the
# fixtures in mcp/test_route.py, not a measurement.
_UNTAGGED_ERROR_SHARE = 0.25
_UNTAGGED_GRAMMARS = ("python", "typescript", "rust")


def _error_lines(code: str, lang: str) -> int:
    import code_check
    root = code_check.parse_tree(code, lang)
    if not root.has_error:
        return 0
    rows: set[int] = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            rows.update(range(n.start_point[0], n.end_point[0] + 1))
            continue
        if n.has_error:
            stack.extend(n.children)
    return len(rows)


def reads_as_code(body: str) -> str | None:
    """The grammar an UNTAGGED block parses in (mostly), or None."""
    lines = [ln for ln in (body or "").split("\n") if ln.strip()]
    if not lines:
        return None
    for lang in _UNTAGGED_GRAMMARS:
        try:
            bad = _error_lines(body, lang)
        except Exception:                                        # noqa: BLE001
            continue
        if bad <= _UNTAGGED_ERROR_SHARE * len(lines):
            return lang
    return None


def placeholder_only(body: str, tag: str) -> bool:
    """A block that parses to nothing but comments -- LiveBench's
    "```python\\n# YOUR CODE HERE\\n```" -- is where code goes, not code.
    Read with the tag's grammar when code_check has one."""
    import code_check
    lang = code_check.normalize_language(tag)
    if lang not in code_check.CODE_LANGS:
        return False
    try:
        root = code_check.parse_tree(body, lang)
    except Exception:                                            # noqa: BLE001
        return False
    kids = [n for n in root.named_children]
    return bool(kids) and all("comment" in n.type for n in kids)


def prompt_code(text: str) -> list[str]:
    """The languages of the code blocks `text` carries, in order ("?" for an
    untagged block that parses as code). Prose, logs and output in a fence
    are not code, and neither is a placeholder that holds only comments."""
    import code_check
    out: list[str] = []
    for b in code_check.fenced_blocks(text or ""):
        body = b["code"]
        if not body.strip():
            continue
        tag = (b.get("lang") or "").strip().lower()
        if tag in _TEXT_TAGS:
            continue
        if tag in _CODE_TAGS:
            if not placeholder_only(body, tag):
                out.append(tag)
            continue
        if not tag or tag not in _TEXT_TAGS:
            if reads_as_code(body):
                out.append(tag or "?")
    return out

# ------------------------------------------------------- the instruction ---


# Not "code" (a noun far more often: "In the TSL node code, the method ..."),
# not "extend" ("which class does it extend" -- context-economy ce12, ce26).
_WRITE = (r"write|implement|complete|finish|fix|create|generate|build|"
          r"rewrite|port|refactor|define|declare|expose|add|make|"
          r"translate|convert|produce|draft|modify|update")
_NOUN = (r"function|method|class|component|hook|type|interface|struct|enum|"
         r"trait|impl|program|script|module|solution|code|tests?|query|"
         r"algorithm|shader|kernel|library|crate|package|api|endpoint|cli|"
         r"parser|macro|schema|binding|wrapper|handler|reducer|migration|"
         r"snippet|regex|lines")
# _CODE_TASK's shape without its ``` alternative (a fence is decided by
# parsing, above), widened: "In Rust, define a struct", "Expose a Rust
# histogram to C", "write in Python the remaining lines of the program".
# The noun may not follow a hyphen: "make changes to non-test files" (the
# SWE-agent task statement) is not a request for tests.
_CODE_REQUEST = re.compile(
    r"\b(" + _WRITE + r")\b[^.?!\n]{0,80}?(?<![\w-])(" + _NOUN + r")s?\b",
    re.I)
# Three more shapes of a code request, each from the benchmark prompts the
# proxy is measured on (bench/domain/tasks*.jsonl), each one an imperative:
#   "..., export `GAUSS_WEIGHTS`: a WGSL constant"   export + a backticked name,
#                                                    at a clause head (so "why
#                                                    does three export `X`?"
#                                                    is not one)
#   "Write a Rust string interner", "Write a         a write verb, then a
#    reusable React 19 error boundary"               language name
#   "I need two Rust functions callable from C"      need/want + a count or
#                                                    article + a code noun
_EXPORT_NAME = re.compile(
    r"(?:^|[.;:!\n(]\s*|,\s*|\b(?:and|then|also)\s+)(?i:exports?)\b[^\n]{0,80}?"
    r"`[A-Za-z_$][\w$.]*")
_LANGS = (r"TypeScript|JavaScript|Rust|Python|Go|C\+\+|C#|C|WGSL|GLSL|HLSL|"
          r"React|Vue|Svelte|SQL|Bash|Lua|Zig|Kotlin|Java|Swift|Ruby|PHP|"
          r"CUDA|Solidity|Haskell|OCaml|Elixir|Node(?:\.js)?|Deno")
_WRITE_IN_LANG = re.compile(
    r"(?i:\b(?:write|implement|build|create|code|port|rewrite|translate|"
    r"convert)\s+(?:me\s+)?(?:(?:a|an|the|some|this|that|it)\s+)?"
    r"(?:[\w-]+\s+){0,3}?(?:in(?:to)?\s+)?)(?:" + _LANGS + r")(?![\w+#])")
_NEED_CODE = re.compile(
    r"\b(?:i|we)\s+(?:need|want|would\s+like)\s+(?:a|an|one|two|three|four|"
    r"some|\d+)\s+(?:[\w-]+\s+){0,3}?(?:" + _NOUN + r")s?\b", re.I)
# A write verb whose object is a SOURCE FILE (#14, docs/SELF-IMPROVEMENT-
# LOG.md): "implement the task by writing the whole of stats.py with
# write_file" names no code noun -- the file is the code. The live gate's
# agent-loop task routed `prose` on its first turn, both runs. A file name
# counts only with an extension in a code language (README.md, notes.txt
# and data.json are not code); a question about a file ("what does utils.py
# do?") is not a request (_asked). The verb list is _WRITE plus "writing",
# the form the live request used.
_SOURCE_EXT = (r"py|pyi|ts|tsx|mts|cts|js|jsx|mjs|cjs|rs|c|h|cc|cpp|cxx|hpp|"
               r"hh|go|java|kt|kts|swift|rb|php|lua|zig|wgsl|glsl|hlsl|cs|fs|"
               r"sh|bash|ps1|sql|vue|svelte|dart|scala|hs|ml|ex|exs|jl|nim|"
               r"cu|sol")
_WRITE_FILE = re.compile(
    r"\b(" + _WRITE + r"|writing)\b[^.?!\n]{0,80}?(?<![\w./-])"
    r"([\w./-]*[\w-]\.(?:" + _SOURCE_EXT + r"))(?![\w.])", re.I)
# Work on code the prompt carries: with code present, the object is implied
# ("fix it", "make this button nicer", "complete the given function").
_EDIT_VERB = re.compile(
    r"\b(?:" + _WRITE + r"|change|clean\s+up|simplify|optimi[sz]e|rename|"
    r"remove|delete|replace|continue|port|adapt|migrate|correct|debug|"
    r"improve|speed\s+up|tidy|reformat|format|restructure)\b", re.I)
_QUESTION_LEAD = re.compile(
    r"(?:^|[.;:!\n]\s*)(?:what|where|which|who|whom|whose|why|how|when|is|are|"
    r"was|were|does|do|did|can|could|should|would|will|has|have|explain|"
    r"describe|tell me|show me|list|compare|find|locate|look up|search|name|"
    r"give|identify|state)\b"
    r"|\b(?:to|and|then)\s+(?:search|find|locate|look up|grep)\b", re.I)
# A code request inside a sentence that ends in "?" is a question about code
# ("How do I write a function that ...?") unless it is asked as a request.
_REQUEST_LEAD = re.compile(r"^\W*(?:can|could|would|will)\s+you\b|^\W*please\b",
                           re.I)


def _prose_of(text: str) -> str:
    """The instruction without code: fences and backticked spans removed."""
    return selection._BACKTICK.sub(" ", selection._FENCE.sub(" ", text or ""))


def _asked(text: str, m) -> bool:
    """Is the match in a sentence that is a request, not a question?"""
    start = max(text.rfind(c, 0, m.start()) for c in ".!?\n") + 1
    ends = [i for i in (text.find(c, m.end()) for c in ".!?\n") if i >= 0]
    end = min(ends) if ends else len(text)
    sentence = text[start:end + 1]
    return not sentence.rstrip().endswith("?") or bool(
        _REQUEST_LEAD.search(sentence))


def code_request(instruction: str) -> str | None:
    """The phrase that asks for code to be written, or None."""
    prose = _prose_of(instruction)
    # the backticked name is the point of the export shape, so it reads the
    # instruction with fences removed but backticks kept
    raw = selection._FENCE.sub(" ", instruction or "")
    for rx, text in ((_CODE_REQUEST, prose), (_WRITE_IN_LANG, prose),
                     (_NEED_CODE, prose), (_WRITE_FILE, raw),
                     (_EXPORT_NAME, raw)):
        for m in rx.finditer(text):
            if _asked(text, m):
                return " ".join(m.group(0).split())[:80]
    return None


def is_question(instruction: str) -> bool:
    prose = _prose_of(instruction).strip()
    return "?" in prose or bool(_QUESTION_LEAD.search(prose))

# -------------------------------------------------------------- decision ---


def _last_user(messages: list[dict]) -> str:
    whole, _ctx, _speaking = selection.question_of(messages or [])
    return whole


def classify(messages: list[dict], *, client_tools: list[str] | None = None,
             util: dict | None = None, gate: dict | None = None,
             dbs: dict[str, str] | None = None) -> dict:
    """{class, because, signals} for one request. Pure: no network, no model.

    `client_tools` names the tools the CLIENT sent (never ours); `util` is
    proxy.utility_of's decision (header applied); `gate` is the tool gate's
    decision for this request, or None when none was computed; `dbs` the
    symbol tables selection.defined_symbols may read (None: the held package
    store alone, read-only).
    """
    whole = _last_user(messages)
    instruction, attached = selection.instruction_of(whole)
    last = ends_on(messages)
    sig: dict = {"ends_on": last, "client_tools": len(client_tools or []),
                 "instruction_chars": len(instruction),
                 "attached_chars": len(attached)}

    def out(cls: str, because: str) -> dict:
        return {"class": cls, "because": because, "signals": sig}

    # 1. utility ------------------------------------------------------------
    if util and util.get("utility"):
        sig["utility_form"] = (util.get("signals") or {}).get("form")
        return out("utility", util.get("because") or "a client side call")

    # 2. agent_step -----------------------------------------------------------
    if client_tools and last == "tool":
        return out("agent_step", "the conversation ends on a client tool "
                                 "result: a step in the client's own loop")
    notice = harness_notice(instruction) if client_tools and _has_answers(
        messages) else None
    sig["harness_notice"] = notice
    if notice:
        return out("agent_step", f"the last user turn is a harness {notice} "
                                 f"with no new task, and the client sent its "
                                 f"own tools: the in-flight loop continues")
    local = selection.acts_locally(instruction) if client_tools else None
    sig["acts_locally"] = local
    if local:
        return out("agent_step", f"the instruction asks to act on the user's "
                                 f"machine ({local!r}) and the client sent its "
                                 f"own tools")
    if last == "tool":
        # A tool result with no client tools: a client replaying a tool loop
        # of its own (the conversation carries results but no tool list).
        return out("agent_step", "the conversation ends on a tool result: a "
                                 "step in a loop, not a new request")

    # 3-4. code -------------------------------------------------------------
    langs = prompt_code(whole)
    req = code_request(instruction)
    question = is_question(instruction)
    sig.update(prompt_code=langs, code_request=req, question=question)
    if langs:
        edit = _EDIT_VERB.search(_prose_of(instruction))
        sig["edit_verb"] = edit.group(0) if edit else None
        if req or edit or not question:
            why = (f"the prompt carries code ({', '.join(langs[:4])}) and "
                   + (f"asks for work on it ({(req or edit.group(0))!r})"
                      if (req or edit) else "is not a question about it"))
            return out("code_edit", why)
    if req:
        return out("code_generation", f"the instruction asks for new code "
                                      f"({req!r})")

    # 5. library_question -----------------------------------------------------
    situation = (gate or {}).get("situation")
    offered = bool((gate or {}).get("offer"))
    readable = offered and situation in selection.READABLE_GATE
    held: dict = {}
    if question and offered and not readable:
        # The same readability deep thinking uses (selection.decide): a held
        # source DEFINES a name the question uses ("the Trait type" -- koota).
        # The gate's own probe is narrower on purpose (it only offers).
        try:
            held = selection.defined_symbols(
                instruction, dbs if dbs is not None else selection.symbol_dbs())
        except Exception:                                        # noqa: BLE001
            held = {}
        readable = bool(held)
    sig.update(gate=situation, readable=readable,
               held_symbols={k: v[:4] for k, v in held.items()})
    if question and readable:
        why = ("a held source defines "
               + "; ".join(f"{k}: {', '.join(v[:3])}" for k, v in held.items())
               if held else f"the tools are offered because {situation}")
        return out("library_question", f"a question, and {why}: indexed "
                                       f"source bears on it")
    # 6. prose ----------------------------------------------------------------
    if question and gate is None:
        why = "a question, but no tool gate was computed for it"
    elif question:
        why = (f"a question, but the gate's situation ({situation}) is not "
               f"evidence that indexed source bears on it")
    else:
        why = "not a code request, not a question about indexed source"
    return out("prose", why)


def is_code(route: dict | None) -> bool:
    return bool(route) and route.get("class") in CODE_CLASSES


def log_line(route: dict) -> str:
    return f"route: {route['class']} -- {route['because'][:200]}"
