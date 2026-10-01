#!/usr/bin/env python
"""THE CODE-WORK ROUTER: one class per request, decided once, in prepare.

WHY IT EXISTS (operator, 2026-09-24). Features that apply to one kind of
request read ONE class from here instead of each re-deciding "is this code?"
with its own words. It was built for the code pipelines -- fan-out and the
repair pass, REMOVED 2026-09-29 (docs/REMOVED.md) -- and today the skills
and the agent step's thinking cap read it.

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

  skills            skill_select (SKIP_CLASSES, the per-turn engine)
  thinking cap      an agent_step thinks at most tiers.AGENT_STEP_THINKING
                    (proxy.prepare); every other turn USER_TURN_THINKING
  x_yamadori.route  recorded on every response

A header that forces an augmentation (X-Yamadori-Features) still forces it,
whatever the class.

THE SIGNALS ARE DETERMINISTIC, AND EACH ONE IS JUSTIFIED

  ends on a tool result   STRUCTURE: the role of the last non-system message.
                          A client tool result means the model is mid-way
                          through the client's loop (selection.question_of
                          uses the same fact). A
                          harness's synthetic tool-media turn (OpenCode, Pi,
                          Cline; image_input.TOOL_MEDIA_TURNS) counts as
                          the tool result it carries.
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
  a code request          a write verb, then a code noun within 80
                          characters (fan-out's old word rule), widened by the
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

import image_input  # noqa: E402
import selection  # noqa: E402

CLASSES = ("utility", "agent_step", "code_edit", "code_generation",
           "library_question", "prose")
CODE_CLASSES = frozenset({"code_generation", "code_edit"})

# ------------------------------------------------------------ structure ----


def ends_on(messages: list[dict]) -> str | None:
    """The role of the last non-system message: "user", "tool" (a client
    tool result; "function" is the legacy name), "assistant", or None.

    A harness's SYNTHETIC TOOL-MEDIA TURN is "tool" (docs/VISION.md 5a):
    OpenCode, Pi and Cline carry a tool's image in a user message after the
    tool result, because a Chat Completions tool message holds text only.
    It is the tool result, not a new request (image_input.TOOL_MEDIA_TURNS,
    each row read from the harness's source)."""
    msgs = messages or []
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            continue
        if role == "user" and image_input.is_tool_media(msgs, i):
            return "tool"
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
# Fan-out's old code-task shape (a write verb, then a code noun) without its ``` alternative (a fence is decided by
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

# ----------------------------------------------------------- work intent ---
# DOES THIS USER TURN ASK FOR SOMETHING TO BE MADE OR CHANGED? (operator,
# 2026-09-27: "'Go implement' is not the trigger, every natural language
# variation of implement, build, make is.") Read by skill_select's server-tool
# recall (trigger 4: a user turn after an answer that sends the model off to
# build names yama_plan). The embedder and E1 cannot be run offline and have
# no head for this; the router's own STRUCTURE is reused instead: a clause is
# a request for work when its HEAD -- after discourse openers ("ok", "great",
# "now", "then") -- is
#   an imperative work verb                "build the village", "wire up X"
#   a request lead + a work verb           "can you add ...", "please create"
#   a desire lead + a work verb or object  "I want a leaderboard", "I'd like
#                                          you to set up ..."
#   let's + a work verb                    "let's do it", "let's build ..."
#   a sequencing adverb + a noun phrase    "now the HUD part", "next up: the
#   (an elliptical "build X next")         particle effects"
# and the clause is not a question (a "?" without a request lead, or an
# interrogative head), not negated ("don't build", "hold off") and not an
# information verb ("explain", "show me", "summarize", "check"). The work
# verbs are the router's own sets (_WRITE, selection._ACT_VERB, _EDIT_VERB)
# plus the pro-verbs that carry an earlier proposal forward (do, go ahead,
# proceed, carry on, ship, execute) and the phrasal verbs of making (put
# together, whip up, knock out, hook up, flesh out). Measured in-sample on
# bench/skills/work_intent.jsonl (mcp/test_tool_recall.py [intent]).
_PRO_VERBS = (r"do|go\s+ahead|go\s+for\s+it|proceed|carry\s+on|continue|"
              r"keep\s+going|ship|execute|start|begin|get\s+(?:going|started)|"
              r"tackle|finish|complete|put\s+together|whip\s+up|knock\s+out|"
              r"hook\s+up|wire\s+up|flesh\s+out|code(?:\s+up)?|land|roll\s+out|"
              r"turn|integrate|extend|implement|build|make|add|create|write|"
              r"scaffold|set\s*up|spin\s+up")
# selection._ACT_VERB less `run`: running something makes and changes
# nothing ("can you run the tests?" is a check).
_ACT_WORK = "|".join(v for v in selection._ACT_VERB[3:-1].split("|")
                     if v != "run")
_WORK_VERB = (r"(?:" + _PRO_VERBS + r"|" + _WRITE + r"|"
              + _ACT_WORK + r"|change|clean\s+up|simplify|"
              r"optimi[sz]e|replace|adapt|migrate|correct|improve|speed\s+up|"
              r"tidy|restructure|redo|rework|swap|move|hook)")
# What a clause may open with before its head: assent, praise, sequencing.
_OPENER = (r"(?:(?:ok(?:ay)?|alright|all\s+right|right|great|nice|cool|good|"
           r"perfect|awesome|excellent|sounds\s+good|looks\s+good|lgtm|yes|"
           r"yeah|yep|yup|sure|so|and|also|then|now|next|please|pls|thanks|"
           r"thank\s+you)\b[\s,.!:;-]*)*")
_REQUEST_HEAD = (r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
                 r"please\s+|pls\s+|go\s+ahead\s+and\s+|time\s+to\s+|"
                 r"(?:i|we)(?:\s+(?:want|need|would\s+like)|['’]d\s+like)"
                 r"\s+(?:you\s+to\s+|to\s+have\s+you\s+)?|"
                 r"let'?s\s+|let\s+us\s+|let’s\s+)")
_WORK_CLAUSE = re.compile(
    r"^" + _OPENER + r"(?:" + _REQUEST_HEAD + r")?(" + _WORK_VERB + r")\b",
    re.I)
# A desire for a THING: "I want a leaderboard that ...".
_WANT_THING = re.compile(
    r"^" + _OPENER + r"(?:i|we)(?:\s+(?:want|need|would\s+like)|['’]d\s+"
    r"like)\s+(?:a|an|the|some|another)\s+(?!(?:explanation|summary|"
    r"overview|review|answer|list|comparison|breakdown|update|status|"
    r"word|look|idea|sense|minute|second|moment|break|hand|hint|chance)\b)"
    r"\w+", re.I)
# "now the HUD part", "next: the inventory screen", "next up: ...": a
# sequencing adverb heading a verbless noun phrase.
_NEXT_PIECE = re.compile(
    r"^" + _OPENER.replace("now|next|", "") + r"(?:now|next(?:\s+up)?|then)"
    r"\s*[:,-]?\s*(?:on\s+to\s+|onto\s+)?(?:the|a|an|some|our|my)\s+"
    r"[\w-]+(?:\s+[\w-]+){0,3}\s*[.!]*$", re.I)
_INFO_VERB = re.compile(
    r"\b(?:explain|describe|tell|show|summari[sz]e|list|compare|review|"
    r"check|verify|look|read|analy[sz]e|clarify|walk\s+me|remind|think|"
    r"say|mean|know|see|understand|wonder)\b", re.I)
_NEGATED = re.compile(r"^" + _OPENER + r"(?:don'?t|do\s+not|don’t|never|"
                      r"hold\s+off|wait|stop|no\b|not\s+yet)", re.I)
_INTERROGATIVE = re.compile(
    r"^" + _OPENER + r"(?:what|where|which|who|whom|whose|why|how|when|is|"
    r"are|was|were|does|did|do\s+you|should|has|have|am)\b", re.I)


def _clauses(prose: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+|\s+[-–—]\s+", prose or "")
    return [p.strip() for p in parts if p and p.strip()]


def work_intent(instruction: str) -> str | None:
    """The clause of a user turn that asks for something to be made or
    changed, or None. Pure; reads the prose (code removed first)."""
    for c in _clauses(_prose_of(instruction)):
        if _NEGATED.match(c) or _INTERROGATIVE.match(c):
            continue
        m = _WORK_CLAUSE.match(c)
        if m:
            head = c[:m.end()]
            asked = c.rstrip().endswith("?")
            if asked and not re.search(r"(?:can|could|would|will)\s+you",
                                       head, re.I):
                continue
            if _INFO_VERB.fullmatch(m.group(1).split()[0]):
                continue
            return " ".join(c.split())[:80]
        if (_WANT_THING.match(c) or _NEXT_PIECE.match(c)) and \
                not c.rstrip().endswith("?"):
            return " ".join(c.split())[:80]
    return None

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
    media_row = image_input.last_is_tool_media(messages)
    sig: dict = {"ends_on": last, "client_tools": len(client_tools or []),
                 "instruction_chars": len(instruction),
                 "attached_chars": len(attached)}
    if media_row:
        sig["tool_media_turn"] = media_row["harness"]
    media_why = (f"the conversation ends on {media_row['harness']}'s "
                 f"synthetic turn carrying a tool result's media: a step in "
                 f"the client's own loop" if media_row else None)

    def out(cls: str, because: str) -> dict:
        return {"class": cls, "because": because, "signals": sig}

    # 1. utility ------------------------------------------------------------
    if util and util.get("utility"):
        sig["utility_form"] = (util.get("signals") or {}).get("form")
        return out("utility", util.get("because") or "a client side call")

    # 2. agent_step -----------------------------------------------------------
    if client_tools and last == "tool":
        return out("agent_step", media_why or "the conversation ends on a "
                                 "client tool result: a step in the client's "
                                 "own loop")
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
        return out("agent_step", media_why or "the conversation ends on a "
                                 "tool result: a step in a loop, not a new "
                                 "request")

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
        # The readability selection's legacy path used: a held
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
