#!/usr/bin/env python
"""The selection engine: per request, whether skills may run -- and the
reading of a request that the route and the utility rule share.

    select / decide   may skills be attached at all? (the tier allows, a
                      header forces; the skills system abstains per skill)
    utility_call      is this a client's own side call (a title, a
                      classifier, a summary of the conversation)?
    question_of /     the user's question, its context, and the instruction
    instruction_of    apart from an attachment the harness inlined after it
    acts_locally      does the instruction ask to act on the user's machine?
    symbol_dbs /      does a held package DEFINE a name the question uses?
    defined_symbols   (mcp/route.py's library_question)
    rule_baseline     the six-line regex E1 is compared against (mcp/e1.py)

D0 (which systems the caller ALLOWS) is the tier, `mcp/tiers.py`. This module
never widens it: a tier flag means ALLOWED, not ON, and nothing decided here
exceeds what the client asked for. The one exception is deliberate and loud
-- a flag set in `X-Yamadori-Features` (`tier["overridden"]`) is FORCED on
or off, because a benchmark arm must be able to say "on, always" or "off,
always" and mean it.

REMOVED 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37): the
deep-thinking decision (the triggers' record, and the LEGACY path: the regex
+ symbol lookup + Laya's route_in head, two signals where disagreement
escalated -- held-out 91/120 against 89/120 for the rule alone, p=0.79, not
significant), the fan-out rule (design / code-task words, then the route's
code classes; its only measurement a null, 7/8 vs 7/8 at 3.2x wall clock,
n=8) and the Laya service call.

THE HARD-SLICE CHECK: THE SYMBOL LOOKUP (defined_symbols; mcp/route.py)

The regex reads surface cues ("our", "src/", a `.ts` path). The questions it
cannot read are the ones about a LIBRARY'S source phrased as general
questions -- "what is the default value of `Object3D.DEFAULT_UP`" has no
cue at all. Whether a held package DEFINES a name the question uses is not a
reading of the prose; it is a lookup in the index that would answer it.

This probe is WIDER than `domains._symbols`. Code-shaped names (backticks,
an internal capital, a digit, an underscore) are asked of every symbol
table; a capitalised / ALLCAPS word in identifier syntax (`THREE.MOUSE`,
`Loop()`) only of a package the text names. English-shaped ones -- a
capitalised word used as a code noun ("the renderer's Pipelines module") --
count only where a library defines them in its own source, not its examples
or tests: three.js's examples define `For`, `Number` and `String`, and a
LiveCodeBench puzzle's "a Zero Array" and "Group A" matched typegpu's `Array`
and three's `Group` until the code-noun rule. A BARE ALLCAPS WORD IS NOT
PROBED (2026-09-23): "MOUSE" and "API" in a pasted game spec matched three's
`MOUSE` and wgpu-matrix's `API`. Names the platform defines (`Event`,
`Array`, `Math`) never count (domains.PLATFORM_NAMES).

AN AGENT HARNESS ACTING LOCALLY, AND ATTACHMENTS

When the client sent its own tools and the instruction asks to act on the
user's machine ("start this project in ~/Developer/x"), the request is the
client's agent loop to run (`acts_locally`; mcp/route.py). What is read is
the user's instruction, not an attachment the harness inlined after it
(`instruction_of`). Both are explained where they are defined.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# The shortest instruction selection reads as a question.
MIN_QUESTION_CHARS = 8

# How much earlier user text rides along as `context`. The labels' context
# field is the user's own background ("we're on three 0.185.0, the component
# is src/scene/..."), which in a live conversation is their earlier turns.
CONTEXT_CHARS = 1500


# ====================================================== THE REGEX (moved) ===
#
# Moved verbatim from bench/laya_calibration.py on 2026-09-22 so that every
# evaluator runs ONE copy (today E1's, mcp/e1.py: the rule it is compared
# against). bench/data/rule_baseline_golden.json holds its 89
# predictions as captured before the move; mcp/test_selection.py asserts they
# are reproduced exactly.

_SYMBOL_PATTERNS = [
    # A path with a source extension: the strongest "this is our code" signal.
    re.compile(r"\b[\w./\\-]+\.(?:ts|tsx|js|jsx|mjs|cjs|rs|py|wgsl|glsl|toml|json|sqlite3)\b"),
    # CamelCase with at least two humps, not a sentence-initial capital.
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),
    # snake_case / SCREAMING_CASE identifiers.
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
    # dotted or ::-qualified names, and calls.
    re.compile(r"\b\w+\.\w+\(\)|\b\w+::\w+\b|\b[a-z]\w+\(\)"),
]
_PROSE_CAMEL_OK = {"TypeScript", "JavaScript", "WebGPU", "WebGL", "TypeGPU"}


def named_symbols(text: str) -> list[str]:
    """Concrete identifiers the caller actually named. Cheap, no model."""
    out: list[str] = []
    for pat in _SYMBOL_PATTERNS:
        for m in pat.findall(text):
            s = m if isinstance(m, str) else m[0]
            if s in _PROSE_CAMEL_OK or s in out:
                continue
            out.append(s)
    return out[:8]


def _index_facts() -> dict:
    """Index size, read WITHOUT opening it the way code_search does.

    code_search._db() runs CREATE TABLE IF NOT EXISTS on connect, which is a
    write to a file another workstream owns. A read-only URI connection gets
    the same two numbers and cannot touch it.
    """
    db = os.environ.get("CODE_INDEX_DB",
                        os.path.join(ROOT, "index", "code.sqlite3"))
    if not os.path.exists(db):
        return {"exists": False, "chunks": 0, "files": 0}
    try:
        uri = "file:" + os.path.abspath(db).replace("\\", "/") + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            chunks, files = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT path) FROM chunks").fetchone()
        finally:
            con.close()
        return {"exists": chunks > 0, "chunks": int(chunks), "files": int(files)}
    except Exception:                                            # noqa: BLE001
        return {"exists": False, "chunks": 0, "files": 0}


_INDEX = None


def index_facts() -> dict:
    global _INDEX
    if _INDEX is None:
        _INDEX = _index_facts()
    return _INDEX


def signals(item: dict) -> dict:
    """The cheap structured signals. No GPU, no model, ~1 ms per item."""
    import discover
    import domains

    q = item["question"]
    ctx = item.get("context", "") or ""
    blob = f"{q}\n{ctx}"
    msgs = [{"role": "user", "content": blob}]
    try:
        doms = sorted(domains.detect(msgs))
    except Exception:                                            # noqa: BLE001
        doms = []
    try:
        libs = list(dict.fromkeys(discover.imports(blob)))
    except Exception:                                            # noqa: BLE001
        libs = []
    idx = index_facts()
    return {
        "domains": doms,
        "libraries": libs,
        "symbols": named_symbols(blob),
        "code_block": "```" in blob,
        "words": len(q.split()),
        "context_words": len(ctx.split()),
        "index_chunks": idx["chunks"],
        "index_files": idx["files"],
        "index_exists": idx["exists"],
    }


# Without this, "a head gets 62%" is unreadable: the majority class alone
# gets 39%, and six lines of regex may get more than a model does. A router
# is only worth a GPU if it beats the thing you would have written anyway.
_CLARIFY_HINT = re.compile(
    r"^\s*(it'?s broken|fix it|can you (make|check)|why doesn'?t this|"
    r"does this look|which one is better|update the|add the thing|"
    r"should i use the other|is this the right)", re.I)


def rule_baseline(item: dict) -> str:
    s = signals(item)
    if not s["symbols"] and not s["libraries"] and s["words"] <= 8:
        return "clarify"
    if _CLARIFY_HINT.search(item["question"]) and not s["symbols"]:
        return "clarify"
    blob = f"{item['question']}\n{item.get('context', '')}".lower()
    if re.search(r"\b(our|we|us|this repo|codebase|pinned|crates/|src/|mcp/)\b", blob):
        return "investigate"
    if s["symbols"] and re.search(r"\.(ts|tsx|rs|py|wgsl)\b", blob):
        return "investigate"
    return "answer_directly"


# ================================================ THE SYMBOL LOOKUP =========

# domains._TOKEN's shape: identifier-looking, 4..64 characters.
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{2,63}")
_CAMEL = re.compile(r"[a-z][A-Z]")
_DIGIT = re.compile(r"[A-Za-z].*\d|\d.*[A-Za-z]")
_BACKTICK = re.compile(r"`([^`\n]{1,120})`")
# One identifier or dotted path, optionally called: `sizeOf`, `d.vec3f`,
# `tgpu.bindGroupLayout()`. Anything else in backticks is a statement.
_SINGLE_NAME = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:\(\))?")
# Words of the languages themselves, never a library's API: TS/JS and Rust.
_KEYWORDS = frozenset("""
import from export default const let var function return type interface enum
class extends implements new this super as of in for if else while do switch
case break continue async await yield typeof instanceof void null undefined
true false public private protected readonly static declare namespace module
require keyof infer never unknown any string number boolean bigint symbol
object satisfies abstract override get set delete throw try catch finally
fn pub struct impl use mod mut ref self where trait crate unsafe extern match
loop move dyn usize isize u8 u16 u32 u64 i8 i16 i32 i64 f64 bool str
""".split())
# A capitalised word INSIDE a sentence and USED AS A CODE NOUN: after a
# lowercase word, comma or possessive, and followed by "module", "class",
# "method"... -- "the renderer's Pipelines module", "the common Info module".
# Capitalised mid-sentence alone is not enough: LiveCodeBench puzzles say
# "a Zero Array" and "Group A", and typegpu defines `Array`, three `Group`.
_CODE_NOUN = (r"module|class|constant|function|method|node|property|type|"
              r"interface|enum|hook|component|file|export|namespace|instance|"
              r"helper|utility|struct|trait|macro|crate|package|library")
_MID_CAPITAL = re.compile(r"(?<=[a-z,'’] )([A-Z][a-z0-9]{2,40})"
                          r"(?=\s+(?:" + _CODE_NOUN + r")\b)")
# A bare ALLCAPS word is NOT probed (2026-09-23). It used to be, against a
# library's src/, and a Hermes request's pasted spec matched three's `MOUSE`
# (an ALLCAPS heading) and wgpu-matrix's `API` ("Web Audio API"); deep
# thinking (removed 2026-09-29) then ran for minutes on a request to create
# files locally. ALLCAPS
# is how prose writes acronyms and headings. It now counts only in code
# context: backticked (`REVISION`), code-shaped (DEFAULT_UP, PI2), or in
# identifier syntax (THREE.MOUSE, MOUSE.LEFT) -- domains.code_context_names.
MAX_PROBES = 400


def split_probe_tokens(text: str) -> tuple[list[str], list[str]]:
    """(code-shaped names, English-shaped names) in `text`.

    CODE-SHAPED: anything in backticks, identifiers with an internal capital
    (`PMREMGenerator`, `WebGPUBackend`), a digit or an underscore -- English
    words have none of these. A capitalised or ALLCAPS word standing in
    identifier syntax (`THREE.MOUSE`, `Loop()`; domains.code_context_names)
    is code too, and is kept with the plain backticked words (`_LAST_PLAIN`),
    which count only against a package the text names. ENGLISH-SHAPED: a
    capitalised word mid-sentence USED AS A CODE NOUN ("the renderer's
    Pipelines module").
    That is also how prose capitalises, so it counts only against a library's
    own source, never its examples or tests -- see defined_symbols. A bare
    ALLCAPS word (API, MOUSE, POINT) is neither, and names the platform
    defines (`Event`, `Array`, `Math`: domains.PLATFORM_NAMES) never count.
    """
    import domains
    code: dict[str, None] = {}
    words: dict[str, None] = {}
    plain: dict[str, None] = {}

    def add(bag: dict, tok: str) -> None:
        tok = tok.strip("$")
        if tok in domains.PLATFORM_NAMES:
            return
        if (len(tok) >= 2 and tok not in code and tok not in words
                and len(code) + len(words) < MAX_PROBES):
            bag[tok] = None

    def code_shaped(tok: str) -> bool:
        # ALLCAPS is not an internal capital: `REVISION` and `KEYENCE` are
        # English-shaped (below), `PMREMGenerator` is code-shaped.
        return bool(_CAMEL.search(tok)
                    or (tok[1:] != tok[1:].lower() and not tok.isupper())
                    or _DIGIT.search(tok) or "_" in tok.strip("_$"))

    # A backticked NAME counts whole (`LightUniform`, `d.vec3f`, `sizeOf()`),
    # whatever its shape. Inside a backticked STATEMENT, code-shaped names
    # count everywhere; plain-word names (`velocity`, `data`, `tgpu`) are real
    # variables too, so they are KEPT -- in `plain`, which defined_symbols
    # checks only against packages the question itself names or imports.
    # Language keywords never count. The case that shaped this: a prompt
    # backticked `import * as d from 'typegpu/data'`, and `import`/`from`
    # matched definitions and `data`/`velocity` matched three.js and
    # wgpu-matrix -- packages the question never mentioned (bench/domain
    # smoke tg01, 2026-09-22).
    for span in _BACKTICK.findall(text):
        s = span.strip()
        single = _SINGLE_NAME.fullmatch(s)
        for tok in _IDENT.findall(span):
            if tok in _KEYWORDS:
                continue
            if single or code_shaped(tok):
                add(code, tok)
            elif tok not in plain:
                plain[tok] = None
    for tok in _IDENT.findall(text):
        if code_shaped(tok):
            add(code, tok)
    # A capitalised / ALLCAPS word in identifier syntax (THREE.MOUSE, Loop())
    # is code -- but often the USER's code (`LogLevel.Warn`, `React.FC`), and
    # a held package defining the same short name says little: on the
    # benchmark prompts (bench/domain/tasks*) asking every table matched
    # fiber's `React`, @types/three's `Equal`, typegpu's `Warn` and three's
    # `Token` in 21 React / type-challenge tasks. So these go with the
    # backticked plain words: asked only of a package the text names.
    for tok in domains.code_context_names(text):
        if tok not in code and tok not in words and tok not in plain:
            plain[tok] = None
    for tok in _MID_CAPITAL.findall(text):
        add(words, tok)
    _LAST_PLAIN[:] = [t for t in plain if t not in code and t not in words]
    return list(code), list(words)


# The plain-word backticked names of the last split, read by defined_symbols.
# Kept out of the return value so every existing (code, words) caller holds.
_LAST_PLAIN: list[str] = []


def backticked_plain_words(text: str) -> list[str]:
    """Plain-word names from backticked code in `text` (`velocity`, `data`),
    keywords excluded. They count only against packages the text names."""
    split_probe_tokens(text)
    return list(_LAST_PLAIN)


def named_packages(text: str, names) -> set[str]:
    """Which of `names` (held package names) the text names or imports.

    The tool gate's own matcher (domains._named): an alias, else the install
    name as a whole token. This was `n.lower() in text.lower()`, a SUBSTRING
    test, so the numeral in "three paragraphs" named three.js -- the exact
    case HELD_ALIASES exists to refuse -- and "postprocessing" as a word named
    the pmndrs package.
    """
    import domains
    return set(domains._named(text, {n: None for n in names}))


def probe_tokens(text: str) -> list[str]:
    """Every name in `text` worth asking a symbol table about. See the module
    docstring for why this is wider than the tool gate's probe."""
    code, words = split_probe_tokens(text)
    return code + words


# Where a library keeps what is NOT its API. An English-shaped name defined
# only here is not evidence: three.js's examples/jsm/transpiler/AST.js
# defines `For`, `Number` and `String`, and examples/jsm/objects/Water.js
# `Water` -- found when a LiveCodeBench puzzle ("an Array of ... Water")
# matched them. `Pipelines`, `Animation` and `REVISION` live in src/.
_NOT_API = r"(^|/)(examples?|tests?|__tests__|demos?|docs?|benchmarks?|fixtures?)/"


def symbol_dbs(root_db: str | None = None) -> dict[str, str]:
    """{source name: sqlite path} for every symbol table a lookup may read:
    each live held package (newest version) and the bound repository."""
    import domains
    out = {name: vs[0][1] for name, vs in domains.held_sources().items()}
    if root_db and os.path.exists(root_db):
        out["(bound repository)"] = root_db
    return out


def defined_symbols(text: str, dbs: dict[str, str]) -> dict[str, list[str]]:
    """Which of the names in `text` each source's symbol table DEFINES.

    Read-only URIs: nothing here may create or touch an index. A source with
    no `defs` table is skipped -- that is the absence of a signal, not a
    signal against the source.
    """
    code, words = split_probe_tokens(text)
    plain = list(_LAST_PLAIN)
    if not (code or words or plain) or not dbs:
        return {}
    named = named_packages(text, list(dbs))
    not_api = re.compile(_NOT_API)
    out: dict[str, list[str]] = {}
    for name, db in dbs.items():
        # Plain-word names (`velocity`) only against a package the question
        # names; code-shaped and English-shaped as before, against all.
        groups = [(code, False), (words, True)]
        if name in named or name == "(bound repository)":
            groups.append((plain, False))
        try:
            uri = "file:" + os.path.abspath(db).replace("\\", "/") + "?mode=ro"
            con = sqlite3.connect(uri, uri=True)
            try:
                found: set[str] = set()
                for toks, strict in groups:
                    for i in range(0, len(toks), 200):
                        part = toks[i:i + 200]
                        q = ("SELECT name, path FROM defs WHERE name IN (%s)"
                             % ",".join("?" * len(part)))
                        for nm, path in con.execute(q, part):
                            if strict and not_api.search(
                                    (path or "").replace("\\", "/")):
                                continue
                            found.add(nm)
            finally:
                con.close()
        except sqlite3.Error:
            continue
        if found:
            out[name] = sorted(found)
    return out


# ================================== THE INSTRUCTION, NOT THE ATTACHMENT =====
#
# Hermes sends an attached file INLINE in the user turn, after the user's own
# words and a fixed delimiter -- captured from the real producer,
# index/corpus.sqlite3 event 3396, 2026-09-23 (PROTOCOL rule 7):
#
#     @file:`.hermes/attachments/Pasted content (8.8 KB)`
#
#     I want to start this project in ~/Developer/octopus-invaders
#
#     --- Attached Context ---
#
#     📄 @file:`...` (2246 tokens)
#     ```
#     build a space shooter game with vanilla JavaScript ...
#
# DECISION: the signals that ESCALATE -- the regex, the held-symbol lookup,
# the act-locally rule -- read the user's instruction, not the attachment. An attachment
# is material to act on; the instruction says what to do with it. The
# attachment above is a game spec for the user's OWN project: its ALLCAPS
# headings matched held definitions, and the ``` Hermes wraps every
# attachment in read as "a code-writing task". Evidence that an
# attachment really is about a held library still arrives as a FACT: its
# imports are parsed (discover.scan -> the session's packages, and the gate's
# IMPORTS_HELD_SOURCE), and the gate still reads every message, because an
# offer costs a tool list while an escalation costs minutes of the helper
# lane. Only this delimiter is recognised: it is the one format captured from
# a real client. A message without it is read whole, as before.
ATTACHMENT_DELIMITERS = ("\n--- Attached Context ---",)


def instruction_of(q: str) -> tuple[str, str]:
    """(the user's own instruction, the attached context) of one user turn.
    The attachment is "" when the turn carries no recognised delimiter."""
    cut = min((i for i in (q.find(d) for d in ATTACHMENT_DELIMITERS) if i >= 0),
              default=-1)
    if cut < 0:
        return q, ""
    return q[:cut], q[cut:]


# ==================================================== ACTING LOCALLY =========
#
# A turn in an agent harness that asks to act on the user's own machine --
# create, start, scaffold, set up, install, run, build or move a project,
# repo, folder or file -- is work for the CLIENT's tools (file write,
# terminal), which only the client can run. (Deep thinking and fan-out,
# removed 2026-09-29, held such a turn for minutes and could create no file
# on the user's disk.) Live, 2026-09-23: "I want to start this
# project in ~/Developer/octopus-invaders" ran deep thinking for 26,660 prompt
# + 19,515 decoded tokens before it was killed, and the client's tools were
# never called.
#
# The rule needs BOTH halves: the client supplied its own tools (so there is
# an agent loop to hand the work to), and the instruction is a REQUEST to act
# locally -- an action verb at the head of a clause or after a request lead
# ("I want to", "let's", "please", "can you"), whose DIRECT OBJECT is a local
# thing ("this project", "a vite + three.js app", "the files") or which
# names a local place in the same clause (~/, ./, C:\, /home/..., "in this
# folder"). A question ABOUT doing it ("how do I set up ...") is not a
# request; a path alone ("in ~/app/scene.ts, why does X warn?") is not one;
# and "write a function that reads a file" is not one either -- the file is
# not what is being written. Code (fences, backticks) is removed first:
# `npm run build` in a question is a quote, not an instruction.
_LOCAL_PATH = (r"(?:~[\\/]|(?<![\w.])\.{1,2}[\\/]|\b[A-Za-z]:[\\/]|"
               r"(?<![\w.])/(?:home|Users|tmp|mnt|opt|srv|workspace|root)/|"
               r"%USERPROFILE%|\$HOME\b)")
_LOCAL_OBJECT = (r"(?:project|repo(?:sitory)?|(?:sub-?)?folders?|"
                 r"(?:sub-?)?director(?:y|ies)|dir|files?|app|application|"
                 r"workspace|package|monorepo|template|boilerplate|codebase|"
                 r"site|website|game|server|venv|virtualenv|environment|"
                 r"dependencies|tests?|scripts?|build)")
_DETERMINER = r"(?:a|an|the|this|that|these|those|my|our|new|some|\d+)"
# verb + (it|up)? + determiner + up to three modifiers + the object.
_DIRECT_OBJECT = (r"\s+(?:it\s+|up\s+)?" + _DETERMINER
                  + r"\s+(?:[\w.+#/-]+\s+){0,3}?" + _LOCAL_OBJECT + r"\b")
_LOCAL_PLACE = (r"[^.?!\n]{0,100}?(?:" + _LOCAL_PATH
                + r"|\b(?:in|into|inside|under|to|at)\s+(?:this|the|my|a|that|our)"
                r"\s+(?:[\w.+-]+\s+){0,3}?(?:sub-?)?(?:folder|director(?:y|ies)|dir)\b"
                r"|\bon (?:my|this|the) (?:local )?"
                r"(?:machine|computer|disk|laptop|desktop)\b)")
# "your task is (specifically) to" added 2026-09-24 (mcp/route.py): the
# SWE-agent harness states its task that way ("Your task is specifically to
# make changes to non-test files in the current directory", corpus 2279 on)
# with its own bash tool, and the corpus cuts its later numbered steps.
_LEAD = (r"(?:^|[.!?;:\n]\s*|\b(?:and|then)\s+|"
         r"\b(?:please|pls|let'?s|let us|i want to|i'?d like to|i would like to|"
         r"i need to|we need to|want you to|can you|could you|would you|"
         r"will you|help me|go ahead and|now)\s+|"
         r"\byour\s+(?:task|job|goal)\s+is\s+(?:\w+\s+)?to\s+)")
# push / commit / deploy / publish / upload added 2026-09-24 (mcp/route.py):
# "lets push this to a github repo, ... use the gh command" (corpus 3542,
# 3653) is the client's git and gh to run, and read as no request at all.
_ACT_VERB = (r"(?:create|make|start|scaffold|set\s*up|init(?:iali[sz]e)?|"
             r"bootstrap|install|run|build|move|copy|rename|delete|remove|"
             r"clone|generate|write|save|put|mkdir|spin up|kick off|push|"
             r"commit|deploy|publish|upload)")
_ACT_LOCALLY = re.compile(
    _LEAD + r"(" + _ACT_VERB + r")\b(?:" + _DIRECT_OBJECT + r"|"
    + _LOCAL_PLACE + r")", re.I)
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)


def acts_locally(instruction: str) -> str | None:
    """The phrase that asks to act on the user's machine, or None."""
    prose = _BACKTICK.sub(" ", _FENCE.sub(" ", instruction))
    m = _ACT_LOCALLY.search(prose)
    return " ".join(m.group(0).split())[:120] if m else None


# ================================================= CLIENT UTILITY CALLS ======
#
# An agent harness interleaves its main turns with calls of its OWN: a
# command-approval classifier, a session title, a compaction of its history.
# Live, Hermes, 2026-09-23 (corpus turns 510e1d, fbe1dc, 9cca1a, bbddcf, and
# every row since 09-23 whose system prompt is a reviewer or a title namer):
# each got the capability block, our tools and recipes. 9cca1a took 409 s to
# answer one word and CALLED run_check; the compaction bbddcf took 355 s. None
# of them is a task the code tools or a second brain can serve, and every one
# holds the GPU while the user's real turn waits.
#
# THE RULE -- all three, and the first two are STRUCTURE, not wording:
#
#   1. the client sent no tools of its own. A harness's main loop always
#      sends its tools; its side calls send none.
#   2. a single exchange: no assistant or tool message. A side call carries its
#      own one-shot instruction; a conversation that has had answers is a
#      conversation.
#   3. the reply is fixed by a CONTRACT rather than asked for as work:
#      a. `response_format` asks for JSON -- the client declared the shape;
#      b. the instruction fixes the REPLY to a closed form: respond / reply /
#         answer with exactly one word (label, letter, number...), exactly one
#         of a set, JSON only, or yes or no. The verb must address the reply:
#         "print a single integer" is a program's output in a puzzle, and
#         "name it in one word" is a question;
#      c. the job is to summarise a conversation handed over as data: a
#         summarise / compact / condense verb and a conversation noun in the
#         head of the same message.
#
# Why not "no tools" alone: it is also every chat UI and every benchmark
# (342 LiveCodeBench prompts, the domain suite), which are exactly the
# callers the proxy exists to augment. Why not "no tools + first turn": the
# same. Rule 3 is what separates a contract from a task, and it is measured
# on the real corpus and the benchmark prompt sets by mcp/test_utility.py
# (PROTOCOL rule 7) -- see that file for the counts.
#
# A utility call gets the bare model at the tier it resolves to: no capability
# block, no tools, no skills, no seed, and no session (it neither reads nor writes the conversation's offered-tools flag,
# work log or pins -- proxy.session_context). X-Yamadori-Features {"utility":
# true|false} forces it; a header that forces an augmentation ON wins over the
# rule, because a benchmark that forced it meant it.
_REPLY = r"\b(?:respond|reply|answer)\b"
_CLOSED_FORMS = (
    ("one_word", re.compile(
        _REPLY + r"[^.\n]{0,40}?\b(?:exactly|only|just)\s+(?:one|a\s+single|1)"
        r"\s+(?:word|label|letter|token|number|digit|integer|character|"
        r"category)\b", re.I)),
    ("one_of", re.compile(
        _REPLY + r"[^.\n]{0,40}?\b(?:exactly|only)\s+one\s+of\b", re.I)),
    ("json_only", re.compile(
        _REPLY + r"[^.\n]{0,30}?\bjson\b[^.\n]{0,60}?(?:\bonly\b|nothing else|"
        r"no other text)|" + _REPLY + r"[^.\n]{0,20}?\bonly\s+(?:with\s+|in\s+)?"
        r"(?:a\s+|one\s+)?(?:valid\s+)?json\b", re.I)),
    ("yes_no", re.compile(
        _REPLY + r"\s+(?:(?:only|just|with)\s+)*[\"'`*]?(?:yes|true)[\"'`*]?"
        r"\s*(?:or|/)\s*[\"'`*]?(?:no|false)\b", re.I)),
)
_SUMMARISE = re.compile(r"\b(?:summari[sz](?:e|es|ed|ing|ation|er)|"
                        r"compact(?:ion|ing|ed)?|condens(?:e|es|ed|ing))\b", re.I)
_CONVERSATION = re.compile(r"\b(?:conversations?|transcripts?|chat\s+history|"
                           r"dialog(?:ue)?s?|message\s+history)\b", re.I)
# Where a summarising job states itself: the head of the message, before the
# material it hands over. 1,000 characters holds the Hermes compaction's
# whole instruction (its first conversation noun is at ~110).
SUMMARY_HEAD_CHARS = 1000


def closed_form(text: str) -> str | None:
    """The name of the closed reply form `text` asks for, or None."""
    for name, rx in _CLOSED_FORMS:
        if rx.search(text or ""):
            return name
    return None


def summarises_conversation(text: str) -> bool:
    head = (text or "")[:SUMMARY_HEAD_CHARS]
    return bool(_SUMMARISE.search(head) and _CONVERSATION.search(head))


# A TITLE CALL names the conversation; its reply is one line. The harnesses'
# own words, in the head of the system or user text:
#   Hermes    "You name chat sessions. ... write a title" (corpus 510e1d)
#   OpenCode  "You are a title generator. You output ONLY a thread title."
#             plus a user turn "Generate a title for this conversation:"
#             (opencode-ai 1.18.32, SessionPrompt.ensureTitle and the title
#             agent's prompt; corpus 6580, 6584, 6600, ...)
_TITLE = re.compile(
    r"\b(?:title\s+generator|only\s+(?:a|the)\s+(?:thread|session|chat|"
    r"conversation)\s+title|"
    r"(?:generate|write|create)\s+(?:a|an)\s+(?:\w+\s+){0,2}title\s+for\s+"
    r"(?:this|the)\s+(?:conversation|chat|session|thread)|"
    r"name\s+chat\s+sessions)\b", re.I)


def names_a_title(text: str) -> bool:
    return bool(_TITLE.search((text or "")[:SUMMARY_HEAD_CHARS]))


def _harness_compaction(messages: list[dict]) -> bool:
    """A harness's own flattened compaction whose summarise verb is past the
    head (OpenCode's "Create a new anchored summary ..." follows the whole
    transcript): mcp/compaction.harness_of, on the last user turn."""
    import compaction
    users = [m for m in messages if isinstance(m, dict)
             and m.get("role") == "user"]
    if not users:
        return False
    system = next((_text(m) for m in messages if isinstance(m, dict)
                   and m.get("role") in ("system", "developer")), "")
    return compaction.harness_of(_text(users[-1]), system) is not None


# Content-part types that carry an image: the same set as vision.IMAGE_PARTS
# (mcp/test_utility.py asserts they match; not imported, to keep this module
# free of the GPU-side imports vision pulls in).
IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def carries_image(messages: list[dict]) -> bool:
    """Does any message carry an image part?"""
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") in IMAGE_PART_TYPES
                for p in c):
            return True
    return False


def utility_call(messages: list[dict], client_tools: list[str] | None = None,
                 response_format=None) -> dict:
    """{utility, because, signals}: is this a client's own side call rather
    than a task turn? The rule and its evidence are in the note above."""
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    sig: dict = {"client_tools": len(client_tools or []),
                 "single_exchange": not any(r in ("assistant", "tool", "function")
                                            for r in roles),
                 "form": None}
    if carries_image(messages):
        # A REQUEST CARRYING AN IMAGE IS NEVER A SIDE CALL (pre-deploy
        # review, 2026-09-24). Hermes' vision_analyze asks the main provider
        # -- us -- "describe this image" in one exchange with no tools, which
        # read as a side call and went to the bare text model, which cannot
        # see. It is a task for the vision path (yama_describe_image).
        # ONE EXCEPTION (operator, 2026-09-26): a tool-less, single-exchange
        # TITLE or SUMMARY side call stays a side call when it quotes the
        # user's image (OpenCode's title call carries the user's first
        # message, image included, docs/HARNESS-OPENCODE.md 3b). A title
        # needs no vision: the image becomes its text placeholder and
        # nothing is described, and the call never becomes the
        # conversation's first request (no session, no slot pin, no
        # tool offer, no chain salt).
        sig["image"] = True
        form = None
        if not client_tools and sig["single_exchange"]:
            texts = [_text(m) for m in messages if isinstance(m, dict)
                     and m.get("role") in ("system", "developer", "user")]
            form = ("title" if any(names_a_title(t) for t in texts) else
                    "summarise_conversation"
                    if any(summarises_conversation(t) for t in texts)
                    or _harness_compaction(messages) else None)
        if form:
            sig["form"] = form
            return {"utility": True, "signals": sig,
                    "because": (f"no client tools, one exchange, and the "
                                f"reply is fixed by a contract ({form}): a "
                                f"client's own side call; the image it "
                                f"quotes becomes a placeholder, nothing is "
                                f"described (a {form} needs no vision)")}
        return {"utility": False, "signals": sig,
                "because": "the request carries an image: it goes to the "
                           "vision path, never the bare text model"}
    if client_tools:
        return {"utility": False, "signals": sig,
                "because": "the client sent its own tools: an agent turn"}
    if not sig["single_exchange"]:
        return {"utility": False, "signals": sig,
                "because": "the conversation has answers in it: not a side call"}
    rf = response_format if isinstance(response_format, dict) else {}
    if rf.get("type") in ("json_object", "json_schema"):
        sig["form"] = "response_format:" + rf["type"]
    texts = [_text(m) for m in messages
             if isinstance(m, dict) and m.get("role") in ("system", "developer",
                                                           "user")]
    # A TITLE first (2026-09-26): OpenCode's title generator says it is
    # "summarizing" the user's message and names "this conversation", which
    # read as a compaction (corpus 6580 on: 7 OpenCode title calls recorded
    # utility_kind "compaction"). Hermes' title namer is one too.
    if not sig["form"] and any(names_a_title(t) for t in texts):
        sig["form"] = "title"
    if not sig["form"]:
        for t in texts:
            f = closed_form(t)
            if f:
                sig["form"] = f
                break
    if not sig["form"] and (any(summarises_conversation(t) for t in texts)
                            or _harness_compaction(messages)):
        sig["form"] = "summarise_conversation"
    if not sig["form"]:
        return {"utility": False, "signals": sig,
                "because": ("no tools and one exchange, but the reply is not "
                            "fixed by a contract: a task")}
    return {"utility": True, "signals": sig,
            "because": (f"no client tools, one exchange, and the reply is "
                        f"fixed by a contract ({sig['form']}): a client's own "
                        f"side call, answered by the bare model")}


# WHICH KIND OF SIDE CALL (x_yamadori.utility_kind). The contract form already
# says it; this names it for the proxy, which treats one kind differently:
#
#   compaction   summarise_conversation. Served as part of the conversation
#                it summarises, on that conversation's stored prompt and slot
#                (mcp/compaction.py, proxy._serve_compaction). In the corpus:
#                Hermes' "You are a summarization agent creating a context
#                checkpoint" -- one user message, no system prompt, the turns
#                as text. (A compaction that RESENDS the conversation is not a
#                utility call at all; compaction.in_place finds it.)
#                Pi's and OpenCode's summarisers too (compaction.harness_of).
#   title        a title call: Hermes' title namer, OpenCode's title
#                generator (names_a_title; checked before the other forms).
#   classifier   one_word / one_of / yes_no: Hermes' approval reviewer.
#   structured   json_only or a response_format.
#   other        forced on by X-Yamadori-Features with no contract form.
#
# NOT a compaction: the turn that FOLLOWS one. Hermes opens it with
# "[CONTEXT COMPACTION -- REFERENCE ONLY]" and sends it with its tools and the
# history, so it is an agent turn (proxy._continue_after_compaction keeps its
# session), never a utility call. mcp/test_utility.py checks both on the corpus.
UTILITY_KINDS = {"summarise_conversation": "compaction", "title": "title",
                 "one_word": "classifier", "one_of": "classifier",
                 "yes_no": "classifier", "json_only": "structured"}


def utility_kind(util: dict | None) -> str | None:
    """'compaction' | 'title' | 'classifier' | 'structured' | 'other' for a
    utility call (utility_call's or proxy.utility_of's decision); None
    otherwise. `title`: Hermes' title namer and OpenCode's title generator
    (2026-09-26; the namer was `structured` before, by its JSON form)."""
    if not util or not util.get("utility"):
        return None
    form = (util.get("signals") or {}).get("form") or ""
    if form.startswith("response_format:"):
        return "structured"
    return UTILITY_KINDS.get(form, "other")


# ======================================================= THE DECISION =======

def _tool_list(names: list[str] | None, n: int = 4) -> str:
    names = list(names or [])
    return ", ".join(names[:n]) + (f", +{len(names) - n} more"
                                   if len(names) > n else "")


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ""


def question_of(messages: list[dict]) -> tuple[str, str, bool]:
    """(last user turn, earlier user turns as context, is the user speaking).

    The third value is False when the conversation ends on something other
    than a user turn -- a client's tool result, mid-way through its own agent
    loop. That turn continues a task; it does not ask a new question.

    A harness's SYNTHETIC TOOL-MEDIA TURN (OpenCode's "Attached media from
    tool result:", Pi's "Attached image(s) from tool result:", Cline's
    image-only turn; image_input.TOOL_MEDIA_TURNS) is the tool result it
    carries, never the user speaking: it is looked past, as route.ends_on
    does (2026-09-26).
    """
    import image_input
    import message_text
    # A harness's CONTEXT turn (Codex's <environment_context>) is not the
    # user either (message_text.HARNESS_CONTEXT_TURNS).
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"
             and not image_input.is_tool_media(messages, i)
             and message_text.harness_context(m) is None]
    if not users:
        return "", "", False
    last = users[-1]
    earlier = "\n".join(_text(messages[i]) for i in users[:-1])
    speaking = not any(m.get("role") not in ("user", "system", "developer")
                       or image_input.is_tool_media(messages, j)
                       for j, m in enumerate(messages[last + 1:], last + 1))
    return _text(messages[last]), earlier[-CONTEXT_CHARS:], speaking


def _forced(tier: dict, key: str) -> bool:
    return key in (tier.get("overridden") or [])


# Gate situations that are EVIDENCE the request is about source this server
# holds, as opposed to reasons it merely could not rule the tools out.
# domains.tool_admission's other offers -- no domain evidence, a domain some
# held package serves, an unmapped package in the store -- say nothing about
# whether THIS question has anything to read (mcp/route.py's
# library_question).
READABLE_GATE = {"REPOSITORY_BOUND", "IMPORTS_HELD_SOURCE", "NAMES_HELD_SOURCE",
                 "DEFINES_MENTIONED_SYMBOL"}


def decide(messages: list[dict], tier: dict, *,
           client_tools: list[str] | None = None,
           route: dict | None = None) -> dict:
    """{skills, because, signals} for one request.

    `tier` is what the caller is ALLOWED (tiers.resolve, header applied);
    `route` is mcp/route.py's class for the request (proxy.prepare decides
    it once); `client_tools` names the tools the CLIENT sent with the
    request (never ours). Pure: no network call, no model.
    """
    whole, _ctx, speaking = question_of(messages)
    q, attached = instruction_of(whole)
    because: dict[str, str] = {}
    sig: dict = {"question_chars": len(whole), "user_turn": speaking,
                 "instruction_chars": len(q), "attached_chars": len(attached),
                 "client_tools": len(client_tools or []),
                 "forced": sorted(k for k in ("skills",) if _forced(tier, k)),
                 "allowed": {"skills": bool(tier.get("skills"))},
                 "route": (route or {}).get("class")}
    # skills: allowed or forced. The skills system abstains per skill.
    if _forced(tier, "skills"):
        skills_on = bool(tier.get("skills"))
        because["skills"] = (f"forced {'on' if skills_on else 'off'} by "
                             "X-Yamadori-Features")
    elif tier.get("skills"):
        skills_on = True
        because["skills"] = ("allowed; skill_select injects only what its "
                             "applies-when and tests select")
    else:
        skills_on = False
        because["skills"] = (f"tier {tier.get('name', '?')} does not allow "
                             "skills")
    return {"skills": skills_on, "because": because, "signals": sig}


def select(messages: list[dict], tier: dict, *,
           client_tools: list[str] | None = None,
           route: dict | None = None) -> dict:
    """What the proxy calls: decide, and log one line."""
    d = decide(messages, tier, client_tools=client_tools, route=route)
    print("  " + log_line(d), flush=True)
    return d


def log_line(d: dict) -> str:
    return (f"selection: skills={'yes' if d['skills'] else 'no'} | "
            f"{d['because'].get('skills', '')[:220]}")


if __name__ == "__main__":
    import tiers
    q = " ".join(sys.argv[1:]) or "What is the default value of Object3D.DEFAULT_UP?"
    msgs = [{"role": "user", "content": q}]
    d = decide(msgs, tiers.resolve({"reasoning_effort": "max"}))
    print(json.dumps(d, indent=1, default=str))
