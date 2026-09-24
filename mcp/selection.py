#!/usr/bin/env python
"""The selection engine: per request, WHEN each system runs.

docs/SELECTION-BUILD.md is the spec. This module is step 3 and 4 of its build
plan; the decisions it owns are D2-D4 of that plan's table:

    hints         may recipe hints be attached at all?
    investigate   does deep thinking run before the answer?
    fanout_n      how many ways is the answer written?

D0 (which systems the caller ALLOWS) is the tier, `mcp/tiers.py`. D1 (are the
code tools admissible) is `domains.tool_admission`, whose decision arrives
here as `gate`. This module never widens either: a tier flag means ALLOWED,
not ON, and nothing decided here exceeds what the client asked for. The one
exception is deliberate and loud -- a flag set in `X-Yamadori-Features`
(`tier["overridden"]`) is FORCED on or off, because a benchmark arm must be
able to say "deep thinking on, always" or "off, always" and mean it.

WHY decide() IS PURE, AND WHAT PURE MEANS HERE

`decide()` makes no network call and runs no model. It reads the package
store's symbol tables (read-only sqlite, the same files `tool_admission`
reads) and that is all, so a decision can be replayed over thousands of
logged prompts in seconds -- which is how mcp/test_selection.py checks it
against 342 LiveCodeBench prompts and 26 hand-written three.js questions.
The one input that needs a service, Laya's `/route`, is fetched by
`laya_signal()` and handed in. `select()` is the wrapper the proxy calls: it
fetches that signal only when it could change something, then decides.

DEEP THINKING ON THE PROXY'S PATH: TRIGGERS (Phase 0.6, 2026-09-24)

The proxy passes a route and mcp/deep.py's trigger; deep thinking then runs
exactly when a trigger fired (struggle, task kickoff, known-hard area; the
model's own think_deeply call runs during generation), on ANY route class.
The library_question-only gate is gone, and the two-signal rule below is not
consulted there: it is the LEGACY path, kept byte for byte for the offline
evaluators that replay it (no route, no trigger). A header still forces deep
thinking on or off on both paths.

DEEP THINKING, LEGACY PATH: TWO SIGNALS, AND DISAGREEMENT ESCALATES

docs/SELECTION.md pattern 5, the shipped recommendation. The rule signal is
the six-line regex (`rule_baseline`, moved here from bench/laya_calibration.py
so there is ONE copy) plus the symbol lookup. The second signal is the trained
Laya route_in head, served at `/route` with `engine: "trained"`.

    ORIGINAL head (n=89 labels, docs/LAYA.md F11): regex 0.841 +/-0.023 vs
    trained 0.726 +/-0.045 on the same splits.

    RETRAINED 2026-09-22 with 200 package-domain labels, scored on the 120
    HELD-OUT package labels it never saw (bench/eval_route_heldout.py),
    investigate-vs-not: new head 80/120, old head 64/120 (McNemar p=0.011),
    regex 73/120, this module's rule + symbol lookup 89/120. Hard slice
    (n=28): both heads 11, regex and selection 18. The head still does not
    beat the rule, so it never decides alone.

    agree            -> that answer
    disagree         -> investigate (the misfire costs one search loop; the
                        other misfire costs a wrong answer with no receipts)
    head abstains    -> undecided, which is not agreement -> investigate
    Laya down        -> signal recorded as None, the rule decides alone.
                        Never a guess: a missing second opinion is reported
                        as missing, not replaced by a made-up one.

THE HARD-SLICE CHECK: THE SYMBOL LOOKUP

The regex reads surface cues ("our", "src/", a `.ts` path). The questions it
cannot read are the ones about a LIBRARY'S source phrased as general
questions -- "what is the default value of `Object3D.DEFAULT_UP`" has no
cue at all and the regex says `answer_directly` on all 26 of
bench/context_economy_tasks.jsonl. Whether a held package DEFINES a name the
question uses is not a reading of the prose; it is a lookup in the index that
would answer it. So an `answer_directly` from the regex is upgraded to
`investigate` when a held package (or the bound repository) defines a symbol
the question names. `clarify` is not upgraded: an underspecified question
does not become specified because it contains a class name.

This probe is WIDER than `domains._symbols`. Code-shaped names (backticks,
an internal capital, a digit, an underscore) are asked of every symbol
table; a capitalised / ALLCAPS word in identifier syntax (`THREE.MOUSE`,
`Loop()`) only of a package the text names. English-shaped ones -- a
capitalised word used as a code noun ("the renderer's Pipelines module") --
count only where a library defines them in
its own source, not its examples or tests: three.js's examples define `For`,
`Number` and `String`, and a LiveCodeBench puzzle's "a Zero Array" and "Group
A" matched typegpu's `Array` and three's `Group` until the code-noun rule.
A BARE ALLCAPS WORD IS NOT PROBED (2026-09-23): it was, and "MOUSE" and "API"
in a pasted game spec matched three's `MOUSE` and wgpu-matrix's `API` and sent
a request to create files locally into minutes of deep thinking. Names the
platform defines (`Event`, `Array`, `Math`) never count
(domains.PLATFORM_NAMES). mcp/test_selection.py asserts 0/342 LiveCodeBench
prompts fire and 25/26 context-economy questions do with the rule alone:
ce05 names no symbol, and fired before only because the ALLCAPS "TSL"
matched a `TSL` namespace declaration. Held-out labels, investigate-vs-not,
Laya absent: 88/120 before this change, 91/120 after (+4 -1, McNemar
p=0.375, not significant).

AN AGENT HARNESS ACTING LOCALLY, AND ATTACHMENTS

When the client sent its own tools and the instruction asks to act on the
user's machine ("start this project in ~/Developer/x"), neither deep thinking
nor fan-out runs (`acts_locally`). What escalates reads the user's
instruction, not an attachment the harness inlined after it
(`instruction_of`). Both are explained where they are defined.

NOTHING TO READ, NOTHING TO THINK ABOUT

Deep thinking runs only if there is source that bears on THIS question: a
bound repository, a held package the request imports or names, or a held
definition of a name it uses (`READABLE_GATE`, the symbol lookup). The tool
gate also offers for reasons that say nothing about the question -- no
domain evidence, a domain a held package serves, an unmapped package in the
store -- and on 2026-09-22 those opened it for all 342 LiveCodeBench prompts
(koota, mapped to `algorithms`, and four unmapped packages were indexed).
The regex reads "we" in a puzzle as "our code", so 65 of them would have
investigated. Laya is not asked either: a second opinion on whether to read
nothing is not a decision.

FAN-OUT: THE DOCUMENTED RULE, NOT YET CALIBRATED

`mcp/fanout.py`'s docstring states the policy: lookups N=1, design / approach
/ refactor / "how should I" N>1. It was written in prose and implemented
nowhere (PROTOCOL rule 14). It is implemented here as a word rule at n=0
labels -- step 7 of the build plan labels ~40 prompts and replaces it. The
only measurement behind it is the null it cites: 7/8 vs 7/8, zero discordant
pairs, at 3.2x wall clock, on file-location questions (n=8).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")
# A routing call is one forward pass of a 421M encoder. Ten seconds is ~50x the
# measured cost; past it Laya is treated as down for THIS request and the rule
# decides alone. Not a retry loop: one call, then a recorded None.
LAYA_ROUTE_TIMEOUT = float(os.environ.get("YAMADORI_LAYA_ROUTE_TIMEOUT", "10"))

# The shortest question deep thinking will take. shomen's own argument gate
# for `delegate_investigation` (proxy.run_our_tool) uses the same number.
MIN_QUESTION_CHARS = 8

# How much earlier user text rides along as `context`. The labels' context
# field is the user's own background ("we're on three 0.185.0, the component
# is src/scene/..."), which in a live conversation is their earlier turns.
CONTEXT_CHARS = 1500


# ====================================================== THE REGEX (moved) ===
#
# Moved verbatim from bench/laya_calibration.py on 2026-09-22 so that the
# benchmark, the trainer (scripts/train_laya.py scores it on every split) and
# the proxy run ONE copy. bench/data/rule_baseline_golden.json holds its 89
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


# Without this, "Laya gets 62%" is unreadable: the majority class alone gets
# 39%, and six lines of regex may get more than the model does. A router is
# only worth a GPU if it beats the thing you would have written anyway.
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
# thinking then ran for minutes on a request to create files locally. ALLCAPS
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


# ================================================= THE FAN-OUT RULE =========

# fanout.py's docstring rule, as words. n=0 labels; build step 7 replaces it.
_DESIGN = re.compile(
    r"\b(how should (i|we)|design(ing)?|architect(ure|ing)?|approach(es)?|"
    r"refactor\w*|trade-?offs?|pros and cons|(best|cleanest|right) way)\b",
    re.I)

# A request to WRITE code. Operator decision 2026-09-23: at tiers that allow
# fan-out (high, max), code-writing tasks fan out too -- the original and the
# second brain's candidates are checked (they must parse) and the winner is
# DELIVERED (fanout.run, sequential since 2026-09-23; proxy._fan_out). The
# old rule fanned out only on design
# questions, from a null result on file LOOKUPS (7/8 vs 7/8, n=8); it was
# never measured on code generation. A code fence in the request, or a
# write/implement/fix verb followed by a code noun, counts.
_CODE_TASK = re.compile(
    r"```|\b(write|implement|complete|finish|fix|create|generate|build|"
    r"rewrite|port|refactor)\b[^.?!\n]{0,80}?\b(function|method|class|"
    r"component|hook|type|interface|struct|enum|trait|impl|program|script|"
    r"module|solution|code|tests?|query|algorithm|shader|kernel)s?\b",
    re.I)


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
# the design and code-task words, the act-locally rule, and the question Laya
# is asked -- read the user's instruction, not the attachment. An attachment
# is material to act on; the instruction says what to do with it. The
# attachment above is a game spec for the user's OWN project: its ALLCAPS
# headings matched held definitions, and the ``` Hermes wraps every
# attachment in read as "a code-writing task" to _CODE_TASK. Evidence that an
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
# terminal), which only the client can run. Deep thinking reads library
# source for minutes before the first token, and fan-out re-writes a final
# answer; neither can create a file on the user's disk, and both hold the
# turn while the harness waits. Live, 2026-09-23: "I want to start this
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
# each got the capability block, our tools and hints. 9cca1a took 409 s to
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
# block, no tools, no hints, no deep thinking, no fan-out, no repair, and no
# session (it neither reads nor writes the conversation's offered-tools flag,
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
        # see. It is a task for the vision path (describe_image).
        sig["image"] = True
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
    if not sig["form"]:
        for t in texts:
            f = closed_form(t)
            if f:
                sig["form"] = f
                break
    if not sig["form"] and any(summarises_conversation(t) for t in texts):
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
#   classifier   one_word / one_of / yes_no: Hermes' approval reviewer.
#   structured   json_only or a response_format: Hermes' title namer.
#   other        forced on by X-Yamadori-Features with no contract form.
#
# NOT a compaction: the turn that FOLLOWS one. Hermes opens it with
# "[CONTEXT COMPACTION -- REFERENCE ONLY]" and sends it with its tools and the
# history, so it is an agent turn (proxy._continue_after_compaction keeps its
# session), never a utility call. mcp/test_utility.py checks both on the corpus.
UTILITY_KINDS = {"summarise_conversation": "compaction",
                 "one_word": "classifier", "one_of": "classifier",
                 "yes_no": "classifier", "json_only": "structured"}


def utility_kind(util: dict | None) -> str | None:
    """'compaction' | 'classifier' | 'structured' | 'other' for a utility
    call (utility_call's or proxy.utility_of's decision); None otherwise."""
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
    """
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not users:
        return "", "", False
    last = users[-1]
    earlier = "\n".join(_text(messages[i]) for i in users[:-1])
    speaking = not any(m.get("role") not in ("user", "system")
                       for m in messages[last + 1:])
    return _text(messages[last]), earlier[-CONTEXT_CHARS:], speaking


def _forced(tier: dict, key: str) -> bool:
    return key in (tier.get("overridden") or [])


def laya_says_investigate(laya: dict | None) -> bool | None:
    """The head's answer on the binary the rule is compared on.

    True / False when it decided; None when it abstained (undecided, which
    is NOT "no" -- docs/SELECTION.md anti-pattern D) or gave nothing usable.
    """
    if not laya or laya.get("abstain"):
        return None
    choice = laya.get("choice")
    if choice not in ("investigate", "answer_directly", "clarify"):
        return None
    return choice == "investigate"


# Gate situations that are EVIDENCE the request is about source this server
# holds, as opposed to reasons it merely could not rule the tools out.
# domains.tool_admission's other offers -- no domain evidence, a domain some
# held package serves, an unmapped package in the store -- say nothing about
# whether THIS question has anything to read.
READABLE_GATE = {"REPOSITORY_BOUND", "IMPORTS_HELD_SOURCE", "NAMES_HELD_SOURCE",
                 "DEFINES_MENTIONED_SYMBOL"}


def decide(messages: list[dict], tier: dict, gate: dict | None,
           state: dict | None = None, *, laya: dict | None = None,
           laya_status: str = "not consulted",
           dbs: dict[str, str] | None = None,
           client_tools: list[str] | None = None,
           route: dict | None = None,
           trigger: dict | None = None,
           second: str = "Laya") -> dict:
    """{hints, investigate, fanout_n, because, signals} for one request.

    `route` is mcp/route.py's class for the request (proxy.prepare decides
    it once). When given, fan-out READS it instead of re-deciding: it runs on
    code_generation / code_edit only; a header that forces it still forces
    it. None (the default, and every caller before 2026-09-24) keeps the
    word rules.

    `trigger` is mcp/deep.py's decision for the request (Phase 0.6). With a
    route or a trigger, deep thinking runs exactly when a trigger FIRED --
    on any route class; the library_question-only gate is gone (operator,
    2026-09-24) -- and its reason is the trigger's. With neither, the legacy
    rule + symbol lookup + Laya path decides, for the offline evaluators.

    `tier` is what the caller is ALLOWED (tiers.resolve, header applied);
    `gate` is domains.tool_admission's decision, or None when the tier has no
    retrieval; `state` is the session's nebari row. `laya` is the /route
    response or None. `dbs` names the symbol tables to look in; None means
    the held package store (plus nothing -- the proxy passes the bound
    repository's table in explicitly). `client_tools` names the tools the
    CLIENT sent with the request (never ours); a non-empty list means an
    agent harness is driving its own loop. `second` names the second
    signal in the reasons: "Laya" (the default, byte for byte what the
    offline evaluators replay) or "E1" (mcp/e1.py, YAMADORI_E1=1); `laya`
    carries either one's answer.
    """
    st = state or {}
    whole, ctx, speaking = question_of(messages)
    # What escalates reads the instruction; see instruction_of.
    q, attached = instruction_of(whole)
    local = acts_locally(q) if client_tools else None
    because: dict[str, str] = {}
    sig: dict = {"question_chars": len(whole), "user_turn": speaking,
                 "instruction_chars": len(q), "attached_chars": len(attached),
                 "client_tools": len(client_tools or []),
                 "acts_locally": local,
                 "forced": sorted(k for k in ("hints", "investigate", "fanout")
                                  if _forced(tier, k)),
                 "allowed": {"hints": bool(tier.get("hints")),
                             "investigate": bool(tier.get("investigate")),
                             "fanout": int(tier.get("fanout") or 1)},
                 "gate": (gate or {}).get("situation"),
                 "route": (route or {}).get("class"),
                 "laya": laya, "laya_status": laya_status}
    rclass = (route or {}).get("class")

    # ---- hints: allowed or forced. hints.select abstains per hint. ----------
    if _forced(tier, "hints"):
        hints = bool(tier.get("hints"))
        because["hints"] = f"forced {'on' if hints else 'off'} by X-Yamadori-Features"
    elif tier.get("hints"):
        hints = True
        because["hints"] = ("allowed; hints.select attaches only what clears the "
                            "0.55 floor inside the task's domains")
    else:
        hints = False
        because["hints"] = f"tier {tier.get('name', '?')} does not allow hints"

    # ---- deep thinking ------------------------------------------------------
    investigate = False
    rule = None
    held: dict = {}
    trig = trigger or {}
    fired = bool(trig.get("fire")) and trig.get("kind") not in (None, "forced")
    sig["trigger"] = trig.get("kind") if trig.get("fire") else None
    if len(q.strip()) < MIN_QUESTION_CHARS and not fired:
        why = "no question of at least 8 characters to think about"
    elif _forced(tier, "investigate"):
        investigate = bool(tier.get("investigate"))
        why = f"forced {'on' if investigate else 'off'} by X-Yamadori-Features"
    elif not tier.get("investigate"):
        why = f"tier {tier.get('name', '?')} does not allow deep thinking"
    elif route is not None or trigger is not None:
        # PHASE 0.6 (operator, 2026-09-24; docs/SELF-IMPROVEMENT-PLAN.md):
        # deep thinking is no longer limited to library questions. Four
        # TRIGGERS decide it (mcp/deep.py) -- struggle, a task kickoff and a
        # known-hard area here, before main generates; the model's own
        # think_deeply call during generation -- on every route class, the
        # agent path included. The rule, the symbol lookup and Laya are not
        # consulted on this path (a separate evaluation decides Laya vs
        # Tev1); a header still forces it either way (above). The reason is
        # the trigger's, recorded as before.
        investigate = fired
        why = (trig.get("because") or
               "no trigger was evaluated for this request (mcp/deep.py "
               "decides); the model may call think_deeply")
    elif local:
        why = (f"the client sent its own tools ({_tool_list(client_tools)}) and "
               f"the request asks to act on the user's machine ({local!r}): "
               f"that is the client's agent loop to run, and deep thinking "
               f"cannot create or change a local file")
    elif not (gate or {}).get("offer"):
        why = ("the code tools are withheld ("
               + ((gate or {}).get("situation") or "retrieval off")
               + "): nothing indexed that deep thinking could read")
    elif not speaking:
        why = "the conversation ends on a tool result: a task in progress, not a new question"
    else:
        # THE LEGACY PATH: no route and no trigger -- the offline evaluators
        # (bench/eval_route_heldout.py, bench/laya_factcheck/) and every
        # caller before 2026-09-24. The regex + symbol lookup + Laya
        # decision, unchanged, so those measurements still replay.
        rule = rule_baseline({"question": q, "context": ctx})
        if dbs is None:
            dbs = symbol_dbs(None)
        held = defined_symbols(q, dbs)
        situation = (gate or {}).get("situation")
        imported = sorted(set(st.get("packages") or [])
                          & {k for k in dbs if not k.startswith("(")})
        readable = bool(held) or situation in READABLE_GATE or bool(imported)
        sig["readable"] = readable
        if not readable:
            # NOTHING TO READ. The tools were offered for a reason that is not
            # about this question -- no domain evidence, a domain some held
            # package serves, an unmapped package in the store -- and the
            # question names nothing a held package defines. The regex's
            # "investigate" means "this is about OUR code", and with no
            # repository bound there is none to read. Measured necessary on
            # 2026-09-22: indexing koota (mapped to `algorithms`) and four
            # unmapped packages opened the tool gate for all 342 LiveCodeBench
            # prompts, and 65 of them say "we" or "us", which the regex reads
            # as investigate.
            why = (f"regex={rule}, but nothing to read: no repository is "
                   f"bound, the tools were offered only because {situation} "
                   f"and the question names nothing a held source defines")
        else:
            rule_says = rule == "investigate" or (rule == "answer_directly"
                                                  and bool(held))
            rule_why = (f"regex={rule}"
                        + (" + a held source defines "
                           + "; ".join(f"{k}: {', '.join(v[:3])}"
                                       for k, v in held.items())
                           if held else ""))
            sig["consult_laya"] = True
            sig["second_signal"] = second
            other = laya_says_investigate(laya)
            if laya is None:
                investigate = rule_says
                why = (f"{rule_why}; {second}: {laya_status}, so the rule "
                       f"decides alone")
            elif other is None:
                investigate = True
                why = (f"{rule_why}; {second} abstained (margin "
                       f"{laya.get('margin')}) -- undecided is not agreement, "
                       f"so investigate")
            elif other == rule_says:
                investigate = rule_says
                why = f"{rule_why}; {second} agrees ({laya.get('choice')})"
            else:
                investigate = True
                why = (f"{rule_why}; {second} says {laya.get('choice')} -- "
                       f"the signals disagree, which escalates to investigate")
    because["investigate"] = why
    sig["rule"] = rule
    sig["held_symbols"] = {k: v[:8] for k, v in held.items()}

    # ---- fan-out ------------------------------------------------------------
    allowed_n = max(1, int(tier.get("fanout") or 1))
    design = bool(_DESIGN.search(q))
    code_task = bool(_CODE_TASK.search(q))
    sig["design"] = design
    sig["code_task"] = code_task
    if _forced(tier, "fanout"):
        fanout_n = allowed_n
        because["fanout"] = f"forced to {fanout_n} by X-Yamadori-Features"
    elif allowed_n <= 1:
        fanout_n = 1
        because["fanout"] = f"tier {tier.get('name', '?')} allows one answer"
    elif not speaking:
        fanout_n = 1
        because["fanout"] = "a task in progress, not a new question"
    elif local:
        fanout_n = 1
        because["fanout"] = (f"the client sent its own tools and the request "
                             f"asks to act on the user's machine ({local!r}): "
                             f"one answer, the client's loop does the work")
    elif route is not None:
        # The route decides, not the words (operator, 2026-09-24): the code
        # classes fan out -- candidates are graded by the code check -- and
        # nothing else does, design questions included.
        if rclass in ("code_generation", "code_edit"):
            fanout_n = allowed_n
            because["fanout"] = (f"route {rclass}: candidates are checked and "
                                 f"the best-supported parsing code is "
                                 f"delivered")
        else:
            fanout_n = 1
            because["fanout"] = (f"route {rclass}: fan-out runs on "
                                 f"code_generation / code_edit only")
    elif design:
        fanout_n = allowed_n
        because["fanout"] = ("a design/approach question: room for different "
                             "answers (word rule, n=0 labels -- build step 7)")
    elif code_task:
        fanout_n = allowed_n
        because["fanout"] = ("a code-writing task: candidates are checked and "
                             "the best-supported parsing code is selected and "
                             "delivered (operator decision 2026-09-23)")
    else:
        fanout_n = 1
        because["fanout"] = ("not a design/approach question: a single right "
                             "answer leaves no variance for variants to use "
                             "(fanout.py: 7/8 vs 7/8, 0 discordant, n=8)")

    _ = st  # the session row is part of the contract; nothing reads it yet
    return {"hints": hints, "investigate": investigate, "fanout_n": fanout_n,
            "because": because, "signals": sig}


# ====================================================== THE SERVICE CALL ===

def laya_signal(question: str, context: str = "", url: str | None = None,
                timeout: float | None = None) -> tuple[dict | None, str]:
    """(the trained head's /route answer, status). (None, why) when it did not
    answer -- down, slow, no trained head, or a malformed reply. Never a guess.
    With YAMADORI_E1=1 Laya is off the request path: nothing is sent.
    """
    import e1
    if not e1.laya_allowed():
        return None, "not consulted: YAMADORI_E1=1 (E1 is the second signal)"
    body = json.dumps({"task": "route_in", "engine": "trained",
                       "question": question, "context": context}).encode()
    req = urllib.request.Request((url or LAYA_URL) + "/route", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(
                req, timeout=LAYA_ROUTE_TIMEOUT if timeout is None else timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:                                       # noqa: BLE001
        return None, f"down ({type(e).__name__}: {str(e)[:120]})"
    if not isinstance(d, dict) or d.get("engine") != "trained" or "choice" not in d:
        return None, "answered without a trained route_in decision: " + json.dumps(d)[:160]
    keep = {k: d.get(k) for k in ("choice", "probabilities", "margin",
                                  "abstain", "gate", "engine",
                                  "artefact_version", "elapsed_ms")}
    return keep, "answered"


def select(messages: list[dict], tier: dict, gate: dict | None,
           state: dict | None = None, *, root_db: str | None = None,
           laya_url: str | None = None,
           client_tools: list[str] | None = None,
           route: dict | None = None,
           trigger: dict | None = None) -> dict:
    """What the proxy calls: decide, fetch the second signal only if the
    decision reached the two-signal stage, decide again with it, and log one
    line.

    Laya is consulted whenever deep thinking is genuinely on the table --
    allowed, not forced, tools offered, something to read -- even when the
    rule already says investigate. The outcome cannot change then, but the
    log carries both signals, and that pair is the data pattern 5 needs to be
    judged on. Deciding twice costs a few symbol-table reads, not a model.
    """
    try:
        dbs = symbol_dbs(root_db)
    except Exception:                                            # noqa: BLE001
        dbs = {}
    d = decide(messages, tier, gate, state, dbs=dbs,
               laya_status="not consulted", client_tools=client_tools,
               route=route, trigger=trigger)
    if d["signals"].get("consult_laya"):
        q, ctx, _ = question_of(messages)
        # The instruction, not the attachment: Laya's window is ~512 tokens
        # and it silently drops the tail (AGENTS.md), so an 8.8 KB paste is
        # judged on whatever of it fits.
        q, _attached = instruction_of(q)
        import e1
        if e1.enabled():
            # E1 (mcp/e1.py) in Laya's place: its route_in head on one
            # embedding of the same instruction and context. None when the
            # head is not servable or the embedder is down -- the rule then
            # decides alone, as it does when Laya is down.
            laya, status = e1.route_signal(q, ctx)
            second = "E1"
        else:
            laya, status = laya_signal(q, ctx, url=laya_url)
            second = "Laya"
        d = decide(messages, tier, gate, state, laya=laya,
                   laya_status=status, dbs=dbs, client_tools=client_tools,
                   route=route, trigger=trigger, second=second)
    else:
        d["signals"]["laya_status"] = ("not consulted: "
                                       + d["because"]["investigate"][:120])
    print("  " + log_line(d), flush=True)
    return d


def log_line(d: dict) -> str:
    s = d["signals"]
    laya = s.get("laya")
    lay = (f"{laya.get('choice')}{'?' if laya.get('abstain') else ''}"
           f"({laya.get('margin')})" if laya else f"None [{s.get('laya_status')}]")
    name = "e1" if s.get("second_signal") == "E1" else "laya"
    held = ";".join(f"{k}:{','.join(v[:3])}"
                    for k, v in (s.get("held_symbols") or {}).items()) or "-"
    return (f"selection: investigate={'yes' if d['investigate'] else 'no'} "
            f"fanout={d['fanout_n']} hints={'yes' if d['hints'] else 'no'} | "
            f"rule={s.get('rule')} held={held} {name}={lay} | "
            f"{d['because'].get('investigate', '')[:220]}")


if __name__ == "__main__":
    import tiers
    q = " ".join(sys.argv[1:]) or "What is the default value of Object3D.DEFAULT_UP?"
    msgs = [{"role": "user", "content": q}]
    import domains
    g = domains.tool_admission(msgs, None)
    d = decide(msgs, tiers.resolve({"reasoning_effort": "max"}), g,
               laya_status="not consulted: self-test")
    print(json.dumps(d, indent=1, default=str))
