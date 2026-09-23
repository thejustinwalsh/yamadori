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

DEEP THINKING: TWO SIGNALS, AND DISAGREEMENT ESCALATES

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
an internal capital, a digit, an underscore) are asked of every symbol table.
English-shaped ones -- a capitalised word used as a code noun ("the
renderer's Pipelines module"), an ALLCAPS word (`REVISION`, `TSL`) -- count
only where a library defines them in its own source, not its examples or
tests: three.js's examples define `For`, `Number` and `String`, and a
LiveCodeBench puzzle's "a Zero Array" and "Group A" matched typegpu's `Array`
and three's `Group` until the code-noun rule. 26/26 context-economy
questions fire and 0/342 LiveCodeBench prompts do, with the tool gate open.

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
_ALLCAPS = re.compile(r"\b([A-Z][A-Z0-9_]{2,40})\b")
MAX_PROBES = 400


def split_probe_tokens(text: str) -> tuple[list[str], list[str]]:
    """(code-shaped names, English-shaped names) in `text`.

    CODE-SHAPED: anything in backticks, and identifiers with an internal
    capital (`PMREMGenerator`, `WebGPUBackend`), a digit or an underscore --
    English words have none of these. ENGLISH-SHAPED: a capitalised word
    mid-sentence (`Pipelines`, `Animation`) or an ALLCAPS word (`REVISION`,
    `TSL`). Those are also how prose capitalises, so they count only against a
    library's own source, never its examples or tests -- see defined_symbols.
    """
    code: dict[str, None] = {}
    words: dict[str, None] = {}
    plain: dict[str, None] = {}

    def add(bag: dict, tok: str) -> None:
        tok = tok.strip("$")
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
    for tok in _MID_CAPITAL.findall(text):
        add(words, tok)
    for tok in _ALLCAPS.findall(text):
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
    """Which of `names` (held package names) the text names or imports."""
    import domains
    low = text.lower()
    out = set()
    for n in names:
        alias = domains.HELD_ALIASES.get(n)
        if (alias and re.search(alias, text, re.I)) or n.lower() in low:
            out.add(n)
    return out


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


# ======================================================= THE DECISION =======

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
           dbs: dict[str, str] | None = None) -> dict:
    """{hints, investigate, fanout_n, because, signals} for one request.

    `tier` is what the caller is ALLOWED (tiers.resolve, header applied);
    `gate` is domains.tool_admission's decision, or None when the tier has no
    retrieval; `state` is the session's nebari row. `laya` is the /route
    response or None. `dbs` names the symbol tables to look in; None means
    the held package store (plus nothing -- the proxy passes the bound
    repository's table in explicitly).
    """
    st = state or {}
    q, ctx, speaking = question_of(messages)
    because: dict[str, str] = {}
    sig: dict = {"question_chars": len(q), "user_turn": speaking,
                 "forced": sorted(k for k in ("hints", "investigate", "fanout")
                                  if _forced(tier, k)),
                 "allowed": {"hints": bool(tier.get("hints")),
                             "investigate": bool(tier.get("investigate")),
                             "fanout": int(tier.get("fanout") or 1)},
                 "gate": (gate or {}).get("situation"),
                 "laya": laya, "laya_status": laya_status}

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
    if len(q.strip()) < MIN_QUESTION_CHARS:
        why = "no question of at least 8 characters to think about"
    elif _forced(tier, "investigate"):
        investigate = bool(tier.get("investigate"))
        why = f"forced {'on' if investigate else 'off'} by X-Yamadori-Features"
    elif not tier.get("investigate"):
        why = f"tier {tier.get('name', '?')} does not allow deep thinking"
    elif not (gate or {}).get("offer"):
        why = ("the code tools are withheld ("
               + ((gate or {}).get("situation") or "retrieval off")
               + "): nothing indexed that deep thinking could read")
    elif not speaking:
        why = "the conversation ends on a tool result: a task in progress, not a new question"
    else:
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
            other = laya_says_investigate(laya)
            if laya is None:
                investigate = rule_says
                why = f"{rule_why}; Laya: {laya_status}, so the rule decides alone"
            elif other is None:
                investigate = True
                why = (f"{rule_why}; Laya abstained (margin "
                       f"{laya.get('margin')}) -- undecided is not agreement, "
                       f"so investigate")
            elif other == rule_says:
                investigate = rule_says
                why = f"{rule_why}; Laya agrees ({laya.get('choice')})"
            else:
                investigate = True
                why = (f"{rule_why}; Laya says {laya.get('choice')} -- the "
                       f"signals disagree, which escalates to investigate")
    because["investigate"] = why
    sig["rule"] = rule
    sig["held_symbols"] = {k: v[:8] for k, v in held.items()}

    # ---- fan-out ------------------------------------------------------------
    allowed_n = max(1, int(tier.get("fanout") or 1))
    design = bool(_DESIGN.search(q))
    sig["design"] = design
    if _forced(tier, "fanout"):
        fanout_n = allowed_n
        because["fanout"] = f"forced to {fanout_n} by X-Yamadori-Features"
    elif allowed_n <= 1:
        fanout_n = 1
        because["fanout"] = f"tier {tier.get('name', '?')} allows one answer"
    elif not speaking:
        fanout_n = 1
        because["fanout"] = "a task in progress, not a new question"
    elif design:
        fanout_n = allowed_n
        because["fanout"] = ("a design/approach question: room for different "
                             "answers (word rule, n=0 labels -- build step 7)")
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
    """
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
           laya_url: str | None = None) -> dict:
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
               laya_status="not consulted")
    if d["signals"].get("consult_laya"):
        q, ctx, _ = question_of(messages)
        laya, status = laya_signal(q, ctx, url=laya_url)
        d = decide(messages, tier, gate, state, laya=laya,
                   laya_status=status, dbs=dbs)
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
    held = ";".join(f"{k}:{','.join(v[:3])}"
                    for k, v in (s.get("held_symbols") or {}).items()) or "-"
    return (f"selection: investigate={'yes' if d['investigate'] else 'no'} "
            f"fanout={d['fanout_n']} hints={'yes' if d['hints'] else 'no'} | "
            f"rule={s.get('rule')} held={held} laya={lay} | "
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
