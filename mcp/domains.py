#!/usr/bin/env python
"""Which recipes are even eligible for this task.

THE PROBLEM

The corpus spans memory layout in Zig, WGSL workgroup sizing, TypeScript
variance and -- now -- colour and type scale. A ranker over all of it will
sometimes surface type-scale advice while someone writes a database
migration, because "hierarchy" and "spacing" embed near plenty of things.

Relevance ranking cannot fix that. A cosine score says how similar two texts
are, not whether a rule is ADMISSIBLE, and the failure is not that the design
recipe ranked third instead of thirtieth -- it is that it was ever a
candidate.

So domain is a filter applied BEFORE ranking, and it is hard: a recipe whose
domains do not intersect the task's is not ranked low, it is absent.

HOW THE TASK'S DOMAIN IS DECIDED

From what the caller actually sent, in the order the evidence is trustworthy:

  imports     `discover.py` parses them with tree-sitter. `react` means
              frontend; `tokio` means backend. This is the strongest signal
              because it is a fact about the code rather than a reading of
              the prose.
  extensions  .tsx and .css say something .rs does not.
  words       weakest, and used only to ADD domains, never to remove one.
              "make the button feel nicer" carries no imports at all.

Nothing is inferred from the absence of a signal. A task with no evidence
gets the permissive set, because a stack that refuses to answer until it has
classified you is worse than one that occasionally offers an irrelevant hint.

FAIL OPEN OR FAIL CLOSED

Recipes without domain tags fail OPEN -- eligible everywhere -- because six of
the eight corpora predate this field and silently dropping 600 recipes would
be a worse bug than the one this prevents. The exception is the visual design
corpus, which fails CLOSED: it is tagged at collection time, and the whole
reason it exists is that it must not leak into backend work.

THE SECOND CONSUMER: WHICH REQUESTS ARE OFFERED THE CODE TOOLS

`tool_admission()` at the bottom of this file. Same argument, different
object: a tool definition that cannot return anything is not a low-ranked
tool, it is an inadmissible one. See its docstring for the decision rule and
the measurement behind it. The proxy calls it from `prepare()`; nothing else
decides whether the capability block and our tool list are attached.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The fixed vocabulary. Kept small on purpose: a taxonomy nobody can hold in
# their head is one that gets tagged inconsistently.
DOMAINS = {
    "visual-design", "ui-component", "motion", "web-frontend",
    "native-mobile", "accessibility", "brand",
    "systems", "gpu", "types", "algorithms", "backend", "tooling",
}

# Corpora that must never leak outside their domains, even untagged.
STRICT_FILES = {"design_visual"}

# Package -> domains. Only packages whose presence genuinely implies a domain;
# a guess here is worse than no entry, because it silently filters recipes.
PACKAGE_DOMAINS = {
    "react": {"web-frontend", "ui-component"},
    "react-dom": {"web-frontend", "ui-component"},
    "next": {"web-frontend"},
    "@react-three/fiber": {"web-frontend", "gpu"},
    "@react-three/drei": {"web-frontend", "gpu"},
    "three": {"gpu", "web-frontend"},
    "typegpu": {"gpu"},
    "@webgpu/types": {"gpu"},
    # Indexed 2026-09-22 (mcp/deps.py index). Each one left unmapped reopened
    # the tool gate for every task -- HELD_SOURCE_UNMAPPED offers -- so all
    # 342 LiveCodeBench prompts were offered ~2,700 tokens of tools again.
    "@types/three": {"gpu", "web-frontend", "types"},
    "@typegpu/noise": {"gpu"},
    "@typegpu/three": {"gpu", "web-frontend"},
    "@react-three/postprocessing": {"gpu", "web-frontend"},
    "postprocessing": {"gpu", "web-frontend"},
    "three-mesh-bvh": {"gpu", "web-frontend"},
    "wgpu-matrix": {"gpu"},
    "type-fest": {"types"},
    "framer-motion": {"motion", "web-frontend", "ui-component"},
    "tailwindcss": {"visual-design", "web-frontend"},
    "styled-components": {"visual-design", "web-frontend"},
    # An ECS state library for games and React UIs -- NOT "algorithms". That
    # entry was harmless while koota was unindexed; once held (2026-09-22) it
    # made every competitive-programming prompt "match a held source" whose
    # code cannot answer an algorithm puzzle.
    "koota": {"web-frontend"},
    "tokio": {"backend", "systems"},
    "axum": {"backend"},
    "actix-web": {"backend"},
    "sqlx": {"backend"},
    "diesel": {"backend"},
    "serde": {"backend", "systems"},
    "numpy": {"algorithms"},
    "pandas": {"algorithms"},
}

EXTENSION_DOMAINS = {
    ".tsx": {"web-frontend", "ui-component"}, ".jsx": {"web-frontend", "ui-component"},
    ".css": {"visual-design", "web-frontend"}, ".scss": {"visual-design", "web-frontend"},
    ".html": {"web-frontend"},
    ".rs": {"systems"}, ".c": {"systems"}, ".h": {"systems"},
    ".cpp": {"systems"}, ".hpp": {"systems"}, ".zig": {"systems"},
    ".wgsl": {"gpu"}, ".glsl": {"gpu"}, ".hlsl": {"gpu"},
    ".swift": {"native-mobile"}, ".kt": {"native-mobile"},
    ".sql": {"backend"},
}

# Weak, additive only. These never remove a domain, because a backend task
# that happens to say "layout" is still a backend task.
#
# STEMS NEED `\w*`. `animat`, `typograph` and `accessib` were written as stems
# and wrapped in `\b...\b`, so the closing boundary demanded the word END
# there: "animation", "typography" and "accessibility" never matched, and the
# only strings that could were the stems themselves. Found while wiring the
# tool gate; asserted in mcp/test_domains.py.
#
# `gpu` HAD NO WORD PATTERN AT ALL, and it is the domain of both packages this
# server holds source for (three, typegpu). A prose question about shaders
# with no import line therefore carried no GPU evidence, and the tool gate --
# which offers tools when a request's domains meet a held package's -- would
# have had nothing to meet. Every term below is one that `bench/
# context_economy_tasks.jsonl` (26 hand-written three.js questions) or a TSL
# user actually writes; `tsl` is three.js's shading language.
#
# `algorithms` gained the phrases a competitive-programming prompt is made of.
# Before this the 342 LiveCodeBench prompts in bench/data/test5+test6 were
# classified almost entirely by "input" matching `ui-component` (278/342
# carried nothing else) -- the right gate outcome for the wrong reason, and
# one that a later fix to "input" would have silently reversed.
#
# A WORD COUNTS ONLY IN ITS DOMAIN SENSE (2026-09-23). The Rust C-ABI task
# bench/domain/tasks/rs12 ("a point parser ... after trimming ASCII
# whitespace") was classified `visual-design`, and nothing else, because of
# the bare word "whitespace". In the benchmark prompts (bench/domain/tasks*,
# 5 hits) it meant characters every time -- a parser's, a type challenge's
# TrimLeft -- and never layout. Same rule as `vertex` above: a word whose
# first sense in programming text is not the domain's keeps only the phrases
# that carry the domain's sense. "whitespace" (characters) needs a layout
# phrase; "hierarchy" (class / type / module hierarchy) needs "visual" and
# kin; "contrast" loses the English connective ("in contrast", "by
# contrast"). The design-sense phrases have NO measured positive in this
# repo's real input (n=0): they are kept so a design question phrased that
# way still matches, not because one was observed.
WORD_DOMAINS = {
    "visual-design": r"\b(colou?r|palette|typograph\w*|font|spacing|"
                     r"(?<!in )(?<!by )contrast|dark mode|"
                     r"(?:visual|typographic|information|content|heading) "
                     r"hierarchy|"
                     r"(?:more|less|generous|enough|extra|negative) white ?space|"
                     r"white ?space (?:between|around))\b",
    "ui-component": r"\b(button|modal|dropdown|form field|tooltip|navbar|"
                    r"component|input)\b",
    "motion": r"\b(animat\w*|transition|easing|keyframe|spring)\b",
    "accessibility": r"\b(a11y|accessib\w*|screen reader|aria|wcag|focus ring)\b",
    "brand": r"\b(brand|voice|tone of voice|identity)\b",
    "backend": r"\b(migration|endpoint|database|schema|query plan|index|"
               r"server|api route)\b",
    "algorithms": r"\b(complexity|big-?o|algorithms?|data structures?|sort|"
                  r"graph|competitive programming|standard input|"
                  r"sample input|sample output)\b",
    # No bare `vertex`/`vertices`: they are graph theory first. Measured, they
    # were the ONLY gpu terms to fire on the 342 LiveCodeBench prompts (149 and
    # 92 hits) and pulled 22 algorithm puzzles into `gpu`. "vertex shader" and
    # "vertex buffer" keep the GPU sense.
    "gpu": r"\b(shaders?|gpus?|webgpu|webgl\d?|wgsl|glsl|hlsl|tsl|"
           r"vertex (?:shaders?|buffers?|attributes?)|"
           r"fragment shader|textures?|meshe?s?|renderers?|"
           r"render targets?|render pipelines?|workgroups?|compute shaders?|"
           r"uniform buffers?|draw calls?)\b",
}

# Where an untagged recipe belongs, derived from the fields those corpora DO
# carry. Inferred rather than guessed: these map a corpus's own labels.
AREA_DOMAINS = {
    "threejs-tsl": {"gpu"}, "webgpu-typegpu": {"gpu"}, "wgsl": {"gpu"},
    "react-perf": {"web-frontend", "ui-component"},
    "typescript-perf": {"web-frontend", "types"},
    "ecs-dod": {"algorithms", "systems"},
    "r3f-core": {"web-frontend", "gpu"}, "r3f-perf": {"web-frontend", "gpu"},
    "r3f-hooks": {"web-frontend", "gpu"}, "r3f-webgpu": {"web-frontend", "gpu"},
    "r3f-version": {"web-frontend", "gpu"}, "drei": {"web-frontend", "gpu"},
    "data_oriented_design": {"algorithms", "systems"},
    "naming_api_style": {"tooling"},
}
LANGUAGE_DOMAINS = {
    "Rust": {"systems"}, "C": {"systems"}, "C++": {"systems"},
    "Zig": {"systems"}, "TypeScript": {"web-frontend", "types"},
}


# DERIVED DOMAINS (#20 in docs/SELF-IMPROVEMENT-LOG.md, 2026-09-24). A package
# indexed with no PACKAGE_DOMAINS entry reopened the tool gate for EVERY
# request (HELD_SOURCE_UNMAPPED) until someone mapped it by hand: the Octopus
# pilot indexed @pmndrs/glyph and three-flatland and the gate offered library
# help to all 342 LiveCodeBench prompts again. So a held package with no entry
# is mapped from its OWN IMPORTS, read from its indexed source files (the
# index's package_files, in the store's _src cache) by discover.imports -- a
# parser, not a pattern (PROTOCOL rule 8): the union of the HAND-MAPPED
# domains of every package it imports in at least DERIVE_MIN_FILES of its
# files. One level, through the hand map only, so a derivation never rests on
# another derivation. The indexer keeps no package.json, so keywords and peer
# dependencies are not available as evidence; a package whose imports name
# nothing mapped stays unmapped, and the gate says so as before. Computed on
# the first request after the package appears (held_sources is re-read on the
# directory listing), cached by the index file's mtime. DERIVE_MIN_FILES is a
# CHOICE: one stray import (a test helper, a vendored util) should not assign a
# domain.
DERIVE_MIN_FILES = 2
DERIVE_MAX_FILES = 600
_DERIVE_FENCE = {".ts": "ts", ".tsx": "tsx", ".mts": "ts", ".cts": "ts",
                 ".js": "js", ".mjs": "js", ".cjs": "js", ".jsx": "tsx",
                 ".py": "py", ".rs": "rs"}
_DERIVED_CACHE: dict = {}


def derived_domains(name: str, db: str) -> dict:
    """{domains, imports, files}: `name`'s domains derived from the packages
    its own source files import (see DERIVED DOMAINS). Never raises."""
    import discover
    try:
        key = (os.path.abspath(db), os.stat(db).st_mtime_ns)
    except OSError:
        return {"domains": [], "imports": {}, "files": 0}
    if key in _DERIVED_CACHE:
        return _DERIVED_CACHE[key]
    stem = os.path.basename(db)[:-len(".sqlite3")]
    src = os.path.join(os.path.dirname(db), "_src", stem)
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            files = [r[0] for r in con.execute(
                "SELECT path FROM package_files ORDER BY path")]
        finally:
            con.close()
    except sqlite3.Error:
        files = []
    counts: dict[str, int] = {}
    read = 0
    for path in files[:DERIVE_MAX_FILES]:
        lang = _DERIVE_FENCE.get(os.path.splitext(path)[1].lower())
        if not lang:
            continue
        try:
            with open(os.path.join(src, path), encoding="utf-8",
                      errors="replace") as f:
                head = "".join(f.readline() for _ in range(120))
        except OSError:
            continue
        read += 1
        for pkg in set(discover.imports(f"```{lang}\n{head}\n```")):
            if pkg != name:
                counts[pkg] = counts.get(pkg, 0) + 1
    doms: set[str] = set()
    used = {}
    for pkg, n in sorted(counts.items()):
        if n >= DERIVE_MIN_FILES and pkg in PACKAGE_DOMAINS:
            doms |= PACKAGE_DOMAINS[pkg]
            used[pkg] = n
    out = {"domains": sorted(doms), "imports": used, "files": read}
    _DERIVED_CACHE[key] = out
    return out


def package_domains(name: str, held: dict | None = None) -> set[str]:
    """A package's domains: its PACKAGE_DOMAINS entry, else, for a HELD
    package, derived from its own imports (DERIVED DOMAINS). Empty when
    neither says anything."""
    if name in PACKAGE_DOMAINS:
        return set(PACKAGE_DOMAINS[name])
    try:
        versions = (held if held is not None else held_sources()).get(name)
    except Exception:                                            # noqa: BLE001
        versions = None
    if not versions:
        return set()
    return set(derived_domains(name, versions[0][1])["domains"])


def _imported_domains(pkg: str) -> set[str]:
    """The domains a request's import of `pkg` implies: the hand map, else a
    held package's derived domains."""
    if pkg in PACKAGE_DOMAINS:
        return PACKAGE_DOMAINS[pkg]
    try:
        held = held_sources()
    except Exception:                                            # noqa: BLE001
        return set()
    return package_domains(pkg, held) if pkg in held else set()


def _blob(messages: list[dict]) -> str:
    text_all = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        if isinstance(c, str) and c:
            text_all.append(c)
    return "\n".join(text_all)


def detect(messages: list[dict], imports: list[str] | None = None) -> set[str]:
    """Domains this task plausibly belongs to, from what the caller sent.

    `imports`, when given, replaces the tree-sitter pass over the messages.
    The proxy has already parsed every import in the conversation once per
    request (`discover.scan`, accumulated by `nebari`), and parsing a long
    agent transcript twice to learn the same package names is pure cost.
    """
    import discover

    found: set[str] = set()
    blob = _blob(messages)

    for pkg in (discover.imports(blob) if imports is None else imports):
        found |= _imported_domains(pkg)
    for ext, doms in EXTENSION_DOMAINS.items():
        if re.search(re.escape(ext) + r"\b", blob):
            found |= doms
    low = blob.lower()
    for dom, pat in WORD_DOMAINS.items():
        if re.search(pat, low):
            found.add(dom)
    return found


def strong_evidence(messages: list[dict],
                    imports: list[str] | None = None) -> set[str]:
    """Domains from FACTS about code only -- imports and file extensions.

    `detect()` also reads words, which the module docstring ranks weakest and
    allows only to ADD domains, never to remove one. A filter that restricts
    on a word-only task set does exactly that: measured on the 89 hint probes
    (cached harness scores, bench/data/hint_collapse_runs.json), filtering on
    word-derived domains took bucket argmax from 64 to 59, and filtering with
    the strict corpora closed on no evidence took it to 41. So a caller that
    RESTRICTS asks this first, and restricts only when it is non-empty.
    """
    import discover

    blob = _blob(messages)
    found: set[str] = set()
    for pkg in (discover.imports(blob) if imports is None else imports):
        found |= _imported_domains(pkg)
    for ext, doms in EXTENSION_DOMAINS.items():
        if re.search(re.escape(ext) + r"\b", blob):
            found |= doms
    return found


def recipe_domains(recipe: dict, source_file: str = "") -> set[str] | None:
    """Domains a recipe declares, or infers from the labels it does carry.

    None means "untagged and unmappable", which the gate treats as eligible
    everywhere -- see the fail-open note in the module docstring.
    """
    d = recipe.get("domains")
    if isinstance(d, list) and d:
        return {x for x in d if x in DOMAINS} or None
    area = recipe.get("area") or ""
    if area in AREA_DOMAINS:
        return set(AREA_DOMAINS[area])
    lang = recipe.get("language") or ""
    if lang in LANGUAGE_DOMAINS:
        return set(LANGUAGE_DOMAINS[lang])
    return None


def eligible(recipe: dict, task: set[str], source_file: str = "") -> bool:
    """May this recipe be offered for a task in these domains?

    Applied before ranking, not after. A relevance score cannot express
    "inadmissible", and the bug being prevented is candidacy, not rank.
    """
    stem = os.path.basename(source_file).replace(".jsonl", "")
    doms = recipe_domains(recipe, source_file)
    if doms is None:
        # Untagged. Permissive, except for corpora that exist precisely
        # because they must not leak.
        return stem not in STRICT_FILES
    if not task:
        # No evidence about the task. Refusing to answer until the caller has
        # been classified is worse than an occasional irrelevant hint --
        # except, again, for the strict corpora.
        return stem not in STRICT_FILES
    return bool(doms & task)


def filter_recipes(recipes: list[dict], task: set[str],
                   source_file: str = "") -> list[dict]:
    return [r for r in recipes if eligible(r, task, source_file)]


# ---------------------------------------------------------------------------
# Tool admission: may this request be offered the code-intelligence tools?
# ---------------------------------------------------------------------------

# Where the dependency indexes live. None means `deps.STORE`, which is the
# directory `deps.db_path()` opens -- i.e. what `packages.search_discovered`
# actually searches. Reading any other directory would answer "what could be
# searched" from a place the search never looks. Tests point this at a temp
# directory; nothing else should set it.
PACKAGE_STORE: str | None = None

# How a request names a held package WITHOUT an import line. Keyed by the
# package's install name. A bare word is never an alias on its own when it is
# also English: "three" appears in "explain quicksort in three paragraphs",
# which is a real logged request (index/corpus.sqlite3), so three.js is
# matched by its spellings and its sub-paths, never by the numeral.
HELD_ALIASES = {
    "three": r"\bthree(?:\.js|js)\b|\bthree/(?:webgpu|tsl|addons|examples|src)\b"
             r"|\btsl\b",
    "typegpu": r"\btypegpu\b|\btgpu\b",
    "@react-three/fiber": r"@react-three/fiber|\br3f\b|\breact-three-fiber\b",
    "@react-three/drei": r"@react-three/drei|\bdrei\b",
    "koota": r"\bkoota\b",
    # Only unambiguous spellings. Bare "postprocessing" is an ordinary word in
    # shader questions, so that package is reached by import or by the
    # scoped @react-three name, never by the word alone.
    "@react-three/postprocessing": r"@react-three/postprocessing",
    "wgpu-matrix": r"\bwgpu-matrix\b",
    "three-mesh-bvh": r"\bthree-mesh-bvh\b|\bmesh-bvh\b",
    "type-fest": r"\btype-fest\b",
    "@typegpu/noise": r"@typegpu/noise",
    "@typegpu/three": r"@typegpu/three",
    # The unscoped pmndrs package. With no alias, `_named` fell back to the
    # bare install name, so the ordinary word "postprocessing" in a shader
    # question named this package. Reached by a spelling that can only be
    # the package: quoted or backticked as a module, its repo, a version pin.
    "postprocessing": r"""['"`]postprocessing['"`]|\bpmndrs/postprocessing\b"""
                      r"|\bpostprocessing@\d",
}

# An identifier-shaped token: camelCase, snake_case, or letters with a digit.
# Plain words are excluded because every package defines `run`, `map` and
# `sort`, and "Where" at the start of a sentence is Capitalised. What is left
# -- `Object3D`, `painterSortStable`, `DEFAULT_UP`, `WGSLNodeBuilder` -- is a
# name someone copied out of code, and if a held package DEFINES it, that is
# the strongest evidence available that the question is about that package.
_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{3,63}")
_CAMEL = re.compile(r"[a-z][A-Z]")
_DIGIT = re.compile(r"[A-Za-z].*\d|\d.*[A-Za-z]")
# Bounded because a 65k-token agent transcript yields thousands of tokens and
# the lookup is per request. The most recent text is scanned first, so the cap
# drops the oldest turns, which describe the current task least.
MAX_SYMBOL_PROBES = 400

# ---------------------------------------------------------------------------
# A NAME IS EVIDENCE ONLY IN CODE CONTEXT (2026-09-23)
#
# Live, 2026-09-23: a Hermes request -- "I want to start this project in
# ~/Developer/..." above an 8.8 KB pasted game spec -- logged
# `held=@types/three:MOUSE;three:MOUSE;wgpu-matrix:API`. Both are English
# in that text (an ALLCAPS spec heading, "Web Audio API"), and both are
# defined by a held package (three's src/constants.js `MOUSE`, wgpu-matrix's
# `API` type), so the symbol lookup called the request "about" three.js and
# deep thinking ran for minutes on a task that needed the client's own file
# tools. A capitalised or ALLCAPS word -- API, MOUSE, POINT, Event -- is
# spelled the same in prose and in code; what makes it a name is WHERE it
# stands. So such a word counts only in code context: backticked (selection
# reads backticks itself), or in identifier syntax:
#
#   Foo.bar / x.Foo    member access (THREE.MOUSE, MOUSE.LEFT, Info.autoReset)
#                      -- not a file or product spelling (Node.js, Three.js)
#   Foo(               a call or constructor (Loop(), new Pipelines())
#   Foo:: / ::Foo      a path
#   extends / implements / instanceof Foo, import Foo from, import { Foo }
#
# A fenced block is NOT treated as context word-for-word: Hermes wraps every
# pasted attachment in ```, so "every capitalised word inside a fence" would
# be every heading of a pasted spec. Code inside a fence is caught by the
# syntax above. Names that are code-shaped on their own (camelCase,
# snake_case, a digit) are unchanged: English words do not look like that.
#
# THE PLATFORM STOPLIST. In identifier syntax, `Array.from(...)`, `new
# Event('x')` and `Math.random()` are code -- but the code of the JS / DOM /
# Rust standard library, which every program uses and which a held package
# re-declaring (typegpu's `Array`, @types/three's `Event`) does not make the
# question about. A name the platform itself defines is never evidence for a
# held package, in any context. Deliberately NOT listed: `Node` and `Path`,
# which three.js defines as core API (src/nodes/core/Node.js,
# src/extras/core/Path.js) and a TSL question uses constantly; the DOM's
# `Node` is rarely what a request here writes.
PLATFORM_NAMES = frozenset("""
Array Object Function String Number Boolean Symbol BigInt Math Date JSON
Promise Map Set WeakMap WeakSet WeakRef Error TypeError RangeError SyntaxError
ReferenceError RegExp Reflect Proxy Intl Atomics ArrayBuffer DataView Iterator
Event EventTarget CustomEvent Element Document Window Image Audio Blob
File FileReader URL Request Response Headers Worker WebSocket Performance
Console Buffer Storage Location History Navigator Screen Selection Range Text
Comment Attr Record Partial Required Readonly Pick Omit Exclude Extract
Parameters Awaited NonNullable InstanceType ReturnType Uppercase Lowercase
Vec Box Option Result Some None Ok Err Rc Arc Cell RefCell Mutex HashMap
HashSet BTreeMap Self Default Clone Copy Debug Display Drop Send Sync Sized
Into From Iterator
NaN Infinity API GPU CPU URL URI HTTP HTTPS JSON HTML CSS DOM SVG XML UTF ASCII
""".split())

_CAP_NAME = r"[A-Z][A-Za-z0-9_$]{2,63}"
# What follows a dot when the dot is a file extension or a product spelling
# (Node.js, Three.js, index.html, example.com), not member access.
_NOT_MEMBER = (r"(?:js|jsx|ts|tsx|mjs|cjs|json|md|rs|py|css|html?|wasm|toml|"
               r"ya?ml|txt|io|org|com|net|dev|ai|app|sh|exe)\b")
_CONTEXT = [
    re.compile(r"\b(" + _CAP_NAME + r")\.(?!" + _NOT_MEMBER + r")(?=[A-Za-z_$])"),
    re.compile(r"(?<=[A-Za-z0-9_$\])]\.)(" + _CAP_NAME + r")\b"),
    re.compile(r"\b(" + _CAP_NAME + r")\((?!s\))"),
    re.compile(r"\b(" + _CAP_NAME + r")::"),
    re.compile(r"::(" + _CAP_NAME + r")\b"),
    re.compile(r"\b(?:extends|implements|instanceof)\s+(" + _CAP_NAME + r")\b"),
    re.compile(r"\bimport\s+(" + _CAP_NAME + r")\s+from\b"),
]
_IMPORT_BRACES = re.compile(r"\bimport\s*(?:type\s*)?\{([^}]{0,600})\}")


def code_context_names(text: str) -> list[str]:
    """Capitalised / ALLCAPS names that stand in identifier syntax in `text`.

    These are the names that are NOT code-shaped on their own and so are
    evidence only where they stand (see above). Platform names are dropped.
    Order of first appearance, no duplicates.
    """
    out: dict[str, None] = {}
    for pat in _CONTEXT:
        for m in pat.finditer(text):
            out.setdefault(m.group(1), None)
    for m in _IMPORT_BRACES.finditer(text):
        for tok in re.findall(r"\b(" + _CAP_NAME + r")\b", m.group(1)):
            out.setdefault(tok, None)
    return [t for t in out if t not in PLATFORM_NAMES]

_HELD_CACHE: dict = {}


def _unslug(stem: str) -> str:
    """Invert `deps.slug`: `react-three__fiber` -> `@react-three/fiber`."""
    return "@" + stem.replace("__", "/", 1) if "__" in stem else stem


def _store() -> str:
    if PACKAGE_STORE:
        return PACKAGE_STORE
    import deps
    return deps.STORE


def held_sources(store: str | None = None) -> dict[str, list[tuple[str, str]]]:
    """Every package index this server can actually search: {name: [(version, db)]}.

    ALIVE, NOT MERELY PRESENT. A database with zero chunks is counted as
    absent. PROTOCOL rule 1: both package indexes once held nothing but zero
    vectors and reported success; a file existing is not an index working.
    Chunk count is the floor the symbol table and BM25 both need -- a package
    index is built with INDEX_NO_EMBED=1, so non-zero vectors are not required
    for the searches that serve a remote caller.

    Cached on the directory's (name, mtime, size) listing, so indexing a new
    package is picked up on the next request with no restart.
    """
    root = store or _store()
    try:
        names = sorted(fn for fn in os.listdir(root)
                       if fn.endswith(".sqlite3") and "@" in fn)
    except OSError:
        return {}
    sig = []
    for fn in names:
        try:
            st = os.stat(os.path.join(root, fn))
            sig.append((fn, st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    key = (os.path.abspath(root), tuple(sig))
    if key in _HELD_CACHE:
        return _HELD_CACHE[key]

    out: dict[str, list[tuple[str, str]]] = {}
    for fn, _m, _s in sig:
        db = os.path.join(root, fn)
        stem, _, ver = fn[:-len(".sqlite3")].rpartition("@")
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error:
            n = 0
        # HELD MEANS USABLE, not "has chunks". A package that shipped only
        # dist/ used to index as its README -- 9 KB of fiber 10's 4.7 MB, zero
        # definitions -- and still counted as held, so the gate offered tools
        # that could only ever find nothing (2026-09-22). deps.index_health
        # requires definitions and a chunk from a code file; the cheap form
        # reads counts only.
        if n > 0:
            try:
                import deps
                if not deps.index_health(db, measure_bytes=False,
                                         scan_vectors=False)["ok"]:
                    n = 0
            except Exception:                                    # noqa: BLE001
                n = 0
        if n > 0:
            out.setdefault(_unslug(stem), []).append((ver, db))

    def vkey(v: str) -> tuple:
        return tuple(int(p) if p.isdigit() else -1 for p in v.split("."))

    out = {k: sorted(v, key=lambda x: vkey(x[0]), reverse=True)
           for k, v in sorted(out.items())}
    _HELD_CACHE.clear()
    _HELD_CACHE[key] = out
    return out


def _named(text: str, held: dict) -> list[str]:
    """Held packages the text names outright."""
    hit = []
    for name in held:
        pat = HELD_ALIASES.get(name) or (
            r"(?<![\w@/-])" + re.escape(name) + r"(?![\w-])")
        if re.search(pat, text, re.IGNORECASE):
            hit.append(name)
    return hit


def _symbols(text: str, held: dict) -> dict[str, list[str]]:
    """Code-shaped names in the text that a held package defines.

    Only names that are code-shaped on their own (camelCase, snake_case, a
    digit): English-shaped words -- API, MOUSE, POINT, Event -- are never
    probed here, and neither are platform names (PLATFORM_NAMES: `WeakMap`,
    `ArrayBuffer`, `HashMap`). Capitalised words in identifier syntax
    (`code_context_names`) are NOT added: measured on the benchmark prompts
    (bench/domain/tasks*), asking every table about them offered tools to 21
    React and type-challenge tasks on fiber's `React`, @types/three's `Equal`,
    typegpu's `Warn` -- the user's own code sharing a short name with a
    library. selection asks them only of a package the text names.
    """
    seen: list[str] = []
    have: set[str] = set()
    # Newest text last in the blob; walk it backwards so the cap keeps it.
    for tok in reversed(_TOKEN.findall(text)):
        if tok in have:
            continue
        if not (_CAMEL.search(tok) or _DIGIT.search(tok)
                or "_" in tok.strip("_$")):
            continue
        have.add(tok)
        seen.append(tok)
        if len(seen) >= MAX_SYMBOL_PROBES:
            break
    seen = [t for t in seen if t not in PLATFORM_NAMES]
    if not seen:
        return {}
    out: dict[str, list[str]] = {}
    for name, versions in held.items():
        db = versions[0][1]
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                found: set[str] = set()
                for i in range(0, len(seen), 200):
                    part = seen[i:i + 200]
                    q = ("SELECT DISTINCT name FROM defs WHERE name IN (%s)"
                         % ",".join("?" * len(part)))
                    found |= {r[0] for r in con.execute(q, part)}
            finally:
                con.close()
        except sqlite3.Error:
            # No symbol table is not evidence against the package; it only
            # means this signal cannot speak for it.
            continue
        if found:
            out[name] = sorted(found)
    return out


def _decision(offer: bool, situation: str, because: str, evidence: dict,
              retryable: bool | None = None,
              remedies: list[dict] | None = None) -> dict:
    d = {"offer": offer, "situation": situation, "because": because,
         "evidence": evidence}
    if not offer:
        d["retryable"] = bool(retryable)
        d["remedies"] = remedies or []
    return d


def tool_admission(messages: list[dict], root: str | None, *,
                   discovered: list[str] | None = None,
                   offered_before: bool = False,
                   store: str | None = None) -> dict:
    """Should this request carry the capability block and our tool list?

    A FACT ABOUT WHAT IS INDEXED, NOT A GUESS ABOUT WHAT WOULD HELP.

    WHAT IT COSTS WHEN THE ANSWER IS WRONGLY "YES", MEASURED.
    bench/lcb_after.jsonl, n=3 LiveCodeBench problems, three arms each. On
    abc390_b the augmented arm's prompt was 3,145 tokens against 416 for
    `minimal` and 374 for `aug_off`: 2,729 (87%) or 2,771 (88%) of it was the
    capability block and twelve tool definitions, and it called no tool. The
    other two problems carried 5,164 and 6,894 prompt tokens and ran 849s and
    889s. n=3 says the overhead is real, not how often the loops happen.

    WHY THE OLD GATE NEVER FIRED. `proxy.should_offer_tools` withheld only when
    `code_search.has_index()` was false. That reads `cs.INDEX_DB`, which in the
    proxy process is index/code.sqlite3 -- the server's OWN source, 7,742
    chunks of this repository, present on every deployment. But a caller with
    no repository is never searched against it: `run_our_tool` points
    CODE_INDEX_DB at `_no_repository_bound.sqlite3`, precisely so our source is
    not disclosed. So the gate asked about a database no caller's search can
    reach, got "yes" every time, and the LiveCodeBench arm above was offered
    tools while every index tool was guaranteed to return NO_INDEX.

    WHAT A CALLER WITHOUT A REPOSITORY CAN ACTUALLY REACH is the package store:
    `packages.search_discovered` serves the libraries the conversation imports,
    and `bind_project_context` lets the model name one. Nothing else. So the
    question this function answers is: does anything in the request point at a
    package this server holds a live index of?

    KNOWN, 2026-09-22, AND NOT FIXED HERE: that path is currently unreachable.
    `proxy.run_our_tool` returns `no_index_error` for any index tool when
    `db is None` -- i.e. whenever no repository is bound -- BEFORE it reaches
    the `packages.search_discovered` fallback further down. Verified
    in-process: a session whose state imports `three` gets NO_INDEX from
    find_definition_opt, while search_discovered called directly returns the
    three@0.185.1 source. This gate deliberately decides on what the package
    store COULD serve, not on that bug: deciding on the bug would withhold
    tools from every three.js question and stay wrong after the fix. Until
    run_our_tool is repaired, an offered no-repository request still costs the
    block and gets NO_INDEX -- which is exactly what every request got before
    this gate, so nothing is worse and every withheld request is better.

    THE RULE, in order. The first that applies decides.

      repository bound         offer. Its own index answers; this gate never
                               second-guesses a bound repository.
      offered earlier          offer. Once a conversation has had the tools it
                               keeps them: withdrawing them mid-session changes
                               the system prefix (a full prefill, ~51s at 65k)
                               and takes `read_rings` away exactly when a
                               compaction has made the work log necessary.
      nothing held             WITHHOLD. No repository and no live package
                               index: every index tool returns NO_INDEX for
                               any arguments. Not retryable by the caller.
      imports a held package   offer. search_discovered will serve it.
      names a held package     offer. "three.js", "TSL", "typegpu".
      a held package defines   offer. `Object3D`, `painterSortStable`: a code-
      a symbol it mentions     shaped name that the package's own symbol table
                               holds. Decided from the index itself.
      a held package has no    offer. Its domains are unknown, so no request
      domain mapping           can be ruled out by domain. A package counts as
                               mapped by PACKAGE_DOMAINS or by the domains
                               DERIVED from its own imports (derived_domains,
                               2026-09-24); only one that neither maps turns
                               this gate off.
      no domain evidence       offer. This module's own rule: "Nothing is
                               inferred from the absence of a signal."
      domains meet a held      offer.
      package's domains
      otherwise                WITHHOLD. The request carries domain evidence,
                               none of it is a domain any held package serves,
                               and nothing names or imports one.

    ONLY THE LAST RULE USES DOMAIN, AND ONLY TO WITHHOLD ON POSITIVE EVIDENCE.
    An earlier version of the proxy's gate rejected domain outright, fearing a
    misread domain would withhold `find_references` from someone asking about
    their own code. That cannot happen here: with no repository the caller's
    own code is not indexed and `find_references` returns NO_INDEX regardless,
    and with a repository this function returns at the first rule.

    WHERE THE EVIDENCE COMES FROM. Evidence that can only lead to OFFERING --
    imports, names, symbols, a domain match -- is read from every message,
    system prompt included: a harness that says "this is a three.js project"
    has said something true. Evidence that can lead to WITHHOLDING is read
    only from the task (every role but system). A harness's boilerplate says
    "server" and "input" on every request and is not a statement about the
    task; letting it withhold would turn a harness's wording into a gate.

    MEASURED AGAINST REAL INPUT (PROTOCOL rule 7) -- see mcp/test_domains.py,
    which replays these against the live package store when it is present:
    all 26 hand-written three.js questions in bench/context_economy_tasks.jsonl
    are offered, and all 342 LiveCodeBench prompts in bench/data/test5+test6
    are withheld. What has NOT been measured is the end-to-end effect on answer
    quality; that needs the GPU and is listed in the report that shipped this.
    """
    held = held_sources(store)
    names_held = sorted(held)
    everything = _blob(messages)
    task_msgs = [m for m in messages if m.get("role") != "system"]
    ev: dict = {"held": {k: [v for v, _db in vs] for k, vs in held.items()}}

    if root:
        return _decision(True, "REPOSITORY_BOUND",
                         "a repository is bound; its own index answers", ev)
    if offered_before:
        return _decision(True, "OFFERED_EARLIER_THIS_SESSION",
                         "this conversation already had the tools; withdrawing "
                         "them would change the prefix and remove the work log",
                         ev)
    if not held:
        return _decision(
            False, "NOTHING_HELD",
            ("no repository is bound and the server holds no live package "
             "index, so every index tool would return NO_INDEX for any "
             "arguments"), ev, retryable=False,
            remedies=[
                {"fixable_by": "operator",
                 "action": ("index a dependency into the package store "
                            f"({_store()}) with scripts/index_code.py"),
                 "effect": "requests about that library are offered the tools"},
                {"fixable_by": "user",
                 "action": "run the client where the server can see the repository",
                 "effect": "the repository is indexed and the tools are offered"}])

    imported = sorted(set(discovered or []) & set(held))
    if imported:
        ev["imports"] = imported
        return _decision(True, "IMPORTS_HELD_SOURCE",
                         "the conversation imports " + ", ".join(imported), ev)
    named = _named(everything, held)
    if named:
        ev["names"] = named
        return _decision(True, "NAMES_HELD_SOURCE",
                         "the conversation names " + ", ".join(named), ev)
    syms = _symbols(everything, held)
    if syms:
        ev["symbols"] = {k: v[:8] for k, v in syms.items()}
        return _decision(True, "DEFINES_MENTIONED_SYMBOL",
                         "a held package defines "
                         + "; ".join(f"{k}: {', '.join(v[:4])}"
                                     for k, v in syms.items()), ev)
    # Mapped by hand, else derived from the package's own imports (DERIVED
    # DOMAINS); only a package neither maps is unmapped.
    doms = {n: package_domains(n, held) for n in names_held}
    derived = {n: derived_domains(n, held[n][0][1])
               for n in names_held if n not in PACKAGE_DOMAINS and doms[n]}
    if derived:
        ev["derived"] = {n: {"domains": d["domains"], "imports": d["imports"]}
                         for n, d in derived.items()}
    unmapped = [n for n in names_held if not doms[n]]
    if unmapped:
        ev["unmapped"] = unmapped
        return _decision(True, "HELD_SOURCE_UNMAPPED",
                         "held package(s) with no domain mapping, so no request "
                         "can be ruled out by domain: " + ", ".join(unmapped), ev)

    served = set().union(*(doms[n] for n in names_held))
    ev["held_domains"] = sorted(served)
    anywhere = detect(messages, imports=list(discovered or []))
    if anywhere & served:
        ev["domains"] = sorted(anywhere)
        return _decision(True, "DOMAIN_MATCHES_HELD_SOURCE",
                         "request domains " + ", ".join(sorted(anywhere & served))
                         + " are served by a held package", ev)
    task = detect(task_msgs)
    ev["domains"] = sorted(task)
    if not task:
        return _decision(True, "NO_DOMAIN_EVIDENCE",
                         "the task carries no domain evidence; nothing is "
                         "inferred from an absence", ev)
    return _decision(
        False, "DOMAIN_OUTSIDE_HELD_SOURCES",
        (f"no repository is bound; the task's domains ({', '.join(sorted(task))}) "
         f"are not served by any held package ({', '.join(names_held)}: "
         f"{', '.join(sorted(served))}), and it neither imports, names nor "
         f"mentions a symbol from one"),
        ev, retryable=True,
        remedies=[
            {"fixable_by": "user",
             "action": "name the library, or paste code that imports it",
             "effect": ("the next turn re-decides, and offers the tools if the "
                        "server holds that library")},
            {"fixable_by": "operator",
             "action": ("index the library into the package store (its "
                        "domains are derived from its own imports; map it "
                        "in domains.PACKAGE_DOMAINS only if that is wrong)"),
             "effect": "requests in its domain are offered the tools"}])


if __name__ == "__main__":
    cases = [
        ("a React button", [{"role": "user", "content":
            "```tsx\nimport { useState } from 'react'\n```\nMake this button "
            "feel nicer, the colour contrast looks off"}]),
        ("a Rust backend", [{"role": "user", "content":
            "```rs\nuse tokio::net::TcpListener;\nuse sqlx::PgPool;\n```\n"
            "Write the database migration for the users table"}]),
        ("a WGSL shader", [{"role": "user", "content":
            "```wgsl\n@compute @workgroup_size(64)\n```\nspeed this up"}]),
        ("bare prose", [{"role": "user", "content": "how do I reverse a list"}]),
    ]
    samples = [
        ({"domains": ["visual-design"], "recipe": "colour rule"}, "design_visual.jsonl"),
        ({"language": "Rust", "recipe": "memory layout"}, "systems.jsonl"),
        ({"area": "wgsl", "recipe": "workgroup sizing"}, "web_gpu.jsonl"),
        ({"recipe": "untagged general advice"}, "complexity.jsonl"),
    ]
    for label, msgs in cases:
        task = detect(msgs)
        print(f"\n  {label}\n    detected: {sorted(task) or '(none)'}")
        for rec, src in samples:
            tag = (rec.get("domains") or rec.get("language")
                   or rec.get("area") or "untagged")
            ok = eligible(rec, task, src)
            print(f"      {'offer ' if ok else 'BLOCK '} {str(tag):<16} {src}")
