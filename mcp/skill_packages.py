#!/usr/bin/env python
"""WHICH HELD PACKAGE IS IN PLAY, AND ITS SKILL.

    python mcp/skill_packages.py vocab [--samples 20]   the vocabulary report
    python mcp/skill_packages.py detect FILE            detect over a text file
    python mcp/skill_packages.py mapping                 package -> skill

Operator, 2026-09-27: "if the prompt references something related to r3f or
Koota or TSL, it gets the skill, the skill is the best way to use this tech
... discover by a cloud of related words and methods what package it is
likely using, and give it the skill."

THREE PARTS, each pure except for reading the held package sources:

  1. THE VOCABULARY (vocabulary()). For every HELD package
     (index/packages/_src/<slug>@<version>/, the sources deps.py fetched; read
     only, every held version of a package merged), the names it exports:
     the names in its `export` statements and the members its declaration
     files (.d.ts) declare; a package with no declarations (three) is read
     from its JavaScript exports -- three's addons (examples/jsm, the
     `three/addons` entry) included, their relative `export *` followed --
     and from its @types package. `three/tsl` is its own unit: three's TSL
     entry (src/Three.TSL.js), the classes three/webgpu exports that the
     WebGL entry (src/Three.js) does not, and the members of the node
     materials (src/materials/nodes/*.d.ts: colorNode, positionNode ...);
     three keeps the rest. Members are read only inside `{ }` bodies (a
     tuple's element labels are not members), and a bundler chunk's short
     aliases (`TraitValue as P`) resolve to the declared name.
     A symbol DETECTS a package only if exactly ONE package exports it (a
     structural rule, no threshold: `world`, `query`, `get` fall out on
     their own; React's own types -- web/node_modules/@types/react,
     react-dom, react-reconciler -- take part in that test as a library
     that is never in play, so fiber's vendored reconciler names are not
     fiber's), and it is neither
       - a JavaScript/TypeScript global or platform API: every name the
         TypeScript compiler's own lib files declare (web/node_modules/
         typescript/lib/lib.*.d.ts: ES, DOM, WebWorker ...), @types/node's,
         and @webgpu/types' (held, but it types the browser's WebGPU API);
       - a plain English word IN PROSE (in a code-like context -- an import,
         `<Canvas`, `Fn(`, `<Html`, `.name`, a fence, written code, an error
         line -- a unique name counts even when it is also a word: Canvas,
         Fn, Html, Bloom are their packages' core API; coordinator,
         2026-09-27): a WORD-SHAPED name (all lower-case, or
         Capitalized: `texture`, `Position`, `Canvas`) whose lower-case form
         is an ordinary word of the lexicon: the words used in PROSE (code
         fences and backtick spans removed) of the READMEs of the npm
         packages the dashboard installs (not the held ones' own) and of
         this Python environment's package descriptions and documentation
         (pydoc_data.topics), plus the whole-word tokens of the pinned
         Qwen3-8B tokenizer (a byte-pair vocabulary makes a whole word one
         token only when it is frequent in web text). A name with inner
         capitals, digits, `_` or `$` (useFrame, vec3) is never a plain
         word. Derived, not listed.
  2. DETECTION (detect()). Over a turn's evidence -- the user turn, pasted
     code, the newest tool calls and results, error lines; our own injected
     craft block stripped and a dependency's own files set aside
     (skill_select.evidence_view) -- a package is IN PLAY on
       name      its name or alias in the user's own words (skill_classify.
                 asked_terms' vocabulary: r3f, koota, pmndrs math, three.js,
                 TSL, typegpu; the exact npm name for the rest), never
                 negated ("without koota")
       implied   the implication table (skill_select.IMPLIES) at package
                 level: R3F v10 is WebGPU-first, its materials three.js TSL
       import    an import or require of it (a subpath counts: three/tsl)
       pin       a dependency pin (package.json, `name@1.2.3`, npm install)
       symbol    a unique exported symbol in code-like context -- a fence,
                 a tool call's written code (string literals aside), a
                 backtick span, a call `name(`, a JSX tag `<Name`, a member
                 `.name` (not a file extension: `Chart.js`), a code-SHAPED
                 name in prose (useFrame, GLTFLoader) -- or in an error
                 line; never a config file's keys (skill_select.
                 evidence_view's config rule: tsconfig's moduleResolution)
     Each package carries why: the name, import, pin or symbol that put it
     in play, and where.
  3. PACKAGE -> SKILL (CANONICAL) and THE INJECTION RULE (decide()): a
     package in play whose canonical skill has not been given in this
     conversation gets the BODY; one in play again after its skill was
     given gets a RECALL line only on one of the selection chart's recall
     events for it (skill_chart.AREA_MACHINE: given --ASKED | ERROR |
     PHASE--> recall_eligible; the fade, FADE_TOKENS, was removed from
     skill_chart on 2026-09-27); otherwise nothing. State only, no numbers.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import sys
import threading
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import package_registry  # noqa: E402  (stdlib-only at import)

# A CANDIDATE vocabulary under evaluation (package_registry.floor, through
# using()): its held dirs and vocabulary replace the promoted ones while
# entered. Process-global, one lock.
_OVERRIDE: dict = {"held": None, "vocab": None}
_USING = threading.RLock()

SRC = os.environ.get("YAMADORI_PACKAGES_SRC") or os.path.join(
    ROOT, "index", "packages", "_src")
TS_LIB = os.environ.get("YAMADORI_TS_LIB") or os.path.join(
    ROOT, "web", "node_modules", "typescript", "lib")
NODE_TYPES = os.environ.get("YAMADORI_NODE_TYPES") or os.path.join(
    ROOT, "web", "node_modules", "@types", "node")
# The web-text tokenizer tokenizer_words() reads: the pinned Qwen3-8B
# tokenizer.json (models/manifest.yaml qwen3-8b-tokenizer). It was CLM's
# (mcp/clm.py TOKENIZER_PATH, same default path and override) until CLM was
# removed, 2026-09-29.
WORDS_TOKENIZER = os.environ.get("YAMADORI_CLM_TOKENIZER") or os.path.join(
    os.environ.get("YAMADORI_MODELS_DIR",
                   "C:/Users/jwals/textgen/user_data/models"),
    "Qwen3-8B", "tokenizer.json")
# The prose corpus for "a plain English word" (see the module docstring).
PROSE_GLOBS = (
    # the README of every npm package the dashboard installs, and the long
    # description of every package in this Python environment: developers'
    # English at large. Not this repo's docs: they name the held packages'
    # APIs in running prose (updateEach, useFrame, fbm).
    os.path.join(ROOT, "web", "node_modules", "*", "[Rr][Ee][Aa][Dd][Mm][Ee]*.md"),
    os.path.join(ROOT, "web", "node_modules", "@*", "*",
                 "[Rr][Ee][Aa][Dd][Mm][Ee]*.md"),
    os.path.join(os.path.dirname(os.__file__), "site-packages", "*.dist-info",
                 "METADATA"))

TSL = "three/tsl"
# A held package that types a PLATFORM API, not a library: its names are
# excluded like the TypeScript lib's, and it is never "in play".
PLATFORM_PACKAGES = ("@webgpu/types",)
# Libraries that are NOT held but that the held ones are built on and ship
# types from (@react-three/fiber 9 vendors react-reconciler's declarations;
# fiber and drei declare React props): their names are in the uniqueness
# test -- a name React also declares is shared, not fiber's -- and they are
# never "in play". Read from the dashboard's own installed types
# (web/node_modules/@types), read only.
REFERENCE_TYPES = tuple(os.path.join(ROOT, "web", "node_modules", "@types", n)
                        for n in ("react", "react-dom", "react-reconciler"))

_ID = re.compile(r"[A-Za-z_$][\w$]*")


# ---------------------------------------------------------------------------
# 1. The vocabulary.
# ---------------------------------------------------------------------------
def package_of_dir(d: str) -> tuple[str, str]:
    """('@react-three/fiber', '10.0.0-alpha.5') for
    'react-three__fiber@10.0.0-alpha.5' (deps.slug's inverse); an @types
    package names the package it types."""
    name, _, ver = os.path.basename(d).rpartition("@")
    name = "@" + name.replace("__", "/") if "__" in name else name
    if name.startswith("@types/"):
        name = name[len("@types/"):]
    return name, ver


def held_from_dirs(dirs) -> dict[str, list[str]]:
    """{package: [source dirs]} for source dir basenames under SRC (those
    that exist), in sorted order."""
    out: dict[str, list[str]] = defaultdict(list)
    for b in sorted(dirs or ()):
        d = os.path.join(SRC, b)
        if os.path.isdir(d):
            out[package_of_dir(d)[0]].append(d)
    return dict(out)


def held() -> dict[str, list[str]]:
    """{package: [source dirs]} for every held package (all versions): the
    PROMOTED manifest's dirs (package_registry's held.json) when it exists,
    else every dir under SRC. While a candidate is evaluated (using()), its
    dirs."""
    if _OVERRIDE["held"] is not None:
        return dict(_OVERRIDE["held"])
    m = package_registry.held_manifest()
    if m is not None:
        return held_from_dirs(m["dirs"])
    out: dict[str, list[str]] = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(SRC, "*@*"))):
        if os.path.isdir(d):
            out[package_of_dir(d)[0]].append(d)
    return dict(out)


def held_versions(pkg: str) -> list[str]:
    return sorted(package_of_dir(d)[1] for d in held().get(pkg, [])
                  if not os.path.basename(d).startswith("types__"))


def _read(p: str) -> str:
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


_EXPORT_DECL = re.compile(
    r"^\s*export\s+(?:declare\s+)?(?:default\s+)?(?:abstract\s+)?"
    r"(?:async\s+)?(?:function\*?|const|let|var|class|interface|type|enum|"
    r"namespace)\s+([A-Za-z_$][\w$]*)", re.M)
_EXPORT_LIST = re.compile(r"\bexport\s+(?:type\s+)?\{([^}]*)\}", re.S)
_EXPORT_NS = re.compile(r"\bexport\s+\*\s+as\s+([A-Za-z_$][\w$]*)")
_MEMBER = re.compile(
    r"^[ \t]+(?:(?:readonly|static|public|abstract|override|declare|get|"
    r"set|async)\s+)*([A-Za-z_$][\w$]*)\??\s*(?:<[^>\n]*>)?\s*[(:]", re.M)
_PRIVATE = re.compile(r"^[ \t]+(?:private|protected|#)")
_DECLARED = re.compile(
    r"^\s*(?:export\s+)?declare\s+(?:function|const|let|var|class|"
    r"interface|type|enum|namespace|abstract\s+class)\s+"
    r"([A-Za-z_$][\w$]*)|^\s*(?:interface|type|class)\s+([A-Za-z_$][\w$]*)",
    re.M)


def exports_of(text: str) -> set[str]:
    """The public names a module's export statements give. In `export { a
    as b }` the public name is `b` -- unless `a` is declared in this same
    file: then the file is a bundler's chunk re-exporting its own
    declarations under short aliases (koota's `TraitValue as P`), which the
    entry module imports back under their names, and `a` is the name."""
    out = set(_EXPORT_DECL.findall(text)) | set(_EXPORT_NS.findall(text))
    local = None
    for body in _EXPORT_LIST.findall(text):
        for part in body.split(","):
            part = re.sub(r"^\s*type\s+", "", part.strip())
            if not part:
                continue
            a, _, b = part.partition(" as ")
            a, b = a.strip(), b.strip()
            if b:
                if local is None:
                    local = {x or y for x, y in _DECLARED.findall(text)}
                name = a if a in local else b
            else:
                name = a
            if _ID.fullmatch(name) and name != "default":
                out.add(name)
    return out


def _line_brackets(text: str) -> list[str]:
    """For each line, the innermost open bracket at its start ('{', '[',
    '(' or '' at top level). Strings, template literals and comments are
    skipped; a declaration file has few of them."""
    out = []
    stack: list[str] = []
    i, n = 0, len(text)
    out.append("")
    quote = None
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            elif c == "\n":
                out.append(stack[-1] if stack else "")
                if quote != "`":
                    quote = None
            i += 1
            continue
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.extend([stack[-1] if stack else ""] * text.count("\n", i, j))
            i = j
            continue
        if c in "'\"`":
            quote = c
        elif c in "{[(":
            stack.append(c)
        elif c in "}])" and stack:
            stack.pop()
        elif c == "\n":
            out.append(stack[-1] if stack else "")
        i += 1
    return out


def members_of(text: str) -> set[str]:
    """The member names a declaration file declares inside `{ ... }`
    bodies (an interface, a class, an object type): public ones only
    (private and protected lines skipped); a tuple's element labels
    (`[e1: number, ...]`) and a function's parameters are not members."""
    out = set()
    lines = text.splitlines()
    inner = _line_brackets(text)
    for k, line in enumerate(lines):
        if k < len(inner) and inner[k] != "{":
            continue
        s = line.lstrip()
        if not s or s.startswith(("*", "/", "#")) or _PRIVATE.match(line):
            continue
        m = _MEMBER.match(line)
        if m:
            out.add(m.group(1))
    return out


def declared_of(text: str) -> set[str]:
    """Every name a lib file declares: its declarations and members."""
    out = set()
    for a, b in _DECLARED.findall(text):
        out.add(a or b)
    return out | members_of(text) | exports_of(text)


def _files(d: str, exts: tuple) -> list[str]:
    out = []
    for dp, dn, fn in os.walk(d):
        dn[:] = [x for x in dn if x not in ("node_modules",
                                            "test", "tests", "__tests__")]
        for f in fn:
            if f.endswith(exts):
                out.append(os.path.join(dp, f))
    return sorted(out)


def package_names(dirs: list[str]) -> set[str]:
    """Exports and declared members of one package over its dirs: .d.ts
    files when the dir has any, else its JavaScript exports."""
    names: set[str] = set()
    for d in dirs:
        dts = _files(d, (".d.ts", ".d.mts", ".d.cts"))
        if dts:
            for p in dts:
                t = _read(p)
                names |= exports_of(t) | members_of(t)
        else:
            for p in _files(d, (".js", ".mjs")):
                names |= exports_of(_read(p))
    return names


_STAR = re.compile(r"""^\s*export\s+\*\s+from\s+['"](\.[^'"]+)['"]""", re.M)


def module_exports(path: str, seen: set | None = None) -> set[str]:
    """A JavaScript module's exports, following its relative `export *
    from './x.js'` re-exports (three's entry files are built from them)."""
    seen = set() if seen is None else seen
    path = os.path.normpath(path)
    if path in seen:
        return set()
    seen.add(path)
    t = _read(path)
    out = exports_of(t)
    for rel in _STAR.findall(t):
        out |= module_exports(os.path.join(os.path.dirname(path), rel), seen)
    return out


def tsl_names(three_dirs: list[str]) -> set[str]:
    """three's TSL unit (see the module docstring)."""
    out: set[str] = set()
    webgl: set[str] = set()
    webgpu: set[str] = set()
    for d in three_dirs:
        base = os.path.basename(d)
        src = os.path.join(d, "src")
        if base.startswith("types__"):
            for p in _files(os.path.join(src, "materials", "nodes"),
                            (".d.ts",)):
                out |= members_of(_read(p))
            continue
        out |= exports_of(_read(os.path.join(src, "Three.TSL.js")))
        webgl |= module_exports(os.path.join(src, "Three.js"))
        webgpu |= module_exports(os.path.join(src, "Three.WebGPU.js"))
    return out | (webgpu - webgl - {"TSL"})


def platform_names(h: dict | None = None) -> set[str]:
    """JavaScript/TypeScript globals and platform APIs (derived: the
    TypeScript lib files, @types/node, the held platform type packages)."""
    out: set[str] = set()
    for p in sorted(glob.glob(os.path.join(TS_LIB, "lib.*.d.ts"))):
        out |= declared_of(_read(p))
    for p in _files(NODE_TYPES, (".d.ts",)):
        out |= declared_of(_read(p))
    h = held() if h is None else h
    for pk in PLATFORM_PACKAGES:
        for d in h.get(pk, []):
            for p in _files(d, (".d.ts",)):
                out |= declared_of(_read(p))
    return out


_FENCE = re.compile(r"```.*?```", re.S)
_TICK = re.compile(r"`[^`\n]*`")


def _held_readme(path: str, names=None) -> bool:
    """A README of a package this stack holds (the dashboard installs three
    and @react-three/*): the source of the symbols, not English at large."""
    p = path.replace("\\", "/")
    names = set(held()) if names is None else set(names)
    return any(f"/node_modules/{n}/" in p for n in names)


def prose_words(names=None) -> set[str]:
    """The words used as ORDINARY words in English prose: code fences and
    backtick spans removed, and a word followed by `(` or joined to `.`,
    `_`, `/`, `$`, `@`, `#` or `-` is not prose. A word-shaped one (all
    lower-case, or Capitalized) is kept lower-cased; one with inner capitals
    (WebGPU, JavaScript) is kept as written. `names`: the held packages whose
    READMEs are left out (default: held())."""
    names = set(held()) if names is None else set(names)
    texts = []
    for g in PROSE_GLOBS:
        texts += [_read(p) for p in sorted(glob.glob(g))
                  if not _held_readme(p, names)]
    try:
        import pydoc_data.topics as topics
        texts += list(topics.topics.values())
    except Exception:                                            # noqa: BLE001
        pass
    out: set[str] = set()
    rx = re.compile(r"(?<![\w.$/@#-])([A-Za-z]+)(?![\w(.$/-])")
    for t in texts:
        t = _TICK.sub(" ", _FENCE.sub(" ", t))
        for w in rx.findall(t):
            word = w.islower() or (w[:1].isupper() and w[1:].islower())
            out.add(w.lower() if word else w)
    return out


def tokenizer_words() -> set[str]:
    """The whole words of a web-text tokenizer's vocabulary: every
    space-prefixed token that is a lower-case or Capitalized word, lower-
    cased. A byte-pair vocabulary merges a whole word into one token only
    when the word is frequent in its training text, so these are the words
    common in text at large ("water", "decoration", "spawn"), a rare API
    name ("updateEach", "metalness") is not one. The pinned Qwen3-8B
    tokenizer (WORDS_TOKENIZER), read only; absent, the prose corpus stands
    alone."""
    try:
        with open(WORDS_TOKENIZER, encoding="utf-8") as f:
            vocab = json.load(f)["model"]["vocab"]
    except Exception:                                            # noqa: BLE001
        return set()
    out = set()
    for t in vocab:
        if t.startswith("Ġ"):
            w = t[1:]
            if w.isalpha() and (w.islower() or (w[:1].isupper()
                                                and w[1:].islower())):
                out.add(w.lower())
    return out


def is_english(name: str, prose: set[str]) -> bool:
    """A plain English word: a word-shaped name -- all lower-case
    (`texture`) or Capitalized (`Position`, `Canvas`) -- whose lower-case
    form is used as an ordinary word in the lexicon. A name with inner
    capitals, digits, `_` or `$` (useFrame, vec3, BRDF_GGX) never is."""
    if not name.isalpha():
        return False
    if name.islower() or (name[:1].isupper() and name[1:].islower()):
        return name.lower() in prose
    # Inner capitals: a coined name, never a plain word. (Tried 2026-09-27:
    # counting one as English when prose uses it as written dropped useFrame
    # and MeshBasicMaterial, which the three.js ecosystem's READMEs write
    # unquoted, to lose WebGPU and WebGL.)
    return False


_VOCAB: dict | None = None
_FILE_VOCAB: dict = {"key": None, "vocab": None}


def promoted_vocabulary() -> dict | None:
    """The PROMOTED vocabulary (package_registry.VOCABULARY), re-read when
    the file's stat changes; None when there is none."""
    path = package_registry.VOCABULARY
    key = package_registry._stat_key(path)
    if key is None:
        return None
    if _FILE_VOCAB["key"] != (path, key):
        try:
            with open(path, encoding="utf-8") as f:
                v = json.load(f).get("vocabulary")
        except (OSError, ValueError, AttributeError):
            v = None
        if not isinstance(v, dict) or "symbols" not in v:
            return None
        _FILE_VOCAB.update(key=(path, key), vocab=v)
    return _FILE_VOCAB["vocab"]


def vocabulary(rebuild: bool = False) -> dict:
    """{"symbols": {symbol: package}, "per_package": {package: {"names",
    "unique", "dropped_shared", "dropped_platform", "dropped_english"}},
    "packages": [...]}. The promoted vocabulary.json when it exists (never
    built on the request path then); else built once per process from the
    held sources. `rebuild` builds from the held sources whatever exists.
    While a candidate is evaluated (using()), the candidate's."""
    global _VOCAB
    if _OVERRIDE["vocab"] is not None:
        return _OVERRIDE["vocab"]
    if not rebuild:
        v = promoted_vocabulary()
        if v is not None:
            return v
        if _VOCAB is not None:
            return _VOCAB
    _VOCAB = build_vocabulary(held())
    return _VOCAB


@contextlib.contextmanager
def using(vocabulary=None, held_dirs=None, entries=()):
    """Detect under a CANDIDATE: its vocabulary, its held dirs and its
    registry entries (package_registry.using) in place of the promoted
    ones, until the block exits. Process-global while entered; the process's
    own vocabulary and registry are untouched and come back on exit."""
    with _USING:
        prev = dict(_OVERRIDE)
        _OVERRIDE["vocab"] = vocabulary
        _OVERRIDE["held"] = (held_from_dirs(held_dirs)
                             if held_dirs is not None else None)
        try:
            with package_registry.using(entries):
                yield
        finally:
            _OVERRIDE.update(prev)


def build_vocabulary(h: dict[str, list[str]]) -> dict:
    """The vocabulary of the held sources `h` ({package: [dirs]}): pure
    with respect to the process (nothing cached)."""
    names: dict[str, set[str]] = {}
    for pkg, dirs in h.items():
        if pkg in PLATFORM_PACKAGES:
            continue
        names[pkg] = package_names(dirs)
    for d in REFERENCE_TYPES:
        if os.path.isdir(d):
            names["(ref) " + os.path.basename(d)] = package_names([d])
    if "three" in h:
        tsl = tsl_names(h["three"])
        names[TSL] = tsl
        names["three"] = names.get("three", set()) - tsl
    owners: dict[str, list[str]] = defaultdict(list)
    for pkg, ns in names.items():
        for n in ns:
            owners[n].append(pkg)
    platform = platform_names(h)
    english = prose_words(h) | tokenizer_words()
    symbols: dict[str, str] = {}
    # Unique names that are plain English words: they detect only in a
    # code-like context (an import, `<Canvas`, `Fn(`, `.name`, a fence,
    # written code, an error line), never in prose.
    code_only: dict[str, str] = {}
    per: dict[str, dict] = {pkg: {"names": len(ns), "unique": 0,
                                  "dropped_shared": 0, "dropped_platform": 0,
                                  "dropped_english": 0}
                            for pkg, ns in names.items()}
    for n, pk in owners.items():
        if len(pk) > 1:
            for p in pk:
                per[p]["dropped_shared"] += 1
            continue
        p = pk[0]
        if p.startswith("(ref) "):
            continue
        if n in platform:
            per[p]["dropped_platform"] += 1
        elif is_english(n, english):
            per[p]["dropped_english"] += 1
            code_only[n] = p
        else:
            symbols[n] = p
            per[p]["unique"] += 1
    return {"symbols": symbols, "code_symbols": {**symbols, **code_only},
            "per_package": per,
            "packages": sorted(p for p in names
                               if not p.startswith("(ref) ")),
            "platform": len(platform),
            "english": len(english)}


# ---------------------------------------------------------------------------
# 2. Detection.
# ---------------------------------------------------------------------------
# Names and aliases, in the user's own words: skill_classify.asked_terms (its
# vocabulary, its negation and ownership rules, and the version written after
# a name), each term mapped to the held package it names. `threejs` is split
# by the form the user wrote (TSL, three/tsl, three/webgpu are TSL) and `r3f`
# by its npm scope (@react-three/drei is drei). Every other held package is
# named by its exact npm name. Read from the package registry
# (package_registry: its SEED is the hand-written map, r3f ->
# @react-three/fiber, koota, pmndrs_math -> math, threejs -> three, typegpu;
# an onboarded package adds its own term), live.
TERM_PACKAGE = package_registry.live("term_package")
_TSL_FORM = re.compile(r"^(?:TSL|three/(?:tsl|webgpu))$", re.I)
# An identifier whose SHAPE is code wherever it stands -- an inner capital
# (useFrame, updateEach, MeshBVH), an underscore, a digit or a `$` -- is
# code-like context by itself, in prose too ("the particles stutter in
# useFrame").
_CODE_SHAPED = re.compile(r"(?<![\w$])([A-Za-z_$]*(?:[a-z][A-Z]|[A-Z]{2}[a-z]"
                          r"|[_$\d])[\w$]*)")


def names_in(prose: str) -> list[tuple[str, str, str | None]]:
    """[(package, the form written, version)] named in the user's prose."""
    import skill_classify
    out = []
    term_pkg = dict(TERM_PACKAGE)
    for term, e in skill_classify.asked_terms(prose).items():
        pkg = term_pkg.get(term)
        form = str(e.get("form") or "")
        if term == "threejs" and _TSL_FORM.match(form):
            pkg = TSL
        if term == "r3f" and form.lower().startswith("@react-three/"):
            pkg = package_of_specifier(form) or pkg
        if pkg:
            out.append((pkg, form, e.get("version")))
    named = set(term_pkg.values())
    for pk in held():
        if pk in named or pk in PLATFORM_PACKAGES or \
                "/" not in pk and "-" not in pk:
            # a one-word npm name (math, postprocessing, three) is an
            # ordinary word in prose; it counts as an import or a pin
            continue
        if re.search(r"(?<![\w/@-])" + re.escape(pk) + r"(?![\w/-])", prose):
            out.append((pk, pk, None))
    return out
# IMPLIES at package level (skill_select.IMPLIES: "@react-three/fiber 10
# readme: WebGPU support is first class ... ThreeJS WebGPU features/Nodes";
# the operator's own example, "r3f v10 -> TSL").
PACKAGE_IMPLIES = ({"when": "@react-three/fiber", "min_version": 10,
                    "then": TSL,
                    "why": "R3F v10: WebGPU first class, materials are "
                           "three.js TSL node graphs (skill_select.IMPLIES)"},)

_IMPORT = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(?\s*|\brequire\s*\(\s*)['"]"""
    r"""([^'"\s]+)['"]""")
_PIN_JSON = re.compile(r'"(@?[a-z0-9][\w.-]*(?:/[\w.-]+)?)"\s*:\s*"'
                       r'(?:[~^=<>]*\s*|npm:|workspace:)?v?(\d[\w.+-]*)"')
_PIN_AT = re.compile(r"(?<![\w/@.-])(@?[a-z0-9][\w.-]*(?:/[\w.-]+)?)@"
                     r"(\d+(?:\.\d+)*(?:-[\w.]+)?)")
_INSTALL = re.compile(r"\b(?:npm\s+(?:i|install|add)|pnpm\s+(?:add|i|install)"
                      r"|yarn\s+add|bun\s+add)\s+([^\n;&|]+)")
# A call: the name then `(` with no space ("score (top left)" in prose is a
# parenthesis, not a call).
_CALL = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)\(")
# Reference positions in code besides calls, tags and members.
_NEW_EXTENDS = re.compile(r"\b(?:new|extends|implements)\s+([A-Za-z_$][\w$]*)")
_IMPORT_BRACES = re.compile(r"\bimport\s*(?:type\s*)?\{([^}]*)\}")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_QUOTED = re.compile(r"'([^'\n]*)'|\"([^\"\n]*)\"|`([^`\n]*)`")
_JSX = re.compile(r"<([A-Z][\w$]*)")
_MEMBER_REF = re.compile(r"\.([A-Za-z_$][\w$]*)")
# A one-line string literal in code.
_STRING = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"")
# `.word` ending a word: a file extension when mimetypes knows it.
_FILE_EXT_RX = re.compile(r"\.([A-Za-z0-9]+)(?![\w$(])")


def package_of_specifier(spec: str) -> str | None:
    """The held package an import specifier names ('three/tsl' is TSL,
    'three/addons/...' is three, '@react-three/fiber/webgpu' is fiber)."""
    s = spec.strip()
    if s in ("three/tsl", "three/webgpu") or s.startswith(("three/tsl/",
                                                           "three/webgpu/")):
        return TSL
    pk = held()
    parts = s.split("/")
    base = "/".join(parts[:2]) if s.startswith("@") else parts[0]
    if base.startswith("@types/"):
        base = base[len("@types/"):]
    if base in pk and base not in PLATFORM_PACKAGES:
        return base
    return None


def _subpath_version(pkg: str, spec: str) -> str | None:
    """The held version whose files have the specifier's subpath
    (@react-three/fiber/webgpu exists only in v10's tree)."""
    sub = spec[len(pkg):].strip("/")
    if not sub:
        return None
    first = sub.split("/")[0]
    have = []
    for d in held().get(pkg, []):
        if os.path.exists(os.path.join(d, first)) or os.path.exists(
                os.path.join(d, "dist", first)):
            have.append(package_of_dir(d)[1])
    return have[0] if len(have) == 1 else None


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(str(p.get("text") or "") for p in c
                         if isinstance(p, dict))
    return str(c or "")


def evidence_pieces(messages: list[dict]) -> list[tuple[str, str]]:
    """[(where, text)] of the turn's NEWEST evidence: `user` (the user's
    prose), `code` (fences, a tool call's arguments), `tool` (tool results),
    `error` (error lines), `config` (a manifest or config file read or
    written: its pins and imports count, its keys are no symbols -- a
    tsconfig's `moduleResolution` is not type-fest's). Our own craft block
    is stripped and a dependency's own files set aside
    (skill_select.evidence_view; with `config`, what it sets aside as a
    config file's content is what is marked `config` here)."""
    import skill_classify
    import skill_select
    msgs = [m for m in messages or [] if isinstance(m, dict)]
    fresh = skill_select.fresh_messages(msgs)
    view, _info = skill_select.evidence_view(fresh)
    cfg, _cinfo = skill_select.evidence_view(fresh, config=True)
    pairs = skill_select._pair_results(view)
    out: list[tuple[str, str]] = []
    for i, (m, mc) in enumerate(zip(view, cfg)):
        role = m.get("role")
        t = _text(m)
        is_cfg = role == "tool" and t.strip() and not _text(mc).strip()
        is_read = role == "tool" and _is_read(pairs.get(i))
        for f in _FENCE.findall(t):
            out.append(("config" if is_cfg else "code", f.strip("`")))
        if role == "user":
            out.append(("user", t))
        elif role == "tool":
            # A file the agent READ is code: its comments are no evidence
            # and a word-shaped name counts only in reference position.
            out.append(("config" if is_cfg else "code" if is_read
                        else "tool", t))
        calls_c = mc.get("tool_calls") or []
        for k, c in enumerate(m.get("tool_calls") or []):
            a = str(((c or {}).get("function") or {}).get("arguments") or "")
            ac = str((((calls_c[k] if k < len(calls_c) else c) or {}).get(
                "function") or {}).get("arguments") or "")
            if a:
                try:
                    obj = json.loads(a)
                    a = "\n".join(str(v) for v in (obj.values() if
                                                   isinstance(obj, dict)
                                                   else [obj]))
                except ValueError:
                    pass
                out.append(("config" if ac == "{}" and a.strip() else "code",
                            a))
        if role in ("user", "tool"):
            errs = skill_classify._error_lines(t)
            if errs:
                out.append(("error", "\n".join(errs)))
    return out


def _is_read(call: dict | None) -> bool:
    if not call:
        return False
    try:
        import tool_code
        return tool_code.read_target(call) is not None
    except Exception:                                            # noqa: BLE001
        return False


# A name the PROJECT declares: a function, class, type, interface, enum or
# variable declared in code of this conversation (written, read, pasted).
_DECLARES = re.compile(r"\b(?:function\*?|class|interface|type|enum|const|"
                       r"let|var)\s+([A-Za-z_$][\w$]*)")


def declared_names(messages: list[dict]) -> set[str]:
    """Every name the conversation's own code declares (the client's
    messages: tool calls' written code, file reads, pasted fences). Such a
    name is the project's, not a package's, even when a held package also
    exports it (the agent's own `rand()`, its own `View` trait)."""
    out: set[str] = set()
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        texts = [_text(m)]
        for c in m.get("tool_calls") or []:
            texts.append(str(((c or {}).get("function") or {}).get(
                "arguments") or ""))
        for t in texts:
            if "function" in t or "const" in t or "class" in t or \
                    "let" in t or "var" in t or "type" in t:
                out |= set(_DECLARES.findall(t.replace("\\n", "\n")))
    return out


# The names an import binds, and the module each comes from:
#   import { a, b as c } from 'm'      a, c        (named; `b as c` binds c)
#   import d, { e } from 'm'           d, e        (default + named)
#   import * as ns from 'm'            ns          (namespace)
#   const { f, g: h } = require('m')   f, h
#   const k = require('m')             k
_IMPORT_STMT = re.compile(
    r"""\bimport\s+(?:type\s+)?([^'";]*?)\s+from\s*['"]([^'"\s]+)['"]""",
    re.S)
_REQUIRE_STMT = re.compile(
    r"""\b(?:const|let|var)\s+(\{[^}]*\}|[A-Za-z_$][\w$]*)\s*=\s*"""
    r"""require\s*\(\s*['"]([^'"\s]+)['"]\s*\)""")


def _bound(clause: str) -> tuple[list[str], list[str]]:
    """(named or default names, namespace names) an import clause binds."""
    named, spaces = [], []
    clause = clause.strip()
    m = re.search(r"\*\s*as\s+([A-Za-z_$][\w$]*)", clause)
    if m:
        spaces.append(m.group(1))
    braces = re.search(r"\{([^}]*)\}", clause)
    if braces:
        for part in braces.group(1).split(","):
            part = re.sub(r"^\s*type\s+", "", part.strip())
            if not part:
                continue
            a, _, b = part.partition(" as ")
            if ":" in a and not b:                       # require { g: h }
                a, _, b = a.partition(":")
            name = (b or a).strip()
            if _ID.fullmatch(name):
                named.append(name)
    head = re.sub(r"\{[^}]*\}|\*\s*as\s+[A-Za-z_$][\w$]*", "",
                  clause).strip(" ,")
    if head and _ID.fullmatch(head):
        spaces.append(head)          # a default import: used as `d.x` too
        named.append(head)
    return named, spaces


def imported_names(messages: list[dict]) -> dict[str, str]:
    """{name: module} for every name the conversation's code imports
    (written, read or pasted), and for every `ns.member` used through a
    namespace or default import, the member too. A name bound by an import
    of module X belongs to X wherever it is used; it never counts for
    another package (`import { createWorld } from "bitecs"` is bitecs's
    createWorld, not koota's)."""
    out: dict[str, str] = {}
    texts = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        texts.append(_text(m))
        for c in m.get("tool_calls") or []:
            texts.append(str(((c or {}).get("function") or {}).get(
                "arguments") or ""))
    spaces: dict[str, str] = {}
    for t in texts:
        if "import" not in t and "require" not in t:
            continue
        t = t.replace("\\n", "\n").replace('\\"', '"')
        for clause, mod in _IMPORT_STMT.findall(t):
            named, ns = _bound(clause)
            for n in named:
                out.setdefault(n, mod)
            for n in ns:
                spaces.setdefault(n, mod)
        for clause, mod in _REQUIRE_STMT.findall(t):
            named, ns = _bound(clause if clause.startswith("{")
                               else clause)
            for n in named:
                out.setdefault(n, mod)
            if not clause.startswith("{"):
                spaces.setdefault(clause, mod)
    if spaces:
        rx = re.compile(r"(?<![\w$.])(" + "|".join(
            re.escape(n) for n in spaces) + r")\.([A-Za-z_$][\w$]*)")
        for t in texts:
            for ns, member in rx.findall(t):
                out.setdefault(member, spaces[ns])
    return out


def _symbols_in(where: str, text: str) -> list[tuple[str, bool]]:
    """Candidate symbols, each with whether its context is CODE-LIKE (a
    fence, written code, an error line, a backtick span, a call `name(`, a
    JSX tag `<Name`, a member `.name`) or prose (a code-shaped name standing
    in prose). In code, a string literal's contents are not references (an
    import path, a message) and `.ext` at the end of a word is a file
    extension."""
    return [(n, True) for n in _code_like(where, text)] + [
        (n, False) for n in (_CODE_SHAPED.findall(text)
                             if where not in ("code", "error", "config")
                             else [])]


def _word_shaped(n: str) -> bool:
    return n.isalpha() and (n.islower() or (n[:1].isupper()
                                            and n[1:].islower()))


def _references(text: str) -> set[str]:
    """The names in REFERENCE position: a call, a JSX tag, a member (not a
    file extension), `new` / `extends` / `implements`, an import's braces."""
    got = set(_CALL.findall(text)) | set(_JSX.findall(text)) | set(
        _NEW_EXTENDS.findall(text))
    for body in _IMPORT_BRACES.findall(text):
        got |= {p.split(" as ")[0].strip() for p in body.split(",")}
    for m in _MEMBER_REF.finditer(text):
        end = m.end()
        if not (m.group(1).lower() in _extensions()
                and (end >= len(text) or not text[end].isalnum())):
            got.add(m.group(1))
    return got


def _code_like(where: str, text: str) -> list[str]:
    """In code (a fence, written code): every name -- comments and string
    literals aside -- but a WORD-SHAPED one (`tree`, `inner`, `Canvas`)
    only in reference position (`<Canvas`, `Fn(`, `.name`, an import): a
    local variable named `tree` is no use of a package. In an error line:
    a word-shaped name only quoted or in reference position (the message
    around it is prose: "... type declarations")."""
    if where == "config":
        return []
    if where in ("code", "error"):
        refs: set[str] = set()
        if where == "code":
            text = _COMMENT.sub(" ", _STRING.sub(" ", text))
        else:
            for m in _QUOTED.finditer(text):
                refs |= set(_ID.findall(next(g for g in m.groups()
                                             if g is not None)))
        text = _FILE_EXT_RX.sub(lambda m: m.group(0)
                                if m.group(1).lower() not in _extensions()
                                else " ", text)
        refs |= _references(text)
        return [n for n in _ID.findall(text)
                if not _word_shaped(n) or n in refs]
    got = []
    for span in _TICK.findall(text):
        got += _ID.findall(span)
    got += _CALL.findall(text) + _JSX.findall(text)
    # `.name` is a member -- unless it ends a word as a file extension does
    # (`Chart.js`, `scene.glb`): the extensions Python's mimetypes knows.
    for m in _MEMBER_REF.finditer(text):
        name = m.group(1)
        end = m.end()
        if name.lower() in _extensions() and (end >= len(text)
                                              or not text[end].isalnum()):
            continue
        got.append(name)
    return got


_EXT: set[str] | None = None


def _extensions() -> set[str]:
    global _EXT
    if _EXT is None:
        import mimetypes
        mimetypes.init()
        _EXT = {k.lstrip(".").lower() for k in mimetypes.types_map} | {
            k.lstrip(".").lower() for k in mimetypes.common_types}
    return _EXT


def detect(messages: list[dict]) -> dict[str, dict]:
    """{package: {"why": [{how, what, where}], "version": str | None,
    "events": [ASKED | ERROR]}} for every held package in play at this
    turn."""
    v = vocabulary()
    vocab, code_vocab = v["symbols"], v["code_symbols"]
    pieces = evidence_pieces(messages)
    declared = declared_names(messages)
    imported = imported_names(messages)
    found: dict[str, dict] = {}

    def add(pkg, how, what, where, version=None, weak=False):
        e = found.setdefault(pkg, {"why": [], "version": None,
                                   "events": set()})
        key = (how, what)
        if key not in {(w["how"], w["what"]) for w in e["why"]}:
            e["why"].append({"how": how, "what": what, "where": where,
                             "strength": "weak" if weak else "strong"})
        if version and not e["version"]:
            e["version"] = version
        if how == "name":
            e["events"].add("ASKED")
        if where == "error":
            e["events"].add("ERROR")

    for where, text in pieces:
        if where == "user":
            for pkg, form, ver in names_in(text):
                add(pkg, "name", form, where, ver)
        for spec in _IMPORT.findall(text):
            pkg = package_of_specifier(spec)
            if pkg:
                add(pkg, "import", spec, where,
                    _subpath_version(pkg, spec) if pkg != TSL else None)
        for rx in (_PIN_JSON, _PIN_AT):
            for name, ver in rx.findall(text):
                pkg = package_of_specifier(name)
                if pkg:
                    add(pkg, "pin", f"{name}@{ver}", where, ver)
        for line in _INSTALL.findall(text):
            for tok in line.split():
                name = tok.rsplit("@", 1)[0] if tok.count("@") > (
                    1 if tok.startswith("@") else 0) else tok
                pkg = package_of_specifier(name)
                if pkg:
                    add(pkg, "pin", tok, where)
        for sym, code_like in _symbols_in(where, text):
            if sym in declared:
                continue
            if sym in imported:
                # The import's own module decides: a name bound by an
                # import of another module (held or not) is that module's.
                own = package_of_specifier(imported[sym])
                if own is None or own != (code_vocab.get(sym)
                                          or vocab.get(sym)):
                    continue
            pkg = (code_vocab if code_like else vocab).get(sym)
            if pkg:
                # A symbol read in prose (a code-shaped name standing in a
                # sentence, or in a tool's plain output) is WEAK: the
                # decider confirms it. In code, an error line or a
                # backticked span it is strong.
                add(pkg, "symbol", sym, where,
                    weak=not code_like or where == "tool")
    # The implication table, at package level.
    for row in PACKAGE_IMPLIES:
        e = found.get(row["when"])
        if not e:
            continue
        v = e.get("version") or ""
        try:
            major = int(re.match(r"\d+", v).group(0)) if v else None
        except AttributeError:
            major = None
        if major is not None and major >= row["min_version"]:
            add(row["then"], "implied", f"{row['when']} v{major}",
                "implied")
    for e in found.values():
        e["events"] = sorted(e["events"])
        e["strength"] = "strong" if any(w.get("strength") != "weak"
                                        for w in e["why"]) else "weak"
    return found


# ---------------------------------------------------------------------------
# 3. Package -> skill, and the injection rule.
# ---------------------------------------------------------------------------
# PROPOSED for the operator (2026-09-27). The upstream authors' own SKILL.md
# where it was ingested (skills/ingested/pmndrs/spec.json: koota's and
# math's SKILL.md were DECOMPOSED into atomic skills; the canonical one is
# the skill of the SKILL.md's first section); else the best existing skill
# for using the package, by its description. A package with no row has no
# canonical skill in the store (listed by mapping_report()). The rows live
# in the package registry (package_registry.SEED's `canonical`, and an
# onboarded package's), read live.
CANONICAL = package_registry.live("canonical")


# The selector's AREAS whose skills are PACKAGE skills: they reach a turn
# only through this rule (skill_select.decide), never through the asked
# slots, the implication table or the evidence rows. skill_classify's terms
# that name a held package (TERM_PACKAGE), live.
PACKAGE_AREAS = package_registry.LiveKeys("term_package",
                                          package_registry._term_package)

_LEADS: dict = {"key": None, "map": {}}


def lead_skills(pool: list[dict]) -> dict[str, list[str]]:
    """{package: [skill names]} for the armed skills whose SKILL.md says
    `metadata.yamadori.lead_for: <package>` (a string or a list): the
    package's LEAD skill, written for it by the skill pipeline. Parsed once
    per armed set."""
    key = tuple(sorted((x.get("id"), x.get("version")) for x in pool))
    if _LEADS["key"] == key:
        return _LEADS["map"]
    import skill_md
    out: dict[str, list[str]] = {}
    for x in pool:
        lead = x.get("lead_for")
        if lead is None:
            # (a row from before skills.armed carried `lead_for`)
            text = x.get("text") or ""
            if "lead_for" not in text:
                continue
            try:
                ours = (skill_md.parse(text) or {}).get("yamadori") or {}
            except Exception:                                    # noqa: BLE001
                continue
            lead = ours.get("lead_for")
        for pk in ([lead] if isinstance(lead, str) else lead or []):
            if isinstance(pk, str) and pk.strip():
                out.setdefault(pk.strip(), []).append(x["name"])
    for v in out.values():
        v.sort()
    _LEADS.update(key=key, map=out)
    return out


def canonical_skill(pkg: str, version: str | None = None,
                    pool: list[dict] | None = None) -> str | None:
    """The package's skill: its LEAD skill in the armed set (lead_skills;
    one per package, the first by name if several claim it), else the
    CANONICAL mapping (by major version where it has one)."""
    if pool is not None:
        leads = lead_skills(pool).get(pkg)
        if leads:
            return leads[0]
    row = CANONICAL.get(pkg)
    if not row:
        return None
    major = (re.match(r"\d+", version or "") or [None])[0]
    return (row.get("by_major") or {}).get(major) or row["skill"]


def new_state() -> dict:
    return {"v": 1, "given": {}, "req": 0}


def decide(in_play: dict[str, dict], state: dict | None, *,
           phase_changed: bool = False, pool: list[dict] | None = None
           ) -> tuple[list[dict], dict]:
    """(decisions, new state). Pure. `in_play` is detect()'s result; `pool`
    the armed skills (for a LEAD skill, and to drop a skill that is not
    armed). Each decision: {package, skill, form: body | recall | none,
    why, evidence}."""
    import skill_chart
    st = json.loads(json.dumps(state or new_state()))
    st["req"] = int(st.get("req") or 0) + 1
    out: list[dict] = []
    for pkg in sorted(in_play):
        e = in_play[pkg]
        skill = canonical_skill(pkg, e.get("version"), pool)
        because = [f"{w['how']} {w['what']}" for w in e["why"]][:4]
        if not skill:
            out.append({"package": pkg, "skill": None, "form": "none",
                        "why": "no canonical skill for this package",
                        "evidence": because})
            continue
        if pool is not None and skill not in {x.get("name") for x in pool}:
            out.append({"package": pkg, "skill": skill, "form": "none",
                        "why": "its skill is not armed", "evidence": because})
            continue
        g = st["given"].get(skill)
        if g is None:
            st["given"][skill] = {"package": pkg, "req": st["req"]}
            out.append({"package": pkg, "skill": skill, "form": "body",
                        "why": "in play; its skill was not given in this "
                               "conversation", "evidence": because})
            continue
        events = list(e.get("events") or []) + (["PHASE"] if phase_changed
                                                 else [])
        ev = next((x for x in ("ASKED", "ERROR", "PHASE") if x in events
                   and skill_chart.transition("given", x) is not None), None)
        if ev:
            g["recalled"] = st["req"]
            out.append({"package": pkg, "skill": skill, "form": "recall",
                        "event": ev,
                        "why": f"given before; {ev} brings it back "
                               "(given --" + ev + "--> recall_eligible)",
                        "evidence": because})
        else:
            out.append({"package": pkg, "skill": skill, "form": "none",
                        "why": "given before; no recall event this turn",
                        "evidence": because})
    return out, st


def mapping_report(pool: list[dict] | None = None) -> list[dict]:
    """Every held package with its canonical skill, the skill's body size
    in tokens (skill_limits.tokens) and whether it is over the body cap."""
    import skill_limits as L
    import skills
    pool = skills.armed() if pool is None else pool
    by = {s["name"]: s for s in pool}
    cap = getattr(L, "SKILL_TOKENS_HARD", None)
    rows = []
    for pkg in vocabulary()["packages"]:
        row = CANONICAL.get(pkg) or {}
        names = [row["skill"]] if row else []
        names += [n for n in (row.get("by_major") or {}).values()
                  if n not in names]
        for n in names or [None]:
            s = by.get(n) if n else None
            tok = L.tokens(s["body"]) if s else None
            rows.append({"package": pkg, "skill": n,
                         "armed": bool(s), "tokens": tok,
                         "over_cap": bool(cap and tok and tok > cap),
                         "why": row.get("why")})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("vocab", "detect", "mapping"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--samples", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "vocab":
        v = vocabulary()
        print(f"platform names {v['platform']}, prose words {v['english']}")
        by = defaultdict(list)
        for s, p in v["symbols"].items():
            by[p].append(s)
        for p in v["packages"]:
            c = v["per_package"][p]
            print(f"  {p:<30} names {c['names']:>6}  unique {c['unique']:>5}"
                  f"  shared {c['dropped_shared']:>5}  platform "
                  f"{c['dropped_platform']:>4}  english "
                  f"{c['dropped_english']:>4}")
        for p in ("@react-three/fiber", "koota", "math", TSL):
            got = sorted(by.get(p, []))
            step = max(1, len(got) // a.samples)
            print(f"\n  {p}: " + ", ".join(got[::step][:a.samples]))
        return 0
    if a.cmd == "mapping":
        for r in mapping_report():
            print(f"  {r['package']:<30} {str(r['skill']):<44} "
                  f"{'' if r['tokens'] is None else r['tokens']:>4} "
                  f"{'OVER CAP' if r['over_cap'] else ''}")
        return 0
    text = _read(a.path)
    got = detect([{"role": "user", "content": text}])
    print(json.dumps(got, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
