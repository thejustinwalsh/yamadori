#!/usr/bin/env python
"""When does a skill apply? One vocabulary, read from both sides.

THE RULE (operator, 2026-09-24)

Skills apply by ARTIFACT -- what is being created or edited -- and by broad
context. A skill's applies-when is a small structured rule:

    {"applies_to": {"artifacts": ["tests"], "languages": ["typescript"],
                    "frameworks": []},
     "domains": [...],
     "triggers": [{"text": "Use when writing Vitest tests", "quote": "...",
                   "origin": "description"}],
     "text": "tests (TypeScript)", "evidence": [...]}

The SAME tables classify a source when it is ingested and read a request when
it arrives, so the two sides cannot disagree about what "TypeScript" or
"slides" means.

ARTIFACTS

    code, tests, docs, slides, ui_design, data, config, prose

`code` is implied by any programming language or framework. The others are
named by nouns ("README", "slide deck", "SQL"), file paths and extensions
(`deck.pptx`, `schema.sql`, `Dockerfile`), and by an artifact VERB in front
of a noun ("write a README", "make slides").

TRIGGERS

A SKILL.md's frontmatter `description` IS its trigger text (Anthropic Agent
Skills: "what the skill does and when to use it", at most 1,024 characters).
So triggers come, with a verbatim quote, from the description, from the
lines under a "When to use" heading, and, failing both, from the rule itself.
They are what the request is EMBEDDED against (skill_select), never the
skill's body. A poisoned description can therefore buy a skill a place on
the CANDIDATE list, never an injection: an embedding match without
deterministic evidence goes to the confirming fallback (skill_select's
decision table).

SIGNAL STRENGTH

    fact    structure: a fence tag, an import, a file path or extension in
            the request or in a tool call, the router's code class
    phrase  an artifact verb in front of an artifact noun
    word    a bare name in prose

HOW A SOURCE IS CLASSIFIED

Deterministic, with a verbatim quote per term. A term needs `MIN_HITS` hits
(1 when declared by the corpus itself, as the migration does) and at least
`SHARE` of its kind's top count; at most `MAX_TERMS` per kind, so a rule
cannot be stuffed into matching everything. YAML frontmatter counts for
triggers only, not for terms.
"""
from __future__ import annotations

import collections
import functools
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import package_registry  # noqa: E402
import skill_limits as L  # noqa: E402

MIN_HITS = 2
SHARE = 0.25
MAX_TERMS = 2
STRENGTH = {"fact": 3, "phrase": 2, "word": 1}


@dataclass(frozen=True)
class Term:
    id: str
    name: str
    kind: str                       # language | framework
    words: str
    case: bool = False
    exts: tuple = ()
    fences: tuple = ()
    packages: tuple = ()
    domains: frozenset = field(default_factory=frozenset)


# THE HAND-WRITTEN TERMS. VOCAB (below) is these plus one framework term per
# package the registry holds that is not already one of these
# (package_registry; operator, 2026-09-27: "The taxonomy grows automatically
# from package names plus the aliases the operator types"), rebuilt by
# refresh() when the registry changes.
HAND_VOCAB: tuple[Term, ...] = (
    Term("typescript", "TypeScript", "language",
         r"\btypescript\b|\btsconfig(?:\.json)?\b",
         exts=("ts", "tsx", "mts", "cts"), fences=("ts", "typescript", "tsx"),
         packages=("typescript",), domains=frozenset({"types", "web-frontend"})),
    Term("javascript", "JavaScript", "language", r"\bjavascript\b|\becmascript\b",
         exts=("js", "mjs", "cjs", "jsx"),
         fences=("js", "javascript", "jsx", "mjs"),
         domains=frozenset({"web-frontend"})),
    Term("rust", "Rust", "language",
         r"\brust\b|\bcargo\.toml\b|\brustc\b|\brustup\b",
         exts=("rs",), fences=("rust", "rs"), domains=frozenset({"systems"})),
    # C is a letter: only phrases that cannot mean anything else, and
    # case-sensitive.
    Term("c", "C", "language",
         r"\bC (?:ABI|code|headers?|library|libraries|functions?|structs?"
         r"|compilers?|strings?|types?|programs?|APIs?|calling convention)\b"
         r"|\bin C\b|\bC(?:89|99|11|17|23)\b|\bANSI C\b",
         case=True, exts=("c", "h"), fences=("c", "h"),
         domains=frozenset({"systems"})),
    Term("cpp", "C++", "language", r"C\+\+", case=True,
         exts=("cpp", "cc", "cxx", "hpp", "hh"),
         fences=("cpp", "c++", "cxx", "hpp"), domains=frozenset({"systems"})),
    Term("zig", "Zig", "language", r"\bzig\b", exts=("zig",), fences=("zig",),
         domains=frozenset({"systems"})),
    Term("wgsl", "WGSL", "language", r"\bwgsl\b", exts=("wgsl",),
         fences=("wgsl",), domains=frozenset({"gpu"})),
    Term("glsl", "GLSL", "language", r"\bglsl\b", exts=("glsl", "vert", "frag"),
         fences=("glsl",), domains=frozenset({"gpu"})),
    Term("python", "Python", "language", r"\bpython\b", exts=("py",),
         fences=("python", "py"), domains=frozenset({"backend"})),
    Term("sql", "SQL", "language", r"\bsql\b", exts=("sql",),
         fences=("sql", "postgresql", "sqlite", "mysql"),
         domains=frozenset({"backend"})),
    Term("css", "CSS", "language", r"\bcss\b", exts=("css", "scss"),
         fences=("css", "scss"),
         domains=frozenset({"visual-design", "web-frontend"})),
    # A page: the entry point of every browser app (2026-09-26, the
    # "Browser app entry point" skill needs html AND javascript).
    Term("html", "HTML", "language",
         r"\bhtml\b|\bindex\.html?\b|<!doctype html",
         exts=("html", "htm"), fences=("html", "htm", "xhtml"),
         domains=frozenset({"web-frontend"})),
    Term("react", "React", "framework", r"\breact\b(?![- ]three)",
         packages=("react", "react-dom", "next"),
         domains=frozenset({"web-frontend", "ui-component"})),
    Term("r3f", "React Three Fiber", "framework",
         r"\breact[- ]three[- ]fiber\b|\br3f\b|@react-three/[\w.-]*",
         packages=("@react-three/fiber", "@react-three/drei",
                   "@react-three/postprocessing"),
         domains=frozenset({"gpu", "web-frontend"})),
    Term("threejs", "three.js", "framework",
         r"\bthree\.?js\b|\bthree js\b|\bTSL\b|\bthree/(?:webgpu|tsl)\b",
         packages=("three", "@types/three"),
         domains=frozenset({"gpu", "web-frontend"})),
    # pmndrs' ECS and math engine (operator, 2026-09-26: "we want a skill for
    # Koota", "pmndrs just released npm math, we want that too"). `math` is
    # a common word and Python's stdlib: its words are only the forms that
    # cannot mean anything else (the repo, a subpath, a pinned version), and
    # its import is a JS/TS one (discover judges stdlib names per grammar).
    Term("koota", "Koota", "framework", r"\bkoota\b", packages=("koota",),
         domains=frozenset({"web-frontend"})),
    # "pmndrs math" is how the operator names it in prose (2026-09-27:
    # "Build it with r3f (react-three-fiber) v10 and Koota and pmndrs math").
    Term("pmndrs_math", "math (pmndrs)", "framework",
         r"\bpmndrs/math\b|\bmath/(?:noise|random|time|shapes|geometry|color|ik)\b"
         r"|(?<![\w.])math@\d|\bnpm (?:package )?[`'\"]?math\b"
         r"|(?i:\bpmndrs(?:'s)?[ -]+math\b)",
         case=True, packages=("math",), domains=frozenset({"gpu", "web-frontend"})),
    Term("typegpu", "TypeGPU", "framework", r"\btypegpu\b|\btgpu\b",
         packages=("typegpu", "@typegpu/noise", "@typegpu/three"),
         domains=frozenset({"gpu"})),
    Term("webgpu", "WebGPU", "framework", r"\bwebgpu\b",
         packages=("@webgpu/types", "wgpu-matrix"), domains=frozenset({"gpu"})),
    Term("wasm_bindgen", "wasm-bindgen", "framework", r"\bwasm[-_]bindgen\b",
         packages=("wasm-bindgen", "wasm_bindgen"),
         domains=frozenset({"systems", "web-frontend"})),
    Term("emscripten", "Emscripten", "framework",
         r"\bemscripten\b|\bemcc\b|\bembind\b", domains=frozenset({"systems"})),
    Term("webassembly", "WebAssembly", "framework", r"\bwebassembly\b|\bwasm\b",
         exts=("wasm", "wat"), fences=("wat", "wasm"),
         domains=frozenset({"systems"})),
    Term("cbindgen", "cbindgen", "framework", r"\bcbindgen\b",
         domains=frozenset({"systems"})),
    Term("tailwind", "Tailwind CSS", "framework", r"\btailwind(?:css)?\b",
         packages=("tailwindcss",),
         domains=frozenset({"visual-design", "web-frontend"})),
    Term("vitest", "Vitest", "framework", r"\bvitest\b", packages=("vitest",),
         domains=frozenset({"tooling"})),
    Term("jest", "Jest", "framework", r"\bjest\b", packages=("jest",),
         domains=frozenset({"tooling"})),
    Term("pytest", "pytest", "framework", r"\bpytest\b", packages=("pytest",),
         domains=frozenset({"tooling"})),
)
VOCAB: tuple[Term, ...] = HAND_VOCAB


@dataclass(frozen=True)
class Artifact:
    id: str
    name: str
    nouns: str                      # regex over prose
    exts: tuple = ()
    paths: str = ""                 # regex over a path
    domains: frozenset = field(default_factory=frozenset)


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("code", "code",
             r"\b(?:functions?|methods?|class(?:es)?|modules?|components?"
             r"|programs?|scripts?|hooks?|structs?|interfaces?|endpoints?"
             r"|algorithms?|implementations?|refactor(?:ing)?|source code)\b"),
    Artifact("tests", "tests",
             r"\b(?:unit|integration|e2e|end-to-end|snapshot|regression"
             r"|property[- ]based)\s+tests?\b|\btest (?:suites?|cases?|files?"
             r"|coverage)\b|\btests?\b(?= for\b)|\bvitest\b|\bjest\b|\bpytest\b"
             r"|\bplaywright\b|\bcypress\b|\btesting[- ]library\b",
             paths=r"\.(?:test|spec)\.\w+$|_test\.\w+$|(?:^|/)test_\w+\.py$"
                   r"|(?:^|/)(?:__tests__|tests?)/",
             domains=frozenset({"tooling"})),
    Artifact("docs", "documentation",
             r"\breadme\b|\bdocumentation\b|\bdocstrings?\b|\bchangelog\b"
             r"|\bapi reference\b|\bmarkdown\b|\bjsdoc\b|\brustdoc\b"
             r"|\btechnical writing\b|\buser guide\b",
             exts=("md", "mdx", "rst", "adoc"),
             paths=r"(?:^|/)(?:README|CHANGELOG|CONTRIBUTING)(?:\.\w+)?$"
                   r"|(?:^|/)docs/"),
    Artifact("slides", "slides",
             r"\bslides?\b|\bslide ?decks?\b|\bdecks?\b(?! of cards)"
             r"|\bpresentations?\b|\bkeynote\b|\bpowerpoint\b|\bpptx?\b"
             r"|\breveal\.js\b",
             exts=("pptx", "ppt", "key", "odp")),
    Artifact("ui_design", "UI and visual design",
             # Not bare "layout": that is memory layout (Rust type layout, bind
             # group layouts) in this repo's own corpora.
             r"\bui\b|\buser interfaces?\b|\bvisual design\b"
             r"|\b(?:page|css|grid|flex(?:box)?|visual|ui|screen|responsive)"
             r" layouts?\b"
             r"|\bmockups?\b|\bwireframes?\b|\bfigma\b|\bdesign systems?\b"
             r"|\btypography\b|\bcolou?r palettes?\b|\bstyling\b"
             r"|\blanding pages?\b|\bdashboards?\b",
             exts=("css", "scss", "fig", "sketch"),
             domains=frozenset({"visual-design", "ui-component"})),
    Artifact("data", "data and SQL",
             r"\bsql\b|\bdatabase schemas?\b|\b(?:database|schema|sql) "
             r"migrations?\b|\bcsv\b|\bdataframes?\b|\bpandas\b|\betl\b"
             r"|\bpostgres(?:ql)?\b|\bsqlite\b|\bmysql\b|\bsql quer(?:y|ies)\b",
             exts=("sql", "csv", "parquet", "tsv"),
             domains=frozenset({"backend"})),
    Artifact("config", "configuration and infrastructure",
             r"\bdockerfile\b|\bdocker[- ]compose\b|\bkubernetes\b|\bk8s\b"
             r"|\bhelm charts?\b|\bterraform\b|\bci(?:/cd)? (?:pipelines?"
             r"|config)\b|\bgithub actions\b|\bnginx\b|\bansible\b"
             r"|\bconfig(?:uration)? files?\b",
             exts=("yaml", "yml", "toml", "ini", "tf", "hcl", "conf"),
             paths=r"(?:^|/)Dockerfile$|docker-compose|\.github/workflows/",
             domains=frozenset({"tooling"})),
    Artifact("prose", "prose writing",
             r"\bessays?\b|\bblog posts?\b|\barticles?\b|\bemails?\b"
             r"|\bcover letters?\b|\bnewsletters?\b|\bpress releases?\b"
             r"|\bshort stor(?:y|ies)\b|\bcopywriting\b|\bmarketing copy\b"),
)

# ---------------------------------------------------------------------------
# THE TAXONOMY (operator, 2026-09-26: "well tagged, categorized and compact").
#
# Every skill is filed on four fixed axes, and the same tables read a
# request, so the two sides cannot disagree:
#
#   artifact    what is made or edited            ARTIFACTS above
#   language    the language it is written in     VOCAB, kind language
#   framework   the library or tool it uses       VOCAB, kind framework
#   phase       where in the work the model is    PHASES below
#
# plus two finer gates a rule MAY carry:
#
#   situations  facts about the conversation (SITUATIONS below): an error
#               in the output, a path:line reference, several source files
#   topics      API names and terms (`useActionState`, `repr(C)`, "prefix
#               sums"): a code-shaped topic is a FACT when it appears; a
#               plain word needs the rule's primary key as well
#
# A rule with no phases applies in every phase; one with no topics or
# situations is gated by its artifact / language / framework alone.
# ---------------------------------------------------------------------------
PHASES = ("plan", "implement", "debug", "verify", "refactor", "review")

# Request side: which phase the work is in. Flat text (PROTOCOL rule 8
# allows keyword tests for this kind of label); several may hold at once.
_PHASE_RX = {
    "plan": re.compile(
        r"\b(?:plan(?:ning)?|design (?:the|a|an)|architect(?:ure)?|approach"
        r"|how should (?:i|we)|outline|break (?:it|this) down|spec(?:ification)?"
        r"|roadmap)\b", re.I),
    # A FAILURE REPORT, not a word: a spec that says "no console errors" or
    # "the tests must not fail" is a build task (replay of the Octopus V0
    # spec, 2026-09-26). So bare "error"/"fail" do not count; an error's
    # own text, a report phrase or an explicit ask to fix does.
    "debug": re.compile(
        r"(?:\b\w*Error\b:|\bexception\b:|\btraceback\b|\bstack ?trace\b"
        r"|\bfails? with\b|\bfailed with\b|\bis failing\b|\bbroken\b"
        r"|\bbugs?\b|\bcrash(?:es|ed|ing)?\b|\b(?:does ?n[o']t|did ?n[o']t"
        r"|not) work(?:ing)?\b|\bblank (?:page|screen|canvas)\b|\bnothing "
        r"(?:is )?(?:drawn|renders?|happens|shows|moves)\b|\bis not a "
        r"function\b|\bis not defined\b|\bcannot read propert|\bundefined is "
        r"not\b|\bstill broken\b|\bsame error\b|\bregression\b|\bwrong "
        r"(?:output|result)\b|\bfix (?:the|this|these|it|them)\b|\bdebug"
        # a compiler's coded error pasted into the turn (tsc, C#, rustc)
        r"|\berror\s+(?-i:[A-Z]{1,3})\d{3,5}\s*:|\berror\[E\d{3,5}\]:)",
        re.I),
    "verify": re.compile(
        r"\b(?:tests?|testing|verify|verif(?:y|ied|ication)|check (?:that"
        r"|whether|if)|make sure|assert(?:ion)?s?|playwright|puppeteer"
        r"|headless|smoke[- ]test|reproduce|validate)\b", re.I),
    "refactor": re.compile(
        r"\b(?:refactor(?:ing)?|clean(?:ing)? ?up|rename|restructur(?:e|ing)"
        r"|simplif(?:y|ying)|extract (?:a |the )?(?:function|module|component"
        r"|method|class)|split (?:the |this )?(?:file|module))\b", re.I),
    "review": re.compile(
        r"\b(?:review|audit|critique|code review|look over)\b", re.I),
}

# Request side: facts about the conversation that a situation gate reads.
SITUATIONS = ("error_output", "call_mismatch", "file_line", "multi_file",
              "screenshot_made")
# A screenshot or image file the conversation's tool calls made or named.
_SHOT = re.compile(r"\bscreenshot\b|[\w./-]+\.(?:png|jpe?g|webp)\b", re.I)
# A compiler's CODED error: tsc `error TS2339:`, C# `error CS0103:`, MSVC
# `error C2065:`, rustc `error[E0425]:` (2026-09-27: a tsc failure did not
# put the request in the debug phase -- `error:` needs the colon right after
# the word).
_CODED_ERROR = (r"\berror\s+(?-i:[A-Z]{1,3})\d{3,5}\s*:"
                r"|\berror\[E\d{3,5}\]:")
_ERROR_OUT = re.compile(
    r"(?:\b\w*Error\b:|\bTraceback \(most recent call last\)|\bexit (?:code|"
    r"status)[: ]+[1-9]|(?-i:\bFAIL(?:ED)?\b)|\bpanicked at\b"
    r"|\berror(?:\[E\d+\])?:|" + _CODED_ERROR +
    # ESLint's stylish lines (`  14:5  error  'x' is not defined  no-undef`)
    # and its summary (`3 problems (3 errors, 0 warnings)`); esbuild / Vite
    # (`[ERROR] Could not resolve "x"`).
    r"|(?m:^\s*\d+:\d+\s+error\s{2,}\S)|\b\d+ problems? \(\d+ errors?\b"
    r"|(?-i:\[ERROR\])"
    r"|\bUncaught\b|\bis not a function\b|\bis not defined\b|\bcannot read "
    r"propert)", re.I)
_CALL_MISMATCH = re.compile(
    r"\bis not a function\b|\bis not defined\b|\bhas no exported member\b"
    r"|\bdoes not provide an export named\b|\bis not exported\b"
    r"|\bundefined is not an? (?:function|object)\b|\bcannot read "
    r"propert(?:y|ies) of undefined\b|\bno attribute\b|\bnot callable\b",
    re.I)
_FILE_LINE = re.compile(
    r"\b[\w./-]+\.(?:js|mjs|cjs|jsx|ts|tsx|py|rs|c|h|cpp|go|java|rb|php|css"
    r"|html)[:(](\d+)\b|\bline \d+ (?:of|in) [\w./-]+\.\w{1,5}\b", re.I)
_SOURCE_EXT = ("js", "mjs", "cjs", "jsx", "ts", "tsx", "py", "rs", "c", "h",
               "cpp", "go", "java", "rb", "php", "zig", "wgsl", "glsl")
_SOURCE_PATH = re.compile(
    r"(?<![\w/.-])((?:[\w-]+/)*[\w-]+\.(?:" + "|".join(_SOURCE_EXT)
    + r"))\b")

# A topic that looks like code: camelCase, snake_case, dotted, a call, a
# decorator or scoped package, a macro. Such a string does not appear in
# prose by accident, so its presence is a fact.
_CODE_SHAPED = re.compile(
    r"^(?:@[\w-]+/[\w.-]+|[A-Za-z_$][\w$]*(?:[A-Z_.$][\w$]*|\(\)?|!|<[^>]*>)"
    r"[\w$.()<>!]*|[a-z]+[A-Z]\w*)$")


# Latin abbreviations are prose even though `e.g` has a dot (replay of the
# V4 stack, 2026-09-26: the migrated topic "e.g" was a FACT in every spec).
_PROSE_ABBREV = {"e.g", "i.e", "etc", "vs", "cf", "n.b", "a.k.a"}


_PACKAGE_NAME = re.compile(r"^@[\w-]+/[\w.-]+$")
# So is a project file's name (`tsconfig.json`, `package.json`): a spec that
# lists its files names them whatever it is about (the same replay: a
# TypeScript skill about synthetic events on the V4 spec's file list).
_FILE_NAME = re.compile(r"^[\w.-]+\.(?:json|jsonc|ts|tsx|js|mjs|cjs|md|toml"
                        r"|ya?ml|html?|css|lock)$", re.I)


# PLATFORM AND CORE APIS (2026-09-27, the daily-work selection audit): a
# topic every web or React program touches -- `localStorage`, `fetch`,
# `useEffect`, `setTimeout` -- says nothing about a skill's SUBJECT. "Add a
# dark-mode toggle ... remembers the choice in localStorage" injected an
# auth-token skill (topics localStorage, sessionStorage) as a FACT. Such a
# topic counts as a PLAIN word in match(): it confirms the primary key and
# needs a second topic, never a fact on its own. The list is the web
# platform's globals, the language's core objects and React's built-in
# hooks, plus the topics the armed store files under three or more areas
# (useEffect, useState, useRef, startTransition, sRGB). KEPT FOR THE
# OPERATOR (docs/CONSTANTS-AUDIT.md "COMMON_API", 2026-09-27): without it
# "remembers the choice in localStorage" makes an auth-token craft a FACT
# again (mcp/test_skill_turns.py [precision]).
COMMON_API = frozenset("""
localStorage sessionStorage fetch setTimeout setInterval clearTimeout
clearInterval requestAnimationFrame cancelAnimationFrame addEventListener
removeEventListener dispatchEvent console.log console.error console.warn
JSON.parse JSON.stringify Promise.all Promise.resolve Promise.race
Object.keys Object.values Object.entries Object.assign Object.freeze
Array.from Array.isArray Math.PI Math.random Math.floor Math.ceil Math.min
Math.max Math.abs Math.sqrt Number.isFinite document.querySelector
document.getElementById document.createElement document.body window.location
window.addEventListener navigator.userAgent structuredClone queueMicrotask
toString valueOf Float32Array Float64Array Uint8Array Uint32Array Int32Array
ArrayBuffer useState useEffect useRef useMemo useCallback useContext
useReducer useLayoutEffect useId startTransition sRGB
""".split())


def strong_topic(topic: str, rule: dict | None = None) -> bool:
    """A code-shaped topic that is evidence about a skill's SUBJECT: not a
    platform or core API every program uses (COMMON_API) -- unless the
    rule's primary framework is a SPECIFIC one (koota, R3F, pmndrs math):
    inside a narrow framework, useEffect is the subject (spawning koota
    entities on mount), where in React or TypeScript it is in every file."""
    t = (topic or "").strip()
    if not code_shaped(t):
        return False
    if t.rstrip("()") not in COMMON_API:
        return True
    fw = applies_to(rule)["frameworks"] if rule else []
    return bool(fw) and fw[0] not in HOST_FRAMEWORKS


# Frameworks that HOST others (half the store names them). KEPT FOR THE
# OPERATOR (docs/CONSTANTS-AUDIT.md "HOST_FRAMEWORKS", 2026-09-27): without
# it "React" in the user's prose becomes an asked FACT, which moves every
# React mention from the ask path (the last resort confirms) to the fact
# path (the asked slot's relevance floor decides).
HOST_FRAMEWORKS = frozenset({"react"})




def code_shaped(topic: str) -> bool:
    t = (topic or "").strip()
    if t.lower().rstrip(".") in _PROSE_ABBREV:
        return False
    # A scoped package name (`@types/react`) is in every manifest and pin
    # list of its projects: the term it maps to is the evidence, and the
    # topic only confirms (replay of the V4 stack, 2026-09-26: a React 19
    # test-migration skill became a FACT on a spec's pinned @types/react).
    if _PACKAGE_NAME.match(t) or _FILE_NAME.match(t):
        return False
    return bool(t) and bool(_CODE_SHAPED.match(t)) and not t.isupper() \
        and len(t) >= 3


@functools.lru_cache(maxsize=8192)
def _topic_rx(topic: str):
    t = topic.strip()
    if code_shaped(t):
        core = re.escape(t.rstrip("()"))
        if t.endswith(("()", "(")):
            # A call (`select()`, `Fn()`, `color()`) is a call in the text:
            # the bare word `select` is not (replay of the V4 stack,
            # 2026-09-26: a TSL skill matched "no-select" in a CSS rule).
            return re.compile(r"(?<![\w$-])" + core + r"\s*\(")
        return re.compile(r"(?<![\w$])" + core + r"(?![\w$])")
    return re.compile(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", re.I)


VERBS = (r"\b(?:write|writing|create|creating|generate|make|making|draft"
         r"|drafting|build|building|add|update|edit|editing|fix|refactor"
         r"|rewrite|review|improve|design|designing|produce|author|convert"
         r"|format|restyle|document)\b")

ART_BY_ID = {a.id: a for a in ARTIFACTS}
_NOUNS = {a.id: re.compile(a.nouns, re.I) for a in ARTIFACTS}
_PHRASE = {a.id: re.compile(VERBS + r"(?:\W+\w+){0,5}?\W+(?:" + a.nouns + ")",
                            re.I) for a in ARTIFACTS}
_APATH = {a.id: re.compile(a.paths) for a in ARTIFACTS if a.paths}
_EXT_ART = {e: a.id for a in ARTIFACTS for e in a.exts}
_EQ_CACHE: "dict[str, tuple[str, dict]]" = {}


def _registry_terms() -> tuple[Term, ...]:
    """One framework Term per registry package that is not already filed
    under a hand-written term: its id the package's term, its WORD rule the
    one stored with it (package_registry.words_rule: the exact npm name and
    the operator's typed aliases), its package the npm name, its domains
    the entry's (none unless it says). Packages under one new term share
    it."""
    hand = {t.id for t in HAND_VOCAB}
    hand_pkgs = {p for t in HAND_VOCAB for p in t.packages}
    rows: dict[str, dict] = {}
    for name, e in package_registry.load().items():
        term = e.get("term")
        if e.get("seed") or not term or term in hand or name in hand_pkgs:
            continue
        rule = e.get("words") or package_registry.words_rule(
            name, e.get("aliases") or ())
        try:
            rx = str(rule["regex"])
            case = bool(rule.get("case"))
            re.compile(rx, 0 if case else re.I)
        except (re.error, KeyError, TypeError):
            continue
        r = rows.setdefault(term, {"name": e.get("label") or name,
                                   "words": [], "case": case,
                                   "packages": [], "domains": set()})
        part = rx if case == r["case"] else (
            "(?i:" + rx + ")" if r["case"] else "(?-i:" + rx + ")")
        if part not in r["words"]:
            r["words"].append(part)
        r["packages"].append(name)
        r["domains"] |= set(e.get("domains") or ())
    return tuple(Term(tid, r["name"], "framework", "|".join(r["words"]),
                      case=r["case"], packages=tuple(r["packages"]),
                      domains=frozenset(r["domains"]))
                 for tid, r in rows.items())


def _rebuild(vocab: tuple) -> None:
    """VOCAB and every table derived from it."""
    global VOCAB, BY_ID, _WORDS, _FENCE_TO, _PKG_TO, _EXT_TO, _ALL_EXT
    global _EXT_RX
    VOCAB = vocab
    BY_ID = {t.id: t for t in VOCAB}
    _WORDS = {t.id: re.compile(t.words, 0 if t.case else re.I)
              for t in VOCAB}
    _FENCE_TO = {f: t.id for t in VOCAB for f in t.fences}
    _PKG_TO = {p: t.id for t in HAND_VOCAB for p in t.packages}
    for t in VOCAB[len(HAND_VOCAB):]:
        for p in t.packages:
            _PKG_TO.setdefault(p, t.id)
    _EXT_TO = {e: t.id for t in VOCAB for e in t.exts}
    _ALL_EXT = sorted(set(_EXT_TO) | set(_EXT_ART), key=len, reverse=True)
    # A file path shape: a word character, a dot, a known extension, a
    # boundary. Flat text (PROTOCOL rule 8 allows it for path shapes).
    _EXT_RX = re.compile(r"(?<=[\w\]\)])\.(" + "|".join(
        map(re.escape, _ALL_EXT)) + r")\b(?![\w-])")
    _EQ_CACHE.clear()


_REFRESH: dict = {"key": None}


def refresh() -> bool:
    """Rebuild VOCAB and its tables when the package registry changed (one
    os.stat when it did not: package_registry.state_key), so a
    long-running worker or proxy picks up a promoted package's term without
    a restart. True when it rebuilt. Called from taxonomy(), asked_terms(),
    negated_terms(), classify(), request_signals() and the other readers of
    the terms."""
    key = package_registry.state_key()
    if _REFRESH["key"] == key:
        return False
    with package_registry._LOCK:
        if _REFRESH["key"] == key:
            return False
        _rebuild(HAND_VOCAB + _registry_terms())
        _REFRESH["key"] = key
    return True


_rebuild(HAND_VOCAB)
_PATH_RX = re.compile(r"[\w.-]*(?:/[\w.-]+)+|\b[\w-]+\.\w{1,6}\b")
_FENCE_RX = re.compile(r"^[ \t]{0,3}(?:`{3,}|~{3,})[ \t]*([\w+#.-]+)", re.M)
_FRONTMATTER = re.compile(r"\A﻿?---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_WHEN_HEAD = re.compile(r"^\s{0,3}#{1,6}\s*(?:when to use|use (?:this|it) when"
                        r"|triggers?|use cases?|when to apply)\b.*$", re.I | re.M)

CODE_ROUTE_CLASSES = {"code_generation", "code_edit"}


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------
def applies_to(rule: dict | None) -> dict:
    a = (rule or {}).get("applies_to") or {}
    return {"artifacts": list(a.get("artifacts") or []),
            "languages": list(a.get("languages") or []),
            "frameworks": list(a.get("frameworks") or [])}


def names(ids) -> list[str]:
    return [BY_ID[i].name if i in BY_ID else ART_BY_ID[i].name
            for i in ids if i in BY_ID or i in ART_BY_ID]


def condition_text(rule: dict) -> str:
    """The human `applies when:` line for a rule, with its gates."""
    base = _condition_base(rule)
    g = gates(rule)
    bits = []
    if g["all_of"]:
        bits.append("with " + " and ".join(names(g["all_of"])))
    if g["phases"]:
        bits.append("while " + " or ".join(_PHASE_ING[p] for p in
                                            g["phases"]))
    if g["situations"]:
        bits.append("when " + " or ".join(_SITUATION_TEXT[s] for s in
                                          g["situations"]))
    if g["topics"]:
        bits.append("about " + ", ".join(g["topics"][:4]))
    if g["tools_all"] or g["tools_any"]:
        bits.append("when the client offers " + (
            " and ".join(g["tools_all"]) if g["tools_all"] else
            " or ".join(g["tools_any"])))
    if g["tools_none"]:
        bits.append("not with " + " or ".join(g["tools_none"]))
    if not base and bits:
        base = "the work"
    return base + ("; " + "; ".join(bits) if bits and base else "")


_PHASE_ING = {"plan": "planning", "implement": "implementing",
              "debug": "debugging", "verify": "verifying",
              "refactor": "refactoring", "review": "reviewing"}
_SITUATION_TEXT = {"error_output": "the output shows an error",
                   "call_mismatch": "a call does not match its callee",
                   "file_line": "a defect is located to a file and line",
                   "multi_file": "several source files are involved",
                   "screenshot_made": "a screenshot or image was made"}


def _condition_base(rule: dict) -> str:
    a = applies_to(rule)
    arts = [x for x in a["artifacts"] if x != "code"]
    terms = a["frameworks"] + a["languages"]
    if rule.get("any") and (terms or arts):
        return f"{' or '.join(names(arts + terms))} is being worked on"
    if arts:
        tail = f" ({', '.join(names(terms))})" if terms else ""
        return f"{' or '.join(names(arts))} is being created or edited{tail}"
    if a["frameworks"]:
        return f"{' or '.join(names(a['frameworks']))} is being used"
    if a["languages"]:
        return f"{' or '.join(names(a['languages']))} is being used"
    if "code" in a["artifacts"]:
        return "code is being written or edited"
    doms = rule.get("domains") or []
    if doms:
        return f"the task involves {' or '.join(sorted(doms))} work"
    return ""


def empty(rule: dict | None) -> bool:
    a = applies_to(rule)
    g = gates(rule)
    return not (a["artifacts"] or a["languages"] or a["frameworks"]
                or (rule or {}).get("domains")
                or g["tools_any"] or g["tools_all"]
                or any(code_shaped(t) for t in (rule or {}).get("topics")
                       or []))


def gates(rule: dict | None) -> dict:
    """The rule's finer gates, cleaned: phases, situations, all_of, topics."""
    r = rule or {}

    def names(key):
        return [str(t).strip() for t in r.get(key) or []
                if str(t).strip()][:12]
    return {
        "phases": [p for p in r.get("phases") or [] if p in PHASES],
        # THE CLIENT'S TOOLS (operator, 2026-09-26: skills help the model
        # discover what its harness offers). Tool names are how selection
        # sees WHICH harness it serves: Hermes offers vision_analyze and
        # terminal, OpenCode read/bash/edit, Codex view_image.
        "tools_any": names("tools_any"),
        "tools_all": names("tools_all"),
        "tools_none": names("tools_none"),
        "situations": [s for s in r.get("situations") or []
                       if s in SITUATIONS],
        "all_of": [t for t in r.get("all_of") or []
                   if t in BY_ID or t in ART_BY_ID],
        "topics": [str(t).strip() for t in r.get("topics") or []
                   if str(t).strip()],
    }


def category(rule: dict | None) -> dict:
    """The skill's place on the taxonomy's four axes, plus its gates: what
    the dashboard browses by and what the SKILL.md frontmatter records."""
    a = applies_to(rule)
    g = gates(rule)
    return {"artifact": a["artifacts"], "language": a["languages"],
            "framework": a["frameworks"], "phase": g["phases"],
            "domain": list((rule or {}).get("domains") or [])}


def tags_of(rule: dict | None, extra=()) -> list[str]:
    """Hermes-style tags: the axis names, then topics."""
    a = applies_to(rule)
    g = gates(rule)
    out: list[str] = []
    for x in (names(a["frameworks"] + a["languages"])
              + [ART_BY_ID[i].name for i in a["artifacts"] if i in ART_BY_ID]
              + g["phases"] + list(extra) + g["topics"]):
        x = str(x).strip()
        if x and x.lower() not in {t.lower() for t in out}:
            out.append(x)
    return out


def taxonomy() -> dict:
    """The vocabulary (the hand-written terms plus one per registry
    package), for the dashboard and the prompts."""
    refresh()
    return {"artifact": [{"id": a.id, "name": a.name} for a in ARTIFACTS],
            "language": [{"id": t.id, "name": t.name} for t in VOCAB
                         if t.kind == "language"],
            "framework": [{"id": t.id, "name": t.name} for t in VOCAB
                          if t.kind == "framework"],
            "phase": list(PHASES), "situation": list(SITUATIONS),
            "tools": "the client's tool names: tools_any / tools_all / "
                     "tools_none"}


# Language keywords and everyday names: in backticks they are code, but
# they appear in almost every request of their language, so they gate
# nothing.
_KEYWORDS = set("""await async return const let var null undefined true false
this self new function class import export default key ref refs props state
type types string number boolean void any unknown never object array list
dict map set int float str bool char none nil some ok err result option
unsafe extern struct enum impl trait fn mut pub use mod crate super static
value values item items data error errors event events index main init
config input output file files path name id test tests div span
function functions component components list lists number numbers sort
sorts write class classes method methods module modules object objects
call calls code guide general usage basics""".split())


def extract_topics(texts, limit: int | None = None) -> list[str]:
    """API names from `code spans` and code-shaped tokens in some text, in
    order of first appearance: the topics a migrated or distilled skill is
    gated by. Language and framework names are the rule's own axes, never
    topics. Every one found (no count or length caps since 2026-09-27,
    docs/CONSTANTS-AUDIT.md); `limit` is for a caller that wants the first
    few (the primary topic)."""
    refresh()
    vocab = {t.name.lower() for t in VOCAB} | {t.id for t in VOCAB}         | _KEYWORDS
    out: list[str] = []

    def add(tok: str):
        tok = tok.strip().strip(".,;:")
        tok = re.sub(r"\(.*\)$", "()", tok) if tok.endswith(")") else tok
        tok = re.sub(r"<.*>$", "", tok)
        if not tok or tok.lower() in vocab:
            return
        if tok.lower() not in {x.lower() for x in out}:
            out.append(tok)

    for text in texts:
        text = str(text or "")
        for span in re.findall(r"`([^`\n]+)`", text):
            span = span.strip()
            # A span is a topic when it is one identifier-ish token.
            m = re.match(r"^(@?[\w$./-]+(?:\(\))?)", span)
            if m and (code_shaped(m.group(1)) or re.match(
                    r"^[a-z][\w$]*$", m.group(1))):
                add(m.group(1))
        for tok in re.findall(r"(?<![`\w$])([A-Za-z_$][\w$]*(?:\.[\w$]+)+"
                              r"|[a-z]+[A-Z][\w$]*)(?![`\w$])", text):
            if code_shaped(tok):
                add(tok)
        if limit is not None and len(out) >= limit:
            break
    return out if limit is None else out[:limit]


# ---------------------------------------------------------------------------
# Source side
# ---------------------------------------------------------------------------
def frontmatter(text: str) -> tuple[dict, str]:
    """(fields, body). A YAML parser if one is installed, else `key: value`
    lines; a folded or literal block (`>` / `|`) is joined either way."""
    m = _FRONTMATTER.match(text or "")
    if not m:
        return {}, text or ""
    raw = m.group(1)
    body = "\n" * m.group(0).count("\n") + text[m.end():]
    fields: dict = {}
    try:
        import yaml
        got = yaml.safe_load(raw)
        if isinstance(got, dict):
            fields = {str(k): v for k, v in got.items()}
    except Exception:                                            # noqa: BLE001
        key = None
        for line in raw.split("\n"):
            mm = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
            if mm:
                key = mm.group(1)
                fields[key] = mm.group(2).strip().strip("\"'").lstrip(">|")
            elif key and line.startswith((" ", "\t")):
                fields[key] = (str(fields[key]) + " " + line.strip()).strip()
    return fields, body


def _line_at(text: str, pos: int) -> str:
    a = text.rfind("\n", 0, pos) + 1
    b = text.find("\n", pos)
    return text[a:b if b >= 0 else len(text)].strip()[:240]


def _imports(code: str) -> list[str]:
    try:
        import discover
        return list(discover.imports(code) or [])
    except Exception:                                            # noqa: BLE001
        return []


def _sentences(s: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", " ".join(str(s).split()))
    return [p.strip() for p in parts if len(p.strip()) >= 12]


def triggers_of(text: str, fields: dict) -> list[dict]:
    """Trigger lines with verbatim quotes: the WHOLE description (one
    trigger: the SKILL.md description IS the trigger condition, bounded by
    its 1,024 characters), then every bullet under a "When to use"
    heading, to the next heading. No sentence split, no length floor, no
    count cap (removed 2026-09-27, docs/CONSTANTS-AUDIT.md)."""
    out: list[dict] = []
    desc = fields.get("description")
    if isinstance(desc, str) and desc.strip():
        d = " ".join(desc.split())[:L.DESCRIPTION_CHARS]
        out.append({"text": d, "quote": d, "origin": "description"})
    for m in _WHEN_HEAD.finditer(text):
        rest = text[m.end():]
        stop = re.search(r"^\s{0,3}#{1,6}\s", rest, re.M)
        block = rest[:stop.start()] if stop else rest
        for ln in block.split("\n"):
            mm = re.match(r"^\s*(?:[-*+]|\d+\.)\s+(\S.*)$", ln)
            if mm:
                t = mm.group(1).strip()
                out.append({"text": t, "quote": t, "origin": "when-to-use"})
    seen, uniq = set(), []
    for t in out:
        k = t["text"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def _source_hits(body: str) -> dict[str, dict]:
    """term or artifact id -> {count, how, quote}."""
    import skill_screen
    hits: dict[str, dict] = {}

    def add(tid, how, quote, n=1):
        h = hits.setdefault(tid, {"count": 0, "how": set(), "quote": None})
        h["count"] += n
        h["how"].add(how)
        if h["quote"] is None and quote:
            h["quote"] = quote

    for _line, info, code in skill_screen.fences(body):
        tag = info.split()[0] if info else ""
        tid = _FENCE_TO.get(tag)
        if tid:
            add(tid, "fence", f"```{info}".strip())
        if code.strip() and tid in (None, "typescript", "javascript", "rust",
                                    "python"):
            for pkg in _imports(code):
                pid = _PKG_TO.get(pkg)
                if pid:
                    add(pid, "import", next((ln.strip() for ln in
                                             code.split("\n") if pkg in ln),
                                            pkg)[:240])
    prose = skill_screen._outside_fences(body)
    for t in VOCAB:
        ms = list(_WORDS[t.id].finditer(prose))
        if ms:
            add(t.id, "word", _line_at(prose, ms[0].start()), len(ms))
    for a in ARTIFACTS:
        if a.id == "code":
            continue
        ms = list(_NOUNS[a.id].finditer(prose))
        if ms:
            add(a.id, "word", _line_at(prose, ms[0].start()), len(ms))
    return hits


def classify(text: str, *, declared: dict[str, str] | None = None) -> dict:
    """The applies-when rule for a source, with evidence and triggers.

    `declared` maps a term or artifact id to the verbatim line of the source
    that declares it (the migration's `language: Rust` header); a declared
    id counts as MIN_HITS.
    """
    refresh()
    import domains as domain_gate
    fields, body = frontmatter(text or "")
    hits = _source_hits(body)
    for tid, line in (declared or {}).items():
        if tid in BY_ID or tid in ART_BY_ID:
            h = hits.setdefault(tid, {"count": 0, "how": set(), "quote": None})
            h["how"].add("declared")
            h["quote"] = line
            h["count"] = max(h["count"], MIN_HITS)
    a = {"artifacts": [], "languages": [], "frameworks": []}
    evidence = []
    kinds = (("framework", "frameworks", lambda i: i in BY_ID
              and BY_ID[i].kind == "framework"),
             ("language", "languages", lambda i: i in BY_ID
              and BY_ID[i].kind == "language"),
             ("artifact", "artifacts", lambda i: i in ART_BY_ID))
    for kind, key, belongs in kinds:
        cands = sorted(((i, h) for i, h in hits.items()
                        if belongs(i) and h["count"] >= MIN_HITS),
                       key=lambda x: (-x[1]["count"], x[0]))
        if not cands:
            continue
        top = cands[0][1]["count"]
        for i, h in cands:
            if len(a[key]) >= MAX_TERMS:
                break
            if h["count"] >= SHARE * top:
                a[key].append(i)
                evidence.append({"term": i, "name": names([i])[0],
                                 "kind": kind, "count": h["count"],
                                 "how": sorted(h["how"]), "quote": h["quote"]})
    # A non-code artifact outranks every term in match(), so on a source that
    # is about a language or framework it must earn that: declared, or at
    # least as prominent as the top term. "layout" is memory layout in the
    # Rust and C corpora and "UI" is in every React page; without this the
    # migration dry run classified 20 of 67 code groups as UI design.
    top_term = max([hits[i]["count"] for i in a["frameworks"] + a["languages"]]
                   or [0])
    if top_term:
        keep = [i for i in a["artifacts"]
                if "declared" in hits[i]["how"] or hits[i]["count"] >= top_term]
        dropped = [i for i in a["artifacts"] if i not in keep]
        a["artifacts"] = keep
        evidence = [e for e in evidence if e.get("term") not in dropped]
    if (a["languages"] or a["frameworks"]) and not a["artifacts"]:
        a["artifacts"].append("code")
    doms: set[str] = set()
    for i in a["frameworks"] + a["languages"]:
        doms |= BY_ID[i].domains
    for i in a["artifacts"]:
        doms |= ART_BY_ID[i].domains
    if not doms and not a["artifacts"]:
        doms = set(domain_gate.detect([{"role": "user", "content": body}]))
        if doms:
            evidence.append({"term": None, "kind": "domain",
                             "how": ["domain words"], "domains": sorted(doms)})
    rule = {"applies_to": a, "domains": sorted(doms & set(domain_gate.DOMAINS)),
            "evidence": evidence}
    rule["text"] = condition_text(rule)
    trig = triggers_of(body, fields)
    if not trig and rule["text"]:
        trig = [{"text": rule["text"], "quote": None, "origin": "derived"}]
    rule["triggers"] = trig
    if isinstance(fields.get("name"), str):
        rule["declared_name"] = fields["name"][:64]
    # ESCALATE (Phase 0.6, trigger 3; operator 2026-09-24): a skill whose
    # frontmatter says `escalate: true` declares that its area needs deep
    # thinking; mcp/deep.py runs it once per skill per conversation when the
    # skill is injected at a tier that allows deep thinking. A poisoned
    # source can at most spend one read-only second-brain run per
    # conversation with it; the screen still decides whether it arms.
    esc = fields.get("escalate")
    if esc is True or str(esc).strip().lower() in ("true", "yes", "1"):
        rule["escalate"] = True
    return rule


_NOT_FOR = re.compile(r"(?:^|(?<=[.;:!?]))\s*Not (?:for|when)\b[^.]*\.?")


def strip_boundary(description: str) -> str:
    """The description without its "Not for / Not when ..." sentence."""
    return _NOT_FOR.sub("", description or "").strip()


def with_gates(rule: dict, *, phases=(), situations=(), all_of=(),
               topics=(), description: str = "", tools_any=None,
               tools_all=None, tools_none=None) -> dict:
    """`rule` with the finer gates set (each checked against the fixed
    vocabulary) and, when given, the description as its first trigger --
    the SKILL.md description IS the trigger condition. The tool gates are
    kept as the rule has them unless given (None)."""
    out = dict(rule)
    given = {"tools_any": tools_any, "tools_all": tools_all,
             "tools_none": tools_none}
    g = gates({"phases": list(phases), "situations": list(situations),
               "all_of": list(all_of), "topics": list(topics),
               **{k: list(v if v is not None else out.get(k) or [])
                  for k, v in given.items()}})
    for k, v in g.items():
        if v:
            out[k] = v
        else:
            out.pop(k, None)
    if description:
        trig = [t for t in out.get("triggers") or []
                if isinstance(t, dict) and t.get("origin") != "description"]
        # A description's "Not for ..." boundary sentence (distil/4,
        # decompose/3) says where the skill does NOT apply: it stays in the
        # SKILL.md for the model to read, and out of the trigger text, whose
        # words would otherwise pull the embedding toward the very
        # situation the boundary excludes.
        d = " ".join(strip_boundary(description).split())[
            :L.DESCRIPTION_CHARS]
        desc = [{"text": d, "quote": None, "origin": "description"}] \
            if d else []
        out["triggers"] = desc + trig
    out["text"] = condition_text(out)
    return out


def verified_all_of(proposed, applies: dict, text: str
                    ) -> tuple[list[str], dict, dict]:
    """MULTI-DOMAIN FILING (operator, 2026-09-27: "Why can't we tag skills
    that apply to multiple domains?"). The tag stage (skill_prompts tag/5)
    may propose `all_of`: languages or frameworks that must ALSO be in use,
    for a skill about using them TOGETHER (koota inside React, pmndrs math
    with three.js). The model proposes; the text decides. A proposed term is
    kept only when
      - it is a language or framework of the fixed vocabulary, and
      - the text (the skill, or the source it was made from) names it
        (_WORDS, the same test a tag's framework must pass).
    A kept term leaves the OR list (applies_to's languages / frameworks):
    "koota or React" becomes "koota, with React". When that would leave the
    OR list with no language or framework -- nothing left to be the primary
    key -- the proposal is refused whole and the rule stays as it was.

    Returns (all_of, applies, record {proposed, kept, dropped[{term, why}]})."""
    refresh()
    a = {k: list(v) for k, v in (applies or {}).items()}
    rec: dict = {"proposed": [str(x) for x in proposed or []][:12],
                 "kept": [], "dropped": []}
    kept: list[str] = []
    for x in rec["proposed"]:
        term = BY_ID.get(x)
        if term is None or term.kind not in ("language", "framework"):
            rec["dropped"].append({"term": x, "why": "not a language or "
                                   "framework of the vocabulary"})
        elif not _WORDS[x].search(text or ""):
            rec["dropped"].append({"term": x, "why": "the text does not "
                                   "name it"})
        elif x not in kept:
            kept.append(x)
    if not kept:
        return [], a, rec
    rest = {k: [t for t in a.get(k) or [] if t not in kept]
            for k in ("languages", "frameworks")}
    if not (rest["languages"] or rest["frameworks"]):
        rec["dropped"] += [{"term": x, "why": "it would leave no other "
                            "language or framework to key the skill on"}
                           for x in kept]
        return [], a, rec
    a.update(rest)
    rec["kept"] = kept
    return kept, a, rec


def rule_from_metadata(aw: dict) -> dict:
    """A rule from a SKILL.md's `metadata.yamadori.applies_when` (an
    authored or compiled skill states its own; nothing is inferred)."""
    refresh()
    aw = aw or {}
    a = {"artifacts": [x for x in aw.get("artifacts") or [] if x in ART_BY_ID],
         "languages": [x for x in aw.get("languages") or []
                       if x in BY_ID and BY_ID[x].kind == "language"],
         "frameworks": [x for x in aw.get("frameworks") or []
                        if x in BY_ID and BY_ID[x].kind == "framework"]}
    if (a["languages"] or a["frameworks"]) and not a["artifacts"]:
        a["artifacts"] = ["code"]
    doms: set[str] = set(aw.get("domains") or [])
    for i in a["frameworks"] + a["languages"]:
        doms |= BY_ID[i].domains
    for i in a["artifacts"]:
        doms |= ART_BY_ID[i].domains
    rule = {"applies_to": a, "domains": sorted(doms), "evidence": [
        {"term": i, "name": names([i])[0], "how": ["declared"],
         "kind": "declared", "quote": None}
        for i in a["frameworks"] + a["languages"] + a["artifacts"]]}
    if aw.get("any"):
        rule["any"] = True
    trig = [{"text": str(t), "quote": None,
             "origin": "declared"} for t in aw.get("triggers") or []
            if str(t).strip()]
    rule["triggers"] = trig
    if aw.get("escalate"):
        rule["escalate"] = True
    return with_gates(rule, tools_any=aw.get("tools_any") or (),
                      tools_all=aw.get("tools_all") or (),
                      tools_none=aw.get("tools_none") or (),
                      phases=aw.get("phases") or (),
                      situations=aw.get("situations") or (),
                      all_of=aw.get("all_of") or (),
                      topics=aw.get("topics") or ())


def metadata_of_rule(rule: dict) -> dict:
    """The inverse of rule_from_metadata: what a SKILL.md records."""
    a = applies_to(rule)
    g = gates(rule)
    out = {"artifacts": a["artifacts"], "languages": a["languages"],
           "frameworks": a["frameworks"]}
    for k in ("phases", "situations", "all_of", "topics", "tools_any",
              "tools_all", "tools_none"):
        if g[k]:
            out[k] = g[k]
    if rule.get("domains"):
        out["domains"] = list(rule["domains"])
    trig = [t.get("text") for t in rule.get("triggers") or []
            if isinstance(t, dict) and t.get("origin") != "description"
            and t.get("text")]
    if trig:
        out["triggers"] = trig
    if rule.get("any"):
        out["any"] = True
    if rule.get("escalate"):
        out["escalate"] = True
    out["text"] = rule.get("text") or condition_text(rule)
    return out


def rule_of_condition(line: str) -> dict:
    """An `applies when:` line a PERSON wrote, read back into a rule. `any`:
    "React or TypeScript" means either one."""
    refresh()
    import domains as domain_gate
    a = {"artifacts": [], "languages": [], "frameworks": []}
    evidence = []
    for t in VOCAB:
        if _WORDS[t.id].search(line) or re.search(
                r"(?<![\w.])" + re.escape(t.name) + r"(?![\w])", line, re.I):
            key = "frameworks" if t.kind == "framework" else "languages"
            if len(a[key]) < MAX_TERMS:
                a[key].append(t.id)
                evidence.append({"term": t.id, "name": t.name, "kind": t.kind,
                                 "how": ["operator"], "quote": line.strip()})
    for art in ARTIFACTS:
        if (art.id != "code" and _NOUNS[art.id].search(line)) or re.search(
                r"(?<![\w])" + re.escape(art.name) + r"(?![\w])", line, re.I):
            if len(a["artifacts"]) < MAX_TERMS:
                a["artifacts"].append(art.id)
                evidence.append({"term": art.id, "name": art.name,
                                 "kind": "artifact", "how": ["operator"],
                                 "quote": line.strip()})
    doms = {d for d in domain_gate.DOMAINS
            if re.search(r"(?<![\w-])" + re.escape(d) + r"(?![\w-])", line,
                         re.I)}
    for i in a["frameworks"] + a["languages"]:
        doms |= BY_ID[i].domains
    rule = {"applies_to": a, "domains": sorted(doms), "evidence": evidence,
            "any": True}
    rule["text"] = condition_text(rule)
    rule["triggers"] = [{"text": line.strip(), "quote": None,
                         "origin": "operator"}] if line.strip() else []
    return rule


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------
def _text_of(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c
                     if isinstance(x, dict) and isinstance(x.get("text"), str))
    return c if isinstance(c, str) else ""


def _tool_call_paths(messages: list[dict]) -> list[str]:
    """String arguments of the conversation's tool calls that look like file
    paths: the files an agent is reading and writing are facts about the
    artifact."""
    out = []
    for m in messages or []:
        for tc in m.get("tool_calls") or []:
            args = ((tc or {}).get("function") or {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_": args}
            stack = [args]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
                elif isinstance(x, str) and len(x) < 400 and "\n" not in x \
                        and re.search(r"[\w-]\.\w{1,6}$|/", x):
                    out.append(x.strip())
                elif isinstance(x, str) and "\n" in x:
                    # A PATCH names its files in its own headers: Codex
                    # writes every file through apply_patch, one multi-line
                    # V4A argument ("*** Add File: js/audio.js"), 2026-09-26.
                    # tool_code's parser, the one that checks those writes.
                    import tool_code
                    if tool_code.looks_like_patch(x):
                        out += [f["path"] for f in tool_code.parse_patch(x)
                                if f.get("path")]
    return out[-50:]


# A package named with an exact version in prose: `koota 0.6.6`,
# `@react-three/fiber 10.0.0-alpha.5`, `math@0.1.0`. Flat text (PROTOCOL
# rule 8 keeps regexes for version strings); the name must be a package
# the vocabulary knows, so "version 1.2.3" names nothing.
_PROSE_PIN = re.compile(
    r"(?<![\w@/.-])(@?[a-z0-9][\w.-]*(?:/[\w.-]+)?)(?:@| )v?"
    r"(\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?)(?![\w.-])")


def _pinned(text: str) -> dict[str, str]:
    """{package: version} pinned in the text: manifest lines (discover's
    reader) and a spec's `name 1.2.3` pins."""
    out: dict[str, str] = {}
    try:
        import discover
        out.update(discover.versions(text or ""))
    except Exception:                                            # noqa: BLE001
        pass
    for m in _PROSE_PIN.finditer(text or ""):
        out.setdefault(m.group(1), m.group(2))
    return out


def window_start(messages: list[dict]) -> int:
    """Where the WORK being evidenced begins (a structural boundary, since
    2026-09-27; the 8-message window and 40,000-character cut were ours,
    docs/CONSTANTS-AUDIT.md "WORKED_MESSAGES / WORKED_CHARS", "_evidence
    window"): the last user turn -- or, for a request that ends ON a user
    turn, the user turn before it (the exchange that turn answers)."""
    msgs = messages or []
    users = [i for i, m in enumerate(msgs) if isinstance(m, dict)
             and m.get("role") == "user"]
    if not users:
        return 0
    if users[-1] == len(msgs) - 1:
        return users[-2] if len(users) > 1 else 0
    return users[-1]


def _worked_code(messages: list[dict]) -> list[tuple[str, str]]:
    """(how, text): the tool results and the multi-line string arguments of
    the tool calls (the code written or patched) of the current work
    (window_start)."""
    results, written = [], []
    for m in (messages or [])[window_start(messages or []):]:
        if m.get("role") == "tool":
            results.append(_text_of(m))
        for tc in m.get("tool_calls") or []:
            args = ((tc or {}).get("function") or {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_": args}
            stack = [args]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
                elif isinstance(x, str) and "\n" in x:
                    written.append(x)
    out = []
    if results:
        out.append(("tool result", "\n".join(results)))
    if written:
        out.append(("written", "\n".join(written)))
    return out


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            t = _text_of(m)
            if t.strip():
                return t
    return ""


# THE EMBEDDING QUERY (2026-09-27; #61's open gap). The embedding stage
# embedded the last user text alone, so a short follow-up ("the bullets never
# move ...") or an agent step, whose evidence is in the recent tool results
# and the code the agent wrote, embedded far from the skill that evidence
# names (koota-queries-and-systems at 0.35 < EMB_LOW: fact -> ask). The query
# is now the last user text PLUS a digest of the same newest evidence the
# deterministic stages read (the last two tool results and the last two
# writes, within the last WORKED_MESSAGES messages): the error lines, the
# packages imported (and the languages/frameworks they are), the API names
# and the source files named. Deterministic, bounded, and the user text stays
# FIRST and dominant when it is substantive.
#
# THE DIGEST JOINS ONLY WHEN THE TURN IS ABOUT THE EVIDENCE: an agent step
# (the evidence is newer than the last user turn), a failure report or fix
# ask (the debug phase's own phrases, or an error's text), or a user turn
# that names something in it (a file, an API name, a package, a word of its
# identifiers). A short NEW instruction after a tool result ("make the
# background dark blue") is not about it: with the digest it embedded near
# the previous step's koota skills (0.34 -> 0.55, the same replay), which
# would have injected them without the fallback. Every number is a CHOICE.
EMBED_QUERY_CHARS = 2000          # the whole query (the old cut of the text)
DIGEST_HEAD = "Recent tool output and code:"
_EQ_CACHE_MAX = 64
# A location in front of an error line (`src/a.ts:14:5 - `): the file is in
# `files`; the message is the evidence.
_LOC_PREFIX = re.compile(r"^(?:[\w./\\-]+\.\w{1,5}[:(]\d+(?:[:,]\d+)*\)?"
                         r"\s*(?:-\s*)?)+")
# Words too common to say a user turn is about a tool result (with the
# code vocabulary's own generic words, _KEYWORDS).
_PLAIN_WORDS = frozenset("""that this these those with from have what when
then there they them their into your make made like just also only some more
than does dont should would could about after before still again please
thanks same work works working next want need needs using used were been being
will very much many other each every where which while here back down over
under well good better right left take give keep look show shows seen""".split())


def _evidence(messages: list[dict]) -> tuple[list[tuple[str, str]], bool]:
    """((kind, text) newest first, whether it is newer than the last user
    turn): the tool results and the code the tool calls wrote in the
    current work (window_start: a tool result from an earlier task names
    nothing)."""
    msgs = messages or []
    lu = next((i for i in range(len(msgs) - 1, -1, -1)
               if msgs[i].get("role") == "user"), -1)
    out: list[tuple[str, str]] = []
    newest = -1
    start = window_start(msgs)
    for i in range(len(msgs) - 1, start - 1, -1):
        m = msgs[i]
        if m.get("role") == "tool":
            t = _text_of(m)[-8000:]
            if t.strip():
                out.append(("tool result", t))
                newest = max(newest, i)
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            got = [t[-8000:] for _h, t in _worked_code([m])]
            if got:
                out += [("written", t) for t in got]
                newest = max(newest, i)
    return out, newest > lu


def _error_lines(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or not (_ERROR_OUT.search(s) or _CALL_MISMATCH.search(s)):
            continue
        s = _LOC_PREFIX.sub("", s).strip()
        if s:
            out.append(" ".join(s.split())[:160])
    return out


def _about_evidence(user: str, texts: list[str], agent_step: bool
                    ) -> tuple[bool, str]:
    """ALWAYS JOIN (2026-09-27): the evidence of the current work goes with
    every turn. The rule that joined only a failure report, an agent step
    or a turn sharing a 4+-letter plain word with it was ours
    (docs/CONSTANTS-AUDIT.md "_about_evidence join rule")."""
    if agent_step:
        return True, "agent step: the evidence is newer than the user turn"
    return True, "the current work's evidence"


def embedding_query(messages: list[dict]) -> tuple[str, dict]:
    """(the text the embedding stage embeds, what it was built from). The
    record carries counts and source kinds only, never the text: the
    caller's code is not stored (nebari's rule), and x_yamadori.skills is
    stored with the turn."""
    refresh()
    last = _last_user_text(messages)
    ev, agent_step = _evidence(messages)
    h = hashlib.sha1(last.encode("utf-8", "replace"))
    h.update(b"\x02" if agent_step else b"\x03")
    for kind, t in ev:
        h.update(b"\x00" + kind.encode() + b"\x01"
                 + t.encode("utf-8", "replace"))
    key = h.hexdigest()
    hit = _EQ_CACHE.get(key)
    if hit is not None:
        return hit[0], dict(hit[1], cache="hit")
    # A FIXED DIGEST (2026-09-27): every error line, import, API name and
    # source file of the current work's evidence, after the user text; the
    # whole query cut at EMBED_QUERY_CHARS, the user text first. The digest
    # budgets (800 / 300 characters around a 400-character "substantive"
    # user text) and the per-kind counts (3 / 8 / 10 / 4) were ours
    # (docs/CONSTANTS-AUDIT.md "EMBED_* / DIGEST_*").
    user = last.strip()
    kinds = collections.Counter(k for k, _t in ev)
    rec = {"user_chars": 0, "digest_chars": 0,
           "sources": dict(sorted(kinds.items())), "digest": "none",
           "because": "no recent tool result or written code",
           "errors": 0, "frameworks": 0, "imports": 0, "names": 0,
           "files": 0, "cap": EMBED_QUERY_CHARS}
    digest = ""
    if ev:
        joined, because = _about_evidence(user, [t for _k, t in ev],
                                          agent_step)
        rec.update(digest="joined" if joined else "withheld",
                   because=because)
        if joined:
            digest, counts = _digest(ev)
            rec.update(counts)
    query = ((user + "\n\n" + digest).strip() if digest else user)[
        :EMBED_QUERY_CHARS]
    rec.update(user_chars=min(len(user), len(query)),
               digest_chars=max(len(query) - len(user) - 2, 0)
               if digest else 0)
    if len(_EQ_CACHE) >= _EQ_CACHE_MAX:
        _EQ_CACHE.pop(next(iter(_EQ_CACHE)))
    _EQ_CACHE[key] = (query, rec)
    return query, dict(rec, cache="miss")


def _digest(ev: list[tuple[str, str]]) -> tuple[str, dict]:
    """(the digest text, its counts): error lines, what the imports are,
    the packages, API names and source files, newest evidence first."""
    errors: list[str] = []
    pkgs: list[str] = []
    fw: list[str] = []
    for _kind, t in ev:
        for e in _error_lines(t):
            if e not in errors:
                errors.append(e)
        if re.search(r"\bimport\b|\brequire\(|\buse \w", t):
            for p in _imports(t):
                if p and p not in pkgs:
                    pkgs.append(p)
    for p in pkgs:
        tid = _PKG_TO.get(p)
        name = BY_ID[tid].name if tid in BY_ID else None
        if name and name not in fw:
            fw.append(name)
    names = extract_topics([t for _k, t in ev])
    files: list[str] = []
    for _kind, t in ev:
        for p in _SOURCE_PATH.findall(t):
            b = p.replace("\\", "/").rsplit("/", 1)[-1]
            if b not in files:
                files.append(b)
    lines = []
    if errors:
        lines.append("errors: " + " | ".join(errors))
    if fw:
        lines.append("uses: " + ", ".join(fw))
    if pkgs:
        lines.append("imports: " + ", ".join(pkgs))
    if names:
        lines.append("names: " + ", ".join(names))
    if files:
        lines.append("files: " + ", ".join(files))
    digest = DIGEST_HEAD + "\n" + "\n".join(lines) if lines else ""
    return digest, {"errors": len(errors), "frameworks": len(fw),
                    "imports": len(pkgs), "names": len(names),
                    "files": len(files)}


def request_signals(messages: list[dict],
                    route_class: str | None = None,
                    tools: list[str] | None = None) -> dict:
    """{terms: {id: {strength, how: [...]}}, artifacts: {id: {...}},
    domains: [...], query: text to embed}."""
    refresh()
    import domains as domain_gate
    blob = "\n".join(_text_of(m) for m in messages or []
                     if m.get("role") in ("user", "system"))
    # The phrase and noun reads look at the last user turn only, and at its
    # head: a pasted file is not the instruction.
    last = _last_user_text(messages)[:8000]
    terms: dict[str, dict] = {}
    arts: dict[str, dict] = {}

    def add(table, i, strength, how):
        e = table.setdefault(i, {"strength": strength, "how": []})
        if STRENGTH[strength] > STRENGTH[e["strength"]]:
            e["strength"] = strength
        if how not in e["how"] and len(e["how"]) < 3:
            e["how"].append(how)

    for m in _FENCE_RX.finditer(blob):
        tid = _FENCE_TO.get(m.group(1).lower())
        if tid:
            add(terms, tid, "fact", f"fence {m.group(1).lower()}")
    if re.search(r"\bimport\b|\brequire\(|\buse \w|\bfrom \S+ import\b", blob):
        for pkg in _imports(blob):
            tid = _PKG_TO.get(pkg)
            if tid:
                add(terms, tid, "fact", f"import {pkg}")
    # THE CODE THE AGENT IS WORKING ON: imports in the recent tool results
    # (a file it read, a build log quoting a line) and in the code its tool
    # calls wrote are facts about the project (replay of the V4 stack,
    # 2026-09-26: a step that read an R3F component carried no R3F fact).
    worked = _worked_code(messages)
    for how, text in worked:
        if re.search(r"\bimport\b|\brequire\(|\buse \w", text):
            for pkg in _imports(text):
                tid = _PKG_TO.get(pkg)
                if tid:
                    add(terms, tid, "fact", f"import {pkg} ({how})")
        for pkg, ver in _pinned(text).items():
            tid = _PKG_TO.get(pkg)
            if tid:
                add(terms, tid, "fact", f"pinned {pkg}@{ver} ({how})")
    # A PINNED DEPENDENCY is a fact about the project: a manifest line
    # ("koota": "0.6.6") or a spec's pin list ("koota 0.6.6, math 0.1.0").
    for pkg, ver in _pinned(blob).items():
        tid = _PKG_TO.get(pkg)
        if tid:
            add(terms, tid, "fact", f"pinned {pkg}@{ver}")
    paths = _tool_call_paths(messages)
    for src, text in (("request", blob), ("tool call", "\n".join(paths))):
        for m in _EXT_RX.finditer(text):
            ext = m.group(1)
            if ext in _EXT_TO:
                add(terms, _EXT_TO[ext], "fact", f"{src} file .{ext}")
            if ext in _EXT_ART:
                add(arts, _EXT_ART[ext], "fact", f"{src} file .{ext}")
    import itertools
    for p in paths + [x.group(0) for x in
                      itertools.islice(_PATH_RX.finditer(blob), 500)]:
        for aid, rx in _APATH.items():
            if rx.search(p):
                add(arts, aid, "fact", f"path {p[-60:]}")
    for t in VOCAB:
        # A mention behind a negation ("no React", "without R3F", "instead
        # of koota") is not evidence FOR the term (2026-10-07, from the
        # activation cases of the gap-fill skills: "plain three.js (no
        # React, no React Three Fiber)" selected the React skills).
        for m in _WORDS[t.id].finditer(blob):
            if _NEG_ADJ.search(blob[max(0, m.start() - 80):m.start()]):
                continue
            add(terms, t.id, "word", f"word {m.group(0)!r}")
            break
    # THE EXPLICIT ASK: named by the user in the current request's prose.
    # A SPECIFIC framework asked for is a FACT ("build it with koota"); a
    # language or a host framework (React) named in prose stays a WORD --
    # half the store is filed under them, and "my React page" is no
    # evidence for an ungated React skill (HOST_FRAMEWORKS, above).
    asked = asked_terms(_last_user_text(messages))
    for tid, e in asked.items():
        t = BY_ID.get(tid)
        specific = bool(t and t.kind == "framework"
                        and tid not in HOST_FRAMEWORKS)
        add(terms, tid, "fact" if specific else "word",
            f"asked {e['form']!r}"
            + (f" v{e['version']}" if e.get("version") else ""))
    # KEPT FOR THE OPERATOR (docs/CONSTANTS-AUDIT.md "VERBS + phrase
    # window", 2026-09-27): with nouns only, "make me a slide deck" is a
    # WORD, which the selector's language round does not open (ART_OPENING),
    # so no slides or docs craft could be chosen for it.
    for a in ARTIFACTS:
        m = _PHRASE[a.id].search(last)
        if m:
            add(arts, a.id, "phrase", f"phrase {m.group(0)[-60:]!r}")
        elif a.id != "code":
            m = _NOUNS[a.id].search(last)
            if m:
                add(arts, a.id, "word", f"word {m.group(0)!r}")
    if route_class in CODE_ROUTE_CLASSES:
        add(arts, "code", "fact", f"route {route_class}")
    if terms and "code" not in arts:
        best = max(terms.values(), key=lambda e: STRENGTH[e["strength"]])
        add(arts, "code", best["strength"], "a language or framework")
    try:
        doms = sorted(domain_gate.detect(messages))
    except Exception:                                            # noqa: BLE001
        doms = []
    # THE TAXONOMY'S REQUEST SIDE: phases, situations, and the text topics
    # are looked for in (the last user turn, the recent messages' text and
    # the conversation's tool-call arguments).
    recent = "\n".join(_text_of(m) for m in (messages or [])[-8:]
                       if m.get("role") in ("tool", "assistant", "user"))
    # The code the agent WROTE is text too: its API names are topics (a
    # terrain file written with simplex2d.create names the noise skill).
    written = "\n".join(t for how, t in worked if how == "written")
    topic_text = "\n".join([last, recent[-20000:], "\n".join(paths),
                            written[-20000:]])
    # THE NEWEST EVIDENCE: the last user turn, the last two tool results
    # and the code the last two assistant turns wrote. A topic found here
    # ranks its skill higher (match), so a later turn about a failing koota
    # query is not outranked by what the opening spec also named.
    fresh = [last]
    for m in reversed((messages or [])[window_start(messages or []):]):
        if m.get("role") == "tool":
            fresh.append(_text_of(m)[-8000:])
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            fresh += [t[-8000:] for _h, t in _worked_code([m])]
    phases: dict[str, str] = {}
    for p, rx in _PHASE_RX.items():
        m = rx.search(last)
        if m:
            phases[p] = f"phrase {m.group(0)[-40:]!r}"
    tail_tool = "\n".join(_text_of(m) for m in (messages or [])[-4:]
                          if m.get("role") == "tool")[-8000:]
    if "debug" not in phases and _ERROR_OUT.search(tail_tool):
        phases["debug"] = "an error in a recent tool result"
    # Code work that is not debugging, verifying, refactoring or reviewing
    # is being implemented -- planned too, maybe: a build spec that says
    # "outline" or "spec" is still a build (replay of the V4 stack,
    # 2026-09-26: the Octopus spec read as `plan` only).
    if route_class in CODE_ROUTE_CLASSES or ("code" in arts and not (
            set(phases) & {"debug", "verify", "refactor", "review"})):
        phases.setdefault("implement", f"route {route_class}"
                          if route_class in CODE_ROUTE_CLASSES else
                          "code is being written")
    situ: dict[str, str] = {}
    err_blob = last + "\n" + tail_tool
    m = _ERROR_OUT.search(err_blob)
    if m:
        situ["error_output"] = f"{m.group(0)[:40]!r}"
    m = _CALL_MISMATCH.search(err_blob)
    if m:
        situ["call_mismatch"] = f"{m.group(0)[:40]!r}"
    m = _FILE_LINE.search(last + "\n" + recent[-20000:])
    if m:
        situ["file_line"] = f"{m.group(0)[:60]!r}"
    # The client's tools: those the request offers, and any the history
    # shows it calling (a call proves the harness offered it).
    tool_names = {str(t) for t in tools or [] if t}
    for mm in messages or []:
        for tc in mm.get("tool_calls") or []:
            n = ((tc or {}).get("function") or {}).get("name")
            if n:
                tool_names.add(str(n))
    shot = _SHOT.search("\n".join(paths) + "\n" + tail_tool)
    if shot:
        situ["screenshot_made"] = f"{shot.group(0)[-50:]!r}"
    srcs = {p.lower() for p in _SOURCE_PATH.findall(
        "\n".join(paths) + "\n" + blob[:40000])}
    if len(srcs) >= 2:
        situ["multi_file"] = f"{len(srcs)} source files"
    # `query` stays the last user text: the fallback's durable record and
    # what idle-time learning turns into triggers read it, and E1 and Laya
    # were built on it. The EMBEDDING stage embeds `embed_query`.
    eq, efrom = embedding_query(messages)
    return {"terms": terms, "artifacts": arts, "domains": doms,
            "phases": phases, "situations": situ, "asked": asked,
            # Named only behind "without" / "instead of" in the user's own
            # words (the selector's category rounds drop them when a bare
            # word is all the evidence there is).
            "negated": sorted(negated_terms(_last_user_text(messages))),
            # What the request RULES OUT (negated names, and the framework
            # layers a "vanilla" / "plain" stack leaves out) and the major
            # versions it STATES (match's two new gates).
            "ruled_out": sorted(ruled_out(_last_user_text(messages))),
            "versions": stated_versions(asked, blob),
            "tools": sorted(tool_names),
            "topic_text": topic_text, "fresh_text": "\n".join(fresh),
            "query": last[:2000], "embed_query": eq, "embed_from": efrom}


# The signal fields that carry the request's TEXT (the caller's messages and
# code). Never written to a durable record: the fallback's `features` keep
# the rest (terms, phases, counts).
TEXT_FIELDS = ("query", "topic_text", "fresh_text", "embed_query")
# The first word of an evidence string (`how`): its KIND. Anything else is
# recorded as "other" -- the quoted part (a matched phrase, a path, a
# package) never reaches a durable record.
_HOW_KINDS = {"fence", "import", "pinned", "request", "tool", "path", "word",
              "phrase", "route", "a", "asked"}


def durable_signals(sig: dict) -> dict:
    """The request's signals as a DURABLE record may hold them (2026-09-27;
    the fallback's `features` and the router labels): term and artifact ids
    with their strength and the KINDS of their evidence, phase and
    situation NAMES, domain ids, the client's tool names and the embedding
    query's counts. Never text: not the TEXT_FIELDS, not a matched phrase,
    not a path:line, not an error's words. Idempotent (a record already
    reduced reduces to itself)."""
    sig = sig or {}

    def table(t):
        out = {}
        for k, v in (t or {}).items():
            v = v if isinstance(v, dict) else {}
            kinds = set()
            for h in v.get("how") or []:
                w = str(h).split(" ", 1)[0]
                kinds.add(w if w in _HOW_KINDS else "other")
            out[str(k)] = {"strength": v.get("strength"),
                           "how": sorted(kinds)}
        return out

    def names(x):
        return sorted(str(k) for k in (x or {}))

    ef = sig.get("embed_from") or {}
    return {"terms": table(sig.get("terms")),
            "artifacts": table(sig.get("artifacts")),
            "domains": sorted(str(d) for d in sig.get("domains") or []),
            "phases": names(sig.get("phases")),
            "situations": names(sig.get("situations")),
            "tools": sorted(str(t) for t in sig.get("tools") or [])[:40],
            "embed_from": {k: v for k, v in ef.items()
                           if isinstance(v, (int, float, bool)) or (
                               isinstance(v, dict) and all(isinstance(n, int)
                                                           for n in v.values()))
                           or k in ("digest", "cache")}}


# A line of CODE or of a tool's output inside a user turn (a CHOICE): a
# fenced block, an indented line, a line that opens with a declaration or a
# location, or ends like a statement. Durable records keep the user's
# words, never the code or the log they pasted.
_FENCED = re.compile(r"(?ms)^[ \t]{0,3}(`{3,}|~{3,}).*?(?:^[ \t]{0,3}\1[^\n]*$|\Z)")
_CODE_LINE = re.compile(
    r"^(?:\s{4,}|\t)\S"
    r"|^\s*(?:def|class)\s+\w+\s*[(:]"
    r"|^\s*import\s+(?:\{|\*|\w+\s*(?:,|from\b|$))|^\s*from\s+\S+\s+import\b"
    r"|^\s*(?:const|let|var)\s+[\w{\[]+\s*[:=]|^\s*(?:async\s+)?function\b\s*\w*\("
    r"|^\s*(?:fn|pub fn|impl|struct|enum)\s+\w+|^\s*#include\b"
    r"|[;{}]\s*$|=>|^\s*[)\]}]"
    r"|^\s*(?:at\s+\S+[:(]|File \"|Traceback\b|[\w./\\-]+\.\w{1,5}[:(]\d+)"
    r"|^\s*\w*(?:Error|Exception)\b:")


def prose_excerpt(text: str) -> str:
    """The user's words in a turn, without the code or output it carries
    (fenced blocks and code- or log-shaped lines dropped), whitespace kept
    per line. What a durable record may keep of a request (2026-09-27)."""
    t = _FENCED.sub("\n", text or "")
    keep = [ln for ln in t.split("\n") if ln.strip()
            and not _CODE_LINE.search(ln)]
    return "\n".join(" ".join(ln.split()) for ln in keep).strip()


# ---------------------------------------------------------------------------
# THE EXPLICIT ASK (operator, 2026-09-27: "ensure the engine isn't just
# injecting 'use this shit' blindly, it's more about I asked for it,
# reinforce it"). A framework, library, package or language the USER names
# in their own words in the current request is a FACT for that area -- a
# scoped package name, a library's name, "v10", "TSL", "koota", "WebGPU",
# "pmndrs math". Only their PROSE counts: a name that is merely present in a
# pasted manifest, lockfile, file listing, fence or log is not an ask (the
# manifest rule, _pinned, still reads those as project facts).
# ---------------------------------------------------------------------------
# A line of a pasted manifest, lockfile or file listing (a CHOICE): a JSON
# key line ("koota": "0.6.6"), a TOML/INI assignment, a lockfile entry
# (koota@0.6.6), a dependency-table heading, a tree line or a bare path.
_ARTEFACT_LINE = re.compile(
    r"""^\s*"[@\w./-]+"\s*:\s*|^\s*[@\w./-]+\s*=\s*["'{\[\d^~]"""
    r"|^\s*[@\w./-]+@[\^~<>=\d*]|^\s*\[?(?:dev|peer|optional)?[Dd]ependencies\]?\s*:?\s*$"
    r"|^\s*[│├└─|`+\\ -]*[\w@.-]*[/.][\w@./-]*/?\s*$"
    r"|^\s*[│├└]")
# A negation in front of a name ("without React", "not three.js", "instead
# of koota"): the user is asking NOT to use it. It reaches the name within
# its CLAUSE (no ., ;, :, !, ? or newline between): a structure, since
# 2026-09-27 -- the 24- and 60-character windows were ours
# (docs/CONSTANTS-AUDIT.md "_NEGATION + window").
_NEGATION = re.compile(
    r"\b(?:without|not|no|instead of|rather than|avoid(?:ing)?|don'?t use"
    r"|do not use|never use|drop(?:ping)?|remove|removing|replace|replacing"
    r"|migrate (?:away )?from|except)\b[^.;:!?\n]*$", re.I)
# A name the user OWNS already ("my React page", "our three.js scene",
# "this Rust error"): the project's context, not a stack choice -- the
# determiner, and at most one word, right before the name (the grammar of
# a noun phrase; the 24-character window was ours, removed 2026-09-27).
_CONTEXT_MENTION = re.compile(r"\b(?:my|our|this|these|the|their|your|"
                              r"existing|current)\s+(?:\w+\s+)?$", re.I)
# A version right after the name: "v10", "10", "0.6.6", "10.0.0-alpha.5".
_ASK_VERSION = re.compile(r"^\W{0,3}(?:v|version\s*)?(\d+(?:\.\d+)*"
                          r"(?:-[\w.]+)?)\b", re.I)


def user_prose(text: str) -> str:
    """The user's own words in a turn: fenced blocks, code- and log-shaped
    lines (prose_excerpt) and pasted manifest, lockfile and listing lines
    (_ARTEFACT_LINE) dropped."""
    t = prose_excerpt(text or "")
    return "\n".join(ln for ln in t.split("\n")
                     if not _ARTEFACT_LINE.search(ln))


# A HARNESS'S OWN TURN in the user role: a bracket tag opening the text --
# Hermes' "[CONTEXT COMPACTION -- REFERENCE ONLY]" summary, "[IMPORTANT:
# Background process ...]", "[STILL IN PROGRESS ...]", "[System: ...]". Its
# words are the harness's, not the user's ask (2026-09-27, pagoda-h4: the
# compaction summary's boilerplate -- "previous", "appear", "run", "note" --
# picked a TypeScript 6.0 craft for an asked slot). A structure: the tag's
# shape, no word list.
HARNESS_NOTICE = re.compile(r"^\s*\[(?:[A-Z][A-Za-z]*[ :\-—–]|[A-Z]{2,})")


def is_harness_notice(text: str) -> bool:
    return bool(HARNESS_NOTICE.match(text or ""))


def asked_terms(text: str) -> dict[str, dict]:
    """{term id: {form, version}} -- what the user ASKS for by name in
    their own words (never negated). `version` is the one written right
    after it ("r3f v10" -> "10"), or None. A harness's own turn (a bracket
    tag, is_harness_notice) asks for nothing."""
    refresh()
    if is_harness_notice(text):
        return {}
    prose = user_prose(text)
    out: dict[str, dict] = {}
    if not prose.strip():
        return out
    for t in VOCAB:
        for m in _WORDS[t.id].finditer(prose):
            if _NEGATION.search(_clause_before(prose, m.start())):
                continue
            v = _ASK_VERSION.match(prose[m.end():m.end() + 24])
            # CONTEXT, not a choice: "my React settings page", "in this
            # three.js scene" -- the project it is, not what to build it
            # with. Both are facts; a context mention's area gets a skill
            # only when the skill is about the request (skill_select).
            ctx = bool(_CONTEXT_MENTION.search(
                _clause_before(prose, m.start())))
            a = max(prose.rfind(x, 0, m.start()) for x in ".!?\n") + 1
            b = min([i for i in (prose.find(x, m.end()) for x in ".!?\n")
                     if i >= 0] or [len(prose)])
            out[t.id] = {"form": " ".join(m.group(0).split())[:40],
                         "version": v.group(1) if v else None,
                         "at": m.start(),
                         "mode": "context" if ctx else "choice",
                         "sentence": prose[a:b].strip()[:400]}
            break
    # A version that follows a parenthesised alias belongs to the name
    # before it: "r3f (react-three-fiber) v10" -- the parenthesis and the
    # version right after it (a structure; the 60- and 40-character windows
    # were ours, removed 2026-09-27).
    for tid, e in out.items():
        if e["version"] is None:
            rx = _WORDS[tid]
            for m in rx.finditer(prose):
                tail = prose[m.end():]
                mm = re.match(r"\s*\([^)\n]*\)\s*(?:v|version\s*)?"
                              r"(\d+(?:\.\d+)*(?:-[\w.]+)?)\b", tail, re.I)
                if mm:
                    e["version"] = mm.group(1)
                    break
    return out


def _clause_before(prose: str, at: int) -> str:
    """The text of `at`'s clause before it (back to the last ., ;, :, !, ?
    or newline)."""
    a = max(prose.rfind(x, 0, at) for x in ".;:!?\n") + 1
    return prose[a:at]


def negated_terms(text: str) -> set[str]:
    """The term ids the user names in their own words ONLY behind a
    negation ("without React or three.js", "instead of koota"): what the
    request asks NOT to use. A name also named plainly is not negated."""
    refresh()
    prose = user_prose(text)
    out: set[str] = set()
    for t in VOCAB:
        ms = list(_WORDS[t.id].finditer(prose))
        if ms and all(_NEGATION.search(_clause_before(prose, m.start()))
                      for m in ms):
            out.add(t.id)
    return out


# A request that says it is "vanilla", "plain" or "standalone" X is a stack
# without a framework layer on top: it rules out React and R3F as surely as
# "no React" does ("plain HTML and JavaScript", "vanilla JavaScript",
# "a standalone Three.js page"). The names are the stack words, not ours.
_VANILLA = re.compile(
    r"\b(?:vanilla|plain|pure|standalone)\s+(?:(?:html|css)\s*(?:,|and|\+|&)?"
    r"\s*)*(?:three\.?js|javascript|js|typescript|ts|html|webgl|webgpu)\b",
    re.I)
# The framework layers a vanilla stack leaves out.
_LAYERS = ("react", "r3f")
# What a name IMPLIES for a gate: React Three Fiber is React ("needs React"
# must hold for a request that names R3F).
IMPLIES_FOR_GATES = {"react": ("r3f",)}


# A negation that GOVERNS the name: right before it, or before a list it is in
# ("no React", "without React or three.js", "no React, no R3F", "instead of
# koota"). `_NEGATION` (negated_terms) reads any "not" earlier in the clause
# ("a function is not memoised ... React" is no negation of React); a gate
# that drops a skill must not.
_NEG_ADJ = re.compile(
    r"\b(?:without|no|instead of|rather than|avoid(?:ing)?|don'?t use|do not "
    r"use|never use|not using|not use|except|excluding)\s+"
    r"(?:(?:the|a|an|any|using|use|plain|vanilla|raw|my|our)\s+)?"
    r"(?:[\w.+/@-]+(?:\s+[\w.+/@-]+){0,2}\s*(?:,|/|\bor\b|\band\b)\s*"
    r"(?:(?:no|without)\s+)?)*$", re.I)


def _adjacent_negated(prose: str) -> set[str]:
    """The term ids every mention of which sits right behind a governing
    negation (_NEG_ADJ)."""
    out: set[str] = set()
    for t in VOCAB:
        ms = list(_WORDS[t.id].finditer(prose))
        if ms and all(_NEG_ADJ.search(prose[max(0, m.start() - 80):m.start()])
                      for m in ms):
            out.add(t.id)
    return out


def ruled_out(text: str) -> set[str]:
    """The term ids the request rules out: the names right behind a
    negation (_adjacent_negated) and, for a "vanilla" / "plain" stack, the
    framework layers it does not use -- unless the request plainly names
    one."""
    prose = user_prose(text)
    out = _adjacent_negated(prose)
    if prose.strip() and _VANILLA.search(prose):
        asked = asked_terms(text)
        for tid in _LAYERS:
            if tid not in asked:
                out.add(tid)
    return out


def _major(v) -> int | None:
    m = re.match(r"\D*(\d+)", str(v or ""))
    return int(m.group(1)) if m else None


def stated_versions(asked: dict, blob: str) -> dict[str, int]:
    """{term id: major} the request states: "R3F v10", "React 17" (asked_
    terms' version) and a pinned dependency (package.json, a spec's pin
    list). The user's words win over a pin."""
    out: dict[str, int] = {}
    for pkg, ver in _pinned(blob).items():
        tid = _PKG_TO.get(pkg)
        mj = _major(ver)
        if tid and mj is not None:
            out.setdefault(tid, mj)
    for tid, e in (asked or {}).items():
        mj = _major(e.get("version"))
        if mj is not None:
            out[tid] = mj
    return out


@functools.lru_cache(maxsize=8192)
def _versions_in(text: str) -> tuple[tuple[str, int], ...]:
    """((term id, major),) a text names ONE major for: "React Three Fiber
    v10", "React 19" -- the version right after the name, as asked_terms
    reads a request. A term with two different majors in the text is left
    out (a skill that names v9 and v10 is about the move, not a version)."""
    found: dict[str, set[int]] = {}
    for t in VOCAB:
        if t.kind != "framework":
            continue
        for m in _WORDS[t.id].finditer(text):
            v = _ASK_VERSION.match(text[m.end():m.end() + 24])
            mj = _major(v.group(1)) if v else None
            if mj is not None:
                found.setdefault(t.id, set()).add(mj)
    return tuple((tid, next(iter(s))) for tid, s in sorted(found.items())
                 if len(s) == 1)


def rule_versions(rule: dict) -> dict[str, int]:
    """The major version a skill is about, per framework: the rule's own
    `versions` ({term id: major}) when it has them, else the one major its
    "applies when" text and trigger texts name for a framework."""
    got: dict[str, int] = {}
    for tid, mj in ((rule or {}).get("versions") or {}).items():
        mm = _major(mj)
        if mm is not None:
            got[tid] = mm
    texts = [str((rule or {}).get("text") or "")]
    texts += [str(t.get("text") or "") for t in (rule or {}).get(
        "triggers") or [] if isinstance(t, dict)]
    seen: dict[str, set[int]] = {}
    for tx in texts:
        for tid, mj in _versions_in(tx):
            seen.setdefault(tid, set()).add(mj)
    for tid, s in seen.items():
        if len(s) == 1:
            got.setdefault(tid, next(iter(s)))
    return got


def match(rule: dict, sig: dict) -> dict:
    """{score, strength, why} for one skill rule against one request.

    The PRIMARY key must match: a non-code artifact when the rule names one
    (a slides skill applies to slides whatever the language); else its
    frameworks; else its languages; else `code`; else its domains. Anything
    else it names that also matches adds to the score. `strength` is the
    primary match's strongest evidence; None means no match.
    """
    none = {"score": 0.0, "strength": None, "why": []}
    g = gates(rule)
    terms, arts = sig.get("terms") or {}, sig.get("artifacts") or {}
    # GATES FIRST: each one the rule carries must hold, whatever the
    # primary key says.
    # A NEGATED name or a "vanilla" stack rules out the skills gated on it
    # ("no React" and a skill about React / R3F do not meet).
    out_ids = set(sig.get("ruled_out") or [])
    named = set(g["all_of"]) | set(applies_to(rule)["frameworks"])
    gone = sorted(out_ids & named)
    if gone:
        return dict(none, gate=f"the request rules out {names(gone)[0]}")
    # A VERSION the request states against the one the skill is about:
    # "R3F v9" does not meet a v10 skill, "React 17" does not meet React 19.
    sv = sig.get("versions") or {}
    for tid, major in rule_versions(rule).items():
        if tid in sv and sv[tid] != major:
            return dict(none, gate=f"the request is {names([tid])[0]} "
                        f"{sv[tid]}, the skill is {major}")
    for t in g["all_of"]:
        if t not in terms and t not in arts and not any(
                i in terms for i in IMPLIES_FOR_GATES.get(t, ())):
            return dict(none, gate=f"needs {names([t])[0]}")
    if g["phases"] and not set(g["phases"]) & set(sig.get("phases") or {}):
        return dict(none, gate=f"phase is not {'/'.join(g['phases'])}")
    have = set(sig.get("tools") or [])
    if g["tools_all"] and not set(g["tools_all"]) <= have:
        return dict(none, gate="the client does not offer "
                    + "/".join(sorted(set(g["tools_all"]) - have)))
    if g["tools_any"] and not set(g["tools_any"]) & have:
        return dict(none, gate="the client offers none of "
                    + "/".join(g["tools_any"]))
    if g["tools_none"] and set(g["tools_none"]) & have:
        return dict(none, gate="the client offers "
                    + "/".join(sorted(set(g["tools_none"]) & have)))
    if g["situations"] and not set(g["situations"]) & set(
            sig.get("situations") or {}):
        return dict(none, gate=f"no {'/'.join(g['situations'])}")
    topic_hits: list[str] = []
    code_hit = False
    if g["topics"]:
        text = sig.get("topic_text") or sig.get("query") or ""
        for t in g["topics"]:
            if _topic_rx(t).search(text):
                topic_hits.append(t)
                code_hit = code_hit or strong_topic(t, rule)
        if not topic_hits:
            return dict(none, gate="no topic of it appears")
        plain = [t for t in g["topics"] if not strong_topic(t, rule)]
        if not code_hit and len(topic_hits) < min(2, len(plain)):
            # One common word ("search", "state") is not evidence: plain
            # topics need two of them (replay of 2026-09-26: a single
            # plain word put an algorithms skill on a game spec). KEPT FOR
            # THE OPERATOR (docs/CONSTANTS-AUDIT.md "plain topics need two",
            # 2026-09-27): without it react-dev-blog-react-19-2-conditional-
            # rendering fails its own activation near miss and would be
            # quarantined on its next arm.
            return dict(none, gate="one plain topic word is not enough")
    base_m = _primary(rule, terms, arts, sig)
    a0 = applies_to(rule)
    if base_m["strength"] is None and not (a0["artifacts"] or a0["languages"]
                                           or a0["frameworks"]
                                           or (rule or {}).get("domains")) \
            and (g["tools_all"] or g["tools_any"]):
        # A rule keyed on the harness itself: the tools it offers are a fact
        # about the request (they are in the request body).
        base_m = {"score": float(STRENGTH["fact"]),
                  "strength": "fact",
                  "why": [f"tool {t}" for t in (g["tools_all"]
                                                or g["tools_any"])[:3]]}
    if base_m["strength"] is None and not code_hit:
        return base_m
    strength = base_m["strength"]
    if code_hit and base_m["strength"] is None:
        # A code-shaped topic stands alone only when the request does not
        # name OTHER languages or frameworks instead of the rule's own (a
        # TypeScript typing skill is no evidence for a vanilla-JS game
        # that uses the same DOM API). KEPT FOR THE OPERATOR
        # (docs/CONSTANTS-AUDIT.md "other-language gate", 2026-09-27):
        # without it the pmndrs math and koota crafts fail their own
        # activation near misses ("squaredDistance ... in this Rust crate")
        # and would be quarantined on their next arm.
        a = applies_to(rule)
        own = set(a["languages"] + a["frameworks"])
        if own and set(terms) and not own & set(terms):
            return dict(none, gate="the request names other languages")
    if code_hit:
        strength = "fact"
    elif topic_hits:
        if strength is None:
            return none
        # Plain words confirm the primary key; they never make it a fact:
        # the embedding stage or the fallback decides (verdict "ask").
        strength = "word"
    why = list(base_m["why"])
    why += [f"topic {t}" for t in topic_hits[:3]]
    why += [f"phase {p}" for p in g["phases"]
            if p in (sig.get("phases") or {})][:1]
    why += [f"situation {s}" for s in g["situations"]
            if s in (sig.get("situations") or {})][:1]
    # `score` is the strength's ORDINAL (fact 3, phrase 2, word 1): the
    # weights that summed topics, gates and fresh hits (1.0 / 0.25 / 0.5)
    # were ours (removed 2026-09-27, docs/CONSTANTS-AUDIT.md "match score
    # weights"); skill_select.confidence orders by strength, then the count
    # of code-shaped topics present, then the cosine.
    score = float(STRENGTH[strength])
    return {"score": score, "strength": strength, "why": why,
            "topics": topic_hits,
            "strong_topics": [t for t in topic_hits
                              if strong_topic(t, rule)]}


def _primary(rule: dict, terms: dict, arts: dict, sig: dict) -> dict:
    """The primary key's match (the pre-taxonomy rule, unchanged)."""
    a = applies_to(rule)
    noncode = [x for x in a["artifacts"] if x != "code"]

    def hits(ids, table):
        return [(i, table[i]) for i in ids if i in table]

    # The strength is kept (an ordinal); the base weights per key kind
    # (1.5 / 2.0 / 1.0 / 0.75, 0.25 per secondary, 0.5 per domain) were
    # ours (removed 2026-09-27, docs/CONSTANTS-AUDIT.md "_primary base
    # weights"): `score` is the ordinal.
    if rule.get("any"):
        prim = hits(a["frameworks"] + a["languages"], terms) + hits(
            noncode, arts)
        sec = []
    elif noncode:
        prim = hits(noncode, arts)
        sec = hits(a["frameworks"] + a["languages"], terms)
    elif a["frameworks"]:
        prim = hits(a["frameworks"], terms)
        sec = hits(a["languages"], terms)
    elif a["languages"]:
        prim, sec = hits(a["languages"], terms), []
    elif "code" in a["artifacts"]:
        prim, sec = hits(["code"], arts), []
    else:
        doms = sorted(set(rule.get("domains") or [])
                      & set(sig.get("domains") or []))
        if not doms:
            return {"score": 0.0, "strength": None, "why": []}
        return {"score": float(STRENGTH["word"]), "strength": "word",
                "why": [f"domain {d}" for d in doms]}
    if not prim:
        return {"score": 0.0, "strength": None, "why": []}
    strength = max((e["strength"] for _i, e in prim),
                   key=lambda s: STRENGTH[s])
    why = [f"{names([i])[0]} ({e['how'][0]})" for i, e in prim + sec]
    return {"score": float(STRENGTH[strength]), "strength": strength,
            "why": why}


if __name__ == "__main__":
    refresh()
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
            print(json.dumps(classify(f.read()), indent=2, default=list))
    else:
        for t in VOCAB:
            print(f"  {t.id:<13} {t.kind:<9} {t.name}")
        for a in ARTIFACTS:
            print(f"  {a.id:<13} artifact  {a.name}")
