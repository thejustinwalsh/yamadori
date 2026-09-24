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

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


VOCAB: tuple[Term, ...] = (
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
    Term("react", "React", "framework", r"\breact\b(?![- ]three)",
         packages=("react", "react-dom", "next"),
         domains=frozenset({"web-frontend", "ui-component"})),
    Term("r3f", "React Three Fiber", "framework",
         r"\breact[- ]three[- ]fiber\b|\br3f\b|@react-three/",
         packages=("@react-three/fiber", "@react-three/drei",
                   "@react-three/postprocessing"),
         domains=frozenset({"gpu", "web-frontend"})),
    Term("threejs", "three.js", "framework",
         r"\bthree\.js\b|\bthreejs\b|\bTSL\b|\bthree/(?:webgpu|tsl)\b",
         packages=("three", "@types/three"),
         domains=frozenset({"gpu", "web-frontend"})),
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

VERBS = (r"\b(?:write|writing|create|creating|generate|make|making|draft"
         r"|drafting|build|building|add|update|edit|editing|fix|refactor"
         r"|rewrite|review|improve|design|designing|produce|author|convert"
         r"|format|restyle|document)\b")

BY_ID = {t.id: t for t in VOCAB}
ART_BY_ID = {a.id: a for a in ARTIFACTS}
_WORDS = {t.id: re.compile(t.words, 0 if t.case else re.I) for t in VOCAB}
_NOUNS = {a.id: re.compile(a.nouns, re.I) for a in ARTIFACTS}
_PHRASE = {a.id: re.compile(VERBS + r"(?:\W+\w+){0,5}?\W+(?:" + a.nouns + ")",
                            re.I) for a in ARTIFACTS}
_APATH = {a.id: re.compile(a.paths) for a in ARTIFACTS if a.paths}
_FENCE_TO = {f: t.id for t in VOCAB for f in t.fences}
_PKG_TO = {p: t.id for t in VOCAB for p in t.packages}
_EXT_TO = {e: t.id for t in VOCAB for e in t.exts}
_EXT_ART = {e: a.id for a in ARTIFACTS for e in a.exts}
_ALL_EXT = sorted(set(_EXT_TO) | set(_EXT_ART), key=len, reverse=True)
# A file path shape: a word character, a dot, a known extension, a boundary.
# Flat text (PROTOCOL rule 8 allows it for path shapes).
_EXT_RX = re.compile(r"(?<=[\w\]\)])\.(" + "|".join(map(re.escape, _ALL_EXT))
                     + r")\b(?![\w-])")
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
    """The human `applies when:` line for a rule."""
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
    return not (a["artifacts"] or a["languages"] or a["frameworks"]
                or (rule or {}).get("domains"))


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
    """Trigger lines with verbatim quotes: the description's sentences, then
    the bullets under a "When to use" heading."""
    out: list[dict] = []
    desc = fields.get("description")
    if isinstance(desc, str) and desc.strip():
        d = desc.strip()[:L.DESCRIPTION_CHARS]
        for sent in _sentences(d):
            out.append({"text": sent[:L.TRIGGER_CHARS], "quote": sent,
                        "origin": "description"})
    for m in _WHEN_HEAD.finditer(text):
        rest = text[m.end():]
        stop = re.search(r"^\s{0,3}#{1,6}\s", rest, re.M)
        block = rest[:stop.start() if stop else 2000]
        for ln in block.split("\n"):
            mm = re.match(r"^\s*(?:[-*+]|\d+\.)\s+(.{8,})$", ln)
            if mm:
                t = mm.group(1).strip()
                out.append({"text": t[:L.TRIGGER_CHARS], "quote": t,
                            "origin": "when-to-use"})
    seen, uniq = set(), []
    for t in out:
        k = t["text"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:L.MAX_TRIGGERS]


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


def rule_of_condition(line: str) -> dict:
    """An `applies when:` line a PERSON wrote, read back into a rule. `any`:
    "React or TypeScript" means either one."""
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
    rule["triggers"] = [{"text": line.strip()[:L.TRIGGER_CHARS], "quote": None,
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
    return out[-50:]


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            t = _text_of(m)
            if t.strip():
                return t
    return ""


def request_signals(messages: list[dict],
                    route_class: str | None = None) -> dict:
    """{terms: {id: {strength, how: [...]}}, artifacts: {id: {...}},
    domains: [...], query: text to embed}."""
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
        m = _WORDS[t.id].search(blob)
        if m:
            add(terms, t.id, "word", f"word {m.group(0)!r}")
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
    return {"terms": terms, "artifacts": arts, "domains": doms,
            "query": last[:2000]}


def match(rule: dict, sig: dict) -> dict:
    """{score, strength, why} for one skill rule against one request.

    The PRIMARY key must match: a non-code artifact when the rule names one
    (a slides skill applies to slides whatever the language); else its
    frameworks; else its languages; else `code`; else its domains. Anything
    else it names that also matches adds to the score. `strength` is the
    primary match's strongest evidence; None means no match.
    """
    a = applies_to(rule)
    terms, arts = sig.get("terms") or {}, sig.get("artifacts") or {}
    noncode = [x for x in a["artifacts"] if x != "code"]

    def hits(ids, table):
        return [(i, table[i]) for i in ids if i in table]

    if rule.get("any"):
        prim = hits(a["frameworks"] + a["languages"], terms) + hits(
            noncode, arts)
        base, sec = 1.5, []
    elif noncode:
        prim, base = hits(noncode, arts), 1.5
        sec = hits(a["frameworks"] + a["languages"], terms)
    elif a["frameworks"]:
        prim, base = hits(a["frameworks"], terms), 2.0
        sec = hits(a["languages"], terms)
    elif a["languages"]:
        prim, base, sec = hits(a["languages"], terms), 1.0, []
    elif "code" in a["artifacts"]:
        prim, base, sec = hits(["code"], arts), 0.75, []
    else:
        doms = sorted(set(rule.get("domains") or [])
                      & set(sig.get("domains") or []))
        if not doms:
            return {"score": 0.0, "strength": None, "why": []}
        return {"score": 0.5 * len(doms), "strength": "word",
                "why": [f"domain {d}" for d in doms]}
    if not prim:
        return {"score": 0.0, "strength": None, "why": []}
    strength = max((e["strength"] for _i, e in prim),
                   key=lambda s: STRENGTH[s])
    why = [f"{names([i])[0]} ({e['how'][0]})" for i, e in prim + sec]
    return {"score": base * len(prim) + 0.25 * len(sec),
            "strength": strength, "why": why}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
            print(json.dumps(classify(f.read()), indent=2, default=list))
    else:
        for t in VOCAB:
            print(f"  {t.id:<13} {t.kind:<9} {t.name}")
        for a in ARTIFACTS:
            print(f"  {a.id:<13} artifact  {a.name}")
