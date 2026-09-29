#!/usr/bin/env python
"""THE PACKAGE REGISTRY: which packages the detector, the taxonomy and the
selector know, as data (docs/PACKAGE-ONBOARDING.md section 5, item F).

    python mcp/package_registry.py show       the merged registry
    python mcp/package_registry.py migrate    write packages.json from SEED

Four files in DIR (YAMADORI_PKG_REGISTRY_DIR, else index/packages -- the
same directory deps.STORE defaults to), each written atomically (a temp
file, then os.replace):

  packages.json         {npm name: entry}: the packages onboarding added,
                        and extra versions / onboarding / licence of a SEED
                        package. Absent: SEED alone, which is exactly the
                        tables that were hand-written before (below), so a
                        missing file changes nothing.
  held.json             the PROMOTED held manifest: {"dirs": [source dir
                        basenames under skill_packages.SRC], "packages":
                        {npm name: [versions]}, "signature", "promoted_at",
                        "by", ...}. skill_packages.held() reads it; absent,
                        the glob of SRC as before. freeze_held() writes it
                        from today's glob, so a package fetched later is not
                        held until promoted.
  vocabulary.json       the promoted vocabulary (skill_packages.vocabulary()'s
                        shape) with the held signature; skill_packages loads
                        it (re-read when its stat changes) and never builds
                        on the request path while it exists.
  vocabulary.next.json  the CANDIDATE build_candidate() wrote: the vocabulary,
                        its held dirs and entries, its signature and the DIFF
                        (per added package; the names existing packages lose
                        or gain). The operator's force reads it
                        (next_candidate()).

ENTRY: {"term", "label", "aliases", "area", "versions", "built_on",
"built_on_from", "canonical"?, "onboarding"?, "licence"?, "words"?,
"seed", "names_term", "names_area", "term_note"?}. `term` is the
skill_classify taxonomy term the package is filed under (a SEED package's
hand-written term; a new package's slug_term()); `area` the selector's
package area (skill_match.PACKAGE_AREA); `names_term` / `names_area`: the
package is the one its term / area names (skill_packages.TERM_PACKAGE,
skill_match.AREA_PACKAGE); `words` the taxonomy term's WORD rule
(words_rule(), stored at add time so no request builds the English lexicon).

THE OPERATOR'S RULES (2026-09-27) this module applies:
  4  a candidate vocabulary is promoted only if the standing detection
     labels (bench/skills/package_detect_labels.jsonl) lose no true positive
     and gain no false positive (floor()); held otherwise, recorded with the
     rows and names, and the operator may force it (promote(forced=True)).
  7  the taxonomy grows from package names plus the aliases the operator
     typed: one framework term per onboarded package, its words from the
     exact npm name and those aliases only; a one-word npm name that is an
     English word (skill_packages.is_english over prose_words() |
     tokenizer_words()) gets pmndrs_math's treatment -- only `name@<digit>`,
     a subpath `name/<x>`, `npm install name` / `npm i name`.
  -  no IMPLIES row is ever added (skill_select.IMPLIES is not touched here).
  9  no size cap.

A CANDIDATE is evaluated in this process through skill_packages.using() and
using() here: both are PROCESS-GLOBAL while they are entered. using() holds
the registry lock for its whole block, so another thread's load() /
skill_classify.refresh() WAITS for the evaluation to end rather than seeing
the candidate; a thread that reads skill_classify's tables or
skill_packages' vocabulary directly in that window can see it. The
onboarding vocab stage runs in the worker, not the proxy.

Imports at module level are stdlib only: skill_classify imports this module
while it is itself being imported.
"""
from __future__ import annotations

import contextlib
import copy
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import threading
from collections.abc import Mapping, Set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DIR = os.environ.get("YAMADORI_PKG_REGISTRY_DIR") or os.path.join(
    ROOT, "index", "packages")
PACKAGES = os.path.join(DIR, "packages.json")
HELD = os.path.join(DIR, "held.json")
VOCABULARY = os.path.join(DIR, "vocabulary.json")
NEXT = os.path.join(DIR, "vocabulary.next.json")

# The standing detection labels (operator decision 4) and the system text
# bench/skills/eval_packages.py wraps each of them in (replay_selection.
# DAILY_SYSTEM).
LABELS = os.path.join(ROOT, "bench", "skills", "package_detect_labels.jsonl")
LABEL_SYSTEM = ("You are Hermes, an AI coding agent. You work in the user's "
                "project with your tools.")

TSL = "three/tsl"          # skill_packages.TSL: three's TSL unit, its own row

# ---------------------------------------------------------------------------
# SEED: the hand-written rows, migrated from skill_packages.TERM_PACKAGE,
# skill_packages.CANONICAL, skill_match.PACKAGE_AREA / AREA_PACKAGE,
# bench/skills/eval_packages.py PACKAGE_AREA (the same map) and
# skill_select.BUILT_ON. Kept in code so a missing packages.json changes
# nothing.
# ---------------------------------------------------------------------------
_R3F_BUILT_ON = ('@react-three/fiber readme: "react-three-fiber is a React '
                 'renderer for threejs" (skill_select.BUILT_ON, hand row)')
_TGPU_BUILT_ON = ('typegpu readme: "TypeGPU is a modular and open-ended '
                  'toolkit for WebGPU" (skill_select.BUILT_ON, hand row)')


def _seed(term, label, area, *, names_term=False, names_area=False,
          built_on=(), built_on_from="", canonical=None) -> dict:
    e = {"term": term, "label": label, "aliases": [], "area": area,
         "versions": [], "built_on": list(built_on),
         "built_on_from": built_on_from, "seed": True,
         "names_term": names_term, "names_area": names_area}
    if canonical is not None:
        e["canonical"] = canonical
    return e


SEED: dict[str, dict] = {
    "@react-three/fiber": _seed(
        "r3f", "React Three Fiber", "r3f", names_term=True, names_area=True,
        built_on=("react", "three"), built_on_from=_R3F_BUILT_ON,
        canonical={"skill": "r3f-v10-setup-21",
                   "by_major": {"9": "r3f-v9-setup-22",
                                "10": "r3f-v10-setup-21"},
                   "why": "the setup skill of the major in play (v10 "
                          "when no version is known: the newest "
                          "held)"}),
    "@react-three/drei": _seed(
        "r3f", "drei", "r3f",
        canonical={"skill": "r3f-drei-pairing-8",
                   "why": "drei's version pairing with fiber"}),
    "@react-three/postprocessing": _seed(
        "r3f", "@react-three/postprocessing", "r3f"),
    "koota": _seed(
        "koota", "Koota", "koota", names_term=True, names_area=True,
        canonical={"skill": "koota-traits-and-entities",
                   "why": "pmndrs/koota skills/koota/SKILL.md @ v0.6.6, first "
                          "section (Glossary / Trait types / Entities)"}),
    "math": _seed(
        "pmndrs_math", "math (pmndrs)", "pmndrs_math", names_term=True,
        names_area=True,
        canonical={"skill": "math-data-oriented-functions",
                   "why": "pmndrs/math skills/math/SKILL.md @ 0.1.0, first "
                          "section (Types / Style)"}),
    "three": _seed("threejs", "three.js", "threejs", names_term=True),
    TSL: _seed(
        "threejs", "three.js TSL", "threejs", names_area=True,
        canonical={"skill": "threejs-llms-full-tsl-e-g",
                   "why": "the TSL skill skill_select.IMPLIES already names "
                          "for TSL (three.js docs/llms-full.txt); no TSL "
                          "overview skill is in the store"}),
    "typegpu": _seed(
        "typegpu", "TypeGPU", "typegpu", names_term=True, names_area=True,
        built_on=("@webgpu/types",), built_on_from=_TGPU_BUILT_ON,
        canonical={"skill": "typegpu-migration-0-12-import-pattern-2",
                   "why": "the only TypeGPU skill about using the package as "
                          "a whole (its import pattern at 0.12, the held "
                          "version)"}),
    "@typegpu/three": _seed(
        "typegpu", "@typegpu/three", "typegpu",
        canonical={"skill": "typegpu-three-tsl-integration-2",
                   "why": "the @typegpu/three docs' skill"}),
    "@typegpu/noise": _seed("typegpu", "@typegpu/noise", "typegpu"),
}
# What a packages.json row may change on a SEED package.
SEED_FILE_KEYS = ("versions", "onboarding", "licence")


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _stat_key(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, getattr(st, "st_ino", 0))


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: str, obj) -> None:
    """Atomic: a temp file beside it, then os.replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _uniq(xs) -> list:
    out = []
    for x in xs or []:
        if x not in out:
            out.append(x)
    return out


def npm_of_dir(basename: str) -> tuple[str, str]:
    """('@types/three', '0.186.0') for 'types__three@0.186.0' (deps.slug's
    inverse, WITHOUT folding @types into the package it types --
    skill_packages.package_of_dir does that)."""
    name, _, ver = os.path.basename(basename).rpartition("@")
    return ("@" + name.replace("__", "/") if "__" in name else name), ver


# ---------------------------------------------------------------------------
# THE OVERLAY (a candidate's entries, while floor() evaluates it).
# ---------------------------------------------------------------------------
_LOCK = threading.RLock()
_OVERLAY: list[dict] = []
_GEN = [0]


@contextlib.contextmanager
def using(entries=()):
    """Evaluate with `entries` added to the registry (process-global while
    entered; see the module docstring)."""
    global _OVERLAY
    with _LOCK:
        prev = _OVERLAY
        _OVERLAY = [dict(e) for e in entries or ()]
        _GEN[0] += 1
        try:
            yield
        finally:
            _OVERLAY = prev
            _GEN[0] += 1


def state_key():
    """Changes whenever what load() returns may change: packages.json's stat
    and the overlay's generation. One os.stat."""
    return (PACKAGES, _stat_key(PACKAGES), _GEN[0])


# ---------------------------------------------------------------------------
# load() and the tables derived from it.
# ---------------------------------------------------------------------------
_CACHE: dict = {"key": object(), "data": {}, "derived": {}}


def _normalise(name: str, e: dict) -> dict:
    """A non-seed entry with every key present (a hand-written row with no
    term gets its bare slug: add_package() is where collisions are
    resolved)."""
    term = e.get("term") or _slug(name)
    out = {"term": term, "label": e.get("label") or name,
           "aliases": [str(a) for a in e.get("aliases") or [] if str(a).strip()],
           "area": e.get("area", term),
           "versions": _uniq(e.get("versions") or []),
           "built_on": _uniq(e.get("built_on") or []),
           "built_on_from": e.get("built_on_from") or "",
           "seed": False,
           "names_term": bool(e.get("names_term", True)),
           "names_area": bool(e.get("names_area", True))}
    for k in ("canonical", "onboarding", "licence", "words", "term_note",
              "domains"):
        if e.get(k) is not None:
            out[k] = copy.deepcopy(e[k])
    return out


def _merge(data: dict, name: str, row: dict) -> None:
    if not isinstance(row, dict):
        return
    if name in SEED:
        e = data[name]
        for k in SEED_FILE_KEYS:
            if k == "versions":
                e["versions"] = _uniq(list(e["versions"])
                                      + list(row.get("versions") or []))
            elif row.get(k) is not None:
                e[k] = copy.deepcopy(row[k])
        return
    if name in data:          # a second row for a registered package
        e = data[name]
        e["versions"] = _uniq(list(e["versions"])
                              + list(row.get("versions") or []))
        for k in ("onboarding", "licence"):
            if row.get(k) is not None:
                e[k] = copy.deepcopy(row[k])
        return
    data[name] = _normalise(name, row)


def _load() -> dict:
    key = state_key()
    with _LOCK:
        if _CACHE["key"] == key:
            return _CACHE["data"]
        data = copy.deepcopy(SEED)
        raw = _read_json(PACKAGES)
        if isinstance(raw, dict):
            for name in sorted(raw):
                _merge(data, name, raw[name])
        for e in _OVERLAY:
            name = e.get("name")
            if name:
                _merge(data, name, e)
        _CACHE.update(key=key, data=data, derived={})
        return data


def load() -> dict:
    """{npm name: entry}: SEED merged with packages.json (and a candidate's
    entries while using() is entered). Re-read when the file's stat
    changes. A copy: the caller may change it."""
    return copy.deepcopy(_load())


def entry(name: str) -> dict | None:
    e = _load().get(name)
    return copy.deepcopy(e) if e is not None else None


def _derived(name: str, fn):
    data = _load()
    with _LOCK:
        d = _CACHE["derived"]
        if name not in d:
            d[name] = fn(data)
        return d[name]


def _term_package(data: dict) -> dict:
    out: dict = {}
    for name, e in data.items():
        if e.get("names_term") and e.get("term"):
            out.setdefault(e["term"], name)
    return out


def _package_area(data: dict) -> dict:
    return {name: e["area"] for name, e in data.items() if e.get("area")}


def _area_package(data: dict) -> dict:
    out: dict = {}
    for name, e in data.items():
        if e.get("area") and e.get("names_area"):
            out.setdefault(e["area"], name)
    # Every area a package is filed under has a package that names it
    # (skill_match.plan_turn reads AREA_PACKAGE[area] for each).
    for name, e in data.items():
        if e.get("area"):
            out.setdefault(e["area"], name)
    return out


def _canonical(data: dict) -> dict:
    return {name: e["canonical"] for name, e in data.items()
            if isinstance(e.get("canonical"), dict)}


def _hand_term_of_package(npm: str) -> str | None:
    """The hand-written taxonomy term that files `npm` (skill_classify's
    Term.packages), if any."""
    import skill_classify
    for t in skill_classify.HAND_VOCAB:
        if npm in t.packages:
            return t.id
    return None


def _built_on_terms(data: dict) -> dict:
    out: dict[str, list] = {}
    for name, e in data.items():
        term = e.get("term")
        if not term or not e.get("built_on"):
            continue
        for b in e["built_on"]:
            be = data.get(b)
            bt = be.get("term") if be else _hand_term_of_package(b)
            if bt and bt != term:
                row = out.setdefault(term, [])
                if bt not in row:
                    row.append(bt)
    return {t: tuple(v) for t, v in out.items()}


def term_package() -> dict:
    """{taxonomy term: the npm name it names} (skill_packages.TERM_PACKAGE)."""
    return dict(_derived("term_package", _term_package))


def package_area() -> dict:
    """{npm name: selector area} (skill_match.PACKAGE_AREA)."""
    return dict(_derived("package_area", _package_area))


def area_package() -> dict:
    """{selector area: npm name} (skill_match.AREA_PACKAGE)."""
    return dict(_derived("area_package", _area_package))


def canonical() -> dict:
    """{npm name: canonical skill row} (skill_packages.CANONICAL)."""
    return copy.deepcopy(_derived("canonical", _canonical))


def built_on_terms() -> dict:
    """{term: (terms it is built on)} (skill_select.BUILT_ON): each entry's
    `built_on` npm names mapped to their terms (a registered package's term,
    else the hand-written term that files it)."""
    return dict(_derived("built_on", _built_on_terms))


class LiveMap(Mapping):
    """A read-only mapping that reads the registry on every access (one
    os.stat; the table is rebuilt only when the registry changed). Keeps a
    module-level name (TERM_PACKAGE, AREA_PACKAGE ...) importable and
    current in a long-running process."""

    def __init__(self, table: str, fn):
        self._table, self._fn = table, fn

    def _d(self) -> dict:
        return _derived(self._table, self._fn)

    def __getitem__(self, k):
        return self._d()[k]

    def __contains__(self, k):
        return k in self._d()

    def __iter__(self):
        return iter(list(self._d()))

    def __len__(self):
        return len(self._d())

    def __repr__(self):
        return f"LiveMap({self._d()!r})"


class LiveKeys(Set):
    """The keys of a registry table, read live (skill_packages.
    PACKAGE_AREAS)."""

    def __init__(self, table: str, fn):
        self._m = LiveMap(table, fn)

    def __contains__(self, k):
        return k in self._m

    def __iter__(self):
        return iter(self._m)

    def __len__(self):
        return len(self._m)

    def __repr__(self):
        return f"LiveKeys({sorted(self._m)!r})"


def live(table: str) -> LiveMap:
    """LiveMap over one of: term_package, package_area, area_package,
    canonical, built_on."""
    fns = {"term_package": _term_package, "package_area": _package_area,
           "area_package": _area_package, "canonical": _canonical,
           "built_on": _built_on_terms}
    return LiveMap(table, fns[table])


# ---------------------------------------------------------------------------
# The taxonomy term and its WORD rule (operator decision 7).
# ---------------------------------------------------------------------------
def _slug(npm: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", npm.lower().lstrip("@")).strip("_")
    return s or "package"


def _term_for(name: str, data: dict | None = None) -> tuple[str, str]:
    """(term, note). A registered package keeps its term. A package a
    hand-written term already files (skill_classify Term.packages) is that
    term. Else its slug, suffixed `_pkg`, `_pkg2` ... when the slug is
    already a term of something else -- and the note says so."""
    data = _load() if data is None else data
    if name in data:
        return data[name]["term"], ""
    hand = _hand_term_of_package(name)
    if hand:
        return hand, (f"filed under skill_classify's term {hand!r}, which "
                      f"already lists {name!r} among its packages")
    import skill_classify
    taken = {t.id: t for t in skill_classify.HAND_VOCAB}
    used = {e["term"] for n, e in data.items() if n != name}
    base = _slug(name)
    if base not in taken and base not in used:
        return base, ""
    k = 1
    while True:
        cand = f"{base}_pkg" + ("" if k == 1 else str(k))
        if cand not in taken and cand not in used:
            what = (f"skill_classify's {taken[base].kind} term"
                    if base in taken else "another registered package's term")
            return cand, (f"slug {base!r} is {what}; suffixed "
                          f"deterministically to {cand!r}")
        k += 1


def slug_term(npm_name: str) -> str:
    """The taxonomy term id for an npm name: "@pmndrs/uikit" ->
    "pmndrs_uikit", "three-flatland" -> "three_flatland". Never another
    package's term (see _term_for)."""
    return _term_for(npm_name)[0]


_ENGLISH: dict = {"words": None}


def english_words() -> set[str]:
    """skill_packages' English lexicon: prose_words() | tokenizer_words()
    (read once per process: seconds, never on the request path -- the rule
    is stored in the entry when the package is added)."""
    with _LOCK:
        if _ENGLISH["words"] is None:
            import skill_packages as SP
            _ENGLISH["words"] = SP.prose_words() | SP.tokenizer_words()
        return _ENGLISH["words"]


def words_rule(name: str, aliases=(), *, english: set | None = None
               ) -> dict:
    """{"regex", "case", "why"}: the taxonomy term's WORD rule, from the
    exact npm name and the operator's typed aliases only (operator decision
    7). A ONE-WORD npm name (lower-case letters and digits, no scope, no
    separator) that is an English word gets only the forms that cannot mean
    anything else: `name@<digit>`, a subpath `name/<x>`, `npm install name`
    / `npm i name` -- skill_classify's hand rule for `math`. Any other name
    matches as written (case-insensitive: npm names are lower-case), not
    inside a longer name. Aliases are used AS TYPED: escaped, word-bounded,
    case as typed."""
    import skill_packages as SP
    esc = re.escape(name)
    one_word = re.fullmatch(r"[a-z][a-z0-9]*", name) is not None
    lex = english_words() if english is None else english
    is_eng = one_word and SP.is_english(name, lex)
    if is_eng:
        parts = [rf"(?<![\w.@/-]){esc}@\d",
                 rf"(?<![\w.@/-]){esc}/[A-Za-z_$][\w.$-]*",
                 rf"\bnpm\s+(?:install|i)\s+[`'\"]?{esc}(?![\w/-])"]
        why = [f"npm name {name!r} is a one-word English word "
               "(skill_packages.is_english over prose_words | "
               "tokenizer_words): only name@<digit>, name/<subpath>, "
               "npm install|i name"]
    else:
        parts = [rf"(?i:(?<![\w.@/-]){esc}(?![\w-]))"]
        why = [f"the exact npm name {name!r}"
               + ("" if not one_word else " (not an English word)")]
    typed = [a for a in (str(x).strip() for x in aliases or ()) if a]
    for a in typed:
        parts.append(rf"(?<!\w){re.escape(a)}(?!\w)")
    if typed:
        why.append("the aliases the operator typed, as typed: "
                   + ", ".join(repr(a) for a in typed))
    rx = "|".join(parts)
    re.compile(rx)
    return {"regex": rx, "case": True, "why": "; ".join(why)}


def catalogue_sha() -> str:
    """8 hex characters over the taxonomy catalogue the tag prompt renders
    (skill_prompts.tag_catalogue())."""
    import skill_prompts
    return hashlib.sha256(skill_prompts.tag_catalogue().encode(
        "utf-8")).hexdigest()[:8]


def add_package(name: str, *, version: str, aliases=(), built_on=(),
                built_on_from: str = "", onboarding=None, licence=None,
                label: str | None = None) -> dict:
    """The registry entry for a package (with its `name`), NOT written:
    build_candidate() takes it and promote() writes it. A SEED package
    only gains the version, onboarding and licence."""
    data = _load()
    if name in data:
        e = copy.deepcopy(data[name])
        e["versions"] = _uniq(list(e["versions"]) + [version])
        if onboarding is not None:
            e["onboarding"] = onboarding
        if licence is not None:
            e["licence"] = licence
        if not e.get("seed"):
            e["aliases"] = _uniq(list(e.get("aliases") or [])
                                 + [str(a) for a in aliases or ()
                                    if str(a).strip()])
            e["words"] = words_rule(name, e["aliases"])
        e["name"] = name
        return e
    term, note = _term_for(name, data)
    hand = _hand_term_of_package(name)
    ali = _uniq(str(a) for a in aliases or () if str(a).strip())
    e = {"name": name, "term": term, "label": label or name,
         "aliases": ali,
         # A package a hand-written, non-package term already files (react,
         # tailwind, webgpu ...) opens no package area of its own.
         "area": None if hand else term,
         "versions": [version], "built_on": _uniq(built_on),
         "built_on_from": built_on_from, "seed": False,
         "names_term": not hand, "names_area": not hand,
         "words": words_rule(name, ali)}
    if note:
        e["term_note"] = note
    if onboarding is not None:
        e["onboarding"] = onboarding
    if licence is not None:
        e["licence"] = licence
    return e


# ---------------------------------------------------------------------------
# The held manifest, the candidate vocabulary, the floor, promotion.
# ---------------------------------------------------------------------------
def held_manifest() -> dict | None:
    m = _read_json(HELD)
    return m if isinstance(m, dict) and isinstance(m.get("dirs"), list) \
        else None


def _src() -> str:
    import skill_packages as SP
    return SP.SRC


def _glob_dirs() -> list[str]:
    src = _src()
    try:
        names = os.listdir(src)
    except OSError:
        return []
    return sorted(n for n in names if "@" in n
                  and os.path.isdir(os.path.join(src, n)))


def current_held_dirs() -> list[str]:
    m = held_manifest()
    return list(m["dirs"]) if m else _glob_dirs()


def _packages_of(dirs) -> dict:
    out: dict[str, list] = {}
    for d in dirs:
        n, v = npm_of_dir(d)
        out.setdefault(n, []).append(v)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _listing(dirs) -> list:
    src = _src()
    rows = []
    for d in sorted(dirs):
        root = os.path.join(src, d)
        for dp, dn, fn in os.walk(root):
            dn.sort()
            for f in sorted(fn):
                p = os.path.join(dp, f)
                try:
                    size = os.path.getsize(p)
                except OSError:
                    continue
                rows.append([d, os.path.relpath(p, root).replace("\\", "/"),
                             size])
    return rows


def signature(dirs, entries=()) -> str:
    """sha256 over the held dirs' file listing (path and size) and the
    registry entries."""
    h = hashlib.sha256()
    h.update(json.dumps(_listing(dirs), separators=(",", ":")).encode())
    h.update(json.dumps(sorted((dict(e) for e in entries or ()),
                               key=lambda e: str(e.get("name"))),
                        sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def freeze_held(by: str = "freeze") -> dict:
    """held.json from today's glob of the source dir, when it is absent
    (idempotent: an existing manifest is returned as is)."""
    with _LOCK:
        m = held_manifest()
        if m:
            return m
        dirs = _glob_dirs()
        m = {"dirs": dirs, "packages": _packages_of(dirs),
             "signature": signature(dirs), "promoted_at": _now(), "by": by,
             "why": "frozen from the source dir as found (the packages held "
                    "before onboarding)"}
        _write_json(HELD, m)
        return m


def _by_package(symbols: dict) -> dict:
    out: dict[str, set] = {}
    for n, p in symbols.items():
        out.setdefault(p, set()).add(n)
    return out


def _diff(current: dict, cand: dict, added: set) -> dict:
    per = {}
    code_only = _by_package({n: p for n, p in cand["code_symbols"].items()
                             if n not in cand["symbols"]})
    for p in sorted(added):
        c = cand["per_package"].get(p)
        if c is None:
            continue
        per[p] = {"names": c["names"], "unique": c["unique"],
                  "code_only": len(code_only.get(p, ())),
                  "dropped_shared": c["dropped_shared"],
                  "dropped_platform": c["dropped_platform"],
                  "dropped_english": c["dropped_english"]}
    cur_by = _by_package(current.get("code_symbols")
                         or current.get("symbols") or {})
    new_by = _by_package(cand["code_symbols"])
    lost, gained = {}, {}
    # Every package held now (a new VERSION of one is listed too: what its
    # old version detected and the new one does not).
    for p in sorted(set(current.get("packages") or []) | set(cur_by)):
        a, b = cur_by.get(p, set()), new_by.get(p, set())
        if a - b:
            lost[p] = sorted(a - b)
        if b - a:
            gained[p] = sorted(b - a)
    return {"per_package": per, "lost": lost, "gained": gained}


def build_candidate(add_dirs=(), remove_dirs=(), entries=()) -> dict:
    """The CANDIDATE vocabulary: the held dirs with `add_dirs` added and
    `remove_dirs` removed, the registry `entries` to add, its signature and
    the diff against the current vocabulary. Writes vocabulary.next.json;
    changes nothing the process serves from."""
    import skill_packages as SP
    src = _src()
    missing = [d for d in add_dirs or ()
               if not os.path.isdir(os.path.join(src, d))]
    if missing:
        raise ValueError(f"not in the source dir {src}: {missing}")
    dirs = sorted((set(current_held_dirs()) - set(remove_dirs or ()))
                  | set(add_dirs or ()))
    ents = []
    for e in entries or ():
        e = dict(e)
        if not e.get("name"):
            raise ValueError("a registry entry needs its npm `name`")
        ents.append(e)
    vocab = SP.build_vocabulary(SP.held_from_dirs(dirs))
    current = SP.vocabulary()
    added = {SP.package_of_dir(d)[0] for d in add_dirs or ()}
    if "three" in added:
        added.add(SP.TSL)
    cand = {"vocabulary": vocab, "held": dirs, "entries": ents,
            "signature": signature(dirs, ents),
            "diff": _diff(current, vocab, added),
            "added_dirs": sorted(add_dirs or ()),
            "removed_dirs": sorted(remove_dirs or ()),
            "built_at": _now()}
    _write_json(NEXT, cand)
    return cand


def next_candidate() -> dict | None:
    c = _read_json(NEXT)
    return c if isinstance(c, dict) else None


def label_rows(path: str | None = None) -> list[dict]:
    out = []
    with open(path or LABELS, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                out.append(json.loads(ln))
    return out


def floor(candidate: dict, labels: str | None = None) -> dict:
    """Operator decision 4: detection over the standing labels, under the
    current vocabulary and registry and under the candidate's. Held when a
    row's expected package is detected now and not by the candidate (a lost
    true positive) or an unexpected package is detected by the candidate and
    not now (a gained false positive)."""
    import skill_packages as SP
    path = labels or LABELS
    rows = label_rows(path)

    def msgs(text):
        return [{"role": "system", "content": LABEL_SYSTEM},
                {"role": "user", "content": text}]
    now = {r["id"]: set(SP.detect(msgs(r["text"]))) for r in rows}
    with SP.using(candidate["vocabulary"], candidate["held"],
                  candidate.get("entries") or ()):
        then = {r["id"]: set(SP.detect(msgs(r["text"]))) for r in rows}
    lost, gained = [], []
    for r in rows:
        want = set(r.get("expect") or [])
        a, b = now[r["id"]], then[r["id"]]
        for p in sorted(want & (a - b)):
            lost.append({"id": r["id"], "package": p})
        for p in sorted((b - a) - want):
            gained.append({"id": r["id"], "package": p})
    try:
        shown = os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        shown = path
    return {"passed": not lost and not gained, "rows": len(rows),
            "lost_tp": lost, "gained_fp": gained, "labels": shown,
            "signature": candidate.get("signature")}


def promote(candidate: dict, *, by: str, forced: bool = False,
            reason: str = "") -> dict:
    """Write packages.json (the candidate's entries), vocabulary.json and
    held.json, each atomically (held.json last). Refused (ValueError) when
    the candidate does not pass floor() and `forced` is False. Returns the
    held manifest."""
    fl = floor(candidate)
    if not fl["passed"] and not forced:
        raise ValueError(
            "the candidate vocabulary is held: it loses "
            f"{len(fl['lost_tp'])} true positive(s) and gains "
            f"{len(fl['gained_fp'])} false positive(s) on {fl['labels']} "
            f"(lost {fl['lost_tp'][:10]}, gained {fl['gained_fp'][:10]}); "
            "the operator may force it")
    with _LOCK:
        raw = _read_json(PACKAGES)
        raw = raw if isinstance(raw, dict) else {}
        for e in candidate.get("entries") or ():
            name = e["name"]
            row = {k: v for k, v in e.items() if k != "name"}
            if name in SEED:
                old = raw.get(name) or {}
                keep = {k: row[k] for k in SEED_FILE_KEYS
                        if row.get(k) is not None}
                keep["versions"] = _uniq(list(old.get("versions") or [])
                                         + [v for v in row.get("versions")
                                            or [] if v not in
                                            SEED[name]["versions"]])
                raw[name] = {**old, **keep}
            elif name in raw:
                old = raw[name]
                row["versions"] = _uniq(list(old.get("versions") or [])
                                        + list(row.get("versions") or []))
                raw[name] = {**old, **row}
            else:
                raw[name] = row
        dirs = list(candidate["held"])
        at = _now()
        _write_json(PACKAGES, raw)
        _write_json(VOCABULARY, {"signature": candidate["signature"],
                                 "held": dirs, "promoted_at": at, "by": by,
                                 "vocabulary": candidate["vocabulary"]})
        m = {"dirs": dirs, "packages": _packages_of(dirs),
             "signature": candidate["signature"], "promoted_at": at,
             "by": by, "forced": bool(forced), "reason": reason,
             "floor": {k: fl[k] for k in ("passed", "rows", "lost_tp",
                                          "gained_fp", "labels")}}
        _write_json(HELD, m)
        nxt = next_candidate()
        if nxt and nxt.get("signature") == candidate["signature"]:
            try:
                os.remove(NEXT)
            except OSError:
                pass
        return m


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("show", "migrate"))
    a = ap.parse_args(argv)
    if a.cmd == "migrate":
        raw = _read_json(PACKAGES)
        raw = raw if isinstance(raw, dict) else {}
        for name, e in SEED.items():
            raw.setdefault(name, copy.deepcopy(e))
        _write_json(PACKAGES, raw)
        print(f"wrote {PACKAGES}: {len(raw)} package(s)")
        return 0
    data = load()
    print(f"registry {DIR}: {len(data)} package(s); packages.json "
          f"{'present' if os.path.exists(PACKAGES) else 'absent'}; held.json "
          f"{'present' if os.path.exists(HELD) else 'absent'}")
    for name, e in data.items():
        print(f"  {name:<30} term {e['term']:<16} area {str(e['area']):<14}"
              f" {'seed' if e.get('seed') else 'onboarded'}"
              f" versions {','.join(e['versions']) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
