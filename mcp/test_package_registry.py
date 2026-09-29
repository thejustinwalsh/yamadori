#!/usr/bin/env python
"""The package registry (mcp/package_registry.py). No GPU, no network; every
store is a temp dir -- the registry (YAMADORI_PKG_REGISTRY_DIR), the held
sources (YAMADORI_PACKAGES_SRC, a tiny fake package tree), the TypeScript
lib and @types/node (empty) -- and the English lexicon is a fixed set.

    python mcp/test_package_registry.py      -> "N/M checks passed"

  1. SEED ONLY: with no packages.json the tables are the hand-written ones
     (TERM_PACKAGE, PACKAGE_AREA / AREA_PACKAGE, BUILT_ON, CANONICAL) and
     the taxonomy is the hand-written VOCAB; `migrate` changes nothing.
  2. TERMS AND WORDS: slug terms, collisions, the one-word English rule,
     aliases as typed.
  3. THE HELD MANIFEST: freeze_held() from the glob; a package fetched later
     is not held until promoted.
  4. THE CANDIDATE: the diff (unique names, the names an existing package
     loses), vocabulary.next.json, the process untouched.
  5. THE FLOOR: pass and hold on label files; the real labels pass.
  6. PROMOTION: a held candidate is refused, a forced one written (atomic,
     recorded); the vocabulary file is loaded, not built; the new term is in
     the taxonomy, asked_terms, detection, the selector's maps; tag_version
     moves with the catalogue while the prompt pin holds.
  7. The name-major filter reads `package_version`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402
_TMP = offline_stores.isolate("yamadori_test_package_registry_")
REG = os.path.join(_TMP, "registry")
SRC = os.path.join(_TMP, "_src")
os.environ["YAMADORI_PKG_REGISTRY_DIR"] = REG
os.environ["YAMADORI_PACKAGES_SRC"] = SRC
os.environ["YAMADORI_TS_LIB"] = os.path.join(_TMP, "no_ts_lib")
os.environ["YAMADORI_NODE_TYPES"] = os.path.join(_TMP, "no_node_types")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# The held package before onboarding, and one fetched later.
_write(os.path.join(SRC, "fakeecs@1.0.0", "index.d.ts"),
       "export declare function spawnThing(): void;\n"
       "export declare function sharedHelper(): void;\n"
       "export declare function zorbify(): void;\n")
NEW_DTS = ("export declare function sharedHelper(): void;\n"
           "export declare function blorpify(): void;\n"
           "export declare function thing(): void;\n")

import package_registry as R  # noqa: E402
import skill_packages as SP  # noqa: E402

# The English lexicon, fixed (the real one reads READMEs and a tokenizer).
ENGLISH = {"tweak", "hello", "world", "thing", "the", "loop"}
SP.prose_words = lambda names=None: set(ENGLISH)
SP.tokenizer_words = lambda: set()
SP.REFERENCE_TYPES = ()

import skill_classify as C  # noqa: E402
import skill_match as M  # noqa: E402
import skill_prompts as P  # noqa: E402
import skill_select as S  # noqa: E402

_results: list[tuple[bool, str, str]] = []

OLD_TERM_PACKAGE = {"r3f": "@react-three/fiber", "koota": "koota",
                    "pmndrs_math": "math", "threejs": "three",
                    "typegpu": "typegpu"}
OLD_PACKAGE_AREA = {"@react-three/fiber": "r3f", "@react-three/drei": "r3f",
                    "@react-three/postprocessing": "r3f", "koota": "koota",
                    "math": "pmndrs_math", "three/tsl": "threejs",
                    "three": "threejs", "typegpu": "typegpu",
                    "@typegpu/three": "typegpu", "@typegpu/noise": "typegpu"}
OLD_AREA_PACKAGE = {"r3f": "@react-three/fiber", "koota": "koota",
                    "pmndrs_math": "math", "threejs": "three/tsl",
                    "typegpu": "typegpu"}
OLD_BUILT_ON = {"typegpu": ("webgpu",), "r3f": ("react", "threejs")}
OLD_CANONICAL = {"koota": "koota-traits-and-entities",
                 "math": "math-data-oriented-functions",
                 "@react-three/fiber": "r3f-v10-setup-21",
                 "three/tsl": "threejs-llms-full-tsl-e-g",
                 "@react-three/drei": "r3f-drei-pairing-8",
                 "typegpu": "typegpu-migration-0-12-import-pattern-2",
                 "@typegpu/three": "typegpu-three-tsl-integration-2"}


def check(ok, name, detail=""):
    _results.append((bool(ok), name, str(detail)[:300]))


def user(text):
    return [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": text}]


def labels_file(name, rows):
    p = os.path.join(_TMP, name)
    _write(p, "// a comment line\n" + "\n".join(json.dumps(r) for r in rows)
           + "\n")
    return p


PASS_LABELS = labels_file("pass.jsonl", [
    {"id": "p1", "text": "```ts\nspawnThing()\n```", "expect": ["fakeecs"]},
    {"id": "p2", "text": "hello world", "expect": [], "near": True}])
HOLD_LABELS = labels_file("hold.jsonl", [
    {"id": "h1", "text": "```ts\nsharedHelper()\n```", "expect": ["fakeecs"]},
    {"id": "h2", "text": "```ts\nblorpify()\n```", "expect": [],
     "near": True}])


def framework_ids():
    return {x["id"] for x in C.taxonomy()["framework"]}


# ============================================================ 1. seed ==
def test_seed():
    check(not os.path.exists(R.PACKAGES), "[seed] no packages.json in the "
          "temp registry", R.PACKAGES)
    check(R.term_package() == OLD_TERM_PACKAGE and dict(SP.TERM_PACKAGE)
          == OLD_TERM_PACKAGE, "[seed] term -> package is the hand-written "
          "TERM_PACKAGE", R.term_package())
    check(R.package_area() == OLD_PACKAGE_AREA and dict(M.PACKAGE_AREA)
          == OLD_PACKAGE_AREA, "[seed] package -> area is the hand-written "
          "PACKAGE_AREA (skill_match and eval_packages)", R.package_area())
    check(R.area_package() == OLD_AREA_PACKAGE and dict(M.AREA_PACKAGE)
          == OLD_AREA_PACKAGE, "[seed] area -> package is the hand-written "
          "AREA_PACKAGE (threejs -> three/tsl)", R.area_package())
    check(R.built_on_terms() == OLD_BUILT_ON and dict(S.BUILT_ON)
          == OLD_BUILT_ON, "[seed] BUILT_ON is the two hand rows, in order",
          R.built_on_terms())
    check({k: v["skill"] for k, v in SP.CANONICAL.items()} == OLD_CANONICAL
          and SP.CANONICAL["@react-three/fiber"]["by_major"]
          == {"9": "r3f-v9-setup-22", "10": "r3f-v10-setup-21"},
          "[seed] CANONICAL rows unchanged (by_major kept)",
          dict(SP.CANONICAL))
    check(set(SP.PACKAGE_AREAS) == set(OLD_TERM_PACKAGE),
          "[seed] PACKAGE_AREAS is the terms that name a package",
          set(SP.PACKAGE_AREAS))
    C.refresh()
    check(C.VOCAB == C.HAND_VOCAB and set(C.BY_ID) == {
        t.id for t in C.HAND_VOCAB}, "[seed] the taxonomy is the "
        "hand-written VOCAB", len(C.VOCAB))
    before = json.dumps(R.load(), sort_keys=True)
    tv = P.tag_version()
    rc = R.main(["migrate"])
    check(rc == 0 and os.path.exists(R.PACKAGES), "[migrate] writes "
          "packages.json from SEED")
    check(json.dumps(R.load(), sort_keys=True) == before and C.refresh()
          and C.VOCAB == C.HAND_VOCAB and P.tag_version() == tv,
          "[migrate] the migrated file changes nothing: the same registry, "
          "taxonomy and tag version")
    with open(R.PACKAGES, encoding="utf-8") as f:
        raw = json.load(f)
    check(set(raw) == set(R.SEED) and all(raw[k]["seed"] for k in raw),
          "[migrate] one row per SEED package, marked seed", sorted(raw))


# ================================================== 2. terms and words ==
def test_terms_and_words():
    check(R.slug_term("@pmndrs/uikit") == "pmndrs_uikit"
          and R.slug_term("three-flatland") == "three_flatland",
          "[slug] @pmndrs/uikit -> pmndrs_uikit, three-flatland -> "
          "three_flatland")
    check(R.slug_term("koota") == "koota", "[slug] a registered package "
          "keeps its term")
    e = R.add_package("rust", version="1.0.0")
    check(e["term"] == "rust_pkg" and "language term" in e.get(
        "term_note", ""), "[slug] a slug that is another term's id (the "
        "Rust language) is suffixed deterministically, and the entry says "
        "so", e)
    e = R.add_package("tailwindcss", version="4.0.0")
    check(e["term"] == "tailwind" and e["area"] is None
          and not e["names_term"], "[slug] a package a hand-written term "
          "already files (tailwindcss) takes that term and opens no package "
          "area", e)
    w = R.words_rule("tweak", [], english=ENGLISH)
    rx = re.compile(w["regex"], 0 if w["case"] else re.I)
    hits = [bool(rx.search(t)) for t in (
        "pin tweak@2 now", "import { a } from 'tweak/core'",
        "npm install tweak", "then npm i tweak")]
    miss = [bool(rx.search(t)) for t in (
        "tweak the loop", "Tweak it", "tweaks", "npm install tweaker")]
    check(all(hits) and not any(miss), "[words] a one-word English npm name "
          "gets only name@<digit>, a subpath, npm install|i name",
          (hits, miss, w))
    check("English" in w["why"], "[words] and says why", w["why"])
    e = R.add_package("tweak", version="1.0.0")
    check("English" in e["words"]["why"] and not re.search(
        e["words"]["regex"], "tweak the loop"), "[words] add_package applies "
        "the English rule automatically (skill_packages.is_english)",
        e["words"])
    w = R.words_rule("zorbl", ["ZB Kit"], english=ENGLISH)
    rx = re.compile(w["regex"], 0 if w["case"] else re.I)
    check(bool(rx.search("use Zorbl here")) and bool(rx.search("the ZB Kit "
          "renders")) and not rx.search("zorbls") and not rx.search(
          "the ZB Kitchen") and not rx.search("the zb kit"),
          "[words] a non-English name matches as written (any case); an "
          "alias as typed, word-bounded", w)
    w = R.words_rule("@pmndrs/uikit", [], english=ENGLISH)
    rx = re.compile(w["regex"], 0 if w["case"] else re.I)
    check(bool(rx.search("import { Root } from '@pmndrs/uikit'"))
          and bool(rx.search("@pmndrs/uikit/react")) and not rx.search(
              "@pmndrs/uikit-lucide"), "[words] a scoped name, not inside a "
          "longer name", w)


# ================================================== 3. held manifest ==
def test_held():
    check(set(SP.held()) == {"fakeecs"} and R.held_manifest() is None,
          "[held] no held.json: the glob of the source dir", SP.held())
    m = R.freeze_held(by="test")
    check(m["dirs"] == ["fakeecs@1.0.0"] and m["packages"] == {
        "fakeecs": ["1.0.0"]} and os.path.exists(R.HELD),
        "[held] freeze_held writes held.json from today's glob", m)
    _write(os.path.join(SRC, "newpkg@2.0.0", "index.d.ts"), NEW_DTS)
    check(set(SP.held()) == {"fakeecs"}, "[held] a package fetched later is "
          "NOT held until promoted (held() reads held.json)", SP.held())
    check(R.freeze_held(by="again") == m, "[held] freeze_held is idempotent")


# ===================================================== 4. candidate ==
STATE: dict = {}


def test_candidate():
    v0 = SP.vocabulary()
    check("spawnThing" in v0["symbols"] and "sharedHelper" in v0["symbols"]
          and "blorpify" not in v0["symbols"], "[vocab] the current "
          "vocabulary is the held fake package's", sorted(v0["symbols"]))
    ent = R.add_package("newpkg", version="2.0.0", aliases=["Newpkg Thing"],
                        built_on=["react"], built_on_from="package.json "
                        "peerDependencies at abc123",
                        licence={"spdx": "MIT", "quote": "MIT",
                                 "where": "package.json"})
    check(ent["term"] == "newpkg" and ent["area"] == "newpkg"
          and ent["versions"] == ["2.0.0"], "[entry] a new package's term "
          "and area are its slug", ent)
    cand = R.build_candidate(add_dirs=["newpkg@2.0.0"], entries=[ent])
    STATE["cand"], STATE["ent"] = cand, ent
    check(cand["held"] == ["fakeecs@1.0.0", "newpkg@2.0.0"],
          "[candidate] the held dirs plus the added one", cand["held"])
    per = cand["diff"]["per_package"].get("newpkg") or {}
    check(per.get("unique") == 1 and per.get("code_only") == 1
          and per.get("dropped_shared") == 1 and per.get("dropped_english")
          == 1, "[candidate] per added package: unique (blorpify), "
          "code-only / English (thing), shared (sharedHelper)", per)
    check(cand["diff"]["lost"] == {"fakeecs": ["sharedHelper"]},
          "[candidate] the names an existing package LOSES (uniqueness is "
          "relative)", cand["diff"]["lost"])
    nxt = R.next_candidate()
    check(nxt and nxt["signature"] == cand["signature"]
          and len(cand["signature"]) == 64, "[candidate] "
          "vocabulary.next.json holds the candidate and its signature")
    check(SP.vocabulary() is v0 and "blorpify" not in SP.vocabulary()[
        "symbols"] and "newpkg" not in R.load() and "newpkg" not in
          framework_ids(), "[candidate] the process's vocabulary, registry "
          "and taxonomy are untouched")
    with SP.using(cand["vocabulary"], cand["held"], cand["entries"]):
        inside = ("newpkg" in framework_ids(), "newpkg" in SP.held(),
                  SP.vocabulary()["symbols"].get("blorpify"))
    check(inside == (True, True, "newpkg") and "newpkg" not in
          framework_ids() and "newpkg" not in SP.held(),
          "[using] a candidate is seen inside using() and gone after",
          inside)


# ========================================================= 5. floor ==
def test_floor():
    cand = STATE["cand"]
    f = R.floor(cand, labels=PASS_LABELS)
    check(f["passed"] and f["rows"] == 2 and not f["lost_tp"]
          and not f["gained_fp"], "[floor] no lost true positive, no gained "
          "false positive: passes", f)
    f = R.floor(cand, labels=HOLD_LABELS)
    check(not f["passed"] and f["lost_tp"] == [{"id": "h1", "package":
                                                "fakeecs"}]
          and f["gained_fp"] == [{"id": "h2", "package": "newpkg"}],
          "[floor] a lost true positive and a gained false positive hold "
          "it, with the rows and packages", f)
    f = R.floor(cand)
    n = len(R.label_rows())
    check(f["passed"] and f["rows"] == n and n > 0 and f["labels"]
          == "bench/skills/package_detect_labels.jsonl",
          "[floor] the standing labels (bench/skills/package_detect_labels."
          "jsonl) run under both and pass", f)
    check("newpkg" not in framework_ids() and "newpkg" not in SP.held(),
          "[floor] the process state is back after the floor")


# ===================================================== 6. promotion ==
def test_promote():
    cand, _ent = STATE["cand"], STATE["ent"]
    tv0 = P.tag_version()
    R.LABELS = PASS_LABELS
    c0 = R.build_candidate()
    m0 = R.promote(c0, by="test")
    check(os.path.exists(R.VOCABULARY) and m0["dirs"] == ["fakeecs@1.0.0"]
          and m0["floor"]["passed"] and not m0["forced"],
          "[promote] a candidate that passes is written", m0)
    real_build = SP.build_vocabulary

    def no_build(*a, **k):
        raise AssertionError("built on the request path")
    SP.build_vocabulary = no_build
    try:
        v = SP.vocabulary()
        ok = v["symbols"].get("spawnThing") == "fakeecs"
    except AssertionError as e:
        ok, v = False, str(e)
    finally:
        SP.build_vocabulary = real_build
    check(ok, "[promote] vocabulary() loads vocabulary.json; it is not "
          "built while the file exists", v)
    R.LABELS = HOLD_LABELS
    with open(R.HELD, encoding="utf-8") as fh:
        held_before = fh.read()
    with open(R.PACKAGES, encoding="utf-8") as fh:
        pk_before = fh.read()
    try:
        R.promote(cand, by="test")
        refused = False
    except ValueError as e:
        refused = "held" in str(e)
    with open(R.HELD, encoding="utf-8") as fh:
        held_after = fh.read()
    with open(R.PACKAGES, encoding="utf-8") as fh:
        pk_after = fh.read()
    check(refused and held_after == held_before and pk_after == pk_before,
          "[promote] a candidate the floor holds is refused and nothing is "
          "written")
    m = R.promote(cand, by="operator", forced=True, reason="forced by test")
    R.LABELS = PASS_LABELS
    left = [n for n in os.listdir(REG) if n.endswith(".tmp")]
    check(m["forced"] and m["reason"] == "forced by test"
          and not m["floor"]["passed"] and m["by"] == "operator"
          and m["dirs"] == ["fakeecs@1.0.0", "newpkg@2.0.0"]
          and m["packages"]["newpkg"] == ["2.0.0"] and not left,
          "[promote] forced: written with who, why and the floor it failed; "
          "no temp file left", (m, left))
    with open(R.VOCABULARY, encoding="utf-8") as fh:
        vf = json.load(fh)
    check(vf["signature"] == cand["signature"] and not os.path.exists(
        R.NEXT), "[promote] vocabulary.json carries the signature; the "
          "promoted next file is cleared")
    with open(R.PACKAGES, encoding="utf-8") as fh:
        raw = json.load(fh)
    check(raw.get("newpkg", {}).get("term") == "newpkg"
          and raw["newpkg"].get("licence", {}).get("spdx") == "MIT"
          and "koota" in raw, "[promote] packages.json gains the entry "
          "beside the migrated seed rows", raw.get("newpkg"))
    # --- what reads the registry sees it, with no restart ----------------
    check(set(SP.held()) == {"fakeecs", "newpkg"} and SP.vocabulary()[
        "symbols"].get("blorpify") == "newpkg", "[promote] held() and "
          "vocabulary() read the promoted files")
    check(R.term_package().get("newpkg") == "newpkg"
          and SP.TERM_PACKAGE.get("newpkg") == "newpkg"
          and M.AREA_PACKAGE.get("newpkg") == "newpkg"
          and M.PACKAGE_AREA.get("newpkg") == "newpkg"
          and "newpkg" in SP.PACKAGE_AREAS,
          "[promote] TERM_PACKAGE, PACKAGE_AREA, AREA_PACKAGE, PACKAGE_AREAS "
          "carry it", R.area_package())
    check(R.built_on_terms().get("newpkg") == ("react",)
          and S.BUILT_ON.get("newpkg") == ("react",)
          and S.BUILT_ON.get("r3f") == ("react", "threejs"),
          "[promote] BUILT_ON merges the entry's built_on (npm -> term) with "
          "the seed rows", dict(S.BUILT_ON))
    check("newpkg" in framework_ids() and C.BY_ID["newpkg"].kind
          == "framework" and C.BY_ID["newpkg"].packages == ("newpkg",)
          and C._PKG_TO.get("newpkg") == "newpkg"
          and C.VOCAB[:len(C.HAND_VOCAB)] == C.HAND_VOCAB,
          "[taxonomy] refresh() adds one framework term for the package; the "
          "hand terms stay first", [t.id for t in C.VOCAB])
    a = C.asked_terms("Build the HUD with newpkg v2 please.")
    b = C.asked_terms("Wire the Newpkg Thing into the menu.")
    n = C.negated_terms("do it without newpkg")
    check(a.get("newpkg", {}).get("version") == "2" and "newpkg" in b
          and n == {"newpkg"}, "[taxonomy] asked_terms finds the name and "
          "the typed alias; negation holds", (a, b, n))
    got = SP.detect(user("Build the HUD with newpkg v2 please."))
    got2 = SP.detect(user("```ts\nblorpify()\n```"))
    check(got.get("newpkg", {}).get("version") == "2" and "newpkg" in got2,
          "[detect] the package is in play by name (with its version) and "
          "by its unique symbol", (got, got2))
    tv1 = P.tag_version()
    check(P.TAG_VERSION == "tag/6" and tv0 != tv1 and tv1.startswith(
        "tag/6+") and len(tv1) == len("tag/6+") + 8 and "newpkg" in
          P.tag_system(), "[tag] tag_version names the catalogue: it moved "
          "with the new term; TAG_VERSION stays tag/6", (tv0, tv1))
    with open(os.path.join(HERE, "fixtures", "skill_prompt_pins.json"),
              encoding="utf-8") as fh:
        pins = json.load(fh)
    reg = {r["name"]: r for r in P.registry()}
    check(reg["tag"]["sha256"] == pins["tag"]["sha256"]
          and reg["tag"]["version"] == pins["tag"]["version"]
          and P.CATALOGUE_SLOT in reg["tag"]["text"]
          and reg["tag"]["recorded_version"] == tv1,
          "[tag] the prompt pin is the template: adding a term does not "
          "break it", (reg["tag"]["sha256"], pins["tag"]))
    # --- a seed row cannot be rewritten by the file ------------------------
    raw["koota"] = {"term": "hijack", "area": "elsewhere",
                    "versions": ["9.9.9"], "onboarding": "ds-1"}
    R._write_json(R.PACKAGES, raw)
    k = R.entry("koota")
    check(k["term"] == "koota" and k["area"] == "koota"
          and "9.9.9" in k["versions"] and k["onboarding"] == "ds-1"
          and "hijack" not in framework_ids(),
          "[seed] packages.json may add a seed package's versions / "
          "onboarding / licence, never change its term or area", k)
    # --- add_package on a registered package adds the version --------------
    e2 = R.add_package("newpkg", version="2.1.0")
    check(e2["versions"] == ["2.0.0", "2.1.0"] and e2["term"] == "newpkg",
          "[entry] a registered package gains the version, keeps its term",
          e2)


# =================================================== 7. the name-major ==
def test_majors():
    got = [S.skill_majors({"name": "r3f-v9-setup-22"}),
           S.skill_majors({"name": "fiber-setup",
                           "package_version": "10.0.0-alpha.5"}),
           S.skill_majors({"name": "r3f-v9-setup-22",
                           "package_version": "10.0.0"}),
           S.skill_majors({"name": "koota-traits", "package_version": None})]
    check(got == [{9}, {10}, {10}, set()], "[major] the armed row's "
          "package_version decides the major; else the name", got)


def main() -> int:
    for fn in (test_seed, test_terms_and_words, test_held, test_candidate,
               test_floor, test_promote, test_majors):
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            check(False, f"{fn.__name__} itself raised",
                  f"{type(e).__name__}: {e}")
    fails = 0
    for ok, name, detail in _results:
        print(f"  {'pass' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"   <- {detail}"))
        fails += not ok
    n = len(_results)
    print(f"\n  {n - fails}/{n} checks passed")
    shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
