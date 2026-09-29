#!/usr/bin/env python
"""The package evaluation (mcp/package_eval.py). No GPU, no network, no
model: a fake GitHub raw transport, a fake held package source tree, every
store a temp dir, a deterministic embedder, and the regression scripts'
output canned (the selection child runs for real, on a copy of the temp
store, with the stub decider).

    python mcp/test_package_eval.py      -> "N/M checks passed"

  1. STATISTICS: exact McNemar on known tables; the fewest discordant pairs
     that reach p < ALPHA; the smallest-detectable-difference statement and
     the "too small to support a claim" statement (PROTOCOL rule 4).
  2. THE RUN: detection with imports stripped and kept (the sanity row),
     per-package n statements, the kNN arm (nested leave-one-group-out),
     detection vs detection + kNN paired with its p and n, selection through
     the subprocess, the model-decider arm recorded "not run", regressions
     parsed, coverage gaps (blind spots, helpers with no skill, onboard
     next).
  3. HONESTY: the key; the same key is not recomputed; a later result on the
     same example set whose code signature changed is in-sample, the change
     named.
  4. NESTED: while choosing k for a held-out group, and while that group
     votes, the index never yields that group (instrumented).
  5. LEAKAGE: a run of lines found in a skill's source is flagged; a run
     under MIN_QUOTE_CHARS is not.
"""
from __future__ import annotations

import hashlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402
_TMP = offline_stores.isolate("yamadori_test_package_eval_")
SRC = os.path.join(_TMP, "_src")
os.environ["YAMADORI_PKG_REGISTRY_DIR"] = os.path.join(_TMP, "registry")
os.environ["YAMADORI_PACKAGES_SRC"] = SRC
os.environ["YAMADORI_TS_LIB"] = os.path.join(_TMP, "no_ts_lib")
os.environ["YAMADORI_NODE_TYPES"] = os.path.join(_TMP, "no_node_types")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "packages")
os.environ["YAMADORI_SKILL_DECIDER"] = "stub"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


_write(os.path.join(SRC, "fakeecs@1.0.0", "index.d.ts"),
       "export declare function spawnThing(): void;\n"
       "export declare function queryThings(): void;\n"
       "export declare function sharedName(): void;\n"
       "export declare const world: number;\n")
_write(os.path.join(SRC, "fakemath@2.0.0", "index.d.ts"),
       "export declare function vecAdd(): void;\n"
       "export declare function vecScale(): void;\n"
       "export declare function sharedName(): void;\n")

import skill_packages as SP  # noqa: E402
SP.prose_words = lambda names=None: {"world", "the", "thing"}
SP.tokenizer_words = lambda: set()
SP.REFERENCE_TYPES = ()

import example_knn as K  # noqa: E402
import onboarding  # noqa: E402
import package_eval as E  # noqa: E402
import package_examples as PE  # noqa: E402
import package_net as net  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    _results.append((bool(ok), name, str(detail)[:400]))


def fake_embed(texts, is_query=False):
    import re

    import numpy as np
    out = np.zeros((len(texts), 128), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[A-Za-z_$][\w$]*", t.split("Query:", 1)[-1]):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 128] += 1
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
        else:
            out[i, 0] = 1.0
    return out


K.EMBED = fake_embed

OWNER, REPO, COMMIT = "acme", "widgets", "beef" + "0" * 36
FILES = {
    "examples/ecs/a.ts": ("import { spawnThing, queryThings } from 'fakeecs'"
                          "\nexport function a() {\n  return spawnThing("
                          "queryThings())\n}\n"),
    "examples/ecs/b.ts": ("import { queryThings } from 'fakeecs'\n"
                          "export const b = () => queryThings()\n"),
    "examples/math.ts": ("import { vecAdd, vecScale } from 'fakemath'\n"
                         "export const m = (x) => vecAdd(x, vecScale(x))\n"),
    "examples/math2.ts": ("import { vecAdd } from 'fakemath'\n"
                          "export function sum(x, y) {\n  return vecAdd(x, "
                          "y)\n}\n"),
    "examples/both.ts": ("import { spawnThing } from 'fakeecs'\n"
                         "import { vecAdd } from 'fakemath'\n"
                         "export const s = spawnThing(vecAdd(1, 2))\n"),
    "examples/ecs2.ts": ("import { spawnThing } from 'fakeecs'\n"
                         "export const t = spawnThing(3)\n"),
    "examples/blind.ts": ("import { sharedName, world } from 'fakeecs'\n"
                          "import pad from 'left-pad-x'\n"
                          "export const z = pad(sharedName(world))\n"),
}


def tree():
    return {"entries": [{"path": p, "type": "blob", "size": len(t.encode()),
                         "sha": PE.git_blob_sha(t.encode())}
                        for p, t in sorted(FILES.items())],
            "truncated": False, "walked": []}


def transport(url, headers, max_bytes):
    for p, t in FILES.items():
        if url == net.raw_url(OWNER, REPO, COMMIT, p):
            return net.Response(200, t.encode(), {}, url)
    return net.Response(404, b"", {}, url)


net.TRANSPORT = transport
PKG = {"ecosystem": "npm", "name": "fakeecs", "version": "1.0.0",
       "commit": COMMIT, "repository": {"owner": OWNER, "repo": REPO,
                                        "directory": None}}
DID = "ds_eval_1"

LABELS_OUT = """vocabulary: 10 unique symbols over 2 packages

== labels: 40 implicit-use prompts, 12 near misses (IN-SAMPLE)
  ALL (package decisions)      precision 0.95 recall 0.90  (tp 36 fp 2 fn 4)
  near misses with any detection: 1/12
    row7: want ['koota'] got -
"""
DAILY_OUT = """pool: 530 armed skills (a copy of the live store)
  FAIL case-3: MUST inject koota-traits
  by kind: MUST 20/21, MUST-NOT 30/30, NOTHING 5/5

55/56 checks passed
"""


def fake_script(argv):
    return (0, LABELS_OUT) if "--labels" in argv else (1, DAILY_OUT)


E.RUN_SCRIPT = fake_script


# ================================================================ tests ==
def test_stats():
    check(abs(E.mcnemar_exact(10, 2) - 158 / 4096) < 1e-12,
          "[mcnemar] b=10 c=2: p = 2*(1+12+66)/2^12 = 0.0386",
          E.mcnemar_exact(10, 2))
    check(E.mcnemar_exact(0, 0) == 1.0 and E.mcnemar_exact(5, 0) == 0.0625
          and E.mcnemar_exact(3, 3) == 1.0,
          "[mcnemar] no discordant pairs -> 1; 5-0 -> 0.0625; 3-3 -> 1",
          (E.mcnemar_exact(5, 0), E.mcnemar_exact(3, 3)))
    check(E.min_discordant(0.05) == 6, "[power] 6 discordant pairs all one "
          "way are the fewest that reach p < 0.05 (2/2^6 = 0.031)",
          E.min_discordant())
    d4, d12 = E.detectable(4), E.detectable(12)
    check(not d4["supports_claim"] and "too small to support a claim" in
          d4["statement"] and "p=0.125" in d4["statement"],
          "[power] n=4 says it cannot support a claim", d4)
    check(d12["supports_claim"] and d12["smallest_difference"] == 0.5 and
          "6/12" in d12["statement"], "[power] n=12: the smallest paired "
          "difference it can detect is 6/12", d12)
    p = E.paired([True, False, True], [True, True, True], "a", "b")
    check(p["only_b"] == 1 and p["only_a"] == 0 and p["n"] == 3 and
          p["p"] == 1.0 and p["detectable"]["n"] == 3,
          "[paired] discordant counts, p and n on the same items", p)


def test_leakage():
    items = [{"id": "f1", "text": "const x = 1\n"
              "export function spawnMany(count) {\n"
              "  return makeThings(count)\n}\n"},
             {"id": "f2", "text": "short()\n"}]
    src = ("# Guide\n\n```ts\nexport function spawnMany(count) {\n"
           "  return makeThings(count)\n}\n```\n")
    got = E.leakage(items, {"skill1": src, "skill2": "short()\n"})
    check(got.get("f1") and got["f1"][0]["skill"] == "skill1" and
          got["f1"][0]["lines"][0] == 2, "[leakage] a run of lines found "
          "in a skill's source is flagged, with where", got)
    check("f2" not in got, "[leakage] a run under MIN_QUOTE_CHARS is not a "
          "quote", got)


def _constructed():
    import math

    import numpy as np
    degs = [0, 10, 20, 80, 90]
    labs = [("A",), ("A",), ("A",), ("B",), ("B",)]
    D = np.vstack([[math.cos(math.radians(d)), math.sin(math.radians(d))]
                   for d in degs]).astype(np.float32)
    nb = K.Neighbours(D, list(range(5)), labs, D.copy(), list(range(5)),
                      labs, [f"g{i}" for i in range(5)])
    nb.file_ids = [f"f{i}" for i in range(5)]
    return nb


ACTIVE: set = set()
VIOL: list = []


class Instrumented(K.Neighbours):
    def ranked(self, f, exclude):
        for g in super().ranked(f, exclude):
            if g in ACTIVE:
                VIOL.append(("ranked", f, g))
            yield g

    def queries_of(self, g):
        if g in ACTIVE:
            VIOL.append(("queries_of", g))
        return super().queries_of(g)


def test_nested():
    base = _constructed()
    nb = Instrumented.__new__(Instrumented)
    nb.__dict__.update(base.__dict__)
    orig = K.logo

    def logo(nb_, folds, *, exclude=frozenset()):
        ACTIVE.clear()
        ACTIVE.update(exclude)
        return orig(nb_, folds, exclude=exclude)
    K.logo = logo
    try:
        items = [{"id": f"f{i}", "group": f"g{i}", "path": f"p{i}",
                  "truth": [lab]} for i, lab in enumerate("AAABB")]
        got = E.knn_arm(items, nb=nb)
    finally:
        K.logo = orig
        ACTIVE.clear()
    check(got["ran"] and len(got["groups"]) == 5,
          "[nested] every group held out once, each with its own k", got)
    check(not VIOL, "[nested] neither the inner k selection nor the vote "
          "of a held-out group ever read that group", VIOL[:5])
    check(got["top1"]["n"] == 5 and got["top1"]["num"] == 5,
          "[nested] top-1 with its n on the constructed case", got["top1"])


def test_run():
    PE.collect(DID, PKG, tree(), linked_dirs=(), beat=lambda s: None)
    K.build(beat=lambda s: None)
    rec = E.run(DID, PKG, beat=lambda s: None)
    check(rec.get("key") and os.path.exists(os.path.join(
        onboarding.out_dir(DID), "eval", f"{rec['key']}.json")),
        "[run] eval/<key>.json written", rec.get("key"))
    check(rec["in_sample"] is False, "[honesty] the first result on an "
          "example set is not in-sample", rec["in_sample"])
    check(rec["sanity"]["ok"] and rec["detection_kept"]["overall"][
        "recall"]["value"] == 1.0, "[detect] imports KEPT: recall 1.0 (the "
        "sanity row)", rec["sanity"])
    ds = rec["detection_stripped"]
    check(ds["overall"]["recall"]["n"] > 0 and "fakeecs" in
          ds["per_package"] and "detectable" in ds["per_package"]["fakeecs"],
          "[detect] imports STRIPPED: per package counts with n and the "
          "detectable statement", ds["overall"])
    small = [p for p, v in ds["per_package"].items()
             if not v["detectable"]["supports_claim"]]
    check(small and all("rate" in ds["per_package"][p] and "precision" not
                        in ds["per_package"][p] for p in small),
          "[power] a package too small to support a claim reports counts, "
          "not a rate", small)
    knn = rec["knn"]
    check(knn["ran"] and knn["top1"]["n"] > 0 and knn["groups"] and
          all("k" in v and "n" in v for v in knn["groups"].values()),
          "[knn] each held-out group's own k (nested), top-1 with its n",
          {k: knn[k] for k in ("top1", "recall_among_voted")})
    cmp_ = rec["detect_vs_detect_knn"]
    check(cmp_["exact_set"]["n"] == knn["items_in_index"] and
          "p" in cmp_["recall_pairs"] and cmp_["exact_set"]["test"]
          .startswith("exact McNemar"), "[paired] detection vs detection + "
          "kNN on the same items, exact McNemar with n", cmp_["exact_set"])
    sel = rec["selection"]
    check(sel.get("decider") == "stub" and sel.get("pool") == 0 and
          sel["area_opened"]["n"] >= 0 and "PROXY" in
          sel["derived_label_delivered"]["note"],
          "[selection] the stub arm ran in the subprocess on a copy of the "
          "store; the derived label says it is a proxy", sel)
    check(rec["selection_model"]["state"] == "not run" and
          "no model decider configured" in rec["selection_model"]["why"],
          "[selection] the model-decider arm is recorded 'not run'",
          rec["selection_model"])
    reg = rec["regressions"]
    check(reg["package_labels"]["tp"] == 36 and reg["package_labels"][
        "near_detected"] == 1 and reg["daily"]["passed"] == 55 and
        reg["daily"]["n"] == 56 and reg["daily"]["by_kind"]["MUST"] ==
        {"passed": 20, "n": 21}, "[regressions] both scripts' counts parsed",
        reg)
    cov = rec["coverage"]
    blind = {b["name"]: b["why"] for b in cov["detector_blind_spots"]}
    check("sharedName" in blind and "not unique" in blind["sharedName"] and
          "English" in blind.get("world", ""), "[coverage] detector blind "
          "spots: a shared name, an English word", blind)
    check([u["package"] for u in cov["onboard_next"]] == ["left-pad-x"],
          "[coverage] packages to onboard next", cov["onboard_next"])
    check(any(h["name"] == "spawnThing" for h in
              cov["helpers_without_skill"]),
          "[coverage] helpers no armed skill names",
          cov["helpers_without_skill"][:3])
    check(rec["summary"]["items"] == len(FILES) and
          rec["summary"]["seen_by_skill_source"]["n"] == len(FILES),
          "[summary] every number carries its n", rec["summary"])
    again = E.run(DID, PKG, beat=lambda s: None)
    check(again.get("cached") and again["key"] == rec["key"],
          "[honesty] the same key is not recomputed", again.get("key"))
    orig = E.git_head
    E.git_head = lambda root=E.ROOT: "0123456789abcdef"
    try:
        later = E.run(DID, PKG, beat=lambda s: None, selection_on=False,
                      regressions_on=False)
    finally:
        E.git_head = orig
    check(later["key"] != rec["key"] and later["in_sample"] is True and
          later["in_sample_changes"] == ["code"] and
          later["in_sample_since"] == rec["key"],
          "[honesty] a later result on the same example set after the code "
          "changed is in-sample, the change named",
          {k: later.get(k) for k in ("in_sample", "in_sample_changes")})
    sk = E.run("ds_none", None)
    check(sk.get("skipped") and sk["key"] is None,
          "[skip] no package -> skipped", sk)


def main() -> int:
    for t in (test_stats, test_leakage, test_nested, test_run):
        try:
            t()
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{t.__name__} raised {type(e).__name__}: {e}",
                  traceback.format_exc()[-1500:])
    bad = [r for r in _results if not r[0]]
    for ok, name, detail in _results:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}"
              + ("" if ok else f"\n        {detail}"))
    print(f"\n{len(_results) - len(bad)}/{len(_results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
