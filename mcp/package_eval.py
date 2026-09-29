#!/usr/bin/env python
"""THE PACKAGE EVALUATION: detection, the example kNN and selection, on the
package's own example code, leave-one-GROUP-out.

    python mcp/package_eval.py show <onboarding id>     the latest result

docs/PACKAGE-ONBOARDING.md section 7 (item K), with operator decision 1
(2026-09-27) in place of the design's fixed held-out set: "no fraction is
invented. Use leave-one-GROUP-out cross-validation over the example groups
(every group is held out once, and its votes come from an index built
without it). k for the vote is chosen inside the training folds only."

EVERY kept example file of the onboarding is an ITEM (package_examples'
manifest); its truth is the held packages its OWN imports name. Deterministic
and offline: no model, no GPU (the kNN reads the vectors example_knn stored;
nothing is embedded here).

| measure | how | truth |
|---|---|---|
| detection, imports STRIPPED | each file as an agent step (a write_file call writing it; import / require lines removed, package_examples.strip_imports) through skill_packages.detect | the file's import labels; precision / recall per package and overall |
| detection, imports KEPT | the same with its imports | the SANITY row: recall must be 1.0 -- less is reported as a BUG, not a result |
| kNN | the file's group held out: its vote comes from every OTHER group (and k for it is chosen by an inner leave-one-group-out over those other groups only -- nested) | top-1 of the tally's order; recall of the true packages among every voted package |
| detection + kNN | a package counts as found by detect OR by the tally's top | PAIRED with detection alone on the same items (PROTOCOL rule 6): exact McNemar, n printed |
| selection | each file (imports stripped) through skill_select.decide / plan_turn with the STUB decider, in a SUBPROCESS on a COPY of the store (decide writes selection rows) | package area opened; the lead given on its first appearance; a delivered skill among the file's DERIVED skill labels (a PROXY: a skill of the file's package whose verified code-shaped topics the file uses); delivered skills of packages the file does not use (precision) |
| selection, model decider | only when YAMADORI_SKILL_DECIDER names a non-stub decider: a gpu job (idle-gated), DECIDER_RUNS runs side by side (PROTOCOL rule 10) | as above |
| regressions | bench/skills/eval_packages.py --labels and bench/skills/test_daily_eval.py as subprocesses on this store | their counts; a new MUST / MUST-NOT miss against the previous result |

PROTOCOL RULE 4: every rate carries its n, and each package states the
smallest paired difference its n can detect; a package whose n cannot reach
p < ALPHA even with every item discordant one way says so and reports counts,
not a rate.

HONESTY (7.2): the result KEY is sha256 over the example manifest's SHA, the
vocabulary signature, the kNN index signature, the armed set's signature,
the git HEAD of the code (read, never required: "unknown" without git; the
HEAD commit, not the working tree) and EVAL_VERSION. The same key is not
recomputed. A later result on the same example set whose other signatures
changed is marked `in_sample: true`, the changes named. The leakage column
"seen by a skill source": a file with a run of consecutive lines found in a
skill's source under the quote check's normalisation (skill_builder._norm)
of at least skill_builder.MIN_QUOTE_CHARS characters -- the quote check's own
minimum.

COVERAGE GAPS (7.3): helpers with no skill, detector blind spots, packages to
onboard next, skills with no example, black-hole candidates among the
onboarding's skills (skill_boundaries' rule).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

EVAL_VERSION = "package_eval/1"
# The conventional two-sided significance level, the one the p-values this
# repo reports are read against (AGENTS.md: p=0.0001, p=0.003, p=0.011):
# operator, 2026-09-27, "keep ours (no external evidence)".
ALPHA = 0.05
ALPHA_SOURCE = ("the conventional two-sided level (the repo's reported "
                "p-values are read against it): operator, 2026-09-27: keep "
                "ours (no external evidence)")
DECIDER_QUEUE = "package.evaluate_decider"
# docs/PACKAGE-ONBOARDING.md 7.1: "n >= 2 runs reported side by side
# (PROTOCOL rule 10)".
DECIDER_RUNS = 2
# The Hermes tool names the replay harness offers
# (bench/skills/replay_selection.py HERMES_TOOLS; not imported: importing it
# copies the live store).
TOOLS = ["terminal", "read_file", "write_file", "patch", "search_files",
         "skills_list", "skill_view", "vision_analyze"]
SYSTEM = ("You are Hermes, an AI coding agent. You work in the user's "
          "project with your tools.")        # replay_selection.DAILY_SYSTEM
STEP_USER = "Continue the task."
PROXY_NOTE = ("derived skill labels are a PROXY for 'this skill would have "
              "helped': a skill of the file's package whose verified "
              "code-shaped topics the file uses")

# Regressions: the scripts and their output lines.
EVAL_PACKAGES = os.path.join(ROOT, "bench", "skills", "eval_packages.py")
DAILY_EVAL = os.path.join(ROOT, "bench", "skills", "test_daily_eval.py")
_ALL_LINE = re.compile(r"ALL \(package decisions\)\s+precision ([\d.]+) "
                       r"recall ([\d.]+)\s+\(tp (\d+) fp (\d+) fn (\d+)\)")
_NEAR_LINE = re.compile(r"near misses with any detection: (\d+)/(\d+)")
_CHECKS_LINE = re.compile(r"^(\d+)/(\d+) checks passed", re.M)
_KIND = re.compile(r"(MUST-NOT|MUST|NOTHING|form/recall) (\d+)/(\d+)")


# ---------------------------------------------------------------------------
# Statistics (PROTOCOL rules 4 and 6).
# ---------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: the binomial test on the discordant pairs
    (b one way, c the other) at 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def min_discordant(alpha: float = ALPHA) -> int:
    """The fewest discordant pairs, all one way, that reach p < alpha."""
    d = 1
    while mcnemar_exact(d, 0) >= alpha:
        d += 1
    return d


def detectable(n: int, alpha: float = ALPHA) -> dict:
    """What n paired items can support (PROTOCOL rule 4)."""
    d = min_discordant(alpha)
    if n < d:
        p = mcnemar_exact(n, 0)
        return {"n": n, "supports_claim": False, "min_discordant": d,
                "statement": f"n={n}: too small to support a claim -- even a "
                             f"difference on every item ({n} discordant "
                             f"pairs, all one way) gives exact McNemar "
                             f"p={p:.3g}, not below {alpha}"}
    return {"n": n, "supports_claim": True, "min_discordant": d,
            "smallest_difference": round(d / n, 4),
            "statement": f"n={n}: the smallest paired difference it can "
                         f"detect is {d}/{n} = {d / n:.1%} ({d} discordant "
                         f"pairs all one way: p={mcnemar_exact(d, 0):.3g} < "
                         f"{alpha})"}


def rate(num: int, n: int) -> dict:
    return {"value": round(num / n, 4) if n else None, "num": num, "n": n}


def paired(a: list[bool], b: list[bool], label_a: str, label_b: str) -> dict:
    """Paired binary outcomes on the same items: the discordant counts and
    the exact McNemar p, n printed."""
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = len(a)
    return {"n": n, "both": both, "neither": neither,
            f"only_{label_a}": only_a, f"only_{label_b}": only_b,
            label_a: rate(both + only_a, n), label_b: rate(both + only_b, n),
            "p": mcnemar_exact(only_a, only_b),
            "test": "exact McNemar (two-sided binomial on the discordant "
                    "pairs)", "alpha": ALPHA, "alpha_source": ALPHA_SOURCE,
            "detectable": detectable(n)}


# ---------------------------------------------------------------------------
# Items.
# ---------------------------------------------------------------------------
def agent_step(text: str, ext: str) -> list[dict]:
    """One example file as an agent step: a write_file call writing it (the
    path neutral: an example's own path could name its package)."""
    args = json.dumps({"path": f"src/example{ext}", "content": text})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": STEP_USER},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_0", "type": "function",
                             "function": {"name": "write_file",
                                          "arguments": args}}]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": f"Wrote {len(text)} characters."}]


def items_of(did: str) -> list[dict]:
    import package_examples as PE
    out = []
    for row in PE.manifest_rows(did):
        text = PE.read_stored(row)
        if text is None:
            continue
        facts = PE.file_facts(text)
        out.append({"id": row.get("id") or PE.file_id(
            row["repository"], row["path"], row.get("blob") or ""),
            "group": PE.group_hash(row["repository"], row["group"]),
            "group_path": row["group"], "path": row["path"],
            "repository": row["repository"],
            "ext": os.path.splitext(row["path"])[1] or ".ts",
            "text": text, "stripped": PE.strip_imports(text),
            "truth": facts["packages"], "bindings": facts["bindings"],
            "unheld": facts["unheld"], "identifiers": facts["identifiers"]})
    return out


def detect_set(text: str, ext: str) -> list[str]:
    import skill_packages as SP
    return sorted(SP.detect(agent_step(text, ext)))


def score_detection(items: list[dict], found: dict) -> dict:
    """Per package and overall precision / recall with their n; the per
    package n statement (PROTOCOL rule 4)."""
    per: dict[str, dict] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "n_true": 0})
    misses = []
    for it in items:
        have, want = set(found[it["id"]]), set(it["truth"])
        for p in have | want:
            k = "tp" if p in have and p in want else "fp" if p in have \
                else "fn"
            per[p][k] += 1
            if k == "fn":
                misses.append({"item": it["id"], "path": it["path"],
                               "package": p})
        for p in want:
            per[p]["n_true"] += 1
    tot = {k: sum(v[k] for v in per.values()) for k in ("tp", "fp", "fn")}
    out = {"overall": {"precision": rate(tot["tp"], tot["tp"] + tot["fp"]),
                       "recall": rate(tot["tp"], tot["tp"] + tot["fn"]),
                       "items": len(items), **tot},
           "per_package": {}, "misses": misses}
    for p, v in sorted(per.items()):
        det = detectable(v["n_true"])
        row = {"tp": v["tp"], "fp": v["fp"], "fn": v["fn"],
               "n": v["n_true"], "detectable": det}
        if det["supports_claim"]:
            row["precision"] = rate(v["tp"], v["tp"] + v["fp"])
            row["recall"] = rate(v["tp"], v["tp"] + v["fn"])
        else:
            row["rate"] = "not reported: " + det["statement"]
        out["per_package"][p] = row
    return out


# ---------------------------------------------------------------------------
# The kNN arm (nested leave-one-group-out on the stored vectors).
# ---------------------------------------------------------------------------
def knn_arm(items: list[dict], idx: dict | None = None,
            nb=None) -> dict:
    """{ran, why, per_item {id: {k, tally, order, top}}, groups {group: {k,
    n}}, top1, recall}. For each held-out group g: k_g by an inner
    leave-one-group-out over every group but g (never g), then g's files
    vote against every group but g."""
    import example_knn as K
    if nb is None:
        if idx is None:
            st = K.index_state()
            if not st.get("fresh"):
                return {"ran": False, "why": "the example kNN index is not "
                        f"fresh ({st.get('why')})"}
            idx = K.load()
        if idx is None:
            return {"ran": False, "why": "no example kNN index"}
        if not idx["rows"]:
            return {"ran": False, "why": "the index holds no labelled chunk"}
        nb = K.neighbours_of(idx)
    fpos = {fid: f for f, fid in enumerate(nb.file_ids or [])}
    gpos = {g: i for i, g in enumerate(nb.groups)}
    per_item, per_group = {}, {}
    for it in items:
        f = fpos.get(it["id"])
        g = gpos.get(it["group"])
        if f is None or g is None:
            continue
        if g not in per_group:
            sel = K.logo(nb, range(len(nb.groups)), exclude={g})
            per_group[g] = {"k": sel["k"], "n": sel["n"],
                            "folds": sel["folds"], "k_max": sel["k_max"]}
        k = per_group[g]["k"]
        if k is None:
            per_item[it["id"]] = {"k": None, "tally": {}, "order": [],
                                  "top": [], "groups": []}
            continue
        t = K.tally(nb, f, k, {g})
        per_item[it["id"]] = dict(t, k=k)
    judged = [it for it in items if it["id"] in per_item and it["truth"]]
    top1 = sum(1 for it in judged if per_item[it["id"]]["order"][:1]
               and per_item[it["id"]]["order"][0] in it["truth"])
    rec_num = sum(len(set(it["truth"]) & set(per_item[it["id"]]["tally"]))
                  for it in judged)
    rec_n = sum(len(it["truth"]) for it in judged)
    return {"ran": bool(per_item), "why": None if per_item else
            "none of the onboarding's files is in the index",
            "per_item": per_item,
            "groups": {nb.groups[g]: v for g, v in per_group.items()},
            "top1": rate(top1, len(judged)),
            "recall_among_voted": rate(rec_num, rec_n),
            "items_in_index": len(per_item), "items": len(items)}


def combined(items, det, knn) -> dict:
    """detection vs detection + kNN, paired on the same items."""
    rows = [it for it in items if it["id"] in knn.get("per_item", {})]
    # (1) per (item, true package): found?
    a, b = [], []
    for it in rows:
        top = set(knn["per_item"][it["id"]]["top"])
        for p in it["truth"]:
            a.append(p in det[it["id"]])
            b.append(p in det[it["id"]] or p in top)
    # (2) per item: the predicted package SET equals the truth (the cost of
    # a false package counts here).
    ea, eb = [], []
    for it in rows:
        top = set(knn["per_item"][it["id"]]["top"])
        d = set(det[it["id"]])
        ea.append(d == set(it["truth"]))
        eb.append((d | top) == set(it["truth"]))
    return {"recall_pairs": paired(a, b, "detect", "detect_knn"),
            "exact_set": paired(ea, eb, "detect", "detect_knn")}


# ---------------------------------------------------------------------------
# Selection (a subprocess on a copy of the store).
# ---------------------------------------------------------------------------
def _store_env(tmp: str, decider: str) -> dict:
    import jobs
    import offline_stores
    import skills
    db = os.path.join(tmp, "jobs.sqlite3")
    if os.path.exists(jobs.DB):
        # A consistent snapshot (sqlite's backup API: the worker's database
        # may hold writes in its WAL that a file copy would miss).
        import sqlite3
        src = sqlite3.connect(f"file:{jobs.DB}?mode=ro", uri=True)
        dst = sqlite3.connect(db)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    sk = os.path.join(tmp, "skills")
    if os.path.isdir(skills.STORE):
        shutil.copytree(skills.STORE, sk)
    else:
        os.makedirs(sk, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in offline_stores.STORES
           or k in ("YAMADORI_ONBOARDING_DIR", "YAMADORI_EXAMPLES_DIR",
                    "YAMADORI_EXAMPLE_KNN_DIR")}
    env.update(YAMADORI_JOBS_DB=db, YAMADORI_SKILLS_DIR=sk,
               YAMADORI_SKILL_DECIDER=decider)
    if decider == "stub":
        env["YAMADORI_GPU_ROOM"] = "0"
    return env


def run_selection(items: list[dict], *, decider: str = "stub") -> dict:
    """The selection child over the items (imports stripped), on a copy of
    the store. {ok, why, pool, items: {id: {decisions, questions}}}."""
    import skill_packages as SP
    tmp = tempfile.mkdtemp(prefix="yamadori_pkg_eval_sel_")
    try:
        env = _store_env(tmp, decider)
        inp = {"tools": TOOLS,
               "vocabulary": SP.vocabulary(),
               "held_dirs": sorted(os.path.basename(d) for ds in
                                   SP.held().values() for d in ds),
               "items": [{"id": it["id"],
                          "messages": agent_step(it["stripped"], it["ext"])}
                         for it in items]}
        ip, op = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
        with open(ip, "w", encoding="utf-8") as f:
            json.dump(inp, f)
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--selection-child", ip, op], env=env,
                           capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0 or not os.path.exists(op):
            return {"ok": False, "why": f"the selection child exited "
                    f"{r.returncode}: " + (r.stderr or r.stdout)[-1500:]}
        with open(op, encoding="utf-8") as f:
            return dict(json.load(f), ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _selection_child(ip: str, op: str) -> int:
    """In the subprocess: the store is the copy; every other store is a temp
    path (offline_stores); the embedding stage and the fallback model are
    not run (replay_selection's rule)."""
    import offline_stores
    offline_stores.isolate("yamadori_pkg_eval_child_")
    import route
    import skill_packages as SP
    import skill_select
    import skills
    if os.environ.get("YAMADORI_SKILL_DECIDER", "stub") == "stub":
        skill_select.best_cosines = lambda q, pool: ({}, {
            "ok": False, "why": "offline evaluation: the embedding stage is "
                                "not run"})
    skill_select.ask_fallback = lambda s, u: (_ for _ in ()).throw(
        RuntimeError("offline evaluation: the fallback is not run"))
    with open(ip, encoding="utf-8") as f:
        data = json.load(f)
    pool = skills.armed()
    out = {"pool": len(pool),
           "decider": os.environ.get("YAMADORI_SKILL_DECIDER") or "stub",
           "items": {}}
    with SP.using(vocabulary=data["vocabulary"],
                  held_dirs=data["held_dirs"]):
        for it in data["items"]:
            msgs = it["messages"]
            try:
                rc = route.classify(msgs, client_tools=data["tools"],
                                    util={"utility": False}, gate=None,
                                    dbs={})["class"]
            except Exception:                                    # noqa: BLE001
                rc = None
            _t, rec, _st = skill_select.decide(msgs, rc, pool, data["tools"],
                                               None, key=f"pe:{it['id']}")
            out["items"][it["id"]] = {
                "route": rc,
                "decisions": [{k: d.get(k) for k in ("id", "name", "form",
                                                     "package", "trigger")}
                              for d in rec.get("decisions") or []],
                "questions": [{k: q.get(k) for k in (
                    "kind", "area", "packages", "hard", "leads",
                    "first_appearance", "picks")}
                    for q in (rec.get("match") or {}).get("questions")
                    or []],
                "packages": [p.get("package") for p in rec.get("packages")
                             or []]}
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return 0


def score_selection(items: list[dict], sel: dict, pool: list[dict]) -> dict:
    """The selection measures from the child's records."""
    import example_knn as K
    import skill_match
    import skill_packages as SP
    pk_of = {s["id"]: K.skill_packages_of(s) for s in pool}
    area_n = area_hit = lead_n = lead_hit = der_n = der_hit = 0
    deliv = wrong = 0
    wrong_rows = []
    for it in items:
        r = (sel.get("items") or {}).get(it["id"])
        if r is None:
            continue
        opened = {q["area"] for q in r["questions"]
                  if q.get("kind") == "package"}
        given = [d for d in r["decisions"] if d.get("form") in ("body",
                                                                "recall")]
        given_names = {d["name"] for d in given}
        for p in it["truth"]:
            a = skill_match.PACKAGE_AREA.get(p)
            if a:
                area_n += 1
                area_hit += a in opened
            lead = SP.canonical_skill(p, None, pool)
            if lead and any(s["name"] == lead for s in pool):
                lead_n += 1
                lead_hit += any(d["name"] == lead and d.get("form") == "body"
                                for d in given)
        der = set(K.derived_skills(it["truth"], it["identifiers"], pool,
                                   pk_of=pk_of))
        if der:
            der_n += 1
            der_hit += bool(der & {d["id"] for d in given})
        for d in given:
            pks = pk_of.get(d["id"]) or set()
            if not pks:
                continue
            deliv += 1
            if not (pks & set(it["truth"])):
                wrong += 1
                wrong_rows.append({"item": it["id"], "path": it["path"],
                                   "skill": d["name"],
                                   "packages": sorted(pks)})
        it["_given"] = sorted(given_names)
    return {"pool": sel.get("pool"), "decider": sel.get("decider"),
            "area_opened": rate(area_hit, area_n),
            "lead_on_first_appearance": rate(lead_hit, lead_n),
            "derived_label_delivered": dict(rate(der_hit, der_n),
                                            note=PROXY_NOTE),
            "package_skill_precision": dict(
                rate(deliv - wrong, deliv),
                note="delivered skills of a package the file does not use "
                     "count against it"),
            "wrong_package_deliveries": wrong_rows,
            "imports": "stripped", "items": len(items)}


def decider_arm(did: str, key: str) -> dict:
    """The model-decider arm: recorded "not run" unless a non-stub decider
    is configured AND its gpu job is registered with the worker."""
    name = (os.environ.get("YAMADORI_SKILL_DECIDER") or "stub").strip()
    if name.lower() == "stub":
        return {"state": "not run", "why": "not run: no model decider "
                "configured (YAMADORI_SKILL_DECIDER is stub)"}
    try:
        import onboarding
        registered = DECIDER_QUEUE in onboarding.HANDLERS
    except Exception:                                            # noqa: BLE001
        registered = False
    if not registered:
        return {"state": "not run", "decider": name,
                "why": f"a model decider is configured ({name}); its arm is "
                       f"a gpu job ({DECIDER_QUEUE}, idle-gated) that the "
                       "worker does not register yet"}
    import jobs
    jid = jobs.add(DECIDER_QUEUE, {"idle": True, "did": did, "key": key,
                                   "runs": DECIDER_RUNS}, lane="gpu",
                   dataset=did, stage="evaluate")
    return {"state": "queued", "decider": name, "job": jid,
            "runs": DECIDER_RUNS}


def handle_decider(job: dict, ctx=None) -> dict:
    """The gpu job: the selection child DECIDER_RUNS times with the
    configured decider, reported side by side into eval/<key>.json."""
    import onboarding
    import skills
    p = job.get("payload") or {}
    did, key = p["did"], p["key"]
    name = (os.environ.get("YAMADORI_SKILL_DECIDER") or "stub").strip()
    items = items_of(did)
    pool = skills.armed()
    runs = []
    for i in range(int(p.get("runs") or DECIDER_RUNS)):
        if ctx is not None:
            ctx.beat(f"decider run {i + 1}")
        sel = run_selection(items, decider=name)
        runs.append(score_selection(items, sel, pool) if sel.get("ok")
                    else {"ok": False, "why": sel.get("why")})
    rec = onboarding.read_json(did, os.path.join("eval", f"{key}.json"), {})
    rec["selection_model"] = {"decider": name, "runs": runs,
                              "n_runs": len(runs),
                              "rule": "PROTOCOL rule 10: runs side by side"}
    onboarding.write_json(did, os.path.join("eval", f"{key}.json"), rec)
    return {"runs": len(runs)}


# ---------------------------------------------------------------------------
# Regressions (subprocesses; the scripts copy the store themselves).
# ---------------------------------------------------------------------------
def parse_labels(out: str) -> dict:
    m = _ALL_LINE.search(out or "")
    n = _NEAR_LINE.search(out or "")
    rows = [ln.strip() for ln in (out or "").splitlines()
            if re.match(r"^    \S+: want ", ln)]
    return {"precision": float(m.group(1)) if m else None,
            "recall": float(m.group(2)) if m else None,
            "tp": int(m.group(3)) if m else None,
            "fp": int(m.group(4)) if m else None,
            "fn": int(m.group(5)) if m else None,
            "near_detected": int(n.group(1)) if n else None,
            "near_n": int(n.group(2)) if n else None,
            "wrong_rows": rows, "parsed": bool(m)}


def parse_daily(out: str) -> dict:
    ms = _CHECKS_LINE.findall(out or "")
    kinds = {}
    for line in (out or "").splitlines():
        if line.strip().startswith("by kind:"):
            kinds = {k: {"passed": int(a), "n": int(b)}
                     for k, a, b in _KIND.findall(line)}
    fails = [ln.strip()[5:] for ln in (out or "").splitlines()
             if ln.strip().startswith("FAIL ")]
    passed, n = (int(ms[-1][0]), int(ms[-1][1])) if ms else (None, None)
    return {"passed": passed, "n": n, "by_kind": kinds, "fails": fails,
            "parsed": bool(ms)}


def _run_script(argv: list[str]) -> tuple[int, str]:
    import jobs
    import skills
    env = dict(os.environ, YAMADORI_REPLAY_DB=os.path.abspath(jobs.DB),
               YAMADORI_REPLAY_SKILLS_DIR=os.path.abspath(skills.STORE))
    r = subprocess.run([sys.executable] + argv, env=env, capture_output=True,
                       text=True, cwd=ROOT)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


RUN_SCRIPT = _run_script      # tests replace it


def regressions(previous: dict | None = None) -> dict:
    out = {}
    code, text = RUN_SCRIPT([EVAL_PACKAGES, "--labels"])
    out["package_labels"] = dict(parse_labels(text), exit=code,
                                 script="bench/skills/eval_packages.py "
                                        "--labels")
    code, text = RUN_SCRIPT([DAILY_EVAL])
    out["daily"] = dict(parse_daily(text), exit=code,
                        script="bench/skills/test_daily_eval.py")
    prev_fails = set((((previous or {}).get("regressions") or {}).get(
        "daily") or {}).get("fails") or [])
    new = [f for f in out["daily"]["fails"] if f not in prev_fails
           and ("MUST" in f)]
    out["new_must_misses"] = new if previous else None
    out["compared_with"] = (previous or {}).get("key")
    return out


# ---------------------------------------------------------------------------
# Leakage (7.2) and coverage (7.3).
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    import skill_builder
    return skill_builder._norm(s)


def leakage(items: list[dict], sources: dict[str, str]) -> dict:
    """{item id: [{skill, lines}]}: a run of consecutive non-blank lines of
    the file, each found among a skill source's lines and together found in
    the source, of at least MIN_QUOTE_CHARS normalised characters (the quote
    check's rule and minimum)."""
    import skill_builder
    need = skill_builder.MIN_QUOTE_CHARS
    norm = {sid: _norm(t) for sid, t in sources.items() if t}
    line_sets = {sid: {_norm(ln) for ln in t.splitlines() if _norm(ln)}
                 for sid, t in sources.items() if t}
    out: dict[str, list] = {}
    for it in items:
        lines = it["text"].splitlines()
        nl = [_norm(ln) for ln in lines]
        for sid, ls in line_sets.items():
            hit = None
            for i in range(len(nl)):
                if not nl[i] or nl[i] not in ls:
                    continue
                run, j = [], i
                while j < len(nl) and nl[j] and nl[j] in ls:
                    run.append(nl[j])
                    j += 1
                    if len(" ".join(run)) >= need:
                        break
                joined = " ".join(run)
                if len(joined) >= need and joined in norm[sid]:
                    hit = [i + 1, j]
                    break
            if hit:
                out.setdefault(it["id"], []).append({"skill": sid,
                                                     "lines": hit})
    return out


def _skill_sources(pool: list[dict], packages: set, did: str) -> dict:
    """{skill id: its source text} for the armed skills of these packages
    and the onboarding's own skills."""
    import example_knn as K
    import skill_pipeline
    ids = {s["id"]: s for s in pool if K.skill_packages_of(s) & packages}
    try:
        import onboarding
        for s in onboarding.skills_of(did):
            if s["id"] not in ids and s.get("served_version"):
                ids[s["id"]] = {"id": s["id"], "version": s["served_version"]}
    except Exception:                                            # noqa: BLE001
        pass
    out = {}
    for sid, s in ids.items():
        try:
            _raw, text, _kind = skill_pipeline.source_texts(sid,
                                                            s.get("version"))
        except Exception:                                        # noqa: BLE001
            continue
        if text:
            out[sid] = text
    return out


def coverage(items: list[dict], pool: list[dict], pkg_name: str,
             did: str) -> dict:
    import example_knn as K
    import skill_packages as SP
    v = SP.vocabulary()
    sym, code_sym = v.get("symbols") or {}, v.get("code_symbols") or {}
    uses: dict[tuple, set] = collections.defaultdict(set)
    for it in items:
        refs = set(re.findall(r"[A-Za-z_$][\w$]*",
                              _code_only(it["stripped"])))
        for b, p in it["bindings"].items():
            if b in refs:
                uses[(p, b)].add(it["group"])
    texts = {s["id"]: " ".join([str(s.get("body") or "")] + [
        str(x.get("text") if isinstance(x, dict) else x)
        for x in s.get("items") or []] + K.code_topics(s)) for s in pool}
    helpers, blind = [], []
    for (p, b), gs in uses.items():
        rx = re.compile(r"(?<![\w$])" + re.escape(b) + r"(?![\w$])")
        if not any(rx.search(t) for t in texts.values()):
            helpers.append({"package": p, "name": b, "groups": len(gs)})
        owner = sym.get(b)
        if owner == p:
            continue
        if code_sym.get(b) == p:
            why = ("a plain English word: detects only in a code-like "
                   "context")
        elif owner:
            why = f"attributed to {owner} by the vocabulary"
        else:
            why = ("not unique to it: shared with another package, a "
                   "platform API, or not in the vocabulary")
        blind.append({"package": p, "name": b, "groups": len(gs),
                      "why": why})
    key = (lambda x: (-x["groups"], x["package"], x["name"]))
    unheld: dict[str, set] = collections.defaultdict(set)
    for it in items:
        for u in it["unheld"]:
            unheld[u].add(it["group"])
    ids_all = set()
    for it in items:
        ids_all |= set(it["identifiers"])
    no_example, no_topic = [], []
    for s in pool:
        if pkg_name not in K.skill_packages_of(s):
            continue
        topics = K.code_topics(s)
        if not topics:
            no_topic.append(s["name"])
        elif not any(K.topic_in(t, ids_all) for t in topics):
            no_example.append({"skill": s["name"], "topics": topics})
    return {"helpers_without_skill": sorted(helpers, key=key),
            "detector_blind_spots": sorted(blind, key=key),
            "onboard_next": sorted(({"package": u, "groups": len(g)}
                                    for u, g in unheld.items()),
                                   key=lambda x: (-x["groups"],
                                                  x["package"])),
            "skills_without_example": no_example,
            "skills_without_code_topic": sorted(no_topic),
            "black_hole_candidates": black_holes_of(did)}


def _code_only(text: str) -> str:
    import package_examples as PE
    return PE.code_only(text)


def black_holes_of(did: str) -> dict:
    """skill_boundaries.black_holes' rule, restricted to the onboarding's
    own skills as the CAPTURING side (every armed skill's should-cases, one
    pass per new skill instead of the whole store squared)."""
    try:
        import onboarding
        import skill_boundaries as B
        import skill_classify as C
        import skill_select
        import skill_tests
        mine = {s["id"] for s in onboarding.skills_of(did)}
        pool = B._served_rows()
        new = [s for s in pool if s["id"] in mine]
        if not new:
            return {"ran": True, "rows": [], "why": "the onboarding has no "
                    "armed skill"}
        areas = {s["id"]: B.area_of(s.get("rule") or {}) for s in pool}
        got: dict[str, dict] = {}
        for owner in pool:
            for c in B._should_cases(owner["id"], owner.get("served")):
                sig = C.request_signals(skill_tests.messages_of(c),
                                        c.get("route_class"), c.get("tools"))
                for s in new:
                    if s["id"] == owner["id"] or \
                            areas[s["id"]] & areas[owner["id"]]:
                        continue
                    det = C.match(s.get("rule") or {}, sig)
                    if skill_select.verdict(det["strength"], None) == "none":
                        continue
                    g = got.setdefault(s["id"], {"id": s["id"],
                                                 "name": s["name"],
                                                 "captured": 0,
                                                 "areas": set()})
                    g["captured"] += 1
                    g["areas"].add(",".join(sorted(areas[owner["id"]]))
                                   or "-")
        rows = sorted(({**g, "areas": sorted(g["areas"])}
                       for g in got.values()),
                      key=lambda g: (-len(g["areas"]), -g["captured"],
                                     g["name"]))
        return {"ran": True, "rows": rows,
                "rule": "skill_boundaries.black_holes (never retired "
                        "automatically)"}
    except Exception as e:                                       # noqa: BLE001
        return {"ran": False, "why": f"{type(e).__name__}: {e}"[:300]}


# ---------------------------------------------------------------------------
# Keys and the in-sample rule (7.2).
# ---------------------------------------------------------------------------
def git_head(root: str = ROOT) -> str:
    """The HEAD commit, read from .git (no git process); "unknown" when
    there is none."""
    g = os.path.join(root, ".git")
    try:
        with open(os.path.join(g, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head or "unknown"
        ref = head[4:].strip()
        p = os.path.join(g, *ref.split("/"))
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read().strip() or "unknown"
        with open(os.path.join(g, "packed-refs"), encoding="utf-8") as f:
            for ln in f:
                if ln.strip().endswith(" " + ref):
                    return ln.split()[0]
    except OSError:
        pass
    return "unknown"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()[:16]


def signatures(did: str, pool: list[dict]) -> dict:
    import example_knn as K
    import package_examples as PE
    import skill_packages as SP
    v = SP.vocabulary()
    idx = K.load()
    return {"examples": PE.manifest_sha256(did) or "none",
            "vocabulary": _sha({"symbols": v.get("symbols"),
                                "code_symbols": v.get("code_symbols")}),
            "knn": (idx or {}).get("signature") or "none",
            "armed": _sha(sorted(f"{s['id']}@{s.get('version')}"
                                 for s in pool)),
            "code": git_head(), "eval": EVAL_VERSION}


def result_key(sigs: dict) -> str:
    return _sha(sigs)


def _previous(did: str) -> list[dict]:
    import onboarding
    d = os.path.join(onboarding.out_dir(did), "eval")
    out = []
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    except OSError:
        return []
    for n in names:
        r = onboarding.read_json(did, os.path.join("eval", n))
        if isinstance(r, dict) and r.get("signatures"):
            out.append(r)
    out.sort(key=lambda r: r.get("at") or 0)
    return out


def in_sample_of(did: str, sigs: dict, key: str) -> tuple[bool, list, str | None]:
    """(in_sample, the changed signatures, the first result's key): a result
    on the same example set after an earlier result was read is in-sample
    when anything else changed since that first result."""
    prev = [r for r in _previous(did) if r.get("key") != key and
            (r.get("signatures") or {}).get("examples") == sigs["examples"]]
    if not prev:
        return False, [], None
    first = prev[0]
    changed = [k for k in ("vocabulary", "knn", "armed", "code", "eval")
               if (first.get("signatures") or {}).get(k) != sigs.get(k)]
    return True, changed, first.get("key")


# ---------------------------------------------------------------------------
# The stage.
# ---------------------------------------------------------------------------
def run(did: str, pkg: dict | None, *, beat=None, regressions_on: bool = True,
        selection_on: bool = True) -> dict:
    """Evaluate one onboarding (section 7, leave-one-group-out). Writes
    eval/<key>.json; returns {key, in_sample, summary, ...} or {skipped}."""
    import onboarding
    import skills
    beat = beat or (lambda s: None)
    why = None
    if not pkg:
        why = "no package: nothing to evaluate"
    elif pkg.get("ecosystem") != "npm":
        why = f"not supported for {pkg.get('ecosystem')}: JS/TS packages only"
    items = items_of(did) if not why else []
    if not why and not items:
        why = "no kept example files (examples.json says why)"
    if why:
        return {"skipped": why, "key": None, "in_sample": None,
                "summary": {"items": 0}}
    pool = skills.armed()
    sigs = signatures(did, pool)
    key = result_key(sigs)
    have = onboarding.read_json(did, os.path.join("eval", f"{key}.json"))
    if isinstance(have, dict) and have.get("key") == key:
        return dict(have, cached=True)
    prev = _previous(did)
    in_sample, changed, first = in_sample_of(did, sigs, key)
    beat(f"evaluate: detection on {len(items)} files")
    det_s = {it["id"]: detect_set(it["stripped"], it["ext"]) for it in items}
    det_k = {it["id"]: detect_set(it["text"], it["ext"]) for it in items}
    stripped = score_detection(items, det_s)
    kept = score_detection(items, det_k)
    sanity = {"recall": kept["overall"]["recall"],
              "ok": kept["overall"]["fn"] == 0,
              "why": ("the import rule found every imported held package"
                      if kept["overall"]["fn"] == 0 else
                      "BUG, not a result: with its imports kept, detect "
                      "missed imported held packages (listed)"),
              "misses": kept["misses"]}
    beat("evaluate: kNN, nested leave-one-group-out")
    knn = knn_arm(items)
    comb = combined(items, det_s, knn) if knn.get("ran") else {
        "ran": False, "why": knn.get("why")}
    sel = None
    if selection_on:
        beat("evaluate: selection (stub decider, a copy of the store)")
        got = run_selection(items)
        sel = (score_selection(items, got, pool) if got.get("ok") else
               {"ran": False, "why": got.get("why")})
    else:
        sel = {"ran": False, "why": "switched off by the caller"}
    reg = None
    if regressions_on:
        beat("evaluate: regressions (package labels, daily eval)")
        reg = regressions(prev[-1] if prev else None)
    truth_pk = set()
    for it in items:
        truth_pk |= set(it["truth"])
    beat("evaluate: leakage and coverage")
    leak = leakage(items, _skill_sources(pool, truth_pk | {pkg["name"]},
                                         did))
    cov = coverage(items, pool, pkg["name"], did)
    groups = {it["group"] for it in items}
    summary = {
        "items": len(items), "groups": len(groups),
        "items_with_held_import": sum(1 for it in items if it["truth"]),
        "detect_stripped_recall": stripped["overall"]["recall"],
        "detect_stripped_precision": stripped["overall"]["precision"],
        "detect_kept_sanity": sanity["ok"],
        "knn_top1": knn.get("top1"),
        "paired_p": ((comb.get("exact_set") or {}).get("p")
                     if knn.get("ran") else None),
        "paired_n": ((comb.get("exact_set") or {}).get("n")
                     if knn.get("ran") else None),
        "seen_by_skill_source": rate(len(leak), len(items)),
        "in_sample": in_sample}
    rec = {"key": key, "did": did, "package": f"{pkg['name']}@"
           f"{pkg.get('version')}", "eval_version": EVAL_VERSION,
           "signatures": sigs, "in_sample": in_sample,
           "in_sample_changes": changed, "in_sample_since": first,
           "split": {"rule": "leave-one-group-out (operator decision 1, "
                             "2026-09-27): every group held out once; its "
                             "votes come from every other group; k chosen "
                             "by an inner leave-one-group-out on those "
                             "other groups only",
                     "groups": len(groups)},
           "detection_stripped": stripped, "detection_kept": kept,
           "sanity": sanity, "knn": knn, "detect_vs_detect_knn": comb,
           "selection": sel, "selection_model": decider_arm(did, key),
           "regressions": reg,
           "leakage": {"rule": "a run of consecutive lines, each among a "
                               "skill source's lines and together in the "
                               "source, of at least MIN_QUOTE_CHARS "
                               "normalised characters (skill_builder)",
                       "items": leak,
                       "seen_by_skill_source": rate(len(leak), len(items))},
           "coverage": cov,
           "per_item": [{"id": it["id"], "path": it["path"],
                         "group": it["group"], "truth": it["truth"],
                         "detect_stripped": det_s[it["id"]],
                         "detect_kept": det_k[it["id"]],
                         "knn_top": ((knn.get("per_item") or {}).get(
                             it["id"]) or {}).get("top"),
                         "seen_by_skill_source": it["id"] in leak,
                         "selected": it.get("_given")}
                        for it in items],
           "summary": summary, "at": time.time()}
    onboarding.write_json(did, os.path.join("eval", f"{key}.json"), rec)
    return rec


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--selection-child":
        sys.exit(_selection_child(sys.argv[2], sys.argv[3]))
    if len(sys.argv) >= 3 and sys.argv[1] == "show":
        import onboarding
        print(json.dumps(onboarding._latest_eval(sys.argv[2]), indent=1,
                         default=str))
    else:
        print(__doc__)
