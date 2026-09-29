#!/usr/bin/env python
"""MEASURE A MODEL'S RENDERING PROFILE (GPU): the same skill items, rendered
each way the injector can render them, each paired with the same probe
WITHOUT them -- the per-model half of the operator's "hand skills based on
the same source evidence over to each model with the best results".

    python bench/skills/render_probe.py --model bonsai --effort medium \\
        [--skills 12] [--variants list,table,first_person,top1] [--dry-run]
    python bench/skills/render_probe.py --model bonsai --report

THE PAIR, AS skill_prove MAKES IT (mcp/skill_prove.py: SkillsBench
2602.12670 and ASI 2504.06821 -- a skill needs execution-checked proof): a
probe task from the skill's OWN should-cases (skill_prove.floor_probes: a
case that quotes an item is skipped), checks BY CODE (skill_prove.
floor_checks: each DO / WHEN item's code spans must appear, each DO NOT
item's must not; every fenced block must parse -- skill_prove.run_check),
the SAME seed and sampling on every side, the answer allowance tiers.A_MIN,
thinking capped at tiers.HELPER_THINKING (the pipeline's cap for this kind
of job, so every model is measured under one bound). Only the injection
differs:

  without       the probe alone
  list          skill_inject.render, format list, the note voice at the
                end of the user turn (the research default)
  table         format table, the note voice
  first_person  the same items in the model's own voice ("I'll ...", "I
                won't ..."), injected as TEXT at the user turn's end like
                list and table
  first_person_prefill  the same items as ONE line of the model's own
                reasoning, prefilled (an assistant turn with reasoning_content
                and no content -- proxy.directive_prefill's channel, kept by
                the operator 2026-09-29); it COUNTS only once bench/skills/
                prefill_check.py has shown the served template leaves the
                think block open after it and the generation continues it
  top1          format list, ONLY the item behind the probe's first code
                check (size: one relevant item vs the skill's all)

Every variant injects the skill's items (the doubt filter applied, at most
the profile's max_items, in the skill's order) through skill_inject.render
itself: what is measured is what serves.

THE DECISION, per variant, paired per probe and per check (skill_prove.pair):
`better` (failed without, passed with), `worse` (passed without, failed
with), net = better - worse. The profile's pick is the variant with the
largest net among those with no more worse than better; a tie goes to the
research default (list, note). A model where every variant's net is below
zero is reported as HURT by injection -- a finding for the operator, never
hidden by a default.

ONE GPU CONSUMER AT A TIME: refuses unless `--model` is loaded (GET /running,
which never loads anything).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "inject", "results")
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)
# The stores this bench reads go through replay_selection's isolation (a
# temp copy of the jobs database; the skill library read in place): the
# bench never writes a live store.
import replay_selection  # noqa: E402,F401

PROBE_VERSION = "render-probe/1"
VARIANTS = ("list", "table", "first_person", "first_person_prefill",
            "top1")
AREAS = ("koota", "pmndrs_math", "r3f", "threejs", "react", "typegpu",
         "webgpu", "typescript", "rust", "emscripten", "python", "c")


def library_rows() -> list[dict]:
    """Armed skills as (row, ver) with their tests.json, from the store."""
    import skill_md
    import skills
    by_name = {}
    for p in glob.glob(os.path.join(skills.LIBRARY if hasattr(
            skills, "LIBRARY") else os.path.join(ROOT, "index", "skills",
                                                 "library"), "*",
            "SKILL.md")):
        folder = os.path.dirname(p)
        by_name[os.path.basename(folder)] = folder
    out = []
    for s in skills.armed():
        folder = by_name.get(s.get("name") or "")
        if not folder:
            continue
        try:
            with open(os.path.join(folder, "tests.json"),
                      encoding="utf-8") as f:
                tests = json.load(f)
            with open(os.path.join(folder, "SKILL.md"),
                      encoding="utf-8") as f:
                text = f.read()
        except (OSError, ValueError):
            continue
        ver = {"text": text, "classify": s.get("rule") or {},
               "tests": tests, "validate": {"items": skill_md.parse(text)
                                            .get("items") or []}}
        out.append({"row": s, "ver": ver})
    return out


def pick_skills(n: int) -> list[dict]:
    """Up to n skills, at most 2 per area in AREAS order, each with a probe
    whose language is known and at least one present/absent code check."""
    import skill_prove as SP
    import skill_select
    per = collections.Counter()
    out = []
    for x in sorted(library_rows(), key=lambda x: x["row"]["id"]):
        area = skill_select.area_of(x["row"].get("rule") or {})
        if area not in AREAS or per[area] >= 2:
            continue
        probes, _sk = SP.floor_probes(x["ver"])
        probes = [p for p in probes if p["language"]]
        checks, _dr = SP.floor_checks(SP._items(x["ver"]),
                                      SP._topics(x["ver"]))
        checks = [c for c in checks if not __import__("re").search(
            c["pattern"], probes[0]["task"])] if probes else []
        if not probes or not checks:
            continue
        per[area] += 1
        x.update(area=area, probe=probes[0], checks=checks)
        out.append(x)
    out.sort(key=lambda x: (AREAS.index(x["area"]), x["row"]["id"]))
    return out[:n]


def rendering(x: dict, variant: str, profile: dict) -> dict:
    import skill_inject as I
    items = [it for it in I.skill_items(x["row"]) if not it["doubt"]]
    if variant == "top1":
        first = x["checks"][0].get("item") or 1
        items = [items[min(first, len(items)) - 1]] if items else []
        prof = dict(profile, format="list", voice="note")
    elif variant == "table":
        prof = dict(profile, format="table", voice="note")
    elif variant == "first_person":
        prof = dict(profile, format="list", voice="first_person")
    elif variant == "first_person_prefill":
        prof = dict(profile, format="list", voice="first_person_prefill")
    else:
        prof = dict(profile, format="list", voice="note")
    items = items[:max(int(prof.get("max_items") or I.DEFAULT_MAX_ITEMS), 1)]
    return I.render(items, prof)


def body(task: str, r: dict | None, model: str, effort: str, seed: int
         ) -> dict:
    import skill_prove as SP
    import tiers
    import model as M
    user = task + ((r or {}).get("text") or "")
    b = M.shape({"messages": [{"role": "system",
                               "content": SP.ANSWER_SYSTEM},
                              {"role": "user", "content": user}],
                 "max_tokens": tiers.A_MIN}, effort=effort, role="helper",
                step_cap=tiers.HELPER_THINKING)
    b.pop("_share", None)
    b["model"] = model
    b["seed"] = seed
    if r and r.get("prefill"):
        # the prefill channel as proxy.directive_prefill delivers it: an
        # assistant turn with the line as reasoning_content and no content
        b["messages"].append({"role": "assistant", "content": "",
                              "reasoning_content": r["prefill"]})
    return b


def generate(b: dict) -> tuple[str | None, str, float, dict]:
    """(answer or None, why, seconds, gen): `gen` is what the generation
    cost -- finish_reason, completion tokens, reasoning characters and the
    reasoning tokens where the server reports them -- so a rendering that
    only wins by thinking longer is visible (coordinator, 2026-09-29)."""
    import model as M
    t0 = time.time()
    try:
        d = M.post(b)
    except Exception as e:                                       # noqa: BLE001
        return None, f"not run: {type(e).__name__}: {e}"[:200], \
            round(time.time() - t0, 2), {}
    ch = ((d.get("choices") or [{}])[0]) if isinstance(d, dict) else {}
    msg = ch.get("message") or {}
    usage = d.get("usage") or {}
    gen = {"finish_reason": ch.get("finish_reason"),
           "completion_tokens": usage.get("completion_tokens"),
           "reasoning_tokens": (usage.get("completion_tokens_details")
                                or {}).get("reasoning_tokens"),
           "reasoning_chars": len(msg.get("reasoning_content") or ""),
           "content_chars": len(msg.get("content") or "")}
    try:
        return M.answer(d), "answered", round(time.time() - t0, 2), gen
    except M.BudgetEvent as e:
        return None, f"budget event: {e}"[:200], \
            round(time.time() - t0, 2), gen


def run(model: str, effort: str, n: int, variants) -> list[dict]:
    import skill_inject as I
    import skill_prove as SP
    prof = I.profile_for(model)
    rows = []
    for x in pick_skills(n):
        p = x["probe"]
        seed = SP.seed_of(x["row"]["id"], 1, p["task"])
        packages = SP._packages(x["row"]["id"], x["row"].get("rule") or {})
        base, why0, s0, g0 = generate(body(p["task"], None, model, effort,
                                           seed))
        rec = {"v": PROBE_VERSION, "model": model, "effort": effort,
               "skill": x["row"]["id"], "name": x["row"].get("name"),
               "area": x["area"], "probe_sha": SP._sha(p["task"])[:16],
               "language": p["language"], "seed": seed,
               "without": {"answered": base is not None, "why": why0,
                           "s": s0, "gen": g0}, "variants": {}}
        chk = [{"id": "parse", "kind": "parse"}] + [
            dict(c, id=f"{c['kind']}{k}") for k, c in enumerate(x["checks"])]
        w0 = {c["id"]: SP.run_check(c, base, p["language"], packages)[0]
              if base is not None else None for c in chk}
        for v in variants:
            r = rendering(x, v, prof)
            ans, why, s, g1 = generate(body(p["task"], r, model, effort,
                                            seed))
            pairs = []
            for c in chk:
                w1 = SP.run_check(c, ans, p["language"], packages)[0] \
                    if ans is not None else None
                pairs.append({"check": c["id"], "kind": c["kind"],
                              "without": w0[c["id"]], "with": w1,
                              "pair": SP.pair(w0[c["id"]], w1)})
            rec["variants"][v] = {"answered": ans is not None, "why": why,
                                  "s": s, "gen": g1, "chars": r["chars"],
                                  "tokens": r["tokens"], "pairs": pairs}
        rows.append(rec)
        print(f"  {x['row'].get('name')}: " + ", ".join(
            f"{v} {sum(q['pair'] == 'better' for q in rec['variants'][v]['pairs'])}"
            f"/{sum(q['pair'] == 'worse' for q in rec['variants'][v]['pairs'])}"
            for v in variants), flush=True)
    return rows


def prefill_confirmed(model: str) -> bool:
    """bench/skills/prefill_check.py's verdict for `model` (its results
    file): the think block stays open and the generation continues it."""
    try:
        with open(os.path.join(RESULTS, f"prefill_check_{model}.json"),
                  encoding="utf-8") as f:
            return bool(json.load(f).get("confirmed"))
    except (OSError, ValueError):
        return False


def decide(rows: list[dict], model: str | None = None) -> dict:
    """Per variant: better, worse, same, not_run, net; and the pick. The
    prefill variant is eligible only where prefill_check confirmed the
    engine's behaviour for the model."""
    tally = {}
    for r in rows:
        for v, x in r["variants"].items():
            t = tally.setdefault(v, collections.Counter())
            for q in x["pairs"]:
                t[q["pair"]] += 1
            t["probes"] += 1
            t["unanswered"] += int(not x["answered"])
            g, g0 = x.get("gen") or {}, (r.get("without") or {}).get(
                "gen") or {}
            t["reasoning_chars"] += int(g.get("reasoning_chars") or 0)
            t["reasoning_chars_without"] += int(g0.get("reasoning_chars")
                                                or 0)
            t["length_finishes"] += int(g.get("finish_reason") == "length")
    res = {v: {"better": t["better"], "worse": t["worse"],
               "same": t["same"], "not_run": t["not_run"],
               "net": t["better"] - t["worse"], "probes": t["probes"],
               "unanswered": t["unanswered"],
               "length_finishes": t["length_finishes"],
               # thinking beside the win: mean reasoning characters WITH
               # the rendering vs the same probes WITHOUT it
               "reasoning_chars_mean": round(t["reasoning_chars"]
                                             / max(t["probes"], 1)),
               "reasoning_chars_mean_without": round(
                   t["reasoning_chars_without"] / max(t["probes"], 1))}
           for v, t in tally.items()}
    pre_ok = prefill_confirmed(model or (rows[0]["model"] if rows else ""))
    ok = {v: x for v, x in res.items() if x["worse"] <= x["better"]
          and x["unanswered"] < x["probes"]
          and (v != "first_person_prefill" or pre_ok)}
    if ok:
        top = max(x["net"] for x in ok.values())
        tied = sorted(v for v, x in ok.items() if x["net"] == top)
        pick = "list" if "list" in tied else tied[0]
    else:
        pick = None
    hurt = bool(res) and all(x["net"] < 0 for x in res.values())
    return {"variants": res, "pick": pick, "hurt": hurt,
            "prefill_confirmed": pre_ok,
            "n_probes": len(rows)}


def held_out(rows: list[dict], model: str | None = None) -> dict:
    """LEAVE-ONE-AREA-OUT (coordinator, 2026-09-29: a choice is selected on
    some data and scored on data it never saw; a setting that does not hold
    up held out does not ship): for each skill AREA, the variant is picked
    (decide) on the other areas' probes and its paired checks are scored on
    the held-out area's. Pooled over areas: better and worse among the
    checks that changed, and the share better with its Wilson 95% interval.
    SHIPS when that interval's lower end is above 0.5 -- the pick makes
    more checks pass than fail on probes it never saw."""
    import math
    by_area = collections.defaultdict(list)
    for r in rows:
        by_area[r["area"]].append(r)
    better = worse = 0
    picks = collections.Counter()
    for area, test in by_area.items():
        train = [r for r in rows if r["area"] != area]
        pick = decide(train, model)["pick"] if train else None
        picks[str(pick)] += 1
        if not pick:
            continue
        for r in test:
            for q in (r["variants"].get(pick) or {}).get("pairs") or []:
                better += q["pair"] == "better"
                worse += q["pair"] == "worse"
    n = better + worse
    z = 1.959964
    if n:
        p = better / n
        den = 1 + z * z / n
        mid = p + z * z / (2 * n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        ci = [round((mid - half) / den, 4), round((mid + half) / den, 4)]
    else:
        ci = [None, None]
    return {"areas": len(by_area), "picks": dict(picks), "better": better,
            "worse": worse, "share_better_ci95": ci,
            "ships": ci[0] is not None and ci[0] > 0.5}


def running(upstream: str) -> list[str]:
    with urllib.request.urlopen(f"{upstream}/running", timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = d.get("running") if isinstance(d, dict) else d
    return [str((x or {}).get("model")) for x in rows or []
            if isinstance(x, dict)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--skills", type=int, default=16)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"render_{a.model}.jsonl")
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    if a.report:
        with open(out, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        print(json.dumps({"all": decide(rows, a.model),
                          "held_out_by_area": held_out(rows, a.model)},
                         indent=1))
        return 0
    if a.dry_run:
        xs = pick_skills(a.skills)
        print(json.dumps({"skills": [{"name": x["row"].get("name"),
                                      "area": x["area"],
                                      "language": x["probe"]["language"],
                                      "checks": len(x["checks"])}
                                     for x in xs],
                          "generations": len(xs) * (1 + len(variants))},
                         indent=1))
        return 0
    import model as M
    have = running(M.UPSTREAM)
    if a.model not in have:
        print(f"REFUSED: {a.model} is not loaded (GET /running: {have})")
        return 2
    import cancel
    import max_mode
    with cancel.bound(cancel.Token()):
        max_mode.set_current(a.model)
        rows = run(a.model, a.effort, a.skills, variants)
    with open(out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"all": decide(rows, a.model),
                      "held_out_by_area": held_out(rows, a.model)},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
