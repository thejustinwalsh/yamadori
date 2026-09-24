#!/usr/bin/env python
"""The benchmark page's server half (mcp/dash_results.py), on fixtures.

No GPU, no server, no network. Every section reads files that a running
benchmark is still writing, so each one is checked on the shapes it will
actually meet: absent, empty, half-written, running, finished, frozen.

WHAT IS ASSERTED

  - a section with nothing to read says `empty` and how to produce it, and a
    half-written JSON file costs one poll, never the section
  - LiveBench: the frozen run is a record and gains no arm from a progress
    log; a run with only progress logs is `running` with its counts; the
    retired pre-fix arm never appears; each arm's legend comes from the
    harness's own arm table (a tier arm reads "auto" for what selection
    decides)
  - SWE-bench: agent/eval errors are counted apart and never scored; Wilson
    bounds are None at n=0 (not a measured zero); the leaderboard is parsed
    from the doc's section 7 with its wrapped caption
  - images: what an image costs is read off images that were made; a
    killed run is listed with its kill, not folded into the VRAM figure
  - domain: a stale row is excluded and its reason reported; a contaminated
    domain is flagged, not dropped; no comparison carries a verdict
  - the published model card says it was not measured here
  - one section raising cannot take the others down
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dash_results as dr  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def _w(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _jl(path: str, rows: list[dict]) -> None:
    _w(path, "".join(json.dumps(r) + "\n" for r in rows))


def _walk(o, key: str) -> bool:
    """Whether `key` appears anywhere in a nested payload."""
    if isinstance(o, dict):
        return key in o or any(_walk(v, key) for v in o.values())
    if isinstance(o, list):
        return any(_walk(v, key) for v in o)
    return False


# --------------------------------------------------------------- tables ---
def test_markdown_tables():
    doc = ("# T\n\n## 7. Published scores\n\n**A. Same scaffold: mini v2, bash only.\n"
           "These are like for like.**\n\n| model | open weights | Verified | date |\n"
           "|---|---|---|---|\n| M1 | no | 76.8 | 2026 |\n| M2 | yes | **1,234.5** | x |\n\n"
           "Prose.\n\n**B. Older family.**\n\n| model | size | Verified |\n|---|---:|---|\n"
           "| M3 | 24B | 56.4 |\n\n## 8. Other\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    t = dr.md_tables(doc)
    check(len(t) == 3, "three tables found", str(len(t)))
    check(t[0]["caption"].endswith("These are like for like.**"),
          "a bold caption that wraps onto the next line is read whole", t[0]["caption"])
    check(t[0]["heading"] == "7. Published scores", "each table keeps its heading")
    check(len(t[0]["rows"]) == 2, "the separator row is not a data row")
    check(dr.num("**1,234.5**") == 1234.5 and dr.num("–") is None and dr.num(None) is None,
          "num() reads bold, comma'd numbers and returns None for a dash")
    d = tempfile.mkdtemp(prefix="yamadori_test_dr_")
    try:
        p = os.path.join(d, "SWE.md")
        _w(p, doc)
        lb = dr.swebench_leaderboard(p)
        keys = [x["key"] for x in lb["tables"]]
        check(keys == ["A", "B"], "section 7 tables A and B only (section 8 ignored)", str(keys))
        a = lb["tables"][0]
        check(a["same_scaffold"] and not lb["tables"][1]["same_scaffold"],
              "only table A is marked same-scaffold")
        check(a["rows"][0] == {"model": "M1", "verified": 76.8,
                               "detail": {"open weights": "no", "date": "2026"}},
              "a leaderboard row keeps its model, score and other columns", json.dumps(a["rows"][0]))
        check(lb["n_instances"] == 500, "the leaderboard is labelled as the full 500")
        check(dr.swebench_leaderboard(os.path.join(d, "missing.md"))["tables"] == [],
              "a missing doc gives no tables, not an exception")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------ swebench ---
def test_swebench():
    d = tempfile.mkdtemp(prefix="yamadori_test_dr_")
    try:
        doc = os.path.join(d, "SWE.md")
        _w(doc, "## 7. Published scores\n\n**A. Same.**\n\n| model | Verified |\n|---|---|\n| X | 70.0 |\n")
        e = dr.swebench_summary(os.path.join(d, "none"), doc)
        check(e["state"] == "empty" and "run.py" in e["how"],
              "no run: empty, and says how to produce one")
        check(e["leaderboard"]["tables"][0]["rows"][0]["verified"] == 70.0,
              "the published leaderboard is shown even before any run")
        run = os.path.join(d, "r1")
        _w(os.path.join(run, "instances.json"), json.dumps({"subset": "verified-mini",
                                                           "order": [f"i{i}" for i in range(50)]}))
        base = {"run_id": "r1", "arm": "bonsai", "steps": 10, "seconds": 100.0,
                "prompt_tokens": 1000, "completion_tokens": 50, "exit_status": "Submitted",
                "finish_reasons": {"tool_calls": 10},
                "x_yamadori": {"turns": 10, "hint_turns": 0, "tiers": {"max": 10}}}
        _jl(os.path.join(run, "results.jsonl"), [
            dict(base, instance="i0", eval_status="unresolved"),
            dict(base, instance="i0", eval_status="resolved"),      # re-run wins
            dict(base, instance="i1", eval_status="empty_patch"),
            dict(base, instance="i2", eval_status="agent_error", steps=3),
            dict(base, instance="i3", eval_status="eval_error"),
        ])
        with open(os.path.join(run, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write('{"run_id": "r1", "arm": "bon')                 # killed mid-line
        _w(os.path.join(run, "run.log"), "a\n04:37 STOPPED by operator decision: x\n")
        s = dr.swebench_summary(d, doc)
        a = s["arms"][0]
        check(s["state"] == "ready" and s["planned"] == 50, "planned count from instances.json")
        check(s["bad_lines"] == 1, "a half-written last line is counted, not fatal", str(s["bad_lines"]))
        check((a["resolved"], a["scored"], a["attempted"]) == (1, 2, 4),
              "resolved of scored: empty_patch scores as unresolved; agent/eval errors are not scored",
              str((a["resolved"], a["scored"], a["attempted"])))
        check(a["outcomes"].get("agent_error") == 1 and a["outcomes"].get("eval_error") == 1,
              "errors are counted per kind")
        lo, hi = a["lo"], a["hi"]
        check(lo is not None and 0 < lo < 0.5 < hi <= 1, "Wilson bounds from the harness", f"{lo} {hi}")
        check(s["stopped"] and "STOPPED" in s["stopped"], "the run log's stop line is carried")
        check(a["mechanisms"]["turns"] == 40 and a["tiers_reported"] == {"max": 40},
              "x_yamadori counters are summed over the instances")
        check(dr._wilson(0, 0) == (None, None), "Wilson at n=0 is (None, None), never a measured 0")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------- livebench ---
def _lb_summary(arms: dict) -> dict:
    return {"run_id": "x", "release": ["2024-11-25"], "population": {"coding": 128},
            "arms": arms, "arm_labels": {a: f"label {a}" for a in arms}}


def _lb_arm(score, n=21):
    return {"overall": {"score": score, "ci95": [score - 10, score + 10],
                        "categories_included": ["coding"]},
            "categories": {"coding": {"score": score, "ci95": [score - 10, score + 10], "n": n,
                                      "tasks": {"LCB_generation": {"score": score,
                                                                   "ci95": [0, 100], "n": n}},
                                      "finish_reason": {"stop": n}}},
            "status": {"ok": n}, "finish_reason": {"stop": n}}


def test_livebench():
    d = tempfile.mkdtemp(prefix="yamadori_test_dr_")
    try:
        e = dr.livebench_summary(os.path.join(d, "none"))
        check(e["state"] == "empty" and e["how"], "no runs: empty, with how")

        old = os.path.join(d, "lb-1")
        _w(os.path.join(old, "FROZEN"), "frozen: min_p 0.05 record")
        _w(os.path.join(old, "summary.json"), json.dumps(_lb_summary({"bonsai": _lb_arm(70.0)})))
        _w(os.path.join(old, "comparison.json"), json.dumps({"same_questions": {"bonsai": {
            "Ref A": {"overall": 60.0, "n": 21, "missing": 0, "categories": {"coding": 60.0}},
            "Ref B": {"overall": 90.0, "n": 21, "missing": 0, "categories": {"coding": 90.0}}}}}))
        _w(os.path.join(old, "logs", "progress_yamadori_coding.log"),
           "[02:00:00] yamadori/coding chunk 1 rc=0 5s  answered 2/21 err_rows 0  finish=[]\n")
        _jl(os.path.join(old, "rows_bonsai_coding.jsonl"),
            [{"arm": "bonsai", "id": str(i), "status": "ok"} for i in range(21)])

        cur = os.path.join(d, "lb-2")
        _w(os.path.join(cur, "condition.json"), json.dumps({"min_p": 0.0}))
        _w(os.path.join(cur, "logs", "progress_bonsai_instruction_following.log"),
           "[08:02:21] bonsai/instruction_following chunk 1 rc=0 1747s  answered 4/200 err_rows 1  "
           "finish=['stop']\n")
        _w(os.path.join(cur, "logs", "progress_yamadori_coding.log"), "starting\n")
        _w(os.path.join(cur, "prefix_yamadori_features_arm", "README.txt"), "PRE-FIX")
        _w(os.path.join(cur, "logs", "progress_yamadori_features_prefix_coding.log"),
           "[07:30:00] x answered 9/21 err_rows 0\n")
        _w(os.path.join(cur, "summary.json"), '{"run_id": "lb-2", "arms": {')   # mid-write
        os.utime(old, (1, 1))

        s = dr.livebench_summary(d)
        runs = {r["run_id"]: r for r in s["runs"]}
        check(s["current"] == "lb-2", "current = newest run without FROZEN", str(s["current"]))
        f = runs["lb-1"]
        check(f["frozen"] and [a["arm"] for a in f["arms"]] == ["bonsai"],
              "a frozen run gains no arm from a progress log",
              str([a["arm"] for a in f["arms"]]))
        check(f["arms"][0]["overall"]["score"] == 70.0 and f["state"] == "ready",
              "the frozen run's scores are read from its summary")
        check([r["model"] for r in f["same_question_refs"]["bonsai"]] == ["Ref B", "Ref A"],
              "same-question references, highest first")
        h = f["arms"][0]["mechanism_health"]
        check(h == {"answers": 21, "recorded": 0},
              "rows scored before mechanisms were recorded say so (recorded 0)", json.dumps(h))
        c = runs["lb-2"]
        names = [a["arm"] for a in c["arms"]]
        check(c["state"] == "running", "only progress logs: running", c["state"])
        check(not any(n.startswith("yamadori_features") for n in names),
              "the retired pre-fix arm never appears", str(names))
        p = {(x["arm"], x["category"]): x for x in c["progress"]}
        q = p.get(("bonsai", "instruction_following")) or {}
        check((q.get("answered"), q.get("of"), q.get("err_rows"), q.get("at")) == (4, 200, 1, "08:02:21"),
              "progress: answered/of/err_rows/time from the last progress line", json.dumps(q))
        check((p.get(("yamadori", "coding")) or {}).get("answered") == 0,
              "a progress log with no progress line reads 0 answered")
        leg = {x["arm"]: x for x in c["legend"]}
        check(leg.get("bonsai", {}).get("kind") == "features" and leg["bonsai"]["retrieval"] == "off",
              "bonsai's legend is its forced-off header", json.dumps(leg.get("bonsai")))
        y = leg.get("yamadori", {})
        check(y.get("kind") == "tier" and y.get("tier") == "max" and y.get("deep_thinking") == "auto"
              and y.get("thinking_sent") == "xhigh",
              "yamadori's legend is tier max: deep thinking auto, xhigh sent", json.dumps(y))

        # the same run once the summary lands, with a paired block
        summ = _lb_summary({"bonsai": _lb_arm(50.0, 2), "yamadori": _lb_arm(100.0, 2)})
        summ["paired_yamadori_minus_bonsai"] = {
            "n_pairs": 2, "diff_overall": 50.0, "ci95_overall": [0.0, 100.0],
            "categories": {"coding": {"n_pairs": 2, "a_score": 50.0, "b_score": 100.0, "diff": 50.0,
                                      "ci95": [0, 100], "binary": True, "b_only_correct": 1,
                                      "a_only_correct": 0, "mcnemar_p": 1.0}}}
        summ["arms"]["bonsai"]["mechanism_health"] = {
            "answers": 2, "stack_errors": 0, "mechanisms": {"hints": {"allowed": 0, "ran": 0}}}
        _jl(os.path.join(cur, "rows_bonsai_coding.jsonl"),
            [{"arm": "bonsai", "id": str(i), "status": "ok"} for i in range(3)])
        _w(os.path.join(cur, "summary.json"), json.dumps(summ))
        bh = next(a for a in {r["run_id"]: r for r in dr.livebench_summary(d)["runs"]}["lb-2"]["arms"]
                  if a["arm"] == "bonsai")["mechanism_health"]
        check((bh.get("recorded"), bh.get("answers")) == (2, 3),
              "summary.json's health is used: its answers are the recorded n, beside the answers scored",
              json.dumps(bh))
        mech = dr._bench_sub("livebench", "mechanisms")
        xy = {"tier": "max", "hops": 2, "tools_gate": {"offer": True},
              "selection": {"investigate": True, "fanout_n": 3,
                            "signals": {"allowed": {"investigate": True, "fanout": 3, "hints": True}}},
              "investigate": {"ran": True, "ok": True, "hops": 4, "injected": True, "cited": ["a"]},
              "fanout": {"n": 3, "asked": 3, "selection": "compile", "winner": 1}}
        _jl(os.path.join(cur, "rows_yamadori_coding.jsonl"),
            [{"arm": "yamadori", "id": "1", "status": "ok", "mechanisms": mech.record(xy)},
             {"arm": "yamadori", "id": "2", "status": "ok", "mechanisms": mech.record(dict(
                 xy, investigate={"ran": True, "ok": False, "why": "timeout"}))}])
        c = {r["run_id"]: r for r in dr.livebench_summary(d)["runs"]}["lb-2"]
        yh = next(a for a in c["arms"] if a["arm"] == "yamadori")["mechanism_health"]
        dt = (yh.get("mechanisms") or {}).get("deep_thinking") or {}
        check(yh.get("recorded") == 2 and yh.get("stack_errors") == 1 and dt.get("ran") == 2
              and dt.get("produced") == 1,
              "mechanism health is mechanisms.health over the rows: ran 2, produced 1, one stack error",
              json.dumps(yh)[:300])
        pr = c["paired"][0] if c["paired"] else {}
        check(c["state"] == "ready" and (pr.get("a"), pr.get("b")) == ("bonsai", "yamadori"),
              "paired_<b>_minus_<a> is read as b against a", json.dumps(pr)[:200])
        cc = pr.get("categories", {}).get("coding", {})
        check((cc.get("b_only_correct"), cc.get("a_only_correct")) == (1, 0),
              "the discordant counts ride with the paired difference")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------ imagegen ---
def test_imagegen():
    d = tempfile.mkdtemp(prefix="yamadori_test_dr_")
    try:
        p = os.path.join(d, "results.jsonl")
        check(dr.imagegen_summary(p)["state"] == "empty", "no file: empty")
        ok = {"config": "c", "size": "1024x1024", "steps": 20, "rc": 0, "image": "x.png",
              "wall_s": 110.0, "sample_s": 100.0, "sec_per_step": 5.0, "decode_s": 7.0,
              "delta_peak_mib": 6300, "min_free_mib": 4900, "peak_temp_c": 80}
        _jl(p, [ok, dict(ok, wall_s=120.0, errors=["vae retile"]),
                dict(ok, killed=True, rc=1, image=None, delta_peak_mib=9300, min_free_mib=190,
                     kill_why="free 190 MiB"),
                dict(ok, skipped=True, why="CUDA1 free 5082 MiB < need 6000"),
                {"probe": "idle_after_generation", "base_used_mib": 4931,
                 "idle_after_mib": [5250, 5250, 5260]},
                {"config": "sd-server c", "images": [{"ok": True, "seconds_client": 104.0},
                                                     {"ok": True, "seconds_client": 106.0}]}])
        s = dr.imagegen_summary(p)
        c = s["configs"][0]
        check(c["images"] == 2 and c["status"] == {"ok": 1, "errors_logged": 1, "killed": 1, "skipped": 1},
              "status per row: ok, errors logged, killed, skipped", json.dumps(c["status"]))
        check(c["median_wall_s"] == 115.0, "median seconds over images made")
        check(c["max_delta_peak_mib"] == 6300 and c["min_free_mib"] == 4900,
              "VRAM is read off images made, not the killed run")
        check(c["kills"] == [{"why": "free 190 MiB", "min_free_mib": 190, "delta_peak_mib": 9300}],
              "the kill is listed with its own reading", json.dumps(c["kills"]))
        check(s["server"][0]["images"] == 2 and s["server"][0]["median_seconds"] == 105.0,
              "the served path's images are counted apart")
        check(s["idle_probes"][0]["idle_after_mib"] == 5250, "idle probe median")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------- speed ---
def test_speed():
    d = tempfile.mkdtemp(prefix="yamadori_test_dr_")
    try:
        doc = os.path.join(d, "MTP.md")
        _w(doc, "## 6. Speed on the A4000\n\n| run | binary / file | head | BI | TS | bash | prose | mean | "
                "acceptance TS / bash / prose |\n|---|---|---|---|---:|---:|---:|---:|---|\n"
                "| E | **production** / Abliterated | – | – | 32.36 | 32.37 | 32.40 | **32.38** | – |\n"
                "| H | new / graft | on | 1 | 57.03 | 55.88 | 51.78 | **54.89** | 0.837 / 0.793 / 0.669 |\n\n"
                "## 7. Draft acceptance\n\n### Lean file (`x.gguf`)\n\n| content | n prompts | acceptance | "
                "per prompt | tok/s head on | tok/s head off | speedup |\n|---|---:|---:|---|---:|---:|---:|\n"
                "| TypeScript | 3 | **0.839** (655/781) | 0.84 | 60.0 | 43.5 | 1.38x |\n"
                "| image | 3 | **0.726** (644/887) | 0.72 | 54.1 | not run | – |\n")
        m = dr.mtp_tables(doc)
        b = m["builds"]["rows"]
        check(m["section6"] == "Speed on the A4000", "the card is carried from the heading")
        check([(r["run"], r["mean_tps"]) for r in b] == [("E", 32.38), ("H", 54.89)],
              "build rows: run and mean tok/s", json.dumps(b)[:200])
        check(b[0]["build"] == "production / Abliterated", "bold is stripped from the build name")
        a = m["acceptance"][0]["rows"]
        check((a[0]["acceptance"], a[0]["accepted"], a[0]["drafted"]) == (0.839, 655, 781),
              "acceptance keeps its accepted/drafted counts")
        check(a[1]["tps_off"] is None, "'not run' reads as None, never 0")
        check(dr.mtp_tables(os.path.join(d, "no.md"))["builds"] is None, "a missing doc: no table")
        lc = dr.longctx_summary(os.path.join(d, "longctx"))
        check(lc["state"] == "empty" and "run.py" in lc["how"], "no long-context run: empty with how")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# -------------------------------------------------------------- domain ---
def _drow(task, arm, outcome, domain="typescript", contaminated=False, stage="test", **kw):
    return dict({"task": task, "arm": arm, "outcome": outcome, "domain": domain,
                 "contaminated": contaminated, "grade": {"stage": stage},
                 "usage": {"prompt_tokens": 100, "completion_tokens": 50}, "seconds": 10.0,
                 "x_yamadori": {"hops": 1}}, **kw)


def test_domain():
    d = tempfile.mkdtemp(prefix="yamadori_test_dr_")
    try:
        man = {"group": "g1", "effort": "medium", "runs": [{"arms": ["A0", "A6"]}]}
        for sub, rows in (("g1a", [_drow("t1", "A0", "pass"), _drow("t1", "A6", "fail", stage="compile"),
                                   _drow("t9", "A0", "pass", domain="three_tsl", contaminated=True),
                                   _drow("t9", "A6", "pass", domain="three_tsl", contaminated=True)]),
                          ("g1b", [_drow("t2", "A0", "fail"), _drow("t2", "A6", "pass"),
                                   _drow("t3", "A0", "pass", stale_code={"reason": "pre-fix fan-out"}),
                                   _drow("t3", "A0", "pass")])):
            _jl(os.path.join(d, sub, "rows.jsonl"), rows)
            _w(os.path.join(d, sub, "manifest.json"), json.dumps(man))
        _jl(os.path.join(d, "other", "rows.jsonl"), [_drow("t1", "A0", "pass")])
        s = dr.domain_summary(d, "g1")
        check(s["run_id"] == "g1" and sorted(s["merged_run_dirs"]) == ["g1a", "g1b"],
              "the named group is chosen and merged, not the newest dir",
              str((s.get("run_id"), s.get("merged_run_dirs"))))
        check(s["stale_rows"] == 1 and s["stale"] == [{"reason": "pre-fix fan-out", "rows": 1,
                                                       "arms": {"A0": 1}}],
              "a stale row is excluded and its reason reported", json.dumps(s.get("stale")))
        bd = s["by_domain"]
        check(bd.get("three_tsl", {}).get("contaminated") is True
              and bd.get("typescript", {}).get("contaminated") is False,
              "the contaminated domain is flagged, not dropped", str(list(bd)))
        check(s["excluded_contaminated"]["domains"] == ["three_tsl"],
              "the lcb-shaped headline still excludes it")
        check(not _walk({k: s[k] for k in ("pairs", "headline", "contaminated_block")}, "verdict"),
              "no comparison carries a verdict")
        leg = {x["arm"]: x for x in s["legend"]}
        check(leg.get("A6", {}).get("deep_thinking") == "on" and leg.get("A0", {}).get("fanout") == "1",
              "the legend comes from bench/domain/run.py ARMS", json.dumps(s["legend"])[:300])
        check("A0" in s["triggers"], "mechanism triggers ride with the section")
        e = dr.domain_summary(os.path.join(d, "none"), "g1")
        check(e["state"] == "empty", "no run: empty")

        # A newer group that has only planned (manifest, no rows yet) is the
        # current one, and says so; the older group is listed with its
        # exclusions.
        _w(os.path.join(d, "g2a", "manifest.json"), json.dumps(
            {"group": "g2", "temperature": 1.0, "server_sampling": {"min_p": "0.0"},
             "runs": [{"arms": ["A0", "S0"], "tasks": ["t1", "t2", "t3"]}]}))
        r = dr.domain_summary(d, None)
        check(r["state"] == "running" and r["group"] == "g2",
              "unpinned: the newest group, running when it holds no rows", str((r.get("state"), r.get("group"))))
        check(r["current"]["planned_pairs"] == 6 and r["current"]["scored_pairs"] == 0,
              "planned pairs from the manifest, none scored", json.dumps(r.get("current"))[:200])
        gs = {g["group"]: g for g in r["groups"]}
        check(gs.get("g1", {}).get("stale") == [{"reason": "pre-fix fan-out", "rows": 1, "arms": {"A0": 1}}]
              and gs["g1"]["scored_pairs"] == 7,
              "every group is listed with its stale exclusions and scored pairs", json.dumps(gs.get("g1"))[:300])
        check(gs.get("g2", {}).get("conditions", {}).get("temperature") == 1.0,
              "a group's sampling conditions come from its manifest")
        check([x["arm"] for x in r["legend"]] == ["A0", "S0"], "a running group's legend is its planned arms")

        # every row stale: nothing to headline, the reason is shown
        _jl(os.path.join(d, "g3a", "rows.jsonl"),
            [_drow("t1", "A0", "pass", stale_code={"reason": "temperature 0.0"})])
        _w(os.path.join(d, "g3a", "manifest.json"), json.dumps({"group": "g3", "runs": [{"arms": ["A0"]}]}))
        r = dr.domain_summary(d, "g3")
        check(r["state"] == "running" and r["current"]["stale"][0]["reason"] == "temperature 0.0",
              "a group whose every row is stale has nothing to score and shows why",
              json.dumps(r.get("current"))[:200])
    finally:
        shutil.rmtree(d, ignore_errors=True)


# -------------------------------------------------------------- whole ---
def test_results_isolation_and_card():
    card = dr.model_card_summary()
    check(card["measured_here"] is False and card["publisher"] == "PrismML"
          and card["source"].startswith("https://huggingface.co/prism-ml/"),
          "the model card is labelled published by PrismML, not measured here")
    check(all(len(r["values"]) == len(card["columns"]) for r in card["rows"]),
          "every card row has a value per column")

    def boom():
        raise RuntimeError("fixture")
    saved = dr.SECTIONS
    try:
        dr.SECTIONS = (("bad", boom), ("model_card", dr.model_card_summary))
        r = dr.results()
        b = r["sections"]["bad"]
        check(b["state"] == "error" and "RuntimeError: fixture" in b["error"] and "boom" in b["component"],
              "a raising section is an error naming its function", json.dumps(b))
        check(r["sections"]["model_card"]["state"] == "ready", "the other sections still render")
    finally:
        dr.SECTIONS = saved
    names = [n for n, _ in dr.SECTIONS]
    check(all(n in names for n in ("livebench", "swebench", "speed", "imagegen", "model_card", "domain")),
          "results() carries every benchmark section", str(names))


def main() -> int:
    for fn in (test_markdown_tables, test_swebench, test_livebench, test_imagegen,
               test_speed, test_domain, test_results_isolation_and_card):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
