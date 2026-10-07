#!/usr/bin/env python
"""The package skills probe's summary (bench/mcp/package_skills_probe.py),
offline: a synthetic relay file and a synthetic project, no Pi, no proxy.

    python bench/skills/test_package_skills_probe.py   -> "N/M checks passed"

GATES: the plan runs nothing and states the arms, the n and the GPU
estimate; the arms differ only in the package_skills switch and mode; a
trial's per-request records (package tool calls, sections, triggers, craft
calls by name and by question, recall after a section) are read from
x_yamadori; an arm whose proxy did not run the arm's mode is invalid; the
pitfall rules read the files the model wrote (applies / pitfall / good) for
the cases whose skill was delivered or pulled; the report counts per arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_pkgskills_probe_")
sys.path.insert(0, os.path.join(ROOT, "bench", "mcp"))
sys.path.insert(0, HERE)

import package_skills_probe as PP  # noqa: E402

CHECKS: list = []


def check(name, ok, detail=None):
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


def row(x: dict) -> dict:
    return {"method": "POST", "path": "/v1/chat/completions",
            "response": {"x_yamadori": x}}


def main() -> int:
    args = argparse.Namespace(arms=None, trials=5, steps=15, tag="t",
                              max_minutes=None, key_file=None)
    text = PP.plan(args)
    check("the plan names every arm, the n, the steps and the GPU estimate",
          all(f'"package_skills_mode": "{m}"' in text
              for m in ("inject", "router", "both"))
          and "20 trials" in text and "h of an exclusive GPU" in text
          and "nothing was run" in text, text[:300])
    f = {a: PP.arm_features(a) for a in PP.ARMS}
    base = dict(f["control"])
    check("the arms differ only in the package_skills switch and mode",
          "package_skills" not in f["control"]
          and all({k: v for k, v in f[a].items() if not k.startswith(
              "package_skills")} == base for a in ("inject", "router", "both"))
          and [f[a]["package_skills_mode"] for a in ("inject", "router",
                                                      "both")]
          == ["inject", "router", "both"])
    # a synthetic trial
    log = os.path.join(_TMP, "relay.jsonl")
    proj = os.path.join(_TMP, "project")
    os.makedirs(os.path.join(proj, "src"))
    os.makedirs(os.path.join(proj, "node_modules", "x"))
    with open(os.path.join(proj, "src", "inv.ts"), "w") as fh:
        fh.write("import type { Entity } from 'koota';\n"
                 "import { Inventory } from './traits';\n"
                 "export function pickUp(player: Entity, item: any) {\n"
                 "  const inventory = player.get(Inventory)!;\n"
                 "  inventory.items.push(item);\n"
                 "  player.changed(Inventory);\n}\n")
    with open(os.path.join(proj, "node_modules", "x", "i.ts"), "w") as fh:
        fh.write("export const bad = 1;\n")
    mcp_on = {"switch": {"on": True, "source": "header"}, "offered": list(
        PP.LP.TOOLS), "on_main": list(PP.LP.TOOLS)}
    sw = {"on": True, "source": "header", "mode": "router",
          "mode_source": "header"}
    rows = [
        row({"tier": "medium", "mcp": dict(mcp_on, calls=[]),
             "skills": {"package_switch": sw, "package": []},
             "craft": {"tool": True, "reads": []}}),
        row({"tier": "medium", "mcp": dict(mcp_on, calls=[{
            "tool": "yama_find_package", "args": {"query": "koota"},
            "names": ["koota"], "found": [{"name": "koota",
                                           "version": "0.6.6"}],
            "ok": True}]),
            "skills": {"package_switch": sw, "package": [{
                "tool": "yama_find_package", "package": "koota", "major": 0,
                "mode": "router", "chars": 400, "why": "first appearance",
                "skills": [{"id": "a", "name": "koota-traits-and-entities",
                            "form": "router"},
                           {"id": "b", "name": "koota-react-integration",
                            "form": "router"}]}]},
            "craft": {"tool": True, "reads": []}}),
        row({"tier": "medium", "mcp": dict(mcp_on, calls=[]),
             "skills": {"package_switch": sw, "package": [{
                 "event": "recall_after_section",
                 "craft": "koota-traits-and-entities", "found": True,
                 "listed_by_router": True, "mode": "router",
                 "package": "koota", "steps_later": 1, "question": True},
                 {"trigger": "install", "package": "koota",
                  "evidence": "install line: koota", "skills": []}]},
             "craft": {"tool": True, "reads": [{
                 "how": "question", "name": "koota-traits-and-entities",
                 "found": "a", "query": {
                     "question": "how do I signal an array change in koota?",
                     "answered": True, "p": 0.7, "gate_p": 0.9,
                     "untuned": True, "shortlist": ["a", "b"]}}],
                 "query": [{}]}}),
    ]
    with open(log, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ids = {"relay_out": log, "run_dir": proj}

    class W:
        stop_reason = "max_requests"
    s = PP.summarise_steps(ids, W(), "router", 15)
    check("the trial's records are read per request: the package call, the "
          "section with its skills and forms, the trigger, the craft call by "
          "question (verbatim), the recall after the section",
          s["package_tool_called"] and s["first_package_call_step"] == 2
          and s["section_delivered"]
          and s["sections"][0]["skills"][0]["form"] == "router"
          and s["triggers"][0]["trigger"] == "install"
          and s["craft_calls"][0]["how"] == "question"
          and s["craft_calls"][0]["question"].startswith("how do I signal")
          and s["craft_calls"][0]["gate_p"] == 0.9
          and s["recall_after_section"][0]["steps_later"] == 1
          and s["recall_called"] and s["recall_after_section_called"]
          and s["question_calls"] == 1
          and s["crafts_pulled"] == ["koota-traits-and-entities"]
          and s["router_listed_pulled"] == ["koota-traits-and-entities"]
          and s["invalid"] is None, json.dumps(s)[:700])
    pat = PP.pattern_use(
        proj, {"koota-traits-and-entities"}, {"koota-traits-and-entities"})
    mine = [c for c in pat["cases"] if c["case"]
            == "koota-inventory-change-flag"]
    check("the pitfall rules read the files the model wrote (node_modules "
          "left out): the koota case applies, the pitfall is avoided and the "
          "good pattern is used",
          mine and mine[0]["applies"] and not mine[0]["pitfall"]
          and mine[0]["good"] and mine[0]["via"] == "pulled"
          and "bad = 1" not in PP.project_answer(proj), json.dumps(pat)[:400])
    s2 = PP.summarise_steps(ids, W(), "inject", 15)
    check("an arm the proxy did not run (the arm says inject, the proxy ran "
          "router) is invalid, with why",
          s2["invalid"] and "ran" in s2["invalid"][0], s2["invalid"])
    ctrl = PP.summarise_steps(ids, W(), "control", 15)
    check("a control arm with the channel on is invalid",
          ctrl["invalid"] and "control arm" in ctrl["invalid"][0],
          ctrl["invalid"])
    rep = PP.report([dict(s, outcome="ok", arm="router"),
                     dict(s, outcome="invalid", arm="router")])
    check("the report counts valid trials per arm: rates, questions, crafts "
          "pulled, the rule outcomes where a rule applied",
          rep["router"]["n_valid_trials"] == 1
          and rep["router"]["package_tool_call_rate"] == 1.0
          and rep["router"]["question_calls_total"] == 1
          and rep["router"]["crafts_pulled"] == {
              "koota-traits-and-entities": 1}
          and rep["router"]["pitfall_rules"].get("applies", 0) >= 1
          and rep["router"][
              "recall_after_section_rate_of_trials_with_a_section"] == 1.0,
          json.dumps(rep)[:400])
    ok = sum(1 for _n, v in CHECKS if v)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
