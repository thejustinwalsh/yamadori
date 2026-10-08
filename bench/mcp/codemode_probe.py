#!/usr/bin/env python
"""THE TOOLBOX PROBE: does the model find and use the server toolbox (`yama_compose_tools`, mcp/codemode.py,
docs/CODEMODE.md), and is it better than today's individual tools?

    python bench/mcp/codemode_probe.py                  # print the plan; runs nothing
    python bench/mcp/codemode_probe.py --run --harness pi|hermes --trials N --steps S --max-minutes M
                       [--arms off,on,only] [--tag T] [--key-file F]
    python bench/mcp/codemode_probe.py --report [--tag-filter T]

Operator, 2026-10-07 (verbatim): "pi now has codemode ... we should make codemode in our server. QuickJS sandbox is a
good idea, I love this." THE TOOL RECIPE rule 7: the toolbox (switch `codemode`, tiers.OFF_BY_DEFAULT) ships off and is
turned on only after this probe passes. THIS SCRIPT RUNS NOTHING UNLESS --run IS GIVEN; the coordinator runs it with
the operator's go and an idle GPU (AGENTS.md "One GPU consumer at a time"). Nothing here has been run.

THE THREE ARMS (the coordinator's, 2026-10-07; X-Yamadori-Features, tier medium; the package skills channel stays at
its default, as in production):
  off    today's tools: {"mcp_tools": true, "skills": false, "codemode": false}
  on     the toolbox ADDED beside the individual lookups: + {"codemode": true, "codemode_mode": "on"}
  only   the toolbox INSTEAD of the MCP lookups:          + {"codemode": true, "codemode_mode": "only"}
Arms are INTERLEAVED (trial 1 of every arm, then trial 2 ...). Sampling is unseeded: n beside every number.

ONE TRIAL = the pagoda prompt's FIRST --steps REQUESTS (byte for byte, bench/mcp/lookup_probe.py's):
  --harness pi      Pi 0.87.1 in the harness box, `--thinking medium`; the header is Pi's provider header
  --harness hermes  Hermes with the octopus profile (Responses wire), `--max-turns` S; the recording relay ADDS the
                    header (bench/octopus/relay.py --inject-header; Hermes has no setting for it)
every request through the recording relay to :1234, so x_yamadori of each request is on record.

RECORDED per trial (bench/mcp/results/codemode_probe.jsonl), from x_yamadori of each of the first S requests:
  prompt tokens our additions cost   codemode.ours {tools, tool_tokens, system_tokens, tokens} of the FIRST request
                                     (the proxy's estimate; recorded with the toolbox on or off)
  hops                               x_yamadori.hops per request, summed; tool_turns.turns, and whether the loop hit the
                                     cap (tool_turns.hit) on any request
  context added by our tool results  x_yamadori.tools[].chars summed (each hidden hop's result, the toolbox's included)
  did the model call the toolbox     codemode.runs; the first step that did; what it wrote (codemode.heads: the first
                                     120 characters of each script; the full text is in the corpus tool_call rows)
  script failure rate                codemode.failed / reasons per run (js_error, syntax_error, timeout, ...)
  inner calls                        codemode.calls[{tool, ms, ok}] vs the direct package calls (mcp.calls less the
                                     inner ones)
  DISCOVERY                          codemode.searches / describes and codemode.discovery[]: each tools.search (the query,
                                     the functions it returned) and tools.describe (the name, whether it existed);
                                     `discovery_hit`: a function the model CALLED (inner or direct) in a request was
                                     among the hits of a search in that request; `describe_misses`, `empty_searches`;
                                     `first_call_tool` (the pagoda's vague names are meant to start at yama_find_package)
  the pitfall rules on written code  bench/skills/pitfall_rules/* (package_skills_probe.pattern_use) on the files the
                                     model wrote, over the skills DELIVERED (package sections) or PULLED (craft reads,
                                     inside a script too)
A trial where the header did not reach the proxy, the arm's switch or mode is not what the arm says, or the toolbox was
not on main in a toolbox arm is `invalid` with why.

REPORT: per harness and arm: n valid trials, mean prompt tokens our additions cost, mean hops, trials that hit the
cap, mean tool-result characters, the toolbox call rate, script failure rate, discovery rates (searched, described,
discovery_hit of the trials that searched), direct lookups per trial, the pitfall outcomes.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))

import lookup_probe as LP  # noqa: E402
import package_skills_probe as PSP  # noqa: E402

RESULTS = os.path.join(HERE, "results", "codemode_probe.jsonl")
ARMS = ("off", "on", "only")
HARNESSES = ("pi", "hermes")
COMPOSE = "yama_compose_tools"
CRAFT_TOOL = "yama_recall_craft"
S_PER_REQUEST = PSP.S_PER_REQUEST
S_FIXED = PSP.S_FIXED


def arm_features(arm: str) -> dict:
    f = dict(LP.FEATURES)
    f["codemode"] = arm != "off"             # off is FORCED off (the default, but the arm says so)
    if arm != "off":
        f["codemode_mode"] = arm
    return f


def _x(row: dict) -> dict:
    return (row.get("response") or {}).get("x_yamadori") or {}


def summarise_requests(rows: list[dict], arm: str, steps: int, project: str, harness: str, stop: str | None) -> dict:
    rows = rows[:steps]
    first = _x(rows[0]) if rows else {}
    cm0 = first.get("codemode") or {}
    hops = turns = tool_chars = 0
    cap_hit = False
    runs, failed, reasons, searches, describes = 0, 0, [], 0, 0
    heads, inner, direct, discovery_events = [], [], 0, []
    first_compose_step = None
    discovery_hit = False
    describe_misses = empty_searches = 0
    delivered: set = set()
    pulled: set = set()
    sections: list = []
    first_call_tool = None
    for i, r in enumerate(rows, 1):
        x = _x(r)
        hops += int(x.get("hops") or 0)
        tt = x.get("tool_turns") or {}
        turns += int(tt.get("turns") or 0)
        cap_hit = cap_hit or bool(tt.get("hit"))
        tool_chars += sum(int(t.get("chars") or 0) for t in x.get("tools") or [])
        cm = x.get("codemode") or {}
        if cm.get("runs"):
            runs += int(cm["runs"])
            first_compose_step = first_compose_step or i
        failed += sum(1 for f_ in [cm.get("failed")] if f_)
        reasons += list(cm.get("reasons") or [])
        searches += int(cm.get("searches") or 0)
        describes += int(cm.get("describes") or 0)
        heads += list(cm.get("heads") or [])
        calls_here = list(cm.get("calls") or [])
        inner += [{"step": i, "tool": c.get("tool"), "ok": c.get("ok"), "ms": c.get("ms")} for c in calls_here]
        mcp_calls = (x.get("mcp") or {}).get("calls") or []
        direct += max(len(mcp_calls) - sum(1 for c in calls_here if c.get("tool") != CRAFT_TOOL
                                           and str(c.get("tool", "")).startswith("yama_")
                                           and c.get("tool") in {m.get("tool") for m in mcp_calls}), 0)
        if first_call_tool is None:
            seq = [c.get("tool") for c in calls_here] or [c.get("tool") for c in mcp_calls]
            first_call_tool = seq[0] if seq else None
        hits_here: set = set()
        for d in cm.get("discovery") or []:
            discovery_events.append({"step": i, **d})
            if d.get("kind") == "search":
                hits_here |= set(d.get("hits") or [])
                empty_searches += int(not d.get("hits"))
            elif d.get("kind") == "describe":
                describe_misses += int(d.get("ok") is False)
        called_here = {c.get("tool") for c in calls_here} | {c.get("tool") for c in mcp_calls}
        if hits_here and hits_here & called_here:
            discovery_hit = True
        for p in (x.get("skills") or {}).get("package") or []:
            if p.get("tool") or p.get("trigger"):
                sections.append({"step": i, **{k: p.get(k) for k in ("tool", "package", "major", "mode", "skills",
                                                                     "chars", "trigger") if k in p}})
        for rd in (x.get("craft") or {}).get("reads") or []:
            if rd.get("name"):
                pulled.add(rd["name"])
    delivered = {s["name"] for sec in sections for s in sec.get("skills") or []}
    invalid = []
    if not rows:
        invalid.append("no request reached the relay")
    else:
        sw = cm0.get("switch") or {}
        if first.get("tier") != LP.EFFORT:
            invalid.append(f"tier {first.get('tier')}, not {LP.EFFORT}")
        if arm == "off":
            if sw.get("on") or cm0.get("offered"):
                invalid.append(f"the off arm ran with the toolbox on ({sw})")
        else:
            if sw.get("source") != "header" or not sw.get("on") or sw.get("mode") != arm:
                invalid.append(f"the arm says {arm}, the proxy ran {sw or 'no codemode record'}")
            if not cm0.get("offered"):
                invalid.append(f"the toolbox was not offered ({cm0.get('why')})")
        mcp0 = first.get("mcp") or {}
        if not set(LP.TOOLS) <= set(mcp0.get("offered") or []):
            invalid.append(f"the MCP tools were not all offered ({mcp0.get('offered')})")
    pattern = PSP.pattern_use(project, delivered, pulled) if rows else {}
    return {
        "arm": arm, "harness": harness, "steps": steps, "requests": len(rows), "stop": stop,
        "offer": {k: cm0.get(k) for k in ("switch", "offered", "mode", "kept", "why", "declared", "left_out",
                                          "budget_tokens", "tokens", "hidden")},
        "ours": cm0.get("ours"),
        "ours_tokens": (cm0.get("ours") or {}).get("tokens"),
        "hops": hops, "tool_turns": turns, "cap_hit": cap_hit, "tool_result_chars": tool_chars,
        "compose_called": runs > 0, "compose_runs": runs, "first_compose_step": first_compose_step,
        "scripts": heads, "script_failures": sum(1 for r in reasons if r not in ("completed", "exit")),
        "script_reasons": reasons, "inner_calls": inner, "direct_package_calls": direct,
        "searches": searches, "describes": describes, "discovery": discovery_events,
        "searched": searches > 0, "described": describes > 0, "discovery_hit": discovery_hit,
        "describe_misses": describe_misses, "empty_searches": empty_searches, "first_call_tool": first_call_tool,
        "sections": sections, "crafts_pulled": sorted(pulled), "crafts_delivered": sorted(delivered),
        "pattern": pattern, "invalid": invalid or None}


# --------------------------------------------------------------------- Pi ----
def run_pi_trial(arm: str, n: int, a) -> dict:
    saved = (LP.FEATURES, LP.Watch, LP.summarise)
    LP.FEATURES = arm_features(arm)
    LP.Watch = lambda ids, _mr, mm: PSP.StepWatch(ids, a.steps, mm)
    LP.summarise = lambda ids, watch: summarise_requests(
        LP._gens(LP._rows(ids["relay_out"])), arm, a.steps, ids["run_dir"], "pi", watch.stop_reason)
    try:
        ns = argparse.Namespace(**{**vars(a), "tag": f"{a.tag}-{arm}", "max_requests": a.steps})
        row = LP.run_trial(n, ns)
    finally:
        LP.FEATURES, LP.Watch, LP.summarise = saved
    row["arm"], row["harness"], row["steps_bound"] = arm, "pi", a.steps
    return row


# ------------------------------------------------------------------ Hermes ---
def run_hermes_trial(arm: str, n: int, a) -> dict:
    """Hermes (octopus profile, Responses wire): the recording relay adds the features header."""
    R, pagoda = LP._mods()
    path, sha = pagoda.write_prompt()
    rid = pagoda.run_id(f"{a.tag}-{arm}", "medium", n)
    feats = json.dumps(arm_features(arm))
    real = R.relay_cmd
    R.relay_cmd = lambda out_path: real(out_path) + ["--inject-header", f"{LP.HEADER}={feats}"]
    meta = {"task": pagoda.TASK, "variant": pagoda.TASK, "harness": "hermes", "arm": "medium", "rep": n,
            "tag": a.tag, "features": arm_features(arm), "prompt_sha256": sha, "protocol": "single-shot"}
    try:
        row = R.run_one(rid, path, "medium", a.steps, int(a.max_minutes * 60), meta)
    finally:
        R.relay_cmd = real
    if row.get("outcome") == "not_run":
        return dict(row, arm=arm, harness="hermes")
    relay_out = os.path.join(R.LOGS_DIR, rid, "relay.jsonl")
    s = summarise_requests(LP._gens(LP._rows(relay_out)), arm, a.steps, os.path.join(R.RUNS_DIR, rid), "hermes",
                           "exited")
    return dict(row, arm=arm, harness="hermes", steps_bound=a.steps, tag=a.tag, trial=n, features=arm_features(arm),
                outcome="invalid" if s["invalid"] else "ok", **{k: v for k, v in s.items() if k != "arm"})


def run_trial(arm: str, n: int, a) -> dict:
    return run_pi_trial(arm, n, a) if a.harness == "pi" else run_hermes_trial(arm, n, a)


# ------------------------------------------------------------------ report ---
def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def report(rows: list[dict]) -> dict:
    out: dict = {}
    for h in HARNESSES:
        for arm in ARMS:
            r = [x for x in rows if x.get("arm") == arm and x.get("harness", "pi") == h
                 and x.get("outcome") == "ok"]
            n = len(r)
            if not n:
                continue
            runs = sum(x.get("compose_runs") or 0 for x in r)
            searched = [x for x in r if x.get("searched")]
            pat = collections.Counter()
            for x in r:
                for c in (x.get("pattern") or {}).get("cases") or []:
                    if c.get("applies"):
                        pat["applies"] += 1
                        pat["pitfall_avoided"] += int(not c["pitfall"])
                        pat["good_used"] += int(c["good"])
            out[f"{h}/{arm}"] = {
                "n_valid_trials": n,
                "mean_prompt_tokens_our_additions_cost": _mean([x.get("ours_tokens") for x in r]),
                "mean_hops": _mean([x.get("hops") for x in r]),
                "trials_that_hit_the_tool_turn_cap": sum(bool(x.get("cap_hit")) for x in r),
                "mean_tool_result_chars": _mean([x.get("tool_result_chars") for x in r]),
                "toolbox_call_rate": round(sum(bool(x.get("compose_called")) for x in r) / n, 3),
                "scripts": runs,
                "script_failure_rate": round(sum(x.get("script_failures") or 0 for x in r) / runs, 3) if runs else None,
                "searched_rate": round(len(searched) / n, 3),
                "described_rate": round(sum(bool(x.get("described")) for x in r) / n, 3),
                "discovery_hit_rate_of_trials_that_searched": round(
                    sum(bool(x.get("discovery_hit")) for x in searched) / len(searched), 3) if searched else None,
                "describe_misses": sum(x.get("describe_misses") or 0 for x in r),
                "empty_searches": sum(x.get("empty_searches") or 0 for x in r),
                "mean_direct_package_calls": _mean([x.get("direct_package_calls") for x in r]),
                "first_call_tools": dict(collections.Counter(x.get("first_call_tool") for x in r).most_common(5)),
                "pitfall_rules": dict(pat)}
    return out


# -------------------------------------------------------------------- plan ---
def plan(a) -> str:
    arms = [x for x in (a.arms or ",".join(ARMS)).split(",") if x in ARMS]
    trials = a.trials if a.trials is not None else "<--trials>"
    steps = a.steps if a.steps is not None else "<--steps>"
    est = ""
    if a.trials is not None and a.steps is not None:
        per = a.steps * S_PER_REQUEST + S_FIXED
        est = (f"\nGPU time ESTIMATE per harness: {a.trials} trials x {len(arms)} arms, each ~{per} s "
               f"({a.steps} requests x {S_PER_REQUEST} s + {S_FIXED} s; package_skills_probe's figures) = "
               f"~{a.trials * len(arms) * per / 3600:.1f} h of an exclusive GPU. A toolbox call adds no generation; "
               f"its lookups and any jjava reads are the server's.")
    return "\n".join([
        "THE TOOLBOX PROBE -- the plan (nothing was run, written or started)",
        "",
        f"arms ({len(arms)}): " + ", ".join(f"{x} {json.dumps(arm_features(x))}" for x in arms),
        f"harnesses: {', '.join(HARNESSES)} (one --harness per invocation); trials per arm: {trials}; steps "
        f"(requests) per trial: {steps}; interleaved",
        "per trial: the pagoda prompt, the recording relay, stopped after --steps requests; Pi through its provider "
        "header, Hermes through the relay's --inject-header",
        f"one row per trial appended to {RESULTS}",
        est,
        "",
        "BEFORE RUNNING (the coordinator):",
        "  - the proxy restarted with this code (mcp/codemode.py, mcp/codemode_sandbox.py, wasmtime 49.0.0 in the "
        "stack interpreter and bin/qjs-wasi.wasm: `python mcp/codemode_sandbox.py fetch`) and the MCP host up (GET /dash/api/mcp: packagelens "
        "`ready`); an idle GPU and the operator's go",
        f"  - then: python bench/mcp/codemode_probe.py --run --harness pi --trials {trials} --steps {steps} "
        f"--max-minutes <M> --tag {a.tag}",
        f"  - and:  python bench/mcp/codemode_probe.py --run --harness hermes --trials {trials} --steps {steps} "
        f"--max-minutes <M> --tag {a.tag}",
        "  - then: python bench/mcp/codemode_probe.py --report [--tag-filter T]"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", help="run the trials (default: print the plan)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--harness", choices=HARNESSES, default="pi")
    ap.add_argument("--trials", type=int, help="trials per arm")
    ap.add_argument("--steps", type=int, help="requests per trial")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tag", default="codemode")
    ap.add_argument("--tag-filter", help="with --report: only the rows of this tag")
    ap.add_argument("--key-file")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if a.report:
        rows = LP._rows(RESULTS)
        if a.tag_filter:
            rows = [r for r in rows if str(r.get("tag") or "").startswith(a.tag_filter)]
        print(json.dumps(report(rows), indent=1))
        return 0
    if not a.run:
        print(plan(a))
        return 0
    missing = [f"--{k.replace('_', '-')}" for k in ("trials", "steps", "max_minutes") if getattr(a, k) is None]
    if missing:
        print(f"--run needs {', '.join(missing)} (no defaults: the bounds are the coordinator's)", file=sys.stderr)
        return 2
    arms = [x for x in a.arms.split(",") if x in ARMS]
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    for n in range(1, a.trials + 1):
        for arm in arms:
            row = run_trial(arm, n, a)
            row.setdefault("t_logged", time.time())
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row.get(k) for k in (
                "run_id", "arm", "harness", "outcome", "why", "stop", "requests", "ours_tokens", "hops", "cap_hit",
                "compose_called", "searches", "describes", "discovery_hit", "script_failures", "invalid")},
                indent=1), flush=True)
            if row.get("outcome") == "not_run":
                return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
