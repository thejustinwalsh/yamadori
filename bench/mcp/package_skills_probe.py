#!/usr/bin/env python
"""THE PACKAGE SKILLS PROBE: do the skills that ride in our package tools'
results change what the model does in its first steps, and which of the two
delivery designs does it better?

    python bench/mcp/package_skills_probe.py                  # print the plan; runs nothing
    python bench/mcp/package_skills_probe.py --run --trials N --steps S --max-minutes M
                       [--arms control,inject,router,both] [--tag T] [--key-file F]

Operator, 2026-10-06 (verbatim): "If the model uses some of our other tools
or mcps this may be a good point to skill up", and: "research showed
injection did work better than discovery, but I don't know that research
measured discovery in the way I am proposing it, if a model was told the tool
to call and given a few router like decisions maybe it would be inclined to
call the craft tool for more of the skill". And "a natural language query the
model asks that is weighed against the skills". THE TOOL RECIPE rule 7: the
channel (mcp/package_skills.py, switch `package_skills`) ships off and is
turned on only after this probe passes. THIS IS A COMPARISON OF TWO DELIVERY
DESIGNS the operator asked to test -- not an on/off arm. THIS SCRIPT RUNS
NOTHING UNLESS --run IS GIVEN; the coordinator runs it with the operator's go
and an idle GPU (AGENTS.md "One GPU consumer at a time").

THE ARMS (X-Yamadori-Features on Pi's one provider header; tier medium):
  control   {"mcp_tools": true, "skills": false}: the model plus our MCP tools
            (lookup_probe.py's own configuration, byte for byte)
  inject    + {"package_skills": true, "package_skills_mode": "inject"}:
            the lead skill and strongest items ride in the package tool's result
  router    + mode "router": no skill body; a decision-router table, one row
            per armed, proven craft, each "call yama_recall_craft with ..."
  both      + mode "both": the lead's body and the router table for the rest
yama_recall_craft is on main in the three package-skills arms (it answers a
craft's name or a QUESTION: mcp/craft_query.py, one jjava choice without a
"none" option, then one relevance gate).

ONE TRIAL = lookup_probe.py's: the pagoda prompt byte for byte, Pi 0.87.1 in
the harness box, `--thinking medium`, every request through the recording relay
to :1234 -- but it is NOT stopped at the first install: it runs the model's
FIRST --steps REQUESTS (the coordinator's number; no default), or --max-minutes.
Arms are INTERLEAVED (trial 1 of every arm, then trial 2 ...) so drift in the
box or the model's load spreads over the arms. Sampling is unseeded, as in
lookup_probe.py (Pi sends no seed): the arms are compared at n trials each,
not paired by seed -- say n beside every number.

RECORDED per trial (bench/mcp/results/package_skills_probe.jsonl), per
request of the first S: from x_yamadori of that request --
  package tool called        mcp.calls (tool, args, names, found), and the step
                             of the first one                    [lookup rate]
  which skills rode          skills.package: tool, package, major, mode, each
                             skill and its form (body | recall | router), chars
  triggers                   skills.package entries with `trigger` (install,
                             import, error, new_area) and their evidence
  craft calls                craft.reads / craft.query: by NAME or by QUESTION
                             (the question verbatim, to 200 characters, from the
                             probe's own prompt -- synthetic prompts only), what
                             was returned (craft, answered, jjava's choice p and
                             the relevance gate's p, untuned), the step
  recall after a section     skills.package `recall_after_section`: which craft,
                             whether the router listed it, steps later
and, after the run, from the project the model wrote:
  did the next code use the craft's pattern   the pitfall rules
                             (bench/skills/pitfall_rules/*.py through
                             pitfall_harness.check_answer) of every pitfall case
                             whose skill was DELIVERED or PULLED, on the files
                             Pi wrote: {case, applies (the code uses the case's
                             package), pitfall, good, via}; and, weak (the
                             injector's own H1 rule agreed with the rubric at
                             kappa 0.08, n=470), whether a pulled craft's code
                             spans appear in the code (`span_hits`)
Summary fields: package_tool_called, first_package_call_step,
section_delivered, recall_called, recall_after_section, question_calls,
crafts_pulled, router_listed_pulled, pattern. A trial where the header did not
reach the proxy, the tools were not on main, or the arm's switch/mode is not
what the arm says is `invalid` with why.

REPORT: `--report` reads the results and prints, per arm, n valid trials, the
package-tool call rate, the section-delivered rate, the yama_recall_craft call
rate after a section (the router arms' point), the share of those calls that
were questions, the crafts pulled, and the pitfall-rule outcomes where a rule
applied (pitfall avoided / good used out of applies).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))

import lookup_probe as LP  # noqa: E402

RESULTS = os.path.join(HERE, "results", "package_skills_probe.jsonl")
ARMS = ("control", "inject", "router", "both")
CRAFT_TOOL = "yama_recall_craft"
QUESTION_CHARS = 200
# Files whose code the pitfall rules read (the model's project; node_modules
# and build output left out).
CODE_EXT = {".ts": "ts", ".tsx": "tsx", ".js": "js", ".jsx": "jsx",
            ".mjs": "js", ".json": "json", ".wgsl": "wgsl", ".html": "html"}
SKIP_DIRS = {"node_modules", "dist", "build", ".git", ".vite"}
# Wall seconds per request, for the plan's estimate: the median of
# wall_s / requests over the lookup probe's ok trials with at least 2
# requests (bench/mcp/results/lookup_probe.jsonl, n=5: 20, 30, 40, 69, 105 s).
S_PER_REQUEST = 40
S_FIXED = 30      # box start-up and teardown per trial (an estimate: the 1-request trials took 32 and 139 s)


def arm_features(arm: str) -> dict:
    f = dict(LP.FEATURES)
    # The channel is ON by default since 2026-10-06 (commit 9398750), so every
    # arm FORCES its own setting: the control sends package_skills false.
    f["package_skills"] = arm != "control"
    if arm != "control":
        f["package_skills_mode"] = arm
    return f


class StepWatch(LP.Watch):
    """Stops at the request bound or the time bound -- NOT at the first
    install: this probe runs the model's first --steps requests."""

    def check(self) -> bool:
        if len(LP._gens(LP._rows(self.ids["relay_out"]))) >= self.max_requests:
            self.stop_reason = "max_requests"
            return True
        if time.time() - self.t0 >= self.max_s:
            self.stop_reason = "max_minutes"
            return True
        return False


# ---------------------------------------------------------------- the code --
def project_answer(project: str) -> str:
    """The files the model wrote as one pitfall-harness answer: each file's
    name on the line before its fenced block (pitfall_harness.blocks)."""
    parts = []
    for dp, dn, fn in os.walk(project):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            lang = CODE_EXT.get(os.path.splitext(f)[1].lower())
            if not lang or f in ("package-lock.json",):
                continue
            p = os.path.join(dp, f)
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if len(text) > 200_000:
                continue
            rel = os.path.relpath(p, project).replace("\\", "/")
            parts.append(f"{rel}\n```{lang}\n{text}\n```")
    return "\n\n".join(parts)


def pattern_use(project: str, delivered: set, pulled: set,
                cases: list | None = None, answer: str | None = None) -> dict:
    """Did the model's code use the crafts' patterns? {cases: [{case, skills,
    via, applies, pitfall, good}], span_hits: {craft: [spans found]}}, over
    the pitfall cases whose skills were delivered or pulled. `applies`: the
    code uses one of the case's packages (a rule on code that never touches
    the library says nothing). Errors are recorded, never raised."""
    out: dict = {"cases": [], "span_hits": {}}
    try:
        import pitfall_harness as PH
        code = answer if answer is not None else project_answer(project)
        for case in cases if cases is not None else PH.load_cases():
            names = {PH._skill_ref(x)[0] for x in case.get("skills") or []}
            via = ("pulled" if names & pulled else
                   "delivered" if names & delivered else None)
            if not via:
                continue
            chk = PH.check_answer(case, code)
            pk = case.get("packages") or []
            out["cases"].append({
                "case": case["id"], "skills": sorted(names), "via": via,
                "applies": any(re.search(
                    r"['\"]" + re.escape(p) + r"(?:/[^'\"]*)?['\"]", code)
                    for p in pk) if pk else bool(code.strip()),
                "pitfall": chk["pitfall"], "good": chk["good"]})
    except Exception as e:                                       # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out
    try:
        import skill_inject
        import skills
        by = {s["name"]: s for s in skills.armed()}
        code = answer if answer is not None else project_answer(project)
        for n in sorted(pulled):
            s = by.get(n)
            if s is None:
                continue
            spans = sorted({sp for it in skill_inject.skill_items(s)
                            for sp in it["spans"]})
            hit = [sp for sp in spans if sp in code]
            out["span_hits"][n] = {"spans": len(spans), "found": hit[:12]}
    except Exception as e:                                       # noqa: BLE001
        out["span_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


# --------------------------------------------------------------- the trial ---
def summarise_steps(ids: dict, watch, arm: str, steps: int,
                    answer: str | None = None) -> dict:
    rows = LP._gens(LP._rows(ids["relay_out"]))[:steps]
    first = ((rows[0].get("response") or {}).get("x_yamadori") or {}) \
        if rows else {}
    mcp0 = first.get("mcp") or {}
    pkg_calls, sections, triggers, craft_calls, follow = [], [], [], [], []
    for i, r in enumerate(rows, 1):
        x = (r.get("response") or {}).get("x_yamadori") or {}
        for c in (x.get("mcp") or {}).get("calls") or []:
            pkg_calls.append({"step": i, **{k: c.get(k) for k in (
                "tool", "args", "names", "found", "ok") if k in c}})
        for p in (x.get("skills") or {}).get("package") or []:
            if p.get("event") == "recall_after_section":
                follow.append({"step": i, **{k: p.get(k) for k in (
                    "craft", "found", "listed_by_router", "mode", "package",
                    "steps_later", "question") if k in p}})
            elif p.get("trigger"):
                triggers.append({"step": i, **{k: p.get(k) for k in (
                    "trigger", "package", "major", "evidence", "mode",
                    "skills", "chars", "why") if k in p}})
            elif p.get("tool"):
                sections.append({"step": i, **{k: p.get(k) for k in (
                    "tool", "package", "version", "major", "major_from",
                    "mode", "mode_requested", "skills", "chars", "why")
                    if k in p}})
        cr = x.get("craft") or {}
        for rd in cr.get("reads") or []:
            q = rd.get("query") if isinstance(rd.get("query"), dict) else None
            craft_calls.append({
                "step": i, "how": rd.get("how"), "craft": rd.get("name"),
                "found": bool(rd.get("found")),
                "question": ((q or {}).get("question") or "")[:QUESTION_CHARS]
                if q else None,
                "answered": (q or {}).get("answered") if q else None,
                "choice_p": (q or {}).get("p") if q else None,
                "gate_p": (q or {}).get("gate_p") if q else None,
                "untuned": (q or {}).get("untuned") if q else None,
                "shortlist": len((q or {}).get("shortlist") or []) if q else None})
    delivered = {s["name"] for sec in sections + triggers
                 for s in sec.get("skills") or []}
    pulled = {c["craft"] for c in craft_calls if c.get("craft")}
    listed = {s["name"] for sec in sections for s in sec.get("skills") or []
              if s.get("form") == "router"}
    invalid = []
    if not rows:
        invalid.append("no request reached the relay")
    else:
        if (mcp0.get("switch") or {}).get("source") != "header":
            invalid.append(f"the header did not reach the proxy (switch "
                           f"{mcp0.get('switch')})")
        if not set(LP.TOOLS) <= set(mcp0.get("on_main") or []):
            invalid.append(f"the MCP tools were not all on main "
                           f"({mcp0.get('on_main')})")
        if first.get("tier") != LP.EFFORT:
            invalid.append(f"tier {first.get('tier')}, not {LP.EFFORT}")
        sw = (first.get("skills") or {}).get("package_switch") or {}
        if arm == "control":
            if sw.get("on"):
                invalid.append(f"the control arm ran with the channel on "
                               f"({sw})")
        else:
            if not sw.get("on") or sw.get("mode") != arm:
                invalid.append(f"the arm says {arm}, the proxy ran "
                               f"{sw or 'no package_switch'}")
            if not (first.get("craft") or {}).get("tool"):
                invalid.append("yama_recall_craft was not offered (x_yamadori"
                               ".craft.tool false)")
    proj = pattern_use(ids["run_dir"], delivered, pulled, answer=answer)
    q_calls = [c for c in craft_calls if c.get("how") == "question"]
    return {
        "arm": arm, "steps": steps, "requests": len(rows),
        "offer": {k: mcp0.get(k) for k in ("switch", "offered", "on_main",
                                           "why")},
        "tier": first.get("tier"),
        "package_switch": (first.get("skills") or {}).get("package_switch"),
        "stop": watch.stop_reason,
        "package_calls": pkg_calls, "sections": sections,
        "triggers": triggers, "craft_calls": craft_calls,
        "recall_after_section": follow,
        "package_tool_called": bool(pkg_calls),
        "first_package_call_step": pkg_calls[0]["step"] if pkg_calls else None,
        "section_delivered": any(s.get("chars") for s in sections),
        "recall_called": bool(craft_calls),
        "recall_after_section_called": bool(follow),
        "question_calls": len(q_calls),
        "crafts_pulled": sorted(pulled),
        "router_listed_pulled": sorted(pulled & listed),
        "pattern": proj, "invalid": invalid or None}


def run_trial(arm: str, n: int, a) -> dict:
    saved = (LP.FEATURES, LP.Watch, LP.summarise)
    LP.FEATURES = arm_features(arm)
    LP.Watch = lambda ids, _mr, mm: StepWatch(ids, a.steps, mm)
    LP.summarise = lambda ids, watch: summarise_steps(ids, watch, arm,
                                                      a.steps)
    try:
        ns = argparse.Namespace(**{**vars(a), "tag": f"{a.tag}-{arm}",
                                   "max_requests": a.steps})
        row = LP.run_trial(n, ns)
    finally:
        LP.FEATURES, LP.Watch, LP.summarise = saved
    row["arm"] = arm
    row["steps_bound"] = a.steps
    return row


# ------------------------------------------------------------------ report ---
def report(rows: list[dict]) -> dict:
    out: dict = {}
    for arm in ARMS:
        r = [x for x in rows if x.get("arm") == arm
             and x.get("outcome") == "ok"]
        n = len(r)
        if not n:
            continue

        def rate(key):
            return round(sum(bool(x.get(key)) for x in r) / n, 3)
        sect = [x for x in r if x.get("section_delivered")]
        calls = collections.Counter(c.get("craft") for x in r
                                    for c in x.get("craft_calls") or []
                                    if c.get("craft"))
        pat = collections.Counter()
        for x in r:
            for c in (x.get("pattern") or {}).get("cases") or []:
                if c.get("applies"):
                    pat["applies"] += 1
                    pat["pitfall_avoided"] += int(not c["pitfall"])
                    pat["good_used"] += int(c["good"])
                    pat[f"applies_{c['via']}"] += 1
        out[arm] = {
            "n_valid_trials": n,
            "package_tool_call_rate": rate("package_tool_called"),
            "section_delivered_rate": rate("section_delivered"),
            "recall_call_rate": rate("recall_called"),
            "recall_after_section_rate_of_trials_with_a_section": round(
                sum(bool(x.get("recall_after_section_called"))
                    for x in sect) / len(sect), 3) if sect else None,
            "question_calls_total": sum(x.get("question_calls") or 0
                                        for x in r),
            "crafts_pulled": dict(calls.most_common(12)),
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
        est = (f"\nGPU time ESTIMATE: {a.trials} trials x {len(arms)} arms = "
               f"{a.trials * len(arms)} trials, each ~{per} s ({a.steps} requests x "
               f"{S_PER_REQUEST} s + {S_FIXED} s; S_PER_REQUEST is the median of "
               f"the lookup probe's ok trials with >= 2 requests, n=5, "
               f"bench/mcp/results/lookup_probe.jsonl) = "
               f"~{a.trials * len(arms) * per / 3600:.1f} h of an exclusive "
               f"GPU. Each question to yama_recall_craft adds two typed jjava "
               f"questions (the choice and the gate; ~0.6 s each, decide_turn's "
               f"measured p90 of 577 ms per question) and no generation.")
    return "\n".join([
        "THE PACKAGE SKILLS PROBE -- the plan (nothing was run, written or started)",
        "",
        f"arms ({len(arms)}): " + ", ".join(
            f"{x} {json.dumps(arm_features(x))}" for x in arms),
        f"trials per arm: {trials}; steps (requests) per trial: {steps}; "
        f"interleaved: trial 1 of each arm, then trial 2 ...",
        "per trial: lookup_probe.py's (pagoda prompt, Pi 0.87.1 in the box, --thinking "
        "medium, the recording relay), stopped after --steps requests -- not at the install",
        f"one row per trial appended to {RESULTS}",
        est,
        "",
        "BEFORE RUNNING (the coordinator):",
        "  - the proxy restarted with this code and the tier-medium MCP host up "
        "(GET /dash/api/mcp: packagelens `ready`); the armed library as the operator wants it measured",
        "  - an idle GPU (both cards: jjava reads the question path on bonsai-a4000) and the operator's go",
        f"  - then: python bench/mcp/package_skills_probe.py --run --trials {trials} "
        f"--steps {steps} --max-minutes <M> --tag {a.tag}",
        "  - then: python bench/mcp/package_skills_probe.py --report"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="run the trials (default: print the plan)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--trials", type=int, help="trials per arm")
    ap.add_argument("--steps", type=int, help="requests per trial")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tag", default="pkgskills")
    ap.add_argument("--tag-filter", help="with --report: only the rows of this run's tag")
    ap.add_argument("--key-file")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if a.report:
        rows = LP._rows(RESULTS)
        if a.tag_filter:
            # rows of one run: their tag is "<tag>-<arm>"
            rows = [r for r in rows if str(r.get("tag") or "").startswith(a.tag_filter + "-")]
        print(json.dumps(report(rows), indent=1))
        return 0
    if not a.run:
        print(plan(a))
        return 0
    missing = [f"--{k.replace('_', '-')}" for k in ("trials", "steps",
                                                    "max_minutes")
               if getattr(a, k) is None]
    if missing:
        print(f"--run needs {', '.join(missing)} (no defaults: the bounds are "
              f"the coordinator's)", file=sys.stderr)
        return 2
    arms = [x for x in a.arms.split(",") if x in ARMS]
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    for n in range(1, a.trials + 1):
        for arm in arms:
            row = run_trial(arm, n, a)
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row.get(k) for k in (
                "run_id", "arm", "outcome", "why", "stop", "requests",
                "package_tool_called", "section_delivered", "recall_called",
                "question_calls", "crafts_pulled", "invalid")}, indent=1),
                flush=True)
            if row.get("outcome") == "not_run":
                return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
