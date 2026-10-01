#!/usr/bin/env python
"""THE SERVER-TOOL TRIGGERS and THE WORK'S OWN EVIDENCE (skill_select,
2026-09-27). No GPU, no network.

    python mcp/test_tool_recall.py      -> "N/M checks passed"

The recall LINES were retired the same day (operator, after pagoda-h5).
The jobs the triggers fired (deep thinking, the plan: mcp/deep.py) were
removed 2026-09-29 (docs/REMOVED.md); the DETECTION,
skill_select.server_tool_triggers, remains and is gated here. "Names X"
below means "the first candidate names X".

WHAT IS GATED

  [retired]   no TOOL_RECALL_* text is registered or defined, and the
              selector's text names no server tool.
  [intent]    route.work_intent on bench/skills/work_intent.jsonl: every
              labelled paraphrase gets its label (IN-SAMPLE: the set was
              written before the detector, and the detector adjusted after
              its first pass -- 46/47 positives, 47/47 negatives on the
              first 94 rows; 5 hard rows added after). Precision and recall
              are printed.
  [probe]     a call that reads inside node_modules/<pkg>/ names
              yama_think_deeply and the package ONCE per package; a
              package.json version grep is not a probe; a scratch file that
              imports the package shares the key.
  [scratch]   a throwaway write (/tmp, _tmp_*, *.tmp.json) names
              yama_think_deeply; `cd DIR && cat > x.tmp.json` resolves to DIR.
  [plan]      the plan's FILES (plan_files_of): a project write in a
              directory the plan never named names yama_plan once per
              directory; every file written names it once more.
  [implement] a user turn after an answer that asks for something to be
              built names yama_plan; the first turn, a harness notice and a
              question do not.
  [offered]   nothing names a tool main does not have.
  [evidence]  pagoda-h4's blind injections, each with the control that
              shows the check can fail: a dependency's own files, an error
              located inside a dependency, a config file's keys, a craft
              about an area the conversation does not use, our own block
              handed back, a craft whose only evidence is its area's file.
  [proxy]     the user-turn tail selects on the CLIENT's messages, never the
              ledger-restored ones.
"""
from __future__ import annotations

import offline_stores  # noqa: I001  (before any store-owning module)

_TMP = offline_stores.isolate("yamadori_test_tool_recall_")

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
os.environ.setdefault("YAMADORI_SKILL_DECIDER", "stub")

import route  # noqa: E402
import skill_classify as C  # noqa: E402
import skill_prompts as P  # noqa: E402
import skill_select as S  # noqa: E402

# The deterministic stages only: the embedder and the fallback model are
# never reached from an offline suite (replay_selection's rule).
S.best_cosines = lambda q, pool: ({}, {
    "ok": False, "why": "offline suite: the embedding stage is not run"})
S.ask_fallback = lambda s, u: (_ for _ in ()).throw(
    RuntimeError("offline suite: the fallback is not run"))

_results: list[tuple[bool, str, str]] = []
# The tools' names, literal since mcp/deep.py was removed (2026-09-29), as
# skill_select.server_tool_triggers names them.
THINK, PLAN = "yama_think_deeply", "yama_plan"
# deep.KICKOFF_PLAN_ARGS as it was: the inserted kickoff call's arguments.
KICKOFF_PLAN_ARGS = {"task": "The task in the user's message above."}
BOTH = {THINK, PLAN}


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, str(detail)[:400]))
    return bool(ok)


# ------------------------------------------------------------- the pool --
def mk(sid, name, desc, applies, items, **gates):
    rule = C.rule_from_metadata(dict(applies, **gates))
    rule = C.with_gates(rule, description=desc, **{
        k: v for k, v in gates.items() if k in ("phases", "situations",
                                                "all_of", "topics")})
    body = "\n".join([name] + [f"- {it['form']}: {it['text']}"
                               for it in items])
    return {"id": sid, "version": "1", "name": name, "title": name,
            "description": desc, "rule": rule, "items": items,
            "body": body, "text": body}


DO = [{"form": "DO", "situation": "", "text": "do the thing well."}]
ONCLICK = mk("r_onclick", "react-event-handlers-onclick",
             "Use when passing a handler to onClick or onChange in JSX.",
             {"frameworks": ["react"], "languages": ["typescript"]}, DO,
             topics=["onClick", "onChange", "handleClick"])
REACTNODE = mk("r_node", "react-reactnode-children",
               "Use when annotating children as ReactNode in a component.",
               {"frameworks": ["react"], "languages": ["typescript"]}, DO,
               topics=["ReactNode", "ReactElement"])
TS6 = mk("t_ts6", "typescript-6-upgrade-module-flags",
         "Use when upgrading to TypeScript 6.0 and moduleResolution or "
         "allowSyntheticDefaultImports changes.",
         {"languages": ["typescript"]}, DO,
         topics=["moduleResolution", "allowSyntheticDefaultImports",
                 "esModuleInterop"])
TGPU = mk("g_unwrap", "typegpu-pipeline-unwrap",
          "Use when you need a TypeGPU pipeline with the raw WebGPU API "
          "(root.unwrap).", {"languages": ["typescript"]}, DO,
          topics=["root.unwrap", "pipeline.with", "setPipeline"])
URLSTATE = mk("r_url", "react-url-state",
              "Use when you need to store data in URL query strings.",
              {"frameworks": ["react"], "languages": ["typescript"]}, DO)
MATH = mk("m_rand", "math-seeded-random",
          "Use when drawing seeded random numbers with pmndrs math: "
          "mulberry32.create and mulberry32.sample.",
          {"frameworks": ["pmndrs_math"]}, DO,
          topics=["mulberry32.create", "mulberry32.sample"])
POOL = [ONCLICK, REACTNODE, TS6, TGPU, URLSTATE, MATH]
TOOLS = ["terminal", "read_file", "write_file", "patch"]


def U(text: str) -> list[dict]:
    return [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": text}]


_N = [0]


def step(msgs, name, args, result):
    _N[0] += 1
    cid = f"c{_N[0]}"
    return msgs + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": cid, "content": result}]


def decide(msgs, st, *, tools=BOTH, plan=None, pool=POOL, key=None):
    _N[0] += 1
    return S.decide(msgs, None, pool, TOOLS, st, key=key or f"k{_N[0]}",
                    server_tools=tools, plan_files=plan)


def recalls(rec):
    return [(d["name"], d["trigger"]) for d in rec.get("decisions") or []
            if d.get("form") == "tool_recall"]


def names(rec):
    return [d["name"] for d in rec.get("decisions") or []
            if d.get("form") != "tool_recall"]


PAGODA = ("Design a voxel pagoda garden. Build it with r3f "
          "(react-three-fiber) v10 and Koota and pmndrs math.")


# ============================================================= wording ==
def fires(msgs, st, *, tools=BOTH, plan=None, intent=None):
    """The server-tool trigger the proxy would fire for this request
    (skill_select.server_tool_triggers; deep.decide fires the first
    candidate and marks its key once the job ran): ([(tool, trigger)],
    the candidate or None, the state)."""
    st = {} if st is None else st
    kind = "user" if msgs[-1].get("role") == "user" else "step"
    c = S.server_tool_triggers(msgs, st, kind=kind,
                               think_ok=THINK in (tools or ()),
                               plan_ok=PLAN in (tools or ()),
                               plan_files=plan, intent=intent)
    if not c:
        return [], None, st
    S.mark_trigger_fired(st, c[0]["key"])
    return [(c[0]["tool"], c[0]["trigger"])], c[0], st


def test_wording():
    """RETIRED 2026-09-27 (operator, after pagoda-h5): the SERVER-TOOL
    RECALL lines. No TOOL_RECALL_* text is registered, and the selector's
    text never carries one -- the triggers run the job instead (deep.decide
    kind "auto")."""
    rows = {r["name"] for r in P.craft_registry()}
    check(not any(n.startswith("tool_recall") for n in rows)
          and not any(n.startswith("TOOL_RECALL") for n in dir(P)),
          "[retired] no TOOL_RECALL_* text is registered or defined",
          sorted(rows))
    m = U(PAGODA)
    m = step(m, "terminal", {"command": "ls node_modules/math/; cat "
                             "node_modules/math/package.json | head -40"},
             '{"output": "dist\\npackage.json"}')
    t, r, st = decide(m, None)
    check("server tool" not in t and not recalls(r),
          "[retired] a package probe: the selector's text names no server "
          "tool (the proxy runs the job itself)", t[-200:])

# ============================================================== intent ==
def test_intent():
    path = os.path.join(ROOT, "bench", "skills", "work_intent.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                rows.append(json.loads(ln))
    tp = fp = fn = tn = 0
    for r in rows:
        got = bool(route.work_intent(r["text"]))
        tp += got and r["intent"]
        fp += got and not r["intent"]
        fn += (not got) and r["intent"]
        tn += (not got) and not r["intent"]
        check(got == r["intent"], f"[intent] {r['text']!r} -> "
              f"{'intent' if r['intent'] else 'no intent'}",
              route.work_intent(r["text"]))
    check(sum(r["intent"] for r in rows) >= 40
          and sum(not r["intent"] for r in rows) >= 40,
          "[intent] at least 40 positive and 40 negative wordings",
          len(rows))
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    print(f"  [intent] in-sample on {len(rows)} rows: precision {p:.3f} "
          f"({tp}/{tp + fp}), recall {rc:.3f} ({tp}/{tp + fn})")


# =============================================================== probe ==
def test_probe_and_scratch():
    m = U(PAGODA)
    f, c, st = fires(m, None)
    check(not f, "[probe] the first user turn fires nothing (its plan is "
          "the kickoff's)", f)
    m = step(m, "terminal", {"command": "cd /tmp/r3fcheck && grep "
                             "'\"version\"' node_modules/three/package.json"},
             '{"output": "  \\"version\\": \\"0.186.1\\","}')
    f, c, st = fires(m, st)
    check(not f, "[probe] a package.json VERSION grep is not a probe", f)
    m = step(m, "terminal", {"command": "cd /tmp/math-check && ls "
                             "node_modules/math/; cat node_modules/math/"
                             "package.json | head -40"},
             '{"output": "dist\\npackage.json\\n{\\"name\\": \\"math\\"}"}')
    f, c, st = fires(m, st)
    check(f == [(THINK, "probe")] and c.get("package") == "math"
          and c.get("job") == "investigate" and c.get("key") == "think:math",
          "[probe] the first read inside node_modules/math fires deep "
          "thinking on math", (f, c))
    ev = (c or {}).get("evidence") or {}
    check(ev.get("package") == "math" and any(
        "node_modules/math" in p for p in ev.get("paths") or [])
          and "text" not in ev,
          "[probe] recorded with its evidence: the package and paths",
          ev)
    m = step(m, "read_file", {"path": "/workspace/pagoda/node_modules/math/"
                              "dist/src/random/index.d.ts"},
             "1|export * as mulberry32 from './mulberry32.js';")
    f, c, st = fires(m, st)
    check(not f, "[probe] the second math probe: nothing (once per "
          "package)", f)
    m = step(m, "terminal", {"command": "cd /workspace/pagoda && cat > "
                             "/tmp/t.ts << 'EOF'\nimport { mulberry32 } "
                             "from 'math/random'\nconst s = mulberry32."
                             "create(42)\nEOF\nnpx tsc --noEmit /tmp/t.ts"},
             '{"output": "../../tmp/t.ts(1,28): error TS2307: Cannot find '
             "module 'math/random'.\"}")
    f, c, st = fires(m, st)
    check(not f, "[scratch] /tmp/t.ts importing math after the math probe: "
          "nothing (the same key, think:math)", f)
    m = step(m, "terminal", {"command": "cd /workspace/pagoda && cat > "
                             "tsconfig.tmp.json << 'EOF'\n{\"compilerOptions"
                             "\": {\"module\": \"NodeNext\"}}\nEOF\nnpx tsc "
                             "-p tsconfig.tmp.json"}, '{"output": ""}')
    f, c, st = fires(m, st)
    check(f == [(THINK, "scratch")] and not c.get("package"),
          "[scratch] tsconfig.tmp.json (it imports nothing) fires deep "
          "thinking on the library it tests", (f, c))
    ev = (c or {}).get("evidence") or {}
    check(ev.get("path") == "/workspace/pagoda/tsconfig.tmp.json",
          "[scratch] `cd DIR && cat > x.tmp.json` is DIR/x.tmp.json", ev)
    m = step(m, "write_file", {"path": "/workspace/pagoda/tsconfig.tmp.json",
                               "content": "{}\n"}, '{"bytes_written": 3}')
    f, c, st = fires(m, st)
    check(not f, "[scratch] the same directory again: nothing", f)
    # A conversation whose FIRST sign is a scratch file importing a pinned
    # package: the package is named, trigger scratch.
    m2 = U(PAGODA)
    m2 = step(m2, "write_file", {"path": "/workspace/pagoda/package.json",
                                 "content": '{"dependencies": {"math": '
                                 '"0.1.0", "koota": "0.6.6"}}\n'},
              '{"bytes_written": 60}')
    m2 = step(m2, "write_file", {"path": "/workspace/pagoda/_tmp_test.ts",
                                 "content": "import { simplex2d } from "
                                 "'math/noise'\nconsole.log(simplex2d)\n"},
              '{"bytes_written": 70}')
    f, c, _st = fires(m2, None)
    check(f == [(THINK, "scratch")] and c.get("package") == "math",
          "[scratch] _tmp_test.ts importing math/noise names math", (f, c))
    # A candidate is not fired until the job ran: without the mark it is
    # offered again (a busy lane skips a request, not the trigger).
    st3: dict = {}
    c1 = S.server_tool_triggers(m2, st3, kind="step")
    c2 = S.server_tool_triggers(m2, st3, kind="step")
    check(c1 and c2 and c1[0]["key"] == c2[0]["key"],
          "[scratch] an unmarked candidate is offered again", (c1, c2))
    # Only yama_plan on main: no think trigger; no server tools: nothing.
    for tools, why in ((None, "no server tools"), ({PLAN}, "only yama_plan")):
        f, c, _st = fires(m2, None, tools=tools)
        check(not f, f"[offered] {why}: no yama_think_deeply trigger", f)


# ================================================================ plan ==
PLAN_TEXT = ("The plan for this task, written by a second model on the "
             "Yamadori server.\n\nFILES\n"
             "- `app/package.json` — deps\n"
             "- `app/src/main.tsx` — entry\n"
             "- `app/src/scene/App.tsx` & `app/src/scene/Tree.tsx` — the scene"
             "\nORDER\n- write the config\nKEY DECISIONS\n- three@0.185.1 "
             "pinned (reasoning)\nRISKS\n- none")


def _with_plan(msgs):
    return msgs + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "kp", "type": "function", "function": {
            "name": PLAN, "arguments": json.dumps(KICKOFF_PLAN_ARGS)}}]},
        {"role": "tool", "tool_call_id": "kp", "content": PLAN_TEXT}]


def test_plan():
    files = S.plan_files_of(_with_plan(U("Build the app.")))
    check(files == ["app/package.json", "app/src/main.tsx",
                    "app/src/scene/App.tsx", "app/src/scene/Tree.tsx"],
          "[plan] plan_files_of reads the FILES section's heads (two files "
          "joined by &; a pinned version is no file)", files)
    m = U("Build the app.")
    f, c, st = fires(m, None, plan=files)
    m = step(m, "write_file", {"path": "/w/app/package.json", "content":
                               "{}\n"}, '{"bytes_written": 3}')
    f, c, st = fires(m, st, plan=files)
    check(not f, "[plan] a planned file: nothing", f)
    m = step(m, "write_file", {"path": "/w/app/src/layout/builder.ts",
                               "content": "export const b = 1\n"},
             '{"bytes_written": 20, "lint": {"status": "error", "output": '
             '"x.ts(1,1): error TS2304: Cannot find name"}}')
    f, c, st = fires(m, st, plan=files)
    check(f == [(PLAN, "next_piece")] and c.get("piece") == "app/src/layout"
          and c.get("job") == "plan",
          "[plan] a write in a directory the plan never named fires "
          "yama_plan for the next piece (a lint error in the result does "
          "not un-write it)", (f, c))
    m = step(m, "write_file", {"path": "/w/app/src/layout/grid.ts",
                               "content": "export const g = 1\n"},
             '{"bytes_written": 20}')
    f, c, st = fires(m, st, plan=files)
    check(not f, "[plan] the same new directory again: nothing", f)
    m = step(m, "write_file", {"path": "/tmp/probe.ts", "content": "x\n"},
             '{"bytes_written": 2}')
    m = step(m, "patch", {"path": "/w/app/src/main.tsx", "old_string": "a",
                          "new_string": "b"},
             '{"success": false, "error": "Could not find a match"}')
    f, c, st = fires(m, st, plan=files)
    check((PLAN, "plan_done") not in f, "[plan] a failed patch is no write",
          f)
    for p in ("app/src/main.tsx", "app/src/scene/App.tsx",
              "app/src/scene/Tree.tsx"):
        m = step(m, "write_file", {"path": "/w/" + p, "content": "x\n"},
                 '{"bytes_written": 2}')
        f, c, st = fires(m, st, plan=files)
    check(f == [(PLAN, "plan_done")] and c.get("files") == files,
          "[plan] the last planned file written: yama_plan, once", (f, c))
    m = step(m, "write_file", {"path": "/w/app/src/scene/App.tsx",
                               "content": "y\n"}, '{"bytes_written": 2}')
    f, c, st = fires(m, st, plan=files)
    check(not f, "[plan] after it: nothing more", f)
    f, c, _st = fires(m, dict(st, fired={}), plan=files, tools={THINK})
    check(not f, "[offered] no yama_plan on main: no plan trigger", f)


# =========================================================== implement ==
def test_implement():
    m = U(PAGODA)
    f, c, st = fires(m, None)
    check(not f, "[implement] the conversation's first turn: nothing (the "
          "kickoff planned it)", f)
    m = m + [{"role": "assistant", "content": "Here is the plan: ..."}]
    for text, want in (("Great, now build the village next.", True),
                       ("Why does the build fail on Windows?", False),
                       ("[IMPORTANT: Background process proc_1 completed "
                        "normally (exit code 0).]", False),
                       ("ok now the audio system", True)):
        mm = m + [{"role": "user", "content": text}]
        f, c, st2 = fires(mm, json.loads(json.dumps(st)))
        got = f == [(PLAN, "implement")]
        check(got == want and (not want or (c.get("evidence") or {}).get(
            "intent") == "rule"),
              f"[implement] {text[:40]!r} -> "
              f"{'yama_plan' if want else 'nothing'} (the rule decides "
              f"when no decider answers)", (f, c))
        if want:
            f2, _c2, _ = fires(mm, st2)
            check(not f2, "[implement] the same user turn again: nothing "
                  "(once per user turn)", f2)
    # The DECIDER's build intent wins over the rule (operator, 2026-09-27:
    # the decider is the default for every judgment question).
    mm = m + [{"role": "user", "content": "Great, now build the village "
                                         "next."}]
    f, c, _ = fires(mm, json.loads(json.dumps(st)), intent=lambda t: False)
    check(not f, "[implement] the decider says no build intent: nothing, "
          "whatever the rule says", f)
    mm = m + [{"role": "user", "content": "Why does the build fail?"}]
    f, c, _ = fires(mm, json.loads(json.dumps(st)), intent=lambda t: True)
    check(f == [(PLAN, "implement")]
          and (c.get("evidence") or {}).get("intent") == "decider",
          "[implement] the decider says build intent: yama_plan, recorded "
          "as the decider's", (f, c))

# ============================================================ evidence ==
def test_evidence():
    base = U(PAGODA)
    # 1. A dependency's own files (h4 step 13: koota's README).
    readme = ("1|Koota is an ECS-based state management library.\n77|"
              "createRoot(document.getElementById('root')!)\n131|  return "
              "<button onClick={destroyAllShips}>Boom!</button>\n")
    _t, r, _ = decide(step(base, "read_file", {
        "path": "/tmp/koota-check/node_modules/koota/README.md"}, readme),
        None, tools=None)
    check("react-event-handlers-onclick" not in names(r),
          "[evidence] a dependency's README (onClick in its example) opens "
          "no React craft", names(r))
    check((r.get("set_aside") or {}).get("dependency_reads") == ["koota"],
          "[evidence] recorded: the dependency read set aside",
          r.get("set_aside"))
    code = ("import { useState } from 'react'\nexport function B() {\n"
            "  return <button onClick={handleClick}>Go</button>\n}\n")
    m = step(base, "write_file", {"path": "/w/pagoda/package.json",
                                  "content": '{"dependencies": {"react": '
                                  '"19.2.0"}}\n'}, '{"bytes_written": 40}')
    _t, r, _ = decide(step(m, "read_file", {"path": "/w/pagoda/src/B.tsx"},
                           code), None, tools=None)
    check("react-event-handlers-onclick" in names(r),
          "[evidence] CONTROL: the same onClick in the project's own file "
          "does", names(r))
    # 2. An error located inside a dependency (h4 step 89).
    lint = json.dumps({"bytes_written": 3033, "lint": {
        "status": "error", "output": "node_modules/@types/react/index.d.ts"
        "(436,10): error TS2456: Type alias 'ReactNode' circularly references"
        " itself.\nnode_modules/@types/react/index.d.ts(48,7): error TS2304: "
        "Cannot find name 'Iterable'."}})
    _t, r, _ = decide(step(m, "write_file", {
        "path": "/w/pagoda/src/palette.ts", "content": "export const P = []"
        "\n"}, lint), None, tools=None)
    check("react-reactnode-children" not in names(r),
          "[evidence] an error inside node_modules/@types/react names no "
          "craft of ours (ReactNode is the dependency's own symbol)",
          names(r))
    own = lint.replace("node_modules/@types/react/index.d.ts",
                       "src/scene/App.tsx")
    _t, r, _ = decide(step(m, "write_file", {
        "path": "/w/pagoda/src/palette.ts", "content": "export const P = []"
        "\n"}, own), None, tools=None)
    check("react-reactnode-children" in names(r),
          "[evidence] CONTROL: the same error in the project's App.tsx does",
          names(r))
    # 3. A config file's schema (h4 step 67), and the error that names it.
    tsconfig = ('{"compilerOptions": {"moduleResolution": "Bundler", '
                '"allowSyntheticDefaultImports": true}}\n')
    _t, r, _ = decide(step(m, "write_file", {
        "path": "/w/pagoda/tsconfig.json", "content": tsconfig},
        '{"bytes_written": 90}'), None, tools=None)
    check("typescript-6-upgrade-module-flags" not in names(r),
          "[evidence] writing tsconfig.json opens no TypeScript craft",
          names(r))
    _t, r, _ = decide(step(m, "write_file", {
        "path": "/w/pagoda/src/cfg.ts", "content": "export const c = " +
        json.dumps(tsconfig) + "\n// moduleResolution allowSyntheticDefault"
        "Imports\n"}, '{"bytes_written": 90}'), None, tools=None)
    check("typescript-6-upgrade-module-flags" in names(r),
          "[evidence] CONTROL: the same keys in a source file do", names(r))
    err = step(m, "terminal", {"command": "npx tsc --noEmit"},
               '{"output": "error TS5110: Option \'module\' must be set to '
               '\'NodeNext\' when option \'moduleResolution\' is set to '
               '\'NodeNext\'.", "exit_code": 2}')
    _t, r, _ = decide(err, None, tools=None)
    got = {d["name"]: d["trigger"] for d in r.get("decisions") or []}
    check(got.get("typescript-6-upgrade-module-flags") == "error",
          "[evidence] a tsc error naming moduleResolution brings it on the "
          "error", got)
    # 4. An area the conversation does not use (h4 relay 206/210: TypeGPU).
    tsx = ("import { useFrame } from '@react-three/fiber'\n// root.unwrap("
           "pipeline) and pipeline.with(pass) are TypeGPU's\n")
    _t, r, _ = decide(step(m, "write_file", {
        "path": "/w/pagoda/src/Scene.tsx", "content": tsx},
        '{"bytes_written": 90}'), None, tools=None)
    check("typegpu-pipeline-unwrap" not in names(r),
          "[evidence] a TypeGPU craft (filed under TypeScript alone) stays "
          "out of an r3f project that does not use TypeGPU", names(r))
    check(any("typegpu" in (x.get("why") or "") for x in r.get("skipped")
              or []), "[evidence] recorded: held back, about typegpu",
          r.get("skipped"))
    _t, r, _ = decide(step(m, "write_file", {
        "path": "/w/pagoda/src/Scene.tsx", "content":
        "import tgpu from 'typegpu'\n" + tsx}, '{"bytes_written": 90}'),
        None, tools=None)
    check("typegpu-pipeline-unwrap" in names(r),
          "[evidence] CONTROL: with `import tgpu from 'typegpu'` it goes in",
          names(r))
    # 5. Our own block handed back (the ledger-restored shape): not the
    # user's words, so TypeGPU is not "named by the user".
    ours = ("\n\n---\n" + P.CRAFT_HEADER + "\n\ntypegpu-pipeline-unwrap\n- "
            "DO: call root.unwrap(pipeline) with TypeGPU.\n")
    view = S.evidence_view(U(PAGODA + ours))[0]
    used = S.areas_in_use(view, C.request_signals(view))
    check("typegpu" not in used and {"r3f", "koota", "pmndrs_math"} <= used,
          "[evidence] our craft block in a user turn names no area", used)
    # 6. A craft whose only evidence is its area's file (h4 step 65).
    _t, r, _ = decide(step(m, "write_file", {
        "path": "/w/pagoda/vite.config.ts", "content": "import react from "
        "'@vitejs/plugin-react'\nexport default { plugins: [react()] }\n"},
        '{"bytes_written": 90}'), None, tools=None)
    check("react-url-state" not in names(r),
          "[evidence] writing vite.config.ts brings no topic-less React "
          "craft (URL state)", names(r))
    # 7. The error's own package.
    check(S.error_names(MATH, "error TS2307: Cannot find module "
                              "'math/random' or its types.")
          == ["names pmndrs_math"],
          "[evidence] an error naming module 'math/random' names the math "
          "craft", S.error_names(MATH, "Cannot find module 'math/random'"))


# =============================================================== proxy ==
def test_proxy_selects_on_the_clients_messages():
    import proxy
    import skill_select
    seen = {}
    real = skill_select.attach

    def spy(augmented, messages, sel, body=None):
        seen["augmented"] = augmented
        seen["messages"] = messages
        seen["body"] = body
        return augmented, {"on": True, "ids": []}
    raw = U("Build it with r3f v10.")
    restored = [dict(raw[0]), dict(raw[1], content=raw[1]["content"] + (
        "\n\n---\n" + P.CRAFT_HEADER + "\n\nroot.unwrap TypeGPU"))]
    skill_select.attach = spy
    try:
        proxy._SKILL_CTX.value = {"lineage": "", "key": "k", "raw": raw}
        proxy._skills_tail(restored, {"skills": True}, {"class": "prose"},
                           [], "acct")
    finally:
        skill_select.attach = real
        proxy._SKILL_CTX.value = None
    check(seen.get("messages") and "TypeGPU" not in json.dumps(
        seen["messages"]) and "TypeGPU" in json.dumps(seen["augmented"]),
          "[proxy] the tail matches on the client's messages and appends "
          "to the restored ones", json.dumps(seen.get("messages"))[:200])


def main() -> int:
    for fn in (test_wording, test_intent, test_probe_and_scratch, test_plan,
               test_implement, test_evidence,
               test_proxy_selects_on_the_clients_messages):
        print(f"\n--- {fn.__name__} ---")
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
    return 1 if fails else 0


if __name__ == "__main__":
    # jjava is always in scope where skills serve (operator, 2026-09-29):
    # offline, the STUBBED decider answers (mcp/decider_stub.py).
    import decider_stub
    with decider_stub.installed():
        sys.exit(main())
