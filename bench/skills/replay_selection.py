#!/usr/bin/env python
"""Which skills WOULD have been selected: the Octopus V0 run and a corpus
sample, replayed offline against the live skill store.

    python bench/skills/replay_selection.py [--run v0e-V0-xhigh-1]
        [--sample 20] [--seed 7] [--out PATH]
    python bench/skills/replay_selection.py --daily [-v] [--embed]

OFFLINE. The store is read (index/jobs.sqlite3, read only in effect: the
selection log is written to a temp copy); nothing reaches :1234, :11434 or
:1235. The embedding stage and the fallback model are NOT run -- the
deterministic stages decide, and an `ask` (a candidate the live cascade
would confirm or refuse with the embedder and the fallback) is reported as
such, never counted as injected.

  octopus   the Hermes session of an Octopus run (session*.jsonl under
            C:/Users/jwals/octo/logs/<run>/): the request at each user turn
            (where injection is decided) and, for the record, at every agent
            step (the skills a step's context would select if it were a user
            turn -- injection is NOT decided there).
  corpus    a seeded sample of the corpus's user-speaking task turns
            (index/corpus.sqlite3 `turn` events: the first 2,000 characters
            of the request, not utility calls), for a by-eye precision read.

`--transcript RUN` replays a run's whole Hermes stream request by request
(`transcript`, below).

The Octopus prompt is test material: this prints skill NAMES and counts per
request, never the prompt's text.

`--user-only` embeds the last user text alone (the query before
2026-09-27) for a before/after. `--embed` runs the embedding stage as well (the resident embedder on the
A4000, a GPU consumer: only when nothing else runs). The trigger vectors
come from YAMADORI_SKILL_TRIGGER_CACHE when it names a cache built for the
same armed set (`python mcp/skill_select.py --refresh-triggers` builds the
live one), else they are embedded here.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# YAMADORI_REPLAY_DB / YAMADORI_REPLAY_SKILLS_DIR replay another store (a
# temp store a change was tried in: mcp/skill_offline.py --temp --copy-live)
# instead of the live one. Either way it is copied first.
LIVE_DB = os.environ.get("YAMADORI_REPLAY_DB") or os.path.join(
    ROOT, "index", "jobs.sqlite3")
_TMP = tempfile.mkdtemp(prefix="yamadori_replay_sel_")
shutil.copyfile(LIVE_DB, os.path.join(_TMP, "jobs.sqlite3"))
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["YAMADORI_SKILLS_DIR"] = os.environ.get(
    "YAMADORI_REPLAY_SKILLS_DIR") or os.path.join(ROOT, "index", "skills")
# --embed reaches the A4000 through gpu_room (its lease, its room decision);
# without it nothing may reach llama-swap at all.
os.environ["YAMADORI_GPU_ROOM"] = "1" if "--embed" in sys.argv else "0"
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import route  # noqa: E402
import skill_limits as L  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402

_BEST_COSINES = skill_select.best_cosines   # --embed restores it
skill_select.best_cosines = lambda q, pool: ({}, {
    "ok": False, "why": "offline replay: the embedding stage is not run"})
skill_select.ask_fallback = lambda s, u: (_ for _ in ()).throw(
    RuntimeError("offline replay: the fallback is not run"))


def decide(messages: list[dict], pool: list[dict],
           offered: list[str] | None = None) -> dict:
    clean = [m for m in messages if m.get("role") in ("system", "user",
                                                      "assistant", "tool")]
    tools = []
    for m in clean:
        for c in m.get("tool_calls") or []:
            n = ((c or {}).get("function") or {}).get("name")
            if n and n not in tools:
                tools.append(n)
    try:
        rc = route.classify(clean, client_tools=tools, util={"utility": False},
                            gate=None, dbs={})["class"]
    except Exception:                                            # noqa: BLE001
        rc = None
    chosen, rec = skill_select.select(clean, rc, pool,
                                      tools=list(offered or tools))
    names = {s["id"]: s["name"] for s in pool}
    fb = rec.get("fallback") or {}
    return {"route": rc, "injected": [c["name"] for c in chosen],
            "ask": [names.get(i, i) for i in fb.get("asked") or []],
            "candidates": rec["candidates"]}


def _hermes_messages(session: dict) -> list[dict]:
    msgs = [{"role": "system", "content": session.get("system_prompt") or ""}]
    for m in session.get("messages") or []:
        mm = {"role": m.get("role"), "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = m["tool_calls"]
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except ValueError:
                    calls = []
            mm["tool_calls"] = calls
        msgs.append(mm)
    return msgs


def _session(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


def octopus(run: str, pool: list[dict]) -> dict:
    """Prompt 1 and its steps from session.jsonl; prompt 2 as the resumed
    request (prompt 1's history + prompt.p2.md as the next user turn) and
    its steps from session.p2.jsonl (Hermes compacted before prompt 2, so
    that session opens on the compaction summary)."""
    logs = os.path.join("C:/Users/jwals/octo/logs", run)
    p1 = os.path.join(logs, "session.jsonl")
    if not os.path.exists(p1):
        return {"error": f"no session.jsonl under {logs}"}
    m1 = _hermes_messages(_session(p1))
    out = {"session": "session.jsonl (+ session.p2.jsonl)",
           "messages": len(m1), "user_turns": [], "steps": {}}
    step_names = collections.Counter()
    steps = 0

    def walk(msgs, label):
        nonlocal steps
        for i, m in enumerate(msgs):
            if i == 0:
                continue
            upto = msgs[:i + 1]
            if m["role"] == "user":
                d = decide(upto, pool)
                out["user_turns"].append({"index": f"{label}@{i}", **d})
            elif m["role"] == "tool" and (i + 1 == len(msgs) or msgs[i + 1][
                    "role"] != "tool"):
                steps += 1
                for n in decide(upto, pool)["injected"]:
                    step_names[n] += 1
    p1q = os.path.join(logs, "prompt.md")
    if os.path.exists(p1q):
        # The session files open on Hermes' compaction summaries (both
        # sessions were compacted mid-run); the ORIGINAL first request is
        # the system prompt and prompt.md.
        with open(p1q, encoding="utf-8") as f:
            q1 = f.read()
        d = decide([m1[0], {"role": "user", "content": q1}], pool)
        out["user_turns"].append({"index": "p1 (original request)", **d})
    walk(m1, "p1-session (after a compaction)")
    p2q = os.path.join(logs, "prompt.p2.md")
    if os.path.exists(p2q):
        with open(p2q, encoding="utf-8") as f:
            q2 = f.read()
        d = decide(m1 + [{"role": "user", "content": q2}], pool)
        out["user_turns"].append({"index": "p2 (resumed request)", **d})
    p2 = os.path.join(logs, "session.p2.jsonl")
    if os.path.exists(p2):
        m2 = _hermes_messages(_session(p2))
        out["messages"] += len(m2)
        walk(m2, "p2-session")
    out["steps"] = {"requests": steps, "would_select": dict(
        step_names.most_common(15))}
    return out


def corpus(n: int, seed: int, pool: list[dict]) -> list[dict]:
    db = os.path.join(ROOT, "index", "corpus.sqlite3")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = [json.loads(p) for (p,) in con.execute(
            "SELECT payload FROM events WHERE kind='turn' ORDER BY id")]
    finally:
        con.close()
    eligible = [r for r in rows if str(r.get("utility")) in ("False", "false")
                and r.get("ends_on") == "user" and len(str(r.get(
                    "request") or "")) > 40]
    rnd = random.Random(seed)
    picks = rnd.sample(eligible, min(n, len(eligible)))
    out = []
    for r in picks:
        msgs = [{"role": "system", "content": r.get("system_head") or ""},
                {"role": "user", "content": r.get("request") or ""}]
        import ast
        try:
            offered = ast.literal_eval(str(r.get("client_tools") or "[]"))
        except (ValueError, SyntaxError):
            offered = []
        d = decide(msgs, pool, offered)
        out.append({"request": str(r.get("request"))[:160].replace("\n", " "),
                    "traffic": r.get("traffic"), **d})
    return out


# ------------------------------------------------------------ the stack --
# Operator, 2026-09-26: "we want our react-three-fiber, TSL and three.js
# skills loading and matching, we want a skill for Koota" -- the V4 stack
# (R3F 10 alpha + koota + pmndrs/math; since 2026-09-27 the V4 and pagoda
# prompts name it in ONE sentence, variants.V4_TECH, with no pins or rules:
# replays before that date ran the longer prompts). `--stack` replays the V4 prompt, the pagoda
# task's stack prompt and hand-written agent steps a run on this stack
# produces (a tool result showing an R3F component, a TSL build error, a
# koota query failing, a terrain written with math/noise). The steps are
# OURS, written for this replay (PROTOCOL rule 7 wants captured input: no
# V4 run exists yet; the Hermes system prompt and tool names ARE captured,
# from the v0f run's session). Names and verdicts are printed, never the
# Octopus prompt's text.
FAMILIES = (("koota", ("koota",)), ("math", ("math",)),
            ("r3f", ("r3f", "react-three")), ("tsl", ("tsl", "node-material")),
            ("three", ("three", "webgpu")))


def family(name: str) -> str | None:
    n = name.lower()
    for fam, keys in FAMILIES:
        if any(k in n for k in keys):
            return fam
    return None


def _hermes_context(run: str = "v0f-V0-xhigh-1") -> tuple[str, list[str]]:
    """The Hermes system prompt and tool names of a real run (captured)."""
    p = os.path.join("C:/Users/jwals/octo/logs", run, "session.jsonl")
    try:
        s = _session(p)
    except OSError:
        return "You are Hermes, an AI coding agent.", ["terminal", "read_file",
                                                       "write_file", "patch",
                                                       "search_files"]
    try:
        tools = [t["function"]["name"] for t in json.loads(
            s.get("tool_names") or "{}").get("tools") or []]
    except (ValueError, KeyError, TypeError):
        tools = []
    return s.get("system_prompt") or "", tools


def _call(i: int, name: str, args: dict) -> dict:
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


SCENE_TSX = """import { useFrame } from '@react-three/fiber/webgpu'
import { useQuery } from 'koota/react'
import { vec2 } from 'math'
import { world, Position, Velocity, IsOctopus } from '../world'
import { octopusMaterial } from '../materials'

export function Scene() {
  const octopi = useQuery(IsOctopus, Position)
  useFrame((_, delta) => {
    world.query(Position, Velocity).updateEach(([p, v]) => {
      vec2.scaleAndAdd(p, p, v, delta)
    })
  })
  return <instancedMesh args={[undefined, undefined, 256]} material={octopusMaterial} />
}
"""
MATERIALS_ERR = """> space-shooter@0.0.0 build
> tsc --noEmit && vite build

src/materials.ts:14:5 - error TS2339: Property 'colorNode' does not exist on type 'MeshBasicMaterial'.

14   mat.colorNode = texture(grid, uv()).mul(tint)
       ~~~~~~~~~

Found 1 error in src/materials.ts:14
"""
KOOTA_ERR = """[browser console]
Uncaught TypeError: Cannot read properties of undefined (reading 'x')
    at src/game.ts:31:17
    at updateEach (koota.js:1:5210)

src/game.ts:28-33:
  world.query(IsBullet, Position, Velocity).updateEach(([bullet, pos, vel]) => {
    pos.y -= vel.y * delta
  })
"""
TERRAIN_TS = """import { simplex2d, fbm } from 'math/noise'
import { mulberry32, random } from 'math/random'
import { trait } from 'koota'

export const Voxel = trait({ x: 0, y: 0, z: 0, color: 0 })
const noise = simplex2d.create(1234)
const rng = mulberry32.create(99)
export function heightAt(x: number, z: number): number {
  return fbm((f) => simplex2d.sample(noise, x * f, z * f), 5, 2, 0.5)
}
"""


def stack_prompts() -> dict[str, str]:
    """{label: prompt} -- V4 (test material: never printed) and the pagoda
    task's two prompts (bench/voxel/run.py)."""
    import importlib.util
    out: dict[str, str] = {}
    sys.path.insert(0, os.path.join(ROOT, "bench", "octopus"))
    import variants
    if os.path.isfile(variants.ORIGINAL):
        out["octopus V4 (user turn 1)"] = variants.build(variants.fetch(), "V4")[0]
    spec = importlib.util.spec_from_file_location(
        "voxel_run", os.path.join(ROOT, "bench", "voxel", "run.py"))
    vx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vx)
    tasks = getattr(vx, "TASKS", None) or {"single-html": vx.PROMPT}
    out["pagoda single-html (original)"] = tasks["single-html"]
    if "r3f-stack" in tasks:
        out["pagoda r3f-stack"] = tasks["r3f-stack"]
    return out


def stack_scenarios() -> list[tuple[str, list[dict], list[str]]]:
    system, tools = _hermes_context()
    sysm = {"role": "system", "content": system}
    prompts = stack_prompts()
    out = [(label, [sysm, {"role": "user", "content": text}], tools)
           for label, text in prompts.items()]
    v4 = prompts.get("octopus V4 (user turn 1)")
    base = [sysm, {"role": "user", "content": v4 or "Build the game."}]
    steps = [
        ("step: read_file shows an R3F component",
         [_call(1, "read_file", {"path": "src/components/Scene.tsx"})], SCENE_TSX),
        ("step: a TSL material fails tsc",
         [_call(2, "terminal", {"command": "npm run build"})], MATERIALS_ERR),
        ("step: a koota query throws",
         [_call(3, "terminal", {"command": "node scripts/smoke.mjs"})], KOOTA_ERR),
    ]
    for label, calls, result in steps:
        msgs = base + [{"role": "assistant", "content": "", "tool_calls": calls},
                       {"role": "tool", "tool_call_id": calls[0]["id"],
                        "content": result}]
        out.append((label, msgs, tools))
    pag = prompts.get("pagoda r3f-stack")
    if pag:
        msgs = [sysm, {"role": "user", "content": pag},
                {"role": "assistant", "content": "", "tool_calls": [
                    _call(4, "write_file", {"path": "src/terrain.ts",
                                            "content": TERRAIN_TS})]},
                {"role": "tool", "tool_call_id": "call_4",
                 "content": '{"bytes_written": 412}'}]
        out.append(("step: pagoda terrain written with math/noise", msgs, tools))
    # A later user turn in the same V4 conversation (where injection is
    # decided again): the history carries the koota query that failed.
    msgs = base + [{"role": "assistant", "content": "", "tool_calls": [
        _call(5, "read_file", {"path": "src/game.ts"})]},
        {"role": "tool", "tool_call_id": "call_5", "content": KOOTA_ERR},
        {"role": "assistant", "content": "The game builds and runs."},
        {"role": "user", "content": "The bullets never move and the console "
                                    "shows a TypeError in game.ts."}]
    out.append(("user turn 2: bullets never move (koota)", msgs, tools))
    return out


def stack_row(messages: list[dict], pool: list[dict], tools: list[str]) -> dict:
    import skill_classify
    clean = [m for m in messages if m.get("role") in ("system", "user",
                                                      "assistant", "tool")]
    try:
        rc = route.classify(clean, client_tools=tools, util={"utility": False},
                            gate=None, dbs={})["class"]
    except Exception:                                            # noqa: BLE001
        rc = None
    chosen, rec = skill_select.select(clean, rc, pool, tools=list(tools))
    sig = skill_classify.request_signals(clean, rc, tools)
    cos, emb = skill_select.best_cosines(sig.get("embed_query")
                                         or sig["query"], pool)
    fams: dict[str, list[str]] = {}
    for s in pool:
        fam = family(s.get("name") or "")
        if not fam:
            continue
        det = skill_classify.match(s.get("rule") or {}, sig)
        c = cos.get(s["id"]) if emb.get("ok") else None
        v = skill_select.verdict(det["strength"], c)
        if v != "none":
            fams.setdefault(fam, []).append(
                f"{s['name']}:{v}" + (f"@{c:.2f}" if c is not None else ""))
    names = {s["id"]: s["name"] for s in pool}
    fb = rec.get("fallback") or {}
    return {"route": rc, "injected": [c["name"] for c in chosen],
            "ask": [names.get(i, i) for i in fb.get("asked") or []],
            "candidates": rec["candidates"],
            "terms": sorted(sig["terms"]), "phases": sorted(sig["phases"]),
            "families": {k: sorted(v) for k, v in fams.items()},
            "embedding": {"ok": emb.get("ok"), "why": emb.get("why"),
                          "query": sig.get("embed_from")},
            "dropped": [names.get(d["id"], d["id"]) for d in rec.get("dropped") or []]}


def stack(pool: list[dict]) -> list[dict]:
    rows = []
    for label, msgs, tools in stack_scenarios():
        rows.append({"scenario": label, **stack_row(msgs, pool, tools)})
    return rows


# ------------------------------------------------------------ daily work --
# Operator, 2026-09-27: "evaluate whether this model is useful in daily work
# (not only the pmndrs stack)" and "ensure the engine isn't just injecting
# 'use this shit' blindly, it's more about I asked for it, reinforce it".
# `--daily` replays bench/skills/daily_eval.jsonl -- everyday requests across
# the operator's work, each labelled FROM THE REQUEST'S INTENT: the skills
# (or areas) that MUST go in because the user asked for that stack or topic,
# the areas that MUST NOT (unrelated stacks), "nothing" cases, and
# SEQUENCES that exercise the per-turn engine (skill_select.decide: a body
# the first time, a recall line when an error brings the area back, nothing
# on an unrelated step). Deterministic stages only unless --embed (the
# fallback model is never run). Fails on any MUST-NOT, NOTHING or MUST miss.
DAILY = os.path.join(HERE, "daily_eval.jsonl")
HERMES_TOOLS = ["terminal", "read_file", "write_file", "patch",
                "search_files", "skills_list", "skill_view",
                "vision_analyze"]
DAILY_SYSTEM = ("You are Hermes, an AI coding agent. You work in the user's "
                "project with your tools.")


def _source_text(name: str) -> str:
    """Test material named, never printed: the Octopus spec, the pagoda
    prompt."""
    import importlib.util
    if name in ("octopus", "octopus_v4"):
        sys.path.insert(0, os.path.join(ROOT, "bench", "octopus"))
        import variants
        if name == "octopus_v4":
            # The exact V4 prompt: the original with its one-line tech
            # statement replaced by the operator's (variants.V4_TECH).
            return variants.build(variants.fetch(), "V4")[0]
        with open(variants.ORIGINAL, encoding="utf-8") as f:
            return f.read()
    spec = importlib.util.spec_from_file_location(
        "voxel_run", os.path.join(ROOT, "bench", "voxel", "run.py"))
    vx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vx)
    if name == "pagoda_stack":
        return vx.TASKS["r3f-stack"]
    return vx.PROMPT


def _daily_rows(path: str = DAILY) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                rows.append(json.loads(ln))
    return rows


def _user_text(row: dict) -> str:
    if row.get("source"):
        base = _source_text(row["source"])
        return base + ("\n\n" + row["append"] if row.get("append") else "")
    return row.get("user") or ""


def _judge(exp: dict, got: list[dict], text: str, pool_area: dict,
           label: str) -> tuple[list[tuple[bool, str]], set, set]:
    """[(ok, what)] for one request's expectations, the areas it injected,
    and the areas it was expected to."""
    checks = []
    names = [g["name"] for g in got]
    areas = {pool_area.get(n) for n in names}
    want_areas = set()
    for m in exp.get("must") or []:
        if m.startswith("area:"):
            a = m[5:]
            want_areas.add(a)
            checks.append((a in areas, f"{label}: MUST inject from area {a}"))
        elif m not in pool_area and m in (exp.get("pending_reingest") or {}):
            # PENDING RE-INGESTION (2026-09-28, skills/replacements/
            # 2026-09-28-assured-voice.json): the named craft was archived
            # when its replacement batch armed; until the row names the
            # replacement, the batch's area stands in for it. While the
            # named craft is still armed, the named check above applies.
            a = exp["pending_reingest"][m][5:]
            want_areas.add(a)
            checks.append((a in areas, f"{label}: MUST inject from area {a} "
                           f"(pending re-ingestion of {m})"))
        else:
            want_areas.add(pool_area.get(m))
            checks.append((m in names, f"{label}: MUST inject {m}"))
    for a in exp.get("must_not_areas") or []:
        bad = [n for n in names if pool_area.get(n) == a]
        checks.append((not bad, f"{label}: MUST NOT inject area {a}"
                       + (f" (got {bad})" if bad else "")))
    for n in exp.get("must_not") or []:
        checks.append((n not in names, f"{label}: MUST NOT inject {n}"))
    if exp.get("nothing"):
        checks.append((not got, f"{label}: injects NOTHING"
                       + (f" (got {names})" if got else "")))
    for n, form in (exp.get("forms") or {}).items():
        g = next((x for x in got if x["name"] == n), None)
        checks.append((g is not None and g["form"] == form,
                       f"{label}: {n} as a {form}"
                       + (f" (got {g['form'] if g else 'nothing'})")))
    for n, sub in (exp.get("recall_has") or {}).items():
        checks.append((sub in text, f"{label}: the recall of {n} names "
                       f"{sub!r}"))
    return checks, {a for a in areas if a}, {a for a in want_areas if a}


def daily(pool: list[dict], path: str = DAILY, verbose: bool = False
          ) -> dict:
    """Run the daily eval; {checks: [(ok, what)], per_area, requests}."""
    import skill_classify
    pool_area = {s["name"]: skill_select.area_of(s.get("rule") or {})
                 for s in pool}
    by_name = {s["name"]: s for s in pool}
    checks: list[tuple[bool, str]] = []
    per = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    requests = []
    for row in _daily_rows(path):
        tools = row.get("tools") or HERMES_TOOLS
        sysm = {"role": "system", "content": row.get("system")
                or DAILY_SYSTEM}
        msgs = [sysm] + list(row.get("prior") or [])
        turns = row.get("turns") or [{"user": _user_text(row),
                                      "expect": row}]
        state = None
        for ti, t in enumerate(turns):
            if "user" in t:
                msgs = msgs + [{"role": "user", "content": t["user"]}]
            if "reply" in t:
                msgs = msgs + [{"role": "assistant", "content": t["reply"]}]
            if "step" in t:
                st_ = dict(t["step"])
                res_ = str(st_.get("result") or "")
                if res_.startswith("FILL:"):
                    # A long, unrelated tool output (npm, a linter): the
                    # conversation grows past the fade without the area.
                    n = int(res_[5:])
                    line = ("npm warn deprecated inflight@1.0.6: this module "
                            "is not supported, and leaks memory\n")
                    st_["result"] = (line * (n // len(line) + 1))[:n]
                cid = f"call_{ti}"
                msgs = msgs + [{"role": "assistant", "content": "",
                                "tool_calls": [_call(ti, st_["name"],
                                                     st_.get("args") or {})]},
                               {"role": "tool", "tool_call_id": cid,
                                "content": st_.get("result") or ""}]
            if "expect" not in t:
                continue
            clean = [m for m in msgs if m.get("role") in (
                "system", "user", "assistant", "tool")]
            try:
                rc = route.classify(clean, client_tools=tools,
                                    util={"utility": False}, gate=None,
                                    dbs={})["class"]
            except Exception:                                    # noqa: BLE001
                rc = None
            # STAGE 1 is what this eval judges (which skills are offered,
            # as a body or a recall); the injector's gate is jjava's, run
            # here by the STUB decider (mcp/decider_stub.py, pass_all), and
            # tuned on its own labels (bench/skills/inject_labels.py).
            import decider_stub
            with decider_stub.installed():
                text, rec, state = skill_select.decide(
                    clean, rc, pool, tools, state, key=f"{row['id']}:{ti}")
            got = [{"name": d["name"], "form": d["form"],
                    "trigger": d["trigger"], "slot": d.get("slot")}
                   for d in rec.get("stage1") or []]
            label = row["id"] + (f"#{ti}" if len(turns) > 1 else "")
            cs, areas, want = _judge(t["expect"], got, text, pool_area, label)
            checks += cs
            # Precision counts an injected area against what was asked for
            # (must) and allowed; recall only against what was asked for.
            allowed = set(t["expect"].get("allow_areas") or [])
            fp_areas = sorted(a for a in areas - want - allowed)
            for a in areas | want:
                if a in areas and a in want:
                    per[a]["tp"] += 1
                elif a in areas and a not in allowed:
                    per[a]["fp"] += 1
                elif a in want:
                    per[a]["fn"] += 1
            nm = {s["id"]: s["name"] for s in pool}
            requests.append({"id": label, "cat": row.get("cat"),
                             "route": rc, "got": got, "fp_areas": fp_areas,
                             # THE SELECTOR (2026-09-27): each round's
                             # survivors, each question's options (names)
                             # with the decider's probabilities and pick,
                             # and the statechart's state and transitions.
                             "rounds": [(r["name"], r["survivors"], r["how"])
                                        for r in rec.get("rounds") or []],
                             "questions": [{
                                 "kind": q["kind"], "area": q["area"],
                                 "options": [nm.get(o, o) for o in
                                             q["options"]],
                                 "p": [q["probabilities"].get(o, 0.0)
                                       for o in q["options"]],
                                 "pick": nm.get(q["pick"], q["pick"]),
                                 "decider": q["decider"]}
                                 for q in rec.get("questions") or []],
                             "chart": {k: (rec.get("chart") or {}).get(k)
                                       for k in ("state", "area", "areas",
                                                 "transitions")},
                             "tokens": sum(L.tokens(skill_select.injected_text(
                                 by_name[g["name"]])) if g["form"] == "body"
                                           else 0 for g in got
                                           if g["name"] in by_name),
                             "asked": sorted((skill_classify.asked_terms(
                                 skill_classify._last_user_text(clean))
                                 or {}))})
    return {"checks": checks, "per_area": dict(per), "requests": requests}


def print_selector(r: dict) -> None:
    """One request's rounds (survivors per round), questions (options,
    probabilities, pick) and statechart."""
    print("      rounds: " + " -> ".join(f"{n} {k}" + ("" if h in (
        "pattern", "store", "chart") else f" ({h})") for n, k, h in
        r.get("rounds") or []))
    for q in r.get("questions") or []:
        opts = ", ".join(f"{o} {p:.2f}" for o, p in zip(q["options"],
                                                         q["p"]))
        print(f"      Q {q['kind']}:{q['area']} [{q['decider']}] -> "
              f"{q['pick'] or 'none'}   {{{opts}}}")
    ch = r.get("chart") or {}
    if ch.get("state") or ch.get("transitions"):
        tr = "; ".join(f"{t.get('area') or 'top'} {t['from']} --{t['event']}"
                       f"--> {t['to']}" for t in ch.get("transitions") or [])
        print(f"      chart: {ch.get('state')}({ch.get('area') or ''}) "
              f"areas {ch.get('areas') or {}}" + (f"; {tr}" if tr else ""))


def print_daily(res: dict, verbose: bool = False, show: str = "") -> int:
    bad = [c for c in res["checks"] if not c[0]]
    for r in res["requests"]:
        if verbose or (show and any(s and s in r["id"]
                                    for s in show.split(","))):
            print(f"  {r['id']:<28} [{r['cat']}/{r['route']}] asked "
                  f"{r['asked'] or '-'} -> "
                  + (", ".join(f"{g['name']}({g['form']},{g['trigger']})"
                               for g in r["got"]) or "-")
                  + (f"  ~{r['tokens']} tok" if r["tokens"] else ""))
            print_selector(r)
    print("\n  per area (requests):  area  precision  recall  (tp/fp/fn)")
    for a, v in sorted(res["per_area"].items()):
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else None
        rc = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None
        print(f"    {a:<12} {'-' if p is None else f'{p:.2f}':>9} "
              f"{'-' if rc is None else f'{rc:.2f}':>7}   "
              f"({v['tp']}/{v['fp']}/{v['fn']})")
    fps = [(r["id"], r["fp_areas"], [g["name"] for g in r["got"]])
           for r in res["requests"] if r.get("fp_areas")]
    if fps:
        print("\n  unlabelled areas injected (count against precision):")
        for rid, fa, names in fps:
            print(f"    {rid}: {fa} <- {names}")
    for ok, what in res["checks"]:
        if not ok or verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    kinds = collections.Counter()
    kinds_ok = collections.Counter()
    for ok, what in res["checks"]:
        k = ("MUST-NOT" if "MUST NOT" in what else "MUST" if "MUST inject"
             in what else "NOTHING" if "NOTHING" in what else "form/recall")
        kinds[k] += 1
        kinds_ok[k] += bool(ok)
    print("\n  by kind: " + ", ".join(f"{k} {kinds_ok[k]}/{kinds[k]}"
                                      for k in ("MUST", "MUST-NOT",
                                                "NOTHING", "form/recall")
                                      if kinds[k]))
    nq = sum(len(r.get("questions") or []) for r in res["requests"])
    print(f"  decider {os.environ.get('YAMADORI_SKILL_DECIDER') or 'stub'}: "
          f"{nq} questions over {len(res['requests'])} requests")
    n = len(res["checks"])
    print(f"\n{n - len(bad)}/{n} checks passed")
    return 1 if bad else 0


# ------------------------------------------------------------ transcript --
# THE WHOLE CONVERSATION, REQUEST BY REQUEST (2026-09-27): an Octopus/pagoda
# run's Hermes stream (hermes.jsonl: its text, tool_use and tool_result
# events -- the session export keeps only what the last compaction left)
# replayed through skill_select.decide with the state carried, main offered
# yama_think_deeply and yama_plan, and the kickoff plan's FILES read
# READ-ONLY from the live ledger (index/nebari.sqlite3, the hop recorded
# inside the run's time window). Prints each request that injected craft or
# a server-tool recall, by message index. `--no-server-tools` replays the
# craft alone.
def _stream_messages(run_dir: str) -> list[dict]:
    evs = []
    with open(os.path.join(run_dir, "hermes.jsonl"), encoding="utf-8") as f:
        for ln in f:
            try:
                evs.append(json.loads(ln))
            except ValueError:
                pass
    with open(os.path.join(run_dir, "prompt.md"), encoding="utf-8") as f:
        prompt = f.read()
    msgs = [{"role": "system", "content": DAILY_SYSTEM},
            {"role": "user", "content": prompt}]
    cur, pending, n = None, [], 0
    for e in evs:
        t = e.get("type")
        if t in ("text", "tool_use"):
            if cur is None:
                cur = {"role": "assistant", "content": "", "tool_calls": []}
                msgs.append(cur)
                pending = []
            if t == "text":
                cur["content"] += e.get("text") or ""
            else:
                n += 1
                cur["tool_calls"].append({
                    "id": f"call_{n}", "type": "function", "function": {
                        "name": e.get("name"),
                        "arguments": json.dumps(e.get("input") or {})}})
                pending.append(f"call_{n}")
        elif t == "tool_result":
            msgs.append({"role": "tool", "tool_call_id": pending.pop(0)
                         if pending else f"call_x{len(msgs)}",
                         "content": e.get("output") or ""})
            cur = None
    for m in msgs:
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            m.pop("tool_calls", None)
    return msgs


def _run_plan_files(run_dir: str) -> list[str] | None:
    """The kickoff plan's FILES from the live ledger, read only."""
    try:
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        db = os.path.join(ROOT, "index", "nebari.sqlite3").replace("\\", "/")
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute(
            "select text from additions where kind='hops' and ts between ? "
            "and ? and text like '%FILES%' order by ts",
            (meta["t_start"], meta["t_end"])).fetchall()
        con.close()
    except Exception:                                            # noqa: BLE001
        return None
    for (text,) in rows:
        stack = [json.loads(text)]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                stack += list(x.values())
            elif isinstance(x, list):
                stack += x
            elif isinstance(x, str) and "\nFILES\n" in "\n" + x:
                got = skill_select.plan_files_of([
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "p", "function": {"name": "yama_plan",
                                                 "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "p", "content": x}])
                if got:
                    return got
    return None


def transcript(run: str, pool: list[dict], server: bool = True) -> dict:
    run_dir = run if os.path.isdir(run) else os.path.join(
        "C:/Users/jwals/octo/logs", run)
    msgs = _stream_messages(run_dir)
    plan = _run_plan_files(run_dir) if server else None
    tools = HERMES_TOOLS
    state, out = None, []
    for i in range(2, len(msgs) + 1):
        last = msgs[i - 1]
        if last["role"] == "assistant" or (
                last["role"] == "tool" and i < len(msgs)
                and msgs[i]["role"] == "tool"):
            continue
        upto = msgs[:i]
        try:
            rc = route.classify(upto, client_tools=tools,
                                util={"utility": False}, gate=None,
                                dbs={})["class"]
        except Exception:                                        # noqa: BLE001
            rc = None
        import decider_stub
        with decider_stub.installed():
            _t, rec, state = skill_select.decide(
                upto, rc, pool, tools, state, key=f"t:{i}",
                server_tools=({"yama_think_deeply", "yama_plan"} if server
                              else None), plan_files=plan)
        got = [{"name": d["name"], "trigger": d["trigger"],
                "form": d["form"], "evidence": d.get("evidence")}
               for d in rec.get("stage1") or []]
        if got:
            out.append({"index": i - 1, "kind": rec.get("kind"),
                        "got": got})
    return {"run": run, "messages": len(msgs), "plan_files": plan,
            "requests": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="",
                    help="replay a run's whole Hermes stream request by "
                         "request (a run name under C:/Users/jwals/octo/logs "
                         "or a directory)")
    ap.add_argument("--no-server-tools", action="store_true",
                    help="--transcript: main is offered no server tool")
    ap.add_argument("--run", default="v0e-V0-xhigh-1")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--stack", action="store_true",
                    help="only the V4 stack scenarios (R3F 10 + koota + TSL + math)")
    ap.add_argument("--user-only", action="store_true",
                    help="embed the last user text alone (the query before "
                         "2026-09-27), for a before/after on the same store")
    ap.add_argument("--daily", action="store_true",
                    help="the daily-work selection eval "
                         "(bench/skills/daily_eval.jsonl); exit 1 on any "
                         "MUST, MUST-NOT or NOTHING miss")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--show", default="",
                    help="--daily: print the rounds, questions and chart of "
                         "the requests whose id contains one of these "
                         "(comma-separated)")
    ap.add_argument("--decider", default="",
                    help="the first decider (stub | lexical | clm; "
                         "YAMADORI_SKILL_DECIDER); the gate runs the stub")
    ap.add_argument("--embed", action="store_true",
                    help="run the embedding stage too (the embedder on the A4000 "
                         "via llama-swap: a GPU consumer -- only when nothing "
                         "else is running; the fallback model is still not run)")
    a = ap.parse_args(argv)
    if a.decider:
        os.environ["YAMADORI_SKILL_DECIDER"] = a.decider
    if a.embed:
        skill_select.best_cosines = _BEST_COSINES
    if a.user_only:
        import skill_classify
        skill_classify.embedding_query = lambda msgs: (
            skill_classify._last_user_text(msgs)[:2000], {"user_only": True})
    pool = skills.armed()
    print(f"pool: {len(pool)} armed skills (live store, read offline)")
    if a.transcript:
        res = transcript(a.transcript, pool, not a.no_server_tools)
        print(f"== {res['run']}: {res['messages']} messages; plan files "
              f"{len(res['plan_files'] or [])}")
        for r in res["requests"]:
            print(f"  @{r['index']:<4} {r['kind']:<5} " + ", ".join(
                f"{g['name']}({g['form']},{g['trigger']})" for g in r["got"]))
            for g in r["got"]:
                if g["form"] == "tool_recall":
                    print(f"         {json.dumps(g['evidence'])[:200]}")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
        return 0
    if a.daily:
        res = daily(pool)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in res.items() if k != "checks"},
                          f, indent=1)
        return print_daily(res, a.verbose, a.show)
    if a.stack:
        rows = stack(pool)
        for r in rows:
            print(f"\n== {r['scenario']}: route {r['route']}; terms {r['terms']}; "
                  f"phases {r['phases']}; {r['candidates']} candidate(s)")
            print(f"   injected: {r['injected'] or '-'}")
            print(f"   ask:      {r['ask'] or '-'}")
            if r["dropped"]:
                print(f"   dropped:  {r['dropped']}")
            for fam, _k in FAMILIES:
                print(f"   {fam:6s} {r['families'].get(fam) or '-'}")
            q = (r.get("embedding") or {}).get("query") or {}
            print(f"   query:    {json.dumps(q, sort_keys=True)}"
                  + ("" if (r.get("embedding") or {}).get("ok") else
                     f" (embedding not run: {r['embedding'].get('why')})"))
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1)
        return 0
    oc = octopus(a.run, pool)
    print(f"\n== Octopus {a.run} ({oc.get('session')}, "
          f"{oc.get('messages')} messages)")
    for u in oc.get("user_turns") or []:
        print(f"  user turn @{u['index']}: route {u['route']}; injected "
              f"{u['injected'] or '-'}; ask {u['ask'] or '-'}; "
              f"{u['candidates']} candidate(s)")
    st = oc.get("steps") or {}
    print(f"  agent steps: {st.get('requests')} requests; skills their "
          f"context would select (not injected at steps): "
          f"{st.get('would_select')}")
    cs = corpus(a.sample, a.seed, pool)
    print(f"\n== corpus sample (n={len(cs)}, seed {a.seed})")
    for i, c in enumerate(cs, 1):
        print(f"  {i:>2}. [{c['traffic']}/{c['route']}] {c['request'][:110]!r}"
              f"\n      injected {c['injected'] or '-'}"
              + (f"; ask {c['ask']}" if c["ask"] else ""))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"octopus": oc, "corpus": cs}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
