#!/usr/bin/env python
"""The pagoda through Hermes or Pi: bench/voxel's r3f-stack prompt, run like
Octopus V4, measured like bench/voxel.

    python bench/octopus/run.py --task pagoda --arm xhigh --tag pagoda-h1
    python bench/octopus/run.py --task pagoda --harness pi --arm xhigh --tag pagoda-p1
    python bench/octopus/run.py --task pagoda [--harness pi] --arm max --tag T --print
    python bench/octopus/run.py --measure pagoda-h1-pagoda-xhigh-1   # re-measure

PI (2026-09-28): the same prompt as Pi's message, byte for byte, through
bench/sandbox/harness_box.py `run pi` (Pi 0.87.1 in the box, the default
loadout, the browser sidecar), with a fresh run folder as the project (/work,
Pi's current directory: the prompt's pagoda/ lands in it), `--thinking` =
the arm, every request through the same recording relay (the box's one
forward targets the relay's port), Pi's model entry built at run time from
the proxy's /v1/models row (pi_models), then the same measure(). The harness
is recorded in the run row and in pagoda.jsonl.

WHY (operator, 2026-09-27). The pagoda sent as ONE chat request with no
client tools (bench/voxel/results/pagoda-r3f-1, xhigh) routed `prose`, drew
four pictures with generate_image, hit the tool-turn cap and landed with a
written-out tool call as its answer. Through a real harness the model has
files and a terminal and builds the project the prompt asks for.

HOW IT RUNS: exactly as Octopus V4 (run.py run_one): the Windows Hermes on
the dogfood profile, speaking RESPONSES (make_profile.py --wire responses;
run.py refuses the task on a chat-wire profile), the default loadout (the
browser in the sandbox, the LSP fallback, files and terminal in the docker
sandbox on its --internal network), a fresh run folder mounted at
/workspace, SINGLE-SHOT: one prompt, no graded follow-up (the grader is a
measurement only). The prompt is bench/voxel/run.py TASKS["r3f-stack"],
byte for byte (its sha256 is recorded and checked): nothing of ours is
added -- no folder name, no port, no layout. The model chooses where the
project goes (grade.project_root finds it: a package.json or an index.html,
shallowest first).

HOW IT IS MEASURED (measure(); nothing measured ever goes back to a model):
  1. serve: the Octopus grader's app container (grade.app_up, V4's rules:
     npm install + npm run build when there is a package.json -- tsc only
     with a tsconfig.json -- then dist/ on :3001, else `vite` dev on :3001,
     else the folder as is) on a sandbox network. The prompt names no port;
     :3001 is the grader's. What the model's README says about running it is
     RECORDED (readme_run), not followed: a README is text, not a command.
  2. the page: bench/voxel/check.py's page checker (check_served ->
     page_check.py --url) in the app container's network namespace against
     http://localhost:3001/ -- the voxel sequence: load, t05/t10/t15 shots,
     fps, three.js scene stats, orbit/zoom shots -- then check.analyse (PIL
     over the shots: non-blank, colours, orbit by pixels) with the gate log.
  3. the stack: bench/voxel/check.py stack_static (the Octopus grader's V4
     checks: r3f v10 Canvas, koota world/traits/queries, pmndrs math used),
     with the version npm installed.
  4. the proxy side: relay_summary over the relay's rows (x_yamadori
     verbatim): route classes, skills injected with their triggers, deep
     thinking, our tools and the client's calls, tools withheld, landings,
     tokens.
Output: <logs>/<run_id>/measure.json, screenshots under
C:\\Users\\jwals\\octo\\grade\\<run_id>\\shots, one row appended to
bench/octopus/results/pagoda.jsonl.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import grade as G  # noqa: E402  (imports bench/octopus/run.py as `run` first)
import sandbox_net  # noqa: E402

TASK = "pagoda"
VOXEL_TASK = "r3f-stack"
RESULTS = os.path.join(HERE, "results", "pagoda.jsonl")
PROMPT_PATH = os.path.join(ROOT, "index", "octopus", "prompts", "pagoda.md")
URL = "http://localhost:3001/"
_VC = None


def voxel_check():
    """bench/voxel/check.py, loaded by path (bench/voxel has its own run.py:
    `run` stays bench/octopus/run.py, which grade imported above)."""
    global _VC
    if _VC is None:
        key = "voxel_check"
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                key, os.path.join(ROOT, "bench", "voxel", "check.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[key] = mod
            spec.loader.exec_module(mod)
        _VC = sys.modules[key]
    return _VC


# THE PROMPT FILE A RUN USES INSTEAD (run.py --prompt-file; operator, 2026-09-28: "Give it the prompt with the real
# package names and commands to install them ... exact versions"). None: bench/voxel's r3f-stack prompt.
PROMPT_OVERRIDE: str | None = None


def prompt() -> tuple[str, str]:
    """(the prompt, its sha256): bench/voxel/run.py TASKS["r3f-stack"], or the --prompt-file's bytes."""
    if PROMPT_OVERRIDE:
        raw = open(PROMPT_OVERRIDE, "rb").read()
        return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    return voxel_check()._voxel_run().task_prompt(VOXEL_TASK)


def write_prompt(path: str = PROMPT_PATH) -> tuple[str, str]:
    """The prompt file Hermes reads (--query-file), written only when it
    differs; (path, sha256), the file's bytes checked against the prompt.
    With --prompt-file, that file itself (never written)."""
    if PROMPT_OVERRIDE:
        return PROMPT_OVERRIDE, prompt()[1]
    text, sha = prompt()
    try:
        same = open(path, encoding="utf-8").read() == text
    except OSError:
        same = False
    if not same:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != sha:
        raise SystemExit(f"{path}: sha256 {got} != the prompt's {sha}")
    return path, sha


def profile_wire(path: str) -> str:
    """"responses" when the profile's octo-relay provider speaks Responses
    (make_profile.py --wire responses), else "chat"."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return "missing"
    m = re.search(r"(?ms)^providers:\n  octo-relay:\n(?P<b>(?:    [^\n]*\n)+)", text + "\n")
    if m and re.search(r"(?m)^    api_mode:\s*codex_responses\b", m.group("b")):
        return "responses"
    return "chat"


HARNESSES = ("hermes", "pi")


def run_id(tag: str, arm: str, rep: int, harness: str = "hermes") -> str:
    """Hermes keeps the ids it always had; another harness is named in the id
    so two harnesses' runs of one tag never share a folder."""
    return f"{tag}-{TASK}-{arm}-{rep}" if harness == "hermes" else f"{tag}-{TASK}-{harness}-{arm}-{rep}"


# ------------------------------------------------------------------ Pi (the harness box)
# Pi 0.87.1 in the harness box (bench/sandbox/harness_box.py `run pi`, the
# default loadout: read bash edit write grep find, skills page-check and
# type-check, the browser sidecar), with a fresh run folder as the project:
# /work is Pi's current directory, so the prompt's "pagoda/ folder in the
# current directory" lands in the run folder, as Hermes' /workspace does.
PI_MODEL_ID = "yamadori"           # the id in the host test's models.json (docs/HARNESS-PI.md)


def pi_args(effort: str, prompt_text: str) -> list[str]:
    """Pi's argv: JSON event mode, print (one prompt, then exit), the thinking
    level. `--thinking` is Pi's CLI flag (dist/cli/args.js; levels off minimal
    low medium high xhigh max, docs/cli.md); the model entry's
    thinkingLevelMap sends xhigh as reasoning_effort "xhigh" and max as "max"
    (pi-ai dist/api/openai-completions.js: `thinkingLevelMap[level] ?? level`;
    xhigh/max are offered only when the map names them, models.js
    getSupportedThinkingLevels). The prompt is the message itself, byte for
    byte (an `@file` argument would wrap it: `<file name="...">` + content,
    dist/cli/file-processor.js:60)."""
    return ["--mode", "json", "--thinking", effort, "-p", prompt_text]


def advertised_card(port: int, key: str, model: str = PI_MODEL_ID, timeout: float = 30) -> dict:
    """GET /v1/models through the harness's forward target (the relay on
    127.0.0.1:<port>, which forwards to the proxy): the chat model's row --
    its window (`context_length`, the main share, mcp/catalog.py
    context_window, THE LIMIT ENFORCED), its output ceiling
    (`max_completion_tokens`, catalog.max_output) and its input modalities.
    Read at RUN time: the window changes with deploys. Raises on anything
    but a row with a window."""
    import urllib.request
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    rows = [m for m in data.get("data") or [] if isinstance(m, dict)]
    row = next((m for m in rows if m.get("id") == model), None)
    if row is None:
        raise SystemExit(f"/v1/models lists no {model!r} (ids: {[m.get('id') for m in rows]})")
    if not isinstance(row.get("context_length"), int):
        raise SystemExit(f"/v1/models {model!r} carries no context_length: {json.dumps(row)[:400]}")
    return row


def pi_models(src: dict, card: dict, model: str = PI_MODEL_ID) -> tuple[dict, dict]:
    """The host test's models.json with the model entry's window, output
    ceiling and inputs taken from the proxy's advertised card (never a stored
    number: full.json's 132096 / 26419 were the window of an earlier deploy).
    Returns (config, record of what changed)."""
    import copy
    cfg = copy.deepcopy(src)
    rec: dict = {"card": {k: card.get(k) for k in ("id", "context_length", "max_completion_tokens",
                                                     "modalities")}}
    found = 0
    for prov in (cfg.get("providers") or {}).values():
        for m in prov.get("models") or []:
            if m.get("id") != model:
                continue
            found += 1
            before = {k: m.get(k) for k in ("contextWindow", "maxTokens", "input")}
            m["contextWindow"] = int(card["context_length"])
            if isinstance(card.get("max_completion_tokens"), int):
                m["maxTokens"] = int(card["max_completion_tokens"])
            inputs = (card.get("modalities") or {}).get("input")
            if isinstance(inputs, list) and inputs:
                m["input"] = list(inputs)
            rec.setdefault("changed", []).append(
                {"before": before, "after": {k: m.get(k) for k in ("contextWindow", "maxTokens", "input")}})
    if found != 1:
        raise SystemExit(f"the Pi models.json has {found} entries for {model!r} (need exactly 1)")
    return cfg, rec


# ------------------------------------------------------------------ proxy side
def _xy(row: dict) -> dict:
    return ((row.get("response") or {}).get("x_yamadori")) or {}


def relay_summary(log_dir: str, sfx: str = "") -> dict:
    """What the proxy did over the run, from the relay's rows."""
    path = os.path.join(log_dir, f"relay{sfx}.jsonl")
    rows = []
    try:
        for ln in open(path, encoding="utf-8", errors="replace"):
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    except OSError:
        return {"error": f"no relay rows at {path}"}
    gens = [r for r in rows if r.get("method") == "POST"
            and str(r.get("path", "")).endswith(("/responses", "/chat/completions"))]
    routes, tools_ours, client_calls, status = Counter(), Counter(), Counter(), Counter()
    skills, deep_runs, think, withheld, markup, landings = [], [], [], None, [], 0
    tok = Counter()
    utility = Counter()
    efforts = Counter()          # what the harness sent (a clamp would show here)
    for i, r in enumerate(gens):
        status[str(r.get("status"))] += 1
        efforts[str((r.get("request") or {}).get("reasoning_effort"))] += 1
        resp = r.get("response") or {}
        for n in resp.get("tool_calls") or []:
            client_calls[n] += 1
        x = _xy(r)
        if not x:
            continue
        ut = x.get("utility")
        if (ut.get("utility") if isinstance(ut, dict) else bool(ut)):
            utility[x.get("utility_kind") or "utility"] += 1
        routes[(x.get("route") or {}).get("class")] += 1
        for t in x.get("tools") or []:
            tools_ours[t.get("name")] += 1
        if withheld is None and x.get("tools_withheld") is not None:
            withheld = x.get("tools_withheld")
        sk = x.get("skills") or {}
        if sk.get("ids") and not sk.get("replayed"):
            skills.append({"request": i, "kind": sk.get("kind"),
                           "route": (x.get("route") or {}).get("class"),
                           "skills": [{k: m.get(k) for k in ("name", "trigger", "slot",
                                                              "decided_by", "tokens")}
                                      for m in sk.get("matched") or []]
                           or [{"name": n} for n in sk.get("names") or []],
                           "tokens": sk.get("tokens")})
        d = x.get("deep") or {}
        if d.get("fire"):
            deep_runs.append({"request": i, "kind": d.get("kind"), "job": d.get("job")})
        for c in ((d.get("think_tool") or {}).get("calls") or []):
            think.append({"request": i, "ran": c.get("ran"), "ok": c.get("ok"),
                          "seconds": c.get("seconds"), "hops": c.get("hops")})
        if (x.get("tool_turns") or {}).get("hit"):
            landings += 1
        if x.get("tool_markup"):
            markup.append({"request": i, **x["tool_markup"]})
        u = x.get("usage") or {}
        s = u.get("summed") or {}
        for k in ("prompt_tokens", "completion_tokens"):
            tok[k] += int(s.get(k) or 0)
        for g in u.get("per_generation") or []:
            tok["cached_tokens"] += int(g.get("cached_tokens") or 0)
        tok["generations"] += int(u.get("generations") or 0)
    first = _xy(gens[0]) if gens else {}
    return {"requests": len(gens), "status": dict(status), "efforts_sent": dict(efforts),
            "first_route": (first.get("route") or {}).get("class"),
            "first_route_because": (first.get("route") or {}).get("because"),
            "routes": dict(routes), "utility": dict(utility),
            "tools_withheld": withheld, "craft": first.get("craft"),
            "skills_injections": skills, "skills_injected": len(skills),
            "deep_runs": deep_runs, "think_deeply": think,
            "our_tools": dict(tools_ours), "client_calls": dict(client_calls),
            "images": sum(len(_xy(r).get("images") or []) for r in gens),
            "tool_turn_cap_hits": landings, "tool_markup": markup,
            "tokens_main": dict(tok)}


# ------------------------------------------------------------------ the app
def readme_run(proj: str) -> dict:
    """What the project's README says about running it (recorded only)."""
    for fn in ("README.md", "readme.md", "README.MD", "README"):
        p = os.path.join(proj, fn)
        if os.path.isfile(p):
            text = G.read(p)
            lines = [ln.strip() for ln in text.splitlines()
                     if re.search(r"\b(npm|pnpm|yarn|npx|vite|http\.server|serve|"
                                  r"localhost|port)\b", ln, re.I)]
            return {"file": fn, "mentions_3001": "3001" in text,
                    "ports": sorted(set(re.findall(r"(?:localhost|127\.0\.0\.1|port)"
                                                   r"[:\s]*(\d{4,5})", text, re.I))),
                    "run_lines": lines[:8]}
    return {"file": None}


def _wait_200(name: str, tries: int = 90) -> bool:
    for _ in range(tries):
        r = G.docker(["exec", name, "sh", "-c",
                      "curl -s -o /dev/null -w '%{http_code}' " + URL], timeout=60)
        if r.stdout.strip() == "200":
            return True
        time.sleep(2)
    return False


def measure(rid: str, run_dir: str, log_dir: str, run_row: dict | None = None,
            write: bool = True) -> dict:
    """Serve the model's project, check the page, check the stack, sum the
    proxy side. Never raises; a step that could not run is its `error`."""
    VC = voxel_check()
    t0 = time.time()
    row: dict = {"run_id": rid, "task": TASK, "measured_at": t0,
                 "prompt_sha256": prompt()[1], "run_dir": run_dir}
    if run_row:
        row.update({k: run_row.get(k) for k in ("wall_s", "exit", "killed_by_runner",
                                                  "session_id", "outcome", "why", "harness",
                                                  "effort", "tools")})
        row["stack_deltas"] = run_row.get("stack")
    try:
        row["proxy"] = relay_summary(log_dir)
    except Exception as e:                                       # noqa: BLE001
        row["proxy"] = {"error": f"{type(e).__name__}: {e}"}
    proj = G.project_root(run_dir, "V4") if os.path.isdir(run_dir) else None
    row["project"] = proj
    if proj is None:
        row["error"] = "no package.json or index.html anywhere in the run folder"
    else:
        row["project_rel"] = os.path.relpath(proj, run_dir)
        row["files"] = G.files_and_refs("V4", proj)
        row["readme_run"] = readme_run(proj)
        out = os.path.join(G.GRADE_DIR, rid)
        shots = os.path.join(out, "shots")
        app: dict = {}
        gate = None
        try:
            app = G.app_up("V4", proj, out, re.sub(r"[^a-z0-9-]", "", rid.lower()))
            if app.get("serve_mode") and _wait_200(app["container"]):
                row["served"] = {"ok": True, "mode": app["serve_mode"], "url": ":3001"}
                row["page"] = VC.check_served(app["container"], URL, shots, rid)
            else:
                row["served"] = {"ok": False, "stage": app.get("stage"),
                                 "server_log": G.read(os.path.join(out, "server.log"))[-800:]}
        except Exception as e:                                   # noqa: BLE001
            row["error"] = f"serve: {type(e).__name__}: {e}"[:400]
        finally:
            if app.get("container"):
                G.docker(["rm", "-f", app["container"]], timeout=120)
            if app.get("net_tag"):
                try:
                    gate = sandbox_net.down(app["net_tag"])
                except Exception as e:                           # noqa: BLE001
                    row["gate_error"] = str(e)[:200]
        row["build"] = {k: app.get(k) for k in ("mode", "stage", "serve_mode", "npm_ls",
                                                 "docker_rc", "error")}
        for step in ("npm_install", "tsc", "build"):
            if isinstance(app.get(step), dict):
                row["build"][step] = {k: app[step].get(k) for k in
                                      ("rc", "error_ts", "errors", "seconds", "tail")}
        if os.path.isfile(os.path.join(shots, "browser.json")):
            try:
                an = VC.analyse(shots, gate)
                row["summary"] = an.get("summary")
                row["shots"] = {n: s for n, s in (an.get("shots") or {}).items()}
                row["screenshots"] = sorted(os.path.join(shots, f) for f in os.listdir(shots)
                                            if f.endswith(".png"))
                b = json.load(open(os.path.join(shots, "browser.json"), encoding="utf-8"))
                row["console_errors"] = [c.get("text", "")[:300] for c in b.get("console") or []
                                         if c.get("type") == "error"][:20]
                row["page_errors"] = [e.get("text", "")[:300]
                                      for e in b.get("page_errors") or []][:20]
            except Exception as e:                               # noqa: BLE001
                row["analysis_error"] = f"{type(e).__name__}: {e}"[:400]
        try:
            row["static"] = VC.stack_summary(VC.stack_static(proj, app))
        except Exception as e:                                   # noqa: BLE001
            row["static"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    row["measure_s"] = round(time.time() - t0, 1)
    if write:
        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if os.path.isdir(log_dir):
            with open(os.path.join(log_dir, "measure.json"), "w", encoding="utf-8") as f:
                json.dump(row, f, indent=1, ensure_ascii=False)
    return row


def brief(row: dict) -> dict:
    """The lines worth printing."""
    s = row.get("summary") or {}
    p = row.get("proxy") or {}
    return {"run_id": row.get("run_id"), "project": row.get("project_rel"),
            "harness": row.get("harness"), "wall_s": row.get("wall_s"), "exit": row.get("exit"),
            "efforts_sent": p.get("efforts_sent"),
            "served": row.get("served"), "build": {k: (row.get("build") or {}).get(k)
                                                   for k in ("mode", "stage")},
            "nonblank": s.get("nonblank"), "voxels": s.get("voxels"),
            "fps": s.get("fps"), "page_errors": s.get("page_errors"),
            "console_errors": s.get("console_errors"), "orbit": s.get("orbit"),
            "stack": (row.get("static") or {}).get("checks"),
            "first_route": p.get("first_route"), "routes": p.get("routes"),
            "tools_withheld": p.get("tools_withheld"), "skills_injected": p.get("skills_injected"),
            "deep_runs": p.get("deep_runs"), "think_deeply": p.get("think_deeply"),
            "our_tools": p.get("our_tools"), "client_calls": p.get("client_calls"),
            "tokens_main": p.get("tokens_main"), "screenshots": row.get("screenshots"),
            "error": row.get("error")}


if __name__ == "__main__":
    print(json.dumps({"prompt_sha256": prompt()[1], "prompt": prompt()[0]}, indent=1))
