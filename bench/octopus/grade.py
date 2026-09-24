#!/usr/bin/env python
"""Deterministic grader for Octopus Invaders runs. No model is ever asked.

    python bench/octopus/grade.py pilot-V0-xhigh-1            # a Hermes run
    python bench/octopus/grade.py --fixture C:/Users/jwals/octo/fixtures/x --variant V1 --id fx-v1
    python bench/octopus/grade.py RUN_ID --no-runtime         # skip the browser

Use the stack interpreter (tree-sitter-language-pack, PIL, numpy). Builds and
the browser run in Docker (Docker Desktop), never on the host: npm install runs code
the model chose (lifecycle scripts), and the game runs in Chromium.

THE FIVE PARTS (every check says HOW it was verified: `runtime`, `static-parse`
(tree-sitter over the project's JS/TS), `static-text` (flat text: README,
package.json), `pixel` (PIL over a screenshot, a heuristic with its
threshold stated here)):

1. completion   how the Hermes session ended (finished / turn_cap / budget /
                error / not_run), model calls, wall time -- from Hermes'
                stream-json, its exit code, and the relay's rows.
2. build        files per the variant's structure; TS variants: npm install,
                tsc --noEmit, npm run build (exit codes, `error TS` counts),
                in a COPY of the project (never the model's tree), in the
                pinned node image with network (npm needs it). V0: every
                script index.html references exists.
3. runtime      bench/octopus/browser_check.py in the pinned Playwright
                image: 15 s idle (console errors), start screen, click,
                mousemove, ESC, 30 s of play (errors, FPS), a game-over
                attempt, screenshots. Served from dist/ (a static server) when
                the build produced one, else from `vite` dev (labelled).
                WebGPU: whatever the page asked getContext() for is recorded
                (the container has no GPU; three's WebGPURenderer falls back
                to WebGL2 on SwiftShader, so FPS is software-rendered).
4. spec         a FIXED checklist (SPEC below) from the prompt's own
                requirements, each labelled by method.
5. stack        per request, from the relay's rows (x_yamadori verbatim):
                tier, route classes, tool_code repairs, deep thinking,
                fan-out, compactions, cache reused vs processed, tool turns,
                energy; and run.py's ledger deltas (tokens by account, GPU Wh).

OUTPUT: bench/octopus/results/grades.jsonl (one row per grade; a re-grade
appends), screenshots copied to index/octopus/grades/<id>/ (gitignored).

A 429 is "not run", never a failure. An exception in any part is recorded
as that part's `error`, never as a failed check (PROTOCOL rule 3).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import run as runmod        # noqa: E402  (IMAGE, OCTO, RUNS_DIR, LOGS_DIR)
import variants             # noqa: E402

RESULTS = os.path.join(HERE, "results")
GRADES = os.path.join(RESULTS, "grades.jsonl")
SHOTS = os.path.join(ROOT, "index", "octopus", "grades")
GRADE_DIR = os.path.join(runmod.OCTO, "grade")
# mcr.microsoft.com/playwright/python v1.63.0-noble (pinned by digest in
# playwright.Dockerfile) ships the browsers but not the Python package, so a
# one-line derived image adds playwright==1.63.0. Built on first use.
PW_IMAGE = "octo-playwright:1.63.0"
PW_DOCKERFILE = os.path.join(HERE, "playwright.Dockerfile")


def ensure_pw_image() -> None:
    if docker(["image", "inspect", PW_IMAGE], timeout=60).returncode == 0:
        return
    r = docker(["build", "-q", "-t", PW_IMAGE, "-f", PW_DOCKERFILE, HERE], timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("could not build " + PW_IMAGE + ": " + r.stderr[-400:])

EXPECTED = {
    "V0": ["index.html", "README.md", "css/styles.css", "js/config.js", "js/game.js",
           "js/player.js", "js/enemies.js", "js/particles.js", "js/background.js",
           "js/ui.js", "js/audio.js"],
    "V1": ["index.html", "package.json", "tsconfig.json", "vite.config.ts", "README.md",
           "src/main.tsx", "src/App.tsx", "src/styles.css", "src/config.ts", "src/game.ts",
           "src/player.ts", "src/enemies.ts", "src/particles.ts", "src/background.ts",
           "src/audio.ts", "src/components/Scene.tsx", "src/components/Hud.tsx"],
    "V2": ["index.html", "package.json", "tsconfig.json", "README.md", "src/main.ts",
           "src/styles.css", "src/config.ts", "src/textures.ts", "src/game.ts",
           "src/player.ts", "src/enemies.ts", "src/particles.ts", "src/background.ts",
           "src/ui.ts", "src/audio.ts"],
}
EXPECTED["V3"] = EXPECTED["V1"] + ["public/fonts/JetBrainsMono-Regular.ttf"]

# ------------------------------------------------------------------ helpers


def docker(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def read(path: str, limit: int | None = None) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit) if limit else f.read()
    except OSError:
        return ""


def jsonl(path: str) -> list[dict]:
    out = []
    for ln in read(path).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            out.append({"_unparsed": ln[:300]})
    return out


def check(name: str, ok, method: str, evidence=None) -> dict:
    return {"check": name, "pass": (None if ok is None else bool(ok)),
            "method": method, "evidence": evidence}


# --------------------------------------------------------------- 1 completion


def completion(log_dir: str, sfx: str = "") -> dict:
    try:
        meta = json.load(open(os.path.join(log_dir, f"meta{sfx}.json"), encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    events = jsonl(os.path.join(log_dir, f"hermes{sfx}.jsonl"))
    err = read(os.path.join(log_dir, f"hermes{sfx}.err"))
    relay = jsonl(os.path.join(log_dir, f"relay{sfx}.jsonl"))
    chats = [r for r in relay if r.get("path", "").endswith("/chat/completions")
             and r.get("method") == "POST"]
    statuses = {}
    for r in chats:
        statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1
    exit_code = meta.get("exit")
    wall = None
    if meta.get("t_start") and meta.get("t_end"):
        wall = round(float(meta["t_end"]) - float(meta["t_start"]), 1)
    types = {}
    for e in events:
        t = str(e.get("type") or e.get("event") or "?")
        types[t] = types.get(t, 0) + 1
    final = next((e for e in reversed(events) if isinstance(e, dict)
                  and (e.get("type") or e.get("event")) in
                  ("result", "final", "done", "session_end", "end")), None)
    blob = (json.dumps(final) if final else "") + "\n" + err[-6000:]
    low = blob.lower()
    if statuses.get("429") and not any(s == "200" for s in statuses):
        outcome = "not_run"
    elif meta.get("stopped_by_operator"):
        outcome = "stopped_by_operator"   # not a model result (meta.stop_note)
    elif meta.get("killed_by_runner"):
        outcome = "budget"             # run.py killed it at run_budget + 900 s
    elif re.search(r"max[_ -]?turns|maximum (tool[- ]calling )?iterations|turn (cap|limit)",
                   low):
        outcome = "turn_cap"
    elif re.search(r"run[_ -]?budget|wall[- ]clock budget|budget (exceeded|reached)", low):
        outcome = "budget"
    elif exit_code == 0 and final is not None and not re.search(
            r'"(error|is_error)"\s*:\s*true|"subtype"\s*:\s*"error', blob):
        outcome = "finished"
    elif exit_code == 0 and final is None and events:
        outcome = "finished_unconfirmed"
    else:
        outcome = "error"
    ut = [r for r in chats if ((r.get("response") or {}).get("x_yamadori") or {})
          .get("utility")]
    return {"outcome": outcome, "exit": exit_code, "wall_s": wall,
            "model_calls": len(chats), "utility_calls": len(ut),
            "http_status": statuses, "event_types": types,
            "final_event": (final if final and len(json.dumps(final)) < 4000 else
                            (json.dumps(final)[:4000] if final else None)),
            "stderr_tail": err[-1500:],
            "t_start": meta.get("t_start"), "session_id": meta.get("session_id"),
            "killed_by_runner": meta.get("killed_by_runner")}


# -------------------------------------------------------------------- 5 stack


def stack(log_dir: str, run_row: dict | None, sfx: str = "") -> dict:
    relay = jsonl(os.path.join(log_dir, f"relay{sfx}.jsonl"))
    chats = [r for r in relay if r.get("path", "").endswith("/chat/completions")]
    tiers, routes, efforts = {}, {}, {}
    cache = {"prompt": 0, "reused": 0, "processed": 0, "requests": 0}
    wh = 0.0
    wh_n = 0
    tool_code = {"requests_checked": 0, "files": 0, "errors_before": 0,
                 "errors_after": 0, "fixed": 0, "rounds": 0, "stopped": {},
                 "unknown_calls": 0}
    deep = {"ran": 0, "skipped": 0, "searches": 0, "facts": 0, "unverified": 0}
    fan = {"requests": 0, "ran": 0}
    comp = []
    turns = {"hit": 0, "max_turns": 0}
    finishes = {}
    first_tier = None
    for r in chats:
        x = ((r.get("response") or {}).get("x_yamadori")) or {}
        req = r.get("request") or {}
        e = str(req.get("reasoning_effort"))
        efforts[e] = efforts.get(e, 0) + 1
        fr = str((r.get("response") or {}).get("finish_reason"))
        finishes[fr] = finishes.get(fr, 0) + 1
        if not x:
            continue
        t = str(x.get("tier"))
        tiers[t] = tiers.get(t, 0) + 1
        if first_tier is None:
            first_tier = {"tier": x.get("tier"), "effort_sent": x.get("effort_sent"),
                          "tier_requested": x.get("tier_requested"),
                          "request_effort": req.get("reasoning_effort")}
        rc = str((x.get("route") or {}).get("class"))
        routes[rc] = routes.get(rc, 0) + 1
        c = x.get("cache") or {}
        if c.get("prompt") is not None:
            cache["requests"] += 1
            for k in ("prompt", "reused", "processed"):
                cache[k] += int(c.get(k) or 0)
        en = x.get("energy") or {}
        if en.get("wh") is not None:
            wh += float(en["wh"])
            wh_n += 1
        tc = x.get("tool_code")
        if tc:
            tool_code["requests_checked"] += 1
            tool_code["files"] += len(tc.get("files") or [])
            tool_code["errors_before"] += int(tc.get("errors_before") or 0)
            tool_code["errors_after"] += int(tc.get("errors_after") or 0)
            tool_code["rounds"] += int(tc.get("rounds") or 0)
            s = str(tc.get("stopped"))
            tool_code["stopped"][s] = tool_code["stopped"].get(s, 0) + 1
            if s == "fixed":
                tool_code["fixed"] += 1
            tool_code["unknown_calls"] += len(tc.get("unknown") or [])
        inv = x.get("investigate")
        if isinstance(inv, dict) and inv:
            if inv.get("ran"):
                deep["ran"] += 1
                deep["searches"] += int(inv.get("searches") or 0)
                h = inv.get("handoff") or {}
                deep["facts"] += int(h.get("facts") or 0)
                deep["unverified"] += int(h.get("unverified") or 0)
            elif inv.get("why"):
                deep["skipped"] += 1
        f = x.get("fanout")
        if f:
            fan["requests"] += 1
            if isinstance(f, dict) and (f.get("n") or 0) > 1:
                fan["ran"] += 1
        if x.get("compaction") or x.get("utility_kind") == "compaction":
            cc = x.get("compaction") or {}
            comp.append({"t": r.get("t0"), "mode": cc.get("mode"), "shape": cc.get("shape"),
                         "answer": cc.get("answer"), "thinking": cc.get("thinking"),
                         "finish": (r.get("response") or {}).get("finish_reason"),
                         "cache": x.get("cache")})
        tt = x.get("tool_turns") or {}
        if tt.get("hit"):
            turns["hit"] += 1
        turns["max_turns"] = max(turns["max_turns"], int(tt.get("turns") or 0))
    comp_tok = sum(int(((r.get("response") or {}).get("usage") or {}).get("completion_tokens")
                       or 0) for r in chats)
    peak = max((int(((r.get("response") or {}).get("usage") or {}).get("prompt_tokens") or 0)
                for r in chats), default=0)
    out = {"requests": len(chats), "completion_tokens": comp_tok, "peak_prompt_tokens": peak,
           "tiers": tiers, "first_request": first_tier,
           "efforts_sent": efforts, "routes": routes, "finish_reasons": finishes,
           "cache": {**cache, "reuse_pct": (round(100 * cache["reused"] / cache["prompt"], 1)
                                            if cache["prompt"] else None)},
           "energy_wh_requests": round(wh, 2), "energy_requests_with_wh": wh_n,
           "tool_code": tool_code, "deep_thinking": deep, "fanout": fan,
           "compactions": comp, "proxy_tool_turns": turns}
    if run_row:
        out["ledger"] = run_row.get("stack")
    return out


# ------------------------------------------------------------ 2 build + 3 runtime


def project_root(run_dir: str) -> str | None:
    """The directory holding the game: space-shooter/ if present, else the
    shallowest dir with an index.html (outside node_modules/dist)."""
    cand = os.path.join(run_dir, "space-shooter")
    if os.path.isfile(os.path.join(cand, "index.html")):
        return cand
    best = None
    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".git")]
        depth = dirpath[len(run_dir):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        if "index.html" in filenames and (best is None or depth < best[0]):
            best = (depth, dirpath)
    return best[1] if best else None


def files_and_refs(variant: str, proj: str) -> dict:
    res: dict = {"project": proj}
    present = {f: os.path.isfile(os.path.join(proj, *f.split("/"))) for f in EXPECTED[variant]}
    res["files_expected"] = len(present)
    res["files_present"] = sum(present.values())
    res["files_missing"] = [f for f, ok in present.items() if not ok]
    src_files = []
    for dirpath, dirnames, filenames in os.walk(proj):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".git")]
        src_files += [os.path.join(dirpath, fn) for fn in filenames
                      if fn.endswith((".js", ".ts", ".tsx", ".jsx", ".mjs"))]
    res["source_files"] = len(src_files)
    res["source_lines"] = sum(read(p).count("\n") for p in src_files)
    if not variants.VARIANTS[variant]["ts"]:
        html = read(os.path.join(proj, "index.html"))
        refs = re.findall(r"""<(?:script[^>]*\bsrc|link[^>]*\bhref)\s*=\s*["']([^"']+)["']""",
                          html, flags=re.I)
        local = [r for r in refs if not re.match(r"^(https?:)?//", r)]
        clean = [re.split(r"[?#]", r)[0] for r in local]
        res["html_refs"] = clean
        res["html_refs_missing"] = [r for r in clean if not os.path.isfile(
            os.path.join(proj, *r.lstrip("./").split("/")))]
        res["external_refs"] = [r for r in refs if r not in local]
        res["serve_ok"] = bool(clean) and not res["html_refs_missing"]
    return res


def app_up(variant: str, proj: str, out: str, gid: str) -> dict:
    """Start the app container (serve_app.sh): builds a copy, then serves
    :3001. Returns build results; the container keeps running for runtime()."""
    ts = variants.VARIANTS[variant]["ts"]
    name = f"octo-app-{gid}"[:60]
    docker(["rm", "-f", name], timeout=60)
    for fn in os.listdir(out) if os.path.isdir(out) else []:
        p = os.path.join(out, fn)
        if os.path.isfile(p):
            os.remove(p)
    os.makedirs(out, exist_ok=True)
    r = docker(["run", "-d", "--name", name, "--cpus", "4", "--memory", "6g",
                "-v", f"{proj}:/src:ro", "-v", f"{out}:/out", "-v", f"{HERE}:/grader:ro",
                runmod.IMAGE, "bash", "/grader/serve_app.sh", "ts" if ts else "js"],
               timeout=300)
    res: dict = {"container": name, "docker_rc": r.returncode}
    if r.returncode != 0:
        res["error"] = (r.stderr or r.stdout)[-600:]
        return res
    stage = ""
    t_end = time.time() + 45 * 60
    while time.time() < t_end:
        stage = read(os.path.join(out, "stage")).strip()
        if stage.startswith("serving") or stage == "failed":
            break
        st = docker(["inspect", "-f", "{{.State.Running}}", name], timeout=60).stdout.strip()
        if st != "true":
            stage = stage or "container exited"
            break
        time.sleep(5)
    res["stage"] = stage
    if ts:
        def rc(step):
            t = read(os.path.join(out, f"{step}.rc")).strip()
            return int(t) if t.lstrip("-").isdigit() else None
        for step in ("npm_install", "tsc", "build"):
            log = read(os.path.join(out, f"{step}.log"))
            res[step] = {"rc": rc(step),
                         "error_ts": len(re.findall(r"error TS\d+", log)),
                         "errors": len(re.findall(r"(?im)^\s*(npm )?(ERR!|error)\b", log)),
                         "tail": log[-1500:]}
        for step in ("npm_install", "build"):
            sec = read(os.path.join(out, f"{step}.s")).strip()
            res[step]["seconds"] = int(sec) if sec.isdigit() else None
        try:
            res["npm_ls"] = {k: v.get("version") for k, v in json.load(open(
                os.path.join(out, "npm_ls.json"))).get("dependencies", {}).items()}
        except (OSError, ValueError, AttributeError):
            res["npm_ls"] = None
    res["serve_mode"] = {"serving static": "static", "serving dev": "vite dev"}.get(stage)
    return res


def runtime(app: dict, out: str) -> dict:
    if not app.get("serve_mode"):
        return {"ran": False, "why": f"nothing to serve (stage: {app.get('stage')})"}
    name = app["container"]
    ok = False
    for _ in range(90):
        r = docker(["exec", name, "sh", "-c",
                    "curl -s -o /dev/null -w '%{http_code}' http://localhost:3001/"], timeout=60)
        if r.stdout.strip() == "200":
            ok = True
            break
        time.sleep(2)
    if not ok:
        return {"ran": False, "why": "server never answered 200 on :3001",
                "server_log": read(os.path.join(out, "server.log"))[-800:]}
    shots = os.path.join(out, "shots")
    os.makedirs(shots, exist_ok=True)
    ensure_pw_image()
    r = docker(["run", "--rm", "--network", f"container:{name}", "--ipc=host",
                "-v", f"{HERE}:/grader:ro", "-v", f"{shots}:/out", PW_IMAGE,
                "python3", "/grader/browser_check.py", "--url", "http://localhost:3001/",
                "--out", "/out"], timeout=900)
    res = {"ran": True, "mode": app["serve_mode"], "browser_rc": r.returncode,
           "browser_out": (r.stdout + r.stderr)[-1500:]}
    bj = os.path.join(shots, "browser.json")
    if not os.path.isfile(bj):
        res["error"] = "browser_check produced no browser.json"
        return res
    res["browser"] = json.load(open(bj, encoding="utf-8"))
    res["shots_dir"] = shots
    return res



# --------------------------------------------------------------- image math


def _img(path):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype("int16")


def img_stats(path) -> dict:
    import numpy as np
    a = _img(path)
    flat = a.reshape(-1, 3)
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    mode = vals[counts.argmax()]
    off = (np.abs(flat - mode).sum(axis=1) > 24).mean()
    return {"std": round(float(a.std()), 2), "mode_rgb": [int(x) for x in mode],
            "non_mode_frac": round(float(off), 4)}


def meanabs(p1, p2, band=None) -> float:
    import numpy as np
    a, b = _img(p1), _img(p2)
    if band:
        h = a.shape[0]
        a, b = a[int(h * band[0]):int(h * band[1])], b[int(h * band[0]):int(h * band[1])]
    return round(float(np.abs(a - b).mean()), 3)


def ship_shift(left, right) -> dict:
    """Bright pixels that APPEARED on the right and DISAPPEARED from the left
    between the two shots, in the bottom 40%: their centroids should be far
    apart (a ship that follows the mouse), while stars scrolling down add
    symmetric noise. Threshold: centroid gap > 25% of the width, both masses
    > 80 px."""
    import numpy as np
    a, b = _img(left), _img(right)
    h, w = a.shape[:2]
    a, b = a[int(h * 0.6):], b[int(h * 0.6):]
    la, lb = a.max(axis=2) > 90, b.max(axis=2) > 90
    appeared, gone = lb & ~la, la & ~lb
    out = {"appeared_px": int(appeared.sum()), "gone_px": int(gone.sum())}
    if appeared.sum() and gone.sum():
        xa = float(np.nonzero(appeared)[1].mean()) / w
        xg = float(np.nonzero(gone)[1].mean()) / w
        out.update(appeared_cx=round(xa, 3), gone_cx=round(xg, 3), gap=round(xa - xg, 3))
    # Second signal: the ship's spec color. Tier 1 is cyan-edged (#4ECDC4),
    # so the centroid of cyan pixels in the bottom band follows it. Planets
    # and nebulae scroll through the band too, which is what defeated the
    # first signal on the V0 reference fixture (ship visibly moved 240 ->
    # 1035 px, bright-pixel gap 0.12).
    def cyan_cx(img):
        r, g, bl = img[..., 0], img[..., 1], img[..., 2]
        m = (g > 150) & (bl > 140) & (r < 150) & (abs(g - bl) < 70)
        return (float(np.nonzero(m)[1].mean()) / w, int(m.sum())) if m.sum() else (None, 0)
    ca, na = cyan_cx(a)
    cb, nb = cyan_cx(b)
    out.update(cyan_left_cx=None if ca is None else round(ca, 3), cyan_left_px=na,
               cyan_right_cx=None if cb is None else round(cb, 3), cyan_right_px=nb)
    bright = bool(out.get("gap", 0) > 0.25 and out["appeared_px"] > 80 and out["gone_px"] > 80)
    cyan = bool(ca is not None and cb is not None and na > 30 and nb > 30 and cb - ca > 0.25)
    out["pass"] = bright or cyan
    out["rule"] = ("mouse at x=19% then 81% of the width, bottom band: bright-pixel "
                   "appeared/gone centroid gap > 0.25, or cyan (#4ECDC4-ish) centroid "
                   "shift > 0.25")
    return out


# ------------------------------------------------------------ static parse


class Src:
    """Everything the spec checks read from the project's JS/TS, by parse."""

    def __init__(self, proj_unc: str):
        from tree_sitter_language_pack import get_parser
        self.strings: list[str] = []
        self.numbers: list[float] = []
        self.calls: list[str] = []
        self.imports: list[str] = []
        self.idents: set[str] = set()
        self.assign: list[tuple[str, str]] = []
        self.mod5: list[str] = []
        self.const5: list[str] = []
        self.grids = 0
        self.speed_array = False
        self.text = ""
        self.files: list[str] = []
        self.parse_errors = 0
        for dirpath, dirnames, filenames in os.walk(proj_unc):
            dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".git")]
            for fn in filenames:
                ext = os.path.splitext(fn)[1]
                lang = {".js": "javascript", ".mjs": "javascript", ".jsx": "javascript",
                        ".ts": "typescript", ".tsx": "tsx"}.get(ext)
                if not lang or fn.endswith(".d.ts"):
                    continue
                path = os.path.join(dirpath, fn)
                code = read(path)
                self.files.append(os.path.relpath(path, proj_unc).replace("\\", "/"))
                self.text += "\n" + code
                tree = get_parser(lang).parse(code.encode("utf-8"))
                if tree.root_node.has_error:
                    self.parse_errors += 1
                self._walk(tree.root_node, code.encode("utf-8"))

    def _t(self, n, src) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def _walk(self, root, src):
        stack = [root]
        while stack:
            n = stack.pop()
            t = n.type
            if t in ("string", "template_string"):
                self.strings.append(self._t(n, src)[1:-1])
            elif t == "number":
                try:
                    self.numbers.append(float(self._t(n, src).replace("_", "")))
                except ValueError:
                    pass
            elif t == "call_expression":
                f = n.child_by_field_name("function")
                if f is not None:
                    self.calls.append(self._t(f, src))
            elif t == "import_statement":
                s = n.child_by_field_name("source")
                if s is not None:
                    self.imports.append(self._t(s, src).strip("'\""))
            elif t in ("identifier", "property_identifier", "type_identifier",
                       "shorthand_property_identifier"):
                self.idents.add(self._t(n, src))
            elif t == "assignment_expression":
                l, r = n.child_by_field_name("left"), n.child_by_field_name("right")
                if l is not None and r is not None:
                    self.assign.append((self._t(l, src), self._t(r, src)))
            elif t == "binary_expression":
                op = n.child_by_field_name("operator")
                r = n.child_by_field_name("right")
                if op is not None and self._t(op, src) == "%" and r is not None \
                        and self._t(r, src).strip() == "5":
                    self.mod5.append(self._t(n, src)[:80])
            elif t in ("variable_declarator", "pair", "public_field_definition"):
                k = n.child_by_field_name("name") or n.child_by_field_name("key")
                v = n.child_by_field_name("value")
                if k is not None and v is not None and self._t(v, src).strip() == "5" \
                        and re.search(r"boss", self._t(k, src), re.I):
                    self.const5.append(self._t(k, src))
            elif t == "array":
                kids = [c for c in n.named_children]
                vals = [self._t(c, src) for c in kids]
                if len(vals) >= 8 and all(v in ("0", "1") for v in vals):
                    self.grids += 1
                elif kids and all(c.type == "string" for c in kids):
                    rows = [v.strip("'\"`") for v in vals]
                    if len(rows) >= 6 and all(len(r) >= 6 and set(r) <= set("01.#X ")
                                              for r in rows):
                        self.grids += 1
                self._speeds(kids, src)
            elif t == "object":
                vals = []
                for c in n.named_children:
                    if c.type == "pair" and c.child_by_field_name("value") is not None:
                        vals.append(c.child_by_field_name("value"))
                self._speeds(vals, src)
            stack.extend(n.children)

    def _speeds(self, kids, src) -> None:
        """Speeds 8, 10, 12, 14 in order as the elements (array) or values
        (object) of one literal, bare or as `speed: N` inside each."""
        nums = []
        for c in kids:
            if c.type == "number":
                try:
                    nums.append(int(float(self._t(c, src))))
                except ValueError:
                    pass
            else:
                m = re.search(r"speed\s*:\s*(\d+)", self._t(c, src))
                if m:
                    nums.append(int(m.group(1)))
        if [8, 10, 12, 14] == [x for x in nums if x in (8, 10, 12, 14)][:4]:
            self.speed_array = True

    def has_num(self, *xs) -> bool:
        return all(any(abs(n - x) < 1e-9 for n in self.numbers) for x in xs)

    def has_str(self, s: str) -> bool:
        s = s.lower()
        return any(s in x.lower() for x in self.strings)

    def has_ident(self, *names) -> bool:
        return all(n in self.idents for n in names)

    def imports_pkg(self, pkg: str) -> bool:
        return any(i == pkg or i.startswith(pkg + "/") for i in self.imports)


# -------------------------------------------------------------------- 4 spec


def spec(variant: str, pu: str, b: dict, rt: dict) -> list[dict]:
    s = Src(pu)
    br = (rt or {}).get("browser") or {}
    states = br.get("states") or {}
    shots = (rt or {}).get("shots_dir")
    ts = variants.VARIANTS[variant]["ts"]
    out: list[dict] = []

    def txt(label):
        st = states.get(label) or {}
        return ((st.get("dom_text") or "") + "\n" + "\n".join(st.get("fill_recent") or [])).upper()

    # --- game content (static-parse) ---
    out.append(check("enemy_sizes_36_48_20_150", s.has_num(36, 48, 20, 150), "static-parse",
                     "number literals 36, 48, 20, 150"))
    kinds = [k for k in ("small", "medium", "baby", "boss")
             if any(k in i.lower() for i in s.idents) or s.has_str(k)]
    out.append(check("enemy_types_4", len(kinds) == 4, "static-parse",
                     f"names found: {kinds}"))
    out.append(check("bullet_speeds_4_tiers", s.speed_array, "static-parse",
                     "an array literal holding speeds 8, 10, 12, 14 in order"))
    out.append(check("boss_every_5_levels", bool(s.mod5 or s.const5), "static-parse",
                     {"mod5": s.mod5[:3], "boss_const5": s.const5[:3]}))
    out.append(check("lerp_0_35", s.has_num(0.35), "static-parse", "number literal 0.35"))
    circle = bool(re.search(r"(\w+)\s*\*\s*\1\s*\+\s*(\w+)\s*\*\s*\2", s.text)
                  or "Math.hypot" in s.text)
    out.append(check("circle_collision", circle, "static-text",
                     "dx*dx + dy*dy or Math.hypot in source"))
    out.append(check("pixel_grids_ge_4", s.grids >= 4, "static-parse",
                     f"{s.grids} 0/1 grid rows or grids (>=8 cells) found"))
    if ts:
        hard = s.has_ident("NearestFilter")
        hard_ev = "identifier NearestFilter"
    else:
        hard = any(l.endswith("imageSmoothingEnabled") and r.strip() == "false"
                   for l, r in s.assign)
        hard_ev = "assignment *.imageSmoothingEnabled = false"
    out.append(check("hard_pixels", hard, "static-parse", hard_ev))
    out.append(check("bg_color_0D1117_source", s.has_str("0d1117"), "static-parse",
                     "string literal containing #0D1117"))
    out.append(check("unleash_mode", any("unleash" in i.lower() for i in s.idents)
                     or s.has_str("unleash"), "static-parse", "identifier/string 'unleash'"))
    out.append(check("combo_counter", s.has_str("combo"), "static-parse", "string 'COMBO'"))
    out.append(check("restart_text", s.has_str("click to restart"), "static-parse",
                     "string 'CLICK TO RESTART'"))
    webaudio_static = (any(c.endswith("createOscillator") for c in s.calls)
                       and ("AudioContext" in s.idents))
    out.append(check("web_audio_static", webaudio_static, "static-parse",
                     "AudioContext + createOscillator() calls"))
    readme = read(os.path.join(pu, "README.md"))
    out.append(check("readme_run_3001", "3001" in readme, "static-text", "README mentions 3001"))
    css_html = ""
    for dirpath, dirnames, filenames in os.walk(pu):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".git")]
        css_html += "".join(read(os.path.join(dirpath, f)) for f in filenames
                            if f.endswith((".css", ".html")))
    if variant == "V3":
        mono = os.path.isfile(os.path.join(pu, "public", "fonts", "JetBrainsMono-Regular.ttf"))
    else:
        mono = s.has_str("monospace") or "monospace" in css_html
    out.append(check("monospace_font", mono, "static-parse+text",
                     "'monospace' in a JS/TS string or any .css/.html (V3: the JetBrains "
                     "Mono TTF in public/fonts)"))

    # --- stack-specific ---
    if variant == "V0":
        out.append(check("particles_draw_ctx",
                         any(re.fullmatch(r"(this\.)?particle\w*\.draw", c) for c in s.calls),
                         "static-parse", "a call particle*.draw(...) (particles / particleSystem)"))
        ext = [i for i in s.imports if not i.startswith((".", "/"))]
        out.append(check("no_libraries", not ext and not b.get("external_refs"),
                         "static-parse", {"bare_imports": ext,
                                          "external_scripts": b.get("external_refs")}))
    if variant in ("V1", "V3"):
        out.append(check("r3f_canvas_useFrame",
                         s.imports_pkg("@react-three/fiber") and "useFrame" in s.idents
                         and "Canvas" in s.idents, "static-parse",
                         "imports @react-three/fiber; Canvas and useFrame used"))
        out.append(check("instanced_or_points",
                         bool({"instancedMesh", "InstancedMesh", "points", "Points"} & s.idents),
                         "static-parse", "instancedMesh or points in source"))
    if variant == "V2":
        out.append(check("three_flatland_sprites",
                         s.imports_pkg("three-flatland") and s.has_ident("Sprite2D")
                         and s.has_ident("SpriteGroup"), "static-parse",
                         "imports three-flatland; Sprite2D and SpriteGroup used"))
        out.append(check("webgpu_renderer", s.imports_pkg("three/webgpu")
                         and "WebGPURenderer" in s.idents, "static-parse",
                         "imports three/webgpu; WebGPURenderer used"))
    if variant == "V3":
        out.append(check("glyph_used", s.imports_pkg("@pmndrs/glyph"), "static-parse",
                         "imports @pmndrs/glyph (any subpath)"))
    if ts:
        try:
            pkg = json.load(open(os.path.join(pu, "package.json"), encoding="utf-8"))
        except (OSError, ValueError):
            pkg = {}
        want = variants.VERSIONS[variant]
        bad = []
        for sect in ("dependencies", "devDependencies"):
            have = pkg.get(sect) or {}
            for k, v in want[sect].items():
                got = have.get(k) or (pkg.get("dependencies", {}) | pkg.get(
                    "devDependencies", {})).get(k)
                if got != v:
                    bad.append(f"{k}: want {v}, got {got}")
        extra = sorted((set(pkg.get("dependencies", {})) | set(pkg.get("devDependencies", {})))
                       - set(want["dependencies"]) - set(want["devDependencies"]))
        out.append(check("package_pins_exact", pkg and not bad and not extra, "static-text",
                         {"mismatch": bad, "extra": extra}))

    # --- runtime ---
    if not rt or not rt.get("ran") or not br:
        for name in ("no_console_errors_15s", "canvas_not_blank", "start_screen",
                     "click_starts_game", "mousemove_moves_ship", "esc_pause",
                     "play_30s_no_errors", "web_audio_runtime", "hud_positions",
                     "bg_color_runtime"):
            out.append(check(name, None, "runtime", "not run: " + str((rt or {}).get("why")
                                                                  or (rt or {}).get("error"))))
        return out

    def is_noise(c):
        return "favicon.ico" in (c.get("text") or "")
    loaded_t = next((e["t"] for e in br.get("events", []) if e["name"] in
                     ("loaded", "load_failed")), 0)
    early = [c for c in br.get("console", []) if c["type"] == "error"
             and c["t"] <= loaded_t + 15 and not is_noise(c)]
    early += [e for e in br.get("page_errors", []) if e["t"] <= loaded_t + 15]
    out.append(check("no_console_errors_15s", not early and not br.get("load_error"),
                     "runtime", {"errors": [e.get("text") for e in early][:5],
                                 "load_error": br.get("load_error")}))
    p = lambda n: os.path.join(shots, f"{n}.png")                 # noqa: E731
    st = img_stats(p("start")) if os.path.isfile(p("start")) else {}
    out.append(check("canvas_not_blank", st.get("std", 0) >= 4 and
                     st.get("non_mode_frac", 0) >= 0.005, "pixel",
                     {**st, "threshold": "std >= 4 and >= 0.5% pixels off the mode color"}))
    bg = st.get("mode_rgb")
    out.append(check("bg_color_runtime", bg is not None and
                     sum(abs(x - y) for x, y in zip(bg, (13, 17, 23))) <= 24, "pixel",
                     {"mode_rgb": bg, "want": [13, 17, 23]}))
    t0 = txt("start")
    if variant == "V3":
        ok = (s.has_str("octopus invaders") and s.has_str("click to start")
              and st.get("std", 0) >= 4)
        out.append(check("start_screen", ok, "static-parse+pixel",
                         "glyph text is not in the DOM or 2D canvas: strings in source "
                         "and a non-blank start frame"))
    else:
        flat = re.sub(r"\s+", "", t0)
        out.append(check("start_screen", "OCTOPUSINVADERS" in flat and "CLICKTOSTART" in flat,
                         "runtime", {"rule": "title and CLICK TO START in DOM text or "
                                             "fillText, whitespace ignored",
                                     "title": "OCTOPUSINVADERS" in flat,
                                     "click_to_start": "CLICKTOSTART" in flat,
                                     "start_click": br.get("start_click")}))
    t1 = txt("after_click")
    d_click = meanabs(p("start"), p("after_click")) if os.path.isfile(p("after_click")) else None
    if variant == "V3":
        out.append(check("click_starts_game", d_click is not None and d_click > 2.0, "pixel",
                         {"meanabs_start_vs_after_click": d_click, "threshold": 2.0}))
    else:
        # The start screen's own text (title / CLICK TO START / START GAME) was
        # shown before the click and is gone after it. A HUD word is not
        # evidence: canvas games draw the HUD under the menu (seen on the V0
        # reference fixture).
        def marks(t):
            f = re.sub(r"\s+", "", t)
            return {m for m in ("OCTOPUSINVADERS", "CLICKTOSTART", "STARTGAME") if m in f}
        m0, m1 = marks(t0), marks(t1)
        started = bool(m0) and not m1
        out.append(check("click_starts_game", started, "runtime",
                         {"start_marks_before": sorted(m0), "start_marks_after": sorted(m1),
                          "start_click": br.get("start_click"),
                          "meanabs_start_vs_after_click": d_click}))
    if os.path.isfile(p("ship_left")) and os.path.isfile(p("ship_right")):
        sh = ship_shift(p("ship_left"), p("ship_right"))
        out.append(check("mousemove_moves_ship", sh["pass"], "pixel", sh))
    d_pause = meanabs(p("pause_a"), p("pause_b")) if os.path.isfile(p("pause_b")) else None
    d_res = meanabs(p("resumed_a"), p("resumed_b")) if os.path.isfile(p("resumed_b")) else None
    paused_txt = "PAUSE" in txt("paused")
    frozen = d_pause is not None and d_res is not None and d_pause < 0.3 and d_res >= 0.3
    out.append(check("esc_pause", frozen or paused_txt, "runtime+pixel",
                     {"meanabs_while_paused": d_pause, "meanabs_after_resume": d_res,
                      "pause_text": paused_txt,
                      "threshold": "paused < 0.3 and resumed >= 0.3, or PAUSE text"}))
    pe = br.get("play_new_errors") or {}
    out.append(check("play_30s_no_errors", pe.get("console_errors", 1) == 0
                     and pe.get("page_errors", 1) == 0, "runtime", pe))
    au = (states.get("end") or states.get("after_play") or {}).get("audio") or {}
    out.append(check("web_audio_runtime", (au.get("osc_start", 0) + au.get("buf_start", 0)) > 0,
                     "runtime", au))
    out.append(hud_check(variant, br, s))
    return out


def hud_check(variant: str, br: dict, s: "Src") -> dict:
    W, H = br.get("viewport") or (1280, 720)
    if variant == "V0":
        fills = br.get("fill_all") or []

        def find(word):
            return [f for f in fills if f["t"].upper().startswith(word)]
        sc, lv, co = find("SCORE"), find("LEVEL"), find("COMBO")
        ev = {"score": sc[:1], "level": lv[:1], "combo": co[:1]}

        def at_y40(f):
            return abs(f["y"] - 40) <= 2 and "20px" in f["font"] and "monospace" in f["font"]
        ok = (any(abs(f["x"] - 20) <= 2 and at_y40(f) for f in sc)
              and any(abs(f["x"] - f["cw"] / 2) <= 4 and f["align"] == "center" and at_y40(f)
                      for f in lv)
              and any(f["x"] >= f["cw"] * 0.7 and at_y40(f) for f in co))
        return check("hud_positions", ok, "runtime",
                     {**ev, "rule": "SCORE x=20 y=40, LEVEL centered y=40, COMBO right "
                                    "y=40, all 20px monospace (fillText args)"})
    if variant in ("V1", "V2"):
        d = br.get("dom_hud") or []

        def pick(word):
            return [e for e in d if e["t"].upper().startswith(word) and e["visible"]]
        sc, lv, co = pick("SCORE"), pick("LEVEL"), pick("COMBO")

        def y40(e):
            return e["y"] <= 40 <= e["y"] + e["h"] + 6 and e["font_size"] == "20px" \
                and "mono" in e["font_family"].lower()
        ok = (any(abs(e["x"] - 20) <= 12 and y40(e) for e in sc)
              and any(abs(e["x"] + e["w"] / 2 - W / 2) <= 40 and y40(e) for e in lv)
              and any(e["x"] + e["w"] >= W - 120 and y40(e) for e in co))
        return check("hud_positions", ok, "runtime",
                     {"score": sc[:1], "level": lv[:1], "combo": co[:1],
                      "rule": "DOM boxes: SCORE left 20, LEVEL centered, COMBO right, "
                              "y=40 inside the box, 20px monospace"})
    ok = s.has_str("score") and s.has_str("level") and s.has_num(40, 20)
    return check("hud_positions", ok, "static-parse",
                 "V3 text is glyph geometry: strings SCORE/LEVEL and numbers 20, 40 "
                 "in source only (positions not verified)")


# --------------------------------------------------------------------- main


def grade(gid: str, variant: str, run_dir: str, log_dir: str | None,
          do_runtime: bool, sfx: str = "") -> dict:
    """`gid` is `<run_id>` or, under the iterative protocol, `<run_id>@p<n>`
    (the state after prompt n); `sfx` names prompt n's log files ('' or
    '.p<n>', run.sfx_of)."""
    t0 = time.time()
    run_id, _, pn = gid.partition("@p")
    prompt_no = int(pn) if pn.isdigit() else 1
    row: dict = {"grade_id": gid, "variant": variant, "run_dir": run_dir,
                 "prompt_no": prompt_no, "graded_at": t0, "grader": "bench/octopus/grade.py"}
    run_row = None
    if os.path.isfile(runmod.RUNS):
        for r in jsonl(runmod.RUNS):
            if (r.get("run_id") == run_id and r.get("outcome") != "not_run"
                    and int(r.get("prompt_no") or 1) == prompt_no):
                run_row = r
    if log_dir:
        try:
            row["completion"] = completion(log_dir, sfx)
        except Exception as e:                             # noqa: BLE001
            row["completion"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            row["stack"] = stack(log_dir, run_row, sfx)
        except Exception as e:                             # noqa: BLE001
            row["stack"] = {"error": f"{type(e).__name__}: {e}"}
    proj = project_root(run_dir) if os.path.isdir(run_dir) else None
    row["project"] = proj
    out = os.path.join(GRADE_DIR, gid.replace("@", "_"))
    b, rt, app = {}, None, {}
    partial = os.path.join(run_dir, "space-shooter")
    if proj is None and os.path.isdir(partial):
        # A run that ended before index.html existed: nothing to build or
        # serve, but the files it wrote still get the static checks.
        row["project_partial"] = partial
        b = files_and_refs(variant, partial)
        row["build"] = {**b, "error": "no index.html: not built, not served (partial project)"}
        try:
            row["spec"] = spec(variant, partial, b, {"ran": False,
                                                    "why": "no index.html (partial project)"})
        except Exception as e:                             # noqa: BLE001
            row["spec"] = [{"check": "spec", "pass": None, "method": "error",
                            "evidence": f"{type(e).__name__}: {e}"}]
    elif proj is None:
        row["build"] = {"error": "no index.html anywhere in the run directory"}
    else:
        try:
            b = files_and_refs(variant, proj)
            app = app_up(variant, proj, out, re.sub(r"[^a-z0-9-]", "", gid.lower()))
            b.update(app)
            row["build"] = b
        except Exception as e:                             # noqa: BLE001
            row["build"] = {**b, "error": f"{type(e).__name__}: {e}"}
        if do_runtime:
            try:
                rt = runtime(app, out)
            except Exception as e:                         # noqa: BLE001
                rt = {"ran": False, "error": f"{type(e).__name__}: {e}"}
        if app.get("container"):
            docker(["rm", "-f", app["container"]], timeout=120)
        try:
            row["spec"] = spec(variant, proj, b, rt or {"ran": False, "why": "--no-runtime"})
        except Exception as e:                             # noqa: BLE001
            row["spec"] = [{"check": "spec", "pass": None, "method": "error",
                            "evidence": f"{type(e).__name__}: {e}"}]
    if rt:
        br = rt.pop("browser", None) or {}
        row["runtime"] = {**rt, **{k: br.get(k) for k in (
            "fps_idle", "fps_play", "webgpu_probe", "load_error", "gameover_seen_text",
            "browser_version", "play_new_errors")},
            "context_types": sorted(set(((br.get("states") or {}).get("end") or {})
                                        .get("ctx") or [])),
            "console_errors": [c for c in br.get("console", []) if c["type"] == "error"][:20],
            "page_errors": br.get("page_errors", [])[:20],
            "requests_failed": br.get("requests_failed", [])[:20]}
        if rt.get("shots_dir") and os.path.isdir(rt["shots_dir"]):
            dst = os.path.join(SHOTS, gid.replace("@", "_"))
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(rt["shots_dir"]):
                shutil.copy2(os.path.join(rt["shots_dir"], fn), os.path.join(dst, fn))
            row["runtime"]["screenshots"] = dst
    checks = row.get("spec") or []
    row["spec_passed"] = sum(1 for c in checks if c.get("pass") is True)
    row["spec_failed"] = sum(1 for c in checks if c.get("pass") is False)
    row["spec_unknown"] = sum(1 for c in checks if c.get("pass") is None)
    row["spec_total"] = len(checks)
    row["grade_s"] = round(time.time() - t0, 1)
    os.makedirs(RESULTS, exist_ok=True)
    with open(GRADES, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--variant", choices=list(variants.VARIANTS))
    ap.add_argument("--fixture", help="a directory to grade as if it were a run")
    ap.add_argument("--id")
    ap.add_argument("--no-runtime", action="store_true")
    a = ap.parse_args()
    if a.fixture:
        if not (a.variant and a.id):
            ap.error("--fixture needs --variant and --id")
        row = grade(a.id, a.variant, os.path.abspath(a.fixture), None, not a.no_runtime)
    else:
        if not a.run_id:
            ap.error("run_id required")
        variant = a.variant or a.run_id.split("-")[1]
        row = grade(a.run_id, variant, os.path.join(runmod.RUNS_DIR, a.run_id),
                    os.path.join(runmod.LOGS_DIR, a.run_id), not a.no_runtime)
    summary = {k: row.get(k) for k in ("grade_id", "variant", "project", "spec_passed",
                                        "spec_failed", "spec_unknown", "spec_total",
                                        "grade_s")}
    summary["completion"] = (row.get("completion") or {}).get("outcome")
    b = row.get("build") or {}
    summary["build"] = {k: (b.get(k) or {}).get("rc") if isinstance(b.get(k), dict) else b.get(k)
                        for k in ("files_present", "files_expected", "npm_install", "tsc",
                                  "build", "serve_ok", "stage", "error")}
    rt = row.get("runtime") or {}
    summary["runtime"] = {k: rt.get(k) for k in ("ran", "mode", "why", "error", "fps_play",
                                                 "context_types", "screenshots")}
    summary["failed"] = [c["check"] for c in row.get("spec") or [] if c.get("pass") is False]
    summary["unknown"] = [c["check"] for c in row.get("spec") or [] if c.get("pass") is None]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
