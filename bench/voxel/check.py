#!/usr/bin/env python
"""The voxel check: every rep's page.html in the sandboxed browser, measured.
No model is asked anything, and nothing found here ever goes back to a model
(graders only MEASURE).

    python bench/voxel/check.py --tag vx1                  # every unchecked rep
    python bench/voxel/check.py --tag vx1 --force          # re-check all
    python bench/voxel/check.py --html bench/voxel/fixtures/sample_three.html \
        --out C:/tmp/vx-smoke                              # one page (a smoke)

Use the stack interpreter (PIL, numpy). CPU only: Chromium renders with
SwiftShader in Docker, so it is not a GPU consumer -- but run it after the
generation run, not beside it, so it cannot slow a timed rep.

PER PAGE (bench/voxel/page_check.py has the fixed sequence):
  1. a sandbox network (bench/sandbox/sandbox_net.py, prefix `voxel`): an
     `--internal` Docker network whose one way out is the egress gate
     (global addresses on 80/443 only; the Windows host unreachable).
  2. the pinned Playwright image (bench/octopus/playwright.Dockerfile ->
     octo-playwright:1.63.0, built on first use) on that network only, with
     the page in a directory of its own (index.html and nothing else) and
     Chromium's proxy set to the gate. The page may fetch only CDN_HOSTS
     (page_check.py); everything else is aborted and recorded.
  3. host side, PIL over each screenshot (thresholds are CHOICES, stated at
     NONBLANK_*): background share, colour count, non-blank; pixel change
     between shots -- the orbit evidence for a page with no three.js camera.
  4. the gate's own log (sandbox_net.down): which hosts it let through.

TASK r3f-stack (run.py --task r3f-stack; the rep has project/, not page.html;
the prompt names the stack in one sentence and nothing else, so the project
is whatever the model wrote):
  1. build: bench/octopus/serve_app.sh in the Octopus grader's pinned node
     image on a `voxel` sandbox network -- `npm build` when the project has a
     package.json (npm install, tsc --noEmit only with a tsconfig.json, npm
     run build, in a COPY; exit codes, `error TS` counts, tails; dist/ copied
     to reps/<rep>/build/dist), else `js build` (the project itself, with
     its index.html, is the page). A failed build is a measurement.
  2. static: the Octopus grader's V4 stack checks (bench/octopus/grade.py
     v4_stack_checks over grade.Src -- what the sentence asks: r3f major 10
     with a Canvas, koota world / traits / queries, pmndrs math imported and
     used), over the extracted project, with the version npm installed.
  3. the built dist/ through the same page_check.py sequence as a page.
     `python bench/voxel/check.py --project DIR --out DIR` checks one project.

OUTPUT: reps/<rep>/check/{browser.json, check.json, *.png} (gitignored) and
one row per check appended to results/<tag>/checks.jsonl (tracked).
An exception is the check's `error`, never a measurement (PROTOCOL rule 3).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(ROOT, "bench", "sandbox"))
import sandbox_net  # noqa: E402  (the internal network + egress gate)

CHECK_VERSION = 1
PW_IMAGE = "octo-playwright:1.63.0"        # the Octopus grader's image, one pin
PW_DOCKERFILE = os.path.join(ROOT, "bench", "octopus", "playwright.Dockerfile")
NET_PREFIX = "voxel"
PAGE_TIMEOUT_S = 420
SHOTS = ("t05", "t10", "t15", "t16", "orbit_right", "orbit_high", "zoom_out")
# NON-BLANK (CHOICES, 2026-09-26): a shot is blank when >= 99% of its pixels
# sit within 8 levels (per channel) of its most common colour, or when its
# mean channel standard deviation is under 2.
NONBLANK_BG_SHARE = 0.99
NONBLANK_TOL = 8
NONBLANK_MIN_STD = 2.0
# ORBIT BY PIXELS (only when no three.js camera was read): a drag changed
# the picture by > 3x the no-input change and by > 2 mean levels.
ORBIT_PIXEL_FACTOR = 3.0
ORBIT_PIXEL_MIN = 2.0


def docker(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def ensure_pw_image() -> None:
    if docker(["image", "inspect", PW_IMAGE], 60).returncode == 0:
        return
    r = docker(["build", "-q", "-t", PW_IMAGE, "-f", PW_DOCKERFILE,
                os.path.dirname(PW_DOCKERFILE)], 1800)
    if r.returncode != 0:
        raise RuntimeError("could not build " + PW_IMAGE + ": " + r.stderr[-400:])


# ------------------------------------------------------------ pixel math --
def _img(path):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype("int16")


def img_stats(path: str) -> dict:
    """Background share, colour count and non-blank for one screenshot."""
    import numpy as np
    a = _img(path)
    flat = a.reshape(-1, 3)
    keys = (flat[:, 0].astype(np.int64) << 16) | (flat[:, 1].astype(np.int64) << 8) | flat[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode = int(vals[counts.argmax()])
    bg = np.array([(mode >> 16) & 255, (mode >> 8) & 255, mode & 255], dtype=np.int16)
    near = (np.abs(flat - bg) <= NONBLANK_TOL).all(axis=1)
    bg_share = float(near.mean())
    q = flat >> 3                                           # 5 bits per channel
    qk = (q[:, 0].astype(np.int64) << 10) | (q[:, 1].astype(np.int64) << 5) | q[:, 2]
    qv, qc = np.unique(qk, return_counts=True)
    colours = int((qc >= max(1, flat.shape[0] // 10000)).sum())   # >= 0.01% of pixels
    std = float(flat.std(axis=0).mean())
    return {"bg_rgb": [int(x) for x in bg], "bg_share": round(bg_share, 4),
            "colours_q5": colours, "std": round(std, 2),
            "nonblank": bg_share < NONBLANK_BG_SHARE and std >= NONBLANK_MIN_STD}


def img_diff(p1: str, p2: str) -> float | None:
    try:
        a, b = _img(p1), _img(p2)
    except (OSError, ValueError):
        return None
    if a.shape != b.shape:
        return None
    return round(float(abs(a - b).mean()), 3)


def pixel_evidence(shots_dir: str) -> dict:
    p = lambda n: os.path.join(shots_dir, f"{n}.png")      # noqa: E731
    have = {n for n in SHOTS if os.path.isfile(p(n))}
    d = {}
    for a, b, k in (("t05", "t10", "anim_t05_t10"), ("t10", "t15", "anim_t10_t15"),
                    ("t15", "t16", "no_input"), ("t16", "orbit_right", "drag_right"),
                    ("orbit_right", "orbit_high", "drag_up"), ("orbit_high", "zoom_out", "wheel")):
        d[k] = img_diff(p(a), p(b)) if a in have and b in have else None
    base = d.get("no_input")
    moved = [d.get("drag_right"), d.get("drag_up")]
    if base is None or any(m is None for m in moved):
        d["orbit"] = None
    else:
        d["orbit"] = any(m > max(ORBIT_PIXEL_FACTOR * base, ORBIT_PIXEL_MIN) for m in moved)
    return d


# -------------------------------------------------------------- summary --
def summarise(b: dict, shots: dict, pix: dict, gate: dict | None) -> dict:
    """The checks.jsonl row's measurements, from browser.json + the pixels."""
    st = (b.get("stats") or {})
    sc = st.get("scene") or {}
    reqs = b.get("requests") or []
    net = [r for r in reqs if r.get("decision") in ("cdn", "blocked")]
    cam = b.get("camera_input") or {}
    if cam.get("camera"):
        orbit, orbit_by = cam.get("orbit"), "camera"
    else:
        orbit, orbit_by = pix.get("orbit"), ("pixels" if pix.get("orbit") is not None else None)
    default_shots = [n for n in ("t05", "t10", "t15") if n in shots]
    return {
        "loaded": "load_error" not in b and bool(b.get("events")),
        "load_error": b.get("load_error"),
        "first_draw_s": b.get("first_draw_s"),
        "page_errors": len(b.get("page_errors") or []),
        "console_errors": sum(1 for c in b.get("console") or [] if c.get("type") == "error"),
        "console_warnings": sum(1 for c in b.get("console") or [] if c.get("type") == "warning"),
        "first_errors": ([e.get("text", "")[:200] for e in (b.get("page_errors") or [])[:3]]
                         + [c.get("text", "")[:200] for c in (b.get("console") or [])
                            if c.get("type") == "error"][:3])[:4],
        "nonblank": any(shots[n].get("nonblank") for n in default_shots) if default_shots else None,
        "nonblank_t15": (shots.get("t15") or {}).get("nonblank"),
        "colours_q5_t15": (shots.get("t15") or {}).get("colours_q5"),
        "fps": (b.get("fps") or {}).get("fps"),
        "max_frame_gap_ms": (b.get("fps") or {}).get("max_gap_ms"),
        "software_rendered": True,
        "webgl": (b.get("probe") or {}).get("webgl"),
        "gl_renderer": (b.get("probe") or {}).get("renderer"),
        "contexts": st.get("contexts") or [],
        "gl_per_frame": b.get("gl_per_frame"),
        "three": {"revision": st.get("revision"), "scenes": st.get("scenes_observed"),
                  "renders": st.get("renders")} if st.get("revision") or st.get("scenes_observed") else None,
        "voxels": sc.get("voxels"),
        "box_meshes": sc.get("box_meshes"), "box_instances": sc.get("box_instances"),
        "meshes": sc.get("meshes"), "instanced_meshes": sc.get("instanced_meshes"),
        "instances": sc.get("instances"),
        "cube_equiv_nonbox": sc.get("cube_equiv_nonbox"),
        "triangles": sc.get("triangles"), "objects": sc.get("objects"),
        "points": sc.get("points"), "point_vertices": sc.get("point_vertices"),
        "lights": sc.get("lights"),
        "distinct_colors": sc.get("distinct_colors"), "colors_capped": sc.get("colors_capped"),
        "material_colors": sc.get("material_colors"), "instance_colors": sc.get("instance_colors"),
        "vertex_colors": sc.get("vertex_colors"),
        "bbox_voxels": (sc.get("bbox_voxels") or {}).get("size"),
        "bbox_all": (sc.get("bbox_all") or {}).get("size"),
        "stats_truncated": sc.get("truncated"),
        "shadows": any((r or {}).get("shadow_map") for r in st.get("renderers") or []),
        "orbit": orbit, "orbit_by": orbit_by,
        "needs_network": bool(net),
        "cdn_hosts": sorted({r["host"] for r in reqs if r.get("decision") == "cdn"}),
        "blocked_hosts": sorted({r["host"] for r in reqs if r.get("decision") == "blocked"}),
        "failed_requests": sum(1 for r in reqs if r.get("failure")
                               or (r.get("status") or 0) >= 400),
        "gate": ({k: gate.get(k) for k in ("allowed_hosts", "allowed", "denied", "denied_sample")}
                 if gate else None),
        "check_wall_s": b.get("wall_s"),
    }


def analyse(shots_dir: str, gate: dict | None = None) -> dict:
    """Read browser.json + screenshots in shots_dir; write check.json."""
    bj = os.path.join(shots_dir, "browser.json")
    b = json.load(open(bj, encoding="utf-8")) if os.path.isfile(bj) else {}
    shots = {}
    for n in SHOTS:
        p = os.path.join(shots_dir, f"{n}.png")
        if os.path.isfile(p):
            try:
                shots[n] = img_stats(p)
            except (OSError, ValueError) as e:
                shots[n] = {"error": str(e)[:200]}
    pix = pixel_evidence(shots_dir)
    out = {"check_version": CHECK_VERSION, "shots": shots, "pixels": pix,
           "summary": summarise(b, shots, pix, gate) if b else None,
           "browser_json": os.path.isfile(bj)}
    with open(os.path.join(shots_dir, "check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


# --------------------------------------------------------------- the run --
def net_tag(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", label)[:60]


def check_page(html_path: str, shots_dir: str, label: str,
               timeout: float = PAGE_TIMEOUT_S, seed: int = 1) -> dict:
    """One page through the sandboxed browser. Never raises. `html_path` is a
    single HTML file (served as index.html, alone) or a DIRECTORY served as
    it is -- a built Vite dist/ (index.html + assets/)."""
    os.makedirs(shots_dir, exist_ok=True)
    for fn in os.listdir(shots_dir):
        fp = os.path.join(shots_dir, fn)
        if os.path.isfile(fp):
            os.remove(fp)
    rec: dict = {"label": label, "started_at": time.time()}
    tag = net_tag(label)
    stage = tempfile.mkdtemp(prefix="voxel-page-")
    if os.path.isdir(html_path):
        shutil.copytree(html_path, stage, dirs_exist_ok=True)
    else:
        shutil.copyfile(html_path, os.path.join(stage, "index.html"))
    name = f"voxel-pw-{tag}"[:60]
    gate = None
    try:
        ensure_pw_image()
        net = sandbox_net.up(tag, prefix=NET_PREFIX)
        rec["sandbox_net"] = {k: net.get(k) for k in ("ok", "network", "gate", "error", "ready_s")}
        if not net["ok"]:
            rec["error"] = f"sandbox network: {net.get('error')}"
            return rec
        argv = ["run", "--rm", "--name", name, "--ipc=host", "--cpus", "6", "--memory", "6g",
                *sandbox_net.network_args(tag, "voxel-browser", NET_PREFIX),
                "-v", f"{stage}:/page:ro", "-v", f"{HERE}:/checker:ro", "-v", f"{shots_dir}:/out",
                PW_IMAGE, "python3", "/checker/page_check.py", "--page", "/page", "--out", "/out",
                "--proxy", sandbox_net.PROXY_URL, "--seed", str(seed)]
        try:
            r = docker(argv, timeout)
            rec["browser_rc"] = r.returncode
            rec["browser_out"] = (r.stdout + r.stderr)[-1500:]
        except subprocess.TimeoutExpired:
            docker(["rm", "-f", name], 60)
            rec["timeout"] = f"browser did not finish in {timeout:.0f}s (partial browser.json kept)"
    except Exception as e:                                       # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    finally:
        try:
            gate = sandbox_net.down(tag, prefix=NET_PREFIX)
            rec["gate"] = gate
        except Exception as e:                                   # noqa: BLE001
            rec["gate_error"] = str(e)[:200]
        shutil.rmtree(stage, ignore_errors=True)
    try:
        rec["analysis"] = analyse(shots_dir, gate)
        if not rec["analysis"]["browser_json"] and "error" not in rec:
            rec["error"] = "page_check produced no browser.json"
    except Exception as e:                                       # noqa: BLE001
        rec["error"] = f"analysis: {type(e).__name__}: {e}"[:400]
    rec["wall_s"] = round(time.time() - rec["started_at"], 1)
    return rec


def check_served(container: str, url: str, shots_dir: str, label: str,
                 timeout: float = PAGE_TIMEOUT_S, seed: int = 1) -> dict:
    """A SERVED app through the same page_check.py sequence (bench/octopus
    pagoda.py: the model's project served on :3001 by the Octopus grader's
    serve_app.sh). The browser runs in `container`'s network namespace --
    the app's sandbox network, whose gate is its proxy -- and loads `url`
    (loopback: the app's own server). Never raises; the caller owns the app
    container and its network, and analyses after taking the gate's log
    down (analyse(shots_dir, gate))."""
    os.makedirs(shots_dir, exist_ok=True)
    for fn in os.listdir(shots_dir):
        fp = os.path.join(shots_dir, fn)
        if os.path.isfile(fp):
            os.remove(fp)
    rec: dict = {"label": label, "url": url, "container": container,
                 "started_at": time.time()}
    name = f"voxel-pw-{net_tag(label)}"[:60]
    try:
        ensure_pw_image()
        argv = ["run", "--rm", "--name", name, "--ipc=host", "--cpus", "6", "--memory", "6g",
                "--network", f"container:{container}",
                "-v", f"{HERE}:/checker:ro", "-v", f"{shots_dir}:/out",
                PW_IMAGE, "python3", "/checker/page_check.py", "--url", url, "--out", "/out",
                "--proxy", sandbox_net.PROXY_URL, "--seed", str(seed)]
        try:
            r = docker(argv, timeout)
            rec["browser_rc"] = r.returncode
            rec["browser_out"] = (r.stdout + r.stderr)[-1500:]
        except subprocess.TimeoutExpired:
            docker(["rm", "-f", name], 60)
            rec["timeout"] = f"browser did not finish in {timeout:.0f}s (partial browser.json kept)"
    except Exception as e:                                       # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    if not os.path.isfile(os.path.join(shots_dir, "browser.json")) and "error" not in rec:
        rec["error"] = "page_check produced no browser.json"
    rec["wall_s"] = round(time.time() - rec["started_at"], 1)
    return rec


# ------------------------------------------------------- task r3f-stack --
# The extracted project is BUILT exactly as the Octopus grader builds a V4
# project (bench/octopus/serve_app.sh `npm` or `js`, grade.build_mode, in its
# pinned node image, on a sandbox network; build-only mode: dist/ is copied
# out and the container exits), its stack is MEASURED statically with the
# Octopus grader's own V4 checks (grade.v4_stack_checks over grade.Src, with
# the build's npm ls), and the built dist/ goes through the same
# page_check.py sequence. A build
# that ran and failed is a MEASUREMENT (build.outcome "failed"); only a
# build that could not run (Docker, the gate, a timeout) is an `error`.
BUILD_TIMEOUT_S = 45 * 60      # serve_app.sh's own step timeouts sum to 40 min
OCTOPUS = os.path.join(ROOT, "bench", "octopus")


def _voxel_run():
    """bench/voxel/run.py (this directory's `run`, never bench/octopus's)."""
    mine = os.path.normcase(os.path.join(HERE, "run.py"))
    m = sys.modules.get("run")
    if m is not None and os.path.normcase(os.path.abspath(getattr(m, "__file__", "") or "")) == mine:
        return m
    if m is None:
        sys.path.insert(0, HERE)
        import run as runmod
        return runmod
    if "voxel_run" not in sys.modules:
        spec = importlib.util.spec_from_file_location("voxel_run", mine)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["voxel_run"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["voxel_run"]


_grade = None


def octopus_grade():
    """bench/octopus/grade.py, loaded by path. It does `import run` for
    bench/octopus/run.py, which here would find bench/voxel/run.py; so the
    octopus run module stands in under that name while grade.py loads."""
    global _grade
    if _grade is None:
        octo = _voxel_run().octopus_run()
        saved = sys.modules.get("run")
        sys.modules["run"] = octo
        try:
            spec = importlib.util.spec_from_file_location(
                "octopus_grade", os.path.join(OCTOPUS, "grade.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            if saved is None:
                sys.modules.pop("run", None)
            else:
                sys.modules["run"] = saved
        _grade = mod
    return _grade


def stack_static(proj: str, build: dict | None = None) -> dict:
    """The Octopus grader's V4 stack checks over the extracted project, by
    parse (tree-sitter); `build` (read_build's record) gives the version npm
    installed. Measured, never fed back."""
    g = octopus_grade()
    s = g.Src(proj)
    checks = g.v4_stack_checks(s, proj, build)
    return {"grader": "bench/octopus/grade.py v4_stack_checks",
            "checks": checks,
            "passed": sum(c["pass"] is True for c in checks), "total": len(checks),
            "files_parsed": len(s.files), "parse_errors": s.parse_errors,
            "source_lines": s.text.count("\n")}


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def read_build(out: str) -> dict:
    """What serve_app.sh left in `out`, parsed as grade.app_up parses it."""
    res: dict = {"stage": _read(os.path.join(out, "stage")).strip()}
    for step in ("npm_install", "tsc", "build"):
        log = _read(os.path.join(out, f"{step}.log"))
        t = _read(os.path.join(out, f"{step}.rc")).strip()
        res[step] = {"rc": int(t) if t.lstrip("-").isdigit() else None,
                     "error_ts": len(re.findall(r"error TS\d+", log)),
                     "errors": len(re.findall(r"(?im)^\s*(npm )?(ERR!|error)\b", log)),
                     "first_ts_errors": [ln.strip()[:240] for ln in log.splitlines()
                                         if re.search(r"error TS\d+", ln)][:5],
                     "tail": log[-1500:]}
    for step in ("npm_install", "build"):
        sec = _read(os.path.join(out, f"{step}.s")).strip()
        res[step]["seconds"] = int(sec) if sec.isdigit() else None
    try:
        res["npm_ls"] = {k: (v or {}).get("version") for k, v in json.load(open(
            os.path.join(out, "npm_ls.json"), encoding="utf-8")).get("dependencies", {}).items()}
    except (OSError, ValueError, AttributeError):
        res["npm_ls"] = None
    dist = os.path.join(out, "dist")
    files = [os.path.join(dp, fn) for dp, _d, fns in os.walk(dist) for fn in fns]
    res["dist"] = os.path.isfile(os.path.join(dist, "index.html"))
    res["dist_files"] = len(files)
    res["dist_bytes"] = sum(os.path.getsize(p) for p in files)
    res["outcome"] = ("built" if res["stage"] == "built" and res["dist"]
                      else "failed" if res["stage"] in ("built", "failed") else None)
    return res


def build_summary(b: dict) -> dict:
    """The checks.jsonl row's build fields (the tails stay in check.json)."""
    st = lambda k: b.get(k) or {}                                  # noqa: E731
    return {"outcome": b.get("outcome"), "stage": b.get("stage"),
            "npm_install_rc": st("npm_install").get("rc"),
            "npm_install_s": st("npm_install").get("seconds"),
            "tsc_rc": st("tsc").get("rc"), "tsc_error_ts": st("tsc").get("error_ts"),
            "build_rc": st("build").get("rc"), "build_error_ts": st("build").get("error_ts"),
            "build_s": st("build").get("seconds"),
            "first_ts_errors": (st("tsc").get("first_ts_errors")
                                or st("build").get("first_ts_errors") or [])[:3],
            "dist_files": b.get("dist_files"), "npm_ls": b.get("npm_ls"),
            "error": b.get("error")}


def build_project(proj: str, out: str, label: str,
                  timeout: float = BUILD_TIMEOUT_S) -> dict:
    """npm install / tsc / npm run build of a COPY of `proj` in the Octopus
    grader's pinned node image, on a `voxel` sandbox network (the internet
    through the allow-public-only gate; npm needs it). Never raises."""
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    rec: dict = {"started_at": time.time()}
    tag = net_tag("b-" + label)
    name = f"voxel-build-{tag}"[:60]
    try:
        net = sandbox_net.up(tag, prefix=NET_PREFIX)
        rec["sandbox_net"] = {k: net.get(k) for k in ("ok", "network", "gate", "error", "ready_s")}
        if not net["ok"]:
            rec["error"] = f"sandbox network: {net.get('error')}"
        else:
            image = _voxel_run().octopus_run().IMAGE
            mode = "npm" if os.path.isfile(os.path.join(proj, "package.json")) else "js"
            rec.update(image=image, mode=mode)
            argv = ["run", "--rm", "--name", name, "--cpus", "4", "--memory", "6g",
                    *sandbox_net.network_args(tag, "voxel-build", NET_PREFIX),
                    *sandbox_net.env_args(),
                    "-v", f"{proj}:/src:ro", "-v", f"{out}:/out", "-v", f"{OCTOPUS}:/grader:ro",
                    image, "bash", "/grader/serve_app.sh", mode, "build"]
            try:
                r = docker(argv, timeout)
                rec["docker_rc"] = r.returncode
                rec["docker_out"] = (r.stdout + r.stderr)[-800:]
            except subprocess.TimeoutExpired:
                docker(["rm", "-f", name], 60)
                rec["error"] = f"build container did not finish in {timeout:.0f}s"
    except Exception as e:                                       # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    finally:
        try:
            gate = sandbox_net.down(tag, prefix=NET_PREFIX)
            rec["gate"] = {k: gate.get(k) for k in ("allowed_hosts", "allowed", "denied",
                                                    "denied_sample")} if gate else None
        except Exception as e:                                   # noqa: BLE001
            rec["gate_error"] = str(e)[:200]
    rec.update(read_build(out))
    if "error" not in rec and rec["outcome"] is None:
        rec["error"] = (f"the build container left no verdict (stage {rec['stage']!r}, "
                        f"docker rc {rec.get('docker_rc')}): {rec.get('docker_out', '')[-300:]}")
    rec["wall_s"] = round(time.time() - rec["started_at"], 1)
    return rec


def check_project(proj: str, rep_dir: str, label: str, timeout: float = PAGE_TIMEOUT_S,
                  seed: int = 1, build_timeout: float = BUILD_TIMEOUT_S,
                  builder=build_project, pager=None) -> dict:
    """One r3f-stack rep: static stack checks, the build, then page_check on
    the built dist/. Writes <rep>/check/check.json (with `build` and
    `static`) unless the build could not RUN (an error: a rerun retries it).
    Never raises."""
    pager = pager or check_page
    shots = os.path.join(rep_dir, "check")
    rec: dict = {"label": label, "task": "r3f-stack", "started_at": time.time()}
    rec["build"] = builder(proj, os.path.join(rep_dir, "build"), label, build_timeout)
    try:
        rec["static"] = stack_static(proj, rec["build"])
    except Exception as e:                                       # noqa: BLE001
        rec["static"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    dist = os.path.join(rep_dir, "build", "dist")
    if rec["build"].get("error"):
        rec["error"] = "build: " + str(rec["build"]["error"])[:400]
    if os.path.isfile(os.path.join(dist, "index.html")):
        page = pager(dist, shots, label, timeout, seed)
        rec["page"] = {k: v for k, v in page.items() if k != "analysis"}
        rec["analysis"] = page.get("analysis")
        for k in ("timeout", "browser_rc"):
            if k in page:
                rec[k] = page[k]
        if page.get("error"):
            rec["error"] = "page: " + str(page["error"])[:400]
    else:
        rec["page"] = {"ran": False, "why": "no dist/index.html: the build "
                       + ("could not run" if rec.get("error") else "failed")}
        os.makedirs(shots, exist_ok=True)
        for fn in os.listdir(shots):
            fp = os.path.join(shots, fn)
            if os.path.isfile(fp):
                os.remove(fp)
    cj = os.path.join(shots, "check.json")
    if rec.get("error") and not rec.get("analysis"):
        if os.path.isfile(cj):
            os.remove(cj)                  # not measured: stays in the to-check list
    else:
        out = dict(rec.get("analysis") or {"check_version": CHECK_VERSION, "shots": {},
                                           "pixels": None, "summary": None,
                                           "browser_json": False})
        out.update(task="r3f-stack", build=rec["build"], static=rec["static"],
                   page_check=rec["page"])
        os.makedirs(shots, exist_ok=True)
        with open(cj, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    rec["wall_s"] = round(time.time() - rec["started_at"], 1)
    return rec


def stack_summary(static: dict | None) -> dict:
    st = static or {}
    if st.get("error"):
        return {"error": st["error"]}
    r3f = next((c for c in st.get("checks") or [] if c["check"] == "r3f_v10_canvas"), {})
    ev = r3f.get("evidence") or {}
    return {"checks": {c["check"]: c["pass"] for c in st.get("checks") or []},
            "passed": st.get("passed"), "total": st.get("total"),
            "parse_errors": st.get("parse_errors"), "files_parsed": st.get("files_parsed"),
            "r3f": {k: ev.get(k) for k in ("declared", "installed", "major")}}


def reps_to_check(run_dir: str, force: bool = False) -> list[dict]:
    """The last summary row per (arm, rep) that has a page (single-html) or
    a project (r3f-stack), unchecked unless force."""
    runmod = _voxel_run()
    rows = runmod.last_rows(runmod.read_rows(os.path.join(run_dir, "summary.jsonl")))
    out = []
    for (_arm, _rep), r in sorted(rows.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if not (r.get("page") or r.get("project")):
            continue
        cj = os.path.join(run_dir, r["dir"], "check", "check.json")
        if os.path.isfile(cj) and not force:
            continue
        out.append(r)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--html", help="check one file instead of a run")
    ap.add_argument("--project", help="check one r3f-stack project directory instead of a run")
    ap.add_argument("--out", help="with --html / --project: where the shots, build and "
                                  "check.json go")
    ap.add_argument("--timeout", type=float, default=PAGE_TIMEOUT_S)
    ap.add_argument("--build-timeout", type=float, default=BUILD_TIMEOUT_S,
                    help="r3f-stack: the build container's wall-clock limit")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.project:
        if not a.out:
            print("  --project needs --out")
            return 2
        rec = check_project(os.path.abspath(a.project), os.path.abspath(a.out),
                            "project-" + time.strftime("%H%M%S"), a.timeout, a.seed,
                            a.build_timeout)
        print(json.dumps({"error": rec.get("error"), "build": build_summary(rec.get("build") or {}),
                          "stack": stack_summary(rec.get("static")), "page": rec.get("page")},
                         indent=1)[:4000])
        print(json.dumps((rec.get("analysis") or {}).get("summary"), indent=1))
        return 0 if "error" not in rec else 1
    if a.html:
        if not a.out:
            print("  --html needs --out")
            return 2
        rec = check_page(a.html, a.out, "single-" + time.strftime("%H%M%S"), a.timeout, a.seed)
        print(json.dumps({k: v for k, v in rec.items() if k != "analysis"}, indent=1)[:3000])
        print(json.dumps((rec.get("analysis") or {}).get("summary"), indent=1))
        return 0 if "error" not in rec else 1
    if not a.tag:
        print("  --tag or --html is required")
        return 2
    run_dir = os.path.join(RESULTS, a.tag)
    todo = reps_to_check(run_dir, a.force)
    print(f"  {len(todo)} page(s) to check in {os.path.relpath(run_dir, ROOT)}")
    checks = os.path.join(run_dir, "checks.jsonl")
    bad = 0
    for r in todo:
        label = f"{a.tag}-{r['arm']}-r{r['rep']}-a{r['attempt']}"
        print(f"  {label} ...", flush=True)
        task = r.get("task") or "single-html"
        extra = {}
        if task == "r3f-stack":
            rec = check_project(os.path.join(run_dir, r["project"]),
                                os.path.join(run_dir, r["dir"]), label, a.timeout, a.seed,
                                a.build_timeout)
            b = build_summary(rec.get("build") or {})
            extra = {"build": b, "stack": stack_summary(rec.get("static")),
                     "page_checked": bool(rec.get("analysis"))}
            print(f"    build {b.get('outcome')}: install rc {b.get('npm_install_rc')}, tsc rc "
                  f"{b.get('tsc_rc')} ({b.get('tsc_error_ts')} TS errors), build rc "
                  f"{b.get('build_rc')}; stack {extra['stack'].get('passed')}/"
                  f"{extra['stack'].get('total')}", flush=True)
        else:
            rec = check_page(os.path.join(run_dir, r["page"]),
                             os.path.join(run_dir, r["dir"], "check"), label, a.timeout, a.seed)
        s = (rec.get("analysis") or {}).get("summary") or {}
        row = {"tag": a.tag, "task": task, "arm": r["arm"], "rep": r["rep"],
               "attempt": r["attempt"],
               "check_version": CHECK_VERSION, "checked_at": time.time(),
               "error": rec.get("error"), "timeout": rec.get("timeout"),
               "browser_rc": rec.get("browser_rc"), "wall_s": rec.get("wall_s"), **extra, **s}
        with open(checks, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        bad += bool(rec.get("error"))
        print(f"    {'ERROR ' + str(rec.get('error')) if rec.get('error') else 'measured'}: "
              f"nonblank={s.get('nonblank')} voxels={s.get('voxels')} "
              f"colors={s.get('distinct_colors')} fps={s.get('fps')} errors="
              f"{s.get('page_errors')}+{s.get('console_errors')} orbit={s.get('orbit')} "
              f"cdn={s.get('cdn_hosts')} blocked={s.get('blocked_hosts')}", flush=True)
    print(f"  done: {len(todo) - bad} measured, {bad} error(s). Next: "
          f"python bench/voxel/compare.py --tag {a.tag}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
