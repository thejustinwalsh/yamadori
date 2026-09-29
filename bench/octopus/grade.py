#!/usr/bin/env python
"""Deterministic grader for Octopus Invaders runs. No model is ever asked.

    python bench/octopus/grade.py pilot-V0-xhigh-1            # a Hermes run
    python bench/octopus/grade.py --fixture C:/Users/jwals/octo/fixtures/x --variant V1 --id fx-v1
    python bench/octopus/grade.py RUN_ID --no-runtime         # skip the browser

Use the stack interpreter (tree-sitter-language-pack, PIL, numpy; scipy for V4's
pixel checks). Builds and
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
                pinned node image with network (npm needs it) -- the
                sandbox network (sandbox_net.py, #47): the internet
                through an allow-public-only gate, never the host. V0: every
                script index.html references exists. V4 (v4): the layout
                the model chose (build_mode: a package.json -> npm install
                + npm run build, tsc only with a tsconfig.json; else served
                as is), judged by builds_and_serves, not a file list.
3. runtime      bench/octopus/browser_check.py in the pinned Playwright
                image: 15 s idle (console errors), start screen, click,
                mousemove, ESC, 30 s of play (errors, FPS), a game-over
                attempt, screenshots; v3: seeded Math.random and a fake clock
                advanced by the script, so one state plays one game (see
                browser_check.py). Served from dist/ (a static server) when
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
appends), screenshots copied to index/octopus/grades/<id>/v<version>/<stamp>/
(gitignored; v3: one directory per grade row, never overwritten -- up to v2
they went to index/octopus/grades/<id>/ and a re-grade overwrote them; the
state at 2026-09-26 22:12 is kept in index/octopus/grades_backup_20260926-221204).
A grade of prompt n >= 2 of an iterative run carries `assisted` (ASSISTED):
its prompt was built from a grade, which leaks the answer key (operator,
2026-09-26); prompt 1 is the headline. bench/octopus/results/
grade_annotations.jsonl labels the rows written before that field existed.
Each row's `grader` is {path, version, changes} (GRADER_VERSION below; a
bare string is version 1). A change to what a check accepts bumps the
version and is re-applied to every graded state by bench/octopus/regrade.py.

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
import sandbox_net          # noqa: E402  (the app container's network; #47)
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

# GRADER VERSION. Every row carries {"path", "version", "changes"}; a row
# whose `grader` is the bare string "bench/octopus/grade.py" is version 1 (all
# rows before 2026-09-26). A change to what any check accepts bumps the
# version and says why, with the spec text behind it; old rows are never
# rewritten (grades.jsonl is append-only) and a re-grade appends a new row.
# PROTOCOL: these changes were made after results were seen, so each one is
# applied to EVERY graded run and snapshot (bench/octopus/regrade.py) and the
# before/after is reported, never just the run that exposed it.
GRADER_VERSION = 4
_V4_WHY = ("operator, 2026-09-27: V4's prompt is the original with ONE line replaced by "
           "'build a space shooter game with r3f (react-three-fiber) v10 and Koota and pmndrs math. multi-file project structure.' "
           "(variants.V4_TECH); every canvas-specific line stays, and adapting it is part of "
           "the test, so V4 is graded on what the prompt asks, by behaviour and intent. The "
           "2026-09-26 V4 (our pins, layout and stack rules) was never run or graded")
GRADER_CHANGES = {
    4: [
        {"check": "V4 only (V0-V3 unchanged)",
         "change": "every check of V0, V1, V2 and V3 is byte-for-byte the v3 rule (the shared "
                   "code paths take the variant; serve_app.sh's `ts` and `js` modes are "
                   "unchanged): their v3 rows stand, no re-grade is needed. No V4 state has "
                   "been graded",
         "spec": "(scope)", "why": _V4_WHY},
        {"check": "webgpu_canvas, tsl_node_material, package_pins_exact (V4)",
         "change": "REMOVED for V4: the prompt no longer asks for a WebGPU canvas, TSL node "
                   "materials or pinned versions",
         "spec": "build a space shooter game with r3f (react-three-fiber) v10 and Koota and pmndrs math. multi-file project structure.",
         "why": _V4_WHY},
        {"check": "r3f_v10_canvas (V4; replaces r3f_canvas_useFrame and instanced_or_points)",
         "change": "@react-three/fiber major 10 -- the version npm installed (npm ls), else "
                   "what package.json or an import-map URL declares -- and Canvas imported "
                   "from @react-three/fiber (any subpath) and used outside the import. "
                   "useFrame and instancing are the model's choice, not asked",
         "spec": "r3f (react-three-fiber) v10", "why": _V4_WHY},
        {"check": "koota_world_traits_queries, math_used (V4)",
         "change": "koota: unchanged rule (createWorld + trait + a query). math: imported "
                   "(math or math/*, a bare specifier or a CDN URL) AND a name it imports used "
                   "outside the import (v3: the import alone)",
         "spec": "and Koota and pmndrs math", "why": _V4_WHY},
        {"check": "builds_and_serves (V4; replaces the expected file list)",
         "change": "the layout the model chose (grade.build_mode): with a package.json, npm "
                   "install and npm run build exit 0 and dist/ is served (serve_app.sh `npm`: "
                   "tsc only with a tsconfig.json, recorded, not required); without one, every "
                   "local file index.html references exists and the folder is served",
         "spec": "multi-file project structure / serve with python3 -m http.server 3001 when "
                 "done. make sure all imports work and game runs immediately on first load.",
         "why": _V4_WHY + "; the kept PROJECT STRUCTURE names vanilla js/ files the r3f "
                          "request contradicts"},
        {"check": "particles_rendered (V4; replaces particles_draw_ctx)",
         "change": "JUDGEMENT CALL (operator to confirm): the particle pool reaches something "
                   "that draws, by parse -- a particle-named .draw/.render call outside the "
                   "particle module, a JSX element named *particle*, or a file naming particles "
                   "that writes instance/vertex data or maps particles to JSX. A render path, "
                   "not pixels: no calibrated runtime signal tells a particle from a star",
         "spec": "particles.draw(ctx) MUST be called in the game's main draw loop. if you "
                 "forget this, no particles render", "why": _V4_WHY},
        {"check": "octopus_pixel_look (V4; replaces hard_pixels)",
         "change": "from the SCREENSHOTS: >= 2 pixel sprites in one frame (saturated "
                   "components whose outline is >= 90% straight runs of >= 3 px, not a plain "
                   "rectangle, not a glyph in a text line). Thresholds are choices, calibrated "
                   "on the v3 screenshots and synthetic controls (grade.PIX_*)",
         "spec": "render ALL octopus types using grid-based fillRect patterns, NOT ctx.arc() "
                 "or smooth shapes ... this gives authentic pixelated look / set "
                 "ctx.imageSmoothingEnabled = false (pixel art style requires this)",
         "why": _V4_WHY},
        {"check": "scrolls_downward (V4, new)",
         "change": "from screenshot PAIRS with the pointer still: matched tiles vote down / up "
                   "/ sideways; DOWN >= 6 and DOWN >= 2 x (UP + SIDEWAYS). V0-V3 have no "
                   "scroll check (unchanged)",
         "spec": "everything scrolls DOWNWARD (positive y direction) ... do NOT scroll "
                 "sideways or upward", "why": _V4_WHY},
        {"check": "circle_collision (V4)",
         "change": "also a vector library's distance call (distance, squaredDistance, "
                   "distanceTo ...): pmndrs/math vectors are arrays the dx*dx form never names",
         "spec": "use circle-to-circle distance check: dx*dx + dy*dy < (r1+r2)*(r1+r2)",
         "why": _V4_WHY},
        {"check": "hud_positions, start_screen, click_starts_game (V4)",
         "change": "the HUD is judged by V1's DOM rule OR V0's fillText rule, whichever the "
                   "game used; text drawn inside the WebGL scene (drei Text, troika, glyph ...) "
                   "cannot be located: hud_positions not measured, start_screen and "
                   "click_starts_game by V3's rules",
         "spec": "HUD: score (top left), level (top center), combo counter (top right) -- all "
                 "at y=40 with 20px font", "why": _V4_WHY},
        {"check": "hard_pixels, no_libraries, particles_draw_ctx, files_expected (V4)",
         "change": "DROPPED for V4: canvas-API and file-name specifics the r3f request "
                   "replaces (imageSmoothingEnabled, fillRect, a no-libraries rule, the js/ "
                   "file list). The spec constants and content checks are kept as they are",
         "spec": "no libraries. no frameworks. (the replaced line)", "why": _V4_WHY},
    ],
    3: [
        {"check": "ALL runtime checks",
         "change": "the browser run is DETERMINISTIC: the page's Math.random is seeded "
                   "(browser_check.RNG_SEED) and its clock is Playwright's fake clock, paused "
                   "and advanced by the script in 50 ms steps paced to the wall clock, so the "
                   "game gets exactly 60 frames per game second whatever the host's load; "
                   "screenshots are taken at fixed GAME times. v2 waited in wall-clock seconds "
                   "while the game's own Math.random chose the enemies",
         "spec": "ship follows mouse cursor smoothly / pause with ESC key / HUD: score (top "
                 "left), level (top center), combo counter (top right) -- all at y=40",
         "why": "v2 on the UNCHANGED reference fixture, 5 grades: mousemove_moves_ship 1/5, "
                "hud_positions 1/5, esc_pause 0/5 (3/3 on 2026-09-24); v0f@p2's mousemove "
                "flipped between two grades (SELF-IMPROVEMENT-LOG #62)"},
        {"check": "click_starts_game (the start-element fallback)",
         "change": "after clicking a start ELEMENT the grader undoes its own scrollIntoView "
                   "(window.scrollTo(0, 0)) and nudges the pointer 1 px and back, as a "
                   "player's hand does; v2 left the page scrolled and the pointer still",
         "spec": "start screen: ... \"CLICK TO START\" text / ship follows mouse cursor",
         "why": "the reference's Start button is below the fold at 1280x720; the grader's "
                "scroll (32 px) shifted every later screenshot and left the pointer below the "
                "canvas, so the ship flew to (0, 0) until the mouse next moved"},
        {"check": "mousemove_moves_ship",
         "change": "the pointer alternates left / right THREE times (x 30% / 70%, 300 ms of "
                   "game time each); per side the pixel-wise median of its 3 shots; the "
                   "object at the pointer (pixels within +-40 px of it, bottom 45%, that "
                   "differ between pointer-here and pointer-away) must be found again -- same "
                   "colours, or the same object shape -- where the pointer went (>= 100 px, "
                   "both directions); NOT "
                   "MEASURED (None) when the game was over during the sequence. v2: one "
                   "left/right pair, bright-pixel or cyan centroid shift > 25% of the width. "
                   "Calibrated on 2 games and the same 2 with the ship's lerp zeroed "
                   "(grade.ship_follows docstring; n = 1 each)",
         "spec": "ship follows mouse cursor smoothly; ship.x += (mouseX - ship.x) * 0.35 "
                 "per frame",
         "why": "v2's centroids were moved by a cyan planet and by descending enemies "
                "(v0f@p2, index/octopus/grades_backup_20260926-221204/v0f-V0-xhigh-1_p2: the "
                "ship visibly moved 243 -> 1036 px, check failed), and failed whenever the "
                "ship had died (reference). 30% / 70% (v2: 19% / 81%) lie between the "
                "20/40/60/80% columns of a four-enemy W/(n+1) wave; parked at 19% the "
                "reference's ship died under its column-1 ink blobs in the first v3 trial"},
        {"check": "esc_pause",
         "change": "ESC is pressed 2.3 s of game time after the start; NOT MEASURED (None) when "
                   "GAME OVER was showing before ESC; if the game ends after the second ESC, "
                   "the resume half counts as met (the game ran). Thresholds unchanged",
         "spec": "pause with ESC key",
         "why": "v2 pressed ESC on the reference's GAME OVER screen (its ship had died) and "
                "failed it 5/5"},
        {"check": "click_starts_game",
         "change": "the start screen's canvas text must be absent from what was drawn SINCE "
                   "the click (browser_check fill_since, from 100 ms after it); v2 read the "
                   "texts drawn in the last 1500 ms, which v3's 500 ms of game time after the "
                   "click would still contain",
         "spec": "start screen: ... \"CLICK TO START\" text",
         "why": "found by the v3 stability run: v0f@p2's state failed 5/5 with its HUD drawn "
                "and the start text last drawn at the click"},
        {"check": "hud_positions",
         "change": "JUDGEMENT CALL (operator to confirm): COMBO's position is judged when a "
                   "COMBO text was drawn at all; a game that shows the counter only while a "
                   "combo is running passes on SCORE and LEVEL (the evidence says "
                   "combo_drawn: false; combo_counter still checks the counter exists). "
                   "V1/V2 read the DOM HUD right after the 30 s of play, not after the "
                   "game-over attempt",
         "spec": "HUD: score (top left), level (top center), combo counter (top right) -- all "
                 "at y=40 with 20px font; combo multiplier for consecutive kills without "
                 "getting hit",
         "why": "the reference draws COMBO only when combo > 1; v2 passed it only on the 1 "
                "of 5 grades whose random play happened to chain two kills"},
        {"check": "FLAKY",
         "change": "a check listed in grade.FLAKY (not 5/5 or 0/5 on an unchanged state in "
                   "bench/octopus/stability.py) keeps its result in the row, marked "
                   "`flaky`, and is left out of the headline (spec_passed / spec_failed / "
                   "spec_total); spec_flaky lists it",
         "spec": "(measurement hygiene, not a spec item)",
         "why": "a measurement that flips on unchanged code is not a measurement"},
    ],
    2: [
        {"check": "particles_draw_ctx",
         "change": "the receiver may be any object whose name contains 'particle' in ANY "
                   "case (Particles, PARTICLES, ParticleSystem, this.particles, "
                   "game.particleSystem, particles[i]), and the call must be OUTSIDE "
                   "particles.js (the main loop, not the module's own text); v1 was "
                   "case-sensitive and accepted a call anywhere",
         "spec": "particles.draw(ctx) MUST be called in the game's main draw loop",
         "why": "v0f@p1 (Particles.draw(ctx) in Game.draw) failed and became prompt 2's "
                "item 2; v0b (ParticleSystem.draw) and v0e (PARTICLES.draw) failed the same way"},
        {"check": "bullet_speeds_4_tiers",
         "change": "a `speed` key matches in any case and as a suffix (bulletSpeed, "
                   "BULLET_SPEED, \"speed\"); v1 matched lower-case `speed:` only",
         "spec": "BULLET SPEEDS ... tier 1: speed 8 ... tier 4: speed 14"},
        {"check": "boss_every_5_levels",
         "change": "an ASSIGNMENT of 5 to a boss-named target (this.bossEvery = 5, "
                   "Game.BOSS_INTERVAL = 5) counts like a declaration or object key",
         "spec": "boss octopus every 5 levels"},
        {"check": "circle_collision",
         "change": "operands may be member expressions (this.dx*this.dx) and a square may "
                   "be written x**2 or Math.pow(x, 2)",
         "spec": "use circle-to-circle distance check: dx*dx + dy*dy < (r1+r2)*(r1+r2)"},
        {"check": "pixel_grids_ge_4",
         "change": "JUDGEMENT CALL (operator to confirm): grids whose cells are palette "
                   "digits 0-9 count (string rows of digits, or a 2D literal of >= 6 rows x "
                   ">= 6 single-digit cells); v1 took only 0/1 (and '.#X ') cells",
         "spec": "render ALL octopus types using grid-based fillRect patterns ... define "
                 "pixel grids (e.g. 8x8 or 10x10 arrays of 0s and 1s) for each octopus type"},
        {"check": "bg_color_0D1117_source",
         "change": "#0D1117 in a .css/.html file counts too (v1: JS/TS strings only)",
         "spec": "dark space background (#0D1117)"},
        {"check": "monospace_font",
         "change": "'monospace' in .css/.html matches in any case (CSS keywords are "
                   "case-insensitive)",
         "spec": "all UI text in monospace font"},
    ],
}


def _sha(path: str) -> str | None:
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


def grader_info() -> dict:
    # v3: which code graded the row (the browser script drives the game)
    return {"path": "bench/octopus/grade.py", "version": GRADER_VERSION,
            "changes": GRADER_CHANGES.get(GRADER_VERSION, []),
            "sha256_16": {"grade.py": _sha(os.path.abspath(__file__)),
                          "browser_check.py": _sha(os.path.join(HERE, "browser_check.py"))}}


# v3: checks that are not 5/5 or 0/5 on an unchanged state
# (bench/octopus/stability.py, 5 grades each of the reference fixture and of
# v0f@p2's state). Each entry names its evidence. Their results stay in the
# row, marked `flaky`, and are left out of the headline (spec_passed /
# spec_failed / spec_total), listed in spec_flaky.
# EMPTY on evidence: stability batch 20260926-232208 (browser_check sha256
# a850d6eac21d0be7; run beside a full re-grade, so under extra CPU load):
# all 27 checks 5/5 or 0/5 on both states.
FLAKY: dict[str, str] = {}

# Operator, 2026-09-26: feeding the grader's failed checks back as a
# follow-up prompt leaks the answer key. A grade of the state after prompt
# n >= 2 of an iterative run is ASSISTED; the headline is prompt 1.
ASSISTED = "assisted: graded follow-up"


def headline(checks: list[dict]) -> dict:
    """The row's counts. v3: a check in FLAKY is marked (`flaky`: why), keeps
    its result, and is left out of spec_passed / spec_failed / spec_unknown /
    spec_total; spec_flaky lists it with its result."""
    for c in checks:
        if c.get("check") in FLAKY:
            c["flaky"] = FLAKY[c["check"]]
    head = [c for c in checks if not c.get("flaky")]
    return {"spec_passed": sum(1 for c in head if c.get("pass") is True),
            "spec_failed": sum(1 for c in head if c.get("pass") is False),
            "spec_unknown": sum(1 for c in head if c.get("pass") is None),
            "spec_total": len(head),
            "spec_flaky": [{"check": c["check"], "pass": c.get("pass")}
                           for c in checks if c.get("flaky")]}


def assisted_label(gid: str) -> str | None:
    _run, _, pn = gid.partition("@p")
    return ASSISTED if pn.isdigit() and int(pn) >= 2 else None


def shots_dest(gid: str, graded_at: float) -> str:
    """Where a grade's screenshots are kept: one directory per grade ROW,
    under its grader version, never overwritten (v2 re-grades overwrote
    index/octopus/grades/<id>/ and lost the earlier ones)."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(graded_at))
    return os.path.join(SHOTS, gid.replace("@", "_"), f"v{GRADER_VERSION}", stamp)


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
# v4: V4's prompt keeps the vanilla PROJECT STRUCTURE while asking for r3f,
# so no file list is expected; builds_and_serves judges the project instead
EXPECTED["V4"] = []

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
    chats = [r for r in relay if r.get("path", "").endswith(("/chat/completions", "/responses"))
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
    chats = [r for r in relay if r.get("path", "").endswith(("/chat/completions", "/responses"))]
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


def project_root(run_dir: str, variant: str | None = None) -> str | None:
    """The directory holding the game: space-shooter/ if present, else the
    shallowest dir with an index.html (outside node_modules/dist). v4, V4
    only: a package.json marks the project as well as an index.html does
    (a Vite project may keep index.html elsewhere), space-shooter/ first."""
    cand = os.path.join(run_dir, "space-shooter")
    if os.path.isfile(os.path.join(cand, "index.html")):
        return cand
    marks = ("index.html",)
    if variant == "V4":
        if os.path.isfile(os.path.join(cand, "package.json")):
            return cand
        marks = ("package.json", "index.html")
    best = None
    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".git")]
        depth = dirpath[len(run_dir):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        if any(m in filenames for m in marks) and (best is None or depth < best[0]):
            best = (depth, dirpath)
    return best[1] if best else None


def build_mode(variant: str, proj: str) -> str:
    """serve_app.sh's first argument. V0 `js` (served as is), V1-V3 `ts` (npm
    install, tsc, npm run build). v4, V4: the model chose -- `npm` when the
    project has a package.json (install, build; tsc only with a
    tsconfig.json), else `js`."""
    ts = variants.VARIANTS[variant]["ts"]
    if ts is None:
        return "npm" if os.path.isfile(os.path.join(proj, "package.json")) else "js"
    return "ts" if ts else "js"


def files_and_refs(variant: str, proj: str) -> dict:
    res: dict = {"project": proj}
    mode = build_mode(variant, proj)
    if variants.VARIANTS[variant]["ts"] is None:
        res["layout"] = "npm" if mode == "npm" else (
            "static" if os.path.isfile(os.path.join(proj, "index.html")) else "none")
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
    if mode == "js":
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
    mode = build_mode(variant, proj)
    ts = mode in ("ts", "npm")
    name = f"octo-app-{gid}"[:60]
    docker(["rm", "-f", name], timeout=60)
    for fn in os.listdir(out) if os.path.isdir(out) else []:
        p = os.path.join(out, fn)
        if os.path.isfile(p):
            os.remove(p)
    os.makedirs(out, exist_ok=True)
    # The model's code runs here (npm lifecycle scripts, vite, and in
    # runtime() the game in Chromium), so the container is on a sandbox
    # network (sandbox_net.py, SELF-IMPROVEMENT-LOG #47): internet through the
    # allow-public-only gate, the Windows host unreachable. No gate, no build.
    net = sandbox_net.up(name)
    res: dict = {"container": name, "net_tag": name, "sandbox_net": net, "mode": mode}
    if not net["ok"]:
        res["error"] = f"sandbox network: {net.get('error')}"
        return res
    r = docker(["run", "-d", "--name", name, "--cpus", "4", "--memory", "6g",
                *sandbox_net.network_args(name, "octo-app"), *sandbox_net.env_args(),
                "-v", f"{proj}:/src:ro", "-v", f"{out}:/out", "-v", f"{HERE}:/grader:ro",
                runmod.IMAGE, "bash", "/grader/serve_app.sh", mode],
               timeout=300)
    res["docker_rc"] = r.returncode
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
    # In the app's namespace (the sandbox network); Chromium gets the gate as
    # its proxy for anything but loopback, so a CDN the page loads still loads.
    r = docker(["run", "--rm", "--network", f"container:{name}", "--ipc=host",
                "-v", f"{HERE}:/grader:ro", "-v", f"{shots}:/out", PW_IMAGE,
                "python3", "/grader/browser_check.py", "--url", "http://localhost:3001/",
                "--out", "/out", "--proxy", sandbox_net.PROXY_URL], timeout=900)
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


# v3 mousemove rule (GRADER_CHANGES[3]): TRACK the object at the pointer.
FOLLOW_X = {"left": 0.30, "right": 0.70}   # browser_check.py step 3
FOLLOW_BAND = (0.55, 1.0)    # rows: the bottom 45%, where the ship flies
FOLLOW_CORE_PX = 40          # object pixels within +-40 px of the pointer's x
FOLLOW_DIFF = 90             # "differs": channel-sum |at - away| > 90
FOLLOW_TOL = 60              # "the same colour": channel-sum difference < 60
FOLLOW_SLACK_PX = 48         # the displacement may miss the pointer's by 48 px
FOLLOW_DY_PX = 24            # ... and move up/down by 24 px
FOLLOW_MIN_PX = 100          # tracked pixels needed in EACH window
# evidence only (the first v3 draft's rule; it failed v0f's ship, whose
# window was brighter than the ship): bright pixel counts at / away
FOLLOW_HALF_PX = 80
FOLLOW_BRIGHT = 90


def ship_follows(shots: dict[str, list[str]]) -> dict:
    """shots: {"left": [3 paths, pointer at 30%], "right": [3 paths, 70%]}.

    Per side, the pixel-wise MEDIAN of its three shots (taken 0.6 s of game
    time apart) keeps what stays at that pointer position and drops what
    passes through (an enemy, a particle, a star). OBJECT pixels: within
    +-40 px of a pointer position, bottom 45%, where the two medians differ
    (channel sum > 90). For each window W, the object pixels are searched for
    where the pointer went (the pointer's displacement +-48 px, +-24 px up or
    down) in two ways, and the better count is TRACKED:
      colour  the other median has the same colour there (< 60);
      shape   the other window has an object pixel there (colour-free: a
              translucent foreground layer tints the ship differently in the
              two windows, as v0f's nebula does), counting in both windows
              only the object pixels STAYED does not explain.
    Passes when TRACKED >= 100 in both windows: something drawn at the pointer
    went where the pointer went. Mouse-reactive parallax (the spec asks for
    it) shifts a layer by a few pixels: it is found again at no displacement
    (STAYED: same colour at no displacement), never 512 px away.
    A crosshair drawn at the pointer would pass too (not seen in any run).

    Calibration (2026-09-26, browser_check v3, SELF-IMPROVEMENT-LOG #62; n = 1
    each): tracked colour / shape, left | right --
      reference fixture                 1340 / 1340 | 1340 / 1340
      v0f@p2's state                     639 /  205 |  169 /  205
      the reference, ship lerp x 0         0 /    0 |    0 /    0
      v0f@p2, ship lerp x 0 (parallax)     0 /    2 |    0 /    2"""
    import numpy as np
    med = {side: np.median(np.stack([_img(p) for p in paths]), axis=0)
           for side, paths in shots.items()}
    h, w = med["left"].shape[:2]
    y0, y1 = int(h * FOLLOW_BAND[0]), int(h * FOLLOW_BAND[1])
    differs = np.abs(med["left"] - med["right"]).sum(axis=2) > FOLLOW_DIFF
    obj = {}
    for win, fx in FOLLOW_X.items():
        px = int(w * fx)
        m = np.zeros(differs.shape, bool)
        m[y0:y1, max(0, px - FOLLOW_CORE_PX):px + FOLLOW_CORE_PX + 1] = True
        obj[win] = m & differs
    def best(win: str, center: int, how: str, keep=None) -> tuple[int, list[int], object]:
        """The displacement (center +-48, +-24) at which most of window
        win's object pixels are found again in the other side's median."""
        other = "right" if win == "left" else "left"
        m = obj[win] if keep is None else keep[win]
        ys, xs = np.nonzero(m)
        b = (0, [center, 0], np.zeros(len(ys), bool))
        if not len(ys):
            return b
        vals = med[win][ys, xs]
        for dy in range(-FOLLOW_DY_PX, FOLLOW_DY_PX + 1, 2):
            yy = np.clip(ys + dy, 0, h - 1)
            for dx in range(center - FOLLOW_SLACK_PX, center + FOLLOW_SLACK_PX + 1, 2):
                xx = np.clip(xs + dx, 0, w - 1)
                if how == "colour":
                    hit = np.abs(med[other][yy, xx] - vals).sum(axis=1) < FOLLOW_TOL
                else:
                    hit = keep[other][yy, xx]
                n = int(hit.sum())
                if n > b[0]:
                    b = (n, [dx, dy], hit)
        return b

    # STAYED first: object pixels found again, same colour, at no
    # displacement are parallax (a layer shifted a few px); the shape search
    # uses only the pixels it does not explain
    stayed, keep = {}, {}
    for win in FOLLOW_X:
        n, shift, hit = best(win, 0, "colour")
        stayed[win] = (n, shift)
        k = obj[win].copy()
        ys, xs = np.nonzero(obj[win])
        k[ys[hit], xs[hit]] = False
        keep[win] = k
    track: dict[str, dict] = {}
    for win, fx in FOLLOW_X.items():
        other = "right" if win == "left" else "left"
        d = int(w * FOLLOW_X[other]) - int(w * fx)
        col, shp = best(win, d, "colour"), best(win, d, "shape", keep)
        track[win] = {"object_px": int(obj[win].sum()), "tracked_px": max(col[0], shp[0]),
                      "tracked_colour_px": col[0], "tracked_colour_shift": col[1],
                      "tracked_shape_px": shp[0], "tracked_shape_shift": shp[1],
                      "stayed_px": stayed[win][0], "stayed_shift": stayed[win][1]}
    ok = all(t["tracked_px"] >= FOLLOW_MIN_PX for t in track.values())
    # evidence only: bright pixels in each window, pointer at / away, per shot
    feats: dict[str, dict[str, list[int]]] = {}
    for win, fx in FOLLOW_X.items():
        feats[win] = {"at": [], "away": []}
        for side, paths in shots.items():
            for p in paths:
                a = _img(p)
                x0 = max(0, int(w * fx) - FOLLOW_HALF_PX)
                x1 = min(w, int(w * fx) + FOLLOW_HALF_PX)
                n = int((a[y0:y1, x0:x1].max(axis=2) > FOLLOW_BRIGHT).sum())
                feats[win]["at" if side == win else "away"].append(n)
    return {"pass": bool(ok), "track": track, "bright_px_evidence": feats,
            "rule": f"pointer alternates x=30%/70% three times (300 ms of game time each); per "
                    f"side the median of its 3 shots; object = pixels within "
                    f"+-{FOLLOW_CORE_PX} px of the pointer, bottom 45%, differing by > "
                    f"{FOLLOW_DIFF} between the two medians; passes when >= {FOLLOW_MIN_PX} of "
                    f"them are found again (same colour < {FOLLOW_TOL}, or an object pixel "
                    f"there) at the pointer's displacement (+-{FOLLOW_SLACK_PX} px, "
                    f"+-{FOLLOW_DY_PX} px vertical) in BOTH windows"}


# ------------------------------------------------------------ static parse


GRID_CHARS = set("0123456789.#X ")      # v2: digits 2-9 added (palette grids)
# v2: `<anything>.<...particle...>[i]?.draw`, any case, optional chaining.
PARTICLE_DRAW = re.compile(r"(?:[\w$]+(?:\[[^\]]*\])?\??\.)*[\w$]*particle[\w$]*"
                           r"(?:\[[^\]]*\])?\??\.draw", re.I)
_OP = r"[\w$]+(?:\.[\w$]+)*"


def _square(tag: str) -> str:
    return (rf"(?:(?P<{tag}>{_OP})\s*\*\s*(?P={tag})|{_OP}\s*\*\*\s*2"
            rf"|Math\.pow\(\s*{_OP}\s*,\s*2\s*\))")


# v2: dx*dx + dy*dy with member operands, or x**2 / Math.pow(x, 2) squares.
CIRCLE_SUM = re.compile(_square("a") + r"\s*\+\s*" + _square("b"))


def particles_drawn(s: "Src") -> tuple[bool, dict]:
    """particles_draw_ctx (v2): a <particle-named object>.draw(...) call made
    outside particles.js -- the spec's 'called in the game's main draw loop'.
    Returns (pass, evidence)."""
    sites = [c for c in s.call_sites
             if PARTICLE_DRAW.fullmatch(re.sub(r"\s+", "", c["callee"]))]
    main = [c for c in sites
            if not re.match(r"particles?\.", os.path.basename(c["file"]), re.I)]
    return bool(main), {
        "rule": "a call <object named *particle*, any case>.draw(...) outside particles.js",
        "calls": main[:5], "only_in_particles_js": [c for c in sites if c not in main][:3]}


class Src:
    """Everything the spec checks read from the project's JS/TS, by parse."""

    def __init__(self, proj_unc: str):
        from tree_sitter_language_pack import get_parser
        self.strings: list[str] = []
        self.numbers: list[float] = []
        self.calls: list[str] = []
        # v2: each call with where it is (file, line, the enclosing function's
        # name), for checks that care WHERE a call is made.
        self.call_sites: list[dict] = []
        self.imports: list[str] = []
        # the names each import statement brings in, by source
        # (`import { Canvas } from '@react-three/fiber'`: the EXPORTED name)
        self.import_names: dict[str, set[str]] = {}
        # v4: the LOCAL names each import binds, by source (the alias of
        # `{ vec2 as v }`, a default import, `* as M`), and how often each
        # identifier occurs OUTSIDE import statements -- "imported and used"
        self.import_locals: dict[str, set[str]] = {}
        self.ident_uses: dict[str, int] = {}
        # v4: JSX element names (`<Canvas>`, `<Particles />`), with the file
        self.jsx: list[tuple[str, str]] = []
        self.file_text: dict[str, str] = {}
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
                rel = os.path.relpath(path, proj_unc).replace("\\", "/")
                self.files.append(rel)
                self.file_text[rel] = code
                self.text += "\n" + code
                tree = get_parser(lang).parse(code.encode("utf-8"))
                if tree.root_node.has_error:
                    self.parse_errors += 1
                self._walk(tree.root_node, code.encode("utf-8"), rel)

    def _t(self, n, src) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def _enclosing(self, n, src) -> str | None:
        """The name of the nearest named function around node n: a method,
        a function declaration, or a function assigned to a key / variable /
        member (`draw() {}`, `draw: function`, `const draw = () =>`,
        `Game.draw = function`)."""
        p = n.parent
        while p is not None:
            if p.type in ("method_definition", "function_declaration",
                          "generator_function_declaration"):
                k = p.child_by_field_name("name")
                return self._t(k, src) if k is not None else None
            if p.type in ("function_expression", "arrow_function", "function"):
                q = p.parent
                if q is not None and q.type == "pair":
                    k = q.child_by_field_name("key")
                    return self._t(k, src) if k is not None else None
                if q is not None and q.type == "variable_declarator":
                    k = q.child_by_field_name("name")
                    return self._t(k, src) if k is not None else None
                if q is not None and q.type == "assignment_expression":
                    k = q.child_by_field_name("left")
                    return self._t(k, src) if k is not None else None
            p = p.parent
        return None

    def _walk(self, root, src, rel: str = ""):
        stack = [(root, False)]
        while stack:
            n, in_import = stack.pop()
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
                    self.call_sites.append({"callee": self._t(f, src), "file": rel,
                                            "line": n.start_point[0] + 1,
                                            "in": self._enclosing(n, src)})
            elif t == "import_statement":
                s = n.child_by_field_name("source")
                if s is not None:
                    self.imports.append(self._t(s, src).strip("'\""))
                    names = self.import_names.setdefault(self.imports[-1], set())
                    local = self.import_locals.setdefault(self.imports[-1], set())
                    todo = list(n.named_children)
                    while todo:
                        c = todo.pop()
                        if c.type == "import_specifier":
                            k = c.child_by_field_name("name")
                            al = c.child_by_field_name("alias")
                            if k is not None:
                                names.add(self._t(k, src))
                            if (al or k) is not None:
                                local.add(self._t(al or k, src))
                        elif c.type == "namespace_import":
                            local.update(self._t(x, src) for x in c.named_children
                                         if x.type == "identifier")
                        elif c.type == "identifier" and c.parent is not None \
                                and c.parent.type == "import_clause":
                            local.add(self._t(c, src))            # default import
                        elif c.type in ("import_clause", "named_imports"):
                            todo.extend(c.named_children)
            elif t in ("identifier", "property_identifier", "type_identifier",
                       "shorthand_property_identifier"):
                self.idents.add(self._t(n, src))
                if not in_import:
                    k = self._t(n, src)
                    self.ident_uses[k] = self.ident_uses.get(k, 0) + 1
            elif t == "assignment_expression":
                l, r = n.child_by_field_name("left"), n.child_by_field_name("right")
                if l is not None and r is not None:
                    self.assign.append((self._t(l, src), self._t(r, src)))
                    # v2: `this.bossEvery = 5` / `Game.BOSS_INTERVAL = 5`
                    if self._t(r, src).strip() == "5" and re.search(
                            r"boss", self._t(l, src), re.I):
                        self.const5.append(self._t(l, src))
            elif t in ("jsx_opening_element", "jsx_self_closing_element"):
                k = n.child_by_field_name("name")
                if k is not None:
                    self.jsx.append((self._t(k, src), rel))
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
                    # v2: palette digits 2-9 are cells too (v1: "01.#X ")
                    if len(rows) >= 6 and all(len(r) >= 6 and set(r) <= GRID_CHARS
                                              for r in rows):
                        self.grids += 1
                # v2: a 2D literal, >= 6 rows of >= 6 single-digit cells (a 0/1
                # grid whose rows are 6-7 wide, or a palette grid of 0-9), is
                # one grid. (A 0/1 grid with rows of >= 8 cells also counts
                # each row, as in v1.)
                if len(kids) >= 6 and all(c.type == "array" for c in kids):
                    rows2 = [[self._t(x, src) for x in c.named_children] for c in kids]
                    if all(len(r) >= 6 and all(len(v) == 1 and v.isdigit() for v in r)
                           for r in rows2) and any(v != "0" for r in rows2 for v in r):
                        self.grids += 1
                self._speeds(kids, src)
            elif t == "object":
                vals = []
                for c in n.named_children:
                    if c.type == "pair" and c.child_by_field_name("value") is not None:
                        vals.append(c.child_by_field_name("value"))
                self._speeds(vals, src)
            inner = in_import or t == "import_statement"
            stack.extend((c, inner) for c in n.children)

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
                # v2: any case, and as a key suffix (bulletSpeed, BULLET_SPEED,
                # "speed"); v1 was lower-case `speed:` only.
                m = re.search(r"speed\w*[\"']?\s*:\s*(\d+)", self._t(c, src), re.I)
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

    def used_imports(self, pkg: str) -> dict[str, int]:
        """v4: the local names imported from pkg (or a subpath) and how often
        each occurs outside import statements (0: imported, never used)."""
        out: dict[str, int] = {}
        for src_, names in self.import_locals.items():
            if src_ == pkg or src_.startswith(pkg + "/"):
                for k in names:
                    out[k] = self.ident_uses.get(k, 0)
        return out


# -------------------------------------------------------------------- 4 spec


# V4 (grader v4, 2026-09-27; GRADER_CHANGES[4]). V4's prompt is the original
# with ONE line replaced by the operator's "Build it with r3f
# (react-three-fiber) v10 and Koota and pmndrs math." (variants.V4_TECH);
# everything canvas-specific stays in the prompt, and adapting it is part of
# the test. So V4 is graded on what the prompt ASKS, by behaviour and intent:
# the stack the sentence names (these three checks, shared with the pagoda
# task: bench/voxel/check.py), the game spec's constants and content, and the
# game as it runs -- never a file name, a canvas API or a layout. MEASURED
# only, never fed back. The names looked for are the packages' own exports
# (koota 0.6.6 dist, @react-three/fiber 10.0.0-alpha.5), read 2026-09-26.
_KOOTA_QUERY = ("query", "queryFirst", "useQuery", "useQueryFirst")
R3F = "@react-three/fiber"
# a CDN module URL (an import map or a bare URL import) names its package:
# https://esm.sh/@react-three/fiber@10.0.0-alpha.5/..., cdn.jsdelivr.net/npm/
# math@0.1.0/random, unpkg.com/koota ...
_CDN = re.compile(r"^https?://(?:esm\.sh|cdn\.jsdelivr\.net/npm|unpkg\.com|"
                  r"cdn\.skypack\.dev|ga\.jspm\.io/npm:)/?"
                  r"(?P<name>@[^/@]+/[^/@]+|[^/@]+)(?:@(?P<ver>[^/?#]+))?(?P<sub>/[^?#]*)?")
# text drawn INTO the WebGL scene, which neither the DOM nor fillText sees
WEBGL_TEXT = ("troika-three-text", "@react-three/drei", "@pmndrs/glyph",
              "three-mesh-ui", "@react-three/uikit", "three-bmfont-text")


def _pkg_of(source: str) -> tuple[str, str | None]:
    """(package[/subpath], version or None) of an import source: a bare
    specifier as written, a CDN URL by its package path."""
    m = _CDN.match(source or "")
    if not m:
        return source, None
    return m.group("name") + (m.group("sub") or "").rstrip("/"), m.group("ver")


def _is_pkg(source: str, pkg: str) -> bool:
    p, _v = _pkg_of(source)
    return p == pkg or p.startswith(pkg + "/")


def package_json(pu: str) -> dict:
    try:
        pkg = json.load(open(os.path.join(pu, "package.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return pkg if isinstance(pkg, dict) else {}


def semver_major(spec) -> int | None:
    """The major of a version or range as written (10.0.0-alpha.5,
    ^10.0.0-alpha.5, >=10.0.0-alpha.0, 10) -- None for a dist-tag
    (alpha, next, latest) or anything else."""
    m = re.match(r"^\s*(?:npm:[^@\s]+@)?[\^~=v<>\s]*(\d+)(?:[.\s]|$)", str(spec or ""))
    return int(m.group(1)) if m else None


def _page_urls(pu: str) -> list[str]:
    """Module URLs an index.html names (import map values, script src)."""
    html = read(os.path.join(pu, "index.html"))
    return re.findall(r"""["'](https?://[^"'\s]+)["']""", html)


def v4_stack_checks(s: "Src", pu: str | None = None, b: dict | None = None) -> list[dict]:
    """What the operator's sentence asks: r3f v10 with a Canvas, koota, and
    pmndrs math, each IMPORTED AND USED. `b` (the build record) gives the
    version npm actually installed (`npm_ls`), which wins over package.json
    (a dist-tag like "alpha" has no major of its own)."""
    out = []
    pkg = package_json(pu) if pu else {}
    deps = {}
    for k in ("devDependencies", "dependencies"):
        if isinstance(pkg.get(k), dict):
            deps.update(pkg[k])
    declared = deps.get(R3F)
    declared_from = "package.json" if declared is not None else None
    if declared is None and pu:
        # no package.json entry: an import map / CDN URL may carry it
        for u in _page_urls(pu) + [i for i in s.imports if _CDN.match(i)]:
            name, ver = _pkg_of(u)
            if (name == R3F or name.startswith(R3F + "/")) and ver:
                declared, declared_from = ver, "url"
                break
    installed = ((b or {}).get("npm_ls") or {}).get(R3F)
    major = semver_major(installed) if installed else semver_major(declared)
    canvas_from = sorted(src for src, names in s.import_names.items()
                         if _is_pkg(src, R3F) and "Canvas" in names)
    canvas_uses = s.ident_uses.get("Canvas", 0)
    out.append(check("r3f_v10_canvas", major == 10 and bool(canvas_from) and canvas_uses > 0,
                     "static-parse+text",
                     {"rule": "@react-three/fiber major 10 (the version npm installed, else the "
                              "one package.json or an import-map URL declares) and Canvas "
                              "imported from @react-three/fiber (any subpath) and used",
                      "declared": declared, "declared_from": declared_from,
                      "installed": installed, "major": major,
                      "canvas_imported_from": canvas_from, "canvas_uses": canvas_uses}))
    tails = [c.rsplit(".", 1)[-1] for c in s.calls]
    q = sorted({t for t in tails if t in _KOOTA_QUERY})
    each = sorted({t for t in tails if t in ("updateEach", "readEach")})
    koota = any(_is_pkg(i, "koota") for i in s.imports)
    out.append(check("koota_world_traits_queries",
                     koota and "createWorld" in tails and "trait" in tails and bool(q),
                     "static-parse",
                     {"rule": "imports koota; createWorld(...) and trait(...) called; a "
                              "query called (query / queryFirst / useQuery / useQueryFirst)",
                      "imports_koota": koota,
                      "koota_react": any(_is_pkg(i, "koota/react") for i in s.imports),
                      "createWorld": "createWorld" in tails, "trait": "trait" in tails,
                      "queries": q, "each": each}))
    subs = sorted({_pkg_of(i)[0] for i in s.imports if _is_pkg(i, "math")})
    used: dict[str, int] = {}
    for src_, names in s.import_locals.items():
        if _is_pkg(src_, "math"):
            for k in names:
                used[k] = s.ident_uses.get(k, 0)
    out.append(check("math_used", bool(subs) and any(n > 0 for n in used.values()),
                     "static-parse",
                     {"rule": "imports math (pmndrs/math) or a subpath, and uses a name it "
                              "imports (outside the import)", "imports": subs,
                      "names_used": {k: v for k, v in sorted(used.items())},
                      "math_random_calls": sum(c == "Math.random" for c in s.calls)}))
    return out


# circle-to-circle through a vector library (pmndrs/math's vec2.distance /
# squaredDistance, three's distanceTo ...): V4 only, since the sentence asks
# for pmndrs math, whose vectors are arrays the spec's dx*dx form never names
_DIST_CALLS = ("distance", "squaredDistance", "distanceSquared", "distanceTo",
               "distanceToSquared", "distSq", "sqDist", "dist", "hypot")


def particles_rendered(s: "Src") -> tuple[bool, dict]:
    """V4's `particles_rendered` (the spec: "particles.draw(ctx) MUST be called
    in the game's main draw loop. if you forget this, no particles render"),
    stack-agnostic, by parse: the particle pool reaches something that
    draws. Any of
      draw_call  a <particle-named object>.draw/.render(...) call outside a
                 particle module (V0's rule, also .render);
      jsx        a JSX element whose name contains 'particle', any case
                 (<Particles />, <ParticleSystem>) -- a component mounted;
      instances  a file that names particles (any case) and writes per-
                 instance or per-vertex data: setMatrixAt / setColorAt /
                 setXYZ / setXY calls, `.needsUpdate = true`, or a
                 particle collection mapped to JSX (`particles.map(` in a
                 .jsx/.tsx file).
    Evidence of a render path, not of pixels: the runtime cannot tell a
    particle from a star (JUDGEMENT CALL, operator to confirm)."""
    draw = re.compile(r"(?:[\w$]+(?:\[[^\]]*\])?\??\.)*[\w$]*particle[\w$]*"
                      r"(?:\[[^\]]*\])?\??\.(?:draw|render)", re.I)
    calls = [c for c in s.call_sites if draw.fullmatch(re.sub(r"\s+", "", c["callee"]))
             and not re.match(r"particles?\.", os.path.basename(c["file"]), re.I)]
    jsx = sorted({f"<{n}> in {f}" for n, f in s.jsx if "particle" in n.lower()})
    inst = []
    writes = {"setMatrixAt", "setColorAt", "setXYZ", "setXY"}
    for rel, text in s.file_text.items():
        if not re.search(r"particle", text, re.I):
            continue
        how = sorted({c["callee"].rsplit(".", 1)[-1] for c in s.call_sites
                      if c["file"] == rel and c["callee"].rsplit(".", 1)[-1] in writes})
        if re.search(r"\.needsUpdate\s*=\s*true", text):
            how.append("needsUpdate")
        if rel.endswith((".tsx", ".jsx")) and re.search(r"\b\w*particle\w*\.map\(", text, re.I):
            how.append("particles.map(...) to JSX")
        if how:
            inst.append({"file": rel, "writes": how})
    ok = bool(calls or jsx or inst)
    return ok, {"rule": "a particle-named .draw/.render(...) outside the particle module; or a "
                        "JSX element named *particle*; or a file naming particles that writes "
                        "instance/vertex data (setMatrixAt, setColorAt, setXYZ, needsUpdate) "
                        "or maps particles to JSX",
                "draw_calls": [{k: c[k] for k in ("callee", "file", "line")} for c in calls][:5],
                "jsx": jsx[:5], "instance_writes": inst[:5]}


def builds_and_serves(b: dict) -> dict:
    """V4: "a plausible project that builds and serves", in whatever layout
    the model chose (build_mode). npm layout: npm install and npm run build
    exit 0 and the built dist/ is what is served; static layout: every file
    index.html references exists and the directory is served as is. tsc is
    not asked for (the prompt names no language) and is recorded only."""
    layout = b.get("layout")
    ev = {"layout": layout, "stage": b.get("stage"), "serve_mode": b.get("serve_mode")}
    if b.get("error") and not b.get("stage"):
        return check("builds_and_serves", None, "build", {**ev, "not_measured": b["error"]})
    if "stage" not in b:
        return check("builds_and_serves", None, "build",
                     {**ev, "not_measured": "the build was not run"})
    if layout == "npm":
        rc = {k: (b.get(k) or {}).get("rc") for k in ("npm_install", "tsc", "build")}
        ok = rc["npm_install"] == 0 and rc["build"] == 0 and b.get("serve_mode") == "static"
        ev.update(rc=rc, rule="npm install and npm run build exit 0, dist/ served (tsc "
                              "recorded, not required)")
    elif layout == "static":
        ok = bool(b.get("serve_ok")) and b.get("serve_mode") == "static"
        ev.update(html_refs_missing=b.get("html_refs_missing"),
                  rule="no package.json: every local file index.html references exists, served")
    else:
        ok = False
        ev["rule"] = "no package.json and no index.html: nothing to build or serve"
    return check("builds_and_serves", ok, "build", ev)


# ------------------------------------------------ V4 pixel checks (choices)
# octopus_pixel_look: the spec's octopi are pixel art (grids of cells, hard
# edges, "smooth arcs look wrong"). JUDGED FROM THE SCREENSHOTS: saturated
# (max-min > 90, max > 150) connected components of 120+ px and 10-300 px a
# side; a component is a PIXEL SPRITE when >= 90% of its outline lies in
# straight runs of >= 3 px (pixel art drawn at >= 3 px a cell), it is not a
# plain rectangle (fill < 0.9: a health bar, a square placeholder) and it is
# not a glyph in a line of text (>= 3 components on one baseline, heights
# within 25%, gaps < 0.6 x height, mostly distinct shapes). Passes when one
# screenshot holds >= 2 pixel sprites. Calibration (2026-09-27, this code on
# the latest grader-v3 screenshots of each state, n = 1 grade each, and
# synthetic controls; the thresholds are CHOICES set before these numbers,
# not fitted to them). Pixel sprites in the best frame:
#   reference fixture 4 (start 3) | v0f@p1 9, @p2 7 (start 4) | v0e@p1 4,
#   @p2 6 | V2 skeletons (three-flatland pixel sprites) 13        -> pass
#   V1 skeleton (plain squares, fill >= 0.9) 0 | V3 skeleton 0 | the three
#   blank-page states (v0b-final, v0b-snap-0048, v0e-snap1) 0      -> fail
#   the reference start screen (3) blurred, Gaussian r 0.6 / 1 / 1.5 / 2:
#   1 / 1 / 0 / 0; a sprite magnified bilinear 0, bicubic 0, nearest 2;
#   antialiased octopi drawn as ellipses 0
#   "OCTOPUS INVADERS" in bold Consolas / Arial / Courier at 32, 48, 64 px:
#   0 (2-3 letters were sprites before the text-line rule); a formation of 8
#   copies of one sprite, gaps 6 / 14 / 30 px: 8 (not text)
PIX_SAT, PIX_BRIGHT = 90, 150
PIX_MIN_AREA, PIX_MIN_SIDE, PIX_MAX_SIDE = 120, 10, 300
PIX_RUN, PIX_LONG_FRAC, PIX_MAX_FILL = 3, 0.90, 0.90
PIX_MIN_SPRITES = 2
PIX_FRAMES = ("start", "after_click", "ship_left", "ship_right", "ship_left2", "ship_right2",
              "ship_left3", "ship_right3", "play", "play_end", "gameover", "gameover_attempt")


def _runs(line) -> list[int]:
    import numpy as np
    d = np.diff(np.concatenate(([0], line.astype("int8"), [0])))
    return list(np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0])


def pixel_sprites(a) -> dict:
    """The saturated components of one frame and which are pixel sprites."""
    import numpy as np
    from scipy import ndimage
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = (mx - mn > PIX_SAT) & (mx > PIX_BRIGHT)
    lab, _n = ndimage.label(sat)
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        h, w = m.shape
        area = int(m.sum())
        if area < PIX_MIN_AREA or min(h, w) < PIX_MIN_SIDE or max(h, w) > PIX_MAX_SIDE:
            continue
        mp = np.pad(m, 1)
        dv, dh = mp[:, 1:] != mp[:, :-1], mp[1:, :] != mp[:-1, :]
        runs = [r for x in range(dv.shape[1]) for r in _runs(dv[:, x])]
        runs += [r for y in range(dh.shape[0]) for r in _runs(dh[y, :])]
        total = sum(runs)
        long_ = sum(r for r in runs if r >= PIX_RUN)
        comps.append({"x": int(sl[1].start), "y": int(sl[0].start), "w": int(w), "h": int(h),
                      "area": area, "fill": round(area / (h * w), 3),
                      "long_frac": round(long_ / total, 3) if total else 0.0})
    # glyphs in a line of text: >= 3 on one baseline, similar heights, tight
    # gaps, mostly distinct shapes (a wave of one octopus type repeats one)
    text = set()
    for i, c in enumerate(comps):
        row = [j for j, d in enumerate(comps)
               if abs((d["y"] + d["h"]) - (c["y"] + c["h"])) <= 4
               and abs(d["h"] - c["h"]) <= 0.25 * max(c["h"], d["h"])]
        if len(row) < 3:
            continue
        xs = sorted(row, key=lambda j: comps[j]["x"])
        gaps = [comps[k]["x"] - (comps[j]["x"] + comps[j]["w"]) for j, k in zip(xs, xs[1:])]
        shapes = {(comps[j]["w"], comps[j]["h"], round(comps[j]["area"] / 8)) for j in row}
        if (sorted(gaps)[len(gaps) // 2] < 0.6 * c["h"]
                and len(shapes) >= 0.6 * len(row)):
            text.update(row)
    sprites = [c for j, c in enumerate(comps) if j not in text
               and c["long_frac"] >= PIX_LONG_FRAC and c["fill"] < PIX_MAX_FILL]
    return {"components": len(comps), "text_glyphs": len(text), "sprites": sprites}


def octopus_pixel_look(shots: str, states: dict) -> dict:
    frames = {}
    best = None
    for name in PIX_FRAMES:
        p = os.path.join(shots, f"{name}.png")
        if not os.path.isfile(p):
            continue
        r = pixel_sprites(_img(p))
        frames[name] = {"sprites": len(r["sprites"]), "components": r["components"],
                        "text_glyphs": r["text_glyphs"]}
        if best is None or len(r["sprites"]) > len(best[1]["sprites"]):
            best = (name, r)
    if not frames:
        return check("octopus_pixel_look", None, "pixel", "no screenshots")
    ok = best is not None and len(best[1]["sprites"]) >= PIX_MIN_SPRITES
    return check("octopus_pixel_look", ok, "pixel",
                 {"rule": f">= {PIX_MIN_SPRITES} pixel sprites in one screenshot (saturated "
                          f"components, >= {int(PIX_LONG_FRAC * 100)}% of the outline in straight "
                          f"runs >= {PIX_RUN} px, not a plain rectangle, not text)",
                  "best_frame": best[0] if best else None,
                  "examples": (best[1]["sprites"][:4] if best else []), "frames": frames})


# scrolls_downward: "everything scrolls DOWNWARD ... do NOT scroll sideways or
# upward". JUDGED FROM SCREENSHOT PAIRS with the pointer at the SAME place
# (parallax offset equal): ship_left -> ship_left2 -> ship_left3, the same on
# the right (600 ms of game time apart), resumed_a -> resumed_b (1 s). Both
# frames at half size; every 16 px tile of the first that has content (> 2%
# off the median colour) and CHANGED (mean |diff| >= 4) is matched in the
# second within +-20 half-px (every offset); a match counts when it
# is < half the unmoved difference. A vote is DOWN (dy >= 1 half-px, dy >=
# |dx|), UP, or SIDEWAYS (|dx| > |dy|). Passes when DOWN >= 6 and DOWN >= 2 x
# (UP + SIDEWAYS); a pair with GAME OVER showing is skipped (not measured
# when all are). Descending enemies and ink blobs vote down too: the check
# is "the scene moves down", which the spec asks of everything. Calibration
# (2026-09-27, this code on the latest grader-v3 screenshots, n = 1 each;
# CHOICES set before these numbers), down / up / sideways:
#   reference fixture 52 / 2 / 12 | v0e@p1 and @p2 414 / 18 / 54   -> pass
#   v0f@p1 3 / 5 / 15, @p2 3 / 3 / 13 (its stars and planets do not move:
#   checked by eye on resumed_a / resumed_b) | V1 skeleton (squares bob)
#   160 / 164 / 6 | V2 skeletons 119 / 129 / 5 | blank pages 0      -> fail
#   every pair REVERSED (an upward scroll): reference 1 / 73 / 6, v0e@p2
#   5 / 435 / 39                                                    -> fail
SCROLL_PAIRS = (("ship_left", "ship_left2"), ("ship_left2", "ship_left3"),
                ("ship_right", "ship_right2"), ("ship_right2", "ship_right3"),
                ("resumed_a", "resumed_b"))
SCROLL_TILE, SCROLL_R = 16, 20
SCROLL_MIN_DOWN, SCROLL_RATIO = 6, 2.0


def _half(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w].astype("float32")
    return (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4


def motion_votes(a, b) -> list[tuple[int, int]]:
    """(dx, dy) per matched tile, in half-size pixels, a -> b."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    A, B = _half(a), _half(b)
    h, w = A.shape[:2]
    T, R = SCROLL_TILE, SCROLL_R
    bg = np.median(A.reshape(-1, 3), axis=0)
    out = []
    for y in range(int(h * 0.08), int(h * 0.95) - T, T):
        for x in range(0, w - T, T):
            ta = A[y:y + T, x:x + T]
            if (np.abs(ta - bg).sum(axis=2) > 60).mean() < 0.02:
                continue
            same = float(np.abs(ta - B[y:y + T, x:x + T]).sum(axis=2).mean())
            if same < 4:
                continue
            y0, x0 = max(0, y - R), max(0, x - R)
            region = B[y0:min(h, y + T + R), x0:min(w, x + T + R)]
            win = sliding_window_view(region, (T, T, 3))[:, :, 0]
            sad = np.abs(win - ta).sum(axis=-1).mean(axis=(-1, -2))
            iy, ix = np.unravel_index(int(sad.argmin()), sad.shape)
            if float(sad[iy, ix]) < 0.5 * same:
                out.append((int(x0 + ix - x), int(y0 + iy - y)))
    return out


def scrolls_downward(shots: str, states: dict) -> dict:
    down = up = side = 0
    pairs, skipped = {}, []
    for p, q in SCROLL_PAIRS:
        pa, pb = os.path.join(shots, f"{p}.png"), os.path.join(shots, f"{q}.png")
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        if any((states.get(k) or {}).get("game_over_visible") for k in (p, q)):
            skipped.append(f"{p}->{q}")
            continue
        v = motion_votes(_img(pa), _img(pb))
        d = sum(1 for dx, dy in v if dy >= 1 and dy >= abs(dx))
        u = sum(1 for dx, dy in v if dy <= -1 and -dy >= abs(dx))
        sd = sum(1 for dx, dy in v if abs(dx) > abs(dy))
        pairs[f"{p}->{q}"] = {"down": d, "up": u, "sideways": sd, "tiles": len(v)}
        down, up, side = down + d, up + u, side + sd
    ev = {"down": down, "up": up, "sideways": side, "pairs": pairs,
          "skipped_game_over": skipped,
          "rule": f"tiles matched between frames 0.6-1 s apart with the pointer still: DOWN "
                  f">= {SCROLL_MIN_DOWN} and DOWN >= {SCROLL_RATIO:g} x (UP + SIDEWAYS)"}
    if not pairs:
        return check("scrolls_downward", None, "pixel",
                     {**ev, "not_measured": "no frame pair without GAME OVER"})
    ok = down >= SCROLL_MIN_DOWN and down >= SCROLL_RATIO * (up + side)
    return check("scrolls_downward", ok, "pixel", ev)


def webgl_text(s: "Src") -> list[str]:
    return sorted({_pkg_of(i)[0] for i in s.imports
                   if any(_is_pkg(i, p) for p in WEBGL_TEXT)})


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
    circle = bool(CIRCLE_SUM.search(s.text) or "Math.hypot" in s.text)
    if variant == "V4":
        # v4: a vector library's distance (pmndrs/math vec2.distance /
        # squaredDistance, three's distanceTo) is the same circle test
        dist = sorted({c for c in s.calls if c.rsplit(".", 1)[-1] in _DIST_CALLS})
        out.append(check("circle_collision", circle or bool(dist), "static-text",
                         {"rule": "dx*dx + dy*dy (member operands, x**2, Math.pow(x, 2)), "
                                  "Math.hypot, or a distance call (distance, squaredDistance, "
                                  "distanceTo ...) in source", "form": bool(circle),
                          "distance_calls": dist[:5]}))
    else:
        out.append(check("circle_collision", circle, "static-text",
                         "dx*dx + dy*dy (member operands, x**2 and Math.pow(x, 2) too) or "
                         "Math.hypot in source"))
    out.append(check("pixel_grids_ge_4", s.grids >= 4, "static-parse",
                     f"{s.grids} 0/1 grid rows or grids (>=8 cells) found"))
    if variant != "V4":
        # v4: V4 judges the pixel look from the screenshots (octopus_pixel_look)
        if ts:
            hard = s.has_ident("NearestFilter")
            hard_ev = "identifier NearestFilter"
        else:
            hard = any(l.endswith("imageSmoothingEnabled") and r.strip() == "false"
                       for l, r in s.assign)
            hard_ev = "assignment *.imageSmoothingEnabled = false"
        out.append(check("hard_pixels", hard, "static-parse", hard_ev))
    css_html = ""
    for dirpath, dirnames, filenames in os.walk(pu):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".git")]
        css_html += "".join(read(os.path.join(dirpath, f)) for f in filenames
                            if f.endswith((".css", ".html")))
    out.append(check("bg_color_0D1117_source",
                     s.has_str("0d1117") or "0d1117" in css_html.lower(), "static-parse+text",
                     "string literal containing #0D1117, or #0D1117 in any .css/.html"))
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
    if variant == "V3":
        mono = os.path.isfile(os.path.join(pu, "public", "fonts", "JetBrainsMono-Regular.ttf"))
    else:
        mono = s.has_str("monospace") or "monospace" in css_html.lower()
    out.append(check("monospace_font", mono, "static-parse+text",
                     "'monospace' in a JS/TS string or any .css/.html (V3: the JetBrains "
                     "Mono TTF in public/fonts)"))

    # --- stack-specific ---
    if variant == "V0":
        ok, ev = particles_drawn(s)
        out.append(check("particles_draw_ctx", ok, "static-parse", ev))
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
    if variant == "V4":
        out += v4_stack_checks(s, pu, b)
        ok, ev = particles_rendered(s)
        out.append(check("particles_rendered", ok, "static-parse", ev))
        out.append(builds_and_serves(b or {}))
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
                     "bg_color_runtime") + (("octopus_pixel_look", "scrolls_downward")
                                            if variant == "V4" else ()):
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
    # v4: V4 text drawn INTO the WebGL scene (drei Text, troika, glyph ...) is
    # read neither from the DOM nor from fillText: when no start text was
    # found there and the source imports such a renderer, V3's rules apply
    gl_text = (webgl_text(s) if variant == "V4" and not re.search(
        r"OCTOPUSINVADERS|CLICKTOSTART|STARTGAME", re.sub(r"\s+", "", t0)) else [])
    if variant == "V3" or gl_text:
        ok = (s.has_str("octopus invaders") and s.has_str("click to start")
              and st.get("std", 0) >= 4)
        out.append(check("start_screen", ok, "static-parse+pixel",
                         "glyph text is not in the DOM or 2D canvas: strings in source "
                         "and a non-blank start frame"
                         + (f" (V4: WebGL text from {gl_text})" if gl_text else "")))
    else:
        flat = re.sub(r"\s+", "", t0)
        out.append(check("start_screen", "OCTOPUSINVADERS" in flat and "CLICKTOSTART" in flat,
                         "runtime", {"rule": "title and CLICK TO START in DOM text or "
                                             "fillText, whitespace ignored",
                                     "title": "OCTOPUSINVADERS" in flat,
                                     "click_to_start": "CLICKTOSTART" in flat,
                                     "start_click": br.get("start_click")}))
    # v3: after the click, the canvas texts drawn SINCE it (fill_since): only
    # 500 ms of game time pass, inside fill_recent's 1500 ms window
    st1 = states.get("after_click") or {}
    if st1.get("fill_since") is not None:
        t1 = ((st1.get("dom_text") or "") + "\n" + "\n".join(st1["fill_since"])).upper()
    else:
        t1 = txt("after_click")
    d_click = meanabs(p("start"), p("after_click")) if os.path.isfile(p("after_click")) else None
    if variant == "V3" or gl_text:
        out.append(check("click_starts_game", d_click is not None and d_click > 2.0, "pixel",
                         {"meanabs_start_vs_after_click": d_click, "threshold": 2.0,
                          **({"webgl_text": gl_text} if gl_text else {})}))
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
    def over(label):
        # the game was over at this point (browser_check's state; v3)
        st_ = states.get(label) or {}
        if "game_over_visible" in st_:
            return bool(st_["game_over_visible"])
        return "GAME OVER" in txt(label)
    seq = {side: [p(f"ship_{side}" + ("" if i == 1 else str(i))) for i in (1, 2, 3)]
           for side in ("left", "right")}
    if all(os.path.isfile(x) for xs in seq.values() for x in xs):
        # v3: the alternating sequence (GRADER_CHANGES[3])
        fol = ship_follows(seq)
        fol["v2_rule_on_first_pair"] = {k: v for k, v in ship_shift(
            seq["left"][0], seq["right"][0]).items() if k not in ("rule",)}
        ended = [f"ship_{s}" + ("" if i == 1 else str(i)) for i in (1, 2, 3)
                 for s in ("left", "right") if over(f"ship_{s}" + ("" if i == 1 else str(i)))]
        if ended:
            fol["not_measured"] = f"GAME OVER was showing during the sequence: {ended}"
        out.append(check("mousemove_moves_ship", None if ended else fol["pass"], "pixel", fol))
    elif os.path.isfile(p("ship_left")) and os.path.isfile(p("ship_right")):
        sh = ship_shift(p("ship_left"), p("ship_right"))
        sh["rule_version"] = "v2 (a browser.json without the v3 sequence)"
        out.append(check("mousemove_moves_ship", sh["pass"], "pixel", sh))
    d_pause = meanabs(p("pause_a"), p("pause_b")) if os.path.isfile(p("pause_b")) else None
    d_res = meanabs(p("resumed_a"), p("resumed_b")) if os.path.isfile(p("resumed_b")) else None
    paused_txt = "PAUSE" in txt("paused")
    before = over("after_move")
    ended_after_resume = (not over("paused")) and over("resumed")
    frozen = (d_pause is not None and d_res is not None and d_pause < 0.3
              and (d_res >= 0.3 or ended_after_resume))
    ev = {"meanabs_while_paused": d_pause, "meanabs_after_resume": d_res,
          "pause_text": paused_txt, "game_over_before_esc": before,
          "game_over_after_resume": ended_after_resume,
          "threshold": "paused < 0.3 and resumed >= 0.3 (or the game ended after the "
                       "resume), or PAUSE text; not measured if GAME OVER showed before ESC"}
    out.append(check("esc_pause", None if before else (frozen or paused_txt),
                     "runtime+pixel", ev))
    pe = br.get("play_new_errors") or {}
    out.append(check("play_30s_no_errors", pe.get("console_errors", 1) == 0
                     and pe.get("page_errors", 1) == 0, "runtime", pe))
    au = (states.get("end") or states.get("after_play") or {}).get("audio") or {}
    out.append(check("web_audio_runtime", (au.get("osc_start", 0) + au.get("buf_start", 0)) > 0,
                     "runtime", au))
    out.append(hud_check(variant, br, s))
    if variant == "V4":
        out.append(octopus_pixel_look(shots, states))
        out.append(scrolls_downward(shots, states))
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
        score_ok = any(abs(f["x"] - 20) <= 2 and at_y40(f) for f in sc)
        level_ok = any(abs(f["x"] - f["cw"] / 2) <= 4 and f["align"] == "center" and at_y40(f)
                       for f in lv)
        combo_ok = any(f["x"] >= f["cw"] * 0.7 and at_y40(f) for f in co)
        # v3 (JUDGEMENT CALL, GRADER_CHANGES[3]): COMBO judged when drawn at all
        ok = score_ok and level_ok and (combo_ok or not co)
        return check("hud_positions", ok, "runtime",
                     {**ev, "score_ok": score_ok, "level_ok": level_ok,
                      "combo_drawn": bool(co), "combo_ok": combo_ok if co else None,
                      "rule": "SCORE x=20 y=40, LEVEL centered y=40, COMBO right "
                              "y=40, all 20px monospace (fillText args); COMBO judged "
                              "only when a COMBO text was drawn (v3)"})
    if variant in ("V1", "V2"):
        # v3: the DOM HUD right after the 30 s of play (v2: after the
        # game-over attempt, when a game-over screen may have replaced it)
        src = "dom_hud_play" if br.get("dom_hud_play") is not None else "dom_hud"
        d = br.get(src) or []

        def pick(word):
            return [e for e in d if e["t"].upper().startswith(word) and e["visible"]]
        sc, lv, co = pick("SCORE"), pick("LEVEL"), pick("COMBO")

        def y40(e):
            return e["y"] <= 40 <= e["y"] + e["h"] + 6 and e["font_size"] == "20px" \
                and "mono" in e["font_family"].lower()
        score_ok = any(abs(e["x"] - 20) <= 12 and y40(e) for e in sc)
        level_ok = any(abs(e["x"] + e["w"] / 2 - W / 2) <= 40 and y40(e) for e in lv)
        combo_ok = any(e["x"] + e["w"] >= W - 120 and y40(e) for e in co)
        ok = score_ok and level_ok and (combo_ok or not co)
        return check("hud_positions", ok, "runtime",
                     {"score": sc[:1], "level": lv[:1], "combo": co[:1], "source": src,
                      "score_ok": score_ok, "level_ok": level_ok, "combo_drawn": bool(co),
                      "combo_ok": combo_ok if co else None,
                      "rule": "DOM boxes: SCORE left 20, LEVEL centered, COMBO right, "
                              "y=40 inside the box, 20px monospace; COMBO judged only when "
                              "a visible COMBO element exists (v3)"})
    if variant == "V4":
        # v4: whichever the game used -- a DOM overlay (V1's rule) or a 2D
        # canvas (V0's rule); text inside the WebGL scene cannot be located
        dom, fill = hud_check("V1", br, s), hud_check("V0", br, s)
        found = {k: bool(c["evidence"]["score"] or c["evidence"]["level"])
                 for k, c in (("dom", dom), ("canvas_2d", fill))}
        ev = {"dom": dom["evidence"], "canvas_2d": fill["evidence"], "found": found}
        if dom["pass"] or fill["pass"]:
            return check("hud_positions", True, "runtime",
                         {**ev, "judged_by": "dom" if dom["pass"] else "canvas_2d"})
        if not any(found.values()) and s is not None and webgl_text(s):
            return check("hud_positions", None, "runtime",
                         {**ev, "not_measured": "no SCORE/LEVEL in the DOM or 2D canvas; "
                                                f"the HUD is WebGL text ({webgl_text(s)})"})
        return check("hud_positions", False, "runtime", ev)
    ok = s.has_str("score") and s.has_str("level") and s.has_num(40, 20)
    return check("hud_positions", ok, "static-parse",
                 "V3 text is glyph geometry: strings SCORE/LEVEL and numbers 20, 40 "
                 "in source only (positions not verified)")


# --------------------------------------------------------------------- main


def grade(gid: str, variant: str, run_dir: str, log_dir: str | None,
          do_runtime: bool, sfx: str = "", extra: dict | None = None) -> dict:
    """`gid` is `<run_id>` or, under the iterative protocol, `<run_id>@p<n>`
    (the state after prompt n); `sfx` names prompt n's log files ('' or
    '.p<n>', run.sfx_of). `extra` is merged into the row (a re-grade's
    provenance: `regrade`)."""
    t0 = time.time()
    run_id, _, pn = gid.partition("@p")
    prompt_no = int(pn) if pn.isdigit() else 1
    row: dict = {"grade_id": gid, "variant": variant, "run_dir": run_dir,
                 "prompt_no": prompt_no, "graded_at": t0, "grader": grader_info(),
                 **(extra or {})}
    if assisted_label(gid):
        row["assisted"] = assisted_label(gid)
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
    proj = project_root(run_dir, variant) if os.path.isdir(run_dir) else None
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
        if app.get("net_tag"):
            b["sandbox_net_stop"] = sandbox_net.down(app["net_tag"])
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
        row["runtime"]["determinism"] = br.get("determinism")
        if rt.get("shots_dir") and os.path.isdir(rt["shots_dir"]):
            # v3: per grader version and per grade row, never overwritten
            dst = shots_dest(gid, t0)
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(rt["shots_dir"]):
                shutil.copy2(os.path.join(rt["shots_dir"], fn), os.path.join(dst, fn))
            row["runtime"]["screenshots"] = dst
    row.update(headline(row.get("spec") or []))
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
    summary["failed"] = [c["check"] for c in row.get("spec") or [] if c.get("pass") is False
                         and not c.get("flaky")]
    summary["unknown"] = [c["check"] for c in row.get("spec") or [] if c.get("pass") is None]
    summary["flaky"] = row.get("spec_flaky")
    summary["assisted"] = row.get("assisted")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
