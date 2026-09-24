#!/usr/bin/env python
"""The iterative protocol's follow-up prompt, built DETERMINISTICALLY from a
grade (grade.py's row). No model writes it; the same grade gives the same
text, byte for byte.

    python bench/octopus/followup.py GRADE_ID     # print the follow-up for a grade

Why: the prompt's author needed about six prompts on his earlier 9B run
(repo prompts/, "4 iterations. 6 prompts"), and a human steering an agent
tells it what is broken. Here what is broken comes only from the grader: the
build's errors, the browser's console errors, a blank canvas, and the spec
checks that failed. Unknown checks (not measured) are never reported as
failures. Checks the grader can only guess at (`pixel` heuristics) say so.

Order: what stops the game from running first (build, load, console
errors, blank canvas), then the spec checks. Capped at MAX_ITEMS items and
MAX_CHARS characters, so a broken build does not bury the rest.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import variants  # noqa: E402

MAX_ITEMS = 12
MAX_CHARS = 6000

# Our words for each check: what the spec asks, as the fix to make.
WHAT = {
    "enemy_sizes_36_48_20_150": "enemy sizes must be small 36, medium 48, baby 20, boss 150",
    "enemy_types_4": "all four octopus types must exist: small, medium, baby, boss",
    "bullet_speeds_4_tiers": "the four weapon tiers need bullet speeds 8, 10, 12, 14",
    "boss_every_5_levels": "a boss octopus must appear every 5 levels",
    "lerp_0_35": "the ship follows the mouse with lerp factor 0.35",
    "circle_collision": "collisions use the circle-to-circle distance check "
                        "(dx*dx + dy*dy < (r1+r2)*(r1+r2))",
    "pixel_grids_ge_4": "every octopus type is drawn from its own pixel grid of 0s and 1s",
    "hard_pixels": "pixel art needs hard pixels (no smoothing / nearest filtering)",
    "bg_color_0D1117_source": "the space background color is #0D1117",
    "bg_color_runtime": "the rendered background is not #0D1117",
    "unleash_mode": "UNLEASH MODE (the glowing orb power-up) is missing",
    "combo_counter": "the COMBO counter is missing",
    "restart_text": "the game over screen needs \"CLICK TO RESTART\"",
    "web_audio_static": "sounds must be procedural Web Audio (AudioContext, oscillators)",
    "web_audio_runtime": "no sound played while shooting for 30 seconds",
    "readme_run_3001": "the README must say how to run it on port 3001",
    "monospace_font": "all UI text uses a monospace font",
    "particles_draw_ctx": "particles.draw(ctx) is never called in the main draw loop",
    "no_libraries": "no libraries or external scripts are allowed",
    "r3f_canvas_useFrame": "the game must render in an @react-three/fiber <Canvas> driven by useFrame",
    "instanced_or_points": "particles must be drawn through an instancedMesh or points",
    "three_flatland_sprites": "game objects must be three-flatland Sprite2D in SpriteGroups",
    "webgpu_renderer": "the renderer must be three's WebGPURenderer from 'three/webgpu'",
    "glyph_used": "all text must be rendered with @pmndrs/glyph",
    "package_pins_exact": "package.json must list exactly the pinned packages and versions",
    "no_console_errors_15s": "the page logs errors in its first 15 seconds",
    "canvas_not_blank": "the screen is blank on load: nothing is drawn",
    "start_screen": "the start screen must show \"OCTOPUS INVADERS\" and \"CLICK TO START\"",
    "click_starts_game": "clicking the start screen does not start the game",
    "mousemove_moves_ship": "moving the mouse left/right did not move the ship "
                            "(checked from screenshots; a heuristic)",
    "esc_pause": "ESC does not pause the game (the screen keeps changing while paused)",
    "play_30s_no_errors": "errors are thrown during 30 seconds of play",
    "hud_positions": "HUD layout: SCORE at x=20,y=40; LEVEL centered at y=40; COMBO at the "
                     "right edge at y=40; 20px monospace",
}
BLOCKERS = ("no_console_errors_15s", "canvas_not_blank", "click_starts_game",
            "play_30s_no_errors")


def _dedup(msgs: list[str], n: int) -> list[str]:
    out, seen = [], set()
    for m in msgs:
        k = m.strip()[:160]
        if k and k not in seen:
            seen.add(k)
            out.append(m.strip()[:300])
        if len(out) >= n:
            break
    return out


def build(variant: str, row: dict) -> str:
    items: list[str] = []
    b = row.get("build") or {}
    ts = variants.VARIANTS[variant]["ts"]
    if row.get("project") is None:
        items.append("There is no index.html in /workspace: the project was not created "
                     "(expected /workspace/space-shooter/).")
    if b.get("files_missing"):
        items.append("Missing files from the required structure: "
                     + ", ".join(b["files_missing"][:20]))
    if ts:
        for step, label in (("npm_install", "npm install"), ("tsc", "tsc --noEmit"),
                            ("build", "npm run build")):
            s = b.get(step) or {}
            if s.get("rc") not in (0, None):
                tail = "\n".join(ln for ln in (s.get("tail") or "").splitlines()
                                 if "error" in ln.lower())[-1200:]
                items.append(f"`{label}` fails (exit {s.get('rc')}"
                             + (f", {s.get('error_ts')} TypeScript errors" if s.get("error_ts") else "")
                             + "):\n" + (tail or (s.get("tail") or "")[-800:]))
    elif b.get("html_refs_missing"):
        items.append("index.html loads files that do not exist: "
                     + ", ".join(b["html_refs_missing"]))
    rt = row.get("runtime") or {}
    errs = [e.get("text", "") for e in (rt.get("page_errors") or [])]
    errs += [e.get("text", "") for e in (rt.get("console_errors") or [])
             if "favicon" not in e.get("text", "")]
    errs += [f"HTTP {e['status']} for {e['url'].replace('http://localhost:3001', '')}"
             for e in (rt.get("requests_failed") or []) if e.get("status")
             and "favicon" not in e.get("url", "")]
    errs = _dedup(errs, 6)
    if errs:
        items.append("Errors in the browser console:\n" + "\n".join(f"  {e}" for e in errs))
    failed = {c["check"]: c for c in row.get("spec") or [] if c.get("pass") is False}
    for name in BLOCKERS:
        if name in failed and not (name in ("no_console_errors_15s", "play_30s_no_errors")
                                   and errs):
            items.append(WHAT[name])
    for name, c in failed.items():
        if name in BLOCKERS:
            continue
        items.append(WHAT.get(name, name))
    if not items:
        return ""
    items = items[:MAX_ITEMS]
    serve = ("npm install, then npm run build with zero TypeScript errors, then serve with "
             "npm run dev (port 3001)" if ts else "serve with python3 -m http.server 3001")
    text = ("I ran the game in /workspace/space-shooter and checked it against the spec. "
            "These are not working yet:\n\n"
            + "\n".join(f"{i}. {it}" for i, it in enumerate(items, 1))
            + "\n\nFix them without breaking what already works, and keep following the "
              f"original spec. When done: {serve}.")
    return text[:MAX_CHARS]


def main() -> int:
    gid = sys.argv[1]
    row = None
    for ln in open(os.path.join(HERE, "results", "grades.jsonl"), encoding="utf-8"):
        r = json.loads(ln)
        if r.get("grade_id") == gid:
            row = r
    if row is None:
        raise SystemExit(f"no grade {gid}")
    print(build(row["variant"], row) or "(no failures: no follow-up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
