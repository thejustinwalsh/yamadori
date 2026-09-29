#!/usr/bin/env python
"""The Octopus grader's static checks, on small projects written here. No
Docker, no browser, no GPU.

    python bench/octopus/test_grade_checks.py      -> "N/M checks passed"

Grader v2 (2026-09-26, SELF-IMPROVEMENT-LOG #57): `particles_draw_ctx` was
`re.fullmatch(r"(this\\.)?particle\\w*\\.draw", callee)` -- case-sensitive, so
v0f@p1's `Particles.draw(ctx)` in Game.draw() failed, and followup.py sent it
back as prompt 2's item 2 ("particles.draw(ctx) is never called in the main
draw loop"); v0b's `ParticleSystem.draw(ctx)` and v0e's `PARTICLES.draw(ctx)`
failed the same way. The spec's intent is "particles.draw(ctx) MUST be called
in the game's main draw loop": any particle-named object, any case, called
from the game (not from particles.js itself). The audit of the other static
checks found the same class of defect in five more (grade.GRADER_CHANGES).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grade as G  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def project(files: dict[str, str]) -> str:
    d = tempfile.mkdtemp(prefix="octo-grade-")
    for rel, text in files.items():
        p = os.path.join(d, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    return d


def static(files: dict[str, str], variant: str = "V0") -> dict[str, dict]:
    d = project(files)
    try:
        return {c["check"]: c for c in G.spec(variant, d, {}, {"ran": False, "why": "test"})}
    finally:
        shutil.rmtree(d, ignore_errors=True)


PARTICLES_JS = """const Particles = { list: [], draw(ctx) { for (const p of this.list) {
  ctx.fillRect(p.x, p.y, 2, 2); } } };
// NOTE: game.js MUST call particles.draw(ctx) in the main loop
"""


def game(draw_line: str) -> str:
    return ("const Game = {\n  draw() {\n    const ctx = this.ctx;\n    " + draw_line
            + "\n  },\n  loop() { this.draw(); requestAnimationFrame(() => this.loop()); }\n};\n")


def test_particles_draw_ctx():
    cases = [
        ("Particles.draw(ctx) (v0f)", "Particles.draw(ctx);", True),
        ("particles.draw(ctx) (the spec's own spelling)", "particles.draw(ctx);", True),
        ("particleSystem.draw(ctx) (reference fixture)", "particleSystem.draw(ctx);", True),
        ("ParticleSystem.draw(ctx) (v0b)", "ParticleSystem.draw(ctx);", True),
        ("PARTICLES.draw(ctx) (v0e)", "PARTICLES.draw(ctx);", True),
        ("this.particles.draw(ctx)", "this.particles.draw(ctx);", True),
        ("game.particleSystem.draw(ctx)", "game.particleSystem.draw(ctx);", True),
        ("window.Particles?.draw(ctx)", "window.Particles?.draw(ctx);", True),
        ("particles[i].draw(ctx)", "for (let i = 0; i < n; i++) particles[i].draw(ctx);", True),
        ("missing: no particle draw call", "Enemy.draw(ctx); Player.draw(ctx);", False),
        ("missing: only a comment names it", "// Particles.draw(ctx)", False),
        ("missing: a string names it", "console.log('Particles.draw(ctx)');", False),
        ("not the method the spec names: Particles.update(ctx)", "Particles.update(ctx);", False),
    ]
    for name, line, want in cases:
        c = static({"js/particles.js": PARTICLES_JS, "js/game.js": game(line)})["particles_draw_ctx"]
        check(c["pass"] is want, f"particles_draw_ctx: {name} -> {want}", str(c["evidence"])[:300])
    # where: a call only inside particles.js is not the game's main draw loop
    c = static({"js/particles.js": PARTICLES_JS + "Particles.draw(null);\n",
                "js/game.js": game("Player.draw(ctx);")})["particles_draw_ctx"]
    check(c["pass"] is False and c["evidence"]["only_in_particles_js"],
          "particles_draw_ctx: a call only inside particles.js fails", str(c["evidence"]))
    c = static({"js/particles.js": PARTICLES_JS, "js/game.js": game("Particles.draw(ctx);")})
    ev = c["particles_draw_ctx"]["evidence"]["calls"]
    check(ev and ev[0]["file"] == "js/game.js" and ev[0]["in"] == "draw" and ev[0]["line"] == 4,
          "particles_draw_ctx: evidence names the file, line and enclosing function", str(ev))


def test_other_v2_checks():
    base = {"js/particles.js": PARTICLES_JS, "js/game.js": game("Particles.draw(ctx);")}

    def one(name, extra, want):
        c = static({**base, **extra})
        return c[name]

    # bullet speeds: any case, key suffix
    for label, src, want in [
        ("speed: (v1 form)", "const T = [{speed: 8}, {speed: 10}, {speed: 12}, {speed: 14}];", True),
        ("bulletSpeed:", "const T = [{bulletSpeed: 8}, {bulletSpeed: 10}, {bulletSpeed: 12}, "
                         "{bulletSpeed: 14}];", True),
        ("BULLET_SPEED:", "const T = {t1: {BULLET_SPEED: 8}, t2: {BULLET_SPEED: 10}, "
                          "t3: {BULLET_SPEED: 12}, t4: {BULLET_SPEED: 14}};", True),
        ("bare array", "const S = [8, 10, 12, 14];", True),
        ("wrong speeds", "const T = [{bulletSpeed: 8}, {bulletSpeed: 9}, {bulletSpeed: 12}];", False),
    ]:
        c = one("bullet_speeds_4_tiers", {"js/config.js": src}, want)
        check(c["pass"] is want, f"bullet_speeds_4_tiers: {label} -> {want}")
    # boss every 5: declaration, key, assignment (v2), modulo
    for label, src, want in [
        ("const BOSS_EVERY = 5", "const BOSS_EVERY = 5;", True),
        ("this.bossInterval = 5 (v2)", "class G { init() { this.bossInterval = 5; } }", True),
        ("level % 5", "if (level % 5 === 0) spawnBoss();", True),
        ("boss every 4", "this.bossInterval = 4;", False),
    ]:
        c = one("boss_every_5_levels", {"js/config.js": src}, want)
        check(c["pass"] is want, f"boss_every_5_levels: {label} -> {want}")
    # circle collision
    for label, src, want in [
        ("dx*dx + dy*dy", "const hit = dx*dx + dy*dy < r*r;", True),
        ("this.dx*this.dx + this.dy*this.dy (v2)",
         "const hit = this.dx*this.dx + this.dy*this.dy < r*r;", True),
        ("dx**2 + dy**2 (v2)", "const hit = dx ** 2 + dy ** 2 < r * r;", True),
        ("Math.pow (v2)", "const hit = Math.pow(dx, 2) + Math.pow(dy, 2) < r * r;", True),
        ("Math.hypot", "const hit = Math.hypot(dx, dy) < r;", True),
        ("bounding box only", "const hit = a.x < b.x + b.w && a.x + a.w > b.x;", False),
    ]:
        c = one("circle_collision", {"js/game2.js": src}, want)
        check(c["pass"] is want, f"circle_collision: {label} -> {want}")
    # pixel grids
    row01 = '"0110110110"'
    rowpal = '"0012344321"'
    g01 = "[" + ",".join([row01] * 8) + "]"
    gpal = "[" + ",".join([rowpal] * 8) + "]"
    g2d = "[" + ",".join(["[0,1,2,1,0,1]"] * 6) + "]"
    for label, src, want in [
        ("four 0/1 string grids", "\n".join(f"const G{i} = {g01};" for i in range(4)), True),
        ("four palette-digit string grids (v2, the judgement call)",
         "\n".join(f"const G{i} = {gpal};" for i in range(4)), True),
        ("four 6x6 2D palette literals (v2)", "\n".join(f"const G{i} = {g2d};" for i in range(4)),
         True),
        ("three grids", "\n".join(f"const G{i} = {g01};" for i in range(3)), False),
        ("letters are not cells", "const G = [" + ",".join(['"abcdefgh"'] * 8) + "];", False),
    ]:
        c = one("pixel_grids_ge_4", {"js/enemies.js": src}, want)
        check(c["pass"] is want, f"pixel_grids_ge_4: {label} -> {want}", str(c["evidence"]))
    # background color in CSS; monospace in any case
    c = one("bg_color_0D1117_source", {"css/styles.css": "body { background: #0d1117; }"}, True)
    check(c["pass"] is True, "bg_color_0D1117_source: #0D1117 in CSS counts (v2)")
    c = one("bg_color_0D1117_source", {"css/styles.css": "body { background: #000; }"}, False)
    check(c["pass"] is False, "bg_color_0D1117_source: absent everywhere fails")
    c = one("monospace_font", {"css/styles.css": "canvas { font-family: Monospace; }"}, True)
    check(c["pass"] is True, "monospace_font: 'Monospace' in CSS counts (v2)")
    c = one("monospace_font", {"css/styles.css": "canvas { font-family: serif; }"}, False)
    check(c["pass"] is False, "monospace_font: no monospace anywhere fails")


def test_version_and_regrade_plan():
    info = G.grader_info()
    check(info["version"] == G.GRADER_VERSION >= 3 and info["changes"]
          and {c["check"] for c in G.GRADER_CHANGES[2]} >= {"particles_draw_ctx"}
          and {"mousemove_moves_ship", "esc_pause", "hud_positions"}
          <= {c["check"] for c in G.GRADER_CHANGES[3]},
          "every row's grader field: {path, version, changes}; v2 and v3 changes kept",
          str(info)[:200])
    check(G.GRADER_VERSION >= 4 and {"r3f_v10_canvas", "builds_and_serves", "octopus_pixel_look",
                                     "scrolls_downward"} <= {c["check"].split(" ")[0]
                                                              for c in G.GRADER_CHANGES[4]},
          "v4: the V4 checks are listed with their reasons")
    check(all(c.get("spec") and c.get("why") for c in G.GRADER_CHANGES[4]),
          "v4: every change names the spec text and why")
    check(all(c.get("spec") for v in G.GRADER_CHANGES.values() for c in v),
          "every change (every version) names the spec text behind it")
    check(set((info.get("sha256_16") or {})) == {"grade.py", "browser_check.py"}
          and all(info["sha256_16"].values()),
          "v3: every row records which grade.py and browser_check.py graded it")
    # PROTOCOL: a grader change is applied to EVERY graded state. Any grade id
    # whose rows are all from an older grader must be in regrade.PLAN.
    import regrade
    planned = {p[0] for p in regrade.PLAN}
    by_id: dict[str, set] = {}
    for r in G.jsonl(G.GRADES):
        g = r.get("grader")
        v = g.get("version") if isinstance(g, dict) else 1
        by_id.setdefault(r.get("grade_id"), set()).add(v)
    stale = sorted(gid for gid, vs in by_id.items()
                   if max(vs) < G.GRADER_VERSION and gid not in planned)
    check(not stale, "every graded id with only old-grader rows is in regrade.PLAN", str(stale))


V4_PKG = json.dumps({"name": "space-shooter", "private": True, "type": "module",
                     "scripts": {"build": "vite build"},
                     "dependencies": {"@react-three/fiber": "10.0.0-alpha.5", "react": "19.2.8",
                                      "three": "0.185.1", "koota": "0.6.6", "math": "0.1.0"}})
V4_OK = {
    "package.json": V4_PKG,
    "src/App.jsx": """import { Canvas } from '@react-three/fiber'
import { WorldProvider } from 'koota/react'
import { world } from './world'
import { Scene } from './Scene'
export function App() {
  return <WorldProvider world={world}><Canvas orthographic><Scene /></Canvas></WorldProvider>
}
""",
    "src/world.js": """import { createWorld, trait } from 'koota'
export const Position = trait({ x: 0, y: 0 })
export const Velocity = trait({ x: 0, y: 0 })
export const world = createWorld()
""",
    "src/Scene.jsx": """import { useFrame } from '@react-three/fiber'
import { vec2 } from 'math'
import { world, Position, Velocity } from './world'
export function Scene() {
  useFrame((_, delta) => {
    world.query(Position, Velocity).updateEach(([p, v]) => { p.x += v.x * delta })
  })
  return null
}
export const near = (a, b, r) => vec2.distance(a, b) < r
""",
}


def test_v4_stack_checks():
    """V4 (grader v4, 2026-09-27): what the operator's sentence asks -- r3f
    v10 with a Canvas, koota, pmndrs math IMPORTED AND USED -- and nothing it
    does not (no WebGPU canvas, TSL, pins, useFrame or instancing)."""
    got = static(V4_OK, "V4")
    for name in ("r3f_v10_canvas", "koota_world_traits_queries", "math_used"):
        check(got.get(name, {}).get("pass") is True, f"V4 {name}: a project using the "
              "stack passes", str(got.get(name)))
    gone = {"webgpu_canvas", "tsl_node_material", "package_pins_exact", "r3f_canvas_useFrame",
            "instanced_or_points", "hard_pixels", "no_libraries", "particles_draw_ctx"}
    check(not gone & set(got), "V4 carries none of the checks the prompt no longer asks for",
          str(sorted(gone & set(got))))
    v1 = static(V4_OK, "V1")
    check({"r3f_canvas_useFrame", "instanced_or_points", "package_pins_exact", "hard_pixels"}
          <= set(v1) and "r3f_v10_canvas" not in v1 and "particles_rendered" not in v1,
          "V1 keeps its v3 checks; the V4 checks are V4's only")

    def r3f(pkg_deps, b=None, files=None):
        f = dict(V4_OK, **(files or {}))
        f["package.json"] = json.dumps({"name": "x", "dependencies": pkg_deps})
        d = project(f)
        try:
            s = G.Src(d)
            return {c["check"]: c for c in G.v4_stack_checks(s, d, b)}
        finally:
            shutil.rmtree(d, ignore_errors=True)
    for label, deps, b, want in [
        ("10.0.0-alpha.5 declared", {"@react-three/fiber": "10.0.0-alpha.5"}, None, True),
        ("^10.0.0-alpha.5 declared", {"@react-three/fiber": "^10.0.0-alpha.5"}, None, True),
        ("^9.8.1 declared (v9, the npm `latest`)", {"@react-three/fiber": "^9.8.1"}, None, False),
        ("the `alpha` dist-tag, npm installed 10.0.0-alpha.5", {"@react-three/fiber": "alpha"},
         {"npm_ls": {"@react-three/fiber": "10.0.0-alpha.5"}}, True),
        ("the `alpha` dist-tag, not built (no major known)", {"@react-three/fiber": "alpha"},
         None, False),
        ("^10 declared but npm installed 9.8.1: the install wins",
         {"@react-three/fiber": "^10.0.0-alpha.5"}, {"npm_ls": {"@react-three/fiber": "9.8.1"}},
         False),
        ("not in package.json", {"react": "19.2.8"}, None, False),
    ]:
        c = r3f(deps, b)["r3f_v10_canvas"]
        check(c["pass"] is want, f"r3f_v10_canvas: {label} -> {want}", str(c["evidence"]))
    c = r3f({"@react-three/fiber": "10.0.0-alpha.5"}, None, {
        "src/App.jsx": "import { Canvas } from '@react-three/fiber'\nexport const A = 1\n"})
    check(c["r3f_v10_canvas"]["pass"] is False, "r3f_v10_canvas: Canvas imported, never used "
          "-> fails", str(c["r3f_v10_canvas"]["evidence"]))
    c = r3f({"@react-three/fiber": "10.0.0-alpha.5"}, None, {
        "src/Scene.jsx": V4_OK["src/Scene.jsx"].replace(
            "export const near = (a, b, r) => vec2.distance(a, b) < r\n", "")})
    check(c["math_used"]["pass"] is False
          and c["math_used"]["evidence"]["names_used"] == {"vec2": 0},
          "math_used: `math` imported but no imported name used -> fails",
          str(c["math_used"]["evidence"]))
    c = r3f({"@react-three/fiber": "10.0.0-alpha.5"}, None, {
        "src/rand.js": "import * as R from 'math/random'\nexport const g = R.mulberry32.create(1)\n"})
    check(c["math_used"]["pass"] is True and c["math_used"]["evidence"]["names_used"].get("R") == 1,
          "math_used: a namespace import of a subpath, used", str(c["math_used"]["evidence"]))
    c = r3f({"@react-three/fiber": "10.0.0-alpha.5"}, None, {
        "src/world.js": "export const state = { ships: [] }\n"})
    check(c["koota_world_traits_queries"]["pass"] is False, "koota: no koota world -> fails")
    # a no-build project: an import map names the CDN versions
    page = {"index.html": '<script type="importmap">{"imports": {"@react-three/fiber": '
                          '"https://esm.sh/@react-three/fiber@10.0.0-alpha.5", "math/": '
                          '"https://esm.sh/math@0.1.0/"}}</script>'
                          '<script type="module" src="./main.js"></script>',
            "main.js": "import { Canvas } from '@react-three/fiber'\n"
                       "import { createWorld, trait } from 'https://esm.sh/koota@0.6.6'\n"
                       "import { mulberry32 } from 'https://esm.sh/math@0.1.0/random'\n"
                       "const w = createWorld(); const T = trait({}); w.query(T)\n"
                       "export const r = mulberry32.create(1); export const C = Canvas\n"}
    d = project(page)
    try:
        c = {x["check"]: x for x in G.v4_stack_checks(G.Src(d), d, None)}
    finally:
        shutil.rmtree(d, ignore_errors=True)
    check(all(c[k]["pass"] for k in ("r3f_v10_canvas", "koota_world_traits_queries",
                                      "math_used"))
          and c["r3f_v10_canvas"]["evidence"]["declared_from"] == "url",
          "an import-map project (no package.json): versions from the URLs, CDN imports count",
          str({k: v["evidence"] for k, v in c.items()})[:600])


def test_v4_game_checks():
    """V4's game checks judge intent, not the canvas API or a file name."""
    base = dict(V4_OK)
    for label, extra, want in [
        ("a particle-named .draw(ctx) on a 2D overlay", {"src/fx.js":
         "export function frame(ctx, particleSystem) { particleSystem.draw(ctx) }\n"}, True),
        ("a <Particles /> component mounted", {"src/Game.jsx":
         "import { Particles } from './particles'\nexport const G = () => <Particles />\n"}, True),
        ("instance data written in the particle file", {"src/particles.js":
         "export function sync(mesh, particles, m) { particles.forEach((p, i) => "
         "mesh.setMatrixAt(i, m)); mesh.instanceMatrix.needsUpdate = true }\n"}, True),
        ("a particle pool nobody draws", {"src/particles.js":
         "export const particles = []\nexport function spawn(p) { particles.push(p) }\n"}, False),
    ]:
        c = static({**base, **extra}, "V4")["particles_rendered"]
        check(c["pass"] is want, f"particles_rendered: {label} -> {want}", str(c["evidence"]))
    c = static({**base, "src/hit.js": "import { vec2 } from 'math'\n"
                "export const hit = (a, b, r) => vec2.squaredDistance(a, b) < r * r\n"}, "V4")
    check(c["circle_collision"]["pass"] is True
          and c["circle_collision"]["evidence"]["distance_calls"] == ["vec2.distance",
                                                                        "vec2.squaredDistance"],
          "circle_collision V4: a vector library's distance call counts",
          str(c["circle_collision"]["evidence"]))
    c0 = static({"js/particles.js": PARTICLES_JS, "js/game.js": game("Particles.draw(ctx);"),
                 "js/hit.js": "const hit = (a, b) => vec2.distance(a, b) < 3;\n"}, "V0")
    check(c0["circle_collision"]["pass"] is False, "circle_collision V0: unchanged (a distance "
          "call is not the spec's form there)")
    # builds_and_serves: whatever layout the model chose
    ok_npm = {"layout": "npm", "stage": "serving static", "serve_mode": "static",
              "npm_install": {"rc": 0}, "tsc": {"rc": None}, "build": {"rc": 0}}
    for label, b, want in [
        ("npm: install 0, build 0, dist served (no tsc)", ok_npm, True),
        ("npm: the build fails, vite dev serves", {**ok_npm, "build": {"rc": 1},
                                                   "serve_mode": "vite dev"}, False),
        ("static: every referenced file exists", {"layout": "static", "stage": "serving static",
                                                  "serve_mode": "static", "serve_ok": True}, True),
        ("static: a referenced file is missing", {"layout": "static", "stage": "serving static",
                                                  "serve_mode": "static", "serve_ok": False}, False),
        ("neither package.json nor index.html", {"layout": "none", "stage": "failed"}, False),
        ("the build was not run", {}, None),
        ("the sandbox network could not start", {"error": "sandbox network: x"}, None),
    ]:
        c = G.builds_and_serves(b)
        check(c["pass"] is want, f"builds_and_serves: {label} -> {want}", str(c["evidence"]))
    d = project({"space-shooter/package.json": "{}", "space-shooter/public/index.html": "x"})
    e = project({"space-shooter/index.html": "x"})
    try:
        check(G.build_mode("V4", os.path.join(d, "space-shooter")) == "npm"
              and G.build_mode("V4", os.path.join(e, "space-shooter")) == "js"
              and G.build_mode("V1", e) == "ts" and G.build_mode("V0", d) == "js",
              "build_mode: V4 npm with a package.json, js without; V0 js and V1 ts as before")
        check(G.project_root(d, "V4") == os.path.join(d, "space-shooter")
              and G.project_root(d) == os.path.join(d, "space-shooter", "public"),
              "project_root: V4 takes the package.json's folder; V0-V3 unchanged")
        fr = G.files_and_refs("V4", os.path.join(d, "space-shooter"))
        check(fr["layout"] == "npm" and fr["files_expected"] == 0 and G.EXPECTED["V4"] == [],
              "V4 expects no file list; the layout is recorded", str(fr))
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(e, ignore_errors=True)


def test_v4_prompt_is_one_sentence():
    """V4 = the original with ONLY its one-line tech statement replaced by
    the operator's sentence (2026-09-27). Built from the fetched original
    only when it is present -- a missing original is SKIPPED, not passed."""
    import variants as V
    check(V.V4_TECH == "build a space shooter game with r3f (react-three-fiber) v10 and Koota and pmndrs math. multi-file project structure."
          and V.VARIANTS["V4"]["pairs"] == [(V.O_TECH, V.V4_TECH)]
          and V.VERSIONS["V4"] == {} and V.VARIANTS["V4"]["ts"] is None,
          "V4 is one pair (the tech line -> the operator's sentence), no pins, ts decided "
          "by the project")
    check(not any(hasattr(V, n) for n in ("stack_rules", "_V4_STRUCTURE", "_V4_CANVAS")),
          "our stack rules, layout and canvas block are gone")
    if not os.path.isfile(V.ORIGINAL):
        print("  SKIPPED the V4 build: the original is not fetched here")
        return
    orig = V.fetch()
    text, meta = V.build(orig, "V4")
    a, b = orig.split("\n"), text.split("\n")
    changed = [(i, x, y) for i, (x, y) in enumerate(zip(a, b), 1) if x != y]
    check(len(a) == len(b) and len(changed) == 1 and changed[0][0] == 1
          and changed[0][2] == V.V4_TECH and meta["lines_kept_verbatim"] == len(a) - 1,
          "V4 differs from V0 in line 1 only", str(changed)[:300])
    check(all(w in text for w in ("PROJECT STRUCTURE", "ctx.fillRect", "requestAnimationFrame",
                                  "in game.js init()", "python3 -m http.server 3001")),
          "every canvas-specific line stays (adapting it is part of the test)")


def _grid_sprite(a, x, y, cell, col):
    grid = ["0011111100", "0111111110", "1101111011", "1111111111",
            "1111111111", "0110110110", "1100000011", "1010000101"]
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == "1":
                a[y + r * cell:y + (r + 1) * cell, x + c * cell:x + (c + 1) * cell] = col


def test_v4_pixel_checks():
    import numpy as np
    from PIL import Image, ImageDraw
    d = tempfile.mkdtemp(prefix="octo-v4px-")
    try:
        a = np.zeros((720, 1280, 3), dtype="uint8")
        a[:] = BG
        _grid_sprite(a, 200, 100, 5, (255, 105, 180))
        _grid_sprite(a, 500, 100, 5, (0, 170, 255))
        _write(d, "start", a)
        c = G.octopus_pixel_look(d, {})
        check(c["pass"] is True and c["evidence"]["best_frame"] == "start",
              "octopus_pixel_look: two pixel-grid octopi -> passes", str(c["evidence"])[:300])
        im = Image.new("RGB", (1280, 720), BG)
        dr = ImageDraw.Draw(im)
        for x0, col in ((200, (255, 105, 180)), (500, (0, 170, 255))):
            dr.ellipse((x0, 100, x0 + 50, 140), fill=col)
            dr.ellipse((x0 + 5, 120, x0 + 45, 150), fill=col)
        big = im.resize((2560, 1440)).resize((1280, 720), Image.LANCZOS)
        _write(d, "start", np.asarray(big))
        c = G.octopus_pixel_look(d, {})
        check(c["pass"] is False, "octopus_pixel_look: smooth antialiased octopi -> fails",
              str(c["evidence"])[:300])
        os.remove(os.path.join(d, "start.png"))
        check(G.octopus_pixel_look(d, {})["pass"] is None, "octopus_pixel_look: no screenshot "
              "-> not measured")
        # scrolling: a star field moved by dy between the frames of each pair
        rng = np.random.default_rng(3)
        stars = [(int(rng.integers(40, 680)), int(rng.integers(0, 1280)), int(rng.integers(3, 7)))
                 for _ in range(3000)]

        def field(off):
            f = np.zeros((720, 1280, 3), dtype="uint8")
            f[:] = BG
            for y, x, sz in stars:
                yy = (y + off) % 720
                f[yy:yy + sz, x:x + sz] = (230, 230, 240)
            return f
        for label, dy, want in (("stars move down 10 px", 10, True),
                                ("stars move UP 10 px", -10, False),
                                ("a static background", 0, False)):
            # a frame's place in its chain (left, left2, left3 ...) sets its offset
            for name, (base, k) in {"ship_left": (0, 0), "ship_left2": (0, 1),
                                    "ship_left3": (0, 2), "ship_right": (100, 0),
                                    "ship_right2": (100, 1), "ship_right3": (100, 2),
                                    "resumed_a": (200, 0), "resumed_b": (200, 1)}.items():
                _write(d, name, field(base + k * dy))
            c = G.scrolls_downward(d, {})
            check(c["pass"] is want, f"scrolls_downward: {label} -> {want}",
                  str({k: c["evidence"][k] for k in ("down", "up", "sideways")}))
        over = {p: {"game_over_visible": True} for pr in G.SCROLL_PAIRS for p in pr}
        c = G.scrolls_downward(d, over)
        check(c["pass"] is None and len(c["evidence"]["skipped_game_over"]) == 5,
              "scrolls_downward: GAME OVER in every pair -> not measured")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _src(files):
    d = project(files)
    try:
        return G.Src(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_v4_hud_and_runtime_list():
    def box(t, x, w=120):
        return {"t": t, "x": x, "y": 22, "w": w, "h": 24, "font_size": "20px",
                "font_family": "monospace", "visible": True}
    dom = {"dom_hud_play": [box("SCORE 0", 20), box("LEVEL 1", 580)], "viewport": [1280, 720]}
    c = G.hud_check("V4", dom, _src({"a.js": "x"}))
    check(c["pass"] is True and c["evidence"]["judged_by"] == "dom",
          "hud_positions V4: a DOM overlay HUD -> V1's rule", str(c["evidence"])[:200])
    fill = {"fill_all": [_fill("SCORE: 0", 20), _fill("LEVEL 1", 640, align="center")],
            "viewport": [1280, 720]}
    c = G.hud_check("V4", fill, _src({"a.js": "x"}))
    check(c["pass"] is True and c["evidence"]["judged_by"] == "canvas_2d",
          "hud_positions V4: a 2D-canvas HUD -> V0's rule")
    gl = _src({"a.jsx": "import { Text } from '@react-three/drei'\n"
                        "export const H = () => <Text>SCORE</Text>\n"})
    c = G.hud_check("V4", {"viewport": [1280, 720]}, gl)
    check(c["pass"] is None and "WebGL text" in c["evidence"]["not_measured"],
          "hud_positions V4: no DOM/canvas HUD and drei Text imported -> not measured")
    c = G.hud_check("V4", {"viewport": [1280, 720]}, _src({"a.js": "x"}))
    check(c["pass"] is False, "hud_positions V4: no HUD anywhere -> fails")
    got = static(V4_OK, "V4")
    check({"octopus_pixel_look", "scrolls_downward"} <= set(got)
          and got["octopus_pixel_look"]["pass"] is None,
          "V4 without a runtime lists its pixel checks as not run")


# ------------------------------------------------ v3: the runtime checks (#62)
# Synthetic 1280x720 frames written as PNGs; no browser. The pointer
# positions are browser_check's (30% / 70%); the ship is drawn where the
# spec's lerp puts it once converged: at the pointer.

BG = (13, 17, 23)


def _frame(ship_x=None, planet_dx=0, enemy=None, tint_right=False, text_pause=False):
    import numpy as np
    a = np.zeros((720, 1280, 3), dtype="uint8")
    a[:] = BG
    rng = np.random.default_rng(1)
    for _ in range(300):                      # fixed stars
        y, x = rng.integers(0, 720), rng.integers(0, 1280)
        a[y, x] = (255, 255, 255)
    yy, xx = np.ogrid[:720, :1280]
    for cx, cy, r, col in ((850 + planet_dx, 620, 45, (78, 200, 190)),
                           (380 + planet_dx // 2, 450, 30, (220, 60, 90))):
        a[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = col     # mouse-reactive parallax
    if enemy:
        ex, ey = enemy
        a[ey:ey + 40, ex:ex + 40] = (255, 105, 180)
    if ship_x is not None:                    # 30x40 body, cyan edge, flame
        x0, y0 = int(ship_x) - 15, 577
        a[y0:y0 + 40, x0:x0 + 30] = (38, 50, 56)
        a[y0:y0 + 40, x0] = a[y0:y0 + 40, x0 + 29] = (78, 205, 196)
        a[y0, x0:x0 + 30] = (78, 205, 196)
        a[y0 + 40:y0 + 48, x0 + 10:x0 + 20] = (255, 170, 0)
    if tint_right:                            # a translucent foreground layer (v0f)
        a[:, 640:] = (a[:, 640:] * 0.7 + np.array((90, 50, 140)) * 0.3).astype("uint8")
    if text_pause:
        a[300:340, 560:720] = (255, 255, 255)
    return a


def _write(d, name, arr):
    from PIL import Image
    Image.fromarray(arr).save(os.path.join(d, name + ".png"))
    return os.path.join(d, name + ".png")


def _sequence(d, ship_at, **kw):
    """ship_at(side, i) -> the ship's x in shot i (1..3) with the pointer at
    side; returns the {"left": [...], "right": [...]} ship_follows takes."""
    out = {"left": [], "right": []}
    for i in (1, 2, 3):
        for side, px in (("left", 384), ("right", 896)):
            enemy = (px - 20, 420 + 60 * i) if (i == 2 and side == "left") else None
            name = f"ship_{side}" + ("" if i == 1 else str(i))
            out[side].append(_write(d, name, _frame(
                ship_at(side, i), planet_dx=(12 if side == "right" else -12),
                enemy=enemy, **kw)))
    return out


def test_ship_follows():
    d = tempfile.mkdtemp(prefix="octo-follow-")
    try:
        px = {"left": 384, "right": 896}
        cases = [
            ("the ship follows the pointer (parallax planets, a passing enemy)",
             lambda side, i: px[side], {}, True),
            ("the ship follows, its right window behind a translucent layer (v0f)",
             lambda side, i: px[side], {"tint_right": True}, True),
            ("the ship never moves (parallax still reacts): negative control",
             lambda side, i: 640, {}, False),
            ("the ship stays at the left position whatever the pointer does",
             lambda side, i: 384, {}, False),
            ("no ship at all (dead): negative", lambda side, i: None, {}, False),
            ("the ship only follows to the left", lambda side, i: 384 if side == "left" else 640,
             {}, False),
        ]
        for label, at, kw, want in cases:
            r = G.ship_follows(_sequence(d, at, **kw))
            check(r["pass"] is want, f"ship_follows: {label} -> {want}",
                  str(r["track"]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _runtime_rt(d, states: dict, pause_frozen=True, resume_moves=True, ship=True):
    """A browser run as browser_check v3 writes it: shots + browser.json's
    states (only what spec() reads)."""
    px = {"left": 384, "right": 896}
    _sequence(d, (lambda side, i: px[side]) if ship else (lambda side, i: None))
    for n in ("start", "after_click"):
        _write(d, n, _frame(640))
    _write(d, "pause_a", _frame(640))
    _write(d, "pause_b", _frame(640) if pause_frozen else _frame(700, planet_dx=30))
    _write(d, "resumed_a", _frame(640))
    _write(d, "resumed_b", _frame(700, planet_dx=30) if resume_moves else _frame(640))
    base = {k: {"dom_text": "", "fill_recent": [], "game_over_visible": False}
            for k in ("start", "after_click", "ship_left", "ship_right", "ship_left2",
                      "ship_right2", "ship_left3", "ship_right3", "after_move", "paused",
                      "resumed", "after_play", "end")}
    base["start"]["fill_recent"] = ["OCTOPUS INVADERS", "CLICK TO START"]
    for k, v in states.items():
        base[k].update(v)
    br = {"states": base, "events": [{"t": 0.1, "name": "loaded"}], "console": [],
          "page_errors": [], "play_new_errors": {"console_errors": 0, "page_errors": 0},
          "viewport": [1280, 720], "fill_all": [], "start_click": "center"}
    return {"ran": True, "browser": br, "shots_dir": d}


def _runtime_checks(states, **kw):
    d = tempfile.mkdtemp(prefix="octo-rt-")
    proj = project({"js/particles.js": PARTICLES_JS, "js/game.js": game("Particles.draw(ctx);")})
    try:
        return {c["check"]: c for c in G.spec("V0", proj, {}, _runtime_rt(d, states, **kw))}
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(proj, ignore_errors=True)


def test_runtime_rules_v3():
    GO = {"game_over_visible": True, "fill_recent": ["GAME OVER"]}
    c = _runtime_checks({})
    check(c["mousemove_moves_ship"]["pass"] is True and c["esc_pause"]["pass"] is True,
          "v3 spec(): a game that follows and pauses passes both",
          str((c["mousemove_moves_ship"]["evidence"].get("track"),
               c["esc_pause"]["evidence"])))
    c = _runtime_checks({"ship_right2": GO, "ship_left3": GO, "ship_right3": GO,
                         "after_move": GO}, ship=True)
    check(c["mousemove_moves_ship"]["pass"] is None
          and "GAME OVER" in c["mousemove_moves_ship"]["evidence"]["not_measured"],
          "mousemove: GAME OVER during the sequence -> not measured (None), never failed")
    check(c["esc_pause"]["pass"] is None and c["esc_pause"]["evidence"]["game_over_before_esc"],
          "esc_pause: GAME OVER before ESC -> not measured (None) (v2 failed the reference 5/5 "
          "this way)")
    c = _runtime_checks({}, pause_frozen=False)
    check(c["esc_pause"]["pass"] is False, "esc_pause: frames change while 'paused', no PAUSE "
          "text -> fails")
    c = _runtime_checks({"paused": {"fill_recent": ["PAUSED"]}}, pause_frozen=False)
    check(c["esc_pause"]["pass"] is True, "esc_pause: PAUSE text is enough (unchanged from v2)")
    c = _runtime_checks({"resumed": GO}, resume_moves=False)
    check(c["esc_pause"]["pass"] is True and c["esc_pause"]["evidence"]["game_over_after_resume"],
          "esc_pause: frozen while paused, the game ended after the resume -> the resume half "
          "is met (the game ran)")
    c = _runtime_checks({}, resume_moves=False)
    check(c["esc_pause"]["pass"] is False, "esc_pause: frozen before AND after the second ESC "
          "(a stuck game) still fails")
    c = _runtime_checks({}, ship=False)
    check(c["mousemove_moves_ship"]["pass"] is False, "mousemove: no ship anywhere fails")
    # click_starts_game (v3): what was drawn SINCE the click decides
    c = _runtime_checks({"after_click": {"fill_recent": ["CLICK TO START", "SCORE: 0"],
                                         "fill_since": ["SCORE: 0"]}})
    check(c["click_starts_game"]["pass"] is True, "click_starts_game: the start text only in "
          "the 1500 ms window, not drawn since the click -> started (v0f@p2, v3 stability run)")
    c = _runtime_checks({"after_click": {"fill_recent": ["CLICK TO START"],
                                         "fill_since": ["CLICK TO START"]}})
    check(c["click_starts_game"]["pass"] is False, "click_starts_game: the start text still "
          "drawn after the click -> not started")
    c = _runtime_checks({"after_click": {"fill_recent": ["CLICK TO START"]}})
    check(c["click_starts_game"]["pass"] is False, "click_starts_game: a v2 browser.json (no "
          "fill_since) is read as before")


def _fill(t, x, font="20px monospace", align="left", y=40, cw=1280):
    return {"t": t, "x": x, "y": y, "font": font, "align": align, "cw": cw, "ch": 720}


def test_hud_positions_v3():
    ok_sl = [_fill("SCORE: 0", 20), _fill("LEVEL 1", 640, align="center")]
    c = G.hud_check("V0", {"fill_all": ok_sl, "viewport": [1280, 720]}, None)
    check(c["pass"] is True and c["evidence"]["combo_drawn"] is False,
          "hud_positions V0: SCORE + LEVEL right, COMBO never drawn (shown only in a combo) "
          "-> passes, says combo_drawn false (the judgement call)", str(c["evidence"])[:300])
    c = G.hud_check("V0", {"fill_all": ok_sl + [_fill("COMBO x2", 1260, align="right")],
                           "viewport": [1280, 720]}, None)
    check(c["pass"] is True and c["evidence"]["combo_ok"] is True,
          "hud_positions V0: COMBO drawn at the right edge, y=40 -> passes")
    c = G.hud_check("V0", {"fill_all": ok_sl + [_fill("COMBO x2", 640, y=80)],
                           "viewport": [1280, 720]}, None)
    check(c["pass"] is False, "hud_positions V0: COMBO drawn in the wrong place still fails")
    c = G.hud_check("V0", {"fill_all": [_fill("LEVEL 1", 640, align="center")],
                           "viewport": [1280, 720]}, None)
    check(c["pass"] is False, "hud_positions V0: no SCORE fails")

    def box(t, x, w=120):
        return {"t": t, "x": x, "y": 22, "w": w, "h": 24, "font_size": "20px",
                "font_family": "monospace", "visible": True}
    good = [box("SCORE 0", 20), box("LEVEL 1", 580)]
    c = G.hud_check("V1", {"dom_hud_play": good, "dom_hud": [box("GAME OVER", 500)],
                           "viewport": [1280, 720]}, None)
    check(c["pass"] is True and c["evidence"]["source"] == "dom_hud_play",
          "hud_positions V1: reads the HUD right after play, not the game-over screen")
    c = G.hud_check("V1", {"dom_hud": good, "viewport": [1280, 720]}, None)
    check(c["pass"] is True and c["evidence"]["source"] == "dom_hud",
          "hud_positions V1: a v2 browser.json (no dom_hud_play) still reads dom_hud")


def test_headline_flaky_assisted():
    checks = [{"check": "a", "pass": True}, {"check": "b", "pass": False},
              {"check": "mousemove_moves_ship", "pass": False}, {"check": "c", "pass": None}]
    saved = dict(G.FLAKY)
    try:
        G.FLAKY.clear()
        h = G.headline([dict(c) for c in checks])
        check((h["spec_passed"], h["spec_failed"], h["spec_unknown"], h["spec_total"],
               h["spec_flaky"]) == (1, 2, 1, 4, []), "headline: no flaky checks -> all counted",
              str(h))
        G.FLAKY["mousemove_moves_ship"] = "test"
        cs = [dict(c) for c in checks]
        h = G.headline(cs)
        check((h["spec_passed"], h["spec_failed"], h["spec_total"]) == (1, 1, 3)
              and h["spec_flaky"] == [{"check": "mousemove_moves_ship", "pass": False}]
              and cs[2].get("flaky") == "test",
              "headline: a FLAKY check keeps its result, marked, out of the headline", str(h))
    finally:
        G.FLAKY.clear()
        G.FLAKY.update(saved)
    check(G.assisted_label("v0f-V0-xhigh-1@p2") == G.ASSISTED
          and G.assisted_label("v0f-V0-xhigh-1@p1") is None
          and G.assisted_label("fx-v0-ref") is None,
          "assisted: prompt n >= 2 of an iterative run, never prompt 1 or a fixture")
    a, b = G.shots_dest("x@p1", 1_790_000_000.0), G.shots_dest("x@p1", 1_790_000_001.0)
    check(a != b and f"v{G.GRADER_VERSION}" in a.replace("\\", "/").split("/")
          and "x_p1" in a, "screenshots: one directory per grade row under its grader version",
          a)
    # every existing @p>=2 row is labelled (its own field or an annotation)
    ann = G.jsonl(os.path.join(G.RESULTS, "grade_annotations.jsonl"))
    labelled = {a.get("row") for a in ann if a.get("label") == G.ASSISTED}
    missing = [i for i, r in enumerate(G.jsonl(G.GRADES))
               if G.assisted_label(r.get("grade_id", "")) and not r.get("assisted")
               and i not in labelled]
    check(not missing, "every @p>=2 grade row is labelled assisted", str(missing))


def test_iterative_refused():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "--variant", "V0",
                        "--arm", "xhigh", "--iterative", "2", "--tag", "refusal-test"],
                       capture_output=True, text=True, timeout=120)
    check(r.returncode == 2 and "REFUSED" in r.stderr and "--allow-graded-followups" in r.stderr,
          "run.py --iterative without --allow-graded-followups is refused, and says why",
          (r.stdout + r.stderr)[-300:])
    src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read()
    check('"assisted": ASSISTED' in src and 'meta["graded_followups"]' in src,
          "an allowed iterative run records graded_followups and marks prompts >= 2 assisted")


def test_browser_check_is_deterministic_by_default():
    import ast
    import types
    src = open(os.path.join(HERE, "browser_check.py"), encoding="utf-8").read()
    # the two constants only (browser_check imports playwright, which lives in
    # the grader's container, not on the host)
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        getattr(t, "id", None) in ("RNG_SEED", "SEED_RNG") for t in n.targets)]
    B = types.SimpleNamespace()
    exec(compile(ast.Module(body=keep, type_ignores=[]), "browser_check", "exec"), B.__dict__)
    check('ap.add_argument("--clock", choices=("fake", "real"), default="fake")' in src
          and 'ap.add_argument("--rng", choices=("seeded", "native"), default="seeded")' in src,
          "browser_check: fake clock and seeded Math.random are the defaults")
    check("page.clock.install(time=CLOCK_EPOCH)" in src and "page.clock.pause_at(CLOCK_EPOCH)"
          in src and "page.clock.run_for(" in src, "browser_check: the clock is installed "
          "paused and advanced by the script")
    check(("%d" not in B.SEED_RNG) and str(B.RNG_SEED) in B.SEED_RNG
          and "Math.imul" in B.SEED_RNG, "browser_check: the seed is baked into the init script")
    check(src.index("ctx.add_init_script(SEED_RNG)") < src.index("page = ctx.new_page()"),
          "browser_check: the RNG is seeded before any page script can run")


def test_pagoda_task():
    """bench/octopus/pagoda.py (2026-09-27): the voxel r3f-stack prompt byte
    for byte, the Responses wire required, single-shot only, the proxy
    summary and the README record."""
    import importlib.util
    import subprocess
    import pagoda as PG
    vr_path = os.path.join(os.path.dirname(HERE), "voxel", "run.py")
    spec = importlib.util.spec_from_file_location("voxel_run_for_test", vr_path)
    VR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(VR)
    text, sha = PG.prompt()
    check(text == VR.PAGODA_STACK_PROMPT and sha == VR.PAGODA_STACK_PROMPT_SHA256
          and text.endswith(VR.STACK_SENTENCE + " " + VR.OUTPUT_DIR_SENTENCE)
          and VR.STACK_SENTENCE == ("Build it with r3f (react-three-fiber) v10 and "
                                    "Koota and pmndrs math."),
          "the pagoda prompt IS bench/voxel/run.py TASKS['r3f-stack'], byte for byte", sha)
    d = tempfile.mkdtemp(prefix="octo-pagoda-")
    try:
        p, s2 = PG.write_prompt(os.path.join(d, "pagoda.md"))
        check(s2 == sha and open(p, "rb").read() == text.encode("utf-8"),
              "the prompt file Hermes reads has exactly those bytes")
        base = "model:\n  default: yamadori\nproviders:\n  octo-relay:\n" \
               "    base_url: \"http://127.0.0.1:18234/v1\"\n    api_key: x\n"
        for body, want in ((base + "    api_mode: codex_responses   # --wire responses\n"
                            "\nauxiliary:\n  x: 1\n", "responses"),
                           (base + "\nauxiliary:\n  x: 1\n", "chat"),
                           (base + "\nother:\n    api_mode: codex_responses\n", "chat")):
            f = os.path.join(d, "config.yaml")
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(body)
            check(PG.profile_wire(f) == want, f"profile_wire reads the octo-relay block: {want}")
        check(PG.profile_wire(os.path.join(d, "none.yaml")) == "missing",
              "profile_wire: no profile")
        # the proxy summary over relay rows (x_yamadori verbatim)
        rows = [
            {"method": "GET", "path": "/v1/models", "status": 200, "response": {}},
            {"method": "POST", "path": "/v1/responses", "status": 200, "response": {
                "tool_calls": ["write_file"], "x_yamadori": {
                    "route": {"class": "code_generation", "because": "asks for new code"},
                    "tools_withheld": [{"ours": "generate_image", "because": "software"}],
                    "craft": {"tool": True, "listed": 3},
                    "skills": {"ids": ["a"], "names": ["koota-react"], "tokens": 300,
                               "matched": [{"name": "koota-react", "trigger": "asked",
                                            "slot": "asked", "decided_by": "asked"}]},
                    "deep": {"fire": True, "kind": "kickoff", "job": "plan",
                             "think_tool": {"calls": []}},
                    "usage": {"generations": 1, "summed": {"prompt_tokens": 100,
                                                           "completion_tokens": 20},
                              "per_generation": [{"cached_tokens": 0}]}}}},
            {"method": "POST", "path": "/v1/responses", "status": 200, "response": {
                "tool_calls": ["terminal"], "x_yamadori": {
                    "route": {"class": "agent_step"},
                    "skills": {"ids": ["a"], "replayed": True},
                    "tools": [{"name": "think_deeply"}],
                    "deep": {"fire": False, "think_tool": {"calls": [
                        {"ran": True, "ok": True, "seconds": 9.5, "hops": 4}]}},
                    "usage": {"generations": 2, "summed": {"prompt_tokens": 300,
                                                           "completion_tokens": 50},
                              "per_generation": [{"cached_tokens": 90},
                                                 {"cached_tokens": 200}]}}}},
        ]
        with open(os.path.join(d, "relay.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        s = PG.relay_summary(d)
        check(s["requests"] == 2 and s["first_route"] == "code_generation"
              and s["routes"] == {"code_generation": 1, "agent_step": 1}
              and s["tools_withheld"][0]["ours"] == "generate_image"
              and s["skills_injected"] == 1
              and s["skills_injections"][0]["skills"][0]["trigger"] == "asked"
              and s["deep_runs"] == [{"request": 0, "kind": "kickoff", "job": "plan"}]
              and s["think_deeply"][0]["seconds"] == 9.5
              and s["our_tools"] == {"think_deeply": 1}
              and s["client_calls"] == {"write_file": 1, "terminal": 1}
              and s["tokens_main"]["prompt_tokens"] == 400
              and s["tokens_main"]["cached_tokens"] == 290,
              "relay_summary: routes, withheld tools, skills with triggers (a replay is "
              "not an injection), deep thinking, tools, tokens", json.dumps(s)[:600])
        proj = project({"package.json": "{}", "README.md":
                        "# Pagoda\n\nnpm install\nnpm run dev\nOpen http://localhost:5173\n"})
        r = PG.readme_run(proj)
        check(r["file"] == "README.md" and r["ports"] == ["5173"] and not r["mentions_3001"]
              and "npm run dev" in r["run_lines"],
              "readme_run records what the README says about running it", json.dumps(r))
        shutil.rmtree(proj, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "--task", "pagoda",
                        "--arm", "xhigh", "--iterative", "2", "--tag", "refusal-test"],
                       capture_output=True, text=True, timeout=120)
    check(r.returncode == 2 and "single-shot" in r.stderr,
          "run.py --task pagoda --iterative is refused: single-shot only",
          (r.stdout + r.stderr)[-300:])
    src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read()
    check("octopus\\\\run.py --task" in src and "octopus/run.py --task" in src,
          "a running pagoda task counts as a GPU consumer (BUSY)")
    check("PAGODA_WIRE_REFUSAL" in src and 'wire != "responses"' in src,
          "run.py refuses the pagoda on a chat-wire profile")


def main() -> int:
    for fn in (test_particles_draw_ctx, test_other_v2_checks, test_version_and_regrade_plan,
               test_ship_follows, test_runtime_rules_v3, test_hud_positions_v3,
               test_headline_flaky_assisted, test_iterative_refused,
               test_browser_check_is_deterministic_by_default,
               test_v4_stack_checks, test_v4_game_checks, test_v4_prompt_is_one_sentence,
               test_v4_pixel_checks, test_v4_hud_and_runtime_list, test_pagoda_task):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
