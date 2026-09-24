#!/usr/bin/env python
"""Octopus Invaders variants: one game spec, four tech stacks.

    python bench/octopus/variants.py            # build index/octopus/prompts/V*.md
    python bench/octopus/variants.py --check    # verify only, write nothing

WHAT THIS IS

The operator's stress test. The prompt at SOURCE_URL (a vanilla-JS canvas
space shooter, ~1,400 words) never completed through the proxy. The same game
spec is run with ONLY its tech-stack lines swapped, through a real harness
(Hermes), and every run is graded deterministically by grade.py.

THE ORIGINAL IS NOT IN THIS REPO. It is someone else's work. `fetch()` stores
it under index/octopus/ (gitignored) with its URL, commit sha and sha256, and
the built variant prompts (which contain it) go to index/octopus/prompts/.
What is committed here is OURS: the stack lines, and the rule that says which
original lines they replace.

HOW A VARIANT IS BUILT

Each variant is a list of (original text, replacement text) pairs. Every
original text must occur EXACTLY ONCE in the fetched prompt, as whole lines,
or the build fails (a silent no-op substitution would make a variant that is
secretly the original). After substitution, every line of the original that
was not named in a pair is checked to be present, in order, byte for byte:
the gameplay and visual requirements are word for word by construction, and
the manifest records how many lines were kept vs replaced.

The replaced lines are the stack: the one-line tech statement, PROJECT
STRUCTURE, CANVAS SETUP, the fillRect/ctx instructions for the octopi and the
particle draw call, "canvas-drawn", requestAnimationFrame, and the run/serve
instructions. Nothing about what the game does or looks like is replaced;
file names inside otherwise-kept sentences (`game.js`, `config.js`) are the
exceptions, swapped for their `.ts` counterparts in the TS variants.

PINNED VERSIONS (npm registry, read 2026-09-24; full registry documents kept in
index/octopus/reg_*.json, narrowed while reading, never while fetching --
PROTOCOL rule 16):

  three 0.185.1          three-flatland 0.1.0-alpha.10 peers ^0.185.1 (so
                         <0.186); glyph peers >=0.185.0; r3f v10 >=0.185.0.
                         0.186.1 was published the same day and is not used.
  @react-three/fiber     10.0.0-alpha.5, NOT latest 9.8.0. The choice the
                         operator left open ("unless the repo's recipes say
                         v10"): bench/recipes/r3f*.jsonl hold 48 v10-specific
                         recipes against 9 v9-specific ones, and glyph renders
                         with three/webgpu node materials, which v10's
                         WebGPU canvas provides. glyph's peer range admits
                         both (>=9.7.0 <10 || >=10.0.0-alpha.4).
  @pmndrs/glyph          0.1.0 (published 2026-09-18; the 2026-09-23 upload is
                         the canary 0.1.0-canary-5eccfac5-20260923, not used)
  react / react-dom      19.2.8 (r3f v10 and glyph both peer <19.3; latest
                         19.3.0 is outside that range)
  koota                  0.6.6 (three-flatland's required peer)
  vite                   8.3.1   @vitejs/plugin-react 6.1.1 (peers vite ^8)
  typescript             5.9.3, NOT latest 7.0.2: 7.x is the native compiler
                         and rejects options a 5.x tsconfig commonly carries;
                         a build failure from that would be about tsconfig
                         dialect, not the stack under test.
  @types/react 19.2.18   @types/react-dom 19.2.7   @types/three 0.185.4
  font (V3)              JetBrains Mono v2.304 Regular TTF (glyph loads
                         ttf/otf; it ships no default font)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STORE = os.path.join(ROOT, "index", "octopus")
PROMPTS = os.path.join(STORE, "prompts")

SOURCE_REPO = "sudoingX/octopus-invaders"
SOURCE_PATH = "prompts/octopus_invaders_prompt.md"
SOURCE_SHA = "52cc146e67153269610c316812c322f31baaf517"   # main, 2026-03-19
SOURCE_URL = ("https://raw.githubusercontent.com/sudoingX/octopus-invaders/"
              "main/prompts/octopus_invaders_prompt.md")
SOURCE_SHA256 = "d30c026476f60857b5bc85f06af91622ab58c888487af8114fea6b3d15ce9590"
ORIGINAL = os.path.join(STORE, "octopus_invaders_prompt.md")

FONT_URL = ("https://github.com/JetBrains/JetBrainsMono/raw/v2.304/fonts/ttf/"
            "JetBrainsMono-Regular.ttf")

VERSIONS = {
    "V0": {},
    "V1": {
        "dependencies": {"react": "19.2.8", "react-dom": "19.2.8",
                         "three": "0.185.1", "@react-three/fiber": "10.0.0-alpha.5"},
        "devDependencies": {"vite": "8.3.1", "@vitejs/plugin-react": "6.1.1",
                            "typescript": "5.9.3", "@types/react": "19.2.18",
                            "@types/react-dom": "19.2.7", "@types/three": "0.185.4"},
    },
    "V2": {
        "dependencies": {"three": "0.185.1", "three-flatland": "0.1.0-alpha.10",
                         "koota": "0.6.6"},
        "devDependencies": {"vite": "8.3.1", "typescript": "5.9.3",
                            "@types/three": "0.185.4"},
    },
}
VERSIONS["V3"] = {
    "dependencies": {**VERSIONS["V1"]["dependencies"], "@pmndrs/glyph": "0.1.0"},
    "devDependencies": dict(VERSIONS["V1"]["devDependencies"]),
}

# The node image every run's terminal (and the grader's build) uses.
# Vite 8 needs node ^20.19 || >=22.12. Pinned by digest in run.py's manifest.
NODE_IMAGE = "nikolaik/python-nodejs:python3.12-nodejs22"

# ---------------------------------------------------------------- originals
# The original lines each variant replaces, as (first line, last line,
# sha256[:16] of those lines joined by "\n"), 1-based, in the pinned file.
# Spans, not text: the original is not ours to commit. build() refuses a span
# whose hash does not match, so an edited upstream file cannot be patched
# silently in the wrong place. What each span is, in our words:

O_TECH = (1, 1, "33951257fcb147e7")            # the one-line tech statement
O_STRUCTURE = (3, 16, "79eaf358a5f91979")      # PROJECT STRUCTURE block (js/ files)
O_CANVAS = (20, 23, "ee0233350a01ff9b")        # CANVAS SETUP (canvas size, smoothing)
O_OCTOPUS = (48, 54, "5a8040c1da06b905")       # OCTOPUS RENDERING via ctx.fillRect
O_PARTICLE_DRAW = (66, 66, "d9ad15de83353e4c") # particles.draw(ctx) in the draw loop
O_BULLET_TRAILS = (68, 68, "10b79e239876d40a") # bullet trails "in game.js"
O_SHIP = (109, 109, "6f41d93f2859854c")        # SHIP DESIGN heading, "canvas-drawn"
O_RAF = (133, 133, "7170217643a4a49a")         # 60fps with requestAnimationFrame
O_README = (153, 153, "3ee3a585b2b582a9")      # README: run with python http.server
O_JSDOC = (154, 154, "d8b3a90c59116b9c")       # JSDoc on each JS module
O_CONFIG = (155, 155, "d16418d3d8c8195d")      # config.js sections
O_SERVE = (157, 157, "06d1d66e9671e0d1")       # serve with python http.server 3001

# ------------------------------------------------------------------ ours


def _pins(v: str, extra: str = "") -> str:
    ver = VERSIONS[v]
    dep = ", ".join(f"{k} {x}" for k, x in ver["dependencies"].items())
    dev = ", ".join(f"{k} {x}" for k, x in ver["devDependencies"].items())
    return ("PINNED PACKAGES (package.json lists exactly these, at exactly these "
            "versions: no ^ or ~, no other packages):\n"
            f"- dependencies: {dep}\n"
            f"- devDependencies: {dev}\n"
            "- runtime: node 22 and npm are installed in your environment" + extra)


_SCRIPTS = ('scripts: "dev": "vite --host --port 3001", "build": "tsc --noEmit && '
            'vite build", "preview": "vite preview --host --port 3001"')

_COORDS = ("- all screen positions in this spec (x=20, y=40, top center, right edge, "
           "bottom center, floats up 60px) are CSS pixels from the top-left corner "
           "of the window, y down; map them to your camera")

_R3F_STRUCTURE = """PROJECT STRUCTURE:
space-shooter/
  index.html          -- entry point: one #root div and <script type="module" src="/src/main.tsx">
  package.json        -- the pinned packages above; {scripts}
  tsconfig.json       -- strict mode on, jsx react-jsx, moduleResolution bundler
  vite.config.ts      -- vite config with @vitejs/plugin-react
  README.md           -- what it is, how to run, controls, screenshot placeholder{font_row}
  src/
    main.tsx          -- createRoot(#root).render(<App />), imports styles.css
    App.tsx           -- the fullscreen R3F <Canvas> with an orthographic camera{app_tail}; routes mouse and keyboard input to game.ts
    styles.css        -- fullscreen canvas, cursor hidden during gameplay, no-select, overflow hidden
    config.ts         -- color palette, speeds, enemy stats, sizes, all tuning constants
    game.ts           -- main game loop step, state machine (menu/playing/paused/gameover), collision detection, damage numbers, screen shake
    player.ts         -- ship pixel grids, mouse tracking with lerp, weapons, upgrade tiers, health, engine trail spawning
    enemies.ts        -- pixelated octopus enemy types (pixel grids of 0s and 1s, NOT smooth geometry), wave spawning, boss logic
    particles.ts      -- explosion system, ink splatter, engine trails, bullet trails, spark impacts, powerup sparkles
    background.ts     -- 4-layer parallax (stars, nebula, planets, comets), all scrolling DOWNWARD (vertical shooter), mouse-reactive depth
    audio.ts          -- Web Audio API procedural sounds (laser, explosions, boss music, powerup, hit sound, unleash drone)
    components/
      Scene.tsx       -- draws the world from game state every frame (useFrame): background, octopi, ship, bullets, particles, effects
      Hud.tsx         -- HUD (20px font, y=40 baseline, proper spacing), start/gameover screens, score display, health bar, combo counter{hud_tail}"""

_R3F_CANVAS = """CANVAS SETUP (@react-three/fiber):
- App.tsx renders one <Canvas> from @react-three/fiber that fills the window and resizes with it
- orthographic camera in pixel units: 1 world unit = 1 CSS pixel, the whole window visible
{coords}
- pixel art style requires hard pixels: every texture uses THREE.NearestFilter for magFilter and minFilter (this replaces ctx.imageSmoothingEnabled = false)
- game.ts's update runs once per frame from a single useFrame in Scene.tsx (this replaces the requestAnimationFrame loop); per-frame game state lives in plain TypeScript objects, not React state""".replace("{coords}", _COORDS)

_GRID_OCTOPUS = """OCTOPUS RENDERING (pixel art style):
- render ALL octopus types from pixel grids, NOT circles, spheres or smooth shapes
- define pixel grids (e.g. 8x8 or 10x10 arrays of 0s and 1s) for each octopus type
- calculate cell size from enemy size / grid dimensions
{draw}
- tentacles animate by toggling bottom row cells between frames
- this gives authentic pixelated look. smooth arcs look wrong for this art style"""

_R3F_DRAW = ("- draw each filled cell as a square cellW x cellH at (x + col*cellW, y + "
             "row*cellH): either one instance of an instancedMesh of unit planes per "
             "filled cell, or the grid baked into a DataTexture (one texel per cell, "
             "NearestFilter) on one quad per octopus")
_R3F_PARTICLES = ("- the particle system MUST be drawn every frame: particles.ts keeps the "
                  "pool, and Scene.tsx renders it through one instancedMesh (or "
                  "THREE.Points) whose instance data is rewritten from the pool inside "
                  "useFrame. if you forget this, no particles render")

_TS_README = ("- README.md with: game description, how to run (npm install, then npm run "
              "dev, open http://localhost:3001), controls list, project structure "
              "explanation")
_TS_JSDOC = ("- brief JSDoc comments at top of each TypeScript module explaining its "
             "responsibility")
_TS_CONFIG = "- config.ts should have clear sections with comments for easy tuning"
_TS_SERVE = ("when done: npm install, then npm run build must pass with zero TypeScript "
             "errors, then serve with npm run dev (vite on port 3001). make sure all "
             "imports work and game runs immediately on first load.")
# A kept sentence with one file name swapped: a function of the original line.
def _TS_BULLET_TRAILS(line: str) -> str:
    assert line.count("game.js") == 1, line
    return line.replace("game.js", "game.ts")



_TS_SHIP = "SHIP DESIGN (pixel art, all drawn in code from pixel grids, no image files):"


def _r3f(v: str) -> list[tuple[str, str]]:
    glyph = v == "V3"
    font_row = ("\n  public/fonts/JetBrainsMono-Regular.ttf -- the monospace font every "
                "string is rendered with" if glyph else "")
    app_tail = "" if glyph else ", plus the DOM overlay from Hud.tsx"
    hud_tail = (", drawn inside the R3F scene with @pmndrs/glyph text (no DOM text)"
                if glyph else ", as a DOM overlay above the canvas")
    tech = ("build a space shooter game with React, @react-three/fiber"
            + (", @pmndrs/glyph" if glyph else "")
            + " and TypeScript, bundled with Vite. use exactly the packages and versions "
              "pinned below, no others. multi-file project structure.")
    font_pin = (f"\n- font: JetBrains Mono Regular (v2.304), downloaded once to "
                f"public/fonts/JetBrainsMono-Regular.ttf from {FONT_URL}" if glyph else "")
    structure = (_pins(v, font_pin) + "\n\n" + _R3F_STRUCTURE.format(
        scripts=_SCRIPTS, font_row=font_row, app_tail=app_tail, hud_tail=hud_tail))
    canvas = _R3F_CANVAS
    if glyph:
        canvas += """

TEXT (@pmndrs/glyph):
- ALL text is rendered with @pmndrs/glyph inside the R3F scene: title, "CLICK TO START", HUD (score, level, combo), unleash meter, floating damage numbers, pause text, game over screen
- no DOM text elements and no other text renderer (no HTML overlay text, no canvas 2D fillText, no drei Text)
- every string uses public/fonts/JetBrainsMono-Regular.ttf (this is the monospace font)
- glyph draws with three's node materials (three/webgpu), so the <Canvas> must use R3F's WebGPU renderer (three's WebGPURenderer, which falls back to WebGL2 where WebGPU is missing)"""
    return [
        (O_TECH, tech),
        (O_STRUCTURE, structure),
        (O_CANVAS, canvas),
        (O_OCTOPUS, _GRID_OCTOPUS.format(draw=_R3F_DRAW)),
        (O_PARTICLE_DRAW, _R3F_PARTICLES),
        (O_BULLET_TRAILS, _TS_BULLET_TRAILS),
        (O_SHIP, _TS_SHIP),
        (O_RAF, "- smooth 60fps target (one useFrame drives the whole game loop)"),
        (O_README, _TS_README),
        (O_JSDOC, _TS_JSDOC),
        (O_CONFIG, _TS_CONFIG),
        (O_SERVE, _TS_SERVE),
    ]


_FLAT_STRUCTURE = """PROJECT STRUCTURE:
space-shooter/
  index.html          -- entry point: a #game container for the renderer's canvas, a #ui overlay div for HUD and screens, <script type="module" src="/src/main.ts">
  package.json        -- the pinned packages above; {scripts}
  tsconfig.json       -- strict mode on, moduleResolution bundler, target ES2022 (top-level await)
  README.md           -- what it is, how to run, controls, screenshot placeholder
  src/
    main.ts           -- creates the renderer, orthographic camera and scene, then starts the game
    styles.css        -- fullscreen canvas, cursor hidden during gameplay, no-select, overflow hidden
    config.ts         -- color palette, speeds, enemy stats, sizes, all tuning constants
    textures.ts       -- turns every pixel grid (octopi, ship tiers, bullets, particles, orb) into a pixel-art texture for three-flatland sprites
    game.ts           -- main game loop, state machine (menu/playing/paused/gameover), collision detection, damage numbers, screen shake
    player.ts         -- ship sprite, mouse tracking with lerp, weapons, upgrade tiers, health, engine trail spawning
    enemies.ts        -- pixelated octopus enemy types (three-flatland sprites from pixel-grid textures, NOT smooth geometry), wave spawning, boss logic
    particles.ts      -- explosion system, ink splatter, engine trails, bullet trails, spark impacts, powerup sparkles
    background.ts     -- 4-layer parallax (stars, nebula, planets, comets), all scrolling DOWNWARD (vertical shooter), mouse-reactive depth
    ui.ts             -- HUD (20px font, y=40 baseline, proper spacing), start/gameover screens, score display, health bar, combo counter, in the #ui DOM overlay (three-flatland is a renderer, not a UI toolkit)
    audio.ts          -- Web Audio API procedural sounds (laser, explosions, boss music, powerup, hit sound, unleash drone)"""

_FLAT_CANVAS = """RENDERER SETUP (three.js + three-flatland):
- in main.ts: a WebGPURenderer from 'three/webgpu' (it falls back to WebGL2 where WebGPU is missing), await renderer.init(), size = window.innerWidth x window.innerHeight
- orthographic camera in pixel units: 1 world unit = 1 CSS pixel, the whole window visible
{coords}
- every game object (octopi, ship, bullets, particles, orbs, background layers) is a three-flatland sprite (Sprite2D) in a SpriteGroup for batching, not a hand-built mesh
- pixel art style requires hard pixels: every texture uses THREE.NearestFilter (this replaces ctx.imageSmoothingEnabled = false)
- add window resize listener to update renderer size and camera bounds""".replace("{coords}", _COORDS)

_FLAT_DRAW = ("- textures.ts draws each filled cell as one texel of a texture "
              "(NearestFilter), and each octopus is a three-flatland Sprite2D with that "
              "texture, scaled to its size")
_FLAT_PARTICLES = ("- every particle is a three-flatland sprite in one SpriteGroup that "
                   "MUST be added to the scene, and particles.update() MUST copy each live "
                   "particle's position, scale, color and alpha onto its sprite every "
                   "frame. if you forget this, no particles render")


def _flatland() -> list[tuple[str, str]]:
    tech = ("build a space shooter game with three.js and three-flatland (2D sprites, "
            "tilemaps and effects for three.js, WebGPU/TSL) in TypeScript, bundled with "
            "Vite. no React. use exactly the packages and versions pinned below, no "
            "others. multi-file project structure.")
    return [
        (O_TECH, tech),
        (O_STRUCTURE, _pins("V2") + "\n\n" + _FLAT_STRUCTURE.format(scripts=_SCRIPTS)),
        (O_CANVAS, _FLAT_CANVAS),
        (O_OCTOPUS, _GRID_OCTOPUS.format(draw=_FLAT_DRAW).replace(
            "- tentacles animate by toggling bottom row cells between frames",
            "- tentacles animate by toggling bottom row cells between frames "
            "(one texture per frame, swapped on the sprite)")),
        (O_PARTICLE_DRAW, _FLAT_PARTICLES),
        (O_BULLET_TRAILS, _TS_BULLET_TRAILS),
        (O_SHIP, _TS_SHIP),
        (O_RAF, "- smooth 60fps target with renderer.setAnimationLoop"),
        (O_README, _TS_README),
        (O_JSDOC, _TS_JSDOC),
        (O_CONFIG, _TS_CONFIG),
        (O_SERVE, _TS_SERVE),
    ]


VARIANTS = {
    "V0": {"name": "original: vanilla JS + canvas", "pairs": [], "ts": False},
    "V1": {"name": "React + @react-three/fiber v10 + Vite + TypeScript",
           "pairs": _r3f("V1"), "ts": True},
    "V2": {"name": "three.js + three-flatland + Vite + TypeScript",
           "pairs": _flatland(), "ts": True},
    "V3": {"name": "V1 + @pmndrs/glyph for all text", "pairs": _r3f("V3"), "ts": True},
}

# ------------------------------------------------------------------ build


def fetch(force: bool = False) -> str:
    """The original, pinned to SOURCE_SHA; refuses a changed file."""
    if not os.path.exists(ORIGINAL) or force:
        os.makedirs(STORE, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_SHA}/"
               f"{SOURCE_PATH}")
        data = urllib.request.urlopen(url, timeout=60).read()
        with open(ORIGINAL, "wb") as f:
            f.write(data)
    data = open(ORIGINAL, "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != SOURCE_SHA256:
        raise SystemExit(f"original prompt sha256 {got} != pinned {SOURCE_SHA256}")
    with open(os.path.join(STORE, "SOURCE.json"), "w", encoding="utf-8") as f:
        json.dump({"url": SOURCE_URL, "repo": SOURCE_REPO, "path": SOURCE_PATH,
                   "commit": SOURCE_SHA, "sha256": SOURCE_SHA256,
                   "words": len(data.decode("utf-8").split()),
                   "note": "third-party text; never commit it"}, f, indent=2)
    return data.decode("utf-8")


def build(original: str, v: str) -> tuple[str, dict]:
    lines = original.split("\n")
    spans = sorted(VARIANTS[v]["pairs"], key=lambda p: p[0][0])
    replaced = 0
    last_end = 0
    for (a, b, sha), _new in spans:
        if a <= last_end or b < a or b > len(lines):
            raise SystemExit(f"{v}: bad or overlapping span {a}-{b}")
        got = hashlib.sha256("\n".join(lines[a - 1:b]).encode("utf-8")).hexdigest()
        if got[:16] != sha:
            raise SystemExit(f"{v}: lines {a}-{b} hash {got[:16]} != pinned {sha}: "
                             "the original changed; re-derive the spans")
        last_end = b
        replaced += b - a + 1
    out: list[str] = []
    i = 1
    for (a, b, _sha), new in spans:
        out.extend(lines[i - 1:a - 1])               # kept, verbatim
        old = "\n".join(lines[a - 1:b])
        out.extend((new(old) if callable(new) else new).split("\n"))
        i = b + 1
    out.extend(lines[i - 1:])
    text = "\n".join(out)
    # Every original line outside a span survives, in order (by construction;
    # checked anyway -- a check that cannot fail is not a check, so this one
    # is exercised by --check against the hashes above).
    in_span = {n for (a, b, _s), _ in spans for n in range(a, b + 1)}
    kept = [ln for n, ln in enumerate(lines, 1) if n not in in_span]
    j = 0
    for ln in kept:
        while j < len(out) and out[j] != ln:
            j += 1
        if j == len(out):
            raise SystemExit(f"{v}: kept line lost or reordered")
        j += 1
    total = len(lines)
    return text, {"variant": v, "name": VARIANTS[v]["name"],
                  "lines_original": total, "lines_replaced": replaced,
                  "lines_kept_verbatim": total - replaced,
                  "words": len(text.split()),
                  "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  "versions": VERSIONS[v], "ts": VARIANTS[v]["ts"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refetch", action="store_true")
    a = ap.parse_args()
    original = fetch(force=a.refetch)
    manifest = {"source": json.load(open(os.path.join(STORE, "SOURCE.json"))),
                "node_image": NODE_IMAGE, "font_url": FONT_URL, "variants": {}}
    for v in VARIANTS:
        text, meta = build(original, v)
        manifest["variants"][v] = meta
        print(f"{v}: {meta['lines_kept_verbatim']}/{meta['lines_original']} lines "
              f"verbatim, {meta['lines_replaced']} replaced, {meta['words']} words")
        if not a.check:
            os.makedirs(PROMPTS, exist_ok=True)
            with open(os.path.join(PROMPTS, f"{v}.md"), "w", encoding="utf-8",
                      newline="\n") as f:
                f.write(text)
    if not a.check:
        with open(os.path.join(PROMPTS, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
