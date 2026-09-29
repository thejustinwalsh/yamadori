#!/usr/bin/env python3
"""The runtime half of the grader: drives the built game in headless Chromium.

Runs INSIDE mcr.microsoft.com/playwright/python (grade.py starts it, sharing
the network namespace of the container that serves the game, so the URL is
http://localhost:3001/ and Vite's host check never sees a foreign name).

    python3 browser_check.py --url http://localhost:3001/ --out /out

It measures, it does not judge: every number and screenshot goes to
/out/browser.json and /out/*.png, and grade.py (on the host, with PIL)
turns them into pass/fail with thresholds stated there. The script is fixed:
the same inputs at the same times for every run, so two runs are
comparable.

DETERMINISM (grader v3, 2026-09-26, SELF-IMPROVEMENT-LOG #62). v2's runtime
checks flipped on UNCHANGED code (reference fixture, 5 grades:
mousemove_moves_ship 1/5, hud_positions 1/5, esc_pause 0/5 vs 3/3 on
2026-09-24). Cause, traced frame by frame on the reference: the game's
Math.random picks the enemy types, and whether a medium octopus lands in the
column the ship was parked in (x = 81%) decided whether its ink blob killed the
ship before ESC was pressed -- the reference never removes an enemy bullet
that hits, so one blob takes 10 health per game-loop call and kills in five
frames; and the game advanced by frames while the script waited in wall-clock
seconds, so a loaded host moved every event. Now, by default:
  - Math.random is SEEDED (RNG_SEED, mulberry32) in an init script, before
    any page script;
  - the page's clock is Playwright's FAKE clock, installed paused at a fixed
    epoch: Date, performance.now, setTimeout/setInterval and
    requestAnimationFrame advance only when this script advances them
    (`advance`, in 50 ms steps, PACED to the wall clock so CSS transitions,
    audio and loads -- which run on real time -- see the same real time as
    before). The game gets exactly 60 frames per game second whatever the
    host's load;
  - after the click that starts the game the pointer is nudged 1 px and back,
    as a player's hand does: a game that listens for mousemove on the canvas
    under a start overlay otherwise never learns where the mouse is;
  - the mouse and ESC checks run right after the start (300 ms per mouse
    position, six positions; ESC 2.3 s of game time after the start), and each
    records whether the game was already over (then grade.py says "not
    measured", never "failed").
`--clock real --rng native` is the v2 behaviour, for comparison. FPS under the
fake clock is THROUGHPUT: game frames the page rendered per wall-clock second
while being paced to real time (60 when it keeps up).

Instrumentation (an init script, installed before any page script runs):
  - getContext calls: which context types the page asked for ('2d',
    'webgl2', 'webgpu') -- the renderer actually used, not the one hoped for
  - CanvasRenderingContext2D.fillText: distinct (text, x, y, font, align)
    with first/last seen time -- how a canvas game's text is checked
  - Web Audio: contexts created, oscillator / buffer-source starts
  - a requestAnimationFrame counter of its own: frames the page got (under
    the fake clock: game frames; FPS is frames per wall-clock second)
Console errors and uncaught page errors are recorded with timestamps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from playwright.sync_api import sync_playwright

W, H = 1280, 720

INIT = r"""
(() => {
  const O = window.__octo = {ctx: [], fill: {}, nfill: 0, audio: {contexts: 0,
    osc_start: 0, buf_start: 0, other_start: 0}, frames: [], t0: performance.now()};
  const now = () => Math.round(performance.now());
  try {
    const gc = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...a) {
      try { O.ctx.push(String(type)); } catch (e) {}
      return gc.call(this, type, ...a);
    };
  } catch (e) {}
  try {
    if (window.OffscreenCanvas) {
      const og = OffscreenCanvas.prototype.getContext;
      OffscreenCanvas.prototype.getContext = function (type, ...a) {
        try { O.ctx.push('offscreen:' + String(type)); } catch (e) {}
        return og.call(this, type, ...a);
      };
    }
  } catch (e) {}
  try {
    const ft = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (t, x, y, ...r) {
      try {
        const m = this.getTransform();
        const k = [String(t).slice(0, 60), Math.round(x), Math.round(y), this.font,
                   this.textAlign, this.textBaseline].join('|');
        const e = O.fill[k];
        if (e) { e.last = now(); e.n++; }
        else if (O.nfill < 3000) {
          O.nfill++;
          O.fill[k] = {t: String(t).slice(0, 60), x: +x, y: +y, font: this.font,
            align: this.textAlign, base: this.textBaseline, tx: m.e, ty: m.f,
            sx: m.a, sy: m.d, cw: this.canvas.width, ch: this.canvas.height,
            first: now(), last: now(), n: 1};
        }
      } catch (e) {}
      return ft.call(this, t, x, y, ...r);
    };
  } catch (e) {}
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) {
      const Wrapped = function (...a) { O.audio.contexts++; return new AC(...a); };
      Wrapped.prototype = AC.prototype;
      window.AudioContext = Wrapped;
      window.webkitAudioContext = Wrapped;
      const st = AudioScheduledSourceNode.prototype.start;
      AudioScheduledSourceNode.prototype.start = function (...a) {
        try {
          if (this instanceof OscillatorNode) O.audio.osc_start++;
          else if (this instanceof AudioBufferSourceNode) O.audio.buf_start++;
          else O.audio.other_start++;
        } catch (e) {}
        return st.apply(this, a);
      };
    }
  } catch (e) {}
  const tick = (t) => { O.frames.push(Math.round(t)); if (O.frames.length > 20000)
    O.frames.splice(0, 10000); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
})();
"""


# WebGPU is HIDDEN by default, so three's WebGPURenderer takes its WebGL2
# backend. Measured 2026-09-24 on a skeleton (bench/octopus fixtures
# fx-v2-skel / fx-v2-skel-gl): in this container WebGPU's only adapter is
# SwiftShader; the WebGPU backend drew NOTHING (0.25% of pixels off the
# background, page error "Instance dropped in popErrorScope"), and the same
# code with forceWebGL drew every sprite (2.25%). Grading on that adapter
# would fail every WebGPU variant for the container's reason, not the
# model's. --webgpu allow keeps it for a check of the WebGPU path itself.
HIDE_WEBGPU = r"""
try { Object.defineProperty(Navigator.prototype, 'gpu', {get() { return undefined; },
                                                        configurable: true}); } catch (e) {}
"""


# v3: the page's Math.random, seeded. mulberry32 (a 32-bit state, full
# period); the seed is fixed so every grade of one state plays the same game.
RNG_SEED = 0x5EED0C70
SEED_RNG = r"""
(() => {
  let s = %d >>> 0;
  const r = function random() {
    s = (s + 0x6D2B79F5) >>> 0; let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  try { Object.defineProperty(Math, 'random', {value: r, configurable: true, writable: true}); }
  catch (e) { Math.random = r; }
})();
""" % RNG_SEED
# v3: the fake clock starts here (Date.now() and new Date() read it).
CLOCK_EPOCH = "2026-01-01T00:00:00Z"

GAME_OVER_JS = "GAME OVER"

DOM_HUD = """() => {
  const out = [];
  const all = document.querySelectorAll('body *');
  for (const el of all) {
    if (el.children.length) continue;
    const t = (el.textContent || '').trim();
    if (!/SCORE|LEVEL|COMBO/i.test(t) || t.length > 60) continue;
    const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    out.push({t, x: r.x, y: r.y, w: r.width, h: r.height,
              font_size: cs.fontSize, font_family: cs.fontFamily,
              visible: r.width > 0 && r.height > 0 && cs.visibility !== 'hidden'});
  }
  return out.slice(0, 50);
}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3001/")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--idle", type=float, default=15.0)
    ap.add_argument("--play", type=float, default=30.0)
    ap.add_argument("--die", type=float, default=60.0)
    ap.add_argument("--webgpu", choices=("hide", "allow"), default="hide")
    # v3 determinism (module docstring). `real` / `native` reproduce v2.
    ap.add_argument("--clock", choices=("fake", "real"), default="fake")
    ap.add_argument("--rng", choices=("seeded", "native"), default="seeded")
    # grade.py passes the sandbox network's gate (sandbox_net.py, #47): the
    # container has no route but through it. A Chromium FLAG, not Playwright's
    # proxy option: Playwright adds <-loopback> to the bypass list, which would
    # send localhost:3001 to the gate (refused); the flag keeps Chromium's
    # implicit loopback bypass, as browser_sidecar.py relies on.
    ap.add_argument("--proxy", default=None, help="http://host:port for non-loopback requests")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fake = a.clock == "fake"
    res: dict = {"url": a.url, "viewport": [W, H], "events": [], "console": [],
                 "page_errors": [], "requests_failed": [],
                 "determinism": {"clock": a.clock, "rng": a.rng,
                                 "rng_seed": RNG_SEED if a.rng == "seeded" else None,
                                 "clock_epoch": CLOCK_EPOCH if fake else None,
                                 "paced": "game time advanced in 50 ms steps, paced to "
                                          "the wall clock" if fake else None}}
    T0 = time.time()

    def mark(name, **kw):
        res["events"].append({"t": round(time.time() - T0, 2), "name": name, **kw})

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--enable-unsafe-webgpu", "--enable-features=Vulkan",
            "--autoplay-policy=no-user-gesture-required",
            *([f"--proxy-server={a.proxy}"] if a.proxy else [])])
        res["proxy"] = a.proxy
        res["browser_version"] = browser.version
        ctx = browser.new_context(viewport={"width": W, "height": H})
        if a.webgpu == "hide":
            ctx.add_init_script(HIDE_WEBGPU)
        if a.rng == "seeded":
            ctx.add_init_script(SEED_RNG)
        ctx.add_init_script(INIT)
        res["webgpu_mode"] = a.webgpu
        page = ctx.new_page()
        if fake:
            page.clock.install(time=CLOCK_EPOCH)
            page.clock.pause_at(CLOCK_EPOCH)
        page.on("console", lambda m: res["console"].append(
            {"t": round(time.time() - T0, 2), "type": m.type, "text": m.text[:500]}))
        page.on("pageerror", lambda e: res["page_errors"].append(
            {"t": round(time.time() - T0, 2), "text": str(e)[:800]}))
        page.on("requestfailed", lambda r: res["requests_failed"].append(
            {"t": round(time.time() - T0, 2), "url": r.url[:200],
             "failure": (r.failure or "")[:200]}))
        page.on("response", lambda r: r.status >= 400 and res["requests_failed"].append(
            {"t": round(time.time() - T0, 2), "url": r.url[:200], "status": r.status}))

        chunks: list[float] = []

        def advance(ms: int, each=None, step: int = 50, real_min: float = 0.0) -> float:
            """Advance the game by `ms` of its time; returns wall seconds.
            fake: run the page's clock in `step` ms steps, each paced so the
            wall clock keeps up with game time (never ahead of it). real: sleep.
            `each(done_ms)` runs before every step (input, polls); returning
            True stops early. `real_min`: wait at least this many wall seconds
            in all (the game stays frozen meanwhile: CSS transitions finish)."""
            t_start = time.time()
            done = 0
            while done < ms:
                if each is not None and each(done):
                    break
                n = min(step, ms - done)
                c0 = time.time()
                if fake:
                    page.clock.run_for(n)
                    chunks.append(time.time() - c0)
                    done += n
                    lag = t_start + done / 1000.0 - time.time()
                    if lag > 0:
                        time.sleep(lag)
                else:
                    time.sleep(n / 1000.0)
                    done += n
            rest = t_start + real_min - time.time()
            if rest > 0:
                time.sleep(rest)
            return time.time() - t_start

        def shot(name):
            page.screenshot(path=os.path.join(a.out, f"{name}.png"))
            mark("shot", file=f"{name}.png")

        def state(label, since=None):
            """`since` (page ms): also list the canvas texts drawn at or after
            it (fill_since) -- v3: 500 ms of game time after a click is inside
            fill_recent's 1500 ms, so "recent" would still hold the start
            screen's text the click removed."""
            try:
                s = page.evaluate("""(since) => {
                  const O = window.__octo || {};
                  const t = performance.now();
                  const recent = Object.values(O.fill || {}).filter(e => t - e.last < 1500)
                                    .map(e => e.t);
                  const fsince = since === null ? null : Array.from(new Set(
                    Object.values(O.fill || {}).filter(e => e.last >= since).map(e => e.t)))
                    .slice(0, 200);
                  // VISIBLE text only: innerText keeps a start overlay that was
                  // faded to opacity 0 (seen on the V0 reference fixture).
                  const parts = [];
                  if (document.body) {
                    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                    let n;
                    while ((n = tw.nextNode()) && parts.length < 400) {
                      const t = n.textContent.trim();
                      const el = n.parentElement;
                      if (!t || !el) continue;
                      if (el.checkVisibility && !el.checkVisibility({checkOpacity: true,
                          checkVisibilityCSS: true})) continue;
                      const r = el.getBoundingClientRect();
                      if (r.width === 0 || r.height === 0 || r.bottom < 0 ||
                          r.top > innerHeight * 2 || r.right < 0 || r.left > innerWidth) continue;
                      parts.push(t);
                    }
                  }
                  return {dom_text: parts.join('\\n').slice(0, 3000),
                          fill_recent: Array.from(new Set(recent)).slice(0, 200),
                          fill_since: fsince, since: since,
                          ctx: O.ctx || [], audio: O.audio || {},
                          frames: (O.frames || []).length,
                          canvases: Array.from(document.querySelectorAll('canvas')).map(c => {
                            const r = c.getBoundingClientRect();
                            return {w: c.width, h: c.height, x: r.x, y: r.y,
                                    cw: r.width, ch: r.height}; }),
                          cursor: getComputedStyle(document.body).cursor,
                          game_time_ms: Math.round(t)};
                }""", since)
            except Exception as e:                         # noqa: BLE001
                s = {"error": f"{type(e).__name__}: {e}"[:300]}
            s["t"] = round(time.time() - T0, 2)
            txt = (s.get("dom_text") or "") + " " + " ".join(s.get("fill_recent") or [])
            s["game_over_visible"] = GAME_OVER_JS in txt.upper()
            res.setdefault("states", {})[label] = s
            return s

        def frames_between(t_from_ms, t_to_ms):
            fr = page.evaluate("() => (window.__octo || {}).frames || []")
            return [f for f in fr if t_from_ms <= f <= t_to_ms]

        def fps(t_from_ms, t_to_ms, wall_s, n_chunks0):
            """Frames the page drew in [t_from, t_to] of ITS time, per WALL
            second (under the fake clock: throughput; 60 when it keeps up)."""
            win = frames_between(t_from_ms, t_to_ms)
            gaps = [b - x for x, b in zip(win, win[1:])]
            slow = chunks[n_chunks0:]
            return {"frames": len(win), "seconds": round(wall_s, 2),
                    "game_seconds": round((t_to_ms - t_from_ms) / 1000.0, 2),
                    "fps": round(len(win) / wall_s, 1) if wall_s > 0 else None,
                    "max_gap_ms": max(gaps) if gaps else None,
                    "max_step_wall_ms": round(1000 * max(slow), 1) if slow else None,
                    "clock": a.clock}

        def perf_now():
            return page.evaluate("() => performance.now()")

        # 1. load and idle on the start screen
        try:
            page.goto(a.url, wait_until="load", timeout=60000)
            mark("loaded")
        except Exception as e:                             # noqa: BLE001
            res["load_error"] = f"{type(e).__name__}: {e}"[:500]
            mark("load_failed")
        try:
            # every request the page makes at load settles before game time
            # starts (the fake clock is paused): the same starting state
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:                                  # noqa: BLE001
            mark("networkidle_timeout")
        try:
            res["webgpu_probe"] = page.evaluate("""async () => {
              if (!navigator.gpu) return {navigator_gpu: false};
              try { const ad = await navigator.gpu.requestAdapter();
                    return {navigator_gpu: true, adapter: !!ad,
                            info: ad && ad.info ? {vendor: ad.info.vendor,
                              architecture: ad.info.architecture,
                              description: ad.info.description} : null}; }
              catch (e) { return {navigator_gpu: true, adapter: false, error: String(e)}; }
            }""")
        except Exception as e:                             # noqa: BLE001
            res["webgpu_probe"] = {"error": str(e)[:200]}
        t_idle0 = perf_now()
        n0 = len(chunks)
        page.mouse.move(W / 2, H * 0.8)
        wall = advance(int(a.idle * 1000))
        res["fps_idle"] = fps(t_idle0, perf_now(), wall, n0)
        state("start")
        shot("start")

        def nudge(x, y):
            # v3: a player's hand is never still; one mousemove after the
            # click reaches whatever is under the pointer once an overlay is gone
            page.mouse.move(x + 1, y)
            page.mouse.move(x, y)

        # 2. click to start
        # v3: 500 ms of GAME time after the click, 1.5 s of wall time (start
        # overlays fade on the wall clock; the game waits, frozen)
        t_click = perf_now()
        page.mouse.click(W / 2, H / 2)
        mark("click_start")
        advance(250)
        nudge(W / 2, H / 2)
        advance(250, real_min=1.25)
        s = state("after_click", since=t_click + 100)
        shot("after_click")
        # The spec says "CLICK TO START" (a click anywhere). If the start
        # screen is still up, try once more on a visible element whose text
        # says start -- a button is a deviation, but a game that starts is
        # still graded on everything after. Recorded as start_click.
        res["start_click"] = "center"
        still = (s.get("dom_text") or "").upper()
        if "START" in still:
            box = page.evaluate("""() => {
              const els = Array.from(document.querySelectorAll('button, a, div, span, p, h1, h2, h3'));
              for (const el of els.reverse()) {
                if (el.children.length > 2) continue;
                const t = (el.textContent || '').trim();
                if (!/start/i.test(t) || t.length > 40) continue;
                el.scrollIntoView({block: 'center'});
                const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
                if (r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none')
                  return {x: r.x + r.width / 2, y: r.y + r.height / 2, text: t};
              }
              return null;
            }""")
            if box:
                t_click = perf_now()
                page.mouse.click(box["x"], box["y"])
                mark("click_start_element", text=box["text"])
                # v3: undo OUR scrollIntoView. It scrolled a page a player
                # cannot scroll (the reference: body overflow hidden, 32 px),
                # which shifted every later screenshot and left the pointer
                # below the canvas, where the game never sees it move.
                try:
                    res["scroll_undone"] = page.evaluate(
                        "() => { const s = [scrollX, scrollY]; window.scrollTo(0, 0); "
                        "return s; }")
                except Exception as e:                     # noqa: BLE001
                    res["scroll_undone"] = str(e)[:200]
                advance(250)
                nudge(box["x"], box["y"])
                advance(250, real_min=1.25)
                s2 = state("after_click", since=t_click + 100)
                shot("after_click")
                if (s2.get("dom_text") or "") != (s.get("dom_text") or ""):
                    res["start_click"] = "element: " + box["text"][:40]

        # 3. mouse moves the ship. v3: the pointer ALTERNATES left / right
        # three times (x = 30% / 70%, y = 83%), 300 ms of game time at each (the
        # spec's lerp 0.35 per frame leaves 0.04% of the distance after 18
        # frames). grade.py asks whether what is drawn at each position tracks
        # the pointer across all six shots, so a planet or an enemy passing
        # through one window cannot pass or fail it alone (v2: one left/right
        # pair at 19% / 81%, 1.2 s each, read by colour centroids). 30% / 70%
        # lie between the columns of a W/(n+1) wave for n = 4 (20/40/60/80%),
        # where the reference's ink blobs fall (its ship died parked at 19%).
        for i in (1, 2, 3):
            for side, fx in (("left", 0.30), ("right", 0.70)):
                name = f"ship_{side}" + ("" if i == 1 else str(i))
                page.mouse.move(W * fx, H * 0.83, steps=8)
                advance(300)
                shot(name)
                state(name)
        state("after_move")

        # 4. ESC pause: two frames 1 s of game time apart should be identical
        # while paused, and differ after the second ESC
        page.keyboard.press("Escape")
        mark("esc_1")
        advance(100)
        shot("pause_a")
        advance(1000)
        shot("pause_b")
        state("paused")
        page.keyboard.press("Escape")
        mark("esc_2")
        advance(100)
        shot("resumed_a")
        advance(1000)
        shot("resumed_b")
        state("resumed")

        # 5. 30 s of play: hold fire, sweep across the bottom band
        n_err0 = len(res["console"]), len(res["page_errors"])
        t_play0 = perf_now()
        n0 = len(chunks)
        page.mouse.down()
        k = [0]
        mid = [False]

        def sweep(done):
            x = W / 2 + math.sin(k[0] / 12.0) * W * 0.35
            page.mouse.move(x, H * 0.8)
            k[0] += 1
            if not mid[0] and done >= a.play * 1000 / 2:
                shot("play")
                mid[0] = True
            return False
        wall = advance(int(a.play * 1000), each=sweep)
        page.mouse.up()
        res["fps_play"] = fps(t_play0, perf_now(), wall, n0)
        res["play_new_errors"] = {
            "console_errors": sum(1 for c in res["console"][n_err0[0]:] if c["type"] == "error"),
            "page_errors": len(res["page_errors"]) - n_err0[1]}
        state("after_play")
        shot("play_end")
        try:
            res["dom_hud_play"] = page.evaluate(DOM_HUD)
        except Exception as e:                             # noqa: BLE001
            res["dom_hud_play_error"] = str(e)[:300]

        # 6. try to reach game over: sit in the enemies' path, keep firing off
        got = [False]
        kk = [0]

        def die(done):
            page.mouse.move(W / 2 + math.sin(kk[0] / 3.0) * 120, H * 0.18)
            kk[0] += 1
            if kk[0] % 8 == 0:
                s = state("die_poll")
                if s.get("game_over_visible"):
                    got[0] = True
                    return True
            return False
        advance(int(a.die * 1000), each=die, step=250)
        res["gameover_seen_text"] = got[0]
        mark("die_end", gameover=got[0])
        state("end")
        shot("gameover" if got[0] else "gameover_attempt")
        try:
            res["fill_all"] = page.evaluate(
                "() => Object.values((window.__octo || {}).fill || {}).slice(0, 3000)")
            res["dom_hud"] = page.evaluate(DOM_HUD)
        except Exception as e:                             # noqa: BLE001
            res["collect_error"] = str(e)[:300]
        browser.close()
    res["wall_s"] = round(time.time() - T0, 1)
    with open(os.path.join(a.out, "browser.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"ok": True, "wall_s": res["wall_s"],
                      "page_errors": len(res["page_errors"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
