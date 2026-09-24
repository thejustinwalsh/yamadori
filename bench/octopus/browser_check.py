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

Instrumentation (an init script, installed before any page script runs):
  - getContext calls: which context types the page asked for ('2d',
    'webgl2', 'webgpu') -- the renderer actually used, not the one hoped for
  - CanvasRenderingContext2D.fillText: distinct (text, x, y, font, align)
    with first/last seen time -- how a canvas game's text is checked
  - Web Audio: contexts created, oscillator / buffer-source starts
  - a requestAnimationFrame counter of its own: frames per second the page
    actually got (a busy main thread drops them)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3001/")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--idle", type=float, default=15.0)
    ap.add_argument("--play", type=float, default=30.0)
    ap.add_argument("--die", type=float, default=60.0)
    ap.add_argument("--webgpu", choices=("hide", "allow"), default="hide")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res: dict = {"url": a.url, "viewport": [W, H], "events": [], "console": [],
                 "page_errors": [], "requests_failed": []}
    T0 = time.time()

    def mark(name, **kw):
        res["events"].append({"t": round(time.time() - T0, 2), "name": name, **kw})

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--enable-unsafe-webgpu", "--enable-features=Vulkan",
            "--autoplay-policy=no-user-gesture-required"])
        res["browser_version"] = browser.version
        ctx = browser.new_context(viewport={"width": W, "height": H})
        if a.webgpu == "hide":
            ctx.add_init_script(HIDE_WEBGPU)
        ctx.add_init_script(INIT)
        res["webgpu_mode"] = a.webgpu
        page = ctx.new_page()
        page.on("console", lambda m: res["console"].append(
            {"t": round(time.time() - T0, 2), "type": m.type, "text": m.text[:500]}))
        page.on("pageerror", lambda e: res["page_errors"].append(
            {"t": round(time.time() - T0, 2), "text": str(e)[:800]}))
        page.on("requestfailed", lambda r: res["requests_failed"].append(
            {"t": round(time.time() - T0, 2), "url": r.url[:200],
             "failure": (r.failure or "")[:200]}))
        page.on("response", lambda r: r.status >= 400 and res["requests_failed"].append(
            {"t": round(time.time() - T0, 2), "url": r.url[:200], "status": r.status}))

        def shot(name):
            page.screenshot(path=os.path.join(a.out, f"{name}.png"))
            mark("shot", file=f"{name}.png")

        def state(label):
            try:
                s = page.evaluate("""() => {
                  const O = window.__octo || {};
                  const t = performance.now();
                  const recent = Object.values(O.fill || {}).filter(e => t - e.last < 1500)
                                    .map(e => e.t);
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
                          ctx: O.ctx || [], audio: O.audio || {},
                          canvases: Array.from(document.querySelectorAll('canvas')).map(c => {
                            const r = c.getBoundingClientRect();
                            return {w: c.width, h: c.height, x: r.x, y: r.y,
                                    cw: r.width, ch: r.height}; }),
                          cursor: getComputedStyle(document.body).cursor};
                }""")
            except Exception as e:                         # noqa: BLE001
                s = {"error": f"{type(e).__name__}: {e}"[:300]}
            s["t"] = round(time.time() - T0, 2)
            res.setdefault("states", {})[label] = s
            return s

        def fps(t_from_ms, t_to_ms):
            fr = page.evaluate("() => (window.__octo || {}).frames || []")
            win = [f for f in fr if t_from_ms <= f <= t_to_ms]
            dur = (t_to_ms - t_from_ms) / 1000.0
            gaps = [b - x for x, b in zip(win, win[1:])]
            return {"frames": len(win), "seconds": round(dur, 2),
                    "fps": round(len(win) / dur, 1) if dur > 0 else None,
                    "max_gap_ms": max(gaps) if gaps else None}

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
        page.mouse.move(W / 2, H * 0.8)
        time.sleep(a.idle)
        res["fps_idle"] = fps(t_idle0, perf_now())
        state("start")
        shot("start")

        # 2. click to start
        page.mouse.click(W / 2, H / 2)
        mark("click_start")
        time.sleep(1.5)
        s = state("after_click")
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
                page.mouse.click(box["x"], box["y"])
                mark("click_start_element", text=box["text"])
                time.sleep(1.5)
                s2 = state("after_click")
                shot("after_click")
                if (s2.get("dom_text") or "") != (s.get("dom_text") or ""):
                    res["start_click"] = "element: " + box["text"][:40]

        # 3. mouse moves the ship (control: the same position twice)
        page.mouse.move(W * 0.19, H * 0.83, steps=8)
        time.sleep(1.2)
        shot("ship_left")
        page.mouse.move(W * 0.81, H * 0.83, steps=8)
        time.sleep(1.2)
        shot("ship_right")
        time.sleep(1.2)
        shot("ship_right2")
        state("after_move")

        # 4. ESC pause: two frames 1 s apart should be identical while paused
        page.keyboard.press("Escape")
        mark("esc_1")
        time.sleep(0.8)
        shot("pause_a")
        time.sleep(1.0)
        shot("pause_b")
        state("paused")
        page.keyboard.press("Escape")
        mark("esc_2")
        time.sleep(0.8)
        shot("resumed_a")
        time.sleep(1.0)
        shot("resumed_b")

        # 5. 30 s of play: hold fire, sweep across the bottom band
        n_err0 = len(res["console"]), len(res["page_errors"])
        t_play0 = perf_now()
        page.mouse.down()
        t_end = time.time() + a.play
        k = 0
        mid_done = False
        while time.time() < t_end:
            x = W / 2 + math.sin(k / 12.0) * W * 0.35
            page.mouse.move(x, H * 0.8)
            k += 1
            time.sleep(0.05)
            if not mid_done and time.time() > t_end - a.play / 2:
                shot("play")
                mid_done = True
        page.mouse.up()
        res["fps_play"] = fps(t_play0, perf_now())
        res["play_new_errors"] = {
            "console_errors": sum(1 for c in res["console"][n_err0[0]:] if c["type"] == "error"),
            "page_errors": len(res["page_errors"]) - n_err0[1]}
        state("after_play")
        shot("play_end")

        # 6. try to reach game over: sit in the enemies' path, keep firing off
        t_end = time.time() + a.die
        got = False
        k = 0
        while time.time() < t_end:
            page.mouse.move(W / 2 + math.sin(k / 3.0) * 120, H * 0.18)
            k += 1
            time.sleep(0.25)
            if k % 8 == 0:
                s = state("die_poll")
                txt = (s.get("dom_text") or "") + " " + " ".join(s.get("fill_recent") or [])
                if "GAME OVER" in txt.upper():
                    got = True
                    break
        res["gameover_seen_text"] = got
        mark("die_end", gameover=got)
        state("end")
        shot("gameover" if got else "gameover_attempt")
        try:
            res["fill_all"] = page.evaluate(
                "() => Object.values((window.__octo || {}).fill || {}).slice(0, 3000)")
            res["dom_hud"] = page.evaluate("""() => {
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
            }""")
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
