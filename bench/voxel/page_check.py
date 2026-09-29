#!/usr/bin/env python3
"""The in-container half of the voxel check: loads ONE page in headless
Chromium and measures it. It judges nothing.

Runs INSIDE the pinned Playwright image (bench/octopus/playwright.Dockerfile,
built as octo-playwright:1.63.0), on the run's `--internal` sandbox network
(bench/sandbox/sandbox_net.py); check.py starts it:

    python3 page_check.py --page /page --out /out --proxy http://egress:3128

THE PAGE is served from /page (a directory holding only index.html, or --
task r3f-stack -- a built Vite dist/: index.html plus /assets/*.js, which the
server's root resolves) by a stdlib server on 127.0.0.1 in this container; Chromium bypasses its proxy for
loopback, and everything else goes to the gate, which reaches only global
addresses on 80/443. On top of the gate, a request ROUTE allows only the CDN
hosts in CDN_HOSTS (a CHOICE: where pages load three.js, fonts and the like
from) and aborts the rest; every request is recorded with its decision, so a
page that needs the internet is visible as such.

THE FIXED SEQUENCE (the same inputs at the same times for every page):
  load (<= 90 s), first draw (<= 30 s)
  t05 / t10 / t15   default view, 5, 10 and 15 s after load (no input)
  fps               requestAnimationFrame over t05..t15 (idle); SOFTWARE
                    rendered (SwiftShader: the container has no GPU), so it
                    ranks pages against each other, not against a real card
  stats             after t15: three.js scene stats when three is present
                    (via three's own __THREE_DEVTOOLS__ hook, module or
                    global builds), WebGL draw counters for any page
  t16               2 s after the stats, still no input: the motion baseline
  orbit_right       drag from the centre 1/8 width to the right
  orbit_high        drag from the centre 1/12 height down (orbit up)
  zoom_out          5 wheel ticks away
The camera (three.js) is read before and after each input: `orbit` says
whether the drag moved it beyond the no-input baseline.

Math.random is SEEDED (--seed, default 1; mulberry32) so the same page
builds the same scene on every check. WebGPU is hidden as in
bench/octopus/browser_check.py (its only adapter here is SwiftShader, on
which three's WebGPU backend drew nothing): three falls back to WebGL2.

Writes /out/browser.json (rewritten after every phase, so a page that
hangs the browser still leaves what was measured) and /out/<shot>.png.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import threading
import time
import urllib.parse

W, H = 1280, 720
PORT = 8765
SHOT_TIMES = (5.0, 10.0, 15.0)
# CDN hosts a single-file page may load from (a CHOICE, 2026-09-26). Anything
# else is aborted and recorded; the gate still refuses non-global addresses.
CDN_HOSTS = ("cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com", "esm.sh",
             "cdn.skypack.dev", "ga.jspm.io", "esm.run", "threejs.org",
             "cdn.babylonjs.com", "fonts.googleapis.com", "fonts.gstatic.com")
# Caps so a page with millions of voxels cannot stall the stats pass.
MAX_INSTANCES_SCANNED = 500000
MAX_COLOR_KEYS = 4096
MAX_VERTEX_SAMPLES = 50000


def host_decision(url: str) -> tuple[str, str]:
    """(decision, host): `local` (this container's server), `cdn` (allowed),
    `data` (data:/blob:, no network) or `blocked`."""
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or "").lower()
    if u.scheme in ("data", "blob", "about"):
        return "data", ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return "local", host
    if u.scheme in ("http", "https") and any(
            host == h or host.endswith("." + h) for h in CDN_HOSTS):
        return "cdn", host
    return "blocked", host


def seed_js(seed: int | None) -> str:
    if seed is None:
        return ""
    return ("(() => { let s = %d >>> 0; Math.random = function () { s = (s + 0x6D2B79F5) | 0;"
            " let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;"
            " return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();" % int(seed))


HIDE_WEBGPU = r"""
try { Object.defineProperty(Navigator.prototype, 'gpu', {get() { return undefined; },
                                                        configurable: true}); } catch (e) {}
"""

INIT = r"""
(() => {
  const V = window.__voxel = {frames: [], ctx: [], revision: null, renders: 0,
    gl: {calls: 0, inst_calls: 0, instances: 0, prims: 0}};
  const R = window.__voxelRefs = {scenes: [], renderers: [], cams: new Map()};
  // three.js announces itself and every Scene / WebGLRenderer it builds to
  // __THREE_DEVTOOLS__ (module and global builds alike).
  try {
    const dt = new EventTarget();
    dt.addEventListener('register', (e) => {
      try { V.revision = String(e.detail && e.detail.revision); } catch (x) {} });
    dt.addEventListener('observe', (e) => {
      const o = e.detail;
      try {
        if (!o) return;
        if (o.isScene) { if (R.scenes.length < 50) R.scenes.push(o); return; }
        if (typeof o.render === 'function' && o.domElement && R.renderers.length < 10) {
          R.renderers.push(o);
          const r0 = o.render;
          o.render = function (scene, camera) {
            V.renders++;
            try { if (scene && camera) R.cams.set(scene, camera);
                  if (scene && scene.isScene && !R.scenes.includes(scene) && R.scenes.length < 50)
                    R.scenes.push(scene); } catch (x) {}
            return r0.apply(this, arguments);
          };
        }
      } catch (x) {}
    });
    Object.defineProperty(window, '__THREE_DEVTOOLS__', {value: dt, configurable: true, writable: true});
  } catch (e) {}
  try {
    const gc = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...a) {
      try { if (V.ctx.length < 50) V.ctx.push(String(type)); } catch (e) {}
      return gc.call(this, type, ...a);
    };
  } catch (e) {}
  const G = V.gl;
  const tri = (mode, count) => (mode === 4 ? count / 3 : mode === 5 || mode === 6 ? Math.max(count - 2, 0) : 0);
  for (const C of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
    if (!C) continue;
    const p = C.prototype;
    try {
      const da = p.drawArrays, de = p.drawElements;
      p.drawArrays = function (m, f, c) { G.calls++; G.prims += tri(m, c); return da.apply(this, arguments); };
      p.drawElements = function (m, c) { G.calls++; G.prims += tri(m, c); return de.apply(this, arguments); };
      if (p.drawArraysInstanced) {
        const dai = p.drawArraysInstanced, dei = p.drawElementsInstanced;
        p.drawArraysInstanced = function (m, f, c, n) { G.calls++; G.inst_calls++; G.instances += n;
          G.prims += tri(m, c) * n; return dai.apply(this, arguments); };
        p.drawElementsInstanced = function (m, c, t, o, n) { G.calls++; G.inst_calls++; G.instances += n;
          G.prims += tri(m, c) * n; return dei.apply(this, arguments); };
      }
    } catch (e) {}
  }
  const tick = (t) => { V.frames.push(Math.round(t)); if (V.frames.length > 20000)
    V.frames.splice(0, 10000); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
})();
"""

# The camera of the scene the page renders most: its world matrix, fov, zoom.
POSE_JS = r"""() => {
  const R = window.__voxelRefs; if (!R) return null;
  let best = null, n = -1;
  for (const [scene, cam] of R.cams) {
    let k = 0; try { scene.traverse(() => { k++; }); } catch (e) {}
    if (k > n) { n = k; best = cam; }
  }
  if (!best) return null;
  try { best.updateMatrixWorld && best.updateMatrixWorld(true); } catch (e) {}
  const e = best.matrixWorld && best.matrixWorld.elements;
  return {m: e ? Array.from(e).map(v => Math.round(v * 1e4) / 1e4) : null,
          type: best.type, fov: best.fov, zoom: best.zoom};
}"""

STATS_JS = r"""(caps) => {
  const V = window.__voxel || {}, R = window.__voxelRefs || {scenes: [], renderers: [], cams: new Map()};
  const out = {revision: V.revision || (window.__THREE__ ? String(window.__THREE__) : null)
                 || (window.THREE && window.THREE.REVISION) || null,
               scenes_observed: R.scenes.length, renderers_observed: R.renderers.length,
               renders: V.renders || 0, contexts: V.ctx || []};
  const q = (r, g, b) => ((Math.round(Math.min(Math.max(r, 0), 1) * 255) << 16)
                         | (Math.round(Math.min(Math.max(g, 0), 1) * 255) << 8)
                         | Math.round(Math.min(Math.max(b, 0), 1) * 255));
  const count = (s) => { let k = 0; try { s.traverse(() => { k++; }); } catch (e) {} return k; };
  const scenes = R.scenes.slice();
  for (const s of R.cams.keys()) if (!scenes.includes(s)) scenes.push(s);
  out.scene_sizes = scenes.map(count);
  if (!scenes.length) return out;
  const scene = scenes[out.scene_sizes.indexOf(Math.max(...out.scene_sizes))];
  const st = {objects: 0, visible_objects: 0, meshes: 0, box_meshes: 0, instanced_meshes: 0,
              instances: 0, box_instances: 0, points: 0, point_vertices: 0, lines: 0,
              sprites: 0, triangles: 0, triangles_nonbox: 0, lights: {}, geometry_types: {},
              box_sizes: {}, truncated: false};
  const mat = new Set(), inst = new Set(), vtx = new Set();
  const bbAll = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  const bbVox = bbAll.slice();
  let scanned = 0;
  const grow = (bb, x, y, z) => { if (x < bb[0]) bb[0] = x; if (y < bb[1]) bb[1] = y; if (z < bb[2]) bb[2] = z;
    if (x > bb[3]) bb[3] = x; if (y > bb[4]) bb[4] = y; if (z > bb[5]) bb[5] = z; };
  const apply = (e, x, y, z) => [e[0] * x + e[4] * y + e[8] * z + e[12],
                                 e[1] * x + e[5] * y + e[9] * z + e[13],
                                 e[2] * x + e[6] * y + e[10] * z + e[14]];
  try { scene.updateMatrixWorld && scene.updateMatrixWorld(true); } catch (e) {}
  try { scene.traverse(() => { st.objects++; }); } catch (e) {}
  const visit = (o) => {
    st.visible_objects++;
    if (o.isLight) st.lights[o.type] = (st.lights[o.type] || 0) + 1;
    if (o.isPoints) { st.points++; try { st.point_vertices += o.geometry.attributes.position.count; } catch (e) {} }
    if (o.isLine) st.lines++;
    if (o.isSprite) st.sprites++;
    if (!o.isMesh || !o.geometry) return;
    const g = o.geometry, pos = g.attributes && g.attributes.position;
    if (!pos) return;
    st.geometry_types[g.type] = (st.geometry_types[g.type] || 0) + 1;
    const isBox = g.type === 'BoxGeometry' || g.type === 'BoxBufferGeometry';
    let t = (g.index ? g.index.count : pos.count) / 3;
    if (g.drawRange && isFinite(g.drawRange.count)) t = Math.min(t, g.drawRange.count / 3);
    const n = o.isInstancedMesh ? (o.count || 0) : 1;
    if (o.isInstancedMesh) { st.instanced_meshes++; st.instances += n; if (isBox) st.box_instances += n; }
    else { st.meshes++; if (isBox) st.box_meshes++; }
    st.triangles += t * n; if (!isBox) st.triangles_nonbox += t * n;
    if (isBox && g.parameters && Object.keys(st.box_sizes).length < 50) {
      const k = [g.parameters.width, g.parameters.height, g.parameters.depth].map(v => +(+v).toFixed(3)).join('x');
      st.box_sizes[k] = (st.box_sizes[k] || 0) + n;
    }
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      try { if (m && m.color && mat.size < caps.keys) mat.add(m.color.getHex()); } catch (e) {}
    }
    try {
      if (o.instanceColor && o.instanceColor.array) {
        const a = o.instanceColor.array, lim = Math.min(n, caps.instances);
        for (let i = 0; i < lim && inst.size < caps.keys; i++) inst.add(q(a[3 * i], a[3 * i + 1], a[3 * i + 2]));
      }
    } catch (e) {}
    try {
      const c = g.attributes.color;
      if (c && mats.some(m => m && m.vertexColors)) {
        const isz = c.itemSize, arr = c.array, div = (arr instanceof Float32Array) ? 1 : 255;
        const step = Math.max(1, Math.floor(c.count / caps.vertices));
        for (let i = 0; i < c.count && vtx.size < caps.keys; i += step)
          vtx.add(q(arr[i * isz] / div, arr[i * isz + 1] / div, arr[i * isz + 2] / div));
      }
    } catch (e) {}
    try {
      if (!g.boundingBox && g.computeBoundingBox) g.computeBoundingBox();
      const b = g.boundingBox, w = o.matrixWorld.elements;
      if (!b) return;
      const cs = [];
      for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) cs.push([x, y, z]);
      if (o.isInstancedMesh) {
        const im = o.instanceMatrix.array, lim = Math.min(n, caps.instances - scanned);
        if (lim < n) st.truncated = true;
        for (let i = 0; i < lim; i++) {
          const e = im.subarray(16 * i, 16 * i + 16);
          for (const c of cs) { const p = apply(w, ...apply(e, c[0], c[1], c[2]));
            grow(bbAll, ...p); if (isBox) grow(bbVox, ...p); }
        }
        scanned += Math.max(lim, 0);
      } else {
        for (const c of cs) { const p = apply(w, c[0], c[1], c[2]); grow(bbAll, ...p); if (isBox) grow(bbVox, ...p); }
      }
    } catch (e) {}
  };
  try { (scene.traverseVisible ? scene.traverseVisible(visit) : scene.traverse(visit)); } catch (e) { st.traverse_error = String(e).slice(0, 200); }
  const box = (bb) => isFinite(bb[0]) ? {min: bb.slice(0, 3).map(v => +v.toFixed(3)),
      max: bb.slice(3).map(v => +v.toFixed(3)),
      size: [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]].map(v => +v.toFixed(3))} : null;
  st.bbox_all = box(bbAll); st.bbox_voxels = box(bbVox);
  st.voxels = st.box_meshes + st.box_instances;
  st.cube_equiv_nonbox = Math.round(st.triangles_nonbox / 12);
  st.triangles = Math.round(st.triangles); st.triangles_nonbox = Math.round(st.triangles_nonbox);
  st.material_colors = mat.size; st.instance_colors = inst.size; st.vertex_colors = vtx.size;
  const all = new Set([...mat, ...inst, ...vtx]);
  st.distinct_colors = all.size; st.colors_capped = all.size >= caps.keys;
  try { st.fog = scene.fog ? scene.fog.type || (scene.fog.isFogExp2 ? 'FogExp2' : 'Fog') : null; } catch (e) {}
  try { st.background = scene.background && scene.background.isColor ? scene.background.getHexString()
        : (scene.background ? (scene.background.type || 'texture') : null); } catch (e) {}
  out.scene = st;
  out.renderers = R.renderers.slice(0, 3).map(r => {
    const x = {};
    try { x.render = Object.assign({}, r.info.render); x.memory = Object.assign({}, r.info.memory);
          x.programs = r.info.programs ? r.info.programs.length : null; } catch (e) {}
    try { x.shadow_map = !!(r.shadowMap && r.shadowMap.enabled); x.pixel_ratio = r.getPixelRatio();
          x.tone_mapping = r.toneMapping; x.size = [r.domElement.width, r.domElement.height]; } catch (e) {}
    return x;
  });
  return out;
}"""

GL_JS = "() => Object.assign({}, (window.__voxel || {}).gl || {})"
PROBE_JS = r"""() => {
  const out = {};
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    out.webgl = gl ? (gl instanceof WebGL2RenderingContext ? 'webgl2' : 'webgl') : null;
    const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
    out.renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null;
  } catch (e) { out.error = String(e).slice(0, 200); }
  out.canvases = Array.from(document.querySelectorAll('canvas')).map(c => {
    const r = c.getBoundingClientRect();
    return {w: c.width, h: c.height, x: Math.round(r.x), y: Math.round(r.y),
            cw: Math.round(r.width), ch: Math.round(r.height)}; });
  return out;
}"""


def serve(root: str) -> http.server.ThreadingHTTPServer:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):          # noqa: D401 - the request log is the route's
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT),
                                          functools.partial(Quiet, directory=root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def pose_delta(a: dict | None, b: dict | None) -> float | None:
    if not a or not b or not a.get("m") or not b.get("m"):
        return None
    d = sum((x - y) ** 2 for x, y in zip(a["m"], b["m"])) ** 0.5
    if a.get("zoom") and b.get("zoom"):
        d += abs(a["zoom"] - b["zoom"])
    return round(d, 4)


def orbit_verdict(baseline: float | None, moved: list[float | None]) -> dict:
    """Did an input move the camera beyond what it did with no input?
    A drag counts when its change is > 3x the 1 s no-input change and > 0.01
    (CHOICES)."""
    if baseline is None or any(m is None for m in moved):
        return {"camera": False, "orbit": None, "why": "no three.js camera observed"}
    hit = [m > max(3 * baseline, 0.01) for m in moved]
    return {"camera": True, "orbit": any(hit[:2]), "zoom": bool(hit[2:] and hit[2]),
            "baseline": baseline, "moved": moved}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="/page")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--seed", type=int, default=1, help="Math.random seed; -1 = unseeded")
    ap.add_argument("--webgpu", choices=("hide", "allow"), default="hide")
    # A SERVED APP instead of /page (check.check_served; bench/octopus
    # pagoda.py): the app's own server, reached in its container's network
    # namespace on loopback (a `local` request), nothing served here.
    ap.add_argument("--url", default=None,
                    help="load this loopback URL instead of serving --page")
    a = ap.parse_args()
    if a.url and host_decision(a.url)[0] != "local":
        ap.error("--url must be a loopback URL (the app's own server)")
    from playwright.sync_api import sync_playwright
    os.makedirs(a.out, exist_ok=True)
    seed = None if a.seed < 0 else a.seed
    res: dict = {"viewport": [W, H], "seed": seed, "webgpu_mode": a.webgpu, "proxy": a.proxy,
                 "cdn_hosts_allowed": list(CDN_HOSTS), "events": [], "console": [],
                 "page_errors": [], "requests": [], "shots": {}}
    T0 = time.time()
    out_json = os.path.join(a.out, "browser.json")

    def save():
        res["wall_s"] = round(time.time() - T0, 1)
        with open(out_json + ".tmp", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        os.replace(out_json + ".tmp", out_json)

    def mark(name, **kw):
        res["events"].append({"t": round(time.time() - T0, 2), "name": name, **kw})
        save()

    srv = None if a.url else serve(a.page)
    res["url"] = a.url or f"http://127.0.0.1:{PORT}/index.html"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist",
            "--enable-unsafe-webgpu", "--autoplay-policy=no-user-gesture-required",
            *([f"--proxy-server={a.proxy}"] if a.proxy else [])])
        res["browser_version"] = browser.version
        ctx = browser.new_context(viewport={"width": W, "height": H}, service_workers="block")
        if a.webgpu == "hide":
            ctx.add_init_script(HIDE_WEBGPU)
        if seed is not None:
            ctx.add_init_script(seed_js(seed))
        ctx.add_init_script(INIT)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.on("console", lambda m: len(res["console"]) < 300 and res["console"].append(
            {"t": round(time.time() - T0, 2), "type": m.type, "text": m.text[:500]}))
        page.on("pageerror", lambda e: len(res["page_errors"]) < 100 and res["page_errors"].append(
            {"t": round(time.time() - T0, 2), "text": str(e)[:800]}))
        reqs: dict = {}

        def on_route(route):
            url = route.request.url
            dec, host = host_decision(url)
            rec = {"t": round(time.time() - T0, 2), "url": url[:200], "host": host,
                   "type": route.request.resource_type, "decision": dec}
            if len(res["requests"]) < 500:
                res["requests"].append(rec)
                reqs[url] = rec
            if dec == "blocked":
                route.abort("blockedbyclient")
            else:
                route.continue_()
        ctx.route("**/*", on_route)
        page.on("requestfailed", lambda r: reqs.get(r.url, {}).update(
            failure=(r.failure or "")[:200]))
        page.on("response", lambda r: reqs.get(r.url, {}).update(status=r.status))

        url = res["url"]
        try:
            page.goto(url, wait_until="load", timeout=90000)
            mark("loaded")
        except Exception as e:                                   # noqa: BLE001
            res["load_error"] = f"{type(e).__name__}: {e}"[:500]
            mark("load_failed")
        t_load = time.time()
        try:
            res["probe"] = page.evaluate(PROBE_JS)
        except Exception as e:                                   # noqa: BLE001
            res["probe"] = {"error": str(e)[:200]}
        # first draw: a three render, or any WebGL draw call, or a 2d context
        first = None
        while time.time() - t_load < 30:
            try:
                g = page.evaluate("() => { const V = window.__voxel || {}; return "
                                  "[V.renders || 0, (V.gl || {}).calls || 0, (V.ctx || []).length]; }")
            except Exception:                                    # noqa: BLE001
                g = [0, 0, 0]
            if g[0] or g[1] or g[2]:
                first = round(time.time() - t_load, 2)
                # a 2d context counts on creation (its draws are not hooked)
                res["first_draw_by"] = ("three_render" if g[0] else "webgl_draw" if g[1]
                                        else "canvas_context")
                break
            time.sleep(0.25)
        res["first_draw_s"] = first
        mark("first_draw", s=first)

        def shot(name):
            try:
                page.screenshot(path=os.path.join(a.out, f"{name}.png"), timeout=60000)
                res["shots"][name] = {"t_after_load": round(time.time() - t_load, 2)}
            except Exception as e:                               # noqa: BLE001
                res["shots"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
            mark("shot", file=name)

        def ev(js, *arg, default=None):
            try:
                return page.evaluate(js, *arg)
            except Exception as e:                               # noqa: BLE001
                res.setdefault("eval_errors", []).append(f"{type(e).__name__}: {e}"[:200])
                return default

        poses, gl, perf = {}, {}, {}
        for ts in SHOT_TIMES:
            wait = t_load + ts - time.time()
            if wait > 0:
                time.sleep(wait)
            name = f"t{int(ts):02d}"
            perf[name] = ev("() => performance.now()")
            gl[name] = ev(GL_JS)
            shot(name)
            poses[name] = ev(POSE_JS)
        frames = ev("() => (window.__voxel || {}).frames || []", default=[]) or []
        if perf.get("t05") is not None and perf.get("t15") is not None:
            win = [f for f in frames if perf["t05"] <= f <= perf["t15"]]
            dur = (perf["t15"] - perf["t05"]) / 1000.0
            gaps = [b - x for x, b in zip(win, win[1:])]
            res["fps"] = {"frames": len(win), "seconds": round(dur, 2),
                          "fps": round(len(win) / dur, 2) if dur > 0 else None,
                          "max_gap_ms": max(gaps) if gaps else None,
                          "software_rendered": True}
            g0, g1 = gl.get("t05") or {}, gl.get("t15") or {}
            nf = max(len(win), 1)
            res["gl_per_frame"] = {k: round(((g1.get(k) or 0) - (g0.get(k) or 0)) / nf, 1)
                                   for k in ("calls", "inst_calls", "instances", "prims")}
        res["gl_totals"] = gl.get("t15")
        mark("fps")
        res["stats"] = ev(STATS_JS, {"instances": MAX_INSTANCES_SCANNED, "keys": MAX_COLOR_KEYS,
                                     "vertices": MAX_VERTEX_SAMPLES})
        mark("stats")
        # the no-input baseline, over about as long as one drag and its settle
        poses["pre_t16"] = ev(POSE_JS)
        time.sleep(2.0)
        shot("t16")
        poses["t16"] = ev(POSE_JS)

        def drag(dx, dy):
            try:
                page.mouse.move(W / 2, H / 2)
                page.mouse.down()
                page.mouse.move(W / 2 + dx, H / 2 + dy, steps=20)
                page.mouse.up()
            except Exception as e:                               # noqa: BLE001
                res.setdefault("input_errors", []).append(str(e)[:200])

        drag(W / 8, 0)
        time.sleep(1.5)
        shot("orbit_right")
        poses["orbit_right"] = ev(POSE_JS)
        drag(0, H / 12)
        time.sleep(1.5)
        shot("orbit_high")
        poses["orbit_high"] = ev(POSE_JS)
        try:
            page.mouse.move(W / 2, H / 2)
            for _ in range(5):
                page.mouse.wheel(0, 120)
                time.sleep(0.1)
        except Exception as e:                                   # noqa: BLE001
            res.setdefault("input_errors", []).append(str(e)[:200])
        time.sleep(1.5)
        shot("zoom_out")
        poses["zoom_out"] = ev(POSE_JS)
        res["poses"] = poses
        base = pose_delta(poses.get("pre_t16"), poses.get("t16"))
        res["camera_input"] = orbit_verdict(base, [
            pose_delta(poses.get("t16"), poses.get("orbit_right")),
            pose_delta(poses.get("orbit_right"), poses.get("orbit_high")),
            pose_delta(poses.get("orbit_high"), poses.get("zoom_out"))])
        mark("inputs_done")
        browser.close()
    if srv is not None:
        srv.shutdown()
    save()
    print(json.dumps({"ok": True, "wall_s": res["wall_s"], "page_errors": len(res["page_errors"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
