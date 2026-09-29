#!/usr/bin/env python
"""bench/voxel offline: the request as sent, extraction, the rep record and
its statuses (429 = not run), resume, the preflight refusal, the pixel
measures, the check summary and the compare page (blind mode included).
Fake model responses and a tiny page; no network, no Docker, no GPU.

    python bench/voxel/test_voxel.py      -> "N/M checks passed"
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.error
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check as C  # noqa: E402
import compare as CMP  # noqa: E402
import page_check as P  # noqa: E402
import run as R  # noqa: E402

_results: list[tuple[bool, str, str]] = []
KEY = "sk-test-SECRET-never-printed-123"
VERBATIM = ("Design and create a very creative, elaborate, and detailed voxel art scene of a "
            "pagoda in a beautiful garden with trees, including some cherry blossoms, add a "
            "village with people living in it. Make the scene impressive and varied and use "
            "colorful voxels. Make it really detailed use as many voxels as you want, we have "
            "a power machine here. Create a single HTML file.")
FIXTURE = os.path.join(HERE, "fixtures", "sample_three.html")


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def fake_response(content: str, tier: str = "xhigh", finish: str = "stop",
                  utility: bool = False) -> dict:
    return {"id": "x", "model": "yamadori",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content,
                                     "reasoning_content": "thinking about pagodas"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9000,
                      "completion_tokens_details": {"reasoning_tokens": 3000}},
            "x_yamadori": {"tier": tier, "tier_requested": tier, "utility": utility,
                           "route": {"class": "code_generation"},
                           "selection": {"fanout": 3}, "fanout": {"n": 2},
                           "deep": {"trigger": "none", "fired": False,
                                    "think_tool": {"offered": True, "calls": []}},
                           "usage": {"generations": 1}, "hops": 1,
                           "fold_back": [{"phrase": "Compared two approaches"}],
                           "session": {"source": "minted"}}}


PAGE = "<!DOCTYPE html>\n<html><body><canvas></canvas><script>1</script></body></html>"
ANSWER = ("Today I was inspired by lantern.\n\nAfter thinking deeply, here is the scene.\n\n"
          "```html\n" + PAGE + "\n```\n\nOpen it in a browser.")


def poster_seq(*outs):
    """A fake post(): returns/raises each item in turn, records the calls."""
    calls = []
    seq = list(outs)

    def post(url, key, body, arm, timeout):
        calls.append({"url": url, "key": key, "body": body, "arm": arm,
                      "headers": R.request_headers(key, arm)})
        o = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(o, BaseException):
            raise o
        return o
    post.calls = calls
    return post


def cfg(d: str, **kw) -> dict:
    return dict({"tag": "t", "run_dir": d, "url": "http://fake", "key": KEY,
                 "timeout": 10, "busy_max_s": 60, "max_tokens": None}, **kw)


def all_text(d: str) -> str:
    out = []
    for root, _dirs, files in os.walk(d):
        for fn in files:
            with open(os.path.join(root, fn), encoding="utf-8", errors="replace") as f:
                out.append(f.read())
    return "\n".join(out)


# ------------------------------------------------------------------ tests --
def test_prompt_and_arms():
    check(R.PROMPT == VERBATIM, "the prompt is the public one, verbatim")
    check(R.PROMPT_SHA256 == "2cee15c5c2bfd42e59609e1b948d3ea1d4e4a72d2f136c12e2ea4b09b22fe8c4",
          "the prompt's sha256 is pinned", R.PROMPT_SHA256)
    check(R.DEFAULT_ARMS == ("xhigh",), "the default (and only planned) arm is xhigh "
          "(operator 2026-09-26)", str(R.DEFAULT_ARMS))
    a = R.parse_arm("xhigh")
    check(a["effort"] == "xhigh" and a["features"] is None and a["model"] == "yamadori",
          "xhigh: our stack as a client gets it (no header, model yamadori)", json.dumps(a))
    off = R.parse_arm("xhigh-off")
    check(off["effort"] == "xhigh" and off["features"] == R.ALL_OFF
          and "effort" not in off["features"],
          "xhigh-off: same tier, everything of ours forced off (effort-matched)")
    sib = R.parse_arm("low@sibling-27b")
    check(sib["model"] == "sibling-27b" and sib["effort"] == "low" and sib["arm"] == "low@sibling-27b",
          "ARM@MODEL names a sibling model")
    try:
        R.parse_arm("medium")
        check(False, "an unknown arm is refused")
    except ValueError:
        check(True, "an unknown arm is refused")
    body = R.request_body(a)
    check(set(body) == {"model", "reasoning_effort", "messages"}
          and body["messages"] == [{"role": "user", "content": VERBATIM}],
          "the body is one user message: no system, tools, temperature or max_tokens",
          json.dumps(body)[:200])
    check(R.request_body(a, 40000)["max_tokens"] == 40000, "--max-tokens is sent when asked")
    h = R.request_headers(KEY, a)
    check("X-Yamadori-Features" not in h and h["Authorization"] == "Bearer " + KEY,
          "xhigh sends no features header; the key only as Authorization")
    h2 = R.request_headers(KEY, off)
    check(json.loads(h2["X-Yamadori-Features"]) == R.ALL_OFF, "xhigh-off sends the forced-off header")


def test_extract():
    e = R.extract_html(ANSWER)
    check(e["method"] == "fenced" and e["html"] == PAGE and e["closed_fence"],
          "a fenced html block after the fold-back lines is the page", json.dumps(
              {k: v for k, v in e.items() if k != "html"}))
    e = R.extract_html("Here you go:\n\n" + PAGE + "\n\nEnjoy.")
    check(e["method"] == "raw" and e["html"] == PAGE, "an unfenced document is extracted raw")
    cut = "```html\n<!DOCTYPE html>\n<html><body><script>const a = 1;"
    e = R.extract_html(cut)
    check(e["method"] == "fenced" and e["closed_fence"] is False and not e["has_html_close"]
          and e["html"].startswith("<!DOCTYPE"),
          "an unclosed fence (a cut reply) is extracted as written and flagged")
    multi = ("```html\n<canvas id=c></canvas>\n```\n\n```css\nbody{}\n```\n\n"
             "~~~HTML\n" + PAGE + "\n~~~\n")
    e = R.extract_html(multi)
    check(e["html"] == PAGE and e["page_blocks"] == 2 and e["fenced_blocks"] == 3
          and e["fenced_langs"].get("css") == 1,
          "the largest page block wins; the others are counted", json.dumps(
              {k: v for k, v in e.items() if k != "html"}))
    e = R.extract_html("I cannot do that.")
    check(e["method"] == "none" and e["html"] is None, "no page: method none")
    fx = open(FIXTURE, encoding="utf-8").read()
    e = R.extract_html("```html\n" + fx + "```")
    check(e["three_js"] and "cdn.jsdelivr.net" in e["hosts_named"] and e["has_doctype"],
          "the fixture: three.js and its CDN host are noted", json.dumps(e["hosts_named"]))


def test_run_rep():
    d = tempfile.mkdtemp(prefix="vx-rep-")
    try:
        arm = R.parse_arm("xhigh")
        post = poster_seq((200, fake_response(ANSWER)))
        row = R.run_rep(cfg(d), arm, 1, 1, poster=post, sleep=lambda s: None)
        rdir = os.path.join(d, row["dir"])
        check(row["status"] == "ok" and row["page"].endswith("page.html")
              and open(os.path.join(d, row["page"]), encoding="utf-8").read() == PAGE,
              "200 + stop + the arm's tier: ok, page.html written", row["status"])
        resp = json.load(open(os.path.join(rdir, "response.json"), encoding="utf-8"))
        check(resp["response"]["x_yamadori"]["tier"] == "xhigh"
              and resp["response"]["choices"][0]["message"]["reasoning_content"],
              "response.json keeps the whole body, x_yamadori and reasoning included")
        check(row["usage"]["completion_tokens"] == 9000 and row["wall_s"] >= 0
              and row["x"]["route"] == "code_generation" and row["x"]["fanout_n"] == 2
              and row["x"]["deep"]["think_calls"] == 0,
              "the row carries usage, wall time and the stack's decisions", json.dumps(row["x"])[:300])
        check(KEY not in all_text(d) and KEY not in json.dumps(row), "the key is in no file and no row")
        check(post.calls[0]["body"]["messages"][0]["content"] == VERBATIM,
              "what went out is the verbatim prompt")

        row = R.run_rep(cfg(d), arm, 2, 1, poster=poster_seq((200, fake_response(ANSWER, tier="low"))))
        check(row["status"] == "mismatch" and "tier" in row["mismatch"][0],
              "x_yamadori naming another tier: mismatch (a stack error)", str(row.get("mismatch")))
        row = R.run_rep(cfg(d), arm, 3, 1, poster=poster_seq((200, fake_response(ANSWER, utility=True))))
        check(row["status"] == "mismatch", "served as a utility call: mismatch")
        row = R.run_rep(cfg(d), arm, 4, 1, poster=poster_seq((200, fake_response(ANSWER, finish="length"))))
        check(row["status"] == "length" and row.get("page"), "finish length: done, flagged, page kept")

        waits = []
        post = poster_seq((429, {"error": {"code": "busy"}}))
        row = R.run_rep(cfg(d, busy_max_s=60), arm, 5, 1, poster=post, sleep=waits.append)
        check(row["status"] == "not_run" and len(post.calls) == 3 and row["busy_429s"] == 2
              and not row.get("page"),
              "429 past --busy-max-s: not_run (never a failure), no page",
              f"calls={len(post.calls)} waits={waits}")
        post = poster_seq((429, {}), (200, fake_response(ANSWER)))
        t = iter([100.0, 100.0, 170.0, 170.0])
        row = R.run_rep(cfg(d), arm, 6, 1, poster=post, sleep=lambda s: None, clock=lambda: next(t))
        check(row["status"] == "ok" and row["busy_wait_s"] == R.BUSY_WAIT_S
              and row["wall_s"] == 70.0 - R.BUSY_WAIT_S,
              "429 then 200: ok, and the wait is kept out of wall_s", json.dumps(
                  {k: row[k] for k in ("wall_s", "busy_wait_s")}))
        refused = urllib.error.URLError(ConnectionRefusedError(10061, "refused"))
        row = R.run_rep(cfg(d, busy_max_s=0), arm, 7, 1, poster=poster_seq(refused))
        check(row["status"] == "not_run", "a refused connection past the wait: not_run")
        row = R.run_rep(cfg(d), arm, 8, 1, poster=poster_seq((500, {"error": {"message": "x"}})))
        check(row["status"] == "http_error" and not row.get("page"), "HTTP 500: http_error")
        row = R.run_rep(cfg(d), arm, 9, 1, poster=poster_seq(TimeoutError("timed out")))
        check(row["status"] == "exception" and "TimeoutError" in row["error"], "a timeout: exception")
        row = R.run_rep(cfg(d), arm, 10, 1, poster=poster_seq((200, fake_response("No."))))
        check(row["status"] == "ok" and not row.get("page") and row["extract"]["method"] == "none",
              "an answer with no page: ok, no page, method none")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_plan_resume():
    arms = [R.parse_arm("xhigh")]
    done = {("xhigh", 1): {"status": "ok", "attempt": 1},
            ("xhigh", 2): {"status": "not_run", "attempt": 1},
            ("xhigh", 3): {"status": "exception", "attempt": 2}}
    todo = [(a["arm"], r, k) for a, r, k in R.plan(arms, 4, done, False)]
    check(todo == [("xhigh", 2, 2), ("xhigh", 4, 1)],
          "resume: done skipped, not_run rerun (next attempt), errors kept", str(todo))
    todo = [(a["arm"], r, k) for a, r, k in R.plan(arms, 4, done, True)]
    check(("xhigh", 3, 3) in todo, "--retry-errors reruns a stack error", str(todo))


class FakeOcto:
    def __init__(self, busy=(), idle=True):
        self.busy, self.idle, self.BUSY = list(busy), idle, ()

    def busy_processes(self):
        return self.busy

    def slots_idle(self):
        return self.idle, [(0, not self.idle)]


def test_main_flow():
    d = tempfile.mkdtemp(prefix="vx-main-")
    old = (R.RESULTS, R._octo, R.models_listed, R.post, R.preflight)
    kf = os.path.join(d, "key.txt")
    with open(kf, "w") as f:
        f.write(KEY + "\n")
    orig_pf = R.preflight
    try:
        R.RESULTS = d
        R.preflight = lambda url, key, arms: orig_pf(url, key, arms, sleep=lambda s: None)
        sent = []
        R.post = lambda url, key, body, arm, timeout: (sent.append(body) or (200, fake_response(ANSWER)))
        R.models_listed = lambda url, key: ("200", ["yamadori"])
        R._octo = FakeOcto(busy=["123 python bench/domain/run.py"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = R.main(["--key-file", kf, "--tag", "m1"])
        check(rc == 1 and not sent and not os.path.exists(os.path.join(d, "m1")),
              "another GPU consumer running: refused, nothing sent, nothing written",
              buf.getvalue()[-300:])
        R._octo = FakeOcto(idle=False)
        with redirect_stdout(buf):
            rc = R.main(["--key-file", kf, "--tag", "m1"])
        check(rc == 1 and not sent, "a slot processing: refused")
        R._octo = FakeOcto()
        with redirect_stdout(buf):
            rc = R.main(["--key-file", kf, "--tag", "m1", "--arms", "xhigh@nope"])
        check(rc == 1 and not sent, "a model /v1/models does not list: refused (the proxy "
              "would silently serve yamadori)")
        with redirect_stdout(buf):
            rc = R.main(["--key-file", kf, "--tag", "m1"])
        rows = R.read_rows(os.path.join(d, "m1", "summary.jsonl"))
        man = json.load(open(os.path.join(d, "m1", "manifest.json"), encoding="utf-8"))
        check(rc == 0 and len(sent) == 3 and [r["rep"] for r in rows] == [1, 2, 3]
              and all(r["status"] == "ok" and r["arm"] == "xhigh" for r in rows),
              "the default run: xhigh, 3 reps, 3 rows", f"rc={rc} sent={len(sent)}")
        check(man["prompt"] == VERBATIM and man["arms"]["xhigh"]["effort"] == "xhigh"
              and man["prompt_sha256"] == R.PROMPT_SHA256, "the manifest records prompt and arm")
        check(KEY not in buf.getvalue() and KEY not in all_text(os.path.join(d, "m1")),
              "the key is never printed or written")
        with redirect_stdout(buf):
            R.main(["--key-file", kf, "--tag", "m1"])
        check(len(sent) == 3, "a rerun of a finished tag sends nothing (resume)")
    finally:
        R.RESULTS, R._octo, R.models_listed, R.post, R.preflight = old
        shutil.rmtree(d, ignore_errors=True)


def test_octopus_busy_list():
    o = R.octopus_run()
    check("voxel/run.py" in o.BUSY and "voxel\\run.py" in o.BUSY,
          "bench/octopus/run.py's BUSY list names this runner (other runners refuse beside it)")
    check(callable(o.busy_processes) and callable(o.slots_idle),
          "the busy check is bench/octopus/run.py's own")


def test_page_check_helpers():
    check(P.host_decision("http://127.0.0.1:8765/index.html")[0] == "local"
          and P.host_decision("https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js")
          == ("cdn", "cdn.jsdelivr.net")
          and P.host_decision("https://evil.example.com/x.js")[0] == "blocked"
          and P.host_decision("https://cdn.jsdelivr.net.evil.com/x")[0] == "blocked"
          and P.host_decision("data:image/png;base64,AAAA")[0] == "data",
          "the route: loopback local, CDN allowed, anything else (a look-alike too) blocked")
    a = {"m": [1.0] * 16, "zoom": 1}
    b = {"m": [1.0] * 12 + [4.0, 5.0, 1.0, 1.0], "zoom": 1}
    check(P.pose_delta(a, a) == 0 and P.pose_delta(a, b) == 5.0 and P.pose_delta(a, None) is None,
          "pose delta: the camera matrix distance")
    v = P.orbit_verdict(0.001, [2.0, 0.0, 0.5])
    check(v["orbit"] is True and v["zoom"] is True, "a drag that moves the camera: orbit")
    v = P.orbit_verdict(1.0, [1.5, 1.2, 1.0])
    check(v["orbit"] is False, "an auto-rotating camera the drag does not beat: no orbit")
    check(P.orbit_verdict(None, [None, None, None])["orbit"] is None,
          "no three.js camera: orbit unknown (pixels decide on the host)")
    check("12345" in P.seed_js(12345) and P.seed_js(None) == "", "Math.random seeding")
    node = shutil.which("node")
    if not check(bool(node), "node is available for the JS syntax check"):
        return
    d = tempfile.mkdtemp(prefix="vx-js-")
    try:
        for name, src in (("INIT", P.INIT), ("POSE", "const f = " + P.POSE_JS + ";"),
                          ("STATS", "const f = " + P.STATS_JS + ";"),
                          ("GL", "const f = " + P.GL_JS + ";"), ("PROBE", "const f = " + P.PROBE_JS + ";"),
                          ("SEED", P.seed_js(1)), ("HIDE", P.HIDE_WEBGPU)):
            p = os.path.join(d, name + ".js")
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            r = subprocess.run([node, "--check", p], capture_output=True, text=True)
            check(r.returncode == 0, f"page_check {name} script parses (node --check)",
                  r.stderr[-300:])
        # The stats function on a fake three.js scene, in node: the counts it
        # must report for 400 box instances in 4 colours + 1 box mesh + a plane.
        harness = r"""
const V = {}; globalThis.window = {__voxel: {revision: '170', ctx: ['webgl2'], renders: 9}};
const ident = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const box = (w) => ({type: 'BoxGeometry', parameters: {width: w, height: w, depth: w},
  attributes: {position: {count: 24}}, index: {count: 36}, drawRange: {count: Infinity},
  boundingBox: {min: {x: -w/2, y: -w/2, z: -w/2}, max: {x: w/2, y: w/2, z: w/2}}});
const im = new Float32Array(400 * 16), ic = new Float32Array(400 * 3);
const cols = [[1,0.5,0.6],[0.3,0.6,0.25],[0.5,0.4,0.25],[0.8,0.8,0.8]];
for (let i = 0; i < 400; i++) { im.set(ident, 16*i); im[16*i+12] = i % 20; im[16*i+14] = Math.floor(i/20);
  ic.set(cols[i % 4], 3*i); }
const kids = [
  {isInstancedMesh: true, isMesh: true, count: 400, geometry: box(1), material: {color: {getHex: () => 0xffffff}},
   instanceMatrix: {array: im}, instanceColor: {array: ic}, matrixWorld: {elements: ident}},
  {isMesh: true, geometry: box(2), material: {color: {getHex: () => 0xcc3333}}, matrixWorld: {elements: ident}},
  {isMesh: true, geometry: {type: 'PlaneGeometry', attributes: {position: {count: 4}}, index: {count: 6},
   drawRange: {count: Infinity}, boundingBox: {min: {x:-50,y:0,z:-50}, max: {x:50,y:0,z:50}}},
   material: {color: {getHex: () => 0x33aa33}}, matrixWorld: {elements: ident}},
  {isLight: true, type: 'DirectionalLight'}];
const scene = {isScene: true, traverse(f) { f(scene); kids.forEach(f); },
  traverseVisible(f) { f(scene); kids.forEach(f); }, updateMatrixWorld() {}};
window.__voxelRefs = {scenes: [scene], renderers: [], cams: new Map()};
const out = f({instances: 500000, keys: 4096, vertices: 50000});
console.log(JSON.stringify(out));
"""
        p = os.path.join(d, "stats_run.js")
        with open(p, "w", encoding="utf-8") as f:
            f.write("const f = " + P.STATS_JS + ";\n" + harness)
        r = subprocess.run([node, p], capture_output=True, text=True)
        try:
            s = json.loads(r.stdout.strip().splitlines()[-1])["scene"]
        except (ValueError, IndexError, KeyError):
            s = {}
        check(s.get("voxels") == 401 and s.get("box_instances") == 400 and s.get("box_meshes") == 1
              and s.get("instance_colors") == 4 and s.get("distinct_colors") == 7
              and s.get("bbox_voxels", {}).get("size") == [20.5, 2, 20.5]
              and s.get("bbox_all", {}).get("size") == [100, 2, 100]
              and s.get("lights") == {"DirectionalLight": 1} and s.get("triangles") == 400 * 12 + 12 + 2,
              "the stats function on a fake scene: voxels 401, 4 instance colours, 7 in all, "
              "voxel extent 20.5x2x20.5 vs 100x2x100 with the ground", (r.stdout + r.stderr)[-600:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_check_served():
    """A SERVED app (bench/octopus/pagoda.py): page_check.py --url in the
    app container's network namespace; the same sequence, nothing served by
    the checker, and only a loopback URL accepted."""
    src = open(os.path.join(HERE, "page_check.py"), encoding="utf-8").read()
    check('ap.add_argument("--url"' in src and "srv = None if a.url else serve(a.page)" in src
          and 'host_decision(a.url)[0] != "local"' in src,
          "page_check --url: loads the app's own server, serves nothing, loopback only")
    r = subprocess.run([sys.executable, os.path.join(HERE, "page_check.py"), "--url",
                        "https://example.com/"], capture_output=True, text=True, timeout=60)
    check(r.returncode == 2 and "loopback" in r.stderr,
          "page_check --url refuses a non-loopback URL", (r.stdout + r.stderr)[-300:])
    seen = []
    real_docker, real_pw = C.docker, C.ensure_pw_image

    def fake(args, timeout=120):
        seen.append(list(args))
        return subprocess.CompletedProcess(args, 0, "{}", "")
    d = tempfile.mkdtemp(prefix="vx-served-")
    C.docker, C.ensure_pw_image = fake, (lambda: None)
    try:
        rec = C.check_served("octo-app-x", "http://localhost:3001/", d, "lab")
    finally:
        C.docker, C.ensure_pw_image = real_docker, real_pw
        shutil.rmtree(d, ignore_errors=True)
    argv = seen[0] if seen else []
    check(argv[:1] == ["run"] and "container:octo-app-x" in argv
          and argv[argv.index("--url") + 1] == "http://localhost:3001/"
          and "/checker/page_check.py" in argv and "--proxy" in argv,
          "check_served: page_check.py --url in the app's network namespace, the gate as proxy",
          json.dumps(argv))
    check(rec.get("error") == "page_check produced no browser.json",
          "check_served: no browser.json is an error, never a measurement", json.dumps(rec)[:300])


def _png(path, fill=(135, 206, 235), squares=()):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (320, 180), fill)
    dr = ImageDraw.Draw(im)
    for (x, y, s, c) in squares:
        dr.rectangle([x, y, x + s, y + s], fill=c)
    im.save(path)


def test_check_host_side():
    d = tempfile.mkdtemp(prefix="vx-check-")
    try:
        _png(os.path.join(d, "blank.png"))
        sq = [(20 + 30 * i, 40 + (i % 3) * 30, 25, ((i * 50) % 255, (i * 90) % 255, (i * 30) % 255))
              for i in range(9)]
        _png(os.path.join(d, "scene.png"), squares=sq)
        b, s = C.img_stats(os.path.join(d, "blank.png")), C.img_stats(os.path.join(d, "scene.png"))
        check(not b["nonblank"] and b["bg_share"] == 1.0, "a one-colour shot is blank", json.dumps(b))
        check(s["nonblank"] and s["colours_q5"] >= 9, "a shot with coloured squares is not", json.dumps(s))
        # shots: no motion without input, a big change after the drag
        for n in ("t05", "t10", "t15", "t16"):
            _png(os.path.join(d, n + ".png"), squares=sq)
        moved = [(x + 60, y, s_, c) for (x, y, s_, c) in sq]
        for n in ("orbit_right", "orbit_high", "zoom_out"):
            _png(os.path.join(d, n + ".png"), squares=moved)
        pix = C.pixel_evidence(d)
        check(pix["no_input"] == 0 and pix["drag_right"] > C.ORBIT_PIXEL_MIN and pix["orbit"] is True,
              "pixels: a drag that changes a still picture is orbit evidence", json.dumps(pix))
        browser = {"events": [{"t": 1, "name": "loaded"}], "first_draw_s": 0.4,
                   "page_errors": [{"text": "TypeError: x is undefined"}],
                   "console": [{"type": "error", "text": "Failed to load resource"},
                               {"type": "log", "text": "hi"}],
                   "requests": [{"host": "127.0.0.1", "decision": "local", "status": 200},
                                {"host": "cdn.jsdelivr.net", "decision": "cdn", "status": 200},
                                {"host": "tracker.example.com", "decision": "blocked",
                                 "failure": "net::ERR_BLOCKED_BY_CLIENT"}],
                   "fps": {"fps": 11.5, "max_gap_ms": 240}, "probe": {"webgl": "webgl2"},
                   "stats": {"revision": "170", "scenes_observed": 1, "renders": 50,
                             "scene": {"voxels": 401, "box_meshes": 1, "box_instances": 400,
                                       "distinct_colors": 7, "bbox_voxels": {"size": [20, 2, 20]}}},
                   "camera_input": {"camera": True, "orbit": True}, "wall_s": 40}
        with open(os.path.join(d, "browser.json"), "w") as f:
            json.dump(browser, f)
        out = C.analyse(d, gate={"allowed_hosts": ["cdn.jsdelivr.net:443"], "allowed": 3, "denied": 0})
        sm = out["summary"]
        check(sm["voxels"] == 401 and sm["distinct_colors"] == 7 and sm["bbox_voxels"] == [20, 2, 20]
              and sm["page_errors"] == 1 and sm["console_errors"] == 1 and sm["nonblank"],
              "the summary carries the scene stats, errors and non-blank", json.dumps(sm)[:400])
        check(sm["needs_network"] and sm["cdn_hosts"] == ["cdn.jsdelivr.net"]
              and sm["blocked_hosts"] == ["tracker.example.com"] and sm["failed_requests"] == 1
              and sm["gate"]["allowed_hosts"] == ["cdn.jsdelivr.net:443"],
              "the summary says which CDN hosts loaded, what was blocked, what the gate let through")
        check(sm["orbit"] is True and sm["orbit_by"] == "camera" and sm["software_rendered"],
              "orbit from the three.js camera when there is one; fps labelled software")
        check(os.path.isfile(os.path.join(d, "check.json")), "check.json is written")
        e = os.path.join(d, "empty")
        os.makedirs(e)
        out = C.analyse(e)
        check(out["summary"] is None and not out["browser_json"],
              "no browser.json: no summary (check_page records the error)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fake_run(d: str, arms=("xhigh", "xhigh-off")) -> None:
    os.makedirs(d, exist_ok=True)
    man = {"prompt": VERBATIM, "arms": {a: {"effort": "xhigh", "model": "yamadori",
                                             "features": None} for a in arms}}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    for a in arms:
        for rep in (1, 2):
            rdir = f"reps/{a}__r{rep}__a1"
            os.makedirs(os.path.join(d, rdir, "check"), exist_ok=True)
            with open(os.path.join(d, rdir, "page.html"), "w") as f:
                f.write(PAGE)
            for n in ("t05", "t10", "t15", "orbit_right"):
                _png(os.path.join(d, rdir, "check", n + ".png"),
                     squares=[(10 * rep, 10, 40, (200, 50, 50))])
            row = {"arm": a, "rep": rep, "attempt": 1, "status": "ok", "finish_reason": "stop",
                   "wall_s": 1234.5, "usage": {"prompt_tokens": 120, "completion_tokens": 9000},
                   "x": {"tier": "xhigh", "route": "code_generation"}, "dir": rdir,
                   "page": rdir + "/page.html", "extract": {"method": "fenced", "chars": 80}}
            R.append_row(os.path.join(d, "summary.jsonl"), row)
            R.append_row(os.path.join(d, "checks.jsonl"),
                         {"arm": a, "rep": rep, "attempt": 1, "voxels": 401 * rep, "orbit": True,
                          "nonblank": True, "distinct_colors": 7,
                          "first_errors": ["<script>alert(1)</script>"]})


def test_compare():
    d = tempfile.mkdtemp(prefix="vx-cmp-")
    try:
        run_dir = os.path.join(d, "c1")
        _fake_run(run_dir)
        out = CMP.build(run_dir, "c1")
        page = open(out, encoding="utf-8").read()
        check("xhigh-off" in page and "completion 9000" in page and "1234.5 s" in page
              and "reps/xhigh__r1__a1/check/t15.png" in page and "orbit_right.png" in page,
              "index.html: arms, tokens, wall time, screenshots (orbit shots when orbit)")
        check("<script>alert(1)</script>" not in page and "&lt;script&gt;" in page,
              "page text from the model's run is escaped")
        check(VERBATIM in page or CMP.esc(VERBATIM) in page, "the prompt is shown verbatim")
        bout, key = CMP.build_blind(run_dir, "c1", seed=7)
        bp = open(bout, encoding="utf-8").read()
        k = json.load(open(key))
        check("xhigh" not in bp and "9000" not in bp and "1234.5" not in bp and "reps/" not in bp,
              "blind page: no arm name, tokens, wall time or rep path", bp[:200])
        check(sorted(k["labels"]) == ["A", "B"] and sorted(k["labels"].values()) == ["xhigh", "xhigh-off"]
              and os.path.isfile(os.path.join(run_dir, "blind", "A-r1", "t15.png"))
              and os.path.isfile(os.path.join(run_dir, "blind", "B-r2", "page.html")),
              "blind: A/B labels, the key in its own file, images and pages copied", json.dumps(k))
        order1 = k["labels"]
        CMP.build_blind(run_dir, "c1", seed=7)
        check(json.load(open(key))["labels"] == order1, "the blind shuffle is reproducible by seed")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reps_to_check():
    d = tempfile.mkdtemp(prefix="vx-todo-")
    try:
        _fake_run(d, arms=("xhigh",))
        R.append_row(os.path.join(d, "summary.jsonl"),
                     {"arm": "xhigh", "rep": 3, "attempt": 1, "status": "not_run", "dir": "reps/x"})
        todo = C.reps_to_check(d)
        check([r["rep"] for r in todo] == [1, 2], "check: reps with a page, not the not_run one")
        with open(os.path.join(d, "reps/xhigh__r1__a1/check/check.json"), "w") as f:
            f.write("{}")
        check([r["rep"] for r in C.reps_to_check(d)] == [2]
              and len(C.reps_to_check(d, force=True)) == 2, "checked reps are skipped unless --force")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------- task r3f-stack --
# 2026-09-27: the public prompt with only its last sentence replaced by the
# operator's (was 9a4d5113...: our pins, layout, stack rules and convention)
STACK_SHA256 = "21dac552a0b3baab5732553c9b7047f1b5632b28838b272d943161c6382b1b5b"


def _variants():
    import importlib.util
    p = os.path.join(HERE, "..", "octopus", "variants.py")
    spec = importlib.util.spec_from_file_location("test_voxel_variants", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stack_prompt():
    V = _variants()
    p = R.PAGODA_STACK_PROMPT
    check(R.PAGODA_STACK_PROMPT_SHA256 == STACK_SHA256
          and R.task_prompt("r3f-stack") == (p, STACK_SHA256),
          "the r3f-stack prompt's sha256 is pinned", R.PAGODA_STACK_PROMPT_SHA256)
    sentence = "Build it with r3f (react-three-fiber) v10 and Koota and pmndrs math."
    out_dir = "Put the project in a pagoda/ folder in the current directory."
    check(p == VERBATIM.replace(" Create a single HTML file.", " " + sentence + " " + out_dir)
          and VERBATIM.endswith(" Create a single HTML file.") and R.STACK_SENTENCE == sentence
          and R.OUTPUT_DIR_SENTENCE == out_dir,
          "r3f-stack = the public prompt with ONLY its last sentence replaced by the "
          "operator's stack sentence plus the output folder (2026-09-27: Hermes needs "
          "the prompt to say where files go)")
    # The Octopus V4 tech line is now "build a space shooter game with r3f
    # (react-three-fiber) v10 and Koota and pmndrs math. multi-file project
    # structure." (variants.V4_TECH, 2026-09-27): the same stack words.
    stack = "with r3f (react-three-fiber) v10 and Koota and pmndrs math."
    check(sentence.endswith(stack) and stack in V.V4_TECH,
          "the same stack words as the Octopus V4 prompt", V.V4_TECH)
    check(not any(w in p for w in ("PINNED", "File:", "REQUIRED", "npm", "WebGPU", "TSL",
                                   "createWorld", "tsconfig", "\n")),
          "nothing of ours: no pins, layout, rules, build gate or delivery convention")
    check(R.TASKS["single-html"] is R.PROMPT and R.DEFAULT_TASK == "single-html"
          and R.PROMPT_SHA256 == R.task_prompt("single-html")[1],
          "single-html is unchanged and the default")
    a = R.parse_arm("xhigh")
    check(R.request_body(a)["messages"][0]["content"] == VERBATIM
          and R.request_body(a, None, "r3f-stack")["messages"][0]["content"] == p
          and set(R.request_body(a, None, "r3f-stack")) == {"model", "reasoning_effort", "messages"},
          "the body carries the task's prompt as the one user message, nothing else")
    try:
        R.task_prompt("nope")
        check(False, "an unknown task is refused")
    except ValueError:
        check(True, "an unknown task is refused")


FILE_PKG = '{\n  "name": "pagoda"\n}'
FILE_IDX = '<!doctype html>\n<div id="root"></div>\n<script type="module" src="/src/main.tsx"></script>'
README = "# Pagoda\n\n```bash\nnpm install\n```\n\ndone"
PROJECT_ANSWER = "\n".join([
    "Today I was inspired by lantern.", "", "Here is the project.", "",
    "File: package.json", "```json", FILE_PKG, "```", "",
    "File: index.html", "", "```html", FILE_IDX, "```", "",
    "```tsx src/main.tsx", "import App from './App'", "```", "",
    '```tsx title="src/App.tsx"', "export default function App() { return null }", "```", "",
    "### `vite.config.ts`", "```ts", "export default {}", "```", "",
    "**tsconfig.json**", "```json", "{}", "```", "",
    "```ts", "// src/world.ts", "export const x = 1", "```", "",
    "File: README.md", "```markdown", README, "```", "",
    "File: /etc/passwd", "```", "root", "```", "",
    "File: ../escape.ts", "```ts", "x", "```", "",
    "File: C:\\Windows\\x.ts", "```ts", "y", "```", "",
    "File: src/App.tsx", "```tsx", "// the second App (last wins)", "```", "",
    "Then run:", "", "```bash", "npm install", "```", ""])


def test_extract_project():
    e = R.extract_project(PROJECT_ANSWER)
    f = e["files"]
    check(sorted(f) == ["README.md", "index.html", "package.json", "src/App.tsx",
                        "src/main.tsx", "src/world.ts", "tsconfig.json", "vite.config.ts"],
          "every named block becomes a file", json.dumps(sorted(f)))
    check(f["package.json"] == FILE_PKG and f["index.html"] == FILE_IDX,
          "convention lines (`File: <path>`, a blank line allowed before the fence)")
    check(f["src/main.tsx"] == "import App from './App'" and e["counts"]["by_source"].get("info") == 2
          and e["counts"]["by_source"].get("convention") == 4,
          "info-string paths (```tsx src/main.tsx, title=\"src/App.tsx\")", json.dumps(e["counts"]))
    check(f["vite.config.ts"] == "export default {}" and f["tsconfig.json"] == "{}"
          and e["counts"]["by_source"].get("heading") == 2,
          "heading / bold / backticked paths on the line before the fence")
    check(f["src/world.ts"].startswith("// src/world.ts") and e["counts"]["by_source"].get("comment") == 1,
          "a first-line comment naming the path (the body is kept as written)")
    check(f["README.md"] == README, "a README's own fenced blocks stay inside it (nesting)",
          repr(f["README.md"]))
    check(f["src/App.tsx"] == "// the second App (last wins)"
          and any(p["kind"] == "duplicate_path" and p["path"] == "src/App.tsx" and p["blocks"] == 2
                  for p in e["problems"]),
          "a repeated path: the last block wins, recorded (title=\"...\" was the first)")
    refused = {p["path"]: p["why"] for p in e["problems"] if p["kind"] == "refused_path"}
    check(refused == {"/etc/passwd": "absolute path", "../escape.ts": "`..` in path",
                      "C:\\Windows\\x.ts": "drive path"} and e["counts"]["refused"] == 3,
          "absolute, `..` and drive paths are refused and recorded, never written", json.dumps(refused))
    check(e["counts"]["unnamed"] == 1 and e["method"] == "mixed" and e["layout"] == "npm"
          and e["counts"]["convention_exact"] == 7 and e["project_root"] == "",
          "the unnamed bash block is counted; a package.json at the root: layout npm",
          json.dumps({k: e[k] for k in ("method", "counts", "layout")}))
    cut = "File: a.ts\n```ts\nconst a = 1\n\nFile: b.ts\n```ts\nconst b = 2\n```\n"
    e = R.extract_project(cut)
    check(e["files"] == {"a.ts": "const a = 1", "b.ts": "const b = 2"}
          and [p["path"] for p in e["problems"] if p["kind"] == "unclosed_fence"] == ["a.ts"],
          "a block left open before the next `File:` line ends there, flagged unclosed",
          json.dumps(e["files"]))
    e = R.extract_project("File: src/App.tsx\n```tsx\nexport const A = 1;")
    check(e["files"] == {"src/App.tsx": "export const A = 1;"} and e["counts"]["unclosed"] == 1
          and e["layout"] == "none",
          "a reply cut mid-file: extracted as written, unclosed, no package.json or index.html")
    nested = "\n".join(["File: pagoda/package.json", "```json", "{}", "```",
                        "File: pagoda/src/App.tsx", "```tsx", "x", "```"])
    e = R.extract_project(nested)
    check(e["project_root"] == "pagoda" and e["layout"] == "npm" and sorted(e["files"]) == [
              "pagoda/package.json", "pagoda/src/App.tsx"],
          "a project under its own folder: the root is found, nothing is moved")
    e = R.extract_project("```tsx:src/a.tsx\na\n```\n```src/b.ts\nb\n```\n```package.json\n{}\n```")
    check(sorted(e["files"]) == ["package.json", "src/a.tsx", "src/b.ts"] and e["method"] == "info",
          "info forms: ```lang:path, ```path", json.dumps(sorted(e["files"])))
    e = R.extract_project("### Three pieces\n```ts\nconst a = 1\n```\n1. Run it\n```bash\nnpm i\n```")
    check(e["files"] == {} and e["method"] == "none" and e["counts"]["unnamed"] == 2,
          "prose headings are not paths; no named block: method none")
    # no layout was asked for: a reply that is one page (an import map, CDN
    # modules) is a project of one file, index.html
    page = ("Here it is.\n\n```html\n<!doctype html>\n<script type=\"importmap\">{}</script>\n"
            "<div id=\"root\"></div>\n</html>\n```\n\n```bash\nnpx serve .\n```\n")
    e = R.extract_project(page)
    check(sorted(e["files"]) == ["index.html"] and e["counts"]["by_source"] == {"page": 1}
          and e["layout"] == "static" and e["counts"]["unnamed"] == 1
          and e["files"]["index.html"].startswith("<!doctype html>"),
          "an unnamed HTML page becomes index.html (source `page`), layout static",
          json.dumps({k: e[k] for k in ("counts", "layout")}))
    e = R.extract_project("File: index.html\n```html\n<!doctype html>\n```\n"
                          "```html\n<!doctype html><p>other</p>\n```\n")
    check(e["files"] == {"index.html": "<!doctype html>"} and "page" not in e["counts"]["by_source"],
          "a named index.html is never replaced by an unnamed page")
    check(R.normalise_path("./src//App.tsx") == ("src/App.tsx", None)
          and R.normalise_path("src\\App.tsx") == ("src/App.tsx", None)
          and R.normalise_path("src/../../x")[1] == "`..` in path"
          and R.normalise_path("~/x.ts")[1] == "absolute path"
          and R.normalise_path("\\\\server\\x.ts")[1] == "absolute path"
          and R.normalise_path("src/a?.ts")[1] is not None,
          "path normalisation: ./ and \\ normalised; .., ~, UNC and bad characters refused")


def test_run_rep_project():
    d = tempfile.mkdtemp(prefix="vx-proj-")
    try:
        arm = R.parse_arm("xhigh")
        post = poster_seq((200, fake_response(PROJECT_ANSWER)))
        row = R.run_rep(cfg(d, task="r3f-stack"), arm, 1, 1, poster=post, sleep=lambda s: None)
        proj = os.path.join(d, row.get("project") or "missing")
        check(row["status"] == "ok" and row["task"] == "r3f-stack"
              and row["prompt_sha256"] == STACK_SHA256 and not row.get("page")
              and row["project"].endswith("/project"),
              "an r3f-stack rep: task and the prompt sha actually sent on the row, project/ not page",
              json.dumps({k: row.get(k) for k in ("status", "task", "project", "page")}))
        check(post.calls[0]["body"]["messages"][0]["content"] == R.PAGODA_STACK_PROMPT,
              "what went out is the r3f-stack prompt")
        check(open(os.path.join(proj, "package.json"), encoding="utf-8").read() == FILE_PKG
              and open(os.path.join(proj, "README.md"), encoding="utf-8", newline="").read() == README
              and not os.path.exists(os.path.join(d, "escape.ts"))
              and not os.path.exists(os.path.join(os.path.dirname(proj), "escape.ts")),
              "files written under reps/<rep>/project/ exactly as extracted; refused paths nowhere")
        ex = json.load(open(os.path.join(d, row["dir"], "extract.json"), encoding="utf-8"))
        check(ex["files_written"] == 8 and "files" not in ex and ex["counts"]["refused"] == 3
              and row["extract"]["layout"] == "npm",
              "extract.json and the row carry counts, problems, missing -- not the file bodies")
        row = R.run_rep(cfg(d), arm, 2, 1, poster=poster_seq((200, fake_response(ANSWER))))
        check(row["task"] == "single-html" and row["prompt_sha256"] == R.PROMPT_SHA256
              and row.get("page") and not row.get("project"),
              "the default task is still single-html, with its own sha on the row")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_task_tags():
    d = tempfile.mkdtemp(prefix="vx-task-")
    old = (R.RESULTS, R._octo, R.models_listed, R.post, R.preflight)
    kf = os.path.join(d, "key.txt")
    with open(kf, "w") as f:
        f.write(KEY + "\n")
    orig_pf = R.preflight
    try:
        R.RESULTS = d
        R.preflight = lambda url, key, arms: orig_pf(url, key, arms, sleep=lambda s: None)
        sent = []
        R.post = lambda url, key, body, arm, timeout: (sent.append(body) or
                                                       (200, fake_response(PROJECT_ANSWER)))
        R.models_listed = lambda url, key: ("200", ["yamadori"])
        R._octo = FakeOcto()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = R.main(["--key-file", kf, "--tag", "s1", "--task", "r3f-stack", "--reps", "2"])
        man = json.load(open(os.path.join(d, "s1", "manifest.json"), encoding="utf-8"))
        rows = R.read_rows(os.path.join(d, "s1", "summary.jsonl"))
        check(rc == 0 and len(sent) == 2 and man["task"] == "r3f-stack"
              and man["prompt"] == R.PAGODA_STACK_PROMPT and man["prompt_sha256"] == STACK_SHA256
              and man["stack_sentence"] == R.STACK_SENTENCE
              and man["replaces"] == "Create a single HTML file."
              and "versions" not in man and "required_files" not in man
              and all(r["task"] == "r3f-stack" and r["prompt_sha256"] == STACK_SHA256 for r in rows),
              "--task r3f-stack: the manifest and every row record the task and the sha sent",
              buf.getvalue()[-300:])
        with redirect_stdout(buf):
            rc = R.main(["--key-file", kf, "--tag", "s1"])
        check(rc == 2 and len(sent) == 2 and "r3f-stack" in buf.getvalue(),
              "a rerun of an r3f-stack tag with the default task: refused, nothing sent")
        with redirect_stdout(buf):
            R.main(["--key-file", kf, "--tag", "h1"])
            rc = R.main(["--key-file", kf, "--tag", "h1", "--task", "r3f-stack"])
        check(rc == 2 and len(sent) == 5, "a single-html tag rerun with --task r3f-stack: refused")
        os.makedirs(os.path.join(d, "legacy"))
        json.dump({"prompt": VERBATIM, "prompt_sha256": R.PROMPT_SHA256},
                  open(os.path.join(d, "legacy", "manifest.json"), "w"))
        check(R.tag_conflict(os.path.join(d, "legacy"), "single-html") is None
              and R.tag_conflict(os.path.join(d, "legacy"), "r3f-stack"),
              "a manifest with no task predates the option: single-html")
        os.makedirs(os.path.join(d, "edited"))
        json.dump({"task": "r3f-stack", "prompt_sha256": "0" * 64},
                  open(os.path.join(d, "edited", "manifest.json"), "w"))
        check("prompt changed" in (R.tag_conflict(os.path.join(d, "edited"), "r3f-stack") or ""),
              "a tag whose prompt sha differs from today's: refused (the prompt changed)")
        os.makedirs(os.path.join(d, "rows"))
        R.append_row(os.path.join(d, "rows", "summary.jsonl"), {"task": "single-html", "rep": 1})
        check(R.tag_conflict(os.path.join(d, "rows"), "r3f-stack")
              and R.tag_conflict(os.path.join(d, "rows"), "single-html") is None,
              "rows of another task (no manifest): refused")
        check(KEY not in buf.getvalue() and KEY not in all_text(os.path.join(d, "s1")),
              "the key is never printed or written (r3f-stack)")
    finally:
        R.RESULTS, R._octo, R.models_listed, R.post, R.preflight = old
        shutil.rmtree(d, ignore_errors=True)


def _write(root: str, files: dict) -> None:
    for p, t in files.items():
        fp = os.path.join(root, *p.split("/"))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(t)


def _v4_pkg(**over) -> str:
    pkg = {"name": "pagoda", "private": True, "type": "module",
           "scripts": {"dev": "vite", "build": "vite build"},
           "dependencies": {"@react-three/fiber": "10.0.0-alpha.5", "react": "19.2.8",
                            "react-dom": "19.2.8", "three": "0.185.1", "koota": "0.6.6",
                            "math": "0.1.0"},
           "devDependencies": {"vite": "8.3.1"}}
    for k, v in over.items():
        pkg[k] = v
    return json.dumps(pkg, indent=2)


GOOD_PROJECT = {
    "src/App.jsx": ("import { Canvas } from '@react-three/fiber'\n"
                    "import { WorldProvider } from 'koota/react'\n"
                    "import { world } from './world'\n"
                    "export default function App() {\n"
                    "  return <WorldProvider world={world}><Canvas /></WorldProvider>\n}\n"),
    "src/world.js": ("import { createWorld, trait } from 'koota'\n"
                     "export const Position = trait({ x: 0, y: 0, z: 0 })\n"
                     "export const world = createWorld()\n"
                     "export function drift(w) {\n"
                     "  w.query(Position).updateEach(([p]) => { p.y += 0.01 })\n}\n"),
    "src/rand.js": ("import { mulberry32 } from 'math/random'\n"
                    "export const rng = mulberry32.create(1)\n"),
}
STACK_CHECKS = {"r3f_v10_canvas", "koota_world_traits_queries", "math_used"}


def test_stack_static():
    d = tempfile.mkdtemp(prefix="vx-static-")
    try:
        _write(d, {**GOOD_PROJECT, "package.json": _v4_pkg()})
        st = C.stack_static(d)
        res = {c["check"]: c["pass"] for c in st["checks"]}
        check(res == {k: True for k in STACK_CHECKS} and st["passed"] == 3
              and st["parse_errors"] == 0,
              "a small project on the named stack passes the Octopus grader's V4 checks "
              "(r3f 10 + Canvas, koota, math imported and used) -- no pins, no TSL, no WebGPU",
              json.dumps(st)[:900])
        check(sys.modules["run"] is R, "loading grade.py leaves `run` as this directory's run.py")
        st = C.stack_static(d, {"npm_ls": {"@react-three/fiber": "9.8.1"}})
        r3f = {c["check"]: c for c in st["checks"]}["r3f_v10_canvas"]
        check(r3f["pass"] is False and r3f["evidence"]["installed"] == "9.8.1",
              "the version npm installed wins over package.json (v9 installed: fails)",
              json.dumps(r3f["evidence"]))
        bad = tempfile.mkdtemp(prefix="vx-static-bad-")
        try:
            _write(bad, {"src/App.jsx": "import { Canvas } from '@react-three/fiber'\n"
                                        "export const A = () => <Canvas />\n",
                         "src/scene.js": "import * as THREE from 'three'\n"
                                         "import { vec3 } from 'math'\n"
                                         "export const r = Math.random()\n",
                         "package.json": _v4_pkg(dependencies={"@react-three/fiber": "^9.8.1",
                                                               "math": "0.1.0"})})
            st = C.stack_static(bad)
            res = {c["check"]: c for c in st["checks"]}
            check(not any(c["pass"] for c in st["checks"]) and st["passed"] == 0,
                  "r3f v9, no koota, math imported but unused: every stack check fails",
                  json.dumps({k: v["pass"] for k, v in res.items()}))
            check(res["math_used"]["evidence"]["names_used"] == {"vec3": 0}
                  and res["math_used"]["evidence"]["math_random_calls"] == 1
                  and res["r3f_v10_canvas"]["evidence"]["major"] == 9,
                  "the evidence names the unused import, Math.random and the major",
                  json.dumps({k: v["evidence"] for k, v in res.items()})[:600])
        finally:
            shutil.rmtree(bad, ignore_errors=True)
        sm = C.stack_summary(C.stack_static(d))
        check(sm["passed"] == 3 and sm["checks"]["r3f_v10_canvas"] is True
              and sm["r3f"]["declared"] == "10.0.0-alpha.5" and "pins_extra" not in sm,
              "the row's stack summary", json.dumps(sm))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fake_build(outcome: str, error: str | None = None):
    calls = []

    def builder(proj, out, label, timeout):
        calls.append(out)
        shutil.rmtree(out, ignore_errors=True)            # as build_project does
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "stage"), "w") as f:
            f.write("" if error else outcome)
        for step, rc in (("npm_install", 0), ("tsc", 0 if outcome == "built" else 2),
                         ("build", 0 if outcome == "built" else 2)):
            with open(os.path.join(out, f"{step}.rc"), "w") as f:
                f.write(str(rc))
            with open(os.path.join(out, f"{step}.log"), "w") as f:
                f.write("" if rc == 0 else "src/App.tsx(3,7): error TS2322: Type 'x' is not ...\n"
                        "src/world.ts(1,1): error TS2307: Cannot find module 'koota'.\n")
        if outcome == "built" and not error:
            _write(out, {"dist/index.html": "<!doctype html>", "dist/assets/index-abc.js": "1"})
        rec = C.read_build(out)
        if error:
            rec["error"] = error
        return rec
    builder.calls = calls
    return builder


def test_check_project():
    d = tempfile.mkdtemp(prefix="vx-cp-")
    try:
        proj = os.path.join(d, "rep", "project")
        _write(proj, {**GOOD_PROJECT, "package.json": _v4_pkg()})
        rep = os.path.join(d, "rep")
        seen = []

        def pager(page, shots, label, timeout, seed):
            seen.append(page)
            os.makedirs(shots, exist_ok=True)
            return {"label": label, "browser_rc": 0,
                    "analysis": {"check_version": 1, "shots": {}, "pixels": {},
                                 "summary": {"nonblank": True, "voxels": 900},
                                 "browser_json": True}}
        rec = C.check_project(proj, rep, "t-r1", builder=_fake_build("built"), pager=pager)
        cj = json.load(open(os.path.join(rep, "check", "check.json"), encoding="utf-8"))
        check(not rec.get("error") and rec["build"]["outcome"] == "built"
              and seen == [os.path.join(rep, "build", "dist")]
              and cj["summary"]["voxels"] == 900 and cj["task"] == "r3f-stack"
              and cj["build"]["dist_files"] == 2 and cj["static"]["passed"] == 3,
              "built: page_check runs on the built dist/; check.json has build and static",
              json.dumps(rec.get("build"))[:400])
        seen.clear()
        rec = C.check_project(proj, rep, "t-r1", builder=_fake_build("failed"), pager=pager)
        cj = json.load(open(os.path.join(rep, "check", "check.json"), encoding="utf-8"))
        b = C.build_summary(rec["build"])
        check(not rec.get("error") and not seen and rec["build"]["outcome"] == "failed"
              and cj["summary"] is None and b["tsc_rc"] == 2 and b["tsc_error_ts"] == 2
              and b["first_ts_errors"][0].startswith("src/App.tsx(3,7): error TS2322")
              and "failed" in rec["page"]["why"],
              "a failed build is a MEASUREMENT (no error), no page check, TS errors counted",
              json.dumps(b)[:400])
        rec = C.check_project(proj, rep, "t-r1", builder=_fake_build("built", error="sandbox network: x"),
                              pager=pager)
        check(rec.get("error", "").startswith("build: sandbox network")
              and not os.path.isfile(os.path.join(rep, "check", "check.json")),
              "a build that could not RUN is an error, and stays unchecked (a rerun retries)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reps_to_check_project():
    d = tempfile.mkdtemp(prefix="vx-todo-p-")
    try:
        R.append_row(os.path.join(d, "summary.jsonl"),
                     {"task": "r3f-stack", "arm": "xhigh", "rep": 1, "attempt": 1, "status": "ok",
                      "dir": "reps/xhigh__r1__a1", "project": "reps/xhigh__r1__a1/project"})
        R.append_row(os.path.join(d, "summary.jsonl"),
                     {"task": "r3f-stack", "arm": "xhigh", "rep": 2, "attempt": 1, "status": "ok",
                      "dir": "reps/xhigh__r2__a1", "extract": {"files_written": 0}})
        check([r["rep"] for r in C.reps_to_check(d)] == [1],
              "check: an r3f-stack rep with a project is checked; one with no files is not")
        row = {"status": "ok", "finish_reason": "stop",
               "extract": {"files_written": 8, "method": "convention", "chars": 900,
                           "layout": "npm", "counts": {"refused": 1}}}
        chk = {"build": {"outcome": "failed", "tsc_rc": 2, "tsc_error_ts": 3,
                         "first_ts_errors": ["src/App.tsx(1,1): error TS2307: <x>"]},
               "stack": {"passed": 2, "total": 3, "checks": {"r3f_v10_canvas": True,
                                                             "math_used": False}},
               "page_checked": False}
        pairs = dict(CMP.stat_pairs(chk, row, True))
        check("8 files" in pairs["project"] and "layout npm" in pairs["project"]
              and pairs["build"].startswith("failed") and "2/3" in pairs["stack"]
              and pairs["render"].startswith("not run") and "voxels" not in pairs,
              "the compare page shows the project, build and stack for an r3f-stack rep",
              json.dumps(pairs)[:500])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_read_build_and_script():
    d = tempfile.mkdtemp(prefix="vx-rb-")
    try:
        b = C.read_build(d)
        check(b["outcome"] is None and b["npm_install"]["rc"] is None and not b["dist"],
              "an empty build dir: no verdict (check_project makes that an error)")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    sh = os.path.join(HERE, "..", "octopus", "serve_app.sh")
    text = open(sh, encoding="utf-8", newline="").read()
    check("\r" not in text and 'MODE=${2:-serve}' in text and 'if [ "$MODE" = build ]' in text
          and 'cp -r dist "$OUT/dist"' in text and "exec python3 -m http.server 3001" in text,
          "serve_app.sh: build-only mode added, LF endings, the serving path kept")
    check('if [ "$1" = ts ] || [ "$1" = npm ]; then' in text
          and 'if [ "$1" = ts ] || [ -f tsconfig.json ]; then' in text
          and 'echo skip > "$OUT/tsc.rc"' in text and 'cp -r . "$OUT/dist"' in text,
          "serve_app.sh: `npm` mode (tsc only with a tsconfig.json) and `js build` (the "
          "project is the page) for a layout the model chose")
    bash = shutil.which("bash")
    if bash:
        r = subprocess.run([bash, "-n", sh], capture_output=True, text=True)
        check(r.returncode == 0, "serve_app.sh parses (bash -n)", r.stderr[-300:])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
        except Exception:                                        # noqa: BLE001
            check(False, f"{t.__name__} raised", traceback.format_exc()[-800:])
    passed = sum(1 for ok, _n, _d in _results if ok)
    for ok, name, detail in _results:
        if not ok:
            print(f"  FAIL: {name}  [{detail}]")
    print(f"{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
