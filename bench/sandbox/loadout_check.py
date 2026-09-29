#!/usr/bin/env python
"""Check each harness's DEFAULT LOADOUT in the harness box, with NO MODEL
(docs/HARNESS-SANDBOX.md "Default loadout"; docs/HARNESSES.md).

    python bench/sandbox/loadout_check.py all --key-file K [--port 18299]
    python bench/sandbox/loadout_check.py opencode --key-file K

A scripted stand-in (scripted_model.py, 127.0.0.1:<port>, the box's one
forward) plays the model: it makes the tool calls a model would make to use
the loadout, in order, and the harness executes them in the box for real.
What is checked, per harness:

  - the tool list the harness offers (names and chars: its token cost);
  - the model's shell serves a page on localhost:5173 (inside the box);
  - the browser opens that page and reads its console error and its
    uncaught exception;
  - the browser cannot reach the host (host.docker.internal:11434,
    192.168.65.254:1234) and does reach a public site through the gate;
  - type errors come back: OpenCode's `write` result (its lsp), and
    `tsc`/`pyright` through the shell for Pi and Codex;
  - the shell cannot reach the host either (#48 probe, curl);
  - an installed package's API comes back without reading node_modules:
    OpenCode's `lsp` hover and goToDefinition on koota's createWorld, Pi's
    `package-api` skill script (--package-dir; see PACKAGE_STEPS), and Pi
    lists every loadout skill in its system prompt.

The key is a dummy (the stand-in ignores it). Every request body the
harness sent is kept under --out, and one row per harness is appended to
bench/sandbox/results/loadout_check.jsonl.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness_box as hb  # noqa: E402

RESULTS = os.path.join(HERE, "results", "loadout_check.jsonl")
PAGE = ("<!doctype html><html><head><title>BoxPage</title></head><body>"
        "<h1>box page</h1><script>console.error('BOX_CONSOLE_ERROR');"
        "setTimeout(function(){ undefinedFn(); }, 10);</script></body></html>")
BAD_TS = 'export const n: number = "str";\n'
BAD_PY = 'def g(x: int) -> int:\n    return x\n\n\ng("s")\n'
LOCAL = "http://localhost:5173/"
HOST_URLS = ("http://host.docker.internal:11434/", "http://192.168.65.254:1234/health")
PUBLIC = "https://example.com/"


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


FIXTURES = (f"mkdir -p /work/site /work/src && echo {_b64(PAGE)} | base64 -d > /work/site/index.html"
            f" && echo {_b64(BAD_TS)} | base64 -d > /work/src/bad.ts"
            f" && echo {_b64(BAD_PY)} | base64 -d > /work/bad.py && echo FIXTURES_OK")
SERVE_BG = ("(setsid python3 -m http.server 5173 --bind 127.0.0.1 --directory /work/site"
            " > /tmp/srv.log 2>&1 < /dev/null &) ; sleep 1")
SERVE_FG = "python3 -m http.server 5173 --bind 127.0.0.1 --directory /work/site"
PROBE = ("curl -s -m 3 -o /dev/null -w 'SERVER_%{http_code}\\n' --noproxy '*' http://127.0.0.1:5173/; "
         "curl -s -m 5 -o /dev/null -w 'PROXIED_HOST_%{http_code}\\n' http://host.docker.internal:11434/; "
         "curl -s -m 5 -o /dev/null -w 'DIRECT_HOST_%{http_code}\\n' --noproxy '*' http://192.168.65.254:11434/; "
         "echo KEYVARS=$(env | grep -c YAMADORI_)")
TYPES = ("tsc --noEmit --strict /work/src/bad.ts; echo TSC_RC=$?; "
         "pyright /work/bad.py; echo PYRIGHT_RC=$?")
AB = "agent-browser --cdp 9322"

# THE PACKAGE-API STEPS (2026-09-29): can the model learn an installed
# package's API without reading node_modules? OpenCode through its `lsp` tool
# (hover + goToDefinition on koota's createWorld), Pi through the skill
# `package-api`'s script. The package is COPIED into the project from an
# install already on this machine (--package-dir, default: the highest-version
# koota in a pagoda run's node_modules; nothing is downloaded); without one, these
# steps are left out and the row says so.
API_USE = "import { createWorld } from 'koota';\nexport const world = createWorld();\n"
API_TSCONFIG = {"compilerOptions": {"target": "ES2022", "module": "ESNext",
                                    "moduleResolution": "bundler", "strict": True,
                                    "skipLibCheck": True, "noEmit": True}}
API_CMD = "node ~/.agents/skills/package-api/api.cjs"
API_PI = (f"cd /work && {API_CMD} koota createWorld World.query; echo API_RC=$?;"
          f" {API_CMD} koota | tail -1; {API_CMD} koota nope; echo MISS_RC=$?")
PACKAGE_STEPS = {
    "opencode": [{"tool": ["lsp"], "args": {"operation": "hover", "filePath": "/work/src/api_use.ts",
                                            "line": 1, "character": 12}},
                 {"tool": ["lsp"], "args": {"operation": "goToDefinition",
                                            "filePath": "/work/src/api_use.ts",
                                            "line": 1, "character": 12}}],
    "pi": [{"tool": ["bash"], "args": {"command": API_PI}}],
}


def default_package_dir() -> str | None:
    """An installed koota already on this machine (a pagoda run's): the highest
    version (a run may have installed an old one: pagoda-p4 has 0.1.12)."""
    import glob

    def version(pj: str) -> tuple:
        try:
            with open(pj, encoding="utf-8") as f:
                v = json.load(f).get("version") or ""
            return tuple(int(x) for x in re.findall(r"\d+", v.split("-")[0])[:3])
        except (OSError, ValueError):
            return ()
    hits = glob.glob(os.path.join(os.path.expanduser("~"), "octo", "runs", "*", "pagoda",
                                  "node_modules", "koota", "package.json"))
    hits.sort(key=lambda pj: (version(pj), os.path.getmtime(pj)), reverse=True)
    return os.path.dirname(hits[0]) if hits else None


def seed_package(proj: str, package_dir: str) -> dict:
    """Copy the package and a file that uses it into the project (mounted at /work)."""
    import shutil
    shutil.copytree(package_dir, os.path.join(proj, "node_modules", "koota"))
    os.makedirs(os.path.join(proj, "src"), exist_ok=True)
    with open(os.path.join(proj, "src", "api_use.ts"), "w", encoding="utf-8", newline="\n") as f:
        f.write(API_USE)
    with open(os.path.join(proj, "tsconfig.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(API_TSCONFIG, f, indent=1)
    with open(os.path.join(package_dir, "package.json"), encoding="utf-8") as f:
        version = json.load(f).get("version")
    return {"package": "koota", "version": version, "from": package_dir}


def script(harness: str, package: bool = False) -> list[dict]:
    """The stand-in's tool calls, in order (scripted_model.py); `package` adds
    the package-API steps at the end."""
    return _script(harness) + (PACKAGE_STEPS.get(harness, []) if package else [])


def _script(harness: str) -> list[dict]:
    if harness == "opencode":
        sh = lambda c: {"tool": ["bash"], "args": {"command": c, "description": "check"}}  # noqa: E731
        nav = lambda u: {"tool": ["playwright_browser_navigate"], "args": {"url": u}}  # noqa: E731
        return [sh(f"{FIXTURES} && {SERVE_BG} && {PROBE}"),
                nav(LOCAL),
                {"tool": ["playwright_browser_console_messages"], "args": {"level": "error"}},
                nav(HOST_URLS[0]), nav(HOST_URLS[1]), nav(PUBLIC),
                {"tool": ["write"], "args": {"filePath": "/work/src/bad2.ts", "content": BAD_TS}},
                {"tool": ["write"], "args": {"filePath": "/work/bad2.py", "content": BAD_PY}},
                {"tool": ["lsp"], "args": {"operation": "hover", "filePath": "/work/src/bad2.ts",
                                           "line": 1, "character": 14}}]
    if harness == "pi":
        sh = lambda c: {"tool": ["bash"], "args": {"command": c}}  # noqa: E731
        return [sh(f"{FIXTURES} && {SERVE_BG} && {PROBE}"),
                sh(f"{AB} open {LOCAL} && sleep 1 && {AB} get title && echo ERRORS: && {AB} errors --json"
                   f" && echo CONSOLE: && {AB} console"),
                sh(f"{AB} open {HOST_URLS[0]}; {AB} get text body; {AB} open {HOST_URLS[1]};"
                   f" {AB} get text body; {AB} open {PUBLIC} && {AB} get title"),
                sh(TYPES),
                {"tool": ["find"], "args": {"pattern": "*.ts"}},
                {"tool": ["grep"], "args": {"pattern": "BOX_CONSOLE_ERROR"}},
                # looking at the page (skill page-check): a screenshot, then `read`
                # returns it as an image part (models.json input has "image")
                sh(f"{AB} open {LOCAL} && {AB} screenshot /tmp/page.png && ls -l /tmp/page.png"),
                {"tool": ["read"], "args": {"path": "/tmp/page.png"}}]
    if harness == "codex":
        ex = lambda c, y=10000: {"tool": ["exec_command"], "args": {"cmd": c, "yield_time_ms": y}}  # noqa: E731
        ns = "mcp__playwright::"
        nav = lambda u: {"tool": [ns + "browser_navigate"], "args": {"url": u}}  # noqa: E731
        return [ex(FIXTURES),
                ex(SERVE_FG, 2000),                    # left running: a unified-exec session
                ex(PROBE),
                nav(LOCAL),
                {"tool": [ns + "browser_console_messages"], "args": {"level": "error"}},
                nav(HOST_URLS[0]), nav(HOST_URLS[1]), nav(PUBLIC),
                ex(TYPES, 60000)]
    raise SystemExit(f"no script for {harness}")


ARGS = {"opencode": ["run", "--format", "json", "--", "check the page"],
        "pi": ["--mode", "json", "-p", "check the page"],
        "codex": ["exec", "--json", "--skip-git-repo-check", "-C", "/work", "check the page"]}


def _tool_results(req: dict) -> list[str]:
    """The tool results in the LAST request, in order, as text."""
    out = []
    if "messages" in req:
        for m in req["messages"]:
            if m.get("role") == "tool":
                c = m.get("content")
                out.append(c if isinstance(c, str) else json.dumps(c))
            elif m.get("role") == "user" and out and isinstance(m.get("content"), list):
                # OpenCode moves tool-result media into a synthetic user message
                out[-1] += " " + json.dumps(m["content"])[:200]
    else:
        for it in req.get("input") or []:
            if isinstance(it, dict) and it.get("type") in ("function_call_output",
                                                           "custom_tool_call_output"):
                o = it.get("output")
                out.append(o if isinstance(o, str) else json.dumps(o))
    return out


def _tools(req: dict) -> list[tuple[str, int]]:
    out = []
    for t in req.get("tools") or []:
        if t.get("type") == "namespace":
            out += [(f"{t['name']}::{u['name']}", len(json.dumps(u))) for u in t.get("tools") or []]
        else:
            name = (t.get("function") or {}).get("name") or t.get("name")
            out.append((name, len(json.dumps(t))))
    return out


def judge_package(harness: str, results: list[str]) -> dict:
    """The package-API steps' results (after the loadout's own steps)."""
    r = results[len(_script(harness)):] + [""] * 3
    if harness == "opencode":
        return {"lsp_hover_package": "createWorld" in r[0] and "World" in r[0]
                and "No LSP server" not in r[0],
                "lsp_definition_package": "node_modules/koota/dist/index.d.ts" in r[1]}
    if harness == "pi":
        return {"package_api_signature": bool(re.search(r"^  createWorld\(.*\): World$", r[0], re.M))
                and "API_RC=0" in r[0],
                "package_api_member": "World.query<T extends QueryParameter[]>" in r[0],
                "package_api_entry_points": "other entry points: koota/react" in r[0],
                "package_api_miss_says_so": "koota exports no nope" in r[0] and "MISS_RC=1" in r[0]}
    return {}


def judge(harness: str, results: list[str]) -> dict:
    r = results + [""] * 10
    allr = "\n".join(results)
    v = {"fixtures_and_server": "FIXTURES_OK" in allr and "SERVER_200" in allr,
         "shell_host_blocked": "PROXIED_HOST_403" in allr and "DIRECT_HOST_000" in allr}
    if harness == "opencode":
        v.update(browser_opened_page="BoxPage" in r[1],
                 browser_console_error="BOX_CONSOLE_ERROR" in r[2],
                 browser_page_exception="undefinedFn" in (r[1] + r[2]),
                 browser_host_blocked=all(re.search(r"ERR_|403|[Ff]orbidden|refused", x)
                                          for x in (r[3], r[4])),
                 browser_public_via_gate="Example Domain" in r[5],
                 lsp_diagnostic_ts="LSP errors detected" in r[6] and "not assignable" in r[6],
                 lsp_diagnostic_py="LSP errors detected" in r[7],
                 lsp_tool_answers=bool(r[8]) and "No LSP server" not in r[8])
    elif harness == "pi":
        v.update(browser_opened_page="BoxPage" in r[1],
                 browser_console_error="BOX_CONSOLE_ERROR" in r[1],
                 browser_page_exception="undefinedFn" in r[1],
                 browser_host_blocked=bool(re.search(r"ERR_|403|[Ff]orbidden|refused", r[2]))
                 and '"ok"' not in r[2] and "Ollama" not in r[2],
                 browser_public_via_gate="Example Domain" in r[2],
                 types_ts="TS2322" in r[3], types_py="reportArgumentType" in r[3],
                 find_tool="bad.ts" in r[4], grep_tool="BOX_CONSOLE_ERROR" in r[5],
                 screenshot_taken="page.png" in r[6] and "No such file" not in r[6],
                 screenshot_read_as_image="image_url" in r[7] or "image/png" in r[7])
    else:
        v.update(browser_opened_page="BoxPage" in r[3],
                 browser_console_error="BOX_CONSOLE_ERROR" in r[4],
                 browser_page_exception="undefinedFn" in (r[3] + r[4]),
                 browser_host_blocked=all(re.search(r"ERR_|403|[Ff]orbidden|refused", x)
                                          for x in (r[5], r[6])),
                 browser_public_via_gate="Example Domain" in r[7],
                 types_ts="TS2322" in r[8], types_py="reportArgumentType" in r[8])
    return v


def check(harness: str, key_file: str, out: str, port: int, timeout: int,
          package_dir: str | None = None) -> dict:
    base = os.path.join(out, harness)
    for sub in ("cap", "proj", "run"):
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", os.path.normpath(os.path.join(base, sub))],
                       capture_output=True)
    os.makedirs(base, exist_ok=True)
    pkg = bool(package_dir) and harness in PACKAGE_STEPS
    seeded = seed_package(os.path.join(base, "proj"), package_dir) if pkg else None
    steps = script(harness, pkg)
    spath = os.path.join(base, "script.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=1)
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"port {port} is taken")
    sm = subprocess.Popen([sys.executable, os.path.join(HERE, "scripted_model.py"), str(port),
                           os.path.join(base, "cap"), spath])
    t0 = time.time()
    try:
        time.sleep(1)
        rc = hb.main(["run", harness, "--key-file", key_file, "--project", os.path.join(base, "proj"),
                      "--run-dir", os.path.join(base, "run"), "--target-port", str(port),
                      "--timeout", str(timeout), "--", *ARGS[harness]])
    finally:
        sm.terminate()
    cap = os.path.join(base, "cap")
    reqs = sorted((f for f in os.listdir(cap) if re.fullmatch(r"req\d+\.json", f)),
                  key=lambda f: int(f[3:-5]))
    bodies = [json.load(open(os.path.join(cap, f), encoding="utf-8")) for f in reqs]
    with_tools = [b for b in bodies if b.get("tools")]
    tools = _tools(with_tools[0]) if with_tools else []
    results = _tool_results(with_tools[-1]) if with_tools else []
    rec = json.loads(open(os.path.join(base, "run", "harness_box.jsonl"), encoding="utf-8")
                     .read().strip().splitlines()[-1])
    row = {"harness": harness, "t": time.time(), "wall_s": round(time.time() - t0, 1), "rc": rc,
           "image": hb.IMAGE, "image_id": rec.get("image_id"),
           "sidecar": {k: (rec.get("browser_sidecar") or {}).get(k)
                       for k in ("image", "image_id", "browser", "ok")},
           "requests": len(bodies), "steps": len(steps), "results": len(results),
           "package": seeded or ("no --package-dir" if harness in PACKAGE_STEPS else None),
           "tools": [n for n, _ in tools], "tool_chars": {n: c for n, c in tools},
           "tools_total_chars": len(json.dumps(with_tools[0]["tools"])) if with_tools else 0,
           "gate": {k: rec.get("stop", {}).get(k) for k in ("allowed", "denied", "forwarded",
                                                             "allowed_hosts", "denied_sample")},
           "verdict": {**judge(harness, results),
                       **(judge_package(harness, results) if pkg else {}),
                       # the loadout skills are listed in Pi's system prompt
                       **({"skills_listed": bool(with_tools) and all(
                           n in json.dumps(with_tools[0].get("messages", [])[:1])
                           for n in hb.loadout_skills("pi"))} if harness == "pi" else {}),
                       # Pi: PI_CACHE_RETENTION=long names the session (prompt_cache_key)
                       **({"prompt_cache_key_sent": bool(with_tools)
                           and all(b.get("prompt_cache_key") for b in with_tools)}
                          if harness == "pi" else {})},
           "result_heads": [x[:300] for x in results]}
    row["ok"] = all(row["verdict"].values()) and len(results) == len(steps)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("harness", choices=[*sorted(hb.HARNESSES), "all"])
    ap.add_argument("--key-file", required=True, help="a DUMMY key (the stand-in ignores it)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("TEMP", "."), "loadout_check"))
    ap.add_argument("--port", type=int, default=18299)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--package-dir", default=None,
                    help="an installed koota to copy into the project for the package-API steps "
                         "(default: the highest-version one in a pagoda run; 'none' leaves the steps out)")
    a = ap.parse_args()
    ok = True
    for h in (sorted(hb.HARNESSES) if a.harness == "all" else [a.harness]):
        pdir = None if a.package_dir == "none" else (a.package_dir or default_package_dir())
        row = check(h, a.key_file, a.out, a.port, a.timeout, pdir)
        ok &= row["ok"]
        print(json.dumps({k: row[k] for k in ("harness", "ok", "rc", "wall_s", "requests", "steps",
                                               "results", "tools_total_chars", "verdict")}, indent=1))
        print("tools: " + ", ".join(f"{n} {c}" for n, c in row["tool_chars"].items()))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
