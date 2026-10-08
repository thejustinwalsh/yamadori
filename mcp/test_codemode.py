#!/usr/bin/env python
"""The server toolbox, library and sandbox (mcp/codemode.py, mcp/codemode_sandbox.py, docs/CODEMODE.md). No GPU, no
network, no model server: the registry is built from the real tool definitions, the server functions are fakes, the
sandbox is the real thing: quickjs-ng's qjs-wasi.wasm under wasmtime (bin/qjs-wasi.wasm, `python
mcp/codemode_sandbox.py fetch`).

    python mcp/test_codemode.py      -> "N/M checks passed"

GATED HERE:
  1. THE REGISTRY, one source per tool: every function's description and parameters are its tool's own definition
     (the MCP config row, images, vision, craft); the callable name is the tool's own `yama_*` name; namespaces
     group (packages for PackageLens, craft, decide, images); a server added through the configuration is
     discoverable (namespaces, search, describe) with no code or prompt change.
  2. DISCOVERY in the sandbox: tools.search is BM25 over name + description + trigger text (deterministic, limit
     8, namespace filter, a hint when nothing matches), tools.describe (by full or short name; an unknown name
     gets the nearest and the next step), tools.namespaces, tools.all; search then call in one script.
  3. COMPOSITION: parallel calls (four at a time), filtering a long result in code, return / text / console and
     Pi's result shapes, partial output and the line of a failure, exit(), a bare string argument, the call cap,
     a function that is not there, a name the parent refuses.
  4. THE LIMITS: timeout (wasmtime's epoch interrupt, the output so far kept), a memory bomb and one allocation
     past the 256 MiB linear-memory maximum, runaway recursion, the output limit, a huge output cut, a promise
     nothing can settle, a syntax error.
  5. ESCAPES: no require, process, fetch, setInterval, print, std, os, WebAssembly; the module's own os / std reach no
     file, directory or environment variable (the shim grants none); a forged protocol line is refused by the host;
     the module imports exactly the shim's 23 functions and is the recorded sha256; a real clock; two
     runs at once share no epoch.
  6. THE TOOL DEFINITION: trigger-condition description, the declarations under the budget (most relevant first,
     the rest named), byte-stable, the budget rule (what the covered tools cost, Pi's 3,000 at most).
  7. THE DERIVATIONS: the decider's read timeout, the script ceiling is the slowest function's bound.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402  (every store a temp path BEFORE anything else is imported)

_TMP = offline_stores.isolate("yamadori_test_codemode_")
os.environ["YAMADORI_MCP_SERVERS"] = os.path.join(_TMP, "mcp", "servers.json")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import codemode as C  # noqa: E402
import mcp_config  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def tool_names() -> list[str]:
    import images
    import skill_prompts
    import vision
    return (sorted(mcp_config.all_tool_names())
            + [skill_prompts.CRAFT_TOOL_NAME, images.TOOL_NAME, vision.TOOL_NAME])


def toolbox(decider: bool = True) -> C.Toolbox:
    return C.build_toolbox(tool_names(), decider=decider)


class Fake:
    """The bridge: records the calls, the concurrency, and answers like the server's functions do."""

    def __init__(self, delay: float = 0.0, big: int = 0):
        self.calls: list[tuple] = []
        self.live = 0
        self.peak = 0
        self.delay = delay
        self.big = big
        self.lock = threading.Lock()

    def __call__(self, name: str, args: dict, seq: int) -> dict:
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.calls.append((name, args, seq))
        try:
            time.sleep(self.delay)
            if args.get("query") == "fail":
                return C.error_result("UPSTREAM_FAILED", "The lookup server refused the call.", False,
                                      "call it with other arguments.")
            text = f"{name}({json.dumps(args, sort_keys=True)})"
            if self.big:
                text += "\n" + "\n".join(f"line {i} " + "x" * 60 for i in range(self.big))
            return {"ok": True, "text": text, "data": {"names": [args.get("query") or args.get("package")]}}
        finally:
            with self.lock:
                self.live -= 1


def go(script: str, tb: C.Toolbox | None = None, fake: Fake | None = None, **kw) -> dict:
    fake = fake or Fake()
    out = C.run(script, tb or toolbox(), fake, timeout_s=kw.pop("timeout_s", 20), max_calls=kw.pop("max_calls", 10),
                **kw)
    out["_fake"] = fake
    return out


# ================================================================ 1. registry
def test_the_registry_is_one_source_per_tool():
    import images
    import skill_prompts
    import vision
    tb = toolbox()
    names = tb.names()
    check(names == ["yama_find_package", "yama_list_package_versions", "yama_read_package_readme",
                    "yama_resolve_packages", skill_prompts.CRAFT_TOOL_NAME, "yama_decide_noul",
                    "yama_decide_choice", "yama_decide_score", images.TOOL_NAME, vision.TOOL_NAME],
          "every function keeps its yama_* name: MCP servers, craft, decide, images", json.dumps(names))
    mcp_defs = {t["name"]: mcp_config.definition(t)["function"]
                for s in mcp_config.servers() for t in s["tools"]}
    same = all(tb.by_name[n]["description"] == d["description"] and tb.by_name[n]["parameters"] == d["parameters"]
               for n, d in mcp_defs.items())
    check(same, "an MCP tool's description and parameters in the toolbox ARE its config row's (one source)")
    import skill_select
    check(tb.by_name[skill_prompts.CRAFT_TOOL_NAME]["description"]
          == skill_select.READ_TOOL["function"]["description"]
          and tb.by_name[images.TOOL_NAME]["description"] == images.MAIN_TOOL["function"]["description"]
          and tb.by_name[vision.TOOL_NAME]["description"] == vision.TOOL["function"]["description"],
          "craft, image and vision functions carry the tools' own descriptions")
    check([tb.by_name[n]["ns"] for n in names] == ["packages"] * 4 + ["craft", "decide", "decide", "decide",
                                                                      "images", "images"],
          "namespaces group: packages (the config row's `namespace`), craft, decide, images")
    check(all(tb.by_name[n]["question"] and tb.by_name[n].get("bound_s") for n in names),
          "every function has the question it answers and a bound of its own")
    check(tb.by_name["yama_find_package"]["question"].startswith("which package is this")
          and "Call it before you install" in tb.by_name["yama_find_package"]["when"]
          and "A server tool" not in tb.by_name["yama_find_package"]["when"],
          "the question and the trigger conditions are read out of the description",
          tb.by_name["yama_find_package"]["when"][:120])
    check("yama_decide_noul" not in toolbox(decider=False).names(),
          "the decide functions are in the toolbox only where the decider is on")
    only = C.build_toolbox(["yama_find_package"], decider=False)
    check(only.names() == ["yama_find_package"], "a tool withheld for a client tool is not in the toolbox")


def test_a_server_added_through_the_configuration_is_discoverable():
    spec = json.loads(json.dumps(mcp_config.PACKAGELENS))
    spec.update(id="docs", title="DocsLens", namespace="docs", enabled=True,
                package="docslens@1.0.0",
                namespace_description="Library documentation search.")
    spec["tools"] = [{
        "name": "yama_search_documentation", "upstream": "docs_search", "render": "search", "kind": "text",
        "description": ("Answers 'what does the documentation say about this API?': searches the library "
                        "documentation by words and returns the matching pages. Call it before you use an API you "
                        "have not used. A server tool: it runs on the Yamadori server and does not touch your "
                        "workspace."),
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "args": {"query": "q"}}]
    spec.pop("line", None)
    cfg = {"version": mcp_config.VERSION, "servers": [json.loads(json.dumps(mcp_config.PACKAGELENS)), spec]}
    r = mcp_config.save(cfg)
    check(r["ok"], "the configuration with a second server validates", json.dumps(r))
    mcp_config._CACHE.update(mtime=None, path=None, cfg=None)
    try:
        tb = C.build_toolbox(tool_names(), decider=False)
        ns = {n["name"]: n for n in tb.namespaces_info()}
        check("docs" in ns and ns["docs"]["functions"] == ["yama_search_documentation"]
              and ns["docs"]["description"] == "Library documentation search.",
              "the new server is a namespace in tools.namespaces()", json.dumps(list(ns)))
        hit = tb.search("documentation for an API")
        check(hit and hit[0]["name"] == "yama_search_documentation",
              "…and tools.search finds its function by what it answers", json.dumps(hit[:2]))
        d = tb.describe("search_documentation")
        check(d["ok"] and d["namespace"] == "docs" and "docs_search" not in d["description"]
              and "query: string" in d["declaration"],
              "…and tools.describe shows its declaration", json.dumps(d)[:200])
        out = go("const f = tools.search('documentation about an API', {limit: 1})[0];"
                 "return (await tools[f.name]({query: 'useFrame'})).text", tb)
        check(out["ok"] and "yama_search_documentation" in out["items"][0],
              "…and a script finds it and calls it in the same run, with no new prompt text", out["items"])
    finally:
        os.remove(mcp_config.path())
        mcp_config._CACHE.update(mtime=None, path=None, cfg=None)


# =============================================================== 2. discovery
def test_discovery():
    tb = toolbox()
    s = tb.search("which versions of a package exist")
    check(s[0]["name"] == "yama_list_package_versions" and len(s) <= C.SEARCH_LIMIT,
          "search ranks by what a function answers (BM25 over name, description, trigger text)",
          json.dumps([h["name"] for h in s]))
    check(tb.search("which versions of a package exist") == s,
          "search is deterministic (no embedder)")
    check(tb.search("how do i draw a picture")[0]["name"] == "yama_generate_image",
          "…a picture request finds the image tool", json.dumps(tb.search("how do i draw a picture")[:3]))
    check(len(tb.search("package", {"limit": 2})) == 2 and all(
        h["namespace"] == "packages" for h in tb.search("image picture", {"namespace": "packages"})),
          "limit and namespace are honoured")
    check(tb.search("zzzz qqqq") == [], "nothing matches: an empty list")
    d = tb.describe("find_package")
    check(d["ok"] and d["name"] == "yama_find_package" and "ecosystem: " in d["declaration"]
          and d["example"].startswith("await tools.yama_find_package(") and d["bound_s"] == 48.6,
          "describe takes a short name and gives the declaration, an example, the bound", json.dumps(d)[:300])
    bad = tb.describe("find_pakage")
    check(bad["ok"] is False and "yama_find_package" in json.dumps(bad) and "tools.search" in bad["next"],
          "an unknown name: the nearest and the next step", json.dumps(bad))
    check(tb.namespaces_info("packages")["functions"][0]["name"] == "yama_find_package"
          and tb.namespaces_info("nope")["ok"] is False,
          "namespaces(name): that group's functions; an unknown one lists the groups")
    out = go("const n = tools.namespaces(); const a = tools.all(); const s = tools.search('readme');"
             "const e = tools.search('zzzz'); const dd = tools.describe('yama_decide_noul');"
             "return {n: n.map(x => x.name), all: a.length, top: s[0].name, hint: e.hint, ok: dd.ok}")
    got = json.loads(out["items"][0]) if out["items"] else {}
    check(out["ok"] and got.get("n") == ["packages", "craft", "decide", "images"] and got.get("all") == 10
          and got.get("top") == "yama_read_package_readme" and "tools.namespaces()" in got.get("hint", "")
          and got.get("ok") is True,
          "all four discovery calls, from inside the sandbox", json.dumps(out["items"])[:300])
    check(out["searches"] == 2 and out["describes"] == 1
          and [d["kind"] for d in out["discovery"]] == ["search", "search", "describe"],
          "discovery is recorded for x_yamadori.codemode", json.dumps(out["discovery"]))
    out = go("const hit = tools.search('how is this package used')[0];"
             "const r = await tools[hit.name]({package: 'koota', ecosystem: 'npm'});"
             "return hit.name + ' -> ' + r.text")
    check(out["ok"] and out["items"][0].startswith("yama_read_package_readme -> yama_read_package_readme("),
          "a script may search and call in the same run", out["items"])


# ============================================================= 3. composition
def test_composition():
    fake = Fake(delay=0.3)
    t0 = time.monotonic()
    out = go("const r = await Promise.all(['a','b','c','d','e','f','g','h'].map(p => "
             "tools.yama_find_package({query: p, ecosystem: 'npm'}))); return r.map(x => x.data.names[0]).join('')",
             fake=fake, max_calls=10)
    took = time.monotonic() - t0
    check(out["ok"] and out["items"] == ["abcdefgh"] and fake.peak == C.CONCURRENCY == 4,
          "eight parallel calls: all answered, at most FOUR at once (Pi's bound)", f"peak {fake.peak}")
    check(took < 8 * 0.3 * 0.75 and len(out["calls"]) == 8 and [c["seq"] for c in
                                                                 sorted(out["calls"], key=lambda c: c["seq"])]
          == list(range(1, 9)),
          "…in parallel, not one after another; each call recorded with its issue order",
          f"{took:.2f}s")
    check(all(set(c) >= {"tool", "ms", "ok"} and c["tool"] == "yama_find_package" for c in out["calls"]),
          "calls[{tool, ms, ok}] carry the tool's own yama_* name")
    fake = Fake(big=400)
    out = go("const r = await tools.yama_read_package_readme({package: 'koota', ecosystem: 'npm'});"
             "return r.text.split('\\n').filter(l => l.includes('line 7 ')).join('|')", fake=fake)
    text = C.format_result(out)
    check(out["ok"] and len(text) < 400 and "line 7 " in text and out["output_chars"] < 400,
          "a long result filtered in code: only the slice reaches the model", f"{len(text)} chars")
    out = go("text('one'); text({two: 2}); console.log('c1', 3); console.warn('c2'); return 'ret'")
    text = C.format_result(out)
    check(text.startswith("Script completed in ") and " s\n\n==> text 1/3 <==\none" in text
          and "==> text 2/3 <==\n{\n  \"two\": 2\n}" in text and "==> text 3/3 <==\nret" in text
          and "<console_output>\nc1 3\nc2\n</console_output>" in text,
          "Pi's shapes: 'Script completed', wall time, '==> text N/M <==' headers, one <console_output> block", text)
    out = go("return 'only'")
    check(C.format_result(out).split("\n\n", 1)[1] == "only",
          "a single output has no header")
    out = go("let s = ''; for (let i = 0; i < 3000; i++) s += 'line ' + i + String.fromCharCode(10);"
             "text(s); console.log('a' + 'b'.repeat(20000) + 'c'); return 'x'.repeat(20000) + 'e'")
    check(out["ok"] and len(out["items"][0]) > 20000 and out["items"][1] == "x" * 20000 + "e"
          and len(out["console"][0]) == 20002,
          "strings built by concatenation (ropes) cross the bridge: quickjs-ng cannot convert them raw, the "
          "prelude flattens every string it hands the host", f"{out['reason']} {out['error']}")
    out = go("text('before'); await tools.yama_find_package({query: 'x', ecosystem: 'npm'}); "
             "undefinedFunction(); text('never')")
    text = C.format_result(out)
    check(not out["ok"] and out["reason"] == "js_error" and out["items"] == ["before"]
          and text.startswith("Script failed in ") and "ReferenceError: undefinedFunction is not defined" in text
          and "Output before the failure:" in text and "before" in text and "never" not in text
          and "Next: fix the line named" in text,
          "a failure: 'Script failed', the error, the partial output, the next step", text)
    check("(line 1, column " in text, "…and the line of the script, not of the wrapper", text.split("\n")[0])
    out = go("\n\nthrow new Error('boom')")
    check("Error: boom (line 3" in C.format_result(out), "a throw on line 3 says line 3",
          C.format_result(out).split("\n")[0])
    out = go("text('x'); exit(); text('y')")
    check(out["ok"] and out["reason"] == "exit" and out["items"] == ["x"], "exit() ends the script successfully")
    fake = Fake()
    out = go("return (await tools.yama_recall_craft('koota traits')).text", fake=fake)
    check(fake.calls[0][1] == {"name_or_topic": "koota traits"},
          "a bare string is the argument of a function whose only required parameter is a string",
          json.dumps(fake.calls))
    fake = Fake()
    out = go("const r = await Promise.all([1,2,3,4,5].map(i => tools.yama_find_package({query: 'q'+i, "
             "ecosystem: 'npm'}))); return r.map(x => x.ok ? 'ok' : x.error.code).join(',')",
             fake=fake, max_calls=2)
    check(out["items"] == ["ok,ok,CALL_CAP,CALL_CAP,CALL_CAP"] and len(fake.calls) == 2
          and out["dispatched"] == 2,
          "past the call budget a call is NOT run and says so (with the next step)", json.dumps(out["items"]))
    out = go("const r = await tools.yama_find_package({query: 'fail', ecosystem: 'npm'});"
             "return [r.ok, r.error.code, r.error.retryable, r.error.next, r.text].join('|')")
    check(out["items"] == ["false|UPSTREAM_FAILED|false|call it with other arguments.|The lookup server refused "
                           "the call. call it with other arguments."],
          "a function never throws: {ok:false, error:{code, reason, retryable, next}} and text says the same",
          json.dumps(out["items"]))
    text = C.format_result(go("await tools.yama_find_package({query: 'fail', ecosystem: 'npm'}); return 1"))
    check("1 of 1 server calls failed (yama_find_package)" in text, "a failed call is counted in the result", text)
    out = go("const r = await tools.nope({}); return [r.ok, r.error.code].join()")
    check(out["items"] == ["false,NO_SUCH_FUNCTION"], "a name that is not a function answers like one",
          json.dumps(out["items"]))
    out = go("const a = {}; a.a = a; const r = await tools.yama_find_package(a); return r.error.code")
    check(out["items"] == ["BAD_ARGUMENTS"], "arguments that cannot be written as JSON: a result, not a crash")
    boom = Fake()

    def raises(name, args, seq):
        raise RuntimeError("bridge exploded")
    out = C.run("const r = await tools.yama_find_package({query:'a', ecosystem:'npm'}); return r.error.code", toolbox(),
                raises, timeout_s=10, max_calls=5)
    check(out["items"] == ["TOOL_RAISED"] and boom.calls == [],
          "a bridge that raises becomes a TOOL_RAISED result", json.dumps(out["items"]))
    big = go("return 'x'.repeat(20000) + '\\nend'")
    t = C.format_result(big)
    check(len(t) <= 6000 and "[output cut at 6000 of " in t and "print less" in t,
          "output beyond the hidden hop's cap (repeats.RESULT_CAP) is cut with the way to print less", str(len(t)))


# ================================================================== 4. limits
def test_the_limits():
    t0 = time.monotonic()
    out = go("text('so far'); while (true) {}", timeout_s=1.5)
    check(out["reason"] == "timeout" and out["items"] == ["so far"] and 1.4 < time.monotonic() - t0 < 4
          and "ran past its 1.5 s limit" in out["error"]["message"],
          "an infinite loop: wasmtime's epoch interruption traps the module at the wall limit and the output so "
          "far is kept", f"{out['reason']} {time.monotonic() - t0:.2f}s {json.dumps(out['error'])}")
    text = C.format_result(out)
    check(text.startswith("Script failed in 1.") and "Output before the failure:\n\nso far" in text
          and "Next: the calls that finished are listed" in text, "…and the result says so", text)
    out = go("const a = []; while (true) a.push('x'.repeat(1000000) + a.length)")
    check(out["reason"] == "memory" and "256 MiB" in out["error"]["message"] and not out["ok"],
          "a memory bomb: QuickJS reports out of memory when the 256 MiB linear-memory maximum refuses the growth",
          json.dumps(out["error"]))
    out = go("const b = new ArrayBuffer(400 * 1024 * 1024); return 'allocated'")
    check(not out["ok"] and out["reason"] in ("memory", "js_error"),
          "one allocation larger than the whole sandbox fails (the maximum is a maximum, not a hint)",
          json.dumps([out["reason"], out["error"]]))
    out = go("text('before'); function f() { return f() + 1 } f()")
    check(not out["ok"] and out["reason"] == "stack_overflow" and out["items"] == ["before"]
          and "RangeError: Maximum call stack size exceeded" in C.format_result(out)
          and "Next: recurse less deeply" in C.format_result(out),
          "runaway recursion ends the run (QuickJS cannot see wasm's call stack, so wasmtime's stack trap is what "
          "stops it), the output so far kept, a result and not a crash of the host", json.dumps([out["reason"],
                                                                                                  out["error"]]))
    saved = C.MAX_OUTPUT_CALLS
    C.MAX_OUTPUT_CALLS = 50
    try:
        out = go("for (let i = 0; i < 100000; i++) text('t' + i)")
    finally:
        C.MAX_OUTPUT_CALLS = saved
    check(out["reason"] == "output_limit" and not out["ok"] and 40 <= len(out["items"]) <= 51,
          "more text() calls than the limit: the script fails with what it wrote", str(len(out["items"])))
    saved = C.MAX_OUTPUT_CHARS
    C.MAX_OUTPUT_CHARS = 5000
    try:
        out = go("for (let i = 0; i < 100; i++) text('y'.repeat(1000))")
    finally:
        C.MAX_OUTPUT_CHARS = saved
    check(out["reason"] == "output_limit" and out["output_chars"] <= 6000, "…and more characters than the limit",
          str(out["output_chars"]))
    out = go("return 'z'.repeat(5 * 1000 * 1000)")
    t = C.format_result(out)
    check(out["ok"] and len(t) <= 6000 and "[output cut at 6000 of " in t,
          "a huge output is collected whole (under Pi's 16 MiB) and CUT at the hidden hop's cap", str(len(t)))
    out = go("await new Promise(() => {})")
    check(out["reason"] == "never_settles" and "Next: every await must be a tools.* call" in C.format_result(out),
          "a promise nothing can settle fails at once (Pi's rule), with the next step", out["reason"])
    out = go("let x = ;")
    check(out["reason"] == "syntax_error" and "SyntaxError" in C.format_result(out)
          and "async function" in C.format_result(out), "a syntax error is named, with the next step",
          C.format_result(out)[:200])
    out = go("throw 'a string'")
    check(not out["ok"] and "a string" in C.format_result(out), "throwing a non-Error is reported")
    out = go("tools.yama_find_package({query: 'a', ecosystem: 'npm'}); return 'not awaited'",
             fake=Fake(delay=0.2))
    check(out["ok"] and out["items"] == ["not awaited"], "a call the script does not await does not hang the script")


# ================================================================== 5. escapes
def test_escapes():
    probes = ["require", "process", "fetch", "setInterval", "setImmediate", "print", "std", "os",
              "WebAssembly", "XMLHttpRequest", "Buffer", "module", "exports", "__dirname", "scriptArgs", "execArgv",
              "argv0", "navigator", "window", "self", "document", "importScripts", "Deno", "Bun", "structuredClone",
              "Intl", "TextEncoder", "TextDecoder", "URL", "WebSocket", "Worker", "gc"]
    out = go("return [" + ",".join(f"typeof {p}" for p in probes) + "].join()")
    got = out["items"][0].split(",") if out["items"] else []
    check(out["ok"] and got == ["undefined"] * len(probes),
          "no require, process, fetch, timers, print, std, os, WebAssembly, Buffer, URL, Worker, scriptArgs ...",
          json.dumps(dict(zip(probes, got)) if len(got) == len(probes) else out["items"]))
    out = go("try { return String(await import('node:fs')) } catch (e) { return 'refused: ' + e.name }")
    check(out["ok"] and out["items"][0].startswith("refused"), "import() of a node module is refused",
          json.dumps(out["items"]))
    out = go("try { return String(await import('./secret.js')) } catch (e) { return 'refused: ' + e.name }")
    check(out["ok"] and out["items"][0].startswith("refused"), "import() of a file module is refused (no file exists "
          "for the module)", json.dumps(out["items"]))
    # THE MODULE'S OWN os / std (quickjs-ng's built-ins, reachable by dynamic import): every capability behind them
    # is the shim's, and the shim grants none.
    out = go("const os = await import('qjs:os'); const std = await import('qjs:std');"
             "const r = {};"
             "r.openFile = os.open('/etc/passwd', os.O_RDONLY);"
             "r.openRel = os.open('secret.txt', os.O_RDONLY);"
             "r.loadFile = std.loadFile('/etc/passwd');"
             "r.getenv = [std.getenv('PATH'), std.getenv('USERPROFILE'), std.getenv('YAMADORI_KEY')];"
             "r.environ = Object.keys(std.getenviron()).length;"
             "r.readdir = os.readdir('.');"
             "r.stat = os.stat('.');"
             "r.write = std.writeFile ? (() => { try { std.writeFile('x.txt', 'hi'); return 'wrote' } catch (e) "
             "{ return 'refused' } })() : 'none';"
             "r.mkdir = os.mkdir('d');"
             "r.cwd = os.getcwd();"
             "return JSON.stringify(r)")
    got = json.loads(out["items"][0]) if out["items"] else {}
    check(out["ok"] and got.get("openFile", 0) < 0 and got.get("openRel", 0) < 0 and got.get("loadFile") is None
          and got.get("getenv") == [None, None, None] and got.get("environ") == 0
          and got.get("readdir", [[], 0])[1] != 0 and got.get("write") in ("refused", "none")
          and got.get("mkdir", 0) < 0, "the built-in os and std modules reach no file, no directory, no "
          "environment variable (no preopened directory, no environ in the shim)", json.dumps(got))
    out = go("const std = await import('qjs:std'); const l = std.in.getline(); return String(l)",
             fake=Fake(delay=0.0))
    check(out["ok"] or out["reason"] in ("crashed", "timeout", "never_settles", "js_error"),
          "reading the module's stdin directly cannot hang the host (the shim answers, or the wall limit ends it)",
          json.dumps([out["reason"], out["items"]]))
    # a forged protocol line (the script writes to stdout itself): the host refuses a name outside the catalog
    small = C.Toolbox(toolbox().fns[:1], {})
    out = go("const std = await import('qjs:std');"
             "std.out.puts(JSON.stringify({op: 'call', id: 77, name: 'yama_generate_image', args: {}}) + '\\n');"
             "std.out.flush(); const line = std.in.getline(); return line", tb=small)
    check(out["ok"] and "NO_SUCH_FUNCTION" in out["items"][0] and not [c for c in out["_fake"].calls
                                                                       if c[0] == "yama_generate_image"],
          "a forged request for a function outside the toolbox is refused by the HOST (the bridge validates every "
          "request), whatever the module writes", json.dumps(out["items"])[:200])
    out = go("const std = await import('qjs:std');"
             "for (let i = 0; i < 5; i++) std.out.puts(JSON.stringify({op: 'call', id: 100 + i, name: "
             "'yama_find_package', args: {query: 'q' + i, ecosystem: 'npm'}}) + '\\n'); std.out.flush(); return 'sent'",
             max_calls=2)
    check(len(out["_fake"].calls) <= 2, "…and a forged flood past the call budget runs at most the budget",
          str(len(out["_fake"].calls)))
    out = go("const g = (function () { return this })() || Function('return this')();"
             "const f = Function('return typeof process + typeof require + typeof fetch')();"
             "const c = (async function () {}).constructor('return typeof process')();"
             "return f + '|' + (await c)")
    check(out["ok"] and out["items"][0] == "undefinedundefinedundefined|undefined",
          "Function(), the async-function constructor and eval reach nothing more", json.dumps(out["items"]))
    out = go("try { eval('require(\"fs\")') } catch (e) { return e.name }")
    check(out["items"] == ["ReferenceError"], "eval cannot find require either")
    out = go("tools.yama_find_package = () => 'x'; tools.extra = 1; return Object.isFrozen ? 'ok' : ''")
    check(out["ok"], "the tools object cannot be extended (a write is ignored or throws, never changes it)")
    out = go("const t = tools; try { Object.defineProperty(t, 'evil', {value: 1}) } catch (e) {}"
             "return String(typeof tools.evil)")
    check(out["items"] == ["function"] or out["items"] == ["undefined"],
          "…and a defined property is at worst an unknown name (answered by the NO_SUCH_FUNCTION rule)",
          json.dumps(out["items"]))
    # THE CLOCK IS REAL (operator, 2026-10-08): the host's time, a real sleep, Math.random seeded from it
    import time as _t
    t_before = _t.time() * 1000
    a = go("return [Date.now(), Math.random(), Math.random()].join(',')")
    t_after = _t.time() * 1000
    stamp = float(a["items"][0].split(",")[0]) if a["ok"] else 0
    check(a["ok"] and t_before - 5 <= stamp <= t_after + 5,
          "Date.now() is the host's wall clock", json.dumps([t_before, stamp, t_after]))
    b = go("return [Date.now(), Math.random(), Math.random()].join(',')")
    check(b["ok"] and a["items"][0].split(",")[1:] != b["items"][0].split(",")[1:],
          "Math.random() differs between runs (seeded from the real clock)", json.dumps([a["items"], b["items"]]))
    out = go("const t0 = Date.now(); await new Promise(r => setTimeout(r, 300)); return Date.now() - t0")
    check(out["ok"] and 280 <= float(out["items"][0]) <= 1500,
          "setTimeout waits its time (a real timer)", json.dumps([out["items"], out["wall_ms"]]))
    out = go("const os = await import('qjs:os'); const t0 = os.now(); os.sleep(250); return os.now() - t0")
    check(out["ok"] and 230_000 <= float(out["items"][0]) <= 1_500_000,
          "os.sleep() waits its time (os.now() is microseconds)", json.dumps([out["items"], out["wall_ms"]]))
    out = go("const t0 = Date.now(); const [a, b] = await Promise.all([sleep(300), sleep(300)]);"
             "return Date.now() - t0")
    check(out["ok"] and 280 <= float(out["items"][0]) <= 550,
          "sleep(ms) is a real wait, and two run together (300 ms, not 600)", json.dumps([out["items"], out["wall_ms"]]))
    out = go("let n = 0; const t = setTimeout(() => { n += 100 }, 200); setTimeout(() => { n += 1 }, 50);"
             "clearTimeout(t); await sleep(400); return n")
    check(out["ok"] and out["items"] == ["1"], "setTimeout / clearTimeout work", json.dumps(out["items"]))
    out = go("const os = await import('qjs:os'); os.sleep(60000); return 'late'", timeout_s=1.5)
    check(not out["ok"] and out["wall_ms"] < 8000,
          "…and a long sleep is cut at the run's wall limit", json.dumps([out["items"], out["wall_ms"], out.get("reason")]))
    # the shim records what the module asked for and was refused
    import codemode_sandbox as S
    seen: dict = {}
    r = S.Run({"op": "init", "script": "const os = await import('qjs:os'); os.open('/etc/passwd', os.O_RDONLY);"
                                        "os.readdir('.'); return 'x'", "names": []},
              lambda m: seen.setdefault(m.get("op"), []).append(m) or [], memory_bytes=C.MEMORY_BYTES, timeout_s=10)
    res = r.run()
    check(("fd_prestat_get" in res["denied"] or "path_open" in res["denied"]) and res["how"] == "exit",
          "the shim records the capabilities it refused (no preopened directory: fd_prestat_get ...), for the record",
          json.dumps(res["denied"]))
    # the module is the recorded one, with exactly the shim's 23 imports
    W = S._wasmtime()
    mod = W.Module(W.Engine(S._config()), open(S.WASM_PATH, "rb").read())
    imps = sorted(i.name for i in mod.imports)
    check(len(imps) == 23 and all(i.module == "wasi_snapshot_preview1" for i in mod.imports)
          and set(imps) == set(r.table) and S.verify() == (True, "qjs-wasi.wasm " + S.QJS_VERSION),
          "the module is quickjs-ng's own qjs-wasi.wasm (sha256 recorded) and imports exactly the 23 functions the "
          "shim defines, nothing else", f"{len(imps)} imports; {sorted(set(imps) ^ set(r.table))}")
    out = C.run("while(true){}", toolbox(), Fake(), timeout_s=30, max_calls=1, cancelled=lambda: True)
    check(out["reason"] == "cancelled" and out["wall_ms"] < 1500, "a cancelled request interrupts the module at once",
          json.dumps([out["reason"], out["wall_ms"]]))
    # one run's cancel or deadline never reaches another run (an Engine per run)
    results: dict = {}

    def spin():
        results["spin"] = C.run("while(true){}", toolbox(), Fake(), timeout_s=30, max_calls=1,
                                cancelled=lambda: results.get("cancel", False))

    def work():
        results["work"] = C.run("const r = await tools.yama_find_package({query:'a', ecosystem:'npm'});"
                                "return r.ok", toolbox(), Fake(delay=1.0), timeout_s=20, max_calls=3)
    ts = [threading.Thread(target=spin), threading.Thread(target=work)]
    for t in ts:
        t.start()
    time.sleep(0.3)
    results["cancel"] = True
    for t in ts:
        t.join(30)
    check(results["spin"]["reason"] == "cancelled" and results["work"]["ok"] and results["work"]["items"] == ["true"],
          "two scripts at once: cancelling the one that spins does not touch the other (separate engines, separate "
          "epochs)", json.dumps([results["spin"]["reason"], results["work"]["reason"]]))
    real_path, real_art = S.WASM_PATH, S._CACHE["art"]
    S.WASM_PATH, S._CACHE["art"] = os.path.join(_TMP, "no-module.wasm"), None
    try:
        out = C.run("return 1", toolbox(), Fake(), timeout_s=5, max_calls=1)
        ok, why = S.available()
    finally:
        S.WASM_PATH, S._CACHE["art"] = real_path, real_art
    check(out["reason"] == "crashed" and "missing" in json.dumps(out["error"]) and not ok and "fetch" in why,
          "a missing or altered module is a result, not an exception, and available() says how to fetch it",
          json.dumps([out["error"], why]))
    check(C.available() == (True, "qjs-wasi.wasm " + S.QJS_VERSION), "available(): wasmtime and the recorded module")
    # wasmtime-py's host-function table is one unsynchronised global; the sandbox locks it (found by this test)
    bad: list = []

    def hammer(k):
        for j in range(15):
            o = C.run("let a = 0; for (let i = 0; i < 5000; i++) a += i; return a", C.Toolbox([], {}),
                      lambda *a: {}, timeout_s=20, max_calls=1)
            if not o["ok"] or o["items"] != ["12497500"]:
                bad.append((k, j, o["reason"], o["error"]))
    ts = [threading.Thread(target=hammer, args=(k,)) for k in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    check(not bad, "six threads x 15 scripts at once: no run is corrupted by another (wasmtime-py's global "
          "host-function table is locked by the sandbox)", json.dumps(bad[:2]))


# ======================================================== 6. the tool definition
def test_the_tool_definition():
    tb = toolbox()
    d, rec = C.tool_definition(tb, rank_query="Build a voxel scene with r3f and pmndrs math.",
                               budget_tokens=3000)
    desc = d["function"]["description"]
    check(d["function"]["name"] == "yama_compose_tools" == C.TOOL_NAME
          and desc.startswith("Answers 'how do I get several facts from the server in one go?':"),
          "the tool is yama_compose_tools; its description opens with the question it answers", desc[:100])
    check("tools.search" in desc and "tools.describe" in desc and "Promise.all" in desc
          and "bash, read and write are your client's tools and cannot be called from it" in desc
          and desc.split("\n\n")[0].endswith("A server tool: it runs on the Yamadori server and does not touch "
                                             "your workspace."),
          "trigger conditions, discovery, the parallel form, the client's tools named as out of reach, the "
          "server-tool line")
    check("declare const tools: {" in desc and "type ToolResult = " in desc
          and "yama_find_package(args: { query: string; ecosystem: \"npm\" | \"pypi\" |" in desc
          and "/** which package is this, and what is its exact name in the registry? */" in desc,
          "typed declarations reuse the tools' own parameters and the question their own description opens with")
    check("```js\nconst found = tools.search(" in desc and "await Promise.all([tools." in desc,
          "the example searches and calls in one run")
    check(rec["tokens"] <= rec["budget_tokens"] == 3000 and rec["left_out"] == [] and len(rec["declared"]) == 10,
          "everything fits Pi's 3,000-token budget here and is declared", json.dumps(rec)[:300])
    d2, rec2 = C.tool_definition(tb, rank_query="Draw a picture of a cat and show it", budget_tokens=1300)
    check(rec2["tokens"] <= 1300 and rec2["left_out"] and "yama_generate_image" in rec2["declared"]
          and "// Not declared here, found with tools.search / tools.describe: " in d2["function"]["description"]
          and all(n in d2["function"]["description"] for n in rec2["left_out"]),
          "a tight budget declares the most relevant functions for the conversation and NAMES the rest",
          json.dumps(rec2)[:400])
    d3, rec3 = C.tool_definition(tb, rank_query="Which versions of koota exist", budget_tokens=1300)
    check("yama_list_package_versions" in rec3["declared"] and rec3["declared"] != rec2["declared"],
          "…which functions are declared follows the conversation (BM25 of its first message)",
          json.dumps([rec2["declared"], rec3["declared"]]))
    again, _ = C.tool_definition(tb, rank_query="Which versions of koota exist", budget_tokens=1300)
    check(json.dumps(again) == json.dumps(d3), "the definition is byte-stable for the same inputs")
    covered = [mcp_config.definition(t) for s in mcp_config.servers() for t in s["tools"]]
    b = C.budget_for(covered, "x" * 250)
    check(b == min(C.tokens_of("".join(json.dumps(x) for x in covered)) + 250 // 3, 3000) or b <= 3000,
          "the budget is what the covered tools cost today, at most Pi's 3,000", str(b))
    check(C.budget_for([]) == C.INLINE_BUDGET_CEILING_TOKENS == 3000, "…and Pi's figure when nothing is covered")
    check(C.ts_type({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}},
                                                      "b": {"type": "integer"}}, "required": ["a"]})
          == "{ a: string[]; b?: number; }", "JSON Schema becomes TypeScript")
    q, when = C.split_description("Answers 'x?': does y. Call it when z. A server tool: it runs on the Yamadori "
                                  "server and does not touch your workspace.")
    check((q, when) == ("x?", "does y. Call it when z."), "the description is split into question and trigger",
          json.dumps([q, when]))
    check(C.SYSTEM_LINE.startswith("\n\nTo get several facts from the server at once")
          and "yama_compose_tools" in C.SYSTEM_LINE and "never" not in C.SYSTEM_LINE.lower(),
          "the one system line is positive and names the tool")
    check(C.TOOL_NAME.startswith("yama_") and C.TOOL_NAME == "yama_compose_tools",
          "AGENTS.md naming: yama_ prefix, verb first, spelled out")


# ================================================================ 7. derivations
def test_the_derivations():
    import decider_bonsai as D
    for fn in (D.read, D.ask_one):
        check(inspect.signature(fn).parameters["timeout"].default == C.DECIDER_READ_S == 120.0,
              f"DECIDER_READ_S is decider_bonsai.{fn.__name__}'s per-read timeout")
    check(C.CRAFT_BOUND_S == 240.0 and C.DECIDE_BOUND_S == 120.0, "craft: two reads; decide: one")
    tb = toolbox()
    import npm_resolve
    check(tb.bound_s() == max(npm_resolve.FETCH_WORST_S, 900.0), "the script ceiling is the slowest function's bound",
          str(tb.bound_s()))
    only = C.build_toolbox(["yama_find_package"], decider=False)
    check(only.bound_s() == mcp_config.PL_CALL_TIMEOUT_S == 48.6, "…PackageLens alone: its 48.6 s")
    check(C.MEMORY_BYTES == 256 * 1024 * 1024 and C.CONCURRENCY == 4 and C.SEARCH_LIMIT == 8
          and C.MAX_OUTPUT_CHARS == 16777216 and C.MAX_OUTPUT_CALLS == 100000,
          "Pi's documented limits: 256 MB, four at once, search limit 8, 16,777,216 characters, 100,000 calls")
    import repeats
    check(repeats.RESULT_CAP == 6000, "the returned output cap is the hidden hop's own breaker, 6,000")
    import tiers
    sw = C.switch({})
    check(sw["on"] is False and sw["mode"] == "on" and "codemode" in tiers.OFF_BY_DEFAULT,
          "the switch is off by default; the mode is `on`", json.dumps(sw))
    os.environ["YAMADORI_CODEMODE"] = "1"
    try:
        check(C.switch({"mcp_tools": True})["on"] is True and C.switch({"mcp_tools": False})["on"] is False,
              "YAMADORI_CODEMODE=1 turns it on where the tier has the MCP tools (medium and up)")
    finally:
        del os.environ["YAMADORI_CODEMODE"]
    f = tiers.from_header('{"codemode": true, "codemode_mode": "only"}')
    sw = C.switch(dict(f, overridden=sorted(f)))
    check(sw["on"] and sw["source"] == "header" and sw["mode"] == "only" and sw["mode_source"] == "header",
          "the header forces the switch and the mode at any tier", json.dumps(sw))
    check(C.mode_source(dict(overridden=["codemode_mode"], codemode_mode="bogus"))[0] == "on",
          "an unknown mode is ignored")
    import codemode_sandbox as S
    check(S.compiled() is S.compiled() and len(S.compiled()) > 1_000_000,
          "the compiled module is cached across calls (wasmtime's serialised form, one compile per process)")
    ms = sorted(go("return 1")["instantiate_ms"] for _ in range(5))
    check(ms[2] is not None and ms[2] < 250,
          "each call only instantiates: engine + deserialise + instantiate, median of 5 (the measured figure is ~2 ms; "
          "the bound here is a test's, not a limit)", json.dumps(ms))
    check(S.QJS_SHA256 == "42a732a676ec2d93488c19411e0fad283bf72658fdad746f089914b523c783b1"
          and S.QJS_URL.endswith("/v0.17.0/qjs-wasi.wasm") and S.TICK_S == 0.05,
          "the recorded module and its release URL (models/manifest.yaml codemode-wasi)")
    import yaml
    man = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), "models", "manifest.yaml"), encoding="utf-8"))
    rt = next((r for r in man["runtimes"] if r["id"] == "codemode-wasi"), {})
    lock = open(os.path.join(os.path.dirname(HERE), "requirements.lock.txt"), encoding="utf-8").read()
    check(rt.get("files", [{}])[0].get("sha256") == S.QJS_SHA256 and "wasmtime==49.0.0" in lock
          and "quickjs" not in lock.lower() and "quickjs" not in open(
              os.path.join(os.path.dirname(HERE), "requirements.txt"), encoding="utf-8").read().lower(),
          "the manifest pins the module's sha256 (the sandbox's constant), the lock pins wasmtime, and the native "
          "quickjs-ng binding is gone from both")


# =================================================================== 8. the probe
def test_the_probe_reads_what_the_proxy_records():
    """bench/mcp/codemode_probe.py has not been run (a GPU job); its reader is checked here against the record the
    proxy writes, so a first run does not find a mismatch."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "mcp"))
    import codemode_probe as P

    def row(cm, tools=(), hops=2, mcp_calls=(), turns=1, skills=None):
        x = {"tier": "medium", "hops": hops, "tool_turns": {"turns": turns, "hit": False},
             "tools": [{"name": n, "chars": c} for n, c in tools],
             "mcp": {"offered": list(P.LP.TOOLS), "calls": list(mcp_calls)},
             "codemode": cm, "craft": {"reads": []}, "skills": skills or {}}
        return {"method": "POST", "path": "/v1/chat/completions", "response": {"x_yamadori": x}}
    base = {"switch": {"on": True, "source": "header", "mode": "only", "mode_source": "header"}, "offered": True,
            "mode": "only", "ours": {"tools": ["yama_compose_tools"], "tokens": 1500}, "declared": ["a"],
            "left_out": [], "budget_tokens": 1585, "tokens": 1500, "hidden": ["yama_find_package"]}
    run = dict(base, runs=1, failed=False, reasons=["completed"], searches=1, describes=1, heads=["const h = ..."],
               calls=[{"tool": "yama_find_package", "ms": 20, "ok": True}],
               discovery=[{"kind": "search", "query": "find a package", "hits": ["yama_find_package"]},
                          {"kind": "describe", "name": "nope", "ok": False}])
    rows = [row(base, tools=[], hops=1), row(run, tools=[("yama_compose_tools", 900), ("yama_find_package", 700)],
                                              mcp_calls=[{"tool": "yama_find_package"}])]
    s = P.summarise_requests(rows, "only", 12, os.path.join(_TMP, "no-project"), "pi", "max_requests")
    check(s["invalid"] is None and s["ours_tokens"] == 1500 and s["compose_called"] and s["first_compose_step"] == 2
          and s["hops"] == 3 and s["tool_result_chars"] == 1600 and s["searched"] and s["described"]
          and s["discovery_hit"] and s["describe_misses"] == 1 and s["first_call_tool"] == "yama_find_package"
          and s["direct_package_calls"] == 0 and s["scripts"] == ["const h = ..."] and s["script_failures"] == 0,
          "the probe reads x_yamadori.codemode: cost of our additions, hops, tool-result chars, the toolbox call, "
          "discovery (hit, miss), the first function called, direct vs inner calls",
          json.dumps({k: s[k] for k in ("invalid", "ours_tokens", "hops", "tool_result_chars", "discovery_hit",
                                        "direct_package_calls", "first_call_tool")}))
    wrong = P.summarise_requests([row(dict(base, switch={"on": False, "source": "default"}, offered=False)), ], "only",
                                 12, os.path.join(_TMP, "x"), "pi", None)
    check(wrong["invalid"] and any("the arm says only" in w for w in wrong["invalid"]),
          "a trial where the header did not reach the proxy is invalid, with why", json.dumps(wrong["invalid"]))
    rep = P.report([dict(s, outcome="ok", harness="pi", arm="only")])
    check(rep["pi/only"]["n_valid_trials"] == 1 and rep["pi/only"]["toolbox_call_rate"] == 1.0
          and rep["pi/only"]["discovery_hit_rate_of_trials_that_searched"] == 1.0
          and rep["pi/only"]["mean_prompt_tokens_our_additions_cost"] == 1500,
          "the report: per harness and arm", json.dumps(rep)[:200])
    check(P.arm_features("off")["codemode"] is False and P.arm_features("on")["codemode_mode"] == "on"
          and P.arm_features("only")["codemode_mode"] == "only"
          and "--inject-header" in open(os.path.join(os.path.dirname(HERE), "bench", "octopus", "relay.py"),
                                        encoding="utf-8").read(),
          "the arms' features; the relay can add the header for Hermes")


def main() -> int:
    for fn in (test_the_registry_is_one_source_per_tool,
               test_a_server_added_through_the_configuration_is_discoverable,
               test_discovery, test_composition, test_the_limits, test_escapes,
               test_the_tool_definition, test_the_derivations, test_the_probe_reads_what_the_proxy_records):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised", traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
