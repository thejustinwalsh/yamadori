#!/usr/bin/env python
"""Executable graders for the target-domain task set (bench/domain/tasks.jsonl).

WHAT THIS IS

`grade(task, answer_text)` takes a model's whole reply, pulls the code out of
it, writes it into a per-task sandbox, runs a real toolchain against it, and
says which stage it died at:

  extract   no usable fenced code block in the reply
  compile   the answer itself does not compile / parse / link its imports
  test      it compiles, but the grader's checks against it fail
  error     the GRADER could not run (toolchain missing, sandbox unwritable).
            This is never a model failure and a benchmark must count it
            separately -- PROTOCOL rule 3: never let an exception become a
            wrong answer.

It never raises. Every path returns a dict.

WHY THE STAGES ARE SEPARATE

"Did not answer in code", "wrote code that does not compile" and "wrote code
that compiles and is wrong" are three different findings about a model. A
grader that folds them together cannot tell a formatting problem from a
knowledge problem, and a grader that folds `error` into any of them scores
its own bugs as model failures -- the mistake this repo keeps making.

GRADER KINDS

  ts     solution.ts + the task's test.ts, `tsc --strict` (TypeScript 7.0.2,
         pinned in env/package.json) over both. An error located in
         solution.ts is `compile`; an error located only in test.ts (a failed
         type-level assertion, a missing export, an unused @ts-expect-error)
         is `test`. If `runtime` is set the emitted JS is then run under node
         and a non-zero exit is `test`. typegpu tasks use this kind, compiled
         against typegpu@0.12.5 from node_modules, which is byte-identical to
         the indexed `index/packages/_src/typegpu@0.12.5` (checked by
         test_grade.py).
  rust   the answer becomes src/lib.rs of a throwaway crate sharing one warm
         CARGO_TARGET_DIR. `host_test` (a Rust file) is mounted as a
         cfg(test) child module, so it can see private items; errors located
         in lib.rs are `compile`, errors in the grader module are `test`.
         `wasm_test` (an .mjs file) runs under node against the
         wasm32-unknown-unknown build -- raw exports and memory, or, with
         `bindgen: true`, the wasm-bindgen 0.2.128 CLI output.
  tsl    three.js TSL version markers. CONTAMINATED: three.js is in every
         training set. The code is parsed with tree-sitter (PROTOCOL rule 8),
         never pattern-matched. A syntax error, or a named import that
         three@0.185.1 does not export, is `compile`; a required r185 name
         missing, or a deprecated/removed name present, is `test`. Export
         lists are read from the indexed source under the package store.
  tc_tests
         the type-challenges subset (tasks_type_challenges.jsonl; MIT,
         CONTAMINATED -- public since 2020). solution.ts + the challenge's
         own upstream test-cases.ts, `tsc --noEmit --strict` (same pinned
         TypeScript), with @type-challenges/utils vendored into the sandbox.
         The answer is a global script, as in the upstream playground; if it
         is a module instead, the entry names it declares are exported and
         imported into the test file, so `export type X` is not penalised.
         Errors in solution.ts are `compile`, in test-cases.ts `test`.
  react  React 19.2.8 components and hooks (tasks/react-*, indexed in
         tasks_react.jsonl; invented here, not contaminated). The ```tsx
         answer is typechecked strict with the task's test.tsx, then run
         under vitest 5 + jsdom + @testing-library/react in a separate warm
         sandbox (_work/react, pinned by env/react/). See grade_react.py.

WHICH CODE BLOCK IS GRADED

Fenced blocks are read with a line scanner (fences are flat text; their
CONTENTS are parsed with tree-sitter). Blocks tagged with a language the
domain does not accept are ignored; untagged blocks are accepted. Of the
rest:
  - one block: that block.
  - several: the LAST block that declares every `entry` symbol, since a
    reply that revises itself ends with the revision; failing that, all
    candidate blocks concatenated in order, so the failure surfaces at the
    compile/test stage with the real reason rather than as `extract`.
An unterminated final fence (a truncated reply) is still extracted.

USAGE

    python bench/domain/grade.py --setup          # install / warm toolchains
    python bench/domain/grade.py --build-index    # tasks/*/task.json -> tasks.jsonl
    python bench/domain/grade.py ID ANSWER_FILE   # grade one reply
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TASKS_JSONL = os.path.join(HERE, "tasks.jsonl")
TASK_DIR = os.path.join(HERE, "tasks")
ENV_DIR = os.path.join(HERE, "env")          # tracked: package.json, locks
LIB_DIR = os.path.join(HERE, "lib")          # tracked: shared test helpers
WORK = os.environ.get("DOMAIN_WORK", os.path.join(HERE, "_work"))
NODE_DIR = os.path.join(WORK, "node")
RUST_DIR = os.path.join(WORK, "rust")
CARGO_TARGET = os.path.join(RUST_DIR, "target")

# Mirrors mcp/deps.py STORE. Not imported: grading must not pull in the
# server's module graph.
PKG_STORE = os.environ.get("YAMADORI_PKG_DIR",
                           os.path.join(ROOT, "index", "packages"))
THREE_SRC = os.path.join(PKG_STORE, "_src", "three@0.185.1")

PINS = {"typescript": "7.0.2", "typegpu": "0.12.5"}
WASM_BINDGEN_VERSION = "0.2.128"

T_TSC = 120
T_NODE = 30
T_CARGO = 600          # a cold wasm-bindgen build; warm is a few seconds
T_TEST = 60

LANGS = {
    "ts": {"ts", "js", ""},
    "tsl": {"ts", "js", ""},
    "rust": {"rust", ""},
    "react": {"ts", "js", ""},   # ```tsx / ```jsx alias to ts / js
}
_ALIAS = {
    "ts": "ts", "typescript": "ts", "tsx": "ts", "mts": "ts", "cts": "ts",
    "js": "js", "javascript": "js", "jsx": "js", "mjs": "js", "cjs": "js",
    "node": "js", "rust": "rust", "rs": "rust", "": "",
}


# ---------------------------------------------------------------- tools ---

def _which(name: str, *fallbacks: str) -> str | None:
    for p in fallbacks:
        if p and os.path.isfile(p):
            return p
    return shutil.which(name)


def _home(*p: str) -> str:
    return os.path.join(os.path.expanduser("~"), *p)


NODE = os.environ.get("DOMAIN_NODE") or _which(
    "node", os.path.join(os.environ.get("APPDATA", ""), "nvm", "v24.21.0",
                         "node.exe"))
NPM = _which("npm", os.path.join(os.path.dirname(NODE or ""), "npm.cmd"))
CARGO = os.environ.get("DOMAIN_CARGO") or _which(
    "cargo", _home(".cargo", "bin", "cargo.exe"), _home(".cargo", "bin", "cargo"))
RUSTC = _which("rustc", _home(".cargo", "bin", "rustc.exe"),
               _home(".cargo", "bin", "rustc"))
WASM_BINDGEN = os.environ.get("DOMAIN_WASM_BINDGEN") or _which(
    "wasm-bindgen", _home(".cargo", "bin", "wasm-bindgen.exe"),
    _home(".cargo", "bin", "wasm-bindgen"))
TSC = os.path.join(NODE_DIR, "node_modules", "typescript", "bin", "tsc")


def run(cmd: list[str], cwd: str, timeout: float,
        env: dict | None = None) -> dict:
    """Run a command; kill the whole tree on timeout. Never raises."""
    t0 = time.perf_counter()
    e = dict(os.environ)
    if env:
        e.update(env)
    if NODE:  # npm / tsc shims look node up on PATH
        e["PATH"] = os.path.dirname(NODE) + os.pathsep + e.get("PATH", "")
    try:
        p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    except OSError as ex:
        return {"rc": None, "out": "", "err": f"could not start {cmd[0]}: {ex}",
                "secs": 0.0, "timeout": False, "spawn_failed": True}
    try:
        out, err = p.communicate(timeout=timeout)
        to = False
    except subprocess.TimeoutExpired:
        to = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)],
                           capture_output=True)
        else:
            p.kill()
        try:
            out, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
    return {"rc": p.returncode, "out": out.decode("utf-8", "replace"),
            "err": err.decode("utf-8", "replace"),
            "secs": time.perf_counter() - t0, "timeout": to,
            "spawn_failed": False}


# ----------------------------------------------------------- extraction ---

_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([^\s`{]*)")


def code_blocks(text: str) -> list[dict]:
    """Every fenced block: [{lang, code, closed}] in order."""
    blocks, cur = [], None
    for line in (text or "").splitlines():
        if cur is None:
            m = _OPEN.match(line)
            if m:
                tag = m.group(2).strip().lower().lstrip(".")
                cur = {"fence": m.group(1), "tag": tag,
                       "lang": _ALIAS.get(tag, tag), "lines": []}
            continue
        s = line.strip()
        f = cur["fence"]
        if s and set(s) == {f[0]} and len(s) >= len(f):
            blocks.append({"lang": cur["lang"], "tag": cur["tag"],
                           "code": "\n".join(cur["lines"]), "closed": True})
            cur = None
        else:
            cur["lines"].append(line)
    if cur is not None and cur["lines"]:
        blocks.append({"lang": cur["lang"], "tag": cur["tag"],
                       "code": "\n".join(cur["lines"]), "closed": False})
    return blocks


_parsers: dict = {}


def parse(code: str, lang: str):
    """Tree-sitter root node. Raises if the grammar pack is missing: the
    caller turns that into stage `error`, never into a model failure."""
    if lang not in _parsers:
        from tree_sitter_language_pack import get_parser
        _parsers[lang] = get_parser(lang)
    return _parsers[lang].parse(code.encode("utf-8")).root_node


def _txt(n) -> str:
    return n.text.decode("utf-8", "replace")


_TS_DECL = {"function_declaration", "generator_function_declaration",
            "class_declaration", "abstract_class_declaration",
            "interface_declaration", "type_alias_declaration",
            "enum_declaration", "function_signature"}
_RS_DECL = {"function_item", "struct_item", "enum_item", "union_item",
            "type_item", "const_item", "static_item", "trait_item",
            "mod_item", "function_signature_item"}


def declared_names(code: str, family: str) -> set[str]:
    """Top-level names a block declares (exported or not)."""
    names: set[str] = set()
    if family == "rust":
        root = parse(code, "rust")
        for n in root.children:
            if n.type in _RS_DECL:
                nm = n.child_by_field_name("name")
                if nm is not None:
                    names.add(_txt(nm))
            elif n.type == "foreign_mod_item":
                body = n.child_by_field_name("body")
                for c in (body.children if body else []):
                    nm = c.child_by_field_name("name")
                    if nm is not None:
                        names.add(_txt(nm))
        return names
    root = parse(code, "tsx" if family == "tsx" else "typescript")
    stack = list(root.children)
    while stack:
        n = stack.pop()
        if n.type == "export_statement":
            stack.extend(n.children)
        elif n.type in _TS_DECL:
            nm = n.child_by_field_name("name")
            if nm is not None:
                names.add(_txt(nm))
        elif n.type in ("lexical_declaration", "variable_declaration"):
            for c in n.children:
                if c.type == "variable_declarator":
                    nm = c.child_by_field_name("name")
                    if nm is not None and nm.type == "identifier":
                        names.add(_txt(nm))
    return names


def extract(answer_text: str, family: str, entry: list[str]) -> dict:
    """{ok, code, detail}. family is ts | tsl | rust."""
    blocks = code_blocks(answer_text)
    if not blocks:
        return {"ok": False, "detail": "no fenced code block in the reply"}
    accept = LANGS[family]
    cands = [b for b in blocks if b["lang"] in accept]
    if not cands:
        tags = sorted({b["tag"] or "(untagged)" for b in blocks})
        return {"ok": False, "detail": f"{len(blocks)} block(s) tagged "
                f"{', '.join(tags)}; none is {family} code"}
    cands = [b for b in cands if b["code"].strip()]
    if not cands:
        return {"ok": False, "detail": "the only matching code block is empty"}
    if len(cands) == 1:
        return {"ok": True, "code": cands[0]["code"],
                "detail": "1 block", "truncated": not cands[0]["closed"]}
    want = set(entry or [])
    pf = {"rust": "rust", "react": "tsx"}.get(family, "ts")
    if want:
        for i in range(len(cands) - 1, -1, -1):
            if want <= declared_names(cands[i]["code"], pf):
                return {"ok": True, "code": cands[i]["code"],
                        "detail": f"block {i + 1} of {len(cands)} "
                                  f"(last to declare {sorted(want)})",
                        "truncated": not cands[i]["closed"]}
    return {"ok": True, "code": "\n\n".join(b["code"] for b in cands),
            "detail": f"{len(cands)} blocks concatenated "
                      f"(no single block declares {sorted(want)})",
            "truncated": not cands[-1]["closed"]}


# ---------------------------------------------------------------- setup ---

_ENV_STATE: dict = {}


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def node_ready() -> str | None:
    """None if the node sandbox is usable, else the reason."""
    if not NODE:
        return "node not found (set DOMAIN_NODE)"
    for pkg, ver in PINS.items():
        got = _read_json(os.path.join(NODE_DIR, "node_modules", pkg,
                                      "package.json")).get("version")
        if got != ver:
            return f"{pkg} in {NODE_DIR} is {got!r}, need {ver} " \
                   f"(run grade.py --setup)"
    return None


def rust_ready(wasm: bool, bindgen: bool) -> str | None:
    key = ("rust", wasm, bindgen)
    if key in _ENV_STATE:
        return _ENV_STATE[key]
    why = None
    if not CARGO or not RUSTC:
        why = "cargo/rustc not found"
    else:
        r = run([RUSTC, "--version"], HERE, 30)
        m = re.search(r"rustc 1\.(\d+)", r["out"])
        if not m or int(m.group(1)) < 77:
            why = f"rustc too old for offset_of!: {r['out'].strip()}"
        elif wasm:
            sr = run([RUSTC, "--print", "sysroot"], HERE, 30)["out"].strip()
            if not os.path.isdir(os.path.join(sr, "lib", "rustlib",
                                              "wasm32-unknown-unknown")):
                why = ("wasm32-unknown-unknown target not installed "
                       "(rustup target add wasm32-unknown-unknown)")
    if not why and bindgen:
        if not WASM_BINDGEN:
            why = "wasm-bindgen CLI not found"
        else:
            v = run([WASM_BINDGEN, "--version"], HERE, 30)["out"]
            if WASM_BINDGEN_VERSION not in v:
                why = f"wasm-bindgen CLI is {v.strip()!r}, " \
                      f"need {WASM_BINDGEN_VERSION}"
    if not why and not os.path.isfile(os.path.join(ENV_DIR, "Cargo.lock")):
        why = "env/Cargo.lock missing (run grade.py --setup)"
    _ENV_STATE[key] = why
    return why


def setup(verbose: bool = True) -> int:
    """Install the pinned node packages and warm the cargo cache."""
    def say(*a):
        if verbose:
            print(*a, flush=True)
    os.makedirs(NODE_DIR, exist_ok=True)
    for fn in ("package.json", "package-lock.json"):
        src = os.path.join(ENV_DIR, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(NODE_DIR, fn))
    if node_ready():
        cmd = "ci" if os.path.isfile(os.path.join(NODE_DIR,
                                                  "package-lock.json")) else "install"
        say(f"npm {cmd} in {NODE_DIR}")
        r = run([NPM, cmd, "--no-audit", "--no-fund"], NODE_DIR, 600)
        say(r["out"][-800:], r["err"][-800:])
    why = node_ready()
    say("node:", why or "ready")

    # Rust: one template crate resolves the lockfile and warms both targets.
    tpl = os.path.join(RUST_DIR, "_template")
    os.makedirs(os.path.join(tpl, "src"), exist_ok=True)
    with open(os.path.join(tpl, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(_cargo_toml("t_template", {"wasm-bindgen":
                                           f"={WASM_BINDGEN_VERSION}"}))
    with open(os.path.join(tpl, "src", "lib.rs"), "w", encoding="utf-8") as f:
        f.write("use wasm_bindgen::prelude::*;\n#[wasm_bindgen]\n"
                "pub fn ping() -> u32 { 1 }\n")
    lock = os.path.join(ENV_DIR, "Cargo.lock")
    if os.path.isfile(lock):
        shutil.copy2(lock, os.path.join(tpl, "Cargo.lock"))
    env = {"CARGO_TARGET_DIR": CARGO_TARGET}
    for args in (["build", "--lib"],
                 ["build", "--lib", "--target", "wasm32-unknown-unknown"],
                 ["test", "--lib", "--no-run"]):
        r = run([CARGO] + args, tpl, T_CARGO, env)
        say("cargo", " ".join(args), "->", r["rc"], r["err"][-400:])
    if not os.path.isfile(lock) and os.path.isfile(os.path.join(tpl, "Cargo.lock")):
        os.makedirs(ENV_DIR, exist_ok=True)
        shutil.copy2(os.path.join(tpl, "Cargo.lock"), lock)
    _ENV_STATE.clear()
    rwhy = rust_ready(True, True)
    say("rust:", rwhy or "ready")
    return 0 if not why and not rwhy else 1


# --------------------------------------------------------------- common ---

def _res(passed: bool, stage: str, detail: str, **kw) -> dict:
    d = {"passed": passed, "stage": stage, "detail": detail[:4000]}
    d.update(kw)
    return d


def _p(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(HERE, rel)


def _fresh_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def _safe(task_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", task_id.lower())


def _tail(s: str, n: int = 1500) -> str:
    s = s.strip()
    return s if len(s) <= n else "..." + s[-n:]


# -------------------------------------------------------------------- ts ---

_TSC_ERR = re.compile(r"^(.*?)\((\d+),(\d+)\): error (TS\d+): (.*)$")


def _ts_options(g: dict, runtime: bool) -> dict:
    """The compiler options kind `ts` grades with. One function, so the
    public check (public_check) compiles the answer exactly as grading does."""
    opts = {
        "strict": True, "target": "es2023", "module": "nodenext",
        "moduleResolution": "nodenext", "skipLibCheck": True,
        "allowImportingTsExtensions": True, "types": ["node"],
        "lib": ["es2023"],
    }
    if runtime:
        opts.update({"outDir": "out", "rootDir": ".", "noEmitOnError": True,
                     "rewriteRelativeImportExtensions": True})
    else:
        opts["noEmit"] = True
    if g.get("webgpu"):
        opts["types"] = ["node", "@webgpu/types"]
        opts["lib"] = ["es2023", "dom"]
    opts.update(g.get("compilerOptions") or {})
    return opts


def grade_ts(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    why = node_ready()
    if why:
        return _res(False, "error", why, error=True)
    box = os.path.join(NODE_DIR, "tasks", _safe(task["id"]))
    _fresh_dir(box)
    with open(os.path.join(box, "solution.ts"), "w", encoding="utf-8") as f:
        f.write(code + "\n")
    shutil.copy2(_p(g["test"]), os.path.join(box, "test.ts"))
    shutil.copy2(os.path.join(LIB_DIR, "assert_types.ts"),
                 os.path.join(box, "assert_types.ts"))
    runtime = bool(g.get("runtime"))
    opts = _ts_options(g, runtime)
    with open(os.path.join(box, "tsconfig.json"), "w", encoding="utf-8") as f:
        json.dump({"compilerOptions": opts,
                   "files": ["solution.ts", "test.ts", "assert_types.ts"]}, f,
                  indent=1)
    r = run([NODE, TSC, "-p", "tsconfig.json"], box, T_TSC)
    timings["tsc"] = round(r["secs"], 3)
    if r["spawn_failed"]:
        return _res(False, "error", r["err"], error=True)
    if r["timeout"]:
        return _res(False, "compile", "tsc timed out", timeout=True)
    if r["rc"] != 0:
        errs = [_TSC_ERR.match(l) for l in (r["out"] + r["err"]).splitlines()]
        errs = [m for m in errs if m]
        if not errs:
            return _res(False, "error", "tsc failed with no located error: "
                        + _tail(r["out"] + r["err"]), error=True)
        in_sol = [m for m in errs if os.path.basename(m.group(1)) == "solution.ts"]
        in_test = [m for m in errs if os.path.basename(m.group(1)) == "test.ts"]
        other = [m for m in errs if m not in in_sol and m not in in_test]
        fmt = lambda ms: "\n".join(m.group(0) for m in ms[:12])  # noqa: E731
        if in_sol:
            return _res(False, "compile", fmt(in_sol))
        if in_test:
            return _res(False, "test", "type-level check failed:\n" + fmt(in_test))
        return _res(False, "error", "tsc error outside the answer and the "
                    "grader:\n" + fmt(other), error=True)
    if not runtime:
        return _res(True, "test", "typechecks, all type-level assertions hold")
    r = run([NODE, os.path.join("out", "test.js")], box, T_NODE)
    timings["node"] = round(r["secs"], 3)
    if r["spawn_failed"]:
        return _res(False, "error", r["err"], error=True)
    if r["timeout"]:
        return _res(False, "test", f"runtime test timed out after {T_NODE}s",
                    timeout=True)
    if r["rc"] != 0:
        return _res(False, "test", "runtime assertion failed: "
                    + _failure_line(r["err"] or r["out"]))
    return _res(True, "test", "typechecks and runtime assertions pass")


# ------------------------------------------------------------------ rust ---

def _cargo_toml(name: str, deps: dict) -> str:
    lines = ["[package]", f'name = "{name}"', 'version = "0.0.0"',
             'edition = "2021"', "", "[lib]", 'path = "src/lib.rs"',
             'crate-type = ["cdylib", "rlib"]', "", "[dependencies]"]
    for k, v in (deps or {}).items():
        lines.append(f'{k} = "{v}"')
    lines += ["", "[profile.dev]", "debug = false", ""]
    return "\n".join(lines)


def _cargo_errors(out: str) -> list[dict]:
    errs = []
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") != "compiler-message":
            continue
        msg = m.get("message") or {}
        if msg.get("level") != "error":
            continue
        if (msg.get("message") or "").startswith("aborting due to"):
            continue
        spans = [s for s in msg.get("spans") or [] if s.get("is_primary")]
        errs.append({"file": os.path.basename(spans[0]["file_name"]) if spans
                     else None, "text": msg.get("rendered")
                     or msg.get("message") or ""})
    return errs


def grade_rust(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    host = g.get("host_test")
    wasm = g.get("wasm_test")
    bindgen = bool(g.get("bindgen"))
    why = rust_ready(bool(wasm), bindgen)
    if why:
        return _res(False, "error", why, error=True)
    name = "t_" + _safe(task["id"])
    box = os.path.join(RUST_DIR, name)
    os.makedirs(os.path.join(box, "src"), exist_ok=True)
    for fn in os.listdir(os.path.join(box, "src")):
        os.remove(os.path.join(box, "src", fn))
    deps = dict(g.get("deps") or {})
    if bindgen:
        deps.setdefault("wasm-bindgen", f"={WASM_BINDGEN_VERSION}")
    with open(os.path.join(box, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(_cargo_toml(name, deps))
    shutil.copy2(os.path.join(ENV_DIR, "Cargo.lock"),
                 os.path.join(box, "Cargo.lock"))
    lib = code.rstrip() + "\n"
    if host:
        shutil.copy2(_p(host), os.path.join(box, "src", "__grader.rs"))
        lib += '\n#[cfg(test)]\n#[path = "__grader.rs"]\nmod __grader;\n'
    if g.get("wasm_support"):
        shutil.copy2(os.path.join(LIB_DIR, "wasm_support.rs"),
                     os.path.join(box, "src", "__support.rs"))
        lib += '\n#[path = "__support.rs"]\nmod __support;\n'
    with open(os.path.join(box, "src", "lib.rs"), "w", encoding="utf-8") as f:
        f.write(lib)
    env = {"CARGO_TARGET_DIR": CARGO_TARGET, "CARGO_TERM_COLOR": "never"}
    base = [CARGO]

    if host:
        r = run(base + ["test", "--lib", "--no-run", "--offline",
                        "--message-format=json"], box, T_CARGO, env)
        timings["cargo_build_host"] = round(r["secs"], 3)
        if r["spawn_failed"]:
            return _res(False, "error", r["err"], error=True)
        if r["timeout"]:
            return _res(False, "compile", "cargo timed out", timeout=True)
        if r["rc"] != 0:
            errs = _cargo_errors(r["out"])
            if not errs:
                return _res(False, "error", "cargo failed with no compiler "
                            "error: " + _tail(r["err"]), error=True)
            in_lib = [e for e in errs if e["file"] == "lib.rs"]
            if in_lib:
                return _res(False, "compile",
                            "\n".join(e["text"] for e in in_lib[:6]))
            return _res(False, "test", "grader does not compile against the "
                        "answer (wrong name, signature, visibility or ABI):\n"
                        + "\n".join(e["text"] for e in errs[:6]))
        r = run(base + ["test", "--lib", "--offline", "-q", "--",
                        "__grader::", "--test-threads=1"], box, T_TEST, env)
        timings["cargo_test"] = round(r["secs"], 3)
        if r["timeout"]:
            return _res(False, "test", "host tests timed out", timeout=True)
        if r["rc"] != 0:
            return _res(False, "test", "host test failed:\n"
                        + _tail(r["out"] + "\n" + r["err"]))
        ran = [int(n) for n in re.findall(r"test result: ok\. (\d+) passed",
                                          r["out"])]
        if not ran or sum(ran) == 0:
            # A check that cannot fail is not a check (PROTOCOL rule 16).
            return _res(False, "error", "grader ran zero host tests:\n"
                        + _tail(r["out"]), error=True)

    if wasm:
        r = run(base + ["build", "--lib", "--offline", "--target",
                        "wasm32-unknown-unknown", "--message-format=json"],
                box, T_CARGO, env)
        timings["cargo_build_wasm"] = round(r["secs"], 3)
        if r["spawn_failed"]:
            return _res(False, "error", r["err"], error=True)
        if r["timeout"]:
            return _res(False, "compile", "cargo (wasm) timed out", timeout=True)
        if r["rc"] != 0:
            errs = _cargo_errors(r["out"])
            if not errs:
                return _res(False, "error", "wasm build failed with no "
                            "compiler error: " + _tail(r["err"]), error=True)
            return _res(False, "compile", "wasm32 build failed:\n"
                        + "\n".join(e["text"] for e in errs[:6]))
        art = os.path.join(CARGO_TARGET, "wasm32-unknown-unknown", "debug",
                           name + ".wasm")
        if not os.path.isfile(art):
            return _res(False, "error", f"no wasm artifact at {art}", error=True)
        shutil.copy2(_p(wasm), os.path.join(box, "grader.mjs"))
        shutil.copy2(os.path.join(LIB_DIR, "wasm_helpers.mjs"),
                     os.path.join(box, "wasm_helpers.mjs"))
        target = art
        if bindgen:
            pkg = os.path.join(box, "pkg")
            _fresh_dir(pkg)
            r = run([WASM_BINDGEN, "--target", "nodejs", "--out-dir", pkg, art],
                    box, T_TEST)
            timings["wasm_bindgen"] = round(r["secs"], 3)
            if r["rc"] != 0:
                return _res(False, "test", "wasm-bindgen rejected the module:\n"
                            + _tail(r["err"]))
            with open(os.path.join(pkg, "package.json"), "w") as f:
                f.write('{"type": "commonjs"}\n')
            target = os.path.join(pkg, name + ".js")
        r = run([NODE, "grader.mjs", target], box, T_NODE)
        timings["node"] = round(r["secs"], 3)
        if r["spawn_failed"]:
            return _res(False, "error", r["err"], error=True)
        if r["timeout"]:
            return _res(False, "test", "wasm test timed out", timeout=True)
        if r["rc"] != 0:
            return _res(False, "test", "wasm test failed: "
                        + _failure_line(r["err"] or r["out"]))
    if not host and not wasm:
        return _res(False, "error", "task has neither host_test nor wasm_test",
                    error=True)
    return _res(True, "test", "compiles and every grader check passes")


# ------------------------------------------------------------------- tsl ---

_EXPORTS: dict = {}


def _module_exports(path: str, seen: set) -> set[str]:
    """Names an ES module exports, following `export * from` recursively."""
    path = os.path.normpath(path)
    if path in seen or not os.path.isfile(path):
        return set()
    seen.add(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    root = parse(src, "javascript")
    out: set[str] = set()
    for n in root.children:
        if n.type != "export_statement":
            continue
        source = n.child_by_field_name("source")
        clause = next((c for c in n.children if c.type == "export_clause"), None)
        decl = n.child_by_field_name("declaration")
        if clause is not None:
            for sp in clause.children:
                if sp.type != "export_specifier":
                    continue
                alias = sp.child_by_field_name("alias")
                nm = sp.child_by_field_name("name")
                out.add(_txt(alias if alias is not None else nm))
        elif source is not None:
            ns = next((c for c in n.children if c.type == "namespace_export"),
                      None)
            if ns is not None:
                ids = [c for c in ns.children if c.type in ("identifier",
                                                            "string")]
                if ids:
                    out.add(_txt(ids[-1]).strip("'\""))
            else:   # export * from '...'
                rel = _txt(source).strip("'\"")
                out |= _module_exports(os.path.join(os.path.dirname(path), rel),
                                       seen)
        elif decl is not None:
            if decl.type in ("lexical_declaration", "variable_declaration"):
                for c in decl.children:
                    if c.type == "variable_declarator":
                        out.add(_txt(c.child_by_field_name("name")))
            else:
                nm = decl.child_by_field_name("name")
                if nm is not None:
                    out.add(_txt(nm))
    return out


def three_exports() -> dict[str, set[str]]:
    """{specifier: names} for three@0.185.1, from the indexed source."""
    if _EXPORTS:
        return _EXPORTS
    src = os.path.join(THREE_SRC, "src")
    tsl_build = os.path.join(THREE_SRC, "build", "three.tsl.js")
    ex = {
        "three": _module_exports(os.path.join(src, "Three.js"), set()),
        "three/webgpu": _module_exports(os.path.join(src, "Three.WebGPU.js"),
                                        set()),
        "three/tsl": _module_exports(tsl_build, set()),
    }
    _EXPORTS.update(ex)
    return _EXPORTS


def tsl_facts(code: str) -> dict:
    """Parse once; everything the checks need."""
    root = parse(code, "typescript")
    facts = {"syntax": [], "imports": [], "ident": set(), "prop": set(),
             "call": set()}
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            facts["syntax"].append(f"line {n.start_point[0] + 1}: "
                                   f"{'missing ' + n.type if n.is_missing else 'unparseable'}")
        t = n.type
        if t == "import_statement":
            srcn = n.child_by_field_name("source")
            spec = _txt(srcn).strip("'\"") if srcn is not None else ""
            for c in n.children:
                if c.type != "import_clause":
                    continue
                for cc in c.children:
                    if cc.type == "named_imports":
                        for sp in cc.children:
                            if sp.type == "import_specifier":
                                nm = sp.child_by_field_name("name")
                                facts["imports"].append((spec, _txt(nm)))
        elif t in ("identifier", "shorthand_property_identifier",
                   "shorthand_property_identifier_pattern"):
            facts["ident"].add(_txt(n))
        elif t == "property_identifier":
            facts["prop"].add(_txt(n))
        elif t == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None:
                if fn.type == "identifier":
                    facts["call"].add(_txt(fn))
                elif fn.type == "member_expression":
                    p = fn.child_by_field_name("property")
                    if p is not None:
                        facts["call"].add(_txt(p))
        stack.extend(n.children)
    return facts


def grade_tsl(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    if not os.path.isdir(THREE_SRC):
        return _res(False, "error", f"three@0.185.1 source not held at "
                    f"{THREE_SRC}", error=True)
    ex = three_exports()
    if min(len(v) for v in ex.values()) < 50:
        return _res(False, "error", "three@0.185.1 export lists look empty: "
                    + str({k: len(v) for k, v in ex.items()}), error=True)
    t0 = time.perf_counter()
    f = tsl_facts(code)
    timings["parse"] = round(time.perf_counter() - t0, 4)
    if f["syntax"]:
        return _res(False, "compile", "syntax error: " + "; ".join(f["syntax"][:5]))
    imported = {nm for _, nm in f["imports"]}
    bad = [f"{nm} from '{spec}'" for spec, nm in f["imports"]
           if spec in ex and nm not in ex[spec]]
    if bad and g.get("check_imports", True):
        return _res(False, "compile", "imports that three@0.185.1 does not "
                    "export: " + ", ".join(bad))
    used = f["ident"] | f["prop"] | imported
    hits = sorted(n for n in g.get("forbid", []) if n in used)
    if hits:
        return _res(False, "test", "uses pre-r185 / deprecated name(s): "
                    + ", ".join(hits))
    missing = []
    for req in g.get("require", []):
        nm, how = req["name"], req.get("as", "any")
        where = {"import": imported, "call": f["call"], "property": f["prop"],
                 "ident": f["ident"], "any": used}[how]
        if nm not in where:
            missing.append(f"{nm} (as {how})")
        frm = req.get("from")
        if frm and (frm, nm) not in f["imports"]:
            missing.append(f"{nm} imported from '{frm}'")
    if missing:
        return _res(False, "test", "missing the r185 API: " + ", ".join(missing))
    return _res(True, "test", "uses the r185 names and no deprecated ones")


# -------------------------------------------------------------- tc_tests ---

TC_DIR = os.path.join(HERE, "tasks_type_challenges")
TC_UTILS = os.path.join(TC_DIR, "utils")
TC_JSONL = os.path.join(HERE, "tasks_type_challenges.jsonl")


def _tc_decl_names(n) -> list[str]:
    """Names one top-level TS statement declares (incl. `declare ...` and
    `namespace X {}`, which tree-sitter wraps in an expression_statement)."""
    if n.type == "ambient_declaration":
        return [x for c in n.children for x in _tc_decl_names(c)]
    if n.type == "expression_statement":
        return [x for c in n.children if c.type == "internal_module"
                for x in _tc_decl_names(c)]
    if n.type in _TS_DECL or n.type in ("internal_module", "module"):
        nm = n.child_by_field_name("name")
        return [_txt(nm)] if nm is not None else []
    if n.type in ("lexical_declaration", "variable_declaration"):
        out = []
        for c in n.children:
            if c.type == "variable_declarator":
                nm = c.child_by_field_name("name")
                if nm is not None and nm.type == "identifier":
                    out.append(_txt(nm))
        return out
    return []


def tc_names(code: str) -> dict:
    """{declared, exported, module} for the top level of a TS source."""
    root = parse(code, "typescript")
    declared: set[str] = set()
    exported: set[str] = set()
    module = False
    for n in root.children:
        if n.type == "import_statement":
            module = True
        elif n.type == "export_statement":
            module = True
            decl = n.child_by_field_name("declaration")
            if decl is not None:
                names = _tc_decl_names(decl)
                declared.update(names)
                exported.update(names)
            for c in n.children:
                if c.type == "export_clause":
                    for sp in c.children:
                        if sp.type == "export_specifier":
                            al = sp.child_by_field_name("alias")
                            nm = sp.child_by_field_name("name")
                            exported.add(_txt(al if al is not None else nm))
        else:
            declared.update(_tc_decl_names(n))
    return {"declared": declared, "exported": exported, "module": module}


def _tc_options(g: dict) -> dict:
    opts = {"strict": True, "noEmit": True, "target": "esnext",
            "lib": ["esnext"], "module": "esnext",
            "moduleResolution": "bundler", "skipLibCheck": True, "types": []}
    opts.update(g.get("compilerOptions") or {})
    return opts


def _tc_box(box: str) -> None:
    """A fresh sandbox with @type-challenges/utils vendored into it."""
    _fresh_dir(box)
    util_dst = os.path.join(box, "node_modules", "@type-challenges", "utils")
    os.makedirs(util_dst, exist_ok=True)
    for fn in ("index.d.ts", "package.json"):
        shutil.copy2(os.path.join(TC_UTILS, fn), os.path.join(util_dst, fn))


def grade_tc(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    why = node_ready()
    if why:
        return _res(False, "error", why, error=True)
    if not os.path.isfile(os.path.join(TC_UTILS, "index.d.ts")):
        return _res(False, "error", f"vendored @type-challenges/utils missing "
                    f"at {TC_UTILS}", error=True)
    box = os.path.join(NODE_DIR, "tasks", _safe(task["id"]))
    _tc_box(box)
    with open(_p(g["test"]), encoding="utf-8") as f:
        test_src = f.read()
    sol = code.rstrip() + "\n"
    names = tc_names(code)
    if names["module"]:
        # The upstream playground treats the answer as a global script. An
        # answer written as a module gets the entry names it declares
        # exported and imported, so `export type X` scores like `type X`.
        entry = [e for e in (g.get("entry") or []) if e in names["declared"]]
        add = [e for e in entry if e not in names["exported"]]
        if add:
            sol += "\nexport { " + ", ".join(add) + " }\n"
        if entry:
            test_src = ("import { " + ", ".join(entry)
                        + " } from './solution'\n" + test_src)
    with open(os.path.join(box, "solution.ts"), "w", encoding="utf-8") as f:
        f.write(sol)
    with open(os.path.join(box, "test-cases.ts"), "w", encoding="utf-8") as f:
        f.write(test_src)
    opts = _tc_options(g)
    with open(os.path.join(box, "tsconfig.json"), "w", encoding="utf-8") as f:
        json.dump({"compilerOptions": opts,
                   "files": ["solution.ts", "test-cases.ts"]}, f, indent=1)
    r = run([NODE, TSC, "-p", "tsconfig.json"], box, T_TSC)
    timings["tsc"] = round(r["secs"], 3)
    if r["spawn_failed"]:
        return _res(False, "error", r["err"], error=True)
    if r["timeout"]:
        return _res(False, "compile", "tsc timed out", timeout=True)
    if r["rc"] == 0:
        return _res(True, "test", "typechecks, every upstream test case holds")
    errs = [_TSC_ERR.match(l) for l in (r["out"] + r["err"]).splitlines()]
    errs = [m for m in errs if m]
    if not errs:
        return _res(False, "error", "tsc failed with no located error: "
                    + _tail(r["out"] + r["err"]), error=True)
    base = lambda m: os.path.basename(m.group(1))                # noqa: E731
    in_sol = [m for m in errs if base(m) == "solution.ts"]
    in_test = [m for m in errs if base(m) == "test-cases.ts"]
    other = [m for m in errs if m not in in_sol and m not in in_test]
    fmt = lambda ms: "\n".join(m.group(0) for m in ms[:12])      # noqa: E731
    if in_sol:
        return _res(False, "compile", fmt(in_sol))
    if in_test:
        return _res(False, "test", "type-level test case failed:\n"
                    + fmt(in_test))
    return _res(False, "error", "tsc error outside the answer and the "
                "test cases:\n" + fmt(other), error=True)


# ----------------------------------------------------------------- react ---

def _grade_react(task: dict, code: str, timings: dict) -> dict:
    """Kind `react` (tasks/react-*, tasks_react.jsonl): see grade_react.py."""
    import grade_react
    return grade_react.grade_react(task, code, timings)


# ----------------------------------------------------------------- entry ---

FAMILY = {"ts": "ts", "rust": "rust", "tsl": "tsl", "tc_tests": "ts",
          "react": "react"}
GRADERS = {"ts": grade_ts, "rust": grade_rust, "tsl": grade_tsl,
           "tc_tests": grade_tc, "react": _grade_react}


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class _TaskLock:
    """Cross-process lock per task sandbox.

    Each task grades in a fixed directory (and, for Rust, a fixed crate name
    whose artifact lands in the shared target dir). Two arms of a benchmark
    grading the same task at once would otherwise overwrite each other's
    answer mid-run and score one reply with the other's code.
    """

    STALE = 1800.0

    def __init__(self, task_id: str):
        os.makedirs(os.path.join(WORK, "locks"), exist_ok=True)
        self.path = os.path.join(WORK, "locks", _safe(task_id) + ".lock")

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    with open(self.path, encoding="ascii") as f:
                        holder = int(f.read().strip() or 0)
                    dead = holder and not _pid_alive(holder)
                    if dead or time.time() - os.path.getmtime(self.path) \
                            > self.STALE:
                        os.remove(self.path)   # holder died mid-grade
                        continue
                except (OSError, ValueError):
                    continue
                if time.time() - t0 > self.STALE:
                    raise TimeoutError(f"task lock {self.path} held too long")
                time.sleep(0.05)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def grade(task: dict, answer_text: str) -> dict:
    """Grade one model reply against one task. Never raises."""
    t0 = time.perf_counter()
    timings: dict = {}
    try:
        kind = (task.get("grader") or {}).get("kind")
        if kind not in GRADERS:
            out = _res(False, "error", f"unknown grader kind {kind!r}",
                       error=True)
        else:
            try:
                ex = extract(answer_text, FAMILY[kind],
                             task["grader"].get("entry") or [])
            except Exception as e:                              # noqa: BLE001
                ex = {"ok": False, "crash": f"extractor crashed: {e!r}"}
            if ex.get("crash"):
                out = _res(False, "error", ex["crash"], error=True)
            elif not ex["ok"]:
                out = _res(False, "extract", ex["detail"])
            else:
                with _TaskLock(task["id"]):
                    out = GRADERS[kind](task, ex["code"], timings)
                out["extracted"] = ex["detail"]
                if ex.get("truncated"):
                    out["truncated"] = True
    except Exception:                                           # noqa: BLE001
        out = _res(False, "error", "grader crashed:\n"
                   + traceback.format_exc()[-1500:], error=True)
    out.setdefault("error", False)
    out["seconds"] = round(time.perf_counter() - t0, 3)
    out["timings"] = timings
    return out


# ---------------------------------------------------------- public check ---
#
# What a self-check arm's `check_solution` tool runs (bench/domain/run.py,
# arms S*). It is the grader's extract + compile stages ONLY, against the same
# pinned toolchains and packages, with NOTHING the grader holds back: no
# test.ts / test-cases.ts / test.tsx, no host_test module, no wasm_test, no
# TSL require/forbid lists. The sandbox it compiles in never contains a
# hidden file, so no error it returns can quote one. test_public_check.py
# proves it both ways: every reference passes, every wrong_compile fails, and
# every wrong_test answer -- wrong only by the hidden tests -- PASSES.

PUBLIC_MAX_LINES = 30
PUBLIC_MAX_CHARS = 4000


def _public(ok: bool, stage: str, tool: str, errors: list[str] | None = None,
            **kw) -> dict:
    errs = [e.rstrip() for e in (errors or []) if e.strip()]
    lines: list[str] = []
    for e in errs:
        lines += e.splitlines()
    shown, size = [], 0
    for ln in lines:
        if len(shown) >= PUBLIC_MAX_LINES or size + len(ln) > PUBLIC_MAX_CHARS:
            break
        shown.append(ln)
        size += len(ln) + 1
    d = {"ok": ok, "stage": stage, "tool": tool, "n_errors": len(errs),
         "errors": shown, "truncated": len(shown) < len(lines),
         "checker_error": False}
    d.update(kw)
    return d


def _checker_error(tool: str, why: str) -> dict:
    """The CHECKER could not run. Never the model's fault (PROTOCOL rule 3)."""
    return _public(False, "error", tool, [why], checker_error=True)


def _scrub(text: str, *dirs: str) -> str:
    """Sandbox paths out of compiler output: the answer's file name stays."""
    dirs = dirs + (os.path.expanduser("~"),)
    for d in dirs:
        for form in (d, d.replace("\\", "/")):
            text = text.replace(form + os.sep, "").replace(form + "/", "")
            text = text.replace(form, ".")
    return text


def _tsc_public(box: str, opts: dict, files: list[str], tool: str,
                timings: dict, tsc: str | None = None) -> dict:
    with open(os.path.join(box, "tsconfig.json"), "w", encoding="utf-8") as f:
        json.dump({"compilerOptions": opts, "files": files}, f, indent=1)
    r = run([NODE, tsc or TSC, "-p", "tsconfig.json"], box, T_TSC)
    timings["tsc"] = round(r["secs"], 3)
    if r["spawn_failed"]:
        return _checker_error(tool, r["err"])
    if r["timeout"]:
        return _public(False, "compile", tool, [f"tsc timed out after {T_TSC}s"])
    if r["rc"] == 0:
        return _public(True, "compile", tool)
    # box, then its sandbox root (node_modules/... stays readable), then WORK
    out = _scrub(r["out"] + r["err"], box, os.path.dirname(os.path.dirname(box)),
                 WORK)
    errs: list[str] = []
    for ln in out.splitlines():
        if _TSC_ERR.match(ln):
            errs.append(ln)
        elif errs and ln.startswith(" "):          # a continuation line
            errs[-1] += "\n" + ln
    if not errs:
        return _checker_error(tool, "tsc failed with no located error: "
                              + _tail(out, 600))
    return _public(False, "compile", tool, errs)


def _public_ts(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    tool = f"tsc --strict (TypeScript {PINS['typescript']}" + (
        f", typegpu@{PINS['typegpu']}, @webgpu/types" if g.get("webgpu") else "") + ")"
    why = node_ready()
    if why:
        return _checker_error(tool, why)
    box = os.path.join(NODE_DIR, "public", _safe(task["id"]))
    _fresh_dir(box)
    with open(os.path.join(box, "solution.ts"), "w", encoding="utf-8") as f:
        f.write(code + "\n")
    opts = _ts_options(g, runtime=False)
    return _tsc_public(box, opts, ["solution.ts"], tool, timings)


def _public_tc(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    tool = f"tsc --strict (TypeScript {PINS['typescript']})"
    why = node_ready()
    if why:
        return _checker_error(tool, why)
    if not os.path.isfile(os.path.join(TC_UTILS, "index.d.ts")):
        return _checker_error(tool, f"vendored @type-challenges/utils missing at "
                              f"{TC_UTILS}")
    # The template is in the prompt; its declared names are public. The test
    # cases (which of them are used, and how) are not.
    want: list[str] = []
    if g.get("template") and os.path.isfile(_p(g["template"])):
        with open(_p(g["template"]), encoding="utf-8") as f:
            want = sorted(tc_names(f.read())["declared"])
    have = tc_names(code)["declared"]
    missing = [w for w in want if w not in have]
    box = os.path.join(NODE_DIR, "public", _safe(task["id"]))
    _tc_box(box)
    with open(os.path.join(box, "solution.ts"), "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")
    out = _tsc_public(box, _tc_options(g), ["solution.ts"], tool, timings)
    if missing and not out["checker_error"]:
        extra = [f"solution.ts: the template declares `{m}`; your code does not "
                 f"declare it" for m in missing]
        out = _public(False, "compile", tool,
                      extra + ([] if out["ok"] else out["errors"]))
    return out


def _public_react(task: dict, code: str, timings: dict) -> dict:
    import grade_react
    tool = (f"tsc --strict (TypeScript {grade_react.PINS['typescript']}, "
            f"react@{grade_react.PINS['react']}, @types/react@"
            f"{grade_react.PINS['@types/react']})")
    why = grade_react.ready()
    if why:
        return _checker_error(tool, why)
    box = os.path.join(grade_react.SANDBOX, "public", _safe(task["id"]))
    _fresh_dir(box)
    with open(os.path.join(box, "solution.tsx"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(code.rstrip() + "\n")
    opts = dict(grade_react.TSCONFIG["compilerOptions"])
    opts.update(task["grader"].get("compilerOptions") or {})
    return _tsc_public(box, opts, ["solution.tsx"], tool, timings,
                       tsc=grade_react.TSC)


def _public_tsl(task: dict, code: str, timings: dict) -> dict:
    tool = "tree-sitter parse + three@0.185.1 export check"
    if not os.path.isdir(THREE_SRC):
        return _checker_error(tool, f"three@0.185.1 source not held at {THREE_SRC}")
    ex = three_exports()
    if min(len(v) for v in ex.values()) < 50:
        return _checker_error(tool, "three@0.185.1 export lists look empty")
    t0 = time.perf_counter()
    f = tsl_facts(code)
    timings["parse"] = round(time.perf_counter() - t0, 4)
    if f["syntax"]:
        return _public(False, "compile", tool,
                       ["syntax error: " + s for s in f["syntax"]])
    bad = [f"`{nm}` is not exported by '{spec}' in three@0.185.1"
           for spec, nm in f["imports"] if spec in ex and nm not in ex[spec]]
    if bad and task["grader"].get("check_imports", True):
        return _public(False, "compile", tool, bad)
    return _public(True, "compile", tool)


def _public_rust(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    host, wasm, bindgen = g.get("host_test"), g.get("wasm_test"), bool(g.get("bindgen"))
    targets = (["host"] if host else []) + (["wasm32-unknown-unknown"] if wasm else [])
    tool = "cargo " + " + ".join(
        "test --no-run (host)" if t == "host" else f"build --target {t}"
        for t in targets) + (f", wasm-bindgen {WASM_BINDGEN_VERSION}" if bindgen else "")
    why = rust_ready(bool(wasm), bindgen)
    if why:
        return _checker_error(tool, why)
    if not targets:
        return _checker_error(tool, "task has neither host_test nor wasm_test")
    name = "t_" + _safe(task["id"])
    box = os.path.join(RUST_DIR, name)
    os.makedirs(os.path.join(box, "src"), exist_ok=True)
    for fn in os.listdir(os.path.join(box, "src")):
        os.remove(os.path.join(box, "src", fn))       # no grader module left
    deps = dict(g.get("deps") or {})
    if bindgen:
        deps.setdefault("wasm-bindgen", f"={WASM_BINDGEN_VERSION}")
    with open(os.path.join(box, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(_cargo_toml(name, deps))
    shutil.copy2(os.path.join(ENV_DIR, "Cargo.lock"), os.path.join(box, "Cargo.lock"))
    lib = code.rstrip() + "\n"
    if g.get("wasm_support"):
        # Part of the build the answer is graded in (an allocator), not a test.
        shutil.copy2(os.path.join(LIB_DIR, "wasm_support.rs"),
                     os.path.join(box, "src", "__support.rs"))
        lib += '\n#[path = "__support.rs"]\nmod __support;\n'
    with open(os.path.join(box, "src", "lib.rs"), "w", encoding="utf-8") as f:
        f.write(lib)
    env = {"CARGO_TARGET_DIR": CARGO_TARGET, "CARGO_TERM_COLOR": "never"}
    for t in targets:
        cmd = ([CARGO, "test", "--lib", "--no-run"] if t == "host" else
               [CARGO, "build", "--lib", "--target", t])
        r = run(cmd + ["--offline", "--message-format=json"], box, T_CARGO, env)
        timings[f"cargo_{'host' if t == 'host' else 'wasm'}"] = round(r["secs"], 3)
        if r["spawn_failed"]:
            return _checker_error(tool, r["err"])
        if r["timeout"]:
            return _public(False, "compile", tool, ["cargo timed out"])
        if r["rc"] != 0:
            errs = _cargo_errors(r["out"])
            if not errs:
                return _checker_error(tool, "cargo failed with no compiler error: "
                                      + _tail(_scrub(r["err"], box, CARGO_TARGET), 600))
            return _public(False, "compile", tool,
                           [_scrub(e["text"], box, CARGO_TARGET) for e in errs])
    return _public(True, "compile", tool)


PUBLIC = {"ts": _public_ts, "rust": _public_rust, "tsl": _public_tsl,
          "tc_tests": _public_tc, "react": _public_react}


def public_check(task: dict, code: str) -> dict:
    """Does this code compile, with nothing hidden in the sandbox?

    `code` is what the model passed to `check_solution`: raw source, or a
    reply holding fenced blocks (then extracted exactly as grade() would).
    Returns {ok, stage (extract|compile|error), tool, n_errors, errors
    (first PUBLIC_MAX_LINES lines), truncated, checker_error, seconds}.
    `checker_error` means the harness failed, never the code. Never raises.
    """
    t0 = time.perf_counter()
    timings: dict = {}
    kind = (task.get("grader") or {}).get("kind")
    try:
        if kind not in PUBLIC:
            out = _checker_error("?", f"unknown grader kind {kind!r}")
        else:
            src = code or ""
            if code_blocks(src):
                ex = extract(src, FAMILY[kind], task["grader"].get("entry") or [])
                src = ex.get("code") if ex["ok"] else None
                detail = ex.get("detail")
            else:
                detail = "no code" if not src.strip() else None
                src = src if src.strip() else None
            if src is None:
                out = _public(False, "extract", "extract",
                              [f"no usable code in `code`: {detail}"])
            else:
                with _TaskLock(task["id"]):
                    out = PUBLIC[kind](task, src, timings)
    except Exception:                                           # noqa: BLE001
        out = _checker_error("?", "public check crashed:\n"
                             + traceback.format_exc()[-800:])
    out["seconds"] = round(time.perf_counter() - t0, 3)
    out["timings"] = timings
    return out


def load_tasks(path: str = TASKS_JSONL) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_task_dirs() -> list[dict]:
    """The same rows, read from tasks/*/task.json (the authoring source)."""
    rows = [_read_json(p) for p in
            sorted(glob.glob(os.path.join(TASK_DIR, "*", "task.json")))]
    rows = [t for t in rows if t.get("domain") != "react"]  # tasks_react.jsonl
    return sorted(rows, key=lambda t: (_ORDER.get(t.get("domain"), 9),
                                       t.get("id", "")))


def _failure_line(text: str) -> str:
    """Put the assertion message first; node buries it under a stack frame."""
    for line in text.splitlines():
        if (re.search(r"(AssertionError|GraderFailure|Error)\b.*?:", line)
                and "triggerUncaughtException" not in line):
            return line.strip() + "\n" + _tail(text)
    return _tail(text)


_ORDER = {"typescript": 0, "typegpu": 1, "rust_wasm": 2, "three_tsl": 3}


def build_index() -> int:
    """Collect tasks/*/task.json into tasks.jsonl, validating the schema."""
    rows, problems = [], []
    need = ("id", "domain", "prompt", "grader", "reference", "contaminated",
            "needs_retrieval", "difficulty")
    for p in sorted(glob.glob(os.path.join(TASK_DIR, "*", "task.json"))):
        t = _read_json(p)
        if t.get("domain") == "react":
            continue    # indexed by grade_react.py --build-index
        miss = [k for k in need if k not in t]
        if miss:
            problems.append(f"{p}: missing {miss}")
            continue
        if not os.path.isfile(_p(t["reference"])):
            problems.append(f"{p}: reference {t['reference']} not found")
        if t["domain"] == "three_tsl" and not t["contaminated"]:
            problems.append(f"{p}: three_tsl must be contaminated")
        rows.append({k: t[k] for k in need} | {k: v for k, v in t.items()
                                                if k not in need})
    rows.sort(key=lambda t: (_ORDER.get(t["domain"], 9), t["id"]))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate task ids")
    with open(TASKS_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} tasks -> {TASKS_JSONL}")
    for pr in problems:
        print("  PROBLEM", pr)
    return 1 if problems else 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["--setup"]:
        return setup()
    if a[:1] == ["--build-index"]:
        return build_index()
    if len(a) != 2:
        print(__doc__)
        return 2
    pool = load_tasks()
    if os.path.isfile(TC_JSONL):
        pool += load_tasks(TC_JSONL)
    if os.path.isfile(os.path.join(HERE, "tasks_react.jsonl")):
        pool += load_tasks(os.path.join(HERE, "tasks_react.jsonl"))
    task = next((t for t in pool if t["id"] == a[0]), None)
    if task is None:
        print(f"no task {a[0]!r}")
        return 2
    with open(a[1], encoding="utf-8") as f:
        print(json.dumps(grade(task, f.read()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
