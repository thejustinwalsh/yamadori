#!/usr/bin/env python
"""A style / modernness score for TypeScript and React answers. SEPARATE from
pass/fail: it is recorded beside the grade and never changes an outcome.

Deterministic and CPU-only, computed at grade time on the extracted answer:

  lint     ESLint 10.11.0 with typescript-eslint 8.70.1 `strict-type-checked`
           (type-aware: a TS program is built for the answer alone) plus
           eslint-plugin-react-hooks 7.1.1 `recommended`. typescript-eslint
           needs the JS TypeScript API, which TypeScript 7 (the Go port the
           graders compile with) does not ship, so the linter's program uses
           TypeScript 6.0.3. It lints; it never decides whether code compiles.
           The answer is compiled with the SAME compiler options its grader
           uses. Pinned in env/lint/package.json (+ lock), installed only
           under _work/lint (`python bench/domain/grade_style.py --setup`).
  react19  a small tree-sitter rule set (PROTOCOL rule 8: parsed, not
           pattern-matched) for React 19 idioms:
             flags    forwardRef (a plain `ref` prop since 19), class_component
                      (error boundaries exempt: there is no function
                      equivalent), defaultProps (on a function component),
                      legacy_render (ReactDOM.render / hydrate from
                      'react-dom'), legacy_dom_api (findDOMNode,
                      unmountComponentAtNode), effect_derived_state (an effect
                      whose whole body is one setState call -- derived state
                      belongs in render; detectable cases only)
             credits  useActionState, useOptimistic, use() -- counted only
                      where the task's prompt names them (`called_for`)

Kinds scored: ts (typescript, typegpu), react, tc_tests. Others are skipped.
Result: {lint_errors, lint_warnings, lint_rules{rule: n}, modern_flags[],
modern_credits[], modern_used[], called_for[], tool, seconds, style_error,
skipped}. `style_error` means the SCORER failed, never the answer.

    python bench/domain/grade_style.py --setup
    python bench/domain/grade_style.py ID ANSWER_FILE
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import grade as G  # noqa: E402

ENV_DIR = os.path.join(G.ENV_DIR, "lint")
SANDBOX = os.path.join(G.WORK, "lint")
NODE_MODULES = os.path.join(SANDBOX, "node_modules")
ESLINT = os.path.join(NODE_MODULES, "eslint", "bin", "eslint.js")
PINS = {"eslint": "10.11.0", "typescript-eslint": "8.70.1",
        "eslint-plugin-react-hooks": "7.1.1", "typescript": "6.0.3",
        "@types/react": "19.2.18", "typegpu": "0.12.5"}
TOOL = ("eslint 10.11.0 + typescript-eslint 8.70.1 strict-type-checked + "
        "eslint-plugin-react-hooks 7.1.1 recommended (TypeScript 6.0.3) + "
        "react19 idiom rules")
T_LINT = 180
KINDS = ("ts", "react", "tc_tests")

ESLINT_CONFIG = """\
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';

export default [
  ...tseslint.configs.strictTypeChecked,
  reactHooks.configs.flat.recommended,
  {
    languageOptions: {
      parserOptions: { project: './tsconfig.json', tsconfigRootDir: import.meta.dirname },
    },
  },
];
"""


def ready() -> str | None:
    if not G.NODE:
        return "node not found (set DOMAIN_NODE)"
    for pkg, ver in PINS.items():
        got = G._read_json(os.path.join(NODE_MODULES, pkg, "package.json")).get("version")
        if got != ver:
            return (f"{pkg} in {SANDBOX} is {got!r}, need {ver} "
                    f"(run grade_style.py --setup)")
    return None


def setup(verbose: bool = True) -> int:
    os.makedirs(SANDBOX, exist_ok=True)
    for fn in ("package.json", "package-lock.json"):
        src = os.path.join(ENV_DIR, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(SANDBOX, fn))
    if ready():
        cmd = "ci" if os.path.isfile(os.path.join(SANDBOX, "package-lock.json")) else "install"
        r = G.run([G.NPM, cmd, "--no-audit", "--no-fund"], SANDBOX, 600)
        if verbose:
            print(r["out"][-800:], r["err"][-800:])
    why = ready()
    if verbose:
        print("lint sandbox:", why or "ready")
    return 1 if why else 0


# ------------------------------------------------------------ react19 -----
_CREDITS = ("useActionState", "useOptimistic", "use")
_SIDE = re.compile(r"\b(fetch|await|setTimeout|setInterval|subscribe|"
                   r"addEventListener|requestAnimationFrame|then)\b")


def _callee(n) -> str:
    fn = n.child_by_field_name("function")
    return G._txt(fn) if fn is not None else ""


def _effect_derived(call) -> bool:
    """useEffect(() => { setX(expr) }, deps): the whole body is one setState."""
    args = call.child_by_field_name("arguments")
    fns = [c for c in (args.children if args else [])
           if c.type in ("arrow_function", "function_expression", "function")]
    if not fns:
        return False
    body = fns[0].child_by_field_name("body")
    if body is None:
        return False
    if body.type == "statement_block":
        stmts = [c for c in body.children if c.is_named and c.type != "comment"]
        if len(stmts) != 1 or stmts[0].type != "expression_statement":
            return False
        expr = next((c for c in stmts[0].children if c.is_named), None)
    else:
        expr = body
    if expr is None or expr.type != "call_expression":
        return False
    name = _callee(expr)
    return bool(re.fullmatch(r"set[A-Z]\w*", name)) and not _SIDE.search(G._txt(expr))


def react19(code: str, prompt: str) -> dict:
    root = G.parse(code, "tsx")
    flags: list[str] = []
    used: set[str] = set()
    dom_imports: set[str] = set()
    react_names: set[str] = {"React"}
    classes: set[str] = set()
    stack = [root]
    nodes = []
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children)
    for n in nodes:                                  # imports first
        if n.type == "import_statement":
            src = n.child_by_field_name("source")
            spec = G._txt(src).strip("'\"") if src is not None else ""
            for c in n.children:
                if c.type != "import_clause":
                    continue
                for cc in c.children:
                    if cc.type == "named_imports":
                        for sp in cc.children:
                            if sp.type == "import_specifier":
                                nm = G._txt(sp.child_by_field_name("name"))
                                if spec == "react-dom":
                                    dom_imports.add(nm)
                    elif cc.type in ("identifier", "namespace_import") and spec == "react-dom":
                        dom_imports.add("ReactDOM")
                    elif cc.type == "identifier" and spec == "react":
                        react_names.add(G._txt(cc))
    for n in nodes:
        t = n.type
        if t == "call_expression":
            name = _callee(n)
            base = name.split(".")[-1]
            obj = name.rsplit(".", 1)[0] if "." in name else ""
            if base == "forwardRef" and (not obj or obj in react_names):
                flags.append("forwardRef")
            if base in ("render", "hydrate") and (
                    (obj and obj in ("ReactDOM",) and "ReactDOM" in dom_imports)
                    or (not obj and base in dom_imports)):
                flags.append("legacy_render")
            if base in ("findDOMNode", "unmountComponentAtNode") and (
                    obj == "ReactDOM" or base in dom_imports):
                flags.append("legacy_dom_api")
            if base in ("useEffect", "useLayoutEffect") and _effect_derived(n):
                flags.append("effect_derived_state")
            if base in _CREDITS and (not obj or obj in react_names):
                used.add(base)
        elif t in ("class_declaration", "class", "abstract_class_declaration"):
            her = next((c for c in n.children if c.type == "class_heritage"), None)
            if her is not None and re.search(r"\bextends\s+(\w+\.)?(Pure)?Component\b",
                                             G._txt(her)):
                nm = n.child_by_field_name("name")
                if nm is not None:
                    classes.add(G._txt(nm))
                body = G._txt(n)
                if not re.search(r"\b(componentDidCatch|getDerivedStateFromError)\b", body):
                    flags.append("class_component")
        elif t == "assignment_expression":
            left = n.child_by_field_name("left")
            if left is not None and left.type == "member_expression":
                prop = left.child_by_field_name("property")
                obj = left.child_by_field_name("object")
                if (prop is not None and G._txt(prop) == "defaultProps"
                        and obj is not None and G._txt(obj) not in classes):
                    flags.append("defaultProps")
    called = [c for c in _CREDITS
              if (re.search(r"\buse\(" if c == "use" else rf"\b{c}\b", prompt or ""))
              or (c == "use" and re.search(r"`use`", prompt or ""))]
    return {"modern_flags": sorted(flags), "modern_used": sorted(used),
            "called_for": called,
            "modern_credits": sorted(c for c in used if c in called)}


# --------------------------------------------------------------- lint -----
def _tsconfig(task: dict) -> tuple[dict, str]:
    g = task["grader"]
    kind = g["kind"]
    if kind == "react":
        import grade_react
        return dict(grade_react.TSCONFIG["compilerOptions"],
                    **(g.get("compilerOptions") or {})), "solution.tsx"
    if kind == "tc_tests":
        return G._tc_options(g), "solution.ts"
    return G._ts_options(g, runtime=False), "solution.ts"


def lint(task: dict, code: str) -> dict:
    why = ready()
    if why:
        return {"style_error": why}
    opts, fname = _tsconfig(task)
    opts = dict(opts, noEmit=True)
    box = os.path.join(SANDBOX, "tasks", G._safe(task["id"]))
    G._fresh_dir(box)
    with open(os.path.join(box, fname), "w", encoding="utf-8", newline="\n") as f:
        f.write(code.rstrip() + "\n")
    with open(os.path.join(box, "tsconfig.json"), "w", encoding="utf-8") as f:
        json.dump({"compilerOptions": opts, "files": [fname]}, f, indent=1)
    with open(os.path.join(box, "eslint.config.mjs"), "w", encoding="utf-8") as f:
        f.write(ESLINT_CONFIG)
    r = G.run([G.NODE, ESLINT, "--format", "json", fname], box, T_LINT)
    if r["spawn_failed"] or r["timeout"]:
        return {"style_error": "eslint " + ("timed out" if r["timeout"] else r["err"][:300])}
    try:
        rep = json.loads(r["out"] or "[]")
    except ValueError:
        return {"style_error": "eslint gave no JSON (rc=%s): %s"
                % (r["rc"], G._tail(r["out"] + r["err"], 400))}
    msgs = [m for f in rep for m in f.get("messages") or []]
    rules: dict[str, int] = {}
    for m in msgs:
        k = m.get("ruleId") or ("parse" if m.get("fatal") else "?")
        rules[k] = rules.get(k, 0) + 1
    return {"lint_errors": sum(1 for m in msgs if m.get("severity") == 2),
            "lint_warnings": sum(1 for m in msgs if m.get("severity") == 1),
            "lint_rules": dict(sorted(rules.items(), key=lambda kv: -kv[1])),
            "lint_fatal": any(m.get("fatal") for m in msgs)}


def style(task: dict, answer_text: str) -> dict:
    """The style record for one reply. Never raises; never a grade."""
    t0 = time.perf_counter()
    kind = (task.get("grader") or {}).get("kind")
    out: dict = {"tool": TOOL, "style_error": None, "skipped": None}
    try:
        if kind not in KINDS:
            out["skipped"] = f"kind {kind!r} is not TypeScript"
        else:
            ex = G.extract(answer_text, G.FAMILY[kind],
                           task["grader"].get("entry") or [])
            if not ex["ok"]:
                out["skipped"] = "no code extracted"
            else:
                with G._TaskLock("lint-" + task["id"]):
                    out.update(lint(task, ex["code"]))
                out.update(react19(ex["code"], task.get("prompt") or ""))
    except Exception:                                           # noqa: BLE001
        out["style_error"] = "style scorer crashed: " + traceback.format_exc()[-600:]
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["--setup"]:
        return setup()
    if len(a) != 2:
        print(__doc__)
        return 2
    pool = G.load_tasks()
    for p in (os.path.join(HERE, "tasks_react.jsonl"), G.TC_JSONL):
        if os.path.isfile(p):
            pool += G.load_tasks(p)
    task = next((t for t in pool if t["id"] == a[0]), None)
    if task is None:
        print(f"no task {a[0]!r}")
        return 2
    with open(a[1], encoding="utf-8") as f:
        print(json.dumps(style(task, f.read()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
