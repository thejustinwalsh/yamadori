#!/usr/bin/env python
"""The code parsers (mcp/code_check.py). No GPU, no network.

    python mcp/test_code_check.py      -> "N/M checks passed"

WHAT THIS GATES

code_check is only the parse layer now (2026-09-29, docs/REMOVED.md): the
formatters, lint, style inference, continuation analysis, review_answer, the
check_code tool and the answer-repair pass were removed with the code check
of client writes. What stays is read by mcp/route.py (does the prompt carry
code that parses?), mcp/tool_code.py (an .html write's inline scripts) and
the skills pipeline's proof (mcp/skill_prove.py).

  1. Syntax errors come back with the right line and column, per language,
     and valid code in every language comes back clean (a false syntax error
     would send a model to "fix" correct code). Python's compiler is the
     authority where tree-sitter lags.
  2. Fenced code blocks: info strings, tildes, nested fences, unclosed
     fences, and the lines each block spans.
  3. normalize_language, parse_tree and the MAX_ERRORS cap.
  4. THE CONTRACT: no user code is executed and nothing outside a temp dir
     (or the interpreter's own library) is opened. Asserted with sentinel
     files the code under check would create, and a Python audit hook over
     every parse.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_code_check_")

import code_check as cc  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def first(errs: list[dict]) -> dict:
    return errs[0] if errs else {}


# ---------------------------------------------------------------------------
# 1. Syntax errors, per language
# ---------------------------------------------------------------------------
BROKEN = {
    # language: (code, expected line of the first error)
    "python":     ("x = 1\ny = (2,\nz = 3\n", 2),
    "typescript": ("const a: number = 1;\nfunction f(x: number) {\n  return x\n", 3),
    "tsx":        ("const a = 1;\nconst b = <div>{a</div>;\n", 2),
    "javascript": ("let a = 1;\nlet b = [1, 2;\n", 2),
    "jsx":        ("const a = 1;\nconst x = <p>{a</p>;\n", 2),
    "rust":       ("fn main() {\n    let x = 1\n    let y = 2;\n}\n", 2),
    "c":          ("int main(void) {\n  int x = 1\n  return x;\n}\n", 2),
    "cpp":        ("class A {\n  int x;\n};\nint f( { return 1; }\n", 4),
    "json":       ('{\n  "a": 1,\n  "b": ,\n}\n', 3),
    "toml":       ("[a]\nb = 1\nc = \n", 3),
    "yaml":       ("a: 1\nb: [1, 2\nc: 3\n", 3),
}
VALID = {
    "python":     "def f(xs):\n    return [x * 2 for x in xs]\n",
    "typescript": "export function f(x: number): number {\n  return x + 1;\n}\n",
    "tsx":        "export const A = ({ n }: { n: number }) => <div>{n}</div>;\n",
    "javascript": "const f = (a) => a.map((x) => x + 1);\n",
    "jsx":        "const X = () => <p className=\"a\">hi</p>;\n",
    "rust":       "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
    "c":          "int add(int a, int b) { return a + b; }\n",
    "cpp":        "#include <vector>\nint n(const std::vector<int>& v) { return (int)v.size(); }\n",
    "json":       '{"a": [1, 2, {"b": null}]}\n',
    "toml":       "[tool.x]\na = 1\nb = \"s\"\n",
    "yaml":       "a: 1\nb:\n  - x\n  - y\n",
}


def test_syntax_errors_have_positions_per_language():
    for lang, (code, line) in BROKEN.items():
        errs = cc.syntax_errors(code, lang)
        check(bool(errs), f"{lang}: a syntax error is found", repr(code))
        e = first(errs)
        check(e.get("line") == line,
              f"{lang}: the first error is on line {line}",
              json.dumps(errs)[:200])
        check(isinstance(e.get("col"), int) and e["col"] >= 1
              and bool(e.get("message")),
              f"{lang}: with a column and a message", json.dumps(e))
    for lang, code in VALID.items():
        errs = cc.syntax_errors(code, lang)
        check(errs == [], f"{lang}: valid code has no errors",
              json.dumps(errs)[:200])


def test_python_uses_the_compiler_as_the_authority():
    # A compile-stage error tree-sitter cannot see.
    errs = cc.syntax_errors("class A:\n    pass\nreturn 1\n", "python")
    check(first(errs).get("message", "").startswith("'return' outside")
          and first(errs).get("line") == 3,
          "'return' outside function is reported at its line",
          json.dumps(errs))
    errs = cc.syntax_errors("def f():\n    x = 1\n      y = 2\n", "python")
    check(first(errs).get("line") == 3 and "indent" in first(errs)["message"],
          "an unexpected indent is reported at its line", json.dumps(errs))
    # Newer syntax the compiler accepts: never a false syntax error.
    errs = cc.syntax_errors("type Pair[T] = tuple[T, T]\n"
                            "def first[T](p: Pair[T]) -> T:\n    return p[0]\n",
                            "python")
    check(errs == [], "code the compiler accepts is clean even if a grammar "
          "lags", json.dumps(errs))
    col = cc.syntax_errors("s = 'é' + (\n", "python")
    check(bool(col), "a non-ASCII line still yields an error", json.dumps(col))
    ts = cc.tree_errors("x = 'é';  y = (\n", "python")
    check(all(e["col"] >= 1 for e in ts), "tree-sitter columns are characters "
          "and 1-based", json.dumps(ts))


# ---------------------------------------------------------------------------
# 2. Fenced code blocks
# ---------------------------------------------------------------------------
def test_fenced_blocks():
    fb = cc.fenced_blocks("a\n```python\nx = 1\n```\n~~~ts\nlet a\n~~~\n"
                          "````md\n```js\nno\n```\n````\n```\nopen")
    check([b["lang"] for b in fb] == ["python", "ts", "md", ""],
          "fences: info strings, tildes, a longer outer fence, unclosed",
          json.dumps(fb))
    check(fb[2]["code"] == "```js\nno\n```" and fb[3]["closed"] is False,
          "an inner fence stays content; an unclosed fence runs to the end",
          json.dumps(fb))
    check([(b["line"], b["end_line"]) for b in fb]
          == [(3, 4), (6, 7), (9, 12), (14, 15)],
          "each block says where its body starts and its closing fence is",
          json.dumps([(b["line"], b["end_line"]) for b in fb]))


# ---------------------------------------------------------------------------
# 3. Languages, the tree, the cap
# ---------------------------------------------------------------------------
def test_languages_the_tree_and_the_cap():
    for tag, want in (("py", "python"), (".ts", "typescript"),
                      ("TSX", "tsx"), ("node", "javascript"),
                      ("rs", "rust"), ("c++", "cpp"), ("yml", "yaml"),
                      ("python", "python"), ("cobol", None), ("", None),
                      (None, None), (7, None)):
        check(cc.normalize_language(tag) == want,
              f"normalize_language({tag!r}) is {want!r}",
              repr(cc.normalize_language(tag)))
    check(set(cc.ALIASES.values()) <= set(cc.LANGS),
          "every alias names a language the parsers know")
    check(cc.CODE_LANGS <= set(cc.LANGS)
          and not {"json", "toml", "yaml"} & cc.CODE_LANGS,
          "the CODE languages are known and exclude the config formats",
          str(sorted(cc.CODE_LANGS)))
    root = cc.parse_tree(VALID["typescript"], "typescript")
    check(root.type == "program" and not root.has_error,
          "parse_tree: valid TypeScript is a clean program", root.type)
    bad = cc.parse_tree(BROKEN["javascript"][0], "javascript")
    check(bad.has_error, "parse_tree: broken JavaScript has an error node")
    many = "\n".join(f"let a{i} = [1, 2;" for i in range(40)) + "\n"
    errs = cc.syntax_errors(many, "javascript")
    check(0 < len(errs) <= cc.MAX_ERRORS,
          f"at most MAX_ERRORS ({cc.MAX_ERRORS}) errors per text",
          str(len(errs)))


# ---------------------------------------------------------------------------
# 4. THE CONTRACT: nothing executed, nothing read outside a temp dir
# ---------------------------------------------------------------------------
_AUDIT: dict = {"on": False, "events": []}


def _hook(event, args):
    if _AUDIT["on"] and event in ("open", "os.system", "subprocess.Popen",
                                  "exec", "os.exec", "os.spawn", "os.posix_spawn",
                                  "os.startfile", "os.remove", "os.rename",
                                  "shutil.rmtree"):
        _AUDIT["events"].append((event, args))


sys.addaudithook(_hook)


def test_no_user_code_runs_and_nothing_outside_temp_is_opened():
    s1 = os.path.join(_TMP, "SENTINEL_open")
    s2 = os.path.join(_TMP, "SENTINEL_system")
    s3 = os.path.join(_TMP, "SENTINEL_js")
    s4 = os.path.join(_TMP, "SENTINEL_rs")
    marker = "yamadori_marker_7f3a"
    # Every payload is VALID code that would create a sentinel if it ran.
    # (It once was not: a Windows path inside a Python string literal made
    # the payload a SyntaxError, so it could never have run and the check
    # below could never have failed. Asserted now.)
    cmd2 = json.dumps(f'echo ran > "{s2}"')
    cmd4 = json.dumps(f'echo ran > "{s4}"')
    payloads = {
        "python": (f"# {marker}\nopen({s1!r}, 'w').write('ran')\n"
                   f"import os; os.system({cmd2})\n"
                   f"__import__('os').system({cmd2})\n"),
        "javascript": (f"// {marker}\nrequire('fs').writeFileSync("
                       f"{json.dumps(s3)}, 'ran');\n"
                       f"require('child_process').execSync({cmd2});\n"),
        "typescript": (f"// {marker}\nimport * as fs from 'fs';\n"
                       f"fs.writeFileSync({json.dumps(s3)}, 'ran');\n"),
        "rust": (f"// {marker}\nfn main() {{ std::fs::write({json.dumps(s4)}, "
                 f"\"ran\").unwrap(); std::process::Command::new(\"cmd\")"
                 f".arg(\"/c\").arg(\"echo ran\").status().unwrap(); }}\n"),
        "c": (f"// {marker}\n#include <stdlib.h>\nint main(void) {{ "
              f"system({cmd4}); return 0; }}\n"),
    }
    for lang, code in payloads.items():
        check(cc.syntax_errors(code, lang) == [],
              f"the {lang} payload is valid code (so only not running it "
              f"keeps its sentinel absent)",
              json.dumps(cc.syntax_errors(code, lang))[:200])
    # Warm-up outside the audit: first use imports modules and loads
    # grammars, which legitimately open library files.
    for lang, code in payloads.items():
        cc.syntax_errors(code, lang)
        cc.fenced_blocks(f"```{lang}\n{code}\n```")
    cc.syntax_errors("class A:\n    pass\nreturn 1\n", "python")
    _AUDIT["events"].clear()
    _AUDIT["on"] = True
    try:
        for lang, code in payloads.items():
            cc.syntax_errors(code, lang)
            cc.tree_errors(code, lang)
            for b in cc.fenced_blocks(f"```{lang}\n{code}\n```"):
                cc.syntax_errors(b["code"], cc.normalize_language(b["lang"]))
        # A compile-STAGE error carries no source text, and CPython then
        # tries to open the file named in compile() to quote the line. With
        # "<check_code>" that was a read in the server's working directory.
        cc.syntax_errors("class A:\n    pass\nreturn 1\n", "python")
    finally:
        _AUDIT["on"] = False
    events = list(_AUDIT["events"])
    for s in (s1, s2, s3, s4):
        check(not os.path.exists(s), f"sentinel {os.path.basename(s)} was "
              f"never created: the code was not run")
    check(not [e for e in events if e[0] in ("os.system", "exec", "os.exec",
                                             "os.spawn", "os.posix_spawn",
                                             "os.startfile",
                                             "subprocess.Popen")],
          "no os.system / exec / spawn / subprocess during any parse",
          str([e[0] for e in events][:10]))
    tmp = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
    lib = [os.path.normcase(os.path.abspath(p)) for p in
           {sys.prefix, sys.base_prefix, sys.exec_prefix}]
    opened = []
    for ev, args in events:
        if ev != "open" or not args or not isinstance(args[0], (str, bytes)):
            continue
        p = args[0].decode() if isinstance(args[0], bytes) else args[0]
        opened.append(os.path.normcase(os.path.abspath(p)))
    outside = [p for p in opened
               if not p.startswith(tmp) and not any(p.startswith(x) for x in lib)]
    check(not outside, "every file opened is in a temp dir (or the "
          "interpreter's own library)", str(outside[:5]))
    check(os.path.normcase(os.path.abspath(cc._COMPILE_NAME)).startswith(tmp)
          and not os.path.exists(os.path.dirname(cc._COMPILE_NAME)),
          "the name compile() may try to open is under the temp dir and "
          "its directory never exists", cc._COMPILE_NAME)


TESTS = [
    test_syntax_errors_have_positions_per_language,
    test_python_uses_the_compiler_as_the_authority,
    test_fenced_blocks,
    test_languages_the_tree_and_the_cap,
    test_no_user_code_runs_and_nothing_outside_temp_is_opened,
]


def main() -> int:
    for t in TESTS:
        try:
            t()
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{t.__name__} raised",
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    passed = sum(1 for ok, _, _ in _results if ok)
    for ok, name, detail in _results:
        if not ok:
            print(f"FAIL  {name}\n      {detail[:1500]}")
    print(f"\n{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
