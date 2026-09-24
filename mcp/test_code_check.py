#!/usr/bin/env python
"""check_code (mcp/code_check.py), its gating in the proxy, and the opt-in
repair pass, against a fake upstream. No GPU, no network.

    python mcp/test_code_check.py      -> "N/M checks passed"

WHAT THIS GATES

  1. Syntax errors come back with the right line and column, per language,
     and valid code in every language comes back clean (a false syntax error
     would send a model to "fix" correct code).
  2. Style is inferred from the existing code (2 vs 4 spaces vs tabs, quotes,
     semicolons, trailing commas, width), and explicit config text wins.
  3. Continuation mode catches the LiveBench 2026-09-23 failure shapes -- a
     body not indented under the prefix's open header, a continuation
     dedented out of a `while` (an infinite loop), out of the method (a
     `return` in the class body), an unclosed docstring carried over from
     the prefix, and a re-declared signature -- and passes their correct
     versions. The prefixes are reduced skeletons of lb-20260923 rows
     3cd8c16a, 45d51d09, 5c9d7ade and eae49210 (the question text is
     LiveBench's and is not copied here); the continuations are the model's
     own answers, verbatim. THE GENERAL RULE (operator, 2026-09-23): when
     the question carries fenced code, an answer's block parses if it
     parses on its own OR appended after one of the question's blocks, with
     no trigger words; repair flags it only when neither does, and says
     both results. A question without code keeps the old reading.
  4. A missing or broken formatter is reported, never raised.
  5. THE CONTRACT: no user code is executed and nothing outside a temp dir
     is opened. Asserted with sentinel files the code under check tries to
     create, and a Python audit hook over every check call.
  6. check_code is offered by the tier's rule, independent of the code-tool
     gate, and the bare-model arm stays bare.
  7. The repair loop: feedback with the exact errors, the model's own
     corrected answer delivered untouched, and x_yamadori.repair recording
     rounds / errors_before / errors_after / why it stopped. Blocking and
     streamed paths, against a fake upstream.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Every store is a temp path, set BEFORE import: the corpus is training data
# for the decision model and test runs have polluted it before.
_TMP = tempfile.mkdtemp(prefix="yamadori_test_code_check_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.makedirs(os.environ["YAMADORI_PKG_DIR"], exist_ok=True)
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
os.environ.pop("YAMADORI_DELEGATE_TOOL", None)
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")

import code_check as cc  # noqa: E402
import tiers  # noqa: E402

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
# 2. Style inference and config precedence
# ---------------------------------------------------------------------------
PY2 = "def f(a):\n  if a:\n    return 'x'\n  return 'y'\n"
PY4 = 'def f(a):\n    if a:\n        return "x"\n    return "y"\n'
PYTAB = "def f(a):\n\tif a:\n\t\treturn 1\n\treturn 2\n"
TS_NOSEMI = ("import { a } from 'b'\n"
             "const o = {\n    x: 1,\n    y: 2,\n}\n"
             "export function f(n: number) {\n    return n + 1\n}\n")
TS_SEMI = ('import { a } from "b";\n'
           "const o = {\n  x: 1,\n  y: 2\n};\n"
           "export function f(n: number) {\n  return n + 1;\n}\n")


def test_style_is_inferred_from_the_existing_code():
    s = cc.infer_style(PY2, "python")
    check(s.get("indent") == "space" and s.get("indent_width") == 2,
          "2-space Python", json.dumps(s))
    check(s.get("quotes") == "single", "single quotes", json.dumps(s))
    s = cc.infer_style(PY4, "python")
    check(s.get("indent_width") == 4 and s.get("quotes") == "double",
          "4-space Python, double quotes", json.dumps(s))
    s = cc.infer_style(PYTAB, "python")
    check(s.get("indent") == "tab", "tab-indented Python", json.dumps(s))
    s = cc.infer_style(TS_NOSEMI, "typescript")
    check(s.get("indent_width") == 4 and s.get("quotes") == "single"
          and s.get("semicolons") is False and s.get("trailing_commas") == "all",
          "TS: 4 spaces, single quotes, no semicolons, trailing commas",
          json.dumps(s))
    s = cc.infer_style(TS_SEMI, "typescript")
    check(s.get("indent_width") == 2 and s.get("quotes") == "double"
          and s.get("semicolons") is True and s.get("trailing_commas") == "none",
          "TS: 2 spaces, double quotes, semicolons, no trailing commas",
          json.dumps(s))
    jsdoc = "/**\n * doc\n * more\n */\nfunction f() {\n  return 1;\n}\n"
    s = cc.infer_style(jsdoc, "javascript")
    check(s.get("indent_width") == 2, "JSDoc ' *' alignment is not an indent "
          "of 1", json.dumps(s))
    long_ = "\n".join(f"x{i} = {'1 + ' * 23}1" for i in range(12))
    s = cc.infer_style(long_, "python")
    check(s.get("width") == 100, "width: the smallest standard limit that "
          "holds the longest line", json.dumps(s))
    check("width" not in cc.infer_style(PY2, "python"),
          "no width claimed from four lines")
    check(cc.infer_style("", "python") == {}, "no evidence, no claims")


def test_config_text_wins_over_inference():
    ec = "root = true\n[*]\nindent_style = space\nindent_size = 3\n" \
         "[*.md]\nindent_size = 8\n"
    style, src, notes = cc.resolve_style("python", "x = 1\n", PY2, ec)
    check(style["indent_width"] == 3 and src["indent_width"] == "config",
          ".editorconfig beats the context's 2 spaces", json.dumps([style, src]))
    check(style["quotes"] == "single" and src["quotes"] == "context",
          "keys the config does not set still come from context",
          json.dumps(src))
    check("editorconfig" in notes.get("config_text", ""),
          "and the result says how the config was read", json.dumps(notes))
    pr = '{"singleQuote": false, "semi": true, "tabWidth": 2, "printWidth": 100}'
    style, src, _ = cc.resolve_style("typescript", "let a = 1\n", TS_NOSEMI, pr)
    check(style["quotes"] == "double" and style["semicolons"] is True
          and style["indent_width"] == 2 and style["width"] == 100,
          ".prettierrc JSON beats inference", json.dumps(style))
    pp = "[tool.ruff]\nline-length = 100\n[tool.ruff.format]\nquote-style = 'single'\n"
    style, src, _ = cc.resolve_style("python", "x = 1\n", PY4, pp)
    check(style["width"] == 100 and style["quotes"] == "single"
          and style["indent_width"] == 4,
          "pyproject [tool.ruff] beats inference; the rest is inferred",
          json.dumps(style))
    rf = "max_width = 80\ntab_spaces = 2\n"
    style, _, _ = cc.resolve_style("rust", "fn a() {}\n", None, rf)
    check(style["width"] == 80 and style["indent_width"] == 2,
          "rustfmt.toml is read", json.dumps(style))
    style, _, notes = cc.resolve_style("python", "x = 1\n", None, "hello")
    check("no recognised" in notes.get("config_text", ""),
          "unrecognised config text is said to be ignored", json.dumps(notes))


# ---------------------------------------------------------------------------
# 3. Continuation mode: the LiveBench failure shapes
# ---------------------------------------------------------------------------
# lb-20260923 3cd8c16a: the continuation dedents out of `while left <= right`,
# leaving `mid = ...` as the loop's whole body -- an infinite loop.
LOOP_PREFIX = (
    "class Solution(object):\n"
    "    def earliestSecondToMarkIndices(self, nums, changeIndices):\n"
    "        def check(t):\n"
    "            return t >= len(nums)\n"
    "\n"
    "        left, right = sum(nums)+len(nums), len(changeIndices) \n"
    "        while left <= right:\n"
    "            mid = left+(right-left)//2")
LOOP_MODEL = (
    "        if check(mid):\n"
    "            right = mid - 1\n"
    "        else:\n"
    "            left = mid + 1\n"
    "        if left > len(changeIndices):\n"
    "            return -1\n"
    "        return left")
LOOP_RIGHT = (
    "            if check(mid):\n"
    "                right = mid - 1\n"
    "            else:\n"
    "                left = mid + 1\n"
    "        if left > len(changeIndices):\n"
    "            return -1\n"
    "        return left")

# lb-20260923 45d51d09: dedented out of the method; `return` lands in the class.
METHOD_PREFIX = (
    "class Solution(object):\n"
    "    def maximumLength(self, nums):\n"
    "        cnt = {}\n"
    "        for x in nums:\n"
    "            cnt[x] = cnt.get(x, 0) + 1\n"
    "        result = 0\n"
    "        for x in cnt.keys():\n"
    "            l = cnt[x]")
METHOD_MODEL = (
    "    result = max(result, l)\n"
    "    for i in range(3):\n"
    "        l += 2\n"
    "        result = max(result, l)\n"
    "    return result")

# lb-20260923 5c9d7ade: the prefix ends inside an open docstring.
DOC_PREFIX = (
    "class Solution(object):\n"
    "    def returnToBoundaryCount(self, nums):\n"
    '        """\n'
    "        :type nums: List[int]\n"
    "        :rtype: int")
DOC_MODEL = (
    "        position = 0\n"
    "        count = 0\n"
    "        for num in nums:\n"
    "            position += num\n"
    "            if position == 0:\n"
    "                count += 1\n"
    "        return count")
DOC_RIGHT = '        """\n' + DOC_MODEL

# lb-20260923 eae49210: the prefix ends on a `while` header; the body is not
# indented under it.
HEADER_PREFIX = (
    "class Solution(object):\n"
    "    def minimumAddedCoins(self, coins, target):\n"
    "        coins.sort()\n"
    "        result = reachable = 0\n"
    "        for x in coins:\n"
    "            while not reachable >= x-1:\n"
    "                result += 1\n"
    "                reachable += reachable+1\n"
    "            reachable += x\n"
    "        while not reachable >= target:")
HEADER_MODEL = (
    "    result += 1\n"
    "    reachable += reachable + 1\n"
    "return result")
HEADER_RIGHT = (
    "            result += 1\n"
    "            reachable += reachable + 1\n"
    "        return result")

# Re-declared signature: the model writes the method header again.
SIG_PREFIX = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    '        """Return the indices of the two numbers."""')
SIG_MODEL = (
    "    def twoSum(self, nums, target):\n"
    "        seen = {}\n"
    "        for i, x in enumerate(nums):\n"
    "            if target - x in seen:\n"
    "                return [seen[target - x], i]\n"
    "            seen[x] = i")
SIG_RIGHT = "\n".join(SIG_MODEL.split("\n")[1:])


def kinds(res: dict) -> set[str]:
    return {d.get("kind") for d in res["defects"]}


def test_continuation_catches_the_livebench_shapes():
    r = cc.check(LOOP_MODEL, "python", mode="continuation", prefix=LOOP_PREFIX)
    check(not r["ok"] and "loop_cannot_end" in kinds(r),
          "3cd8c16a: dedenting out of the while is an infinite loop",
          json.dumps(r["defects"])[:300])
    d = [x for x in r["defects"] if x["kind"] == "loop_cannot_end"]
    check(bool(d) and d[0]["where"] == "prefix" and d[0]["line"] == 7
          and "left" in d[0]["message"] and "right" in d[0]["message"],
          "and names the loop, its line and the names it never changes",
          json.dumps(d))
    text = cc.render(r)
    check("closes `while left <= right`" in text and "column 12" in text,
          "and says the first line closes the loop, and which column is "
          "inside it", text)
    r = cc.check(LOOP_RIGHT, "python", mode="continuation", prefix=LOOP_PREFIX)
    check(r["ok"], "the correct continuation passes",
          json.dumps(r["defects"] + r["syntax_errors"])[:300])

    r = cc.check(METHOD_MODEL, "python", mode="continuation",
                 prefix=METHOD_PREFIX)
    e = first(r["syntax_errors"])
    check(not r["ok"] and "outside function" in e.get("message", "")
          and e.get("where") == "code" and e.get("line") == 5,
          "45d51d09: `return` in the class body, at YOUR line 5",
          json.dumps(r["syntax_errors"]))

    r = cc.check(DOC_MODEL, "python", mode="continuation", prefix=DOC_PREFIX)
    check(not r["ok"] and "unclosed_string_from_prefix" in kinds(r),
          "5c9d7ade: the unclosed docstring from the prefix is reported",
          json.dumps(r["defects"])[:300])
    msg = next((x["message"] for x in r["defects"]
                if x["kind"] == "unclosed_string_from_prefix"), "")
    check('`        """`' in msg and "prefix line 3" in msg,
          "with the exact closing line and where it opened", msg)
    check((r.get("continuation") or {}).get("open_string", {}).get("docstring"),
          "and it is identified as a docstring")
    r = cc.check(DOC_RIGHT, "python", mode="continuation", prefix=DOC_PREFIX)
    check(r["ok"], "closing the docstring first passes",
          json.dumps(r["defects"] + r["syntax_errors"])[:300])

    r = cc.check(HEADER_MODEL, "python", mode="continuation",
                 prefix=HEADER_PREFIX)
    check(not r["ok"] and "body_not_indented" in kinds(r),
          "eae49210: the body is not indented under the prefix's while",
          json.dumps(r["defects"])[:300])
    msg = next((x["message"] for x in r["defects"]
                if x["kind"] == "body_not_indented"), "")
    check("while not reachable >= target" in msg and "column 12" in msg
          and "column 4" in msg,
          "and names the header, the column it needs and the one it got", msg)
    r = cc.check(HEADER_RIGHT, "python", mode="continuation",
                 prefix=HEADER_PREFIX)
    check(r["ok"], "the correct continuation passes",
          json.dumps(r["defects"] + r["syntax_errors"])[:300])

    r = cc.check(SIG_MODEL, "python", mode="continuation", prefix=SIG_PREFIX)
    check(not r["syntax_errors"], "(a re-declared method still compiles)")
    check(not r["ok"] and "redeclares_signature" in kinds(r),
          "a re-declared signature is caught although it compiles",
          json.dumps(r["defects"])[:300])
    r = cc.check(SIG_RIGHT, "python", mode="continuation", prefix=SIG_PREFIX)
    check(r["ok"], "the body alone passes",
          json.dumps(r["defects"] + r["syntax_errors"])[:300])


def test_continuation_in_brace_languages():
    pre = ("function sum(xs: number[]): number {\n  let s = 0;\n"
           "  for (const x of xs) {")
    ok = "    s += x;\n  }\n  return s;\n}"
    bad = "    s += x;\n  return s;\n}"
    r = cc.check(ok, "typescript", mode="continuation", prefix=pre)
    check(r["ok"], "TS continuation that closes both blocks passes",
          json.dumps(r["syntax_errors"]))
    check((r["continuation"].get("prefix_leaves_open") or {}).get("}") == 2,
          "the prefix is reported to leave two braces open",
          json.dumps(r["continuation"]))
    r = cc.check(bad, "typescript", mode="continuation", prefix=pre)
    check(not r["ok"] and r["syntax_errors"], "one brace short is an error",
          json.dumps(r["syntax_errors"]))
    rs = "fn main() {\n    let v = vec![1, 2];\n    for x in &v {"
    r = cc.check('        println!("{}", x);\n    }\n}', "rust",
                 mode="continuation", prefix=rs)
    check(r["ok"], "Rust continuation passes", json.dumps(r["syntax_errors"]))


def test_whole_mode_python_excerpts_and_loops():
    excerpt = "    def f(self):\n        return 1\n"
    r = cc.check(excerpt, "python", run_format=False)
    check(r["ok"] and "dedented" in r["notes"],
          "an indented excerpt is checked dedented, not called broken",
          json.dumps(r["notes"]))
    loop = "def f(n):\n    i = 0\n    while i < n:\n        print(i)\n"
    r = cc.check(loop, "python", run_format=False)
    check("loop_cannot_end" in kinds(r), "whole mode: a loop that cannot end",
          json.dumps(r["defects"]))
    fine = [
        "def f(q):\n    while q:\n        q.pop()\n",
        "def f(n):\n    while n > 0:\n        n -= 1\n",
        "def f():\n    while True:\n        if g():\n            break\n",
        "def f(s):\n    while s.ready:\n        step(s)\n",
        "def f(h):\n    import heapq\n    while h:\n        heapq.heappop(h)\n",
        "def f(a):\n    i = 0\n    while i < len(a):\n        for j in a:\n            break\n        i += 1\n",
        "x = 0\ndef bump():\n    global x\n    x += 1\nwhile x < 3:\n    bump()\n",
        "def f(it):\n    while it:\n        yield it\n",
    ]
    for src in fine:
        r = cc.check(src, "python", run_format=False)
        check("loop_cannot_end" not in kinds(r),
              f"no false alarm: {src.splitlines()[1].strip()} ...",
              json.dumps(r["defects"]))
    nested_break = ("def f(n):\n    i = 0\n    while i < n:\n"
                    "        for j in range(3):\n            break\n")
    r = cc.check(nested_break, "python", run_format=False)
    check("loop_cannot_end" in kinds(r),
          "a break of an INNER loop does not end the outer one",
          json.dumps(r["defects"]))


# ---------------------------------------------------------------------------
# Real-shaped answers: fences, the question's prefix, the repair verdict
# ---------------------------------------------------------------------------
def completion_question(prefix: str) -> str:
    """The SHAPE of a LiveBench coding_completion prompt (its words are
    paraphrased): instructions, the starter code fenced, a closing note."""
    return ("### Instructions: complete the program.\n### Question:\n"
            "(problem statement)\n\n### Format: You will use the following "
            "starter code.\n```python\n" + prefix + "\n```\n\n### Answer: "
            "(only the missing portion)\n")


def test_fenced_blocks_and_review_answer():
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

    msgs = [{"role": "user", "content": completion_question(HEADER_PREFIX)}]
    p = cc.prefix_from_messages(msgs)
    check(p and p["code"] == HEADER_PREFIX, "the starter code is recovered "
          "verbatim from the question", json.dumps(p)[:200])
    check([b["code"] for b in cc.prompt_code(msgs)] == [HEADER_PREFIX],
          "prompt_code: the question's fenced code, verbatim")
    r = cc.review_answer(f"```python\n{HEADER_MODEL}\n```", msgs)
    check(r["checked"] == 1 and r["count"] >= 1 and r["feedback"]
          and r["check_mode"] == "neither"
          and "does not parse on its own" in r["feedback"]
          and "Nor does it parse appended directly after code block 1 of "
              "the question" in r["feedback"],
          "a continuation broken both ways: flagged, with both results",
          json.dumps(r)[:400])
    r2 = cc.review_answer(f"```python\n{HEADER_RIGHT}\n```", msgs)
    check(r2["count"] == 0 and r2["feedback"] is None
          and r2["check_mode"] == "appended",
          "its correct version parses appended: clean", json.dumps(r2)[:300])
    whole = ("```python\nclass Solution(object):\n    def minimumAddedCoins("
             "self, coins, target):\n        return 0\n```")
    r3 = cc.review_answer(whole, msgs)
    check(r3["count"] == 0 and r3["check_mode"] == "standalone",
          "a whole program that stands alone is clean, 'standalone'",
          json.dumps(r3)[:300])
    r4 = cc.review_answer("The answer is 42.", msgs)
    check(r4["checked"] == 0 and r4["count"] == 0
          and r4["check_mode"] is None,
          "an answer with no code checks nothing; check_mode None")
    r5 = cc.review_answer("```json\n{\"a\": ...}\n```", msgs)
    check(r5["checked"] == 0, "config-format blocks are not repaired")
    lcb = [{"role": "user", "content": "Solve it.\n```python\nclass Solution:"
            "\n    def f(self, n: int) -> int:\n        \n```"}]
    r6 = cc.review_answer("```python\nclass Solution:\n    def f(self, n: int)"
                          " -> int:\n        return n\n```", lcb)
    check(r6["count"] == 0, "LCB-style: a full program after a stub header "
          "is clean", json.dumps(r6)[:300])
    r7 = cc.review_answer("```python\ndef f(:\n    pass\n```",
                          [{"role": "user", "content": "write f"}])
    check(r7["count"] >= 1 and "line 1" in (r7["feedback"] or "")
          and r7["check_mode"] is None,
          "no code in the question: a broken block is reviewed on its own, "
          "check_mode None", json.dumps(r7)[:300])
    check(r["key"] != r2["key"] and r["key"] == cc.review_answer(
        f"```python\n{HEADER_MODEL}\n```", msgs)["key"],
          "the review key identifies the code, and only the code")


# The 128 LiveBench coding questions of 2026-09-23, cached by the stack-tuning
# analysis. Read for the measurement only; no question text is copied here.
LB_QUESTIONS = os.path.join(HERE, "..", "bench", "analysis",
                            "stack_tuning_0923", "data",
                            "lb_questions_slim.json")

# A WHOLE-CLASS REWRITE of the 3cd8c16a skeleton, the shape the fan-out
# tie-breaker delivered on 2026-09-23 (6 of 10 completion answers; 5 of the
# 6 scored). It parses on its own.
LOOP_REWRITE = (LOOP_PREFIX.replace("len(changeIndices) \n",
                                    "len(changeIndices)\n")
                + "\n" + LOOP_RIGHT)


def test_the_general_rule():
    """Operator decision, 2026-09-23: no trigger words, no intent guessing.
    When the question carries fenced code, an answer's code parses if it
    parses on its own OR appended after one of the question's blocks; repair
    flags it only when neither does."""
    msgs = [{"role": "user", "content": completion_question(DOC_PREFIX)}]
    pcode = cc.prompt_code(msgs)

    # 5c9d7ade: the open-docstring fragment fails both ways.
    g = cc.check_against_prompt(DOC_MODEL, "python", pcode)
    check(not g["parses"] and g["check_mode"] == "neither"
          and g["prefix_block"] == 1 and g["appended"] is not None,
          "the open-docstring fragment parses neither alone nor appended",
          json.dumps({k: g[k] for k in ("parses", "check_mode",
                                        "prefix_block")}))
    r = cc.review_answer(f"```python\n{DOC_MODEL}\n```", msgs)
    fb = r["feedback"] or ""
    check(r["count"] >= 1 and r["check_mode"] == "neither"
          and "does not parse on its own" in fb
          and "outside function" in fb
          and "Nor does it parse appended" in fb and "never closes it" in fb,
          "repair flags it and states both results: alone and appended",
          fb[:900])

    # A valid continuation fragment parses as "appended".
    r = cc.review_answer(f"```python\n{DOC_RIGHT}\n```", msgs)
    check(r["count"] == 0 and r["check_mode"] == "appended",
          "a valid continuation parses appended: clean", json.dumps(r)[:300])

    # A whole rewrite parses as "standalone".
    lmsgs = [{"role": "user", "content": completion_question(LOOP_PREFIX)}]
    check(cc.syntax_errors(LOOP_REWRITE, "python") == [],
          "(fixture: the rewrite parses on its own)")
    r = cc.review_answer(f"```python\n{LOOP_REWRITE}\n```", lmsgs)
    check(r["count"] == 0 and r["check_mode"] == "standalone",
          "a whole-class rewrite parses standalone: clean",
          json.dumps(r)[:300])
    # So does the 5c9d7ade delivered shape: a rewrite that re-opens the
    # docstring. The rule does not know how any grader joins; parsing on
    # its own is enough. (LiveBench appended it and it failed there.)
    doc_rewrite = ("class Solution(object):\n"
                   "    def returnToBoundaryCount(self, nums):\n" + DOC_MODEL)
    r = cc.review_answer(f"```python\n{doc_rewrite}\n```", msgs)
    check(r["count"] == 0 and r["check_mode"] == "standalone",
          "(a rewrite is judged on its own; no grader is modelled)",
          json.dumps(r)[:300])
    # Only parsing decides when the question carries code: 3cd8c16a's
    # fragment parses appended, so its infinite loop (a defect) is no
    # longer a reason to repair. Pinned so the trade is visible.
    r = cc.review_answer(f"```python\n{LOOP_MODEL}\n```", lmsgs)
    check(r["count"] == 0 and r["check_mode"] == "appended",
          "(a fragment that parses appended is not repaired for a defect)",
          json.dumps(r)[:300])

    # The last block first, then the others; other languages never.
    two = [{"role": "user", "content": "Context:\n```python\n" + DOC_PREFIX
            + "\n```\nand also\n```python\nx = 1\n```\n```json\n{}\n```"}]
    g = cc.check_against_prompt(DOC_RIGHT, "python", cc.prompt_code(two))
    check(g["parses"] and g["check_mode"] == "appended"
          and g["prefix_block"] == 1,
          "with several blocks, the one it parses after is found and named",
          json.dumps({k: g[k] for k in ("check_mode", "prefix_block")}))
    g = cc.check_against_prompt(DOC_MODEL, "python", cc.prompt_code(two))
    check(g["check_mode"] == "neither" and g["prefix_block"] == 2,
          "failing everywhere, it reports the last python block tried first",
          json.dumps({k: g[k] for k in ("check_mode", "prefix_block")}))

    # A code task with no code in the prompt: exactly the old reading.
    plain = [{"role": "user", "content": "write f"}]
    loop = "def f(n):\n    i = 0\n    while i < n:\n        print(i)\n"
    for code in ("def f(:\n    pass\n", "def f(x):\n    return x\n", loop):
        new = cc.review_answer(f"```python\n{code}```", plain)
        want = cc.check(code, "python", run_format=False)
        check(new["check_mode"] is None
              and new["count"] == len(want["syntax_errors"])
              + len(want["defects"]),
              "no code in the question: checked on its own, the same count "
              "as before (defects included)", json.dumps(new)[:200])
    # Prose: null.
    r = cc.review_answer("The answer is 42.", msgs)
    check(r["checked"] == 0 and r["check_mode"] is None,
          "prose: nothing checked, check_mode None", json.dumps(r))

    # Phrasing does not matter: "finish this function" and a partial def
    # behave exactly as the LiveBench-shaped question does.
    casual = [{"role": "user", "content": "finish this function\n\n```python\n"
               + DOC_PREFIX + "\n```"}]
    for code in (DOC_MODEL, DOC_RIGHT, doc_rewrite, HEADER_MODEL):
        a = cc.review_answer(f"```python\n{code}\n```", msgs)
        b = cc.review_answer(f"```python\n{code}\n```", casual)
        check((a["count"], a["check_mode"], a["feedback"])
              == (b["count"], b["check_mode"], b["feedback"]),
              f"'finish this function' == the LiveBench phrasing "
              f"({a['check_mode']})", json.dumps([a, b])[:400])

    # Measured on the real prompts, when the cached file is present: the
    # reference remainder of every completion, checked against the full
    # question and against the starter code alone with every word of the
    # instructions removed. Identical, and every one parses appended.
    if os.path.exists(LB_QUESTIONS):
        with open(LB_QUESTIONS, encoding="utf-8") as f:
            qs = [q for q in json.load(f)
                  if q.get("task") == "coding_completion"]
        same, appended = 0, 0
        for q in qs:
            full = cc.prompt_code([{"role": "user",
                                    "content": q["turns"][0]}])
            bare = cc.prompt_code([{"role": "user", "content":
                                    "```python\n" + q["partial_solution"]
                                    + "\n```"}])
            ga = cc.check_against_prompt(q["remainder"], "python", full)
            gb = cc.check_against_prompt(q["remainder"], "python", bare)
            same += (ga["check_mode"], ga["parses"]) == (gb["check_mode"],
                                                         gb["parses"])
            appended += ga["check_mode"] == "appended"
        check(qs and same == len(qs) == appended,
              "LiveBench 2026-09-23: every completion's reference remainder "
              "parses appended, identically with the instructions removed",
              f"{same} same, {appended} appended, of {len(qs)}")


# ---------------------------------------------------------------------------
# 4. Formatters: present, missing, broken
# ---------------------------------------------------------------------------
def test_formatters_follow_the_inferred_style():
    for lang, code, ctx, want, why in (
            ("python", 'def f(a,b):\n    return "x"+a\n', PY2,
             "def f(a, b):\n  return 'x' + a\n", "ruff: 2 spaces, single quotes from context"),
            ("typescript", 'const o = {a:1,b:"x"}\nexport function f(n:number){return n}\n',
             TS_NOSEMI,
             "const o = { a: 1, b: 'x' }\nexport function f(n: number) {\n    return n\n}\n",
             "prettier: 4 spaces, single quotes, no semicolons from context"),
            ("rust", "fn main(){let x=1;println!(\"{}\",x);}\n",
             "fn a() {\n  let y = 2;\n  if y > 1 {\n    b();\n  }\n}\n",
             'fn main() {\n  let x = 1;\n  println!("{}", x);\n}\n',
             "rustfmt: 2 spaces from context")):
        name = cc.LANGS[lang]["formatter"]
        st = cc.formatter_status(name)
        if not st["available"]:
            r = cc.check(code, lang, context=ctx)
            check(r["formatted"] is None and r["formatter"].get("remedy"),
                  f"{name} not installed here: reported with a remedy",
                  json.dumps(r["formatter"]))
            print(f"  note: {name} is not installed; its output was not checked")
            continue
        r = cc.check(code, lang, context=ctx)
        check(r["formatted"] == want, why,
              json.dumps({"got": r["formatted"], "fmt": r["formatter"]}))
        check(r["diff_summary"].get("changed_lines", 0) > 0,
              f"{lang}: the diff summary counts the changed lines",
              json.dumps(r["diff_summary"]))
        again = cc.check(want, lang, context=ctx)
        check(again["diff_summary"].get("changed_lines") == 0
              and "already matches" in cc.render(again),
              f"{lang}: formatted code is reported as already matching",
              cc.render(again)[:300])
    r = cc.check("def f(:\n  pass\n", "python", context=PY2)
    check(r["formatted"] is None and "fix the syntax" in cc.render(r),
          "no formatting is attempted on code that does not parse",
          cc.render(r))
    ex = "    def f(self):\n        return  1\n"
    r = cc.check(ex, "python")
    if r["formatter"].get("available"):
        check(r["formatted"] == "    def f(self):\n        return 1\n",
              "an excerpt is formatted dedented and put back at its indent",
              repr(r["formatted"]))


def test_a_missing_or_broken_formatter_is_reported_not_raised():
    real = cc._which
    try:
        cc._which = lambda name: None
        for lang in ("python", "typescript", "rust", "c"):
            r = cc.check(VALID[lang], lang)
            f = r["formatter"]
            check(r["ok"] and r["formatted"] is None and f["available"] is False
                  and f.get("remedy", "").startswith("operator"),
                  f"{lang}: formatter missing -> syntax still checked, "
                  f"remedy names the operator", json.dumps(f))
            text = cc.render(r)
            check("not installed" in text and "complete without it" in text,
                  f"{lang}: and the model is told so in plain words", text)
        cc._which = lambda name: os.path.join(_TMP, "no-such-formatter.exe")
        r = cc.check(VALID["python"], "python")
        check(r["ok"] and r["formatted"] is None
              and "could not start" in (r["formatter"].get("error") or ""),
              "a formatter that cannot start is reported, not raised",
              json.dumps(r["formatter"]))
    finally:
        cc._which = real
    r = cc.check('{"a": 1}', "json")
    check(r["formatted"] is None and "syntax check only" in cc.render(r),
          "JSON/TOML/YAML are syntax-only by design", cc.render(r))


# ---------------------------------------------------------------------------
# 5. THE CONTRACT: nothing executed, nothing read outside a temp dir
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
        cc.check(code, lang, context=code)
    cc.check(payloads["python"], "python", mode="continuation",
             prefix="def f():\n    x = 1")
    cc.run_tool({"code": payloads["python"], "language": "python"})
    _AUDIT["events"].clear()
    _AUDIT["on"] = True
    try:
        for lang, code in payloads.items():
            cc.check(code, lang, context=code)
            cc.run_tool({"code": code, "language": lang, "context": code})
        cc.check(payloads["python"], "python", mode="continuation",
                 prefix="def f():\n    x = 1")
        cc.review_answer("```python\n" + payloads["python"] + "\n```",
                         [{"role": "user", "content": "go"}])
        # A compile-STAGE error carries no source text, and CPython then
        # tries to open the file named in compile() to quote the line. With
        # "<check_code>" that was a read in the server's working directory.
        cc.check("class A:\n    pass\nreturn 1\n", "python")
    finally:
        _AUDIT["on"] = False
    events = list(_AUDIT["events"])
    for s in (s1, s2, s3, s4):
        check(not os.path.exists(s), f"sentinel {os.path.basename(s)} was "
              f"never created: the code was not run")
    check(not [e for e in events if e[0] in ("os.system", "exec", "os.exec",
                                             "os.spawn", "os.posix_spawn",
                                             "os.startfile")],
          "no os.system / exec / spawn during any check",
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
    # subprocess opens the null device for a formatter's stdin.
    null = {os.path.normcase(os.path.abspath(os.devnull)),
            os.path.normcase("\\\\.\\nul"), os.path.normcase(os.devnull)}
    outside = [p for p in opened if p not in null
               and not p.startswith(tmp) and not any(p.startswith(x) for x in lib)]
    check(not outside, "every file opened is in a temp dir (or the "
          "interpreter's own library)", str(outside[:5]))
    check(any(p.startswith(tmp) for p in opened),
          "(the audit saw the formatter's temp files, so it was watching)",
          str(opened[:3]))
    pops = [a for e, a in events if e == "subprocess.Popen"]
    allowed = {"ruff", "ruff.exe", "node", "node.exe", "rustfmt",
               "rustfmt.exe", "clang-format", "clang-format.exe"}
    exes = []
    for a in pops:
        argv = a[1] if len(a) > 1 else []
        if isinstance(argv, (str, bytes)):
            # On Windows the audit event carries the joined command line.
            import shlex
            argv = [x.strip('"') for x in shlex.split(
                argv if isinstance(argv, str) else argv.decode(), posix=False)]
        argv = list(argv)
        exes.append(os.path.basename(str(a[0] or argv[0])).lower())
        cwd = a[2] if len(a) > 2 else None
        check(cwd is not None and os.path.normcase(os.path.abspath(cwd))
              .startswith(tmp), f"{exes[-1]} runs with a temp dir as cwd",
              str(cwd))
        check(not any(marker in str(x) for x in argv),
              f"{exes[-1]}: the code never travels on a command line")
        joined = " ".join(str(x) for x in argv)
        if exes[-1].startswith("ruff"):
            check("--isolated" in argv, "ruff runs --isolated (no project "
                  "config)", joined)
        if exes[-1].startswith("node"):
            check("--no-config" in argv and "--no-editorconfig" in argv,
                  "prettier runs with --no-config --no-editorconfig", joined)
        if exes[-1].startswith("rustfmt"):
            k = argv.index("--config-path") if "--config-path" in argv else -1
            check(k >= 0 and os.path.normcase(os.path.abspath(argv[k + 1]))
                  .startswith(tmp), "rustfmt reads only our config file", joined)
    check(set(exes) <= allowed, "only formatters are ever started",
          str(sorted(set(exes))))
    check(bool(pops), "(formatters did run under the audit)", str(len(pops)))
    for name, exe in (("ruff", "ruff"), ("prettier", "node"),
                      ("rustfmt", "rustfmt")):
        if cc.formatter_status(name)["available"]:
            check(any(e.startswith(exe) for e in exes),
                  f"(and {name} was among them, so its flags were checked)",
                  str(sorted(set(exes))))


# ---------------------------------------------------------------------------
# The tool: arguments, failures, description
# ---------------------------------------------------------------------------
def test_the_tool_answers_every_call():
    for args, code in (({}, "BAD_ARGUMENTS"),
                       ({"code": "x = 1"}, "UNSUPPORTED_LANGUAGE"),
                       ({"code": "x = 1", "language": "cobol"},
                        "UNSUPPORTED_LANGUAGE"),
                       ({"code": 5, "language": "python"}, "BAD_ARGUMENTS"),
                       ({"code": "x", "language": "python",
                         "mode": "continuation"}, "BAD_ARGUMENTS"),
                       ({"code": "x", "language": "python", "mode": "diff"},
                        "BAD_ARGUMENTS"),
                       ({"code": "x" * (cc.MAX_CHARS + 1),
                         "language": "python"}, "TOO_LARGE")):
        out = cc.run_tool(args)
        try:
            d = json.loads(out)
        except ValueError:
            d = {}
        check(d.get("ok") is False and d.get("error") == code
              and d.get("retryable") is True
              and (d.get("remedies") or [{}])[0].get("fixable_by") == "agent",
              f"{code} for {sorted(args)}: retryable, fixable by the agent",
              out[:200])
    d = json.loads(cc.run_tool({"code": "x", "language": "cobol"}))
    check("python" in d.get("supported", []) and "rust" in d["supported"],
          "an unsupported language lists what IS supported", json.dumps(d))
    rec: list = []
    out = cc.run_tool({"code": HEADER_MODEL, "language": "py",
                       "mode": "continuation", "prefix": HEADER_PREFIX}, rec)
    check(out.startswith("check_code: python, continuation after a 10-line "
                         "prefix -- ") and "problem" in out,
          "a continuation result leads with what was checked and the verdict",
          out[:200])
    check("your line 1" in out and "column 12" in out,
          "and points at the line and the column to use", out)
    check(rec == [{"language": "python", "mode": "continuation", "ok": False,
                   "errors": 2}], "each call is recorded, without the code",
          json.dumps(rec))
    out = cc.run_tool({"code": VALID["rust"], "language": "rs"})
    check(out.startswith("check_code: rust -- parses"), "a clean result is "
          "one line to read first", out[:120])
    desc = cc.TOOL["function"]["description"]
    for phrase in ("Before you answer with code", "continuation", "prefix",
                   "context", "never runs the code", "syntax errors with line",
                   "checks only the code text you pass",
                   "read it with your client's own file tools"):
        check(phrase in desc, f"the description triggers on / says {phrase!r}")
    check(cc.TOOL["function"]["name"] == "check_code",
          "named per the conventions: verb first, spelled out")


# ---------------------------------------------------------------------------
# 6. Gating in the proxy
# ---------------------------------------------------------------------------
import proxy  # noqa: E402
import shomen  # noqa: E402

proxy.PREAMBLE = False
BONSAI = {"retrieval": False, "hints": False, "investigate": False,
          "fanout": 1, "effort": "medium"}
LCB_PROMPT = ("You are given an integer array nums. Return the number of "
              "subarrays whose sum is even.\n```python\nclass Solution:\n"
              "    def countEven(self, nums: List[int]) -> int:\n        \n```")


def names_of(out: dict) -> list[str]:
    return [t.get("function", {}).get("name") for t in out.get("tools") or []]


def prep(effort: str | None, header: dict | None, tools=None,
         content: str = LCB_PROMPT) -> dict:
    body = {"model": "yamadori",
            "messages": [{"role": "user", "content": content}],
            "_client_ip": "127.0.0.1"}
    if effort:
        body["reasoning_effort"] = effort
    if header is not None:
        body["_features"] = json.dumps(header)
    if tools:
        body["tools"] = tools
    return proxy.prepare(body)


def test_gating():
    # Since 2026-09-24 (operator) the check_code TOOL is never on main, at
    # any tier: the proxy checks code itself. `_tool_code` (client writes are
    # checked) from `medium`; `_fixup` (the second brain repairs) from `high`.
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        out = prep(effort, {"hints": False})
        check("check_code" not in names_of(out)
              and not [n for n in names_of(out) if n.startswith("find_")],
              f"{effort}: no check_code tool and no code tool on main",
              str(names_of(out)))
        want_check = effort in ("medium", "high", "xhigh", "max")
        want_fix = effort in ("high", "xhigh", "max")
        check(out["_tool_code"] is want_check and out["_fixup"] is want_fix,
              f"{effort}: writes checked={want_check}, repaired={want_fix}",
              f"{out['_tool_code']} {out['_fixup']}")
    out = prep(None, BONSAI)
    check(names_of(out) == [] and out["_repair"] is False,
          "the bonsai arm (retrieval forced off) stays bare", str(names_of(out)))
    out = prep(None, dict(BONSAI, check_code=True))
    check(names_of(out) == [] and out["_tool_code"] is True
          and not [m for m in out["messages"] if m.get("role") == "system"],
          "bonsai + check_code forced on: the write check, no tool, no "
          "system text", str(names_of(out)))
    out = prep(None, dict(BONSAI, check_code=True, repair=True))
    check(out["_repair"] is True and out["_fixup"] is True,
          "repair forced on by the header")
    out = prep("high", {"check_code": False, "hints": False, "fanout": 1})
    check(out["_tool_code"] is False, "the check forced off at high")
    mine = {"type": "function", "function": {
        "name": "check_code", "description": "the client's own",
        "parameters": {"type": "object", "properties": {}}}}
    out = prep("low", {"hints": False}, tools=[mine])
    got = [t for t in out["tools"] if t["function"]["name"] == "check_code"]
    check(len(got) == 1 and got[0]["function"]["description"]
          == "the client's own",
          "a client's own check_code passes through untouched")
    check("check_code" not in [t["function"]["name"]
                               for t in proxy.deep_thinking_tools()],
          "deep thinking is not given check_code")
    check("check_code" not in proxy.OUR_NAMES,
          "and it is not a tool the proxy runs any more (2026-09-24)")
    check(tiers.from_header('{"check_code": true, "repair": true}')
          == {"check_code": True, "repair": True}, "the header keys parse")
    check(tiers.from_header('{"check_code": "yes", "repair": 1}') is None,
          "non-boolean values are dropped, not coerced")
    # Operator, 2026-09-23: repair is ON at high and max, off below.
    for name in tiers.ORDER:
        want = name in ("high", "xhigh", "max")
        check(tiers.TIERS[name]["repair"] is want,
              f"repair is {'on' if want else 'off'} in tier {name} "
              f"unless a header forces it the other way")


# ---------------------------------------------------------------------------
# 7. The repair, against a fake upstream: a final answer's broken code goes
# to the second brain's fixup job (shomen.run, faked at shomen._post); main
# generates once (operator, 2026-09-24).
# ---------------------------------------------------------------------------
_script: list[dict] = []
_seen: list[dict] = []
HELPER: list[str] = []
_helper_seen: list[dict] = []


def reply(content: str = "", calls: list | None = None,
          finish: str | None = None) -> dict:
    return {"content": content, "calls": calls or [],
            "finish": finish or ("tool_calls" if calls else "stop")}


def _sse(r: dict) -> bytes:
    def ev(delta=None, finish=None):
        return b"data: " + json.dumps({"id": "u", "object": "chat.completion.chunk",
                                       "choices": [{"index": 0, "delta": delta or {},
                                                    "finish_reason": finish}]}).encode() + b"\n\n"
    out = [ev({"role": "assistant"})]
    s = r["content"]
    out += [ev({"content": s[i:i + 9]}) for i in range(0, len(s), 9)]
    for i, c in enumerate(r["calls"]):
        out.append(ev({"tool_calls": [dict(c, index=i)]}))
    out.append(ev({}, r["finish"]))
    out.append(b"data: " + json.dumps({"id": "u", "choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 10,
        "total_tokens": 110}}).encode() + b"\n\n")
    return b"".join(out) + b"data: [DONE]\n\n"


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/upstream/"):
            # proxy._warm: a render, then a zero-token /completion.
            data = json.dumps({"prompt": "".join(
                json.dumps(m) + "<|im_end|>" for m in body.get("messages")
                or [])} if self.path.endswith("apply-template")
                else {}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        _seen.append(body)
        if not _script:
            self.send_response(500)
            self.end_headers()
            return
        data = _sse(_script.pop(0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
proxy.UPSTREAM = f"http://127.0.0.1:{_srv.server_address[1]}"
# The second brain's door (mcp/model.py) goes to the same fake: no
# offline test may reach the live model server.
import model as _model  # noqa: E402
_model.UPSTREAM = proxy.UPSTREAM


def _helper_post(path, payload, timeout=3600):
    _helper_seen.append(json.loads(json.dumps(payload)))
    text = HELPER.pop(0) if HELPER else "(no code)"
    return {"choices": [{"message": {"content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


shomen._post = _helper_post
proxy._draw_seed = lambda prompt=None: {"word": "cedar", "token_id": 1,
                                        "u32": 2}

BROKEN_ANSWER = f"```python\n{HEADER_MODEL}\n```"
FIXED_ANSWER = f"Here it is:\n```python\n{HEADER_RIGHT}\n```"


def run_complete(script: list[dict], header: dict,
                 helper: list[str] = ()) -> dict:
    _script[:] = script
    _seen.clear()
    HELPER[:] = list(helper)
    _helper_seen.clear()
    return proxy.complete({
        "model": "yamadori", "_client_ip": "127.0.0.1",
        "_features": json.dumps(header),
        "messages": [{"role": "user",
                      "content": completion_question(HEADER_PREFIX)}]})


def test_the_fake_upstream_is_alive():
    _script[:] = [reply("pong")]
    _seen.clear()
    d = proxy._post("/v1/chat/completions", {"messages": []})
    check(d["choices"][0]["message"]["content"] == "pong",
          "fake upstream answers through the proxy's own reader",
          json.dumps(d)[:200])


def test_repair_runs_on_the_second_brain_and_delivers_in_place():
    d = run_complete([reply(BROKEN_ANSWER)], dict(BONSAI, repair=True),
                     [f"```python\n{HEADER_RIGHT}\n```"])
    check(len(_seen) == 1, "main generates ONCE: no feedback turn in its "
          "context", str(len(_seen)))
    user = (_helper_seen[0]["messages"][-1]["content"] if _helper_seen
            else "")
    check(len(_helper_seen) == 1 and "column 12" in user
          and "while not reachable >= target" in user
          and completion_question(HEADER_PREFIX)[:60] in user
          and "Inspiration word: cedar" in user,
          "the fixup job reads the request, the block and its exact "
          "problem, and the seed", user[:500])
    content = d["choices"][0]["message"]["content"]
    check(HEADER_RIGHT in content and HEADER_MODEL not in content
          # A repair note carries no seed line since 2026-09-24.
          and content.endswith("\n\nRepaired: 1 "
                               "code block that did not parse, fixed in 1 "
                               "round; the code above is the corrected "
                               "version."),
          "the repaired code is delivered IN PLACE, with the note",
          content[-300:])
    rp = (d.get("x_yamadori") or {}).get("repair") or {}
    check(rp.get("enabled") is True and rp.get("rounds") == 1
          and rp.get("errors_before", 0) >= 1 and rp.get("errors_after") == 0
          and rp.get("stopped") == "fixed" and rp.get("blocks_checked") == 1
          and rp.get("into") == "in_place",
          "x_yamadori.repair: rounds 1, errors before > 0, after 0, fixed, "
          "in place", json.dumps(rp))
    check(rp.get("check_mode") == "neither",
          "x_yamadori.repair.check_mode: the broken block parsed neither way",
          json.dumps(rp))
    check(not any(k.startswith("_") for k in rp), "no bookkeeping leaks into "
          "the record", json.dumps(rp))
    ck = (d.get("x_yamadori") or {}).get("check_code") or {}
    check(ck.get("offered") is False, "no check_code tool is offered",
          json.dumps(ck))


def test_repair_off_changes_nothing():
    d = run_complete([reply(BROKEN_ANSWER)], BONSAI)
    check(len(_seen) == 1 and not _helper_seen,
          "repair off: one generation, no fixup", str(len(_seen)))
    check(d["choices"][0]["message"]["content"] == BROKEN_ANSWER,
          "and the answer is returned exactly as written, broken or not")
    check((d.get("x_yamadori") or {}).get("repair") is None,
          "x_yamadori.repair is null when the pass is off")
    tools = _seen[0].get("tools") or []
    check(tools == [], "the bare arm sends no tools upstream", str(tools)[:100])


def test_repair_stops_at_the_cap_and_verifies_clean_code():
    bad = f"```python\n{HEADER_MODEL}\n```"
    d = run_complete([reply(BROKEN_ANSWER)], dict(BONSAI, repair=True),
                     [bad, bad, bad])
    rp = (d.get("x_yamadori") or {}).get("repair") or {}
    check(len(_seen) == 1 and len(_helper_seen) == 3
          and rp.get("stopped") == "not_fixed" and rp.get("rounds") == 3,
          "the fixup job stops at REPAIR_ROUNDS", json.dumps(rp))
    content = d["choices"][0]["message"]["content"]
    check(content.startswith(BROKEN_ANSWER)
          and "does not parse, and the repair did not fix it" in content,
          "and the answer stands, with a note that says so", content[-200:])
    d = run_complete([reply(FIXED_ANSWER)], dict(BONSAI, repair=True))
    rp = (d.get("x_yamadori") or {}).get("repair") or {}
    check(len(_seen) == 1 and not _helper_seen and rp.get("stopped") == "clean"
          and rp.get("rounds") == 0 and rp.get("errors_before") == 0
          and d["choices"][0]["message"]["content"]
          == FIXED_ANSWER + "\n\nVerified: the python code above parses.",
          "a clean first answer: a one-line 'Verified' note, no generation",
          json.dumps(rp))
    d = run_complete([reply("It is 42.")], dict(BONSAI, repair=True))
    rp = (d.get("x_yamadori") or {}).get("repair") or {}
    check(rp.get("stopped") == "no_code" and len(_seen) == 1
          and rp.get("check_mode") is None
          and d["choices"][0]["message"]["content"] == "It is 42.",
          "an answer without code is left alone", json.dumps(rp))
    d = run_complete([reply("```python\nx = (\n", finish="length")],
                     dict(BONSAI, repair=True))
    rp = (d.get("x_yamadori") or {}).get("repair") or {}
    check(rp.get("stopped") is None and len(_seen) == 1 and not _helper_seen,
          "a budget event is not repaired", json.dumps(rp))


def test_check_code_forced_on_offers_no_tool():
    d = run_complete([reply(FIXED_ANSWER)], dict(BONSAI, check_code=True))
    first_tools = [t["function"]["name"] for t in (_seen[0].get("tools") or [])]
    ck = (d.get("x_yamadori") or {}).get("check_code") or {}
    check(first_tools == [] and ck.get("offered") is False,
          "bonsai+check: no tool reaches the model; the proxy checks client "
          "writes itself", str(first_tools))


def test_repair_on_the_streamed_path():
    _script[:] = [reply(BROKEN_ANSWER)]
    _seen.clear()
    HELPER[:] = [f"```python\n{HEADER_RIGHT}\n```"]
    _helper_seen.clear()
    events = []
    for b in proxy.stream_body({
            "model": "yamadori", "stream": True, "_client_ip": "127.0.0.1",
            "_features": json.dumps(dict(BONSAI, repair=True)),
            "messages": [{"role": "user",
                          "content": completion_question(HEADER_PREFIX)}]},
            "yamadori"):
        if b.strip() == b"data: [DONE]":
            continue
        events.append(json.loads(b[6:].decode()))
    content = "".join((e["choices"][0]["delta"].get("content") or "")
                      for e in events if e.get("choices"))
    check(len(_seen) == 1, "streamed: main generates once", str(len(_seen)))
    check(content.startswith(BROKEN_ANSWER + "\n\nRepaired: 1 code block")
          and content.rstrip().endswith(f"{HEADER_RIGHT.rstrip()}\n```"),
          "the delivered first answer is followed by the repaired block",
          content[-300:])
    xs = [e.get("x_yamadori") for e in events if e.get("x_yamadori")]
    rp = (xs[-1] if xs else {}).get("repair") or {}
    check(rp.get("rounds") == 1 and rp.get("stopped") == "fixed"
          and rp.get("into") == "appended",
          "the final chunk carries x_yamadori.repair", json.dumps(rp))


# ---------------------------------------------------------------------------
TESTS = [
    test_syntax_errors_have_positions_per_language,
    test_python_uses_the_compiler_as_the_authority,
    test_style_is_inferred_from_the_existing_code,
    test_config_text_wins_over_inference,
    test_continuation_catches_the_livebench_shapes,
    test_continuation_in_brace_languages,
    test_whole_mode_python_excerpts_and_loops,
    test_fenced_blocks_and_review_answer,
    test_the_general_rule,
    test_formatters_follow_the_inferred_style,
    test_a_missing_or_broken_formatter_is_reported_not_raised,
    test_no_user_code_runs_and_nothing_outside_temp_is_opened,
    test_the_tool_answers_every_call,
    test_gating,
    test_the_fake_upstream_is_alive,
    test_repair_runs_on_the_second_brain_and_delivers_in_place,
    test_repair_off_changes_nothing,
    test_repair_stops_at_the_cap_and_verifies_clean_code,
    test_check_code_forced_on_offers_no_tool,
    test_repair_on_the_streamed_path,
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
    _srv.shutdown()
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
