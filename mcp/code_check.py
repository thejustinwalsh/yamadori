#!/usr/bin/env python
"""The code parsers: does this code parse, and where are the code blocks in a
text.

  normalize_language   a fence tag or an extension -> a canonical language
  syntax_errors        every syntax error (a reference parser where the
                       language has one -- Python's compiler, json, tomllib,
                       yaml -- else tree-sitter), capped at MAX_ERRORS
  parse_tree           tree-sitter's root node
  fenced_blocks        Markdown fenced code blocks

Read by mcp/route.py (does the prompt carry code that parses?), mcp/tool_code.py
(an .html write's inline scripts) and the skills pipeline's proof
(mcp/skill_prove.py).

REMOVED 2026-09-29 (docs/REMOVED.md; the way back is commit e360d37): the
check built on these for the code check of client writes and the answer
repair -- the formatters (prettier, ruff format, rustfmt, clang-format, and
the pinned tools/format), the lint, the style inference, the continuation
analysis of a completion prompt, the answer review (review_answer) -- and
the check_code tool's surface (TOOL, run_tool, render), off main since
2026-09-24.

THE CONTRACT (hard)

  - The server never reads the user's disk. Code arrives only as TEXT.
    Nothing here takes a path.
  - The server never executes user code. Python is checked with `compile()`
    (which builds a code object and does not run it) and every language with
    tree-sitter. JSON/TOML/YAML use `json.loads`, `tomllib.loads` and
    `yaml.safe_load` -- the safe loader, which constructs no Python objects.

Parsing follows PROTOCOL rule 8: structure is read with a parser (tree-sitter,
Python's own compiler), never with regular expressions. Markdown fences are
flat text and are read line by line.
"""
from __future__ import annotations

import json
import os
import re
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Errors reported per text: enough to locate the first few.
MAX_ERRORS = 8

# canonical name -> tree-sitter grammar and file extension.
LANGS = {
    "python":     {"grammar": "python",     "ext": "py"},
    "typescript": {"grammar": "typescript", "ext": "ts"},
    "tsx":        {"grammar": "tsx",        "ext": "tsx"},
    "javascript": {"grammar": "javascript", "ext": "js"},
    "jsx":        {"grammar": "javascript", "ext": "jsx"},
    "rust":       {"grammar": "rust",       "ext": "rs"},
    "c":          {"grammar": "c",          "ext": "c"},
    "cpp":        {"grammar": "cpp",        "ext": "cpp"},
    "json":       {"grammar": "json",       "ext": "json"},
    "toml":       {"grammar": "toml",       "ext": "toml"},
    "yaml":       {"grammar": "yaml",       "ext": "yaml"},
}
ALIASES = {
    "py": "python", "python3": "python", "py3": "python",
    "ts": "typescript", "mts": "typescript", "cts": "typescript",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "node": "javascript", "ecmascript": "javascript",
    "rs": "rust",
    "h": "c",
    "c++": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp",
    "hxx": "cpp",
    "yml": "yaml",
}
# The languages that are CODE: a JSON or YAML block in a chat answer is very
# often an illustration with `...` in it, not code to check.
CODE_LANGS = {"python", "typescript", "tsx", "javascript", "jsx", "rust",
              "c", "cpp"}


def normalize_language(name) -> str | None:
    if not isinstance(name, str):
        return None
    k = name.strip().lower().lstrip(".")
    k = ALIASES.get(k, k)
    return k if k in LANGS else None


# ---------------------------------------------------------------------------
# Syntax
# ---------------------------------------------------------------------------
_parsers: dict = {}


def _parser(grammar: str):
    if grammar not in _parsers:
        from tree_sitter_language_pack import get_parser
        _parsers[grammar] = get_parser(grammar)
    return _parsers[grammar]


def parse_tree(code: str, lang: str):
    """Tree-sitter root node for `code` in canonical language `lang`."""
    return _parser(LANGS[lang]["grammar"]).parse(code.encode("utf-8")).root_node


def _col(lines_b: list[bytes], row: int, byte_col: int) -> int:
    """1-based CHARACTER column from tree-sitter's 0-based byte column."""
    if 0 <= row < len(lines_b):
        return len(lines_b[row][:byte_col].decode("utf-8", "replace")) + 1
    return byte_col + 1


def tree_errors(code: str, lang: str) -> list[dict]:
    """ERROR and MISSING nodes, outermost only, in document order."""
    root = parse_tree(code, lang)
    if not root.has_error:
        return []
    lines_b = code.encode("utf-8").split(b"\n")
    out: list[dict] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.is_missing:
            row, bc = n.start_point
            out.append({"line": row + 1, "col": _col(lines_b, row, bc),
                        "message": f"missing {n.type!r}",
                        "source": "tree-sitter"})
            continue
        if n.type == "ERROR":
            row, bc = n.start_point
            snippet = (n.text or b"").decode("utf-8", "replace").strip()
            snippet = snippet.splitlines()[0][:40] if snippet else ""
            out.append({"line": row + 1, "col": _col(lines_b, row, bc),
                        "message": (f"unexpected {snippet!r}" if snippet
                                    else "unexpected end of input"),
                        "source": "tree-sitter"})
            continue
        if n.has_error:
            stack.extend(reversed(n.children))
    out.sort(key=lambda e: (e["line"], e["col"]))
    return out


# The filename handed to compile(). When a compile-stage error (e.g. `return`
# outside a function) carries no source text, CPython tries to OPEN the named
# file to quote the offending line. "<check_code>" was therefore an attempted
# read of a file of that name in the server's working directory -- found by
# the audit hook in mcp/test_code_check.py. This path is inside the temp dir
# and its directory is never created, so the attempt fails without touching
# anything that exists.
_COMPILE_NAME = os.path.join(tempfile.gettempdir(),
                             "yamadori-check_code-never-created", "code.py")


def _toml_position(e: Exception) -> tuple[int, int, str]:
    """tomllib's error position. Python 3.14 adds lineno/colno; 3.13 only
    writes them into the message, "(at line 3, column 5)" -- a flat string,
    so it is read as one."""
    msg = str(e)
    if getattr(e, "lineno", None):
        return e.lineno, getattr(e, "colno", 1) or 1, getattr(e, "msg", msg)
    m = re.search(r"\(at line (\d+), column (\d+)\)", msg)
    if m:
        return int(m.group(1)), int(m.group(2)), msg[:m.start()].strip()
    return 1, 1, msg


def _reference_error(code: str, lang: str) -> tuple[bool, dict | None]:
    """(has_reference_parser, error) from the language's own parser."""
    if lang == "python":
        try:
            # compile(), not exec(): a code object is built and discarded.
            compile(code, _COMPILE_NAME, "exec", dont_inherit=True)
            return True, None
        except SyntaxError as e:          # includes IndentationError, TabError
            return True, {"line": e.lineno or 1, "col": e.offset or 1,
                          "message": e.msg, "source": "python"}
        except (ValueError, RecursionError, MemoryError, OverflowError) as e:
            return True, {"line": 1, "col": 1,
                          "message": f"{type(e).__name__}: {e}",
                          "source": "python"}
    if lang == "json":
        try:
            json.loads(code)
            return True, None
        except json.JSONDecodeError as e:
            return True, {"line": e.lineno, "col": e.colno, "message": e.msg,
                          "source": "json"}
    if lang == "toml":
        import tomllib
        try:
            tomllib.loads(code)
            return True, None
        except tomllib.TOMLDecodeError as e:
            line, col, msg = _toml_position(e)
            if "end of document" in msg or line > code.count("\n") + 1:
                line = max(code.rstrip("\n").count("\n") + 1, 1)
            return True, {"line": line, "col": col, "message": msg,
                          "source": "toml"}
    if lang == "yaml":
        try:
            import yaml
        except ImportError:
            return False, None
        try:
            list(yaml.safe_load_all(code))     # the SAFE loader: no objects
            return True, None
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            return True, {"line": (mark.line + 1) if mark else 1,
                          "col": (mark.column + 1) if mark else 1,
                          "message": str(getattr(e, "problem", None) or e),
                          "source": "yaml"}
    return False, None


def syntax_errors(code: str, lang: str) -> list[dict]:
    """Every syntax error found, capped at MAX_ERRORS.

    Where the language has a reference parser (Python's compiler, json,
    tomllib, yaml) IT decides whether the code is valid: tree-sitter grammars
    can lag the language (new syntax), and a false syntax error would send a
    model to "fix" correct code. When the reference parser rejects the code,
    its error comes first and tree-sitter adds positions on other lines. For
    the brace languages tree-sitter is the parser.
    """
    has_ref, ref = _reference_error(code, lang)
    try:
        ts = tree_errors(code, lang)
    except Exception as e:                                       # noqa: BLE001
        ts = [] if has_ref else [{"line": 1, "col": 1, "source": "tree-sitter",
                                  "message": f"parser unavailable: {type(e).__name__}"}]
    if has_ref:
        if ref is None:
            return []
        seen = {ref["line"]}
        extra = [e for e in ts if e["line"] not in seen]
        return ([ref] + extra)[:MAX_ERRORS]
    return ts[:MAX_ERRORS]


# ---------------------------------------------------------------------------
# Fenced code blocks
# ---------------------------------------------------------------------------
def fenced_blocks(text: str) -> list[dict]:
    """Markdown fenced code blocks, CommonMark-style: an opening run of 3+
    backticks or tildes (indented at most 3 spaces) and a closing run of the
    same character at least as long. An unclosed fence runs to the end --
    and says so (`closed` False): an unclosed fence is a truncated reply. `fence` is the opening run itself.

    `line` is the 1-based line of the first body line; `end_line` is the
    1-based line of the closing fence (one past the last line when the
    fence is unclosed), so the fence spans lines line-1 .. end_line."""
    out = []
    lines = (text or "").split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.lstrip(" ")
        ind = len(ln) - len(s)
        if ind <= 3 and (s.startswith("```") or s.startswith("~~~")):
            ch = s[0]
            n = len(s) - len(s.lstrip(ch))
            info = s[n:].strip()
            if ch == "`" and "`" in info:
                i += 1
                continue
            body, j, closed = [], i + 1, False
            while j < len(lines):
                t = lines[j].lstrip(" ")
                if (len(lines[j]) - len(t)) <= 3 and t.startswith(ch * n) \
                        and not t.lstrip(ch).strip():
                    closed = True
                    break
                body.append(lines[j][ind:] if lines[j][:ind].strip() == ""
                            else lines[j])
                j += 1
            out.append({"lang": (info.split() or [""])[0].lower(),
                        "code": "\n".join(body), "line": i + 2,
                        "end_line": j + 1, "closed": closed,
                        "fence": ch * n})
            i = j + 1
            continue
        i += 1
    return out
