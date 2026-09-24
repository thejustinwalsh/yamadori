#!/usr/bin/env python
"""check_code: does this code parse, does it fit where it goes, and what would
it look like in the existing file's style.

WHY THIS EXISTS

LiveBench coding, bare model, 2026-09-23 (run lb-20260923, bonsai arm): of the
five coding_completion misses, four were mechanical, not reasoning. Writing a
program from scratch scored 10/11 (LCB_generation) in the same run. The
completions were wrong in ways a parser sees in milliseconds:

  - the continuation was dedented out of the `while` loop the prefix had just
    opened, so the loop body never changed its condition: an infinite loop
  - the continuation dedented out of the method, so `return` landed in the
    class body: SyntaxError
  - the prefix ended inside an open docstring and the continuation never
    closed it, so the whole answer was string contents
  - the prefix ended on `while ...:` and the continuation's first line was
    not indented under it

The n is tiny (5 misses of 10 completions) and the categories are the
operator's reading of those five; nothing here claims a score change. That is
what the `bonsai+check` LiveBench arm is for (bench/livebench/drive.py).

THE CONTRACT (hard)

  - The server never reads the user's disk. Code arrives only as TEXT in a
    tool argument or the conversation. Nothing here takes a path.
  - The server never executes user code. Python is checked with `compile()`
    (which builds a code object and does not run it) and every language with
    tree-sitter. JSON/TOML/YAML use `json.loads`, `tomllib.loads` and
    `yaml.safe_load` -- the safe loader, which constructs no Python objects.
  - Formatters run in a subprocess, on a file in a fresh temp directory that
    is also their working directory, with a timeout, a minimal environment
    whose HOME/APPDATA point into that directory, and every project-config
    lookup switched off: `ruff --isolated`, `prettier --no-config
    --no-editorconfig`, `rustfmt --config-path <our file>`, `clang-format
    --style={...}`. The only configuration any formatter sees is what the
    caller passed as `config_text` or what was inferred from `context`.
    None of these formatters has a network feature; nothing here opens a
    socket.

WHAT IT NEVER DOES

It never alters the model's code silently. A formatted version is OFFERED in
the result; any change to an answer is made by the model, after feedback
(`review_answer`, used by the proxy's opt-in repair pass).

Parsing follows PROTOCOL rule 8: structure is read with a parser (tree-sitter,
Python's own tokenizer and compiler), never with regular expressions.
Markdown fences and config files are flat text and are read line by line.
"""
from __future__ import annotations

import difflib
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tokenize
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

TOOL_NAME = "check_code"

# Pinned formatters live here: tools/format/package.json + package-lock.json
# (prettier). ruff is the stack interpreter's own (the same one
# scripts/run_tests.py lints with); rustfmt comes from rustup; clang-format is
# optional and absent on this machine as of 2026-09-23.
FORMAT_DIR = os.environ.get("YAMADORI_FORMAT_DIR",
                            os.path.join(ROOT, "tools", "format"))
FORMAT_TIMEOUT = float(os.environ.get("YAMADORI_FORMAT_TIMEOUT", "20"))

# Bigger than any single file a model should be pasting into one call, small
# enough that parsing and a formatter run stay well under a second.
MAX_CHARS = 200_000
MAX_ERRORS = 8

# canonical name -> tree-sitter grammar, formatter, file extension, and
# whether formatting is offered at all (config formats: syntax only).
LANGS = {
    "python":     {"grammar": "python",     "formatter": "ruff",         "ext": "py"},
    "typescript": {"grammar": "typescript", "formatter": "prettier",     "ext": "ts"},
    "tsx":        {"grammar": "tsx",        "formatter": "prettier",     "ext": "tsx"},
    "javascript": {"grammar": "javascript", "formatter": "prettier",     "ext": "js"},
    "jsx":        {"grammar": "javascript", "formatter": "prettier",     "ext": "jsx"},
    "rust":       {"grammar": "rust",       "formatter": "rustfmt",      "ext": "rs"},
    "c":          {"grammar": "c",          "formatter": "clang-format", "ext": "c"},
    "cpp":        {"grammar": "cpp",        "formatter": "clang-format", "ext": "cpp"},
    "json":       {"grammar": "json",       "formatter": None,           "ext": "json"},
    "toml":       {"grammar": "toml",       "formatter": None,           "ext": "toml"},
    "yaml":       {"grammar": "yaml",       "formatter": None,           "ext": "yaml"},
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
# The repair pass looks only at code. A JSON or YAML block in a chat answer is
# very often an illustration with `...` in it, and nagging about that would be
# a false alarm; the tool still checks them when asked.
CODE_LANGS = {"python", "typescript", "tsx", "javascript", "jsx", "rust",
              "c", "cpp"}
BRACE_LANGS = {"typescript", "tsx", "javascript", "jsx", "rust", "c", "cpp"}


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
# Style inference
# ---------------------------------------------------------------------------
STD_WIDTHS = (80, 88, 100, 120)


def _leading(line: str) -> str:
    return line[:len(line) - len(line.lstrip(" \t"))]


def _indent_width(ws: str, tab: int = 8) -> int:
    return len(ws.expandtabs(tab))


def infer_indent(text: str) -> dict:
    """Indent unit from the most common positive leading-whitespace delta.

    Tabs win when more indented lines start with a tab than with a space.
    Lines that continue a block comment (` * ...`) are skipped: their one
    space is alignment, not an indent level.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.lstrip().startswith("*")]
    tabs = sum(1 for ln in lines if ln.startswith("\t"))
    spaces = sum(1 for ln in lines if ln.startswith(" "))
    if not tabs and not spaces:
        return {}
    if tabs > spaces:
        return {"indent": "tab", "_evidence": {"tab_lines": tabs,
                                               "space_lines": spaces}}
    deltas: Counter = Counter()
    prev = None
    for ln in lines:
        if ln.startswith("\t"):
            prev = None
            continue
        w = len(_leading(ln))
        if prev is not None and w > prev:
            deltas[w - prev] += 1
        prev = w
    if not deltas:
        return {}
    usual = [(d, c) for d, c in deltas.most_common() if d in (2, 3, 4, 8)]
    unit = (usual or deltas.most_common())[0][0]
    return {"indent": "space", "indent_width": unit,
            "_evidence": {"deltas": dict(deltas.most_common(4))}}


def infer_width(text: str) -> dict:
    lines = [ln.expandtabs(4) for ln in text.splitlines() if ln.strip()]
    if len(lines) < 10:
        return {}          # too little to say anything about a line limit
    longest = max(len(ln) for ln in lines)
    for w in STD_WIDTHS:
        if longest <= w:
            return {"width": w, "_evidence": {"longest_line": longest}}
    return {"width": longest, "_evidence": {"longest_line": longest}}


def _python_quotes(text: str) -> Counter:
    q: Counter = Counter()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            s = None
            if tok.type == tokenize.STRING:
                s = tok.string.lstrip("rRbBuUfF")
            elif tok.type == getattr(tokenize, "FSTRING_START", -1):
                s = tok.string.lstrip("rRbBuUfF")
            if not s or s[:3] in ('"""', "'''"):
                continue
            if s[0] in "'\"":
                q[s[0]] += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass                        # a partial file: keep what was read
    return q


_JS_STMTS = {"expression_statement", "lexical_declaration",
             "variable_declaration", "return_statement", "import_statement",
             "throw_statement", "break_statement", "continue_statement",
             "type_alias_declaration", "do_statement"}
_LISTS = {"array", "object", "arguments", "formal_parameters", "named_imports",
          "object_pattern", "array_pattern", "object_type", "tuple_type",
          "export_clause",
          # rust
          "array_expression", "field_declaration_list",
          "field_initializer_list", "enum_variant_list", "parameters",
          "tuple_expression", "use_list"}


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _tree_style(text: str, lang: str) -> dict:
    out: dict = {}
    try:
        root = parse_tree(text, lang)
    except Exception:                                            # noqa: BLE001
        return out
    q: Counter = Counter()
    semi: Counter = Counter()
    trail: Counter = Counter()
    for n in _walk(root):
        t = n.type
        if lang in ("typescript", "tsx", "javascript", "jsx"):
            if t == "string" and n.text:
                c = n.text[:1].decode("utf-8", "replace")
                if c in "'\"":
                    q[c] += 1
            elif t in _JS_STMTS and n.text and not n.has_error:
                semi["yes" if n.text.rstrip().endswith(b";") else "no"] += 1
        if t in _LISTS and n.start_point[0] != n.end_point[0] \
                and n.named_child_count and not n.has_error:
            kids = n.children
            if len(kids) >= 2:
                before_close = kids[-2]
                trail["yes" if before_close.type == "," else "no"] += 1
    if q:
        out["quotes"] = "single" if q["'"] > q['"'] else "double"
        out.setdefault("_evidence", {})["quotes"] = dict(q)
    if semi:
        out["semicolons"] = semi["yes"] >= semi["no"]
        out.setdefault("_evidence", {})["semicolons"] = dict(semi)
    if trail:
        out["trailing_commas"] = "all" if trail["yes"] > trail["no"] else "none"
        out.setdefault("_evidence", {})["trailing_commas"] = dict(trail)
    return out


def infer_style(text: str, lang: str) -> dict:
    """Style facts evident in `text`. A key is present only with evidence."""
    if not text or not text.strip():
        return {}
    style: dict = {}
    ev: dict = {}
    for part in (infer_indent(text), infer_width(text)):
        ev.update(part.pop("_evidence", {}) or {})
        style.update(part)
    if lang == "python":
        q = _python_quotes(text)
        if q:
            style["quotes"] = "single" if q["'"] > q['"'] else "double"
            ev["quotes"] = dict(q)
    elif lang in BRACE_LANGS:
        part = _tree_style(text, lang)
        ev.update(part.pop("_evidence", {}) or {})
        style.update(part)
    if ev:
        style["_evidence"] = ev
    return style


# ---------------------------------------------------------------------------
# Explicit configuration text (wins over inference)
# ---------------------------------------------------------------------------
def _expand_braces(pat: str) -> list[str]:
    """`*.{js,ts}` -> [`*.js`, `*.ts`] (one level, as editorconfig uses)."""
    a, b = pat.find("{"), pat.find("}")
    if a == -1 or b < a:
        return [pat]
    return [pat[:a] + alt + pat[b + 1:] for alt in pat[a + 1:b].split(",")]


def _parse_editorconfig(text: str, ext: str) -> dict:
    """The keys that apply to a file named check.<ext>. Later sections win,
    as in editorconfig itself. Not configparser: editorconfig allows keys
    before any section and repeated sections."""
    found: dict = {}
    applies = False          # the preamble (root = true) applies to no file
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            pat = line[1:-1].strip()
            name = f"check.{ext}"
            applies = any(fnmatch.fnmatch(name, p.lstrip("/").split("/")[-1])
                          for p in _expand_braces(pat))
            continue
        if applies and "=" in line:
            k, v = line.split("=", 1)
            found[k.strip().lower()] = v.strip().lower()
    style: dict = {}
    if found.get("indent_style") in ("tab", "space"):
        style["indent"] = found["indent_style"]
    for key in ("indent_size", "tab_width"):
        if found.get(key, "").isdigit():
            style["indent_width"] = int(found[key])
            break
    if found.get("max_line_length", "").isdigit():
        style["width"] = int(found["max_line_length"])
    if found.get("quote_type") in ("single", "double"):
        style["quotes"] = found["quote_type"]
    return style


def _from_prettier(d: dict) -> dict:
    s: dict = {}
    if isinstance(d.get("useTabs"), bool):
        s["indent"] = "tab" if d["useTabs"] else "space"
    if isinstance(d.get("tabWidth"), int):
        s["indent_width"] = d["tabWidth"]
    if isinstance(d.get("printWidth"), int):
        s["width"] = d["printWidth"]
    if isinstance(d.get("singleQuote"), bool):
        s["quotes"] = "single" if d["singleQuote"] else "double"
    if isinstance(d.get("semi"), bool):
        s["semicolons"] = d["semi"]
    if d.get("trailingComma") in ("all", "none", "es5"):
        s["trailing_commas"] = d["trailingComma"]
    return s


def _from_toml(d: dict) -> dict:
    s: dict = {}
    ruff = ((d.get("tool") or {}).get("ruff") if isinstance(d.get("tool"), dict)
            else None) or (d if ("line-length" in d or "format" in d) else None)
    if isinstance(ruff, dict):
        if isinstance(ruff.get("line-length"), int):
            s["width"] = ruff["line-length"]
        if isinstance(ruff.get("indent-width"), int):
            s["indent_width"] = ruff["indent-width"]
        fmt = ruff.get("format") if isinstance(ruff.get("format"), dict) else {}
        if fmt.get("quote-style") in ("single", "double"):
            s["quotes"] = fmt["quote-style"]
        if fmt.get("indent-style") in ("tab", "space"):
            s["indent"] = fmt["indent-style"]
    # rustfmt.toml
    if isinstance(d.get("max_width"), int):
        s["width"] = d["max_width"]
    if isinstance(d.get("tab_spaces"), int):
        s["indent_width"] = d["tab_spaces"]
    if isinstance(d.get("hard_tabs"), bool):
        s["indent"] = "tab" if d["hard_tabs"] else "space"
    if isinstance(d.get("edition"), str):
        s["edition"] = d["edition"]
    return s


def parse_config(text: str | None, lang: str) -> tuple[dict, str | None]:
    """(style, what it was read as). Tries JSON (.prettierrc), TOML
    (pyproject / ruff.toml / rustfmt.toml), editorconfig, then YAML
    (.prettierrc.yaml). An unrecognised text yields ({}, None)."""
    if not text or not text.strip():
        return {}, None
    ext = LANGS[lang]["ext"]
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            s = _from_prettier(d)
            if s:
                return s, "prettier (json)"
    except ValueError:
        pass
    try:
        import tomllib
        d = tomllib.loads(text)
        s = _from_toml(d)
        if s:
            return s, "toml (pyproject/ruff/rustfmt)"
    except Exception:                                            # noqa: BLE001
        pass
    s = _parse_editorconfig(text, ext)
    if s:
        return s, "editorconfig"
    try:
        import yaml
        d = yaml.safe_load(text)
        if isinstance(d, dict):
            s = _from_prettier(d)
            if s:
                return s, "prettier (yaml)"
    except Exception:                                            # noqa: BLE001
        pass
    return {}, None


# Formatter defaults, stated so a result can say where each value came from.
DEFAULTS = {
    "python":     {"indent": "space", "indent_width": 4, "width": 88, "quotes": "double"},
    "typescript": {"indent": "space", "indent_width": 2, "width": 80, "quotes": "double",
                   "semicolons": True, "trailing_commas": "all"},
    "rust":       {"indent": "space", "indent_width": 4, "width": 100},
    "c":          {"indent": "space", "indent_width": 2, "width": 80},
}
for _k in ("tsx", "javascript", "jsx"):
    DEFAULTS[_k] = DEFAULTS["typescript"]
DEFAULTS["cpp"] = DEFAULTS["c"]
STYLE_KEYS = ("indent", "indent_width", "width", "quotes", "semicolons",
              "trailing_commas", "edition")


def resolve_style(lang: str, code: str, context: str | None,
                  config_text: str | None) -> tuple[dict, dict, dict]:
    """(style, source per key, notes). Precedence: config > context > the
    code itself > formatter default. Width never comes from the code itself:
    a snippet's longest line says nothing about the file's limit."""
    cfg, cfg_kind = parse_config(config_text, lang)
    ctx = infer_style(context or "", lang)
    own = infer_style(code, lang)
    own.pop("width", None)
    style: dict = {}
    src: dict = {}
    for key in STYLE_KEYS:
        for name, layer in (("config", cfg), ("context", ctx), ("code", own),
                            ("default", DEFAULTS.get(lang, {}))):
            if key in layer:
                style[key] = layer[key]
                src[key] = name
                break
    notes = {}
    if config_text and not cfg:
        notes["config_text"] = ("no recognised settings (read as prettier "
                                "JSON/YAML, TOML, editorconfig); inferred "
                                "style used instead")
    elif cfg_kind:
        notes["config_text"] = f"read as {cfg_kind}"
    return style, src, notes


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def _which(name: str) -> str | None:
    if name == "ruff":
        try:
            import ruff  # the stack interpreter's pinned ruff
            p = ruff.find_ruff_bin()
            if p and os.path.exists(p):
                return p
        except Exception:                                        # noqa: BLE001
            pass
        return shutil.which("ruff")
    if name == "prettier":
        js = os.path.join(FORMAT_DIR, "node_modules", "prettier", "bin",
                          "prettier.cjs")
        return js if os.path.exists(js) and shutil.which("node") else None
    if name == "rustfmt":
        p = shutil.which("rustfmt")
        if p:
            return p
        home = os.path.join(os.path.expanduser("~"), ".cargo", "bin",
                            "rustfmt" + (".exe" if os.name == "nt" else ""))
        return home if os.path.exists(home) else None
    if name == "clang-format":
        return shutil.which("clang-format")
    return None


REMEDY = {
    "ruff": "operator: install ruff into the stack interpreter "
            "(it ships with the env scripts/run_tests.py uses)",
    "prettier": "operator: run `npm ci` in tools/format (pinned by its "
                "package-lock.json); node must be on PATH",
    "rustfmt": "operator: `rustup component add rustfmt`",
    "clang-format": "operator: install LLVM's clang-format and put it on PATH",
}


def formatter_status(name: str | None) -> dict:
    if not name:
        return {"name": None, "available": False}
    path = _which(name)
    out = {"name": name, "available": bool(path)}
    if not path:
        out["remedy"] = REMEDY[name]
    return out


def _isolated_env(tmp: str) -> dict:
    """A minimal environment. HOME and the config roots point into `tmp`, so
    a formatter that looks for a user-level config finds nothing. rustup's
    proxy still needs to find its toolchains, so RUSTUP_HOME/CARGO_HOME are
    passed as they are."""
    home = os.path.expanduser("~")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": tmp, "TMP": tmp, "TMPDIR": tmp,
        "HOME": tmp, "USERPROFILE": tmp, "APPDATA": tmp,
        "LOCALAPPDATA": tmp, "XDG_CONFIG_HOME": tmp,
        "RUSTUP_HOME": os.environ.get("RUSTUP_HOME")
        or os.path.join(home, ".rustup"),
        "CARGO_HOME": os.environ.get("CARGO_HOME")
        or os.path.join(home, ".cargo"),
        "NO_COLOR": "1",
    }
    return {k: v for k, v in env.items() if v}


def _command(name: str, exe: str, path: str, lang: str, style: dict,
             tmp: str) -> tuple[list[str], bool]:
    """(argv, output_on_stdout). False means the file is rewritten in place."""
    width = int(style.get("width") or 0)
    iw = int(style.get("indent_width") or 4)
    tabs = style.get("indent") == "tab"
    if name == "ruff":
        cmd = [exe, "format", "--isolated", "--no-cache", "--quiet",
               "--config", f"indent-width={iw}",
               "--config", f'format.indent-style="{"tab" if tabs else "space"}"',
               "--config", f'format.quote-style="{style.get("quotes") or "double"}"']
        if width:
            cmd += ["--line-length", str(width)]
        return cmd + [path], False
    if name == "prettier":
        node = shutil.which("node") or "node"
        cmd = [node, exe, "--no-config", "--no-editorconfig",
               "--ignore-path", os.path.join(tmp, ".no-ignore"),
               "--tab-width", str(iw)]
        if tabs:
            cmd.append("--use-tabs")
        if width:
            cmd += ["--print-width", str(width)]
        if style.get("quotes") == "single":
            cmd.append("--single-quote")
        if style.get("semicolons") is False:
            cmd.append("--no-semi")
        if style.get("trailing_commas") in ("all", "none", "es5"):
            cmd += ["--trailing-comma", style["trailing_commas"]]
        return cmd + [path], True
    if name == "rustfmt":
        conf = os.path.join(tmp, "rustfmt.toml")
        with open(conf, "w", encoding="utf-8") as f:
            f.write(f"tab_spaces = {iw}\nhard_tabs = {str(tabs).lower()}\n")
            if width:
                f.write(f"max_width = {width}\n")
        edition = str(style.get("edition") or "2021")
        return [exe, "--edition", edition, "--config-path", conf, path], False
    if name == "clang-format":
        spec = (f"{{BasedOnStyle: LLVM, IndentWidth: {iw}, "
                f"UseTab: {'Always' if tabs else 'Never'}, TabWidth: {iw}"
                + (f", ColumnLimit: {width}" if width else "") + "}")
        return [exe, f"--style={spec}", path], True
    raise ValueError(name)


def run_formatter(code: str, lang: str, style: dict) -> tuple[str | None, dict]:
    """Format `code` in a fresh temp dir. Never raises: a missing or failing
    formatter is reported in the info dict, and `formatted` is None."""
    name = LANGS[lang]["formatter"]
    info = formatter_status(name)
    if not info["available"]:
        return None, info
    exe = _which(name)
    with tempfile.TemporaryDirectory(prefix="yamadori_fmt_") as tmp:
        path = os.path.join(tmp, f"check.{LANGS[lang]['ext']}")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        try:
            cmd, stdout = _command(name, exe, path, lang, style, tmp)
            r = subprocess.run(
                cmd, cwd=tmp, env=_isolated_env(tmp), capture_output=True,
                stdin=subprocess.DEVNULL, timeout=FORMAT_TIMEOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            info["error"] = f"timed out after {FORMAT_TIMEOUT:.0f}s"
            return None, info
        except OSError as e:
            info["error"] = f"could not start: {e.strerror or e}"
            info["available"] = False
            info["remedy"] = REMEDY[name]
            return None, info
        if r.returncode != 0:
            err = (r.stderr or r.stdout or b"").decode("utf-8", "replace")
            info["error"] = (err.strip().splitlines() or ["exit "
                                                          f"{r.returncode}"])[0][:200]
            return None, info
        if stdout:
            out = r.stdout.decode("utf-8", "replace")
        else:
            with open(path, encoding="utf-8") as f:
                out = f.read()
    return out.replace("\r\n", "\n"), info


# ---------------------------------------------------------------------------
# A light lint, for whole files (the proxy's tool-call check, mcp/tool_code.py)
# ---------------------------------------------------------------------------
# flake8's "fatal" selection, the one its own docs and GitHub's starter
# workflow run to stop a build: E9 (syntax / IO errors), F63 (invalid
# comparisons: `is` with a literal), F7 (statements in the wrong place:
# `return`/`yield` outside a function, `break` outside a loop), F82 (undefined
# names). These are errors in any file, not style; nothing here reports
# unused imports or line length. Python only: no offline linter for TS/JS
# or Rust is installed (tools/format holds prettier alone), so those get the
# parser and the formatter's own parser, nothing more.
LINT_SELECT = "E9,F63,F7,F82"


def lint_errors(code: str, lang: str) -> list[dict]:
    """ruff's LINT_SELECT findings for a whole Python file, [] for any other
    language or when ruff is unavailable. Code goes in on stdin; ruff runs
    --isolated in a fresh temp dir, like the formatters. Never raises."""
    if lang != "python":
        return []
    exe = _which("ruff")
    if not exe:
        return []
    with tempfile.TemporaryDirectory(prefix="yamadori_lint_") as tmp:
        try:
            r = subprocess.run(
                [exe, "check", "--isolated", "--no-cache", "--select",
                 LINT_SELECT, "--output-format", "json", "--stdin-filename",
                 "check.py", "-"],
                input=code.encode("utf-8"), cwd=tmp, env=_isolated_env(tmp),
                capture_output=True, timeout=FORMAT_TIMEOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (subprocess.TimeoutExpired, OSError):
            return []
    try:
        found = json.loads(r.stdout.decode("utf-8", "replace") or "[]")
    except ValueError:
        return []
    out = []
    for f in found if isinstance(found, list) else []:
        loc = f.get("location") or {}
        out.append({"line": int(loc.get("row") or 1),
                    "col": int(loc.get("column") or 1),
                    "message": f"{f.get('code') or 'lint'} {f.get('message')}",
                    "source": "ruff"})
    out.sort(key=lambda e: (e["line"], e["col"]))
    return out[:MAX_ERRORS]


_PRETTIER_POS = re.compile(r"\((\d+):(\d+)\)\s*$")


def formatter_parse_error(res: dict) -> dict | None:
    """The formatter's own parser rejecting code tree-sitter accepted, as a
    syntax error -- or None. prettier parses with TypeScript's / Babel's
    parser and rustfmt with rustc's, both stricter than tree-sitter's
    error-tolerant grammars. Only a failure that SAYS it is a parse error
    counts; a timeout or a missing formatter is not the code's fault."""
    f = res.get("formatter") or {}
    err = f.get("error") or ""
    if not err or res.get("formatted") is not None:
        return None
    if f.get("name") == "prettier" and "SyntaxError" in err:
        m = _PRETTIER_POS.search(err)
        msg = err.split("SyntaxError:", 1)[1].strip()
        return {"line": int(m.group(1)) if m else 1,
                "col": int(m.group(2)) if m else 1,
                "message": _PRETTIER_POS.sub("", msg).strip()[:200],
                "source": "prettier"}
    if f.get("name") == "rustfmt" and err.startswith("error"):
        return {"line": 1, "col": 1, "message": err[:200], "source": "rustfmt"}
    return None


def diff_summary(before: str, after: str | None) -> dict:
    if after is None:
        return {"changed_lines": None}
    a, b = before.rstrip("\n").split("\n"), after.rstrip("\n").split("\n")
    if a == b:
        return {"changed_lines": 0}
    changed, first, ws_only = 0, None, True
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
        if first is None:
            first = i1 + 1
        if tag != "replace" or (i2 - i1) != (j2 - j1) or any(
                x.strip() != y.strip() for x, y in zip(a[i1:i2], b[j1:j2])):
            ws_only = False
    return {"changed_lines": changed, "of": len(a), "first_changed_line": first,
            "whitespace_only": ws_only}


# ---------------------------------------------------------------------------
# Python structure: open blocks at the end of a prefix, loops that cannot end
# ---------------------------------------------------------------------------
_COMPOUND = ("if", "elif", "else", "for", "while", "try", "except", "finally",
             "with", "def", "class", "async", "match", "case")


def python_open_state(prefix: str) -> dict:
    """What is still open at the end of `prefix`, read with Python's tokenizer.

    Returns levels (indent columns of the open blocks, outermost first),
    blocks (header text, line and body column for each open block), whether
    the last statement is a header still waiting for its body, the last
    statement's column, and any string or brackets left open.
    """
    state = {"levels": [0], "blocks": [], "awaits_body": False,
             "last_indent": 0, "last_line": 0, "last_header": None,
             "open_string": None, "open_brackets": [], "tokenized": True}
    lines = prefix.split("\n")
    levels = [0]
    blocks: list[dict] = []
    pending = None           # the logical line just finished (a header?)
    stmt_start = None        # (row, col, text) of the current logical line
    last_sig = None
    brackets: list[tuple[str, int, int]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(prefix).readline):
            tt = tok.type
            if tt == tokenize.INDENT:
                levels.append(_indent_width(tok.string))
                blocks.append(dict(pending or {"header": "?", "line": 0},
                                   body_col=levels[-1]))
                continue
            if tt == tokenize.DEDENT:
                if len(levels) > 1:
                    levels.pop()
                    blocks.pop()
                continue
            if tt in (tokenize.NL, tokenize.COMMENT, tokenize.ENCODING,
                      tokenize.ENDMARKER):
                continue
            if tt == tokenize.NEWLINE:
                if stmt_start:
                    row, col, text = stmt_start
                    ends_colon = (last_sig is not None
                                  and last_sig.type == tokenize.OP
                                  and last_sig.string == ":")
                    first_word = text.split(None, 1)[0] if text.split() else ""
                    first_word = first_word.rstrip(":(")
                    header = ends_colon and first_word in _COMPOUND
                    pending = {"header": text.strip(), "line": row, "col": col}
                    state.update(levels=list(levels),
                                 blocks=[dict(b) for b in blocks],
                                 awaits_body=header, last_indent=col,
                                 last_line=row,
                                 last_header=pending if header else None)
                stmt_start = None
                last_sig = None
                continue
            if stmt_start is None:
                row = tok.start[0]
                stmt_start = (row, tok.start[1],
                              lines[row - 1] if row - 1 < len(lines) else "")
            if tt == tokenize.OP and tok.string in "([{":
                brackets.append((tok.string, tok.start[0], tok.start[1] + 1))
            elif tt == tokenize.OP and tok.string in ")]}" and brackets:
                brackets.pop()
            last_sig = tok
    except tokenize.TokenError as e:
        msg = e.args[0] if e.args else ""
        pos = e.args[1] if len(e.args) > 1 else (0, 0)
        if "string" in msg:
            row = pos[0] if isinstance(pos, tuple) else 0
            state["open_string"] = _find_open_string(lines, row)
        state["open_brackets"] = [{"char": c, "line": r, "col": k}
                                  for c, r, k in brackets]
    except (IndentationError, SyntaxError) as e:
        state["tokenized"] = False
        state["tokenize_error"] = f"{e.msg} (line {e.lineno})"
    return state


def _find_open_string(lines: list[str], row: int) -> dict | None:
    """The unterminated string's delimiter and where it starts. The tokenizer
    reports the row; the delimiter is the triple quote on that row (or the
    last one before it, for a string that began earlier)."""
    for r in range(min(row, len(lines)), 0, -1):
        text = lines[r - 1]
        best = None
        for delim in ('"""', "'''"):
            k = text.rfind(delim)
            if k != -1 and (best is None or k > best[1]):
                best = (delim, k)
        if best:
            delim, k = best
            return {"delimiter": delim, "line": r, "col": k + 1,
                    "indent": len(_leading(text).expandtabs(8)),
                    "docstring": text[:k].strip() == ""}
    return None


_PURE_BUILTINS = {"len", "min", "max", "abs", "range", "int", "str", "float",
                  "bool", "sum", "sorted", "list", "dict", "set", "tuple",
                  "isinstance", "enumerate", "zip", "any", "all", "divmod",
                  "pow", "round", "hash", "ord", "chr", "repr", "reversed",
                  "map", "filter", "type", "id", "print", "bin", "hex"}


def _idents(node) -> set[str]:
    out = set()
    for n in _walk(node):
        if n.type == "identifier":
            p = n.parent
            if p is not None and p.type == "attribute" and \
                    p.child_by_field_name("attribute") == n:
                continue
            if p is not None and p.type == "keyword_argument" and \
                    p.child_by_field_name("name") == n:
                continue
            out.add(n.text.decode("utf-8", "replace"))
    return out


def _loop_exits(body) -> bool:
    """break (of THIS loop), return, raise, yield or await anywhere in body."""
    stack = [(c, False) for c in body.children]
    while stack:
        n, nested_loop = stack.pop()
        t = n.type
        if t in ("function_definition", "class_definition", "lambda"):
            continue
        if t in ("return_statement", "raise_statement", "yield", "await"):
            return True
        if t == "break_statement" and not nested_loop:
            return True
        inner = nested_loop or t in ("for_statement", "while_statement")
        stack.extend((c, inner) for c in n.children)
    return False


def _touched(body) -> tuple[set[str], bool, bool]:
    """(names the body may change, has an impure call, declares global/nonlocal)."""
    touched: set[str] = set()
    impure = False
    scoped = False
    for n in _walk(body):
        t = n.type
        if t in ("assignment", "augmented_assignment"):
            left = n.child_by_field_name("left")
            if left is not None:
                touched |= _idents(left)
        elif t == "named_expression":
            nm = n.child_by_field_name("name")
            if nm is not None:
                touched |= _idents(nm)
        elif t in ("for_statement", "for_in_clause"):
            left = n.child_by_field_name("left")
            if left is not None:
                touched |= _idents(left)
        elif t in ("delete_statement", "as_pattern_target"):
            touched |= _idents(n)
        elif t in ("global_statement", "nonlocal_statement"):
            scoped = True
        elif t == "call":
            fn = n.child_by_field_name("function")
            args = n.child_by_field_name("arguments")
            # print(i) and len(xs) change neither i nor xs; any other call
            # may mutate what it is handed.
            pure = (fn is not None and fn.type == "identifier"
                    and fn.text.decode("utf-8", "replace") in _PURE_BUILTINS)
            if args is not None and not pure:
                touched |= _idents(args)        # f(xs) may mutate xs
            if fn is not None and fn.type == "attribute":
                obj = fn.child_by_field_name("object")
                if obj is not None:
                    touched |= _idents(obj)     # xs.pop() mutates xs
                impure = True
            elif not pure:
                impure = True
    return touched, impure, scoped


def _scope_locals(node) -> set[str]:
    """Names bound in the enclosing function (parameters and assignments)."""
    fn = node.parent
    while fn is not None and fn.type not in ("function_definition", "module"):
        fn = fn.parent
    if fn is None:
        return set()
    names: set[str] = set()
    if fn.type != "function_definition":
        # Module level: any call may rebind a module global, so nothing
        # there counts as out of reach.
        return set()
    params = fn.child_by_field_name("parameters")
    if params is not None:
        names |= _idents(params)
    scope = fn.child_by_field_name("body")
    if scope is not None:
        t, _, _ = _touched(scope)
        names |= t
    return names


def loops_that_cannot_end(code: str) -> list[dict]:
    """`while` loops whose body can never make the condition false.

    Deliberately narrow, because a false alarm here would send a model to
    rewrite a correct loop. A loop is reported only when ALL of these hold:
    its condition contains no call; its body has no break (of this loop),
    return, raise, yield or await; no name in the condition is assigned,
    deleted, used as a method receiver or passed to a call in the body; the
    body declares nothing global or nonlocal; and, if the body makes any call
    that could have side effects, every name in the condition is a local of
    the enclosing function (so no call can reach it).

    This is the LiveBench completion failure of 2026-09-23: a continuation
    dedented out of `while left <= right:` left `mid = ...` as the loop's
    whole body.
    """
    try:
        root = parse_tree(code, "python")
    except Exception:                                            # noqa: BLE001
        return []
    out = []
    for n in _walk(root):
        if n.type != "while_statement" or n.has_error:
            continue
        cond = n.child_by_field_name("condition")
        body = n.child_by_field_name("body")
        if cond is None or body is None:
            continue
        if any(x.type == "call" for x in _walk(cond)):
            continue
        if _loop_exits(body):
            continue
        names = _idents(cond)
        touched, impure, scoped = _touched(body)
        if scoped or names & touched:
            continue
        if impure and names - _scope_locals(n):
            continue
        header = n.text.decode("utf-8", "replace").split("\n", 1)[0].strip()
        what = (", ".join(sorted(names)) if names else "nothing")
        out.append({"line": n.start_point[0] + 1,
                    "end_line": n.end_point[0] + 1,
                    "kind": "loop_cannot_end",
                    "message": (f"`{header}` can never end: its body never "
                                f"changes {what} and has no break, return or "
                                f"raise")})
    return out


# ---------------------------------------------------------------------------
# Continuation
# ---------------------------------------------------------------------------
def join_continuation(prefix: str, cont: str) -> str:
    """Exactly how a completion grader joins them: prefix, newline, code.
    LiveBench's coding_completion does `partial_solution + '\\n' + answer`."""
    return prefix + "\n" + cont


def _first_code_line(cont: str) -> tuple[int, str] | None:
    for i, ln in enumerate(cont.split("\n"), 1):
        s = ln.strip()
        if s and not s.startswith("#"):
            return i, ln
    return None


def _where(line: int, p_lines: int) -> dict:
    if line > p_lines:
        return {"line": line - p_lines, "where": "code"}
    return {"line": line, "where": "prefix"}


def _header_name(text: str) -> str | None:
    """`def foo(...)` / `class Foo:` -> the name, via the tokenizer."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(
                text.strip() + "\n").readline)
                if t.type in (tokenize.NAME, tokenize.OP)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    words = [t.string for t in toks]
    if words[:1] == ["async"]:
        words = words[1:]
    if len(words) >= 2 and words[0] in ("def", "class"):
        return f"{words[0]} {words[1]}"
    return None


def _describe_blocks(blocks: list[dict]) -> str:
    return " > ".join(f"`{b['header'].rstrip(':')}` (line {b['line']}, body col "
                      f"{b['body_col']})" for b in blocks)


def analyse_python_continuation(prefix: str, cont: str) -> dict:
    """Defects of `cont` written to follow `prefix`, and the facts behind them.

    Every defect names what the prefix left open and where the continuation
    went instead. Line numbers are relative to the continuation ("your line")
    unless the problem is inside the prefix.
    """
    p_lines = prefix.count("\n") + 1
    joined = join_continuation(prefix, cont)
    st = python_open_state(prefix)
    defects: list[dict] = []
    notes: list[str] = []
    first = _first_code_line(cont)
    first_col = _indent_width(_leading(first[1])) if first else None
    info = {"prefix_lines": p_lines, "open_blocks": st["blocks"],
            "awaits_body": st["awaits_body"], "open_string": st["open_string"],
            "open_brackets": st["open_brackets"], "first_col": first_col}

    if first is None:
        defects.append({"line": 1, "where": "code", "kind": "empty",
                        "message": "the continuation has no code in it"})
        return {"defects": defects, "syntax_errors": [], "info": info,
                "notes": notes}

    fl, ftext = first
    os_ = st["open_string"]
    if os_ and not ftext.strip().startswith(os_["delimiter"]) \
            and os_["delimiter"] not in cont:
        close = " " * os_["indent"] + os_["delimiter"]
        defects.append({
            "line": fl, "where": "code", "kind": "unclosed_string_from_prefix",
            "message": (f"the prefix leaves a {os_['delimiter']} "
                        f"{'docstring' if os_['docstring'] else 'string'} "
                        f"open (prefix line {os_['line']}) and your code never "
                        f"closes it, so all of it is inside that string. "
                        f"Close it first, on its own line: `{close}`")})

    name = _header_name(ftext)
    if name:
        for b in st["blocks"]:
            if _header_name(b["header"]) == name:
                defects.append({
                    "line": fl, "where": "code", "kind": "redeclares_signature",
                    "message": (f"your code starts by declaring `{name}` "
                                f"again; the prefix already opened it at "
                                f"prefix line {b['line']}. Write only the "
                                f"lines that come after the prefix.")})
                break

    if not st["tokenized"]:
        # The prefix's own indentation is inconsistent, so there is no
        # block structure to measure the continuation against.
        notes.append("the prefix itself does not tokenize ("
                     + st.get("tokenize_error", "?") + "); indentation was "
                     "not compared")
    elif not os_ and not st["open_brackets"]:
        levels = st["levels"]
        if st["awaits_body"]:
            h = st["last_header"] or {}
            unit = infer_indent(prefix).get("indent_width") or 4
            if first_col <= st["last_indent"]:
                defects.append({
                    "line": fl, "where": "code", "kind": "body_not_indented",
                    "message": (f"the prefix ends with `{h.get('header', '')}` "
                                f"(prefix line {h.get('line')}), which needs "
                                f"its body indented to column "
                                f"{st['last_indent'] + unit}; your first line "
                                f"is at column {first_col}")})
        elif first_col not in levels:
            defects.append({
                "line": fl, "where": "code", "kind": "indent_matches_no_block",
                "message": (f"your first line is at column {first_col}; at "
                            f"the end of the prefix the open blocks start "
                            f"lines at columns "
                            f"{', '.join(str(x) for x in reversed(levels))}")})
        elif first_col < st["last_indent"]:
            closed = [b for b in st["blocks"] if b["body_col"] > first_col]
            if closed:
                notes.append(
                    "your first line (column " f"{first_col}) closes "
                    + ", ".join(f"`{b['header'].rstrip(':')}` (prefix line "
                                f"{b['line']})" for b in reversed(closed))
                    + "; lines meant to be inside "
                    + ("it" if len(closed) == 1 else "them")
                    + " need column "
                    + str(closed[-1]["body_col"]))

    errs = []
    for e in syntax_errors(joined, "python"):
        errs.append(dict(e, **_where(e["line"], p_lines)))
    # A loop the prefix left open and the continuation closed without ever
    # updating its condition. Loops wholly inside the prefix are not the
    # continuation's doing and are not reported.
    if not errs:
        for lp in loops_that_cannot_end(joined):
            if lp["end_line"] >= p_lines:
                defects.append(dict(lp, **_where(lp["line"], p_lines)))
    return {"defects": defects, "syntax_errors": errs, "info": info,
            "notes": notes}


def _open_braces(code: str, lang: str) -> Counter:
    c: Counter = Counter()
    try:
        root = parse_tree(code, lang)
    except Exception:                                            # noqa: BLE001
        return c
    for n in _walk(root):
        if n.is_missing and n.type in ("}", ")", "]"):
            c[n.type] += 1
    return c


def analyse_brace_continuation(prefix: str, cont: str, lang: str) -> dict:
    p_lines = prefix.count("\n") + 1
    joined = join_continuation(prefix, cont)
    errs = [dict(e, **_where(e["line"], p_lines))
            for e in syntax_errors(joined, lang)]
    open_p = _open_braces(prefix, lang)
    info = {"prefix_lines": p_lines,
            "prefix_leaves_open": dict(open_p) or None}
    return {"defects": [], "syntax_errors": errs, "info": info, "notes": []}


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
def _dedent_python(code: str) -> tuple[str, int]:
    """A method or block excerpt is valid Python once its common indent is
    removed; checking it raw would report `unexpected indent` for correct
    code. Returns (dedented, columns removed)."""
    ded = textwrap.dedent(code)
    if ded == code:
        return code, 0
    before = [ln for ln in code.split("\n") if ln.strip()]
    after = [ln for ln in ded.split("\n") if ln.strip()]
    removed = (len(before[0]) - len(after[0])) if before and after else 0
    return ded, removed


def _reindent(code: str, cols: int) -> str:
    if not cols:
        return code
    pad = " " * cols
    return "\n".join(pad + ln if ln.strip() else ln for ln in code.split("\n"))


def check(code: str, language: str, context: str | None = None,
          config_text: str | None = None, mode: str = "whole",
          prefix: str | None = None, run_format: bool = True) -> dict:
    """Parse `code`, infer the target style, and offer a formatted version.

    Returns {ok, language, mode, syntax_errors, defects, formatted, style,
    style_source, diff_summary, formatter, notes} and, in continuation mode,
    `continuation` with what the prefix left open. `ok` means no syntax
    error and no defect. Pure: reads no file, runs no user code.
    """
    lang = normalize_language(language)
    if lang is None:
        raise ValueError(f"unsupported language {language!r}")
    code = (code or "").replace("\r\n", "\n")
    notes: dict = {}
    res: dict = {"language": lang, "mode": mode, "defects": [],
                 "formatted": None}
    style_ctx = context if context else (prefix if mode == "continuation" else None)
    style, src, snotes = resolve_style(lang, code, style_ctx, config_text)
    notes.update(snotes)
    ev = style.pop("_evidence", None)
    res["style"] = {k: v for k, v in style.items() if not k.startswith("_")}
    res["style_source"] = src
    if ev:
        res["style_evidence"] = ev

    if mode == "continuation":
        prefix = (prefix or "").replace("\r\n", "\n")
        if lang == "python":
            a = analyse_python_continuation(prefix, code)
        else:
            a = analyse_brace_continuation(prefix, code, lang)
        res["syntax_errors"] = a["syntax_errors"]
        res["defects"] = a["defects"]
        res["continuation"] = a["info"]
        if a["notes"]:
            notes["continuation"] = a["notes"]
        res["formatter"] = formatter_status(LANGS[lang]["formatter"])
        res["diff_summary"] = {"changed_lines": None}
        notes["formatted"] = ("not offered for a continuation: a formatter "
                              "would reflow the prefix too; match the columns "
                              "reported above")
    else:
        body, removed = (_dedent_python(code) if lang == "python"
                         else (code, 0))
        if removed:
            notes["dedented"] = (f"checked with its common indent of {removed} "
                                 f"columns removed (an excerpt from inside a "
                                 f"block)")
        res["syntax_errors"] = [dict(e) for e in syntax_errors(body, lang)]
        if lang == "python" and not res["syntax_errors"]:
            res["defects"] = loops_that_cannot_end(body)
        fmt_name = LANGS[lang]["formatter"]
        if fmt_name is None:
            res["formatter"] = {"name": None, "available": False}
            notes["formatted"] = "syntax check only for this language"
        elif res["syntax_errors"]:
            res["formatter"] = formatter_status(fmt_name)
            notes["formatted"] = "not run: fix the syntax errors first"
        elif not run_format:
            res["formatter"] = formatter_status(fmt_name)
        else:
            out, finfo = run_formatter(body, lang, style)
            res["formatter"] = finfo
            if out is not None:
                res["formatted"] = _reindent(out, removed) if removed else out
        res["diff_summary"] = diff_summary(code, res["formatted"])
    res["ok"] = not res["syntax_errors"] and not res["defects"]
    res["notes"] = notes
    return res


# ---------------------------------------------------------------------------
# What the model reads
# ---------------------------------------------------------------------------
def _style_line(res: dict) -> str:
    s, src = res.get("style") or {}, res.get("style_source") or {}
    bits = []
    if "indent" in s:
        if s["indent"] == "tab":
            bits.append(f"indent tabs ({src.get('indent')})")
        else:
            bits.append(f"indent {s.get('indent_width')} spaces "
                        f"({src.get('indent_width', src.get('indent'))})")
    for key, label in (("quotes", "quotes"), ("semicolons", "semicolons"),
                       ("trailing_commas", "trailing commas"),
                       ("width", "width")):
        if key in s:
            v = s[key]
            v = ("yes" if v else "no") if isinstance(v, bool) else v
            bits.append(f"{label} {v} ({src.get(key)})")
    return ", ".join(bits)


def render(res: dict) -> str:
    """Compact, actionable text: what is wrong, where, and what to do."""
    lang, mode = res["language"], res["mode"]
    problems = len(res["syntax_errors"]) + len(res["defects"])
    if mode == "continuation":
        c = res.get("continuation") or {}
        head = (f"check_code: {lang}, continuation after a "
                f"{c.get('prefix_lines')}-line prefix -- ")
    else:
        head = f"check_code: {lang} -- "
    head += "parses" if res["ok"] else (
        f"{problems} problem{'s' if problems != 1 else ''}")
    out = [head]
    items = []
    for e in res["syntax_errors"]:
        where = e.get("where")
        loc = (f"prefix line {e['line']}" if where == "prefix"
               else f"your line {e['line']}" if where == "code"
               else f"line {e['line']}")
        items.append((e["line"] if where != "prefix" else -1,
                      f"  {loc}, col {e.get('col', 1)}: {e['message']}"))
    for d in res["defects"]:
        where = d.get("where")
        loc = (f"prefix line {d['line']}" if where == "prefix"
               else f"your line {d['line']}" if where == "code"
               else f"line {d['line']}")
        items.append((d["line"] if where != "prefix" else -1,
                      f"  {loc}: {d['message']}"))
    out += [t for _, t in sorted(items, key=lambda x: x[0])]
    if mode == "continuation":
        c = res.get("continuation") or {}
        if c.get("open_blocks"):
            out.append("open at the end of the prefix: "
                       + _describe_blocks(c["open_blocks"]))
        if c.get("open_brackets"):
            out.append("the prefix leaves open: " + ", ".join(
                f"`{b['char']}` (line {b['line']}, col {b['col']})"
                for b in c["open_brackets"]) + " -- your code continues "
                "inside it and must close it")
        if c.get("first_col") is not None:
            out.append(f"your first line starts at column {c['first_col']}")
        if c.get("prefix_leaves_open"):
            out.append("the prefix leaves open: " + ", ".join(
                f"{n} x `{k}`" for k, n in c["prefix_leaves_open"].items())
                + " -- your code must close them")
        for n in (res.get("notes") or {}).get("continuation") or []:
            out.append(n)
    st = _style_line(res)
    if st:
        out.append("style: " + st)
    notes = res.get("notes") or {}
    if notes.get("config_text"):
        out.append("config_text: " + notes["config_text"])
    if notes.get("dedented"):
        out.append(notes["dedented"])
    f = res.get("formatter") or {}
    if res.get("formatted") is not None:
        ds = res.get("diff_summary") or {}
        if not ds.get("changed_lines"):
            out.append(f"formatting: already matches ({f.get('name')})")
        else:
            kind = ("whitespace only" if ds.get("whitespace_only")
                    else "layout and tokens")
            out.append(f"formatting: {ds['changed_lines']} of {ds['of']} lines "
                       f"differ from that style (first at line "
                       f"{ds['first_changed_line']}; {kind}). Formatted by "
                       f"{f.get('name')}:")
            fence = "````" if "```" in res["formatted"] else "```"
            out.append(f"{fence}{lang}\n{res['formatted'].rstrip()}\n{fence}")
    elif f.get("name") and not f.get("available"):
        out.append(f"formatting: not run -- {f['name']} is not installed on "
                   f"this server ({f.get('remedy')}). The syntax check above "
                   f"is complete without it.")
    elif f.get("error"):
        out.append(f"formatting: {f['name']} failed ({f['error']}); the "
                   f"syntax check above still stands")
    elif notes.get("formatted"):
        out.append("formatting: " + notes["formatted"])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The model tool
# ---------------------------------------------------------------------------
_LANG_LIST = ("python, typescript, tsx, javascript, jsx, rust, c, cpp; "
              "json, toml, yaml (syntax only)")

TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Before you answer with code, call this to check it: it parses "
            "the code and returns syntax errors with line numbers, and a "
            "version formatted in the existing file's style when you pass "
            "that file as `context`.\n"
            "Reach for it when: you wrote code that will be appended to "
            "existing code (the rest of a function, a completion of starter "
            "code) -- use mode \"continuation\" and pass the existing code as "
            "`prefix`, and it reports whether the combined program parses, "
            "which block your first line lands in, and any string or bracket "
            "the prefix left open; you are unsure of indentation, bracket "
            "balance or a closing quote; you edited a file and want its "
            "indentation, quotes and semicolons kept.\n"
            f"Languages: {_LANG_LIST}. It checks only the code text you pass "
            "and never runs the code. To check a file in the user's project, "
            "read it with your client's own file tools and pass its "
            "contents; to run it, use your client's own terminal tool."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string",
                         "description": "The code to check. In continuation "
                                        "mode, only the part you wrote."},
                "language": {"type": "string",
                             "description": f"One of: {_LANG_LIST}."},
                "mode": {"type": "string", "enum": ["whole", "continuation"],
                         "description": "\"whole\" (default): the code stands "
                                        "alone. \"continuation\": the code "
                                        "is appended after `prefix`."},
                "prefix": {"type": "string",
                           "description": "Continuation mode: the existing "
                                          "code yours follows, verbatim."},
                "context": {"type": "string",
                            "description": "Existing code from the same file "
                                           "or project, to infer indentation, "
                                           "quotes, semicolons, trailing "
                                           "commas and line width from."},
                "config_text": {"type": "string",
                                "description": "Contents of the project's "
                                               ".editorconfig, .prettierrc, "
                                               "rustfmt.toml or pyproject "
                                               "[tool.ruff], if you have read "
                                               "it. Wins over inference."},
            },
            "required": ["code", "language"],
        },
    },
}


def _error(code: str, reason: str, retryable: bool, action: str,
           fixable_by: str = "agent", **facts) -> str:
    return json.dumps({"tool": TOOL_NAME, "ok": False, "error": code,
                       "reason": reason, "retryable": retryable,
                       "remedies": [{"fixable_by": fixable_by,
                                     "action": action}], **facts})


def run_tool(args: dict, record: list | None = None) -> str:
    """Execute check_code for the model. Never raises, never returns empty."""
    if not isinstance(args, dict):
        args = {}
    code, language = args.get("code"), args.get("language")
    mode = args.get("mode") or "whole"
    for key in ("code", "language", "prefix", "context", "config_text",
                "mode"):
        v = args.get(key)
        if v is not None and not isinstance(v, str):
            return _error("BAD_ARGUMENTS",
                          f"Argument {key!r} must be a string; got "
                          f"{type(v).__name__}. Nothing was checked.", True,
                          f"call check_code again with {key!r} as a string")
    if not code or not code.strip():
        return _error("BAD_ARGUMENTS", "Argument 'code' is missing or empty. "
                      "Nothing was checked.", True,
                      "call check_code again with the code you wrote as `code`")
    lang = normalize_language(language)
    if lang is None:
        return _error("UNSUPPORTED_LANGUAGE",
                      f"{language!r} is not a language this checker parses. "
                      f"Nothing was checked.", True,
                      "call again with one of the supported names",
                      supported=sorted(LANGS))
    if mode not in ("whole", "continuation"):
        return _error("BAD_ARGUMENTS", f"mode must be \"whole\" or "
                      f"\"continuation\"; got {mode!r}.", True,
                      "call again with mode \"whole\", or \"continuation\" "
                      "with `prefix` set")
    prefix = args.get("prefix")
    if mode == "continuation" and not (prefix or "").strip():
        return _error("BAD_ARGUMENTS", "Continuation mode needs `prefix`: the "
                      "existing code your code is appended to. Nothing was "
                      "checked.", True,
                      "call again with `prefix` set to the existing code "
                      "verbatim, or with mode \"whole\"")
    size = sum(len(args.get(k) or "") for k in ("code", "prefix", "context",
                                                "config_text"))
    if size > MAX_CHARS:
        return _error("TOO_LARGE", f"{size} characters in total; the limit is "
                      f"{MAX_CHARS}. Nothing was checked.", True,
                      "check the part you changed, with a shorter `context`")
    try:
        res = check(code, lang, context=args.get("context"),
                    config_text=args.get("config_text"), mode=mode,
                    prefix=prefix)
    except Exception as e:                                       # noqa: BLE001
        return _error("CHECK_FAILED", f"the checker itself failed "
                      f"({type(e).__name__}); this says nothing about your "
                      f"code", False, "check the proxy log",
                      fixable_by="operator")
    if record is not None:
        record.append({"language": lang, "mode": mode, "ok": res["ok"],
                       "errors": len(res["syntax_errors"]) + len(res["defects"])})
    return render(res)


# ---------------------------------------------------------------------------
# Answers: fenced blocks, the prefix a question carried, the repair verdict
# ---------------------------------------------------------------------------
def fenced_blocks(text: str) -> list[dict]:
    """Markdown fenced code blocks, CommonMark-style: an opening run of 3+
    backticks or tildes (indented at most 3 spaces) and a closing run of the
    same character at least as long. An unclosed fence runs to the end --
    and says so (`closed` False): a caller that WRITES the code anywhere
    uses closed blocks only (shomen.fixup; an unclosed fence is a truncated
    reply). `fence` is the opening run itself.

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


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return ""


def prefix_from_messages(messages: list[dict]) -> dict | None:
    """The last fenced block of the last user message: the starter code a
    completion question carries, if it carries one."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            blocks = [b for b in fenced_blocks(_text_of(m.get("content")))
                      if b["code"].strip()]
            return blocks[-1] if blocks else None
    return None


def prompt_code(messages: list[dict]) -> list[dict]:
    """The code the question carries: every non-empty fenced block of the
    LATEST user message, in order; [] when it has none.

    No words are read. Whether an answer continues this code is decided by
    parsing (check_against_prompt), never by how the question is phrased, so
    "finish this function" and a benchmark's template are treated alike.
    """
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return [b for b in fenced_blocks(_text_of(m.get("content")))
                    if b["code"].strip()]
    return []


def _appendable(block: dict, lang: str) -> bool:
    """A question block `lang` code can be appended to: tagged `lang`, or
    untagged. A block in another language (a log, JSON, a shell line) is
    never tried."""
    tag = (block.get("lang") or "").strip()
    return not tag or normalize_language(tag) == lang


def check_against_prompt(code: str, lang: str, blocks: list[dict]) -> dict:
    """THE GENERAL RULE for code written in answer to a question that
    carries code (operator decision, 2026-09-23).

    A code block PARSES if it parses on its own, or when appended directly
    after one of the question's blocks (join_continuation: block, newline,
    code) -- the last block first, then the others, latest back. Nothing
    guesses the question's intent: a whole program and a continuation of
    the question's code both pass, and only code that parses NEITHER way
    fails. "Parses" means no syntax error; a defect (a loop that cannot
    end, a re-declared signature) is a warning, never a failure.

    Returns {parses, check_mode, result, alone, appended, prefix_block}:
      check_mode  None          the question carries no code (plain check)
                  "standalone"  it parses on its own
                  "appended"    it parses only appended after a block
                  "neither"     it parses neither way
      result      the check() result of the form that parsed; the
                  standalone one when none did
      alone       the standalone check() result
      appended    the continuation check() result: of the block it parsed
                  after, or, for "neither", of the first block tried (the
                  last one); None when no block was tried
      prefix_block  that block's 1-based index among `blocks`, or None
    Pure: reads no file, runs no code.
    """
    alone = check(code, lang, run_format=False)
    out = {"parses": not alone["syntax_errors"], "check_mode": None,
           "result": alone, "alone": alone, "appended": None,
           "prefix_block": None}
    if not blocks:
        return out
    if not alone["syntax_errors"]:
        return dict(out, check_mode="standalone")
    first = None
    for i in range(len(blocks) - 1, -1, -1):
        if not _appendable(blocks[i], lang):
            continue
        res = check(code, lang, mode="continuation", prefix=blocks[i]["code"],
                    run_format=False)
        if first is None:
            first = (res, i + 1)
        if not res["syntax_errors"]:
            return dict(out, parses=True, check_mode="appended", result=res,
                        appended=res, prefix_block=i + 1)
    return dict(out, check_mode="neither",
                appended=first[0] if first else None,
                prefix_block=first[1] if first else None)


def check_mode_of(modes: list) -> str | None:
    """One check_mode for an answer's checked blocks: the blocks' shared
    mode, "mixed" when they differ, None when nothing was checked or the
    question carried no code."""
    names = {m for m in modes if m}
    if not names:
        return None
    return names.pop() if len(names) == 1 else "mixed"


def _problem_lines(res: dict) -> str:
    """render() without its head line and without the style and formatting
    lines: what the model must fix, not how it might be laid out."""
    text = render(res)
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return "\n".join(ln for ln in body.split("\n")
                     if not ln.startswith(("style:", "formatting:",
                                           "config_text:")))


def _as_problems(res: dict) -> list[dict]:
    out = [dict(e) for e in res.get("syntax_errors") or []]
    for d in res.get("defects") or []:
        if isinstance(d, dict):
            out.append({"line": d.get("line", 1), "col": d.get("col", 1),
                        "message": d.get("message") or d.get("kind")
                        or "defect"})
    return out


def block_problems(code: str, lang: str, pcode: list[dict]) -> list[dict]:
    """What review_answer would flag in one block, as [{line, col,
    message}] ([] when it passes): the second brain's fixup job re-checks a
    corrected block with this."""
    g = check_against_prompt(code, lang, pcode)
    if g["check_mode"] is None:
        return _as_problems(g["alone"])
    if g["parses"]:
        return []
    return _as_problems(g["alone"]) or [
        {"line": 1, "col": 1, "message": "does not parse"}]


def review_answer(content: str, messages: list[dict]) -> dict:
    """What the repair pass needs to know about an answer's code blocks.

    Returns {blocks, checked, count, key, feedback, check_mode, flagged,
    langs}. `count` is
    the number of problems across flagged blocks; `feedback` is the text for
    the model when count > 0; `key` identifies the code so an identical
    re-answer can be recognised; `check_mode` is check_mode_of the blocks'
    modes (check_against_prompt). `flagged` lists each failing block as
    {index, lang, code, errors} for the second brain's fixup job; `langs`
    the checked blocks' languages.

    When the question carries code (prompt_code), a block is flagged only
    when it parses NEITHER on its own NOR appended after the question's
    code, and the feedback states both results. When it carries none, the
    reading is the one it always was: the block on its own, syntax errors
    and defects both counted.
    """
    blocks = fenced_blocks(content or "")
    pcode = prompt_code(messages)
    last = pcode[-1] if pcode else None
    checked, count, parts, modes = 0, 0, [], []
    flagged, langs = [], []
    h = hashlib.sha1()
    for k, b in enumerate(blocks, 1):
        lang = normalize_language(b["lang"])
        if lang is None and not b["lang"] and last:
            lang = normalize_language(last["lang"])
        if lang not in CODE_LANGS or not b["code"].strip():
            continue
        checked += 1
        if lang not in langs:
            langs.append(lang)
        h.update(b["code"].encode("utf-8"))
        try:
            g = check_against_prompt(b["code"], lang, pcode)
        except Exception:                                        # noqa: BLE001
            continue
        modes.append(g["check_mode"])
        if g["check_mode"] is None:
            res = g["alone"]
            n = len(res["syntax_errors"]) + len(res["defects"])
            if n:
                count += n
                parts.append(f"code block {k} ({lang}, checked on its own):\n"
                             + _problem_lines(res))
                flagged.append({"index": k, "lang": lang, "code": b["code"],
                                "errors": _as_problems(res)})
            continue
        if g["parses"]:
            continue
        a, j = g["alone"], g["appended"]
        n = len(a["syntax_errors"]) + len(a["defects"])
        if j is not None:
            n = min(n, len(j["syntax_errors"]) + len(j["defects"]))
        count += max(n, 1)
        # Both readings, as the feedback text gives them: on its own, and
        # appended after the question's code (the one a continuation needs).
        errs = _as_problems(a) + [
            dict(e, message=f"appended after code block "
                            f"{g['prefix_block']} of the question: "
                            f"{e['message']}")
            for e in (_as_problems(j) if j is not None else [])]
        flagged.append({"index": k, "lang": lang, "code": b["code"],
                        "errors": errs or [{"line": 1, "col": 1,
                                            "message": "does not parse"}]})
        text = (f"code block {k} ({lang}) does not parse on its own:\n"
                + _problem_lines(a))
        if j is not None:
            text += (f"\n\nNor does it parse appended directly after code "
                     f"block {g['prefix_block']} of the question:\n"
                     + _problem_lines(j))
        else:
            text += (f"\n\nThe question's code is not {lang}, so it was not "
                     f"tried appended to it.")
        parts.append(text)
    feedback = None
    if count:
        feedback = ("Your answer's code was checked mechanically before "
                    "delivery. It does not work as written:\n\n"
                    + "\n\n".join(parts)
                    + "\n\nWrite your complete answer again with these fixed, "
                      "in the same form the question asked for.")
    return {"blocks": len(blocks), "checked": checked, "count": count,
            "key": h.hexdigest() if checked else None, "feedback": feedback,
            "check_mode": check_mode_of(modes), "flagged": flagged,
            "langs": langs}


if __name__ == "__main__":
    # A smoke check: python -m code_check < file.py  [language]
    src = sys.stdin.read()
    print(render(check(src, sys.argv[1] if len(sys.argv) > 1 else "python")))
