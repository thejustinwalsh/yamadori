#!/usr/bin/env python
"""The voxel-scene benchmark: one public prompt, one chat request per rep,
through the door users use (the proxy on :1234).

    python bench/voxel/run.py --key-file K --tag vx1                 # xhigh, 3 reps
    python bench/voxel/run.py --key-file K --tag vx1 --reps 5        # 2 more reps
    python bench/voxel/run.py --key-file K --tag vx2 --arms xhigh,xhigh-off
    python bench/voxel/run.py --preflight-only --key-file K
    python bench/voxel/run.py --list-arms

Then `check.py --tag vx1` (the sandboxed browser) and `compare.py --tag vx1`
(the page the operator judges). docs/BENCH-VOXEL.md is the plan.

THE TASK is a prompt people post online, sent VERBATIM (PROMPT below; its
sha256 is pinned by test_voxel.py). Single shot: one user message, no system
prompt, no tools, no follow-ups, no temperature and no max_tokens -- what a
chat client sends. `--max-tokens` exists for a later arm and is recorded.

THE STACK OPTION (`--task r3f-stack`; `single-html` stays the default): the
same prompt with ONLY its last sentence changed -- "Create a single HTML
file." becomes the operator's "Build it with r3f (react-three-fiber) v10 and
Koota and pmndrs math." (2026-09-27). Nothing of ours is added: no pins, no
layout, no build gate, no delivery convention. The model decides the
project; extract_project() takes whatever layout it writes (one fenced
block per file, the path from a `File:` line, the info string, a heading, a
first-line comment, or -- for an unnamed page -- index.html) and writes it
under reps/<rep>/project/; check.py builds and measures it. A tag holds ONE
task: a rerun with another --task (or a changed prompt) is refused.

THE ARMS (ARMS below). An arm is a tier (the body's `reasoning_effort`), an
optional X-Yamadori-Features header (an effort-matched pair: the same tier
with everything of ours forced off) and a model id (`ARM@MODEL`: a sibling
model once one is configured; the proxy resolves an UNKNOWN name to
`yamadori` silently, so preflight refuses a model /v1/models does not list).
The default is `xhigh` alone (operator, 2026-09-26: "our xHigh thinking is
all I care to test anymore, until it actually works"). The rest are kept for
later and run only when named.

WHAT A REP LEAVES (results/<tag>/, docs/BENCH-VOXEL.md "Files"):
  manifest.json           the task, the prompt, its sha256, the arms as sent
                          (tracked)
  summary.jsonl           one row per attempt, append-only (tracked): task,
                          prompt sha256 sent, status, HTTP status, finish,
                          usage, wall time, the x_yamadori decisions that
                          matter here, extraction
  reps/<arm>__r<n>__a<k>/ (gitignored) response.json -- the whole response
                          body, x_yamadori included --, page.html (the
                          extracted file) or project/ (r3f-stack), extract.json

NOT RUN IS NOT A RESULT. A 429 (every lane busy) or a refused connection
(the proxy restarting) generated nothing: the rep is waited on for at most
--busy-max-s and then recorded `not_run`, and a rerun of the same tag runs it
again. `http_error`, `exception` and `mismatch` (x_yamadori says the stack
did not run the arm asked for: another tier, or served as a utility call) are
stack errors, never model outcomes (PROTOCOL rule 3); `--retry-errors` reruns
them. `ok` and `length` (finish_reason length: the page may be cut) are done.

ONE GPU CONSUMER AT A TIME (AGENTS.md, "Before you claim anything works" 4):
preflight refuses while another benchmark or live suite runs (the BUSY list
of bench/octopus/run.py, which lists this runner too) or while any
llama-server slot is processing on two reads 5 s apart; the busy-process
check is repeated before every rep, and a run stops (resumable) when it trips.

The key is read from --key-file or YAMADORI_TEST_KEY, sent only as a header,
never printed and never written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
PROXY = os.environ.get("YAMADORI_PROXY", "http://127.0.0.1:1234")

# THE TASK, verbatim as posted (operator, 2026-09-26). Do not edit a
# character: test_voxel.py pins its sha256.
PROMPT = ("Design and create a very creative, elaborate, and detailed voxel art "
          "scene of a pagoda in a beautiful garden with trees, including some cherry "
          "blossoms, add a village with people living in it. Make the scene impressive "
          "and varied and use colorful voxels. Make it really detailed use as many "
          "voxels as you want, we have a power machine here. Create a single HTML file.")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


# THE STACK OPTION (task `r3f-stack`). Operator, 2026-09-27: "So I just said
# build the pagoda and octopus-invaders with r3f (react-three-fiber) v10 and
# Koota and pmndrs math." ... "That changes the prompt." So the ONLY change to
# the public prompt is its last sentence: "Create a single HTML file." becomes
# STACK_SENTENCE, the operator's words. Nothing else of ours: no pins, no
# layout, no rules, no build gate, no delivery convention -- the model decides
# the project, and extract_project() takes whatever layout it writes (one
# fenced block per file, the path from any of the usual places). The first
# version (2026-09-26) appended our pins, a required layout, the Octopus V4
# stack rules and a `File: <path>` convention; it was never run.
# test_voxel.py pins its sha256.
PROMPT_LAST_SENTENCE = " Create a single HTML file."
assert PROMPT.endswith(PROMPT_LAST_SENTENCE), "the verbatim prompt's last sentence moved"
PAGODA_CONTENT = PROMPT[: -len(PROMPT_LAST_SENTENCE)]
STACK_SENTENCE = "Build it with r3f (react-three-fiber) v10 and Koota and pmndrs math."
# The output directory (operator, 2026-09-27: "If you aren't in the right
# directory in the harness then yes you have to name an output directory in
# the prompt"). Hermes' docker backend does not tell the model its cwd
# (agent/prompt_builder.py: "the model can `whoami && pwd` when a task
# actually needs them"), and pagoda-h1 built in /tmp/voxbuild, outside the
# mounted /workspace. Octopus names its own folder (space-shooter/).
OUTPUT_DIR_SENTENCE = "Put the project in a pagoda/ folder in the current directory."
PAGODA_STACK_PROMPT = PAGODA_CONTENT + " " + STACK_SENTENCE + " " + OUTPUT_DIR_SENTENCE
PAGODA_STACK_PROMPT_SHA256 = hashlib.sha256(PAGODA_STACK_PROMPT.encode("utf-8")).hexdigest()

TASKS = {"single-html": PROMPT, "r3f-stack": PAGODA_STACK_PROMPT}
DEFAULT_TASK = "single-html"


def task_prompt(task: str) -> tuple[str, str]:
    """(prompt text, its sha256) for a task."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {', '.join(TASKS)}")
    p = TASKS[task]
    return p, hashlib.sha256(p.encode("utf-8")).hexdigest()


DEFAULT_MODEL = "yamadori"
# Everything of ours forced off, as bench/domain's A0 (selection._forced): the
# tier's thinking is kept, so `<tier>-off` is the effort-matched control of
# `<tier>`. Retrieval, deep thinking, fan-out, check_code and repair were
# removed 2026-09-29 (docs/REMOVED.md); skills, the concept seed and the MCP
# host's tools are what is left to force off.
ALL_OFF = {"skills": False, "seed": False, "mcp_tools": False}
ARMS: dict[str, dict] = {
    "xhigh": {"effort": "xhigh", "features": None,
              "note": "our full stack at medium thinking (the planned arm)"},
    "max": {"effort": "max", "features": None,
            "note": "our full stack at xhigh thinking"},
    "low": {"effort": "low", "features": None,
            "note": "the model as it ships (the benchmark baseline); the same "
                    "thinking effort the template gets at xhigh"},
    "xhigh-off": {"effort": "xhigh", "features": dict(ALL_OFF),
                  "note": "xhigh with everything of ours forced off (effort-matched "
                          "control of xhigh)"},
    "max-off": {"effort": "max", "features": dict(ALL_OFF),
                "note": "max with everything of ours forced off (effort-matched "
                        "control of max)"},
}
DEFAULT_ARMS = ("xhigh",)
DEFAULT_REPS = 3

# A 27B writing a large page with thinking: the main share is ~132K tokens,
# and a blocking request says nothing until it ends. Four hours covers the
# whole share at ~9 tok/s (the slowest decode seen on the shared card).
DEFAULT_TIMEOUT = 4 * 3600.0
BUSY_WAIT_S = 30.0
DONE = ("ok", "length")
ERRORS = ("http_error", "exception", "mismatch")


# ------------------------------------------------------------------ arms --
def parse_arm(spec: str) -> dict:
    """`xhigh` or `xhigh@some-model` -> the arm as it will be sent."""
    name, _, model = spec.strip().partition("@")
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
    a = ARMS[name]
    feats = dict(a["features"]) if a["features"] is not None else None
    return {"arm": spec.strip(), "base": name, "effort": a["effort"],
            "features": feats, "model": model or DEFAULT_MODEL, "note": a["note"]}


def request_body(arm: dict, max_tokens: int | None = None, task: str = DEFAULT_TASK) -> dict:
    body = {"model": arm["model"], "reasoning_effort": arm["effort"],
            "messages": [{"role": "user", "content": task_prompt(task)[0]}]}
    if max_tokens:
        body["max_tokens"] = int(max_tokens)
    return body


def request_headers(key: str, arm: dict) -> dict:
    h = {"Content-Type": "application/json", "Connection": "close",
         "Authorization": f"Bearer {key}"}
    if arm["features"] is not None:
        h["X-Yamadori-Features"] = json.dumps(arm["features"], sort_keys=True)
    return h


# ------------------------------------------------------------ extraction --
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([\w+.#-]*)[^\n]*\n", re.M)
_DOC_START = re.compile(r"<!doctype\s+html|<html[\s>]", re.I)
_DOC_END = re.compile(r"</html\s*>", re.I)


def fenced_blocks(text: str) -> list[dict]:
    """Every fenced block: {lang, body, closed, start}. An unclosed fence runs
    to the end of the text (a reply cut off mid-file)."""
    out, pos = [], 0
    while True:
        m = _FENCE.search(text, pos)
        if not m:
            return out
        fence, lang = m.group(1), m.group(2).lower()
        close = re.compile(r"^[ \t]*" + re.escape(fence[0]) + "{" + str(len(fence))
                           + r",}[ \t]*$", re.M)
        c = close.search(text, m.end())
        body = text[m.end(): c.start()] if c else text[m.end():]
        if c and body.endswith("\n"):          # the line break before the closing fence
            body = body[:-1]
        out.append({"lang": lang, "body": body, "closed": bool(c), "start": m.start()})
        if not c:
            return out
        pos = c.end()


def _is_page(body: str) -> bool:
    return bool(_DOC_START.search(body)) or bool(re.search(r"<(body|canvas|script)[\s>]", body, re.I))


def extract_html(content: str) -> dict:
    """The single HTML file in an answer. Returns {html, method, ...}; html is
    None when there is none. Method: `fenced` (a fenced block that is a page:
    the largest such, the rest counted), `raw` (a <!doctype/<html> ... </html>
    span in the text) or `none`. Nothing is repaired: an unclosed fence or a
    missing </html> is extracted as written and flagged."""
    blocks = fenced_blocks(content or "")
    langs: dict[str, int] = {}
    for b in blocks:
        langs[b["lang"] or "(none)"] = langs.get(b["lang"] or "(none)", 0) + 1
    pages = [b for b in blocks if _is_page(b["body"])]
    rec: dict = {"fenced_blocks": len(blocks), "fenced_langs": langs,
                 "page_blocks": len(pages)}
    if pages:
        best = max(pages, key=lambda b: len(b["body"]))
        html = best["body"]
        rec.update(method="fenced", lang=best["lang"], closed_fence=best["closed"])
    else:
        m = _DOC_START.search(content or "")
        if not m:
            rec.update(html=None, method="none", chars=0)
            return rec
        e = None
        for e in _DOC_END.finditer(content, m.start()):
            pass
        html = content[m.start(): e.end()] if e else content[m.start():]
        rec.update(method="raw", closed_fence=None)
    rec.update(html=html, chars=len(html),
               has_doctype=bool(re.search(r"<!doctype\s+html", html, re.I)),
               has_html_close=bool(_DOC_END.search(html)),
               other_blocks=len(blocks) - (1 if pages else 0),
               # flat-text hints only (the browser check records what loads)
               three_js=bool(re.search(r"\bthree(\.module)?(\.min)?\.js\b|['\"]three['\"]|"
                                       r"\bTHREE\.", html)),
               hosts_named=sorted(set(h.lower() for h in re.findall(
                   r"['\"(]https?://([\w.-]+)", html)))[:30])
    return rec


# ---------------------------------------------------- project extraction --
# Task r3f-stack: the answer is a project, one fenced block per file, in
# whatever layout the model chose (the prompt states none). A block's path
# comes from, in order: a `File: <path>` line (the nearest non-blank line
# before the fence; the first prompt asked for it, and models write it
# unasked), the fence's info string (```tsx src/App.tsx, ```tsx:src/App.tsx,
# title="src/App.tsx"), a heading / bold / backticked path on the line before
# the fence, or a first-line comment naming the path (// src/App.tsx, /* ...
# */, <!-- ... -->). When no block is named index.html, the largest UNNAMED
# block that is an HTML page becomes index.html (source `page`: a reply that
# is one page with an import map is a project of one file).
# Nothing is repaired: a body is written as extracted; a path that is
# absolute, has a drive or `..` is refused and recorded; a repeated path keeps
# its LAST block and is recorded.
_PFENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\n]*)$")
_CONVENTION = re.compile(r"^[ \t]*(?:[#>*_-]+[ \t]*)*File:[ \t]*[`*_\"']*([^`*\"'\s]+)[`*_\"']*"
                         r"[ \t]*:?[ \t]*$", re.I)
_CONVENTION_EXACT = re.compile(r"^File: (\S+)$")
_INFO_ATTR = re.compile(r"""\b(?:title|file|filename|path|name)\s*=\s*["']?([^"'\s]+)""", re.I)
_COMMENT_PATH = re.compile(r"^\s*(?://+|/\*+|<!--|#)\s*(?:(?:file(?:name)?|path)\s*:\s*)?"
                           r"[`\"']?([^`\"'\s*]+?)[`\"']?\s*(?:\*+/|-->)?\s*$", re.I)
_WIN_BAD = set('<>:"|?*')
PROJECT_MAX_FILES = 400        # a breaker, not a gate: a reply with more is recorded, cut


def _pathlike(tok: str) -> bool:
    """A file path's shape: no spaces, not a URL, and a last segment with an
    extension that starts with a letter (App.tsx, vite.config.ts) or a
    dotfile (.gitignore). Absolute and `..` paths are pathlike on purpose:
    they are then REFUSED and recorded, never silently unnamed."""
    tok = tok.strip()
    if (not tok or len(tok) > 240 or re.search(r"\s", tok)
            or re.match(r"^[a-z][a-z0-9+.-]*://", tok, re.I)
            or not re.fullmatch(r"[\w@.$+~/\\:\[\]-]+", tok)):
        return False
    last = re.split(r"[\\/]", tok)[-1]
    return bool(re.fullmatch(r"[\w@$+~\[\]-]+(?:\.[A-Za-z][A-Za-z0-9]*)+", last)
                or re.fullmatch(r"\.[A-Za-z][\w.-]*", last))


def project_blocks(text: str) -> list[dict]:
    """Every fenced block with its info string and the lines before it:
    {info, lang, body, closed, line, before}. Fences of the same character and
    at least the opener's length NEST: an inner fence with an info string
    (```bash inside a README's ```markdown) opens a level and a bare one closes
    it, so a README's own code blocks stay inside it. An unclosed block runs
    to the end of the text (a reply cut off mid-file) -- except that an
    opening fence right after a `File:` line is the NEXT file, so the block
    before it ends there, unclosed (flagged, not repaired)."""
    lines = text.split("\n")

    def before(i: int) -> str:
        k = i - 1
        while k >= 0 and not lines[k].strip():
            k -= 1
        return lines[k] if k >= 0 else ""

    out, i = [], 0
    while i < len(lines):
        m = _PFENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, info = m.group(1), m.group(2).strip()
        if fence[0] == "`" and "`" in info:          # inline code, not a fence
            i += 1
            continue
        depth, j, nested = 0, i + 1, 0
        close, next_file = None, None
        while j < len(lines):
            n = _PFENCE.match(lines[j])
            if n and n.group(1)[0] == fence[0] and len(n.group(1)) >= len(fence):
                if n.group(2).strip():
                    if depth == 0 and _CONVENTION.match(before(j)):
                        next_file = j
                        break
                    depth += 1
                    nested += 1
                elif depth:
                    depth -= 1
                else:
                    close = j
                    break
            j += 1
        end = close if close is not None else (next_file if next_file is not None else len(lines))
        body_lines = lines[i + 1: end]
        if next_file is not None:          # drop the File: line (and blanks) that belong to the next
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            if body_lines and _CONVENTION.match(body_lines[-1]):
                body_lines.pop()
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
        lang = re.split(r"[\s:{]", info, 1)[0].lower() if info else ""
        out.append({"info": info, "lang": lang, "body": "\n".join(body_lines),
                    "closed": close is not None, "line": i + 1, "before": before(i),
                    "nested_fences": nested})
        if close is not None:
            i = close + 1
        elif next_file is not None:
            i = next_file
        else:
            return out
    return out


def _path_from_info(info: str) -> str | None:
    if not info:
        return None
    m = _INFO_ATTR.search(info)
    if m:
        return m.group(1)
    toks = info.split()
    first = toks[0]
    if ":" in first and not re.match(r"^[A-Za-z]:[\\/]", first):   # ```tsx:src/App.tsx
        _lang, _, rest = first.partition(":")
        if _pathlike(rest):
            return rest
    for t in toks[1:]:                                             # ```tsx src/App.tsx
        if _pathlike(t.strip("`'\"")):
            return t.strip("`'\"")
    if len(toks) == 1 and _pathlike(first):                        # ```src/App.tsx
        return first
    return None


def _path_from_heading(line: str) -> str | None:
    """A heading / bold / list line naming one path: `### src/App.tsx`,
    `**src/App.tsx**`, `` `src/App.tsx` ``, `1. **`src/App.tsx`**:`."""
    s = line.strip()
    if not s or len(s) > 200:
        return None
    ticked = [t for t in re.findall(r"`([^`\n]+)`", s) if _pathlike(t.strip())]
    if len(ticked) == 1:
        return ticked[0].strip()
    core = re.sub(r"^(?:#{1,6}\s*|>\s*|[-*+]\s+|\d+[.)]\s+)+", "", s)
    core = core.strip().strip("*_`").strip().rstrip(":").strip().strip("*_`").strip()
    if _pathlike(core):
        return core
    return None


def _path_from_comment(body: str) -> str | None:
    first = next((ln for ln in body.split("\n") if ln.strip()), "")
    m = _COMMENT_PATH.match(first)
    if m and _pathlike(m.group(1)) and ("/" in m.group(1) or "." in m.group(1)):
        return m.group(1)
    return None


def normalise_path(raw: str) -> tuple[str | None, str | None]:
    """(path, None) or (None, why refused). Relative, forward slashes, no
    `./`, no `..`, no absolute or drive path, no character Windows refuses."""
    p = (raw or "").strip().strip("`'\"").replace("\\", "/")
    if not p:
        return None, "empty"
    if re.match(r"^[A-Za-z]:", p):
        return None, "drive path"
    if p.startswith("/") or p.startswith("~"):
        return None, "absolute path"
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if any(x == ".." for x in parts):
        return None, "`..` in path"
    if not parts:
        return None, "empty"
    if any(c in _WIN_BAD or ord(c) < 32 for x in parts for c in x):
        return None, "character a file name cannot hold"
    return "/".join(parts), None


def extract_project(content: str) -> dict:
    """The project in an answer: {files: {path: text}, method, counts,
    problems, ...}. `method` is the path source every named block used
    (convention / info / heading / comment), `mixed`, or `none`."""
    blocks = project_blocks(content or "")
    files: dict[str, str] = {}
    sources: dict[str, int] = {}
    problems: list[dict] = []
    seen: dict[str, int] = {}
    unnamed: list[dict] = []
    exact = 0
    for b in blocks:
        raw, src = None, None
        m = _CONVENTION.match(b["before"])
        if m:
            raw, src = m.group(1), "convention"
            exact += bool(_CONVENTION_EXACT.match(b["before"]))
        if raw is None:
            raw = _path_from_info(b["info"])
            src = "info" if raw else None
        if raw is None:
            raw = _path_from_heading(b["before"])
            src = "heading" if raw else None
        if raw is None:
            raw = _path_from_comment(b["body"])
            src = "comment" if raw else None
        if raw is None:
            unnamed.append({"line": b["line"], "lang": b["lang"], "chars": len(b["body"])})
            continue
        path, why = normalise_path(raw)
        if why:
            problems.append({"kind": "refused_path", "path": raw[:200], "why": why,
                             "line": b["line"], "source": src})
            continue
        if not b["closed"]:
            problems.append({"kind": "unclosed_fence", "path": path, "line": b["line"]})
        if path in files:
            seen[path] = seen.get(path, 1) + 1
        elif len(files) >= PROJECT_MAX_FILES:
            problems.append({"kind": "too_many_files", "path": path, "line": b["line"]})
            continue
        files[path] = b["body"]
        sources[src] = sources.get(src, 0) + 1
    for path, n in seen.items():
        problems.append({"kind": "duplicate_path", "path": path, "blocks": n,
                         "kept": "last"})
    if unnamed:
        problems.append({"kind": "unnamed_blocks", "blocks": unnamed[:20], "n": len(unnamed)})
    if not any(f == "index.html" or f.endswith("/index.html") for f in files):
        pages = [b for b in blocks if b["line"] in {u["line"] for u in unnamed}
                 and _is_page(b["body"])]
        if pages:
            best = max(pages, key=lambda b: len(b["body"]))
            files["index.html"] = best["body"]
            sources["page"] = sources.get("page", 0) + 1
            unnamed = [u for u in unnamed if u["line"] != best["line"]]
            problems = [p for p in problems if p["kind"] != "unnamed_blocks"]
            if unnamed:
                problems.append({"kind": "unnamed_blocks", "blocks": unnamed[:20],
                                 "n": len(unnamed)})
            if not best["closed"]:
                problems.append({"kind": "unclosed_fence", "path": "index.html",
                                 "line": best["line"]})
    root = project_root_of(list(files))
    rel = {f[len(root) + 1:] if root else f for f in files
           if not root or f.startswith(root + "/")}
    layout = ("npm" if "package.json" in rel else "static" if "index.html" in rel
              else "none")
    method = ("none" if not files else next(iter(sources)) if len(sources) == 1 else "mixed")
    return {"files": files, "method": method,
            "counts": {"fenced_blocks": len(blocks), "named": sum(sources.values()),
                       "files": len(files), "unnamed": len(unnamed),
                       "refused": sum(p["kind"] == "refused_path" for p in problems),
                       "duplicates": len(seen), "by_source": sources,
                       "convention_exact": exact,
                       "unclosed": sum(p["kind"] == "unclosed_fence" for p in problems)},
            "problems": problems, "project_root": root, "layout": layout,
            "paths": sorted(files)[:PROJECT_MAX_FILES],
            "chars": sum(len(t) for t in files.values())}


def project_root_of(paths: list[str]) -> str:
    """Where the project sits among the extracted paths: "" when package.json
    or index.html is at the top, else the shallowest directory holding
    package.json (else index.html) -- a reply that wrote `pagoda/src/...`.
    The same idea as bench/octopus/grade.py project_root; nothing is moved."""
    for name in ("package.json", "index.html"):
        if name in paths:
            return ""
        dirs = sorted((p.count("/"), p.rsplit("/", 1)[0]) for p in paths
                      if p.endswith("/" + name))
        if dirs:
            return dirs[0][1]
    return ""


def write_project(files: dict[str, str], dest: str) -> list[dict]:
    """Write the extracted files under dest (a fresh directory), each body
    exactly as extracted. Returns the files that could not be written (a
    path that is also a directory ...)."""
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    bad = []
    for path, text in files.items():
        target = os.path.join(dest, *path.split("/"))
        if not os.path.abspath(target).startswith(os.path.abspath(dest) + os.sep):
            bad.append({"path": path, "why": "outside the project"})     # defence in depth
            continue
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        except OSError as e:
            bad.append({"path": path, "why": f"{type(e).__name__}: {e}"[:200]})
    return bad


# ------------------------------------------------------------- preflight --
_octo = None


def octopus_run():
    """bench/octopus/run.py, loaded under another name (this file is `run`
    too): its BUSY list, busy_processes() and slots_idle() are THE check."""
    global _octo
    if _octo is None:
        path = os.path.join(ROOT, "bench", "octopus", "run.py")
        spec = importlib.util.spec_from_file_location("octopus_run", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _octo = mod
    return _octo


def models_listed(url: str, key: str) -> tuple[str, list[str]]:
    req = urllib.request.Request(f"{url}/v1/models",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.load(r)
        return "200", [m.get("id") for m in d.get("data") or [] if isinstance(m, dict)]
    except urllib.error.HTTPError as e:
        return str(e.code), []
    except Exception as e:                                       # noqa: BLE001
        return f"down: {type(e).__name__}", []


def preflight(url: str, key: str, arms: list[dict], sleep=time.sleep) -> dict:
    o = octopus_run()
    out: dict = {"busy": o.busy_processes()}
    try:
        a, s1 = o.slots_idle()
        sleep(5)
        b, s2 = o.slots_idle()
        out.update(slots=[s1, s2], slots_idle=a and b)
    except Exception as e:                                       # noqa: BLE001
        out.update(slots_error=f"{type(e).__name__}: {e}"[:200], slots_idle=False)
    out["proxy"], listed = models_listed(url, key)
    out["models"] = listed
    out["unlisted_models"] = sorted({a["model"] for a in arms} - set(listed))
    out["ok"] = (not out["busy"] and out["slots_idle"] and out["proxy"] == "200"
                 and not out["unlisted_models"])
    return out


# --------------------------------------------------------------- request --
def post(url: str, key: str, body: dict, arm: dict, timeout: float) -> tuple[int, object]:
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=request_headers(key, arm))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode("utf-8", "replace"), e.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def post_admitted(url, key, body, arm, timeout, busy_max_s, sleep=time.sleep, poster=post):
    """(status, body, seconds waited, 429s). A 429 or a refused connection is
    waited on up to busy_max_s; past it the status is returned as it is
    (429) or the refusal re-raised -- both become `not_run`."""
    waited, n = 0.0, 0
    while True:
        try:
            status, d = poster(url, key, body, arm, timeout)
        except urllib.error.URLError as e:
            refused = isinstance(getattr(e, "reason", None), ConnectionRefusedError)
            if not refused or waited >= busy_max_s:
                raise
            n += 1
            sleep(BUSY_WAIT_S)
            waited += BUSY_WAIT_S
            continue
        if status != 429 or waited >= busy_max_s:
            return status, d, waited, n
        n += 1
        sleep(BUSY_WAIT_S)
        waited += BUSY_WAIT_S


def x_summary(x: dict | None) -> dict:
    """The decisions that matter for reading a rep, from x_yamadori (the
    whole record is in response.json)."""
    if not isinstance(x, dict):
        return {}
    skills = x.get("skills") if isinstance(x.get("skills"), dict) else {}
    session = x.get("session") if isinstance(x.get("session"), dict) else {}
    mcp = x.get("mcp") if isinstance(x.get("mcp"), dict) else {}
    route = x.get("route") if isinstance(x.get("route"), dict) else {}
    usage = x.get("usage") if isinstance(x.get("usage"), dict) else {}
    return {
        "tier": x.get("tier"), "tier_requested": x.get("tier_requested"),
        "effort_sent": x.get("effort_sent"), "utility": x.get("utility"),
        "route": route.get("class"),
        "skills": skills.get("ids") or skills.get("names") or [],
        "seed": (session.get("seed") or {}).get("word")
        if isinstance(session.get("seed"), dict) else None,
        "mcp_calls": len(mcp.get("calls") or []),
        "images": len(x.get("images") or []),
        "hops": x.get("hops"),
        "generations": usage.get("generations"),
        "budget": x.get("budget"),
        "cache": x.get("cache"),
        "session_source": session.get("source"),
        "energy": x.get("energy"),
    }


def verify(arm: dict, x: dict | None) -> list[str]:
    """Where x_yamadori says the stack did not run the arm asked for."""
    if not isinstance(x, dict):
        return ["no x_yamadori on the response"]
    bad = []
    if x.get("utility"):
        bad.append("served as a utility call (tier overridden to minimal)")
    if x.get("tier") != arm["effort"]:
        bad.append(f"tier {x.get('tier')!r}, arm asked {arm['effort']!r}")
    return bad


def rep_dirname(arm: str, rep: int, attempt: int) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]', '-', arm)}__r{rep}__a{attempt}"


def read_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def last_rows(rows: list[dict]) -> dict:
    """(arm, rep) -> the last row for it."""
    out = {}
    for r in rows:
        out[(r.get("arm"), r.get("rep"))] = r
    return out


def append_row(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_rep(cfg: dict, arm: dict, rep: int, attempt: int, poster=post,
            sleep=time.sleep, clock=time.time) -> dict:
    """One rep: one request, its files, its summary row. Never raises."""
    task = cfg.get("task") or DEFAULT_TASK
    body = request_body(arm, cfg.get("max_tokens"), task)
    sent_sha = hashlib.sha256(body["messages"][0]["content"].encode("utf-8")).hexdigest()
    rdir_rel = os.path.join("reps", rep_dirname(arm["arm"], rep, attempt))
    rdir = os.path.join(cfg["run_dir"], rdir_rel)
    os.makedirs(rdir, exist_ok=True)
    row: dict = {"tag": cfg["tag"], "task": task, "arm": arm["arm"], "rep": rep,
                 "attempt": attempt, "effort": arm["effort"], "model": arm["model"],
                 "features_sent": arm["features"], "max_tokens_sent": body.get("max_tokens"),
                 "prompt_sha256": sent_sha, "started_at": clock(),
                 "dir": rdir_rel.replace(os.sep, "/")}
    t0 = clock()
    status, d, err, waited, n429 = None, None, None, 0.0, 0
    try:
        status, d, waited, n429 = post_admitted(cfg["url"], cfg["key"], body, arm,
                                                cfg["timeout"], cfg["busy_max_s"],
                                                sleep=sleep, poster=poster)
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), ConnectionRefusedError):
            status = "refused"
        err = f"{type(e).__name__}: {e}"[:300]
    except Exception as e:                                       # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]
    row["wall_s"] = round(clock() - t0 - waited, 2)
    row["busy_wait_s"], row["busy_429s"] = round(waited, 1), n429
    row["http_status"] = status
    content, reasoning, finish, x, usage = "", "", "", None, {}
    if isinstance(d, dict):
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish = ch.get("finish_reason") or ""
        x = d.get("x_yamadori")
        usage = d.get("usage") or {}
        row["model_returned"] = d.get("model")
    with open(os.path.join(rdir, "response.json"), "w", encoding="utf-8") as f:
        json.dump({"request": {"body": body, "features": arm["features"]},
                   "http_status": status, "error": err,
                   "response": d if isinstance(d, dict) else (str(d)[:20000] if d else None)},
                  f, ensure_ascii=False, indent=1)
    row["finish_reason"] = finish
    row["usage"] = usage
    row["x_usage"] = (x or {}).get("usage") if isinstance(x, dict) else None
    row["reasoning_chars"] = len(reasoning)
    row["content_chars"] = len(content)
    row["x"] = x_summary(x)
    if status == 429 or status == "refused":
        row["status"] = "not_run"
    elif err:
        row["status"], row["error"] = "exception", err
    elif status != 200:
        row["status"] = "http_error"
        row["error_body"] = (json.dumps(d) if isinstance(d, dict) else str(d))[:500]
    else:
        mism = verify(arm, x)
        row["mismatch"] = mism
        row["status"] = "mismatch" if mism else ("length" if finish == "length" else "ok")
    if row["status"] not in ("not_run",) and status == 200 and task == "r3f-stack":
        ex = extract_project(content)
        files = ex.pop("files")
        if files:
            bad = write_project(files, os.path.join(rdir, "project"))
            if bad:
                ex["problems"].append({"kind": "unwritable", "files": bad[:20], "n": len(bad)})
            ex["files_written"] = len(files) - len(bad)
            proj = rdir_rel + "/project" + ("/" + ex["project_root"] if ex["project_root"] else "")
            row["project"] = proj.replace(os.sep, "/")
        else:
            ex["files_written"] = 0
        row["extract"] = ex
        with open(os.path.join(rdir, "extract.json"), "w", encoding="utf-8") as f:
            json.dump(ex, f, indent=1)
    elif row["status"] not in ("not_run",) and status == 200:
        ex = extract_html(content)
        html = ex.pop("html")
        row["extract"] = ex
        with open(os.path.join(rdir, "extract.json"), "w", encoding="utf-8") as f:
            json.dump(ex, f, indent=1)
        if html is not None:
            with open(os.path.join(rdir, "page.html"), "w", encoding="utf-8",
                      newline="") as f:
                f.write(html)
            row["page"] = (rdir_rel + "/page.html").replace(os.sep, "/")
    return row


# ------------------------------------------------------------------ main --
def read_key(path: str | None) -> str | None:
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    return os.environ.get("YAMADORI_TEST_KEY") or None


def plan(arms: list[dict], reps: int, done: dict, retry_errors: bool) -> list[tuple[dict, int, int]]:
    """(arm, rep, attempt) still to run: a rep whose last row is done is
    skipped; `not_run` always reruns; errors only with retry_errors."""
    todo = []
    for a in arms:
        for rep in range(1, reps + 1):
            last = done.get((a["arm"], rep))
            if last:
                st = last.get("status")
                if st in DONE or (st in ERRORS and not retry_errors):
                    continue
            attempt = (last.get("attempt", 0) if last else 0) + 1
            todo.append((a, rep, attempt))
    return todo


def tag_conflict(run_dir: str, task: str) -> str | None:
    """Why this tag cannot take `task`, or None. A tag holds ONE task and one
    prompt: its manifest (and every summary row) must name the same task and
    prompt sha256. A manifest or row with no task predates the option and is
    single-html."""
    want_sha = task_prompt(task)[1]
    p = os.path.join(run_dir, "manifest.json")
    if os.path.isfile(p):
        try:
            m = json.load(open(p, encoding="utf-8"))
        except ValueError:
            return "manifest.json does not parse; refusing to guess its task"
        have = m.get("task") or DEFAULT_TASK
        if have != task:
            return f"this tag holds task {have!r}; --task {task!r} would mix results"
        if m.get("prompt_sha256") and m["prompt_sha256"] != want_sha:
            return (f"this tag's prompt sha256 is {m['prompt_sha256'][:12]}..., the "
                    f"{task!r} prompt is now {want_sha[:12]}...: the prompt changed; use a new tag")
    for r in read_rows(os.path.join(run_dir, "summary.jsonl")):
        if (r.get("task") or DEFAULT_TASK) != task:
            return (f"summary.jsonl holds a {r.get('task') or DEFAULT_TASK!r} row; "
                    f"--task {task!r} would mix results")
        if r.get("prompt_sha256") and r["prompt_sha256"] != want_sha:
            return "summary.jsonl holds a row sent with another prompt sha256; use a new tag"
    return None


def write_manifest(run_dir: str, tag: str, arms: list[dict], reps: int,
                   max_tokens: int | None, task: str = DEFAULT_TASK) -> None:
    p = os.path.join(run_dir, "manifest.json")
    m = {}
    if os.path.isfile(p):
        try:
            m = json.load(open(p, encoding="utf-8"))
        except ValueError:
            m = {}
    prompt, sha = task_prompt(task)
    m.update(tag=tag, task=task, prompt=prompt, prompt_sha256=sha,
             reps=max(reps, m.get("reps", 0)),
             max_tokens_sent=max_tokens, runner="bench/voxel/run.py")
    if task == "r3f-stack":
        m.update(stack_sentence=STACK_SENTENCE, replaces=PROMPT_LAST_SENTENCE.strip())
    known = m.setdefault("arms", {})
    for a in arms:
        known[a["arm"]] = {k: a[k] for k in ("base", "effort", "features", "model", "note")}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", help="results/<tag>/ (a rerun resumes it)")
    ap.add_argument("--task", default=DEFAULT_TASK, choices=tuple(TASKS),
                    help="single-html: the public prompt verbatim (default); r3f-stack: its "
                         "content sentences + our stack requirement (docs/BENCH-VOXEL.md)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help=f"comma-separated ARM or ARM@MODEL; known: {', '.join(ARMS)}")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--key-file")
    ap.add_argument("--url", default=PROXY)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="send max_tokens (default: none, as a chat client)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--busy-max-s", type=float, default=15 * 60.0,
                    help="how long a 429 / refused connection is waited on before not_run")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--list-arms", action="store_true")
    a = ap.parse_args(argv)
    if a.list_arms:
        for name, v in ARMS.items():
            dflt = " (default)" if name in DEFAULT_ARMS else ""
            print(f"  {name:10s} effort={v['effort']:6s} features={json.dumps(v['features'])}"
                  f"{dflt}\n             {v['note']}")
        return 0
    try:
        arms = [parse_arm(s) for s in a.arms.split(",") if s.strip()]
    except ValueError as e:
        print(f"  {e}")
        return 2
    if a.tag:
        why = tag_conflict(os.path.join(RESULTS, a.tag), a.task)
        if why:
            print(f"  REFUSED: results/{a.tag}: {why}. Nothing was sent.")
            return 2
    try:
        key = read_key(a.key_file)
    except OSError as e:
        print(f"  cannot read --key-file: {e.strerror or e}")
        return 2
    if not key:
        print("  no key: pass --key-file or set YAMADORI_TEST_KEY")
        return 2
    pf = preflight(a.url, key, arms)
    print("  preflight: " + json.dumps({k: v for k, v in pf.items() if k != "slots"}))
    if a.preflight_only:
        return 0 if pf["ok"] else 1
    if not pf["ok"]:
        print("  REFUSED: another GPU consumer, a busy slot, the proxy, or an unlisted "
              "model (see above). Nothing was sent.")
        return 1
    if not a.tag:
        print("  --tag is required")
        return 2
    run_dir = os.path.join(RESULTS, a.tag)
    os.makedirs(os.path.join(run_dir, "reps"), exist_ok=True)
    write_manifest(run_dir, a.tag, arms, a.reps, a.max_tokens, a.task)
    summary = os.path.join(run_dir, "summary.jsonl")
    todo = plan(arms, a.reps, last_rows(read_rows(summary)), a.retry_errors)
    print(f"  {len(todo)} rep(s) to run in {os.path.relpath(run_dir, ROOT)} (task {a.task})")
    cfg = {"tag": a.tag, "run_dir": run_dir, "url": a.url, "key": key, "task": a.task,
           "timeout": a.timeout, "busy_max_s": a.busy_max_s, "max_tokens": a.max_tokens}
    o = octopus_run()
    for arm, rep, attempt in todo:
        busy = o.busy_processes()
        if busy:
            print(f"  STOPPED before {arm['arm']} r{rep}: another GPU consumer started: "
                  f"{busy[:3]} (rerun the same --tag to resume)")
            return 3
        print(f"  {arm['arm']} r{rep} a{attempt} ...", flush=True)
        row = run_rep(cfg, arm, rep, attempt, poster=post)
        append_row(summary, row)
        u = row.get("usage") or {}
        ex = row.get("extract") or {}
        got = (f"project={ex.get('files_written')} files, layout {ex.get('layout')}"
               if a.task == "r3f-stack" else f"page={'yes' if row.get('page') else 'no'}")
        print(f"    {row['status']}  http={row.get('http_status')} finish={row.get('finish_reason')!r} "
              f"wall={row.get('wall_s')}s completion={u.get('completion_tokens')} "
              f"{got} ({ex.get('method')}, {ex.get('chars')} chars)", flush=True)
    rows = last_rows(read_rows(summary))
    nr = sum(1 for r in rows.values() if r.get("status") == "not_run")
    print(f"  done: {len(rows)} rep(s) recorded, {nr} not run. Next: "
          f"python bench/voxel/check.py --tag {a.tag}")
    return 3 if nr else 0


if __name__ == "__main__":
    sys.exit(main())
