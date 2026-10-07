#!/usr/bin/env python
"""The skill pipeline's job handlers. `mcp/worker.py` claims and runs them.

    fetch -> screen -> screen_model -> licence -> distil | decompose ->
    classify -> tests -> validate -> arm          watch (scheduled)

See `skills.py` for the paths each kind of version walks, the states and the
no-review decision. Each handler reads its version, does one stage, and
writes what it found onto the version; `advance_after` (called by the worker
once the job is `done`) enqueues the next stage. A stage never enqueues its
own successor, for the reason datasets.py gives: a stage is left only when
its job is done. `run_inline` runs the same handlers in order without the
queue (the migration, an authored install).

HOW A STAGE ENDS

  the version moves on        the handler returns; advance_after enqueues
                              the next stage
  quarantined / failed        the handler records it on the version and
                              RETURNS -- the job is done, the pipeline
                              stopped, and the reason is on the skill
  retryable (network, model)  the handler RAISES; the worker retries it and,
                              once attempts run out, the job is `errored`
                              and the version stays where it was, visible

A permanent refusal is a fact about the SKILL, not a job error, so it never
raises: raising would put it in `errored`, where an operator reads "job
failed" instead of "the screen quarantined this skill because ...".

THE ONE DOOR AND THE PROMPTS

Every model call is `ask_model`, which goes through `mcp/model.py`
(`tiers.apply`, the same shaping as a client request at the same effort).
Every prompt is a versioned template in `skill_prompts.py` (distil,
decompose, tag, tests, faithful; the screen's is skill_screen's), and the
version is recorded on what it produced. Tests replace `ask_model`. The
model PROPOSES; each proposal is verified deterministically (quotes in the
source, tags in the fixed vocabulary, topics in the skill, regexes that
compile) before it is kept.

FETCHING

http(s) only; robots.txt is read and obeyed for our user agent (401/403 on
robots.txt disallows everything, any other 4xx allows everything, a 5xx or
no answer is retryable -- the conventions of RFC 9309); a size cap; scripts
and binaries are refused by extension, content type and a NUL byte, and are
never stored. A GitHub page or folder is resolved to its raw file
(`skills.normalise_url`). A frontier skill's scripts/ are never fetched:
scripts are data, never armed as executable.
"""
from __future__ import annotations

import hashlib
import html.parser
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402
import skill_builder  # noqa: E402
import skill_classify  # noqa: E402
import skill_learn  # noqa: E402
import skill_limits  # noqa: E402
import skill_prompts  # noqa: E402
import skill_prove  # noqa: E402
import skill_screen  # noqa: E402
import skills  # noqa: E402

UA = "yamadori-skill-worker/1"
UA_TOKEN = "yamadori-skill-worker"
FETCH_TIMEOUT = 30
FETCH_MAX_BYTES = int(os.environ.get("YAMADORI_SKILL_FETCH_MAX_BYTES",
                                     str(1024 * 1024)))
ROBOTS_MAX_BYTES = 512 * 1024
ROBOTS_TTL = 3600.0
# One chunk is one model call (worker.CHUNK_CHARS's reasoning: ~3k tokens of
# source leaves the thinking model room to answer).
CHUNK_CHARS = int(os.environ.get("YAMADORI_SKILL_CHUNK_CHARS", "12000"))
# The worker's own bound on a source (worker.MAX_CHUNKS, 60 chunks, its
# variable YAMADORI_EXTRACT_MAX_CHUNKS): one bound for both paths (the 8 here
# was ours, docs/CONSTANTS-AUDIT.md).
MAX_CHUNKS = int(os.environ.get("YAMADORI_EXTRACT_MAX_CHUNKS", "60"))
# The ANSWER allowance for each call (tiers.A_MIN's 2,048, the stack's
# answer floor; decompose uses it too since 2026-09-27, not twice it);
# thinking is added by tiers.budget.
DISTIL_MAX_TOKENS = int(os.environ.get("YAMADORI_SKILL_DISTIL_MAX_TOKENS",
                                       "2048"))
SCREEN_MAX_TOKENS = int(os.environ.get("YAMADORI_SKILL_SCREEN_MAX_TOKENS",
                                       "1024"))
MODEL_TIMEOUT = int(os.environ.get("YAMADORI_SKILL_MODEL_TIMEOUT", "3600"))
# The model-assisted screen. On unless switched off; switching it off is
# recorded on every version it skips.
MODEL_SCREEN = os.environ.get("YAMADORI_SKILL_MODEL_SCREEN", "1") != "0"
# The other model-assisted halves (skill_prompts): tagging in classify, test
# proposals in tests, the faithfulness check in validate. Each is skipped for
# a compiled or edited version (its author wrote it), and each can be
# switched off; the skip is recorded on the version.
MODEL_TAG = os.environ.get("YAMADORI_SKILL_MODEL_TAG", "1") != "0"
MODEL_TESTS = os.environ.get("YAMADORI_SKILL_MODEL_TESTS", "1") != "0"
MODEL_VALIDATE = os.environ.get("YAMADORI_SKILL_MODEL_VALIDATE", "1") != "0"

REFUSED_EXT = {".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat",
               ".cmd", ".vbs", ".exe", ".msi", ".dll", ".so", ".dylib",
               ".bin", ".py", ".pyc", ".js", ".mjs", ".cjs", ".jar", ".class",
               ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
               ".whl", ".deb", ".rpm", ".apk", ".dmg", ".iso", ".wasm",
               ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
ALLOWED_TYPES = {"text/plain", "text/markdown", "text/x-markdown",
                 "text/html", "application/xhtml+xml", "text/x-rst",
                 "application/markdown"}


def describe(situation: str, retryable: bool, remedy: str, owner: str) -> str:
    """worker.describe's shape, so the dashboard prints one format."""
    return (f"{situation} | retryable: {'yes' if retryable else 'no'} | "
            f"remedy ({owner}): {remedy}")


class Refused(Exception):
    """A permanent refusal while fetching: recorded on the version, not
    raised out of the handler."""


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------
_ROBOTS: dict[str, tuple[float, object]] = {}


def robots_allows(url: str) -> dict:
    """{allowed, robots_url, status, why}. Raises RuntimeError when robots.txt
    cannot be read for a reason that may pass (5xx, no answer)."""
    p = urllib.parse.urlsplit(url)
    base = f"{p.scheme}://{p.netloc}"
    robots_url = f"{base}/robots.txt"
    hit = _ROBOTS.get(base)
    if hit and time.time() - hit[0] < ROBOTS_TTL:
        verdict = hit[1]
    else:
        req = urllib.request.Request(robots_url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                body = r.read(ROBOTS_MAX_BYTES).decode("utf-8", "replace")
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(body.splitlines())
            verdict = ("parsed", 200, rp)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                verdict = ("disallow_all", e.code, None)
            elif 400 <= e.code < 500:
                verdict = ("allow_all", e.code, None)
            else:
                raise RuntimeError(describe(
                    f"{robots_url} answered HTTP {e.code}", True,
                    "retried automatically; robots.txt must be readable "
                    "before the page is fetched", "worker")) from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(describe(
                f"{robots_url} did not answer ({e})", True,
                "retried automatically; check the host is reachable",
                "worker")) from e
        _ROBOTS[base] = (time.time(), verdict)
    kind, status, rp = verdict
    if kind == "parsed":
        ok = rp.can_fetch(UA_TOKEN, url)
        why = ("robots.txt allows it" if ok else
               f"robots.txt disallows {p.path or '/'} for {UA_TOKEN}")
    elif kind == "disallow_all":
        ok, why = False, f"robots.txt answered HTTP {status}: disallow all"
    else:
        ok, why = True, f"robots.txt answered HTTP {status}: allow all"
    return {"allowed": ok, "robots_url": robots_url, "status": status,
            "why": why}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_source(url: str, beat=None) -> tuple[bytes, dict]:
    """GET one skill source under every rule above. Raises Refused for a
    permanent refusal, RuntimeError for a retryable one."""
    if not re.match(r"^https?://", url or ""):
        raise Refused(f"{url!r} is not an http(s) URL")
    ext = os.path.splitext(urllib.parse.urlsplit(url).path)[1].lower()
    if ext in REFUSED_EXT:
        raise Refused(f"{ext} files are scripts or binaries and are never "
                      "ingested as skills")
    robots = robots_allows(url)
    if not robots["allowed"]:
        raise Refused(f"{robots['why']} ({robots['robots_url']})")
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "text/markdown, "
                                               "text/plain, text/html;q=0.8"})
    try:
        resp = urllib.request.urlopen(req, timeout=FETCH_TIMEOUT)
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500 and e.code not in (408, 425, 429):
            raise Refused(f"GET {url} answered HTTP {e.code}") from e
        raise RuntimeError(describe(f"GET {url} answered HTTP {e.code}", True,
                                    "retried automatically", "worker")) from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(describe(f"GET {url} failed: {e}", True,
                                    "retried automatically", "worker")) from e
    with resp:
        ctype = resp.headers.get_content_type()
        charset = resp.headers.get_content_charset()
        final = resp.geturl()
        if ctype not in ALLOWED_TYPES and not (
                ctype == "application/octet-stream"
                and os.path.splitext(urllib.parse.urlsplit(final).path)[1]
                .lower() in (".md", ".markdown", ".txt", ".mdx")):
            raise Refused(f"{url} is {ctype}, not text, markdown or HTML")
        h = hashlib.sha256()
        buf = bytearray()
        while True:
            block = resp.read(64 * 1024)
            if not block:
                break
            buf += block
            h.update(block)
            if len(buf) > FETCH_MAX_BYTES:
                raise Refused(f"{url} is larger than {FETCH_MAX_BYTES:,} "
                              "bytes (YAMADORI_SKILL_FETCH_MAX_BYTES); a skill "
                              "is a compact document")
            if beat:
                beat(f"fetched {len(buf):,} bytes")
        status = resp.status
    raw = bytes(buf)
    if b"\x00" in raw[:8192]:
        raise Refused(f"{url} contains NUL bytes: a binary, not a document")
    return raw, {"url": url, "final_url": final, "status": status,
                 "content_type": ctype, "charset": charset,
                 "fetched_at": time.time(), "robots": robots}


# ---------------------------------------------------------------------------
# Source text. HTML is reduced to text with hidden elements, comments,
# scripts and styles DROPPED -- the screen reports them; the model never
# reads them.
# ---------------------------------------------------------------------------
class _PageText(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template",
            "iframe", "object", "embed", "nav", "footer"}
    BLOCK = {"p", "div", "li", "pre", "br", "tr", "h1", "h2", "h3", "h4",
             "h5", "h6", "section", "article", "blockquote", "table", "dt",
             "dd", "ul", "ol", "main"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[tuple[str, bool]] = []

    def _skipping(self) -> bool:
        return any(h for _t, h in self.stack)

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        hidden = (tag in self.SKIP or "hidden" in a
                  or a.get("aria-hidden", "").lower() == "true"
                  or bool(skill_screen._HIDDEN_STYLE.search(a.get("style", ""))))
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag in self.VOID:
            return
        self.stack.append((tag, hidden))
        if tag == "pre" and not self._skipping():
            self.out.append("\n```\n")

    def handle_endtag(self, tag):
        # Pop to the matching open tag: real pages leave elements unclosed,
        # and popping blindly would un-hide text inside a hidden element.
        idx = next((i for i in range(len(self.stack) - 1, -1, -1)
                    if self.stack[i][0] == tag), None)
        if tag in self.VOID or idx is None:
            return
        if tag == "pre" and not self._skipping():
            self.out.append("\n```\n")
        del self.stack[idx:]
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self._skipping():
            self.out.append(data)


def page_text(raw_html: str) -> str:
    p = _PageText()
    p.feed(raw_html)
    p.close()
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.out))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def source_texts(sid: str, v: int) -> tuple[str, str, str]:
    """(raw, text the model reads, kind) for a version. An edit's source is
    the text the operator submitted. The text the model reads -- and every
    quote is verified against -- has zero-width typography stripped
    (skill_screen.strip_typography; the screen records the removal)."""
    raw, text, kind = _source_texts(sid, v)
    return raw, skill_screen.strip_typography(text)[0], kind


def _source_texts(sid: str, v: int) -> tuple[str, str, str]:
    ver = skills.version(sid, v) or {}
    if ver.get("origin") == "edit":
        t = (ver.get("distil") or {}).get("submitted") or ""
        return t, t, "markdown"
    got = skills.source_raw(sid, v)
    if got is None:
        return "", "", "text"
    raw_b, meta = got
    raw = raw_b.decode(meta.get("charset") or "utf-8", errors="replace")
    ctype = meta.get("content_type") or ""
    if meta.get("kind") in ("migration", "dataset", "authored"):
        # Our own rendering of reviewed recipe rows: plain text, never
        # rendered as markdown (skill_screen.check_hidden_html).
        return raw, raw, "text"
    if "html" in ctype or re.search(r"(?i)<html|<body", raw[:2000]):
        return raw, page_text(raw), "html"
    return raw, raw, "markdown"


def chunks_of(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraph-bounded chunks (worker.chunks_of's rule)."""
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.append(para[:size])
            para = para[size:]
        if cur and len(cur) + len(para) + 2 > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def parse_object(reply: str) -> dict:
    reply = re.sub(r"(?s)<think>.*?</think>", "", reply or "")
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(reply[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


# The pipeline job this thread is running (`skill.<purpose>`), as shomen
# names its job on the thread (shomen._CURRENT): ask_model reads it for the
# job's thinking cap.
import threading as _threading  # noqa: E402

_CURRENT = _threading.local()


def job_name(purpose: str) -> str:
    return f"skill.{purpose or 'model'}"


def shaped_body(system: str, user: str, *, max_tokens: int,
                purpose: str = "") -> dict:
    """The upstream body of one pipeline stage's generation: the HELPER role
    and the job's thinking cap, the same path shomen's hops take
    (model.shape(role="helper", step_cap=tiers.JOB_THINKING.get(job))). A
    pipeline job has no JOB_THINKING row of its own, so tiers.budget caps it
    at HELPER_THINKING (6,144: the operator's 1.5x of 2026-09-25). Evidence
    for a cap at that level: natural thinking ran 682-2,826 tokens on coding
    prompts, 5 of 7 within 2,048 (docs/CONSTRAINTS.md 1b, n=7). Before
    2026-09-27 these calls ran as role main with no job: thinking was the
    main share less the prompt (~125k tokens), capped by nothing."""
    import model
    import tiers
    job = job_name(purpose)
    body = model.shape({"messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
                        "max_tokens": max_tokens},
                       effort="medium", role="helper",
                       step_cap=tiers.JOB_THINKING.get(job),
                       nudge=tiers.helper_nudge(job))
    # Accounted as internal work in the token ledger, as before (the helper
    # share is where its thinking comes from, not the second brain's run).
    body.pop("_share", None)
    return body


def ask_model(system: str, user: str, *, max_tokens: int,
              purpose: str = "") -> str:
    """One generation through the one door (mcp/model.py), at the tier's
    (the vendor's) sampling: no temperature of ours (removed 2026-09-27,
    docs/CONSTANTS-AUDIT.md "pipeline sampling"), as the named helper job
    `skill.<purpose>` (shaped_body: the helper thinking cap). Replaced in
    tests. A length finish is a budget event and is retried, never read as
    an answer."""
    import model
    prev, _CURRENT.job = getattr(_CURRENT, "job", None), job_name(purpose)
    try:
        content = model.answer(model.post(
            shaped_body(system, user, max_tokens=max_tokens, purpose=purpose),
            timeout=MODEL_TIMEOUT))
    except model.BudgetEvent as e:
        raise RuntimeError(describe(
            f"the {purpose or 'skill'} call hit the token limit: {e}", True,
            "lower YAMADORI_SKILL_CHUNK_CHARS or raise the stage's "
            "MAX_TOKENS", "operator")) from e
    finally:
        _CURRENT.job = prev
    if not (content or "").strip():
        raise RuntimeError(describe(
            f"the model wrote nothing for the {purpose or 'skill'} call", True,
            "retried automatically; if it repeats, check the model on "
            "llama-swap", "operator"))
    return content


# ---------------------------------------------------------------------------
# Licences. Established only from a verbatim quote: a SKILL.md's own
# `license:` line, a licence statement in the source, a LICENSE file next
# to it (same host; a GitHub repo's raw LICENSE), or the operator's own
# statement (skills.set_licence). A guessed licence is worse than none
# (datasets.py): a source whose licence nobody established FAILS at this
# stage with the remedy, and a no-derivatives licence fails outright --
# distilling it may be the prohibited act.
# ---------------------------------------------------------------------------
SPDX = (
    (r"\bMIT\b(?: License)?", "MIT"),
    (r"\bApache(?: License)?,? (?:Version )?2\.0\b|\bApache-2\.0\b",
     "Apache-2.0"),
    (r"\bBSD[- ]3[- ]Clause\b", "BSD-3-Clause"),
    (r"\bBSD[- ]2[- ]Clause\b", "BSD-2-Clause"),
    (r"\bISC License\b|\bISC\b", "ISC"),
    (r"\bMozilla Public License,? (?:Version )?2\.0\b|\bMPL-2\.0\b",
     "MPL-2.0"),
    (r"\bCC[- ]BY[- ]SA[- ]4\.0\b|Attribution-ShareAlike 4\.0",
     "CC-BY-SA-4.0"),
    (r"\bCC[- ]BY[- ]NC[- ]?(?:SA[- ]?)?4\.0\b|NonCommercial",
     "CC-BY-NC-4.0"),
    (r"\bCC[- ]BY[- ]ND[- ]4\.0\b|NoDerivatives", "CC-BY-ND-4.0"),
    (r"\bCC[- ]BY[- ]4\.0\b|Creative Commons Attribution 4\.0", "CC-BY-4.0"),
    (r"\bCC0\b|\bCC0-1\.0\b|public domain dedication", "CC0-1.0"),
    (r"\bUnlicense\b", "Unlicense"),
    (r"\bAGPL(?:-?3\.0)?\b|Affero General Public License", "AGPL-3.0"),
    (r"\bLGPL(?:-?[23]\.\d)?\b|Lesser General Public License", "LGPL"),
    (r"\bGPL(?:-?[23]\.0)?\b|GNU General Public License", "GPL"),
)
_SPDX = [(re.compile(p, re.I), sid) for p, sid in SPDX]
_LICENCE_LINE = re.compile(
    r"licen[cs]|copyright|all rights reserved|public domain|creative commons"
    r"|\bcc[- ]?by|\bcc0\b|spdx|redistribut", re.I)
NO_DERIVATIVES = ("CC-BY-ND-4.0",)
# A URL is an address, never a licence statement (2026-10-07: a react.dev page
# whose HTML example plays `.../media/cc0-videos/flower.mp4` was recorded as
# CC0-1.0, quote and all). Every URL, and every bare host/path, is blanked
# from a line before the line is read for a licence; the quote stays the
# line as it is in the source.
_URLISH = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://|www\.)[^\s\"'<>)\]]*"
    r"|[\w-]+(?:\.[\w-]+)+/[^\s\"'<>)\]]*", re.I)


def licence_of(text: str) -> dict | None:
    """{spdx, quote, where} from a verbatim line of the text, or None. A URL
    inside a line is not read: only the words around it."""
    import skill_md
    fm, _body = skill_md.split(text or "")
    for key in ("license", "licence"):
        v = fm.get(key)
        if isinstance(v, str) and v.strip():
            m = re.search(r"(?im)^" + key + r"\s*:\s*(.+)$", text or "")
            quote = m.group(0).strip() if m else f"{key}: {v}"
            spdx = next((sid for rx, sid in _SPDX if rx.search(v)),
                        v.strip()[:80])
            return {"spdx": spdx, "quote": quote, "where": "frontmatter"}
    for line in (text or "").split("\n"):
        scan = _URLISH.sub(" ", line)
        if not _LICENCE_LINE.search(scan):
            continue
        for rx, sid in _SPDX:
            if rx.search(scan):
                return {"spdx": sid, "quote": line.strip()[:600],
                        "where": "source"}
    return None


# WHERE A REPOSITORY STATES ITS LICENCE (coordinator, 2026-10-07: react.dev
# carries ONLY LICENSE-DOCS.md at its pinned commit, CC BY 4.0, and the stage
# looked for LICENSE* alone, so three of its pages failed with "no licence").
# For a source on GitHub the stage reads the repository's own licence files AT
# THE PINNED REF the source URL names, in this order, and fills the licence
# only from a verbatim quote of the file it reads (licence_of: its SPDX line
# or the licence's own name sentence), recorded with that file's URL:
#   1. a DOCS licence file, only when the source is a DOCUMENTATION page
#      (THE RULE below);
#   2. the general files: LICENSE*, LICENCE*, COPYING*.
# A docs licence file is a statement about the docs, so it is never read for a
# code path: a repository whose LICENSE-DOCS covers `docs/` says nothing about
# `src/app.js`.
DOCS_LICENCE_FILES = ("LICENSE-DOCS.md", "LICENSE-DOCS", "LICENSE-DOCS.txt",
                      "LICENSE_DOCS.md", "LICENCE-DOCS.md", "LICENCE-DOCS",
                      "LICENSE-DOCUMENTATION.md")
GENERAL_LICENCE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE",
                         "LICENCE.md", "LICENCE.txt", "COPYING",
                         "COPYING.md", "COPYING.txt")
# THE RULE that ties a docs licence file to a path: the path is a text or
# markup document (DOC_EXTENSIONS) under a directory that holds documentation
# (DOC_DIRS: any segment of the path, `src/content/...` included). A README
# at the root, source code and data are not documentation pages. The lists
# are the usual names of documentation trees (docs/, content/, website/, the
# react.dev layout src/content/); a path outside them falls back to the
# general files.
DOC_EXTENSIONS = (".md", ".mdx", ".markdown", ".rst", ".adoc", ".txt",
                  ".html", ".htm")
DOC_DIRS = ("docs", "doc", "documentation", "content", "website", "wiki",
            "guide", "guides", "manual", "handbook", "blog")
MAX_LICENCE_FETCHES = 16


def github_source(url: str) -> dict | None:
    """{owner, repo, ref, path} for a GitHub-hosted source URL (raw, blob or
    tree), else None."""
    import urllib.parse
    p = urllib.parse.urlsplit(url or "")
    host = (p.hostname or "").lower()
    parts = [x for x in p.path.split("/") if x]
    if host == "raw.githubusercontent.com" and len(parts) >= 3:
        return {"owner": parts[0], "repo": parts[1], "ref": parts[2],
                "path": "/".join(parts[3:])}
    if host == "github.com" and len(parts) >= 5 and parts[2] in (
            "blob", "tree"):
        return {"owner": parts[0], "repo": parts[1], "ref": parts[3],
                "path": "/".join(parts[4:])}
    return None


def is_docs_path(path: str) -> bool:
    segs = [x.lower() for x in (path or "").split("/") if x]
    if not segs or not segs[-1].endswith(DOC_EXTENSIONS):
        return False
    return any(sg in DOC_DIRS for sg in segs[:-1])


def licence_files_for(url: str) -> list[dict]:
    """[{url, kind, rule}] in the order they are read: for a GitHub source,
    the repository's licence files at the URL's ref (docs files first, only
    for a documentation path); for any other host, worker.licence_candidates
    as before."""
    g = github_source(url)
    if g is None:
        import worker
        return [{"url": u, "kind": "general",
                 "rule": "the host's own LICENSE beside the page or at its "
                         "root (worker.licence_candidates)"}
                for u in worker.licence_candidates(url)]
    base = (f"https://raw.githubusercontent.com/{g['owner']}/{g['repo']}/"
            f"{g['ref']}/")
    out: list[dict] = []
    if is_docs_path(g["path"]):
        out += [{"url": base + n, "kind": "docs",
                 "rule": f"{g['path']} is a documentation page (a text or "
                         "markup file under a docs directory), so the "
                         "repository's docs licence file applies to it"}
                for n in DOCS_LICENCE_FILES]
    out += [{"url": base + n, "kind": "general",
             "rule": "the repository's own licence file at the pinned ref"}
            for n in GENERAL_LICENCE_FILES]
    return out[:MAX_LICENCE_FETCHES]


# CC BY asks for attribution (credit, the licence's link, a note of changes);
# the provenance records what it asks for (attribution_of).
_CC_URL = {"CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
           "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
           "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/"}


def attribution_of(url: str, lic: dict) -> dict | None:
    """What a CC BY licence asks the reuser to carry: who to credit (the
    repository and the commit), the licence and its link, where the licence
    text was read, and that the source was changed. None for a licence that
    asks nothing of the kind."""
    spdx = str((lic or {}).get("spdx") or "")
    if not spdx.startswith("CC-BY"):
        return None
    g = github_source(url)
    credit = (f"{g['owner']}/{g['repo']}" if g else str(url or "")[:120])
    commit = g["ref"] if g and re.fullmatch(r"[0-9a-f]{40}", g["ref"]) \
        else None
    out = {"credit": credit, "source_url": url, "licence": spdx,
           "licence_url": _CC_URL.get(spdx),
           "licence_file": (lic or {}).get("where"),
           "changes": "distilled into short DO / DO NOT lines; the source's "
                      "wording is shortened and rearranged",
           "no_endorsement": True}
    if commit:
        out["commit"] = commit
    return {k: v for k, v in out.items() if v is not None}


def handle_licence(job: dict, ctx) -> dict:
    t = _target(job, "licence")
    if t is None:
        return _skipped(job, "licence")
    sid, v, ver = t
    s = skills.get(sid) or {}
    op = (s.get("meta") or {}).get("licence")
    searched = []
    lic = None
    if op:
        lic = {"spdx": op["spdx"], "quote": op["quote"], "where": "operator",
               "by": op.get("by")}
    if lic is None:
        raw, _text, _kind = source_texts(sid, v)
        lic = licence_of(raw)
        searched.append("the source text" + (" (found)" if lic else ""))
    pkg_lic = (s.get("meta") or {}).get("package_licence")
    if lic is None and isinstance(pkg_lic, dict) and pkg_lic.get("quote"):
        # A source a PACKAGE ONBOARDING read from the package's own tarball
        # (no tag for the version: docs/PACKAGE-ONBOARDING.md 2.1): the
        # package's licence, from the verbatim quote resolve established,
        # with where it was found.
        lic = {"spdx": pkg_lic.get("spdx"), "quote": pkg_lic["quote"],
               "where": pkg_lic.get("where") or "the package's licence"}
        searched.append("the package's licence (found)")
    url = s.get("source_url") or ""
    if lic is None and url:
        for cf in licence_files_for(url)[:MAX_LICENCE_FETCHES]:
            cand = cf["url"]
            ctx.beat(f"looking for a licence at {cand}")
            try:
                raw_l, _meta = fetch_source(cand)
            except (Refused, RuntimeError) as e:
                searched.append(f"{cand} ({str(e)[:80]})")
                continue
            got = licence_of(raw_l.decode("utf-8", "replace"))
            searched.append(f"{cand} ({'found' if got else 'no licence line'})")
            if got:
                lic = dict(got, where=cand, file_kind=cf["kind"],
                           file_rule=cf["rule"])
                break
    if lic is None and ver.get("origin") in ("ingest",) and not url:
        # Pasted by an operator: their material, their call. Recorded as
        # such, never as a licence the text grants.
        lic = {"spdx": "operator-supplied", "quote": None,
               "where": "pasted by an operator; the text states no licence"}
    rec = dict(lic or {}, searched=searched)
    skills.update_version(sid, v, licence=rec)
    if lic is None:
        skills.fail(sid, v, "licence: no licence could be established from "
                    "a verbatim quote (searched " + "; ".join(searched)[:300]
                    + "). Remedy (operator): POST /dash/api/skill/licence "
                    "{id, licence, quote}, then re-run the licence stage")
        return {"failed": "no licence"}
    if lic.get("spdx") in NO_DERIVATIVES and lic.get("where") != "operator":
        skills.fail(sid, v, f"licence: {lic['spdx']} forbids derivatives; "
                    "distilling it may be the prohibited act. Remedy "
                    "(operator): state a licence that allows it, or drop "
                    "the source")
        return {"failed": "no-derivatives"}
    return {"licence": lic.get("spdx"), "where": lic.get("where")}


# ---------------------------------------------------------------------------
# Handlers.
# ---------------------------------------------------------------------------
def _target(job: dict, stage: str) -> tuple[str, int, dict] | None:
    p = job.get("payload") or {}
    sid, v = p.get("skill"), p.get("version")
    ver = skills.version(sid, v) if sid and v else None
    if ver is None or ver["state"] != "running" or ver["stage"] != stage:
        return None
    return sid, int(v), ver


def _skipped(job: dict, stage: str) -> dict:
    p = job.get("payload") or {}
    ver = skills.version(p.get("skill"), p.get("version")) if p.get("skill") \
        else None
    return {"skipped": (f"v{p.get('version')} of {p.get('skill')} is "
                        f"{ver['state']} at {ver['stage']}, not running at "
                        f"{stage}" if ver else "no such skill version")}


def _compiled(ver: dict) -> bool:
    return ver.get("origin") in skills.COMPILED or bool(
        (ver.get("distil") or {}).get("compiled"))


def _operator(ver: dict) -> bool:
    return ver.get("origin") == "edit" or ver.get("origin") == "authored"


def handle_fetch(job: dict, ctx) -> dict:
    t = _target(job, "fetch")
    if t is None:
        return _skipped(job, "fetch")
    sid, v, _ver = t
    s = skills.get(sid) or {}
    try:
        raw, meta = fetch_source(s.get("source_url") or "", ctx.beat)
    except Refused as e:
        skills.fail(sid, v, f"fetch refused: {e}")
        return {"refused": str(e)}
    # THE CLEANING STEP (mcp/skill_clean.py): badge images and HTML comments
    # out of a PINNED source, before it is screened or distilled. Only for a
    # skill whose meta names the pin (`fetch_clean`), and only for bytes that
    # hash to it; every other source reaches the screen as it arrived.
    cfg = (s.get("meta") or {}).get("fetch_clean")
    cleaned = None
    if cfg:
        import skill_clean
        raw, cleaned = skill_clean.apply(raw, meta, cfg)
        meta = dict(meta, clean=cleaned)
        if cleaned.get("applied"):
            meta["charset"] = "utf-8"
    rec = skills.store_source(sid, v, raw, meta)
    out = {"bytes": rec["bytes"], "sha256": rec["sha256"],
           "robots": meta["robots"]["why"]}
    if cleaned:
        out["clean"] = {k: cleaned.get(k) for k in (
            "applied", "removed", "hosts", "why", "raw_bytes",
            "cleaned_bytes") if cleaned.get(k) is not None}
    return out


MAX_SOURCE_CHARS = CHUNK_CHARS * MAX_CHUNKS


def handle_screen(job: dict, ctx) -> dict:
    t = _target(job, "screen")
    if t is None:
        return _skipped(job, "screen")
    sid, v, _ver = t
    raw, text, kind = source_texts(sid, v)
    if not text.strip():
        skills.fail(sid, v, "the source is empty after extraction; nothing "
                            "to screen or distil")
        return {"failed": "empty source"}
    if len(text) > MAX_SOURCE_CHARS:
        skills.fail(sid, v, f"the source is {len(text):,} characters of text, "
                            f"over {MAX_SOURCE_CHARS:,} ({MAX_CHUNKS} chunks "
                            "of YAMADORI_SKILL_CHUNK_CHARS): submit a narrower "
                            "page, or raise YAMADORI_EXTRACT_MAX_CHUNKS")
        return {"failed": "too long"}
    ctx.beat(f"screening {len(text):,} characters")
    res = skill_screen.screen(raw, text, kind)
    screen_rec: dict = {"deterministic": res}
    cl = ((_ver.get("fetched") or {}).get("clean") or {})
    if cl:
        # what the fetch removed before this screen ran (skill_clean): the
        # screen above saw the cleaned text, every rule at full strength
        screen_rec["cleaning"] = {k: cl.get(k) for k in (
            "v", "applied", "removed", "hosts", "comment_words", "why",
            "raw_sha256", "cleaned_sha256") if cl.get(k) is not None}
    skills.update_version(sid, v, screen=screen_rec)
    if not res["ok"]:
        skills.quarantine(sid, v, "screen: " + skill_screen.summary(res))
        return {"quarantined": len(res["quarantine"])}
    return {"ok": True, "notes": len(res["notes"])}


def handle_screen_model(job: dict, ctx) -> dict:
    t = _target(job, "screen_model")
    if t is None:
        return _skipped(job, "screen_model")
    sid, v, ver = t
    screen = dict(ver.get("screen") or {})
    if not MODEL_SCREEN:
        screen["model"] = {"skipped": "YAMADORI_SKILL_MODEL_SCREEN=0"}
        skills.update_version(sid, v, screen=screen)
        return {"skipped_model_screen": True}
    _raw, text, _kind = source_texts(sid, v)
    s = skills.get(sid) or {}
    parts = chunks_of(text)
    norm = skill_builder._norm(text)
    findings, verdicts = [], []
    for i, part in enumerate(parts):
        ctx.beat(f"model screen {i + 1}/{len(parts)}")
        reply = ask_model(skill_screen.SCREEN_SYSTEM,
                          skill_screen.screen_user(part, s.get("name") or ""),
                          max_tokens=SCREEN_MAX_TOKENS,
                          purpose="screen")
        try:
            got = skill_screen.read_verdict(parse_object(reply), norm)
        except (ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(describe(
                f"the model screen's reply for chunk {i + 1} was not a "
                f"verdict ({e}); it began {reply[:120]!r}", True,
                "retried automatically; if it repeats, switch the model "
                "screen off with YAMADORI_SKILL_MODEL_SCREEN=0 and say so",
                "operator")) from e
        verdicts.append(got["verdict"])
        findings += got["quarantine"]
    screen["model"] = {"verdicts": verdicts, "quarantine": findings,
                       "chunks": len(parts), "ok": not findings}
    skills.update_version(sid, v, screen=screen)
    if findings:
        skills.quarantine(sid, v, "model screen: "
                          + skill_screen.summary({"quarantine": findings}))
        return {"quarantined": len(findings)}
    return {"ok": True, "chunks": len(parts)}


def _goal(sid: str) -> str:
    return str(((skills.get(sid) or {}).get("meta") or {}).get("goal") or "")


def handle_distil(job: dict, ctx) -> dict:
    t = _target(job, "distil")
    if t is None:
        return _skipped(job, "distil")
    sid, v, _ver = t
    _raw, text, _kind = source_texts(sid, v)
    s = skills.get(sid) or {}
    parts = chunks_of(text)
    parsed, replies = [], []
    for i, part in enumerate(parts):
        ctx.beat(f"distil {i + 1}/{len(parts)}")
        reply = ask_model(
            skill_prompts.DISTIL_SYSTEM,
            skill_prompts.distil_user(
                part, name=s.get("name") or "", goal=_goal(sid),
                part=f"{i + 1} of {len(parts)}" if len(parts) > 1 else ""),
            max_tokens=DISTIL_MAX_TOKENS, purpose="distil")
        replies.append(reply[:20000])
        parsed.append(skill_prompts.parse_skill_reply(reply))
    merged = dict(skill_builder.merge(parsed))
    for k in ("name", "description", "section"):
        merged[k] = next((p[k] for p in parsed if p.get(k)), "")
    for k in ("phases", "topics"):
        merged[k] = list(dict.fromkeys(x for p in parsed for x in p.get(k)
                                       or []))
    skills.update_version(sid, v, distil={
        "parsed": merged, "replies": replies, "chunks": len(parts),
        "replies_chars": [len(r) for r in replies],
        "prompt": skill_prompts.DISTIL_VERSION})
    return {"chunks": len(parts), "items_proposed": len(merged["items"])}


def _names_package(text: str, name: str) -> bool:
    """The source names the package as a word of its own (`koota`,
    `@react-three/fiber`, `math`), not inside another word."""
    name = (name or "").strip()
    return bool(name) and bool(re.search(
        r"(?<![\w@/.-])" + re.escape(name) + r"(?![\w/-])", text or "", re.I))


def _choose_lead(chunks: list[list[dict]], text: str, declared: str
                 ) -> dict:
    """THE PACKAGE'S LEAD (decompose/3): the first block marked `lead:` in
    the FIRST part is the lead; every other `lead:` mark is cleared. The
    package it leads is the one the operator declared on the source
    (meta.package), else the model's `lead:` value; either must be named in
    the source. Returns the record kept on the version."""
    rec: dict = {"declared": declared or None, "proposed": None,
                 "package": None, "block": None}
    for i, blocks in enumerate(chunks):
        for b in blocks:
            mark = str(b.get("lead") or "").strip().strip("`\"'")
            b["lead"] = ""
            if not mark:
                continue
            if i > 0 or rec["proposed"] is not None:
                rec.setdefault("cleared", []).append(
                    {"name": b.get("name"), "part": i + 1})
                continue
            rec["proposed"] = mark
            pkg = (declared or mark).strip().lower()
            if _names_package(text, pkg):
                b["lead"] = pkg
                rec.update(package=pkg, block=b.get("name"))
            else:
                rec["why"] = (f"the source does not name {pkg!r}; the block "
                              "is kept as an ordinary skill")
    if rec["proposed"] is None:
        rec["why"] = "the model marked no lead in the first part"
    return rec


def _run_child(cid: str, v: int, errors: list[dict]) -> None:
    """Run a child's stages inline. A retryable error (the faithfulness
    check could not run, the model is down) leaves THAT child where it
    stopped -- running, visible, re-runnable from its stage -- and the
    other children go on: the source is decomposed either way."""
    try:
        run_inline(cid, v)
    except Exception as e:                                       # noqa: BLE001
        ver = skills.version(cid, v) or {}
        errors.append({"id": cid, "version": v, "stage": ver.get("stage"),
                       "error": f"{type(e).__name__}: {e}"[:300]})


def handle_decompose(job: dict, ctx, *, inline: bool = False) -> dict:
    """A frontier SKILL.md becomes several atomic child skills (each its own
    skill with its own pipeline: classify -> tests -> validate -> arm) and
    ONE LEAD skill for the package (decompose/3: what the package is and its
    core pattern; `metadata.yamadori.lead_for`); the source's version is
    marked `decomposed` and lists them.

    A LATER version of the source (a watch found it changed, or an operator
    re-run) SUPERSEDES the children of the earlier one: a block whose name
    matches an existing child (the lead matches the earlier lead) becomes a
    NEW VERSION of that child -- its served version keeps serving until the
    new one arms (skills.arm supersedes it), as a watched URL skill's does;
    a new name is a new child; an earlier child that no block matches is
    ARCHIVED (its section is gone from the source; nothing is deleted)."""
    import skill_md
    t = _target(job, "decompose")
    if t is None:
        return _skipped(job, "decompose")
    sid, v, _ver = t
    _raw, text, _kind = source_texts(sid, v)
    s = skills.get(sid) or {}
    parts = chunks_of(text)
    per_part, replies = [], []
    for i, part in enumerate(parts):
        ctx.beat(f"decompose part {i + 1}")
        reply = ask_model(skill_prompts.DECOMPOSE_SYSTEM,
                          skill_prompts.decompose_user(
                              part, name=s.get("name") or "",
                              goal=_goal(sid),
                              part=f"{i + 1} of {len(parts)}"
                              if len(parts) > 1 else ""),
                          max_tokens=DISTIL_MAX_TOKENS,
                          purpose="decompose")
        replies.append(reply[:30000])
        per_part.append(skill_prompts.parse_decompose(reply))
    lead = _choose_lead(per_part, text, str(((s.get("meta") or {})
                                             .get("package")) or ""))
    blocks = [b for bs in per_part for b in bs]
    # Chunks of one source can propose the same skill twice: first wins.
    seen: set[str] = set()
    uniq = []
    for b in blocks:
        key = (b.get("name") or b.get("title") or "").strip().lower()
        if b.get("items") and key not in seen:
            seen.add(key)
            uniq.append(b)
    # Every child the source proposes (no MAX_CHILDREN cap since 2026-09-27).
    blocks = uniq
    if lead.get("block") and not any(b.get("lead") for b in blocks):
        lead["why"] = "the lead block had no items"
        lead["package"] = None
    skills.update_version(sid, v, distil={
        "replies": replies, "prompt": skill_prompts.DECOMPOSE_VERSION,
        "lead": lead,
        "blocks": [{k: b.get(k) for k in ("name", "section", "title",
                                          "lead")}
                   | {"items": len(b["items"])} for b in blocks]})
    if not blocks:
        skills.fail(sid, v, "decompose: the model proposed no skill with "
                    "items; nothing to validate")
        return {"failed": "no blocks"}
    # The children of earlier versions of this source, still in service.
    earlier = [c for c in skills.children(sid)
               if int(((c.get("meta") or {}).get("parent") or {})
                      .get("version") or 0) < v
               and c.get("status") != "archived"]

    def _was_lead(c: dict) -> bool:
        cv = skills.version(c["id"], c.get("latest_version") or 1) or {}
        return bool(((cv.get("distil") or {}).get("parsed") or {})
                    .get("lead"))
    by_name = {skill_md.to_name(c["name"]): c for c in earlier}
    old_lead = next((c for c in earlier if _was_lead(c)), None)
    kids, versioned, used = [], [], set()
    child_errors: list[dict] = []
    for b in blocks:
        prev = by_name.get(skill_md.to_name(b.get("name") or b.get("title")
                                            or "skill"))
        if prev is None and b.get("lead") and old_lead is not None:
            prev = old_lead
        if prev is not None and prev["id"] not in used:
            used.add(prev["id"])
            nv = skills.new_child_version(prev["id"], sid, v, parsed=b,
                                          section=b.get("section") or "",
                                          enqueue_first=not inline)
            kids.append(prev["id"])
            versioned.append({"id": prev["id"], "version": nv})
            if inline:
                _run_child(prev["id"], nv, child_errors)
            continue
        c = skills.create_child(sid, v, parsed=b,
                                section=b.get("section") or "",
                                enqueue_first=not inline)
        kids.append(c["id"])
        if inline:
            _run_child(c["id"], 1, child_errors)
    retired, to_retire = [], []
    onboarding = (s.get("meta") or {}).get("onboarding")
    for c in earlier:
        if c["id"] not in used:
            why = (f"superseded: its section is not in v{v} of its source "
                   f"({s.get('name')})")
            if onboarding:
                # A PACKAGE ONBOARDING's new version (docs/PACKAGE-ONBOARDING
                # .md 4.3): the child keeps serving until the onboarding's
                # skills JOIN -- every replacement armed or stopped -- and its
                # retire stage archives it then.
                to_retire.append({"id": c["id"], "why": why, "version": v})
                continue
            skills.archive(c["id"], reason=why, author="pipeline:decompose")
            retired.append(c["id"])
    if to_retire:
        prev = [x for x in ((s.get("meta") or {}).get("to_retire") or [])
                if x.get("id") not in {t["id"] for t in to_retire}]
        skills.update_meta(sid, to_retire=prev + to_retire)
    skills.mark_decomposed(sid, v, kids, f"decomposed into {len(kids)} "
                           "atomic skill(s)" + (f" (lead: {lead['package']})"
                                                if lead.get("package")
                                                else ""))
    out = {"children": kids, "lead": lead.get("package")}
    if child_errors:
        out["child_errors"] = child_errors
    if versioned or retired or to_retire:
        out.update(new_versions=versioned, archived=retired)
    if to_retire:
        out["to_retire"] = [t["id"] for t in to_retire]
    return out


def _skill_text(parsed: dict) -> str:
    import skill_md
    return "\n".join([parsed.get("title") or ""]
                     + [skill_md.item_line(it) for it in
                        parsed.get("items") or []])


def _verified_topics(topics, *texts, source: str | None = None,
                     dropped: list | None = None) -> list[str]:
    """The proposed topics the text names. With `source` (distil/5, tag/6:
    topics are API names and symbols, 2026-09-27), a topic must also be
    CODE-SHAPED (skill_classify.code_shaped: a plain word is no evidence --
    the matcher needs two, and the koota run's plain topics quarantined
    their own skills) and appear in the SOURCE the skill was made from;
    each refusal is recorded in `dropped`."""
    blob = "\n".join(t or "" for t in texts)
    out = []
    for t in topics or []:
        t = str(t).strip().strip("`")
        if not t or t in out:
            continue
        rx = skill_classify._topic_rx(t)
        why = None
        if source is not None and not skill_classify.code_shaped(t):
            why = "not code-shaped (a plain word)"
        elif not rx.search(blob):
            why = "the skill does not name it"
        elif source is not None and not rx.search(source):
            why = "the source does not name it"
        if why:
            if dropped is not None:
                dropped.append({"topic": t, "why": why})
            continue
        out.append(t)
    return out[:6]


def _quotes_text(parsed: dict) -> str:
    return "\n".join(str(it.get("quote") or "")
                     for it in parsed.get("items") or [])


# ---------------------------------------------------------------------------
# REVIEW (operator, 2026-09-28): the second pass over a draft, before it is
# tagged. The model proposes a verdict per item (skill_prompts.REVIEW_SYSTEM:
# keep / rewrite / drop); the code below decides what stands. Its record --
# the items before and after, every verdict, every refusal -- is the
# version's `review` column. The deterministic floor runs after it in
# validate (skill_limits.doubt), whatever the review said.
# ---------------------------------------------------------------------------
MODEL_REVIEW = os.environ.get("YAMADORI_SKILL_MODEL_REVIEW", "1") != "0"
_TOKEN = re.compile(r"`([^`]+)`|([A-Za-z_$@][\w$.@/<>!-]*(?:\(\))?)")


def code_names(text: str) -> set[str]:
    """The code an item names: its backticked spans and its code-shaped
    words (skill_classify.code_shaped), without trailing punctuation."""
    import skill_classify
    out = set()
    for tick, word in _TOKEN.findall(text or ""):
        t = (tick or word or "").strip().rstrip(".,;:")
        if tick or skill_classify.code_shaped(t):
            out.add(t)
    return out


def review_items(items: list[dict], got: dict) -> tuple[list[dict], dict]:
    """(the items that stand, the record) after the review's verdicts.
    Every refusal keeps the item as it was, with the reason recorded."""
    import skill_md
    by_i: dict[int, dict] = {}
    for v in (got or {}).get("items") or []:
        try:
            by_i[int(v.get("i"))] = v
        except (TypeError, ValueError):
            continue
    kept: list[dict] = []
    verdicts: list[dict] = []
    refused: list[dict] = []
    for n, it in enumerate(items, 1):
        line = skill_md.item_line(it)
        v = by_i.get(n) or {}
        verdict = str(v.get("verdict") or "").strip().lower()
        rec = {"i": n, "item": line[:300], "verdict": verdict or "keep"}
        if verdict == "drop":
            reason = str(v.get("reason") or "").strip().lower()
            if reason not in skill_prompts.REVIEW_DROP_REASONS:
                refused.append(dict(rec, why="a drop needs a reason from "
                                    "skill_prompts.REVIEW_DROP_REASONS"))
                kept.append(it)
                continue
            if reason == "no_action" and code_names(line) \
                    and not skill_limits.doubt(line):
                refused.append(dict(rec, why="it names code and states no "
                                    "doubt: an action by construction"))
                kept.append(it)
                continue
            verdicts.append(dict(rec, reason=reason))
            continue
        if verdict == "rewrite":
            text = str(v.get("text") or "").strip()
            new = skill_md.parse_items("- " + text.lstrip("- ").strip())
            why = None
            if len(new) != 1:
                why = "the rewrite is not one DO / WHEN / DO NOT item"
            else:
                new_line = skill_md.item_line(new[0])
                was_names = code_names(line + " " + (it.get("quote") or ""))
                new_names = code_names(new_line)
                if skill_limits.doubt(new_line):
                    why = f"the rewrite is doubt ({skill_limits.doubt(new_line)})"
                elif len(new_line) > len(line):
                    why = "the rewrite is longer than the item"
                elif new_names - was_names:
                    why = ("the rewrite names code the item and its quote do "
                           "not: " + ", ".join(sorted(new_names - was_names))
                           [:120])
                elif code_names(line) and not (new_names & code_names(line)):
                    why = "the rewrite keeps none of the item's code names"
            if why:
                refused.append(dict(rec, why=why, text=text[:300]))
                kept.append(it)
                continue
            kept.append(dict(new[0], quote=it.get("quote") or "",
                             **{k: it[k] for k in ("provenance",)
                                if k in it}))
            verdicts.append(dict(rec, after=new_line[:300]))
            continue
        if verdict not in ("", "keep"):
            refused.append(dict(rec, why=f"unknown verdict {verdict!r}; kept"))
        elif n not in by_i:
            rec["verdict"] = "keep (not judged)"
        kept.append(it)
        verdicts.append(rec)
    before = [skill_md.item_line(it) for it in items]
    after = [skill_md.item_line(it) for it in kept]
    return kept, {"prompt": skill_prompts.REVIEW_VERSION,
                  "before": before, "after": after, "verdicts": verdicts,
                  "refused": refused,
                  "tokens_before": skill_limits.tokens("\n".join(before)),
                  "tokens_after": skill_limits.tokens("\n".join(after))}


def handle_review(job: dict, ctx) -> dict:
    """The review pass: the draft's items (a distilled or decomposed skill's
    `parsed`, a compiled one's `skill`, an edit's `parsed`) read against the
    definition; what stands replaces the draft's items, and the record goes
    in the version's `review` column. Inline (a migration or authored
    install: no model) and switched off, it records why it did not run and
    the draft goes on unchanged -- validate's floor still applies."""
    t = _target(job, "review")
    if t is None:
        return _skipped(job, "review")
    sid, v, ver = t
    dist = dict(ver.get("distil") or {})
    key = "skill" if (_compiled(ver) and dist.get("skill")) else "parsed"
    draft = dict(dist.get(key) or {})
    items = list(draft.get("items") or [])
    if not items:
        skills.update_version(sid, v, review={"skipped": "no items"})
        return {"skipped": "no items"}
    no_model = inline_without_model(ver)
    if no_model or not MODEL_REVIEW:
        why = no_model or "switched off (YAMADORI_SKILL_MODEL_REVIEW=0)"
        skills.update_version(sid, v, review={
            "prompt": skill_prompts.REVIEW_VERSION, "skipped": why})
        return {"skipped": why}
    # A model that cannot answer (ask_model's RuntimeError: down, the token
    # limit) RAISES, retryable, like the faithfulness check: the review is
    # not skipped because the engine was busy. A reply that is not the JSON
    # the template asks for is recorded and the draft goes on unchanged
    # (validate's floor applies).
    try:
        got = skill_prompts.parse_object(ask_model(
            skill_prompts.REVIEW_SYSTEM,
            skill_prompts.review_user(draft.get("title") or "", items,
                                      description=draft.get("description")
                                      or ""),
            max_tokens=SCREEN_MAX_TOKENS, purpose="review"))
    except ValueError as e:
        rec = {"prompt": skill_prompts.REVIEW_VERSION,
               "error": f"{type(e).__name__}: {e}"[:300],
               "kept": "the draft, unchanged; validate's floor applies"}
        skills.update_version(sid, v, review=rec)
        return {"error": rec["error"]}
    kept, rec = review_items(items, got)
    draft["items"] = kept
    dist[key] = draft
    # An edit's items are verified by the quotes they carried: a rewritten
    # line keeps its item's quote (skill_builder.validate, known_quotes).
    if dist.get("known_quotes") is not None:
        kq = dict(dist["known_quotes"])
        for it in kept:
            import skill_md
            kq.setdefault(skill_md.item_line(it), it.get("quote") or "")
        dist["known_quotes"] = kq
    skills.update_version(sid, v, distil=dist, review=rec)
    return {"before": len(rec["before"]), "after": len(rec["after"]),
            "rewritten": sum(1 for x in rec["verdicts"] if x.get("after")),
            "dropped": sum(1 for x in rec["verdicts"]
                           if x["verdict"] == "drop"),
            "refused": len(rec["refused"]),
            "tokens": [rec["tokens_before"], rec["tokens_after"]]}


def handle_classify(job: dict, ctx) -> dict:
    """Place the skill on the taxonomy. Deterministic evidence from the
    source; the distil's own proposals and the `tag` template's, each
    VERIFIED (a term the source names, a topic the skill contains, a phase
    or situation from the fixed lists). A compiled or edited version states
    its rule; it is re-checked, not re-derived."""
    import skill_classify as C
    t = _target(job, "classify")
    if t is None:
        return _skipped(job, "classify")
    sid, v, ver = t
    dist = ver.get("distil") or {}
    given = ver.get("classify") or {}
    if _compiled(ver) or ver.get("origin") == "edit":
        rule = C.rule_from_metadata(C.metadata_of_rule(given)) if given \
            else {}
        if given.get("triggers"):
            rule["triggers"] = given["triggers"]
        if not rule or C.empty(rule):
            skills.fail(sid, v, "classify: the rule names no artifact, "
                        "language, framework, domain or code-shaped topic "
                        "the selector can detect")
            return {"failed": "no applies-when"}
        skills.update_version(sid, v, classify=rule)
        return {"applies_when": rule["text"], "given": True}
    parsed = dist.get("parsed") or {}
    _raw, source, _kind = source_texts(sid, v)
    full_source = source
    skill_text = _skill_text(parsed)
    topic_text = skill_text + "\n" + _quotes_text(parsed)
    topics_dropped: list[dict] = []
    if ver.get("origin") == "decomposed":
        # One atomic part of a larger source: its evidence is its own
        # section's skill, not the whole document (a debugging skill whose
        # parent shows pytest examples is not a Python-tests skill).
        source = f"{parsed.get('description') or ''}\n{skill_text}"
    rule = C.classify(source, declared=((ver.get("meta") or {}).get(
        "declared") or {}))
    phases = [p for p in parsed.get("phases") or [] if p in C.PHASES]
    topics = _verified_topics(parsed.get("topics"), topic_text,
                              source=full_source, dropped=topics_dropped)
    if not topics:
        topics = _verified_topics(C.extract_topics([skill_text]), topic_text,
                                  source=full_source)
    # The template version AND the catalogue it rendered (package_registry
    # .catalogue_sha): a record says exactly which catalogue filed it, and a
    # term an onboarding adds needs no hand-bumped version.
    tag: dict = {"prompt": skill_prompts.tag_version()}
    if MODEL_TAG:
        try:
            got = skill_prompts.parse_object(ask_model(
                skill_prompts.tag_system(), skill_prompts.tag_user(
                    f"{parsed.get('description') or ''}\n{skill_text}"),
                max_tokens=SCREEN_MAX_TOKENS,
                purpose="tag"))
            tag["reply"] = {k: got.get(k) for k in (
                "artifacts", "languages", "frameworks", "phases",
                "situations", "topics", "description", "all_of")}
        except (ValueError, RuntimeError) as e:
            tag["error"] = f"{type(e).__name__}: {e}"[:200]
            got = {}
        # A term the model names joins the rule only when the SOURCE or
        # the skill names it too: the model proposes, the text decides.
        blob = source + "\n" + skill_text
        a = dict(C.applies_to(rule))
        for key, kind in (("languages", "language"),
                          ("frameworks", "framework")):
            for x in got.get(key) or []:
                term = C.BY_ID.get(str(x))
                if term and term.kind == kind and x not in a[key] and \
                        C._WORDS[term.id].search(blob) and len(a[key]) < 2:
                    a[key].append(term.id)
        for x in got.get("artifacts") or []:
            if x in C.ART_BY_ID and x not in a["artifacts"] and x != "code" \
                    and C._NOUNS[x].search(blob) and len(a["artifacts"]) < 2:
                a["artifacts"].append(x)
        if not (a["languages"] or a["frameworks"] or a["artifacts"]) and                 "code" in (got.get("artifacts") or []):
            # Language-agnostic advice about code (debugging, testing
            # habits): the model says so and nothing contradicts it.
            a["artifacts"] = ["code"]
        if (a["languages"] or a["frameworks"]) and not a["artifacts"]:
            a["artifacts"] = ["code"]
        # THE OR LIST MEANS "ANY ONE BRINGS IT INTO PLAY" (tag/5). The
        # deterministic filing adds every language the text names ("... in
        # TypeScript"), so a koota skill was filed "koota or TypeScript",
        # and a code-shaped topic it shares with another library (useQuery:
        # React Query's too) then selected it for any TypeScript request.
        # When the tag keys the skill on a framework and does not list a
        # language, that language is its host, not an alternative: it
        # leaves the OR list (recorded). Narrowing only -- the tag never
        # adds a language the text does not name (above).
        if a["frameworks"] and "languages" in got and got.get("frameworks"):
            said = {str(x) for x in got.get("languages") or []}
            host = [x for x in a["languages"] if x not in said]
            if host:
                a["languages"] = [x for x in a["languages"] if x in said]
                tag["host_languages"] = host
        # MULTI-DOMAIN (tag/5): the terms that must ALSO be in use, each
        # verified against the text (skill_classify.verified_all_of).
        all_of, a, tag["all_of"] = C.verified_all_of(
            got.get("all_of") or [], a, blob)
        rule["applies_to"] = a
        phases = phases or [p for p in got.get("phases") or []
                            if p in C.PHASES]
        topics = topics or _verified_topics(
            got.get("topics"), topic_text, source=full_source,
            dropped=topics_dropped)
        situ = [x for x in got.get("situations") or [] if x in C.SITUATIONS]
    else:
        situ, all_of = [], []
        tag["skipped"] = "YAMADORI_SKILL_MODEL_TAG=0"
    desc = parsed.get("description") or (tag.get("reply") or {}).get(
        "description") or ""
    rule = C.with_gates(rule, phases=phases, situations=situ, topics=topics,
                        all_of=all_of, description=desc)
    # A trigger DERIVED from the filing says what the filing says now (the
    # tag stage may have changed the OR list: "Koota or React is being used"
    # read as a trigger for a skill filed "koota, with React").
    base = C.condition_text({"applies_to": rule.get("applies_to"),
                             "domains": rule.get("domains")})
    for trig in rule.get("triggers") or []:
        if isinstance(trig, dict) and trig.get("origin") == "derived" \
                and base:
            trig["text"] = base
    if topics_dropped:
        tag["topics_dropped"] = topics_dropped
    rule["_tag"] = tag
    if C.empty(rule):
        skills.update_version(sid, v, classify=rule)
        skills.fail(sid, v, "classify: the source names no language, "
                    "framework or domain the selector can detect, and the "
                    "skill no code-shaped topic, so it could never be "
                    "selected. Edit it with an applies_when naming one of: "
                    + ", ".join(x.name for x in C.VOCAB))
        return {"failed": "no applies-when"}
    skills.update_version(sid, v, classify=rule)
    return {"applies_when": rule["text"]}


def gates_text(rule: dict) -> str:
    """The skill's own gates, as the tests template reads them (tests/3):
    every should-case must satisfy them."""
    import skill_classify as C
    a = C.applies_to(rule)
    g = C.gates(rule)
    lines = []
    libs = C.names(a["frameworks"] + a["languages"])
    if libs:
        lines.append("libraries: " + " or ".join(libs) + (
            " -- with " + " and ".join(C.names(g["all_of"]))
            if g["all_of"] else ""))
    if g["topics"]:
        lines.append("topics (name at least one exactly as written): "
                     + ", ".join(g["topics"]))
    if g["phases"]:
        lines.append("phases: " + ", ".join(g["phases"]))
    if g["situations"]:
        lines.append("situations: " + ", ".join(
            C._SITUATION_TEXT.get(x, x) for x in g["situations"]))
    return "\n".join(lines)


# A gate the skill's OWN should-cases contradict is dropped (2026-09-27,
# after the koota live run: 5 of 7 quarantines were a should-case the skill's
# own phase or situation gate refused). Keyed by the gate and the refusal
# skill_classify.match gives for it, exactly.
_OWN_GATES = (("phases", "phase is not {}"), ("situations", "no {}"))


def _reconcile_gates(rule: dict, tests: dict) -> tuple[dict, list[dict]]:
    """(rule, dropped): each of _OWN_GATES that refused one of the skill's
    own should-cases is removed from the rule, and recorded."""
    import skill_classify as C
    import skill_tests
    should = [c for c in ((tests.get("activation") or {}).get("should")
                          or []) if isinstance(c, dict)]
    dropped = []
    for key, prefix in _OWN_GATES:
        if not C.gates(rule)[key]:
            continue
        refusal = prefix.format("/".join(C.gates(rule)[key]))
        hit = [c for c in should
               if skill_tests.run_case(c, rule).get("gate") == refusal]
        if not hit:
            continue
        was = C.gates(rule)[key]
        rule = {k: v for k, v in rule.items() if k != key}
        rule["text"] = C.condition_text(rule)
        dropped.append({"gate": key, "was": was, "cases": len(hit),
                        "example": str(hit[0].get("text") or "")[:120]})
    return rule, dropped


def handle_tests(job: dict, ctx) -> dict:
    """The skill's tests: the deterministic floor from its rule, what it
    arrived with (compiled, authored, an edit), and -- for a distilled or
    decomposed skill -- the `tests` template's proposals."""
    import skill_tests
    t = _target(job, "tests")
    if t is None:
        return _skipped(job, "tests")
    sid, v, ver = t
    rule = ver.get("classify") or {}
    dist = ver.get("distil") or {}
    parsed = dist.get("parsed") or (dist.get("skill") or {})
    given = ver.get("tests") or {}
    # A LEAD is pushed when its package comes into play, never matched: its
    # tests are about the PACKAGE (skill_pipeline._lead_activation), so they
    # are written from the package alone -- no phase, topic or situation
    # near misses, which no detector reads.
    lead = bool(parsed.get("lead"))
    gen_rule = {k: v for k, v in rule.items() if k not in (
        "phases", "situations", "topics", "all_of")} if lead else rule
    base = skill_tests.generate(gen_rule, description=parsed.get(
        "description") or "", title=parsed.get("title") or "")
    tests = skill_tests.merge(given, base) if given else base
    model = {"prompt": skill_prompts.TESTS_VERSION}
    if not (_compiled(ver) or ver.get("origin") == "edit") and MODEL_TESTS:
        try:
            got = skill_prompts.parse_object(ask_model(
                skill_prompts.TESTS_SYSTEM,
                skill_prompts.tests_user(
                    f"description: {parsed.get('description') or ''}\n"
                    f"applies when: {rule.get('text') or ''}\n"
                    + _skill_text(parsed), gates=gates_text(gen_rule)),
                max_tokens=SCREEN_MAX_TOKENS,
                purpose="tests"))
            tests = skill_tests.merge(tests, skill_tests.from_model(got))
            model["proposed"] = {k: len(got.get(k) or []) for k in
                                 ("should", "should_not", "behaviour")}
        except (ValueError, RuntimeError) as e:
            model["error"] = f"{type(e).__name__}: {e}"[:200]
    beh, dropped = skill_tests.clean_behaviour(tests.get("behaviour"))
    tests["behaviour"] = beh
    tests["model"] = model
    if dropped:
        tests["dropped"] = dropped
    if not (_compiled(ver) or ver.get("origin") == "edit" or lead):
        rule2, gone = _reconcile_gates(rule, tests)
        if gone:
            tests["gates_dropped"] = gone
            skills.update_version(sid, v, classify=rule2)
    skills.update_version(sid, v, tests=tests)
    act = tests.get("activation") or {}
    return {"should": len(act.get("should") or []),
            "should_not": len(act.get("should_not") or []),
            "behaviour": len(beh)}


# THE FAITHFULNESS CHECK IS MANDATORY for a model-written skill (operator,
# 2026-09-27, from docs/research/SKILLS-RESEARCH.md Part 4.5): SkillsBench
# 2602.12670 -- self-generated skills cost -8.1 to -11.5 pp, their packs
# "confidently wrong"; ASI 2504.06821 -- verification added +4.2; Dynamic
# Cheatsheet 2504.07952 -- small models write flawed memory. A version whose
# items a model wrote never arms with the check switched off, errored, or
# silent on an item.
# CAVEAT, recorded on every verdict (FAITHFUL_JUDGE): the judge is the same
# ternary model that wrote the items, reading each item beside its verbatim
# quote -- weaker than ASI's execution-based check.
FAITHFUL_JUDGE = ("the model that wrote the items (skill_pipeline.ask_model), "
                  "reading each item beside its verbatim quote; weaker than "
                  "an execution-based check (ASI 2504.06821)")


def _faithful(items: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """(kept, dropped, record) after the `faithful` template. RAISES a
    retryable RuntimeError when the check could not run or its reply is not
    a verdict: a skill never arms unchecked. An item the reply gives no
    verdict for is dropped (silence is not a verdict)."""
    rec = {"prompt": skill_prompts.FAITHFUL_VERSION, "judge": FAITHFUL_JUDGE}
    try:
        got = skill_prompts.parse_object(ask_model(
            skill_prompts.FAITHFUL_SYSTEM, skill_prompts.faithful_user(items),
            max_tokens=SCREEN_MAX_TOKENS, purpose="faithful"))
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(describe(
            f"the faithfulness check could not run ({type(e).__name__}: "
            f"{str(e)[:160]})", True, "retried automatically; the skill does "
            "not arm unchecked", "worker")) from e
    verdict: dict[int, tuple[bool, str]] = {}
    for x in got.get("items") or []:
        if (isinstance(x, dict) and str(x.get("i") or "").isdigit()
                and isinstance(x.get("faithful"), bool)):
            verdict.setdefault(int(x["i"]), (x["faithful"],
                                             str(x.get("why") or "")[:160]))
    kept, dropped = [], []
    for n, it in enumerate(items, 1):
        v = verdict.get(n)
        if v is None:
            dropped.append({"item": skill_builder.item_line(it)[:300],
                            "why": "the faithfulness check gave no verdict "
                                   "for it"})
        elif v[0] is False:
            dropped.append({"item": skill_builder.item_line(it)[:300],
                            "why": f"the model judged it unfaithful to its "
                                   f"quote: {v[1]}"})
        else:
            kept.append(it)
    rec["unfaithful"] = sum(1 for n in range(1, len(items) + 1)
                            if verdict.get(n, (True,))[0] is False)
    rec["no_verdict"] = sum(1 for n in range(1, len(items) + 1)
                            if n not in verdict)
    rec["faithful"] = len(kept)
    return kept, dropped, rec


def _provenance(sid: str, ver: dict, s: dict) -> dict:
    fetched = ver.get("fetched") or {}
    lic = ver.get("licence") or {}
    meta = s.get("meta") or {}
    par = meta.get("parent") or {}
    prov = {"kind": ver.get("origin") if ver.get("origin") != "ingest"
            else s.get("source_kind"),
            "source": (meta.get("provenance") or {}).get("source")
            or s.get("source_url") or s.get("name"),
            "url": s.get("source_url") or fetched.get("url"),
            "fetched_at": fetched.get("fetched_at") or fetched.get(
                "received_at"),
            "sha256": ver.get("source_sha256") or fetched.get("sha256")}
    for k in ("repo", "path", "rows", "files", "log_rows", "dataset"):
        if (meta.get("provenance") or {}).get(k):
            prov[k] = meta["provenance"][k]
    if par:
        prov["source"] = (skills.get(par["skill"]) or {}).get("name") \
            or prov.get("source")
        prov["parent"] = par.get("skill")
        prov["section"] = par.get("section")
        pv = skills.version(par["skill"], par.get("version") or 1) or {}
        lic = lic or pv.get("licence") or {}
        prov["url"] = prov["url"] or (skills.get(par["skill"]) or {}).get(
            "source_url")
        prov["sha256"] = prov["sha256"] or pv.get("source_sha256")
    given = (meta.get("provenance") or {}).get("licence")
    if given and not lic:
        lic = given
    if not lic and ver.get("origin") == "edit":
        # An edit changes the skill's text or rule, never where it came
        # from: the licence is the edited version's (2026-09-26: an edited
        # fetched skill was written out as `license: unspecified`).
        ef = (ver.get("meta") or {}).get("edited_from")
        prev = skills.version(sid, ef) if ef else None
        while prev and not prev.get("licence") and prev.get("origin") == "edit":
            ef = (prev.get("meta") or {}).get("edited_from")
            prev = skills.version(sid, ef) if ef else None
        lic = (prev or {}).get("licence") or {}
    if lic:
        prov["licence"] = {k: lic.get(k) for k in ("spdx", "quote", "where")
                           if lic.get(k)}
        att = attribution_of(prov.get("url") or "", lic)
        if att:
            prov["attribution"] = att
    return {k: v for k, v in prov.items() if v not in (None, "", [], {})}


# ONE QUOTE-REPAIR ROUND (2026-09-27, after the koota live run: 14 items
# dropped for a quote under MIN_QUOTE_CHARS or not in the source). Before an
# item is dropped for its quote, the item goes back to the model ONCE with
# its exact failing quote and the source section (skill_prompts
# quote_repair/1); a new quote is kept only when it verifies exactly as any
# quote does (skill_builder: normalised whitespace, in the source, at least
# MIN_QUOTE_CHARS). A failed repair leaves the item to be dropped as before.
def _quote_fails(quote: str, src_norm: str) -> bool:
    qn = skill_builder._norm(quote)
    return len(qn) < skill_builder.MIN_QUOTE_CHARS or qn not in src_norm


def _section_of(source: str, section: str, hints: list[str]) -> str:
    """The part of the source a repair reads: all of it when it fits one
    chunk; else the sections whose headings the child's `section` names;
    else the chunk sharing most words with the items and their quotes."""
    if len(source) <= CHUNK_CHARS:
        return source
    names = [x.strip().lower() for x in re.split(r"[/|,;]", section or "")
             if x.strip()]
    if names:
        parts = re.split(r"(?m)^(?=#{1,6}\s)", source)
        hit = [pt for pt in parts
               if pt.startswith("#") and any(
                   n in pt.split("\n", 1)[0].lstrip("#").strip().lower()
                   for n in names)]
        text = "\n".join(hit)
        if text.strip():
            return text[:CHUNK_CHARS]
    words = set(re.findall(r"[A-Za-z_]\w{3,}", " ".join(hints).lower()))
    best = max(chunks_of(source), key=lambda c: len(
        words & set(re.findall(r"[A-Za-z_]\w{3,}", c.lower()))))
    return best[:CHUNK_CHARS]


def _repair_quotes(parsed: dict, source: str) -> tuple[dict, dict | None]:
    """(parsed with repaired quotes, record) -- one round, or no change."""
    items = list(parsed.get("items") or [])
    src_norm = skill_builder._norm(source)
    failing = [n for n, it in enumerate(items)
               if _quote_fails(it.get("quote") or "", src_norm)]
    if not failing:
        return parsed, None
    rec: dict = {"prompt": skill_prompts.QUOTE_REPAIR_VERSION,
                 "asked": len(failing), "repaired": 0, "items": []}
    ask = [{"item": skill_builder.item_line(items[n]),
            "quote": items[n].get("quote") or ""} for n in failing]
    section = _section_of(source, parsed.get("section") or "",
                          [a["item"] + " " + a["quote"] for a in ask])
    try:
        got = skill_prompts.parse_object(ask_model(
            skill_prompts.QUOTE_REPAIR_SYSTEM,
            skill_prompts.quote_repair_user(ask, section),
            max_tokens=SCREEN_MAX_TOKENS, purpose="quote_repair"))
    except (ValueError, RuntimeError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
        return parsed, rec
    by_i = {int(x["i"]): x.get("quote") for x in got.get("items") or []
            if isinstance(x, dict) and str(x.get("i") or "").isdigit()}
    for k, n in enumerate(failing, 1):
        new = by_i.get(k)
        ok = isinstance(new, str) and not _quote_fails(new, src_norm)
        rec["items"].append({"item": ask[k - 1]["item"][:160],
                             "old": ask[k - 1]["quote"][:160],
                             "new": new[:300] if isinstance(new, str)
                             else None, "verified": ok})
        if ok:
            items[n] = dict(items[n], quote=new.strip())
            rec["repaired"] += 1
    return dict(parsed, items=items), rec


def _lead_activation(tests: dict, package: str) -> dict:
    """A LEAD skill's activation tests (2026-09-27): it is PUSHED by
    mcp/skill_packages.py when the package comes into play and never
    matched, so its tests ask the detector that pushes it -- does
    skill_packages.detect put `package` in play for each should-case, and
    not for each near miss -- instead of the matcher's gates (which
    quarantined koota's lead on its own should-cases in the live run).
    run()'s shape, `by` naming the detector."""
    import skill_limits as L
    import skill_packages
    import skill_tests
    act = (tests or {}).get("activation") or {}
    should = [c for c in act.get("should") or [] if isinstance(c, dict)]
    should_not = [c for c in act.get("should_not") or []
                  if isinstance(c, dict)]
    want = (package or "").strip().lower()
    cases, failures, ok_n = [], [], 0
    for kind, group in (("should", should), ("should_not", should_not)):
        for c in group:
            found = skill_packages.detect(skill_tests.messages_of(c))
            hit = any(str(k).lower() == want for k in found)
            good = hit if kind == "should" else not hit
            ok_n += good
            cases.append({"kind": kind, "text": str(c.get("text") or "")
                          [:160], "ok": good, "in_play": sorted(found)[:6]})
            if not good:
                failures.append(f"{kind}: {str(c.get('text') or '')[:80]!r} "
                                f"-> {want} {'not ' if kind == 'should' else ''}"
                                "in play")
    n = len(should) + len(should_not)
    if len(should) < L.MIN_SHOULD or len(should_not) < L.MIN_SHOULD_NOT:
        failures.append(f"too few tests: {len(should)} should / "
                        f"{len(should_not)} should_not (at least "
                        f"{L.MIN_SHOULD} / {L.MIN_SHOULD_NOT})")
    return {"passed": not failures, "score": round(ok_n / n, 3) if n
            else 0.0, "n": n, "failures": failures[:8], "cases": cases,
            "by": "skill_packages.detect (a lead is pushed, never matched)"}


def _name_for(s: dict, parsed: dict, dist: dict, title: str) -> str:
    """The SKILL.md name a version is given: the name its creator asked for
    (skills.create's `name`, kept as meta.name_requested) wins over the
    model's and the store's, so a skill built for a case or a list by name is
    still found by it after the distil stage wrote its own."""
    asked = ((s or {}).get("meta") or {}).get("name_requested")
    return (asked or parsed.get("name") or dist.get("name")
            or (s or {}).get("name") or title)


def handle_validate(job: dict, ctx) -> dict:
    """Format, quotes, caps, the item screen, the ecosystem contract, and
    the activation tests. A failed test QUARANTINES; anything else that
    fails FAILS with the reason; a pass writes the SKILL.md."""
    import skill_classify as C
    import skill_md
    import skill_tests
    t = _target(job, "validate")
    if t is None:
        return _skipped(job, "validate")
    sid, v, ver = t
    s = skills.get(sid) or {}
    dist = ver.get("distil") or {}
    rule = ver.get("classify") or {}
    compiled = _compiled(ver)
    # A compiled skill's author vouches for it (the migration carries each
    # row's evidence as its quote where the row has one, and cites the row
    # itself otherwise); a quote that IS given must still verify.
    operator = bool(dist.get("operator")) or _operator(ver) or compiled
    if compiled:
        sk_in = dist.get("skill") or {}
        parsed = {"title": sk_in.get("title") or "",
                  "items": sk_in.get("items") or [],
                  "name": sk_in.get("name"),
                  "description": sk_in.get("description") or ""}
    else:
        parsed = dist.get("parsed") or {}
    if operator and ver.get("origin") == "edit":
        got = skills.source_raw(sid, v)
        source = ""
        if got:
            _r, source, _k = source_texts(sid, got[1]["version"])
    else:
        _raw, source, _kind = source_texts(sid, v)
    repair = None
    if not compiled and not operator and parsed.get("items"):
        parsed, repair = _repair_quotes(parsed, source)
    res = skill_builder.validate(parsed, source=source,
                                 applies_when=rule.get("text") or "",
                                 operator=operator,
                                 known_quotes=dist.get("known_quotes"))
    faithful = None
    if res["ok"] and not compiled and not operator and not MODEL_VALIDATE:
        # Mandatory (FAITHFUL_JUDGE's note): switched off, nothing arms.
        rec = {k: res[k] for k in ("ok", "title", "applies_when", "items",
                                   "dropped", "notes", "counts", "why",
                                   "quarantine")}
        rec["faithful"] = {"skipped": "YAMADORI_SKILL_MODEL_VALIDATE=0"}
        skills.fail(sid, v, "validate: the faithfulness check is required "
                    "before a model-written skill arms, and it is switched "
                    "off (YAMADORI_SKILL_MODEL_VALIDATE=0). Remedy "
                    "(operator): switch it on and re-run validate",
                    validate=rec)
        return {"failed": "faithfulness check switched off"}
    if res["ok"] and not compiled and not operator:
        kept, dropped, faithful = _faithful(res["items"])
        if dropped:
            res["dropped"] += dropped
            res["items"] = kept
            if not kept:
                res["ok"] = False
                res["why"] = ("no item survived the faithfulness check "
                              f"({len(dropped)} dropped)")
    rec = {k: res[k] for k in ("ok", "title", "applies_when", "items",
                               "dropped", "notes", "counts", "why",
                               "quarantine")}
    if faithful:
        rec["faithful"] = faithful
    if repair:
        rec["quote_repair"] = repair
    if res["quarantine"]:
        skills.quarantine(sid, v, "validate: " + res["why"], validate=rec)
        return {"quarantined": len(res["quarantine"])}
    if not res["ok"]:
        skills.fail(sid, v, "validate: " + res["why"], validate=rec)
        return {"failed": res["why"]}
    # The SKILL.md.
    name = skills.set_name(sid, _name_for(s, parsed, dist, res["title"]))
    desc = " ".join(str(parsed.get("description") or "").split())
    if not desc:
        trig = next((x.get("text") for x in rule.get("triggers") or []
                     if isinstance(x, dict) and x.get("text")), "")
        desc = f"Use when {rule.get('text') or trig or res['title']}."
    desc = desc[:skill_limits.DESCRIPTION_CHARS]
    prov = _provenance(sid, ver, s)
    lic = prov.get("licence") or {}
    tests = ver.get("tests") or {}
    # Against the armed pool too (reported, not gated) -- except inline,
    # where the pool is being built one skill at a time and the ranks
    # would describe a half-made store (skill_migrate reports them after).
    pool = None if _INLINE["on"] else [
        x for x in skills.armed() if x["id"] != sid] + [
        {"id": sid, "rule": rule}]
    if parsed.get("lead"):
        act = _lead_activation(tests, str(parsed["lead"]))
    else:
        act = skill_tests.run(tests, rule, skill_id=sid, pool=pool)
    rec["activation"] = act
    # BOUNDARIES, RECORDED -- NOT GATING (skill_boundaries; research Part
    # 4.4): whether the description states a "Not for" boundary, and which
    # of the siblings' own should-cases select this skill too.
    import skill_boundaries
    rec["boundary"] = skill_boundaries.boundary_of(desc)
    try:
        sib = skill_boundaries.sibling_cases(rule, sid)
        rec["siblings"] = {k: sib[k] for k in ("siblings", "cases", "fires")}
        rec["siblings"]["fired"] = [r for r in sib["rows"]
                                    if r["verdict"] != "none"]
    except Exception as e:                                       # noqa: BLE001
        rec["siblings"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    sk = {"name": name, "description": desc,
          "version": f"1.0.{int(v) - 1}", "author": skill_md.AUTHOR,
          "license": lic.get("spdx") or "unspecified",
          "tags": C.tags_of(rule), "related_skills": [],
          "title": res["title"], "when": desc,
          "items": [{"form": it["form"], "situation": it.get("situation")
                     or "", "text": it["text"],
                     "quote": it.get("quote") or "",
                     "ref": it.get("ref") or ""} for it in res["items"]],
          "yamadori": {"id": sid, "revision": int(v), "state": "draft",
                       "category": C.category(rule),
                       "applies_when": C.metadata_of_rule(rule),
                       "escalate": bool(rule.get("escalate")),
                       "provenance": prov,
                       "tests": {"activation": {
                           "passed": act["passed"], "score": act["score"],
                           "n": act["n"]},
                           "behaviour": len(tests.get("behaviour") or [])}}}
    if parsed.get("lead"):
        # decompose/3's lead: the package this skill introduces
        # (skill_pipeline._choose_lead verified the name in the source).
        sk["yamadori"]["lead_for"] = str(parsed["lead"])
    smeta = s.get("meta") or {}
    if smeta.get("onboarding") and smeta.get("package"):
        # Made by a package onboarding: the package and the version it is
        # about travel with the SKILL.md (the selector's asked-major filter
        # reads the version; docs/PACKAGE-ONBOARDING.md 4.3).
        sk["yamadori"]["package"] = str(smeta["package"])
        if smeta.get("package_version"):
            sk["yamadori"]["package_version"] = str(smeta["package_version"])
    problems = skill_md.format_problems(sk)
    text = skill_md.render(sk)
    rec["skill_md_problems"] = problems
    if problems:
        skills.fail(sid, v, "validate: " + "; ".join(problems),
                    validate=rec)
        return {"failed": problems}
    if not act["passed"]:
        skills.update_version(sid, v, text=text,
                              text_sha256=skill_md.sha256(text))
        skills.quarantine(sid, v, "activation tests: "
                          + "; ".join(act["failures"][:4]), validate=rec)
        return {"quarantined": "activation tests", "score": act["score"]}
    skills.update_version(sid, v, validate=rec, text=text,
                          text_sha256=skill_md.sha256(text))
    return {"kept": len(res["items"]), "dropped": len(res["dropped"]),
            "activation": act["score"]}


def handle_arm(job: dict, ctx, *, inline: bool = False) -> dict:
    t = _target(job, "arm")
    if t is None:
        return _skipped(job, "arm")
    sid, v, _ver = t
    skills.arm(sid, v)
    if inline:
        return {"armed": v, "trigger_index": "not built inline (the "
                "request path or the next arm job builds it)"}
    # Pre-build the trigger vectors so the first request after arming does
    # not pay for them. Best effort: the skill is armed either way, and the
    # request path builds them itself if this could not.
    try:
        import skill_select
        index = skill_select.refresh_triggers()
    except Exception as e:                                       # noqa: BLE001
        index = {"not_built": f"{type(e).__name__}: {e}"[:200]}
    return {"armed": v, "trigger_index": index}


def handle_watch(job: dict, ctx) -> dict:
    """Re-fetch a watched source. Unchanged: schedule the next look.
    Changed: a new version, from `screen`, while the served one keeps
    serving."""
    sid = (job.get("payload") or {}).get("skill")
    s = skills.get(sid) if sid else None
    if s is None or not s.get("source_url"):
        return {"skipped": "no such skill, or it has no source URL"}
    now = time.time()
    last = next((x for x in skills.versions(sid) if x.get("source_sha256")),
                None)
    running = [x for x in skills.versions(sid) if x["state"] == "running"]
    if running:
        _reschedule(sid, s, now, {"at": now, "result": "a version is still "
                                  "in the pipeline; looked again later"})
        return {"skipped": f"v{running[0]['version']} is still running"}
    try:
        raw, meta = fetch_source(s["source_url"], ctx.beat)
    except Refused as e:
        _reschedule(sid, s, now, {"at": now, "result": f"refused: {e}"})
        return {"refused": str(e)}
    sha = hashlib.sha256(raw).hexdigest()
    if last and last.get("source_sha256") == sha:
        _reschedule(sid, s, now, {"at": now, "result": "unchanged",
                                  "sha256": sha})
        return {"unchanged": True, "sha256": sha}
    # A frontier SKILL.md goes back through decompose, which supersedes the
    # earlier version's children; anything else is distilled again.
    origin = ("watch_frontier" if s.get("source_kind") == "frontier"
              else "watch")
    v = skills.new_version(sid, origin, author="pipeline",
                           meta={"previous_sha256": (last or {})
                                 .get("source_sha256")})
    skills.store_source(sid, v, raw, meta)
    skills.enqueue(sid, v, skills.PATHS[origin][0])
    _reschedule(sid, s, now, {"at": now, "result": f"changed: v{v}",
                              "sha256": sha})
    return {"changed": True, "version": v, "sha256": sha}


def _reschedule(sid: str, s: dict, now: float, last: dict) -> None:
    con = skills._db()
    try:
        m = dict(s.get("meta") or {})
        m["watch_last"] = last
        secs = s.get("watch_seconds")
        con.execute("UPDATE skills SET next_watch=?, meta=? WHERE id=?",
                    ((now + secs) if secs else None, json.dumps(m), sid))
    finally:
        con.close()


def schedule_watches(now: float | None = None) -> list[str]:
    """Enqueue a watch job for every due skill that has none live. Called by
    the worker once a minute; jobs.py deliberately has no cron."""
    now = now or time.time()
    con = skills._db()
    try:
        due = [r["id"] for r in con.execute(
            "SELECT id FROM skills WHERE watch_seconds IS NOT NULL AND "
            "source_url IS NOT NULL AND next_watch IS NOT NULL AND "
            "next_watch <= ?", (now,))]
    finally:
        con.close()
    out = []
    for sid in due:
        live = [j for j in jobs.listing(dataset=f"skill:{sid}", limit=20)
                if j["queue"] == skills.WATCH[0]
                and j["state"] in ("queued", "running")]
        if live:
            continue
        out.append(jobs.add(skills.WATCH[0], {"skill": sid},
                            lane=skills.WATCH[1], dataset=f"skill:{sid}",
                            stage="watch"))
    return out


def advance_after(job: dict) -> str | None:
    """The worker's hook once a skill job is done: enqueue the next stage."""
    p = job.get("payload") or {}
    stage = p.get("stage")
    if job.get("queue") in (skills.WATCH[0], skill_learn.LEARN[0]) \
            or not stage:
        return None
    nxt = skills.advance(p.get("skill"), p.get("version"), stage)
    if nxt is None:
        # The version stopped (armed, failed, quarantined): a replacement
        # batch it belongs to may now be settled (mcp/skill_rebuild.py:
        # the skills it replaces archived in the same step). Best effort:
        # the job is done either way, and --settle runs it by hand.
        try:
            import skill_rebuild
            skill_rebuild.settle_after(job)
        except Exception as e:                                   # noqa: BLE001
            print(f"[skill_rebuild] settle failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
    return nxt


HANDLERS = {
    skills.JOBS["fetch"][0]: handle_fetch,
    skills.JOBS["screen"][0]: handle_screen,
    skills.JOBS["screen_model"][0]: handle_screen_model,
    skills.JOBS["licence"][0]: handle_licence,
    skills.JOBS["distil"][0]: handle_distil,
    skills.JOBS["decompose"][0]: handle_decompose,
    skills.JOBS["review"][0]: handle_review,
    skills.JOBS["classify"][0]: handle_classify,
    skills.JOBS["tests"][0]: handle_tests,
    skills.JOBS["validate"][0]: handle_validate,
    skills.JOBS["prove"][0]: skill_prove.handle_prove,
    skills.JOBS["arm"][0]: handle_arm,
    skills.WATCH[0]: handle_watch,
    skill_learn.LEARN[0]: skill_learn.handle_learn,
}


# ---------------------------------------------------------------------------
# Inline runs: the same handlers, in order, without the queue -- for the
# migration and an authored install, whose stages need no model and must
# not wait for a worker. The trigger vectors are not built inline (that
# needs the embedder; the next arm job or the request path builds them).
# ---------------------------------------------------------------------------
# "model": whether an inline run has a model for the review and prove
# stages. run_inline(model=None) decides per version: a COMPILED version (a
# migration, an authored install: its items arrive written, and those
# installs run with no model) has none; anything else -- a source distilled
# or decomposed inline by a driver that talks to the model -- has one.
# model=False: the caller has no model for them (skill_offline's written
# replies); both stages record "not run" and validate's floor applies.
_INLINE = {"on": False, "model": None}


def inline_without_model(ver: dict) -> str | None:
    """Why review and prove do not run for this version inline, or None."""
    if not _INLINE["on"]:
        return None
    if _INLINE.get("model") is False:
        return "inline run with no model for this stage (the caller's)"
    if _INLINE.get("model") is None and _compiled(ver):
        return "inline install of a compiled skill: no model"
    return None


class _InlineCtx:
    def beat(self, progress: str | None = None) -> None:
        pass


def run_inline(sid: str, v: int, *, max_steps: int = 20,
               model: bool | None = None) -> dict:
    """Run version v's remaining stages now. Returns {stage: result}.
    `model`: whether the review and prove stages have a model here
    (_INLINE: None decides per version)."""
    out: dict = {}
    was = dict(_INLINE)
    _INLINE["on"] = True
    if model is not None or not was["on"]:
        _INLINE["model"] = model
    try:
        return _run_inline(sid, v, out, max_steps)
    finally:
        _INLINE.update(was)


def _run_inline(sid: str, v: int, out: dict, max_steps: int) -> dict:
    for _ in range(max_steps):
        ver = skills.version(sid, v)
        if not ver or ver["state"] != "running":
            break
        stage = ver["stage"]
        job = {"id": f"inline-{sid}-{v}-{stage}", "queue": skills.JOBS[
            stage][0], "payload": {"skill": sid, "version": v,
                                   "stage": stage}}
        h = HANDLERS[skills.JOBS[stage][0]]
        if stage in ("arm", "decompose"):
            out[stage] = h(job, _InlineCtx(), inline=True)
        else:
            out[stage] = h(job, _InlineCtx())
        if skills.advance(sid, v, stage, enqueue_next=False) is None:
            break
    return out
