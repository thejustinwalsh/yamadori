#!/usr/bin/env python
"""The skill pipeline's job handlers. `mcp/worker.py` claims and runs them.

    fetch -> screen -> screen_model -> classify -> distil -> validate -> arm
    watch (scheduled)

See `skills.py` for the stages, states and the no-review decision. Each
handler here reads its version, does one stage, and writes what it found onto
the version; `advance_after` (called by the worker once the job is `done`)
enqueues the next stage. A stage never enqueues its own successor, for the
reason datasets.py gives: a stage is left only when its job is done.

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

THE ONE DOOR

Every model call is `ask_model`, which goes through `mcp/model.py`
(`tiers.apply`, the same shaping as a client request at the same effort).
Tests replace `ask_model`.

FETCHING

http(s) only; robots.txt is read and obeyed for our user agent (401/403 on
robots.txt disallows everything, any other 4xx allows everything, a 5xx or
no answer is retryable -- the conventions of RFC 9309); a size cap; scripts
and binaries are refused by extension, content type and a NUL byte, and are
never stored. A GitHub page or folder is resolved to its raw file
(`skills.normalise_url`).
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
MAX_CHUNKS = int(os.environ.get("YAMADORI_SKILL_MAX_CHUNKS", "8"))
# The ANSWER allowance for each call; thinking is added by tiers.budget.
DISTIL_MAX_TOKENS = int(os.environ.get("YAMADORI_SKILL_DISTIL_MAX_TOKENS",
                                       "2048"))
SCREEN_MAX_TOKENS = int(os.environ.get("YAMADORI_SKILL_SCREEN_MAX_TOKENS",
                                       "1024"))
MODEL_TIMEOUT = int(os.environ.get("YAMADORI_SKILL_MODEL_TIMEOUT", "3600"))
# The model-assisted screen. On unless switched off; switching it off is
# recorded on every version it skips.
MODEL_SCREEN = os.environ.get("YAMADORI_SKILL_MODEL_SCREEN", "1") != "0"

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
    the text the operator submitted."""
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
    if meta.get("kind") == "migration":
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


def ask_model(system: str, user: str, *, max_tokens: int,
              temperature: float = 0.2, purpose: str = "") -> str:
    """One generation through the one door (mcp/model.py). Replaced in
    tests. A length finish is a budget event and is retried, never read as
    an answer."""
    import model
    try:
        content = model.ask(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            effort="medium", max_tokens=max_tokens, temperature=temperature,
            timeout=MODEL_TIMEOUT)
    except model.BudgetEvent as e:
        raise RuntimeError(describe(
            f"the {purpose or 'skill'} call hit the token limit: {e}", True,
            "lower YAMADORI_SKILL_CHUNK_CHARS or raise the stage's "
            "MAX_TOKENS", "operator")) from e
    if not (content or "").strip():
        raise RuntimeError(describe(
            f"the model wrote nothing for the {purpose or 'skill'} call", True,
            "retried automatically; if it repeats, check the model on "
            "llama-swap", "operator"))
    return content


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
    rec = skills.store_source(sid, v, raw, meta)
    return {"bytes": rec["bytes"], "sha256": rec["sha256"],
            "robots": meta["robots"]["why"]}


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
                            "page, or raise YAMADORI_SKILL_MAX_CHUNKS")
        return {"failed": "too long"}
    ctx.beat(f"screening {len(text):,} characters")
    res = skill_screen.screen(raw, text, kind)
    skills.update_version(sid, v, screen={"deterministic": res})
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
                          max_tokens=SCREEN_MAX_TOKENS, temperature=0.0,
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


def _declared(ver: dict) -> dict:
    return ((ver.get("meta") or {}).get("declared") or {})


def handle_classify(job: dict, ctx) -> dict:
    t = _target(job, "classify")
    if t is None:
        return _skipped(job, "classify")
    sid, v, ver = t
    _raw, text, _kind = source_texts(sid, v)
    rule = skill_classify.classify(text, declared=_declared(ver))
    skills.update_version(sid, v, classify=rule)
    if skill_classify.empty(rule):
        skills.fail(sid, v, "classify: the source names no language, "
                    "framework or domain the selector can detect, so the "
                    "skill could never be selected. Edit it with an "
                    "applies-when line naming one of: "
                    + ", ".join(t.name for t in skill_classify.VOCAB))
        return {"failed": "no applies-when"}
    return {"applies_when": rule["text"]}


def handle_distil(job: dict, ctx) -> dict:
    t = _target(job, "distil")
    if t is None:
        return _skipped(job, "distil")
    sid, v, ver = t
    _raw, text, _kind = source_texts(sid, v)
    s = skills.get(sid) or {}
    rule = ver.get("classify") or {}
    parts = chunks_of(text)
    parsed, replies = [], []
    for i, part in enumerate(parts):
        ctx.beat(f"distil {i + 1}/{len(parts)}")
        reply = ask_model(
            skill_builder.BUILDER_SYSTEM,
            skill_builder.builder_user(
                part, applies_when=rule.get("text") or "",
                name=s.get("name") or "",
                part=f"{i + 1} of {len(parts)}" if len(parts) > 1 else ""),
            max_tokens=DISTIL_MAX_TOKENS, temperature=0.2, purpose="distil")
        replies.append(reply[:20000])
        parsed.append(skill_builder.parse(reply))
    merged = skill_builder.merge(parsed)
    skills.update_version(sid, v, distil={
        "parsed": merged, "replies": replies, "chunks": len(parts),
        "replies_chars": [len(r) for r in replies]})
    return {"chunks": len(parts), "items_proposed": len(merged["items"])}


def handle_validate(job: dict, ctx) -> dict:
    t = _target(job, "validate")
    if t is None:
        return _skipped(job, "validate")
    sid, v, ver = t
    dist = ver.get("distil") or {}
    operator = bool(dist.get("operator"))
    rule = ver.get("classify") or {}
    if operator:
        got = skills.source_raw(sid, v)
        source = ""
        if got:
            _r, source, _k = source_texts(sid, got[1]["version"])
        cond = rule.get("text") or ""
    else:
        _raw, source, _kind = source_texts(sid, v)
        cond = rule.get("text") or ""
    if operator and skill_classify.empty(rule):
        skills.fail(sid, v, "validate: the applies-when line names nothing "
                    "the selector can detect. Name one of: "
                    + ", ".join(t.name for t in skill_classify.VOCAB))
        return {"failed": "no applies-when"}
    res = skill_builder.validate(dist.get("parsed") or {}, source=source,
                                 applies_when=cond, operator=operator,
                                 known_quotes=dist.get("known_quotes"))
    rec = {k: res[k] for k in ("ok", "title", "applies_when", "items",
                               "dropped", "notes", "counts", "why",
                               "quarantine")}
    if res["quarantine"]:
        skills.quarantine(sid, v, "validate: " + res["why"], validate=rec)
        return {"quarantined": len(res["quarantine"])}
    if not res["ok"]:
        skills.fail(sid, v, "validate: " + res["why"], validate=rec)
        return {"failed": res["why"]}
    skills.update_version(sid, v, validate=rec, text=res["text"],
                          text_sha256=hashlib.sha256(
                              res["text"].encode("utf-8")).hexdigest())
    return {"kept": len(res["items"]), "dropped": len(res["dropped"])}


def handle_arm(job: dict, ctx) -> dict:
    t = _target(job, "arm")
    if t is None:
        return _skipped(job, "arm")
    sid, v, _ver = t
    skills.arm(sid, v)
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
    v = skills.new_version(sid, "watch", author="pipeline",
                           meta={"previous_sha256": (last or {})
                                 .get("source_sha256")})
    skills.store_source(sid, v, raw, meta)
    skills.enqueue(sid, v, skills.PATHS["watch"][0])
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
    return skills.advance(p.get("skill"), p.get("version"), stage)


HANDLERS = {
    skills.JOBS["fetch"][0]: handle_fetch,
    skills.JOBS["screen"][0]: handle_screen,
    skills.JOBS["screen_model"][0]: handle_screen_model,
    skills.JOBS["classify"][0]: handle_classify,
    skills.JOBS["distil"][0]: handle_distil,
    skills.JOBS["validate"][0]: handle_validate,
    skills.JOBS["arm"][0]: handle_arm,
    skills.WATCH[0]: handle_watch,
    skill_learn.LEARN[0]: skill_learn.handle_learn,
}
