#!/usr/bin/env python
"""The process that claims jobs from `jobs.py` and runs them.

    python mcp/worker.py            run until interrupted
    python mcp/worker.py --once     drain what is queued now, then exit

WHAT IT IS

One process, one thread per lane slot: `jobs.LANES` is {gpu: 1, cpu: 4,
net: 4}, so nine threads, and the gpu lane can never run two jobs at once
because it has exactly one thread. Two jobs on one card is not slow, it is
wrong -- see `jobs.py`.

Each running job gets a heartbeat thread that calls `jobs.beat()` every
`BEAT_SECONDS` for as long as the handler is alive. A 450-second generation is
one blocking HTTP call and cannot beat from inside itself; without this it
would look exactly like a dead worker and `reclaim()` would hand it to someone
else while it was still running. Handlers ALSO beat with a progress string at
their own checkpoints ("chunk 3/12"), which is what the dashboard prints.

`jobs.reclaim()` runs at startup and then once a minute, so a job orphaned by
a killed worker goes back to the queue with its attempt count intact.

HOW A JOB ENDS

    handler returns          -> jobs.finish()      -> done
    handler raises Permanent -> jobs.fail(retry=False) -> errored
    handler raises anything  -> jobs.fail(retry=True)  -> queued again, or
                                errored once attempts are exhausted

A handler that raises NEVER lands in `done`. That distinction is why jobs.py
exists: a resume path that counted errored rows as done once poisoned a
benchmark file permanently.

Every error string is structured the same way -- the situation, whether it is
retryable as a fact, and a remedy with an owner -- because an operator reading
"failed" on the dashboard with no next step is the loop this repo keeps paying
for.

THE PIPELINE RUNS ITSELF

When a dataset's stage job finishes, the worker calls `datasets.advance()`,
which enqueues the next stage. So once `clarify` is answered, fetch ->
extract -> index -> complete happens unattended. `clarify` answers itself
where the source allows: a finished fetch enqueues `dataset.assist`, which
fills the licence from a verified verbatim quote and proposes the rest, then
advances -- see handle_assist. Review is optional and after
the fact, on /dash; a row marked `reject` there leaves the hints corpus on the
next index.
"""
from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets  # noqa: E402
import jobs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("YAMADORI_DATASETS_DIR",
                          os.path.join(HERE, "..", "index", "datasets"))
UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
EXTRACT_MODEL = os.environ.get("YAMADORI_EXTRACT_MODEL", "bonsai")

# Beat well inside jobs.STALE_SECONDS (1200) so a busy job is never mistaken
# for a dead one, but not so often that nine threads hammer one sqlite file.
BEAT_SECONDS = float(os.environ.get("YAMADORI_BEAT_SECONDS", "60"))
POLL_SECONDS = float(os.environ.get("YAMADORI_POLL_SECONDS", "2"))
RECLAIM_EVERY = 60.0

FETCH_TIMEOUT = 60
MAX_FETCH_BYTES = int(os.environ.get("YAMADORI_FETCH_MAX_BYTES",
                                     str(25 * 1024 * 1024)))
# One chunk is one model call. ~12k characters is ~3k tokens of source, which
# leaves the thinking model room to reason and still answer (the empty-reply
# bug in fdc9067 was a budget that ran out during reasoning).
CHUNK_CHARS = int(os.environ.get("YAMADORI_EXTRACT_CHUNK_CHARS", "12000"))
MAX_CHUNKS = int(os.environ.get("YAMADORI_EXTRACT_MAX_CHUNKS", "60"))
EXTRACT_MAX_TOKENS = int(os.environ.get("YAMADORI_EXTRACT_MAX_TOKENS", "4096"))
MODEL_TIMEOUT = int(os.environ.get("YAMADORI_EXTRACT_TIMEOUT", "3600"))
# A quote shorter than this proves little: "use a" is in every document.
MIN_EVIDENCE_CHARS = 24


class Permanent(Exception):
    """A failure that retrying cannot fix. Lands in `errored` on first raise.

    Carries the three things an operator needs, so the dashboard can print a
    next step rather than a stack trace.
    """

    def __init__(self, situation: str, remedy: str, owner: str = "operator"):
        self.situation, self.remedy, self.owner = situation, remedy, owner
        super().__init__(situation)

    def text(self) -> str:
        return describe(self.situation, False, self.remedy, self.owner)


def describe(situation: str, retryable: bool, remedy: str, owner: str) -> str:
    return (f"{situation} | retryable: {'yes' if retryable else 'no'} | "
            f"remedy ({owner}): {remedy}")


class Context:
    """What a handler gets besides its job row."""

    def __init__(self, job: dict):
        self.job = job

    def beat(self, progress: str | None = None) -> None:
        jobs.beat(self.job["id"], progress)


# ---------------------------------------------------------------------------
# Small helpers shared by handlers.
# ---------------------------------------------------------------------------
def dataset_dir(dataset_id: str) -> str:
    return os.path.join(os.path.abspath(DATA_DIR), dataset_id)


def _dataset(job: dict) -> dict:
    did = job.get("dataset") or (job.get("payload") or {}).get("dataset")
    ds = datasets.get(did) if did else None
    if ds is None:
        raise Permanent(f"job {job['id']} names dataset {did!r}, which does "
                        "not exist", "cancel the job; its dataset was "
                        "deleted or never created")
    return ds


def _merge_counts(dataset_id: str, new: dict) -> None:
    ds = datasets.get(dataset_id) or {}
    counts = dict(ds.get("counts") or {})
    counts.update(new)
    datasets.record_counts(dataset_id, counts)


def _write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# dataset.fetch  (net)
# ---------------------------------------------------------------------------
def handle_fetch(job: dict, ctx: Context) -> dict:
    ds = _dataset(job)
    url = (job["payload"].get("url") or ds.get("source_url") or "").strip()
    if not re.match(r"^https?://", url):
        raise Permanent(f"source {url!r} is not an http(s) URL",
                        "paste the text as the dataset source instead")
    req = urllib.request.Request(url, headers={
        "User-Agent": "yamadori-dataset-worker/1"})
    try:
        resp = urllib.request.urlopen(req, timeout=FETCH_TIMEOUT)
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500 and e.code not in (408, 425, 429):
            raise Permanent(f"GET {url} answered HTTP {e.code}",
                            "check the URL, or paste the text instead") from e
        raise
    h = hashlib.sha256()
    buf = bytearray()
    with resp:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            buf += block
            h.update(block)
            if len(buf) > MAX_FETCH_BYTES:
                raise Permanent(
                    f"{url} is larger than {MAX_FETCH_BYTES:,} bytes",
                    "raise YAMADORI_FETCH_MAX_BYTES or submit a narrower "
                    "source")
            ctx.beat(f"fetched {len(buf):,} bytes")
        meta = {"url": url, "final_url": resp.geturl(),
                "status": resp.status,
                "content_type": resp.headers.get_content_type(),
                "charset": resp.headers.get_content_charset(),
                "bytes": len(buf), "sha256": h.hexdigest(),
                "fetched_at": time.time()}
    d = dataset_dir(ds["id"])
    _write_atomic(os.path.join(d, "source.bin"), bytes(buf))
    _write_atomic(os.path.join(d, "source.json"),
                  json.dumps(meta, indent=2).encode("utf-8"))
    _merge_counts(ds["id"], {"fetched_bytes": len(buf)})
    return meta


# ---------------------------------------------------------------------------
# dataset.extract  (gpu)
# ---------------------------------------------------------------------------
class _Text(html.parser.HTMLParser):
    """HTML to text, keeping block boundaries as newlines."""

    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "li", "pre", "br", "tr", "h1", "h2", "h3", "h4",
             "h5", "h6", "section", "article", "blockquote", "table", "dt",
             "dd"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(s: str) -> str:
    p = _Text()
    p.feed(s)
    p.close()
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def source_text(ds: dict) -> str:
    """The text extract reads: the fetched file, else the pasted source."""
    d = dataset_dir(ds["id"])
    raw_path = os.path.join(d, "source.bin")
    if os.path.exists(raw_path):
        meta = {}
        try:
            with open(os.path.join(d, "source.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass
        ctype = meta.get("content_type") or ""
        with open(raw_path, "rb") as f:
            raw = f.read()
        if ctype == "application/pdf" or raw[:5] == b"%PDF-":
            raise Permanent(
                "the fetched source is a PDF and no PDF text extractor is "
                "installed in the worker's Python",
                "paste the text as the dataset source, or install pypdf and "
                "add PDF support to worker.source_text")
        text = raw.decode(meta.get("charset") or "utf-8", errors="replace")
        if "html" in ctype or re.search(r"(?i)<html|<body", text[:2000]):
            text = html_to_text(text)
        return text
    if ds.get("source_url") and re.match(r"^https?://", ds["source_url"]) \
            and not (ds.get("source") or "").strip():
        raise Permanent(
            f"no fetched copy of {ds['source_url']} exists under "
            f"{d}", "let the dataset.fetch job finish first; if it is "
            "errored, re-run it", owner="worker")
    return (ds.get("source") or ds.get("prompt") or "").strip()


def chunks_of(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries, never mid-paragraph unless one paragraph
    is larger than a chunk on its own."""
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
        if len(cur) + len(para) + 2 > size and cur:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


EXTRACT_SYSTEM = """You turn a technical source into RECIPES: short, \
actionable pieces of advice a programmer applies while writing code.

| field | content |
|---|---|
| recipe | the advice, 1-3 sentences, self-contained, imperative |
| trigger_condition | the situation in which it applies, one sentence |
| category | a short kebab-case topic label |
| confidence | "high" if the source states it outright, "medium" if it is implied |
| evidence | a VERBATIM quote from the source, 30-300 characters, that supports it |

Reply with a JSON array of objects with exactly those fields, and nothing \
else. An empty array is the right answer for a passage with no actionable \
advice. Evidence is checked character for character against the source; \
a row whose quote is not in the source is discarded."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def ask_model(system: str, user: str) -> str:
    """One extraction call, through the one door (mcp/model.py).

    EXTRACT_MAX_TOKENS is the ANSWER allowance -- a JSON array of recipes
    from one chunk -- and the thinking breaker is added on top by the same
    rule every proxy request gets (tiers.budget).
    """
    import model
    try:
        content = model.ask(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            effort="medium", max_tokens=EXTRACT_MAX_TOKENS,
            temperature=0.2, timeout=MODEL_TIMEOUT).strip()
    except model.BudgetEvent as e:
        raise RuntimeError(describe(
            f"extraction hit the token limit: {e}", True,
            "lower YAMADORI_EXTRACT_CHUNK_CHARS so each chunk asks for less, "
            "or raise YAMADORI_EXTRACT_MAX_TOKENS", "operator")) from e
    if not content:
        raise RuntimeError(describe(
            "the model finished without writing anything for this chunk",
            True, "retried automatically; if it repeats, read the chunk",
            "operator"))
    return content


def parse_rows(reply: str) -> list[dict]:
    """The JSON array in a reply, tolerating code fences and prose around it."""
    reply = re.sub(r"(?s)<think>.*?</think>", "", reply)
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array in the reply")
    rows = json.loads(reply[start:end + 1])
    if not isinstance(rows, list):
        raise ValueError("reply JSON is not an array")
    return [r for r in rows if isinstance(r, dict)]


def verified(row: dict, chunk_norm: str) -> bool:
    ev = _norm(str(row.get("evidence") or ""))
    return (len(ev) >= MIN_EVIDENCE_CHARS and ev in chunk_norm
            and bool(str(row.get("recipe") or "").strip()))


def guard_output(path: str, dataset_id: str) -> None:
    """Refuse to overwrite anything this dataset did not write, or anything a
    person has already reviewed."""
    if not os.path.exists(path):
        return
    foreign = reviewed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                foreign += 1
                continue
            if r.get("_dataset") != dataset_id:
                foreign += 1
            elif (r.get("_state") or "unreviewed") != "unreviewed":
                reviewed += 1
    name = os.path.basename(path)
    if foreign:
        raise Permanent(
            f"{name} already holds {foreign} row(s) this dataset did not "
            "write -- probably a hand-collected corpus with the same name",
            "rename the dataset so its recipe file is new")
    if reviewed:
        raise Permanent(
            f"re-extracting would overwrite {reviewed} row(s) in {name} that "
            "a person has already reviewed",
            "move or delete the file deliberately, then re-run the job")


def handle_extract(job: dict, ctx: Context) -> dict:
    ds = _dataset(job)
    text = source_text(ds)
    if not text.strip():
        raise Permanent("the dataset has no source text to read",
                        "give it a URL or paste the text")
    parts = chunks_of(text)
    if len(parts) > MAX_CHUNKS:
        raise Permanent(
            f"the source is {len(parts)} chunks of {CHUNK_CHARS:,} characters,"
            f" over the {MAX_CHUNKS}-chunk limit (each chunk is one model "
            "call)", "split the source, or raise YAMADORI_EXTRACT_MAX_CHUNKS")
    rfile = job["payload"].get("recipe_file") or datasets.recipe_file(ds)
    path = os.path.join(datasets.RECIPES, rfile)
    guard_output(path, ds["id"])

    header = (f"SOURCE: {ds.get('source_name') or ds.get('name')}\n"
              f"AREA: {ds.get('language') or ''}\n\n")
    kept: list[dict] = []
    proposed = dropped = failed = 0
    failures: list[str] = []
    for i, part in enumerate(parts):
        ctx.beat(f"chunk {i + 1}/{len(parts)}: {len(kept)} recipes kept so far")
        try:
            rows = parse_rows(ask_model(EXTRACT_SYSTEM, header + part))
        except (ValueError, RuntimeError, OSError) as e:
            failed += 1
            failures.append(f"chunk {i + 1}: {e}"[:300])
            continue
        norm = _norm(part)
        for r in rows:
            proposed += 1
            if not verified(r, norm):
                dropped += 1
                continue
            kept.append({
                "area": ds.get("language") or ds.get("name"),
                "category": str(r.get("category") or "").strip(),
                "recipe": str(r["recipe"]).strip(),
                "trigger_condition": str(r.get("trigger_condition") or "")
                .strip(),
                "evidence": str(r["evidence"]).strip(),
                "confidence": ("high" if r.get("confidence") == "high"
                               else "medium"),
                "source_url": ds.get("source_url"),
                "source_name": ds.get("source_name"),
                "license_if_known": ds.get("licence"),
                "language": ds.get("language"),
                "domains": ds.get("domains") or [],
                "_state": "unreviewed",
                "_dataset": ds["id"],
                "_job": job["id"],
                "_extracted_at": int(time.time()),
            })

    counts = {"extract_chunks": len(parts), "extract_chunks_failed": failed,
              "extract_proposed": proposed, "extract_kept": len(kept),
              "extract_dropped_unverified": dropped}
    _merge_counts(ds["id"], counts)
    if failed == len(parts):
        # Every call failed: that is the model or the transport, not the
        # source. Retryable, and the first failure says what happened.
        raise RuntimeError(describe(
            f"all {len(parts)} extraction call(s) failed; first: "
            f"{failures[0] if failures else '?'}", True,
            "check llama-swap on :11434 is serving "
            f"{EXTRACT_MODEL!r}", "operator"))
    if not kept:
        raise Permanent(
            f"the model proposed {proposed} recipe(s) and none quoted the "
            f"source verbatim ({dropped} dropped by the evidence check)",
            "inspect the source text; this source may carry no actionable "
            "advice")
    _write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in kept).encode("utf-8"))
    return dict(counts, recipe_file=rfile, failures=failures[:10])


# ---------------------------------------------------------------------------
# dataset.index  (gpu -- the embedding model)
# ---------------------------------------------------------------------------
def handle_index(job: dict, ctx: Context) -> dict:
    import numpy as np

    import hints

    ds = _dataset(job)
    rfile = job["payload"].get("recipe_file") or datasets.recipe_file(ds)
    ctx.beat("embedding the hints corpus")
    rows, mat = hints.refresh()
    mine = sum(1 for r in rows if r.get("_file") == rfile)
    if not mine:
        raise Permanent(
            f"{rfile} contributes no servable rows to the hints corpus "
            f"(looked in {hints.CORPUS})",
            "check extract wrote the file and that not every row is marked "
            "reject", owner="worker")
    if mat is None or mat.shape[0] != len(rows):
        raise RuntimeError(describe(
            f"hints cache has {0 if mat is None else mat.shape[0]} vectors "
            f"for {len(rows)} rows", True, "re-run; if it repeats, delete "
            f"{hints.CACHE} and re-run", "operator"))
    norms = np.linalg.norm(mat, axis=1)
    # PROTOCOL rule 1: an index of zero vectors once reported success and a
    # component was cut on it. Check the vectors, not that the file exists.
    if float(norms.min()) < 0.5:
        raise RuntimeError(describe(
            f"{int((norms < 0.5).sum())} hint vector(s) have norm < 0.5 "
            "(expected 1.0) -- the embedding server returned zeros", True,
            "check the embedding model on llama-swap, then re-run",
            "operator"))
    counts = {"indexed_rows": mine, "corpus_rows": len(rows)}
    _merge_counts(ds["id"], counts)
    return dict(counts, dim=int(mat.shape[1]),
                norm_min=round(float(norms.min()), 4),
                norm_max=round(float(norms.max()), 4),
                cache=os.path.abspath(hints.CACHE),
                note="the proxy holds hints in memory from its first use; "
                     "restart it to serve these rows")


# ---------------------------------------------------------------------------
# dataset.assist  (gpu -- the model; it also fetches a LICENSE file or two)
#
# The model reads the fetched source and fills what the source ESTABLISHES.
#
#   licence       ONLY from a verbatim quote that is in the fetched text or in
#                 a licence file fetched from the same host (or, for
#                 GitHub-hosted docs, the repo's raw LICENSE). The quote is
#                 verified with the same rule as extract's evidence check, the
#                 licence named must sit inside the quote, and the quote must
#                 be about licensing. Anything else: the field stays empty and
#                 missing() says what was searched. A guessed licence is worse
#                 than a missing one.
#   source_name   proposed; `evidence` only if a verbatim line containing it
#                 was quoted and verified.
#   language      proposed. An inference, labelled as one.
#   domains       proposed, and only from domains.DOMAINS; anything else is
#                 recorded as rejected, never stored.
#
# Nothing the model said is stored without its provenance: datasets.answer()
# writes each value and its provenance entry in one update, and never over a
# field a person answered while the model was thinking (only_blank).
#
# When nothing is left missing it advances the dataset itself -- so a
# well-documented source goes fetch -> assist -> extract -> index with no
# human -- EXCEPT when the licence it found is one datasets.RESTRICTED
# surfaces (AGPL, no-redistribution, all rights reserved, ...). Serving rows
# under such a licence is a decision about the rows, and nobody made it: the
# dataset is HELD in clarify with the warning showing and the reason in
# `assist.held`, and a person advances it. HOLD_RESTRICTED turns that off.
# ---------------------------------------------------------------------------
ASSIST_MAX_TOKENS = int(os.environ.get("YAMADORI_ASSIST_MAX_TOKENS", "1024"))
ASSIST_HEAD_CHARS = 8000
ASSIST_TAIL_CHARS = 4000
ASSIST_KEYWORD_LINES = 40
LICENCE_FILE_CHARS = 6000
LICENCE_FETCH_BYTES = 512 * 1024
LICENCE_FETCH_TIMEOUT = 20
LICENCE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
HOLD_RESTRICTED = os.environ.get("YAMADORI_ASSIST_HOLD_RESTRICTED", "1") != "0"
# "MIT License" is 11 characters and a complete statement; below this a
# "quote" is a word, and a word is in every document.
MIN_LICENCE_QUOTE_CHARS = 8
# A quote that fills the licence field must be ABOUT licensing. Flat text, no
# grammar: a keyword test is the right tool here (PROTOCOL rule 8).
LICENCE_WORDS = re.compile(
    r"licen[cs]|copyright|all rights reserved|public domain|creative commons"
    r"|\bcc[- ]?by|\bcc0\b|spdx|redistribut", re.I)


def licence_candidates(url: str) -> list[str]:
    """Where a licence file for this source may live. Same host only, except
    that GitHub-hosted docs (github.com, raw.githubusercontent.com,
    *.github.io) look in the repo's raw LICENSE, which is the same repo."""
    import urllib.parse
    p = urllib.parse.urlsplit(url or "")
    if p.scheme not in ("http", "https") or not p.hostname:
        return []
    host = p.hostname.lower()
    parts = [x for x in p.path.split("/") if x]
    gh = None
    if host == "github.com" and len(parts) >= 2:
        ref = parts[3] if len(parts) >= 4 and parts[2] in ("blob", "tree") \
            else "HEAD"
        gh = (parts[0], parts[1], ref)
    elif host == "raw.githubusercontent.com" and len(parts) >= 3:
        gh = (parts[0], parts[1], parts[2])
    elif host.endswith(".github.io"):
        owner = host[:-len(".github.io")]
        gh = (owner, parts[0] if parts else host, "HEAD")
    if gh:
        o, r, ref = gh
        return [f"https://raw.githubusercontent.com/{o}/{r}/{ref}/{n}"
                for n in LICENCE_NAMES]
    base = f"{p.scheme}://{p.netloc}"
    dirs = parts if p.path.endswith("/") else parts[:-1]
    roots = ([f"{base}/{'/'.join(dirs)}/"] if dirs else []) + [f"{base}/"]
    return [f"{d}{n}" for d in roots for n in LICENCE_NAMES[:3]]


def _allowed_hosts(url: str) -> set[str]:
    import urllib.parse
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return {host, "raw.githubusercontent.com"} if (
        host in ("github.com", "raw.githubusercontent.com")
        or host.endswith(".github.io")) else {host}


def fetch_licence_file(url: str, allowed: set[str]) -> tuple[dict, str]:
    """GET one candidate. Returns (what happened, its text or '')."""
    import urllib.parse
    rec: dict = {"where": url, "kind": "licence_file"}
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "yamadori-dataset-worker/1"})
        with urllib.request.urlopen(req, timeout=LICENCE_FETCH_TIMEOUT) as r:
            final = r.geturl()
            rec.update(status=r.status, final_url=final)
            host = (urllib.parse.urlsplit(final).hostname or "").lower()
            if host not in allowed:
                rec["result"] = f"redirected off-host to {host}; not read"
                return rec, ""
            raw = r.read(LICENCE_FETCH_BYTES + 1)
            ctype = r.headers.get_content_type()
            charset = r.headers.get_content_charset()
    except urllib.error.HTTPError as e:
        rec.update(status=e.code, result=f"HTTP {e.code}")
        return rec, ""
    except (urllib.error.URLError, OSError, ValueError) as e:
        rec["result"] = f"not reachable: {e}"
        return rec, ""
    if len(raw) > LICENCE_FETCH_BYTES:
        raw, rec["truncated"] = raw[:LICENCE_FETCH_BYTES], True
    text = raw.decode(charset or "utf-8", errors="replace")
    rec["sha256"] = hashlib.sha256(raw).hexdigest()
    rec["html"] = bool("html" in ctype
                       or re.search(r"(?i)<html|<body", text[:2000]))
    if rec["html"]:
        # Many sites answer 200 with a page for any path. It is still text
        # from the same site, so it is read -- but it is not a licence file,
        # and the record says so rather than pretending one was found.
        text = html_to_text(text)
        rec["result"] = (f"HTTP {rec['status']} with an HTML page, not a "
                         "licence file (read anyway: same site)")
    else:
        rec["result"] = f"read {len(raw):,} bytes"
    rec["chars"] = len(text)
    return rec, text


def _excerpt(text: str) -> tuple[str, str]:
    """What the model is shown of a long source, and a sentence saying so.

    Licence statements live in headers and footers, so the model sees the
    head and the tail, plus every line anywhere that mentions licensing. The
    quote it returns is verified against the WHOLE text, not this excerpt.
    """
    if len(text) <= ASSIST_HEAD_CHARS + ASSIST_TAIL_CHARS:
        return text, f"all {len(text):,} characters"
    head, tail = text[:ASSIST_HEAD_CHARS], text[-ASSIST_TAIL_CHARS:]
    middle = text[ASSIST_HEAD_CHARS:-ASSIST_TAIL_CHARS]
    lines = [ln.strip() for ln in middle.split("\n")
             if LICENCE_WORDS.search(ln)][:ASSIST_KEYWORD_LINES]
    shown = (head + f"\n\n[... {len(middle):,} characters omitted; the lines "
             "from them that mention licensing follow ...]\n"
             + "\n".join(lines) + "\n[... end of those lines ...]\n\n" + tail)
    return shown, (f"the first {ASSIST_HEAD_CHARS:,} and last "
                   f"{ASSIST_TAIL_CHARS:,} of {len(text):,} characters, plus "
                   f"{len(lines)} line(s) from the middle that mention "
                   "licensing")


def _assist_system() -> str:
    import domains as domain_gate
    vocab = ", ".join(sorted(domain_gate.DOMAINS))
    return f"""You read documents about one technical source and report four \
facts about it as JSON.

| key | what to write | how it is checked |
|---|---|---|
| source_name | the name of the work: its title or project name | your reading; null if the documents do not say |
| source_name_quote | a line copied exactly from a document that contains source_name | verified character for character; null if none |
| language | the programming language or technical area it covers | your reading; null if unclear |
| domains | a list using ONLY these words: {vocab} | other words are dropped |
| licence | the licence name, copied exactly as the document writes it, from inside licence_quote | must appear inside licence_quote |
| licence_quote | the sentence or line that states the licence, copied character for character | verified against the documents |
| licence_quote_url | the DOCUMENT url the quote was copied from | |

A licence whose quote is not in the documents is discarded and a person is \
asked instead, so null licence and null licence_quote are the right answer \
when no document states a licence. A bare copyright line is not a licence; \
"All rights reserved" is one, and is copied like any other.

Reply with one JSON object with exactly those keys, and nothing else."""


def _ask_assist(system: str, user: str) -> str:
    """One call through the one door. Looked up at call time, so a test that
    replaces model.ask replaces exactly this."""
    import model
    try:
        content = (model.ask(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            effort="medium", max_tokens=ASSIST_MAX_TOKENS, temperature=0.0,
            timeout=MODEL_TIMEOUT) or "").strip()
    except model.BudgetEvent as e:
        raise RuntimeError(describe(
            f"the assist hit the token limit: {e}", True,
            "raise YAMADORI_ASSIST_MAX_TOKENS", "operator")) from e
    if not content:
        raise RuntimeError(describe(
            "the model finished without writing anything for the assist",
            True, "retried automatically; if it repeats, check the model on "
            f"{UPSTREAM}", "operator"))
    return content


def parse_object(reply: str) -> dict:
    """The JSON object in a reply, tolerating code fences and prose."""
    reply = re.sub(r"(?s)<think>.*?</think>", "", reply or "")
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(reply[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def _clean(v) -> str:
    if v is None or isinstance(v, (dict, list)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in datasets.NON_ANSWERS else s[:300]


def find_verbatim(quote: str, docs: list[dict],
                  prefer: str | None = None) -> str | None:
    """Where this quote appears, by the extract evidence rule (whitespace-
    collapsed, case-folded substring). The model's claimed location is tried
    first but is not trusted: the location stored is where it was FOUND."""
    q = _norm(quote)
    if not q:
        return None
    order = sorted(docs, key=lambda d: d["where"] != prefer)
    for d in order:
        if q in d["norm"]:
            return d["where"]
    return None


def licence_evidence(reply: dict, docs: list[dict]) -> tuple[dict | None, str]:
    """(provenance entry, '') for a verified licence, or (None, why not)."""
    value = _clean(reply.get("licence"))
    # Not _clean(): that truncates, and a truncated quote is not the quote.
    raw = reply.get("licence_quote")
    quote = "" if isinstance(raw, (dict, list)) else str(raw or "").strip()
    if quote.lower() in datasets.NON_ANSWERS:
        quote = ""
    if not value and not quote:
        return None, ""
    if not quote:
        return None, "the model named a licence but quoted nothing for it"
    if not value:
        return None, "the model quoted text but named no licence from it"
    why = []
    if len(_norm(quote)) < MIN_LICENCE_QUOTE_CHARS:
        why.append(f"the quote is under {MIN_LICENCE_QUOTE_CHARS} characters")
    if not LICENCE_WORDS.search(quote):
        why.append("the quote does not mention a licence")
    if _norm(value) not in _norm(quote):
        why.append("the licence it named is not inside its quote")
    where = find_verbatim(quote, docs,
                          prefer=str(reply.get("licence_quote_url") or ""))
    if where is None:
        why.append("the quote is not in any fetched text, character for "
                   "character")
    if why:
        return None, "; ".join(why)
    return {"provenance": "evidence", "value": value, "quote": quote,
            "found_in": where,
            "claimed_url": str(reply.get("licence_quote_url") or "") or None
            }, ""


def _searched_sentence(searched: list[dict]) -> str:
    bits = []
    for s in searched:
        if s["kind"] == "source":
            bits.append(f"{s['where']} (the model was shown {s['shown']})")
        else:
            bits.append(f"{s['where']} ({s.get('result', '?')})")
    return "; ".join(bits)


def handle_assist(job: dict, ctx: Context) -> dict:
    ds = _dataset(job)
    if ds["stage"] != "clarify":
        return {"skipped": f"the dataset is in {ds['stage']}; the assist only "
                           "fills clarifying answers"}
    open_fields = [m["field"] for m in datasets.missing(ds)
                   if m["field"] in datasets.ASSISTED]
    if not open_fields:
        return {"skipped": "every field the assist may fill was answered "
                           "before it ran"}
    text = source_text(ds)
    if not text.strip():
        raise Permanent("the dataset has no source text to read",
                        "give it a URL or paste the text")
    url = (ds.get("source_url") or "").strip()
    is_web = bool(re.match(r"^https?://", url))
    where = url if is_web and not (ds.get("source") or "").strip() \
        else "the pasted source text"
    shown, how = _excerpt(text)
    docs = [{"where": where, "kind": "source", "text": text, "shown": shown}]
    searched = [{"where": where, "kind": "source", "chars": len(text),
                 "shown": how}]

    if "licence" in open_fields and is_web:
        allowed = _allowed_hosts(url)
        d = dataset_dir(ds["id"])
        for i, cand in enumerate(licence_candidates(url)):
            ctx.beat(f"looking for a licence file: {cand}")
            rec, body = fetch_licence_file(cand, allowed)
            searched.append(rec)
            if not body.strip():
                continue
            loc = rec.get("final_url") or cand
            _write_atomic(os.path.join(d, f"licence-{i}.txt"),
                          body.encode("utf-8"))
            docs.append({"where": loc, "kind": "licence_file", "text": body,
                         "shown": body[:LICENCE_FILE_CHARS]})
            if not rec.get("html"):
                break       # a real licence file: stop looking
    for doc in docs:
        doc["norm"] = _norm(doc["text"])

    user = "\n\n".join(
        f"=== DOCUMENT {i + 1}: {doc['where']} ===\n{doc['shown']}"
        for i, doc in enumerate(docs))
    ctx.beat(f"asking the model about {len(docs)} document(s)")
    reply_text = _ask_assist(_assist_system(), user)
    try:
        reply = parse_object(reply_text)
    except ValueError as e:
        raise RuntimeError(describe(
            f"the assist reply was not a JSON object ({e}); it began "
            f"{reply_text[:160]!r}", True, "retried automatically; if it "
            "repeats, answer the fields by hand", "operator")) from e

    import domains as domain_gate
    answers: dict = {}
    prov: dict = {}
    rejected: dict = {}
    not_found: dict = {}
    looked = _searched_sentence(searched)

    if "source_name" in open_fields:
        v = _clean(reply.get("source_name"))
        if v:
            q = str(reply.get("source_name_quote") or "").strip()
            loc = (find_verbatim(q, docs) if q and _norm(v) in _norm(q)
                   else None)
            answers["source_name"] = v
            prov["source_name"] = (
                {"provenance": "evidence", "quote": q, "found_in": loc} if loc
                else {"provenance": "proposed",
                      "basis": "the model's reading of the source"})
        else:
            not_found["source_name"] = (f"the assist read {looked} and did "
                                        "not propose a name")
    if "language" in open_fields:
        v = _clean(reply.get("language"))
        if v:
            answers["language"] = v
            prov["language"] = {"provenance": "proposed",
                                "basis": "the model's reading of the source"}
        else:
            not_found["language"] = (f"the assist read {looked} and did not "
                                     "propose a language or area")
    if "domains" in open_fields:
        raw = reply.get("domains")
        raw = raw if isinstance(raw, list) else ([raw] if raw else [])
        vals = [str(x).strip() for x in raw if str(x).strip()]
        good = sorted({x for x in vals if x in domain_gate.DOMAINS})
        bad = sorted({x for x in vals if x not in domain_gate.DOMAINS})
        if bad:
            rejected["domains"] = {"values": bad,
                                   "why": "not in domains.DOMAINS"}
        if good:
            answers["domains"] = good
            prov["domains"] = {"provenance": "proposed",
                               "basis": "the model's reading of the source, "
                                        "restricted to domains.DOMAINS"}
        else:
            not_found["domains"] = (f"the assist read {looked} and proposed "
                                    "no tag from domains.DOMAINS")
    if "licence" in open_fields:
        entry, why = licence_evidence(reply, docs)
        if entry:
            answers["licence"] = entry.pop("value")
            prov["licence"] = entry
        else:
            if why:
                rejected["licence"] = {
                    "why": why,
                    "value": _clean(reply.get("licence")),
                    "quote": str(reply.get("licence_quote") or "")[:400],
                    "claimed_url": str(reply.get("licence_quote_url") or "")}
            not_found["licence"] = (
                f"the assist searched {looked}" + (
                    f" -- the model offered a quote that failed verification "
                    f"({why}), so it was discarded" if why else
                    " and found no verbatim licence statement"))

    meta = {"job": job["id"], "at": time.time(), "searched": searched,
            "rejected": rejected, "not_found": not_found, "held": None}
    ds = datasets.answer(ds["id"], answers, provenance=prov, meta=meta,
                         only_blank=True)
    result = {"filled": {k: prov[k]["provenance"] for k in answers},
              "missing": [m["field"] for m in datasets.missing(ds)],
              "documents": len(docs)}
    _merge_counts(ds["id"], {"assist_filled": len(answers),
                             "assist_documents": len(docs)})

    if ds["stage"] != "clarify" or result["missing"]:
        return result
    restricted = [w for w in datasets.warnings(ds) if w["kind"] == "licence"]
    by_model = (ds.get("assist") or {}).get("fields", {}).get(
        "licence", {}).get("provenance") == "evidence"
    if restricted and by_model and HOLD_RESTRICTED:
        held = (f"held in clarify: the licence the assist found carries a "
                f"restriction ({restricted[0]['what']}). Every question is "
                "answered; advancing is a decision about serving these rows, "
                "and a person makes it.")
        datasets.answer(ds["id"], {}, meta={"held": held})
        result["held"] = held
        return result
    try:
        result["advanced_to"] = datasets.advance(ds["id"])["stage"]
    except datasets.Blocked as e:
        result["not_advanced"] = [r.get("what") for r in e.reasons]
    return result


# ---------------------------------------------------------------------------
# dataset.label  (cpu)  and  dataset.train  (gpu)
#
# Refused, on purpose, with the exact gap named. The TARGET format is defined:
# docs/LAYA.md "RUNBOOK" gives each task's label glob and required fields
# (route_in: question/context/label; grounded_excerpt: finding/cites/excerpt/
# label), and scripts/train_laya.py picks up any new file matching the glob.
# What is NOT defined is the step before: what a `laya` dataset's source rows
# are and how they become labelled rows -- the labels are judgements
# (investigate / answer_directly / clarify), and no process here can make
# them. Training is also not automatic in the runbook: a retrain is promoted
# only if it beats the regex and passes the floors in bench/laya_baseline.json
# via bench/test_laya_head.py. A handler that "succeeded" here would either
# invent labels or overwrite a serving artefact without that gate.
# ---------------------------------------------------------------------------
def handle_label(job: dict, ctx: Context) -> dict:
    raise Permanent(
        "a laya dataset has no defined mapping from its source to labelled "
        "rows; the target format exists (docs/LAYA.md RUNBOOK step 1) but the "
        "labels are judgements no job can make",
        "decide what a laya dataset's source is (pre-labelled rows to "
        "validate and copy into bench/laya_<task>_labels_<slug>.jsonl is the "
        "simplest), then implement worker.handle_label", owner="developer")


def handle_train(job: dict, ctx: Context) -> dict:
    raise Permanent(
        "retraining is gated in docs/LAYA.md RUNBOOK steps 2-3: train_laya.py "
        "writes serving artefacts to index/laya/, and a head is promoted only "
        "if it beats the regex and passes bench/test_laya_head.py floors",
        "implement worker.handle_train to train into a staging --out-dir, run "
        "the floors test, and promote only on pass", owner="developer")


HANDLERS = {
    datasets.FETCH[0]: handle_fetch,
    "dataset.extract": handle_extract,
    "dataset.index": handle_index,
    "dataset.label": handle_label,
    "dataset.train": handle_train,
    datasets.ASSIST[0]: handle_assist,
}


def _register_skills() -> None:
    """The skill pipeline's stages (mcp/skill_pipeline.py) run in this same
    worker, on the same lanes. Imported here, not at the top: skill_pipeline
    never imports this module, so the two cannot load each other twice."""
    import skill_pipeline
    HANDLERS.update(skill_pipeline.HANDLERS)


_register_skills()


def _register_prices() -> None:
    """The hosted-API price snapshot (mcp/prices.py): one net-lane job,
    enqueued by the loop below at most once a day."""
    import prices
    HANDLERS[prices.QUEUE] = prices.handle_refresh


_register_prices()


def _register_deep() -> None:
    """Deep thinking's idle-time learner (mcp/deep_learn.py, Phase 0.6):
    one cpu-lane job, enqueued by the loop below only when the stack is
    idle and labelled rows are pending."""
    import deep_learn
    HANDLERS[deep_learn.LEARN[0]] = deep_learn.handle_learn


_register_deep()


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------
def _heartbeat(job_id: str, stop: threading.Event) -> None:
    while not stop.wait(BEAT_SECONDS):
        try:
            jobs.beat(job_id)
        except Exception:                                        # noqa: BLE001
            pass


def advance_after(job: dict) -> str | None:
    """Move the dataset on if this job was the one its stage was waiting for.

    Returns the new stage, or None. A Blocked advance is not an error: after a
    fetch, `clarify` may still have unanswered questions, which is the
    operator's turn, not a failure.
    """
    if str(job.get("queue") or "").startswith("skill."):
        # A skill stage: the version records where it is, and the next
        # stage is enqueued only now that this one's job is done.
        import skill_pipeline
        return skill_pipeline.advance_after(job)
    ds = datasets.get(job.get("dataset") or "") if job.get("dataset") else None
    if ds is None:
        return None
    if job["queue"] == datasets.ASSIST[0]:
        # The assist advances (or HOLDS) the dataset itself; advancing here
        # as well would walk straight past a hold.
        return None
    if job["queue"] == datasets.FETCH[0] and ds["stage"] == "clarify":
        # The source is on disk now, so the model can read it. While an
        # assist is pending it owns the advance; if none is needed (every
        # field it may fill is answered) fall through and advance as before.
        if "job" in datasets.enqueue_assist(ds["id"]):
            return None
    waiting =(ds["stage"] == "clarify" if job["queue"] == datasets.FETCH[0]
               else ds["stage"] == job.get("stage"))
    if not waiting:
        return None
    try:
        return datasets.advance(ds["id"])["stage"]
    except datasets.Blocked:
        return None


def run_one(job: dict) -> str:
    """Run one claimed job to a terminal or re-queued state. Returns it."""
    handler = HANDLERS.get(job["queue"])
    stop = threading.Event()
    beater = threading.Thread(target=_heartbeat, args=(job["id"], stop),
                              daemon=True)
    beater.start()
    try:
        if handler is None:
            raise Permanent(f"no handler for queue {job['queue']!r}; this "
                            f"worker knows {sorted(HANDLERS)}",
                            "add a handler in mcp/worker.py",
                            owner="developer")
        result = handler(job, Context(job))
    except Permanent as e:
        stop.set()
        return jobs.fail(job["id"], e.text(), retry=False)
    except Exception as e:                                       # noqa: BLE001
        stop.set()
        msg = str(e)
        if "| retryable:" not in msg:
            msg = describe(f"{type(e).__name__}: {msg}", True,
                           "retried automatically until attempts run out; "
                           "then read the traceback in the worker log",
                           "worker")
        print(f"[worker] {job['queue']} {job['id']} raised:\n"
              f"{traceback.format_exc()}", file=sys.stderr, flush=True)
        return jobs.fail(job["id"], msg, retry=True)
    finally:
        stop.set()
    result = dict(result or {})
    jobs.finish(job["id"], result)
    try:
        nxt = advance_after(job)
    except Exception as e:                                       # noqa: BLE001
        # The job did its work; failing to enqueue the next stage is a
        # separate fact and must not turn `done` into something else.
        print(f"[worker] advance after {job['id']} failed: {e}",
              file=sys.stderr, flush=True)
        nxt = None
    if nxt:
        result["advanced_to"] = nxt
        jobs.finish(job["id"], result)
    return "done"


def _worker_id(lane: str, slot: int) -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{lane}{slot}"


def lane_loop(lane: str, slot: int, stop: threading.Event,
              once: bool = False) -> int:
    """Claim and run jobs in one lane slot. Returns how many it ran."""
    n = 0
    wid = _worker_id(lane, slot)
    while not stop.is_set():
        job = jobs.claim(lane, wid)
        if job is None:
            if once:
                return n
            stop.wait(POLL_SECONDS)
            continue
        state = run_one(job)
        n += 1
        print(f"[worker] {wid} {job['queue']} {job['id']} -> {state}",
              flush=True)
    return n


def run(once: bool = False, lanes: dict | None = None) -> int:
    """Start one thread per lane slot. With `once`, drain and return."""
    lanes = dict(lanes or jobs.LANES)
    n = jobs.reclaim() + jobs.reclaim_dead_local()
    if n:
        print(f"[worker] reclaimed {n} stale job(s) at startup", flush=True)
    stop = threading.Event()
    counts: list[int] = []

    def target(lane, slot):
        counts.append(lane_loop(lane, slot, stop, once))

    threads = [threading.Thread(target=target, args=(lane, i), daemon=True,
                                name=f"{lane}{i}")
               for lane, k in lanes.items() for i in range(k)]
    for t in threads:
        t.start()
    if once:
        for t in threads:
            t.join()
        # A job finishing can enqueue the next stage after another lane has
        # already found its queue empty; go round until nothing is queued.
        if any(jobs.snapshot()["lanes"].get(lane, {}).get("queued")
               for lane in lanes):
            return sum(counts) + run(once=True, lanes=lanes)
        return sum(counts)

    def _stop(*_):
        stop.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    print(f"[worker] {os.getpid()} lanes {lanes} db {os.path.abspath(jobs.DB)}",
          flush=True)
    last = time.time()
    while not stop.is_set():
        stop.wait(1.0)
        if time.time() - last >= RECLAIM_EVERY:
            last = time.time()
            try:
                n = jobs.reclaim() + jobs.reclaim_dead_local()
                if n:
                    print(f"[worker] reclaimed {n} stale job(s)", flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"[worker] reclaim failed: {e}", file=sys.stderr,
                      flush=True)
            # Watched skills whose re-fetch is due, and -- only when the
            # stack is idle -- learning from skill-selection fallbacks
            # (skill_learn.idle_state). jobs.py has no cron.
            try:
                import skill_learn
                import skill_pipeline
                w = skill_pipeline.schedule_watches()
                if w:
                    print(f"[worker] enqueued {len(w)} skill watch job(s)",
                          flush=True)
                if skill_learn.schedule():
                    print("[worker] idle: enqueued a skill learn job",
                          flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"[worker] skill scheduling failed: {e}",
                      file=sys.stderr, flush=True)
            # Deep thinking's learner (Phase 0.6): thresholds within bounds
            # and description proposals from labelled outcomes, when idle.
            try:
                import deep_learn
                if deep_learn.schedule():
                    print("[worker] idle: enqueued a deep learn job",
                          flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"[worker] deep learn scheduling failed: {e}",
                      file=sys.stderr, flush=True)
            # The hosted-API price snapshot for the dashboard's savings
            # estimate (mcp/prices.py): at most one fetch job a day.
            try:
                import prices
                jid = prices.schedule()
                if jid:
                    print(f"[worker] enqueued {prices.QUEUE} {jid}", flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"[worker] price scheduling failed: {e}",
                      file=sys.stderr, flush=True)
    # Running handlers are not interrupted: their jobs stay `running` with a
    # heartbeat that stops, and reclaim() on the next start returns them.
    return sum(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="claim and run queued jobs")
    ap.add_argument("--once", action="store_true",
                    help="drain what is queued, then exit")
    a = ap.parse_args()
    # Generations the worker runs through mcp/model.py are counted in the
    # token ledger (mcp/token_ledger.py). Here, not on import: tests import
    # this module and must not write the real ledger.
    import token_ledger
    token_ledger.enable()
    print(f"[worker] ran {run(once=a.once)} job(s)")
