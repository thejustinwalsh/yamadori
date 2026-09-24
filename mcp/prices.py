#!/usr/bin/env python
"""Hosted API prices, for the dashboard's "what would this have cost" estimate.

WHERE THE PRICES COME FROM

OpenRouter's public model list, GET https://openrouter.ai/api/v1/models. It
needs no key, and none is ever sent. Prices there are US dollars per token,
as strings; `pricing.input_cache_read`, where a listing has it, is the price
of a prompt token served from the provider's cache. A listing's `overrides`
(a higher price above some prompt length) are NOT applied: the estimate uses
each listing's base price and says so.

HOW THEY ARE KEPT

The WHOLE response is stored, dated, in index/prices/ (PROTOCOL rule 16:
fetch the whole answer, narrow it while reading). Refreshing is a job on the
`net` lane of the durable queue (mcp/jobs.py, queue "prices.refresh"),
enqueued by the worker's loop at most once per REFRESH_SECONDS -- counted
from the last job CREATED, so a failing fetch is retried by the queue's own
attempts and then not again until the next day. Nothing on a request path
fetches: the dashboard only reads the newest file.

THE TWO COMPARISONS

LIKE-FOR-LIKE is `qwen/qwen3.5-27b`: Bonsai 2 27B is a Qwen3.5-27B-family
model (operator, 2026-09-24), and that is OpenRouter's listing of the
Qwen3.5 base at the same size (dense 27B; Hugging Face Qwen/Qwen3.5-27B).
Not qwen3.6-27b or qwen3.8-27b (later generations of the same size), not
qwen3.5-35b-a3b (a mixture-of-experts of another size). On the snapshot of
2026-09-24 it lists NO cache-read price, so cached input is priced at its
uncached input price, and the panel says so.

AVERAGE is the MEDIAN, taken separately for uncached input, cached input
and output, over AVERAGE_SET: eleven ids chosen by hand on 2026-09-24, one
or two current general or coding models per major vendor on OpenRouter.
"Popular" is a judgment, not a measurement: the models endpoint carries no
usage figures. The three medians need not come from the same model. A
member without a cache-read price contributes its uncached input price to
the cached-input median (it bills cached input as input). A member missing
from a snapshot is listed as missing, never silently dropped.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_URL = "https://openrouter.ai/api/v1/models"
SNAPSHOT_DIR = os.environ.get(
    "YAMADORI_PRICES_DIR",
    os.path.abspath(os.path.join(HERE, "..", "index", "prices")))
PREFIX = "openrouter-models-"
QUEUE = "prices.refresh"
REFRESH_SECONDS = 86400          # at most daily
KEEP_SNAPSHOTS = 30
FETCH_TIMEOUT = 60
# A response with fewer listings than this is not a model list (an error
# page, a truncated body) and is refused rather than stored as prices.
MIN_MODELS = 50

LIKE_FOR_LIKE = {
    "id": "qwen/qwen3.5-27b",
    "why": ("Bonsai 2 27B is a Qwen3.5-27B-family model; this is OpenRouter's "
            "listing of the Qwen3.5 base at the same size (dense 27B, "
            "Hugging Face Qwen/Qwen3.5-27B). Not a later generation (3.6, "
            "3.8) and not the 35B-A3B mixture-of-experts."),
}

AVERAGE_SET = (
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5.5",
    "openai/gpt-6-sol",
    "openai/gpt-5.3-codex",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.7",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
    "qwen/qwen3-coder-next",
    "minimax/minimax-m3",
)
AVERAGE_DEFINITION = (
    "median of each price, taken separately, over a named set of 11 current "
    "general and coding models chosen by hand on 2026-09-24 (popularity is "
    "a judgment; the models endpoint has no usage figures)")


# ---------------------------------------------------------------------------
# Fetching and storing.
# ---------------------------------------------------------------------------

def _fetch(url: str = SOURCE_URL, timeout: int = FETCH_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "yamadori-prices/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def validate(doc) -> int:
    """How many listings carry a pricing object; raises if it is not a model
    list worth storing."""
    if not isinstance(doc, dict) or not isinstance(doc.get("data"), list):
        raise ValueError("not an OpenRouter model list: no `data` array")
    n = sum(1 for m in doc["data"] if isinstance(m, dict)
            and isinstance(m.get("id"), str) and isinstance(m.get("pricing"), dict))
    if n < MIN_MODELS:
        raise ValueError(f"only {n} priced listings (a model list has "
                         f"hundreds; refusing to store it as prices)")
    return n


def refresh(fetch=None, now: float | None = None,
            directory: str | None = None) -> dict:
    """Fetch the whole list and store it as today's snapshot. Raises on a
    fetch error or an implausible body; nothing is written then."""
    d = directory or SNAPSHOT_DIR
    now = time.time() if now is None else now
    raw = (fetch or _fetch)()
    doc = json.loads(raw)
    n = validate(doc)
    date = _dt.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{PREFIX}{date}.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"source": SOURCE_URL, "fetched_at": now, "date": date,
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "listings": n, "response": doc}, f)
    os.replace(tmp, path)
    for old in sorted(glob.glob(os.path.join(d, f"{PREFIX}*.json")))[:-KEEP_SNAPSHOTS]:
        try:
            os.remove(old)
        except OSError:
            pass
    return {"path": path, "date": date, "listings": n,
            "like_for_like_listed": any(m.get("id") == LIKE_FOR_LIKE["id"]
                                        for m in doc["data"] if isinstance(m, dict))}


def latest_path(directory: str | None = None) -> str | None:
    files = sorted(glob.glob(os.path.join(directory or SNAPSHOT_DIR,
                                          f"{PREFIX}*.json")))
    return files[-1] if files else None


_cache: dict = {}


def load_latest(directory: str | None = None) -> dict | None:
    """The newest snapshot, parsed once per file version."""
    p = latest_path(directory)
    if not p:
        return None
    try:
        key = (p, os.path.getmtime(p))
    except OSError:
        return None
    if _cache.get("key") != key:
        with open(p, encoding="utf-8") as f:
            snap = json.load(f)
        _cache.update(key=key, snap=snap, table=None)
    return _cache["snap"]


# ---------------------------------------------------------------------------
# The queue: the handler and the at-most-daily schedule.
# ---------------------------------------------------------------------------

def handle_refresh(job: dict, ctx=None) -> dict:
    """Worker handler for QUEUE."""
    return refresh()


def due(now: float | None = None, directory: str | None = None) -> bool:
    """Whether the newest snapshot is at least REFRESH_SECONDS old (or
    absent) AND no refresh job was created in the last REFRESH_SECONDS."""
    import jobs
    now = time.time() if now is None else now
    snap = None
    p = latest_path(directory)
    if p:
        try:
            snap = os.path.getmtime(p)
        except OSError:
            snap = None
    if snap is not None and now - snap < REFRESH_SECONDS:
        return False
    last = jobs.last_created(QUEUE)
    return last is None or now - last >= REFRESH_SECONDS


def schedule(now: float | None = None, directory: str | None = None) -> str | None:
    """Enqueue one refresh on the net lane if due(). Returns the job id."""
    if not due(now, directory):
        return None
    import jobs
    return jobs.add(QUEUE, {"url": SOURCE_URL}, lane="net", max_attempts=3)


# ---------------------------------------------------------------------------
# Reading prices out of a snapshot.
# ---------------------------------------------------------------------------

def _per_token(v) -> float | None:
    """A listed price in USD per token; None when absent or not a price
    (OpenRouter lists -1 for a router whose price varies)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x >= 0 else None


def listing_price(m: dict) -> dict:
    p = m.get("pricing") if isinstance(m.get("pricing"), dict) else {}
    inp, out = _per_token(p.get("prompt")), _per_token(p.get("completion"))
    cache = _per_token(p.get("input_cache_read"))
    return {"id": m.get("id"), "name": m.get("name"),
            "hugging_face_id": m.get("hugging_face_id") or None,
            "input": inp, "output": out,
            "cache_read": cache if cache is not None else inp,
            "cache_read_listed": cache is not None,
            "overrides_ignored": bool(p.get("overrides")),
            "created": m.get("created")}


def table(snap: dict | None) -> dict:
    """{like_for_like, average, snapshot} from a stored snapshot. Prices are
    USD per token; the panel shows them per million."""
    if not snap:
        return {"error": "no price snapshot yet: the worker enqueues "
                         f"{QUEUE} on the net lane at most daily "
                         "(python mcp/prices.py refresh fetches one now)"}
    if _cache.get("snap") is snap and _cache.get("table") is not None:
        return _cache["table"]
    by_id = {m.get("id"): m for m in (snap.get("response") or {}).get("data") or []
             if isinstance(m, dict)}
    lfl = by_id.get(LIKE_FOR_LIKE["id"])
    like = (dict(listing_price(lfl), why=LIKE_FOR_LIKE["why"]) if lfl else
            {"id": LIKE_FOR_LIKE["id"], "missing": True, "why": LIKE_FOR_LIKE["why"]})
    members, missing = [], []
    for mid in AVERAGE_SET:
        if mid in by_id:
            members.append(listing_price(by_id[mid]))
        else:
            missing.append(mid)

    def med(k):
        xs = [m[k] for m in members if m[k] is not None]
        return statistics.median(xs) if xs else None
    avg = {"definition": AVERAGE_DEFINITION, "members": members,
           "missing_members": missing, "n": len(members),
           "input": med("input"), "cache_read": med("cache_read"),
           "output": med("output"),
           "cache_read_listed": sum(1 for m in members if m["cache_read_listed"])}
    out = {"like_for_like": like, "average": avg,
           "snapshot": {"date": snap.get("date"), "fetched_at": snap.get("fetched_at"),
                        "source": snap.get("source") or SOURCE_URL,
                        "listings": snap.get("listings")}}
    if _cache.get("snap") is snap:
        _cache["table"] = out
    return out


def hosted_usd(tokens: dict, price: dict) -> dict | None:
    """What `tokens` (token_ledger kinds) would cost at `price` (USD per
    token). Uncached input = processed + unsplit (a prompt with no split is
    priced at the dearer rate); cached input = the slot-cache hits;
    output = completion, reasoning included."""
    if not price or price.get("missing") is True or price.get("input") is None \
            or price.get("output") is None:
        return None
    uncached = int(tokens.get("prompt_processed", 0)) + int(tokens.get("prompt_unsplit", 0))
    cached = int(tokens.get("prompt_cached", 0))
    out = int(tokens.get("completion", 0))
    parts = {"input": uncached * price["input"],
             "cached_input": cached * price["cache_read"],
             "output": out * price["output"]}
    return {"usd": sum(parts.values()), "parts": parts}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        print(json.dumps(refresh(), indent=1))
    else:
        t = table(load_latest())
        print(json.dumps({k: v for k, v in t.items()}, indent=1, default=str)[:4000])
