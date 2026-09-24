#!/usr/bin/env python
"""/dash/api/tokens: tokens, GPU electricity and the hosted-API comparison,
over ALL TIME, the LAST 30 DAYS and THIS WEEK.

Every number here is read, not fetched: the token ledger (mcp/token_ledger.py,
sqlite), the power ledger (mcp/power.py, in memory in the proxy process) and
the newest stored price snapshot (mcp/prices.py). Cheap enough to poll every
few seconds; that is what makes the counter live.

THE FORMULAS, per window

  tokens          sums of the token ledger's daily rows with day >= start
  hosted cost     (prompt_processed + prompt_unsplit) x input price
                  + prompt_cached x cache-read price
                  + completion x output price
                  over PRICED_ROLES (main, second brain, side calls,
                  internal); warm prefill excluded
  electricity     power ledger cents over the same days, counted from the
                  token ledger's first day (whole days: the first day's
                  electricity starts at midnight, its tokens at first_at,
                  which understates the saving on that day)
  saving          hosted cost - electricity, for each comparison

An ESTIMATE, and labelled one: it excludes hardware purchase and
depreciation, and everything the power ledger does not measure.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import power  # noqa: E402
import prices  # noqa: E402
import token_ledger  # noqa: E402

PATH = "/dash/api/tokens"

BASIS = [
    "ESTIMATE, NOT A BILL",
    "HOSTED COST = UNCACHED INPUT x INPUT PRICE + CACHED INPUT x CACHE-READ PRICE "
    "+ OUTPUT x OUTPUT PRICE, OVER MAIN, SECOND BRAIN, SIDE CALLS AND INTERNAL "
    "GENERATION; WARM PREFILL EXCLUDED",
    "ASSUMES THE HOSTED API WOULD HAVE CACHED EXACTLY THE PROMPT TOKENS OUR SLOTS "
    "REUSED; OUTPUT INCLUDES REASONING",
    "SAVED = HOSTED COST - OUR GPU ELECTRICITY OVER THE SAME DAYS. EXCLUDES "
    "HARDWARE PURCHASE AND DEPRECIATION, AND CPU/RAM/PSU POWER (NOT METERED)",
    "BASE LISTED PRICES; LONG-CONTEXT PRICE TIERS NOT APPLIED",
]


def electricity(days: dict, start: str | None, since: str | None) -> dict:
    """Power-ledger cents and kWh over the days >= max(start, since). `since`
    is the token ledger's first day: electricity before any token was
    recorded is not set against tokens."""
    out = {"cents": None, "kwh": None, "days_measured": 0,
           "measured_hours": 0.0, "gap_hours": 0.0, "from": None}
    if since is None:
        out["why"] = "no tokens recorded yet"
        return out
    lo = max(x for x in (start, since) if x)
    out["from"] = lo
    rows = [power.day_summary(d, v) for d, v in sorted(days.items()) if d >= lo]
    measured = [r for r in rows if r["cents"] is not None]
    out["measured_hours"] = round(sum(r["measured_seconds"] for r in rows) / 3600, 2)
    out["gap_hours"] = round(sum(r["gap_seconds"] for r in rows) / 3600, 2)
    out["days_measured"] = len(measured)
    if measured:
        out["cents"] = round(sum(r["cents"] for r in measured), 4)
        out["kwh"] = round(sum(r["kwh"] or 0 for r in measured), 6)
    else:
        out["why"] = "the power ledger measured nothing on these days"
    return out


def window(w: dict, first_day: str | None, days: dict, table: dict,
           ledger_path: str | None = None) -> dict:
    t = token_ledger.totals(w["start"], path=ledger_path)
    elec = electricity(days, w["start"], first_day)
    usd = None if elec["cents"] is None else elec["cents"] / 100.0
    hosted, saved = {}, {}
    for k in ("like_for_like", "average"):
        h = prices.hosted_usd(t["priced"], table.get(k) or {}) if "error" not in table else None
        hosted[k] = h
        saved[k] = (None if h is None or usd is None else h["usd"] - usd)
    return dict(w, tokens=t["total"], priced=t["priced"], by_role=t["by_role"],
                electricity=elec, hosted=hosted, saved=saved)


def payload(now: float | None = None, ledger_path: str | None = None,
            power_days: dict | None = None, snapshot=None,
            corpus_db: str | None = None) -> dict:
    now = time.time() if now is None else now
    today = _dt.datetime.fromtimestamp(now).date()
    h = token_ledger.history(ledger_path)
    days = power.ledger_days() if power_days is None else power_days
    snap = prices.load_latest() if snapshot is None else snapshot
    table = prices.table(snap)
    snap_age = None
    if isinstance(table.get("snapshot"), dict) and table["snapshot"].get("fetched_at"):
        snap_age = round((now - float(table["snapshot"]["fetched_at"])) / 86400, 2)
    return {
        "at": now,
        "today": today.isoformat(),
        "timezone": token_ledger.TIMEZONE_LABEL,
        "week_start": token_ledger.WEEK_START_LABEL,
        "recording": token_ledger.status(),
        "history": dict(h, gap=token_ledger.gap(h["first_at"], corpus_db),
                        backfill="none: no earlier record carries token counts"),
        "roles": list(token_ledger.ROLES),
        "priced_roles": list(token_ledger.PRICED_ROLES),
        "windows": [window(w, h["first_day"], days, table, ledger_path)
                    for w in token_ledger.windows(today)],
        "prices": dict(table, age_days=snap_age),
        "basis": list(BASIS),
        "estimate": True,
    }


def handle_get(path: str):
    """(status, content_type, body_bytes) or None if this is not ours."""
    if path.rstrip("/") != PATH:
        return None
    try:
        body = payload()
    except Exception as e:                                       # noqa: BLE001
        return 500, "application/json", json.dumps(
            {"error": f"dash_tokens.payload() raised {type(e).__name__}: {e}"}).encode()
    return 200, "application/json", json.dumps(body).encode()


if __name__ == "__main__":
    print(json.dumps(payload(), indent=1, default=str)[:6000])
