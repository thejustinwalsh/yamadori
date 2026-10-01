#!/usr/bin/env python
"""TypeSafe's Python SDK as a client of the Jev API. Run by mcp/test_jev_sdk.py
with the SDK's own venv (tools/typesafe-sdk/py/.venv), never the stack's
Python; imports nothing from this repository.

    python driver.py --base-url URL --plan PLAN.json

The key comes from TYPESAFE_API_KEY (the SDK's own variable) and is never
printed. One JSON line per step on stdout; each step catches its own
failure, so one broken step never hides the others. The test harness reads
the header `x-jev-test` (its scenario) and judges the lines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import typesafe_sdk as ts


def out(rec: dict) -> None:
    print(json.dumps(rec, default=str), flush=True)


def err(e: BaseException) -> dict:
    return {"error_class": type(e).__name__,
            "error_mro": [c.__name__ for c in type(e).__mro__],
            "message": str(e)[:600],
            "status": getattr(e, "status", None),
            "request_id": getattr(e, "request_id", None),
            "retry_after_ms": getattr(e, "retry_after_ms", None),
            "body": getattr(e, "body", None)}


def step(name: str, fn) -> None:
    t0 = time.time()
    try:
        rec = fn()
        rec.update(step=name, ok=True)
    except Exception as e:                                       # noqa: BLE001
        rec = dict(err(e), step=name, ok=False,
                   trace=traceback.format_exc().strip().splitlines()[-3:])
    rec["seconds"] = round(time.time() - t0, 3)
    out(rec)


def answers_of(r) -> dict:
    return {k: dict(a.model_dump(), _class=type(a).__name__)
            for k, a in r.answers.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    with open(a.plan, encoding="utf-8") as f:
        plan = json.load(f)
    live = bool(plan.get("live"))
    out({"step": "sdk", "ok": True, "sdk": "python",
         "version": getattr(ts, "__version__", None)})
    client = ts.TypeSafeClient(base_url=a.base_url)

    def models():
        r = client.models.list(extra_headers={"x-jev-test": "models"})
        return {"class": type(r).__name__,
                "models": [{"class": type(m).__name__, "name": m.name,
                            "description": m.description,
                            "release_date": m.release_date} for m in r.models],
                "request_id": r.request_id,
                "header_request_id":
                    r.raw_http_response.headers.get("x-typesafe-request-id")}
    step("models", models)

    for i, ex in enumerate(plan["examples"]):
        req = ex["request"]

        def one(req=req, i=i):
            r = client.system_one(req["state"], req["questions"],
                                  model=req.get("model"),
                                  extra_headers={"x-jev-test": f"example:{i}"})
            return {"i": i, "class": type(r).__name__, "model": r.model,
                    "usage": {"class": type(r.usage).__name__,
                              "input_tokens": r.usage.input_tokens,
                              "output_tokens": r.usage.output_tokens},
                    "answers": answers_of(r),
                    "split": {"nouls": sorted(r.nouls), "choices": sorted(r.choices),
                              "scores": sorted(r.scores)},
                    "request_id": r.request_id,
                    "header_request_id":
                        r.raw_http_response.headers.get("x-typesafe-request-id")}
        step(f"example:{i}", one)

    def invalid():
        # 11 levels: past Jev's 10, sent as the body (extra_body replaces the
        # questions), so no client-side check can stop it first.
        bad = {"q": {"type": "score", "instructions": "Level?",
                     "criteria": [f"level {n}" for n in range(11)]}}
        client.system_one("state", {"q": {"type": "noul", "instructions": "x"}},
                          extra_body={"questions": bad},
                          extra_headers={"x-jev-test": "invalid"},
                          retry=ts.RetryPolicy(max_retries=0))
        return {"error_class": None, "note": "no exception raised"}
    step("invalid_422", invalid)

    if not live:
        q = {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}

        def busy_no_retry():
            client.system_one("Help!", q, extra_headers={"x-jev-test": "busy_always"},
                              retry=ts.RetryPolicy(max_retries=0))
            return {"error_class": None, "note": "no exception raised"}
        step("busy_no_retry", busy_no_retry)

        def busy_retried():
            r = client.system_one("Help!", q, extra_headers={
                "x-jev-test": f"busy_once:py:{time.time()}"})
            return {"class": type(r).__name__, "request_id": r.request_id}
        step("busy_retried", busy_retried)

        def overload_retried():
            r = client.system_one("Help!", q, extra_headers={
                "x-jev-test": f"overload_once:py:{time.time()}"})
            return {"class": type(r).__name__, "request_id": r.request_id}
        step("overload_retried", overload_retried)
    client.close()
    out({"step": "done", "ok": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
