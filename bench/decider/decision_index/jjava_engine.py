"""Decision Index engine for OUR /jev API (jjava), a thin subclass of the harness's own `http` engine.

The harness's HttpSystemOne posts {model, state, questions} to <base_url>/v1/systemone and records the body as is.
What it does not do, and what this subclass adds (each recorded on the result row under raw["x_adapter"]):

1. 429/529 + Retry-After (our queue / a model off the card): the harness raises (an `error`, retried only on a
   resume, and five errors before a first success stop the run). This waits Retry-After and sends the SAME payload
   again, at most MAX_WAITS times; the waited seconds are recorded and are INCLUDED in the row's wall time, so the
   row's `x_adapter.waited_s` is what to subtract for service time. A valid answer is never retried (rule: no
   correctness-based retry).
2. A 422 whose entries are `too_long` (our Choice 255-option limit, Jev's 32k state+longest question, 64k request)
   is a declared capacity limit -> `Unsupported` (the harness's marker list is Jev's wording; ours differs).
   Every other 422 stays an error.

Payload, model name, and the response are untouched. Use:
  python -m decision_index run --engine jjava_engine:JjavaEngine --option base_url=http://127.0.0.1:1234/jev \
      --option model=jjava-latest ...   (PYTHONPATH must include this directory; key in DECISION_INDEX_API_KEY)
"""
import time

from decision_index.engines.base import Unsupported
from decision_index.engines.http import HttpSystemOne

MAX_WAITS = 8
MAX_WAIT_S = 120


class JjavaEngine(HttpSystemOne):
    name = "jjava"

    def __init__(self, **options):
        super().__init__(**options)
        self.provenance["adapter"] = "bench/decider/decision_index/jjava_engine.py: waits Retry-After on 429/529 (same payload), 422 too_long -> Unsupported"

    def __call__(self, state, questions):
        waited = 0.0
        waits = []
        for attempt in range(MAX_WAITS + 1):
            r = self.client.post("/v1/systemone", json={"model": self.model, "state": state, "questions": questions, **self.extra})
            if r.status_code in (429, 529) and attempt < MAX_WAITS:
                ra = min(MAX_WAIT_S, max(1.0, float(r.headers.get("retry-after", "1") or 1)))
                waits.append({"status": r.status_code, "retry_after_s": ra})
                time.sleep(ra)
                waited += ra
                continue
            break
        if r.status_code == 422:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = None
            if isinstance(detail, list) and detail and all(isinstance(d, dict) and d.get("type") == "too_long" for d in detail):
                raise Unsupported(r.text)
        r.raise_for_status()
        raw = r.json()
        raw["x_adapter"] = {"waited_s": waited, "waits": waits, "request_id": r.headers.get("x-typesafe-request-id")}
        return {k: v for k, v in raw.items() if k != "evaluation_trace"}, raw
