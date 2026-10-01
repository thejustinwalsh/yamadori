#!/usr/bin/env python
"""The Jev API (mcp/jev_api.py; POST /jev/v1/systemone, GET /jev/v1/models,
POST /v1/systemone). No GPU, no network: the model server is a fake behind
decider_bonsai's two doors (_post_default, _upstream), so every request runs
the REAL readout (two orders, averaging, confidence) through server.app.

    python mcp/test_jev_api.py      -> "N/M checks passed"

WHAT THIS GATES (operator, 2026-09-29: "the jjava api exact public api
endpoints that match Jev exposed through our proxy"):

  1. THE DOCS' EXAMPLES: every distinct API response example in
     docs.typesafe.ai (mcp/fixtures/jev_doc_examples.json, 13 of them) is
     reproduced field for field -- the fake reads the example's
     probabilities, and the answer carries exactly Jev's fields with Jev's
     values (confidence within the rounding of the two decimals the docs
     print; `model` is the versioned id that answered, as Jev's is).
  2. VALIDATION: 422 with one {loc, msg, type} entry per offending field,
     read the way TypeSafe's JS SDK reads it (errors.ts extractMessage,
     ported below).
  3. AUTH 401, the lane 429 and the card / model server 529, each with
     Retry-After where Jev's SDKs honour it; x-typesafe-request-id always.
  4. THE MODELS LIST in TypeSafe's ListModelsResponse shape, availability
     and priors in x_yamadori; our OpenAI GET /v1/models unchanged.
  5. THE PATHS: /jev base, the root alias, 404 / 405 in Jev's body.
  6. MORE THAN 26 OPTIONS in two stages (60 and 255), never refused; the
     composite distribution; SCORE LEVELS 2..10.
  7. USAGE counted from the reads the fake saw; the lane's fit; the state
     limit; the corpus row (kind jev_call, traffic class).
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
import sys
import traceback
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_jev_api_")
os.environ["YAMADORI_ACCOUNTS_DIR"] = os.path.join(_TMP, "accounts")
os.environ["YAMADORI_DECIDER_MODELS_DIR"] = os.path.join(_TMP, "decider_models")
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ["YAMADORI_SLOTS"] = "3"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
os.environ.pop("YAMADORI_TIER_MODELS", None)
os.environ.pop("YAMADORI_MAX_MODEL", None)
os.environ.pop("YAMADORI_DECIDER", None)

import accounts  # noqa: E402
import budget  # noqa: E402
import decider_bonsai as D  # noqa: E402
import jev_api as J  # noqa: E402

budget._POOL = 131072          # never /props
_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail="") -> bool:
    _results.append((bool(ok), name, str(detail)[:600]))
    return bool(ok)


# ------------------------------------------------------------ the fake -----
_Q_RE = re.compile(r"^QUESTION: (.*)\n\nOPTIONS:\n(.*)\n\nAnswer with the "
                   r"letter of one option\.$", re.S)


class FakeModel:
    """The model server behind decider_bonsai's doors. `weights(question,
    option_text) -> w` sets each read's distribution (renormalised by the
    decider); every read is logged with the prompt_n / cache_n it reports:
    the slot caches the state (the first two messages) after a read."""

    def __init__(self):
        self.reset()

    def reset(self, weights=None):
        self.weights = weights or (lambda q, t: 1.0)
        self.posts: list[dict] = []
        self.cached_state: str | None = None
        self.tokenize_calls = 0
        self.fail_post: Exception | None = None
        self.fail_tokenize = False

    @staticmethod
    def ntok(s: str) -> int:
        return max(1, len(s) // 4)

    def upstream(self, path, payload=None, timeout=30):
        if path == "/tokenize":
            self.tokenize_calls += 1
            if self.fail_tokenize:
                raise urllib.error.URLError("connection refused")
            s = payload["content"]
            if re.fullmatch(r" ?[A-Z]", s):
                return {"tokens": [1000 + 2 * ord(s[-1]) + (1 if s[0] == " "
                                                            else 0)]}
            if s.strip().lower() in ("yes", "no"):
                return {"tokens": [9000 + len(s)]}
            return {"tokens": list(range(self.ntok(s)))}
        if path == "/slots":
            return [{"id": 2, "n_prompt_tokens": 0}]
        raise AssertionError(path)

    def post(self, body, timeout):
        if self.fail_post is not None:
            raise self.fail_post
        msgs = body["messages"]
        state_part = msgs[0]["content"] + msgs[1]["content"]
        m = _Q_RE.match(msgs[2]["content"])
        assert m, msgs[2]["content"][:200]
        question = m.group(1)
        opts = []
        for line in m.group(2).split("\n"):
            lab, text = line.split(". ", 1)
            opts.append((lab, text))
        ws = [max(float(self.weights(question, t)), 0.0) for _, t in opts]
        z = sum(ws) or 1.0
        probs = [(lab, w / z) for (lab, _), w in zip(opts, ws)]
        top = sorted(({"id": 1000 + 2 * ord(lab) + 1, "token": " " + lab,
                       "logprob": math.log(max(p, 1e-12))}
                      for lab, p in probs), key=lambda x: -x["logprob"])
        k = body["top_logprobs"]
        total = sum(self.ntok(x["content"]) for x in msgs)
        state_tokens = self.ntok(msgs[0]["content"]) + self.ntok(msgs[1]["content"])
        cache = state_tokens if self.cached_state == state_part else 0
        self.cached_state = state_part
        rec = {"model": body["model"], "question": question, "k": k,
               "prompt_n": total - cache, "cache_n": cache,
               "state": msgs[1]["content"], "id_slot": body.get("id_slot")}
        self.posts.append(rec)
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top[:k]}]}}],
                "usage": {"prompt_tokens": total},
                "timings": {"prompt_n": total - cache, "cache_n": cache,
                            "prompt_ms": 1.0}}


FAKE = FakeModel()
D._post_default = FAKE.post
D._upstream = FAKE.upstream

# The release door (decider_bonsai.release -> model.release_slot, since the
# offline suite never enables the proxy's own releases): a fake that records
# the slot, never the model server.
import model as _M  # noqa: E402
RELEASED: list[int] = []


def _fake_release_slot(slot, model=None, timeout=None):
    RELEASED.append(slot)
    return {"ok": True, "cells_before": 0, "ms": 0, "method": "fake"}


_M.release_slot = _fake_release_slot

from starlette.testclient import TestClient  # noqa: E402
import server  # noqa: E402

KEY = accounts.create("jev-tests")
TEST_KEY = accounts.create("live-test-jev")
CLIENT = TestClient(server.app)


def post(body, path="/jev/v1/systemone", key=KEY, raw=None, headers=None):
    h = dict(headers or {})
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    if raw is not None:
        r = CLIENT.post(path, content=raw,
                        headers=dict(h, **{"Content-Type": "application/json"}))
    else:
        r = CLIENT.post(path, json=body, headers=h)
    try:
        data = r.json()
    except ValueError:
        data = None
    return r.status_code, r.headers, data


def js_extract_message(body):
    """TypeSafe JS SDK v0.6.0 src/errors.ts extractMessage +
    describeValidationErrors, line for line (read 2026-09-29)."""
    if isinstance(body, str):
        return body or None
    if not isinstance(body, dict):
        return None
    error, message, detail = body.get("error"), body.get("message"), body.get("detail")
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(message, str):
        return message
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    if isinstance(detail, list):
        parts = []
        for e in detail:
            if not isinstance(e, dict) or not isinstance(e.get("msg"), str):
                continue
            loc = (".".join(str(x) for x in e["loc"] if x != "body")
                   if isinstance(e.get("loc"), list) else "")
            parts.append(f"{loc}: {e['msg']}" if loc else e["msg"])
        return "; ".join(parts) if parts else None
    return None


def weights_for(example: dict):
    """The fake's weights that make the decider read the example's
    probabilities, per question text and printed option."""
    table: dict[str, dict[str, float]] = {}
    for qid, q in example["request"]["questions"].items():
        a = example["response"]["answers"][qid]
        qt = J.as_text(q["instructions"])
        if q["type"] == "noul":
            crit = q.get("criteria") or {}
            yes = "yes" + (f": {J.as_text(crit['true'], one_line=True)}"
                           if crit.get("true") else "")
            no = "no" + (f": {J.as_text(crit['false'], one_line=True)}"
                         if crit.get("false") else "")
            table[qt] = {yes: a["noul"], no: 1.0 - a["noul"]}
        elif q["type"] == "choice":
            table[qt] = {J.option_text(k, d): a["probabilities"][k]
                         for k, d in q["criteria"].items()}
        else:
            table[qt] = {J.as_text(lvl, one_line=True): a["probabilities"][str(i)]
                         for i, lvl in enumerate(q["criteria"])}

    def w(question, text):
        return table[question][text]
    return w


# ----------------------------------------------------------------- tests ----
def test_doc_examples():
    path = os.path.join(HERE, "fixtures", "jev_doc_examples.json")
    with open(path, encoding="utf-8") as f:
        ex = json.load(f)["examples"]
    check(len(ex) == 13, "the fixture holds the docs' 13 distinct API "
          "response examples", len(ex))
    for i, e in enumerate(ex):
        tag = f"[doc {i + 1} {e['source'].rsplit('/', 1)[-1]}]"
        FAKE.reset(weights_for(e))
        st, h, d = post(e["request"])
        if not check(st == 200, f"{tag} 200", (st, d)):
            continue
        doc = e["response"]
        check(set(doc) <= set(d) and set(d) - set(doc) == {"x_yamadori"},
              f"{tag} top level: Jev's fields plus x_yamadori only",
              sorted(d))
        check(d["model"] == "jjava-bonsai",
              f"{tag} model: the versioned id that answered (Jev's is "
              f"{doc['model']})", d["model"])
        check(set(d["usage"]) == {"input_tokens", "output_tokens"} and all(
            isinstance(v, int) and not isinstance(v, bool)
            for v in d["usage"].values()), f"{tag} usage: two integers",
              d["usage"])
        check(list(d["answers"]) == list(e["request"]["questions"]),
              f"{tag} answers keyed by the request's ids, in order",
              list(d["answers"]))
        for qid, want in doc["answers"].items():
            got = d["answers"].get(qid) or {}
            ok = set(got) == set(want) and got.get("type") == want["type"]
            why = []
            if want["type"] == "noul":
                ok &= abs(got["noul"] - want["noul"]) <= 1e-6
            else:
                n = len(want["probabilities"])
                ok &= set(got["probabilities"]) == set(want["probabilities"])
                ok &= all(abs(got["probabilities"][k] - v) <= 1e-6
                          for k, v in want["probabilities"].items())
                ok &= abs(sum(got["probabilities"].values()) - 1.0) <= 1e-5
                tol = n / (n - 1) * 0.005 + 0.005 if n > 1 else 1e-6
                if abs(got["confidence"] - want["confidence"]) > tol:
                    ok = False
                    why.append(f"confidence {got['confidence']} vs "
                               f"{want['confidence']} (tol {tol:.4f})")
                ok &= abs(got["confidence"]
                          - D.confidence(got["probabilities"])) <= 1e-5
                if want["type"] == "choice":
                    ok &= got["choice"] == want["choice"]
                else:
                    tol_s = 0.005 * sum(range(n)) + 0.005
                    ok &= abs(got["score"] - want["score"]) <= tol_s
                    ok &= got["legend"] == want["legend"]
            check(ok, f"{tag} {qid}: exactly Jev's fields and values",
                  f"{why} got {json.dumps(got)} want {json.dumps(want)}")
        check(bool(h.get(J.REQUEST_ID_HEADER, "").startswith("jjava-")),
              f"{tag} x-typesafe-request-id", dict(h))


def _one_noul(model="jev-latest", **kw):
    return dict({"state": "Help! My payouts have been failing for 3 days.",
                 "model": model,
                 "questions": {"is_urgent": {
                     "type": "noul",
                     "instructions": "Does this convey urgency?"}}}, **kw)


def test_answers_carry_only_jevs_fields():
    FAKE.reset(lambda q, t: 3.0 if t.startswith(("yes", "billing", "Very"))
               else 1.0)
    st, _, d = post({"state": {"message": "Refund me", "order": "A-1"},
                     "model": "jjava-latest",
                     "questions": {
                         "n": {"type": "noul", "instructions": "Urgent?",
                               "criteria": {"true": {"what": "time-sensitive"},
                                            "false": "no urgency"}},
                         "c": {"type": "choice", "instructions": "Team?",
                               "criteria": {"billing": "Payments",
                                            "tech": None}},
                         "s": {"type": "score", "instructions": "Anger?",
                               "criteria": ["Calm", {"what": "Frustrated"},
                                            ["Very", "angry"]]}}})
    if not check(st == 200, "mixed request 200", (st, d)):
        return
    a = d["answers"]
    check(list(a["n"]) == ["type", "noul"], "noul: {type, noul} only",
          list(a["n"]))
    check(list(a["c"]) == ["type", "choice", "probabilities", "confidence"],
          "choice: {type, choice, probabilities, confidence} only",
          list(a["c"]))
    check(list(a["s"]) == ["type", "score", "legend", "probabilities",
                           "confidence"],
          "score: {type, score, legend, probabilities, confidence} only",
          list(a["s"]))
    check(a["s"]["legend"] == {"0": "Calm", "1": {"what": "Frustrated"},
                               "2": ["Very", "angry"]},
          "score legend: each level number -> the criteria entry AS GIVEN",
          a["s"]["legend"])
    check(set(a["s"]["probabilities"]) == {"0", "1", "2"},
          "score probabilities keyed by level number as a string")
    check(all(isinstance(v, float) for v in a["c"]["probabilities"].values())
          and isinstance(a["c"]["confidence"], float)
          and isinstance(a["s"]["score"], float)
          and isinstance(a["n"]["noul"], float),
          "every number a float (the Python SDK's models are strict)")
    states = {p["state"] for p in FAKE.posts}
    check(any('"order": "A-1"' in s for s in states),
          "an object state is printed as JSON", list(states)[:1])
    qs = {p["question"] for p in FAKE.posts}
    check(qs == {"Urgent?", "Team?", "Anger?"},
          "each question's instructions reach the model as its question",
          qs)
    x = d["x_yamadori"]
    check(set(x["answers"]) == {"n", "c", "s"}
          and "orders" in x["answers"]["c"]
          and x["answers"]["c"]["readout"] == D.READOUT_VERSION,
          "jjava's diagnostics ride in x_yamadori, per question")
    check(x.get("parallel") is False and "one after another" in x["note"],
          "x_yamadori says the questions ran sequentially (Jev: parallel)")
    printed = [p for p in FAKE.posts if p["question"] == "Team?"]
    check(len(printed) == 2, "a question is read in two orders",
          len(printed))


def test_usage_is_counted_from_the_reads():
    FAKE.reset(lambda q, t: 2.0 if t.startswith("yes") else 1.0)
    body = _one_noul()
    body["questions"]["second"] = {"type": "choice", "instructions": "Team?",
                                   "criteria": {"a": None, "b": None,
                                                "c": None}}
    st, _, d = post(body)
    if not check(st == 200, "usage request 200", (st, d)):
        return
    exp_in = FAKE.posts[0]["cache_n"] + sum(p["prompt_n"] for p in FAKE.posts)
    check(d["usage"]["input_tokens"] == exp_in,
          "input_tokens = the first read's cached prefix + every read's "
          "processed tokens", (d["usage"], exp_in))
    check(d["usage"]["output_tokens"] == len(FAKE.posts) == 4,
          "output_tokens = the reads (one generated token each; 2 "
          "questions x 2 orders)", (d["usage"], len(FAKE.posts)))
    check(FAKE.posts[1]["cache_n"] > 0 and FAKE.posts[-1]["cache_n"] > 0,
          "the state is placed once: later reads reuse it (the fake's cache)",
          [p["cache_n"] for p in FAKE.posts])
    check(all(p["id_slot"] == 2 for p in FAKE.posts),
          "every read on the lane (the child slot, 2 of 3)",
          {p["id_slot"] for p in FAKE.posts})
    lane = d["x_yamadori"]["lane"]
    check(lane["slot"] == 2 and lane["cells"] == budget.LANE_TOKENS
          and lane["fits"] is True, "x_yamadori.lane: slot, cells, fits", lane)
    import slots as _slots
    if _slots.lane_kept():
        check((lane.get("release") or {}).get("skipped", "").startswith(
            "the lane is kept"), "the lane is kept after the call (layout v2)",
              lane.get("release"))
    else:
        check(RELEASED[-1:] == [2] and (lane.get("release") or {}).get("slot") == 2,
              "the lane is cleared once the call is done (slots LANE_KEEP "
              "False: operator 2026-09-29, 'We clear jjava lane too after it "
              "is done')", (RELEASED[-3:], lane.get("release")))


def test_models_and_aliases():
    for alias in J.JEV_ALIASES:
        FAKE.reset()
        st, _, d = post(_one_noul(model=alias))
        check(st == 200 and d["model"] == "jjava-bonsai"
              and d["x_yamadori"]["jjava"].get("alias_of") == J.LATEST
              and d["x_yamadori"]["jjava"]["requested"] == alias,
              f"{alias}: served as jjava-latest, the alias recorded",
              (st, (d or {}).get("x_yamadori", {}).get("jjava")))
    FAKE.reset()
    st, _, d = post(_one_noul(model="jjava-bonsai"))
    check(st == 200 and d["model"] == "jjava-bonsai",
          "jjava-bonsai while bonsai holds the card: 200", (st, d))
    st, h, d = post(_one_noul(model="jjava-flash-next"))
    check(st == 422 and d["detail"][0]["loc"] == ["body", "model"],
          "jjava-flash-next on a stack that does not configure it: 422 on "
          "model", (st, d))
    st, h, d = post(_one_noul(model="gpt-4o"))
    check(st == 422 and "jjava-latest" in d["detail"][0]["msg"],
          "an unknown model: 422 naming the accepted ids", (st, d))
    # the tier table on, flash-next on the card
    real = (J.configured, J.card)
    J.configured = lambda: ["bonsai", "mirai-s", "flash-next"]
    J.card = lambda: {"model": "flash-next", "why": "a side call names no "
                      "conversation: served by flash-next, which is on the "
                      "card", "holder": "flash-next", "enabled": True}
    try:
        FAKE.reset()
        st, h, d = post(_one_noul(model="jjava-bonsai"))
        check(st == 529 and h.get("retry-after", "").isdigit()
              and isinstance(d.get("detail"), str) and "never swaps" in
              d["detail"] and not FAKE.posts,
              "a named model off the card: 529 overloaded with Retry-After, "
              "nothing read, never a swap", (st, dict(h), d))
        st, h, d = post(_one_noul(model="jjava-latest"))
        check(st == 200 and d["model"] == "jjava-flash-next"
              and {p["model"] for p in FAKE.posts} == {"flash-next"},
              "jjava-latest reads whichever model holds the card",
              (st, (d or {}).get("model"), {p["model"] for p in FAKE.posts}))
        FAKE.reset()
        st, h, d = post(_one_noul(model="jjava-flash-next"))
        check(st == 200 and d["model"] == "jjava-flash-next",
              "jjava-flash-next while it holds the card: 200", (st, d))
        r = CLIENT.get("/jev/v1/models",
                       headers={"Authorization": f"Bearer {KEY}"})
        m = r.json()
        names = [x["name"] for x in m["models"]]
        check(names == ["jjava-latest", "jjava-bonsai", "jjava-mirai-s",
                        "jjava-flash-next"],
              "models list: jjava-latest and each configured model", names)
        xm = m["x_yamadori"]["models"]
        check(xm["jjava-flash-next"]["available"] is True
              and xm["jjava-bonsai"]["available"] is False
              and xm["jjava-latest"]["model"] == "flash-next",
              "availability per model (on the card or not)", xm)
    finally:
        J.configured, J.card = real


def test_models_list_shape():
    r = CLIENT.get("/jev/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    check(r.status_code == 200, "GET /jev/v1/models 200", r.status_code)
    m = r.json()
    check(set(m) == {"models", "x_yamadori"} and isinstance(m["models"], list),
          "ListModelsResponse: {models: [...]} (+ x_yamadori)", sorted(m))
    check(all(set(x) == {"name", "description", "release_date"}
              and all(isinstance(v, str) for v in x.values())
              and re.fullmatch(r"\d{4}-\d{2}-\d{2}", x["release_date"])
              for x in m["models"]),
          "every entry exactly ModelMetadata / ModelCard: name, description, "
          "release_date (YYYY-MM-DD strings)", m["models"])
    check([x["name"] for x in m["models"]] == ["jjava-latest", "jjava-bonsai"],
          "one-model stack: jjava-latest and jjava-bonsai",
          [x["name"] for x in m["models"]])
    xb = m["x_yamadori"]["models"]["jjava-bonsai"]
    check(xb["available"] is True and xb["priors_measured"] ==
          {"letter_prior": False, "label_bias": False}
          and xb["tie_band"]["measured"] is True,
          "x_yamadori: availability and whether the model's priors are "
          "measured (bench/decider/results/models/<model>.json)", xb)
    check(m["x_yamadori"]["aliases"] == {a: "jjava-latest"
                                         for a in J.JEV_ALIASES},
          "Jev's ids listed as aliases in x_yamadori, not as models")
    os.makedirs(D.PROFILES_DIR, exist_ok=True)
    with open(os.path.join(D.PROFILES_DIR, "bonsai.json"), "w",
              encoding="utf-8") as f:
        json.dump({"version": D.PROFILE_VERSION, "model": "bonsai",
                   "letter_prior": {"script": "bench/decider/measure_model.py",
                                    "date": "2026-09-29", "n": 50}}, f)
    D._PROFILE_CACHE.clear()
    m = CLIENT.get("/jev/v1/models",
                   headers={"Authorization": f"Bearer {KEY}"}).json()
    check(m["x_yamadori"]["models"]["jjava-bonsai"]["priors_measured"]
          ["letter_prior"] is True,
          "a measured record flips priors_measured", m["x_yamadori"])
    os.remove(os.path.join(D.PROFILES_DIR, "bonsai.json"))
    D._PROFILE_CACHE.clear()
    r = CLIENT.get("/jev/v1/models")
    check(r.status_code in (200, 401), "GET /jev/v1/models without a key "
          "(single-user mode admits; a registry refuses)", r.status_code)
    r = CLIENT.get("/v1/models")
    check(r.status_code == 200 and r.json().get("object") == "list"
          and "data" in r.json(),
          "our GET /v1/models is still OpenAI's list", r.json())


def test_paths():
    FAKE.reset()
    st, h, d = post(_one_noul(), path="/v1/systemone")
    check(st == 200 and d["answers"]["is_urgent"]["type"] == "noul",
          "POST /v1/systemone at the root: the same API", (st, d))
    r = CLIENT.get("/jev/v1/systemone", headers={"Authorization": f"Bearer {KEY}"})
    check(r.status_code == 405 and r.json() == {"detail": "Method Not Allowed"},
          "GET /jev/v1/systemone: 405 in Jev's body", (r.status_code, r.text))
    r = CLIENT.post("/jev/v1/nope", json={})
    check(r.status_code == 404 and r.json() == {"detail": "Not Found"}
          and r.headers.get(J.REQUEST_ID_HEADER),
          "an unknown /jev path: 404 {detail} with a request id",
          (r.status_code, r.text))
    r = CLIENT.get("/")
    eps = r.json().get("endpoints") or []
    check("/jev/v1/systemone" in eps and "/v1/systemone" in eps
          and "/jev/v1/models" in eps, "the service descriptor lists the "
          "Jev routes", eps)


def test_auth():
    st, h, d = post(_one_noul(), key=None)
    check(st == 401 and isinstance(d.get("detail"), str)
          and h.get("www-authenticate") == "Bearer"
          and h.get(J.REQUEST_ID_HEADER),
          "no key: 401 {detail}, WWW-Authenticate, request id", (st, d))
    st, h, d = post(_one_noul(), key="ym-not-a-key")
    check(st == 401 and "unrecognised" in d["detail"],
          "a wrong key: 401", (st, d))
    check(js_extract_message(d) == d["detail"],
          "the JS SDK reads the 401's message", d)
    r = CLIENT.get("/jev/v1/models",
                   headers={"Authorization": "Bearer ym-not-a-key"})
    check(r.status_code == 401, "models list: a wrong key is 401",
          r.status_code)


def _msg(d) -> str:
    return js_extract_message(d) or ""


def test_validation():
    FAKE.reset()
    st, _, d = post({})
    locs = [x["loc"] for x in d.get("detail") or []]
    check(st == 422 and locs == [["body", "state"], ["body", "model"],
                                 ["body", "questions"]]
          and all(x["type"] == "missing" for x in d["detail"]),
          "an empty body: 422 with every missing field", (st, d))
    check(_msg(d) == "state: Field required; model: Field required; "
          "questions: Field required",
          "the JS SDK's message names each field", _msg(d))
    st, _, d = post(None, raw=b'{"state": "x", "model": ')
    check(st == 422 and d["detail"][0]["type"] == "json_invalid"
          and d["detail"][0]["loc"][0] == "body",
          "malformed JSON: 422 json_invalid", (st, d))
    cases = [
        ("empty questions", {"questions": {}}, ["body", "questions"],
         "too_short"),
        ("a number as state", {"state": 7}, ["body", "state"], "state_type"),
        ("an unknown type", {"questions": {"q": {"type": "rank",
                                                 "instructions": "x"}}},
         ["body", "questions", "q", "type"], "union_tag_invalid"),
        ("no instructions", {"questions": {"q": {"type": "noul"}}},
         ["body", "questions", "q", "instructions"], "missing"),
        ("a choice without criteria", {"questions": {"q": {
            "type": "choice", "instructions": "x"}}},
         ["body", "questions", "q", "criteria"], "missing"),
        ("a choice with no options", {"questions": {"q": {
            "type": "choice", "instructions": "x", "criteria": {}}}},
         ["body", "questions", "q", "criteria"], "too_short"),
        ("a choice of 256 options", {"questions": {"q": {
            "type": "choice", "instructions": "x",
            "criteria": {f"o{i}": None for i in range(256)}}}},
         ["body", "questions", "q", "criteria"], "too_long"),
        ("a number as an option's description", {"questions": {"q": {
            "type": "choice", "instructions": "x",
            "criteria": {"a": 1, "b": None}}}},
         ["body", "questions", "q", "criteria", "a"], "criteria_type"),
        ("a noul criteria key that is not true/false", {"questions": {"q": {
            "type": "noul", "instructions": "x",
            "criteria": {"true": "y", "maybe": "z"}}}},
         ["body", "questions", "q", "criteria", "maybe"], "extra_forbidden"),
        ("score criteria as a map", {"questions": {"q": {
            "type": "score", "instructions": "x", "criteria": {"0": "a"}}}},
         ["body", "questions", "q", "criteria"], "list_type"),
        ("a model that is not a string", {"model": 3}, ["body", "model"],
         "string_type"),
    ]
    for name, patch, loc, typ in cases:
        body = _one_noul()
        body.update(patch)
        st, h, d = post(body)
        det = (d or {}).get("detail") or []
        check(st == 422 and any(x["loc"] == loc and x["type"] == typ
                                for x in det)
              and all(set(x) == {"loc", "msg", "type"} for x in det),
              f"422 {name}: {'.'.join(map(str, loc[1:]))} ({typ})", (st, d))
    check(not FAKE.posts and FAKE.tokenize_calls == 0,
          "nothing is sent to the model for a malformed request",
          (len(FAKE.posts), FAKE.tokenize_calls))
    body = _one_noul()
    body["questions"] = {"a": {"type": "score", "instructions": "x",
                               "criteria": ["only"]},
                         "b": {"type": "choice", "instructions": "y"}}
    st, _, d = post(body)
    check(st == 422 and _msg(d) == (
        "questions.a.criteria: A Score should have at least 2 levels; got 1; "
        "questions.b.criteria: Field required"),
          "every offending field in one 422, read as the SDK reads it",
          _msg(d))


def test_score_levels():
    for n, ok in ((1, False), (2, True), (10, True), (11, False)):
        FAKE.reset()
        body = _one_noul()
        body["questions"] = {"s": {"type": "score", "instructions": "Level?",
                                   "criteria": [f"level {i}"
                                                for i in range(n)]}}
        st, _, d = post(body)
        if ok:
            check(st == 200 and len(d["answers"]["s"]["legend"]) == n
                  and abs(sum(d["answers"]["s"]["probabilities"].values())
                          - 1) < 1e-5,
                  f"a Score of {n} levels is answered", (st, d))
        else:
            check(st == 422 and d["detail"][0]["type"] in ("too_short",
                                                           "too_long"),
                  f"a Score of {n} levels is 422 (Jev: 2 to 10)", (st, d))


def test_rounds_past_26():
    FAKE.reset(lambda q, t: 10.0 if t == "opt47" else 1.0)
    body = _one_noul()
    body["questions"] = {"pick": {"type": "choice", "instructions": "Which?",
                                  "criteria": {f"opt{i:02d}": None
                                               for i in range(60)}}}
    st, _, d = post(body)
    if not check(st == 200, "60 options: 200, never refused", (st, d)):
        return
    a = d["answers"]["pick"]
    x = d["x_yamadori"]
    check(a["choice"] == "opt47" and len(a["probabilities"]) == 60
          and abs(sum(a["probabilities"].values()) - 1) < 1e-4,
          "60 options: the choice, a distribution over all 60 summing to 1",
          (a["choice"], sum(a["probabilities"].values())))
    check(x["rounds"] == {"used": True, "questions": ["pick"]},
          "x_yamadori reports that rounds were used", x["rounds"])
    rd = x["answers"]["pick"]["rounds"]
    check([r["stage"] for r in rd] == [1, 1, 1, 2]
          and [r["options"] for r in rd] == [20, 20, 20, 3]
          and rd[-1]["winners"] == ["opt00", "opt20", "opt47"],
          "two stages: three balanced chunks of 20, then the 3 winners",
          [(r["stage"], r["options"], r.get("winners")) for r in rd])
    pf, pc = 10 / 12, 10 / 29
    check(abs(a["probabilities"]["opt47"] - pf * pc) < 1e-5
          and abs(a["probabilities"]["opt00"] - (1 / 12) * (1 / 20)) < 1e-5,
          "P(option) = P_final(its chunk's winner) x P_chunk(option)",
          (a["probabilities"]["opt47"], pf * pc))
    check(abs(a["confidence"] - D.confidence(a["probabilities"])) < 1e-5,
          "confidence over all 60 options (the n-option form)")
    check(len(FAKE.posts) == 8 and d["usage"]["output_tokens"] == 8,
          "(3 chunks + 1 final) x 2 orders = 8 reads, counted in usage",
          (len(FAKE.posts), d["usage"]))
    FAKE.reset(lambda q, t: 25.0 if t == "o200" else 1.0)
    body["questions"] = {"pick": {"type": "choice", "instructions": "Which?",
                                  "criteria": {f"o{i}": None
                                               for i in range(255)}}}
    st, _, d = post(body)
    ok = st == 200
    if ok:
        a = d["answers"]["pick"]
        rd = d["x_yamadori"]["answers"]["pick"]["rounds"]
        ok = (a["choice"] == "o200" and len(a["probabilities"]) == 255
              and len(rd) == 11 and all(2 <= r["options"] <= 26 for r in rd)
              and d["usage"]["output_tokens"] == len(FAKE.posts))
    check(ok, "255 options (Jev's maximum): 10 chunks of <= 26, then the "
          "winners; the choice found", (st, d and d.get("answers")))
    FAKE.reset()
    body["questions"] = {"one": {"type": "choice", "instructions": "Which?",
                                 "criteria": {"only": "the one option"}}}
    st, _, d = post(body)
    check(st == 200 and d["answers"]["one"] == {
        "type": "choice", "choice": "only", "probabilities": {"only": 1.0},
        "confidence": 1.0} and d["x_yamadori"]["answers"]["one"]["reads"]
          == 0 and all(p["question"] != "Which?" for p in FAKE.posts),
          "a single option: probability 1 by definition, nothing read",
          (st, d))
    FAKE.reset()
    body["questions"] = {"q": {"type": "choice", "instructions": "Which?",
                               "criteria": {f"k{i}": None for i in range(26)}}}
    st, _, d = post(body)
    check(st == 200 and d["x_yamadori"]["rounds"]["used"] is False
          and len(FAKE.posts) == 2,
          "26 options: one question, no rounds", (st, len(FAKE.posts)))


def test_busy_and_unavailable():
    # THE QUEUE (operator 2026-10-01): a call whose wait would outlast the SDKs' 10 s timeout is refused at once
    # with the estimate; one that fits waits its turn and is answered
    FAKE.reset()
    with J._q:
        J._running = {"start": time.time(), "est_s": 30.0}
    try:
        st, h, d = post(_one_noul())
    finally:
        with J._q:
            J._running = None
    check(st == 429 and h.get("retry-after") == "30"
          and isinstance(d.get("detail"), str) and not FAKE.posts,
          "a call behind ~30 s of lane work: 429 at once, Retry-After = the estimate (30), nothing read "
          "(never a hang)", (st, dict(h), d))
    FAKE.reset()
    with J._q:
        J._running = {"start": time.time(), "est_s": 0.4}

    def _free():
        with J._q:
            J._running = None
            J._q.notify_all()
    threading.Timer(0.4, _free).start()
    t0 = time.time()
    st, h, d = post(_one_noul())
    q = (d.get("x_yamadori") or {}).get("queue") or {}
    check(st == 200 and q.get("admitted") and 0.2 <= q.get("waited_s", 0) <= 5 and time.time() - t0 < 10,
          "a call behind a short one QUEUES, waits its turn and is answered (x_yamadori.queue)", (st, q))
    FAKE.reset()
    FAKE.fail_post = urllib.error.URLError("connection refused")
    st, h, d = post(_one_noul())
    check(st == 529 and h.get("retry-after", "").isdigit()
          and "MODEL_UNREACHABLE" in d["detail"],
          "the model server unreachable: 529 overloaded with Retry-After",
          (st, dict(h), d))
    FAKE.reset()
    FAKE.fail_tokenize = True
    st, h, d = post(_one_noul())
    check(st == 529 and h.get("retry-after", "").isdigit(),
          "the count cannot be made (model server down): 529", (st, d))
    FAKE.reset()
    real = J.enabled
    J.enabled = lambda: (False, "YAMADORI_DECIDER")
    try:
        st, h, d = post(_one_noul())
    finally:
        J.enabled = real
    check(st == 503 and "YAMADORI_DECIDER=0" in d["detail"] and not FAKE.posts,
          "jjava switched off: 503 saying which switch", (st, d))


def test_state_limit_and_lane():
    FAKE.reset()
    big = "word " * (J.STATE_LIMIT_TOKENS * 4 // 5 + 400)   # > 32,768 fake tokens
    st, _, d = post(_one_noul(state=big))
    check(st == 422 and d["detail"][0]["loc"] == ["body", "state"]
          and d["detail"][0]["type"] == "too_long" and not FAKE.posts,
          "state + the longest question past 32k tokens: 422 on state, "
          "nothing read", (st, (d or {}).get("detail")))
    FAKE.reset()
    mid = "word " * 4000                                     # ~5,000 tokens
    st, _, d = post(_one_noul(state=mid))
    lane = (d or {}).get("x_yamadori", {}).get("lane") or {}
    check(st == 200 and lane.get("fits") is False
          and lane["needed"] > budget.LANE_TOKENS
          and "past the VRAM line" in lane["note"],
          "a state larger than the lane runs anyway, recorded as past the "
          "VRAM line", (st, lane))
    lim = (d or {}).get("x_yamadori", {}).get("limits") or {}
    window = budget.budgets(model="bonsai")["window"]
    check(lim.get("window") == window and lim.get("limit") == min(
        J.STATE_LIMIT_TOKENS, window - lim.get("overhead", 0)),
          "the limit is min(Jev's 32k, the model's window less the template)",
          (lim, window))


def test_corpus_rows():
    db = os.environ["YAMADORI_CORPUS_DB"]
    FAKE.reset()
    post(_one_noul())
    post(_one_noul(), key=TEST_KEY)
    post({})
    con = sqlite3.connect(db)
    rows = [(k, json.loads(p)) for k, p in con.execute(
        "SELECT kind, payload FROM events ORDER BY id")]
    con.close()
    jev = [p for k, p in rows if k == "jev_call"]
    check(jev and all(k in ("jev_call",) for k, _ in rows),
          "every call is a corpus event of kind jev_call, never a turn",
          {k for k, _ in rows})
    last3 = jev[-3:]
    check([p["traffic"] for p in last3] == ["client", "test", "client"],
          "each row carries its traffic class (a live-test account is test)",
          [p.get("traffic") for p in last3])
    check(last3[0]["status"] == 200 and last3[2]["status"] == 422
          and last3[0]["usage"]["output_tokens"] == 2
          and "state" not in last3[0] and last3[0].get("state_sha1"),
          "the row: status, usage, a sha1 of the state (never its text)",
          last3[0])


def main() -> int:
    for fn in (test_doc_examples, test_answers_carry_only_jevs_fields,
               test_usage_is_counted_from_the_reads, test_models_and_aliases,
               test_models_list_shape, test_paths, test_auth, test_validation,
               test_score_levels, test_rounds_past_26,
               test_busy_and_unavailable, test_state_limit_and_lane,
               test_corpus_rows):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
