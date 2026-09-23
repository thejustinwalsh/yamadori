#!/usr/bin/env python
"""The model-assisted clarify step, asserted. No GPU, no network, no real db.

WHAT THIS IS GATING

  1. LICENCE IS NEVER GUESSED. The licence field is filled only from a quote
     that is in the fetched text, or in a LICENSE file fetched from the same
     host, character for character (the extract evidence rule), that is about
     licensing, and that contains the licence it names. A paraphrase, an
     absent quote, a quote naming a different licence -- each leaves the field
     EMPTY, and missing() then says what was searched.
  2. PROVENANCE. Every value the model supplied is stored with where it came
     from: `evidence` (quote + where found) or `proposed` (its reading). An
     operator's value is `operator` and is never overwritten by the model.
  3. UNATTENDED. A source that states its licence runs fetch -> assist ->
     extract -> index -> complete with no human action.
  4. RESTRICTED LICENCES ARE SURFACED, and a model-found one HOLDS the dataset
     in clarify rather than serving the rows on nobody's decision.
  5. THE API carries it: field_states on both dataset endpoints, the minimal
     {url} / {text} submission, and the explicit re-ask.

HOW IT IS ISOLATED

Same as test_worker.py: YAMADORI_JOBS_DB is a temp file before `jobs` is
imported; recipes, datasets dir and hints corpus/cache are temp paths; the
embedder is a fake `code_search`. The model is `model.ask`, replaced -- the
one door, so the assist AND extract both go through the fake. Sources and
LICENSE files come from a local http.server on an ephemeral port.
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import tempfile
import threading
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_assist_")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["YAMADORI_BEAT_SECONDS"] = "0.05"
os.environ["YAMADORI_POLL_SECONDS"] = "0.05"

import numpy as np  # noqa: E402

_fake_cs = types.ModuleType("code_search")


def _embed(texts, is_query=False):
    out = []
    for t in texts:
        v = np.zeros(16, dtype=np.float32)
        v[hash(t) % 16] = 1.0
        out.append(v)
    return np.vstack(out)


_fake_cs.embed = _embed
sys.modules["code_search"] = _fake_cs

import dash_data  # noqa: E402
import datasets  # noqa: E402
import hints  # noqa: E402
import jobs  # noqa: E402
import model  # noqa: E402
import worker  # noqa: E402

RECIPES = os.path.join(_TMP, "recipes")
os.makedirs(RECIPES, exist_ok=True)
datasets.RECIPES = RECIPES
hints.CORPUS = RECIPES
hints.CACHE = os.path.join(_TMP, "hints.npz")
worker.DATA_DIR = os.path.join(_TMP, "datasets")

REAL = os.path.abspath(os.path.join(HERE, "..", "index", "jobs.sqlite3"))

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ---------------------------------------------------------------------------
# The sources. Each page carries the advice extract will quote; they differ
# only in what they say about licensing.
# ---------------------------------------------------------------------------
ADVICE = ("Always validate widget props at the component boundary so that a "
          "malformed config fails loudly instead of rendering half a widget.")


def page(title: str, footer: str = "") -> bytes:
    return (f"<html><head><title>{title}</title></head><body>"
            f"<h1>{title}</h1><p>{ADVICE}</p>"
            f"<footer><p>{footer}</p></footer></body></html>").encode()


MIT = ("MIT License\n\nCopyright (c) 2024 Widget Co\n\nPermission is hereby "
       "granted, free of charge, to any person obtaining a copy of this "
       "software.\n")

PAGES = {
    "/footer/page": page("Widget Handbook",
                         "This documentation is licensed under CC-BY-4.0."),
    "/filed/page": page("Gadget Guide"),
    "/filed/LICENSE": MIT.encode(),
    "/bare/page": page("Bare Notes"),
    "/agpl/page": page("Copyleft Manual",
                       "Licensed under the GNU AGPL-3.0; see COPYING."),
    "/api/page": page("Api Notes"),
}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/offhost/page":
            body = page("Off Host")
        elif self.path.startswith("/offhost/LICENSE"):
            # A redirect to a DIFFERENT hostname for the same server: the
            # assist must not read what it lands on.
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{PORT}"
                                         "/filed/LICENSE")
            self.end_headers()
            return
        else:
            body = PAGES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        ctype = "text/plain" if "LICENSE" in self.path else "text/html"
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
threading.Thread(target=_server.serve_forever, daemon=True).start()
PORT = _server.server_address[1]
BASE = f"http://127.0.0.1:{PORT}"


# ---------------------------------------------------------------------------
# The fake model, through the one door. REPLIES maps a marker that appears in
# the user message (a URL path, or a word in pasted text) to the JSON the
# "model" answers with. Extract calls are answered with one verifiable recipe.
# ---------------------------------------------------------------------------
REPLIES: dict[str, dict | str] = {}
CALLS: list[dict] = []


def fake_ask(messages, **kw):
    system, user = messages[0]["content"], messages[-1]["content"]
    CALLS.append({"system": system, "user": user, "kw": kw})
    if "RECIPES" in system:
        return json.dumps([{
            "recipe": "Validate widget props at the component boundary.",
            "trigger_condition": "writing a widget that takes config",
            "category": "validation", "confidence": "high",
            "evidence": ADVICE[:90]}])
    for marker, reply in REPLIES.items():
        if marker in user:
            if isinstance(reply, Exception):
                raise reply
            return reply if isinstance(reply, str) else json.dumps(reply)
    return json.dumps({})


model.ask = fake_ask


def reply(**kw) -> dict:
    base = {"source_name": None, "source_name_quote": None, "language": None,
            "domains": [], "licence": None, "licence_quote": None,
            "licence_quote_url": None}
    base.update(kw)
    return base


def drain() -> int:
    return worker.run(once=True)


def assist_rows(did: str) -> list[dict]:
    return [j for j in jobs.listing(dataset=did)
            if j["queue"] == datasets.ASSIST[0]]


def states(ds: dict) -> dict:
    return {r["field"]: r for r in datasets.field_states(ds)}


def url_dataset(path: str, **kw) -> dict:
    return datasets.create(f"{BASE}{path}", **kw)


# ---------------------------------------------------------------------------
def test_the_fixture_is_isolated():
    check(os.path.abspath(jobs.DB).startswith(os.path.abspath(_TMP))
          and os.path.abspath(jobs.DB) != REAL,
          "the jobs database is a temp file, not the real index", jobs.DB)
    check(model.ask is fake_ask, "model.ask is the fake: nothing reaches "
          ":11434 or :1234")


def test_a_verbatim_footer_quote_fills_the_licence_and_runs_unattended():
    REPLIES["/footer/page"] = reply(
        source_name="Widget Handbook", source_name_quote="Widget Handbook",
        language="typescript", domains=["types", "made-up-domain"],
        licence="CC-BY-4.0",
        licence_quote="This documentation is licensed under CC-BY-4.0.",
        licence_quote_url=f"{BASE}/footer/page")
    ds = url_dataset("/footer/page")
    check(not assist_rows(ds["id"]),
          "a URL dataset does not enqueue the assist before its fetch is done")
    drain()
    ds = datasets.get(ds["id"])
    a = assist_rows(ds["id"])
    check(len(a) == 1 and a[0]["state"] == "done" and a[0]["lane"] == "gpu",
          "the finished fetch enqueued exactly one assist, on the gpu lane, "
          "and it is done", json.dumps([(j["lane"], j["state"]) for j in a]))
    asked = [c for c in CALLS if "/footer/page" in c["user"]
             and "RECIPES" not in c["system"]]
    check(asked and asked[-1]["kw"].get("effort") == "medium",
          "the assist asked through model.ask at effort medium",
          str(asked and asked[-1]["kw"]))
    check(ds["licence"] == "CC-BY-4.0",
          "a verbatim footer quote fills the licence", ds["licence"])
    st = states(ds)
    lic = st["licence"]
    check(lic["state"] == "evidence"
          and lic["quote"] == "This documentation is licensed under CC-BY-4.0."
          and lic["found_in"] == f"{BASE}/footer/page",
          "and it is stored as EVIDENCE with the quote and where it was found",
          json.dumps(lic))
    check(st["source_name"]["state"] == "evidence",
          "a source name whose verbatim line was quoted is evidence too",
          st["source_name"]["state"])
    check(st["language"]["state"] == "proposed"
          and st["domains"]["state"] == "proposed",
          "language and domains are PROPOSED: the model's reading, labelled",
          json.dumps([st["language"]["state"], st["domains"]["state"]]))
    check(ds["domains"] == ["types"],
          "a domain outside domains.DOMAINS is never stored", str(ds["domains"]))
    check((ds["assist"].get("rejected") or {}).get("domains", {})
          .get("values") == ["made-up-domain"],
          "it is recorded as rejected instead",
          json.dumps(ds["assist"].get("rejected")))
    check(st["source_url"]["state"] == "operator",
          "the URL the operator typed is the operator's", st["source_url"]["state"])
    check(ds["stage"] == "complete",
          "with nothing missing it ran on unattended: fetch -> assist -> "
          "extract -> index -> complete", ds["stage"])
    queues = {j["queue"]: j["state"] for j in jobs.listing(dataset=ds["id"])}
    check(queues == {"dataset.fetch": "done", "dataset.assist": "done",
                     "dataset.extract": "done", "dataset.index": "done"},
          "every stage ran once and is done", json.dumps(queues))
    path = os.path.join(RECIPES, datasets.recipe_file(ds))
    with open(path, encoding="utf-8") as f:
        row = json.loads(f.readline())
    check(row["license_if_known"] == "CC-BY-4.0"
          and row["source_name"] == "Widget Handbook",
          "extracted rows carry the licence the assist established",
          json.dumps({k: row.get(k) for k in
                      ("license_if_known", "source_name")}))


def test_a_licence_file_on_the_same_host_is_read_and_quoted():
    REPLIES["/filed/page"] = reply(
        source_name="Gadget Guide", language="rust", domains=["systems"],
        licence="MIT License", licence_quote="MIT License",
        licence_quote_url=f"{BASE}/filed/LICENSE")
    ds = url_dataset("/filed/page")
    drain()
    ds = datasets.get(ds["id"])
    asked = [c for c in CALLS if "/filed/page" in c["user"]
             and "RECIPES" not in c["system"]][-1]
    check("Permission is hereby granted" in asked["user"],
          "the LICENSE file from the same host was fetched and shown to the "
          "model")
    lic = states(ds)["licence"]
    check(ds["licence"] == "MIT License" and lic["state"] == "evidence"
          and lic["found_in"] == f"{BASE}/filed/LICENSE",
          "the licence is filled from the LICENSE file, found_in naming it",
          json.dumps(lic))
    check(states(ds)["source_name"]["state"] == "proposed",
          "a name given with no quote is PROPOSED, not evidence")
    searched = ds["assist"]["searched"]
    check(searched[0]["kind"] == "source"
          and any(s["where"] == f"{BASE}/filed/LICENSE"
                  and s.get("result", "").startswith("read")
                  for s in searched),
          "what was searched is stored with the dataset",
          json.dumps(searched)[:300])
    check(os.path.exists(os.path.join(worker.dataset_dir(ds["id"]),
                                      "licence-0.txt")),
          "and the licence file it quoted is kept beside the source")


def _needs_licence(ds: dict) -> dict | None:
    return next((m for m in datasets.missing(ds) if m["field"] == "licence"),
                None)


def test_a_quote_that_is_not_in_the_source_leaves_the_licence_missing():
    REPLIES["/bare/page"] = reply(
        source_name="Bare Notes", language="python", domains=["tooling"],
        licence="Apache-2.0",
        licence_quote="Licensed under the Apache License, Version 2.0",
        licence_quote_url=f"{BASE}/bare/page")
    ds = url_dataset("/bare/page")
    drain()
    ds = datasets.get(ds["id"])
    check(ds["licence"] == "",
          "a licence whose quote is not in any fetched text is NOT stored",
          repr(ds["licence"]))
    m = _needs_licence(ds)
    check(m is not None, "missing() asks the operator for it")
    if m:
        check("searched" in m and f"{BASE}/bare/page" in m["searched"]
              and f"{BASE}/bare/LICENSE" in m["searched"]
              and "HTTP 404" in m["searched"],
              "and says what was searched: the page and each LICENSE tried",
              m.get("searched", "")[:300])
        check("failed verification" in m["what"]
              and "not in any fetched text" in m["what"],
              "and why the model's offer was discarded", m["what"][:300])
    check(states(ds)["licence"]["state"] == "needs_you",
          "the field is NEEDS YOU", states(ds)["licence"]["state"])
    check(ds["stage"] == "clarify"
          and not [j for j in jobs.listing(dataset=ds["id"])
                   if j["queue"] == "dataset.extract"],
          "and the dataset stays in clarify: nothing was extracted",
          ds["stage"])
    check(states(ds)["source_name"]["state"] == "proposed",
          "while the fields it may propose are still proposed")


def _pasted(marker: str, body: dict, **kw) -> dict:
    REPLIES[marker] = body
    ds = datasets.create(f"notes {marker}", source=(
        f"{marker} notes. {ADVICE} This page is licensed under CC-BY-4.0."),
        source_url="local:paste", **kw)
    drain()
    return datasets.get(ds["id"])


def test_an_absent_or_non_verbatim_quote_is_discarded_each_way():
    cases = {
        "absentquote": (reply(licence="MIT", licence_quote=None),
                        "quoted nothing"),
        "paraphrase": (reply(licence="CC BY 4.0",
                             licence_quote="licensed under the CC BY 4.0 "
                                           "licence"),
                       "not in any fetched text"),
        "wrongname": (reply(licence="MIT",
                            licence_quote="This page is licensed under "
                                          "CC-BY-4.0."),
                      "not inside its quote"),
        "notlicence": (reply(licence="widget props",
                             licence_quote="validate widget props at the "
                                           "component boundary"),
                       "does not mention a licence"),
    }
    for marker, (body, why) in cases.items():
        ds = _pasted(marker, body)
        rej = (ds["assist"].get("rejected") or {}).get("licence") or {}
        check(ds["licence"] == "" and why in rej.get("why", ""),
              f"{marker}: the licence stays empty ({why})",
              json.dumps([ds["licence"], rej]))
    ds = _pasted("verbatim", reply(
        licence="CC-BY-4.0",
        licence_quote="This page is licensed under   CC-BY-4.0."))
    check(ds["licence"] == "CC-BY-4.0"
          and states(ds)["licence"]["found_in"] == "the pasted source text",
          "the same quote, verbatim up to whitespace, fills it from the "
          "pasted text", json.dumps(states(ds)["licence"]))
    nf = ds["assist"].get("not_found") or {}
    check(ds["stage"] == "clarify" and "source_name" in nf
          and "the pasted source text" in nf["source_name"],
          "with no name proposed it stays in clarify, and says what the "
          "assist read", json.dumps([ds["stage"], nf.get("source_name")]))


def test_a_restricted_licence_is_surfaced_and_held():
    REPLIES["/agpl/page"] = reply(
        source_name="Copyleft Manual", language="c", domains=["systems"],
        licence="GNU AGPL-3.0",
        licence_quote="Licensed under the GNU AGPL-3.0",
        licence_quote_url=f"{BASE}/agpl/page")
    ds = url_dataset("/agpl/page")
    drain()
    ds = datasets.get(ds["id"])
    check(ds["licence"] == "GNU AGPL-3.0"
          and states(ds)["licence"]["state"] == "evidence",
          "an AGPL footer is quoted and filled like any other licence",
          ds["licence"])
    w = datasets.warnings(ds)
    check(any(x["kind"] == "licence" and "AGPL" in x["what"] for x in w),
          "warnings() surfaces the restriction", json.dumps(w))
    check(not datasets.missing(ds) and ds["stage"] == "clarify",
          "every question is answered, yet the dataset is HELD in clarify",
          ds["stage"])
    check("held in clarify" in (ds["assist"].get("held") or ""),
          "and the hold says why", str(ds["assist"].get("held"))[:200])
    check(not [j for j in jobs.listing(dataset=ds["id"])
               if j["queue"] == "dataset.extract"],
          "nothing was extracted on nobody's decision")
    ds = datasets.advance(ds["id"])
    check(ds["stage"] == "extract",
          "a person advancing it is not blocked: the hold is not a gate")


def test_an_operator_answer_is_never_overwritten_and_overrides_are_kept():
    ds = _pasted("operatorfirst", reply(
        source_name="Model Name", language="go",
        licence="CC-BY-4.0",
        licence_quote="This page is licensed under CC-BY-4.0."),
        source_name="Operator Name")
    st = states(ds)
    check(ds["source_name"] == "Operator Name"
          and st["source_name"]["state"] == "operator",
          "a field the operator answered at submission is not overwritten",
          json.dumps([ds["source_name"], st["source_name"]["state"]]))
    check(st["language"]["state"] == "proposed" and ds["language"] == "go",
          "while an unanswered one is proposed")
    ds = datasets.answer(ds["id"], {"licence": "CC0-1.0"})
    lic = states(ds)["licence"]
    check(lic["state"] == "operator"
          and (lic.get("overrode") or {}).get("provenance") == "evidence",
          "an operator override becomes OPERATOR and records the evidence it "
          "replaced", json.dumps(lic))
    # A value changed behind the provenance record's back is not evidence.
    con = datasets._db()
    con.execute("UPDATE datasets SET language='rust' WHERE id=?", (ds["id"],))
    con.close()
    check(states(datasets.get(ds["id"]))["language"]["state"] == "operator",
          "a value that no longer matches its provenance record is not shown "
          "as the model's")


def test_a_failed_model_call_stores_nothing():
    REPLIES["garbled"] = "I think it is probably MIT, no JSON here"
    ds = datasets.create("garbled notes", source=f"garbled. {ADVICE}",
                         source_url="local:paste")
    drain()
    a = assist_rows(ds["id"])
    check(a and a[0]["state"] == "errored"
          and "retryable: yes" in (a[0]["error"] or "")
          and a[0]["attempts"] == a[0]["max_attempts"],
          "an unparseable reply is retried, then errored with a structured "
          "reason -- never done", json.dumps(a and {k: a[0][k] for k in
                                                     ("state", "attempts",
                                                      "error")})[:300])
    ds = datasets.get(ds["id"])
    check(ds["licence"] == "" and not ds["assist"].get("searched"),
          "and nothing the model said was stored")

    REPLIES["overbudget"] = model.BudgetEvent("", {"completion_tokens": 9}, 0)
    ds = datasets.create("overbudget notes", source=f"overbudget. {ADVICE}",
                         source_url="local:paste")
    drain()
    a = assist_rows(ds["id"])
    check(a and a[0]["state"] == "errored"
          and "token limit" in (a[0]["error"] or ""),
          "a token-limit finish is reported as one, with a remedy",
          str(a and a[0]["error"])[:200])


def test_licence_file_locations():
    c = worker.licence_candidates(
        "https://github.com/acme/widgets/blob/main/docs/guide.md")
    check(c[0] == "https://raw.githubusercontent.com/acme/widgets/main/LICENSE",
          "github.com docs look in the repo's raw LICENSE at that ref", c[0])
    c = worker.licence_candidates("https://acme.github.io/widgets/guide/")
    check(c[0] == "https://raw.githubusercontent.com/acme/widgets/HEAD/LICENSE",
          "*.github.io project pages look in that repo", c[0])
    c = worker.licence_candidates("https://docs.example.com/a/b/page.html")
    check(c[0] == "https://docs.example.com/a/b/LICENSE"
          and c[-1] == "https://docs.example.com/LICENSE.txt"
          and all(x.startswith("https://docs.example.com/") for x in c),
          "anything else looks on the same host only", json.dumps(c))
    check(worker.licence_candidates("local:paste") == [],
          "a non-web source looks nowhere")
    rec, body = worker.fetch_licence_file(f"{BASE}/offhost/LICENSE",
                                          {"127.0.0.1"})
    check(body == "" and "off-host" in rec.get("result", ""),
          "a LICENSE that redirects to another host is not read",
          json.dumps(rec))


def test_the_api_carries_provenance_and_takes_the_minimal_submission():
    REPLIES["/api/page"] = reply(source_name="Api Notes", language="rust")
    code, _, body = dash_data.handle_post("/dash/api/dataset",
                                          {"url": f"{BASE}/api/page"})
    j = json.loads(body)
    check(code == 200 and j["ok"]
          and j["dataset"]["source_url"] == f"{BASE}/api/page",
          "POST /dash/api/dataset with just {url} records the dataset",
          body[:200].decode())
    did = j["dataset"]["id"]
    drain()
    code, _, body = dash_data.handle_get(f"/dash/api/datasets/{did}")
    d = json.loads(body)
    fs = {r["field"]: r["state"] for r in d.get("field_states", [])}
    check(code == 200 and fs.get("source_name") == "proposed"
          and fs.get("licence") == "needs_you" and d.get("assist_jobs")
          and "searched" in d.get("assist", {}),
          "GET /dash/api/datasets/{id} carries field_states, the assist "
          "record and the assist job", json.dumps(fs))
    code, _, body = dash_data.handle_get("/dash/api/datasets")
    o = json.loads(body)
    one = next(x for x in o["datasets"] if x["id"] == did)
    check("field_states" in one and "assist" in one
          and o.get("assist", {}).get("lane") == "gpu",
          "and so does the overview, with the assist lane declared")

    code, _, body = dash_data.handle_post(
        "/dash/api/dataset", {"text": "pasted apitext. " + ADVICE,
                              "kind": "laya"})
    j = json.loads(body)
    check(code == 200 and j["dataset"]["source"].startswith("pasted apitext")
          and j["dataset"]["kind"] == "laya"
          and len(j["assist_jobs"]) == 1,
          "{text} becomes the source and enqueues the assist at once",
          body[:300].decode())
    tid = j["dataset"]["id"]
    code, _, body = dash_data.handle_post("/dash/api/dataset/assist",
                                          {"id": tid})
    check(code == 409 and "already" in json.loads(body)["error"],
          "asking again while one is queued is refused, with the reason",
          body.decode()[:200])
    drain()
    code, _, body = dash_data.handle_post("/dash/api/dataset/assist",
                                          {"id": tid})
    check(code == 200 and json.loads(body)["ok"],
          "asking again after it finished enqueues a new one")
    drain()
    code, _, body = dash_data.handle_post("/dash/api/dataset",
                                          {"url": "not a url"})
    check(code == 400, "a {url} that is not a URL is refused", str(code))
    code, _, body = dash_data.handle_post(
        "/dash/api/dataset/answer", {"id": did, "licence": "MIT"})
    fs = {r["field"]: r["state"] for r in json.loads(
        dash_data.handle_get(f"/dash/api/datasets/{did}")[2])["field_states"]}
    check(code == 200 and fs["licence"] == "operator",
          "the existing answer endpoint still works and marks the operator")


def main() -> int:
    for fn in (test_the_fixture_is_isolated,
               test_a_verbatim_footer_quote_fills_the_licence_and_runs_unattended,
               test_a_licence_file_on_the_same_host_is_read_and_quoted,
               test_a_quote_that_is_not_in_the_source_leaves_the_licence_missing,
               test_an_absent_or_non_verbatim_quote_is_discarded_each_way,
               test_a_restricted_licence_is_surfaced_and_held,
               test_an_operator_answer_is_never_overwritten_and_overrides_are_kept,
               test_a_failed_model_call_stores_nothing,
               test_licence_file_locations,
               test_the_api_carries_provenance_and_takes_the_minimal_submission):
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
    print(f"  temp database: {os.path.abspath(jobs.DB)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
