#!/usr/bin/env python
"""HARNESS DECISIONS: every decision point, replayed on REAL request shapes
from each harness. Offline, no GPU, no network.

    python mcp/test_harness_decisions.py         -> "N/M checks passed"
    python mcp/test_harness_decisions.py -v      every check, not only misses
    python mcp/test_harness_decisions.py -k pi   fixtures whose id has "pi"

WHY (operator, 2026-09-26). The server's job is to enhance the model where
it needs help while each harness drives it the way it expects. This week's
harness tests found most bugs at DECISION POINTS misreading another
harness's shapes: Pi's `developer` role (a 502 with the addendum), Pi's and
OpenCode's synthetic image turns routed as prose / counted as kickoffs / as
the user speaking, OpenCode's title call recorded as a compaction, Pi's and
OpenCode's compactions unrecognised, Pi's edit tool caught only by the shape
fallback, .html writes unchecked. Each was found live, one harness at a
time. This gate replays the shapes offline, every harness through every
decision point, so a new harness -- or a change to a classifier -- is
checked against all of them at once (PROTOCOL rule 7: a detector is only
tested by input captured from the real producer).

THE FIXTURES are bench/harness_shapes/<harness>/*.json, one request each:

    {id, harness, harness_version, wire: chat|responses, source,
     headers, body, expect: {<decision point>: <expected decision>}, why}

`source` names the capture it was taken from (relay logs, the Responses
stub captures, the corpus) or says FROM SOURCE when the shape was read from
the harness's code and not seen live. Content is synthetic or a trimmed head
of the harness's own prompt text (the repo is public: no key, no operator
path, no operator image or prompt, no Octopus spec text). Each harness
directory's README.md names its sources and version.

THE EXPECTATIONS are labelled from the HARNESS's source and intent, never
from what the code does today. Where today's code disagrees, the check stays
and KNOWN below marks it with the reason: it prints KNOWN-FAIL, is not
counted as a gating check, and a known failure that starts passing prints
KNOWN-FAIL NOW PASSES (remove its mark). Any other miss is a FAIL.

THE DECISION POINTS (the columns of the matrix printed at the end)

  accepted    the body is served, not refused: api_errors.validate_chat
              (chat), responses_api.to_chat (Responses)
  route       mcp/route.py's class, as proxy.prepare decides it
  utility     selection.utility_call via proxy.utility_of: side call or
              not, and its kind (title / classifier / compaction / ...)
  compaction  compaction.parse_flattened (whose form, how many records) or
              compaction.in_place, and the prepared compaction record
  session     proxy.session_identity's source (prompt_cache_key, header --
              server.session_of_headers -- tool_call_id, minted; None for
              a side call)
  deep        mcp/deep.py's trigger INPUTS: kickoff (deep.kickoff), the
              user speaking (selection.question_of) and the struggle
              signal count (deep.struggle_scan)
  roles       system/developer handling: after prepare exactly one system
              message, first, at the harness's own tier AND at `high` (the
              addendum's tier), rendered by the SERVED chat template
              (mcp/fixtures/bonsai_chat_template.jinja) without raising
  image       vision.normalise's register (each image's `form`) and no
              image part left in what the text model is sent
  tool_code   tool_code.detect / review on the harness's own write/edit
              calls: known-name table vs shape fallback, the unit kind,
              the language, whether it blocks
  skills      skill_select.select with the AUTHORED skills armed in a temp
              store (skills/authored/, deterministic stages: the embedder
              and the fallback are the live stack's and are replaced), both
              called directly and through proxy.prepare's injection: which
              skills a target shape gets, and that a near miss gets none
  project     mcp/progress.py's project: the fixture's conversation
              replayed prefix by prefix (each request that ends on tool
              results, in order, one session) through the wire's
              translation and proxy.prepare -- the working directory, the
              project writes (v0f-V0: a write has to pair with its result
              on the Responses path), and that every tool result goes up
              as the harness sent it (the situation lines were removed
              2026-09-27)

ADDING A HARNESS: drop fixtures under bench/harness_shapes/<name>/ with a
README.md; the matrix grows a row. Every database is a temp file, set
BEFORE proxy is imported.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
SHAPES = os.path.join(ROOT, "bench", "harness_shapes")

_TMP = tempfile.mkdtemp(prefix="yamadori_test_harness_decisions_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json"),
               ("YAMADORI_JOBS_DB", "jobs.sqlite3"),
               ("YAMADORI_SKILLS_DIR", "skills"),
               ("YAMADORI_SKILL_TRIGGER_CACHE", "triggers.npz"),
               ("YAMADORI_SKILL_LABELS", "labels.jsonl"),
               ("YAMADORI_TOKEN_LEDGER", "token_ledger.sqlite3"),
               ("YAMADORI_POWER_LEDGER", "power_ledger.json"),
               ("YAMADORI_ACCOUNTS_DIR", "accounts"),
               ("YAMADORI_INDEX_DIR", "repos"),
               ("YAMADORI_PACKAGES_DIR", "packages"),
               ("YAMADORI_PKG_DIR", "pkgs"),
               ("YAMADORI_MEDIA_DIR", "media"),
               ("YAMADORI_GPU_ROOM_DIR", "gpu_room"),
               ("YAMADORI_DATASETS_DIR", "datasets")):
    os.environ[_k] = os.path.join(_TMP, _v)
for _d in ("YAMADORI_PKG_DIR", "YAMADORI_PACKAGES_DIR", "YAMADORI_INDEX_DIR",
           "YAMADORI_MEDIA_DIR", "YAMADORI_ACCOUNTS_DIR"):
    os.makedirs(os.environ[_d], exist_ok=True)
os.environ["YAMADORI_SKILL_CACHE_SECONDS"] = "0"
os.environ["YAMADORI_GPU_ROOM"] = "0"
# Nothing here may reach a live server: every door is a closed port.
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:9"
os.environ["YAMADORI_MODEL_SERVER"] = "http://127.0.0.1:9"
os.environ["YAMADORI_SLOTS"] = "4"
for _k in ("YAMADORI_E1", "YAMADORI_SKILL_LAYA", "YAMADORI_TEST_KEY",
           "YAMADORI_IMAGEGEN_URL", "X_YAMADORI_FEATURES"):
    os.environ.pop(_k, None)

import jinja2  # noqa: E402

import api_errors  # noqa: E402
import compaction  # noqa: E402
import deep  # noqa: E402
import image_input  # noqa: E402
import jobs  # noqa: E402
import nebari  # noqa: E402
import progress  # noqa: E402
import proxy  # noqa: E402
import responses_api  # noqa: E402
import selection  # noqa: E402
import server  # noqa: E402
import skill_migrate  # noqa: E402
import skill_select  # noqa: E402
import skills  # noqa: E402
import system_roles  # noqa: E402
import tiers  # noqa: E402
import tool_code  # noqa: E402
import model as _model  # noqa: E402

_tmp_root = os.path.abspath(tempfile.gettempdir())
for _p in (jobs.DB, skills.STORE, nebari.DB):
    assert os.path.abspath(_p).startswith(_tmp_root), _p

proxy.UPSTREAM = "http://127.0.0.1:9"
_model.UPSTREAM = "http://127.0.0.1:9"
# The served template's efforts (tiers.accepted_efforts reads them live).
tiers._accepted = ("low", "medium", "xhigh")
# The embedding stage and the fallback need the live stack: replaced, as in
# mcp/test_skill_factory.py. The embedder "does not answer" in a way that is
# NOT retryable (a real decision without stage 2), so the cascade runs its
# deterministic stages only; the fallback confirms nothing.
skill_select.best_cosines = lambda q, pool: ({}, {
    "ok": False, "why": "the embedding stage is not run offline"})
skill_select.ask_fallback = lambda system, user: json.dumps(
    {"use": [], "why": "the offline fallback confirms nothing"})
skill_select.refresh_triggers = lambda: {"rows": 0, "offline": True}
proxy._draw_seed = lambda prompt=None: {"word": "cedar", "token_id": 1,
                                        "u32": 1}

POINTS = ("accepted", "route", "utility", "compaction", "session", "deep",
          "roles", "image", "tool_code", "skills", "project", "tools")
# Decision points checked on EVERY fixture (no `expect` row needed): the
# tool list's no-conflict rule holds for any request of any harness.
UNIVERSAL = {"tools": {}}
# The tools of ours a harness's own list must displace (declared overlaps,
# proxy.TOOL_OVERLAPS): Hermes' vision_analyze answers yama_describe_image's
# question.
UNIVERSAL_WANT = {"hermes": {"tools": {"withheld": ["yama_describe_image"]}}}

# ================================================================ KNOWN ====
# (fixture id, decision point) -> why today's code disagrees with the
# harness. Each is an open finding; the list is the report. "fix list" rows
# are being fixed elsewhere (2026-09-26); "NEW" rows were found by this gate.
KNOWN: dict[tuple[str, str], str] = {
    # The expectation is the harness's intent (one command failing the same
    # way 4 times is a struggle); the code no longer reads these results.
    ("codex/responses/struggle-same-error", "deep"): (
        "NEW 2026-09-27: plain-text tool results never signal since "
        "deep._ERR_TEXT was removed (docs/CONSTANTS-AUDIT.md: only "
        "structured status/error fields count). Codex writes its exit "
        "status as a header line ('Process exited with code 1', "
        "docs/HARNESS-CODEX.md s4, captured), which is not read: an open "
        "decision."),
    ("pi/chat/struggle-same-error", "deep"): (
        "NEW 2026-09-27: plain-text tool results never signal since "
        "deep._ERR_TEXT was removed (docs/CONSTANTS-AUDIT.md). Pi appends "
        "its exit status as a line ('Command exited with code 1', "
        "tools/bash.js), which is not read: an open decision."),
    ("opencode/chat/struggle-same-error", "deep"): (
        "NEW 2026-09-27: plain-text tool results never signal since "
        "deep._ERR_TEXT was removed (docs/CONSTANTS-AUDIT.md). OpenCode's "
        "bash result carries no exit status in its text at all."),
}

# ============================================================= plumbing ====
_TEMPLATE = None


def _template():
    global _TEMPLATE
    if _TEMPLATE is None:
        env = jinja2.Environment()
        env.filters["tojson"] = lambda v, **k: json.dumps(
            v, ensure_ascii=False)
        with open(os.path.join(HERE, "fixtures",
                               "bonsai_chat_template.jinja"),
                  encoding="utf-8") as f:
            _TEMPLATE = env.from_string(f.read())
    return _TEMPLATE


def _raise(msg):
    raise ValueError(msg)


def render(payload: dict) -> str:
    """The prompt llama-server builds (tool-call arguments parsed, as
    llama-server hands them to the template). Raises what the template
    raises: "System message must be at the beginning" was Pi's 502."""
    msgs = []
    for m in payload.get("messages") or []:
        m = dict(m)
        if m.get("role") == "developer":
            # llama-server maps `developer` to `system` before rendering
            # (docs/HARNESS-PI.md: 2,345 prompt tokens either way).
            m["role"] = "system"
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except ValueError:
                        pass
                calls.append(dict(c, function=fn))
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        msgs.append(m)
    kw = {}
    if payload.get("reasoning_effort"):
        kw["reasoning_effort"] = payload["reasoning_effort"]
    ctk = payload.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        kw["enable_thinking"] = ctk["enable_thinking"]
    return _template().render(messages=msgs,
                              tools=payload.get("tools") or None,
                              add_generation_prompt=True,
                              raise_exception=_raise, **kw)


def load(k: str = "") -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(SHAPES, "*", "*.json"))):
        with open(path, encoding="utf-8") as f:
            fx = json.load(f)
        fx["_path"] = os.path.relpath(path, ROOT).replace(os.sep, "/")
        if k in fx["id"]:
            out.append(fx)
    return out


def chat_of(fx: dict) -> tuple[dict | None, str | None]:
    """(the chat body the turn runs on, why it was refused)."""
    body = json.loads(json.dumps(fx["body"]))
    if fx["wire"] == "responses":
        try:
            chat, _ctx = responses_api.to_chat(body)
        except Exception as e:                                   # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"[:300]
        return chat, None
    bad = api_errors.validate_chat(body)
    if bad is not None:
        return None, str(getattr(bad, "message", bad))[:300]
    return body, None


def prepared(fx: dict, chat: dict, effort: str | None = None,
             tag: str = "") -> dict:
    """proxy.prepare on the request as the server would hand it over: the
    session headers read by server.session_of_headers, an account of its
    own per fixture (nothing leaks between fixtures)."""
    body = json.loads(json.dumps(chat))
    if effort:
        body["reasoning_effort"] = effort
    hdr = {str(k).lower(): v for k, v in (fx.get("headers") or {}).items()}
    body["_session_token"], body["_session_header"] = \
        server.session_of_headers(hdr)
    body["_account"] = "hs-" + hashlib.sha1(
        (fx["id"] + tag).encode()).hexdigest()[:10]
    body["_client_ip"] = ""
    if "skills" in (fx.get("expect") or {}):
        # Skills are off at every tier (operator, 2026-09-29: "Stop skills
        # until we have a good skill injector."); a fixture that checks the
        # skills path forces it on, as a benchmark header does.
        feats = json.loads(hdr.get("x-yamadori-features") or "{}")
        body["_features"] = json.dumps(dict(feats, skills=True))
    skill_select._STICKY.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = proxy.prepare(body)
    out["_body_in"] = body
    return out


def _raw(chat: dict) -> list[dict]:
    """The messages as prepare's decisions read them (instruction roles
    mapped by system_roles.one_system, as prepare does first)."""
    msgs, _rec = system_roles.one_system(chat.get("messages") or [])
    return msgs


def _last_user_text(msgs: list[dict]) -> str:
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return selection._text(m)
    return ""


def _calls(msgs: list[dict], where: list) -> dict | None:
    with_calls = [m for m in msgs if isinstance(m, dict)
                  and m.get("role") == "assistant" and m.get("tool_calls")]
    try:
        return with_calls[where[0]]["tool_calls"][where[1]]
    except (IndexError, KeyError, TypeError):
        return None


# ======================================================= the checks ========
# Each returns (ok, got) for one decision point of one fixture.

def c_accepted(fx, want, cx):
    ok = (cx["chat"] is not None) == bool(want)
    return ok, "served" if cx["chat"] is not None else f"refused: {cx['why']}"


def c_route(fx, want, cx):
    got = (cx["out"].get("_route") or {}).get("class")
    return got == want, f"{got} ({(cx['out'].get('_route') or {}).get('because', '')[:90]})"


def c_utility(fx, want, cx):
    util = cx["out"].get("_utility") or {}
    got = {"utility": bool(util.get("utility")),
           "kind": cx["out"].get("_utility_kind")}
    ok = got["utility"] == bool(want.get("utility"))
    if "kind" in want:
        ok = ok and got["kind"] == want["kind"]
    return ok, f"{got} ({str(util.get('because'))[:80]})"


def c_compaction(fx, want, cx):
    raw = cx["raw"]
    text = _last_user_text(raw)
    parsed = compaction.parse_flattened(text) if text else None
    inplace = compaction.in_place(raw)
    kind = cx["out"].get("_utility_kind")
    rec = cx["out"].get("_compaction") or {}
    got = {"parsed": (parsed or {}).get("harness"),
           "records": len((parsed or {}).get("records") or []),
           "in_place": inplace, "kind": kind,
           "served_as": rec.get("harness"), "shape": rec.get("shape")}
    if want.get("none"):
        ok = parsed is None and not inplace and kind != "compaction"
    elif want.get("in_place"):
        ok = inplace and kind == "compaction"
    else:
        ok = (got["parsed"] == want["harness"]
              and got["records"] >= want.get("records_min", 1)
              and kind == "compaction" and rec.get("harness") == want["harness"])
    return ok, str(got)


def c_session(fx, want, cx):
    ses = (cx["out"].get("_ledger") or {}).get("conversation")
    src = (ses or {}).get("source") if ses else None
    ok = src == want.get("source")
    hdr = cx["out"]["_body_in"].get("_session_header")
    if ok and want.get("header"):
        ok = hdr == want["header"]
    return ok, f"source={src} header={hdr or None}"


def c_deep(fx, want, cx):
    raw = cx["raw"]
    ko = deep.kickoff(raw)
    speaking = selection.question_of(raw)[2]
    scan = deep.struggle_scan(raw)
    ev = scan["events"]
    # (`still_broken`, the user_still_broken signal, was removed 2026-09-27.)
    got = {"kickoff": ko["new_task"], "speaking": speaking,
           "struggle": len(ev)}
    ok = True
    for k in ("kickoff", "speaking"):
        if k in want:
            ok = ok and got[k] == want[k]
    if "struggle_min" in want:
        ok = ok and got["struggle"] >= want["struggle_min"]
    if "struggle_max" in want:
        ok = ok and got["struggle"] <= want["struggle_max"]
    kinds = sorted({e["kind"] for e in ev})
    return ok, f"{got} kinds={kinds} kickoff_why={ko['why'][:70]!r}"


def _roles_of(out: dict, addendum: bool) -> tuple[bool, str]:
    msgs = out.get("messages") or []
    inst = [i for i, m in enumerate(msgs) if isinstance(m, dict)
            and m.get("role") in ("system", "developer")]
    first_ok = bool(msgs) and inst == [0] and msgs[0].get("role") == "system"
    if not inst:
        first_ok = True             # no instruction message at all is fine
    why = f"instruction messages at {inst}, roles {[msgs[i].get('role') for i in inst]}"
    if addendum:
        head = selection._text(msgs[0]) if msgs else ""
        has = proxy.ADDENDUM.strip()[:60] in head
        first_ok = first_ok and has
        why += f", addendum in the first: {has}"
    try:
        render(out)
    except Exception as e:                                       # noqa: BLE001
        return False, why + f"; the template raised: {e}"[:200]
    return first_ok, why + "; renders"


def c_roles(fx, want, cx):
    ok1, why1 = _roles_of(cx["out"], False)
    util = bool((cx["out"].get("_utility") or {}).get("utility"))
    if util:
        return ok1, f"as sent: {why1}"
    hi = cx.get("out_high")
    if hi is None:
        hi = cx["out_high"] = prepared(fx, cx["chat"], effort="high",
                                       tag=cx.get("tag", "") + "#high")
    ok2, why2 = _roles_of(hi, True)
    ok = ok1 and ok2
    if want.get("addendum"):
        ok = ok and ok1 and _roles_of(cx["out"], True)[0]
    return ok, f"as sent: {why1} | at high: {why2}"


def c_image(fx, want, cx):
    att = cx["out"].get("_attached") or {}
    forms = sorted({e.get("form") for e in (att.get("images") or {}).values()})
    left = [(i, p.get("type")) for i, m in enumerate(cx["out"].get("messages")
                                                     or [])
            if isinstance(m, dict) and isinstance(m.get("content"), list)
            for p in m["content"] if isinstance(p, dict)
            and p.get("type") in image_input.IMAGE_PARTS]
    ok = forms == sorted(want.get("forms") or []) and not left
    return ok, f"forms={forms} image parts left={left}"


def c_tool_code(fx, want, cx):
    msgs = cx["chat"].get("messages") or []
    oks, gots = [], []
    for spec in want:
        call = _calls(msgs, spec["call"])
        if call is None:
            oks.append(False)
            gots.append(f"no call at {spec['call']}")
            continue
        d = tool_code.detect(call)
        rv = tool_code.review([json.loads(json.dumps(call))])
        units = d.get("units") or []
        g = {"name": d.get("name"), "detected": d.get("detected"),
             "kinds": sorted({u.get("kind") for u in units}),
             "languages": sorted({str(u.get("language")) for u in units}),
             "blocking": bool(rv.get("blocked")), "errors": rv.get("errors")}
        ok = g["detected"] == spec.get("detected", g["detected"]) and bool(units)
        if "kind" in spec:
            ok = ok and g["kinds"] == [spec["kind"]]
        if "language" in spec:
            ok = ok and g["languages"] == [spec["language"]]
        if "blocking" in spec:
            ok = ok and g["blocking"] == spec["blocking"]
        oks.append(ok)
        gots.append(str(g))
    enabled = bool(cx["out"].get("_tool_code"))
    return all(oks) and enabled, f"checked on this request: {enabled}; " \
        + " | ".join(gots)


_POOL: list[dict] | None = None


def c_skills(fx, want, cx):
    pool = _POOL or []
    rc = (cx["out"].get("_route") or {}).get("class")
    tools = proxy.client_tool_names(cx["chat"])
    chosen, rec = skill_select.select(cx["raw"], rc, pool, tools=tools)
    direct = sorted(c.get("name") for c in chosen)
    srec = cx["out"].get("_skills") or {}
    via = sorted(n for n in (srec.get("names") or []) if n)
    ok = True
    for n in want.get("includes") or []:
        ok = ok and n in direct and n in via
    for n in want.get("excludes") or []:
        ok = ok and n not in direct and n not in via
    return ok, (f"select()={direct} prepare={via} "
                f"(prepare: {str(srec.get('why') or 'no decision')[:90]})")


def _prefixes(fx: dict) -> list[dict]:
    """The requests that led to this one: the body cut after each run of
    tool results (Responses function_call_output items, chat `tool`
    messages), in order -- how the harness sent the conversation, one step
    at a time. The last is the fixture's own request."""
    body = fx["body"]
    key = "input" if fx["wire"] == "responses" else "messages"
    items = body.get(key) or []

    def is_result(it):
        return isinstance(it, dict) and (
            it.get("type") in ("function_call_output",
                               "custom_tool_call_output")
            or it.get("role") == "tool")
    out = []
    for k, it in enumerate(items):
        if is_result(it) and (k + 1 == len(items)
                              or not is_result(items[k + 1])):
            out.append(dict(json.loads(json.dumps(body)), **{key: items[:k + 1]}))
    return out


def c_project(fx, want, cx):
    """The PROJECT (mcp/progress.py): every prefix that ends on tool results
    replayed in order through the wire's translation (responses_api.to_chat)
    and proxy.prepare, on one account and session, as the harness sends the
    conversation. Checks the last request's root, the project writes seen
    over all of them, and that the tool result the model is sent is the
    harness's own text (no line of ours: removed 2026-09-27)."""
    recs, last_out, last_chat = [], None, None
    tag = cx.get("tag", "") + "#project"
    for b in _prefixes(fx):
        chat, why = chat_of(dict(fx, body=b))
        if chat is None:
            return False, f"a prefix was refused: {why}"
        last_chat = chat
        last_out = prepared(fx, chat, tag=tag)
        recs.append((last_out.get("_progress") or {}).get("project") or {})
    if not recs:
        return False, "no prefix ends on a tool result"
    last = recs[-1]
    writes = sorted({p for r in recs
                     for p in (r.get("writes") or {}).get("project_writes")
                     or []})
    tail = selection._text((last_out.get("messages") or [{}])[-1])
    sent = selection._text((last_chat.get("messages") or [{}])[-1])
    got = {"root": last.get("root"), "project_writes_seen": writes,
           "scratch": [s.get("path") for r in recs
                       for s in (r.get("writes") or {}).get(
                           "scratch_writes") or []],
           "tool_result_as_sent": tail.startswith(sent)
           and "Unchanged since step" not in tail
           and "No project file has changed" not in tail}
    # Every tool result pairs with the call it answers (the step kind and
    # a write's success read the result by its call's id).
    ids = {c.get("id") for m in last_chat.get("messages") or []
           if m.get("role") == "assistant" for c in m.get("tool_calls") or []}
    got["paired"] = all(m.get("tool_call_id") in ids
                        for m in last_chat.get("messages") or []
                        if m.get("role") == "tool")
    ok = got["tool_result_as_sent"] and got["paired"]
    if "root" in want:
        ok = ok and got["root"] == want["root"]
    if "project_writes_seen" in want:
        ok = ok and writes == sorted(want["project_writes_seen"])
    return ok, json.dumps(got)


def c_tools(fx, want, cx):
    """NO CONFLICTS (operator, 2026-09-27): every tool of ours that goes on
    main -- yama_generate_image, yama_describe_image, yama_think_deeply, yama_recall_craft, the
    delegate arm -- is compared with the harness's own; the final list has
    no duplicate or normalised-duplicate name and no declared overlap
    (proxy.TOOL_OVERLAPS), and each withheld tool names the client tool it
    yielded to. Served at `max` with an image server configured, so every
    tool of ours is a candidate."""
    if not (cx["chat"] or {}).get("tools"):
        return True, "no client tools"
    os.environ["YAMADORI_IMAGEGEN_URL"] = "http://127.0.0.1:9"
    try:
        out = prepared(fx, cx["chat"], effort="max", tag=cx["tag"] + "#tools")
    finally:
        os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
    names = [((t.get("function") or {}).get("name") or t.get("name") or "")
             for t in out.get("tools") or [] if isinstance(t, dict)]
    norm = [proxy._norm_tool(n) for n in names]
    dup = sorted({n for n in norm if norm.count(n) > 1})
    over = [(o, c) for o, cl, _w in proxy.TOOL_OVERLAPS for c in cl
            if o in names and c in names]
    held = out.get("_tools_withheld") or []
    named = all(w.get("client_tool") in names for w in held
                if w.get("client_tool"))
    want_held = set((want or {}).get("withheld") or [])
    got_held = {w["ours"] for w in held}
    ok = not dup and not over and named and want_held <= got_held
    return ok, (f"ours={sorted(out.get('_ours') or [])} withheld="
                f"{[(w['ours'], w.get('client_tool')) for w in held]} "
                f"dup={dup} overlap={over}")


CHECKS = {"accepted": c_accepted, "route": c_route, "utility": c_utility,
          "compaction": c_compaction, "session": c_session, "deep": c_deep,
          "roles": c_roles, "image": c_image, "tool_code": c_tool_code,
          "skills": c_skills, "project": c_project, "tools": c_tools}


def evaluate(fx: dict, points=None, tag: str = "") -> list[tuple]:
    """[(point, ok, got)] for one fixture: its request served once through
    prepare (an account of its own per `tag`), then each decision point."""
    chat, why = chat_of(fx)
    cx = {"chat": chat, "why": why, "tag": tag}
    out_err = None
    if chat is not None:
        try:
            cx["out"] = prepared(fx, chat, tag=tag)
            cx["raw"] = _raw(chat)
        except Exception as e:                                   # noqa: BLE001
            out_err = f"prepare raised {type(e).__name__}: {e}"
            traceback.print_exc()
    res = []
    for p in POINTS:
        if (p not in fx["expect"] and p not in UNIVERSAL) or (
                points and p not in points):
            continue
        want = fx["expect"].get(p)
        if want is None:
            want = ((UNIVERSAL_WANT.get(fx.get("harness")) or {}).get(p)
                    if (fx.get("body") or {}).get("tools") else None) \
                or UNIVERSAL[p]
        if p != "accepted" and (chat is None or out_err):
            ok, g = False, out_err or f"refused: {why}"
        else:
            try:
                if p == "roles":
                    cx.pop("out_high", None)
                    cx["tag"] = tag
                ok, g = CHECKS[p](fx, want, cx)
            except Exception as e:                               # noqa: BLE001
                ok, g = False, f"the check raised {type(e).__name__}: {e}"
                traceback.print_exc()
        res.append((p, ok, g))
    return res


# ============================================================== MUTANTS ====
# THE GATE MUST BE ABLE TO FAIL (PROTOCOL rules 3 and 14). Each bug this
# week's harness tests found is fixed in the tree by now, so its row passes;
# a mutant puts the OLD behaviour back, in-process, and the gate must catch
# it on the fixtures that carry that harness's shape. A mutant the gate
# misses is a FAIL: the column would not have seen the bug.

def _old_add_addendum(messages, think=False):
    """add_addendum before 2026-09-26: only a `system` message was joined."""
    text = proxy.addendum_text(think)
    out = list(messages)
    if out and out[0].get("role") == "system" and isinstance(
            out[0].get("content"), str):
        out[0] = dict(out[0], content=out[0]["content"] + text)
        return out
    return [{"role": "system", "content": text.strip()}] + out


def _old_harness_of(text, system=""):
    t = text or ""
    return "hermes" if compaction._START.search(t) and "summar" in \
        t[:2000].lower() else None


_ORIG_CALL_OF = responses_api._call_of

_ROO_OPENCODE_EDIT = [r for r in tool_code.KNOWN["edit"]
                      if r.get("kind") == "replace"
                      and r.get("path") in ("file_path", "filePath")]

MUTANTS = (
    ("developer role unmapped, addendum joins `system` only",
     [(system_roles, "one_system", lambda msgs: (msgs, {})),
      (proxy, "add_addendum", _old_add_addendum)],
     [("pi/chat/developer-role-high", "roles")]),
    ("synthetic tool-media turns read as user turns",
     [(image_input, "tool_media_turn", lambda messages, i: None)],
     [("opencode/chat/read-image-synthetic-turn", "route"),
      ("opencode/chat/read-image-synthetic-turn", "deep"),
      ("opencode/chat/read-image-synthetic-turn", "image"),
      ("pi/chat/read-image-synthetic-turn", "route"),
      ("pi/chat/read-image-synthetic-turn", "deep"),
      ("pi/chat/read-image-synthetic-turn", "image")]),
    ("a title call read by the summarise rule",
     [(selection, "names_a_title", lambda text: False)],
     [("opencode/chat/title-call", "utility"),
      ("opencode/responses/title-call", "utility")]),
    ("only Hermes' flattened compaction recognised",
     [(compaction, "parse_transcript", lambda text: None),
      (compaction, "harness_of", _old_harness_of)],
     [("pi/chat/compaction-history", "compaction"),
      ("pi/chat/compaction-split-turn", "compaction"),
      ("opencode/chat/compaction", "compaction")]),
    ("Pi's edit rows missing from the known-names table",
     [(tool_code.KNOWN, "edit", _ROO_OPENCODE_EDIT)],
     [("pi/chat/edit-shapes", "tool_code")]),
    (".html writes carry no checkable code",
     [(tool_code, "is_html", lambda path: False)],
     [("opencode/chat/write-html-broken-script", "tool_code"),
      ("pi/chat/write-html", "tool_code")]),
    ("the working directory from the first write's own folder (v0f-V0: a "
     "bare config.js made it .../space-shooter/js)",
     [(progress, "tree_paths", lambda text: ([], set())),
      (progress, "_common_root", lambda a, b: None)],
     [("hermes/responses/reread-unchanged", "project")]),
    ("a Responses call's id is not the call_id its output names (the "
     "suspected v0f-V0 cause: a tool result the project cannot pair)",
     [(responses_api, "_call_of", lambda item, where, custom, flat_of=None:
       dict(_ORIG_CALL_OF(item, where, custom, flat_of),
            id="fc_" + str(item.get("call_id"))))],
     [("hermes/responses/reread-unchanged", "project")]),
    ("tools of ours added whatever the harness offers (no conflict "
     "check): yama_describe_image beside Hermes' vision_analyze",
     [(proxy, "tool_conflicts", lambda ours, client: (list(ours), []))],
     [("hermes/chat/first-turn-task", "tools"),
      ("hermes/chat/agent-step-write-js", "tools")]),
    ("harness session headers ignored",
     [(server, "SESSION_HEADERS", ("x-yamadori-session",))],
     [("opencode/chat/first-turn", "session"),
      ("opencode/chat/fork-own-session", "session"),
      ("pi/chat/affinity-headers", "session")]),
)


@contextlib.contextmanager
def _patched(patches):
    saved = []
    try:
        for obj, name, new in patches:
            if isinstance(obj, dict):
                saved.append((obj, name, obj[name]))
                obj[name] = new
            else:
                saved.append((obj, name, getattr(obj, name)))
                setattr(obj, name, new)
        yield
    finally:
        for obj, name, old in reversed(saved):
            if isinstance(obj, dict):
                obj[name] = old
            else:
                setattr(obj, name, old)


# ================================================================= main ====
def main(argv: list[str]) -> int:
    global _POOL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-v", action="store_true", help="print every check")
    ap.add_argument("-k", default="", help="only fixtures whose id has this")
    a = ap.parse_args(argv)

    with contextlib.redirect_stdout(io.StringIO()):
        got = skill_migrate.install_authored()
    _POOL = skills.armed()
    armed = sorted(s["name"] for s in _POOL)
    fixtures = load(a.k)
    print(f"  {len(fixtures)} fixtures under {os.path.relpath(SHAPES, ROOT)}"
          f"; authored skills armed: {armed} "
          f"({sum(g.get('status') == 'armed' for g in got)}/{len(got)})")
    results: list[dict] = []           # {row, point, id, status, got, want}
    bad_setup = 0
    if not fixtures:
        print("FAIL: no fixtures found")
        bad_setup += 1
    if not armed:
        print("FAIL: no authored skill armed: the skills column cannot be "
              "checked")
        bad_setup += 1
    ids = [fx["id"] for fx in fixtures]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        print(f"FAIL: duplicate fixture id {dup}")
        bad_setup += 1
    by_id = {fx["id"]: fx for fx in fixtures}
    for fx in fixtures:
        row = f"{fx['harness']}/{fx['wire']}"
        for p in fx["expect"]:
            if p not in CHECKS:
                print(f"FAIL: {fx['id']}: unknown decision point {p!r}")
                bad_setup += 1
        for p, ok, g in evaluate(fx):
            want = fx["expect"].get(p, UNIVERSAL.get(p))
            known = KNOWN.get((fx["id"], p))
            status = ("pass" if ok and not known else
                      "now-passes" if ok and known else
                      "known" if known else "FAIL")
            results.append({"row": row, "point": p, "id": fx["id"],
                            "status": status, "got": g, "want": want,
                            "known": known})
            w = json.dumps(want, ensure_ascii=False)
            if status == "FAIL":
                print(f"FAIL: {fx['id']} [{p}] want {w[:160]} -- got {g[:400]}")
                print(f"        why expected: "
                      f"{str((fx.get('why') or {}).get(p, ''))[:200]}")
            elif status == "known":
                print(f"  KNOWN-FAIL  {fx['id']} [{p}] want {w[:120]} -- got "
                      f"{g[:260]}\n              reason: {known}")
            elif status == "now-passes":
                print(f"  KNOWN-FAIL NOW PASSES  {fx['id']} [{p}]: remove "
                      f"its KNOWN mark ({known[:100]})")
            elif a.v:
                print(f"  ok  {fx['id']} [{p}] {g[:200]}")
    for key in KNOWN:
        if key[0] not in ids and not a.k:
            print(f"FAIL: KNOWN names a fixture that does not exist: {key}")
            bad_setup += 1

    # ----------------------------------------------------------- mutants
    caught = 0
    mutant_checks = 0
    for name, patches, targets in MUTANTS:
        for i, p in targets:
            if i not in by_id and not a.k:
                print(f"FAIL: mutant '{name}' names a fixture that does not "
                      f"exist: {i}")
                bad_setup += 1
        live = [(i, p) for i, p in targets if i in by_id]
        if not live:
            continue
        with _patched(patches):
            for i, p in live:
                mutant_checks += 1
                res = evaluate(by_id[i], points=(p,), tag=f"#mut:{name}")
                ok = bool(res) and res[0][1]
                if not ok:
                    caught += 1
                    if a.v:
                        print(f"  ok  mutant '{name}' caught on {i} [{p}]: "
                              f"{res[0][2][:150] if res else 'no result'}")
                else:
                    print(f"FAIL: mutant '{name}' NOT caught on {i} [{p}]: "
                          f"the column passes with the old behaviour back "
                          f"({res[0][2][:200]})")
    print(f"  mutants (the old behaviour of each fixed bug, put back): "
          f"caught {caught}/{mutant_checks}")

    # ---------------------------------------------------------- the matrix
    rows = sorted({r["row"] for r in results})
    w0 = max([len(r) for r in rows] + [14])
    cw = {p: max(len(p), 5) for p in POINTS}
    print()
    print("  " + "harness/wire".ljust(w0) + "  "
          + "  ".join(p.ljust(cw[p]) for p in POINTS))
    for row in rows:
        cells = []
        for p in POINTS:
            st = [r["status"] for r in results
                  if r["row"] == row and r["point"] == p]
            cell = ("-" if not st else "FAIL" if "FAIL" in st else
                    "known" if "known" in st else "pass")
            cells.append(cell.ljust(cw[p]))
        print("  " + row.ljust(w0) + "  " + "  ".join(cells))
    print("  (pass: every check passed; known: a KNOWN failure, not gating; "
          "FAIL: a new miss; -: no fixture checks it)")
    print()
    gating = [r for r in results if r["status"] != "known"]
    passed = sum(r["status"] in ("pass", "now-passes") for r in gating)
    n_known = sum(r["status"] == "known" for r in results)
    print(f"  known failures (listed above, not gating): {n_known}; "
          f"known marks that now pass: "
          f"{sum(r['status'] == 'now-passes' for r in results)}")
    passed += caught
    total = len(gating) + mutant_checks + bad_setup
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total and total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
