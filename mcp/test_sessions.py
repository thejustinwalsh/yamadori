#!/usr/bin/env python
"""A conversation's identity is an explicit session id (#41). No GPU.

    python mcp/test_sessions.py      -> "N/M checks passed"

THE DEFECT (Octopus v0d-V0-xhigh-1, 2026-09-25). A conversation was keyed by
a hash of its account and first two messages (nebari.key_of). Every Octopus
V0 run opens with the same Hermes system prompt and task from the same
account, so run v0d's first request was filed as a continuation of v0b/v0c:
x_yamadori.deep read kickoff done: true, epoch 2, episode 5,
requests_since_run 4 -- and the kickoff plan never ran.

THE RULE (operator, 2026-09-25; mcp/session_id.py): identity comes from
prompt_cache_key, else X-Yamadori-Session, else OUR ID -- carried in the
tool-call ids the proxy returns (`call_<id>_<8 hex>`), or on the line of a
compaction summary the proxy wrote, or recorded with a text-only answer of
ours (proxy THE ANSWER RECORD) -- else a new conversation, minted. No
visible line in answers: the first design put `yamadori session <id>` on the
first answer and broke exact-output callers live ("Reply with exactly: ok"
-> "yamadori session dda0ff896a3c\\n\\nok").

Gated here, through the real complete() / stream_body() and the SERVED chat
template (test_ledger's harness: a fake upstream, a client that keeps what
it was sent as content and tool calls, drops reasoning, and echoes the call
ids it was given, like Hermes):
  1. v0d: a finished conversation, then the same opening -> a new id, a
     fresh deep state (the kickoff fires), no ledger replay, its own slot;
  2. two interleaved conversations with one opening stay apart by id;
  3. a retried opening is a new conversation; the answer the client kept
     names it (answer record);
  4. the compaction summary carries the id, and a continuation that keeps
     only the summary keeps the conversation (the Hermes shapes are in
     mcp/test_utility.py; the in-place form is here);
  5. prompt_cache_key and the header win over our id; the carrier wins over
     the summary line; ids we did not mint are ignored;
  6. the ids round-trip through a Hermes-style history, blocking and
     streamed (the first delta carrying a call carries the id), the
     template renders no id (the prompt is identical to the original-id
     prompt, so the slot's prefix extends), every tool result pairs with a
     call upstream, and an exact-output answer is the model's text byte for
     byte;
  7. a conversation whose first answer makes no call keeps its conversation
     on the next request (the answer record), its slot and its prefix.

Every database is a temp file set BEFORE proxy is imported.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_sessions_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json")):
    os.environ[_k] = os.path.join(_TMP, _v)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import test_ledger as T  # noqa: E402  (its own temp paths + fake upstream)
import deep  # noqa: E402
import nebari  # noqa: E402
import proxy  # noqa: E402
import session_id  # noqa: E402
import shomen  # noqa: E402

_tmp_root = os.path.abspath(tempfile.gettempdir())
for _k in ("YAMADORI_CORPUS_DB", "YAMADORI_NEBARI_DB", "RINGS_DB",
           "CODE_INDEX_DB"):
    assert os.path.abspath(os.environ[_k]).startswith(_tmp_root), _k
assert os.path.abspath(nebari.DB).startswith(_tmp_root), nebari.DB

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def _key(account: str) -> str:
    return (proxy._LAST_SESSION.get(account) or ("", 0, ""))[0]


def _lineage(account: str) -> str:
    return (proxy._LAST_SESSION.get(account) or ("", 0, ""))[2]


def _ses(t: dict) -> dict:
    return (t["d"].get("x_yamadori") or {}).get("session") or {}


def _slot(t: dict):
    return t["gens"][0]["request"].get("id_slot")


def _ids(msg: dict) -> list[str]:
    return [c.get("id") for c in msg.get("tool_calls") or []]


def _pairs_ok(messages: list[dict]) -> tuple[bool, list]:
    """Every tool message answers a call of the assistant turn before it,
    and every call id is unique: the pairing an OpenAI-compatible server
    (llama-server's DeepSeek V4 sort, a strict provider) relies on."""
    bad, open_ids, seen = [], set(), set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            open_ids = set(_ids(m))
            for i in _ids(m):
                if i in seen:
                    bad.append(("duplicate", i))
                seen.add(i)
        elif m.get("role") == "tool":
            if m.get("tool_call_id") not in open_ids:
                bad.append(("unpaired", m.get("tool_call_id")))
    return not bad, bad


SPEC = ("Build a space shooter with vanilla JavaScript and canvas. "
        "PLAYER: moves with arrow keys, fires with space. " * 120)


def _plan_post(real):
    def post(path, payload, timeout=3600):
        if str(payload["messages"][0].get("content", "")).startswith(
                "You are planning"):
            T._helper.append(json.loads(json.dumps(payload)))
            return {"choices": [{"message": {"content": T.PLAN},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return real(path, payload, timeout)
    return post


def test_the_same_opening_after_a_finished_run_is_a_new_conversation():
    """(1) Octopus v0d: run 1 kicks off, works, finishes; run 2 sends the
    byte-identical opening. Run 2 is a NEW conversation."""
    T.slots.reset(n=4)
    T.compaction.reset()
    T._helper.clear()
    real = shomen._post
    shomen._post = _plan_post(real)
    try:
        r1 = T.Unforced("xhigh", "octopus v0")
        acct = r1.body()["_account"]
        t1 = r1.turn([T.reply(" I will start with index.html.", calls=[T.call(
            "write_file", {"path": "index.html", "content": "<canvas>"},
            "v0c-w1")])], user=SPEC)
        lin1 = _lineage(acct)
        r1.tool_result("v0c-w1", "wrote index.html")
        r1.turn([T.reply("", calls=[T.call(
            "write_file", {"path": "js/game.js", "content": "loop()"},
            "v0c-w2")])])
        r1.tool_result("v0c-w2", "wrote js/game.js")
        r1.turn([T.reply("Done: the game is in index.html and js/game.js.")])
        d1 = t1["d"]["x_yamadori"]["deep"]
        check(d1.get("kind") == "kickoff" and d1.get("fire"),
              "run 1: the opening kicks off (the plan job)",
              json.dumps(d1.get("signals", {}).get("kickoff")))
        st1 = deep.load_state(acct, lin1)
        check(int(st1.get("episode") or 0) >= 1 and st1.get("kickoffs"),
              "run 1's deep state has moved on (an episode, a kickoff done)",
              json.dumps({k: st1.get(k) for k in ("episode", "kickoffs")}))

        r2 = T.Unforced("xhigh", "octopus v0")        # the same opening
        n_helper = len(T._helper)
        t2 = r2.turn([T.reply(" Starting with the page.", calls=[T.call(
            "write_file", {"path": "index.html", "content": "<canvas>"},
            "v0d-w1")])], user=SPEC)
        lin2 = _lineage(acct)
    finally:
        shomen._post = real
    x2 = t2["d"]["x_yamadori"]
    d2 = x2.get("deep") or {}
    ko = (d2.get("signals") or {}).get("kickoff") or {}
    check(d2.get("kind") == "kickoff" and d2.get("fire")
          and ko.get("done") is False and len(T._helper) > n_helper,
          "run 2's opening kicks off again (v0d: kickoff done: true, no plan)",
          json.dumps({"fire": d2.get("fire"), "kind": d2.get("kind"),
                      "kickoff": ko}))
    check(d2.get("epoch") == 0 and d2.get("episode") == 0
          and (d2.get("last_run") or {}).get("requests_since_run") is None,
          "run 2's deep state is fresh: epoch 0, episode 0, no run before "
          "(v0d: epoch 2, episode 5, requests_since_run 4)",
          json.dumps({k: d2.get(k) for k in ("epoch", "episode", "last_run")}))
    s1, s2 = _ses(t1), _ses(t2)
    check(lin2 and lin2 != lin1 and s2.get("source") == "minted"
          and s2.get("id") and s2.get("id") != s1.get("id"),
          "run 2 is a new conversation: a new id, a key and lineage of its "
          "own", json.dumps([s1, s2]))
    check(session_id.of_call_id(_ids(r1.msgs[2])[0]) == s1.get("id")
          and session_id.of_call_id(_ids(r2.msgs[2])[0]) == s2.get("id")
          and s1.get("carried") == s2.get("carried") == 1,
          "each run's first call carries its own id",
          json.dumps([_ids(r1.msgs[2]), _ids(r2.msgs[2])]))
    inj = (x2.get("ledger") or {}).get("inject") or {}
    check(inj.get("decided") and not inj.get("replayed"),
          "run 2's opening injection is DECIDED for it, not replayed from "
          "run 1's ledger", json.dumps(inj))
    c2 = x2.get("cache") or {}
    check(_slot(t2) is not None and _slot(t2) != _slot(t1)
          and c2.get("mode") == "pinned" and not c2.get("adopted")
          and not c2.get("evicted"),
          "run 2 gets a slot pin of its own; run 1's slot keeps its prefix",
          json.dumps({"run1": _slot(t1), "run2": _slot(t2), "cache": c2}))
    r2.tool_result("v0d-w1", "wrote index.html")
    t3 = r2.turn([T.reply("Page written.")])
    check(_lineage(acct) == lin2 and _ses(t3).get("source") == "tool_call_id"
          and _ses(t3).get("id") == s2.get("id") and _slot(t3) == _slot(t2),
          "run 2's next request is run 2's: its id read back from the call "
          "id it echoed, on its slot", json.dumps(_ses(t3)))


def test_interleaved_conversations_with_one_opening_keep_apart():
    """(2) A and B open identically and interleave: each keeps its own key,
    slot and deep state by its id."""
    T.slots.reset(n=4)
    T.compaction.reset()
    a = T.Unforced("medium", "interleaved")
    b = T.Unforced("medium", "interleaved")
    acct = a.body()["_account"]
    ask = "List the files in the project."
    ta1 = a.turn([T.reply("", calls=[T.call("write_file", {
        "path": "notes.md", "content": "a"}, "A1")])], user=ask)
    la = _lineage(acct)
    tb1 = b.turn([T.reply("", calls=[T.call("write_file", {
        "path": "todo.md", "content": "b"}, "B1")])], user=ask)
    lb = _lineage(acct)
    a.tool_result("A1", "ok")
    ta2 = a.turn([T.reply("", calls=[T.call("write_file", {
        "path": "notes2.md", "content": "a"}, "A2")])])
    la2 = _lineage(acct)
    b.tool_result("B1", "ok")
    tb2 = b.turn([T.reply("", calls=[T.call("write_file", {
        "path": "todo2.md", "content": "b"}, "B2")])])
    lb2 = _lineage(acct)
    a.tool_result("A2", "ok")
    ta3 = a.turn([T.reply("A is done.")])
    la3 = _lineage(acct)
    b.tool_result("B2", "ok")
    tb3 = b.turn([T.reply("B is done.")])
    lb3 = _lineage(acct)
    check(la and lb and la != lb and la2 == la3 == la and lb2 == lb3 == lb,
          "two conversations with one opening, interleaved: each request "
          "lands in its own conversation",
          json.dumps([la[:8], lb[:8], la2[:8], lb2[:8], la3[:8], lb3[:8]]))
    check(all(_ses(t).get("source") == "tool_call_id"
              for t in (ta2, ta3, tb2, tb3))
          and _ses(ta3)["id"] == _ses(ta1)["id"]
          and _ses(tb3)["id"] == _ses(tb1)["id"]
          and _ses(ta1)["id"] != _ses(tb1)["id"],
          "x_yamadori.session: each by its own id, read from the call ids it "
          "echoed", json.dumps([_ses(ta3), _ses(tb3)]))
    check(_slot(ta1) == _slot(ta2) == _slot(ta3)
          and _slot(tb1) == _slot(tb2) == _slot(tb3)
          and _slot(ta1) != _slot(tb1),
          "each keeps its own slot",
          json.dumps([_slot(t) for t in (ta1, ta2, ta3, tb1, tb2, tb3)]))
    sa, sb = deep.load_state(acct, la), deep.load_state(acct, lb)
    check(sa.get("req") == 3 and sb.get("req") == 3,
          "each deep state counted its own three requests (one shared "
          "state would count six)",
          json.dumps({"A": sa.get("req"), "B": sb.get("req")}))


def test_a_retried_opening_is_a_new_conversation():
    """(3) The same opening sent twice (a retry after a disconnect): two
    conversations, two ids -- nothing is resumed, nothing guessed. The
    answers are the model's text exactly (no line); the one the client kept
    names its conversation (the answer record)."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = T.Unforced("medium", "retry")
    acct = c.body()["_account"]
    c.msgs.append({"role": "user", "content": "Explain the build."})
    T._script[:] = [T.reply("First try.")]
    d1 = proxy.complete(c.body())
    k1 = _key(acct)
    T._script[:] = [T.reply("Second try.")]
    d2 = proxy.complete(c.body())
    k2 = _key(acct)
    s1 = d1["x_yamadori"]["session"]
    s2 = d2["x_yamadori"]["session"]
    check(k1 != k2 and s1["source"] == s2["source"] == "minted"
          and s1["id"] != s2["id"] and not s1.get("line")
          and not s2.get("line"),
          "a retried opening mints another id: a new conversation",
          json.dumps([s1, s2]))
    a1 = d1["choices"][0]["message"]["content"]
    a2 = d2["choices"][0]["message"]["content"]
    check(a1 == "First try." and a2 == "Second try.",
          "each answer is the model's text exactly: no session line",
          json.dumps([a1, a2]))
    # The client keeps the SECOND answer and goes on: that conversation.
    c.msgs.append({"role": "assistant", "content": a2})
    c.msgs.append({"role": "user", "content": "And the tests?"})
    T._script[:] = [T.reply("make test.")]
    d3 = proxy.complete(c.body())
    check(_key(acct) == k2 and d3["x_yamadori"]["session"]["id"] == s2["id"]
          and d3["x_yamadori"]["session"]["source"] == "answer_record",
          "the conversation the client kept goes on under its id (the answer "
          "it kept is on record with it)",
          json.dumps(d3["x_yamadori"]["session"]))


def test_prompt_cache_key_and_the_header_win():
    """(5) An id the client sends itself wins over ours: prompt_cache_key
    first, then X-Yamadori-Session. The carrier wins over a summary line;
    ids we did not mint name nothing."""
    T.slots.reset(n=4)
    c = T.Unforced("medium", "explicit")
    acct = c.body()["_account"]
    msgs = [c.msgs[0], {"role": "user", "content": "hi"}]
    k, s = proxy.session_identity(msgs, acct, header="row-1",
                                  cache_key="conv-123")
    check(s["source"] == "prompt_cache_key"
          and k == session_id.conversation_key(acct, "prompt_cache_key",
                                               "conv-123"),
          "prompt_cache_key wins over the header", json.dumps(s))
    k, s = proxy.session_identity(msgs, acct, header="row-1")
    check(s["source"] == "header" and k == nebari.key_of(msgs, acct, "row-1"),
          "the header: the key as it always was (key_of with the token)",
          json.dumps(s))
    ours = session_id.call_id("0123456789ab")
    carried = msgs + [{"role": "assistant", "content": "", "tool_calls": [
        T.call("write_file", {"path": "a"}, ours)]},
        {"role": "tool", "tool_call_id": ours, "content": "ok"}]
    k1, s1 = proxy.session_identity(carried, acct, cache_key="conv-123")
    k2, s2 = proxy.session_identity(carried, acct, header="row-1")
    k3, s3 = proxy.session_identity(carried, acct)
    check(s1["source"] == "prompt_cache_key" and s2["source"] == "header"
          and s3["source"] == "tool_call_id" and s3["id"] == "0123456789ab"
          and len({k1, k2, k3}) == 3,
          "a history carrying our id: prompt_cache_key, then the header, "
          "win over it", json.dumps([s1, s2, s3]))
    both = carried + [{"role": "user", "content": session_id.line(
        "fedcba987654") + "## Summary\nwrote a"}]
    _k, sb = proxy.session_identity(both, acct)
    only_line = msgs + [{"role": "user", "content": session_id.line(
        "fedcba987654") + "## Summary\nwrote a"}]
    _k, sl = proxy.session_identity(only_line, acct)
    check(sb["source"] == "tool_call_id" and sb["id"] == "0123456789ab"
          and sl["source"] == "summary_line" and sl["id"] == "fedcba987654",
          "the carrier wins over a summary line; a summary line alone names "
          "the conversation", json.dumps([sb, sl]))
    # Ids we did not mint: llama-server's (32 alphanumerics), OpenAI's,
    # Hermes' deterministic fallback (call_<12 hex>), a client's own.
    foreign = ["aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE", "call_abc123XYZdef456",
               "call_0123456789ab", "toolu_01ABC", "call_0123456789ab_xyz"]
    check(all(session_id.of_call_id(i) is None for i in foreign)
          and session_id.of_call_id(ours + "_d2") == "0123456789ab",
          "ids we did not mint name no conversation; Hermes' duplicate "
          "repair (`_d<n>`) keeps ours readable", json.dumps(foreign))
    fh = msgs + [{"role": "assistant", "content": "", "tool_calls": [
        T.call("write_file", {"path": "a"}, foreign[0])]},
        {"role": "tool", "tool_call_id": foreign[0], "content": "ok"}]
    _k, sf = proxy.session_identity(fh, acct)
    check(sf["source"] == "minted",
          "a history whose ids are someone else's: a new conversation, not "
          "a guess", json.dumps(sf))
    # Through complete(): an opening that names itself keeps its text as
    # written; its call ids carry the CARRIER of its key (2026-09-26,
    # session_id.carrier_of: a key that changes mid-conversation -- Hermes'
    # Responses key -- still finds it through them).
    T._script[:] = [T.reply("Hello.", calls=[T.call(
        "write_file", {"path": "x"}, "upstream-id-1")])]
    d = proxy.complete(dict(c.body(), messages=msgs,
                            prompt_cache_key="conv-123"))
    m = d["choices"][0]["message"]
    xs = d["x_yamadori"]["session"]
    check(m["content"] == "Hello."
          and session_id.of_call_id(_ids(m)[0]) == xs.get("carrier")
          == session_id.carrier_of(session_id.conversation_key(
              acct, "prompt_cache_key", "conv-123"))
          and xs["source"] == "prompt_cache_key" and xs["carried"] == 1,
          "an opening that names itself (prompt_cache_key): no line; its "
          "call ids carry its key's carrier", json.dumps(xs))
    T._script[:] = [T.reply("Hello.")]
    d = proxy.complete(dict(c.body(), messages=msgs, _session_token="row-9"))
    check(d["choices"][0]["message"]["content"] == "Hello."
          and d["x_yamadori"]["session"]["source"] == "header",
          "an opening with the header: no line",
          json.dumps(d["x_yamadori"]["session"]))


def test_the_ids_round_trip_and_the_slot_extends():
    """(6) Blocking and streamed: the client stores the calls with the ids it
    was given (content kept, reasoning dropped, as Hermes does) and echoes
    them on its tool results; the template renders no id, so the model
    never sees them and the next request extends what the slot holds."""
    fixture = open(os.path.join(HERE, "fixtures",
                                "bonsai_chat_template.jinja"),
                   encoding="utf-8").read()
    check("tool_call_id" not in fixture and ".id" not in fixture
          and "['id']" not in fixture,
          "the served chat template has no tool_call_id and no call id "
          "(mcp/fixtures/bonsai_chat_template.jinja)")
    T.slots.reset(n=4)
    T.compaction.reset()
    # Blocking.
    c = T.Unforced("medium", "round trip")
    t1 = c.turn([T.reply("Writing it.", reasoning="Write first.", calls=[
        T.call("write_file", {"path": "a.txt", "content": "x"}, "up-A"),
        T.call("write_file", {"path": "b.txt", "content": "y"}, "up-B")])],
        user="Write a.txt and b.txt.")
    sid = _ses(t1).get("id")
    got = _ids(c.msgs[-1])
    check(sid and len(got) == 2 and len(set(got)) == 2
          and all(session_id.of_call_id(i) == sid for i in got)
          and all(len(i) <= 32 and i.startswith("call_") for i in got)
          and _ses(t1).get("carried") == 2,
          "blocking: every call the client gets carries the id, unique, "
          "`call_` kept, 26 characters", json.dumps(got))
    check(c.msgs[-1]["content"] == "Writing it."
          and session_id.PREFIX not in c.msgs[-1]["content"],
          "blocking: the content is the model's text, no line",
          json.dumps(c.msgs[-1]["content"]))
    c.tool_result("up-A", "wrote a.txt")
    c.tool_result("up-B", "wrote b.txt")
    t2 = c.turn([T.reply("Both written.")])
    up = t2["gens"][0]["request"]["messages"]
    ok, bad = _pairs_ok(up)
    check(_ses(t2).get("id") == sid and _ses(t2).get("source")
          == "tool_call_id" and ok
          and [m.get("tool_call_id") for m in up if m.get("role") == "tool"]
          == got,
          "the next request finds the conversation by the ids it echoed; "
          "upstream, every tool result pairs with its call (the ids pass "
          "through as the client sent them)", json.dumps(bad))
    orig = {got[0]: "up-A", got[1]: "up-B"}
    as_generated = [dict(m, tool_calls=[dict(x, id=orig.get(x["id"], x["id"]))
                                        for x in m["tool_calls"]])
                    if m.get("tool_calls") else
                    dict(m, tool_call_id=orig.get(m["tool_call_id"]))
                    if m.get("role") == "tool" else m for m in up]
    req = t2["gens"][0]["request"]
    r_ours = T.render(req)
    r_orig = T.render(dict(req, messages=as_generated))
    check(r_ours == r_orig and not any(i in r_ours for i in got),
          "the template renders no id: the prompt with our ids is the prompt "
          "with the ids llama-server generated, byte for byte",
          f"{len(r_ours)} vs {len(r_orig)} chars")
    T._extends(t1, t2, "[carrier] blocking: the request after the calls")
    # Hermes' duplicate repair and a history where Hermes re-serialised the
    # calls: the id still reads.
    hermes = [dict(m) for m in c.msgs]
    hermes[2] = dict(hermes[2], tool_calls=[dict(x, id=x["id"] + "_d2")
                                            for x in hermes[2]["tool_calls"]])
    for m in hermes:
        if m.get("role") == "tool":
            m["tool_call_id"] = m["tool_call_id"] + "_d2"
    _k, sh = proxy.session_identity(hermes, c.body()["_account"])
    check(sh.get("id") == sid and sh.get("source") == "tool_call_id",
          "a Hermes-repaired id (`_d2`) still names the conversation",
          json.dumps(sh))
    # Streamed.
    msgs = [{"role": "system", "content": T.SYSTEM + " [stream carrier]"},
            {"role": "user", "content": "Write a.txt."}]
    body = {"model": "yamadori", "reasoning_effort": "medium",
            "_account": T.ACCOUNT + "-stream", "_client_ip": "127.0.0.1",
            "tools": [T.WRITE], "messages": json.loads(json.dumps(msgs)),
            "_features": json.dumps({"hints": True, "fanout": 1})}
    T._script[:] = [T.reply("On it.", reasoning="Look at it.", calls=[
        T.call("write_file", {"path": "a.txt", "content": "x"}, "up-S")])]
    n0, w0 = len(T._gens), len(T._warms)
    first_call_delta, content, x_session = None, "", None
    # The calls as a streaming client assembles them: the arguments arrive
    # in later deltas of the same index (the next request renders them).
    assembled: dict[int, dict] = {}
    for b in proxy.stream_body(dict(body, stream=True)):
        for line in b.decode("utf-8").split("\n"):
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip())
            if ev.get("x_yamadori"):
                x_session = ev["x_yamadori"].get("session")
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                content += dl.get("content") or ""
                if dl.get("tool_calls") and first_call_delta is None:
                    first_call_delta = dl["tool_calls"]
                for x in dl.get("tool_calls") or []:
                    a = assembled.setdefault(x.get("index", 0), {
                        "id": x.get("id"), "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    fn = x.get("function") or {}
                    a["id"] = a["id"] or x.get("id")
                    a["function"]["name"] += fn.get("name") or ""
                    a["function"]["arguments"] += fn.get("arguments") or ""
    # The warm the turn scheduled (the model wrote the call's arguments in
    # another order than the sorted one the next request renders): the
    # slot holds what it loaded.
    end = time.time() + 5
    while time.time() < end and len(T._warms) == w0:
        time.sleep(0.02)
    ts1 = {"gens": T._gens[n0:], "warm": T._warms[w0:]}
    sid_s = (x_session or {}).get("id")
    first_ids = [x.get("id") for x in first_call_delta or []]
    check(sid_s and first_ids
          and session_id.of_call_id(first_ids[0]) == sid_s
          and content == "On it.",
          "streamed: the FIRST delta carrying a call carries the id; the "
          "content is the model's text, no line",
          json.dumps({"ids": first_ids, "content": content,
                      "session": x_session}))
    msgs.append({"role": "assistant", "content": content,
                 "tool_calls": [assembled[i] for i in sorted(assembled)]})
    msgs.append({"role": "tool", "tool_call_id": first_ids[0] if first_ids
                 else "", "content": "wrote a.txt"})
    T._script[:] = [T.reply("Done.")]
    n1 = len(T._gens)
    d = T._stream(dict(body, messages=json.loads(json.dumps(msgs))))
    ts2 = {"gens": T._gens[n1:], "warm": []}
    check(d["content"] == "Done.", "streamed: the next answer, no line",
          json.dumps(d["content"]))
    T._extends(ts1, ts2, "[carrier] streamed: the request after the call")
    # EXACT OUTPUT (the live failure): the answer is the model's bytes.
    e = T.Unforced("medium", "exact")
    te = e.turn([T.reply("ok")], user="Reply with exactly: ok")
    T._script[:] = [T.reply("ok")]
    got_s = T._stream(dict(e.body(), messages=[
        e.msgs[0], {"role": "user", "content": "Reply with exactly: ok"}]))
    check(te["d"]["choices"][0]["message"]["content"] == "ok"
          and got_s["content"] == "ok",
          "\"Reply with exactly: ok\" -> \"ok\", blocking and streamed "
          "(live 2026-09-25: \"yamadori session dda0ff896a3c\\n\\nok\")",
          json.dumps([te["d"]["choices"][0]["message"]["content"],
                      got_s["content"]]))


def test_a_first_answer_without_a_call_keeps_its_conversation():
    """(7) A first answer that makes no client call carries nothing. It is
    recorded with the id (the answer record), so the next request is the
    same conversation: its key, slot and prefix."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = T.Unforced("xhigh", "text first")
    acct = c.body()["_account"]
    # Deep thinking forced off: every new task is planned at xhigh
    # (2026-09-27), and this test is about the answer's own text.
    off = {"investigate": False}
    t1 = c.turn([T.reply("It is a canvas game.", reasoning="Look first.")],
                user="What is this project?", features=off)
    k1, sid = _key(acct), _ses(t1).get("id")
    check(c.msgs[-1]["content"] == "It is a canvas game."
          and _ses(t1).get("source") == "minted",
          "the first answer is the model's text", json.dumps(c.msgs[-1]))
    t2 = c.turn([T.reply("", calls=[T.call("write_file", {
        "path": "a.txt", "content": "x"}, "tf1")])], user="Write a.txt.",
                features=off)
    check(_key(acct) == k1 and _ses(t2).get("id") == sid
          and _ses(t2).get("source") == "answer_record"
          and _slot(t2) == _slot(t1)
          and session_id.of_call_id(_ids(c.msgs[-1])[0]) == sid,
          "the next request is the same conversation (answer record), on its "
          "slot, and its first call carries the id from here on",
          json.dumps(_ses(t2)))
    T._extends(t1, t2, "[answer record] the request after a text-only first "
                       "answer")
    c.tool_result("tf1", "wrote a.txt")
    t3 = c.turn([T.reply("Done.")], features=off)
    check(_ses(t3).get("source") == "tool_call_id" and _key(acct) == k1,
          "and after the call: by the carrier", json.dumps(_ses(t3)))
    T._extends(t2, t3, "[answer record] the request after the call")


def test_the_compaction_summary_carries_the_id():
    """(4, in place) A conversation with our id is compacted in place
    (Claude Code / Codex shape); the summary opens with the session line --
    the one visible marker left -- and a continuation that keeps ONLY the
    summary keeps the conversation."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = T.Unforced("medium", "compacted")
    acct = c.body()["_account"]
    t1 = c.turn([T.reply("", calls=[T.call("write_file", {
        "path": "hi.py", "content": "print(1)"}, "cp1")])], user="Write hi.py.")
    sid, k1 = _ses(t1).get("id"), _key(acct)
    c.tool_result("cp1", "wrote hi.py")
    c.turn([T.reply("Wrote hi.py.")])
    t3 = c.turn([T.reply("## Goal\nWrote hi.py.")],
                user="Your task is to create a detailed summary of the "
                     "conversation so far.")
    summary = t3["d"]["choices"][0]["message"]["content"]
    x3 = t3["d"]["x_yamadori"]
    check(x3.get("utility_kind") == "compaction"
          and summary == session_id.line(sid) + "## Goal\nWrote hi.py."
          and _ses(t3).get("line") is True,
          "the in-place compaction's summary opens with the conversation's "
          "session line", json.dumps(summary))
    sent = t3["gens"][0]["request"]["messages"]
    check(not any(session_id.PREFIX in (m.get("content") or "")
                  for m in sent if isinstance(m.get("content"), str)),
          "the model never read a session line")
    cont = T.Unforced("medium", "compacted")
    cont.msgs.append({"role": "user", "content": "Summary of earlier work:\n"
                      + summary})
    t4 = cont.turn([T.reply("", calls=[T.call("write_file", {
        "path": "b.py", "content": "print(2)"}, "cp2")])],
        user="Continue: add b.py.")
    sent4 = t4["gens"][0]["request"]["messages"]
    check(_key(acct) == k1 and _ses(t4).get("id") == sid
          and _ses(t4).get("source") == "summary_line"
          and session_id.of_call_id(_ids(cont.msgs[-1])[0]) == sid
          and not any(session_id.PREFIX in (m.get("content") or "")
                      for m in sent4 if isinstance(m.get("content"), str)),
          "the continuation keeps the conversation by the summary's line "
          "(never shown to the model), and its new call carries the id on",
          json.dumps(_ses(t4)))


def test_harness_session_headers_and_cache_key_alias():
    """OpenCode (docs/HARNESS-OPENCODE.md): its own session id in
    X-Session-Id / x-session-affinity is an explicit session (a --fork with
    the original's tool-call ids stays its own conversation), and its SDK's
    camelCase `promptCacheKey` is the prompt_cache_key."""
    import server
    import session_id as sid
    h = {"x-session-id": "ses_forkB9a8", "x-session-affinity": "ses_forkB9a8"}
    check(server.session_of_headers(h) == ("ses_forkB9a8", "x-session-id"),
          "X-Session-Id is read as the session token",
          str(server.session_of_headers(h)))
    check(server.session_of_headers({"x-yamadori-session": "ours1",
                                     "x-session-id": "theirs"})[0] == "ours1",
          "our X-Yamadori-Session wins over the harness's header")
    check(server.session_of_headers({"x-session-id": "bad id!"}) == ("", ""),
          "a malformed header is ignored")
    check(sid.cache_key_of({"promptCacheKey": "pk1"}) == "pk1"
          and sid.cache_key_of({"prompt_cache_key": "a",
                                "promptCacheKey": "b"}) == "a",
          "promptCacheKey is an alias; prompt_cache_key wins")
    # The fork: the same history (the original's carried tool-call ids) under
    # two different harness session ids must be two conversations.
    carried = sid.call_id("4b1a6b4fe4f0")
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": carried, "type": "function",
                             "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": carried, "content": "ok"}]
    k0, s0 = proxy.session_identity(msgs, "acct", "", "", None)
    ka, sa = proxy.session_identity(msgs, "acct", "ses_orig", "", None)
    kb, sb = proxy.session_identity(msgs, "acct", "ses_fork", "", None)
    check(s0.get("source") == "tool_call_id" and sid.of_call_id(carried)
          == "4b1a6b4fe4f0", "without a header the carried id decides "
          "(the case that merged the fork)", json.dumps(s0))
    check(ka != kb and sa.get("source") == sb.get("source") == "header",
          "a fork (new X-Session-Id, same carried ids) is its own "
          "conversation: the harness's header wins over the carrier",
          json.dumps([sa, sb]))


def main() -> int:
    for fn in (test_harness_session_headers_and_cache_key_alias,
               test_the_same_opening_after_a_finished_run_is_a_new_conversation,
               test_interleaved_conversations_with_one_opening_keep_apart,
               test_a_retried_opening_is_a_new_conversation,
               test_prompt_cache_key_and_the_header_win,
               test_the_ids_round_trip_and_the_slot_extends,
               test_a_first_answer_without_a_call_keeps_its_conversation,
               test_the_compaction_summary_carries_the_id):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        t0 = len(T._results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        # T._extends records into test_ledger's list: fold them in.
        _results.extend(T._results[t0:])
        del T._results[t0:]
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail[:300]}" if not ok and detail else ""))
    T._srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
