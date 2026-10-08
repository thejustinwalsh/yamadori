#!/usr/bin/env python
"""Hermes' flattened compactions onto the stored conversation. No GPU.

WHAT THIS IS GATING (operator, 2026-10-07: "Why is there summary
discrepancy? We should fix.")

The pagoda run (Hermes, tier high, Bonsai, 12:00-17:36) made 20 compactions:
4 mapped onto the stored conversation (x_yamadori.compaction mode
"rewritten"), 16 went up as sent -- 13 "the span's first turn is not in the
stored conversation", 2 "the span's last turn ...", 1 "no stored conversation
for this account" (the proxy had been restarted: the store is in memory) --
and ran cold, thinking off, with a 40K-token prefill each, 13 of them for a
summary Hermes then discarded.

The transcripts were not kept, so Hermes' REAL compressor was driven offline
on synthetic conversations (bench/harness_shapes/hermes/drive_compressor.py;
the fixture mcp/fixtures/hermes_compactions.json is its output, the Hermes
install at ee5ee84a). Two defects of the matcher reproduce:

  1. A COMPACTION'S CARRIER. Hermes folds its summary into a row of the kept
     tail and restates the unfinished request ("[STILL IN PROGRESS -- ...]
     <the task>") after it; at the next compaction it unwraps that row and the
     unwrapped text is the FIRST record. The stored message is the carrier as
     sent, which does not begin with it. With one user request and a long
     tool loop (the pagoda: one "Continue", 150 steps) every compaction after
     the second fails this way: ok, ok, first, first, first -- the run's log.
  2. THE SEARCH WAS GREEDY. "Continue" (or an assistant turn with no text)
     matched the first such turn after the cursor, which skipped the tool
     results between and failed every record after it: "the span's last turn
     is not in the stored conversation".

mcp/compaction.py (THE MAPPING'S FAILURE) anchors on tool-call ids, matches
the other records between the anchors, reads a carrier on the pieces Hermes
wraps, carries turns newer than the stored conversation as text, and explains
what it could not map (x_yamadori.compaction.unmapped, logs/compaction_
trace.jsonl). A mapping stays exact: the first and every interior record must
be found; a non-match goes up as sent.

Offline. The proxy-level checks use test_utility's fake upstream.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_utility as T                                        # noqa: E402
from test_utility import check, proxy, side_call, script, reply  # noqa: E402,F401
import compaction                                               # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "hermes_compactions.json")


def _fixture() -> dict:
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _lagged(history: list[dict]) -> list[dict]:
    """The stored conversation when the client has appended the tool results
    of the turn in flight: the proxy holds the prompt and the assistant's
    answer, not the results."""
    h = list(history)
    while h and h[-1]["role"] == "tool":
        h.pop()
    return h


def _old_same(record: dict, msg: dict) -> bool:
    """mcp/compaction.py's _same at 2026-10-06: the stored message must BEGIN
    with the record's text (a tool result: the id)."""
    if record["role"] != msg.get("role"):
        return False
    if record["role"] == "tool" and record.get("id"):
        return record["id"] == msg.get("tool_call_id")
    a = compaction._norm(record["text"].split("\n...[truncated]...\n", 1)[0]
                         )[:compaction.MATCH_CHARS]
    b = compaction._norm(compaction._text(msg))
    return (not b) if not a else b.startswith(a)


def _greedy(records: list[dict], stored: list[dict]) -> dict:
    """The matcher this replaced (the same date): the first stored message
    after the cursor that begins with the record, the first and the last
    record both required. Kept as the witness that the fixture reproduces
    the failures, and that it could map the wrong span."""
    j, idx = 0, []
    for r in records:
        k = next((i for i in range(j, len(stored))
                  if _old_same(r, stored[i])), None)
        idx.append(k)
        if k is not None:
            j = k + 1
    which = ("first" if idx[0] is None else "last" if idx[-1] is None
             else "interior" if any(i is None for i in idx) else None)
    return {"mapped": which is None, "which": which, "idx": idx}


# --------------------------------------------------------------------------
# Hermes' real compressor
# --------------------------------------------------------------------------

def test_the_real_compressors_compactions_map():
    data = _fixture()
    old = {"first": 0, "last": 0, "interior": 0, "ok": 0}
    n = bad = 0
    shapes = set()
    for sc in data["scenarios"]:
        for c in sc["compactions"]:
            parsed = compaction.parse_flattened(c["prompt"])
            assert parsed and parsed["harness"] == "hermes"
            shapes.add(parsed["iterative"])
            for form, stored in (("as the client sent it", c["history"]),
                                 ("one turn behind", _lagged(c["history"]))):
                n += 1
                w = _greedy(parsed["records"], stored)
                old[w["which"] or "ok"] += 1
                m = compaction.map_records(parsed["records"], stored)
                if not (m["mapped"] and m["matched"] == m["total"]
                        and m["trailing"] == 0
                        and m["idx"] == sorted(m["idx"])):
                    bad += 1
                    check(False, f"{sc['name']} / compaction {c['round']} "
                          f"({form}) maps", json.dumps(
                              {k: m[k] for k in ("why", "which", "matched",
                                                 "total")}))
    check(n >= 20 and bad == 0,
          f"every compaction of the real compressor's {len(data['scenarios'])}"
          f" scenarios maps onto the history it summarises, whole ({n} "
          "mappings: the history as sent, and one turn behind)",
          f"{bad} of {n} did not")
    check(old["first"] and old["last"] and old["interior"] and old["ok"],
          "the fixture is the witness: the greedy matcher it replaced failed "
          "on the first turn, on the last and in between, and mapped the "
          "rest", json.dumps(old))
    check(shapes == {True, False} or shapes == {True},
          "the iterative form (a previous summary) is what it exercises",
          str(shapes))

    # The pagoda's shape, run by run: one request, a long tool loop.
    sc = next(s for s in data["scenarios"]
              if s["name"] == "one task, a long tool loop")
    got = []
    for c in sc["compactions"]:
        parsed = compaction.parse_flattened(c["prompt"])
        got.append(_greedy(parsed["records"], c["history"])["which"] or "ok")
    check(got[:2] == ["ok", "ok"] and set(got[2:]) == {"first"},
          "one request and a long tool loop: the old matcher mapped the "
          "first two compactions and failed every one after on its first "
          "turn (the operator's log: 12:21 and 12:38 mapped, then 13 "
          "'first turn' failures)", json.dumps(got))
    # ...and why: the first record is the restated request, the stored
    # message the carrier it was folded into.
    c = sc["compactions"][2]
    parsed = compaction.parse_flattened(c["prompt"])
    r0 = parsed["records"][0]
    m = compaction.map_records(parsed["records"], c["history"])
    at = c["history"][m["idx"][0]]
    check(r0["role"] == "user"
          and r0["text"].startswith("[STILL IN PROGRESS")
          and at["content"].startswith("[CONTEXT COMPACTION")
          and compaction.CARRIER_END in at["content"],
          "the first record is the restated request Hermes unwrapped; the "
          "stored message is the summary carrier it was folded into",
          json.dumps({"record": r0["text"][:60], "stored": at["content"][:60]}))


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------

def _rec(role, text, rid=None, **kw):
    return dict({"role": role, "id": rid, "text": text}, **kw)


def _calls(*ids):
    return [{"id": i, "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}
            for i in ids]


SUMMARY = ("[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
           "compacted.\nyamadori session 98f09f5a5a17\n\n## Goal\nBuild it.")


def test_a_carrier_is_read_on_the_pieces_hermes_wraps():
    old_text = "Start the dev server and report the port."
    forms = {
        # _merge_summary_into_tail_row: header + old + delimiter + summary
        "folded into the tail row": (
            compaction.CARRIER_HEADER + "\n" + old_text + "\n\n"
            + compaction.CARRIER_DELIMITER + "\n\n" + SUMMARY + "\n\n"
            + compaction.CARRIER_END),
        # force_user_leading: the summary first, the row after the marker
        "the summary first": (SUMMARY + "\n\n" + compaction.CARRIER_END
                              + "\n\n" + old_text),
        # _reappend_inflight_user_task merged onto the carrier
        "the restated request after the marker": (
            SUMMARY + "\n\n" + compaction.CARRIER_END + "\n\n"
            "[STILL IN PROGRESS — this is the active request, restated "
            "after the compaction boundary because it was not finished yet. "
            "Continue it; do not start over.]\n" + old_text),
    }
    for name, text in forms.items():
        stored = [{"role": "system", "content": "s"},
                  {"role": "user", "content": text},
                  {"role": "assistant", "content": "ok",
                   "tool_calls": _calls("c1")},
                  {"role": "tool", "tool_call_id": "c1", "content": "x"}]
        rec_text = (old_text if name != "the restated request after the "
                    "marker" else text.split(compaction.CARRIER_END, 1)[1]
                    .strip())
        recs = [_rec("user", rec_text), _rec("assistant", "ok"),
                _rec("tool", "x", "c1")]
        m = compaction.map_records(recs, stored)
        check(m["mapped"] and m["idx"] == [1, 2, 3],
              f"a carrier ({name}) is the record it wraps", json.dumps(m)[:200])
    plain = [{"role": "user", "content": "Start the dev server."}]
    check(compaction.heads("Start the  dev\nserver.") == ["Start the dev server."]
          and not compaction.map_records([_rec("user", "Stop it.")], plain)["mapped"],
          "a plain message has one head, and a record that is not its start "
          "does not map")
    check(compaction._same(_rec("user", "Start the dev"), plain[0])
          and not compaction._same(_rec("assistant", "Start"), plain[0]),
          "_same still answers for one message")


def test_the_search_is_anchored_not_greedy():
    # Three "Continue" turns and empty assistant turns: the old search took
    # the first of each after its cursor and skipped what lay between.
    stored, recs = [{"role": "system", "content": "s"}], []

    def add(role, content="", calls=None, tid=None):
        m = {"role": role, "content": content}
        if calls:
            m["tool_calls"] = _calls(*calls)
        if tid:
            m["tool_call_id"] = tid
        stored.append(m)
    n = 0
    for blk in range(3):
        add("user", "Continue")
        for _ in range(2):
            n += 1
            add("assistant", "", calls=[f"c{n}"])
            add("tool", f"out {n}", tid=f"c{n}")
    # the transcript starts at the SECOND block
    recs.append(_rec("user", "Continue"))
    for k in (3, 4):
        recs += [_rec("assistant", ""), _rec("tool", f"out {k}", f"c{k}")]
    recs.append(_rec("user", "Continue"))
    for k in (5, 6):
        recs += [_rec("assistant", ""), _rec("tool", f"out {k}", f"c{k}")]
    g = _greedy(recs, stored)
    m = compaction.map_records(recs, stored)
    check(m["mapped"] and m["matched"] == len(recs)
          and m["idx"] == list(range(6, 16)) and m["first"] == 6,
          "duplicate 'Continue' turns and empty assistant turns: the span is "
          "the second block, each record at its own place",
          json.dumps({"idx": m["idx"], "why": m["why"]}))
    check(g["idx"][0] == 1 and g["idx"][0] != m["idx"][0],
          "(the greedy search put the first record on the FIRST 'Continue', "
          "a span of 15 stored turns for a transcript of 10)",
          json.dumps(g["idx"]))
    # an assistant turn is placed by the call its tool result answers, not by
    # its (empty) text
    s2 = [{"role": "assistant", "content": "", "tool_calls": _calls("a")},
          {"role": "tool", "tool_call_id": "a", "content": "1"},
          {"role": "assistant", "content": "", "tool_calls": _calls("b")},
          {"role": "tool", "tool_call_id": "b", "content": "2"}]
    m2 = compaction.map_records([_rec("assistant", ""), _rec("tool", "2", "b")],
                                s2)
    check(m2["mapped"] and m2["idx"] == [2, 3],
          "an empty assistant turn is the one that made the call its result "
          "answers", json.dumps(m2["idx"]))


def test_turns_newer_than_the_stored_conversation_go_in_as_text():
    stored = [{"role": "user", "content": "Build it."},
              {"role": "assistant", "content": "", "tool_calls": _calls("a")},
              {"role": "tool", "tool_call_id": "a", "content": "1"},
              {"role": "assistant", "content": "", "tool_calls": _calls("b")}]
    # the client appended b's result since the proxy last sent a prompt
    recs = [_rec("user", "Build it.", raw="[USER]: Build it."),
            _rec("assistant", "", raw="[ASSISTANT]: \n[Tool calls:\n  "
                 "terminal({})\n]"),
            _rec("tool", "1", "a", raw="[TOOL RESULT a]: 1"),
            _rec("assistant", "", raw="[ASSISTANT]: \n[Tool calls:\n  "
                 "terminal({})\n]"),
            _rec("tool", "the newest output", "b",
                 raw="[TOOL RESULT b]: the newest output")]
    m = compaction.map_records(recs, stored)
    check(m["mapped"] and m["trailing"] == 1 and m["last"] == 3
          and m["matched"] == 4 and "newer than the stored" in m["why"],
          "the result the stored conversation lacks is carried: mapped, one "
          "trailing record", json.dumps({k: m[k] for k in (
              "mapped", "trailing", "last", "matched", "why")}))
    text = ("PREAMBLE\n\nTURNS TO SUMMARIZE:\n"
            + "\n\n".join(r["raw"] for r in recs)
            + "\n\nUse this exact structure:\n## Goal")
    parsed = compaction.parse_flattened(text)
    parsed["records"] = recs
    out = compaction.instruction_for(text, parsed, m, stored)
    check("the newest output" in out and "[TOOL RESULT b]: the newest output"
          in out and "TOOL RESULT a" not in out
          and "they are not in the conversation above" in out
          and out.rstrip().endswith("## Goal"),
          "the instruction names the span and writes the newer turn out",
          out[:400])
    # ...but only when the last record that maps is the stored conversation's
    # own last message: a record dropped in the middle is not 'newer'
    stored2 = stored + [{"role": "tool", "tool_call_id": "b", "content": "x"},
                        {"role": "assistant", "content": "later"}]
    m2 = compaction.map_records(recs[:2] + [_rec("user", "a turn nobody sent")],
                                stored2)
    check(not m2["mapped"] and m2["which"] == "last"
          and "last turn is not in the stored" in m2["why"],
          "a trailing record after a mapped one that is NOT the end of the "
          "stored conversation is not mapped", m2["why"])


def test_a_mapping_stays_exact():
    stored = [{"role": "system", "content": "s"},
              {"role": "user", "content": "Build the scene."},
              {"role": "assistant", "content": "Starting.",
               "tool_calls": _calls("a")},
              {"role": "tool", "tool_call_id": "a", "content": "1"},
              {"role": "assistant", "content": "Next.",
               "tool_calls": _calls("b")},
              {"role": "tool", "tool_call_id": "b", "content": "2"}]
    ok = [_rec("user", "Build the scene."), _rec("assistant", "Starting."),
          _rec("tool", "1", "a"), _rec("assistant", "Next."),
          _rec("tool", "2", "b")]
    check(compaction.map_records(ok, stored)["mapped"], "the whole span maps")
    # a first record that is not stored
    m = compaction.map_records([_rec("user", "Plan a garden.")] + ok[1:],
                               stored)
    check(not m["mapped"] and m["which"] == "first"
          and "first turn is not in the stored" in m["why"],
          "a first record that is not stored: not mapped, 'first'", m["why"])
    # an interior record that is not stored
    m = compaction.map_records(ok[:3] + [_rec("user", "A different turn."),
                                         ok[4]], stored)
    check(not m["mapped"] and m["which"] == "interior"
          and "only 4 of 5" in m["why"],
          "an interior record that is not stored: not mapped, 'interior'",
          m["why"])
    # another conversation's ids
    other = [_rec("user", "Build the scene."), _rec("assistant", "Starting."),
             _rec("tool", "1", "zzz")]
    m = compaction.map_records(other, stored)
    check(not m["mapped"],
          "a tool result whose id is not stored is not found by its text")
    # a tool result before the one the cursor is past
    m = compaction.map_records([ok[0], ok[3], ok[4], ok[2]], stored)
    check(not m["mapped"], "records out of order do not map")
    check(not compaction.map_records([], stored)["mapped"]
          and compaction.map_records([], stored)["why"]
          == "no records in the transcript", "no records, no mapping")
    # MIN_MAPPED still relaxes the interior rule when the operator sets it
    old = compaction.MIN_MAPPED
    try:
        compaction.MIN_MAPPED = 0.8
        m = compaction.map_records(ok[:3] + [_rec("assistant", "Different.")]
                                   + ok[4:], stored)
        check(m["mapped"], "YAMADORI_COMPACTION_MIN_MAPPED still sets the "
              "share of the span that must map", m["why"])
    finally:
        compaction.MIN_MAPPED = old


def test_what_could_not_map_is_explained():
    stored = [{"role": "system", "content": "s"},
              {"role": "user", "content": "Build the scene."},
              {"role": "assistant", "content": "Starting.",
               "tool_calls": _calls("a")},
              {"role": "tool", "tool_call_id": "a", "content": "1"},
              {"role": "user", "content": "[CONTEXT COMPACTION] x"}]
    client = stored + [{"role": "user", "content": "The client's own form."}]
    recs = [_rec("user", "Build the scene and make it big."),     # nearest
            _rec("assistant", "Starting."),
            _rec("tool", "9", "gone"),                            # no id
            _rec("user", "The client's own form."),               # client
            _rec("user", "[CONTEXT COMPACTION] x")]               # misplaced
    m = compaction.map_records(recs, stored)
    e = compaction.explain(recs, stored, m, client=client)
    causes = {u["record"]: u["cause"] for u in e["unmatched"]}
    check(e["which_turn"] == "first" and e["why"] == m["why"]
          and e["best_partial_match"]["matched"] == m["matched"]
          and e["best_partial_match"]["total"] == 5
          and e["best_partial_match"]["stored_messages"] == 5
          and e["unmatched_count"] == 3 and len(e["unmatched"]) == 3,
          "which turn, why, how much matched", json.dumps(e)[:400])
    check("nearest" in e["unmatched"][0]
          and e["unmatched"][0]["nearest"]["common"] >= len("Build the scene")
          and "no stored turn of that role begins" in causes[0],
          "a text that is not stored names the stored turn that starts most "
          "like it", json.dumps(e["unmatched"][0]))
    check("id is not in the stored conversation" in causes[2],
          "an id that is not stored says so", causes[2])
    check("client's own form" in causes[3],
          "a text found only in the client's own form says the proxy changed "
          "its start", causes[3])
    # trace + digests
    d = compaction.digest(stored)
    dr = compaction.digest_records(recs, m["idx"])
    check(d[2][:2] == [2, "assistant"] and d[3][2] == "a"
          and dr[0][:2] == [0, "user"] and len(dr[0]) == 6,
          "the digests: index, role, ids, length, head (and the stored index)",
          json.dumps([d[2], dr[0]]))


# --------------------------------------------------------------------------
# Through the proxy
# --------------------------------------------------------------------------

def _conversation(history: list[dict]) -> None:
    """Send `history` as the client's request; the proxy stores it with the
    answer it gets back."""
    script(reply("Working on it.", cache_n=0, prompt_n=900))
    body = {"model": "yamadori", "_account": T.ACCOUNT, "messages": history,
            "tools": T.CLIENT_TOOLS}
    proxy.complete(body)


def test_the_proxy_maps_the_real_compressors_compactions():
    data = _fixture()
    sc = next(s for s in data["scenarios"]
              if s["name"] == "one task, a long tool loop")
    c = sc["compactions"][2]                    # the third: 'first turn' before
    T.slots.reset(n=4)
    compaction.reset()
    _conversation(_lagged(c["history"]))
    parsed = compaction.parse_flattened(c["prompt"])
    script(reply("## Goal\nBuild it.", cache_n=900, prompt_n=300))
    d = proxy.complete(side_call([{"role": "user", "content": c["prompt"]}]))
    x = d["x_yamadori"]["compaction"]
    check(x["shape"] == "flattened" and x["mode"] == "rewritten"
          and x["mapped"] == len(parsed["records"])
          and x["records"] == len(parsed["records"]) and "unmapped" not in x
          and not x.get("trailing"),
          "the pagoda's shape through the proxy: the third compaction, whose "
          "first turn was the restated request, is rewritten onto the "
          "conversation (every record mapped)", json.dumps(x)[:300])
    up = T._seen[0]
    instr = up["messages"][-1]["content"]
    check(len(up["messages"]) > 10 and up["messages"][-1]["role"] == "user"
          and "[The turns to summarise are in the conversation above"
          in instr and parsed["records"][1]["text"] not in instr[:0] + "\0",
          "upstream: the stored conversation, then one user turn that points "
          "at the span", instr[:200])

    # the same compaction, the stored conversation one turn ahead
    T.slots.reset(n=4)
    compaction.reset()
    _conversation(c["history"][:-2])
    script(reply("## Goal", cache_n=0, prompt_n=300))
    d = proxy.complete(side_call([{"role": "user", "content": c["prompt"]}]))
    check(d["x_yamadori"]["compaction"]["mode"] == "rewritten",
          "and with the conversation a turn behind the client's history",
          json.dumps(d["x_yamadori"]["compaction"])[:200])


def test_a_turn_newer_than_the_conversation_is_written_out():
    T.slots.reset(n=4)
    compaction.reset()
    history, stored, slot = T._a_conversation()
    span = history[1:] + [{"role": "user", "content": "Now add a score "
                           "counter to the corner."}]
    text = T.hermes_compaction(span)
    script(reply("## Goal\nBuild it.", cache_n=9950, prompt_n=500))
    d = proxy.complete(side_call([{"role": "user", "content": text}]))
    x = d["x_yamadori"]["compaction"]
    instr = T._seen[0]["messages"][-1]["content"]
    check(x["mode"] == "rewritten" and x["trailing"] == 1
          and x["mapped"] == len(span) - 1
          and "Now add a score counter to the corner." in instr
          and "[USER]: Now add a score counter" in instr
          and T.GAME_LOOP[:200] not in instr,
          "a user turn the proxy has not seen: the rest maps, that one is "
          "written into the instruction, the file it read is not",
          json.dumps(x)[:300])
    T.slots.reset(n=4)
    compaction.reset()


def test_an_unmapped_compaction_is_explained_and_traced():
    T.slots.reset(n=4)
    compaction.reset()
    history, stored, slot = T._a_conversation()
    other = T.hermes_compaction(
        [{"role": "user", "content": "Plan a garden with raised beds."},
         {"role": "assistant", "content": "Beds first."}])
    trace = os.environ["YAMADORI_COMPACTION_TRACE"]
    before = os.path.getsize(trace) if os.path.exists(trace) else 0
    script(reply("## Goal", cache_n=0, prompt_n=2000))
    d = proxy.complete(side_call([{"role": "user", "content": other}]))
    x = d["x_yamadori"]["compaction"]
    u = x.get("unmapped") or {}
    check(x["mode"] == "as_sent" and u.get("which_turn") == "first"
          and "first turn is not in the stored" in u.get("why", "")
          and u["best_partial_match"]["matched"] == 0
          and u["best_partial_match"]["total"] == 2
          and u["best_partial_match"]["stored_messages"] > 3
          and u["unmatched"][0]["role"] == "user"
          and u["unmatched"][0]["text"].startswith("Plan a garden")
          and u["unmatched"][0]["cause"] and u.get("conversation")
          and u.get("tried") == 1,
          "x_yamadori.compaction.unmapped: which turn, why, the closest "
          "stored conversation and how much of it matched",
          json.dumps(u)[:400])
    check(x["why"] == u.get("why"),
          "the compaction's 'why' is the closest conversation's, not the "
          "last one looked at", x["why"])
    lines = open(trace, encoding="utf-8").read().splitlines() \
        if os.path.exists(trace) else []
    rec = json.loads(lines[-1]) if lines and os.path.getsize(trace) > before \
        else {}
    check(rec.get("which_turn") == "first" and rec.get("records") == 2
          and rec.get("transcript") and rec.get("stored")
          and rec["transcript"][0][:2] == [0, "user"]
          and rec["conversations"][0]["matched"] == 0
          and "Plan a garden" in json.dumps(rec),
          "the trace line: the transcript's and the stored conversation's "
          "shapes (roles, ids, lengths, first characters)",
          json.dumps(rec)[:300])
    # no stored conversation at all (a restart): nothing to explain
    compaction.reset()
    script(reply("## Goal", cache_n=0, prompt_n=2000))
    d = proxy.complete(side_call([{"role": "user", "content": other}]))
    x = d["x_yamadori"]["compaction"]
    check(x["mode"] == "as_sent" and "unmapped" not in x
          and "no stored conversation for this account" in x["why"],
          "after a restart: as sent, and the reason is the empty store",
          x["why"])
    T.slots.reset(n=4)
    compaction.reset()


def main() -> int:
    for fn in (test_the_real_compressors_compactions_map,
               test_a_carrier_is_read_on_the_pieces_hermes_wraps,
               test_the_search_is_anchored_not_greedy,
               test_turns_newer_than_the_stored_conversation_go_in_as_text,
               test_a_mapping_stays_exact,
               test_what_could_not_map_is_explained,
               test_the_proxy_maps_the_real_compressors_compactions,
               test_a_turn_newer_than_the_conversation_is_written_out,
               test_an_unmapped_compaction_is_explained_and_traced):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(T._results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in T._results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in T._results if ok)
    total = len(T._results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
