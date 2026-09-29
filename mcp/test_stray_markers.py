#!/usr/bin/env python
"""Stray template markers in the answer are removed from what the client
gets, and recorded (#12). No GPU.

    python mcp/test_stray_markers.py      -> "N/M checks passed"

THE BEHAVIOUR (diagnosed live 2026-09-25; model behaviour, not a
regression): the model thinks, the FIRST `</think>` ends its reasoning
(llama-server's parser takes it), and in the ANSWER it writes a SECOND
literal `</think>` and usually repeats itself -- 'Done.\\n</think>\\n\\nDone.',
'Done.\\n\\nVerified stats.py ...\\n</think>\\n\\nDone.', or
'<sentence>\\n</think>\\n\\n' before its tool calls. 7 of 40 replays of one
fixed context with reasoning echoed, 3 of 40 stripped.

THE DECISION (operator, 2026-09-25, reversing #12's "delivered as written"):
the proxy removes the marker, and the text after it while it repeats what
was already sent; new text after it goes through; x_yamadori.
template_markers records what was stripped (proxy._StrayMarkers).

GATED HERE:
  1. the filter by itself: the three real shapes (and the triple), new
     text after a marker, markers inside code kept, a marker split across
     two chunks at EVERY position, and one character at a time == the whole
     text on a seeded corpus of random texts and chunkings (output and
     record alike);
  2. _Out: the marker filter composes with the session line, an image line
     and the image-duplicate filter in one deterministic order (image
     duplicates, then markers: a repeat is judged against what the client
     was sent), chunking-independent, the same as the blocking path;
  3. end to end through the real stream_body() / complete() and the SERVED
     chat template (test_ledger's fake upstream, 40-character pieces): no
     content event ever carries a marker, streamed == blocking byte for
     byte, CHANNEL ORDER holds, x_yamadori.template_markers says what was
     stripped, tool calls after the marker are the model's calls untouched;
  4. the marker at every offset against the upstream's chunk boundaries;
  5. the next request still EXTENDS the slot (the ledger keys the turn by
     the client's copy and renders the slot's own text, #10), streamed and
     blocking, and the model's own text is what the next request renders;
  6. with an image: the session line and the image line are never part of
     what a repeat is compared against, and the model's copy of the image
     inside the repeat goes too.
  7. a LANDED answer's written-out tool calls are delivered AS WRITTEN:
     the _ToolMarkup filter and its "[no answer: ...]" notice were REMOVED
     2026-09-27 (docs/CONSTANTS-AUDIT.md: built from one run); streamed ==
     blocking, and the next request extends the slot.

FAIL BEFORE (2026-09-25; the filter replaced by the identity, i.e. the
old "delivered as written"): 58 of 131 checks fail -- every shape, every
offset, the record, the composition result. What still passes is what any
identity keeps: literal code, chunking equality, streamed == blocking, the
calls, and the slot extension (#10's path, which this change relies on).

Every database and store is a temp path set BEFORE proxy is imported.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_stray_markers_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json"),
               ("YAMADORI_MEDIA_DIR", "media"),
               ("YAMADORI_MEDIA_SECRET_FILE", "media_url.key"),
               ("YAMADORI_ACCOUNTS_DIR", "accounts")):
    os.environ[_k] = os.path.join(_TMP, _v)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
os.environ.pop("YAMADORI_VISION", None)

# test_image_emit sets its own temp paths (and test_ledger's fake upstream,
# fake image generator and Hermes-like client) before proxy is imported.
import test_image_emit as E  # noqa: E402
import nebari  # noqa: E402
import proxy  # noqa: E402
import session_id  # noqa: E402

T = E.T


class Conv(E.Conv):
    """test_image_emit's Hermes-like client, with its own prompt_cache_key:
    the conversation's identity does not hang on how the proxy carries its
    own id (mcp/session_id.py), which this suite does not test."""

    def body(self) -> dict:
        return dict(super().body(), prompt_cache_key=self.account)

    def tool_result(self, cid: str, text: str) -> None:
        self.msgs.append({"role": "tool", "tool_call_id": cid,
                          "content": text})


_tmp_root = os.path.abspath(tempfile.gettempdir())
assert os.path.abspath(nebari.DB).startswith(_tmp_root), nebari.DB
for _k in ("YAMADORI_CORPUS_DB", "YAMADORI_NEBARI_DB", "RINGS_DB",
           "CODE_INDEX_DB", "YAMADORI_MEDIA_DIR", "YAMADORI_ACCOUNTS_DIR"):
    assert os.path.abspath(os.environ[_k]).startswith(_tmp_root), _k

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


S = proxy._StrayMarkers
MARKERS = ("</think>", "<think>", "<|im_end|>", "<|im_start|>")

VERIFIED = "Verified stats.py (python): parses, lint clean; sent as written."
# (model content, what the client gets)
SHAPES = [
    ("Done.\n</think>\n\nDone.", "Done."),
    (f"Done.\n\n{VERIFIED}\n</think>\n\nDone.", f"Done.\n\n{VERIFIED}"),
    ("I will read the file first.\n</think>\n\n",
     "I will read the file first."),
    ("Done.\n</think>\n\nDone.\n</think>\n\nDone.", "Done."),
]
NEW_AFTER = [
    ("Done.\n</think>\n\nstats.py is done.", "Done.\n\nstats.py is done."),
    ("Done.\n</think>\n\nIt is done: stats.py implements mean and median.",
     "Done.\n\nIt is done: stats.py implements mean and median."),
    ("The build passed.\n</think>\n\nThe tests fail.",
     "The build passed.\n\nThe tests fail."),
    ("Done.\n</think>\n\nDone.\n\nOne more thing: run it.",
     "Done.\n\nOne more thing: run it."),
    ("</think>\n\nHello.", "Hello."),
]
LITERAL = [
    "Close the block with `</think>` and the turn with `<|im_end|>`.",
    "```python\nTAGS = ('<think>', '</think>')\n```\nThat is the pair.",
    "A partial `</th` and a bare </th at the end stay: </th",
]


def chunked(text: str, cuts: list[int]) -> tuple[str, S]:
    f = S()
    out, prev = [], 0
    for c in sorted(cuts) + [len(text)]:
        out.append(f.feed(text[prev:c]))
        prev = c
    out.append(f.flush())
    return "".join(out), f


def record(f: S) -> tuple:
    return (dict(f.stripped), f.repeat_chars, f.new_after)


# ------------------------------------------------------------------ 1
def test_the_filter_by_itself():
    for text, want in SHAPES + NEW_AFTER:
        got, f = S.clean(text)
        check(got == want, f"whole text: {text!r} -> {want!r}",
              json.dumps({"got": got, "record": record(f)}))
    for text, _want in SHAPES:
        _got, f = S.clean(text)
        check(f.stripped.get("</think>") == text.count("</think>")
              and f.new_after == 0,
              f"recorded: {text.count('</think>')} stripped, the repeat's "
              f"characters counted ({f.repeat_chars})", json.dumps(record(f)))
    _g, f = S.clean(SHAPES[0][0])
    check(f.repeat_chars == len("Done."), "'Done.' after 'Done.': 5 repeated "
          "characters dropped", json.dumps(record(f)))
    for text in LITERAL:
        got, f = S.clean(text)
        check(got == text and not f.stripped,
              f"a marker inside code (or a partial one) is literal and "
              f"stays: {text[:40]!r}", json.dumps(got))
    got, _f = S.clean("Done.<|im_start|>assistant\nDone.")
    check(got == "Done.", "an <|im_start|> takes its role line with it, and "
          "the repeat after it goes", json.dumps(got))
    check(S.clean("Plain answer.\n\nTwo lines.\n")[0]
          == "Plain answer.\n\nTwo lines.\n",
          "text with no marker is untouched (trailing whitespace included)")
    # Split across two chunks at EVERY position, and at every PAIR of
    # positions for the real shapes.
    bad = []
    for text, want in SHAPES + NEW_AFTER:
        whole, wf = S.clean(text)
        for i in range(len(text) + 1):
            got, f = chunked(text, [i])
            if got != whole or record(f) != record(wf):
                bad.append({"text": text, "cut": i, "got": got})
        for i in range(len(text) + 1):
            for j in range(i, len(text) + 1):
                got, f = chunked(text, [i, j])
                if got != whole or record(f) != record(wf):
                    bad.append({"text": text, "cuts": [i, j], "got": got})
    check(not bad, "a marker split across chunks at every position (and "
          "every pair of positions) gives the whole-text result",
          json.dumps(bad[:3]))
    # A streamed feed never releases a marker, nor a repeat it will drop.
    leaked = []
    for text, _want in SHAPES + NEW_AFTER:
        f = S()
        for ch in text:
            piece = f.feed(ch)
            if any(m in f.sent for m in MARKERS) or "<" in piece:
                leaked.append({"text": text, "piece": piece})
    check(not leaked, "fed a character at a time, no piece ever carries a "
          "marker or its '<'", json.dumps(leaked[:3]))
    # One character at a time == the whole text, on a seeded random corpus.
    rng = random.Random(20260925)
    frags = ["Done.", "Done", "stats.py", " is done.", "\n", "\n\n", " ",
             "</think>", "<think>", "<|im_end|>", "<|im_start|>",
             "assistant\n", "</th", "<", "`", "```\n", "Verified it.",
             "It", " works", "~~~\n", "x<y", ">"]
    mism = []
    for _n in range(400):
        text = "".join(rng.choice(frags) for _ in range(rng.randint(1, 14)))
        whole, wf = S.clean(text)
        g = S()
        one = "".join(g.feed(ch) for ch in text) + g.flush()
        if one != whole or record(g) != record(wf):
            mism.append({"text": text, "whole": whole, "one": one})
            continue
        for _k in range(4):
            cuts = sorted(rng.sample(range(len(text) + 1),
                                     min(len(text) + 1, rng.randint(1, 5))))
            got, f = chunked(text, cuts)
            if got != whole or record(f) != record(wf):
                mism.append({"text": text, "cuts": cuts, "got": got,
                             "whole": whole})
                break
    check(not mism, "400 random texts: one character at a time and random "
          "chunkings == the whole text (output and record)",
          json.dumps(mism[:3]))


# ------------------------------------------------------------------ 2
def _out_run(lead: str, block: str, url: str, content: str,
             sizes: list[int]) -> tuple[str, list]:
    """_Out, streamed: the session line, an image line, then the model's
    content in pieces of `sizes` (cycled); what the client was sent."""
    out = proxy._Out(True, lead=lead)
    events = list(out.image(block, url))
    i, k = 0, 0
    while i < len(content):
        n = sizes[k % len(sizes)]
        events += list(out.content(content[i:i + n]))
        i, k = i + n, k + 1
    events += list(out.flush_held())
    events += list(out.flush_lead())
    return "".join(out.shown), events


def test_out_composes_the_filters_in_one_order():
    url = "https://img.example.test/media/abc.png?exp=1&sig=ab"
    block = f"![a fox]({url})\n\n"
    lead = session_id.line("0123456789ab")
    content = (f"Here it is.\n</think>\n\n{block}Here it is.\n\n"
               f"Seen again: ![fox]({url}) in the snow.")
    # The blocking path's order: image duplicates, then markers.
    d = proxy._ImageDedup()
    d.add(url)
    want = lead + block + proxy.strip_stray_markers(d.strip(content))
    got = {}
    for sizes in ([1], [2], [3], [7], [40], [len(content)], [5, 1, 11]):
        shown, events = _out_run(lead, block, url, content, sizes)
        got[str(sizes)] = shown
        check(all(not any(m in (t or "") for m in MARKERS)
                  for _k, t in events),
              f"_Out pieces {sizes}: no content event carries a marker")
    check(len(set(got.values())) == 1 and want in got.values(),
          "_Out: session line, image line, then the model's text through "
          "the marker filter and the duplicate filter -- the same for every "
          "chunking, and what the blocking path builds",
          json.dumps({"want": want, "got": got})[:900])
    check(want == lead + block + "Here it is.\n\nSeen again: in the snow.",
          "the image copies go first, then the marker and its repeat (the "
          "repeat is judged against what the client was sent); the session "
          "and image lines are not part of what it is compared against",
          json.dumps(want))
    # The other order would let the image copy hide the repeat.
    other = lead + block + d.strip(proxy.strip_stray_markers(content))
    check(other != want, "the order matters, and this is the tested one",
          json.dumps(other))


# ------------------------------------------------------------------ 3, 5
def _no_marker_events(t: dict) -> bool:
    return all(not any(m in txt for m in MARKERS)
               for kind, txt, _g in t["stream"]["events"] if kind == "content")


def test_end_to_end_streamed_equals_blocking_and_extends():
    for i, (text, want) in enumerate(SHAPES + NEW_AFTER[:2]):
        # (The imitation filter that also removed SHAPES[1]'s "Verified
        # stats.py ..." line was removed 2026-09-27: the model's line stays.)
        got = {}
        for streamed in (True, False):
            how = "streamed" if streamed else "blocking"
            T.slots.reset(n=4)
            T.compaction.reset()
            c = Conv(f"sm{i}-{int(streamed)}", streamed=streamed)
            c.turn([T.reply("Hello.", reasoning="Greet.")], user="Hi.")
            t = c.turn([T.reply(text, reasoning="It is written; say so.")],
                       user="Is it done?")
            got[how] = t["content"]
            tm = t["x"].get("template_markers") or {}
            check(t["content"] == want,
                  f"[{i}] {how}: the client gets {want!r}",
                  json.dumps({"got": t["content"], "model": text}))
            check(tm.get("stripped", {}).get("</think>")
                  == text.count("</think>") and tm.get("source") == "model"
                  and not tm.get("ours") and not tm.get("in_content"),
                  f"[{i}] {how}: x_yamadori.template_markers records the "
                  f"model's stripped markers", json.dumps(tm))
            if streamed:
                check(_no_marker_events(t)
                      and E.channel_order_holds(t["stream"]["events"]),
                      f"[{i}] streamed: no content event carries a marker; "
                      f"CHANNEL ORDER holds",
                      json.dumps([e[:2] for e in t["stream"]["events"]])[:400])
            t2 = c.turn([T.reply("Good.")], user="Thanks.")
            T._extends(t, t2, f"[{i}] {how}: the next request")
            sent = t2["gens"][0]["request"]["messages"]
            finals = [m for m in sent if m.get("role") == "assistant"]
            check(finals and finals[-1].get("content") == text,
                  f"[{i}] {how}: the next request renders the slot's own text "
                  f"(marker and repeat), keyed by the client's copy",
                  json.dumps(finals[-1] if finals else None)[:300])
            check(not (t["x"].get("warm") or {}).get("sent"),
                  f"[{i}] {how}: no warm needed (the ledger renders what the "
                  f"slot holds)", json.dumps(t["x"].get("warm")))
            t3 = c.turn([T.reply("Bye.")], user="Bye.")
            T._extends(t2, t3, f"[{i}] {how}: the request after that")
        check(got["streamed"] == got["blocking"],
              f"[{i}] streamed == blocking, byte for byte", json.dumps(got))


def test_tool_calls_after_the_marker_are_untouched():
    args = {"path": "notes.txt", "content": "hello\n"}
    for streamed in (True, False):
        how = "streamed" if streamed else "blocking"
        T.slots.reset(n=4)
        T.compaction.reset()
        c = Conv(f"calls-{int(streamed)}", streamed=streamed)
        c.turn([T.reply("Hello.")], user="Hi.")
        call = T.call("write_file", args, "w1")
        t = c.turn([T.reply("I will write the notes file.\n</think>\n\n",
                            reasoning="Write it.", calls=[call])],
                   user="Write notes.txt.")
        kept = c.msgs[-1]
        calls = kept.get("tool_calls") or []
        check(len(calls) == 1 and calls[0]["function"]["name"] == "write_file"
              and json.loads(calls[0]["function"]["arguments"]) == args,
              f"{how}: the call after the marker reaches the client as the "
              f"model wrote it", json.dumps(calls)[:300])
        check(t["content"].startswith("I will write the notes file.")
              and not any(m in t["content"] for m in MARKERS),
              f"{how}: the content before the calls is cleaned (any note of "
              f"ours after it)", json.dumps(t["content"]))
        if streamed:
            ev = t["stream"]["events"]
            last_content = max((k for k, e in enumerate(ev)
                                if e[0] == "content"), default=-1)
            first_call = min((k for k, e in enumerate(ev) if e[0] == "calls"),
                             default=10**9)
            check(_no_marker_events(t) and last_content < first_call,
                  "streamed: no marker in any content event; all content "
                  "before the calls", json.dumps([e[:2] for e in ev])[:400])
        c.tool_result(calls[0]["id"] if calls else "w1",
                      "wrote notes.txt (6 bytes)")
        t2 = c.turn([T.reply("Written.")])
        T._extends(t, t2, f"[calls] {how}: the request after the tool result")


# ------------------------------------------------------------------ 4
def test_the_marker_at_every_offset_of_the_upstreams_chunks():
    """The fake upstream sends 40-character pieces: padding the answer moves
    the marker across every offset of a piece boundary."""
    bad = []
    T.slots.reset(n=4)
    T.compaction.reset()
    for pad in range(0, 42):
        text = "a" * pad + " done.\n</think>\n\n" + "a" * pad + " done."
        want = "a" * pad + " done."
        c = Conv(f"off{pad}", streamed=True)
        c.turn([T.reply("Hi.")], user="Hi.")
        t = c.turn([T.reply(text)], user="Done?")
        if t["content"] != want or not _no_marker_events(t):
            bad.append({"pad": pad, "got": t["content"]})
    check(not bad, "streamed: the marker at every offset of the upstream's "
          "40-character pieces is removed with its repeat",
          json.dumps(bad[:3]))


# ------------------------------------------------------------------ 6
def test_with_an_image_and_the_session_line():
    md = E.md_of("a fox in snow")
    words = f"Here is your fox.\n</think>\n\n{md}\n\nHere is your fox."
    got = {}
    for streamed in (True, False):
        how = "streamed" if streamed else "blocking"
        T.slots.reset(n=4)
        T.compaction.reset()
        c = Conv(f"img-{int(streamed)}", streamed=streamed)
        t = c.turn([T.reply("", calls=[E.draw("a fox in snow", "i1")]),
                    T.reply(words)], user="Draw me a fox.")
        sid = (t["x"].get("session") or {}).get("id") or ""
        tail = md + "\n\nHere is your fox."
        got[how] = t["content"]
        check(t["content"].endswith(tail) and t["content"][:-len(tail)] in (
                  "", session_id.line(sid)),
              f"{how}: session line and image line untouched, the model's "
              f"marker and its repeat (with its image copy) gone",
              json.dumps(t["content"]))
        tm = t["x"].get("template_markers") or {}
        check(tm.get("stripped", {}).get("</think>") == 1,
              f"{how}: recorded", json.dumps(tm))
        t2 = c.turn([T.reply("Glad you like it.")], user="Thanks.")
        T._extends(t, t2, f"[image + marker] {how}: the next request")
    check(got["streamed"] == got["blocking"],
          "with an image: streamed == blocking", json.dumps(got))


# ------------------------------------------------------------------ 7
# A LANDED ANSWER'S WRITTEN-OUT TOOL CALL (2026-09-27): delivered as the
# model wrote it (the _ToolMarkup filter and its notice were removed,
# docs/CONSTANTS-AUDIT.md).
QWEN_CALL = ("<tool_call>\n<function=run_command>\n<parameter=command>\n"
             "ls -la && cat package.json\n</parameter>\n</function>\n"
             "</tool_call>")


def _landing_turn(tag: str, streamed: bool, final: str) -> tuple[dict, "Conv"]:
    """Ten yama_generate_image hops (medium's tool-turn cap, tiers.tool_turn_limit)
    so the loop LANDS, then `final` as the landed hop's content."""
    T.slots.reset(n=4)
    T.compaction.reset()
    c = Conv(tag, streamed=streamed)
    c.turn([T.reply("Hello.")], user="Hi.")
    script = [T.reply("", calls=[E.draw(f"a fox, pose {k}", f"d{k}")])
              for k in range(10)] + [T.reply(final, reasoning="Answer.")]
    t = c.turn(script, user="Draw me ten foxes, one at a time.")
    return t, c


def test_landed_answer_markup_is_delivered_as_written():
    final = "I have the foxes. Let me look at the folder.\n\n" + QWEN_CALL
    got = {}
    for streamed in (True, False):
        how = "streamed" if streamed else "blocking"
        t, c = _landing_turn(f"tm-{int(streamed)}", streamed, final)
        x = t["x"]
        got[how] = t["content"]
        check((x.get("tool_turns") or {}).get("hit") is True,
              f"{how}: the loop landed at the cap",
              json.dumps(x.get("tool_turns")))
        check(t["content"].rstrip().endswith("</tool_call>")
              and "[no answer" not in t["content"]
              and "tool_markup" not in x,
              f"{how}: the landed answer reaches the client as written "
              f"(no filter, no notice, no record)",
              json.dumps(t["content"][-200:]))
        t2 = c.turn([T.reply("Good.")], user="Thanks.")
        T._extends(t, t2, f"[landed] {how}: the next request")
    check(got["streamed"] == got["blocking"],
          "landed: streamed == blocking, byte for byte",
          json.dumps(got)[:600])


def main() -> int:
    for fn in (test_the_filter_by_itself,
               test_landed_answer_markup_is_delivered_as_written,
               test_out_composes_the_filters_in_one_order,
               test_end_to_end_streamed_equals_blocking_and_extends,
               test_tool_calls_after_the_marker_are_untouched,
               test_the_marker_at_every_offset_of_the_upstreams_chunks,
               test_with_an_image_and_the_session_line):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        t0 = len(T._results)
        e0 = len(E._results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        _results.extend(T._results[t0:])
        del T._results[t0:]
        _results.extend(E._results[e0:])
        del E._results[e0:]
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail[:500]}" if not ok and detail else ""))
    T._srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
