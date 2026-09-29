#!/usr/bin/env python
"""Bounded compression of shell results (mcp/result_compress.py), asserted
directly on Hermes-, Pi-, OpenCode- and Codex-shaped results. No GPU, no
network, no proxy.

    python mcp/test_result_compress.py      -> "N/M checks passed"

WHAT THIS GATES (Atomic Agent's compress step; docs/research/ATOMIC-AGENT.md):

  1. SCOPE. Only a shell tool's result (terminal, bash, exec_command), only
     over CAP_CHARS; anything else passes byte for byte.
  2. THE MODEL'S CODE IS NEVER TOUCHED. A result whose command or output
     names a path the model wrote (a write/edit call, a shell redirect
     outside a heredoc body) passes whole; code in a heredoc body does not
     invent paths.
  3. THE CUT. Blank lines dropped, the tail kept on line boundaries within
     the cap; every kept line is a whole line of the original; a
     structured failure keeps its first error line on top; one line says
     what was left out and how to see it.
  4. THE WRAPPER. Hermes' JSON stays valid JSON with exit_code/error
     untouched; Codex's header stays; Pi's trailing exit line survives.
  5. BOUNDED AND DETERMINISTIC. Never over the cap (the caller may lower
     it, never raise it); the same input gives the same bytes.
  6. DECIDE ONCE. compress_last touches only the results the request ends
     on.
  7. THE RECORD carries sizes and why, never text.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_result_compress_")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")

import result_compress as rc  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def call(cid: str, name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def dump(n: int, width: int = 60, prefix: str = "node_modules/koota/dist/x.js"
         ) -> str:
    return "\n".join(f"{prefix}:{i}: " + ("x" * width) for i in range(n))


GREP = {"command": "cd /work/pagoda && grep -rn trait node_modules/koota/dist"}


def test_scope():
    big = dump(400)
    for name in ("read", "read_file", "search_files", "write_file", "grep"):
        out, rec = rc.compress(name, "{}", big)
        check(out is big and rec["why"] == "not_shell",
              f"{name}: not a shell tool, untouched", rec["why"])
    small = dump(20)
    out, rec = rc.compress("bash", json.dumps(GREP), small)
    check(out is small and rec["why"] == "under_cap",
          "under the cap: not a byte changes")
    check(rc.SHELL_TOOLS == {"terminal", "bash", "exec_command"},
          "the shell tools are the verified names only")
    check((rc.CAP_CHARS, rc.TAIL_LINES) == (8000, 500),
          "Atomic's numbers: 8,000 characters, 500 lines")


def test_cut_pi():
    lines = dump(400).split("\n")
    text = "\n\n".join(lines) + "\n\n\nCommand exited with code 0"
    out, rec = rc.compress("bash", json.dumps(GREP), text)
    check(rec["why"] == "compressed" and len(out) <= rc.CAP_CHARS,
          "over the cap: compressed within it", json.dumps(rec))
    check(out.endswith("Command exited with code 0"),
          "Pi's exit line (the tail) survives")
    body = out.split("\n")
    check(body[0].startswith("[") and "earlier lines" in body[0],
          "the first line says what was left out", body[0])
    kept = body[1:]
    orig = set(lines) | {"Command exited with code 0"}
    check(all(k in orig for k in kept), "every kept line is a whole line")
    check(kept[-2] == lines[-1] and "" not in kept,
          "the tail is kept, blank lines dropped")
    n = int(re.search(r"\[(\d+) earlier", body[0]).group(1))
    check(n + len(kept) == len(lines) + 1,
          "the count of lines left out is exact", f"{n}+{len(kept)}")
    check(rec["lines_kept"] == len(kept) and rec["after_chars"] == len(out)
          and rec["before_chars"] == len(text), "the record's sizes are exact")
    out2, _ = rc.compress("bash", json.dumps(GREP), text)
    check(out2 == out, "deterministic: the same input, the same bytes")


def test_cut_hermes():
    body = "Traceback (most recent call last):\nValueError: bad\n" + dump(300)
    text = json.dumps({"output": body, "exit_code": 1, "error": None})
    out, rec = rc.compress("terminal", json.dumps(GREP), text)
    d = json.loads(out)
    check(rec["why"] == "compressed" and len(out) <= rc.CAP_CHARS,
          "Hermes: compressed within the cap", json.dumps(rec))
    check(d["exit_code"] == 1 and d["error"] is None and
          list(d) == ["output", "exit_code", "error"],
          "Hermes' JSON keeps its fields, in order")
    check(d["output"].startswith("key: Traceback (most recent call last)"),
          "a structured failure keeps its first error line on top (Atomic)",
          d["output"][:80])
    ok_text = json.dumps({"output": body, "exit_code": 0, "error": None})
    out, _ = rc.compress("terminal", json.dumps(GREP), ok_text)
    check(not json.loads(out)["output"].startswith("key:"),
          "no key line on a success (Atomic: status error only)")
    check(out.isascii(), "Hermes' ascii JSON stays ascii")


def test_cut_codex_and_opencode():
    head = ("Chunk ID: 87eb5c\nWall time: 0.1 seconds\nProcess exited with "
            "code 0\nOriginal token count: 9000\nOutput:\n")
    text = head + dump(300)
    out, rec = rc.compress("exec_command", json.dumps(GREP), text)
    check(out.startswith(head) and len(out) <= rc.CAP_CHARS,
          "Codex: the header through Output: stays", out[:120])
    out, rec = rc.compress("bash", json.dumps(GREP), dump(300))
    check(rec["why"] == "compressed", "OpenCode plain text compresses")


def test_protected():
    msgs = [{"role": "user", "content": "build"},
            call("c1", "write", {"path": "/work/pagoda/src/main.tsx",
                                 "content": "export {}\n"})]
    w = rc.written_paths(msgs)
    check("src/main.tsx" in w, "a write call's path is written", str(w))
    text = dump(300, prefix="/work/pagoda/src/main.tsx")
    out, rec = rc.compress("bash", json.dumps({"command": "grep -n x src"}),
                           text, w)
    check(out is text and rec["why"] == "protected",
          "an output naming a written file passes whole")
    out, rec = rc.compress("bash", json.dumps(
        {"command": "cat src/main.tsx"}), dump(300), w)
    check(rec["why"] == "protected",
          "a command naming a written file passes whole (cat)")
    heredoc = ("cat > src/App.tsx <<'EOF'\nconst f = () => <div>{a > b}</div>"
               "\nx => y.z\nEOF\nnpx tsc 2>&1 | tee build.log")
    w2 = rc.written_paths([call("c2", "bash", {"command": heredoc})])
    check(w2 == {"src/App.tsx", "build.log"},
          "redirects outside the heredoc body only; `=>` and `2>&1` are not "
          "files", str(sorted(w2)))
    other = dump(300, prefix="node_modules/koota/dist/main.ts")
    out, rec = rc.compress("bash", json.dumps(GREP), other, w)
    check(rec["why"] == "compressed",
          "another package's file of a similar name is not the model's",
          rec["why"])


def test_bounds():
    text = dump(400)
    out, rec = rc.compress("bash", json.dumps(GREP), text, cap_chars=3000)
    check(len(out) <= 3000 and rec["cap"] == 3000,
          "the caller may lower the cap (the window)", str(len(out)))
    out, rec = rc.compress("bash", json.dumps(GREP), text, cap_chars=10 ** 6)
    check(rec["cap"] == rc.CAP_CHARS and len(out) <= rc.CAP_CHARS,
          "the caller never raises it past CAP_CHARS")
    b64 = "B64LEN=90000\n" + "A" * 90000 + "\nEND\ndone"
    out, rec = rc.compress("bash", json.dumps({"command": "base64 x.png"}),
                           b64)
    check(len(out) <= rc.CAP_CHARS and out.endswith("END\ndone"),
          "one line longer than the cap is left out; the lines after it stay",
          out[-40:])
    many = "\n".join("line %05d" % i for i in range(2000))
    out, rec = rc.compress("bash", json.dumps({"command": "yes l"}), many)
    check(rec["lines_kept"] == rc.TAIL_LINES,
          "at most TAIL_LINES lines are kept", str(rec.get("lines_kept")))


def test_compress_last():
    big = dump(300)
    msgs = [{"role": "user", "content": "go"},
            call("a", "bash", GREP),
            {"role": "tool", "tool_call_id": "a", "content": big},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "b", "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps(GREP)}},
                {"id": "c", "type": "function", "function": {
                    "name": "read", "arguments": '{"path": "x"}'}}]},
            {"role": "tool", "tool_call_id": "b", "content": big},
            {"role": "tool", "tool_call_id": "c", "content": big}]
    out, recs = rc.compress_last(msgs)
    check(out[2]["content"] is big,
          "a result the model already read is never shortened later")
    check(len(out[4]["content"]) <= rc.CAP_CHARS and out[5]["content"] is big,
          "the results the request ends on: shell compressed, read untouched")
    check([r["why"] for r in recs] == ["compressed", "not_shell"],
          "one record per trailing result", str(recs))
    check(msgs[4]["content"] is big, "the caller's messages are not mutated")
    out, recs = rc.compress_last(msgs[:3] + [{"role": "user", "content": "x"}])
    check(recs == [], "a request that ends on a user turn compresses nothing")
    # THE REPLAY: a later request (the model then WROTE a file the result
    # names) re-derives the same bytes for that result from its prefix.
    first, _ = rc.compress_at(msgs, 4)
    later = msgs + [call("w", "write", {
        "path": "node_modules/koota/dist/x.js", "content": "y"}),
        {"role": "tool", "tool_call_id": "w", "content": "ok"}]
    again, _ = rc.compress_at(later, 4)
    check(again == first,
          "a decision replays byte for byte on a later request (judged on "
          "the messages up to it)")


_FORBIDDEN = re.compile(r"\b(?:don'?t|do not|never|must not|critical|stop|"
                        r"may|might|perhaps|verify)\b", re.I)


def test_wording_and_record():
    line = rc.omitted_line(120, 9000, 80)
    check(not _FORBIDDEN.findall(line), "the omitted line: no prohibition, "
          "no doubt", line)
    check("run the command again narrowed" in line,
          "it carries the next step", line)
    _, rec = rc.compress("bash", json.dumps({"command": "cat .env"}),
                         "SECRET=1\n" + dump(300))
    blob = json.dumps(rec)
    check("SECRET" not in blob and ".env" not in blob,
          "the record carries no text", blob)


def main() -> int:
    for fn in (test_scope, test_cut_pi, test_cut_hermes,
               test_cut_codex_and_opencode, test_protected, test_bounds,
               test_compress_last, test_wording_and_record):
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
