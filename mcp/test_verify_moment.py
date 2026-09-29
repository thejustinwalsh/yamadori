#!/usr/bin/env python
"""THE VERIFY DIRECTIVE (mcp/verify_moment.py). No GPU, no network: the
decider is a fake model server behind the real decide_turn.Turn.choose.

    python mcp/test_verify_moment.py      -> "N/M checks passed"

GATES
  [kind]      the artifact kind from the files written and planned: the
              first KINDS row that matches, each row naming its signal; a
              React Native manifest beats the .html it also has.
  [moment]    the entry file's first write is a moment once per
              conversation; every planned file written is another (once per
              plan); a user turn is none.
  [options]   the CLIENT's own tools, name + the first sentence of its own
              description, on Hermes-, OpenCode- and Pi-shaped tool lists
              (bench/harness_shapes); more than CHUNK tools: rounds, then
              the winners.
  [choice]    the decider's pick through Turn.choose (the typed readout:
              two orders, no label prior; each round's decision id kept
              for the corpus join); "none", a tie, an unknown kind, an
              unavailable decider and a named line that did not land give
              the GENERIC line, each saying why.
  [skill]     the armed verify craft for the kind, or a recorded gap.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_verify_moment_")
os.environ["YAMADORI_SLOTS"] = "4"

import decider_bonsai as D  # noqa: E402
import decide_turn as T  # noqa: E402
import verify_moment as V  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail="") -> None:
    CHECKS.append((name, bool(ok), str(detail)[:300]))


IDS = {L: 100 + i for i, L in enumerate(D.LETTERS)}
IDS.update({" " + L: 200 + i for i, L in enumerate(D.LETTERS)})


def upstream(path, payload=None, timeout=30):
    if path == "/tokenize":
        s = payload["content"]
        return {"tokens": [IDS[s]] if s in IDS else [1, 2]}
    raise AssertionError(path)


def fake(prefer):
    """A model server whose next token favours (0.8) the option whose text
    `prefer(texts)` names; a blank (label-prior) read is uniform."""
    bodies = []

    def post(body, timeout):
        bodies.append(body)
        m = body["messages"]
        lines = m[2]["content"].split("OPTIONS:\n")[1].splitlines()
        labs = [ln.split(") ", 1) for ln in lines]
        blank = all(t == D.NEUTRAL_STATE for _l, t in labs)
        fav = None if blank else prefer([t for _l, t in labs])
        n = len(labs)
        top = [{"id": IDS[" " + lab], "logprob": math.log(
            (0.8 if t == fav else 0.2 / max(n - 1, 1)) if fav else 1 / n)}
            for lab, t in labs]
        top.sort(key=lambda t: -t["logprob"])
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 60},
            "timings": {"prompt_n": 20, "cache_n": 40, "prompt_ms": 5.0}}
    post.bodies = bodies
    return post


def by_name(word):
    return lambda texts: next((t for t in texts if t.startswith(word)),
                              None)


def shape(harness: str, name: str) -> list[dict]:
    with open(os.path.join(ROOT, "bench", "harness_shapes", harness,
                           name), encoding="utf-8") as f:
        return json.load(f)["body"]["tools"]


HERMES = shape("hermes", "chat__agent-step-write-js.json")
OPENCODE = shape("opencode", "chat__agent-step-write-py.json")
PI = shape("pi", "chat__write-html.json")
# Hermes' browser toolset, as the Octopus default arm turns it on
# (bench/octopus/toolset_arms.py); the descriptions here are ILLUSTRATIVE,
# the fixtures trim theirs.
BROWSER = [{"type": "function", "function": {
    "name": n, "description": d, "parameters": {"type": "object"}}}
    for n, d in (("browser_navigate", "Navigate to a URL in the browser. "
                  "Loads the page."),
                 ("browser_snapshot", "Get a text snapshot of the current "
                  "page."),
                 ("browser_vision", "Take a screenshot of the page and "
                  "describe it."))]


def call(name, args, cid):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def steps(*writes, task="Build a space shooter with canvas."):
    """A conversation of successful writes, one step each, ending on the
    last one's result."""
    m = [{"role": "user", "content": task}]
    for i, (path, content) in enumerate(writes):
        cid = f"w{i}"
        m += [{"role": "assistant", "content": "", "tool_calls": [call(
            "write_file", {"path": path, "content": content}, cid)]},
            {"role": "tool", "tool_call_id": cid,
             "content": '{"bytes_written": 10}'}]
    return m


def test_kind():
    k = V.artifact_kind(V.written(steps(("js/game.js", "x"),
                                         ("index.html", "<canvas>"))))
    check("[kind] an .html page: web_page, naming its signal",
          k["kind"] == "web_page" and ".html" in k["signal"]
          and k["path"] == "index.html", k)
    k = V.artifact_kind(V.written(steps(("vite.config.ts", "x"))))
    check("[kind] a Vite config: web_page", k["kind"] == "web_page", k)
    k = V.artifact_kind(V.written(steps(
        ("index.html", "<div>"),
        ("package.json", '{"dependencies": {"expo": "~51.0.0"}}'))))
    check("[kind] an Expo manifest beats the .html beside it: native_app",
          k["kind"] == "native_app" and "expo" in k["signal"], k)
    k = V.artifact_kind(V.written(steps(("Cargo.toml", "[package]"))))
    check("[kind] Cargo.toml: rust", k["kind"] == "rust", k)
    k = V.artifact_kind(V.written(steps(("tool/main.py", "print(1)"))))
    check("[kind] a .py file: python", k["kind"] == "python", k)
    k = V.artifact_kind(V.written(steps(("README.md", "# x"))))
    check("[kind] nothing it knows: unknown", k["kind"] is None, k)
    k = V.artifact_kind([], ["index.html", "js/game.js"])
    check("[kind] a planned file counts (the plan's FILES)",
          k["kind"] == "web_page", k)
    m = steps(("index.html", "<canvas>"))
    m[-1]["content"] = '{"error": "permission denied"}'
    check("[kind] a failed write is no file", not V.written(m))
    check("[kind] every KINDS row names its signal; every kind has an "
          "ENTRY", all(r.signal for r in V.KINDS)
          and {r.kind for r in V.KINDS} <= set(V.ENTRY))


def test_moment():
    vst: dict = {}
    m = steps(("js/game.js", "loop()"))
    check("[moment] a write that is no entry: none",
          V.moment(m, vst, None) is None)
    m = steps(("js/game.js", "loop()"), ("index.html", "<canvas>"))
    mo = V.moment(m, vst, None)
    check("[moment] the entry's first write: entry_written, once",
          mo and mo["moment"] == "entry_written" and mo["key"] ==
          "verify:entry" and mo["path"] == "index.html"
          and mo["kind"]["kind"] == "web_page", mo)
    check("[moment] the same request again: none (once per moment)",
          V.moment(m, vst, None) is None)
    m2 = steps(("index.html", "<canvas>"), ("index.html", "<canvas2>"))
    check("[moment] a later rewrite of the entry: none",
          V.moment(m2, vst, None) is None)
    auto = {"plan": {"sha": "abc", "files": ["index.html", "js/game.js"]},
            "plan_written": ["index.html", "js/game.js"]}
    m3 = steps(("index.html", "<canvas>"), ("js/game.js", "loop()"))
    mo = V.moment(m3, vst, auto)
    check("[moment] the plan's last file written: piece_done, keyed by the "
          "plan", mo and mo["moment"] == "piece_done"
          and mo["key"] == "verify:piece:abc", mo)
    check("[moment] once per plan", V.moment(m3, vst, auto) is None)
    check("[moment] a user turn is no moment",
          V.moment(m3 + [{"role": "user", "content": "go"}], {}, None)
          is None)


def test_options():
    for tools, name in ((HERMES, "hermes"), (OPENCODE, "opencode"),
                        (PI, "pi")):
        o = V.tool_options(tools)
        check(f"[options] {name}: every client tool, by its own name",
              [x["name"] for x in o] == [
                  (t.get("function") or t)["name"] for t in tools], o)
    long = [{"type": "function", "function": {
        "name": "x", "description": "Runs\na command. Second sentence."}}]
    o = V.tool_options(long)
    check("[options] a description is its first sentence, on one line",
          o[0]["text"] == "x: Runs a command.", o)


def test_choice():
    T.TEMPLATE_CF.clear()
    m = steps(("index.html", "<canvas>"))
    web = {"kind": "web_page", "label": "web page", "verb": "open it"}
    opts = V.tool_options(HERMES + BROWSER)
    post = fake(by_name("browser_navigate"))
    c = V.choose_tool(m, "web page", opts, post=post, upstream=upstream,
                      count=len, on=True)
    line, form, why = V.directive(web, c, {})
    check("[choice] Hermes with its browser, a web page: the decider "
          "picks browser_navigate; the NAMED line names it",
          c["judged"] and c["pick"] == "browser_navigate"
          and form == "named" and line == V.NAMED.format(
              verb="open it", tool="browser_navigate")
          and line[-1].isalpha(), (c, line))
    q = [b for b in post.bodies if "QUESTION" in json.dumps(b)]
    check("[choice] the question names the kind; 'None of these' is an "
          "option and never printed last",
          any("run this web page and show whether it works" in json.dumps(b)
              for b in q)
          and all(not b["messages"][2]["content"].rstrip().endswith(
              V.NONE_OPTION) for b in post.bodies), len(post.bodies))
    check("[choice] each round carries its decision id (the join key into "
          "the decision log); two orders read per round",
          c.get("decision_ids") and all(r.get("decision_id")
                                        for r in c["rounds"])
          and len(post.bodies) == 2 * len(c["rounds"]), c.get("rounds"))
    py = {"kind": "python", "label": "Python program", "verb": "run it"}
    c = V.choose_tool(m, "Python program", V.tool_options(OPENCODE),
                      post=fake(by_name("bash")), upstream=upstream,
                      count=len, on=True)
    line, form, _ = V.directive(py, c, {})
    check("[choice] OpenCode, a Python program: bash, named",
          c["pick"] == "bash" and form == "named"
          and "run it with bash" in line, (c, line))
    c = V.choose_tool(m, "web page", V.tool_options(PI),
                      post=fake(lambda texts: V.NONE_OPTION),
                      upstream=upstream, count=len, on=True)
    line, form, why = V.directive(web, c, {})
    check("[choice] Pi (read/bash/edit/write), a web page: the decider "
          "picks none -> the GENERIC line, saying why",
          c["judged"] and c["pick"] is None and c.get("none")
          and form == "generic" and line == V.GENERIC
          and "none" in why and line[-1].isalpha(), (c, why))
    c = V.choose_tool(m, "web page", V.tool_options(PI),
                      post=fake(lambda texts: None), upstream=upstream,
                      count=len, on=True)
    line, form, why = V.directive(web, c, {})
    check("[choice] a flat read is a tie: GENERIC", c["tie"]
          and form == "generic" and "tie" in why, (c, why))
    many = [{"type": "function", "function": {"name": f"tool_{i}",
                                             "description": "Does a thing."}}
            for i in range(30)] + BROWSER
    post = fake(by_name("browser_navigate"))
    c = V.choose_tool(m, "web page", V.tool_options(many), post=post,
                      upstream=upstream, count=len, on=True)
    check("[choice] 33 tools (more than the 25 single-letter labels "
          "leave): rounds of CHUNK, then the winners",
          V.CHUNK == len(D.LETTERS) - 1
          and c["pick"] == "browser_navigate" and len(c["rounds"]) == 3, c)

    def down(body, timeout):
        raise ConnectionRefusedError("refused")
    c = V.choose_tool(m, "web page", opts, post=down, upstream=upstream,
                      count=len, on=True)
    line, form, why = V.directive(web, c, {})
    check("[choice] the decider unavailable: GENERIC, with its facts",
          not c["judged"] and form == "generic"
          and (c.get("failure") or {}).get("code") == "MODEL_UNREACHABLE",
          (c, why))
    c = V.choose_tool(m, "web page", opts, post=post, upstream=upstream)
    check("[choice] off unless the serving process enabled it",
          not c["judged"] and "off" in c.get("why", ""), c)
    line, form, why = V.directive({"kind": None}, {"judged": True,
                                                   "pick": "x"}, {})
    check("[choice] an unknown kind: GENERIC", form == "generic"
          and "unknown" in why)
    # A NAMED line that did not land: the turn that opened with it called
    # something else -> later moments are GENERIC.
    named = V.NAMED.format(verb="open it", tool="browser_navigate")
    vst = {"named": {"line": named, "tool": "browser_navigate",
                     "moment": "verify:entry"}}
    m2 = m + [{"role": "assistant", "content": named + ".", "tool_calls": [
        call("write_file", {"path": "js/game.js", "content": "x"}, "n1")]},
        {"role": "tool", "tool_call_id": "n1", "content": "ok"}]
    V.observe(m2, vst)
    line, form, why = V.directive(web, {"judged": True,
                                        "pick": "browser_navigate"}, vst)
    check("[choice] a named line that did not land: the next moment is "
          "GENERIC, saying which tool was called instead",
          vst["named"]["landed"] is False and form == "generic"
          and "did not land" in why and "write_file" in why, (vst, why))
    vst2 = {"named": dict(vst["named"])}
    vst2["named"].pop("landed")
    m3 = m + [{"role": "assistant", "content": named + ".", "tool_calls": [
        call("browser_navigate", {"url": "http://localhost"}, "n2")]}]
    V.observe(m3, vst2)
    check("[choice] a named line that landed is recorded so",
          vst2["named"]["landed"] is True and "named_failed" not in vst2)


def test_skill():
    pool = [{"id": "s-web-check", "name": "check-web-page",
             "rule": {"phase": ["verify"], "language": [{"id": "html"}]}},
            {"id": "s-py", "name": "py-impl",
             "rule": {"phase": ["implement"], "language": ["python"]}}]
    check("[skill] the armed verify craft for a web page is found",
          (V.verify_skill("web_page", pool) or {}).get("id")
          == "s-web-check")
    check("[skill] none for python here (only an implement craft): a gap",
          V.verify_skill("python", pool) is None)
    check("[skill] a native app has no taxonomy term yet: a gap",
          V.verify_skill("native_app", pool) is None)


def main() -> int:
    # The decider's slot release goes to the model server: not here.
    D.release = lambda slot, why="": {"released": True, "slot": slot}
    for fn in (test_kind, test_moment, test_options, test_choice,
               test_skill):
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
            check(f"{fn.__name__} itself raised", False)
    ok = 0
    for name, good, detail in CHECKS:
        print(("  pass  " if good else "  FAIL  ") + name
              + ("" if good else f"   <- {detail}"))
        ok += good
    print(f"\n  {ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
