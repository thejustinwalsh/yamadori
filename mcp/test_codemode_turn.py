#!/usr/bin/env python
"""The server toolbox THROUGH THE PROXY (yama_compose_tools on main): the offer and its switch, the hidden hop,
the accounting, the caps and the package skills' sections inside a script, the ledger, the chat and Responses
wires. No GPU, no network, no model server: test_ledger's served template and fake upstream, the fake MCP server,
a fake jjava (jev_api.systemone), a fake image backend, the real QuickJS child process.

    python mcp/test_codemode_turn.py      -> "N/M checks passed"

GATED HERE:
  1. THE OFFER: off by default (no tool, no line, a record that says why); the header turns it on; the tool goes
     AFTER the client's, its description carries the declarations, the one system line is added; decided on the
     first request and KEPT (the tool list and the description are byte-identical on every later request); mode
     `only` hides the MCP lookups it replaces and their system line, and is size-neutral against them; a hidden tool
     called by name still runs; a client tool of the same name withholds the toolbox.
  2. THE HIDDEN HOP: a script's several lookups are ONE hidden hop (tool_turns 1), the client never sees it,
     x_yamadori.codemode.calls / .mcp.calls / .tools carry the tools' own yama_* names; package skills sections
     ride AFTER the output, once, in call order; discovery is recorded; the ledger replays the hop byte for byte
     and the next request EXTENDS the slot (served template).
  3. FAILURES: partial output and the line; a timeout (timeout_ms) with the output so far; an empty script.
  4. THE CAPS INSIDE A SCRIPT: craft.recall obeys the per-request cap of 3 and the do-not-repeat line, and counts
     in x_yamadori.craft; the request's inner server calls are bounded by tool_turn_limit across its scripts;
     compose calls past the tool-turn limit land with the client's tools kept.
  5. jjava (yama_classify_*) through the Jev API's entry; images drawn inside a script are shown by the proxy.
  6. THE WIRES: Responses, blocking and streamed; streamed == blocking.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_package_skills as PK  # noqa: E402  (its fakes: the armed pool, the MCP fixture, test_ledger's template)
import test_mcp_host as H  # noqa: E402
import test_ledger as T  # noqa: E402
import test_responses_api as RT  # noqa: E402
import codemode as C  # noqa: E402
import mcp_config  # noqa: E402
import mcp_host  # noqa: E402
import proxy  # noqa: E402
import tiers  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


class CM(H.Conv):
    """A client with the toolbox on (the header), at medium."""

    def __init__(self, tag: str, cm_mode: str = "on", package_skills: bool = False, **kw):
        f = {"skills": False, "codemode": True, "codemode_mode": cm_mode, "package_skills": package_skills,
             "package_skills_mode": "inject"}
        f.update(kw.pop("features", None) or {})
        super().__init__(tag, features=f, **kw)


def codemode_line_tokens() -> int:
    return len(C.SYSTEM_LINE) // 3 + 1


def compose(script: str, cid: str = "z1", **extra) -> dict:
    return T.call(C.TOOL_NAME, dict({"script": script}, **extra), cid)


def hop(t: dict, which: int, cid: str) -> str:
    msgs = t["gens"][which]["request"]["messages"]
    return next((m["content"] for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == cid), "")


def names_of(gen: dict) -> list[str]:
    return [x["function"]["name"] for x in gen["request"].get("tools") or []]


def fresh() -> None:
    T.slots.reset(n=4)
    H.fresh_host()


def first(tag: str, user: str = "Build a voxel scene with pmndrs math.", cls=CM, **kw):
    fresh()
    c = cls(tag, **kw)
    t = c.turn([T.reply("Ok.")], user=user)
    return c, t


# ================================================================== 1. offer
def test_the_offer():
    fresh()
    off = H.Conv("off", features={"skills": False, "codemode": False})
    t = off.turn([T.reply("Ok.")], user="Build a voxel scene with pmndrs math.")
    g = t["gens"][0]
    cm = t["x"].get("codemode")
    check(C.TOOL_NAME not in names_of(g) and C.SYSTEM_LINE.strip() not in H.system_of(g)
          and cm and cm["offered"] is False and cm["switch"]["on"] is False and cm["switch"]["source"] == "header"
          and "switch codemode off" in (cm.get("why") or ""),
          "switched off (header): no tool, no line; the record says the switch is off and why", json.dumps(cm)[:300])
    c, t = first("on")
    g = t["gens"][0]
    names = names_of(g)
    cm = t["x"]["codemode"]
    check(names[-1] == C.TOOL_NAME and names[:1] == ["write_file"] and set(H.NAMES) <= set(names),
          "the header turns it on: the client's tool first, ours after, the toolbox LAST; the individual lookups "
          "stay (mode on)", json.dumps(names))
    check(C.SYSTEM_LINE.strip() in H.system_of(g) and H.LINE.strip() in H.system_of(g),
          "the one system line is added beside the MCP line")
    d = next(x for x in g["request"]["tools"] if x["function"]["name"] == C.TOOL_NAME)["function"]
    check(d["description"].startswith("Answers 'how do I get several facts from the server in one go?':")
          and "yama_find_package(args:" in d["description"] and "declare const tools" in d["description"],
          "the description opens with the question, then trigger conditions, then the declarations",
          d["description"][:120])
    check(cm["offered"] is True and cm["mode"] == "on" and cm["kept"] is False
          and cm["tokens"] <= cm["budget_tokens"] <= 3000 and set(H.NAMES) <= set(cm["names"])
          and set(cm["declared"]) | set(cm["left_out"]) == set(cm["names"]) and cm["declared"]
          and cm["switch"]["source"] == "header",
          "x_yamadori.codemode: offered, mode, the tokens against the budget, what is declared",
          json.dumps({k: v for k, v in cm.items() if k not in ("names",)})[:400])
    # kept for the conversation
    c.tool_result("w1", "ok")
    t2 = c.turn([T.reply("Fine.")], user="Another thing.")
    g2 = t2["gens"][0]
    d2 = next((x for x in g2["request"]["tools"] if x["function"]["name"] == C.TOOL_NAME), None)
    check(d2 is not None and json.dumps(d2) == json.dumps({"type": "function", "function": d})
          and json.dumps(g2["request"]["tools"]) == json.dumps(g["request"]["tools"])
          and g2["request"]["tools"] == g["request"]["tools"]
          and H.system_of(g2) == H.system_of(g) and t2["x"]["codemode"]["kept"] is True,
          "KEPT: the tool list, the description and the system block are byte-identical on the next request")
    # the continuing conversation that began without a decision gets none
    # only: replaces the lookups, size-neutral against them
    fresh()
    base = H.Conv("size-off", features={"skills": False})
    tb = base.turn([T.reply("Ok.")], user="Build a voxel scene with pmndrs math.")
    only = CM("only", cm_mode="only")
    to = only.turn([T.reply("Ok.")], user="Build a voxel scene with pmndrs math.")
    go = to["gens"][0]
    cmo = to["x"]["codemode"]
    check(names_of(go)[0] == "write_file" and names_of(go)[-1] == C.TOOL_NAME
          and not set(H.NAMES) & set(names_of(go)) and set(cmo["hidden"]) == set(H.NAMES)
          and H.LINE.strip() not in H.system_of(go) and C.SYSTEM_LINE.strip() in H.system_of(go),
          "mode only: the MCP lookups and their system line are hidden; the toolbox and its line replace them",
          json.dumps([names_of(go), cmo.get("hidden")]))
    base_chars = len(json.dumps(tb["gens"][0]["request"]["tools"]))
    only_chars = len(json.dumps(go["request"]["tools"]))
    line = len(H.LINE)
    check(only_chars <= base_chars + line and cmo["tokens"] <= cmo["budget_tokens"],
          "…and it is size-neutral: tools + line are no bigger than the tools + line they replace",
          f"{only_chars} vs {base_chars} + {line}")
    off_ours = (tb["x"]["codemode"].get("ours") or {})
    check(off_ours.get("tokens") and set(H.NAMES) <= set(off_ours["tools"])
          and cmo["ours"]["tokens"] <= off_ours["tokens"] + codemode_line_tokens()
          and C.TOOL_NAME in cmo["ours"]["tools"] and not set(H.NAMES) & set(cmo["ours"]["tools"]),
          "x_yamadori.codemode.ours records what our additions cost the prompt, toolbox on or off (the probe's "
          "figure): mode only is no bigger than today's tools + lines",
          json.dumps([off_ours.get("tokens"), cmo["ours"]["tokens"]]))
    # a hidden tool called by name still runs
    only.tool_result("w1", "ok")
    t = only.turn([T.reply("", calls=[T.call("yama_find_package", {"query": "pmndrs math", "ecosystem": "npm"},
                                              "m1")]), T.reply("Done.")], user="Which package is math?")
    check(hop(t, 1, "m1").startswith("SOURCE: the npm registry") and not t["m"].get("tool_calls"),
          "mode only: a hidden lookup called by name still runs as a hidden hop (never forwarded to the client)",
          hop(t, 1, "m1")[:80])
    # no sandbox engine: no toolbox, and the record says why
    real_available = C.available
    C.available = lambda: (False, "quickjs is not importable (test)")
    try:
        fresh()
        t = CM("noengine").turn([T.reply("Ok.")], user="Build it.")
    finally:
        C.available = real_available
    check(C.TOOL_NAME not in names_of(t["gens"][0]) and "sandbox is not available" in (t["x"]["codemode"]["why"] or ""),
          "the sandbox engine missing: the toolbox is not offered (never a broken tool), and the record says why")
    # a client tool of the same name
    fresh()
    clash = CM("clash", tools=[T.WRITE, {"type": "function", "function": {
        "name": "yama-compose-tools", "description": "x", "parameters": {"type": "object", "properties": {}}}}])
    t = clash.turn([T.reply("Ok.")], user="Build it.")
    check(C.TOOL_NAME not in names_of(t["gens"][0]) and "named" in (t["x"]["codemode"].get("why") or ""),
          "a client tool of the same (normalised) name withholds the toolbox, and the record says so",
          json.dumps(t["x"]["codemode"])[:200])
    # THE HARNESS'S OWN CODE MODE (operator, 2026-10-08): the toolbox is the fallback for harnesses without one
    pi_codemode = {"type": "function", "function": {
        "name": "codemode", "description": "Run JavaScript that calls the other tools.",
        "parameters": {"type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"]}}}
    for mode in ("on", "only"):
        fresh()
        pi = CM(f"pi-{mode}", cm_mode=mode, tools=[T.WRITE, pi_codemode])
        t = pi.turn([T.reply("Ok.")], user="Build a voxel scene with pmndrs math.")
        g = t["gens"][0]
        cmr = t["x"]["codemode"]
        withheld = [w for w in t["x"]["tools_withheld"] if w["ours"] == C.TOOL_NAME]
        check(C.TOOL_NAME not in names_of(g) and set(H.NAMES) <= set(names_of(g)) and "codemode" in names_of(g)
              and cmr["offered"] is False and cmr["why"] == "the client brings its own code mode (codemode)"
              and not cmr.get("hidden") and C.SYSTEM_LINE.strip() not in H.system_of(g)
              and H.LINE.strip() in H.system_of(g)
              and len(withheld) == 1 and withheld[0]["client_tool"] == "codemode"
              and withheld[0]["because"] == C.HARNESS_OVERLAP_WHY,
              f"a Pi-shaped request (a `codemode` tool), switch on, mode {mode}: no yama_compose_tools, the "
              "lookups stay offered as direct tools (nothing is hidden behind a toolbox that is not there), the "
              "MCP line stays; x_yamadori.codemode says why and tools_withheld names the client's tool",
              json.dumps([names_of(g), cmr.get("why"), withheld]))
        pi.tool_result("w1", "ok")
        t2 = pi.turn([T.reply("Fine.")], user="Another thing.")
        check(t2["x"]["codemode"]["kept"] is True and C.TOOL_NAME not in names_of(t2["gens"][0])
              and names_of(t2["gens"][0]) == names_of(g)
              and [w for w in t2["x"]["tools_withheld"] if w["ours"] == C.TOOL_NAME],
              f"mode {mode}: the decision is kept (and still recorded as withheld) on the next request")
    fresh()
    hermes = CM("hermes-like", tools=[T.WRITE, {"type": "function", "function": {
        "name": "terminal", "description": "Run a command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}])
    t = hermes.turn([T.reply("Ok.")], user="Build a voxel scene with pmndrs math.")
    check(names_of(t["gens"][0])[-1] == C.TOOL_NAME and t["x"]["codemode"]["offered"] is True,
          "a Hermes-shaped request (no code mode among its tools) gets the toolbox when the switch is on")
    check(C.HARNESS_TOOLS == ("codemode",) and any(o == C.TOOL_NAME and cl == C.HARNESS_TOOLS
                                                   for o, cl, _w in proxy.TOOL_OVERLAPS),
          "the harness code-mode tool names are ONE constant (codemode.HARNESS_TOOLS) and a declared overlap "
          "(proxy.TOOL_OVERLAPS)")
    # a tool withheld for a client tool is not in the toolbox
    fresh()
    own = CM("own", tools=[T.WRITE, {"type": "function", "function": {
        "name": "smart_search", "description": "x", "parameters": {"type": "object", "properties": {}}}}])
    t = own.turn([T.reply("Ok.")], user="Build it.")
    check("yama_find_package" not in t["x"]["codemode"]["names"] and "yama_find_package" not in names_of(t["gens"][0])
          and "yama_list_package_versions" in t["x"]["codemode"]["names"],
          "a lookup withheld for a client tool that answers the same question is not in the toolbox either",
          json.dumps(t["x"]["codemode"]["names"]))
    # the continuing conversation with no decision
    fresh()
    cont = CM("cont")
    cont.msgs += [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    t = cont.turn([T.reply("Ok.")], user="carry on")
    check(C.TOOL_NAME not in names_of(t["gens"][0]) and "without a toolbox offer" in (t["x"]["codemode"]["why"] or ""),
          "a conversation that began without a decision gets none (the tool list never changes mid-conversation)")


# ============================================================== 2. hidden hop
SCRIPT = ("const hits = tools.search('versions of a package', {limit: 3});"
          "const [f, v, r] = await Promise.all(["
          "tools.yama_find_package({query: 'pmndrs math', ecosystem: 'npm'}),"
          "tools.yama_list_package_versions({package: 'math', ecosystem: 'npm', limit: 2}),"
          "tools.yama_read_package_readme({package: 'math', ecosystem: 'npm'})]);"
          "text(hits.map(h => h.name));"
          "return [f.ok, v.ok, r.ok, f.data.names.join(), f.text.split('\\n')[0]].join(' | ')")


def _session(tag: str, mode: str = "on", package_skills: bool = True, streamed: bool = False):
    fresh()
    c = CM(tag, cm_mode=mode, package_skills=package_skills)
    t1 = c.turn([T.reply("", reasoning="Several lookups at once.", calls=[compose(SCRIPT)]),
                 T.reply("", reasoning="Now write.", calls=[T.call("write_file", H.PKG, "w1")])],
                user="Build a voxel scene with pmndrs math.", streamed=streamed)
    c.tool_result("w1", "wrote package.json")
    t2 = c.turn([T.reply("Done.")], streamed=streamed)
    return c, t1, t2


def test_one_hidden_hop():
    c, t1, t2 = _session("hop")
    out = hop(t1, 1, "z1")
    check(out.startswith("Script completed in ") and "==> text 1/2 <==" in out
          and "yama_list_package_versions" in out.split("==> text 2/2")[0]
          and "true | true | true | math,mathjs | SOURCE: the npm registry" in out,
          "the script's three lookups ran together and its output is the hop's result", out[:420])
    check([x["function"]["name"] for x in t1["m"].get("tool_calls") or []] == ["write_file"],
          "the client never sees the hidden hop: it gets its own write call")
    tt = t1["x"]["tool_turns"]
    check(tt["turns"] == 1 and tt["hit"] is False,
          "ONE yama_compose_tools call is ONE hidden hop toward the tool-turn limit however many lookups",
          json.dumps(tt))
    cm = t1["x"]["codemode"]
    mc = [x["tool"] for x in t1["x"]["mcp"]["calls"]]
    check(sorted(x["tool"] for x in cm["calls"]) == sorted(H.NAMES[:3]) and all(x["ok"] for x in cm["calls"])
          and sorted(mc) == sorted(H.NAMES[:3]) and cm["failed"] is False and cm["runs"] == 1
          and cm["wall_ms"] > 0 and cm["output_chars"] > 0 and cm["searches"] == 1,
          "x_yamadori.codemode.calls and x_yamadori.mcp.calls carry the tools' own yama_* names",
          json.dumps({k: cm[k] for k in ("calls", "runs", "failed", "searches")})[:300])
    ev = [e["name"] for e in t1["x"]["tools"]]
    check(ev.count(C.TOOL_NAME) == 1 and set(H.NAMES[:3]) <= set(ev),
          "x_yamadori.tools lists the compose call and each inner call by name", json.dumps(ev))
    check(cm["discovery"][0]["kind"] == "search" and cm["discovery"][0]["hits"][0] == "yama_list_package_versions",
          "discovery is recorded: what was searched and what it found", json.dumps(cm["discovery"]))
    pk = (t1["x"].get("skills") or {}).get("package") or []
    check(pk and out.count(PK.P.CRAFT_HEADER) == len(pk) and out.index(PK.P.CRAFT_HEADER) > out.rindex("true | true")
          and out.index("- DO: Call vec3.add") > out.rindex("true | true"),
          "the package skills sections ride AFTER the output (never in a JS value): one per lookup the channel "
          "decided on, as the individual tools' do", f"{out.count(PK.P.CRAFT_HEADER)} vs {len(pk)}: {out[-200:]}")
    check(pk and [p["tool"] for p in pk] == H.NAMES[:3] and pk[0]["package"] == "math"
          and pk[0]["skills"][0]["form"] == "body",
          "…decided in the order the calls were ISSUED (the first lookup got the body), whatever order they "
          "finished in", json.dumps([p["tool"] for p in pk]))
    check(T._extends(t1, t2, "[codemode] the request after the toolbox hop"),
          "the request after the hidden hop EXTENDS what the slot holds (served template)")
    check(hop(t2, 0, "z1") == out, "the ledger replays the hop byte for byte")
    # streamed == blocking
    cs_, s1, s2 = _session("hop-s", streamed=True)
    check(T._extends(s1, s2, "[codemode] streamed") and hop(s1, 1, "z1").split(" in ", 1)[1].split("\n", 1)[1]
          == out.split(" in ", 1)[1].split("\n", 1)[1],
          "streamed == blocking (same output; the wall time differs), and the next request extends the slot")


def test_only_mode_session():
    c, t1, t2 = _session("only-hop", mode="only")
    out = hop(t1, 1, "z1")
    check(out.startswith("Script completed") and "true | true | true" in out
          and names_of(t1["gens"][0])[-1] == C.TOOL_NAME and not set(H.NAMES) & set(names_of(t1["gens"][0])),
          "mode only: the whole lookup session runs through the toolbox")
    check(T._extends(t1, t2, "[codemode only] next request"), "…and the next request extends the slot")


# ================================================================ 3. failures
def test_failures():
    fresh()
    c = CM("fail")
    t = c.turn([T.reply("", calls=[compose("text('part one'); await tools.yama_find_package({query: 'x', "
                                           "ecosystem: 'npm'}); nothingHere(); text('lost')", "f1")]),
                T.reply("", calls=[compose("while (true) {}", "f2", timeout_ms=1500)]),
                T.reply("", calls=[T.call(C.TOOL_NAME, {"script": ""}, "f3")]),
                T.reply("", calls=[T.call(C.TOOL_NAME, {"script": "return 'ok'", "timeout_ms": 5000}, "f4")]),
                T.reply("Done.")], user="Try things.")
    a, b = hop(t, 4, "f1"), hop(t, 4, "f2")
    check(a.startswith("Script failed in ") and "ReferenceError: nothingHere is not defined (line 1, column" in a
          and "Output before the failure:\n\npart one" in a and "lost" not in a and "Next: fix the line" in a,
          "a failing script: the error, its line, the output before it, the next step", a[:300])
    check(b.startswith("Script failed in 1.") and "ran past its 1.5 s limit" in b and "Next: the calls that finished"
          in b, "timeout_ms ends a runaway script (the child is killed)", b[:300])
    e = json.loads(hop(t, 4, "f3"))
    check(e["ok"] is False and e["error"] == "BAD_ARGUMENTS" and e["retryable"] is True
          and e["remedies"][0]["fixable_by"] == "agent", "an empty script: a failure envelope with the next step",
          json.dumps(e)[:200])
    check(hop(t, 4, "f4").endswith("\n\nok"), "a script that returns a string: that string")
    cm = t["x"]["codemode"]
    check(cm["runs"] == 3 and cm["failed"] is True and cm["reasons"] == ["js_error", "timeout", "completed"],
          "the record: runs (an empty script never ran), failed, the reason of each run", json.dumps(cm["reasons"]))
    tt = t["x"]["tool_turns"]
    check(tt["turns"] == 4, "each compose call is one tool turn", json.dumps(tt))


# ========================================================== 4. caps in scripts
def test_the_caps_inside_a_script():
    import skill_match
    limit = skill_match.BODIES_PER_DECISION
    fresh()
    c = CM("caps", package_skills=True)
    names = PK.CRAFT_NAMES
    script = ("const names = %s;"
              "const r = await Promise.all(names.slice(0, 6).map(n => tools.yama_recall_craft(n)));"
              "return r.map(x => x.ok ? 'read' : x.error.code).join(',')" % json.dumps(names))
    t1 = c.turn([T.reply("", calls=[compose(script, "k1")]), T.reply("", calls=[T.call("write_file", H.PKG, "w1")])],
                user="Build a voxel scene with koota.")
    out = hop(t1, 1, "k1")
    res = out.split("\n\n", 1)[1].split("\n")[0]
    got = res.split(",")
    check(got.count("read") == limit and got.count("CRAFT_CAP") == 6 - limit,
          f"craft.recall inside a script obeys the per-request cap of {limit}: the rest are CRAFT_CAP results",
          res)
    cr = t1["x"]["craft"]
    check(cr["capped"] == 6 - limit and len(cr["reads"]) == limit,
          "…and counts in x_yamadori.craft (reads, capped)", json.dumps({k: cr[k] for k in ("capped",)}))
    check(t1["x"]["tool_turns"]["turns"] == 1,
          "…and the whole script is one tool turn", json.dumps(t1["x"]["tool_turns"]))
    # the do-not-repeat line, from inside a script (a conversation of its own: the crafts above were read)
    fresh()
    c = CM("repeat", package_skills=True)
    script = ("const a = await tools.yama_recall_craft('%s'); const b = await tools.yama_recall_craft('%s');"
              "return [a.text.slice(0, 40), b.text].join(' || ')" % (names[0], names[0]))
    t2 = c.turn([T.reply("", calls=[compose(script, "k2")]), T.reply("Done.")], user="About koota.")
    res = hop(t2, 1, "k2")
    check(f"Craft {names[0]}" in res.split("||")[0] and "already given above in this conversation" in res
          .split("||")[1] and len(t2["x"]["craft"]["repeat"]) == 1,
          "the same craft twice in a script: the do-not-repeat line the second time, recorded in "
          "x_yamadori.craft.repeat", res[:300])
    # the request's server calls are bounded by tool_turn_limit across its scripts
    lim = tiers.tool_turn_limit(tiers.resolve({"reasoning_effort": "medium"}))
    fresh()
    c = CM("calls")
    many = ("const r = await Promise.all(Array.from({length: %d}, (_, i) => tools.yama_find_package({query: "
            "'math ' + i, ecosystem: 'npm'}))); return r.map(x => x.ok ? 'ok' : x.error.code).join(',')")
    t = c.turn([T.reply("", calls=[compose(many % (lim + 2), "m1")]),
                T.reply("", calls=[compose(many % 2, "m2")]),
                T.reply("Done.")], user="Look up lots.")
    r1 = hop(t, 2, "m1").split("\n\n", 1)[1].split("\n")[0].split(",")
    r2 = hop(t, 2, "m2").split("\n\n", 1)[1].split("\n")[0].split(",")
    check(r1.count("ok") == lim and r1.count("CALL_CAP") == 2 and r2 == ["CALL_CAP", "CALL_CAP"],
          f"inner server calls are bounded by tool_turn_limit ({lim}) across all of a request's scripts; past it "
          "a call is not run and says so", json.dumps([r1, r2]))
    check(len(t["x"]["mcp"]["calls"]) == lim and t["x"]["tool_turns"]["turns"] == 2,
          "…only the calls that ran are recorded; two scripts are two tool turns",
          json.dumps([len(t["x"]["mcp"]["calls"]), t["x"]["tool_turns"]]))
    cm_calls = t["x"]["codemode"]["calls"]
    check(len(cm_calls) == lim, "x_yamadori.codemode.calls: only what ran", str(len(cm_calls)))
    # compose calls past the tool-turn limit land, the client's tools kept
    fresh()
    c = CM("land")
    script = [T.reply("", calls=[compose("return %d" % i, f"l{i}")]) for i in range(lim)]
    script += [T.reply("", calls=[T.call("write_file", H.PKG, "w9")])]
    t = c.turn(script, user="Keep composing.")
    tt = t["x"]["tool_turns"]
    check(tt["hit"] is True and tt["turns"] == lim and [x["function"]["name"] for x in
                                                         t["m"].get("tool_calls") or []] == ["write_file"],
          f"{lim} compose calls hit the tool-turn limit: our tools are withdrawn, the client's kept, its write "
          "flows", json.dumps(tt))
    last = t["gens"][-1]
    check(C.TOOL_NAME not in names_of(last) and "write_file" in names_of(last),
          "…the landing request carries no toolbox and still the client's tools")


# ============================================================ 5. jjava, images
def test_decide_and_images():
    import images
    import jev_api
    import vision
    seen: list = []

    def fake_systemone(body, account, *, route="", request_id=None):
        seen.append((body, account, route))
        q = body["questions"]["q"]
        a = {"noul": {"type": "noul", "noul": 0.93}, "choice": {"type": "choice", "choice": "b",
                                                                "probabilities": {"a": 0.1, "b": 0.9},
                                                                "confidence": 0.8},
             "score": {"type": "score", "score": 2, "probabilities": {"0": 0.1, "2": 0.9}, "confidence": 0.7}}[
            q["type"]]
        if body["state"] == "refuse":
            return 529, {"detail": "jjava's model is not available now"}, {"Retry-After": "7"}
        return 200, {"model": "jjava-bonsai", "answers": {"q": a}, "usage": {}}, {}
    real = (jev_api.enabled, jev_api.systemone, images.configured, images.run_tool, vision.enabled)
    jev_api.enabled = lambda: (True, "test")
    jev_api.systemone = fake_systemone
    images.configured = lambda: True

    def fake_run_tool(args, base, record=None, account=None, options=None):
        if record is not None:
            record.append({"ok": True, "id": "abc"})
        return json.dumps({"tool": images.TOOL_NAME, "ok": True, "markdown": "![a cat](http://x/media/cat.png)",
                           "url": "http://x/media/cat.png", "id": "abc"})
    images.run_tool = fake_run_tool
    vision.enabled = lambda: True
    stub = proxy.run_our_tool
    proxy.run_our_tool = T._real_run_our_tool          # (test_ledger stubs the image tool; this one is faked below)
    try:
        fresh()
        c = CM("jj")
        script = ("const n = await tools.yama_classify_yes_no({state: 'koota is an ECS', question: 'about ECS?'});"
                  "const ch = await tools.yama_classify_choice({state: 'x', question: 'which?', options: "
                  "{a: 'one', b: 'two'}});"
                  "const sc = await tools.yama_classify_score({state: 'x', question: 'how much?', levels: "
                  "['none', 'some', 'lots']});"
                  "const bad = await tools.yama_classify_yes_no({state: 'refuse', question: 'q?'});"
                  "const img = await tools.yama_generate_image({prompt: 'a cat'});"
                  "return JSON.stringify([n.data, ch.data.choice, sc.data.score, bad.error.code, bad.error.retryable,"
                  " bad.error.next, img.ok, img.data.url, img.text.includes('already shown')])")
        t = c.turn([T.reply("", calls=[compose(script, "j1")]), T.reply("Here it is.")],
                   user="Classify and draw.")
        out = hop(t, 1, "j1")
        line = next((ln for ln in out.split("\n") if ln.startswith("[{")), "")
        res = json.loads(line) if line else [out]
        check(res[0] == {"p_true": 0.93} and res[1] == "b" and res[2] == 2 and res[3] == "JJAVA_529"
              and res[4] is True and "7 s" in res[5] and res[6] is True and res[7].endswith("cat.png"),
              "yama_classify_yes_no / choice / score go through the Jev API's entry; a refusal is a result with its "
              "retry; yama_generate_image works inside a script", json.dumps(res))
        body, account, route = seen[0]
        check(body["model"] == "jjava-latest" and body["questions"]["q"] == {"type": "noul", "instructions":
                                                                             "about ECS?"}
              and body["state"] == "koota is an ECS" and route == "/codemode/decide" and account,
              "…the question as jjava takes it, with the account and a codemode route", json.dumps(seen[0])[:200])
        check(seen[1][0]["questions"]["q"]["criteria"] == {"a": "one", "b": "two"}
              and seen[2][0]["questions"]["q"]["criteria"] == ["none", "some", "lots"],
              "…choice options and score levels travel as Jev's criteria")
        shown = (t["m"].get("content") or "")
        check("![a cat](http://x/media/cat.png)" in shown,
              "an image drawn inside a script is SHOWN by the proxy (as yama_generate_image's is)", shown[:160])
        check(shown.count("![a cat]") == 1 and res[8] is True,
              "…and the script's result says the picture is already shown", out[-300:])
        cm = t["x"]["codemode"]
        check("decide" not in json.dumps(cm["names"]).replace("yama_classify", "") and set(
            n for n in cm["names"] if n.startswith("yama_classify")) == set(C.DECIDE_NAMES),
              "the decide functions are in the toolbox (the decider is on)")
        check(any(i.get("emitted") for i in t["x"]["images"]), "x_yamadori.images records when it was shown",
              json.dumps(t["x"]["images"]))
    finally:
        proxy.run_our_tool = stub
        (jev_api.enabled, jev_api.systemone, images.configured, images.run_tool, vision.enabled) = real


def test_store_image_and_the_output_cap_in_a_turn():
    """store/load carry across a conversation's scripts; image() is shown by the proxy; the result may be longer
    than the 6,000-character breaker of the other tools (OUTPUT_CAP)."""
    import base64
    import io
    from PIL import Image
    import images
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 90, 200)).save(buf, format="PNG")
    durl = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    saved_base = images.public_base
    images.public_base = lambda *a, **k: "http://x"
    try:
        fresh()
        c = CM("si")
        s1 = ("store('seen', ['koota', 'zustand']); image(%s); return 'a'.repeat(12000)" % json.dumps(durl))
        t1 = c.turn([T.reply("", calls=[compose(s1, "s1")]), T.reply("Done.")], user="Store a list, show a picture.")
        out1 = hop(t1, 1, "s1")
        check("Script completed" in out1 and len(out1) > 12000 and "[image 1 shown to the user]" in out1
              and "output cut" not in out1,
              "a 12,000-character result comes back whole (the cap is 30,000, not 6,000) and says the image is shown",
              f"{len(out1)} {out1[:120]!r}")
        shown = (t1["m"].get("content") or "")
        check("![image 1](http://x/media/" in shown, "image(v) is SHOWN to the user by the proxy as a picture link",
              shown[:200])
        rec = t1["x"]["codemode"]
        check(rec.get("runs") == 1 and rec.get("images") == 1 and rec.get("stored") == 1,
              "x_yamadori.codemode records the image and the store write", json.dumps(rec)[:300])
        c.tool_result("s1", out1)
        s2 = "return JSON.stringify(load('seen'))"
        t2 = c.turn([T.reply("", calls=[compose(s2, "s2")]), T.reply("Fine.")], user="What did you store?")
        out2 = hop(t2, 1, "s2")
        check('["koota","zustand"]' in out2, "load() in a later script of the same conversation reads it", out2[:200])
    finally:
        images.public_base = saved_base


def test_the_store_survives_a_compaction():
    """store() in a script, an in-place compaction, the client keeps only the summary: load() still reads the
    value (the store hangs off the conversation key, which the summary's session line carries)."""
    fresh()
    c = CM("cmp")
    t1 = c.turn([T.reply("", calls=[compose("store('plan', {step: 3}); return 'stored'", "p1")]),
                 T.reply("", reasoning="Write.", calls=[T.call("write_file", H.PKG, "w1")])],
                user="Remember the plan, then write the file.")
    c.tool_result("w1", "wrote package.json")
    ts = c.turn([T.reply("The user asked to remember a plan; step 3 was stored and package.json written.")],
                user="Your task is to create a detailed summary of the conversation so far.")
    summary = ts["m"].get("content") or ""
    check((ts["x"].get("utility_kind") == "compaction" or ts["x"].get("compaction")) and summary,
          "the summarise turn is served as a compaction", json.dumps(ts["x"].get("compaction"))[:200])
    # the client keeps the system message and the summary only
    c.msgs = [c.msgs[0], {"role": "user", "content": "[CONTEXT COMPACTION] Summary of the earlier conversation:\n"
                          + summary}]
    c.ids = {}
    t2 = c.turn([T.reply("", calls=[compose("return JSON.stringify(load('plan'))", "p2")]), T.reply("Step 3.")],
                user="What was the plan?")
    out = hop(t2, 1, "p2")
    ses = t2["x"].get("session") or {}
    check('{"step":3}' in out, "load() after the compaction reads what was stored before it "
          f"(session source: {ses.get('source')})", out[:200] + " | " + json.dumps(ses)[:200])


def test_an_empty_compaction_summary_is_asked_again():
    """(proxy, 2026-10-15) A compaction whose generation wrote no summary is asked again, once, on the same slot;
    the client gets the second one; a second empty answer is delivered as it is."""
    for label, script, want in (
            ("empty, then a summary", [T.reply("", reasoning="Hmm."), T.reply("The plan was step 3.")],
             "The plan was step 3."),
            ("empty twice", [T.reply(""), T.reply("")], "")):
        fresh()
        c = CM("cr" + label[:3], features={"codemode": False})
        c.turn([T.reply("Noted.")], user="Remember the plan: step 3.")
        t = c.turn(script, user="Your task is to create a detailed summary of the conversation so far.")
        content = (t["m"].get("content") or "")
        body = content.split("\n\n", 1)[1] if content.startswith("yamadori session") else content
        rec = t["x"].get("compaction_retry") or {}
        check(body.strip() == want and rec.get("hop") == 0 and len(t["gens"]) == 2,
              f"{label}: asked again once (2 generations); the client gets {want!r}",
              json.dumps([content[:120], rec, len(t["gens"])]))
    fresh()
    c = CM("crn", features={"codemode": False})
    t = c.turn([T.reply("")], user="Say nothing.")
    check(not t["x"].get("compaction_retry") and len(t["gens"]) == 1,
          "an empty answer to an ordinary turn is NOT retried (only a compaction is)")


def test_a_hidden_hop_keeps_its_argument_order():
    """A hop whose calls' arguments are NOT in sorted key order (the model wrote `query` before `ecosystem`),
    then a TEXT answer: the next request must extend the slot. It did not (2026-10-08, live: reused 5178 of
    15089): prepare sorted the replayed hop's arguments, the slot held the model's order."""
    for label, feats in (("toolbox on", {}), ("toolbox off", {"codemode": False})):
        fresh()
        c = CM("ord" + label[-2:], features=feats)
        t1 = c.turn([T.reply("", reasoning="Look it up.", calls=[
            T.call("yama_find_package", {"query": "pmndrs math", "ecosystem": "npm"}, "m1"),
            T.call("yama_list_package_versions", {"package": "math", "ecosystem": "npm"}, "m2")]),
            T.reply("The latest is 0.1.0.", reasoning="Answer.")], user="Which version of math is latest?")
        t2 = c.turn([T.reply("ok")], user="Thanks. Say ok.")
        check(T._extends(t1, t2, f"[hop then text, {label}]"),
              f"{label}: after two lookups in one hop and a text answer, the next request EXTENDS the slot "
              "(a hidden hop's arguments keep the model's order)")


# ================================================================ 6. the wires
def test_over_responses():
    class PR(RT.RClient):
        def chat_of(self, body):
            chat, ctx_ = RT.R.to_chat(body)
            chat.update(_account=self.account, _client_ip="127.0.0.1", _public_base=RT.BASE, _session_token="",
                        _features=json.dumps({"skills": False, "codemode": True, "codemode_mode": "on",
                                              "package_skills": True, "package_skills_mode": "inject"}))
            return chat, ctx_
    spec = json.loads(json.dumps(mcp_config.PACKAGELENS))
    spec.update(id="packagelens", runtime="local", command=[sys.executable, H.FAKE], call_timeout_s=10.0,
                call_timeout_why="test fixture: the fake answers at once")
    spec.pop("image", None)
    H.fresh_host(spec)
    outs = []
    for stream in (False, True):
        T.slots.reset(n=4)
        T.compaction.reset()
        c = PR(f"cm-{stream}", stream=stream, cache_key=f"cm-{stream}")
        t1 = c.turn([T.reply("", calls=[compose(SCRIPT)]), T.reply("", calls=[T.call("write_file", H.PKG, "r2")])],
                    user="Build a voxel scene with pmndrs math.")
        label = "streamed" if stream else "blocking"
        out = hop(t1, 1, "z1")
        outs.append(out.split(" in ", 1)[1].split("\n", 1)[1])
        tools_sent = names_of(t1["gens"][0])
        check(out.startswith("Script completed") and "true | true | true" in out and tools_sent[-1] == C.TOOL_NAME
              and PK.P.CRAFT_HEADER in out,
              f"responses {label}: the toolbox is offered, the script runs as a hidden hop, the section rides after "
              "the output", out[:160])
        cm = t1["x"].get("codemode") or {}
        check(cm.get("offered") and len(cm.get("calls") or []) == 3 and t1["x"]["tool_turns"]["turns"] == 1,
              f"responses {label}: x_yamadori.codemode and one tool turn", json.dumps(cm)[:160])
        c.tool_output("r2", "wrote package.json")
        t2 = c.turn([T.reply("Done.")])
        T._extends(t1, t2, f"responses {label}: after the toolbox hop")
        check(hop(t2, 0, "z1") == out, f"responses {label}: the hop replays byte for byte")
    check(outs[0] == outs[1], "responses: streamed == blocking")


def main() -> int:
    for fn in (test_the_offer, test_one_hidden_hop, test_only_mode_session, test_failures,
               test_the_caps_inside_a_script, test_decide_and_images,
               test_store_image_and_the_output_cap_in_a_turn, test_the_store_survives_a_compaction,
               test_an_empty_compaction_summary_is_asked_again, test_a_hidden_hop_keeps_its_argument_order,
               test_over_responses):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised", traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name + (f"   <- {detail}" if not ok and detail else ""))
    mcp_host.stop_all()
    T._srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
