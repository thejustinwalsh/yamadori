"""Per-answer mechanism record from x_yamadori, and per-arm mechanism health.

Operator's standing rule (2026-09-23): prove each answer ran at best effort
with every mechanism working, and report evidence only -- no helped/hurt
verdicts. For each mechanism the record says:

  allowed   the tier (or the header) permitted it
  decided   mcp/selection.py chose it for this request (where it decides)
  ran       it executed
  produced  it produced data (non-empty)
  error     it broke -- a tool error or a stack failure. An answer with any
            error is a STACK ERROR: it is not scored and is re-asked
            (drive.py). A mechanism that ran and legitimately found nothing
            is data, not an error.

Fields read, all from mcp/proxy.py:_x_yamadori (schema as of 2026-09-23):
tier, effort_sent, hops, hints[], suppressed_hints[], selection{hints,
investigate, fanout_n, signals{allowed, forced}}, investigate{ran, ok, hops,
injected, cited, unsupported, why}, fanout{n, asked, selection, winner,
winner_index, replaced, appended, candidates[{variant, parses}], error},
check_code{offered, calls, results[]}, repair{enabled, rounds, errors_before,
errors_after, stopped}.

NOT ON THE RESPONSE: per-call retrieval results. x_yamadori carries `hops`
(upstream generations in the tool loop; hops > 1 means tool calls happened)
but not which retrieval tool ran or whether it returned anything. So
retrieval is recorded as `tool_rounds = hops - 1` and its `produced` is
unknown (None), never guessed.
"""
from __future__ import annotations

from collections import defaultdict

MECHS = ("retrieval", "hints", "deep_thinking", "fanout", "check_code", "repair")


def record(xy: dict | None) -> dict:
    """The mechanism record of one answer. `stack_error` lists what broke."""
    if not xy:
        return {"present": False, "stack_error": ["no x_yamadori on the answer"]}
    sel = xy.get("selection") or {}
    sig = sel.get("signals") or {}
    allowed = sig.get("allowed") or {}
    forced = sig.get("forced") or []
    hops = xy.get("hops")
    inv = xy.get("investigate")
    fan = xy.get("fanout")
    cc = xy.get("check_code") or {}
    rep = xy.get("repair")
    hints = xy.get("hints") or []
    errors: list[str] = []

    r = {"present": True, "tier": xy.get("tier"), "effort_sent": xy.get("effort_sent"),
         "forced": sorted(forced), "hops": hops}

    # retrieval: offered by the tools gate; tool rounds from hops. Per-call
    # results come from x_yamadori.tools = [{name, empty, error, chars}] (the
    # deploy of 2026-09-23 morning); before it, `produced` stays None ("--").
    gate = xy.get("tools_gate") or {}
    tools = xy.get("tools")
    rt = {"allowed": gate.get("offer") if gate else False,
          "tool_rounds": (hops - 1) if isinstance(hops, int) else None,
          "calls": None, "nonempty": None, "produced": None, "by_name": None}
    if isinstance(tools, list):
        by_name: dict[str, int] = {}
        for t in tools:
            if not isinstance(t, dict):
                continue
            by_name[t.get("name") or "?"] = by_name.get(t.get("name") or "?", 0) + 1
            if t.get("error"):
                errors.append(f"tool {t.get('name')} error: {str(t.get('error'))[:80]}")
        ok = [t for t in tools if isinstance(t, dict) and not t.get("error")]
        rt.update(calls=len(tools), nonempty=sum(1 for t in ok if not t.get("empty")),
                  produced=any(not t.get("empty") for t in ok), by_name=by_name)
    r["retrieval"] = rt

    # hints: selection decides; injected ones are listed with ids
    hint_ids = [h.get("id") if isinstance(h, dict) else h for h in hints]
    r["hints"] = {"allowed": bool(allowed.get("hints")), "decided": bool(sel.get("hints")),
                  "injected": len(hints), "ids": hint_ids[:20],
                  "suppressed": len(xy.get("suppressed_hints") or [])}

    # deep thinking (shomen): ran, searches (hops), injected, cited
    d = {"allowed": bool(allowed.get("investigate")), "decided": bool(sel.get("investigate")),
         "ran": None, "searches": None, "injected": None, "cited": None}
    if isinstance(inv, dict):
        d.update(ran=bool(inv.get("ran")), searches=inv.get("hops"), injected=bool(inv.get("injected")),
                 cited=len(inv.get("cited") or []) if isinstance(inv.get("cited"), list) else inv.get("cited"),
                 why=inv.get("why"))
        if inv.get("ran") and inv.get("ok") is False:
            errors.append(f"deep thinking failed: {str(inv.get('why') or inv.get('finding') or '')[:120]}")
    elif d["decided"]:
        # selected but no record at all: it did not run and nothing says why
        errors.append("deep thinking selected but no investigate record")
    r["deep_thinking"] = d

    # fan-out: n, selection method, winner delivered, candidates parsing
    f = {"allowed_n": int(allowed.get("fanout") or 1), "decided_n": int(sel.get("fanout_n") or 1),
         "ran": False, "n": None, "method": None, "winner": None, "replaced": None,
         "candidates_parse": None}
    if isinstance(fan, dict):
        cands = fan.get("candidates") or []
        mode = fan.get("mode") or "parallel"
        f.update(ran=not fan.get("error"), n=fan.get("n"), asked=fan.get("asked"),
                 method=fan.get("selection"), winner=fan.get("winner"),
                 replaced=fan.get("replaced"), appended=fan.get("appended"),
                 candidates_parse=[c.get("parses") for c in cands] if cands else None,
                 mode=mode, steps=fan.get("steps"), stop_reason=fan.get("stop_reason"),
                 similarity_ab=fan.get("similarity_ab"),
                 candidate_roles=[c.get("role") or c.get("variant") for c in cands] if cands else None,
                 candidate_seeds=[c.get("seed") for c in cands] if cands else None)
        stop = str(fan.get("stop_reason") or "")
        if fan.get("error"):
            errors.append(f"fan-out error: {fan.get('error')}")
        elif mode == "sequential":
            # n counts candidates INCLUDING the original (A), so 1..3. Fan-out
            # that was decided but never produced a second candidate broke --
            # the helper lane stayed busy or B came back empty -- and is re-asked.
            # "n=1" (nothing to compare) and every graded outcome are data.
            if fan.get("skipped") or stop.startswith("skipped"):
                errors.append(f"fan-out skipped: {fan.get('skipped') or stop}")
                f["ran"] = False
            elif stop == "second candidate failed":
                errors.append("fan-out: the second candidate returned nothing")
            elif f["decided_n"] > 1 and (fan.get("steps") or 0) < 2 and stop != "n=1":
                errors.append(f"fan-out decided {f['decided_n']} but ran {fan.get('steps')} step(s): {stop}")
        elif f["decided_n"] > 1 and not fan.get("n"):
            errors.append(f"fan-out asked {fan.get('asked')} and got 0 candidates")
    r["fanout"] = f

    # check_code: offered, calls, per-call ok
    results = cc.get("results") or []
    r["check_code"] = {"offered": bool(cc.get("offered")), "calls": cc.get("calls") or 0,
                       "ok_calls": sum(1 for x in results if isinstance(x, dict) and x.get("ok"))}
    for x in results:
        if isinstance(x, dict) and x.get("error"):
            errors.append(f"check_code tool error: {str(x.get('error'))[:80]}")

    # repair: rounds, errors before/after, stop reason
    r["repair"] = ({"enabled": True, "rounds": rep.get("rounds"), "errors_before": rep.get("errors_before"),
                    "errors_after": rep.get("errors_after"), "stopped": rep.get("stopped")}
                   if isinstance(rep, dict) else {"enabled": False})

    # tool-turn cap (x_yamadori.tool_turns = {limit, turns, hit}, deployed
    # 2026-09-23 ~11:15). Hitting the cap is DATA -- the loop landed as
    # designed -- never a stack error. None before the deploy.
    tt = xy.get("tool_turns")
    r["tool_turns"] = ({"limit": tt.get("limit"), "turns": tt.get("turns"), "hit": bool(tt.get("hit"))}
                       if isinstance(tt, dict) else None)

    r["stack_error"] = errors
    return r


def health(recs: list[dict]) -> dict:
    """Per-mechanism counts over one arm's answers: allowed / decided / ran / produced data."""
    n = len(recs)
    h: dict = {"answers": n, "stack_errors": sum(1 for r in recs if r.get("stack_error"))}
    c = defaultdict(lambda: defaultdict(int))
    for r in recs:
        if not r.get("present"):
            continue
        rt = r["retrieval"]
        c["retrieval"]["allowed"] += bool(rt["allowed"])
        if rt.get("calls") is None:
            # answer predates x_yamadori.tools: only hops is known
            c["retrieval"]["ran"] += bool(rt["tool_rounds"])
            c["retrieval"]["answers_without_tool_record"] += 1
        else:
            c["retrieval"]["ran"] += rt["calls"] > 0
            c["retrieval"]["produced"] = c["retrieval"].get("produced", 0) + bool(rt["produced"])
            c["retrieval"]["calls_total"] += rt["calls"]
            c["retrieval"]["nonempty_total"] += rt["nonempty"]
        hi = r["hints"]
        c["hints"]["allowed"] += hi["allowed"]; c["hints"]["decided"] += hi["decided"]
        c["hints"]["ran"] += hi["decided"]; c["hints"]["produced"] += hi["injected"] > 0
        c["hints"]["injected_total"] += hi["injected"]
        dt = r["deep_thinking"]
        c["deep_thinking"]["allowed"] += dt["allowed"]; c["deep_thinking"]["decided"] += dt["decided"]
        c["deep_thinking"]["ran"] += bool(dt["ran"])
        c["deep_thinking"]["produced"] += bool(dt["injected"]) or bool(dt["cited"])
        fo = r["fanout"]
        c["fanout"]["allowed"] += fo["allowed_n"] > 1; c["fanout"]["decided"] += fo["decided_n"] > 1
        c["fanout"]["ran"] += bool(fo["ran"]); c["fanout"]["produced"] += bool(fo["n"])
        c["fanout"]["winner_replaced"] += bool(fo.get("replaced"))
        if fo.get("method"):
            c["fanout"][f"method:{fo['method']}"] += 1
        if fo.get("mode"):
            c["fanout"][f"mode:{fo['mode']}"] += 1
        if fo.get("steps"):
            c["fanout"][f"steps:{fo['steps']}"] += 1
        cc = r["check_code"]
        c["check_code"]["allowed"] += cc["offered"]; c["check_code"]["ran"] += cc["calls"] > 0
        c["check_code"]["produced"] += cc["ok_calls"] > 0; c["check_code"]["calls_total"] += cc["calls"]
        rp = r["repair"]
        c["repair"]["allowed"] += rp["enabled"]; c["repair"]["ran"] += bool(rp.get("rounds"))
        c["repair"]["produced"] += bool(rp.get("rounds"))
        if rp.get("stopped"):
            c["repair"][f"stopped:{rp['stopped']}"] += 1
    tts = [r["tool_turns"] for r in recs if r.get("present") and r.get("tool_turns")]
    h["tool_turn_cap"] = ({"answers_with_record": len(tts), "hit": sum(1 for t in tts if t["hit"]),
                           "limits": sorted({t["limit"] for t in tts if t["limit"] is not None}),
                           "max_turns": max((t["turns"] or 0) for t in tts)} if tts else None)
    out = {m: dict(c[m]) for m in MECHS if m in c}
    if "retrieval" in out and "produced" not in out["retrieval"]:
        out["retrieval"]["produced"] = None  # no answer carried x_yamadori.tools
    h["mechanisms"] = out
    return h
