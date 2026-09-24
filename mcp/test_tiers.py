#!/usr/bin/env python
"""`reasoning_effort` as the product dial, asserted. No model, no network.

WHAT THIS IS GATING

`tiers.py` maps whatever a client sends in `reasoning_effort` onto a tier and
then onto a request the chat template will accept. Its promises:

  1. Any spelling resolves -- known aliases to their tier, anything else to
     the default -- and a deployment ceiling caps it.
  2. What reaches the template is always in ACCEPTED_EFFORTS. `max` sent
     straight to the template is a 500; through here it is `xhigh`.
  3. The caller's max_tokens is an ANSWER allowance with a floor (300 tokens
     returns zero characters from a thinking model); thinking is derived from
     the request's share of the pool (role, share_n) minus its prompt and
     that answer, and a per-request cap overrides it.
  4. Overrides break the bundle apart for experiments, and say they did.
  5. The feature header is client-controlled, so malformed input is ignored,
     never raised. (Until 2026-09-22 `[1]` or `{"floor": "big"}` raised.)
  6. `python mcp/tiers.py` runs. Its self-test crashed on a renamed key for
     long enough that PROTOCOL rule 14 was written about it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Read at import. A deployment's own ceiling must not change what is asserted.
os.environ.pop("YAMADORI_TIER_CEILING", None)
os.environ.pop("YAMADORI_TIER_DEFAULT", None)

import budget  # noqa: E402
import tiers  # noqa: E402

# budget.pool_size() would ask the live server. Pinned to the shipped `-c`,
# so every derived number below is the shipped split and nothing leaves
# the box.
POOL, MAIN, HELPER = 147456, 92160, 55296   # 5/8 + 1 x 3/8
budget._POOL = POOL

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ---------------------------------------------------------------------------


def test_the_fixture_uses_the_shipped_settings():
    check(tiers.CEILING == "max" and tiers.DEFAULT == "medium",
          "the ceiling and default are the shipped ones, not the environment's",
          f"{tiers.CEILING}/{tiers.DEFAULT}")


def test_every_spelling_resolves():
    cases = {"none": "minimal", "off": "minimal", "minimal": "minimal",
             "low": "low", "LO": "low", " medium ": "medium", "": "medium",
             "default": "medium", "High": "high", "xhigh": "xhigh",
             "x-high": "xhigh", "ultra": "max", "max": "max",
             "banana": "medium", None: "medium", 7: "medium"}
    for sent, want in cases.items():
        got = tiers.resolve({"reasoning_effort": sent})["name"]
        check(got == want, f"{sent!r} resolves to {want}", got)
    check(tiers.resolve({})["name"] == "medium", "an absent field is the default")


def test_every_alias_lands_on_a_real_tier():
    bad = {k: v for k, v in tiers.ALIASES.items() if v not in tiers.TIERS}
    check(not bad, "every alias target is a tier", str(bad))
    check(list(tiers.TIERS) == tiers.ORDER, "ORDER lists every tier, in order")


def test_the_ceiling_caps_the_request():
    old = tiers.CEILING
    try:
        tiers.CEILING = "high"
        check(tiers.resolve({"reasoning_effort": "max"})["name"] == "high",
              "max is capped to a `high` ceiling")
        check(tiers.resolve({"reasoning_effort": "low"})["name"] == "low",
              "a request under the ceiling is untouched")
        tiers.CEILING = "minimal"
        check(tiers.resolve({"reasoning_effort": "high"})["name"] == "minimal",
              "a `minimal` ceiling caps everything")
    finally:
        tiers.CEILING = old


def _with_template(tpl: str | None):
    """Point tiers.accepted_efforts() at a fake /props answer (None = down)."""
    import io
    import urllib.request

    def fake(url, timeout=None):
        if tpl is None:
            raise OSError("server down")
        return io.BytesIO(json.dumps({"chat_template": tpl}).encode())
    real = urllib.request.urlopen
    urllib.request.urlopen = fake
    tiers._accepted = None
    try:
        return tiers.accepted_efforts()
    finally:
        urllib.request.urlopen = real
        tiers._accepted = None


def test_the_accepted_set_is_read_from_the_served_template():
    guard = ("{%- if resolved_reasoning_effort not in ('xhigh', 'medium', "
             "'low') %}")
    check(_with_template(guard.replace("resolved_reasoning_effort",
                                       "reasoning_effort")
                         ) == ("xhigh", "medium", "low"),
          "the guard clause's list is parsed out of the template")
    check(_with_template("{%- if reasoning_effort not in ('high', 'low') %}")
          == ("high", "low"),
          "a different template's list is followed, not a recalled one")
    check(_with_template(None) == tiers.FALLBACK_EFFORTS,
          "with the server unreachable, the fallback is today's template")
    check(_with_template("no guard here") == tiers.FALLBACK_EFFORTS,
          "and a template with no guard also falls back")


def test_the_template_never_sees_an_effort_it_rejects():
    tiers._accepted = tiers.FALLBACK_EFFORTS          # deterministic: no network
    ok = set(tiers.FALLBACK_EFFORTS)
    for n in tiers.ORDER:
        body = tiers.apply({}, tiers.resolve({"reasoning_effort": n}))
        if not tiers.TIERS[n]["thinks"]:
            check("reasoning_effort" not in body,
                  f"tier {n} (thinking off) sends no effort at all",
                  str(body.get("reasoning_effort")))
            continue
        check(body.get("reasoning_effort") in ok,
              f"tier {n} sends an accepted effort", str(body.get("reasoning_effort")))
    # `high` was an instant 500 on the served template; it rounds UP.
    for sent, want in (("max", "xhigh"), ("MAXIMUM", "xhigh"), ("ultra", "xhigh"),
                       ("high", "xhigh"), ("minimal", "low"),
                       ("none", "low"), ("off", "low"), ("banana", "medium"),
                       ("enhanced", "medium"), (None, "xhigh"), ("", "xhigh"),
                       ("xhigh", "xhigh"), ("low", "low"), ("medium", "medium")):
        got = tiers.safe_effort(sent)
        check(got == want and got in ok,
              f"safe_effort({sent!r}) is {want!r}", repr(got))
    t = tiers.resolve({}, overrides={"effort": "max"})
    check(tiers.apply({}, t)["reasoning_effort"] == "xhigh",
          "an overridden effort still goes through safe_effort")


def test_the_answer_allowance_is_added_to_the_thinking_breaker():
    """docs/CONSTRAINTS.md section 3, mcp/budget.py. The caller's max_tokens is
    the ANSWER; thinking is what the request's share of the pool leaves after
    its prompt and that answer -- never a free-floating number, and never
    taken out of the answer.

        window   = budgets()[role] // share_n
        answer   = max(client max_tokens, A_MIN)
        thinking = max(window - prompt - answer, MIN_THINKING)

    The pool is pinned at 147,456 above, so main is 92,160 and the one helper
    55,296 (5/8 + 3/8, the operator's split of 2026-09-22)."""
    A, M, MSG = tiers.A_MIN, tiers.MIN_THINKING, tiers.BUDGET_MESSAGE
    check(A == 2048 and M == 1024, "the answer minimum is 2048 and the thinking "
          "floor 1024", f"{A}/{M}")
    check(not hasattr(tiers, "R_CAP"),
          "there is no fixed thinking breaker any more (R_CAP is gone)")
    b = budget.budgets()
    check((b["pool"], b["main"], b["helper"]) == (POOL, MAIN, HELPER),
          "the fixture's split is the shipped one", json.dumps(b))

    # --- the derived rule, by role, share and prompt ---------------------
    got = tiers.budget(None)
    check(got == {"max_tokens": MAIN, "reasoning_budget_tokens": MAIN - A,
                  "reasoning_budget_message": MSG},
          "main, no prompt: thinking is main's whole 5/8 minus the answer, "
          "and the total is exactly that share", json.dumps(got))
    got = tiers.budget(None, role="helper")
    check(got["max_tokens"] == HELPER
          and got["reasoning_budget_tokens"] == HELPER - A,
          "a helper's thinking comes out of ITS 3/8, not main's 5/8",
          json.dumps(got))
    got = tiers.budget(None, role="banana")
    check(got["max_tokens"] == MAIN, "an unknown role is a main request",
          json.dumps(got))
    for n in (2, 3, 4):
        got = tiers.budget(None, share_n=n)
        check(got["reasoning_budget_tokens"] == MAIN // n - A
              and got["max_tokens"] == MAIN // n,
              f"share_n={n}: each sample gets 1/{n} of main's share",
              json.dumps(got))
    got = tiers.budget(None, role="helper", share_n=2)
    check(got["max_tokens"] == HELPER // 2, "role and share_n compose",
          json.dumps(got))
    for bad in (0, -3):
        check(tiers.budget(None, share_n=bad)["max_tokens"] == MAIN,
              f"share_n={bad} is treated as 1, never a division error")
    got = tiers.budget(None, prompt_tokens=10000)
    check(got["reasoning_budget_tokens"] == MAIN - 10000 - A
          and got["max_tokens"] + 10000 == MAIN,
          "the prompt comes out of the thinking room: prompt + max_tokens "
          "fills the window exactly", json.dumps(got))
    got = tiers.budget(5000, role="helper", share_n=2, prompt_tokens=3000)
    check(got["reasoning_budget_tokens"] == HELPER // 2 - 3000 - 5000
          and got["max_tokens"] == HELPER // 2 - 3000,
          "role, share_n, prompt and answer all at once", json.dumps(got))
    got = tiers.budget(None, prompt_tokens=MAIN)
    check(got["reasoning_budget_tokens"] == M and got["max_tokens"] == M + A,
          "a prompt that fills the window still leaves the thinking floor and "
          "a full answer", json.dumps(got))

    # --- the answer allowance -------------------------------------------
    # The measured failures: 900 (summarize_text) and 300 returned zero
    # characters, because thinking consumed the whole number.
    for small in (24, 300, 900):
        got = tiers.budget(small)
        check(got["max_tokens"] - got["reasoning_budget_tokens"] == A,
              f"a {small}-token request still leaves a full answer allowance",
              json.dumps(got))
    got = tiers.budget(16000)
    check(got["max_tokens"] - got["reasoning_budget_tokens"] == 16000
          and got["reasoning_budget_tokens"] == MAIN - 16000,
          "a large answer request keeps all of it; thinking gets what is left",
          json.dumps(got))
    for junk in (None, "big", [], 0, -5):
        got = tiers.budget(junk)
        check(got["max_tokens"] == MAIN
              and got["reasoning_budget_tokens"] == MAIN - A,
              f"a max_tokens of {junk!r} is treated as absent, never raised",
              json.dumps(got))

    # --- thinking off ---------------------------------------------------
    check(tiers.budget(100, thinks=False) == {"max_tokens": A},
          "with thinking off, only the answer allowance is sent")
    check(tiers.budget(16000, thinks=False, cap=8192, role="helper",
                       share_n=3, prompt_tokens=50000) == {"max_tokens": 16000},
          "and nothing else -- no cap, share or prompt adds thinking")
    off = tiers.resolve({"reasoning_effort": "low"}, overrides={"thinks": False})
    body = tiers.apply({"max_tokens": 100}, off)
    check(body["max_tokens"] == A and "reasoning_budget_tokens" not in body
          and "reasoning_budget_message" not in body
          and body["enable_thinking"] is False,
          "apply() with thinking off adds no thinking allowance", json.dumps(body))

    # --- the per-request cap overrides the derived room ------------------
    for cap, want_t in ((8192, 8192), (65536, 65536), (10, M),
                        (10**9, POOL), ("x", MAIN - A)):
        got = tiers.budget(None, cap=cap)
        check(got["reasoning_budget_tokens"] == want_t
              and got["max_tokens"] == want_t + A,
              f"cap={cap!r} sends {want_t} (clamped to [{M}, pool])",
              json.dumps(got))
    got = tiers.budget(3000, cap=8192, role="helper", share_n=3,
                       prompt_tokens=50000)
    check(got["reasoning_budget_tokens"] == 8192 and got["max_tokens"] == 11192,
          "a cap overrides the derived room whatever the role, share or "
          "prompt", json.dumps(got))
    t = tiers.resolve({"reasoning_effort": "medium"},
                      overrides={"reasoning_cap": 8192})
    check(tiers.apply({}, t)["reasoning_budget_tokens"] == 8192,
          "apply() passes the tier's reasoning_cap through")
    t = tiers.resolve({}, tiers.from_header('{"reasoning_cap": 8192}'))
    check(tiers.apply({}, t)["reasoning_budget_tokens"] == 8192,
          "and it can be set through X-Yamadori-Features for a benchmark arm")

    # --- apply(): every tier, its own prompt, role and share -------------
    for name in tiers.ORDER:
        if not tiers.TIERS[name]["thinks"]:
            continue    # thinking off: the answer allowance only, tested below
        out = tiers.apply({}, tiers.resolve({"reasoning_effort": name}))
        check(out["max_tokens"] == MAIN
              and out.get("reasoning_budget_tokens") == MAIN - A
              and out.get("reasoning_budget_message") == MSG,
              f"{name}: an empty body gets the main window, with the message",
              json.dumps({k: out.get(k) for k in
                          ("max_tokens", "reasoning_budget_tokens")}))
    msgs = [{"role": "user", "content": "x" * 30000}]
    tools = [{"type": "function", "function": {"name": "f",
                                               "description": "y" * 9000}}]
    est_m = tiers.estimate_prompt_tokens({"messages": msgs})
    est_mt = tiers.estimate_prompt_tokens({"messages": msgs, "tools": tools})
    check(est_m == len(json.dumps(msgs, ensure_ascii=False)) // 3,
          "the prompt estimate is ~chars/3 over the messages", str(est_m))
    check(est_mt == (len(json.dumps(msgs, ensure_ascii=False))
                     + len(json.dumps(tools, ensure_ascii=False))) // 3,
          "and counts the tools too", str(est_mt))
    check(tiers.estimate_prompt_tokens({"model": "x" * 9000,
                                        "max_tokens": 5}) == 0,
          "and nothing outside messages and tools")
    t = tiers.resolve({"reasoning_effort": "medium"})
    out = tiers.apply({"messages": msgs, "tools": tools, "max_tokens": 4000}, t)
    check(out["reasoning_budget_tokens"] == MAIN - est_mt - 4000
          and out["max_tokens"] == MAIN - est_mt,
          "apply() takes its own prompt estimate out of the thinking room",
          json.dumps({k: out.get(k) for k in
                      ("max_tokens", "reasoning_budget_tokens")}))
    out = tiers.apply({"messages": msgs}, t, role="helper", share_n=2)
    check(out["reasoning_budget_tokens"] == HELPER // 2 - est_m - A,
          "apply() honours role and share_n",
          str(out.get("reasoning_budget_tokens")))

    # --- rebudget(): re-derive an already-shaped payload -----------------
    shaped = tiers.apply({"messages": msgs, "max_tokens": 5000}, t)
    before = json.dumps(shaped, sort_keys=True)
    rb = tiers.rebudget(shaped, share_n=3)
    check(json.dumps(shaped, sort_keys=True) == before,
          "rebudget() does not mutate its input")
    check(rb["max_tokens"] - rb["reasoning_budget_tokens"] == 5000,
          "rebudget() keeps the answer allowance it was shaped with",
          json.dumps({k: rb.get(k) for k in
                      ("max_tokens", "reasoning_budget_tokens")}))
    check(rb["reasoning_budget_tokens"] == MAIN // 3 - est_m - 5000,
          "and re-derives thinking for 1/3 of main's share, minus the prompt",
          str(rb.get("reasoning_budget_tokens")))
    check(rb == tiers.apply({"messages": msgs, "max_tokens": 5000}, t,
                            share_n=3),
          "which is exactly what apply() would have given at share_n=3")
    rb = tiers.rebudget(shaped, role="helper")
    check(rb["reasoning_budget_tokens"] == HELPER - est_m - 5000,
          "rebudget() honours role", str(rb.get("reasoning_budget_tokens")))
    small = tiers.rebudget(tiers.apply({"max_tokens": 100}, t), share_n=2)
    check(small["max_tokens"] - small["reasoning_budget_tokens"] == A,
          "a shaped minimum answer stays the minimum", json.dumps(small))
    for plain in (body, {"messages": msgs, "max_tokens": 700}):
        rb = tiers.rebudget(plain, share_n=3)
        check(rb == plain and rb is not plain,
              "a payload with thinking off, or never shaped, comes back as an "
              "unchanged copy", json.dumps(rb)[:120])

    # --- the split is read from budget.py, with a stated fallback --------
    saved = budget.budgets
    try:
        budget.budgets = lambda pool=None: saved(200000)
        got = tiers.budget(None)
        check(got["max_tokens"] == 125000,
              "a different pool moves the window with it (read, not recalled)",
              json.dumps(got))

        def broken(*a, **k):
            raise RuntimeError("budget unavailable")
        budget.budgets = broken
        got = tiers.budget(None, role="helper")
        # The fallback is the split at the live `-c 163840`, not the pinned
        # 147,456 fixture: 5/8 = 102,400 and 3/8 = 61,440.
        check(got["max_tokens"] == 61440 and tiers.budget(None)["max_tokens"]
              == 102400, "an unavailable budget falls back to the shipped "
              "split at the live pool (102,400 / 61,440)", json.dumps(got))
    finally:
        budget.budgets = saved


def test_apply_does_not_mutate_its_input():
    body = {"max_tokens": 10, "messages": []}
    tiers.apply(body, tiers.resolve({}))
    check(body == {"max_tokens": 10, "messages": []}, "the caller's body is untouched",
          json.dumps(body))


def test_thinking_is_off_only_at_minimal():
    # Operator, 2026-09-23: `minimal` is thinking off with the vendor's
    # instruct sampling; every other tier thinks. `low` is the baseline.
    check([n for n in tiers.ORDER if not tiers.TIERS[n]["thinks"]] == ["minimal"],
          "only `minimal` switches thinking off",
          str([n for n in tiers.ORDER if not tiers.TIERS[n]["thinks"]]))
    m = tiers.apply({"max_tokens": 500, "temperature": 1.0},
                    tiers.resolve({"reasoning_effort": "minimal"}))
    check(m.get("chat_template_kwargs", {}).get("enable_thinking") is False
          and m["enable_thinking"] is False and "reasoning_effort" not in m
          and "reasoning_budget_tokens" not in m and m["max_tokens"] == tiers.A_MIN,
          "minimal: the TEMPLATE is told thinking is off (chat_template_kwargs), "
          "no effort, no thinking budget, the answer allowance only",
          json.dumps({k: m.get(k) for k in ("chat_template_kwargs",
                      "enable_thinking", "max_tokens")}))
    check(all(m.get(k) == v for k, v in tiers.VENDOR_SAMPLING_INSTRUCT.items()),
          "minimal: the vendor's instruct sampling, overriding the client's",
          json.dumps({k: m.get(k) for k in tiers.VENDOR_SAMPLING_INSTRUCT}))
    lo = tiers.apply({}, tiers.resolve({"reasoning_effort": "low"}))
    check(lo.get("chat_template_kwargs", {}).get("enable_thinking") is True
          and lo.get("reasoning_effort") == "medium",
          "low: thinking on at medium, told to the template",
          json.dumps({k: lo.get(k) for k in ("chat_template_kwargs",
                      "reasoning_effort")}))
    t = tiers.TIERS["low"]
    check(not any(t[k] for k in ("retrieval", "hints", "investigate",
                                 "check_code", "repair")) and t["fanout"] == 1
          and tiers.TIERS["minimal"]["fanout"] == 1,
          "low: none of our augmentation; neither low nor minimal fans out")
    t = tiers.resolve({"reasoning_effort": "low"}, overrides={"thinks": False})
    body = tiers.apply({"reasoning_effort": "low"}, t)
    check(body["enable_thinking"] is False and "reasoning_effort" not in body,
          "an explicit thinks=False override removes the effort field",
          json.dumps(body))


def test_resolve_returns_a_copy():
    t = tiers.resolve({"reasoning_effort": "high"})
    t["fanout"] = 99
    check(tiers.TIERS["high"]["fanout"] == 3, "mutating a resolved tier leaves "
          "the preset alone", str(tiers.TIERS["high"]["fanout"]))


def test_overrides_break_the_bundle_and_say_so():
    t = tiers.resolve({"reasoning_effort": "high"}, overrides={"fanout": 1})
    check(t["fanout"] == 1 and t["hints"] is True and t["effort"] == "medium",
          "only the overridden field changes", json.dumps(t)[:160])
    check(t["overridden"] == ["fanout"], "and the tier records what was overridden")
    check("overridden" not in tiers.resolve({"reasoning_effort": "high"}),
          "a plain request records no override")
    off, on = tiers.ab("medium", "aug_off"), tiers.ab("medium", "aug_on")
    same = {k for k in ("effort", "thinks", "fanout") if off[k] == on[k]}
    check(same == {"effort", "thinks", "fanout"},
          "the A/B arms differ only in the augmentations", str(same))
    check(off["retrieval"] is False and on["retrieval"] is True,
          "and do differ in them")


def test_the_feature_header_is_parsed_defensively():
    check(tiers.from_header(None) is None and tiers.from_header("") is None,
          "no header, no overrides")
    check(tiers.from_header('{"fanout": 1, "hints": false}')
          == {"fanout": 1, "hints": False}, "a valid header is used")
    check(tiers.from_header('{"thinks": false, "admin": true}') is None,
          "keys outside the allowed set are dropped (thinks is not settable "
          "from a header)")
    for bad in ("not json", "[1, 2]", '"a string"', "42", "null", "true"):
        try:
            out = tiers.from_header(bad)
            check(out is None, f"header {bad!r} is ignored", repr(out))
        except Exception as e:                                   # noqa: BLE001
            check(False, f"header {bad!r} does not raise", f"{type(e).__name__}: {e}")
    out = tiers.from_header('{"floor": "big", "fanout": "3", "hints": 1, '
                            '"retrieval": true, "investigate": "yes", '
                            '"effort": 5}')
    check(out == {"retrieval": True},
          "values of the wrong type are dropped, the right ones kept", repr(out))
    check(tiers.from_header('{"fanout": true}') is None,
          "a boolean is not accepted as a count")
    try:
        t = tiers.resolve({}, tiers.from_header('{"floor": "big"}'))
        tiers.apply({}, t)
        check(True, "a malformed floor cannot reach apply()")
    except Exception as e:                                       # noqa: BLE001
        check(False, "a malformed floor cannot reach apply()",
              f"{type(e).__name__}: {e}")


def test_describe_lists_every_tier():
    d = tiers.describe()
    check(all(n in d for n in tiers.ORDER), "describe() names every tier")
    check("fan-out x3" in d and "deep thinking" in d, "and what each adds")


def test_the_self_test_runs():
    """PROTOCOL rule 14: a self-test that crashes looks like coverage."""
    r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "tiers.py")],
                       capture_output=True, text=True, timeout=60)
    check(r.returncode == 0, "`python mcp/tiers.py` exits 0",
          ((r.stderr or "").strip().splitlines() or [""])[-1])
    check("factorial arm" in r.stdout, "and prints its last line")


def test_vendor_sampling_is_enforced():
    """2026-09-23: the vendor's sampling is written by apply() for every
    request, over whatever the client sent, and recorded."""
    t = tiers.resolve({"reasoning_effort": "medium"})
    for client_temp in (0.0, 0.3):
        out = tiers.apply({"messages": [], "temperature": client_temp,
                           "top_p": 0.95}, t)
        check(out["temperature"] == 1.0 and out["top_p"] == 0.95
              and out["top_k"] == 20 and out["min_p"] == 0.0
              and out["presence_penalty"] == 0.0
              and out["repeat_penalty"] == 1.0,
              f"client temperature {client_temp} -> the vendor's 1.0 and "
              f"the rest of the thinking-mode card",
              json.dumps({k: out.get(k) for k in tiers.VENDOR_SAMPLING}))
        rec = out.get("_sampling") or {}
        check(rec.get("client_overridden") == {"temperature": client_temp},
              "the client's value is reported in client_overridden, and a "
              "value that already matched is not", json.dumps(rec))
        check(rec.get("enforced") == tiers.VENDOR_SAMPLING,
              "and what was enforced is recorded", json.dumps(rec))
    out = tiers.apply({"messages": []}, t)
    check(out["_sampling"]["client_overridden"] == {},
          "a client that sent nothing has nothing overridden")
    off = dict(t, thinks=False)
    out = tiers.apply({"messages": [], "temperature": 1.0}, off)
    check(out["temperature"] == 0.7 and out["top_p"] == 0.80
          and out["presence_penalty"] == 1.5,
          "a non-thinking request gets the card's instruct values",
          json.dumps({k: out.get(k) for k in tiers.VENDOR_SAMPLING}))
    import model
    shaped = model.shape({"messages": [], "temperature": 0.2}, "low")
    check(all(shaped[k] == v for k, v in tiers.VENDOR_SAMPLING.items())
          and shaped["_sampling"]["client_overridden"] == {"temperature": 0.2},
          "internal model.ask/chat calls get the same values (one door)",
          json.dumps({k: shaped.get(k) for k in tiers.VENDOR_SAMPLING}))
    body = {"messages": [], "temperature": 0.0}
    tiers.apply(body, t)
    check(body == {"messages": [], "temperature": 0.0},
          "the caller's body is not mutated")


def test_generic_client_spellings():
    # OpenRouter's object form picks the tier like reasoning_effort.
    check(tiers.resolve({"reasoning": {"effort": "high"}})["name"] == "high",
          "reasoning.effort picks the tier")
    check(tiers.resolve({"reasoning_effort": "low",
                         "reasoning": {"effort": "max"}})["name"] == "low",
          "an explicit reasoning_effort wins over the object form")
    # max_completion_tokens is read as the answer allowance and never passed on.
    t = tiers.resolve({"reasoning_effort": "low"})
    a = tiers.apply({"max_completion_tokens": 5000}, t)
    b = tiers.apply({"max_tokens": 5000}, t)
    check("max_completion_tokens" not in a and a["max_tokens"] == b["max_tokens"],
          "max_completion_tokens is the answer allowance and is not sent upstream",
          json.dumps({k: a.get(k) for k in ("max_tokens", "max_completion_tokens")}))

def main() -> int:
    for fn in (test_generic_client_spellings, test_vendor_sampling_is_enforced,
               test_the_fixture_uses_the_shipped_settings,
               test_every_spelling_resolves,
               test_every_alias_lands_on_a_real_tier,
               test_the_ceiling_caps_the_request,
               test_the_accepted_set_is_read_from_the_served_template,
               test_the_template_never_sees_an_effort_it_rejects,
               test_the_answer_allowance_is_added_to_the_thinking_breaker,
               test_apply_does_not_mutate_its_input,
               test_thinking_is_off_only_at_minimal,
               test_resolve_returns_a_copy,
               test_overrides_break_the_bundle_and_say_so,
               test_the_feature_header_is_parsed_defensively,
               test_describe_lists_every_tier,
               test_the_self_test_runs):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
