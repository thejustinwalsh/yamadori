#!/usr/bin/env python
"""The public model list, asserted. No network.

WHAT THIS IS GATING

`catalog.py` decides what `/v1/models` advertises and what a requested model
name resolves to. Its promises:

  1. One product name is advertised. Internal names (`bonsai`, `embeddings`,
     `reranker`) are never listed -- listing them advertises a bypass around
     retrieval, tiering, secret filtering and the audit log.
  2. The reserved name `metsumi` is neither advertised nor resolvable as ours.
  3. Anything a client sends resolves to a working model: unknown names fall
     back to the product rather than 404, and say they were unknown.
  4. Replies carry the product name back, with the proxy's `_`-prefixed
     bookkeeping stripped.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Read at import. A benchmark shell with it set must not change the assertions.
os.environ.pop("YAMADORI_EXPOSE_INTERNAL", None)
os.environ.pop("YAMADORI_OWNER", None)

import catalog  # noqa: E402
import budget  # noqa: E402

# budget.pool_size() would ask the live server for n_ctx. Pinned to the pool
# config.yaml launches with (-c 163840), so the window is 5/8 of it.
budget._POOL = 163840

INTERNAL_NAMES = {"bonsai", "bonsai-agent", "bonsai-vision", "embeddings",
                  "reranker", "critic-disabled"}

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ---------------------------------------------------------------------------


def test_the_fixture_is_the_shipped_setting():
    check(catalog.EXPOSE_INTERNAL is False, "internal names are not exposed")


def test_only_the_product_is_advertised():
    ids = [m["id"] for m in catalog.public_list()["data"]]
    check(ids == ["yamadori"], "the model list is exactly ['yamadori']", str(ids))
    check(not (set(ids) & INTERNAL_NAMES), "no internal name is listed")
    check("metsumi" not in ids, "the reserved name is not listed")
    payload = catalog.public_list()
    check(payload["object"] == "list"
          and all(m["object"] == "model" and m["owned_by"] == "yamadori"
                  and isinstance(m["created"], int) for m in payload["data"]),
          "the payload is the OpenAI /v1/models shape", json.dumps(payload)[:160])


def test_exposing_internal_names_is_opt_in():
    old = catalog.EXPOSE_INTERNAL
    try:
        catalog.EXPOSE_INTERNAL = True
        ids = [m["id"] for m in catalog.public_list()["data"]]
        check(set(ids) == set(catalog.CATALOG),
              "with the flag, every product variant is listed", str(ids))
        check(not (set(ids) & INTERNAL_NAMES),
              "but still no raw internal model name", str(ids))
        check("metsumi" not in ids, "and still not the reserved name")
    finally:
        catalog.EXPOSE_INTERNAL = old


def test_names_resolve_to_a_working_model():
    cases = {"yamadori": ("bonsai", None, True),
             "yamadori-fast": ("bonsai", "low", True),
             "yamadori-max": ("bonsai", "max", True),
             "yamadori-vision": ("bonsai-vision", None, True),
             " yamadori-max ": ("bonsai", "max", True),
             "bonsai": ("bonsai", None, True),
             "bonsai-agent": ("bonsai", None, True),
             "gpt-4": ("bonsai", None, False),
             "": ("bonsai", None, False),
             None: ("bonsai", None, False),
             "metsumi": ("bonsai", None, False)}
    for name, want in cases.items():
        got = catalog.resolve(name)
        check(got == want, f"{name!r} resolves to {want}", str(got))


def test_bonsai_and_bonsai_agent_are_the_same_process():
    """AGENTS.md: the alias exists only so older clients keep working."""
    check(catalog.resolve("bonsai")[:2] == catalog.resolve("bonsai-agent")[:2]
          == catalog.resolve("yamadori")[:2],
          "bonsai, bonsai-agent and yamadori resolve identically")


def test_replies_carry_the_product_name_and_no_bookkeeping():
    p = {"id": "x", "model": "bonsai", "choices": [], "_transport": {"ms": 3},
         "_debug": 1, "usage": {"_keep": "nested keys are not ours to strip"}}
    out = catalog.rewrite_response(p, "yamadori")
    check(out["model"] == "yamadori", "the model field is the product name")
    check("_transport" not in out and "_debug" not in out,
          "top-level underscored fields are stripped", str(sorted(out)))
    check(out["usage"] == {"_keep": "nested keys are not ours to strip"},
          "nested fields are left alone")
    check(catalog.rewrite_response({"choices": []}, "yamadori") == {"choices": []},
          "a reply with no model field gets none added")
    for odd in (None, [], "text", 3):
        try:
            check(catalog.rewrite_response(odd, "yamadori") == odd,
                  f"a {type(odd).__name__} payload passes through untouched")
        except Exception as e:                                   # noqa: BLE001
            check(False, f"a {type(odd).__name__} payload does not raise",
                  f"{type(e).__name__}: {e}")


def test_describe_and_self_test_run():
    d = catalog.describe()
    check("advertised:" in d and "RESERVED" in d and "yamadori-max" in d,
          "describe() lists advertised, reserved and resolvable names")


def test_the_context_window_is_advertised():
    """Item 4 (2026-09-23): /v1/models said nothing about the window, so a
    harness had to guess when to compact. Every advertised model carries the
    MAIN share of the pool, under the three names clients read, computed from
    budget.py at runtime -- so it follows `-c`."""
    rows = catalog.public_list()["data"]
    want = budget.budgets()["main"]
    check(want == 102400, "at -c 163840 the main share is 102,400", str(want))
    check(all(m.get(f) == want for m in rows for f in catalog.CONTEXT_FIELDS),
          "context_length, max_model_len and context_window = the main share",
          json.dumps(rows)[:300])
    old = budget._POOL
    try:
        budget._POOL = 262144
        grown = catalog.public_list()["data"][0]
        check(grown["context_length"] == int(262144 * budget.MAIN_SHARE)
              == 163840,
              "a larger pool advertises a larger window with no edit",
              json.dumps(grown))
    finally:
        budget._POOL = old
    old_expose = catalog.EXPOSE_INTERNAL
    try:
        catalog.EXPOSE_INTERNAL = True
        rows = {m["id"]: m for m in catalog.public_list()["data"]}
        chat = [n for n in rows if catalog.CATALOG[n][0] == "bonsai"]
        other = [n for n in rows if catalog.CATALOG[n][0] != "bonsai"]
        check(chat and all(rows[n].get("context_length") == want for n in chat),
              "exposed chat variants carry it too", str(chat))
        # The vision copy runs its own -c, and embeddings/reranker are not
        # chat models: the chat card would be false of them.
        check(other and all("context_length" not in rows[n]
                            and "supported_parameters" not in rows[n]
                            for n in other),
              "exposed non-chat names get no chat card", str(other))
    finally:
        catalog.EXPOSE_INTERNAL = old_expose
    real = catalog.context_window
    try:
        catalog.context_window = lambda: (_ for _ in ()).throw(OSError("down"))
        row = catalog.public_list()["data"][0]
        check(row["id"] == "yamadori" and "context_length" not in row,
              "an unknown window is left out, never guessed", json.dumps(row))
    finally:
        catalog.context_window = real
    try:
        from starlette.testclient import TestClient
        import server
        got = TestClient(server.app).get("/v1/models").json()
        check(got["data"][0].get("context_length") == want,
              "the ASGI route serves it", json.dumps(got)[:200])
    except ImportError as e:
        check(False, "the ASGI route could be imported", str(e))


def _walk(node):
    """Every (key, value) in a JSON tree, depth first, in the row's order."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _card(name="yamadori"):
    rows = [m for m in catalog.public_list()["data"] if m["id"] == name]
    return rows[0] if rows else {}


class _Env:
    """Set (or unset, with None) an environment variable for one block."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __enter__(self):
        self.old = os.environ.get(self.name)
        if self.value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.value

    def __exit__(self, *_):
        if self.old is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.old


def test_the_card_is_complete_and_follows_the_budget():
    """Operator, 2026-09-23: "Our harness should not have to dial anything
    in." Every field a harness reads is there, and every number in it
    derives from budget.py / tiers.py at runtime."""
    import tiers
    with _Env("YAMADORI_VISION", None):
        m = _card()
    main = budget.budgets()["main"]
    want_out = max(main // catalog.OUTPUT_FRACTION, tiers.A_MIN)
    for f in ("id", "object", "created", "owned_by", "name", "description",
              *catalog.CONTEXT_FIELDS, *catalog.OUTPUT_FIELDS, "architecture",
              "top_provider", "pricing", "supported_parameters", "x_yamadori"):
        check(f in m, f"the card has `{f}`")
    check(m.get("name") == "Yamadori" and len(m.get("description") or "") > 40,
          "a human name and description", str(m.get("name")))
    check(all(m.get(f) == main for f in catalog.CONTEXT_FIELDS)
          and m["top_provider"]["context_length"] == main,
          "every window field, top_provider included, is the main share",
          json.dumps(m)[:300])
    check(want_out == 20480
          and all(m.get(f) == want_out for f in catalog.OUTPUT_FIELDS)
          and m["top_provider"]["max_completion_tokens"] == want_out,
          "the output ceiling is main / 5 = 20,480 everywhere it is written",
          str(m.get("max_completion_tokens")))
    check(tiers.A_MIN <= want_out < main,
          "the output ceiling sits between A_MIN and the window")
    arch = m.get("architecture") or {}
    check(arch.get("input_modalities") == ["text", "image"]
          and arch.get("output_modalities") == ["text"]
          and arch.get("modality") == "text+image->text",
          "vision on: text + image in, text out", json.dumps(arch))
    params = m.get("supported_parameters") or []
    for p in ("tools", "tool_choice", "reasoning_effort", "max_tokens",
              "response_format", "stop", "stream"):
        check(p in params, f"supported_parameters lists `{p}`")
    # The vendor sampling is enforced (tiers.enforce_sampling): a client that
    # sends these has them overridden, so listing them would be false.
    for p in ("temperature", "top_p", "top_k", "presence_penalty"):
        check(p not in params, f"supported_parameters does not claim `{p}`")
    eff = m["x_yamadori"]["reasoning_effort"]
    check(eff["values"] == ["minimal", "low", "medium", "high", "xhigh", "max"]
          == tiers.ORDER, "every reasoning_effort value is listed", str(eff))
    check(all(tiers.normalise(v) == v for v in eff["values"]),
          "each listed value selects the tier of that name")
    check(eff["default"] == tiers.DEFAULT == "medium",
          "the default effort is the tiers default", str(eff["default"]))
    check(m["x_yamadori"]["answer_allowance_floor"] == tiers.A_MIN,
          "the answer-allowance floor is A_MIN")
    check(not any(k == "max_tokens" for k, _ in _walk(m)),
          "no `max_tokens` key anywhere (Hermes reads it as a window)")
    # How two clients read a row, reimplemented from their source:
    #  - a walker (Hermes Agent 35b14ad, agent/model_metadata.py L458-461
    #    keys, _extract_first_int walks nested dicts, first key wins,
    #    case-insensitive; L832-839 keeps ints in [1024, 10M]): whichever
    #    key it meets first must be the window.
    #  - OpenRouter-shaped (Roo-Code b867ec9, src/api/providers/fetchers/
    #    openrouter.ts L121-123, L211-218).
    walker_keys = {"context_length", "context_window", "context_size",
                   "max_context_length", "max_position_embeddings",
                   "max_model_len", "max_input_tokens", "max_sequence_length",
                   "max_seq_len", "n_ctx_train", "n_ctx", "ctx_size"}
    hits = [v for k, v in _walk(m) if k.lower() in walker_keys]
    check(hits and all(isinstance(v, int) and 1024 <= v <= 10_000_000
                       and v == main for v in hits),
          "every window key a walking client can meet is the main share",
          str(hits))
    import math
    tp = m.get("top_provider") or {}
    roo_max = tp.get("max_completion_tokens") or math.ceil(m["context_length"] * 0.2)
    check(roo_max == want_out and m["context_length"] == main
          and "image" in arch.get("input_modalities", []),
          "an OpenRouter-shaped reader gets window, ceiling and images",
          f"{roo_max}")
    old = budget._POOL
    try:
        budget._POOL = 262144
        g = _card()
        check(g["context_length"] == 163840 and g["max_output_tokens"] == 32768
              and g["top_provider"]["max_completion_tokens"] == 32768,
              "a larger pool moves the window and the ceiling with no edit",
              json.dumps(g)[:200])
        budget._POOL = 147456
        g = _card()
        check(g["context_length"] == 92160 and g["max_output_tokens"] == 18432,
              "at -c 147456: 92,160 and 18,432", json.dumps(g)[:200])
        budget._POOL = 8192
        g = _card()
        check(g["max_output_tokens"] == tiers.A_MIN,
              "a tiny pool never advertises less than A_MIN", str(g))
    finally:
        budget._POOL = old


def test_image_input_follows_the_vision_switch():
    with _Env("YAMADORI_VISION", "0"):
        m = _card()
    arch = m.get("architecture") or {}
    check(arch.get("input_modalities") == ["text"]
          and arch.get("modality") == "text->text",
          "YAMADORI_VISION=0: text in only", json.dumps(arch))
    check("image" not in (m.get("description") or "").lower(),
          "and the description does not offer images")


def test_nothing_internal_leaks():
    """The card is built from our numbers, never copied from the model
    server: no internal model name, no file path, no port, and not the raw
    pool (the conversation gets 5/8 of it)."""
    import re
    import tiers  # noqa: F401
    pool = budget.budgets()["pool"]
    names = ["yamadori", "gpt-4", "bonsai", "yamadori-embed", "", "a/b"]
    bodies = [catalog.public_list()] + [catalog.model_card(n) for n in names]
    for body in bodies:
        rows = body["data"] if "data" in body else [body]
        for r in rows:
            r = {k: v for k, v in r.items() if k != "created"}
            text = json.dumps(r).lower()
            words = set(re.findall(r"[a-z0-9_.\-]+", text))
            leaked = sorted(words & INTERNAL_NAMES)
            check(not leaked, f"{r['id']!r}: no internal model name", str(leaked))
            for bad in ("ternary", "bonsai", "gguf", "mmproj", "llama",
                        "11434", "10001", ":\\\\", "c:/", "/users/",
                        str(pool)):
                check(bad not in text, f"{r['id']!r}: no `{bad}` in the card")


def test_model_detail_route_and_unknown_names():
    import tiers
    list_row = {k: v for k, v in _card().items() if k != "created"}
    for name in ("yamadori", "gpt-4", "openai/gpt-4o", "bonsai",
                 "yamadori-embed", "metsumi"):
        got = {k: v for k, v in catalog.model_card(name).items()
               if k != "created"}
        check(got == list_row,
              f"model_card({name!r}) is the product's card", str(got)[:160])
    # Unknown names still resolve to the chat product for a chat request.
    check(catalog.resolve("gpt-4") == ("bonsai", None, False)
          and catalog.resolve("openai/gpt-4o")[0] == "bonsai",
          "an unknown name still resolves to the product")
    try:
        from starlette.testclient import TestClient
        import server
    except ImportError as e:
        check(False, "the ASGI app could be imported", str(e))
        return
    c = TestClient(server.app)
    for path in ("/v1/models/yamadori", "/v1/models/gpt-4",
                 "/v1/models/openai/gpt-4o"):
        r = c.get(path)
        body = r.json() if r.status_code == 200 else {}
        check(r.status_code == 200 and body.get("id") == "yamadori"
              and body.get("context_length") == budget.budgets()["main"]
              and body.get("max_output_tokens")
              == max(budget.budgets()["main"] // 5, tiers.A_MIN),
              f"GET {path} serves the card", f"{r.status_code} {r.text[:160]}")


def test_raw_server_routes_are_not_served():
    """Operator, 2026-09-23: an OpenAI-compatible proxy serves /v1/models and
    /v1/models/{id}, and no llama.cpp (/props), Ollama (/api/show, /api/tags)
    or LM Studio (/api/v0|v1/models) route. Those describe the raw server:
    the whole pool, the model path. Not served, and never forwarded."""
    import urllib.request
    try:
        from starlette.testclient import TestClient
        import server
    except ImportError as e:
        check(False, "the ASGI app could be imported", str(e))
        return
    calls = []
    real = urllib.request.urlopen

    def spy(req, *a, **k):
        calls.append(getattr(req, "full_url", req))
        raise AssertionError("no upstream call is allowed here")
    urllib.request.urlopen = spy
    try:
        c = TestClient(server.app)
        for path in ("/props", "/v1/props", "/api/show", "/api/tags",
                     "/api/v0/models", "/api/v1/models"):
            for accept in (None, "*/*", "application/json"):
                h = {"accept": accept} if accept else {}
                r = c.get(path, headers=h)
                check(r.status_code == 404 and "n_ctx" not in r.text,
                      f"GET {path} (Accept {accept}) is a 404",
                      f"{r.status_code} {r.text[:120]}")
        r = c.post("/api/show", json={"model": "yamadori"})
        check(r.status_code in (404, 405) and "n_ctx" not in r.text,
              "POST /api/show is refused", f"{r.status_code} {r.text[:120]}")
    finally:
        urllib.request.urlopen = real
    check(not calls, "nothing was forwarded upstream", str(calls))


def main() -> int:
    for fn in (test_the_fixture_is_the_shipped_setting,
               test_only_the_product_is_advertised,
               test_the_context_window_is_advertised,
               test_the_card_is_complete_and_follows_the_budget,
               test_image_input_follows_the_vision_switch,
               test_nothing_internal_leaks,
               test_model_detail_route_and_unknown_names,
               test_raw_server_routes_are_not_served,
               test_exposing_internal_names_is_opt_in,
               test_names_resolve_to_a_working_model,
               test_bonsai_and_bonsai_agent_are_the_same_process,
               test_replies_carry_the_product_name_and_no_bookkeeping,
               test_describe_and_self_test_run):
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
