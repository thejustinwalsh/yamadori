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


def main() -> int:
    for fn in (test_the_fixture_is_the_shipped_setting,
               test_only_the_product_is_advertised,
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
