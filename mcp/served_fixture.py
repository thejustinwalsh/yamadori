"""The served model's state, PINNED for offline suites.

Offline suites exercise code that asks the live stack what it serves:
`tiers.accepted_efforts()` reads the chat template through llama-swap's
`/upstream/bonsai/props`, and `budget.pool_size()` reads n_ctx and the slot
count from llama-server's `/props` (:10001, then llama-swap). The upstream
route LOADS `bonsai` when it is unloaded -- an offline run cut an engine test
window short twice (2026-09-27) -- and a suite whose answer depends on
whether the stack is up is not an offline result (PROTOCOL rule 3).

So a suite that reaches that code calls `pin()` first: the values come from
mcp/fixtures/bonsai_props.json and the template fixture beside it, recorded
from the served model, and `mcp/test_live_stack.py --only served` checks the
fixtures still match what is served. scripts/run_tests.py fails any offline
suite that still connects to a stack port (scripts/offline_guard/).

`no_embedder()` does the same for the resident embedder a suite reaches
through skill selection: an outage the code reports, as it would live.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
PROPS = os.path.join(FIXTURES, "bonsai_props.json")


def props() -> dict:
    """The pinned /props: n_ctx, total_slots, kv_vram_cells, eos_token and
    chat_template."""
    with open(PROPS, encoding="utf-8") as f:
        d = json.load(f)
    with open(os.path.join(FIXTURES, d["chat_template_file"]),
              encoding="utf-8") as f:
        d["chat_template"] = f.read()
    return d


def pin(tiers_module=None) -> dict:
    """Pin what budget and tiers would ask the server for. Returns the props.
    A value a suite pins itself afterwards (a different pool) still wins.
    `tiers_module`: tiers run as a script is `__main__`, not `tiers`."""
    import budget
    tiers = tiers_module
    if tiers is None:
        import tiers
    p = props()
    budget._POOL = int(p["n_ctx"])
    budget._SLOTS = int(p["total_slots"])
    # the tiered cache's VRAM line: the main cap (budget THE CAP LAYOUT)
    budget._LINE = (int(p["kv_vram_cells"])
                    if isinstance(p.get("kv_vram_cells"), int) else None)
    tiers._accepted = (tiers.efforts_in_template(p["chat_template"])
                       or tiers.FALLBACK_EFFORTS)
    pin_vision(p)
    return p


def pin_vision(p: dict | None = None) -> bool:
    """Pin vision.main_sees (whether the main model has a projector: its
    /props `modalities.vision`) to the fixture. Found 2026-09-28:
    proxy.prepare asked the live /props on every request of
    mcp/test_route.py and mcp/test_domains.py (7 refused connects each).
    YAMADORI_MAIN_VISION still overrides it."""
    import model
    import vision
    p = p or props()
    seen = bool((p.get("modalities") or {}).get("vision"))
    vision._sees.update(value=seen, model=model.MODEL,
                        why="pinned: mcp/fixtures/bonsai_props.json "
                            "modalities.vision (served_fixture)")
    return seen


def no_embedder() -> None:
    """Make code_search.embed an outage: skill selection reports it and
    goes on without the embedding, as it does when the embedder is down."""
    import code_search

    def embed(texts, is_query=False):
        raise OSError("offline suite: no embedder (mcp/served_fixture.py)")
    code_search.embed = embed
