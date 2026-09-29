"""Point every live store an offline suite could WRITE at a temp directory.

    import offline_stores                      # before any other mcp module
    _TMP = offline_stores.isolate("yamadori_test_x_")

Found 2026-09-27: mcp/test_domains.py ran proxy.prepare with no
YAMADORI_JOBS_DB, so skill selection wrote its fallback records, counters and
selection rows into the live index/jobs.sqlite3 -- and the worker learned
from them. The stores' paths are read when their modules are IMPORTED, so
this must run first. Only a variable the suite has not set itself is set
(a suite's own fixture path wins), and only for stores the stack WRITES:
nothing a suite legitimately reads as a fixture (the held packages, the
bound repository's code index, the token embeddings, the E1 heads, the
account registry) is moved.

scripts/run_tests.py enforces the rule whether or not a suite uses this: its
offline guard refuses and logs every write under index/ and logs/, and a
before/after snapshot of the live files backs it up.
"""
from __future__ import annotations

import os
import tempfile

# variable -> file or directory name under the temp dir. A store derived
# from another is listed anyway, so it cannot fall back to the live path:
# the skill store, trigger cache and labels sit beside YAMADORI_JOBS_DB.
STORES = {
    "YAMADORI_JOBS_DB": "jobs.sqlite3",          # jobs, skills, skill records
    "YAMADORI_SKILLS_DIR": "skills",
    "YAMADORI_SKILL_TRIGGER_CACHE": "skill_triggers.npz",
    "YAMADORI_SKILL_LABELS": "router_labels.jsonl",
    "YAMADORI_CORPUS_DB": "corpus.sqlite3",      # turns, deep_decisions
    "YAMADORI_NEBARI_DB": "nebari.sqlite3",      # sessions, the ledger
    "RINGS_DB": "rings.sqlite3",                 # the work log
    "YAMADORI_TOKEN_LEDGER": "token_ledger.sqlite3",
    "YAMADORI_SLOTS_STATE": "slots_state.json",
    "CONCEPT_SEED_LAST": "concept_seed_last.json",
    "YAMADORI_POWER_LEDGER": "power_ledger.json",
    "YAMADORI_GPU_ROOM_DIR": "gpu_room",
    "YAMADORI_MEDIA_DIR": "media",
    "YAMADORI_MEDIA_SECRET_FILE": "media_url.key",
    "YAMADORI_PRICES_DIR": "prices",
    "YAMADORI_DATASETS_DIR": "datasets",
    # package onboarding (mcp/onboarding.py): the stage outputs, the example
    # files and the example kNN index.
    "YAMADORI_ONBOARDING_DIR": "onboarding",
    "YAMADORI_EXAMPLES_DIR": "examples",
    "YAMADORI_EXAMPLE_KNN_DIR": "example_knn",
    # the MCP servers' configuration (mcp/mcp_config.py): read as the
    # built-in default when absent; the dashboard's writer writes it.
    "YAMADORI_MCP_SERVERS": "mcp_servers.json",
    # the harness kit (mcp/harness_kit.py): the dashboard's HARNESS TOOLS
    # entries; the API seeds it on first use.
    "YAMADORI_HARNESS_KIT_DB": "harness_kit.sqlite3",
    # the turn decider (mcp/decide_turn.py, mcp/decider_bonsai.py): its
    # disagreement and every-decision logs and the operator's truth labels
    # (bench/decider/label.py writes them; bench/decider/tune.py reads
    # them). The per-model abstain params file went with the decider's
    # built-in abstain (2026-09-29): thresholds are decide_turn.THRESHOLDS.
    "YAMADORI_DECIDER_LOG": "decider_disagreements.jsonl",
    "YAMADORI_DECIDER_DECISIONS": "decider_decisions.jsonl",
    "YAMADORI_DECIDER_LABELS": "decider_labels.jsonl",
}
# NOT moved (read as fixtures; the guard refuses a write to them): the held
# packages and index/packages/registry_history.json (deep.unseen reads it),
# the package registry (packages.json, held.json, vocabulary.json:
# YAMADORI_PKG_REGISTRY_DIR -- detection and the taxonomy read it; only an
# onboarding's vocab stage writes it, and a suite that exercises that stage
# points it at its own temp dir), CODE_INDEX_DB, YAMADORI_ACCOUNTS_DIR,
# YAMADORI_E1_DIR, token_embd.npz.


def isolate(prefix: str = "yamadori_offline_", *, keep: tuple = ()) -> str:
    """Create a temp dir and point every store in STORES that is not
    already set (and not in `keep`) at it. Returns the temp dir."""
    tmp = tempfile.mkdtemp(prefix=prefix)
    for var, name in STORES.items():
        if var in keep or os.environ.get(var):
            continue
        os.environ[var] = os.path.join(tmp, name)
    # No container either: the PROVE stage's type check (mcp/typecheck.py)
    # starts one per check; an offline suite records "not run" instead.
    os.environ.setdefault("YAMADORI_SKILL_TYPECHECK", "0")
    return tmp
