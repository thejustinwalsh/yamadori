#!/usr/bin/env python
"""Accounts, identified by the API key the client already sends.

WHY THIS IS STILL ZERO-CONFIG

Every OpenAI-compatible client has an API key field, already filled in, sent
on every request as `Authorization: Bearer ...`. Nobody experiences typing it
as configuration. So the key IS the account: no login, no separate route per
user, no second thing to set up. A client that was going to work anyway keeps
working, and the server learns who is asking.

WHAT AN ACCOUNT BUYS

  - its own set of repositories, and its own indexes
  - a place to hold a git credential so the server can fetch private code on a
    schedule, without the model or any tool ever seeing it
  - periodic refresh while the user is not working, which is free: indexing is
    tree-sitter plus sqlite since embeddings were cut, so it costs CPU at 3am
    rather than latency at 3pm

WHAT IT MUST NOT BUY

One account's index must never answer another account's question. That is not
a policy, it is a path: indexes live under the account's own directory and a
request can only ever name its own.

PUBLIC PACKAGES ARE SHARED ON PURPOSE. three@0.185.1 is the same source for
everyone, so it lives in a global store. The distinction is exact: public,
version-pinned, registry-published code is shared; anything belonging to an
account is not.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("YAMADORI_ACCOUNTS_DIR",
                       os.path.join(HERE, "..", "index", "accounts"))
REGISTRY = os.path.join(STORE, "accounts.json")

# With no accounts defined the server is single-user and open, which is the
# right default for a box on a private network. Creating one account switches
# it into multi-tenant mode and starts requiring a key.
ANON = "anonymous"


def _load() -> dict:
    try:
        with open(REGISTRY, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save(d: dict) -> None:
    os.makedirs(STORE, exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, REGISTRY)
    try:
        os.chmod(REGISTRY, 0o600)
    except OSError:
        pass


def _hash(key: str) -> str:
    """Keys are stored hashed. A leaked registry then leaks nothing usable."""
    return hashlib.sha256(("yamadori/" + key).encode()).hexdigest()


def multi_tenant() -> bool:
    return bool(_load())


def create(label: str) -> str:
    """Mint a key. Returned once; only its hash is kept."""
    key = "ym-" + secrets.token_urlsafe(32)
    d = _load()
    d[_hash(key)] = {"label": label, "created": time.time(), "uses": 0}
    _save(d)
    return key


def identify(auth_header: str | None) -> tuple[str | None, str]:
    """(account_id, reason). account_id None means reject.

    Comparison is constant-time: a plain == on a secret leaks its prefix
    through timing, and the fix costs nothing.
    """
    if not multi_tenant():
        return ANON, "single-user mode"
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None, "no API key supplied"
    key = auth_header.split(" ", 1)[1].strip()
    h = _hash(key)
    d = _load()
    for known, meta in d.items():
        if hmac.compare_digest(known, h):
            meta["uses"] = meta.get("uses", 0) + 1
            meta["last_seen"] = time.time()
            _save(d)
            return known[:16], f"account {meta.get('label', '?')}"
    return None, "unrecognised API key"


def account_dir(account_id: str) -> str:
    """Where this account's indexes live. Nothing else may read it."""
    safe = "".join(c for c in account_id if c.isalnum() or c in "-_")[:32] or ANON
    p = os.path.join(STORE, safe)
    os.makedirs(p, exist_ok=True)
    return p


# ------------------------------------------------------------ repo watchlist
def watchlist(account_id: str) -> list[dict]:
    p = os.path.join(account_dir(account_id), "repos.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def watch(account_id: str, remote: str, note: str = "") -> None:
    """Track a repository for scheduled refresh.

    Stores the REMOTE, not a credential. Fetching uses whatever git credential
    helper the server is configured with, so no token is ever held here, read
    by a tool, or reachable by the model.
    """
    rows = watchlist(account_id)
    if not any(r["remote"] == remote for r in rows):
        rows.append({"remote": remote, "note": note, "added": time.time()})
        p = os.path.join(account_dir(account_id), "repos.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


def usage(account_id: str) -> dict:
    """Disk actually consumed, so quotas can be enforced on a real number."""
    d = account_dir(account_id)
    total = 0
    for dirpath, _, files in os.walk(d):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return {"account": account_id, "bytes": total,
            "mb": round(total / 1e6, 1), "repos": len(watchlist(account_id))}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "create":
        k = create(sys.argv[2])
        print("  key (shown once, only its hash is stored):")
        print("   ", k)
    else:
        d = _load()
        print(f"  mode: {'multi-tenant' if d else 'single-user (open)'}")
        for h, m in d.items():
            print(f"    {h[:16]}  {m.get('label')}  uses={m.get('uses', 0)}")
