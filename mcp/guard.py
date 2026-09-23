#!/usr/bin/env python
"""What the indexer is allowed to read, and which roots it may touch at all.

TWO SEPARATE PROBLEMS, BOTH REAL

1. SECRETS IN INDEXED FILES. The indexer ingests .json, .yaml, .toml, .ps1
   and .md because projects keep real source and config in them. Projects also
   keep credentials in them. Measured before this existed: secrets.json,
   credentials.toml and deploy.ps1 were all indexed outright, which copies
   their contents into a sqlite database and then feeds them to a model.

2. ATTACKER-CHOSEN ROOTS. Repository detection reads paths from the whole
   conversation, and a conversation contains tool results -- text that came
   out of a repository rather than from the user. A file saying
   "Working directory: C:/Users/you/something-private" is enough to make the
   proxy index that directory. Prompt injection into arbitrary local file
   read, with the contents then summarised back to whoever asked.

The second is the serious one, and it cannot be fixed by pattern matching.
The first fix drew the line between tool results (repository-controlled) and
the system/user messages (called "trusted"), and approved a root on its first
trusted mention. That line was in the wrong place: every message in a request
is written by whoever holds a key, so any caller could name any directory on
this server and read it back through the tools.

CLOSED 2026-09-22. Nothing in a request selects or approves a directory:
proxy.resolve_repo returns no root, and may_establish approves only what the
operator approved on this machine (`python mcp/guard.py approve PATH`). The
caller's own source reaches the model through the caller's harness tools; the
server serves what it holds.
"""
from __future__ import annotations

import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("YAMADORI_INDEX_DIR", os.path.join(HERE, "..", "index", "repos"))
CONSENT = os.path.join(STORE, "approved_roots.json")

# There is no open mode. YAMADORI_TRUST_ALL_ROOTS=1 used to approve every
# root; removed 2026-09-22 with request-declared roots, because a switch that
# approves any directory is one environment variable away from the hole that
# was closed. Roots are approved by the operator only (`approve` below).

# Names that are credentials regardless of extension. Matched on the basename,
# case-insensitively, so `.env.production` and `server.pem` are both caught.
SECRET_NAMES = re.compile(
    r"(^\.env($|\.)|^\.npmrc$|^\.netrc$|^\.pgpass$|^id_(rsa|ecdsa|ed25519)$|"
    r"\.pem$|\.pfx$|\.p12$|\.key$|\.keystore$|\.jks$|"
    r"^terraform\.tfstate|^\.htpasswd$|service[_-]account.*\.json$)",
    re.I)

# Words that mark a credential only in a DATA or CONFIG file's name:
# `secrets.json` and `credentials.toml` were the measured incidents. The same
# words in source file names are ordinary code -- `tokenizer.ts`,
# `password-reset.tsx`, `useApiKey.ts` -- and matching them there silently
# dropped real source from the index, a retrieval bug that is very hard to
# trace back to here. Source files are covered by the content check instead,
# which looks for actual key formats in every file.
SECRET_WORDS = re.compile(
    r"(secret|credential|password|passwd|token|apikey|api[_-]key)", re.I)
DATA_EXTS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
             ".env", ".txt", ".properties", ".xml", ".csv", ".ps1", ".sh",
             ".bat", ".cmd", ""}

# A file that is not named like a secret but contains one. Deliberately narrow:
# a false positive silently drops a real source file from the index, which is
# a retrieval bug that is hard to trace back to here.
SECRET_CONTENT = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\bAKIA[0-9A-Z]{16}\b|"                          # AWS access key id
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|"                # GitHub tokens
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b|"              # Slack
    r"\bsk-[A-Za-z0-9]{32,}\b|"                       # OpenAI-style
    r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.)", # JWT
)

MAX_SNIFF = 65536


def is_secret_path(path: str) -> bool:
    base = os.path.basename(path)
    if SECRET_NAMES.search(base):
        return True
    return (os.path.splitext(base)[1].lower() in DATA_EXTS
            and bool(SECRET_WORDS.search(base)))


def has_secret_content(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return bool(SECRET_CONTENT.search(f.read(MAX_SNIFF)))
    except OSError:
        return False


def should_index(path: str) -> tuple[bool, str]:
    """Called per file. Returns (allowed, reason-if-not).

    The content check runs on EVERY file, source included: a key pasted into
    `config.ts` is as much a leak as one in `config.json`, and the patterns
    are narrow key formats, not words.
    """
    if is_secret_path(path):
        return False, "name looks like a credential"
    if has_secret_content(path):
        return False, "contains a credential-shaped value"
    return True, ""


# ---------------------------------------------------------------- root trust
def _load() -> dict:
    try:
        with open(CONSENT, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save(d: dict) -> None:
    os.makedirs(STORE, exist_ok=True)
    tmp = CONSENT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, CONSENT)


def is_approved(root: str) -> bool:
    return os.path.abspath(root).lower() in {k.lower() for k in _load()}


def approve(root: str, how: str = "manual") -> None:
    d = _load()
    d[os.path.abspath(root)] = {"approved_at": time.time(), "how": how}
    _save(d)


def revoke(root: str) -> bool:
    d = _load()
    for k in list(d):
        if k.lower() == os.path.abspath(root).lower():
            del d[k]
            _save(d)
            return True
    return False


def approved_roots() -> list[str]:
    return sorted(_load())


def may_establish(root: str, from_trusted: bool) -> tuple[bool, str]:
    """Decide whether `root` may be indexed.

    `from_trusted` means the path came from the system prompt -- which the
    harness wrote -- rather than from a tool result or pasted text, which a
    repository can control. An already-approved root needs no evidence; a new
    one does.
    """
    if is_approved(root):
        return True, ""
    # NO APPROVAL FROM A REQUEST. "Trusted" used to mean the path appeared in
    # the system or user message -- both written by any key holder -- and
    # first sight approved it forever. Only the operator approves a root, on
    # this machine, with `python mcp/guard.py approve PATH`.
    return False, ("not an operator-approved root; a request cannot approve "
                   "a directory on this server")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "approve":
        approve(sys.argv[2], "cli")
        print(f"approved {sys.argv[2]}")
    elif len(sys.argv) > 2 and sys.argv[1] == "revoke":
        print("revoked" if revoke(sys.argv[2]) else "was not approved")
    else:
        for r in approved_roots():
            print(" ", r)
