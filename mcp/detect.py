#!/usr/bin/env python
"""Work out which repository a conversation is about, from the conversation.

A chat completion carries no working directory. But an agentic client leaks
its location constantly and cannot help doing so:

  - most harnesses state the cwd in the system prompt outright
  - every file the client's own read tool returns arrives as an absolute path
  - the user pastes stack traces, import errors and diffs, all with paths
  - tool results echo paths back

So the repository is recoverable without anyone configuring anything. Find the
paths, keep the ones that exist on disk, walk up to the nearest .git, and take
the directory that the most evidence points at.

Being wrong is cheap and being silent is not: if this picks the wrong repo the
model gets worse search results, which it can recover from. If it refuses to
guess, the tools are useless until a human configures something, which is the
entire failure this exists to avoid.
"""
from __future__ import annotations

import os
import re
from collections import Counter

# Windows drive paths and POSIX absolute paths. Deliberately greedy about what
# counts as a path character -- false positives get filtered by os.path.exists.
_WIN = re.compile(r"[A-Za-z]:[\\/](?:[^\s\"'<>|?*\n,;:()\[\]]+[\\/])*[^\s\"'<>|?*\n,;:()\[\]]*")
_POSIX = re.compile(r"/(?:[A-Za-z0-9._\-+@]+/)+[A-Za-z0-9._\-+@]*")

# Directories that are never a project root even when a path runs through them.
_NEVER = {"node_modules", ".git", "dist", "build", "target", "__pycache__",
          ".venv", "venv", "site-packages", "installer_files", ".cache"}

MARKERS = (".git", "package.json", "Cargo.toml", "pyproject.toml", "go.mod",
           "pnpm-workspace.yaml", "deno.json", "AGENTS.md", "CLAUDE.md")


def extract_paths(text: str) -> list[str]:
    out = []
    for m in _WIN.finditer(text):
        out.append(m.group(0))
    for m in _POSIX.finditer(text):
        out.append(m.group(0))
    return out


def _root_of(path: str) -> str | None:
    """Walk up from a path to the repository root.

    `.git` decides, and the other markers are only a fallback. A monorepo
    package carries its own package.json, so stopping at the first marker
    found returns `koota/packages/core` when the repository is `koota` --
    which then indexes one package instead of the project and looks, from the
    outside, exactly like retrieval being bad.
    """
    p = os.path.abspath(path.replace("\\", "/").rstrip("/"))
    if os.path.isfile(p):
        p = os.path.dirname(p)

    fallback = None
    seen = 0
    while p and seen < 40:
        seen += 1
        base = os.path.basename(p)
        if base not in _NEVER and os.path.isdir(p):
            if os.path.exists(os.path.join(p, ".git")):
                # Nearest .git wins: for a submodule that is the submodule,
                # which is the right answer, and for a monorepo there is only
                # one anyway.
                return p
            if fallback is None:
                for marker in MARKERS[1:]:
                    if os.path.exists(os.path.join(p, marker)):
                        fallback = p
                        break
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return fallback


def detect_repo(messages: list[dict], trusted_only: bool = False
                ) -> tuple[str | None, dict]:
    """Return (repo_root, evidence). evidence is for logging, not for the model.

    Later messages count for more: a conversation can wander across repos, and
    what the user is talking about NOW is what matters.
    """
    votes: Counter = Counter()
    n = len(messages) or 1
    for i, msg in enumerate(messages):
        role = msg.get("role")
        # The system prompt is written by the harness and the user types their
        # own messages. A tool result is text that came OUT of a repository,
        # so a repository can put a path there and choose what gets indexed.
        if trusted_only and role not in ("system", "user"):
            continue
        content = msg.get("content")
        if isinstance(content, list):      # multimodal content blocks
            content = " ".join(c.get("text", "") for c in content
                               if isinstance(c, dict))
        if not isinstance(content, str) or not content:
            continue
        # Linear recency weight. The system prompt still counts -- it usually
        # holds the cwd -- but a path the user mentioned two turns ago counts
        # for more than one mentioned at the start of a long session.
        weight = 1.0 + 2.0 * (i / n)
        # A harness that states its working directory is the strongest single
        # signal available, so it is worth recognising explicitly.
        for m in re.finditer(r"(?:working directory|cwd|workspace|project root)"
                             r"\s*[:=]\s*([^\n]+)", content, re.I):
            for p in extract_paths(m.group(1)):
                r = _root_of(p)
                if r:
                    votes[r] += 10.0 * weight
        for p in extract_paths(content):
            r = _root_of(p)
            if r:
                votes[r] += weight

    if not votes:
        return None, {"reason": "no existing paths found in the conversation"}
    root, score = votes.most_common(1)[0]
    runner_up = votes.most_common(2)[1] if len(votes) > 1 else None
    return root, {"score": round(score, 1),
                  "considered": len(votes),
                  "runner_up": runner_up[0] if runner_up else None,
                  "margin": round(score - runner_up[1], 1) if runner_up else None}


if __name__ == "__main__":
    import json
    import sys
    msgs = json.load(sys.stdin)
    root, ev = detect_repo(msgs)
    print(json.dumps({"repo": root, "evidence": ev}, indent=2))
