#!/usr/bin/env python
"""Search a repository's dependencies when its own code has no answer.

WHY A FALLBACK AND NOT A MERGED INDEX

Merging a project's source with its dependencies buries the project. `three`
alone is 15,021 chunks against koota's 1,351, so a merged index answers almost
every question with library internals and the user's own code stops winning
anything. Dependencies are consulted only when the repository itself came back
empty.

WHY IT IS LABELLED LOUDLY

An answer drawn from `three@0.185.1` and an answer drawn from the user's own
file look identical once they are both just text with a path. Reading library
internals as if they were project code is a specific and expensive confusion,
so every result says which package and which version it came from.

VERSION IS PART OF THE IDENTITY. An index of three@0.186 answering a question
about three@0.180 is worse than no index, because it is confidently wrong
about an API that moved.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deps  # noqa: E402

# More than a couple of packages per query is a sign the question was not
# really about a dependency, and each one costs a fresh index load.
MAX_PACKAGES = 3


def relevant(root: str, query: str) -> list[tuple[str, str]]:
    """Packages worth consulting for this query, most-imported first.

    A package is only considered if the repository actually imports it and it
    has already been indexed -- this never triggers a download mid-question.
    """
    hits = []
    words = {w.lower() for w in query.replace("/", " ").replace(".", " ").split()}
    for name, version, _count in deps.worth_indexing(root):
        if not deps.is_indexed(name, version):
            continue
        # A query naming the package outright goes first; otherwise fall back
        # to import frequency, which is already the sort order.
        named = name.split("/")[-1].lower() in words
        hits.append((named, name, version))
    hits.sort(key=lambda h: (not h[0],))
    return [(n, v) for _named, n, v in hits[:MAX_PACKAGES]]


def search(root: str, query: str, tool: str, args: dict) -> str | None:
    """Re-run a tool against dependency indexes. None if nothing was found."""
    import code_search as cs

    prev_db = cs.INDEX_DB
    prev_env = os.environ.get("CODE_INDEX_DB")
    out = []
    try:
        for name, version in relevant(root, query):
            db = deps.db_path(name, version)
            os.environ["CODE_INDEX_DB"] = db
            cs.INDEX_DB = db
            try:
                resp = cs.handle({"jsonrpc": "2.0", "id": 1,
                                  "method": "tools/call",
                                  "params": {"name": tool, "arguments": args}})
                text = resp["result"]["content"][0]["text"]
            except Exception:                                    # noqa: BLE001
                continue
            if _is_empty(text):
                continue
            out.append(f"== from dependency {name}@{version}, NOT this "
                       f"repository's own code ==\n{text[:2500]}")
            if out:
                break      # one good source is enough; more is noise
    finally:
        cs.INDEX_DB = prev_db
        if prev_env is not None:
            os.environ["CODE_INDEX_DB"] = prev_env
        else:
            os.environ.pop("CODE_INDEX_DB", None)
    return "\n\n".join(out) if out else None


def _is_empty(text: str) -> bool:
    head = (text or "")[:120].lower()
    return (not text or "no matches" in head or "not found" in head
            or "no definition" in head or "no references" in head
            or "matched 0 of" in head or "no results" in head)


def ensure_for(root: str) -> list[dict]:
    """Index the dependencies this repository imports, in the background.

    Affordable because indexing needs no GPU once embeddings are skipped:
    measured, three@0.185.1 fetched in 0.8s and indexed in 32s. Shared across
    every repository on the same version, so most of these are already done.
    """
    import subprocess
    import threading

    plan = deps.plan(root)
    todo = [p for p in plan if not p["chunks"]]

    def work():
        for p in todo[:deps.__dict__.get("MAX_PACKAGES", 25)]:
            src = deps.fetch(p["name"], p["version"])
            if not src:
                continue
            env = dict(os.environ)
            env["CODE_INDEX_DB"] = deps.db_path(p["name"], p["version"])
            env["INDEX_NO_EMBED"] = "1"
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "scripts", "index_code.py")
            try:
                subprocess.run([sys.executable, script, src], env=env,
                               capture_output=True, text=True, timeout=1800)
            except Exception:                                    # noqa: BLE001
                pass

    if todo:
        threading.Thread(target=work, daemon=True).start()
    return plan


if __name__ == "__main__":
    r = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    for p in deps.plan(r):
        state = f"{p['chunks']} chunks" if p["chunks"] else "not indexed"
        print(f"  {p['name']:<28} {p['version']:<12} {p['imports']:>4} imports  {state}")
