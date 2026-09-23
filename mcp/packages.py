#!/usr/bin/env python
"""Search a repository's dependencies when its own code has no answer.

WHY A FALLBACK AND NOT A MERGED INDEX

Merging a project's source with its dependencies buries the project. `three`
alone is 15,021 chunks against koota's 1,351, so a merged index answers almost
every question with library internals and the user's own code stops winning
anything. Dependencies are consulted only when the repository itself came back
empty.

WHY IT SAYS WHERE IT CAME FROM

This is a projection from a bundled dependency's source back into the answer,
and saying so is useful rather than cautionary. A caller who knows a result
came from `three@0.185.1` can act on it -- check the installed version, read
more of that file, cite it. A caller who does not know has to guess whether a
path belongs to their project.

Earlier wording framed this as a warning ("NOT this repository's own code").
That is negation, which this stack has measured to be read as topic rather
than as a constraint, and it undersells a capability: reading the source of
what you depend on is the point, not a hazard.

VERSION IS PART OF THE IDENTITY. An index of three@0.186 answering a question
about three@0.180 is worse than no index, because it is confidently wrong
about an API that moved.
"""
from __future__ import annotations

import json
import os
import sys
import threading

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
            out.append(f"== projected from {name}@{version} source ==\n"
                       f"{_PAIRING}\n{text[:2500]}")
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
    """Did that search come back with nothing?

    STRUCTURE FIRST, PROSE AS A FALLBACK.

    This was a substring match over the first 120 characters, which worked for
    as long as tool results were sentences. They are now JSON envelopes --
    `{"tool": ..., "ok": true, "matches": 0, ...}` for a real miss and
    `{"ok": false, "error": "NO_INDEX", ...}` for a failure -- and none of the
    phrases below appear in either.

    The consequence was silent and specific: this predicate is the ONLY thing
    that triggers the dependency fallback, the path that makes the service
    useful to a caller whose disk we will never see. It stopped firing for
    find_by_meaning and find_by_pattern the moment their empty results became
    structured, so a remote user asking about a library the server holds got a
    zero-match envelope and no fallback at all.

    A regression introduced by changing a producer without checking its
    consumers. Structured results are read as structure; the prose branch stays
    for the tools that still return sentences.
    """
    if not text:
        return True
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            d = json.loads(stripped)
        except ValueError:
            pass
        else:
            if isinstance(d, dict):
                # A failure is not "empty" in the sense this asks about -- a
                # missing index means the dependency indexes cannot help
                # either, and pretending otherwise starts a pointless search.
                if d.get("ok") is False:
                    return False
                return int(d.get("matches") or 0) == 0
    head = stripped[:120].lower()
    return ("no matches" in head or "not found" in head
            or "no definition" in head or "no references" in head
            or "matched 0 of" in head or "no results" in head)


def ensure_for(root: str) -> list[dict]:
    """Index the dependencies this repository imports, in the background.

    Affordable because indexing needs no GPU once embeddings are skipped:
    measured, three@0.185.1 fetched in 0.8s and indexed in 32s. Shared across
    every repository on the same version, so most of these are already done.
    """
    plan = deps.plan(root)
    # "Not usable", not merely "no chunks": a README-only index built by the
    # old dist/-skipping walk has chunks and must still be rebuilt. Each
    # package is attempted once per process -- this runs on every request,
    # and a package that cannot be made healthy must not be refetched and
    # reindexed on every one of them.
    with _ENSURE_LOCK:
        todo = []
        for p in plan:
            key = (p["name"], p["version"])
            if key in _ATTEMPTED or not deps.needs_index(*key):
                continue
            _ATTEMPTED.add(key)
            todo.append(p)

    def work():
        for p in todo[:deps.__dict__.get("MAX_PACKAGES", 25)]:
            try:
                deps.index_package(p["name"], p["version"], embed=False)
            except Exception:                                    # noqa: BLE001
                pass

    if todo:
        threading.Thread(target=work, daemon=True).start()
    return plan


_ENSURE_LOCK = threading.Lock()
_ATTEMPTED: set[tuple[str, str]] = set()


if __name__ == "__main__":
    r = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    for p in deps.plan(r):
        state = f"{p['chunks']} chunks" if p["chunks"] else "not indexed"
        print(f"  {p['name']:<28} {p['version']:<12} {p['imports']:>4} imports  {state}")


# ---------------------------------------------------------------------------
# Serving a remote caller, who has no repository here
# ---------------------------------------------------------------------------
#
# Everything above assumes a local checkout: `relevant` reads the caller's
# lockfile off disk to decide which dependencies matter. A remote caller has
# no disk here. What they do have is code, and code names its own libraries.
#
# So the same machinery is driven from `discover.scan` instead of a lockfile.
# The one thing imports cannot supply is the version, and that gap is stated
# rather than papered over -- see `banner_for`.


def indexed_versions(name: str) -> list[str]:
    """Versions of a package already indexed on this server, newest first."""
    root = os.path.join(os.path.dirname(os.path.abspath(deps.__file__)),
                        "..", "index", "packages")
    if not os.path.isdir(root):
        return []
    want = deps.slug(name, "").rstrip("@")
    out = []
    for fn in os.listdir(root):
        if not fn.endswith(".sqlite3") or "@" not in fn:
            continue
        stem = fn[:-len(".sqlite3")]
        pkg, _, ver = stem.rpartition("@")
        if pkg == want:
            out.append(ver)

    def key(v: str) -> tuple:
        return tuple(int(p) if p.isdigit() else -1 for p in v.split("."))

    return sorted(out, key=key, reverse=True)


def catalogue() -> dict[str, list[str]]:
    """Every package index this server holds: {name: [versions, newest first]}.

    Exists so an error can name what IS available instead of only what is
    missing. A tool that reports "nothing is indexed" while holding source for
    two hundred libraries has said something true and useless; the actionable
    fact is which of them it can answer from, and how to reach them.
    """
    root = os.path.join(os.path.dirname(os.path.abspath(deps.__file__)),
                        "..", "index", "packages")
    out: dict[str, list[str]] = {}
    if not os.path.isdir(root):
        return out
    for fn in os.listdir(root):
        if not fn.endswith(".sqlite3") or "@" not in fn:
            continue
        pkg, _, ver = fn[:-len(".sqlite3")].rpartition("@")
        out.setdefault(pkg, []).append(ver)

    def key(v: str) -> tuple:
        return tuple(int(p) if p.isdigit() else -1 for p in v.split("."))

    return {k: sorted(v, key=key, reverse=True) for k, v in sorted(out.items())}


def resolve(packages: list[str], versions: dict) -> list[dict]:
    """Match discovered package names to indexes this server actually holds.

    `exact` records whether the caller told us the version or we picked one.
    That distinction has to survive to the answer: three.js renamed half of
    TSL between 0.16x and 0.18x, so an answer from the wrong index is not
    slightly stale, it is about a different API with the same name.
    """
    out = []
    for name in packages[:MAX_PACKAGES]:
        have = indexed_versions(name)
        if not have:
            continue
        stated = (versions or {}).get(name)
        if stated and stated in have:
            out.append({"name": name, "version": stated, "exact": True})
        elif stated:
            # Stated, but not the one we hold. The nearest index is still far
            # better than nothing, and saying so is what makes it safe.
            out.append({"name": name, "version": have[0], "exact": False,
                        "asked_for": stated})
        else:
            out.append({"name": name, "version": have[0], "exact": False})
    return out


# Paths in a projected result name files on THIS SERVER, inside its own copy
# of the library source. The caller has no such file. Their machine has
# node_modules/three/build/three.module.js -- a bundle with entirely different
# line numbers -- or a .d.ts with no implementation in it at all.
#
# So a projected path is readable only by the tool that produced it. Handing
# `src/nodes/tsl/TSLCore.js:1124` to the harness's own file reader gets a
# missing file if you are lucky, and a confidently wrong region of a bundle if
# you are not. The pairing is stated in the RESULT, not only in the tool
# description, because the result is what the model is reading at the moment
# it decides what to do next.
_PAIRING = ("This is a source map for a dependency: the user has the built "
            "bundle, the server has the original source, and these paths are "
            "the server's. The user's node_modules has no file at this path "
            "and its line numbers are different. Read these with "
            "read_file_range, never with your own file tools, and name the "
            "library when you quote one.")


def banner_for(pick: dict) -> str:
    """The heading over a projected result. It must not overstate its source."""
    head = f"== projected from {pick['name']}@{pick['version']} source =="
    if pick.get("asked_for"):
        note = (f"You said {pick['name']}@{pick['asked_for']}. That version is "
                f"not indexed here, so this is {pick['version']}. Check "
                f"anything version sensitive.")
    elif not pick.get("exact"):
        note = ("The version was not stated, so this is the newest indexed. "
                "Ask the caller for the version in their lockfile if the "
                "answer depends on it.")
    else:
        note = ""
    return "\n".join(x for x in (head, note, _PAIRING) if x)


def search_discovered(state: dict, query: str, tool: str,
                      args: dict) -> str | None:
    """Run a tool against the libraries this session is known to be using."""
    import code_search as cs

    picks = resolve(state.get("packages") or [], state.get("versions") or {})
    if not picks:
        return None

    # A query that names a package outright means that package, whatever the
    # import counts say.
    words = {w.lower() for w in query.replace("/", " ").replace(".", " ").split()}
    picks.sort(key=lambda p: p["name"].split("/")[-1].lower() not in words)

    prev_db, prev_env = cs.INDEX_DB, os.environ.get("CODE_INDEX_DB")
    try:
        for pick in picks:
            db = deps.db_path(pick["name"], pick["version"])
            if not os.path.exists(db):
                continue
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
            return f"{banner_for(pick)}\n{text[:3000]}"
    finally:
        cs.INDEX_DB = prev_db
        if prev_env is not None:
            os.environ["CODE_INDEX_DB"] = prev_env
        else:
            os.environ.pop("CODE_INDEX_DB", None)
    return None
