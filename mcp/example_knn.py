#!/usr/bin/env python
"""THE EXAMPLE kNN LABEL INDEX: nearest example code votes packages.

    python mcp/example_knn.py --check      is the index fresh? (exit 1 if not)
    python mcp/example_knn.py --build      rebuild it (the A4000 embedder)

docs/PACKAGE-ONBOARDING.md 6.2-6.5 (item I), with operator decision 1
(2026-09-27), which REPLACES the design's fixed held-out split: "Held-out
split: no fraction is invented. Use leave-one-GROUP-out cross-validation
over the example groups (every group is held out once, and its votes come
from an index built without it). k for the vote is chosen inside the
training folds only. If a fold count is ever needed, it's the number of
groups."

ONE INDEX over every onboarding's kept example files (each
examples/manifest.jsonl under onboarding.root()):

  CHUNKS      scripts/index_code.chunk_file (the chunker every package index
              uses; MAX_CHUNK_LINES sized to the embedder's window) over the
              file with its import / require lines REMOVED
              (package_examples.strip_imports): they are the label's source.
  LABELS      a chunk carries the HELD packages whose imported bindings
              (skill_packages.imported_names over the WHOLE file;
              package_of_specifier for subpaths: three/tsl, three/webgpu ->
              "three/tsl") it references -- several at once (multi-package
              rows). A chunk that references none carries no package
              evidence and is not indexed. Skill labels are not stored: a
              chunk's code-shaped identifiers are, and derived_skills()
              computes its skills from the ARMED set at load.
  DOCUMENTS   embedded PLAIN (code_search.embed, is_query=False), in the
              package indexes' batches (index_code.BATCH), each cut to
              index_code.EMBED_MAX_CHARS for the vector as index_code does.
  QUERIES     carry an instruction, documents none (the Qwen3-Embedding
              card: an instruction on the query side only; research doc
              2.2), in Qwen3-Embedding's get_detailed_instruct form (as
              skill_match.query_text): task TASK, versioned KNN_VERSION --
              UNMEASURED WORDING. A query is CODE with its imports stripped,
              cut to QUERY_CHARS (the bound skill_match.rank_all applies to
              its state). Every kept example FILE is also embedded as a
              query (stored beside the documents), so choosing k and the
              evaluation (package_eval) query exactly what vote() queries at
              serve time -- a file's code -- and need no embedder.
  THE VOTE    rank only, no cosine cut: every group's BEST chunk is its one
              vote; the top-k groups vote their best chunk's packages; the
              tally is ordered by votes, ties by the best rank, then name.
  k           MEASURED by leave-one-group-out over the groups (every group
              held out once; its files query every OTHER group), each k from
              1 to the largest number of groups any package labels,
              maximising the top-1 package accuracy of the vote; ties ->
              the smaller k. Stored with its n and the whole curve.

A GROUP is package_examples' (the directory directly under an examples root,
else the file), identified by repository and path, not commit: the same demo
at two versions is one group, so it never votes for its own earlier copy.

Files: YAMADORI_EXAMPLE_KNN_DIR (default: the directory of skill_match's
cache, index/skills) / example_knn.npz (vectors, chunk ids, signature),
example_knn.rows.jsonl (chunk id, group hash, package labels, code-shaped
identifiers, repository path at the commit -- never example text),
example_knn.files.jsonl (the query files: id, group, labels, path) and
example_knn.json (k, its n and curve, counts).

UPKEEP: the worker's minute (onboarding.tick) calls schedule(), which
enqueues ONE rebuild on the gpu lane with payload {"idle": true} (the
worker's idle gate reads it: the embedder is a GPU consumer) when the index
does not match the manifests and none is queued or running.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QUEUE = "package.example_knn_index"
LANE = "gpu_a4000"  # the embedder, on the A4000: jobs.GPU_SCOPES (2026-09-30)
KNN_VERSION = "knn/1"
# docs/PACKAGE-ONBOARDING.md 6.4; unmeasured wording.
TASK = ("Given code an agent is reading or writing, retrieve example code "
        "that uses the same libraries")
# skill_match.rank_all's bound on its state (`state[:4000]`), the same
# bound for the same embedder.
QUERY_CHARS = 4000
NPZ = "example_knn.npz"
ROWS = "example_knn.rows.jsonl"
FILES = "example_knn.files.jsonl"
META = "example_knn.json"

# Tests set this to a fake: (texts, is_query) -> unit vectors.
EMBED = None


def _embed_fn(embed=None):
    if embed is not None:
        return embed
    if EMBED is not None:
        return EMBED
    import code_search
    return code_search.embed


def cache_dir() -> str:
    d = os.environ.get("YAMADORI_EXAMPLE_KNN_DIR")
    if d:
        return os.path.abspath(d)
    import skill_match
    return os.path.dirname(os.path.abspath(skill_match._cache_path()))


def _path(name: str) -> str:
    return os.path.join(cache_dir(), name)


def query_text(code: str) -> str:
    """Qwen3-Embedding's get_detailed_instruct (as skill_match.query_text),
    with this index's task."""
    return f"Instruct: {TASK}\nQuery:{code}"


def query_of(code: str) -> str:
    """What a query embeds: the code with its imports stripped, cut to
    QUERY_CHARS."""
    import package_examples as PE
    return PE.strip_imports(code or "")[:QUERY_CHARS]


def _ic():
    import deps
    deps.sys_path_scripts()
    import index_code
    return index_code


# ---------------------------------------------------------------------------
# Inputs and freshness.
# ---------------------------------------------------------------------------
def _held_key() -> list[str]:
    import skill_packages as SP
    return sorted(SP.held())


def inputs_signature() -> tuple[str, int]:
    """(signature, manifest files) of what a build would read NOW: every
    manifest's bytes, the held packages (labels depend on them), this
    module's versions and the chunker's window. Cheap: no chunking."""
    import package_examples as PE
    h = hashlib.sha256()
    ms = PE.manifests()
    for did, p in ms:
        try:
            with open(p, "rb") as f:
                h.update(did.encode() + b"\0" + hashlib.sha256(
                    f.read()).hexdigest().encode() + b"\1")
        except OSError:
            continue
    h.update(json.dumps({"held": _held_key(), "knn": KNN_VERSION,
                         "task": TASK, "query_chars": QUERY_CHARS,
                         "rules": PE.RULES_VERSION,
                         "chunk_lines": _ic().MAX_CHUNK_LINES},
                        sort_keys=True).encode())
    return h.hexdigest()[:16], len(ms)


def index_state() -> dict:
    """{fresh, why, rows, path} for the manifests now."""
    import numpy as np
    path = _path(NPZ)
    sig, n_man = inputs_signature()
    out = {"fresh": False, "rows": 0, "path": path, "manifests": n_man,
           "inputs": sig}
    if not os.path.exists(path):
        if n_man == 0:
            return dict(out, fresh=True, why="no example manifests: "
                        "nothing to index")
        return dict(out, why="no index file")
    try:
        z = np.load(path, allow_pickle=False)
        have, rows = str(z["inputs"]), int(z["n"])
    except Exception as e:                                       # noqa: BLE001
        return dict(out, why=f"unreadable: {type(e).__name__}")
    out["rows"] = rows
    if have != sig:
        return dict(out, why="stale: the example manifests, the held "
                             "packages or the index rules changed since it "
                             "was built")
    return dict(out, fresh=True, why="matches the example manifests")


def schedule() -> str | None:
    """Enqueue ONE idle-gated rebuild when the index is stale and none is
    queued or running (the worker's minute)."""
    import jobs
    if index_state()["fresh"]:
        return None
    for st in ("queued", "running"):
        if any(j.get("queue") == QUEUE for j in jobs.listing(state=st,
                                                              limit=500)):
            return None
    return jobs.add(QUEUE, {"idle": True}, lane=LANE, stage="index")


def handle_build(job: dict, ctx=None) -> dict:
    rec = build(beat=getattr(ctx, "beat", None))
    return {k: v for k, v in rec.items() if k != "curve"}


# ---------------------------------------------------------------------------
# Building.
# ---------------------------------------------------------------------------
def _chunks_of(text: str, ext: str) -> list[tuple[int, int, str]]:
    """index_code.chunk_file over `text` (it reads a path: a temp file with
    the same extension, so the AST chunker sees the right grammar)."""
    fd, p = tempfile.mkstemp(suffix=ext, prefix="yamadori_knn_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        return _ic().chunk_file(p)
    finally:
        try:
            os.remove(p)
        except OSError:
            pass


def _labels_of(chunk: str, bindings: dict) -> list[str]:
    import package_examples as PE
    import re
    refs = set(re.findall(r"[A-Za-z_$][\w$]*", PE.code_only(chunk)))
    return sorted({bindings[b] for b in refs if b in bindings})


def collect_rows(beat=None) -> dict:
    """Chunk and label every kept example file (no embedding)."""
    import package_examples as PE
    beat = beat or (lambda s: None)
    files, chunks = [], []
    seen: set = set()
    dropped = {"no_text": 0, "no_label_chunks": 0, "duplicate_file": 0}
    ic = _ic()
    for did, _p in PE.manifests():
        for row in PE.manifest_rows(did):
            fid = row.get("id") or PE.file_id(row["repository"], row["path"],
                                              row.get("blob") or "")
            if fid in seen:
                dropped["duplicate_file"] += 1
                continue
            seen.add(fid)
            text = PE.read_stored(row)
            if text is None:
                dropped["no_text"] += 1
                continue
            beat(f"knn: chunking {row['path'][-120:]}")
            facts = PE.file_facts(text)
            stripped = PE.strip_imports(text)
            ext = os.path.splitext(row["path"])[1] or ".ts"
            gh = PE.group_hash(row["repository"], row["group"])
            files.append({"id": fid, "group": gh, "onboarding": did,
                          "repository": row["repository"],
                          "commit": row.get("commit"), "path": row["path"],
                          "labels": facts["packages"],
                          "query": query_of(text)})
            for s, e, t in _chunks_of(stripped, ext):
                labels = _labels_of(t, facts["bindings"])
                if not labels:
                    dropped["no_label_chunks"] += 1
                    continue
                chunks.append({"id": f"{fid}:{s}-{e}", "group": gh,
                               "file": fid, "labels": labels,
                               "identifiers": PE.identifiers(t),
                               "repository": row["repository"],
                               "commit": row.get("commit"),
                               "path": row["path"], "lines": [s, e],
                               "text": t[:ic.EMBED_MAX_CHARS]})
    return {"files": files, "chunks": chunks, "dropped": dropped}


def _embed_all(texts: list[str], is_query: bool, embed, beat) -> "object":
    import numpy as np
    ic = _ic()
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    out = []
    for i in range(0, len(texts), ic.BATCH):
        beat(f"knn: embedding {'queries' if is_query else 'documents'} "
             f"{i + 1}-{min(i + ic.BATCH, len(texts))}/{len(texts)}")
        batch = texts[i:i + ic.BATCH]
        if is_query:
            batch = [query_text(t) for t in batch]
        out.append(np.asarray(embed(batch, is_query=False), dtype=np.float32))
    mat = np.vstack(out)
    # PROTOCOL rule 2: zero vectors are an outage, never an index.
    if float(np.linalg.norm(mat, axis=1).min()) < 0.5:
        raise RuntimeError("the embedder returned zero vectors")
    return mat


def _signature(chunks: list[dict]) -> str:
    h = hashlib.sha256(KNN_VERSION.encode())
    for c in chunks:
        h.update(f"{c['id']}\0{c['group']}\0{','.join(c['labels'])}\0"
                 f"{hashlib.sha256(c['text'].encode()).hexdigest()}\1"
                 .encode())
    return h.hexdigest()[:16]


def build(beat=None, embed=None) -> dict:
    """(Re)build the ONE index over every onboarding's kept example files;
    choose k by leave-one-group-out. Returns {k, k_n, curve, groups, chunks,
    packages, signature, knn_version, ...}."""
    beat = beat or (lambda s: None)
    fn = _embed_fn(embed)
    inputs, n_man = inputs_signature()
    got = collect_rows(beat)
    files, chunks = got["files"], got["chunks"]
    groups = sorted({c["group"] for c in chunks} | {f["group"] for f in files})
    gidx = {g: i for i, g in enumerate(groups)}
    # Chunks sorted by group, so a group's chunks are one run (reduceat).
    chunks.sort(key=lambda c: (gidx[c["group"]], c["id"]))
    sig = _signature(chunks)
    D = _embed_all([c["text"] for c in chunks], False, fn, beat)
    Q = _embed_all([f["query"] for f in files], True, fn, beat)
    rec = {"knn_version": KNN_VERSION, "task": TASK,
           "task_wording": "unmeasured", "signature": sig, "inputs": inputs,
           "manifests": n_man, "files": len(files), "chunks": len(chunks),
           "groups": len({c["group"] for c in chunks}),
           "dropped": got["dropped"], "built_at": time.time(),
           "query_chars": QUERY_CHARS,
           "query_chars_source": "skill_match.rank_all's state[:4000]"}
    pk: dict[str, dict] = {}
    for c in chunks:
        for p in c["labels"]:
            e = pk.setdefault(p, {"chunks": 0, "groups": set()})
            e["chunks"] += 1
            e["groups"].add(c["group"])
    rec["packages"] = {p: {"chunks": e["chunks"], "groups": len(e["groups"])}
                       for p, e in sorted(pk.items())}
    if chunks and files:
        nb = Neighbours(D, [gidx[c["group"]] for c in chunks],
                        [tuple(c["labels"]) for c in chunks], Q,
                        [gidx[f["group"]] for f in files],
                        [tuple(f["labels"]) for f in files], groups)
        sel = logo(nb, range(len(groups)))
    else:
        sel = {"k": None, "n": 0, "folds": 0, "k_max": 0, "curve": [],
               "why": "no labelled chunk or no query file"}
    rec.update(k=sel["k"], k_n=sel["n"], k_folds=sel["folds"],
               k_max=sel["k_max"], curve=sel["curve"],
               k_why=sel.get("why") or "leave-one-group-out, top-1 package "
               "accuracy, ties -> the smaller k")
    _write(chunks, files, D, Q, rec, groups)
    _LOADED.update(key=None, index=None)
    return rec


def _write(chunks, files, D, Q, rec, groups) -> None:
    import numpy as np
    d = cache_dir()
    os.makedirs(d, exist_ok=True)

    def atomic(name, data: bytes):
        p = os.path.join(d, name)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
    rows = "".join(json.dumps({k: c[k] for k in (
        "id", "group", "file", "labels", "identifiers", "repository",
        "commit", "path", "lines")}, sort_keys=True) + "\n" for c in chunks)
    frows = "".join(json.dumps({k: f[k] for k in (
        "id", "group", "onboarding", "repository", "commit", "path",
        "labels")}, sort_keys=True) + "\n" for f in files)
    atomic(ROWS, rows.encode("utf-8"))
    atomic(FILES, frows.encode("utf-8"))
    rec = dict(rec, rows_sha256=hashlib.sha256(rows.encode()).hexdigest(),
               files_sha256=hashlib.sha256(frows.encode()).hexdigest())
    atomic(META, json.dumps(rec, indent=1, sort_keys=True,
                            default=str).encode("utf-8"))
    tmp = os.path.join(d, f"{NPZ}.{os.getpid()}.tmp.npz")
    np.savez(tmp, docs=D, queries=Q,
             chunk_ids=np.array([c["id"] for c in chunks], dtype=str),
             file_ids=np.array([f["id"] for f in files], dtype=str),
             groups=np.array(groups, dtype=str),
             signature=rec["signature"], inputs=rec["inputs"],
             rows_sha256=rec["rows_sha256"],
             files_sha256=rec["files_sha256"],
             k=-1 if rec["k"] is None else int(rec["k"]), n=len(chunks))
    os.replace(tmp, os.path.join(d, NPZ))


# ---------------------------------------------------------------------------
# Leave-one-group-out.
# ---------------------------------------------------------------------------
class Neighbours:
    """Every query file's groups in rank order (each group by its BEST
    chunk for that query), and the queries of each group. The ONLY way the
    k selection reads the index: ranked() never yields an excluded group
    and queries_of() is asked only for a fold's own group."""

    def __init__(self, D, chunk_group, chunk_labels, Q, file_group,
                 file_labels, groups):
        import numpy as np
        self.groups = list(groups)
        self.chunk_group = list(chunk_group)
        self.chunk_labels = list(chunk_labels)
        self.file_group = list(file_group)
        self.file_labels = list(file_labels)
        G = len(self.groups)
        cg = np.asarray(self.chunk_group, dtype=np.int64)
        # Chunks must be sorted by group (build sorts them).
        order = np.argsort(cg, kind="stable")
        if not np.array_equal(order, np.arange(len(cg))):
            raise ValueError("chunks must be sorted by group")
        present = sorted(set(self.chunk_group))
        starts = np.searchsorted(cg, present)
        lens = np.diff(np.append(starts, len(cg)))
        seg = np.repeat(np.arange(len(present)), lens)
        self._present = np.asarray(present, dtype=np.int64)
        self._pos = {g: j for j, g in enumerate(present)}
        F, P = len(self.file_group), len(present)
        Dm = np.asarray(D, dtype=np.float32)
        Qm = np.asarray(Q, dtype=np.float32)
        # Per query file: each present group's best chunk (its first
        # maximum) and the groups in rank order (best cosine, ties by group
        # order: a stable sort).
        self._arg = np.zeros((F, P), dtype=np.int64)
        self._order = np.zeros((F, P), dtype=np.int64)
        for f in range(F):
            if not P:
                break
            row = Dm @ Qm[f]
            best = np.maximum.reduceat(row, starts)
            hit = np.flatnonzero(row >= best[seg])
            hs = seg[hit]
            first = np.r_[True, hs[1:] != hs[:-1]]
            self._arg[f] = hit[first]
            self._order[f] = np.argsort(-best, kind="stable")
        self._queries: dict[int, list[int]] = {}
        for f, g in enumerate(self.file_group):
            self._queries.setdefault(g, []).append(f)
        self.package_groups: dict[str, set] = {}
        for c, g in enumerate(self.chunk_group):
            for p in self.chunk_labels[c]:
                self.package_groups.setdefault(p, set()).add(g)
        self.G = G
        self.file_ids: list[str] | None = None

    def queries_of(self, g: int) -> list[int]:
        return list(self._queries.get(g, []))

    def truth(self, f: int) -> tuple:
        return self.file_labels[f]

    def ranked(self, f: int, exclude):
        for j in self._order[f]:
            g = int(self._present[j])
            if g not in exclude:
                yield g

    def labels(self, f: int, g: int) -> tuple:
        return self.chunk_labels[int(self._arg[f, self._pos[g]])]

    def k_max(self, exclude) -> int:
        return max((len(gs - set(exclude)) for gs in
                    self.package_groups.values()), default=0)


def tally(nb: Neighbours, f: int, k: int, exclude) -> dict:
    """The vote of the top-k groups for query f: {groups: [{rank, group,
    packages}], tally: {package: votes}, order: [packages by votes, ties by
    best rank, then name], top: [every package at the argmax]}."""
    votes: dict[str, int] = {}
    best: dict[str, int] = {}
    gl = []
    for r, g in enumerate(nb.ranked(f, exclude), start=1):
        if r > k:
            break
        labs = nb.labels(f, g)
        gl.append({"rank": r, "group": nb.groups[g], "packages": list(labs)})
        for p in labs:
            votes[p] = votes.get(p, 0) + 1
            best.setdefault(p, r)
    order = sorted(votes, key=lambda p: (-votes[p], best[p], p))
    mx = max(votes.values(), default=0)
    return {"groups": gl, "tally": {p: votes[p] for p in order},
            "order": order, "top": [p for p in order if votes[p] == mx]}


def logo(nb: Neighbours, folds, *, exclude=frozenset()) -> dict:
    """Leave-one-group-out over `folds` (group indices not in `exclude`):
    each fold's labelled query files vote against every group except the
    fold and `exclude`; every k from 1 to the largest number of groups
    (outside `exclude`) any package labels. {k, n, folds, k_max, curve}.
    Ties in accuracy -> the smaller k."""
    exclude = frozenset(exclude)
    folds = [g for g in folds if g not in exclude]
    K = nb.k_max(exclude)
    if K < 1:
        return {"k": None, "n": 0, "folds": 0, "k_max": 0, "curve": [],
                "why": "no package labels any group outside the held-out "
                       "set"}
    correct = [0] * (K + 1)
    n = 0
    used = 0
    for h in folds:
        qs = [f for f in nb.queries_of(h) if nb.truth(f)]
        if not qs:
            continue
        used += 1
        ex = exclude | {h}
        for f in qs:
            truth = set(nb.truth(f))
            n += 1
            votes: dict[str, int] = {}
            best: dict[str, int] = {}
            tops: list[str] = []
            for g in nb.ranked(f, ex):
                if len(tops) >= K:
                    break
                r = len(tops) + 1
                for p in nb.labels(f, g):
                    votes[p] = votes.get(p, 0) + 1
                    best.setdefault(p, r)
                tops.append(min(votes, key=lambda p: (-votes[p], best[p], p)))
            # Fewer neighbours than K: a larger k is the same vote.
            last = tops[-1] if tops else None
            for kk in range(1, K + 1):
                t = tops[kk - 1] if kk <= len(tops) else last
                if t is not None and t in truth:
                    correct[kk] += 1
    if n == 0:
        return {"k": None, "n": 0, "folds": 0, "k_max": K, "curve": [],
                "why": "no fold has a labelled query file"}
    curve = [{"k": kk, "correct": correct[kk], "n": n,
              "accuracy": round(correct[kk] / n, 4)}
             for kk in range(1, K + 1)]
    best_k = max(curve, key=lambda c: (c["correct"], -c["k"]))["k"]
    return {"k": best_k, "n": n, "folds": used, "k_max": K, "curve": curve}


# ---------------------------------------------------------------------------
# Loading and voting (the selector's side).
# ---------------------------------------------------------------------------
_LOADED: dict = {"key": None, "index": None}


def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def load() -> dict | None:
    """The built index: {docs, queries, rows, files, groups, k, meta,
    signature}, or None when there is none (or its files disagree)."""
    import numpy as np
    p = _path(NPZ)
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (p, st.st_mtime_ns, st.st_size)
    if _LOADED["key"] == key:
        return _LOADED["index"]
    try:
        z = np.load(p, allow_pickle=False)
        with open(_path(ROWS), "rb") as f:
            rows_b = f.read()
        with open(_path(FILES), "rb") as f:
            files_b = f.read()
        if hashlib.sha256(rows_b).hexdigest() != str(z["rows_sha256"]) or \
                hashlib.sha256(files_b).hexdigest() != str(z["files_sha256"]):
            return None
        rows = [json.loads(x) for x in rows_b.decode().splitlines() if x]
        files = [json.loads(x) for x in files_b.decode().splitlines() if x]
        with open(_path(META), encoding="utf-8") as f:
            meta = json.load(f)
        idx = {"docs": z["docs"], "queries": z["queries"], "rows": rows,
               "files": files, "groups": [str(g) for g in z["groups"]],
               "k": None if int(z["k"]) < 0 else int(z["k"]),
               "signature": str(z["signature"]), "inputs": str(z["inputs"]),
               "meta": meta}
    except Exception:                                            # noqa: BLE001
        return None
    _LOADED.update(key=key, index=idx)
    return idx


def neighbours_of(idx: dict) -> Neighbours:
    """The Neighbours over a loaded index (for package_eval)."""
    gidx = {g: i for i, g in enumerate(idx["groups"])}
    nb = Neighbours(idx["docs"], [gidx[r["group"]] for r in idx["rows"]],
                      [tuple(r["labels"]) for r in idx["rows"]],
                      idx["queries"], [gidx[f["group"]] for f in idx["files"]],
                      [tuple(f["labels"]) for f in idx["files"]],
                      idx["groups"])
    nb.file_ids = [f["id"] for f in idx["files"]]
    return nb


def vote(code: str, *, embed=None) -> dict:
    """The vote for one piece of code: {index, k, groups: [{rank, group,
    packages}], tally, top, ok, why}. Rank only, no cosine cut."""
    import numpy as np
    out = {"index": None, "k": None, "groups": [], "tally": {}, "top": [],
           "ok": False, "why": None}
    idx = load()
    if idx is None:
        return dict(out, why="no example kNN index (built by "
                             f"{QUEUE} when the stack is idle)")
    out.update(index=idx["signature"], k=idx["k"])
    if idx["k"] is None:
        return dict(out, why="k was not measured for this index: "
                             + str(idx["meta"].get("k_why") or ""))
    if not idx["rows"]:
        return dict(out, why="the index holds no labelled chunk")
    q_code = query_of(code)
    if not q_code.strip():
        return dict(out, why="no code to query with")
    try:
        q = np.asarray(_embed_fn(embed)([query_text(q_code)],
                                        is_query=False),
                       dtype=np.float32)[0]
    except Exception as e:                                       # noqa: BLE001
        return dict(out, why=f"{type(e).__name__}: {e}"[:200])
    if float(np.linalg.norm(q)) < 0.5:
        return dict(out, why="the query vector is zero")
    sims = idx["docs"] @ q
    best: dict[str, tuple[float, int]] = {}
    for i, (r, s) in enumerate(zip(idx["rows"], sims)):
        g = r["group"]
        if g not in best or float(s) > best[g][0]:
            best[g] = (float(s), i)
    gorder = {g: i for i, g in enumerate(idx["groups"])}
    ranked = sorted(best, key=lambda g: (-best[g][0], gorder.get(g, 0)))
    votes: dict[str, int] = {}
    brank: dict[str, int] = {}
    for r, g in enumerate(ranked[:idx["k"]], start=1):
        row = idx["rows"][best[g][1]]
        labs = row["labels"]
        # The neighbour chunk's code-shaped identifiers (never its text):
        # the selector derives its skill labels from them (6.2, 6.5).
        out["groups"].append({"rank": r, "group": g, "packages": list(labs),
                              "identifiers": list(row.get("identifiers")
                                                  or [])})
        for p in labs:
            votes[p] = votes.get(p, 0) + 1
            brank.setdefault(p, r)
    order = sorted(votes, key=lambda p: (-votes[p], brank[p], p))
    mx = max(votes.values(), default=0)
    out.update(tally={p: votes[p] for p in order},
               top=[p for p in order if votes[p] == mx], ok=True,
               why=f"{len(out['groups'])} of k={idx['k']} groups voted")
    return out


# ---------------------------------------------------------------------------
# Skill labels, derived at load (6.2).
# ---------------------------------------------------------------------------
def topic_in(topic: str, ids) -> bool:
    """A skill's code-shaped topic is USED when every code-shaped name in it
    is among the identifiers (`useFrame()` -> useFrame; `world.query` ->
    its code-shaped parts)."""
    import re
    import skill_classify
    if not skill_classify.code_shaped(topic):
        return False
    parts = [p for p in re.findall(r"[A-Za-z_$][\w$]*", topic)
             if skill_classify.code_shaped(p)]
    return bool(parts) and all(p in ids for p in parts)


def skill_packages_of(s: dict) -> set[str]:
    """The held packages a skill is about: its package metadata, its area,
    the areas its description names (the selector's own reading)."""
    import skill_match
    import skill_select
    out = set()
    if s.get("package"):
        out.add(s["package"])
    areas = {skill_select.area_of(s.get("rule") or {})}
    try:
        areas |= set(skill_select.subject_areas(s))
    except Exception:                                            # noqa: BLE001
        pass
    for p, a in skill_match.PACKAGE_AREA.items():
        if a in areas:
            out.add(p)
    return out


def code_topics(s: dict) -> list[str]:
    import skill_classify
    return [t for t in skill_classify.gates(s.get("rule") or {})["topics"]
            if skill_classify.code_shaped(t)]


def derived_skills(packages, ids, pool: list[dict], *, pk_of=None
                   ) -> list[str]:
    """The skills a piece of example code is labelled with: the ARMED skills
    of its packages whose verified code-shaped topics it uses. A PROXY for
    "this skill would have helped", never a ground truth."""
    packages, ids = set(packages or ()), set(ids or ())
    out = []
    for s in pool:
        pks = (pk_of or {}).get(s["id"])
        if pks is None:
            pks = skill_packages_of(s)
        if not (pks & packages):
            continue
        if any(topic_in(t, ids) for t in code_topics(s)):
            out.append(s["id"])
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        r = build(beat=lambda s: None)
        print(json.dumps({k: v for k, v in r.items() if k != "curve"},
                         indent=1, default=str))
    else:
        st = index_state()
        print(st)
        sys.exit(0 if st["fresh"] else 1)
