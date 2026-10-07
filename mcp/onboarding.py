#!/usr/bin/env python
"""PACKAGE ONBOARDING: a prompt with links -> a held, tested package.

    python mcp/onboarding.py                  every onboarding and its stage
    python mcp/onboarding.py show <id>        one onboarding in full (JSON)

docs/PACKAGE-ONBOARDING.md is the design; this module is its stage handlers.
Operator, 2026-09-27: "all of this should also happen when I add a new skill
prompt on the skills dashboard. We should be able to automate all of this
from a prompt with some links." And: "That is a stateful pipeline in action,
we already have the durable machine too." So there is NO new orchestrator:
an onboarding is a dataset of kind `package` (mcp/datasets.py), its stages
are jobs on mcp/jobs.py's queue, and mcp/worker.py runs them and advances the
dataset the moment a stage's job is `done` (worker.advance_after), exactly as
it does a dataset or a skill version.

    submitted -> resolve -> clarify -> index -> vocab -> examples -> knn
       -> sources -> skills -> retire -> rebuild -> evaluate -> complete

| stage | lane | queue | does |
|---|---|---|---|
| resolve | net | package.resolve | links -> package@version at a commit, the licence from a verbatim quote (package_resolve); N packages -> N-1 sibling onboardings in one group |
| clarify | -- | -- | HUMAN only for a locator (never for a licence: the licence is provenance, recorded when a verbatim quote was found and as "not established" otherwise -- operator, 2026-10-07) |
| index | gpu, IDLE | package.index | the tarball verified against dist.integrity, unpacked, indexed with embeddings (deps.fetch_verified, deps.index_package); skipped when already indexed, healthy and embedded |
| vocab | cpu | package.vocab | the candidate vocabulary with the package; PROMOTED only when the standing detection labels do not get worse (operator decision 4), else HELD with the reason |
| examples | net | package.examples | the package's example code at the commit, labelled by its own imports (package_examples) |
| knn | gpu, IDLE | package.knn | the example kNN label index and its k by leave-one-group-out (example_knn) |
| sources | net | package.sources | the skill sources, tiers 1-3 and the lead fallback; one skill each through the ONE pipeline (package_sources) |
| skills | -- | JOIN | every skill the sources made has stopped (armed, quarantined, failed, decomposed); an errored skill job is a blocker with its remedy |
| retire | cpu | package.retire | what a REPLACE left behind: children whose section is gone, sources whose path is gone, the old version's vocabulary |
| rebuild | -- | JOIN | the skill document index and the example kNN index are fresh (their idle-gated upkeep jobs do the work) |
| evaluate | cpu | package.evaluate | the leave-one-group-out evaluation (package_eval) |

GPU stages wait for an IDLE stack (mcp/idle.py) without burning an attempt
(jobs.defer). A server's rate-limit answer defers the same way, for exactly
as long as the server said (package_net.RateLimited). Every stage is a
function of recorded inputs and writes its outputs atomically under
`index/packages/onboarding/<id>/` (YAMADORI_ONBOARDING_DIR), so a worker
killed mid-stage leaves the old output or none and jobs.reclaim() re-runs it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets  # noqa: E402
import jobs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KIND = "package"


def root() -> str:
    """Where the stage outputs live (read at call time, like jobs.DB, so a
    test's temp path applies)."""
    d = os.environ.get("YAMADORI_ONBOARDING_DIR")
    if d:
        return os.path.abspath(d)
    import deps
    return os.path.join(os.path.abspath(deps.STORE), "onboarding")


def out_dir(did: str) -> str:
    return os.path.join(root(), did)


def write_json(did: str, name: str, obj) -> str:
    p = os.path.join(out_dir(did), name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, p)
    return p


def read_json(did: str, name: str, default=None):
    try:
        with open(os.path.join(out_dir(did), name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _worker():
    import worker
    return worker


def _dataset(job: dict) -> dict:
    return _worker()._dataset(job)


def _merge_counts(did: str, new: dict) -> None:
    _worker()._merge_counts(did, new)


def resolution(did: str) -> dict:
    return read_json(did, "resolution.json", {}) or {}


def package_of(did: str) -> dict | None:
    return resolution(did).get("package")


def _deferred_on_rate_limit(fn):
    """A handler whose GET was told to wait by the server (package_net.
    RateLimited) is DEFERRED until the server's own reset time, without
    spending an attempt (worker.Deferred -> jobs.defer)."""
    def run(job, ctx):
        import package_net
        try:
            return fn(job, ctx)
        except package_net.RateLimited as e:
            raise _worker().Deferred(e.until, e.why) from e
    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    return run


def _unsupported(pkg: dict | None, stage: str) -> str | None:
    """Why this stage does not apply to this onboarding (recorded, not
    errored), or None."""
    if not pkg:
        return (f"no package: the prompt's links resolved to none, so "
                f"{stage} is skipped and its sources run as skill "
                "submissions")
    if pkg.get("ecosystem") != "npm":
        return (f"not supported for {pkg.get('ecosystem')}: "
                + (pkg.get("unsupported") or "JS/TS packages only"))
    return None


# ---------------------------------------------------------------------------
# Submission (the dashboard's PROMPT + LINKS form).
# ---------------------------------------------------------------------------
def submit(prompt: str, *, links=(), aliases=(), replaces: str | None = None,
           author: str = "operator") -> dict:
    """One onboarding for a prompt with links. Records the submission and
    enqueues `package.resolve`; nothing runs here."""
    prompt = (prompt or "").strip()
    links = [str(x).strip() for x in links or () if str(x).strip()]
    aliases = [str(x).strip() for x in aliases or () if str(x).strip()]
    if replaces not in (None, "", "replace", "alongside"):
        raise ValueError("replaces must be 'replace' or 'alongside'")
    import package_resolve
    if not package_resolve.links_of(prompt, links):
        raise ValueError("give a prompt with at least one link (npm, PyPI, "
                         "GitHub, a SKILL.md or a docs page)")
    notes = {"links": links, "aliases": aliases,
             "replaces": replaces or None, "author": author}
    ds = datasets.create(prompt, kind=KIND, notes=json.dumps(notes),
                         source_url=(links[0] if links else ""))
    return ds


# ---------------------------------------------------------------------------
# resolve (net)
# ---------------------------------------------------------------------------
@_deferred_on_rate_limit
def handle_resolve(job: dict, ctx) -> dict:
    import package_resolve
    ds = _dataset(job)
    did = ds["id"]
    notes = datasets.package_notes(ds)
    have = resolution(did)
    if have and not (job.get("payload") or {}).get("re_resolve"):
        # THE FIRST RESOLUTION PINS the version and the commit: a newer
        # `latest` must never silently move a running onboarding.
        return {"kept": "the first resolution pins the version and the "
                        "commit; re-resolve explicitly to move it",
                "package": _spec(have.get("package"))}
    ctx.beat("resolving the links")
    if notes.get("preset"):
        res = notes["preset"]
        how = f"resolved with its group ({notes.get('group')})"
    else:
        res = package_resolve.resolve(ds.get("prompt") or "",
                                      notes.get("links") or (),
                                      notes.get("aliases") or ())
        how = "resolved from the prompt's links"
    pkgs = res.get("packages") or []
    pkg = pkgs[0] if pkgs else None
    sources = [s for s in res.get("sources") or []
               if pkg is None or s.get("attach_to") in (None, pkg["name"])]
    held_versions = []
    if pkg:
        held_versions = _held_versions(pkg["name"])
    rec = {"prompt_sha256": hashlib.sha256((ds.get("prompt") or "")
                                           .encode("utf-8")).hexdigest(),
           "links": res.get("links") or notes.get("links") or [],
           "package": pkg, "sources": sources,
           "unresolved": res.get("unresolved") or [],
           "how": how, "resolved_at": time.time(),
           "group": notes.get("group"),
           "group_size": res.get("group_size") or len(pkgs),
           "replaces": (package_resolve.replaces_rule(
               pkg["name"], pkg["version"], held_versions,
               notes.get("replaces")) if pkg else None)}
    # N packages: N-1 siblings in one group, each resolved from this record.
    siblings = []
    if len(pkgs) > 1 and not notes.get("preset"):
        group = notes.get("group") or did
        rec["group"] = group
        if not notes.get("group"):
            datasets.answer(did, {"notes": json.dumps(dict(notes,
                                                           group=group))})
        for p in pkgs[1:]:
            sib_notes = {k: notes.get(k) for k in ("links", "aliases",
                                                   "replaces", "author")}
            sib_notes.update(group=group, preset={
                "packages": [p], "links": res.get("links"),
                "group_size": len(pkgs),
                "sources": [s for s in res.get("sources") or []
                            if s.get("attach_to") == p["name"]],
                "unresolved": []})
            sib = datasets.create(ds.get("prompt") or "", kind=KIND,
                                  notes=json.dumps(sib_notes),
                                  source_url=(p.get("links") or [None])[0]
                                  or ds.get("source_url") or "")
            siblings.append({"id": sib["id"], "package": _spec(p)})
    rec["siblings"] = siblings
    write_json(did, "resolution.json", rec)
    # The licence: from a verbatim quote, with where it was found.
    fields: dict = {}
    prov: dict = {}
    quotes = (pkg or {}).get("licence_quotes") or []
    write_json(did, "licence.json", {"quotes": quotes,
                                     "chosen": quotes[0] if quotes else None})
    if quotes:
        q = quotes[0]
        fields["licence"] = q["spdx"]
        prov["licence"] = {"provenance": "evidence", "quote": q["quote"],
                           "found_in": q.get("where")}
    if pkg:
        fields["name"] = _spec(pkg)
    if fields:
        datasets.answer(did, fields, provenance=prov,
                        meta={"resolution": {"package": _spec(pkg),
                                             "rules": _rules(pkg)}})
    _merge_counts(did, {"links": len(rec["links"]),
                        "packages": len(pkgs),
                        "sources_linked": len(sources),
                        "unresolved": len(rec["unresolved"])})
    return {"package": _spec(pkg), "rules": _rules(pkg),
            "licence": fields.get("licence"), "siblings": siblings,
            "unresolved": rec["unresolved"], "replaces": rec["replaces"]}


def _spec(pkg: dict | None) -> str | None:
    return f"{pkg['name']}@{pkg.get('version')}" if pkg else None


def _rules(pkg: dict | None) -> dict:
    if not pkg:
        return {}
    return {"version": pkg.get("version_rule"),
            "commit": pkg.get("commit_rule")}


def _held_versions(name: str) -> list[str]:
    out = set()
    try:
        import package_registry
        e = package_registry.entry(name) or {}
        out |= {str(v) for v in e.get("versions") or []}
    except Exception:                                            # noqa: BLE001
        pass
    try:
        import skill_packages
        out |= set(skill_packages.held_versions(name))
    except Exception:                                            # noqa: BLE001
        pass
    return sorted(out)


# ---------------------------------------------------------------------------
# index (gpu, idle-gated)
# ---------------------------------------------------------------------------
@_deferred_on_rate_limit
def handle_index(job: dict, ctx) -> dict:
    import deps
    ds = _dataset(job)
    did = ds["id"]
    pkg = package_of(did)
    why = _unsupported(pkg, "index")
    if why:
        write_json(did, "index.json", {"skipped": why})
        return {"skipped": why}
    name, v = pkg["name"], pkg["version"]
    try:
        import package_registry
        # From here on a fetched package is NOT held until its vocabulary
        # is promoted (docs/PACKAGE-ONBOARDING.md 5.3).
        package_registry.freeze_held()
    except ImportError:
        pass
    db = deps.db_path(name, v)
    if deps.is_indexed(name, v):
        h = deps.index_health(db, measure_bytes=False)
        emb = _meta_of(db).get("embedded")
        if h.get("ok") and emb == "1":
            rec = {"skipped": "already indexed, healthy and embedded",
                   "health": _health_public(h)}
            write_json(did, "index.json", rec)
            return rec
    ctx.beat(f"fetching {name}@{v} (verified against dist.integrity)")
    got = deps.fetch_verified(name, v, tarball=pkg.get("tarball"),
                              integrity=pkg.get("integrity"),
                              unpacked_size=pkg.get("unpacked_size"))
    fetch_rec = {k: got.get(k) for k in (
        "ok", "integrity", "integrity_check", "tarball_bytes",
        "tarball_sha256", "unpacked_bytes", "unpacked_size", "files",
        "licence_files", "reused", "error", "tarball")}
    if not got.get("ok"):
        write_json(did, "index.json", {"fetch": fetch_rec})
        raise _worker().Permanent(
            f"{name}@{v}: {got.get('error')}",
            "check the registry record; a tarball that does not match its "
            "own dist.integrity or unpackedSize is never indexed")
    _licence_from_tarball(did, got)
    ctx.beat(f"indexing {name}@{v} with embeddings (one subprocess)")
    h = deps.index_package(name, v, embed=True, src=got["dest"],
                           from_registry=True, log=lambda s: ctx.beat(s[:200]))
    rec = {"fetch": fetch_rec, "health": _health_public(h)}
    write_json(did, "index.json", rec)
    _merge_counts(did, {"index_chunks": h.get("chunks"),
                        "index_files": h.get("files_selected"),
                        "tarball_bytes": got.get("tarball_bytes"),
                        "unpacked_bytes": got.get("unpacked_bytes")})
    if not h.get("installed"):
        err = str(h.get("error") or "")
        if "embeddings" in err.lower() or "/v1/embeddings" in err:
            raise RuntimeError(_worker().describe(
                f"{name}@{v}: {err}", True, "check the embedding model on "
                "llama-swap, then let it retry", "operator"))
        raise _worker().Permanent(f"{name}@{v} was not installed: {err}",
                                  "read the index health in index.json")
    return rec


def _meta_of(db: str) -> dict:
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return dict(con.execute("SELECT k, v FROM meta").fetchall())
        finally:
            con.close()
    except sqlite3.Error:
        return {}


def _health_public(h: dict) -> dict:
    return {k: h.get(k) for k in ("ok", "installed", "chunks", "defs",
                                  "code_chunks", "zero_vectors", "norm_ok",
                                  "files_selected", "bytes_selected",
                                  "error")}


def _licence_from_tarball(did: str, got: dict) -> None:
    """The LICENSE file the tarball ships, as one more verbatim quote beside
    resolve's (docs/PACKAGE-ONBOARDING.md 2.2)."""
    import skill_pipeline
    rec = read_json(did, "licence.json", {}) or {}
    quotes = list(rec.get("quotes") or [])
    for rel in got.get("licence_files") or []:
        try:
            with open(os.path.join(got["dest"], rel), encoding="utf-8",
                      errors="replace") as f:
                lic = skill_pipeline.licence_of(f.read())
        except OSError:
            continue
        if lic and not any(q.get("kind") == "tarball" for q in quotes):
            quotes.append(dict(lic, where=f"{got.get('tarball')} {rel}",
                               kind="tarball"))
    rec["quotes"] = quotes
    spdx = {q.get("spdx") for q in quotes if q.get("spdx")}
    rec["agree"] = len(spdx) <= 1
    write_json(did, "licence.json", rec)


# ---------------------------------------------------------------------------
# vocab (cpu)
# ---------------------------------------------------------------------------
def handle_vocab(job: dict, ctx) -> dict:
    import deps
    import package_registry
    ds = _dataset(job)
    did = ds["id"]
    pkg = package_of(did)
    why = _unsupported(pkg, "vocab")
    if why:
        write_json(did, "vocab.json", {"state": "skipped", "why": why})
        return {"skipped": why}
    notes = datasets.package_notes(ds)
    name, v = pkg["name"], pkg["version"]
    held = set(package_registry.held_manifest() and
               package_registry.held_manifest().get("dirs") or [])
    add = deps.slug(name, v)
    lic = (read_json(did, "licence.json", {}) or {}).get("chosen")
    built_on = sorted(set(pkg.get("peer_dependencies") or []) & set(
        package_registry.load()))
    import package_resolve
    entry = package_registry.add_package(
        name, version=v, aliases=package_resolve.aliases_for(
            name, notes.get("aliases") or (),
            resolution(did).get("group_size") or 1),
        built_on=built_on,
        built_on_from=(f"package.json peerDependencies at "
                       f"{pkg.get('commit') or v}"),
        onboarding=did, licence=lic)
    have = read_json(did, "vocab.json", {}) or {}
    if have.get("state") in ("promoted", "forced") and add in held:
        return {"kept": f"already {have['state']}", "state": have["state"]}
    ctx.beat(f"building the candidate vocabulary with {name}@{v}")
    cand = package_registry.build_candidate(add_dirs=[add], entries=[entry])
    fl = package_registry.floor(cand)
    rec = {"state": None, "package": _spec(pkg), "add": add,
           "entry": entry, "diff": cand.get("diff"), "floor": fl,
           "signature": cand.get("signature"), "at": time.time()}
    if fl.get("passed"):
        package_registry.promote(cand, by=f"pipeline:onboarding:{did}")
        rec["state"] = "promoted"
    else:
        rec["state"] = "held"
        rec["why"] = _held_why(fl, cand.get("diff") or {})
    write_json(did, "vocab.json", rec)
    per = ((cand.get("diff") or {}).get("per_package") or {}).get(name) or {}
    _merge_counts(did, {"vocab_state": rec["state"],
                        "vocab_unique": per.get("unique"),
                        "vocab_code_only": per.get("code_only")})
    return {"state": rec["state"], "why": rec.get("why"),
            "unique": per.get("unique"), "floor": {
                k: fl.get(k) for k in ("passed", "rows")}}


def _held_why(fl: dict, diff: dict) -> str:
    bits = []
    if fl.get("lost_tp"):
        bits.append("loses " + ", ".join(f"{x['package']} on {x['id']}"
                                         for x in fl["lost_tp"][:6]))
    if fl.get("gained_fp"):
        bits.append("gains " + ", ".join(f"{x['package']} on {x['id']}"
                                         for x in fl["gained_fp"][:6]))
    lost = {p: len(n) for p, n in (diff.get("lost") or {}).items() if n}
    if lost:
        bits.append("names existing packages lose: " + ", ".join(
            f"{p} {n}" for p, n in sorted(lost.items())))
    return ("the standing detection labels get worse: " + "; ".join(bits)
            + " (operator decision 4: held; the operator may force it)")


def promote(did: str, *, author: str) -> dict:
    """The operator forces a HELD vocabulary (docs/PACKAGE-ONBOARDING.md
    5.3), recorded with the author."""
    import package_registry
    rec = read_json(did, "vocab.json", {}) or {}
    if rec.get("state") != "held":
        raise ValueError(f"the vocabulary of {did} is "
                         f"{rec.get('state') or 'not built'}, not held")
    cand = package_registry.next_candidate()
    if not cand or cand.get("signature") != rec.get("signature"):
        raise ValueError("the held candidate is no longer the newest one "
                         "built (another onboarding built after it): re-run "
                         "the vocab stage")
    package_registry.promote(cand, by=author, forced=True,
                             reason=rec.get("why") or "")
    rec.update(state="forced", forced_by=author, forced_at=time.time())
    write_json(did, "vocab.json", rec)
    _merge_counts(did, {"vocab_state": "forced"})
    sweep()
    return rec


# ---------------------------------------------------------------------------
# examples (net) and knn (gpu, idle-gated): package_examples, example_knn
# ---------------------------------------------------------------------------
@_deferred_on_rate_limit
def handle_examples(job: dict, ctx) -> dict:
    import package_examples
    ds = _dataset(job)
    did = ds["id"]
    pkg = package_of(did)
    why = _unsupported(pkg, "examples")
    if why:
        write_json(did, "examples.json", {"skipped": why})
        return {"skipped": why}
    t = tree_of(did)
    linked = (read_json(did, "sources.json", {}) or {}).get(
        "examples_dirs") or linked_dirs(did)
    rec = package_examples.collect(did, pkg, t, linked_dirs=linked,
                                   beat=ctx.beat)
    _merge_counts(did, {k: rec.get(k) for k in
                        ("example_groups", "example_files", "example_chunks",
                         "example_dropped") if k in rec})
    return {k: v for k, v in rec.items() if not isinstance(v, (list, dict))
            or k in ("per_package",)}


def handle_knn(job: dict, ctx) -> dict:
    import example_knn
    ds = _dataset(job)
    did = ds["id"]
    pkg = package_of(did)
    why = _unsupported(pkg, "knn")
    if why:
        write_json(did, "knn.json", {"skipped": why})
        return {"skipped": why}
    rec = example_knn.build(beat=ctx.beat)
    write_json(did, "knn.json", rec)
    _merge_counts(did, {"knn_k": rec.get("k"),
                        "knn_groups": rec.get("groups")})
    return {k: v for k, v in rec.items() if k != "curve"}


def tree_of(did: str) -> dict | None:
    """The repository's tree at the resolved commit, read once per
    onboarding and kept (tree.json): sources and examples read the same
    listing."""
    import package_sources
    t = read_json(did, "tree.json")
    if t is not None:
        return t
    pkg = package_of(did) or {}
    gh_ = pkg.get("repository") or {}
    if not (gh_ and pkg.get("commit")):
        return None
    t = package_sources.tree(gh_["owner"], gh_["repo"], pkg["commit"])
    write_json(did, "tree.json", t)
    return t


def linked_dirs(did: str) -> list[str]:
    return [s.get("path") for s in resolution(did).get("sources") or []
            if s.get("kind") == "github_path" and s.get("is_dir")]


# ---------------------------------------------------------------------------
# sources (net)
# ---------------------------------------------------------------------------
@_deferred_on_rate_limit
def handle_sources(job: dict, ctx) -> dict:
    import package_sources
    ds = _dataset(job)
    did = ds["id"]
    res = resolution(did)
    pkg = res.get("package")
    t = tree_of(did) if pkg else None
    local = None
    if pkg and not pkg.get("commit") and pkg.get("ecosystem") == "npm":
        import deps
        d = os.path.join(os.path.abspath(deps.SRC_CACHE),
                         deps.slug(pkg["name"], pkg["version"]))
        local = d if os.path.isdir(d) else None
    ctx.beat("discovering the skill sources")
    found = package_sources.discover(res, t, local_src=local)
    lic = (read_json(did, "licence.json", {}) or {}).get("chosen")
    made = package_sources.make_skills(
        did, found, package=(pkg or {}).get("name"),
        version=(pkg or {}).get("version"), prompt=ds.get("prompt") or "",
        replaces=res.get("replaces") or {}, package_licence=lic,
        author=f"pipeline:onboarding:{did}")
    rec = dict(found, made=made, at=time.time())
    old = read_json(did, "sources.json", {}) or {}
    if old.get("made"):
        # A re-run adds to what the first run made (make_skills skipped
        # the sources it already created).
        for k in ("created", "repointed", "unchanged"):
            made[k] = list(old["made"].get(k) or []) + [
                x for x in made.get(k) or [] if x not in (old["made"].get(k)
                                                        or [])]
    write_json(did, "sources.json", rec)
    _merge_counts(did, {"sources": len(found.get("chosen") or []),
                        "skills_created": len(made["created"]),
                        "skills_repointed": len(made["repointed"]),
                        "sources_failed": len(made["failed"])})
    if made["failed"] and not (made["created"] or made["repointed"]
                               or made["existing"] or made["unchanged"]):
        raise RuntimeError(_worker().describe(
            f"no source could be made into a skill: {made['failed'][0]}",
            True, "retried automatically; read sources.json", "worker"))
    return {"chosen": len(found.get("chosen") or []),
            "created": len(made["created"]),
            "repointed": len(made["repointed"]),
            "unchanged": len(made["unchanged"]),
            "failed": made["failed"][:5],
            "tier3": {"pages": len(found["tier3"].get("pages") or []),
                      "ingested": bool(found["tier3"].get("ingest")
                                       and found["tier3"].get("pages"))}}


def tier3(did: str, *, author: str) -> dict:
    """Ingest the recorded llms.txt page list (the operator's click;
    docs/PACKAGE-ONBOARDING.md 4.1)."""
    import package_sources
    rec = read_json(did, "sources.json", {}) or {}
    pages = (rec.get("tier3") or {}).get("pages") or []
    if not pages:
        raise ValueError(f"{did} recorded no llms.txt pages")
    ds = datasets.get(did) or {}
    pkg = package_of(did) or {}
    found = {"chosen": [{"tier": 3, "url": p["url"], "frontier": False,
                         "why": f"listed by {rec['tier3'].get('where')}; "
                                f"ingested by {author}"} for p in pages]}
    made = package_sources.make_skills(
        did, found, package=pkg.get("name"), version=pkg.get("version"),
        prompt=ds.get("prompt") or "", replaces={}, package_licence=None,
        author=author)
    rec.setdefault("tier3", {})["ingested_by"] = author
    rec["tier3"]["ingested_at"] = time.time()
    rec.setdefault("made", {}).setdefault("created", []).extend(
        made["created"])
    write_json(did, "sources.json", rec)
    if ds.get("stage") in ("retire", "rebuild", "evaluate", "complete"):
        # The new skills are this onboarding's; its skills JOIN is past, so
        # the evaluation is re-run once they stop (the next evaluate job).
        rec["tier3"]["note"] = ("ingested after the skills stage: re-run "
                                "the evaluate stage once they arm")
        write_json(did, "sources.json", rec)
    return {"created": made["created"], "failed": made["failed"]}


# ---------------------------------------------------------------------------
# The skills JOIN.
# ---------------------------------------------------------------------------
def skills_of(did: str) -> list[dict]:
    import skills
    return [s for s in skills.listing()
            if (s.get("meta") or {}).get("onboarding") == did]


def _skill_waiting(s: dict) -> dict | None:
    """Why this onboarding skill has not stopped, or None when it has."""
    import skills
    latest = skills.version(s["id"], s.get("latest_version")) or {}
    if latest.get("state") == "running":
        return {"what": f"skill {s['name']} ({s['id']}) v{latest['version']} "
                        f"is at {latest.get('stage')}",
                "retryable": True, "owner": "worker",
                "remedy": "wait for mcp/worker.py; its gpu stages wait for "
                          "an idle stack"}
    return None


def _skill_errored(s: dict) -> list[dict]:
    rows = jobs.listing(dataset=f"skill:{s['id']}", state="errored",
                        limit=20)
    out = []
    import skills
    latest = skills.version(s["id"], s.get("latest_version")) or {}
    for j in rows:
        p = j.get("payload") or {}
        if int(p.get("version") or 0) != int(latest.get("version") or -1):
            continue
        out.append({"what": f"skill {s['name']} ({s['id']}): {j['queue']} job "
                            f"{j['id']} is errored",
                    "why": (j.get("error") or "")[:400], "retryable": True,
                    "owner": "operator",
                    "remedy": "fix the cause, then re-run the stage "
                              f"(POST /dash/api/skill/rerun {{id: "
                              f"{s['id']}, stage: {p.get('stage')}}}) or the "
                              f"job (POST /dash/api/dataset/rerun {{job: "
                              f"{j['id']}}})"})
    return out


# ---------------------------------------------------------------------------
# retire (cpu)
# ---------------------------------------------------------------------------
def handle_retire(job: dict, ctx) -> dict:
    import skills
    ds = _dataset(job)
    did = ds["id"]
    res = resolution(did)
    author = f"pipeline:onboarding:{did}"
    archived, kept = [], []
    todo: list[dict] = []
    for s in skills_of(did):
        for x in (s.get("meta") or {}).get("to_retire") or []:
            todo.append(dict(x, source=s["id"]))
    for x in ((read_json(did, "sources.json", {}) or {}).get("made") or {}
              ).get("to_retire") or []:
        todo.append(x)
    for x in todo:
        s = skills.get(x["id"])
        if s is None:
            continue
        if s.get("status") == "archived":
            kept.append({"id": x["id"], "why": "already archived"})
            continue
        skills.archive(x["id"], reason=f"retired by onboarding {did}: "
                       f"{x.get('why')}", author=author)
        archived.append({"id": x["id"], "name": s.get("name"),
                         "why": x.get("why")})
    vocab = None
    rep = res.get("replaces") or {}
    pkg = res.get("package") or {}
    if rep.get("mode") == "replace" and rep.get("old") and \
            not _unsupported(pkg, "retire"):
        vocab = _retire_old_versions(did, pkg, rep["old"], author)
    rec = {"archived": archived, "kept": kept, "vocab": vocab,
           "at": time.time()}
    write_json(did, "retire.json", rec)
    _merge_counts(did, {"retired": len(archived)})
    return {"archived": len(archived), "vocab": (vocab or {}).get("state")}


def _retire_old_versions(did: str, pkg: dict, old: list, author: str) -> dict:
    """REPLACE: the old version leaves the held set only now, after the new
    version's skills stopped -- through the same promotion floor."""
    import deps
    import package_registry
    held = set((package_registry.held_manifest() or {}).get("dirs") or [])
    drop = [deps.slug(pkg["name"], v) for v in old
            if deps.slug(pkg["name"], v) in held]
    if not drop:
        return {"state": "nothing held", "old": old}
    cand = package_registry.build_candidate(remove_dirs=drop)
    fl = package_registry.floor(cand)
    if fl.get("passed"):
        package_registry.promote(cand, by=author)
        return {"state": "removed", "dirs": drop, "floor": fl}
    return {"state": "kept", "dirs": drop, "floor": fl,
            "why": "removing the old version's vocabulary makes the standing "
                   "labels worse; it stays held (operator decision 4)"}


# ---------------------------------------------------------------------------
# evaluate (cpu)
# ---------------------------------------------------------------------------
def handle_evaluate(job: dict, ctx) -> dict:
    import package_eval
    ds = _dataset(job)
    did = ds["id"]
    pkg = package_of(did)
    rec = package_eval.run(did, pkg, beat=ctx.beat)
    _merge_counts(did, {"eval_key": rec.get("key"),
                        "eval_in_sample": rec.get("in_sample")})
    return {k: rec.get(k) for k in ("key", "in_sample", "summary",
                                    "skipped") if k in rec}


HANDLERS = {
    "package.resolve": handle_resolve,
    "package.index": handle_index,
    "package.vocab": handle_vocab,
    "package.examples": handle_examples,
    "package.knn": handle_knn,
    "package.sources": handle_sources,
    "package.retire": handle_retire,
    "package.evaluate": handle_evaluate,
}


def _register_knn_upkeep() -> None:
    """The example kNN upkeep job, and the evaluation's model-decider arm
    (a gpu job, idle-gated; package_eval.decider_arm enqueues it only when
    YAMADORI_SKILL_DECIDER names a decider other than the stub)."""
    try:
        import example_knn
        HANDLERS[example_knn.QUEUE] = example_knn.handle_build
    except ImportError:
        pass
    try:
        import package_eval
        HANDLERS[package_eval.DECIDER_QUEUE] = package_eval.handle_decider
    except (ImportError, AttributeError):
        pass


_register_knn_upkeep()


# ---------------------------------------------------------------------------
# Blockers, the JOINs and the sweep.
# ---------------------------------------------------------------------------
def blockers(ds: dict) -> list[dict]:
    """Why this onboarding cannot leave its stage. Empty means it can."""
    stage = ds.get("stage")
    did = ds["id"]
    if stage in datasets.PACKAGE_ENQUEUE:
        out = [datasets._job_blocker(j)
               for j in datasets._unfinished(did, stage=stage)]
        if not out and stage == "vocab":
            v = read_json(did, "vocab.json", {}) or {}
            if v.get("state") == "held":
                out.append({"what": "the vocabulary is HELD: "
                                    + (v.get("why") or ""),
                            "retryable": False, "owner": "operator",
                            "remedy": "read the diff on the onboarding page; "
                                      "force it with POST /dash/api/skill-"
                                      "factory/onboarding/promote {id}, or "
                                      "fix the labels or aliases and re-run "
                                      "the vocab stage"})
        return out
    if stage == "clarify":
        # The licence never blocks (operator, 2026-10-07): it is provenance,
        # recorded when a verbatim quote was found and as "not established"
        # otherwise; the onboarding asks only for a locator.
        return [m for m in datasets.missing(ds) if m["field"] != "licence"]
    if stage == "skills":
        out = []
        for s in skills_of(did):
            w = _skill_waiting(s)
            if w:
                out.append(w)
            out += _skill_errored(s)
        return out
    if stage == "rebuild":
        out = []
        try:
            import skill_match
            st = skill_match.index_state()
            if not st.get("fresh"):
                out.append({"what": f"the skill document index is not fresh "
                                    f"({st.get('why')})", "retryable": True,
                            "owner": "worker",
                            "remedy": "its upkeep job (skill.match_index) "
                                      "rebuilds it when the stack is idle"})
        except Exception as e:                                   # noqa: BLE001
            out.append({"what": f"the skill document index cannot be read: "
                                f"{type(e).__name__}: {e}"[:300],
                        "retryable": True, "owner": "worker",
                        "remedy": "check the skill store"})
        try:
            import example_knn
            st = example_knn.index_state()
            if not st.get("fresh"):
                out.append({"what": f"the example kNN index is not fresh "
                                    f"({st.get('why')})", "retryable": True,
                            "owner": "worker",
                            "remedy": "its upkeep job (package.example_knn_"
                                      "index) rebuilds it when the stack is "
                                      "idle"})
        except ImportError:
            pass
        return out
    return []


def held_for_person(ds: dict) -> str | None:
    """Nothing holds an onboarding for its licence any more (operator,
    2026-10-07: the licence is provenance only, "WE DONT NEED TO FUCKING
    LICENSE TEXT THAT WE INJECT IT IS FAIR USE"). Kept for the callers that
    ask; always None."""
    return None


def after_job(job: dict) -> str | None:
    """The worker's hook for a package dataset: advance past the stage whose
    job just finished, then on through the jobless stages (clarify, the
    JOINS) while nothing blocks them. Returns the stage it ended at."""
    ds = datasets.get(job.get("dataset") or "")
    if ds is None or ds.get("kind") != KIND:
        return None
    if ds["stage"] != job.get("stage"):
        return None
    return _advance_through(ds["id"])


def _advance_through(did: str) -> str | None:
    moved = None
    for _ in range(len(datasets.PACKAGE_STAGES)):
        ds = datasets.get(did)
        if ds is None or ds["stage"] == "complete":
            break
        if ds["stage"] == "clarify":
            held = held_for_person(ds)
            if held:
                if (ds.get("assist") or {}).get("held") != held:
                    datasets.answer(did, {}, meta={"held": held})
                break
        try:
            ds = datasets.advance(did)
        except datasets.Blocked:
            break
        moved = ds["stage"]
        if moved in datasets.PACKAGE_ENQUEUE:
            break                 # its job runs next; the worker takes it
    return moved


def sweep() -> list[str]:
    """Advance every onboarding whose jobless stage (clarify after an
    answer, a JOIN, a forced vocabulary) has cleared. The worker calls it
    once a minute (onboarding.tick)."""
    lines = []
    for ds in datasets.listing(limit=1000):
        if ds.get("kind") != KIND or ds.get("stage") == "complete":
            continue
        st = ds["stage"]
        if st in datasets.PACKAGE_ENQUEUE:
            # Only a stage whose job is done and whose blocker cleared since
            # (a forced vocabulary).
            if datasets._unfinished(ds["id"], stage=st) or \
                    not [j for j in jobs.listing(dataset=ds["id"], limit=50)
                         if j.get("stage") == st and j["state"] == "done"]:
                continue
        moved = _advance_through(ds["id"])
        if moved:
            lines.append(f"onboarding {ds['id']}: {st} -> {moved}")
    return lines


def tick() -> list[str]:
    """The worker's minute: the example kNN upkeep, then the sweep."""
    out = []
    try:
        import example_knn
        jid = example_knn.schedule()
        if jid:
            out.append(f"enqueued {example_knn.QUEUE} {jid}")
    except ImportError:
        pass
    return out + sweep()


# ---------------------------------------------------------------------------
# The dashboard's views (docs/PACKAGE-ONBOARDING.md 8.1). No filesystem path
# in any of them.
# ---------------------------------------------------------------------------
def summary(ds: dict) -> dict:
    res = resolution(ds["id"])
    pkg = res.get("package") or {}
    jc = datasets.job_counts(ds["id"])
    state = ("errored" if jc.get("errored") else
             "complete" if ds["stage"] == "complete" else
             "running" if jc.get("running") else
             "waiting" if (ds["stage"] in datasets.PACKAGE_JOINS
                           or ds["stage"] == "clarify") else "queued")
    return {"id": ds["id"], "group": res.get("group") or
            datasets.package_notes(ds).get("group"),
            "package": pkg.get("name"), "version": pkg.get("version"),
            "stage": ds["stage"], "state": state, "updated": ds["updated"],
            "created": ds["created"], "counts": ds.get("counts") or {},
            "jobs": jc}


def listing() -> list[dict]:
    return [summary(ds) for ds in datasets.listing(limit=1000)
            if ds.get("kind") == KIND]


def _waiting_reason(did: str) -> str | None:
    for j in jobs.listing(dataset=did, state="queued", limit=20):
        if j.get("not_before") and j["not_before"] > time.time():
            return j.get("progress") or "waiting"
    return None


def detail(did: str) -> dict | None:
    ds = datasets.get(did)
    if ds is None or ds.get("kind") != KIND:
        return None
    import skills
    out = summary(ds)
    rows = datasets.dataset_jobs(did)
    out.update({
        "prompt": ds.get("prompt"), "name": ds.get("name"),
        "licence_value": ds.get("licence"),
        "stages": list(datasets.PACKAGE_STAGES),
        "joins": list(datasets.PACKAGE_JOINS),
        "job_rows": [dict(j, result=(jobs.get(j["id"]) or {}).get("result"))
                     for j in rows],
        "blockers": blockers(ds), "missing": datasets.missing(ds),
        "warnings": datasets.warnings(ds),
        "next_stage": datasets.next_stage(ds),
        "field_states": datasets.field_states(ds),
        "held": held_for_person(ds) or (ds.get("assist") or {}).get("held"),
        "waiting": _waiting_reason(did),
        "notes": {k: v for k, v in datasets.package_notes(ds).items()
                  if k != "preset"},
        "resolution": resolution(did) or None,
        "licence": read_json(did, "licence.json"),
        "index": read_json(did, "index.json"),
        "vocab": read_json(did, "vocab.json"),
        "examples": read_json(did, "examples.json"),
        "knn": read_json(did, "knn.json"),
        "sources": _public_sources(read_json(did, "sources.json")),
        "retire": read_json(did, "retire.json"),
        "eval": _latest_eval(did),
        "reviews": (ds.get("assist") or {}).get("reviews") or [],
    })
    kids = []
    for s in skills_of(did):
        kids.append({"id": s["id"], "name": s["name"], "status": s["status"],
                     "version": s.get("served_version"),
                     "latest_version": s.get("latest_version"),
                     "reason": s.get("reason"),
                     "lead_for": _lead_for(s),
                     "parent": ((s.get("meta") or {}).get("parent") or {})
                     .get("skill"),
                     "source_url": s.get("source_url")})
    out["skills"] = kids
    out["skill_counts"] = {st: sum(1 for k in kids if k["status"] == st)
                           for st in skills.STATUSES}
    return out


def _lead_for(s: dict) -> str | None:
    import skills
    ver = skills.version(s["id"], s.get("served_version")) \
        if s.get("served_version") else None
    txt = (ver or {}).get("text") or ""
    if "lead_for" not in txt:
        return None
    import skill_md
    return ((skill_md.parse(txt) or {}).get("yamadori") or {}).get("lead_for")


def _public_sources(rec: dict | None) -> dict | None:
    if not rec:
        return rec
    rec = dict(rec)
    rec["chosen"] = [{k: v for k, v in c.items() if k != "local"}
                     for c in rec.get("chosen") or []]
    return rec


def _latest_eval(did: str) -> dict | None:
    d = os.path.join(out_dir(did), "eval")
    try:
        names = [n for n in os.listdir(d) if n.endswith(".json")]
    except OSError:
        return None
    if not names:
        return None
    names.sort(key=lambda n: os.path.getmtime(os.path.join(d, n)))
    return read_json(did, os.path.join("eval", names[-1]))


def review(did: str, stage: str, note: str, *, author: str) -> dict:
    """An after-the-fact note on one stage (no review GATE: operator,
    2026-09-24); stored in the dataset's `assist.reviews`, changes nothing
    else."""
    ds = datasets.get(did)
    if ds is None or ds.get("kind") != KIND:
        raise KeyError(f"no such onboarding: {did}")
    if stage not in datasets.PACKAGE_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    note = (note or "").strip()
    if not note:
        raise ValueError("an empty note")
    reviews = list((ds.get("assist") or {}).get("reviews") or [])
    reviews.append({"stage": stage, "note": note[:4000], "by": author,
                    "at": time.time()})
    datasets.answer(did, {}, meta={"reviews": reviews})
    return {"reviews": reviews}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "show":
        print(json.dumps(detail(sys.argv[2]), indent=1, default=str))
    else:
        for r in listing():
            print(f"  {r['id']}  {r['stage']:<9} {r['state']:<8} "
                  f"{r['package']}@{r['version']}")
