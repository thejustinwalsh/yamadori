#!/usr/bin/env python
"""A PACKAGE'S SKILL SOURCES, discovered at the pinned commit, and the skills
made from them through the ONE skills pipeline.

docs/PACKAGE-ONBOARDING.md section 4 (the onboarding's `sources` stage). At
the resolved commit -- or from what the tarball ships when there is no tag
for the version -- in tiers:

| tier | source | path through the pipeline |
|---|---|---|
| 1 | every `SKILL.md` in the repository (the `skills/<x>/SKILL.md` convention: koota, pmndrs/math) and the `references/*.md` beside it | the SKILL.md: frontier (decompose, with `meta.package` so its first part yields the LEAD); each reference: distil |
| 2 | every page the PROMPT links (a SKILL.md or a folder holding one -> frontier; any other page -> distil; a linked folder with no SKILL.md is an examples directory, for the examples stage) | the path its URL implies |
| 3 | `llms.txt` at the repository root, or at the docs site the manifest's `homepage` names: the pages it lists | distil, one skill per page -- ONLY when tier 1 found no SKILL.md (operator, 2026-09-27, decision 3: "llms.txt-listed pages are pulled in only when the package ships no SKILL.md"); otherwise the list is RECORDED and `tier3()` ingests it on the operator's click |
| lead | the README at the commit, when tier 1 found no SKILL.md | frontier with `meta.package` (decompose/4 writes ONE compact lead from the opening sections) |

Each source becomes `skills.create(url=<raw URL at the commit>, goal=<the
prompt's prose>, frontier=..., watch_hours=0, meta={"onboarding", "package",
"package_version"})`: a commit-pinned URL cannot change, and a new VERSION of
the package is a new onboarding. A source already created for this
onboarding (same URL) is not created twice.

A NEW VERSION of a held package (operator, 2026-09-27, decision 2): REPLACE
(the same major) gives the earlier onboarding's skill for the same
repository path a NEW VERSION at the new commit (`skills.repoint`, then
`new_version`, `store_source`, `enqueue`) -- it serves its old version until
the new one arms; an earlier source whose path is gone is recorded
`to_retire`, for the retire stage. ALONGSIDE (a new major) creates new
skills and leaves the old ones armed (the Right Family, Wrong Skill risk is
in the resolution record).

The tree listing is GitHub's (GET, unauthenticated; package_net): one
recursive tree at the commit. A TRUNCATED tree (GitHub truncates very large
ones) is walked non-recursively at the top and recursively below the
directories the sources and examples stages read only (`WALK_BELOW`); what
was walked is recorded, so a miss is visible.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import package_net as net  # noqa: E402

# Directory names whose subtrees are walked when the recursive tree is
# truncated: where SKILL.md files, docs and examples live. The examples
# names are package_examples.EXAMPLE_DIRS (a starting convention the design
# names, docs/PACKAGE-ONBOARDING.md 6.1).
DOC_DIRS = ("skills", "docs", "doc", "documentation")
_LINK = re.compile(r"\[([^\]\n]{0,200})\]\((https?://[^)\s]+)\)")


# ---------------------------------------------------------------------------
# The tree at a commit.
# ---------------------------------------------------------------------------
def walk_below() -> tuple:
    import package_examples
    return DOC_DIRS + tuple(package_examples.EXAMPLE_DIRS)


def tree(owner: str, repo: str, commit: str) -> dict:
    """{entries: [{path, type, sha, size}], truncated, walked, gets}."""
    doc, r = net.gh(f"repos/{owner}/{repo}/git/trees/{commit}?recursive=1")
    if not isinstance(doc, dict) or not isinstance(doc.get("tree"), list):
        return {"entries": [], "truncated": False, "walked": [],
                "error": f"the tree of {owner}/{repo}@{commit[:12]} could not "
                         f"be read (HTTP {r.status})"}
    entries = [{k: e.get(k) for k in ("path", "type", "sha", "size")}
               for e in doc["tree"] if isinstance(e, dict)]
    out = {"entries": entries, "truncated": bool(doc.get("truncated")),
           "walked": []}
    if not out["truncated"]:
        return out
    # Truncated: the top level, then each directory the stages read, whole.
    top, _ = net.gh(f"repos/{owner}/{repo}/git/trees/{commit}")
    rows = [e for e in (top or {}).get("tree") or [] if isinstance(e, dict)]
    have = {e["path"] for e in entries}
    for e in rows:
        if e.get("path") not in have:
            entries.append({k: e.get(k) for k in ("path", "type", "sha",
                                                  "size")})
    names = {n.lower() for n in walk_below()}
    for e in rows:
        if e.get("type") != "tree" or str(e.get("path")).lower() not in names:
            continue
        sub, _ = net.gh(f"repos/{owner}/{repo}/git/trees/{e['sha']}"
                        "?recursive=1")
        got = [x for x in (sub or {}).get("tree") or [] if isinstance(x, dict)]
        for x in got:
            p = f"{e['path']}/{x.get('path')}"
            if p not in have:
                entries.append({"path": p, "type": x.get("type"),
                                "sha": x.get("sha"), "size": x.get("size")})
                have.add(p)
        out["walked"].append({"dir": e["path"], "entries": len(got),
                              "truncated": bool((sub or {}).get("truncated"))})
    return out


def files(t: dict) -> dict[str, dict]:
    return {e["path"]: e for e in t.get("entries") or []
            if e.get("type") == "blob"}


# ---------------------------------------------------------------------------
# Discovery.
# ---------------------------------------------------------------------------
def _in_scope(path: str, directory: str | None) -> bool:
    """A monorepo package (`repository.directory`): its own directory, or a
    top-level skills/ folder (where koota and pmndrs/math keep theirs)."""
    if not directory:
        return True
    d = directory.strip("/") + "/"
    return path.startswith(d) or path.lower().startswith("skills/")


def skill_mds(t: dict, directory: str | None = None) -> list[str]:
    return sorted(p for p in files(t) if p.rsplit("/", 1)[-1] == "SKILL.md"
                  and _in_scope(p, directory))


def references_of(t: dict, skill_md_path: str) -> list[str]:
    base = skill_md_path.rsplit("/", 1)[0] if "/" in skill_md_path else ""
    ref = f"{base}/references/" if base else "references/"
    return sorted(p for p in files(t) if p.startswith(ref)
                  and p.lower().endswith((".md", ".markdown", ".mdx"))
                  and "/" not in p[len(ref):])


def readme_of(t: dict, directory: str | None = None) -> str | None:
    base = (directory.strip("/") + "/") if directory else ""
    for p in files(t):
        if p.startswith(base) and "/" not in p[len(base):] and \
                re.match(r"(?i)^readme(\.md|\.markdown)?$", p[len(base):]):
            return p
    return None


def llms_links(text: str) -> list[dict]:
    """The pages an llms.txt lists: its markdown links, once each."""
    out, seen = [], set()
    for title, url in _LINK.findall(text or ""):
        if url not in seen:
            seen.add(url)
            out.append({"title": title.strip(), "url": url})
    return out


def discover(res: dict, t: dict | None, *, local_src: str | None = None
             ) -> dict:
    """The sources of one package, by tier: {chosen: [{tier, url | path,
    frontier, why}], skipped: [...], tier3: {where, pages}, examples_dirs:
    [...]}. `res` is the resolution's package record plus its attached
    `sources`. No network except the llms.txt read (tier 3)."""
    pkg = res.get("package") or {}
    gh_ = pkg.get("repository") or {}
    commit = pkg.get("commit")
    directory = gh_.get("directory")
    chosen: list[dict] = []
    skipped: list[dict] = []
    examples_dirs: list[str] = []

    def at(path: str) -> str:
        return net.raw_url(gh_["owner"], gh_["repo"], commit, path)

    tier1 = []
    if t is not None and commit and gh_:
        for p in skill_mds(t, directory):
            tier1.append(p)
            chosen.append({"tier": 1, "url": at(p), "path": p,
                           "frontier": True, "why": "a SKILL.md in the "
                           "repository at the commit"})
            for r in references_of(t, p):
                chosen.append({"tier": 1, "url": at(r), "path": r,
                               "frontier": False,
                               "why": f"a reference beside {p}"})
    elif local_src:
        # No tag: what the tarball ships (deps.SOURCE_EXTS keeps .md).
        for dp, _dn, fn in os.walk(local_src):
            for f in fn:
                if f == "SKILL.md":
                    rel = os.path.relpath(os.path.join(dp, f), local_src)
                    tier1.append(rel.replace("\\", "/"))
                    chosen.append({"tier": 1, "local": os.path.join(dp, f),
                                   "path": rel.replace("\\", "/"),
                                   "frontier": True,
                                   "why": "a SKILL.md the tarball ships (no "
                                          "tag for this version)"})
    # Tier 2: the prompt's own links.
    for s in res.get("sources") or []:
        url = s.get("url")
        if s.get("kind") == "github_path":
            path = s.get("path") or ""
            fl = files(t) if t is not None else {}
            if s.get("is_dir"):
                if f"{path.rstrip('/')}/SKILL.md" in fl or t is None:
                    raw = net.raw_url(s["owner"], s["repo"], s["ref"],
                                      f"{path.rstrip('/')}/SKILL.md")
                    chosen.append({"tier": 2, "url": raw, "path": path,
                                   "frontier": True, "why": "the prompt "
                                   "links a folder holding a SKILL.md"})
                else:
                    examples_dirs.append(path.rstrip("/"))
                    skipped.append({"path": path, "why": "a linked folder "
                                    "with no SKILL.md: an examples "
                                    "directory for the examples stage"})
                continue
            raw = url if urllib.parse.urlsplit(url).hostname == \
                "raw.githubusercontent.com" else net.raw_url(
                    s["owner"], s["repo"], s["ref"], path)
            chosen.append({"tier": 2, "url": raw, "path": path,
                           "frontier": path.endswith("SKILL.md"),
                           "why": "the prompt links it"})
        elif url:
            chosen.append({"tier": 2, "url": url, "frontier":
                           bool(re.search(r"/SKILL\.md$", url)),
                           "why": "the prompt links it"})
    # Tier 3 and the lead fallback: only when the package ships no SKILL.md.
    tier3: dict = {"where": None, "pages": [], "ingest": not tier1}
    if t is not None and commit and gh_:
        fl = files(t)
        base = (directory.strip("/") + "/") if directory else ""
        for cand in (f"{base}llms.txt", "llms.txt"):
            if cand in fl:
                r = net.raw(gh_["owner"], gh_["repo"], commit, cand,
                            max_bytes=_doc_cap())
                if r.status == 200:
                    tier3.update(where=at(cand),
                                 pages=llms_links(r.text()))
                break
    if tier3["where"] is None and pkg.get("homepage"):
        home = urllib.parse.urlsplit(str(pkg["homepage"]))
        if home.scheme == "https" and home.hostname:
            url = f"https://{home.hostname}/llms.txt"
            try:
                import skill_pipeline
                raw, _meta = skill_pipeline.fetch_source(url)
                tier3.update(where=url, pages=llms_links(
                    raw.decode("utf-8", "replace")))
            except Exception as e:                               # noqa: BLE001
                if type(e).__name__ == "RateLimited":
                    raise
                tier3["looked"] = f"{url}: {str(e)[:160]}"
    if tier3["pages"]:
        if tier1:
            skipped.append({"tier": 3, "where": tier3["where"],
                            "pages": len(tier3["pages"]),
                            "why": "the package ships a SKILL.md: llms.txt "
                                   "pages are recorded, not ingested "
                                   "(operator decision 3); the dashboard's "
                                   "tier-3 action ingests them"})
        else:
            for pg in tier3["pages"]:
                chosen.append({"tier": 3, "url": pg["url"], "frontier": False,
                               "why": f"listed by {tier3['where']}"})
    if not tier1:
        if t is not None and commit and gh_:
            rd = readme_of(t, directory)
            if rd:
                chosen.append({"tier": "lead", "url": at(rd), "path": rd,
                               "frontier": True, "why": "no SKILL.md: the "
                               "README at the commit yields the package's "
                               "lead"})
        elif local_src:
            for f in sorted(os.listdir(local_src)):
                if re.match(r"(?i)^readme(\.md)?$", f):
                    chosen.append({"tier": "lead",
                                   "local": os.path.join(local_src, f),
                                   "path": f, "frontier": True,
                                   "why": "no SKILL.md and no tag: the "
                                          "tarball's README yields the lead"})
                    break
    # Once each (a prompt link that is also a tier-1 file).
    seen, uniq = set(), []
    for c in chosen:
        key = c.get("url") or c.get("local")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return {"chosen": uniq, "skipped": skipped, "tier3": tier3,
            "examples_dirs": examples_dirs,
            "tree": {"entries": len((t or {}).get("entries") or []),
                     "truncated": bool((t or {}).get("truncated")),
                     "walked": (t or {}).get("walked") or []}}


def _doc_cap() -> int:
    import skill_pipeline
    return skill_pipeline.FETCH_MAX_BYTES


# ---------------------------------------------------------------------------
# Making the skills.
# ---------------------------------------------------------------------------
def repo_path_of(url: str) -> tuple[str, str, str] | None:
    """(owner/repo, ref, path) of a raw GitHub URL, else None."""
    p = urllib.parse.urlsplit(url or "")
    if p.hostname != "raw.githubusercontent.com":
        return None
    parts = [urllib.parse.unquote(x) for x in p.path.split("/") if x]
    if len(parts) < 4:
        return None
    return f"{parts[0]}/{parts[1]}".lower(), parts[2], "/".join(parts[3:])


def goal_of(prompt: str) -> str:
    """The prompt's prose: its links removed (the skill pipeline's `goal`)."""
    return " ".join(re.sub(r"https?://\S+", " ", prompt or "").split())


def make_skills(did: str, found: dict, *, package: str | None,
                version: str | None, prompt: str, replaces: dict,
                package_licence: dict | None, author: str) -> dict:
    """Create (or, REPLACE, re-version) one skill per chosen source.
    Idempotent per onboarding. Returns {created, repointed, unchanged,
    existing, failed, to_retire}."""
    import hashlib
    import skill_pipeline
    import skills
    goal = goal_of(prompt)
    mine = [s for s in skills.listing()
            if (s.get("meta") or {}).get("onboarding") == did]
    have = {s.get("source_url") for s in mine}
    # REPLACE: the earlier onboarding's source skills for this package,
    # by repository path.
    earlier: dict[tuple, dict] = {}
    old_versions = set(replaces.get("old") or []) \
        if replaces.get("mode") == "replace" else set()
    if old_versions and package:
        for s in skills.listing():
            m = s.get("meta") or {}
            if m.get("package") != package or m.get("onboarding") == did \
                    or m.get("parent") or str(m.get("package_version")) \
                    not in old_versions or s.get("status") == "archived":
                continue
            rp = repo_path_of(s.get("source_url") or "")
            if rp:
                earlier[(rp[0], rp[2])] = s
    out = {"created": [], "repointed": [], "unchanged": [], "existing": [],
           "failed": [], "to_retire": []}
    used: set = set()
    meta = {"onboarding": did}
    if package:
        meta.update(package=package, package_version=version)
    for c in found.get("chosen") or []:
        url = c.get("url")
        if url and skills.normalise_url(url) in have:
            out["existing"].append(url)
            continue
        rp = repo_path_of(url) if url else None
        prev = earlier.get((rp[0], rp[2])) if rp else None
        try:
            if prev is not None:
                used.add(prev["id"])
                raw, fmeta = skill_pipeline.fetch_source(url)
                sha = hashlib.sha256(raw).hexdigest()
                last = next((x for x in skills.versions(prev["id"])
                             if x.get("source_sha256")), None)
                skills.repoint(prev["id"], url, onboarding=did,
                               package_version=version, author=author)
                if last and last.get("source_sha256") == sha:
                    out["unchanged"].append(prev["id"])
                    continue
                origin = ("watch_frontier" if prev.get("source_kind")
                          == "frontier" else "watch")
                v = skills.new_version(prev["id"], origin, author=author,
                                       meta={"previous_sha256": (last or {})
                                             .get("source_sha256"),
                                             "onboarding": did})
                skills.store_source(prev["id"], v, raw, fmeta)
                skills.enqueue(prev["id"], v, skills.PATHS[origin][0])
                out["repointed"].append({"id": prev["id"], "version": v})
                continue
            m = dict(meta, onboarding_source={k: c.get(k) for k in
                                              ("tier", "path", "why")})
            if c.get("local"):
                with open(c["local"], encoding="utf-8",
                          errors="replace") as f:
                    text = f.read()
                if package_licence:
                    m["package_licence"] = package_licence
                s = skills.create(text=text, name=c.get("path"),
                                  author=author, goal=goal,
                                  frontier=bool(c.get("frontier")),
                                  meta=m)
            else:
                s = skills.create(url=url, author=author, goal=goal,
                                  frontier=bool(c.get("frontier")),
                                  watch_hours=0, meta=m)
            out["created"].append(s["id"])
        except Exception as e:                                   # noqa: BLE001
            if type(e).__name__ == "RateLimited":
                raise
            out["failed"].append({"source": url or c.get("path"),
                                  "why": f"{type(e).__name__}: {e}"[:300]})
    # An earlier source whose repository path is gone from this version.
    for key, s in earlier.items():
        if s["id"] not in used:
            out["to_retire"].append({"id": s["id"], "why": f"{key[1]} is not "
                                     f"a source of {package}@{version}"})
    return out
