#!/usr/bin/env python
"""A PROMPT WITH LINKS -> package@version at a commit, with a verbatim licence.

    python mcp/package_resolve.py "Add koota 0.6.6 https://github.com/pmndrs/koota"

The package onboarding's first stage (docs/PACKAGE-ONBOARDING.md 2.1-2.3).
Deterministic: nothing here asks a model to name a package, a version, a
commit or a licence -- a wrong one there would be confidently wrong in every
later request. Every choice records the RULE that made it.

LINKS (the prompt's URLs, and the form's `links`):

| link | resolves to |
|---|---|
| npmjs.com/package/<name>[/v/<version>] | npm <name> at that version |
| pypi.org/project/<name>[/<version>] | PyPI <name> at that version |
| github.com/<o>/<r> (root, tree/<ref>, releases/tag/<tag>, commit/<sha>) | the repo at that ref; its package.json at the commit names the package; a name@version on the registry is indexed from the REGISTRY artefact (what gets installed) |
| github.com/<o>/<r>/tree/<ref>/<path>, blob/<ref>/<path>, raw files, docs pages | a SOURCE (a skill source or an examples directory), attached to the package whose repository it is in, else to the first package; with no package at all the onboarding runs its sources as ordinary skill submissions |

`name@1.2.3` pins in the prose (skill_packages._PIN_AT) name npm packages.

VERSION, in order: the version in the link or pin (`link` / `pin`); a MAJOR
the prompt states after the package's name or an alias ("koota v0", "r3f
v10": `major` -- the newest version with that major, stable preferred when
one exists, else the newest prerelease); else the registry's
`dist-tags.latest` (`latest`).

COMMIT, in order: the packument's `versions[v].gitHead` (`gitHead`, the
commit npm recorded at publish); a GitHub link's own ref (`link`); a tag in
the packument's repository among `v<version>`, `<name>@<version>` and the
bare version (`tag <t>`; the forms koota and pmndrs/math use,
skills/ingested/pmndrs/spec.json); else none, recorded "no tag for this
version" -- the docs and examples stages then read what the tarball ships.

LICENCE, from verbatim quotes only (skill_pipeline.licence_of): the
LICENSE / COPYING file at the commit, and the `"license"` line of the
package.json at the commit (or the packument's own `license` field when there
is no commit). No quote: the dataset holds at `clarify` for the operator.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import package_net as net  # noqa: E402

_URL = re.compile(r"https?://[^\s<>\"'`)\]]+")
LICENCE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE",
                 "COPYING")    # worker.LICENCE_NAMES plus the UK spelling
# The Right Family, Wrong Skill risk a new major ALONGSIDE the old one
# carries (operator decision 2, 2026-09-27; research doc 2.2).
ALONGSIDE_RISK = ("Right Family, Wrong Skill (2606.10388: 95.0-95.7% of "
                  "helpful top-three hits also carry the risky sibling): the "
                  "old major's skills stay armed beside the new major's")


def _code_cap() -> int:
    """The per-file cap for a manifest or code file at a commit: the one
    index_code.iter_files already applies (deps.MAX_FILE_BYTES)."""
    import deps
    return deps.MAX_FILE_BYTES


# ---------------------------------------------------------------------------
# Links.
# ---------------------------------------------------------------------------
def links_of(prompt: str, links=()) -> list[str]:
    """The URLs, in the order given, trailing punctuation off, once each."""
    out: list[str] = []
    for u in list(_URL.findall(prompt or "")) + [str(x) for x in links or ()]:
        u = u.strip().rstrip(".,;:!?")
        if u and re.match(r"^https?://", u) and u not in out:
            out.append(u)
    return out


def classify_link(url: str) -> dict:
    p = urllib.parse.urlsplit(url)
    host = (p.hostname or "").lower()
    parts = [urllib.parse.unquote(x) for x in p.path.split("/") if x]
    if host in ("npmjs.com", "www.npmjs.com") and parts[:1] == ["package"]:
        rest = parts[1:]
        if rest and rest[0].startswith("@") and len(rest) >= 2:
            name, rest = f"{rest[0]}/{rest[1]}", rest[2:]
        elif rest:
            name, rest = rest[0], rest[1:]
        else:
            return {"kind": "page", "url": url}
        ver = rest[1] if len(rest) >= 2 and rest[0] == "v" else None
        return {"kind": "npm", "url": url, "name": name, "version": ver}
    if host in ("pypi.org", "www.pypi.org") and parts[:1] == ["project"] \
            and len(parts) >= 2:
        return {"kind": "pypi", "url": url, "name": parts[1],
                "version": parts[2] if len(parts) >= 3 else None}
    if host == "github.com" and len(parts) >= 2:
        o, r = parts[0], re.sub(r"\.git$", "", parts[1])
        rest = parts[2:]
        base = {"owner": o, "repo": r, "url": url}
        if not rest:
            return dict(base, kind="github_repo", ref=None)
        if rest[0] == "commit" and len(rest) >= 2:
            return dict(base, kind="github_repo", ref=rest[1])
        if rest[:2] == ["releases", "tag"] and len(rest) >= 3:
            return dict(base, kind="github_repo", ref="/".join(rest[2:]))
        if rest[0] in ("tree", "blob") and len(rest) >= 2:
            if len(rest) == 2 and rest[0] == "tree":
                return dict(base, kind="github_repo", ref=rest[1])
            return dict(base, kind="github_path", ref=rest[1],
                        path="/".join(rest[2:]),
                        is_dir=rest[0] == "tree")
        return {"kind": "page", "url": url}
    if host == "raw.githubusercontent.com" and len(parts) >= 4:
        return {"kind": "github_path", "url": url, "owner": parts[0],
                "repo": parts[1], "ref": parts[2],
                "path": "/".join(parts[3:]), "is_dir": False}
    return {"kind": "page", "url": url}


def aliases_for(name: str, aliases, packages: int) -> list[str]:
    """The operator's typed aliases that belong to `name`. An alias written
    `alias=package` belongs to that package; an unbound alias belongs to the
    package only when the prompt names ONE (with several, which package a
    bare word means is not written anywhere, and it is not guessed)."""
    out = []
    for a in aliases or ():
        a = str(a).strip()
        if not a:
            continue
        word, eq, pkg = a.partition("=")
        if eq:
            if pkg.strip() == name:
                out.append(word.strip())
        elif packages == 1:
            out.append(a)
    return [x for x in out if x]


def pins_of(prose: str) -> list[tuple[str, str]]:
    import skill_packages
    return [(n, v) for n, v in skill_packages._PIN_AT.findall(prose or "")]


def major_after(prose: str, names) -> int | None:
    """A major the prose states right after one of `names` ("r3f v10",
    "koota version 0"), or None."""
    for n in names:
        if not n:
            continue
        m = re.search(r"(?<![\w@/-])" + re.escape(n) + r"\s+(?:v|version\s*)"
                      r"(\d+)\b", prose or "", re.I)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Versions.
# ---------------------------------------------------------------------------
def semver_key(v: str) -> tuple:
    """Sort key: numeric core, then a release above its prereleases, then
    the prerelease identifiers (numeric ones numerically), as semver 2.0
    orders them."""
    core, _, pre = str(v).partition("-")
    core = core.split("+", 1)[0]
    nums = []
    for x in core.split("."):
        nums.append(int(x) if x.isdigit() else -1)
    while len(nums) < 3:
        nums.append(0)
    if not pre:
        return (tuple(nums), 1, ())
    ids = tuple((0, int(x), "") if x.isdigit() else (1, 0, x)
                for x in pre.split("+", 1)[0].split("."))
    return (tuple(nums), 0, ids)


def major_of(v: str) -> int | None:
    m = re.match(r"v?(\d+)", str(v or ""))
    return int(m.group(1)) if m else None


def choose_version(available, *, latest: str | None, want: str | None,
                   want_rule: str | None, major: int | None) -> dict:
    """{version, rule, why} or {error}. `available` is every published
    version."""
    avail = [str(v) for v in available or []]
    if want:
        if want in avail:
            return {"version": want, "rule": want_rule or "link",
                    "why": f"the {want_rule or 'link'} names {want}"}
        return {"error": f"{want} is not a published version "
                         f"({len(avail)} published)"}
    if major is not None:
        same = [v for v in avail if major_of(v) == major]
        stable = [v for v in same if "-" not in v]
        pick = max(stable or same, key=semver_key) if same else None
        if pick is None:
            return {"error": f"no published version has major {major}"}
        return {"version": pick, "rule": "major",
                "why": f"the prompt states major {major}: the newest "
                       + ("stable" if stable else "prerelease (no stable "
                          "release has that major)")
                       + f" of {len(same)} with it"}
    if latest and latest in avail:
        return {"version": latest, "rule": "latest",
                "why": "the registry's dist-tags.latest"}
    if avail:
        pick = max([v for v in avail if "-" not in v] or avail,
                   key=semver_key)
        return {"version": pick, "rule": "newest",
                "why": "no dist-tags.latest: the newest published version"}
    return {"error": "no published version"}


# ---------------------------------------------------------------------------
# Repository and commit.
# ---------------------------------------------------------------------------
def github_of(repo) -> dict | None:
    """{owner, repo, directory} from a packument `repository` (a string or
    {type, url, directory}), when it is on GitHub."""
    import deps
    key = deps._repo_key(repo)
    if not key or not key.startswith("github.com/"):
        return None
    bits = key.split("/")
    if len(bits) < 3:
        return None
    d = repo.get("directory") if isinstance(repo, dict) else None
    return {"owner": bits[1], "repo": bits[2],
            "directory": str(d).strip("/") if d else None}


def commit_of_ref(owner: str, repo: str, ref: str) -> str | None:
    doc, _r = net.gh(f"repos/{owner}/{repo}/commits/"
                     f"{urllib.parse.quote(ref, safe='')}")
    return (doc or {}).get("sha") if isinstance(doc, dict) else None


def tag_commit(owner: str, repo: str, tag: str) -> str | None:
    """The commit a tag names (an annotated tag is followed to its
    commit)."""
    doc, _r = net.gh(f"repos/{owner}/{repo}/git/ref/tags/"
                     f"{urllib.parse.quote(tag, safe='@')}")
    obj = (doc or {}).get("object") if isinstance(doc, dict) else None
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "tag":
        t, _ = net.gh(f"repos/{owner}/{repo}/git/tags/{obj.get('sha')}")
        obj = (t or {}).get("object") if isinstance(t, dict) else None
        if not isinstance(obj, dict):
            return None
    return obj.get("sha") if obj.get("type") == "commit" else None


def default_branch_commit(owner: str, repo: str) -> tuple[str | None, str]:
    doc, _r = net.gh(f"repos/{owner}/{repo}")
    br = (doc or {}).get("default_branch") if isinstance(doc, dict) else None
    if not br:
        return None, "the repository's default branch could not be read"
    return commit_of_ref(owner, repo, br), f"default branch {br}"


def tag_forms(name: str, version: str) -> list[str]:
    return [f"v{version}", f"{name}@{version}", version]


# ---------------------------------------------------------------------------
# Licence.
# ---------------------------------------------------------------------------
def licence_quotes(gh_: dict | None, commit: str | None, *,
                   manifest_text: str | None = None,
                   manifest_where: str | None = None,
                   field: str | None = None,
                   field_where: str | None = None) -> list[dict]:
    """Every verbatim licence statement found: the LICENSE file at the
    commit, the manifest's licence line. Each {spdx, quote, where}."""
    import skill_pipeline
    out: list[dict] = []
    if gh_ and commit:
        for fn in LICENCE_NAMES:
            r = net.raw(gh_["owner"], gh_["repo"], commit, fn,
                        max_bytes=skill_pipeline.FETCH_MAX_BYTES)
            if r.status != 200 or not r.body:
                continue
            got = skill_pipeline.licence_of(r.text())
            if got:
                out.append(dict(got, where=net.raw_url(
                    gh_["owner"], gh_["repo"], commit, fn), kind="file"))
                break
    if manifest_text:
        for line in manifest_text.splitlines():
            if re.search(r'"licen[cs]e"\s*:', line, re.I):
                got = skill_pipeline.licence_of(line)
                if got:
                    out.append(dict(got, quote=line.strip().rstrip(","),
                                    where=manifest_where, kind="manifest"))
                break
    elif field:
        got = skill_pipeline.licence_of(f"license: {field}") or {
            "spdx": str(field)[:80]}
        out.append({"spdx": got["spdx"], "quote": f'"license": '
                    f'{json.dumps(field)}', "where": field_where,
                    "kind": "registry field"})
    return out


# ---------------------------------------------------------------------------
# Resolution.
# ---------------------------------------------------------------------------
def _npm(name: str, *, want: str | None, rule: str | None, major: int | None,
         gh_ref: dict | None = None) -> dict:
    doc, r = net.npm_packument(name)
    rec: dict = {"ecosystem": "npm", "name": name,
                 "registry": f"{net.REGISTRY}/{name}"}
    if not isinstance(doc, dict) or not doc.get("versions"):
        rec["error"] = (f"npm has no package {name!r} (HTTP {r.status})"
                        if r.status != 200 else
                        f"the packument of {name!r} lists no version")
        return rec
    ch = choose_version(doc["versions"], latest=(doc.get("dist-tags") or {})
                        .get("latest"), want=want, want_rule=rule, major=major)
    if ch.get("error"):
        rec["error"] = ch["error"]
        return rec
    v = ch["version"]
    vd = doc["versions"][v] or {}
    dist = vd.get("dist") or {}
    rec.update(version=v, version_rule=ch["rule"], version_why=ch["why"],
               tarball=dist.get("tarball"), integrity=dist.get("integrity"),
               unpacked_size=dist.get("unpackedSize"),
               published=((doc.get("time") or {}).get(v) or "")[:10] or None,
               peer_dependencies=sorted((vd.get("peerDependencies") or {})),
               homepage=vd.get("homepage") or doc.get("homepage"))
    gh_ = github_of(vd.get("repository") or doc.get("repository"))
    if gh_ is None and gh_ref:
        gh_ = {"owner": gh_ref["owner"], "repo": gh_ref["repo"],
               "directory": None}
    rec["repository"] = gh_
    commit, crule = None, None
    if vd.get("gitHead"):
        commit, crule = vd["gitHead"], "gitHead"
    elif gh_ref and gh_ref.get("commit"):
        commit, crule = gh_ref["commit"], "link"
    elif gh_:
        for t in tag_forms(name, v):
            c = tag_commit(gh_["owner"], gh_["repo"], t)
            if c:
                commit, crule = c, f"tag {t}"
                break
    rec["commit"] = commit
    rec["commit_rule"] = crule or "no tag for this version"
    manifest = None
    if gh_ and commit:
        path = "/".join(x for x in (gh_.get("directory"), "package.json") if x)
        mr = net.raw(gh_["owner"], gh_["repo"], commit, path,
                     max_bytes=_code_cap())
        if mr.status == 200:
            manifest = (mr.text(), net.raw_url(gh_["owner"], gh_["repo"],
                                               commit, path))
    rec["licence_quotes"] = licence_quotes(
        gh_, commit,
        manifest_text=manifest[0] if manifest else None,
        manifest_where=manifest[1] if manifest else None,
        field=None if manifest else (vd.get("license") or doc.get("license")),
        field_where=f"{net.REGISTRY}/{name} versions[{v}].license")
    return rec


def _pypi(name: str, *, want: str | None, rule: str | None,
          major: int | None) -> dict:
    doc, r = net.pypi_project(name)
    rec: dict = {"ecosystem": "python", "name": name,
                 "registry": f"{net.PYPI}/{name}/json"}
    if not isinstance(doc, dict) or not doc.get("releases"):
        rec["error"] = f"PyPI has no project {name!r} (HTTP {r.status})"
        return rec
    info = doc.get("info") or {}
    ch = choose_version(doc["releases"], latest=info.get("version"),
                        want=want, want_rule=rule, major=major)
    if ch.get("error"):
        rec["error"] = ch["error"]
        return rec
    v = ch["version"]
    rec.update(version=v, version_rule=ch["rule"], version_why=ch["why"])
    urls = dict(info.get("project_urls") or {})
    gh_ = None
    for k in ("Source", "Repository", "Source Code", "Code", "Homepage"):
        u = urls.get(k)
        if u and "github.com/" in u:
            gh_ = github_of(u)
            break
    rec["repository"] = gh_
    commit, crule = None, None
    if gh_:
        for t in (f"v{v}", v):
            c = tag_commit(gh_["owner"], gh_["repo"], t)
            if c:
                commit, crule = c, f"tag {t}"
                break
    rec["commit"] = commit
    rec["commit_rule"] = crule or "no tag for this version"
    field = info.get("license")
    quotes = licence_quotes(gh_, commit)
    for c in info.get("classifiers") or []:
        if c.startswith("License ::"):
            import skill_pipeline
            got = skill_pipeline.licence_of(c)
            if got:
                quotes.append(dict(got, quote=c, where=rec["registry"]
                                   + " info.classifiers", kind="classifier"))
                break
    if not quotes and field:
        quotes = licence_quotes(None, None, field=field,
                                field_where=rec["registry"] + " info.license")
    rec["licence_quotes"] = quotes
    rec["unsupported"] = ("index, vocabulary and examples are JS/TS-only "
                          "today (docs/PACKAGE-ONBOARDING.md 2.1, item P)")
    return rec


def _github(link: dict) -> dict:
    """A repository link: its commit, and the package its manifest names."""
    o, r = link["owner"], link["repo"]
    if link.get("ref"):
        commit = commit_of_ref(o, r, link["ref"])
        crule = f"link ref {link['ref']}"
    else:
        commit, crule = default_branch_commit(o, r)
    rec: dict = {"owner": o, "repo": r, "commit": commit,
                 "commit_rule": crule, "url": link["url"]}
    if not commit:
        rec["error"] = f"no commit for {o}/{r} at {link.get('ref') or 'HEAD'}"
        return rec
    mr = net.raw(o, r, commit, "package.json", max_bytes=_code_cap())
    if mr.status == 200:
        try:
            pj = json.loads(mr.text())
        except ValueError:
            pj = {}
        if pj.get("name") and not pj.get("private"):
            rec.update(manifest="package.json", name=pj["name"],
                       version=pj.get("version"), ecosystem="npm")
            return rec
        rec["manifest_note"] = ("package.json at the root is "
                                + ("a private workspace root" if
                                   pj.get("private") else "unnamed")
                                + ": link the package itself")
    pr = net.raw(o, r, commit, "pyproject.toml", max_bytes=_code_cap())
    if pr.status == 200:
        try:
            import tomllib
            pt = tomllib.loads(pr.text())
        except Exception:                                        # noqa: BLE001
            pt = {}
        proj = pt.get("project") or {}
        if proj.get("name"):
            rec.update(manifest="pyproject.toml", name=proj["name"],
                       version=proj.get("version"), ecosystem="python")
    return rec


def resolve(prompt: str, links=(), aliases=()) -> dict:
    """{"packages": [...], "sources": [...], "unresolved": [...]} for one
    submission. Raises package_net.RateLimited when a server says wait."""
    urls = links_of(prompt, links)
    out: dict = {"links": urls, "packages": [], "sources": [],
                 "unresolved": []}
    wanted: list[dict] = []           # {ecosystem, name, version?, rule, link}
    repos: list[dict] = []
    for u in urls:
        c = classify_link(u)
        if c["kind"] == "npm":
            wanted.append({"ecosystem": "npm", "name": c["name"],
                           "version": c.get("version"), "rule": "link"
                           if c.get("version") else None, "link": u})
        elif c["kind"] == "pypi":
            wanted.append({"ecosystem": "python", "name": c["name"],
                           "version": c.get("version"), "rule": "link"
                           if c.get("version") else None, "link": u})
        elif c["kind"] == "github_repo":
            repos.append(c)
        else:
            out["sources"].append(c)
    for n, v in pins_of(prompt):
        if not any(w["name"] == n for w in wanted):
            wanted.append({"ecosystem": "npm", "name": n, "version": v,
                           "rule": "pin", "link": None})
    gh_found: dict[tuple, dict] = {}
    for c in repos:
        g = _github(c)
        gh_found[(c["owner"].lower(), c["repo"].lower())] = g
        if g.get("error") or not g.get("name"):
            out["unresolved"].append({"link": c["url"], "why": g.get("error")
                                      or g.get("manifest_note")
                                      or "no package manifest at the commit",
                                      "github": g})
            continue
        if not any(w["name"] == g["name"] for w in wanted):
            wanted.append({"ecosystem": g["ecosystem"], "name": g["name"],
                           "version": None, "rule": None, "link": c["url"],
                           "github": g, "manifest_version": g.get("version")})
        else:
            for w in wanted:
                if w["name"] == g["name"]:
                    w["github"] = g
    for w in wanted:
        names = [w["name"], w["name"].split("/")[-1]] + aliases_for(
            w["name"], aliases, len(wanted))
        major = None if w.get("version") else major_after(prompt, names)
        g = w.get("github")
        gh_ref = ({"owner": g["owner"], "repo": g["repo"],
                   "commit": g.get("commit")} if g else None)
        if w["ecosystem"] == "npm":
            want = w.get("version")
            rule = w.get("rule")
            if not want and g and w.get("manifest_version") and major is None:
                want, rule = w["manifest_version"], "github manifest"
            rec = _npm(w["name"], want=want, rule=rule, major=major,
                       gh_ref=gh_ref)
            if rec.get("error") and g and rule == "github manifest":
                # The repository's own version is not on the registry: a
                # GitHub-only package at that commit.
                rec = {"ecosystem": "github", "name": w["name"],
                       "version": w.get("manifest_version"),
                       "version_rule": "github manifest",
                       "repository": {"owner": g["owner"], "repo": g["repo"],
                                      "directory": None},
                       "commit": g.get("commit"),
                       "commit_rule": g.get("commit_rule"),
                       "registry_error": rec.get("error"),
                       "unsupported": "a package that is not on the npm "
                                      "registry is not indexed (deps.fetch "
                                      "reads the registry tarball)",
                       "licence_quotes": licence_quotes(
                           {"owner": g["owner"], "repo": g["repo"]},
                           g.get("commit"))}
        else:
            rec = _pypi(w["name"], want=w.get("version"), rule=w.get("rule"),
                        major=major)
        rec["links"] = [x for x in (w.get("link"),) if x]
        if rec.get("error"):
            out["unresolved"].append({"link": w.get("link"), "name":
                                      w["name"], "why": rec["error"]})
            continue
        if not any(p["name"] == rec["name"] for p in out["packages"]):
            out["packages"].append(rec)
    # Sources attach to the package whose repository they are in, else to
    # the first package.
    for s in out["sources"]:
        s["attach_to"] = None
        for p in out["packages"]:
            gh_ = p.get("repository") or {}
            if s.get("owner") and gh_ and \
                    s["owner"].lower() == str(gh_.get("owner")).lower() and \
                    s["repo"].lower() == str(gh_.get("repo")).lower():
                s["attach_to"] = p["name"]
                break
        if s["attach_to"] is None and out["packages"]:
            s["attach_to"] = out["packages"][0]["name"]
    out["gets"] = len(net.LOG)
    return out


def replaces_rule(name: str, version: str, held_versions,
                  asked: str | None = None) -> dict:
    """Operator decision 2 (2026-09-27): "the same major version replaces the
    old one (the old version serves until the new one arms); a new major sits
    alongside (Right Family, Wrong Skill risk noted in the record)". The
    form's explicit choice, when given, wins and says so."""
    held = [str(v) for v in held_versions or [] if str(v) != str(version)]
    same = [v for v in held if major_of(v) == major_of(version)]
    if asked in ("replace", "alongside"):
        mode, why = asked, "the operator chose it on the form"
    elif not held:
        return {"mode": "new", "old": [], "why": f"no version of {name} is "
                                                 "held"}
    elif same:
        mode, why = "replace", (f"{', '.join(same)} has major "
                                f"{major_of(version)}: the same major "
                                "replaces the old version (operator "
                                "decision 2)")
    else:
        mode, why = "alongside", (f"held {', '.join(held)}; major "
                                  f"{major_of(version)} is new: it sits "
                                  "alongside (operator decision 2)")
    out = {"mode": mode, "old": same if mode == "replace" else held,
           "why": why}
    if mode == "alongside":
        out["risk"] = ALONGSIDE_RISK
    return out


if __name__ == "__main__":
    print(json.dumps(resolve(" ".join(sys.argv[1:])), indent=1))
