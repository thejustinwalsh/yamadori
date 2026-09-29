#!/usr/bin/env python
"""The engine source we build, vendored in this repo: engines/src/<engine>.

    python scripts/build_engine.py vendor llama-upstream        # (re)generate it
    python scripts/build_engine.py check                        # offline: every tree
    python scripts/build_engine.py check llama-upstream --derive
    python scripts/build_engine.py build llama-upstream --jobs 8   # NO network
    python scripts/build_engine.py update llama-upstream --to <sha>          # dry run
    python scripts/build_engine.py update llama-upstream --to <sha> --write

WHAT IS VENDORED (docs/ENGINES.md "Vendored source")

engines/src/<engine> is the upstream base commit (its submodules flattened
in) WITH OUR PATCHES APPLIED, minus the paths the manifest's exclusion rule
drops (`vendoring.rules`: CI, docs, media, tokenizer test vocabularies --
nothing the build reads; licence and notice files are always kept). It is
the exact source `build` compiles. The patch series in
engines/patches/<engine>/ stays the record of our changes: `vendor` makes
the tree from base + series, `check --derive` takes the series back out of
the tree offline and must land on the recorded base, and `update` rebases
the series, never the tree.

Every file is stored byte for byte as the upstream git blob (LF where
upstream is LF) and .gitattributes marks engines/src `-text`, so git never
converts it; `build` checks the tree out the way the originals were checked
out (core.autocrlf as the manifest's `clone` says) into a new directory.

TREE HASHES. `tree` / `base_tree` are git tree ids computed over the files
with every file as mode 100644 (Windows records no executable bit): the id
`git rev-parse HEAD:engines/src/<engine>` reports once it is committed from
Windows. `upstream_tree` is the base commit's own tree id on GitHub, and
`patched_tree` the git tree of base + patches before the exclusion, so
either can be re-derived from upstream.

Exit codes: 0 ok, 1 a check failed or a rebase conflicted, 2 refused.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_engine as be  # noqa: E402

ROOT = be.ROOT
SRC_ROOT = os.path.join(ROOT, "engines", "src")

# Kept whatever an exclusion rule says: a vendored tree always carries its
# upstream's licences and notices (LICENSE, licenses/LICENSE-curl,
# tools/mirai-s/NOTICE, COPYING ...).
LICENCE_RE = re.compile(r"^(licen[cs]e|copying|notice|unlicense)([-._ ].*)?$",
                        re.I)

# Every git process here: no conversion, no background gc, long paths.
_GIT_CFG = ["-c", "core.autocrlf=false", "-c", "core.eol=lf",
            "-c", "core.safecrlf=false", "-c", "gc.auto=0",
            "-c", "maintenance.auto=false", "-c", "core.longpaths=true",
            "-c", "advice.detachedHead=false", "-c", "init.defaultBranch=main"]
# Commits made in the rebase work repo never leave it, but pin who and when
# so nothing of this machine's git identity is read, and runs repeat.
_IDENT = {"GIT_AUTHOR_NAME": "yamadori-vendor",
          "GIT_AUTHOR_EMAIL": "vendor@yamadori.invalid",
          "GIT_COMMITTER_NAME": "yamadori-vendor",
          "GIT_COMMITTER_EMAIL": "vendor@yamadori.invalid",
          "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
          "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000"}


class VendorError(be.BuildError):
    """A vendoring step failed; the message says which and what to do."""


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------

def _git_exe(manifest: dict | None = None) -> str:
    for tc in ((manifest or {}).get("toolchains") or {}).values():
        p = (tc.get("git") or {}).get("path")
        if p and os.path.exists(p):
            return p
    return shutil.which("git") or "git"


def git(args: list[str], cwd: str, env: dict | None = None,
        input: bytes | None = None, check: bool = True,
        exe: str = "git") -> subprocess.CompletedProcess:
    e = dict(os.environ)
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
              "GIT_OBJECT_DIRECTORY", "GIT_CONFIG_COUNT"):
        e.pop(k, None)
    e["GIT_TERMINAL_PROMPT"] = "0"
    e.update(env or {})
    r = subprocess.run([exe, *_GIT_CFG, *args], cwd=cwd, env=e, input=input,
                       capture_output=True)
    if check and r.returncode:
        raise VendorError(f"git {' '.join(args[:4])} ... exited {r.returncode} "
                          f"in {cwd}:\n"
                          + r.stderr.decode("utf-8", "replace")[-2500:])
    return r


def out(r: subprocess.CompletedProcess) -> str:
    return r.stdout.decode("utf-8", "replace").strip()


class Cache:
    """One non-bare repository holding every object vendoring needs (all
    llama.cpp forks share it, so their common blobs are stored once), with an
    EMPTY work tree: every index operation uses its own temp index."""

    def __init__(self, path: str, exe: str = "git", log=print):
        self.path = os.path.abspath(path)
        self.repo = os.path.join(self.path, "objects-repo")
        self.hist = os.path.join(self.path, "history.git")
        self.exe = exe
        self.log = log
        if not os.path.isdir(os.path.join(self.repo, ".git")):
            os.makedirs(self.repo, exist_ok=True)
            self.g(["init", "-q", "."])

    def g(self, args, cwd=None, **kw):
        return git(args, cwd or self.repo, exe=self.exe, **kw)

    def has(self, sha: str, kind: str = "commit") -> bool:
        return self.g(["cat-file", "-e", f"{sha}^{{{kind}}}"],
                      check=False).returncode == 0

    def fetch(self, url: str, sha: str) -> None:
        """The pinned commit and its whole tree (depth 1), kept by a ref."""
        if self.has(sha):
            return
        self.log(f"  fetch {url} {sha[:12]}")
        self.g(["fetch", "-q", "--no-tags", "--depth", "1", url,
                f"+{sha}:refs/pins/{sha}"])
        if not self.has(sha):
            raise VendorError(f"{url} did not give commit {sha}")

    def build_number(self, url: str, sha: str) -> int:
        """`git rev-list --count <sha>`, what llama.cpp embeds, from a
        commits-only fetch of the history (no trees, no blobs)."""
        if not os.path.isdir(self.hist):
            git(["init", "-q", "--bare", self.hist], self.path, exe=self.exe)
        r = git(["cat-file", "-e", f"{sha}^{{commit}}"], self.hist,
                exe=self.exe, check=False)
        if r.returncode:
            self.log(f"  fetch history of {sha[:12]} (commits only)")
            git(["fetch", "-q", "--no-tags", "--filter=tree:0", url,
                 f"+{sha}:refs/pins/{sha}"], self.hist, exe=self.exe)
        return int(out(git(["rev-list", "--count", sha], self.hist,
                           exe=self.exe)))

    # -- index work ------------------------------------------------------
    def index(self) -> dict:
        fd, p = tempfile.mkstemp(prefix="vendor-", suffix=".index",
                                 dir=self.path)
        os.close(fd)
        os.remove(p)
        return {"GIT_INDEX_FILE": p}

    def drop(self, env: dict) -> None:
        for p in (env["GIT_INDEX_FILE"], env["GIT_INDEX_FILE"] + ".lock"):
            if os.path.exists(p):
                os.remove(p)

    def flat_tree(self, url: str, commit: str, pins: dict | None,
                  log_subs: dict | None = None) -> str:
        """The commit's tree with every submodule's tree read in at its path
        (fetched at the gitlink's commit, checked against `pins`)."""
        self.fetch(url, commit)
        links = [ln for ln in out(self.g(["ls-tree", "-r", commit])).splitlines()
                 if ln.startswith("160000 ")]
        if not links:
            return out(self.g(["rev-parse", f"{commit}^{{tree}}"]))
        urls = {}
        cfg = self.g(["config", "--blob", f"{commit}:.gitmodules",
                      "--get-regexp", r"^submodule\..*\.(path|url)$"],
                     check=False)
        by_name: dict[str, dict] = {}
        for ln in out(cfg).splitlines():
            key, _, val = ln.partition(" ")
            name, _, field = key[len("submodule."):].rpartition(".")
            by_name.setdefault(name, {})[field] = val
        for d in by_name.values():
            if d.get("path"):
                urls[d["path"]] = d.get("url", "")
        env = self.index()
        try:
            self.g(["read-tree", commit], env=env)
            for ln in links:
                meta, path = ln.split("\t", 1)
                sub_sha = meta.split()[2]
                want = (pins or {}).get(path)
                if pins is not None and want != sub_sha:
                    raise VendorError(
                        f"submodule {path} of {commit[:12]} is at {sub_sha}, "
                        f"the manifest pins {want}")
                sub_url = _resolve_url(url, urls.get(path, ""))
                if not sub_url:
                    raise VendorError(f"no url for submodule {path}")
                sub_tree = self.flat_tree(sub_url, sub_sha, None)
                if log_subs is not None:
                    log_subs[path] = {"repo": sub_url, "commit": sub_sha}
                self.g(["update-index", "--force-remove", path], env=env)
                self.g(["read-tree", f"--prefix={path}/", sub_tree], env=env)
            return out(self.g(["write-tree"], env=env))
        finally:
            self.drop(env)

    def apply(self, tree: str, patches: list[tuple[str, str]],
              reverse: bool = False, exclude: list[str] = ()) -> str:
        """`tree` with the patch files applied in order through a temp index
        (as build_engine's fetch build applies them). [(path, dir)]."""
        env = self.index()
        try:
            self.g(["read-tree", tree], env=env)
            for path, where in patches:
                args = ["apply", "--cached"]
                if reverse:
                    args.append("-R")
                if where and where != ".":
                    args.append(f"--directory={where}")
                args += [f"--exclude={x}" for x in exclude]
                r = self.g(args + ["--check", path], env=env, check=False)
                if r.returncode:
                    raise VendorError(
                        f"patch {os.path.basename(path)} does not "
                        f"{'reverse-' if reverse else ''}apply:\n"
                        + r.stderr.decode("utf-8", "replace")[-2000:])
                self.g(args + [path], env=env)
            return out(self.g(["write-tree"], env=env))
        finally:
            self.drop(env)

    def listing(self, tree: str, repo: str | None = None) -> list[tuple[str, str, str]]:
        """[(mode, blob sha, path)] of every file in a tree."""
        r = git(["ls-tree", "-r", "-z", "--full-tree", tree], repo or self.repo,
                exe=self.exe)
        rows = []
        for rec in r.stdout.split(b"\0"):
            if not rec:
                continue
            meta, path = rec.split(b"\t", 1)
            mode, typ, sha = meta.decode().split()
            if typ != "blob":
                raise VendorError(f"{path.decode()} is a {typ} after flattening")
            rows.append((mode, sha, path.decode("utf-8")))
        return rows

    def write_files(self, rows: list[tuple[str, str, str]], dest: str) -> None:
        """Each blob's raw bytes at dest/path (a symlink as a file holding
        its target, as git on Windows checks one out)."""
        p = subprocess.Popen([self.exe, *_GIT_CFG, "cat-file", "--batch"],
                             cwd=self.repo, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE)
        try:
            for _mode, sha, path in rows:
                p.stdin.write(sha.encode() + b"\n")
                p.stdin.flush()
                head = p.stdout.readline().split()
                if len(head) != 3 or head[1] != b"blob":
                    raise VendorError(f"cat-file {sha} ({path}): {head}")
                size = int(head[2])
                data = p.stdout.read(size)
                p.stdout.read(1)
                full = _long(os.path.join(dest, *path.split("/")))
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as f:
                    f.write(data)
        finally:
            p.stdin.close()
            p.wait()


def _resolve_url(base: str, rel: str) -> str:
    if not rel.startswith("../") and not rel.startswith("./"):
        return rel
    parts = base.rstrip("/").split("/")
    for seg in rel.split("/"):
        if seg == "..":
            parts.pop()
        elif seg not in (".", ""):
            parts.append(seg)
    return "/".join(parts)


def _long(path: str) -> str:
    if os.name == "nt" and len(path) > 240 and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(path)
    return path


# --------------------------------------------------------------------------
# tree hashes and the exclusion rule
# --------------------------------------------------------------------------

def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _nest(rows) -> dict:
    root: dict = {}
    for path, sha in rows:
        node = root
        parts = path.split("/")
        for d in parts[:-1]:
            node = node.setdefault(d, {})
        node[parts[-1]] = sha
    return root


def _tree_id(node: dict) -> str:
    ents = []
    for name, v in node.items():
        nb = name.encode("utf-8")
        if isinstance(v, dict):
            ents.append((nb + b"/", b"40000 " + nb + b"\0"
                         + bytes.fromhex(_tree_id(v))))
        else:
            ents.append((nb, b"100644 " + nb + b"\0" + bytes.fromhex(v)))
    ents.sort(key=lambda e: e[0])
    body = b"".join(e[1] for e in ents)
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def tree_hash_of_rows(rows) -> str:
    """Git tree id of [(path, blob sha)], every file mode 100644."""
    return _tree_id(_nest(rows))


def eol_converted(paths, read) -> list[str]:
    """The paths a checkout of this repo converts to CRLF despite the root
    `engines/src/** -text`: a nested upstream .gitattributes that SETS `text`
    for them (sd-cpp's libwebp: `*.bat text eol=crlf`) outranks the root
    file. Their stored blob is LF; `check` hashes them as git stores them.
    read(path) -> bytes of a .gitattributes in the tree."""
    rules = []
    for p in paths:
        if p.rsplit("/", 1)[-1] != ".gitattributes":
            continue
        base = p[:-len(".gitattributes")]
        for ln in read(p).decode("utf-8", "replace").splitlines():
            parts = ln.split()
            if not parts or parts[0].startswith("#"):
                continue
            attrs = parts[1:]
            if "text" in attrs or any(a.startswith("text=") and a != "text=auto"
                                      for a in attrs):
                rules.append((base, parts[0]))
    out_ = []
    for p in paths:
        for base, pat in rules:
            if not p.startswith(base):
                continue
            rel = p[len(base):]
            name = rel if "/" in pat.strip("/") else rel.rsplit("/", 1)[-1]
            if fnmatch.fnmatchcase(name, pat.lstrip("/")):
                out_.append(p)
                break
    return sorted(out_)


def scan_dir(root: str, lf_paths=()) -> dict:
    """{path: (blob sha, size)} of every file under root, as git stores it:
    the paths in lf_paths (eol_converted) with CRLF read back as LF."""
    lf = set(lf_paths)
    files = {}
    for dirpath, dirnames, filenames in os.walk(_long(root)):
        dirnames.sort()
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, _long(root)).replace(os.sep, "/")
            with open(full, "rb") as f:
                data = f.read()
            if rel in lf:
                data = data.replace(b"\r\n", b"\n")
            files[rel] = (blob_sha(data), len(data))
    return files


def tree_hash_of_dir(root: str) -> tuple[str, int, int]:
    files = scan_dir(root)
    return (tree_hash_of_rows((p, s) for p, (s, _) in files.items()),
            len(files), sum(n for _, n in files.values()))


def is_licence(path: str) -> bool:
    return bool(LICENCE_RE.match(path.rsplit("/", 1)[-1]))


def excluded(path: str, patterns: list[str]) -> bool:
    """Anchored at the tree root, the last matching pattern wins: `dir/` is
    that directory, `!pat` puts a path back, anything else is an fnmatch glob
    over the whole path. A licence or notice file is never excluded."""
    if is_licence(path):
        return False
    hit = False
    for pat in patterns:
        neg = pat.startswith("!")
        p = pat[1:] if neg else pat
        if p.endswith("/"):
            m = path.startswith(p) or path == p[:-1]
        else:
            m = fnmatch.fnmatchcase(path, p)
        if m:
            hit = not neg
    return hit


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def llama_family(e: dict) -> bool:
    """llama.cpp or a fork of it under another name (llama.cpp-mirai-s)."""
    return "/llama.cpp" in str(e.get("upstream", ""))


def settings(manifest: dict) -> dict:
    return manifest.get("vendoring") or {}


def cache_path(manifest: dict) -> str:
    v = settings(manifest)
    if v.get("cache"):
        return v["cache"]
    root = (manifest.get("defaults") or {}).get("engines_root",
                                                 os.path.expanduser("~/engines"))
    return os.path.join(root, ".vendor-cache")


def rule_of(manifest: dict, name: str) -> tuple[str, list[str]]:
    e = manifest["engines"][name]
    rules = settings(manifest).get("rules") or {}
    rule = (e.get("vendor") or {}).get("rule")
    if not rule:
        rule = "llama.cpp" if llama_family(e) else \
            str(e.get("upstream", "")).rstrip("/").rsplit("/", 1)[-1]
    if rule not in rules:
        raise VendorError(f"{name}: no exclusion rule {rule!r} under "
                          "vendoring.rules in the manifest")
    return rule, list(rules[rule] or [])


def patch_list(manifest: dict, name: str, patches_dir: str) -> list[dict]:
    rows = []
    for p in manifest["engines"][name].get("patches") or []:
        path = os.path.join(patches_dir, name, p["file"])
        if p.get("status") == "pending":
            raise VendorError(f"{name}: {p['file']} is pending; export it first")
        if not os.path.exists(path):
            raise VendorError(f"{name}: patch {path} is missing")
        got = be.sha256_file(path)
        if p.get("sha256") and got != p["sha256"]:
            raise VendorError(f"{name}: {p['file']} sha256 {got[:16]}... is not "
                              f"the recorded {p['sha256'][:16]}...")
        rows.append({"file": p["file"], "path": path, "sha256": got,
                     "dir": p.get("dir", ".")})
    return rows


def patches_digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['file']} {r['sha256']}\n".encode())
    return h.hexdigest()


def vendor_dir(name: str, src_root: str = SRC_ROOT) -> str:
    return os.path.join(src_root, name)


# -- textual manifest edits (the file is hand-commented; never re-dumped) --

def _engine_span(lines: list[str], name: str) -> tuple[int, int]:
    start = None
    in_engines = False
    for i, ln in enumerate(lines):
        if ln.rstrip() == "engines:":
            in_engines = True
            continue
        if in_engines and ln.rstrip() == f"  {name}:":
            start = i
            break
    if start is None:
        raise VendorError(f"engine {name} not found in the manifest text")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and len(ln) - len(ln.lstrip()) <= 2:
            end = j
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return start, end


def _block_end(lines: list[str], i: int, indent: int, stop: int) -> int:
    j = i + 1
    while j < stop:
        ln = lines[j]
        if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
            break
        j += 1
    while j > i + 1 and not lines[j - 1].strip():
        j -= 1
    return j


def render_vendor_block(rec: dict, name: str) -> list[str]:
    import yaml
    body = yaml.safe_dump(rec, sort_keys=False, default_flow_style=False,
                          width=4096, allow_unicode=True)
    lines = ["    vendor:",
             f"      # Written by `python scripts/build_engine.py vendor {name}`;",
             "      # `check` verifies the tree against it offline. Do not edit."]
    lines += ["      " + ln for ln in body.rstrip("\n").split("\n")]
    return lines


def write_vendor_block(manifest_path: str, name: str, rec: dict) -> None:
    with open(manifest_path, encoding="utf-8", newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    s, e = _engine_span(lines, name)
    new = render_vendor_block(rec, name)
    at = next((i for i in range(s + 1, e) if lines[i].rstrip() == "    vendor:"),
              None)
    if at is not None:
        lines[at:_block_end(lines, at, 4, e)] = new
    else:
        anchor = next((i for i in range(s + 1, e)
                       if lines[i].startswith("    inputs:")), None)
        if anchor is None:
            anchor = next((i for i in range(s + 1, e)
                           if lines[i].startswith("    toolchain:")), e)
        lines[anchor:anchor] = new
    _atomic_write(manifest_path, nl.join(lines))


def set_engine_scalar(manifest_path: str, name: str, key: str, value: str) -> None:
    with open(manifest_path, encoding="utf-8", newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    s, e = _engine_span(lines, name)
    for i in range(s + 1, e):
        if lines[i].startswith(f"    {key}:"):
            rest = lines[i].split("#", 1)
            comment = ("   #" + rest[1]) if len(rest) > 1 else ""
            lines[i] = f"    {key}: {value}{comment}"
            _atomic_write(manifest_path, nl.join(lines))
            return
    raise VendorError(f"{name}: no `{key}:` line to update")


def set_patch_sha(manifest_path: str, name: str, fname: str, sha: str) -> None:
    with open(manifest_path, encoding="utf-8", newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    s, e = _engine_span(lines, name)
    for i in range(s + 1, e):
        if lines[i].strip() == f"- file: {fname}":
            for j in range(i + 1, e):
                if lines[j].strip().startswith("- file:"):
                    break
                if lines[j].strip().startswith("sha256:"):
                    pre = lines[j][:lines[j].index("sha256:")]
                    lines[j] = f"{pre}sha256: {sha}"
                    _atomic_write(manifest_path, nl.join(lines))
                    return
    raise VendorError(f"{name}: no sha256 line for patch {fname}")


def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".vendor-tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# vendor
# --------------------------------------------------------------------------

def build_info(manifest: dict, name: str, cache: Cache, commit: str,
               use_shipped: bool = True) -> dict | None:
    """What a fetch build embeds in build-info.cpp, for the offline build to
    pass as LLAMA_BUILD_NUMBER / LLAMA_BUILD_COMMIT. None when the original
    embeds "unknown" (built from a copy with no .git: sd-cpp)."""
    e = manifest["engines"][name]
    clone = e.get("clone") or {}
    if not llama_family(e) or not clone.get("keep_git", True):
        return None
    abbrev = int(clone.get("abbrev") or 7)
    short = commit[:abbrev]
    if clone.get("mode") == "shallow":
        return {"number": 1, "commit": short,
                "source": "a depth-1 clone embeds build number 1"}
    if use_shipped:
        tree = (e.get("shipped") or {}).get("build_tree")
        p = os.path.join(tree or "", "common", "build-info.cpp")
        if tree and os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                t = f.read()
            n = re.search(r"LLAMA_BUILD_NUMBER\s*=\s*(\d+)", t)
            c = re.search(r'LLAMA_COMMIT\s*=\s*"([0-9a-f]+)"', t)
            if n and c and commit.startswith(c.group(1)):
                return {"number": int(n.group(1)), "commit": c.group(1),
                        "source": "the shipped build's common/build-info.cpp"}
    n = cache.build_number(e["upstream"], commit)
    return {"number": n, "commit": short,
            "source": "git rev-list --count of the base (commits-only fetch)"}


def derive(manifest: dict, name: str, cache: Cache, patches_dir: str,
           base: str | None = None, pins: dict | None = None) -> dict:
    """Everything `vendor` writes, from upstream + the patch series, as data:
    the rows to write and the record."""
    e = manifest["engines"][name]
    if e.get("kind", "source") != "source":
        raise VendorError(f"{name} is a {e.get('kind')} engine: nothing to vendor")
    base = base or e["base_commit"]
    rule, pats = rule_of(manifest, name)
    subs: dict = {}
    base_tree = cache.flat_tree(e["upstream"], base,
                                e.get("submodules") if pins is None else pins,
                                subs)
    rows = patch_list(manifest, name, patches_dir)
    patched = cache.apply(base_tree, [(r["path"], r["dir"]) for r in rows])
    base_rows = cache.listing(base_tree)
    all_rows = cache.listing(patched)
    keep = [r for r in all_rows if not excluded(r[2], pats)]
    dropped = [r for r in all_rows if excluded(r[2], pats)]
    sizes = _blob_sizes(cache, [r[1] for r in all_rows])
    licences = sorted(r[2] for r in keep if is_licence(r[2]))
    attrs = {p: s for _, s, p in keep if p.rsplit("/", 1)[-1] == ".gitattributes"}
    eol = eol_converted([p for _, _, p in keep],
                        lambda p: cache.g(["cat-file", "blob", attrs[p]]).stdout)
    rec = {
        "rule": rule,
        "path": f"engines/src/{name}",
        "repo": e["upstream"],
        "commit": base,
        "upstream_tree": out(cache.g(["rev-parse", f"{base}^{{tree}}"])),
    }
    if subs:
        rec["submodules"] = subs
    rec.update({
        "patches": len(rows),
        "patches_digest": patches_digest(rows),
        "patched_tree": patched,
        "base_tree": tree_hash_of_rows((p, s) for _, s, p in base_rows
                                       if not excluded(p, pats)),
        "tree": tree_hash_of_rows((p, s) for _, s, p in keep),
        "files": len(keep),
        "bytes": sum(sizes[s] for _, s, _ in keep),
        "excluded_files": len(dropped),
        "excluded_bytes": sum(sizes[s] for _, s, _ in dropped),
        "executable_files": sum(1 for m, _, _ in keep if m == "100755"),
        "licences": licences,
        **({"eol_converted": eol} if eol else {}),
        "build_info": build_info(manifest, name, cache, base,
                                 use_shipped=base == e["base_commit"]),
    })
    return {"rows": keep, "record": rec}


def _blob_sizes(cache: Cache, shas: list[str]) -> dict[str, int]:
    uniq = sorted(set(shas))
    r = cache.g(["cat-file", "--batch-check=%(objectname) %(objectsize)"],
                input=("\n".join(uniq) + "\n").encode())
    sizes = {}
    for ln in out(r).splitlines():
        sha, n = ln.split()
        sizes[sha] = int(n)
    return sizes


def vendor(manifest_path: str, name: str, patches_dir: str = be.PATCHES,
           src_root: str = SRC_ROOT, cache: Cache | None = None,
           check_only: bool = False, log=print) -> dict:
    """(Re)generate src_root/<name> from the manifest, write its record.
    check_only: regenerate in a temp directory and compare, write nothing."""
    manifest = be.load_yaml(manifest_path)
    cache = cache or Cache(cache_path(manifest), _git_exe(manifest), log)
    t0 = time.time()
    d = derive(manifest, name, cache, patches_dir)
    rec = d["record"]
    dest = vendor_dir(name, src_root)
    stage = tempfile.mkdtemp(prefix=f"stage-{name}-", dir=cache.path)
    try:
        cache.write_files(d["rows"], stage)
        got, n, size = tree_hash_of_dir(stage)
        if got != rec["tree"] or n != rec["files"]:
            raise VendorError(f"{name}: the written tree hashes {got} ({n} "
                              f"files), the blobs say {rec['tree']}")
        if check_only:
            have = tree_hash_of_dir(dest)[0] if os.path.isdir(dest) else None
            rec["_vendored_tree_on_disk"] = have
            return rec
        os.makedirs(src_root, exist_ok=True)
        if os.path.exists(dest):
            shutil.rmtree(_long(dest), onerror=be._force_remove)
        shutil.move(stage, dest)
        stage = None
        write_vendor_block(manifest_path, name, rec)
        log(f"  vendored {name}: {rec['files']} files, "
            f"{rec['bytes'] / 1e6:.1f} MB (excluded {rec['excluded_files']} "
            f"files, {rec['excluded_bytes'] / 1e6:.1f} MB), tree {rec['tree']}, "
            f"{time.time() - t0:.0f} s")
        return rec
    finally:
        if stage and os.path.exists(stage):
            shutil.rmtree(_long(stage), onerror=be._force_remove)


# --------------------------------------------------------------------------
# check (offline)
# --------------------------------------------------------------------------

def check(manifest: dict, name: str, patches_dir: str = be.PATCHES,
          src_root: str = SRC_ROOT, derive_base: bool = False,
          scan: dict | None = None, exe: str = "git") -> list[str]:
    """Problems with a vendored tree, [] when it is exactly what the manifest
    records. No network. derive_base: also take the patch series back out of
    the tree (git apply -R) and require the recorded base."""
    e = manifest["engines"][name]
    rec = e.get("vendor") or {}
    if not rec.get("tree"):
        return [f"{name}: no vendor record; run `build_engine.py vendor {name}`"]
    dest = vendor_dir(name, src_root)
    if not os.path.isdir(dest):
        return [f"{name}: {dest} is missing; run `build_engine.py vendor {name}`"]
    problems = []
    try:
        rows = patch_list(manifest, name, patches_dir)
        if patches_digest(rows) != rec.get("patches_digest"):
            problems.append(
                f"{name}: the patch series changed since it was vendored "
                f"({len(rows)} patches now, {rec.get('patches')} then); run "
                f"`build_engine.py vendor {name}`")
    except VendorError as err:
        problems.append(str(err))
    if rec.get("commit") != e.get("base_commit"):
        problems.append(f"{name}: vendored from {rec.get('commit')}, the "
                        f"manifest's base is {e.get('base_commit')}")
    files = scan if scan is not None else scan_dir(
        dest, rec.get("eol_converted") or ())
    got = tree_hash_of_rows((p, s) for p, (s, _) in files.items())
    if got != rec["tree"]:
        problems.append(f"{name}: engines/src/{name} hashes {got}, the "
                        f"manifest records {rec['tree']} ({len(files)} files "
                        f"on disk, {rec.get('files')} recorded): the tree was "
                        f"edited. Change a patch and re-vendor instead.")
    missing = [p for p in rec.get("licences") or [] if p not in files]
    if missing:
        problems.append(f"{name}: licence files missing: {missing}")
    if not any("/" not in p and is_licence(p) for p in files):
        problems.append(f"{name}: no licence file at the tree's root")
    try:
        _, pats = rule_of(manifest, name)
        bad = [p for p in files if excluded(p, pats)]
        if bad:
            problems.append(f"{name}: {len(bad)} files the exclusion rule "
                            f"drops are present, e.g. {bad[:3]}")
    except VendorError as err:
        problems.append(str(err))
    if derive_base and not problems:
        problems += _derive_base_offline(manifest, name, dest, patches_dir, exe)
    return problems


def _derive_base_offline(manifest, name, dest, patches_dir, exe) -> list[str]:
    """Reverse-apply the series to the vendored tree in a throwaway repo and
    compare with the recorded base_tree."""
    rec = manifest["engines"][name]["vendor"]
    _, pats = rule_of(manifest, name)
    rows = patch_list(manifest, name, patches_dir)
    with tempfile.TemporaryDirectory(prefix="vendor-derive-") as tmp:
        gd = os.path.join(tmp, "g.git")
        git(["init", "-q", "--bare", gd], tmp, exe=exe)
        env = {"GIT_DIR": gd, "GIT_WORK_TREE": dest,
               "GIT_INDEX_FILE": os.path.join(tmp, "index")}
        git(["add", "-A", "-f", "."], dest, env=env, exe=exe)
        tree = out(git(["write-tree"], dest, env=env, exe=exe))
        touched = set()
        for r in rows:
            n = git(["apply", "--numstat", "-z", r["path"]], tmp,
                    env={"GIT_DIR": gd}, exe=exe)
            for rec_ in n.stdout.split(b"\0"):
                parts = rec_.decode("utf-8", "replace").split("\t")
                if len(parts) == 3 and parts[2]:
                    touched.add(parts[2])
        skip = sorted(p for p in touched if excluded(p, pats))
        env2 = {"GIT_DIR": gd, "GIT_INDEX_FILE": os.path.join(tmp, "index2")}
        git(["read-tree", tree], tmp, env=env2, exe=exe)
        for r in reversed(rows):
            args = ["apply", "--cached", "-R"] + [f"--exclude={p}" for p in skip]
            if r["dir"] not in (".", ""):
                args.append(f"--directory={r['dir']}")
            res = git(args + [r["path"]], tmp, env=env2, exe=exe, check=False)
            if res.returncode:
                return [f"{name}: {r['file']} does not reverse-apply to the "
                        "vendored tree: " + res.stderr.decode("utf-8", "replace")[-600:]]
        base = out(git(["write-tree"], tmp, env=env2, exe=exe))
        ls = git(["ls-tree", "-r", "-z", base], tmp, env={"GIT_DIR": gd}, exe=exe)
        brows = []
        for x in ls.stdout.split(b"\0"):
            if x:
                meta, p = x.split(b"\t", 1)
                brows.append((p.decode("utf-8"), meta.decode().split()[2]))
        got = tree_hash_of_rows(brows)
    if got != rec.get("base_tree"):
        return [f"{name}: the vendored tree minus the series hashes {got}, "
                f"not the recorded base {rec.get('base_tree')}"]
    return []


# --------------------------------------------------------------------------
# build from the vendored tree (no network)
# --------------------------------------------------------------------------

# Where no process may reach during an offline build: every HTTP(S) client
# that honours the proxy variables (cmake's file(DOWNLOAD), git, curl,
# urllib) gets a closed port.
_NO_NET = {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
           "ALL_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
           "https_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": "",
           "GIT_ALLOW_PROTOCOL": "file"}


def materialize(vendored: str, dest: str, autocrlf: bool, exe: str = "git",
                scratch: str | None = None) -> int:
    """Check the vendored tree out into dest the way a git checkout with
    core.autocrlf=<autocrlf> would (the originals' line endings), leaving no
    .git behind. Returns the number of files."""
    scratch = scratch or os.path.dirname(os.path.abspath(dest))
    gd = tempfile.mkdtemp(prefix="vendor-stage-", dir=scratch)
    try:
        env = {"GIT_DIR": gd, "GIT_WORK_TREE": vendored,
               "GIT_INDEX_FILE": os.path.join(gd, "index")}
        git(["init", "-q", "--bare", gd], scratch, exe=exe)
        git(["add", "-A", "-f", "."], vendored, env=env, exe=exe)
        n = len(out(git(["ls-files"], vendored, env=env, exe=exe)).splitlines())
        os.makedirs(dest, exist_ok=True)
        env2 = dict(env, GIT_WORK_TREE=dest)
        git(["-c", f"core.autocrlf={'true' if autocrlf else 'false'}",
             "checkout-index", "-a", "-f"], dest, env=env2, exe=exe)
        return n
    finally:
        shutil.rmtree(gd, onerror=be._force_remove)


class VendoredBuilder(be.Builder):
    """build_engine's Builder with the source step replaced: the vendored
    tree (patches already in it, hash-checked) is checked out into <out>/src;
    nothing is fetched. llama.cpp's build number and commit, which the fetch
    build reads from git, come from the manifest's vendor.build_info."""

    def __init__(self, *a, src_root: str = SRC_ROOT, inputs_dir: str | None = None,
                 patches_dir: str = be.PATCHES, **kw):
        super().__init__(*a, **kw)
        self.src_root = src_root
        self.inputs_dir = inputs_dir
        self.patches_dir = patches_dir
        self.rec = self.e.get("vendor") or {}
        bi = self.rec.get("build_info") or {}
        self.info_flags = ([f"-DLLAMA_BUILD_NUMBER={bi['number']}",
                            f"-DLLAMA_BUILD_COMMIT={bi['commit']}"] if bi else [])

    def fetch_source(self) -> None:
        problems = check(self.m, self.name, self.patches_dir, self.src_root,
                         exe=self.tc.get("git", {}).get("path", "git"))
        self.check(not problems, "vendored tree is the recorded one "
                   f"({self.rec.get('tree', '')[:12]})", "; ".join(problems))
        n = materialize(vendor_dir(self.name, self.src_root), self.src,
                        bool(self.clone.get("autocrlf", True)),
                        self.tc.get("git", {}).get("path", "git"), self.out)
        self.check(n == self.rec.get("files"), "vendored tree checked out, no "
                   ".git", f"{n} files, autocrlf={self.clone.get('autocrlf', True)}")

    def apply_patches(self) -> None:
        self.check(True, "patches are in the vendored tree",
                   f"{self.rec.get('patches')} patches, digest "
                   f"{self.rec.get('patches_digest', '')[:12]}")

    def place_inputs(self) -> None:
        for key, inp in (self.e.get("inputs") or {}).items():
            if inp.get("url") and not inp.get("path"):
                local = os.path.join(self.inputs_dir or "", f"{key}.tar.gz")
                if not self.inputs_dir or not os.path.exists(local):
                    raise be.BuildError(
                        f"input {key} is a download ({inp['url']}); an offline "
                        f"build takes it from --inputs DIR as {key}.tar.gz "
                        f"(sha256 {inp['sha256'][:16]}...)")
                inp = dict(inp, path=os.path.relpath(local, be.ROOT))
                self.e = dict(self.e, inputs=dict(self.e["inputs"], **{key: inp}))
        super().place_inputs()

    def environment(self) -> dict:
        env = super().environment()
        env.update(_NO_NET)
        # The checkout has no .git: git must not find one above it (this
        # repo, when --out is inside it) and report ITS commit.
        env["GIT_CEILING_DIRECTORIES"] = self.out
        self.check(True, "network fenced for the build",
                   "proxy variables -> 127.0.0.1:9, git file:// only")
        return env

    def configure(self, env: dict, bld: str, flags: list[str]) -> None:
        super().configure(env, bld, list(flags) + self.info_flags)

    configure_only = False

    def go(self) -> dict:
        if not self.configure_only:
            return super().go()
        # Everything up to and including configure and its checks (source,
        # environment, CMakeCache vs the original, native CPU flags,
        # build-info.cpp), without compiling.
        os.makedirs(self.out, exist_ok=True)
        self.say(f"engine {self.name} -> {self.out} (configure only)")
        self.fetch_source()
        self.apply_patches()
        self.place_inputs()
        self.detach_git()
        env = self.environment()
        flags = list(self.e.get("configure") or [])
        self.configure(env, self.bld, flags)
        self.check_cache(self.bld, flags)
        self.check_original_cache(self.bld)
        self.check_log(self.bld)
        self.check_native(self.bld)
        self.check_generated(self.bld, "configure")
        self.report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return self.report

    def check_original_cache(self, bld: str) -> None:
        tree = (self.e.get("shipped") or {}).get("build_tree")
        if not tree or not os.path.exists(os.path.join(tree, "CMakeCache.txt")):
            self.say("original CMakeCache not on disk; option diff skipped")
            return
        old = be.cache_options(os.path.join(tree, "CMakeCache.txt"))
        new = be.cache_options(os.path.join(bld, "CMakeCache.txt"))
        mine = {"LLAMA_BUILD_NUMBER", "LLAMA_BUILD_COMMIT"}
        diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                if k not in mine and old.get(k) != new.get(k)}
        self.report["cache_diff_vs_original"] = diff
        self.check(not diff, f"all {len(old)} CMakeCache options equal the "
                   f"original's, bar the build-info overrides ({tree})",
                   "; ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in diff.items()))


# --------------------------------------------------------------------------
# update: rebase the series onto a new upstream base
# --------------------------------------------------------------------------

def _preamble_and_tail(text: str) -> tuple[str, str]:
    """A patch file's header (everything before its first `diff --git`) and
    a format-patch signature after the diff, if any."""
    i = text.find("\ndiff --git ")
    head = text[:i + 1] if i >= 0 else ""
    tail = ""
    m = re.search(r"\n-- \n[^\n]*\n*$", text)
    if m:
        tail = text[m.start() + 1:]
    return head, tail


def _restat(head: str, stat: str) -> str:
    """Replace a format-patch header's diffstat with the rebased one."""
    m = re.search(r"\n---\n(?: .*\n)+ \d+ files? changed[^\n]*\n", head)
    if not m:
        return head
    return head[:m.start()] + "\n---\n" + stat + head[m.end():]


def update(manifest_path: str, name: str, to: str, write: bool = False,
           patches_dir: str = be.PATCHES, src_root: str = SRC_ROOT,
           cache: Cache | None = None, upstream: str | None = None,
           log=print) -> dict:
    """Rebase the engine's patch series from its base onto `to` with git's
    own 3-way rebase. A conflict, or a patch that becomes empty, stops it and
    is reported -- never resolved or dropped here. On success (and --write)
    writes the changed patch files, the new base and sha256s, and re-vendors."""
    manifest = be.load_yaml(manifest_path)
    cache = cache or Cache(cache_path(manifest), _git_exe(manifest), log)
    work = os.path.join(cache.path, "work", name)
    try:
        return _update(manifest_path, manifest, name, to, write, patches_dir,
                       src_root, cache, upstream, work, log)
    finally:
        if os.path.exists(work):
            shutil.rmtree(_long(work), onerror=be._force_remove)


def _update(manifest_path, manifest, name, to, write, patches_dir, src_root,
            cache, upstream, work, log) -> dict:
    e = manifest["engines"][name]
    url = upstream or e["upstream"]
    old = e["base_commit"]
    rows = patch_list(manifest, name, patches_dir)
    rep: dict = {"engine": name, "from": old, "to": to, "patches": [],
                 "conflict": None, "written": False}
    cache.fetch(e["upstream"], old)
    cache.fetch(url, to)
    if not cache.has(to):
        raise VendorError(f"{to} not found at {url}")
    to = out(cache.g(["rev-parse", f"{to}^{{commit}}"]))
    rep["to"] = to
    old_tree = cache.flat_tree(e["upstream"], old, e.get("submodules"))
    new_subs: dict = {}
    new_tree = cache.flat_tree(url, to, None, new_subs)
    rep["submodules"] = new_subs
    if os.path.exists(work):
        shutil.rmtree(_long(work), onerror=be._force_remove)
    os.makedirs(work)
    w = lambda args, **kw: git(args, work, env=dict(_IDENT, **kw.pop("env", {})),  # noqa: E731
                               exe=cache.exe, **kw)
    w(["init", "-q", "."])
    with open(os.path.join(work, ".git", "objects", "info", "alternates"), "wb") as f:
        f.write(os.path.join(cache.repo, ".git", "objects").replace("\\", "/")
                .encode() + b"\n")
    c_old = out(w(["commit-tree", old_tree, "-m", f"base {old}"]))
    c_new = out(w(["commit-tree", new_tree, "-m", f"base {to}"]))
    w(["checkout", "-q", "-f", "-B", "series", c_old])
    for r in rows:
        args = ["apply", "--index"]
        if r["dir"] not in (".", ""):
            args.append(f"--directory={r['dir']}")
        w(args + [r["path"]])
        w(["commit", "-q", "--allow-empty", "-m", r["file"]])
    res = w(["-c", "rerere.enabled=false", "rebase", "-q", "--empty=keep",
             "--onto", c_new, c_old, "series"],
            env={"GIT_EDITOR": "true", "GIT_SEQUENCE_EDITOR": "true"}, check=False)
    if res.returncode:
        stopped = out(w(["log", "-1", "--format=%s", "REBASE_HEAD"], check=False))
        files = out(w(["diff", "--name-only", "--diff-filter=U"], check=False))
        rep["conflict"] = {"patch": stopped or "?",
                           "files": files.splitlines(),
                           "detail": (res.stdout + res.stderr).decode(
                               "utf-8", "replace")[-1500:]}
        w(["rebase", "--abort"], check=False)
        log(f"  CONFLICT: {stopped} does not rebase onto {to[:12]}; files: "
            f"{files.splitlines()}. Nothing written. Refresh that patch by "
            "hand against the new base (docs/ENGINES.md), then run update again.")
        return rep
    commits = out(w(["rev-list", "--reverse", f"{c_new}..series"])).splitlines()
    if len(commits) != len(rows):
        rep["conflict"] = {"patch": "?", "files": [],
                           "detail": f"{len(rows)} patches became {len(commits)} commits"}
        return rep
    new_files: dict[str, bytes] = {}
    for r, c in zip(rows, commits):
        empty = out(w(["rev-parse", f"{c}^{{tree}}"])) == \
            out(w(["rev-parse", f"{c}^^{{tree}}"]))
        parent_tree = out(w(["rev-parse", f"{c}^^{{tree}}"]))
        try:
            same = cache.apply(parent_tree, [(r["path"], r["dir"])]) == \
                out(w(["rev-parse", f"{c}^{{tree}}"]))
        except VendorError:
            same = False
        stat = w(["diff", "--stat=72", "--summary", f"{c}^", c]).stdout.decode("utf-8")
        summary = [ln.strip() for ln in stat.splitlines() if " changed" in ln]
        row = {"file": r["file"], "status": "unchanged" if same else "rebased",
               "empty": empty, "stat": summary[-1] if summary else ""}
        if empty:
            row["status"] = "EMPTY (upstream already has it)"
        if not same:
            diff = w(["diff", "--no-color", "--no-ext-diff", f"{c}^", c]
                     ).stdout.decode("utf-8")
            with open(r["path"], encoding="utf-8", newline="") as f:
                orig = f.read()
            head, tail = _preamble_and_tail(orig)
            new_files[r["file"]] = (_restat(head, stat) + diff + tail).encode("utf-8")
        rep["patches"].append(row)
        log(f"  {r['file']:70s} {row['status']:10s} {row['stat']}")
    empties = [p["file"] for p in rep["patches"] if p["empty"]]
    if empties:
        rep["conflict"] = {"patch": empties[0], "files": [],
                           "detail": "these patches are empty on the new base "
                                     f"(upstream has them): {empties}. Remove "
                                     "them from the manifest yourself, then "
                                     "run update again."}
        log("  STOP: " + rep["conflict"]["detail"])
        return rep
    rep["build_info"] = build_info(manifest, name, cache, to, use_shipped=False)
    moved = {p: v["commit"] for p, v in new_subs.items()
             if (e.get("submodules") or {}).get(p) != v["commit"]}
    if moved:
        log(f"  submodules move on the new base: {moved}")
    if not write:
        log(f"  dry run: {len(new_files)} of {len(rows)} patch files would be "
            f"rewritten; nothing written (--write to apply)")
        return rep
    if moved:
        log("  REFUSED to write: set the entry's `submodules` pins to the ones "
            "above by hand (a reviewed change), then run update --write again.")
        rep["conflict"] = {"patch": "-", "files": list(moved),
                           "detail": "submodule pins move"}
        return rep
    for fname, data in new_files.items():
        p = os.path.join(patches_dir, name, fname)
        with open(p, "wb") as f:
            f.write(data)
        set_patch_sha(manifest_path, name, fname, hashlib.sha256(data).hexdigest())
    set_engine_scalar(manifest_path, name, "base_commit", to)
    vendor(manifest_path, name, patches_dir, src_root, cache, log=log)
    rep["written"] = True
    log(f"  wrote {len(new_files)} patch files, base_commit {to}, and "
        f"re-vendored engines/src/{name}. The entry's expect_generated pins "
        "(build-info.cpp) are for the old base: re-pin them from the next build.")
    return rep


# --------------------------------------------------------------------------
# CLI (build_engine.py dispatches `vendor`, `check`, `build`, `update` here)
# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="build_engine.py",
                                 description="vendored engine source")
    ap.add_argument("action", choices=["vendor", "check", "build", "update"])
    ap.add_argument("engines", nargs="*")
    ap.add_argument("--manifest", default=be.MANIFEST)
    ap.add_argument("--config", default=be.CONFIG)
    ap.add_argument("--src-root", default=SRC_ROOT)
    ap.add_argument("--all", action="store_true",
                    help="vendor/check: every engine the manifest vendors")
    ap.add_argument("--check-upstream", action="store_true",
                    help="vendor: regenerate in a temp dir from upstream and "
                         "compare with the tree on disk; write nothing")
    ap.add_argument("--derive", action="store_true",
                    help="check: also reverse-apply the series offline and "
                         "require the recorded base")
    ap.add_argument("--to", help="update: the new upstream commit (full SHA)")
    ap.add_argument("--from-repo", help="update: fetch --to from this repo "
                                        "instead of the entry's upstream")
    ap.add_argument("--write", action="store_true", help="update: write it")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--configure-only", action="store_true",
                    help="build: stop after configure and its checks")
    ap.add_argument("--inputs", help="build: directory holding download "
                                     "inputs as <key>.tar.gz")
    a = ap.parse_args(argv)
    manifest = be.load_yaml(a.manifest)
    engines = manifest.get("engines") or {}
    names = a.engines
    if a.all or (a.action == "check" and not names):
        names = [n for n, e in engines.items() if (e or {}).get("vendor")]
    for n in names:
        if n not in engines:
            print(f"  no engine {n!r}; --list shows them")
            return 2
    if not names:
        print(f"  name an engine for {a.action}")
        return 2
    try:
        if a.action == "vendor":
            rc = 0
            for n in names:
                if a.check_upstream:
                    rec = vendor(a.manifest, n, src_root=a.src_root, check_only=True)
                    want = engines[n].get("vendor") or {}
                    same = (rec["tree"] == want.get("tree")
                            == rec.get("_vendored_tree_on_disk"))
                    print(f"  {'ok  ' if same else 'FAIL'}  {n}: upstream + "
                          f"series -> {rec['tree']}; recorded "
                          f"{want.get('tree')}; on disk "
                          f"{rec.get('_vendored_tree_on_disk')}")
                    rc |= 0 if same else 1
                else:
                    vendor(a.manifest, n, src_root=a.src_root)
            return rc
        if a.action == "check":
            bad = 0
            for n in names:
                t0 = time.time()
                p = check(manifest, n, src_root=a.src_root, derive_base=a.derive,
                          exe=_git_exe(manifest))
                bad += bool(p)
                rec = engines[n].get("vendor") or {}
                print(("  ok    " if not p else "  FAIL  ") + n
                      + (f"  tree {rec.get('tree', '')[:12]}, "
                         f"{rec.get('files')} files"
                         + (", base re-derived offline" if a.derive else "")
                         + f" ({time.time() - t0:.1f} s)" if not p else ""))
                for x in p:
                    print("        " + x)
            return 1 if bad else 0
        if a.action == "update":
            if len(names) != 1 or not a.to or len(a.to) != 40:
                print("  update takes one engine and --to <40-char SHA>")
                return 2
            rep = update(a.manifest, names[0], a.to, a.write,
                         src_root=a.src_root, upstream=a.from_repo)
            return 1 if rep["conflict"] else 0
        # build
        if len(names) != 1:
            print("  build takes one engine")
            return 2
        n = names[0]
        e = engines[n]
        if not e.get("vendor"):
            print(f"  REFUSED: {n} is not vendored; `vendor {n}` first, or "
                  "build it the fetching way (build_engine.py {n})")
            return 2
        ui = be.ui_policy_problem(n, e)
        if ui:
            print(f"  REFUSED: {ui}")
            return 2
        root = (manifest.get("defaults") or {}).get("engines_root",
                                                     os.path.expanduser("~/engines"))
        need = [k for k, i in (e.get("inputs") or {}).items()
                if i.get("url") and not i.get("path") and not (
                    a.inputs and os.path.exists(os.path.join(a.inputs, f"{k}.tar.gz")))]
        if need:
            print(f"  REFUSED: {n} needs download inputs {need} "
                  f"({', '.join(e['inputs'][k]['url'] for k in need)}); an "
                  "offline build takes each from --inputs DIR as <key>.tar.gz, "
                  "checked against its sha256")
            return 2
        outdir = a.out or os.path.join(root, f"{n}-{be.engine_hash(manifest, n)}v")
        be.check_out_dir(outdir, be.protected_dirs(manifest, a.config),
                         be.running_exes())
        b = VendoredBuilder(manifest, n, outdir, a.jobs,
                            run_tests=not a.no_tests, src_root=a.src_root,
                            inputs_dir=a.inputs)
        b.configure_only = a.configure_only
        report = b.go()
        import json
        with open(os.path.join(outdir, "build-report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"\n  BUILD OK (vendored, offline): {outdir}")
        return 0
    except be.BuildError as err:
        print(f"\n  FAILED: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
