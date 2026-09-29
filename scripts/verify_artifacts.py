#!/usr/bin/env python
"""Prove that the model files and runtimes the stack loads are the ones recorded.

    python scripts/verify_artifacts.py [--config config.yaml] [--rehash]
                                       [--no-runtimes] [--json]
    python scripts/verify_artifacts.py --write-locks      # refresh lock files

THE ONE FUNCTION. `verify(config_path) -> list[Problem]`, for the deploy
check to call. It reads config.yaml the way llama-swap does (macros expanded,
comment lines dropped), collects every model-file argument of every entry
(`-m`, `--mmproj`, `--kv-mean-center`, `--diffusion-model`, `--vae`, `--llm`,
and any other `.gguf`/`.safetensors` token), and checks each against
models/manifest.yaml (plus the gitignored models/manifest.local.yaml when it
exists): recorded at all, present, the recorded size, the recorded sha256.
Then the artifacts CODE loads (`loaded_by: code`: index/token_embd.npz, the
E1 heads), then the runtimes (the stack interpreter against
requirements.lock.txt, web/dist against its source, llama-swap's version,
SearXNG's commit and settings template, Hermes' commit).

An empty list is the only pass. Every Problem says what is wrong, whether it
blocks (`error`) or is drift around the stack (`warn`), and the remedy with
its owner -- the rule of AGENTS.md "Failure returns carry the next step".

  error  a file config.yaml loads is unrecorded, missing, or not the recorded
         bytes; a manifest entry is malformed; the stack interpreter's
         packages differ from the lock
  warn   the E1 head moved (self-tuning promotes versions by design), the
         dashboard bundle is stale, llama-swap / SearXNG / Hermes differ from
         the record, an in-service entry that config.yaml no longer loads

HASHING. Full sha256 of every referenced file (~45 GB today). A cache in
index/artifact_hashes.json (gitignored) keyed on (size, mtime_ns) skips
files that have not changed since they were last hashed; `--rehash` (or
`rehash=True`) ignores it. The cache defends against accidents, not against
someone who sets mtimes on purpose: run with --rehash when that matters.

The binaries (llama-server, sd-server) are NOT checked here: they are
engines/manifest.yaml's, checked by its own step in scripts/deploy_check.py.

Standard library + PyYAML (already in the stack env).
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MANIFEST = os.path.join(ROOT, "models", "manifest.yaml")
LOCAL_MANIFEST = os.path.join(ROOT, "models", "manifest.local.yaml")
HASH_CACHE = os.path.join(ROOT, "index", "artifact_hashes.json")
DEFAULT_CONFIG = os.path.join(ROOT, "config.yaml")
LOCAL_PREFIX = "#local: "   # a lock line pip cannot install (conda build, editable)

# Flags whose value is a model file, across llama-server and sd-server.
FILE_FLAGS = {
    "-m", "--model", "--mmproj", "--kv-mean-center", "-md", "--model-draft",
    "--diffusion-model", "--vae", "--llm", "--llm_vision", "--clip_l",
    "--clip_g", "--t5xxl", "--taesd", "--control-net", "--upscale-model",
    "--lora", "--lora-model-dir", "--embd-dir", "--clip_vision",
}
FILE_SUFFIXES = (".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
                 ".onnx")

STATUSES = {"in_service", "disabled", "retired", "input", "candidate"}
PROVENANCE = {"pinned", "reproduced", "recipe", "hash_only"}
REQUIRED = ("id", "role", "status", "path", "size", "sha256", "provenance")


@dataclasses.dataclass
class Problem:
    severity: str          # "error" | "warn"
    artifact: str          # manifest id, runtime id, or the path
    message: str           # what is wrong, as a fact
    remedy: str            # what to do, and who does it

    def __str__(self) -> str:
        return f"[{self.severity}] {self.artifact}: {self.message} -- {self.remedy}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ------------------------------------------------------------ manifest ----

def load_manifest(path: str = MANIFEST,
                  local_path: str | None = LOCAL_MANIFEST) -> dict:
    """The tracked manifest with the local one's artifacts and runtimes
    appended (the local file is optional and may hold only those lists)."""
    with open(path, encoding="utf-8") as f:
        m = yaml.safe_load(f) or {}
    m.setdefault("artifacts", [])
    m.setdefault("runtimes", [])
    m["_sources"] = {a.get("id"): path for a in m["artifacts"]}
    if local_path and os.path.exists(local_path):
        with open(local_path, encoding="utf-8") as f:
            loc = yaml.safe_load(f) or {}
        for a in loc.get("artifacts") or []:
            m["artifacts"].append(a)
            m["_sources"][a.get("id")] = local_path
        m["runtimes"].extend(loc.get("runtimes") or [])
    return m


def schema_problems(m: dict) -> list[Problem]:
    out: list[Problem] = []
    seen: set = set()
    for i, a in enumerate(m.get("artifacts") or []):
        aid = a.get("id") or f"artifacts[{i}]"
        src = m.get("_sources", {}).get(a.get("id"), "the manifest")
        fix = f"edit {os.path.relpath(src, ROOT)} (its header lists the fields)"
        for k in REQUIRED:
            if k not in a:
                out.append(Problem("error", aid, f"missing field `{k}`", fix))
        if aid in seen:
            out.append(Problem("error", aid, "duplicate id", fix))
        seen.add(aid)
        if a.get("status") not in STATUSES:
            out.append(Problem("error", aid, f"status {a.get('status')!r} "
                               f"is not one of {sorted(STATUSES)}", fix))
        prov = a.get("provenance") or {}
        if prov.get("status") not in PROVENANCE:
            out.append(Problem("error", aid, f"provenance.status "
                               f"{prov.get('status')!r} is not one of "
                               f"{sorted(PROVENANCE)}", fix))
        if prov.get("status") == "pinned":
            for k in ("repo", "revision", "filename"):
                if not prov.get(k):
                    out.append(Problem("error", aid, f"pinned provenance has no `{k}`", fix))
            if prov.get("revision") and not re.fullmatch(r"[0-9a-f]{40}", str(prov["revision"])):
                out.append(Problem("error", aid, "revision is not a full 40-hex commit", fix))
        if prov.get("status") in ("reproduced", "recipe") and not prov.get("recipe"):
            out.append(Problem("error", aid, "derived artifact has no `recipe`", fix))
        sha = str(a.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            out.append(Problem("error", aid, "sha256 is not 64 lowercase hex", fix))
    return out


# -------------------------------------------------------------- config ----

def _expand(text: str, macros: dict) -> str:
    for _ in range(5):
        new = re.sub(r"\$\{(\w+)\}",
                     lambda mm: str(macros[mm.group(1)]).strip()
                     if mm.group(1) in macros else mm.group(0), text)
        if new == text:
            break
        text = new
    return text


def config_macros(cfg: dict, config_path: str, overrides: dict | None = None) -> dict:
    macros = {k: str(v) for k, v in (cfg.get("macros") or {}).items()}
    macros.setdefault("repo", ROOT.replace("\\", "/"))
    macros.update(overrides or {})
    return macros


def config_files(config_path: str, overrides: dict | None = None) -> list[tuple[str, str, str]]:
    """[(entry, flag, path)] for every model-file argument in config.yaml.
    The first token of each cmd (the binary) is the engines manifest's.
    `overrides` replaces macros (config.template.yaml's are placeholders)."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    macros = config_macros(cfg, config_path, overrides)
    out = []
    for name, entry in (cfg.get("models") or {}).items():
        cmd = (entry or {}).get("cmd") or ""
        lines = [ln for ln in cmd.splitlines() if not ln.strip().startswith("#")]
        toks = [t.strip("\"'") for t in
                shlex.split(_expand("\n".join(lines), macros), posix=False)]
        for i, t in enumerate(toks[1:], start=1):
            prev = toks[i - 1]
            if prev in FILE_FLAGS or t.lower().endswith(FILE_SUFFIXES):
                if t.startswith("-"):
                    continue
                out.append((name, prev if prev in FILE_FLAGS else "", t))
    return out


def norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def resolve(path: str, macros: dict) -> str:
    return os.path.normpath(_expand(path, macros))


# -------------------------------------------------------------- hashing ---

def sha256_file(path: str, cache: dict | None = None) -> str:
    st = os.stat(path)
    key = norm(os.path.abspath(path))
    if cache is not None:
        hit = cache.get(key)
        if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns:
            return hit["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    d = h.hexdigest()
    if cache is not None:
        cache[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                      "sha256": d, "hashed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return d


def _load_cache(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(path: str | None, cache: dict | None) -> None:
    if not path or cache is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def _check_file(a: dict, path: str, cache: dict | None, sev: str = "error") -> list[Problem]:
    aid = a.get("id", path)
    if not os.path.exists(path):
        how = _obtain_hint(a)
        return [Problem(sev, aid, f"{path} does not exist", how)]
    size = os.path.getsize(path)
    if a.get("size") is not None and size != int(a["size"]):
        return [Problem(sev, aid, f"{path} is {size:,} bytes; the manifest "
                        f"records {int(a['size']):,}", _mismatch_hint(a))]
    got = sha256_file(path, cache)
    if got != a.get("sha256"):
        return [Problem(sev, aid, f"{path} sha256 {got[:16]}... is not the "
                        f"recorded {str(a.get('sha256'))[:16]}...", _mismatch_hint(a))]
    return []


def _obtain_hint(a: dict) -> str:
    prov = a.get("provenance") or {}
    if prov.get("status") == "pinned":
        return (f"operator: `python scripts/fetch_models.py fetch {a['id']} "
                f"--dest <dir>` downloads the pinned bytes")
    if prov.get("status") in ("reproduced", "recipe"):
        return "operator: rebuild it with the recipe in models/manifest.yaml (docs/MODELS.md)"
    return "operator: restore it from backup; its source is not recorded (hash only)"


def _mismatch_hint(a: dict) -> str:
    return ("operator: either restore the recorded file (" + _obtain_hint(a).split(": ", 1)[-1]
            + ") or, if the change is intended, record the new file's size, "
            "sha256 and provenance in the manifest (docs/MODELS.md, 'Changing a model')")


# ---------------------------------------------------------------- locks ---

def freeze(interpreter: str, method: str = "pip") -> list[str]:
    """Installed distributions as sorted `name==version` / `name @ url` lines."""
    if method == "pip":
        r = subprocess.run([interpreter, "-m", "pip", "freeze", "--all",
                            "--disable-pip-version-check"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-300:])
        lines = r.stdout.splitlines()
    else:  # importlib.metadata: for a venv without pip (Hermes')
        code = ("import importlib.metadata as m\n"
                "seen=set()\n"
                "for d in m.distributions():\n"
                "  n=d.metadata['Name']\n"
                "  if n and n.lower() not in seen:\n"
                "    seen.add(n.lower()); print(f'{n}=={d.version}')\n")
        r = subprocess.run([interpreter, "-c", code], capture_output=True,
                           text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-300:])
        lines = r.stdout.splitlines()
    return sorted({ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")},
                  key=str.lower)


def python_version(interpreter: str) -> str:
    r = subprocess.run([interpreter, "-c", "import sys; print(sys.version)"],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


def read_lock(path: str) -> tuple[dict, list[str]]:
    header: dict = {}
    reqs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith(LOCAL_PREFIX):
                reqs.append(ln[len(LOCAL_PREFIX):].strip())
                continue
            m = re.match(r"#\s*([a-z_]+):\s*(.*)$", ln)
            if m:
                header[m.group(1)] = m.group(2)
            elif ln.strip() and not ln.startswith("#"):
                reqs.append(ln.strip())
    return header, sorted(set(reqs), key=str.lower)


def write_lock(path: str, rt: dict) -> int:
    interp = rt["interpreter"]
    method = rt.get("freeze", "pip")
    lines = freeze(interp, method)
    extra = []
    conda = rt.get("conda_meta")
    if conda and os.path.isdir(conda):
        extra = sorted(f[:-5] for f in os.listdir(conda) if f.endswith(".json"))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Lock for runtime `{rt['id']}` -- written by "
                "scripts/verify_artifacts.py --write-locks; checked by verify().\n")
        f.write(f"# interpreter: {interp}\n")
        f.write(f"# python: {python_version(interp)}\n")
        f.write(f"# method: {method} freeze\n")
        f.write(f"# written: {time.strftime('%Y-%m-%d')}\n")
        if extra:
            f.write("# conda packages (conda-meta; not installed by pip):\n")
            for e in extra:
                f.write(f"#   {e}\n")
        for ln in lines:
            # A conda-built or editable package has no index to come from:
            # recorded (and compared) as a comment `pip install -r` skips.
            local = ln.startswith("-e ") or " @ file:" in ln
            f.write((LOCAL_PREFIX + ln if local else ln) + "\n")
    return len(lines)


def _lock_problems(rt: dict) -> list[Problem]:
    rid = rt["id"]
    sev = rt.get("severity", "warn")
    interp = rt.get("interpreter")
    lock = os.path.join(ROOT, rt["lock"])
    if not interp or not os.path.exists(interp):
        return [Problem("warn", rid, f"interpreter {interp} is not on this machine",
                        "nothing to check here; on the stack host this must exist")]
    if not os.path.exists(lock):
        return [Problem(sev, rid, f"{rt['lock']} does not exist",
                        "operator: `python scripts/verify_artifacts.py --write-locks`")]
    header, want = read_lock(lock)
    try:
        have = freeze(interp, rt.get("freeze", "pip"))
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        return [Problem(sev, rid, f"could not list installed packages: {e}",
                        "operator: run the interpreter by hand and fix what it reports")]
    out = []
    pv = python_version(interp)
    if header.get("python") and header["python"] != pv:
        out.append(Problem(sev, rid, f"python is {pv!r}; the lock records "
                           f"{header['python']!r}",
                           "operator: reinstall the recorded Python, or re-lock if intended"))
    added = sorted(set(have) - set(want), key=str.lower)
    removed = sorted(set(want) - set(have), key=str.lower)
    if added or removed:
        show = ", ".join([f"+{x}" for x in added[:5]] + [f"-{x}" for x in removed[:5]])
        out.append(Problem(sev, rid, f"installed packages differ from {rt['lock']}: "
                           f"{len(added)} not in the lock, {len(removed)} missing ({show})",
                           f"operator: `{interp} -m pip install -r {rt['lock']}` to "
                           "restore, or `--write-locks` and commit if the change is intended"))
    return out


# -------------------------------------------------------------- runtimes --

def _runtime_problems(rt: dict) -> list[Problem]:
    kind = rt.get("check")
    rid = rt.get("id", "?")
    sev = rt.get("severity", "warn")
    if kind == "lock":
        return _lock_problems(rt)
    if kind == "buildinfo":
        sys.path.insert(0, os.path.join(ROOT, "mcp"))
        try:
            import dash_static
        finally:
            sys.path.pop(0)
        r = dash_static.check_buildinfo(os.path.join(ROOT, rt.get("dist", "web/dist")))
        out = [Problem(sev, rid, p, "contributor: `npm ci && npm run build` in web/, "
                       "commit web/dist") for p in r.get("problems", [])]
        lock = os.path.join(ROOT, rt.get("lock", "web/package-lock.json"))
        g = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch",
                            os.path.relpath(lock, ROOT)], capture_output=True, text=True)
        if g.returncode != 0:
            out.append(Problem(sev, rid, f"{rt.get('lock')} is not tracked by git",
                               "contributor: commit it; web/dist is built from it"))
        return out
    if kind == "version":
        exe = rt.get("path")
        if not exe or not os.path.exists(exe):
            return [Problem("warn", rid, f"{exe} is not on this machine",
                            "nothing to check here")]
        out = []
        try:
            r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
            text = (r.stdout + r.stderr).strip()
        except (OSError, subprocess.SubprocessError) as e:
            text = f"<{e}>"
        if rt.get("version") and rt["version"] not in text:
            out.append(Problem(sev, rid, f"`{exe} --version` says {text[:80]!r}; "
                               f"recorded {rt['version']!r}",
                               "operator: reinstall the recorded release, or record the new one"))
        if rt.get("sha256"):
            got = sha256_file(exe)
            if got != rt["sha256"]:
                out.append(Problem(sev, rid, f"{exe} sha256 {got[:16]}... is not the "
                                   f"recorded {rt['sha256'][:16]}...",
                                   "operator: reinstall the recorded release, or record the new one"))
        return out
    if kind == "files":
        out = []
        for f in rt.get("files") or []:
            p = f["path"]
            if not os.path.exists(p):
                out.append(Problem(sev, rid, f"{p} does not exist",
                                   f"operator: see {rt.get('doc', 'docs/MODELS.md')}"))
                continue
            if f.get("contains"):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    if f["contains"] not in fh.read():
                        out.append(Problem(sev, rid, f"{p} does not contain "
                                           f"{f['contains']!r}",
                                           f"operator: see {rt.get('doc', 'docs/MODELS.md')}"))
            if f.get("sha256") and sha256_file(p) != f["sha256"]:
                out.append(Problem(sev, rid, f"{p} is not the recorded bytes",
                                   f"operator: record the change or restore it "
                                   f"({rt.get('doc', 'docs/MODELS.md')})"))
        return out
    if kind == "git":
        repo = rt.get("path")
        if not repo or not os.path.isdir(repo):
            return [Problem("warn", rid, f"{repo} is not on this machine", "nothing to check here")]
        out = []
        r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        head = r.stdout.strip()
        if head != rt.get("commit"):
            out.append(Problem(sev, rid, f"HEAD is {head[:12] or '?'}; recorded "
                               f"{str(rt.get('commit'))[:12]}",
                               "operator: record the new commit in models/manifest.yaml "
                               "runtimes; results measured before and after are not comparable"))
        d = subprocess.run(["git", "-C", repo, "-c", "core.autocrlf=false", "diff",
                            "--ignore-cr-at-eol", "--quiet"], capture_output=True)
        if d.returncode != 0:
            out.append(Problem(sev, rid, "working tree has changes beyond line endings",
                               "operator: commit, stash or record them; a harness with "
                               "local edits is not the recorded one"))
        return out
    return [Problem("error", rid, f"unknown runtime check {kind!r}",
                    "edit models/manifest.yaml runtimes")]


# ---------------------------------------------------------------- verify --

def verify(config_path: str = DEFAULT_CONFIG, manifest_path: str = MANIFEST,
           local_path: str | None = LOCAL_MANIFEST, *, rehash: bool = False,
           runtimes: bool = True, hash_cache: str | None = HASH_CACHE) -> list[Problem]:
    """Every way what the stack loads differs from what is recorded. [] = pass."""
    try:
        m = load_manifest(manifest_path, local_path)
    except (OSError, yaml.YAMLError) as e:
        return [Problem("error", "manifest", f"{manifest_path} unreadable: {e}",
                        "restore models/manifest.yaml from git")]
    out = schema_problems(m)
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        refs = config_files(config_path)
    except (OSError, yaml.YAMLError, ValueError) as e:
        return out + [Problem("error", "config", f"{config_path} unreadable: {e}",
                              "operator: fix config.yaml")]
    macros = config_macros(cfg, config_path)
    cache = None if rehash else _load_cache(hash_cache)
    if rehash and hash_cache:
        cache = {}
    by_path: dict[str, dict] = {}
    for a in m["artifacts"]:
        if a.get("path"):
            by_path[norm(resolve(str(a["path"]), macros))] = a

    # 1. every file config.yaml passes to a model server
    referenced: dict[str, list[str]] = {}
    for entry, flag, p in refs:
        referenced.setdefault(norm(os.path.normpath(p)), []).append(entry)
    checked: set[str] = set()
    for entry, flag, p in refs:
        k = norm(os.path.normpath(p))
        a = by_path.get(k)
        if a is None:
            if k not in checked:
                out.append(Problem("error", p, f"config.yaml entry `{entry}` loads "
                                   f"it ({flag or 'positional'}) and no manifest entry "
                                   "records it",
                                   "operator: add it to models/manifest.yaml (or "
                                   "manifest.local.yaml if its name is private) with "
                                   "size, sha256 and provenance; docs/MODELS.md"))
                checked.add(k)
            continue
        if k in checked:
            continue
        checked.add(k)
        out += _check_file(a, os.path.normpath(p), cache)

    # 2. manifest entries that say in_service but nothing loads
    for a in m["artifacts"]:
        if not a.get("path"):
            continue
        k = norm(resolve(str(a["path"]), macros))
        loaded_by = a.get("loaded_by", "config")
        if loaded_by == "code":
            if a.get("status") in ("in_service", "candidate", "disabled"):
                sev = a.get("severity", "error")
                out += _check_file(a, resolve(str(a["path"]), macros), cache, sev)
                out += _e1_integrity(a, resolve(str(a["path"]), macros))
            continue
        if a.get("status") == "in_service" and k not in referenced:
            out.append(Problem("warn", a.get("id", "?"), "the manifest says in_service "
                               "but config.yaml loads no such file",
                               "operator: set its status (retired / disabled) or fix "
                               "config.yaml"))
    _save_cache(hash_cache, cache)

    # 3. runtimes
    if runtimes:
        for rt in m["runtimes"]:
            if rt.get("status", "in_service") != "in_service":
                continue
            try:
                out += _runtime_problems(rt)
            except Exception as e:  # a checker bug must not hide as a pass
                out.append(Problem("error", rt.get("id", "?"),
                                   f"runtime check crashed: {type(e).__name__}: {e}",
                                   "fix scripts/verify_artifacts.py"))
    return out


def _e1_integrity(a: dict, path: str) -> list[Problem]:
    """An E1 head carries its own weight hash; the current version is named by
    state.json next to it. Both must agree with the manifest."""
    if a.get("format") != "e1-head" or not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        raw = base64.b64decode(blob["W_b64"])
        if hashlib.sha256(raw).hexdigest() != blob.get("W_sha256"):
            out.append(Problem("error", a["id"], "W_b64 does not match the file's own "
                               "W_sha256: the weights were edited or corrupted",
                               "operator: revert the head (e1.revert) or retrain"))
        state_p = os.path.join(os.path.dirname(path), "state.json")
        with open(state_p, encoding="utf-8") as f:
            cur = json.load(f).get("current")
        if cur != blob.get("version"):
            out.append(Problem(a.get("severity", "warn"), a["id"],
                               f"state.json serves v{cur}; the manifest records "
                               f"v{blob.get('version')}",
                               "operator: E1 self-tuning promoted a new version; record "
                               "its file in models/manifest.yaml (its JSON carries its "
                               "training record), or revert it"))
    except (OSError, ValueError, KeyError) as e:
        out.append(Problem("error", a["id"], f"unreadable E1 head: {e}",
                           "operator: restore index/e1 from backup or retrain"))
    return out


# ------------------------------------------------------------------ CLI ---

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--local", default=LOCAL_MANIFEST)
    ap.add_argument("--rehash", action="store_true")
    ap.add_argument("--no-runtimes", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-locks", action="store_true",
                    help="rewrite every runtime lock from what is installed")
    a = ap.parse_args(argv)
    if a.write_locks:
        m = load_manifest(a.manifest, a.local)
        for rt in m["runtimes"]:
            if rt.get("check") == "lock" and rt.get("interpreter") and os.path.exists(rt["interpreter"]):
                n = write_lock(os.path.join(ROOT, rt["lock"]), rt)
                print(f"wrote {rt['lock']}: {n} packages ({rt['id']})")
        return 0
    t0 = time.time()
    probs = verify(a.config, a.manifest, a.local, rehash=a.rehash,
                   runtimes=not a.no_runtimes)
    if a.json:
        print(json.dumps([p.to_dict() for p in probs], indent=1))
    else:
        for p in probs:
            print(p)
        errs = sum(p.severity == "error" for p in probs)
        print(f"\n{errs} error(s), {len(probs) - errs} warning(s) in {time.time() - t0:.0f}s")
    return 1 if any(p.severity == "error" for p in probs) else 0


if __name__ == "__main__":
    sys.exit(main())
