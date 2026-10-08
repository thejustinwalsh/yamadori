#!/usr/bin/env python
"""Rebuild an inference engine from engines/manifest.yaml, or verify the ones
config.yaml runs.

    python scripts/build_engine.py sd-cpp --jobs 8            # build it
    python scripts/build_engine.py llama-prism --out D:/x     # somewhere else
    python scripts/build_engine.py --verify-only              # every binary
                                                              #   config.yaml
                                                              #   points at
    python scripts/build_engine.py llama-bonsai2 --verify-only
    python scripts/build_engine.py --describe PATH/TO/exe     # hash an exe and
                                                              #   the DLLs it loads
    python scripts/build_engine.py --list

THE VENDORED SOURCE (engines/src; scripts/engine_vendor.py, docs/ENGINES.md
"Vendored source"):

    python scripts/build_engine.py vendor <engine>        # regenerate engines/src/<engine>
    python scripts/build_engine.py check [--derive]       # offline proof of every tree
    python scripts/build_engine.py build <engine>         # build it, NO network
    python scripts/build_engine.py update <engine> --to <sha> [--write]   # rebase the series

WHAT A BUILD DOES, IN ORDER (each step fails loudly; nothing is skipped)

  1. Refuses a pending patch, an output directory that exists and is not
     empty, one inside (or around) a directory a shipped binary lives in, and
     one a running process uses.
  2. Fetches the pinned base commit into <out>/src: `git init` + `git fetch`
     of that SHA (shallow or blob-less, as the manifest's `clone` says, so the
     build number and short hash llama.cpp embeds come out the same), with
     core.autocrlf as recorded (the originals are CRLF checkouts), then the
     submodules, each checked against its recorded SHA.
  3. Applies the patches in order through the index (`git apply --cached`,
     then checked out, so CRLF conversion is the same as a checkout). A patch
     that does not apply stops the build.
  4. Places the pinned inputs (the llama.cpp web UI archive), each checked
     against its SHA-256. An engine whose original was built from a
     non-git copy has its .git removed, so it embeds the same "unknown".
  5. Builds the environment the originals had: vcvars64 on a minimal PATH,
     CUDA's bin first, the tools the manifest says were on PATH (gzip) and
     none of the ones it says were not (npm, pnpm).
  6. Configures with the recorded flags, then checks the CMakeCache against
     them, the native CPU flags GGML_NATIVE resolved to, and the generated
     files the manifest pins (build-info.cpp).
  7. Builds the recorded targets, checks the generated UI source, copies the
     recorded DLLs (each hash-checked), runs the manifest's tests in a
     separate build tree, and prints the SHA-256 of every file the entry
     binary loads next to the shipped one's.

A rebuild is SOURCE-reproducible, not bit-for-bit: MSVC's link stamps a time,
nvcc's fatbins carry temp names, and __FILE__ embeds the source path. That is
why the manifest records the hash of the binary we SHIP, and why deploy_check
verifies that, not a rebuild's. docs/ENGINES.md.

Exit codes: 0 ok, 1 a check failed, 2 refused before doing anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
MANIFEST = os.path.join(ROOT, "engines", "manifest.yaml")
PATCHES = os.path.join(ROOT, "engines", "patches")
CONFIG = os.path.join(ROOT, "config.yaml")
# llama-swap is started by scripts/start-stack.bat, not by config.yaml.
SWAP_EXE = os.path.join(ROOT, "bin", "llama-swap.exe")

# Keys of an engine entry that describe what was SHIPPED, not how to build it:
# they do not change the manifest hash that names a build directory.
_NOT_SOURCE = {"shipped", "previous", "config_refs", "notes", "description",
               # derived from base + patches by `vendor`; not a build input
               "vendor"}
# Environment variables that inject compiler or linker flags behind the
# manifest's back. Removed from every build environment.
_FLAG_VARS = ("CL", "_CL_", "LINK", "_LINK_", "CFLAGS", "CXXFLAGS", "LDFLAGS",
              "CUDAFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
              "CMAKE_GENERATOR", "CMAKE_BUILD_PARALLEL_LEVEL",
              "CMAKE_TOOLCHAIN_FILE", "CCACHE_DIR", "HF_UI_VERSION",
              "HF_WEBUI_VERSION")


# llama.cpp engines build WITHOUT llama-server's web UI (operator,
# 2026-09-25: nothing uses it; the dashboard is the proxy's /dash). Both
# flags: LLAMA_BUILD_UI=OFF alone still downloads the HF prebuilt UI while
# LLAMA_USE_PREBUILT_UI is ON (scripts/ui-assets.cmake). An entry that
# reproduces an older binary with a UI says so in `embeds_ui: <reason>`.
NO_UI_FLAGS = ("-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF")


class BuildError(Exception):
    """A check failed. The message says which, and what to do."""


def is_llama_cpp(entry: dict) -> bool:
    return str(entry.get("upstream", "")).rstrip("/").endswith("/llama.cpp")


def ui_policy_problem(name: str, entry: dict) -> str | None:
    """Why this entry breaks the no-UI default, or None."""
    if not is_llama_cpp(entry) or entry.get("embeds_ui"):
        return None
    flags = list(entry.get("configure") or [])
    missing = [f for f in NO_UI_FLAGS if f not in flags]
    if missing:
        return (f"{name} is a llama.cpp engine without {' '.join(missing)}: "
                "engines build without the web UI (docs/ENGINES.md). Add the "
                "flags, or set `embeds_ui: <reason>` to reproduce a binary "
                "that has one.")
    return None


# --------------------------------------------------------------------------
# hashing and PE imports
# --------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pe_imports(path: str) -> list[str]:
    """The DLL names a PE file imports (normal and delay-load), lower case.
    [] for a file that is not a PE image. Standard library only."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return []
    if len(data) < 0x40 or data[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return []
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    ndd = struct.unpack_from("<I", data, dd - 4)[0]
    secs = []
    sec0 = opt + opt_size
    for i in range(nsec):
        o = sec0 + 40 * i
        vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, o + 8)
        secs.append((vaddr, max(vsize, rsize), raddr))

    def off(rva: int) -> int | None:
        for vaddr, size, raddr in secs:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + raddr
        return None

    def cstr(o: int) -> str:
        end = data.find(b"\0", o)
        return data[o:end].decode("ascii", "replace")

    names: list[str] = []
    # (directory index, descriptor size, offset of the name RVA in it)
    for index, size, name_at in ((1, 20, 12), (13, 32, 4)):
        if index >= ndd:
            continue
        rva = struct.unpack_from("<I", data, dd + 8 * index)[0]
        o = off(rva) if rva else None
        while o is not None and o + size <= len(data):
            desc = data[o:o + size]
            if desc == b"\0" * size:
                break
            nrva = struct.unpack_from("<I", desc, name_at)[0]
            no = off(nrva)
            if no is not None:
                names.append(cstr(no).lower())
            o += size
    return names


def closure(entry: str) -> list[str]:
    """The entry binary plus every file in ITS directory it loads, directly or
    through another such file, by import table. Names, entry first."""
    d = os.path.dirname(entry)
    present = {n.lower(): n for n in os.listdir(d)} if os.path.isdir(d) else {}
    seen = [os.path.basename(entry)]
    todo = [entry]
    while todo:
        for dll in pe_imports(todo.pop()):
            real = present.get(dll)
            if real and real not in seen:
                seen.append(real)
                todo.append(os.path.join(d, real))
    return seen


def describe(entry: str) -> dict:
    """{name: {sha256, size}} for the entry and everything it loads beside
    it -- the shape of a manifest's shipped.files."""
    d = os.path.dirname(entry)
    out = {}
    for name in closure(entry):
        p = os.path.join(d, name)
        out[name] = {"sha256": sha256_file(p), "size": os.path.getsize(p)}
    return out


# --------------------------------------------------------------------------
# manifest and config
# --------------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # MACHINE-LOCAL TOOLCHAIN (2026-10-07): engines/manifest.local.yaml (gitignored; absent = no change) is merged
    # over `defaults` and `toolchains` ONLY, so a machine whose Visual Studio, CUDA and Git live elsewhere can say
    # where (docs/INSTALL.md has an example). The engines, their sources and their recorded hashes are never
    # overridden. The toolchain's recorded VERSIONS (msvc toolset, Windows SDK) are still checked by a build: a
    # different toolchain fails those checks until its versions are stated here, and its binaries are then a
    # rebuild with that toolchain, not the recorded one (docs/ENGINES.md: source-reproducible, not bit-for-bit).
    local = os.path.join(os.path.dirname(os.path.abspath(path)), "manifest.local.yaml")
    if os.path.basename(path) == "manifest.yaml" and os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            over = yaml.safe_load(f) or {}
        for key in ("defaults", "toolchains"):
            if isinstance(over.get(key), dict):
                _deep_merge(data.setdefault(key, {}), over[key])
    return data


def norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(
        path.replace("/", os.sep))))


def _expand(text: str, macros: dict, depth: int = 0) -> str:
    if depth > 10:
        return text
    out = text
    for k, v in macros.items():
        out = out.replace("${" + k + "}", str(v))
    return _expand(out, macros, depth + 1) if out != text else out


def config_binaries(config_path: str) -> list[tuple[str, str]]:
    """[(model id, binary path)] for every model in a llama-swap config:
    the first word of its cmd after macros and comment lines."""
    cfg = load_yaml(config_path)
    macros = {k: v for k, v in (cfg.get("macros") or {}).items()
              if isinstance(v, (str, int, float))}
    out = []
    for mid, m in (cfg.get("models") or {}).items():
        cmd = (m or {}).get("cmd") or ""
        lines = [ln.split(" #")[0].strip() for ln in str(cmd).splitlines()]
        words = " ".join(ln for ln in lines
                         if ln and not ln.startswith("#")).split()
        if not words:
            continue
        exe = _expand(words[0], macros).strip('"').strip("'")
        out.append((str(mid), exe))
    return out


def _blocks(value) -> list[dict]:
    if isinstance(value, list):
        return [b for b in value if isinstance(b, dict)]
    return [value] if isinstance(value, dict) else []


def shipped_entries(manifest: dict) -> dict[str, tuple[str, dict]]:
    """{normalised path: (label, block)} for every binary the manifest pins:
    each engine's `shipped`, and any `previous` block that still records its
    files (a binary config.yaml may run until it is switched)."""
    out = {}
    for name, e in (manifest.get("engines") or {}).items():
        for key in ("shipped", "previous"):
            for b in _blocks((e or {}).get(key)):
                if b.get("path") and b.get("files"):
                    label = name if key == "shipped" else f"{name} (previous)"
                    out.setdefault(norm(b["path"]), (label, b))
    return out


def verify_binaries(manifest: dict, binaries: list[tuple[str, str]]
                    ) -> tuple[list[str], list[str]]:
    """(problems, ok lines) for each (who, path): the path must be an engine's
    shipped path, every file it pins must match size and SHA-256, and every
    DLL it loads from its own directory must be pinned."""
    problems: list[str] = []
    ok: list[str] = []
    table = shipped_entries(manifest)
    by_path: dict[str, list[str]] = {}
    shown_as: dict[str, str] = {}
    for who, path in binaries:
        by_path.setdefault(norm(path), []).append(who)
        shown_as.setdefault(norm(path), path)
    for key, whos in by_path.items():
        who = ", ".join(sorted(whos))
        shown = shown_as[key]
        if key not in table:
            problems.append(
                f"{shown} (used by {who}) is not in engines/manifest.yaml: no "
                "engine records it as shipped. Rebuild it with "
                "scripts/build_engine.py and record it (docs/ENGINES.md), or "
                "point config.yaml at a binary the manifest pins.")
            continue
        engine, s = table[key]
        files = s.get("files") or {}
        d = os.path.dirname(key)
        bad = []
        for fname, want in files.items():
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                bad.append(f"{fname} missing")
                continue
            size = os.path.getsize(p)
            if want.get("size") is not None and size != want["size"]:
                bad.append(f"{fname} size {size} != {want['size']}")
                continue
            got = sha256_file(p)
            if got != want.get("sha256"):
                bad.append(f"{fname} sha256 {got[:16]}... != "
                           f"{str(want.get('sha256'))[:16]}...")
        if os.path.exists(key):
            pinned = {f.lower() for f in files}
            unpinned = [f for f in closure(key) if f.lower() not in pinned]
            bad += [f"{f} is loaded but not pinned" for f in unpinned]
        if bad:
            problems.append(
                f"{shown} (engine {engine}, used by {who}) does not match "
                "the manifest: " + "; ".join(bad) + ". The binary on disk is "
                "not the one recorded -- rebuild with scripts/build_engine.py "
                f"{engine.split(' ')[0]} and update the entry's shipped block, "
                "or restore the recorded binary.")
        else:
            ok.append(f"{shown}: {len(files)} files match engine {engine} "
                      f"(used by {who})")
    return problems, ok


def verify_deploy(config_path: str = CONFIG, manifest_path: str = MANIFEST,
                  extra: list[tuple[str, str]] | None = None
                  ) -> tuple[list[str], list[str]]:
    """Every binary config.yaml points at, plus llama-swap itself."""
    if not os.path.exists(manifest_path):
        return [f"{manifest_path} is missing: nothing pins the engines"], []
    if not os.path.exists(config_path):
        return [f"{config_path} is missing: cannot tell what runs"], []
    bins = config_binaries(config_path)
    bins += extra if extra is not None else [("llama-swap", SWAP_EXE)]
    return verify_binaries(load_yaml(manifest_path), bins)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def engine_hash(manifest: dict, name: str, patches_dir: str = PATCHES) -> str:
    """8 hex chars naming a build: the engine's source-defining keys, its
    toolchain and its patch files' bytes. Recording a new shipped hash does
    not change it; changing a flag, a pin or a patch does."""
    e = manifest["engines"][name]
    src = {k: v for k, v in e.items() if k not in _NOT_SOURCE}
    tc = (manifest.get("toolchains") or {}).get(e.get("toolchain"), {})
    h = hashlib.sha256(json.dumps({"engine": src, "toolchain": tc},
                                  sort_keys=True, default=str).encode())
    for p in e.get("patches") or []:
        path = os.path.join(patches_dir, name, p["file"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:8]


def protected_dirs(manifest: dict, config_path: str = CONFIG) -> list[str]:
    """Directories a running engine lives in: a build must never be written
    into or around one. For .../<checkout>/build*/bin/x.exe that is the
    checkout; otherwise the binary's own directory."""
    paths = []
    for e in (manifest.get("engines") or {}).values():
        for k in ("shipped", "previous"):
            for b in _blocks((e or {}).get(k)):
                if b.get("path"):
                    paths.append(b["path"])
    if os.path.exists(config_path):
        paths += [p for _, p in config_binaries(config_path)]
    paths.append(SWAP_EXE)
    out = []
    for p in paths:
        n = norm(p)
        d = os.path.dirname(n)
        if os.path.basename(d) == "bin" and \
                os.path.basename(os.path.dirname(d)).startswith("build"):
            d = os.path.dirname(os.path.dirname(d))
        if d not in out:
            out.append(d)
    return out


def running_exes() -> list[str]:
    ps = "Get-Process | Where-Object { $_.Path } | ForEach-Object { $_.Path }"
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def check_out_dir(out: str, protected: list[str],
                  running: list[str] | None = None) -> None:
    """Raise BuildError unless `out` is safe to build into."""
    n = norm(out)
    for p in protected:
        if n == p or n.startswith(p + os.sep) or p.startswith(n + os.sep):
            raise BuildError(
                f"refusing {out}: it is {'inside' if n.startswith(p) else 'around'}"
                f" {p}, where a shipped or configured engine lives. Pick a new "
                "directory with --out.")
    if os.path.exists(n) and (not os.path.isdir(n) or os.listdir(n)):
        raise BuildError(f"refusing {out}: it exists and is not empty. Builds "
                         "go into a NEW directory; pass another --out.")
    for exe in running or []:
        if norm(exe).startswith(n + os.sep):
            raise BuildError(f"refusing {out}: running process {exe} is in it.")


class Builder:
    def __init__(self, manifest: dict, name: str, out: str, jobs: int,
                 run_tests: bool = True, log=print):
        self.m = manifest
        self.name = name
        self.e = manifest["engines"][name]
        self.tc = (manifest.get("toolchains") or {}).get(self.e.get("toolchain")) or {}
        self.out = os.path.abspath(out)
        # <out>/src is the checkout and <out>/src/build the build tree, the
        # layout every original has (<checkout>/build). With an --out whose
        # src path is as long as the original's, __FILE__ strings line up
        # byte for byte and a rebuild can be compared with the shipped binary.
        self.src = os.path.join(self.out, "src")
        self.bld = os.path.join(self.src, "build")
        self.jobs = jobs
        self.run_tests = run_tests
        self.log = log
        self.report: dict = {"engine": name, "out": self.out,
                             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             "checks": []}

    # -- helpers ---------------------------------------------------------
    def say(self, msg: str) -> None:
        self.log(f"  {msg}")

    # PORTABLE (2026-10-07, `--portable`): a machine whose MSVC toolset, Windows SDK, CUDA DLLs or CPU differ from the
    # recorded toolchain cannot pass the checks that prove a rebuild's IDENTITY with the shipped one (`identity=True`
    # below). With portable set those are reported as WARN and the build goes on; every check that proves the SOURCE
    # and the CONFIGURATION (the vendored tree, the patches, the CMake flags and cache) still fails the build. The
    # result is a build of the recorded source with this machine's toolchain, not the recorded binary.
    portable = False

    def check(self, ok: bool, what: str, detail: str = "", identity: bool = False) -> None:
        waived = (not ok) and identity and self.portable
        self.report["checks"].append({"ok": ok, "check": what,
                                      "detail": detail, **({"portable_waived": True} if waived else {})})
        self.say(("ok    " if ok else "WARN  " if waived else "FAIL  ") + what
                 + (f"  ({detail})" if detail else "")
                 + ("  [portable: toolchain identity not enforced]" if waived else ""))
        if not ok and not waived:
            raise BuildError(f"{what}: {detail}")

    def run(self, cmd: list[str], cwd: str | None = None,
            env: dict | None = None, log_name: str | None = None) -> str:
        self.say("$ " + " ".join(cmd))
        if log_name:
            path = os.path.join(self.out, log_name)
            with open(path, "w", encoding="utf-8", errors="replace") as lf:
                r = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf,
                                   stderr=subprocess.STDOUT)
            if r.returncode:
                with open(path, encoding="utf-8", errors="replace") as lf:
                    tail = lf.read()[-3000:]
                raise BuildError(f"{cmd[0]} exited {r.returncode}; last output "
                                 f"(full log {path}):\n{tail}")
            return ""
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            raise BuildError(f"{' '.join(cmd)} exited {r.returncode}:\n"
                             f"{(r.stdout or '')[-2000:]}{(r.stderr or '')[-2000:]}")
        return r.stdout

    def git(self, *args: str, cwd: str | None = None) -> str:
        env = dict(os.environ)
        # Every git process here, submodule clones included, checks out the
        # way the originals were checked out.
        env.update({"GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.autocrlf",
                    "GIT_CONFIG_VALUE_0": str(self.clone.get("autocrlf", True)).lower(),
                    "GIT_TERMINAL_PROMPT": "0"})
        return self.run([self.tc.get("git", {}).get("path", "git"), *args],
                        cwd=cwd or self.src, env=env)

    @property
    def clone(self) -> dict:
        return self.e.get("clone") or {}

    # -- steps -----------------------------------------------------------
    def fetch_source(self) -> None:
        base = self.e["base_commit"]
        os.makedirs(self.src)
        self.git("init", "-q", ".")
        self.git("config", "core.autocrlf",
                 str(self.clone.get("autocrlf", True)).lower())
        if self.clone.get("abbrev"):
            self.git("config", "core.abbrev", str(self.clone["abbrev"]))
        self.git("remote", "add", "origin", self.e["upstream"])
        mode = self.clone.get("mode", "blobless")
        if mode == "shallow":
            self.git("fetch", "-q", "--depth", "1", "origin", base)
        elif mode == "blobless":
            self.git("fetch", "-q", "--filter=blob:none", "origin", base)
        else:
            self.git("fetch", "-q", "origin", base)
        self.git("-c", "advice.detachedHead=false", "checkout", "-q", base)
        head = self.git("rev-parse", "HEAD").strip()
        self.check(head == base, "source is the pinned base commit", head)
        subs = self.e.get("submodules") or {}
        if subs:
            self.git("submodule", "update", "--init", "--recursive", "-q")
            for path, want in subs.items():
                got = self.git("rev-parse", "HEAD",
                               cwd=os.path.join(self.src, path)).strip()
                self.check(got == want, f"submodule {path} at its pin", got[:12])

    def apply_patches(self) -> None:
        for p in self.e.get("patches") or []:
            path = os.path.join(PATCHES, self.name, p["file"])
            if not os.path.exists(path):
                raise BuildError(f"patch {path} is missing")
            if p.get("sha256"):
                self.check(sha256_file(path) == p["sha256"],
                           f"patch {p['file']} is the recorded file")
            where = os.path.join(self.src, p.get("dir", "."))
            try:
                self.git("apply", "--cached", "--check", path, cwd=where)
            except BuildError as err:
                raise BuildError(f"patch {p['file']} does not apply to "
                                 f"{self.e['base_commit'][:12]}: {err}") from None
            self.git("apply", "--cached", path, cwd=where)
            status = self.git("diff", "--cached", "--name-status", "HEAD",
                              cwd=where)
            for line in status.splitlines():
                st, _, f = line.partition("\t")
                full = os.path.join(where, f)
                if st.startswith("D") and os.path.exists(full):
                    os.remove(full)
            self.git("checkout-index", "-a", "-f", cwd=where)
            self.check(True, f"patch {p['file']} applied",
                       f"{len(status.splitlines())} files")
        self.report["patched"] = self.git("diff", "--cached", "--stat", "HEAD")

    def place_inputs(self) -> None:
        for key, inp in (self.e.get("inputs") or {}).items():
            os.makedirs(os.path.join(self.out, "inputs"), exist_ok=True)
            dst = os.path.join(self.out, "inputs", f"{key}.tar.gz")
            if inp.get("path"):
                shutil.copyfile(os.path.join(ROOT, inp["path"]), dst)
            else:
                self.say(f"fetch {inp['url']}")
                with urllib.request.urlopen(inp["url"], timeout=300) as r, \
                        open(dst, "wb") as f:
                    shutil.copyfileobj(r, f)
            self.check(sha256_file(dst) == inp["sha256"],
                       f"input {key} is the pinned archive", inp["sha256"][:16])
            target = os.path.join(self.src, inp["extract_to"])
            os.makedirs(target, exist_ok=True)
            with tarfile.open(dst) as t:
                t.extractall(target, filter="data")

    def detach_git(self) -> None:
        """The original was built from a copy with no .git: remove it so the
        build embeds the same "unknown" version the original did."""
        if self.clone.get("keep_git", True):
            return
        for dirpath, dirnames, filenames in os.walk(self.src):
            if ".git" in dirnames:
                shutil.rmtree(os.path.join(dirpath, ".git"),
                              onerror=_force_remove)
                dirnames.remove(".git")
            if ".git" in filenames:
                os.remove(os.path.join(dirpath, ".git"))
        self.check(not os.path.exists(os.path.join(self.src, ".git")),
                   "source detached from git (the original was a copy)")

    def environment(self) -> dict:
        """vcvars64 over a minimal PATH, then CUDA and the recorded tools."""
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        base = dict(os.environ)
        for k in _FLAG_VARS:
            base.pop(k, None)
        git_dir = os.path.dirname(self.tc.get("git", {}).get("path", ""))
        path = [os.path.join(sysroot, "System32"), sysroot,
                os.path.join(sysroot, "System32", "Wbem"),
                os.path.join(sysroot, "System32", "WindowsPowerShell", "v1.0")]
        if git_dir:
            path.append(git_dir.replace("/", "\\"))
        base["PATH"] = ";".join(path)
        bat = os.path.join(self.out, "vcvars-env.bat")
        with open(bat, "w", encoding="ascii") as f:
            f.write(f'@call "{self.tc["vcvars"]}" >nul 2>&1\r\n@set\r\n')
        r = subprocess.run(["cmd", "/d", "/c", bat], env=base,
                           capture_output=True, text=True, errors="replace")
        env = {}
        for line in r.stdout.splitlines():
            k, sep, v = line.partition("=")
            if sep and k:
                env[k] = v
        cuda = self.tc["cuda"]["root"].replace("/", "\\")
        extra_tail = [d.replace("/", "\\")
                      for d in (self.e.get("path_tools") or {}).values() if d]
        env["PATH"] = ";".join([cuda + "\\bin", env.get("PATH", "")]
                               + extra_tail)
        env["CUDAToolkit_ROOT"] = cuda
        env["CUDA_PATH"] = cuda
        if not self.clone.get("keep_git", True):
            env["GIT_CEILING_DIRECTORIES"] = self.out
        self.check(env.get("VCToolsVersion") == self.tc["msvc"]["toolset"],
                   "vcvars64 selects the recorded MSVC toolset",
                   env.get("VCToolsVersion", "vcvars failed"), identity=True)
        want_sdk = self.tc.get("windows_sdk")
        if want_sdk:
            self.check(env.get("WindowsSDKVersion", "").strip("\\") == want_sdk,
                       "vcvars64 selects the recorded Windows SDK",
                       env.get("WindowsSDKVersion", ""), identity=True)
        for tool, d in (self.e.get("path_tools") or {}).items():
            found = shutil.which(tool, path=env["PATH"])
            if d:
                self.check(bool(found), f"{tool} is on PATH, as it was", found or "", identity=True)
            else:
                self.check(not found, f"{tool} is NOT on PATH, as it was not",
                           found or "", identity=True)
        for tool in self.e.get("absent_tools") or []:
            found = shutil.which(tool, path=env["PATH"])
            self.check(not found, f"{tool} is NOT on PATH, as it was not",
                       found or "", identity=True)
        return env

    def configure(self, env: dict, bld: str, flags: list[str]) -> None:
        cmake = self.tc["cmake"]["path"]
        # Backslashes, as the originals' .bat scripts passed it (it is stored
        # verbatim in the CMakeCache).
        cuda = self.tc["cuda"]["root"].replace("/", "\\")
        cmd = [cmake, "-S", self.src, "-B", bld, "-G", "Ninja",
               f"-DCMAKE_MAKE_PROGRAM={self.tc['ninja']['path']}",
               *flags, f"-DCUDAToolkit_ROOT={cuda}"]
        self.run(cmd, env=env,
                 log_name=f"configure-{os.path.basename(bld)}.log")

    def check_cache(self, bld: str, flags: list[str]) -> dict:
        cache = read_cache(os.path.join(bld, "CMakeCache.txt"))
        want = dict(parse_flags(flags))
        want.update({k: str(v) for k, v in (self.e.get("expect_cache") or {}).items()})
        for k, v in want.items():
            self.check(cache.get(k) == v, f"CMakeCache {k}={v}",
                       f"got {cache.get(k)!r}")
        return cache

    def check_original_cache(self, bld: str) -> None:
        """Every BOOL/STRING option in the new CMakeCache equals the one in
        the shipped build tree's (when that tree is still on disk)."""
        tree = (self.e.get("shipped") or {}).get("build_tree")
        if not tree or not os.path.exists(os.path.join(tree, "CMakeCache.txt")):
            self.say("original CMakeCache not on disk; option diff skipped")
            return
        old = cache_options(os.path.join(tree, "CMakeCache.txt"))
        new = cache_options(os.path.join(bld, "CMakeCache.txt"))
        diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                if old.get(k) != new.get(k)}
        self.report["cache_diff_vs_original"] = diff
        self.check(not diff, f"all {len(old)} CMakeCache options equal the "
                   f"original's ({tree})",
                   "; ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in diff.items()), identity=True)

    def check_log(self, bld: str) -> None:
        """Lines the original's configure printed that decide what is built
        in (sd.cpp's "version unknown", "frontend build disabled")."""
        want = self.e.get("expect_configure_log") or []
        if not want:
            return
        path = os.path.join(self.out, f"configure-{os.path.basename(bld)}.log")
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for line in want:
            self.check(line in text, f"configure printed {line!r}")

    def check_native(self, bld: str) -> None:
        nat = (self.e.get("native_cpu") or {}).get("resolved")
        if not nat:
            return
        flags, defines = ninja_cpu_flags(os.path.join(bld, "build.ninja"),
                                         nat.get("object", "ggml-cpu.c.obj"))
        self.check(nat["arch"] in flags.split(),
                   f"GGML_NATIVE resolved to {nat['arch']} on this machine",
                   " ".join(f for f in flags.split() if f.startswith("/arch")), identity=True)
        for d in nat.get("defines") or []:
            self.check(d in defines.split(), f"native define {d}", identity=True)

    def check_generated(self, bld: str, stage: str) -> None:
        for rel, want in (self.e.get("expect_generated") or {}).items():
            if want.get("stage", "configure") != stage:
                continue
            p = os.path.join(bld, rel)
            got = sha256_file(p) if os.path.exists(p) else "missing"
            self.check(got == want["sha256"],
                       f"generated {rel} is byte-identical to the original's",
                       got[:16])

    def check_generated_lacks(self, bld: str) -> None:
        for rel, needle in (self.e.get("expect_generated_lacks") or {}).items():
            p = os.path.join(bld, rel)
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            self.check(needle not in text, f"generated {rel} has no {needle}")

    def build(self, env: dict, bld: str, targets: list[str]) -> None:
        cmd = [self.tc["cmake"]["path"], "--build", bld, "--config", "Release",
               "-j", str(self.jobs)]
        for t in targets:
            cmd += ["--target", t]
        self.run(cmd, env=env, log_name=f"build-{os.path.basename(bld)}.log")

    def post_copies(self) -> None:
        cuda = self.tc["cuda"]["root"]
        for c in self.e.get("post_build_copies") or []:
            src = c["from"].replace("cuda:", cuda.rstrip("/\\") + "/", 1)
            if self.portable and not os.path.exists(src):
                self.say(f"WARN  {os.path.basename(src)} is not in this CUDA toolkit (portable: not copied; "
                         "the server needs its CUDA runtime DLLs on PATH: YAMADORI_CUDA_BIN)")
                continue
            self.check(sha256_file(src) == c["sha256"],
                       f"{os.path.basename(src)} source is the recorded file", identity=True)
            dst = os.path.join(self.bld, c.get("to", "bin"),
                               os.path.basename(src))
            shutil.copyfile(src, dst)
            self.check(sha256_file(dst) == c["sha256"],
                       f"copied {os.path.basename(src)}", identity=True)

    def tests(self, env: dict) -> None:
        t = self.e.get("tests")
        if not t or not self.run_tests:
            if t:
                self.say("tests skipped (--no-tests)")
            return
        bld = os.path.join(self.src, "build-tests")
        flags = list(self.e.get("configure") or []) + list(t.get("configure_extra") or [])
        self.configure(env, bld, flags)
        self.build(env, bld, t.get("targets") or [])
        ctest = os.path.join(os.path.dirname(self.tc["cmake"]["path"]), "ctest.exe")
        self.run([ctest, "--test-dir", bld, "-R", t["ctest_regex"],
                  "--output-on-failure"], env=env, log_name="tests.log")
        with open(os.path.join(self.out, "tests.log"), encoding="utf-8",
                  errors="replace") as f:
            summary = [ln for ln in f.read().splitlines() if "tests passed" in ln]
        self.check(True, "engine tests passed", summary[-1] if summary else "")

    def compare_shipped(self) -> None:
        entry = os.path.join(self.bld, self.e["output"])
        self.check(os.path.exists(entry), f"built {self.e['output']}")
        new = describe(entry)
        old = ((self.e.get("shipped") or {}).get("files")) or {}
        rows = []
        for fname in sorted(set(new) | set(old)):
            n, o = new.get(fname), old.get(fname)
            same = bool(n and o and n["sha256"] == o["sha256"])
            rows.append({"file": fname, "new": n, "shipped": o,
                         "identical": same})
            self.say(f"{fname:28s} new {n['sha256'] if n else '-':64s} "
                     f"{n['size'] if n else '-':>11} | shipped "
                     f"{o['size'] if o else '-':>11} "
                     f"{'IDENTICAL' if same else 'differs'}")
        self.report["files"] = rows
        self.log(f"\n  binary  {entry}\n  sha256  {new[self.e['output'].split('/')[-1]]['sha256']}")

    # -- the whole thing -------------------------------------------------
    def go(self) -> dict:
        os.makedirs(self.out, exist_ok=True)
        self.say(f"engine {self.name} -> {self.out}")
        self.fetch_source()
        self.apply_patches()
        self.place_inputs()
        self.detach_git()
        env = self.environment()
        flags = list(self.e.get("configure") or [])
        self.configure(env, self.bld, flags)
        self.report["cmake_cache"] = {
            k: v for k, v in self.check_cache(self.bld, flags).items()
            if k in dict(parse_flags(flags))}
        self.check_original_cache(self.bld)
        self.check_log(self.bld)
        self.check_native(self.bld)
        self.check_generated(self.bld, "configure")
        self.build(env, self.bld, list(self.e.get("targets") or []))
        self.check_generated(self.bld, "build")
        self.check_generated_lacks(self.bld)
        self.post_copies()
        self.compare_shipped()
        self.tests(env)
        self.report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return self.report


def fetch_release(manifest: dict, name: str, out: str, log=print) -> dict:
    """A `kind: release` engine: download the pinned asset, check it and the
    binary inside it."""
    e = manifest["engines"][name]
    os.makedirs(out, exist_ok=True)
    asset = os.path.join(out, os.path.basename(e["asset"]["url"]))
    log(f"  fetch {e['asset']['url']}")
    with urllib.request.urlopen(e["asset"]["url"], timeout=600) as r, \
            open(asset, "wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256_file(asset)
    if got != e["asset"]["sha256"]:
        raise BuildError(f"asset sha256 {got} != pinned {e['asset']['sha256']}")
    import zipfile
    with zipfile.ZipFile(asset) as z:
        z.extractall(out)
    entry = os.path.join(out, e["output"])
    files = describe(entry)
    want = (e.get("shipped") or {}).get("files") or {}
    for fname, v in files.items():
        same = want.get(fname, {}).get("sha256") == v["sha256"]
        log(f"  {fname:28s} {v['sha256']} {'IDENTICAL' if same else 'DIFFERS'}")
        if not same:
            raise BuildError(f"{fname} from the pinned asset differs from shipped")
    return {"engine": name, "out": out, "files": files}


def _force_remove(func, path, _exc):
    os.chmod(path, 0o700)
    func(path)


def read_cache(path: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line[0] in "#/" or "=" not in line:
                continue
            kt, _, v = line.partition("=")
            k = kt.split(":")[0]
            out[k] = v
    return out


def cache_options(path: str) -> dict[str, str]:
    """The user-settable options of a CMakeCache (BOOL, STRING and
    UNINITIALIZED entries): paths and internals are left out."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line[0] in "#/" or "=" not in line:
                continue
            kt, _, v = line.partition("=")
            k, _, t = kt.partition(":")
            if t in ("BOOL", "STRING", "UNINITIALIZED"):
                out[k] = v.replace("\\", "/")
    return out


def parse_flags(flags: list[str]) -> list[tuple[str, str]]:
    out = []
    for f in flags:
        if not f.startswith("-D") or "=" not in f:
            continue
        k, _, v = f[2:].partition("=")
        out.append((k.split(":")[0], v.strip('"')))
    return out


def ninja_cpu_flags(build_ninja: str, obj: str) -> tuple[str, str]:
    """(FLAGS, DEFINES) of the build statement compiling `obj`."""
    with open(build_ninja, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    flags = defines = ""
    for i, line in enumerate(lines):
        if line.startswith("build ") and obj in line.split(":")[0]:
            for nxt in lines[i + 1:i + 12]:
                s = nxt.strip()
                if s.startswith("FLAGS ="):
                    flags = s[len("FLAGS ="):]
                elif s.startswith("DEFINES ="):
                    defines = s[len("DEFINES ="):].replace("-D", "")
            break
    return flags, defines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

VENDOR_ACTIONS = ("vendor", "check", "build", "update")


def main(argv: list[str]) -> int:
    if argv and argv[0] in VENDOR_ACTIONS:
        # The vendored engine source, engines/src: scripts/engine_vendor.py.
        import engine_vendor
        return engine_vendor.main(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engine", nargs="?", help="engine name in the manifest")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", help="NEW directory to build in (default "
                                  "<engines_root>/<engine>-<manifest hash>)")
    ap.add_argument("--verify-only", action="store_true",
                    help="hash the binaries config.yaml points at (or this "
                         "engine's shipped files) against the manifest")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--portable", action="store_true",
                    help="warn instead of failing where the recorded toolchain's identity (MSVC toolset, Windows "
                         "SDK, CUDA DLL hashes, native CPU flags) differs on this machine; the source and "
                         "configuration checks still fail the build (docs/INSTALL.md)")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--describe", metavar="EXE",
                    help="print the shipped.files block for EXE and exit")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.describe:
        import yaml
        print(yaml.safe_dump({"path": args.describe.replace("\\", "/"),
                              "files": describe(args.describe)},
                             sort_keys=False))
        return 0

    manifest = load_yaml(args.manifest)
    engines = manifest.get("engines") or {}
    if args.list:
        for n, e in engines.items():
            s = (e.get("shipped") or {})
            print(f"  {n:20s} {e.get('kind', 'source'):8s} "
                  f"{str(e.get('base_commit', e.get('version', '')))[:12]:12s} "
                  f"{engine_hash(manifest, n)}  {s.get('path', '')}")
        return 0

    if args.verify_only:
        if args.engine:
            e = engines.get(args.engine)
            if not e:
                print(f"  no engine {args.engine!r}; --list shows them")
                return 2
            problems, ok = verify_binaries(
                manifest, [(args.engine, e["shipped"]["path"])])
        else:
            problems, ok = verify_deploy(args.config, args.manifest)
        for line in ok:
            print("  ok    " + line)
        for line in problems:
            print("  FAIL  " + line)
        print(f"\n  {len(ok)}/{len(ok) + len(problems)} engine binaries match "
              "engines/manifest.yaml")
        return 1 if problems else 0

    if not args.engine or args.engine not in engines:
        print(f"  name an engine ({', '.join(engines)}), or --verify-only")
        return 2
    e = engines[args.engine]
    ui = ui_policy_problem(args.engine, e)
    if ui:
        print(f"  REFUSED: {ui}")
        return 2
    pending = [p["file"] for p in e.get("patches") or []
               if p.get("status") == "pending"]
    if pending:
        print(f"  REFUSED: {args.engine} has pending patches: {pending}. "
              "Export them into engines/patches first.")
        return 2
    root = (manifest.get("defaults") or {}).get("engines_root",
                                                 os.path.expanduser("~/engines"))
    out = args.out or os.path.join(
        root, f"{args.engine}-{engine_hash(manifest, args.engine)}")
    try:
        check_out_dir(out, protected_dirs(manifest, args.config), running_exes())
    except BuildError as err:
        print(f"  REFUSED: {err}")
        return 2
    try:
        if e.get("kind") == "release":
            report = fetch_release(manifest, args.engine, out)
        else:
            b = Builder(manifest, args.engine, out, args.jobs,
                        run_tests=not args.no_tests)
            b.portable = args.portable
            report = b.go()
    except BuildError as err:
        print(f"\n  BUILD FAILED: {err}")
        return 1
    with open(os.path.join(out, "build-report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\n  BUILD OK: {out} (report: build-report.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
