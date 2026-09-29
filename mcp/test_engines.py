#!/usr/bin/env python
"""Engine pinning, asserted offline. Temp files only: nothing here hashes,
runs or loads a live binary.

WHAT THIS IS GATING

engines/manifest.yaml pins every inference engine the stack runs, and
scripts/deploy_check.py refuses a deploy whose config.yaml points at a binary
the manifest does not pin (scripts/build_engine.py verify_deploy). Its
promises:

  1. A binary config.yaml runs must be some engine's `shipped.path`; one that
     is not fails, and the message names it and the remedy.
  2. Every pinned file must match its size and SHA-256; one changed byte
     fails. A missing file fails.
  3. Every DLL the binary loads from its own directory (PE import table,
     transitively, delay-loads included) must be pinned: dropping an unpinned
     DLL beside the exe fails.
  4. deploy_check fails NOT GOOD before it waits for anything.
  5. A build never goes into a non-empty directory, or into/around a
     directory a shipped engine lives in, or one a running process uses; a
     pending patch refuses the build before anything is created.
  6. The committed manifest is well formed: every patch it lists is on disk
     with its recorded hash, every source engine names its base SHA in full.
  7. llama.cpp engines build without the web UI (both LLAMA_BUILD_UI=OFF
     and LLAMA_USE_PREBUILT_UI=OFF) unless the entry says why it embeds one;
     the builder refuses otherwise.
  8. A `previous` block that records its files keeps that binary verifiable
     until config.yaml is switched; one without files pins nothing.

The vendored engine source (engines/src, scripts/engine_vendor.py):

  9. The tree hash is git's own tree id for the same files; the exclusion
     rule is anchored, `!` puts back, licence and notice files always stay.
 10. `vendor` writes base + series minus the excluded paths, deterministically,
     and records it; `check` fails a hand edit, an excluded file, a missing
     licence, and a patch series changed since vendoring; `check --derive`
     takes the series back out offline and lands on the recorded base.
 11. The offline checkout converts line endings as the originals' checkout
     did and leaves no .git.
 12. `update` reports a conflict (patch and file) and writes nothing; a clean
     rebase keeps unchanged patch files byte for byte and rewrites a moved
     one (header kept), the base and the sha256s, then re-vendors.
 13. The committed trees: every source engine config.yaml runs is vendored,
     each tree matches its record, and, once committed, no vendored file is
     left out by a .gitignore (`git add -f`); engines/src is stored -text.
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_engine as be  # noqa: E402
import deploy_check  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def make_pe(imports: list[str], delay: list[str] = ()) -> bytes:
    """A minimal PE32+ image whose import (and delay-import) directories name
    the given DLLs. Enough for build_engine.pe_imports; never executed."""
    data = bytearray(0x600)
    data[0:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x40)
    data[0x40:0x44] = b"PE\0\0"
    coff = 0x44
    struct.pack_into("<HHIIIHH", data, coff, 0x8664, 1, 0, 0, 0, 240, 0x22)
    opt = coff + 20
    struct.pack_into("<H", data, opt, 0x20B)
    struct.pack_into("<I", data, opt + 108, 16)
    dd = opt + 112
    sec = opt + 240
    data[sec:sec + 8] = b".idata\0\0"
    struct.pack_into("<IIII", data, sec + 8, 0x200, 0x1000, 0x200, 0x400)

    def at(rva: int) -> int:
        return rva - 0x1000 + 0x400

    name_rva = 0x1100
    imp_rva, dly_rva = 0x1000, 0x1060
    for i, dll in enumerate(imports):
        struct.pack_into("<I", data, at(imp_rva + 20 * i) + 12, name_rva)
        data[at(name_rva):at(name_rva) + len(dll)] = dll.encode()
        name_rva += 0x10
    for i, dll in enumerate(delay):
        struct.pack_into("<I", data, at(dly_rva + 32 * i) + 4, name_rva)
        data[at(name_rva):at(name_rva) + len(dll)] = dll.encode()
        name_rva += 0x10
    if imports:
        struct.pack_into("<II", data, dd + 8 * 1, imp_rva, 20 * (len(imports) + 1))
    if delay:
        struct.pack_into("<II", data, dd + 8 * 13, dly_rva, 32 * (len(delay) + 1))
    return bytes(data)


class Stack:
    """A temp tree: an engine dir with an exe and its DLLs, a config.yaml
    that runs it through a macro, and a manifest that pins it."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="yamadori-engines-")
        self.bin = os.path.join(self.root, "eng", "build", "bin")
        os.makedirs(self.bin)
        self.write("srv.exe", make_pe(["KERNEL32.dll", "srv-impl.dll"]))
        self.write("srv-impl.dll", make_pe(["ggml.dll"], delay=["cudart.dll"]))
        self.write("ggml.dll", make_pe(["msvcp140.dll"]))
        self.write("cudart.dll", b"not a PE, but loaded")
        self.write("unrelated.exe", make_pe(["srv-impl.dll"]))
        self.exe = os.path.join(self.bin, "srv.exe").replace("\\", "/")
        self.other = os.path.join(self.root, "other.exe").replace("\\", "/")
        with open(self.other, "wb") as f:
            f.write(b"x")
        self.config = os.path.join(self.root, "config.yaml")
        with open(self.config, "w", encoding="utf-8") as f:
            f.write(
                "macros:\n"
                f"  base: \"{os.path.dirname(self.bin).replace(os.sep, '/')}\"\n"
                "  server: \"${base}/bin/srv.exe\"\n"
                "  models: \"C:/models\"\n"
                "models:\n"
                "  chat:\n"
                "    cmd: |\n"
                "      # a comment line first\n"
                "      ${server}\n"
                "      --port ${PORT}\n"
                "      -m ${models}/x.gguf\n"
                "  embed:\n"
                "    cmd: |\n"
                "      ${server}   # trailing comment\n"
                "      --embeddings\n")
        self.manifest_path = os.path.join(self.root, "manifest.yaml")
        self.pin()

    def write(self, name: str, data: bytes) -> None:
        with open(os.path.join(self.bin, name), "wb") as f:
            f.write(data)

    def pin(self, files: dict | None = None) -> None:
        import yaml
        self.manifest = {"engines": {"eng": {
            "kind": "source",
            "base_commit": "0" * 40,
            "shipped": {"path": self.exe,
                        "files": files if files is not None
                        else be.describe(self.exe)}}}}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.manifest, f)

    def verify(self, extra=None):
        return be.verify_deploy(self.config, self.manifest_path,
                                extra if extra is not None else [])

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------------

def test_pe_imports_read_both_tables():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.dll")
        with open(p, "wb") as f:
            f.write(make_pe(["B.dll", "kernel32.dll"], delay=["C.DLL"]))
        got = be.pe_imports(p)
        check(got == ["b.dll", "kernel32.dll", "c.dll"],
              "imports and delay-imports are read, lower-cased", str(got))
        q = os.path.join(d, "not.dll")
        with open(q, "wb") as f:
            f.write(b"MZ" + b"\0" * 100)
        check(be.pe_imports(q) == [], "a file that is not a PE has no imports")
        check(be.pe_imports(os.path.join(d, "missing.dll")) == [],
              "a missing file has no imports")


def test_closure_is_what_the_exe_loads_beside_it():
    s = Stack()
    try:
        got = be.closure(s.exe)
        check(got == ["srv.exe", "srv-impl.dll", "ggml.dll", "cudart.dll"],
              "closure follows imports transitively, delay-loads included, "
              "and skips system DLLs and unrelated files", str(got))
    finally:
        s.close()


def test_config_binaries_expand_macros():
    s = Stack()
    try:
        got = be.config_binaries(s.config)
        want = [("chat", s.exe.replace("\\", "/")), ("embed", s.exe)]
        check([(m, be.norm(p)) for m, p in got]
              == [(m, be.norm(p)) for m, p in want],
              "the first word of each cmd, nested macros expanded, comment "
              "lines and trailing comments ignored", str(got))
    finally:
        s.close()


def test_everything_matching_passes():
    s = Stack()
    try:
        problems, ok = s.verify()
        check(not problems and len(ok) == 1,
              "a config whose binaries all match the manifest passes, "
              "one line per binary (two models, one binary)",
              f"{problems} {ok}")
        check("chat, embed" in (ok[0] if ok else ""),
              "the ok line names every model that uses the binary", str(ok))
    finally:
        s.close()


def test_one_changed_byte_fails():
    s = Stack()
    try:
        p = os.path.join(s.bin, "ggml.dll")
        with open(p, "rb") as f:
            data = bytearray(f.read())
        data[-1] ^= 1
        with open(p, "wb") as f:
            f.write(data)
        problems, _ = s.verify()
        text = " ".join(problems)
        check(len(problems) == 1 and "ggml.dll sha256" in text,
              "one flipped byte in a loaded DLL fails, naming the DLL", text)
        check("engine eng" in text and "build_engine.py eng" in text,
              "the failure names the engine and the remedy", text)
    finally:
        s.close()


def test_size_and_missing_fail():
    s = Stack()
    try:
        s.write("ggml.dll", make_pe(["msvcp140.dll"]) + b"more")
        problems, _ = s.verify()
        check(any("ggml.dll size" in p for p in problems),
              "a changed size fails", str(problems))
        os.remove(os.path.join(s.bin, "cudart.dll"))
        problems, _ = s.verify()
        check(any("cudart.dll missing" in p for p in problems),
              "a pinned file that is gone fails", str(problems))
    finally:
        s.close()


def test_binary_not_in_manifest_fails():
    s = Stack()
    try:
        problems, ok = s.verify(extra=[("llama-swap", s.other)])
        check(len(problems) == 1 and "not in engines/manifest.yaml" in problems[0]
              and "llama-swap" in problems[0],
              "a binary no engine records as shipped fails, naming who uses it",
              str(problems))
        check(len(ok) == 1, "the pinned binary still passes", str(ok))
    finally:
        s.close()


def test_unpinned_loaded_dll_fails():
    s = Stack()
    try:
        files = be.describe(s.exe)
        files.pop("cudart.dll")
        s.pin(files)
        problems, _ = s.verify()
        check(any("cudart.dll is loaded but not pinned" in p for p in problems),
              "a DLL the binary loads from its directory must be pinned",
              str(problems))
    finally:
        s.close()


def test_missing_manifest_or_config_fails():
    s = Stack()
    try:
        problems, _ = be.verify_deploy(s.config, s.manifest_path + ".nope", [])
        check(problems and "missing" in problems[0],
              "no manifest is a failure, not a pass", str(problems))
        problems, _ = be.verify_deploy(s.config + ".nope", s.manifest_path, [])
        check(problems and "missing" in problems[0],
              "no config.yaml is a failure, not a pass", str(problems))
    finally:
        s.close()


def test_deploy_check_engines_check():
    s = Stack()
    try:
        good, lines = deploy_check.engines_check(s.config, s.manifest_path, [])
        check(good and lines and lines[0].startswith("ok"),
              "deploy_check.engines_check passes a matching stack", str(lines))
        s.write("srv.exe", make_pe(["KERNEL32.dll", "srv-impl.dll"]) + b"x")
        good, lines = deploy_check.engines_check(s.config, s.manifest_path, [])
        check(not good and any(x.startswith("FAIL") for x in lines),
              "deploy_check.engines_check fails a changed exe", str(lines))
    finally:
        s.close()


def test_deploy_check_fails_before_waiting():
    calls = []
    saved = (deploy_check.engines_check, deploy_check.services_up,
             deploy_check.record)
    try:
        deploy_check.engines_check = lambda: (False, ["FAIL  (test stub) x.exe is not pinned"])

        def no_services():
            calls.append("services_up")
            raise AssertionError("must not wait for services")
        deploy_check.services_up = no_services
        deploy_check.record = lambda rc, verdict, secs, note="": calls.append(
            ("record", rc, verdict, note))
        rc = deploy_check.main(["--key-file", "unused"])
    finally:
        (deploy_check.engines_check, deploy_check.services_up,
         deploy_check.record) = saved
    check(rc == 1, "deploy_check exits 1 (NOT GOOD) on an unpinned engine",
          str(rc))
    check("services_up" not in calls,
          "it fails before waiting for the stack", str(calls))
    check(any(isinstance(c, tuple) and c[1] == 1 and c[2] == "NOT GOOD"
              and "manifest" in c[3] for c in calls),
          "the verdict is recorded with the reason", str(calls))


def test_out_dir_safety():
    with tempfile.TemporaryDirectory() as d:
        shipped = os.path.join(d, "checkout")
        os.makedirs(shipped)
        prot = [be.norm(shipped)]

        def refused(out, running=None) -> bool:
            try:
                be.check_out_dir(out, prot, running or [])
                return False
            except be.BuildError:
                return True

        check(refused(os.path.join(shipped, "build2")),
              "refuses a directory inside a shipped engine's checkout")
        check(refused(d), "refuses a directory around one")
        busy = os.path.join(d, "busy")
        os.makedirs(busy)
        with open(os.path.join(busy, "f"), "w") as f:
            f.write("x")
        check(refused(busy), "refuses a non-empty directory")
        fresh = os.path.join(d, "fresh")
        check(refused(fresh, [os.path.join(fresh, "bin", "x.exe")]),
              "refuses a directory a running process uses")
        check(not refused(fresh), "accepts a new directory")

        m = {"engines": {"e": {"shipped": {
            "path": os.path.join(shipped, "build", "bin", "x.exe")}}}}
        got = be.protected_dirs(m, os.path.join(d, "no-config.yaml"))
        check(be.norm(shipped) in got,
              "a build*/bin binary protects its whole checkout", str(got))


def test_pending_patch_refuses_before_anything():
    import yaml
    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "m.yaml")
        with open(man, "w", encoding="utf-8") as f:
            yaml.safe_dump({"engines": {"e": {
                "kind": "source", "base_commit": "0" * 40,
                "patches": [{"file": "0001-x.patch", "status": "pending"}]}}}, f)
        out = os.path.join(d, "out")
        rc = be.main(["e", "--manifest", man, "--out", out,
                      "--config", os.path.join(d, "none.yaml")])
        check(rc == 2 and not os.path.exists(out),
              "a pending patch refuses the build and creates nothing", str(rc))


def test_engine_hash_names_the_source_not_the_shipment():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "e"))
        patch = os.path.join(d, "e", "0001-a.patch")
        with open(patch, "w") as f:
            f.write("one")
        m = {"toolchains": {"t": {"cmake": "x"}}, "engines": {"e": {
            "toolchain": "t", "base_commit": "a" * 40,
            "patches": [{"file": "0001-a.patch"}],
            "shipped": {"path": "p", "files": {"x": {"sha256": "1"}}}}}}
        h0 = be.engine_hash(m, "e", d)
        m["engines"]["e"]["shipped"]["files"]["x"]["sha256"] = "2"
        check(be.engine_hash(m, "e", d) == h0,
              "recording a new shipped hash does not rename the build")
        with open(patch, "w") as f:
            f.write("two")
        check(be.engine_hash(m, "e", d) != h0, "a changed patch does")
        m["toolchains"]["t"]["cmake"] = "y"
        check(be.engine_hash(m, "e", d) != h0, "a changed toolchain does")


def test_flag_and_cache_parsing():
    got = dict(be.parse_flags(["-DA=ON", '-DB:STRING="86;120"', "-G", "x"]))
    check(got == {"A": "ON", "B": "86;120"}, "configure flags parse", str(got))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "CMakeCache.txt")
        with open(p, "w") as f:
            f.write("// c\n#c\nGGML_NATIVE:BOOL=ON\nX:UNINITIALIZED=86;120\n")
        c = be.read_cache(p)
        check(c.get("GGML_NATIVE") == "ON" and c.get("X") == "86;120",
              "CMakeCache entries parse", str(c))
        n = os.path.join(d, "build.ninja")
        with open(n, "w") as f:
            f.write("build ggml\\src\\ggml-cpu.dir\\ggml-cpu.c.obj: CC x.c\n"
                    "  DEFINES = -DGGML_AVX2 -DGGML_FMA\n"
                    "  FLAGS = /O2 /arch:AVX2\n")
        flags, defines = be.ninja_cpu_flags(n, "ggml-cpu.c.obj")
        check("/arch:AVX2" in flags.split() and "GGML_FMA" in defines.split(),
              "the native CPU flags are read from build.ninja",
              f"{flags!r} {defines!r}")


def test_previous_blocks_stay_verifiable():
    s = Stack()
    try:
        files = be.describe(s.exe)
        s.manifest["engines"]["eng"]["shipped"] = {"path": s.other,
                                                   "files": be.describe(s.other)}
        s.manifest["engines"]["eng"]["previous"] = [
            {"rebuildable": False, "path": s.exe, "files": files}]
        import yaml
        with open(s.manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(s.manifest, f)
        problems, ok = s.verify()
        check(not problems and ok and "eng (previous)" in ok[0],
              "a binary pinned only as `previous` still verifies, labelled so",
              f"{problems} {ok}")
        s.write("ggml.dll", make_pe(["msvcp140.dll"]) + b"x")
        problems, _ = s.verify()
        check(problems and "build_engine.py eng " in problems[0],
              "a changed `previous` binary fails, and the remedy names the "
              "engine, not the label", str(problems))
        del s.manifest["engines"]["eng"]["previous"][0]["files"]
        with open(s.manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(s.manifest, f)
        problems, _ = s.verify()
        check(problems and "not in engines/manifest.yaml" in problems[0],
              "a `previous` block without files pins nothing", str(problems))
    finally:
        s.close()


def test_llama_cpp_builds_without_the_ui():
    base = {"kind": "source", "base_commit": "0" * 40,
            "upstream": "https://github.com/x/llama.cpp",
            "configure": ["-DLLAMA_BUILD_UI=OFF"]}
    check(be.ui_policy_problem("e", base) is not None,
          "LLAMA_BUILD_UI=OFF alone is refused (the HF prebuilt would still "
          "be downloaded and embedded)")
    ok = dict(base, configure=list(be.NO_UI_FLAGS))
    check(be.ui_policy_problem("e", ok) is None, "both flags OFF passes")
    check(be.ui_policy_problem("e", dict(base, embeds_ui="reproduces X")) is None,
          "an entry that says why it embeds a UI passes")
    sd = dict(base, upstream="https://github.com/leejet/stable-diffusion.cpp")
    check(be.ui_policy_problem("e", sd) is None,
          "the rule is for llama.cpp engines only")
    import yaml
    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "m.yaml")
        with open(man, "w", encoding="utf-8") as f:
            yaml.safe_dump({"engines": {"e": base}}, f)
        out = os.path.join(d, "out")
        rc = be.main(["e", "--manifest", man, "--out", out,
                      "--config", os.path.join(d, "none.yaml")])
        check(rc == 2 and not os.path.exists(out),
              "the builder refuses such an entry and creates nothing", str(rc))


def test_committed_manifest_is_well_formed():
    m = be.load_yaml(be.MANIFEST)
    engines = m.get("engines") or {}
    check(engines, "engines/manifest.yaml lists engines")
    for name, e in engines.items():
        check(be.ui_policy_problem(name, e) is None,
              f"{name}: follows the no-UI default (or says why not)",
              str(be.ui_policy_problem(name, e)))
        check(len(str(e.get("base_commit", ""))) == 40,
              f"{name}: base commit is a full 40-character SHA")
        s = e.get("shipped") or {}
        check(s.get("path") and s.get("files"),
              f"{name}: records a shipped path and its files")
        out = str(e.get("output", "")).split("/")[-1]
        check(out in (s.get("files") or {}),
              f"{name}: the output binary is among the pinned files", out)
        check(all(len(str(v.get("sha256", ""))) == 64
                  for v in (s.get("files") or {}).values()),
              f"{name}: every pinned file has a SHA-256")
        for p in e.get("patches") or []:
            path = os.path.join(be.PATCHES, name, p["file"])
            check(p.get("status") != "pending",
                  f"{name}: {p['file']} is not pending")
            check(os.path.exists(path) and be.sha256_file(path) == p.get("sha256"),
                  f"{name}: {p['file']} is on disk with its recorded hash")
        for key, inp in (e.get("inputs") or {}).items():
            if inp.get("path"):
                p = os.path.join(ROOT, inp["path"])
                check(os.path.exists(p) and be.sha256_file(p) == inp["sha256"],
                      f"{name}: input {key} is on disk with its recorded hash")
            else:
                check(len(str(inp.get("sha256", ""))) == 64 and inp.get("url"),
                      f"{name}: input {key} is pinned by URL and SHA-256")


# --------------------------------------------------------------------------
# vendored source (scripts/engine_vendor.py; docs/ENGINES.md "Vendored source")
# --------------------------------------------------------------------------

import engine_vendor as ev  # noqa: E402


def _git(args, cwd, **kw):
    return ev.git(args, cwd, env=dict(ev._IDENT), **kw)


def test_tree_hash_is_gits_own():
    """tree_hash_of_dir is the id git gives the same files (all 100644)."""
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "w")
        files = {"a.b": b"x\n", "a/c.txt": b"y\r\nz\r\n", "a-b": b"",
                 "a/b/deep.bin": bytes(range(256)), "Z": b"upper\n",
                 "a0/x": b"1"}
        for p, data in files.items():
            full = os.path.join(w, *p.split("/"))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)
        g = os.path.join(d, "g.git")
        _git(["init", "-q", "--bare", g], d)
        env = {"GIT_DIR": g, "GIT_WORK_TREE": w,
               "GIT_INDEX_FILE": os.path.join(d, "idx")}
        ev.git(["add", "-A", "-f", "."], w, env=env)
        want = ev.out(ev.git(["write-tree"], w, env=env))
        got, n, size = ev.tree_hash_of_dir(w)
        check(got == want and n == len(files),
              "the vendored tree hash equals git's write-tree for the same "
              "files (sorting, nesting, CRLF and binary bytes kept)",
              f"{got} vs {want}")


def test_exclusion_rule():
    pats = [".github/", "docs/", "models/", "!models/templates/", "*.gguf"]
    cases = {".github/workflows/x.yml": True, "docs/ops/CUDA.csv": True,
             "docs": True, "src/docs/x.md": False, "models/vocab.gguf": True,
             "models/templates/a.jinja": False, "src/models/qwen.cpp": False,
             "tests/x.gguf": True, "docs/LICENSE": False,
             "models/NOTICE.txt": False, "LICENSE": False}
    bad = {p: ev.excluded(p, pats) for p, want in cases.items()
           if ev.excluded(p, pats) != want}
    check(not bad, "the exclusion rule: anchored dirs, `!` puts back, the "
          "last match wins, licence and notice files always kept", str(bad))
    attrs = {"third/webp/.gitattributes": b"*.bat text eol=crlf\n*.pdf -text\n",
             "third/webm/.gitattributes": b"*.sln eol=crlf\n"}
    paths = list(attrs) + ["third/webp/gradlew.bat", "third/webp/x/y.bat",
                           "third/webm/a.sln", "top.bat", "third/webp/doc.pdf"]
    got = ev.eol_converted(paths, lambda p: attrs[p])
    check(got == ["third/webp/gradlew.bat", "third/webp/x/y.bat"],
          "eol_converted: only paths a nested .gitattributes sets `text` for, "
          "under that directory", str(got))


class Upstream:
    """A throwaway upstream repo, a patch series and a manifest that vendors
    it, all in a temp dir; `vendor`, `check` and `update` run against it."""

    def __init__(self):
        self.d = tempfile.mkdtemp(prefix="yamadori-vendor-")
        self.up = os.path.join(self.d, "up")
        os.makedirs(self.up)
        _git(["init", "-q", "."], self.up)
        self.put({"LICENSE": "MIT\n", "src/a.c": "".join(
            f"int line{i};\n" for i in range(40)),
            "src/b.c": "int b;\n", "docs/big.md": "doc\n",
            "models/v.gguf": "GGUF", "models/templates/t.jinja": "{{x}}\n",
            ".github/ci.yml": "on: push\n"})
        self.base = self.commit("base")
        # the patch: line 20 of src/a.c, exported like `git format-patch`
        work = os.path.join(self.d, "patchwork")
        _git(["clone", "-q", self.up, work], self.d)
        p = os.path.join(work, "src", "a.c")
        with open(p, "rb") as f:
            text = f.read().replace(b"int line20;", b"int line20 = 20; /* ours */")
        with open(p, "wb") as f:
            f.write(text)
        diff = ev.out(_git(["diff"], work)) + "\n"
        self.patches = os.path.join(self.d, "patches")
        os.makedirs(os.path.join(self.patches, "eng"))
        self.patch = os.path.join(self.patches, "eng", "0001-ours.patch")
        with open(self.patch, "wb") as f:
            f.write(("From: Test <test@example.invalid>\nSubject: [PATCH] ours\n"
                     "\n---\n src/a.c | 2 +-\n 1 file changed, 1 insertion(+), 1 "
                     "deletion(-)\n\n" + diff).encode())
        self.src = os.path.join(self.d, "src")
        self.manifest = os.path.join(self.d, "manifest.yaml")
        self.write_manifest()

    def put(self, files: dict) -> None:
        for p, text in files.items():
            full = os.path.join(self.up, *p.split("/"))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(text.encode())

    def commit(self, msg: str) -> str:
        _git(["add", "-A"], self.up)
        _git(["commit", "-q", "-m", msg], self.up)
        return ev.out(_git(["rev-parse", "HEAD"], self.up))

    def write_manifest(self, base: str | None = None) -> None:
        import yaml
        with open(self.manifest, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump({
                "vendoring": {"cache": os.path.join(self.d, "cache"),
                              "rules": {"t": [".github/", "docs/", "models/",
                                              "!models/templates/"]}},
                "engines": {"eng": {
                    "kind": "source",
                    "upstream": "file:///" + self.up.replace("\\", "/"),
                    "base_commit": base or self.base,
                    "patches": [{"file": "0001-ours.patch",
                                 "sha256": be.sha256_file(self.patch)}],
                    "vendor": {"rule": "t"}}}}, sort_keys=False))

    def cache(self):
        return ev.Cache(os.path.join(self.d, "cache"), log=lambda *_: None)

    def vendor(self):
        return ev.vendor(self.manifest, "eng", self.patches, self.src,
                         self.cache(), log=lambda *_: None)

    def check(self, derive=False):
        return ev.check(be.load_yaml(self.manifest), "eng", self.patches,
                        self.src, derive_base=derive)

    def close(self):
        shutil.rmtree(self.d, onerror=be._force_remove)


def test_vendor_then_check_offline():
    u = Upstream()
    try:
        rec = u.vendor()
        tree = os.path.join(u.src, "eng")
        present = sorted(ev.scan_dir(tree))
        check(present == ["LICENSE", "models/templates/t.jinja", "src/a.c",
                          "src/b.c"],
              "vendor writes base + patch minus the excluded paths", str(present))
        with open(os.path.join(tree, "src", "a.c"), "rb") as f:
            check(b"int line20 = 20; /* ours */" in f.read(),
                  "the vendored tree carries the patch")
        m = be.load_yaml(u.manifest)["engines"]["eng"]["vendor"]
        check(m.get("tree") == rec["tree"] and m.get("commit") == u.base
              and m.get("licences") == ["LICENSE"] and m.get("excluded_files") == 3,
              "the manifest records origin, tree hash, licences, exclusions",
              str(m))
        check(u.check() == [], "check passes the tree it wrote", str(u.check()))
        check(u.check(derive=True) == [],
              "check --derive takes the series back out offline and lands on "
              "the recorded base", str(u.check(derive=True)))
        again = u.vendor()
        check(again["tree"] == rec["tree"], "vendor is deterministic")

        p = os.path.join(tree, "src", "b.c")
        with open(p, "ab") as f:
            f.write(b"// edited in place\n")
        got = u.check()
        check(any("hashes" in x and "edited" in x for x in got),
              "a hand edit in the vendored tree fails, with the remedy", str(got))
        u.vendor()
        with open(os.path.join(tree, "docs.md"), "wb") as f:
            f.write(b"x")
        os.makedirs(os.path.join(tree, "docs"))
        with open(os.path.join(tree, "docs", "new.md"), "wb") as f:
            f.write(b"x")
        got = u.check()
        check(any("exclusion rule drops are present" in x for x in got),
              "a file the exclusion rule drops fails", str(got))
        u.vendor()
        os.remove(os.path.join(tree, "LICENSE"))
        got = u.check()
        check(any("licence files missing" in x for x in got)
              and any("no licence file at the tree's root" in x for x in got),
              "a missing upstream licence fails", str(got))
        u.vendor()
        with open(u.patch, "ab") as f:
            f.write(b"\n")
        ev.set_patch_sha(u.manifest, "eng", "0001-ours.patch",
                         be.sha256_file(u.patch))
        got = u.check()
        check(any("patch series changed since it was vendored" in x for x in got),
              "a changed patch series fails until the tree is re-vendored",
              str(got))
        u.vendor()
        check(u.check() == [], "re-vendoring clears it", str(u.check()))
    finally:
        u.close()


def test_materialize_checks_out_like_the_originals():
    with tempfile.TemporaryDirectory() as d:
        v = os.path.join(d, "v")
        os.makedirs(os.path.join(v, "sub"))
        with open(os.path.join(v, "a.txt"), "wb") as f:
            f.write(b"one\ntwo\n")
        with open(os.path.join(v, "sub", "b.bin"), "wb") as f:
            f.write(b"\0\n\1\n")
        with open(os.path.join(v, ".gitignore"), "wb") as f:
            f.write(b"*.bin\n")
        dest = os.path.join(d, "out", "src")
        n = ev.materialize(v, dest, autocrlf=True, scratch=d)
        with open(os.path.join(dest, "a.txt"), "rb") as f:
            a = f.read()
        with open(os.path.join(dest, "sub", "b.bin"), "rb") as f:
            b = f.read()
        check(n == 3 and a == b"one\r\ntwo\r\n" and b == b"\0\n\1\n",
              "the offline checkout converts text to CRLF as the originals' "
              "autocrlf checkout did, keeps binary bytes, and keeps files the "
              "tree's own .gitignore names", f"{n} {a!r} {b!r}")
        check(not os.path.exists(os.path.join(dest, ".git"))
              and not [x for x in os.listdir(d) if x.startswith("vendor-stage")],
              "no .git is left in the checkout or beside it")


def test_update_rebases_reports_conflicts_and_writes():
    u = Upstream()
    try:
        u.vendor()
        before = open(u.manifest, "rb").read()
        patch_before = open(u.patch, "rb").read()
        # upstream moves elsewhere: the series applies as it is
        u.put({"src/b.c": "int b2;\n"})
        clean = u.commit("elsewhere")
        rep = ev.update(u.manifest, "eng", clean, write=False, patches_dir=u.patches,
                        src_root=u.src, cache=u.cache(), log=lambda *_: None)
        check(not rep["conflict"] and rep["patches"][0]["status"] == "unchanged"
              and open(u.manifest, "rb").read() == before,
              "a dry run onto a base the series still applies to reports "
              "`unchanged` and writes nothing", str(rep))
        # upstream edits the very line we patch: a conflict, never resolved
        u.put({"src/a.c": open(os.path.join(u.up, "src", "a.c")).read()
               .replace("int line20;", "long line20;")})
        clash = u.commit("clash")
        rep = ev.update(u.manifest, "eng", clash, write=True, patches_dir=u.patches,
                        src_root=u.src, cache=u.cache(), log=lambda *_: None)
        check(rep["conflict"] and rep["conflict"]["files"] == ["src/a.c"]
              and "0001-ours.patch" in rep["conflict"]["patch"],
              "a conflicting base stops the rebase and names the patch and file",
              str(rep["conflict"]))
        check(open(u.manifest, "rb").read() == before
              and open(u.patch, "rb").read() == patch_before and not rep["written"],
              "a conflict writes nothing, even with --write")
        # upstream edits the context next to our line: git's 3-way rebase
        # carries the patch over and the file is rewritten
        _git(["reset", "-q", "--hard", clean], u.up)
        u.put({"src/a.c": open(os.path.join(u.up, "src", "a.c")).read()
               .replace("int line17;", "int line17 = 17;")
               .replace("int line23;", "int line23 = 23;")})
        near = u.commit("near")
        rep = ev.update(u.manifest, "eng", near, write=True, patches_dir=u.patches,
                        src_root=u.src, cache=u.cache(), log=lambda *_: None)
        m = be.load_yaml(u.manifest)["engines"]["eng"]
        new_patch = open(u.patch, "rb").read()
        check(not rep["conflict"] and rep["written"]
              and rep["patches"][0]["status"] == "rebased"
              and m["base_commit"] == near
              and m["patches"][0]["sha256"] == be.sha256_file(u.patch)
              and new_patch.startswith(b"From: Test <test@example.invalid>\n"
                                       b"Subject: [PATCH] ours\n"),
              "a rebase that changes a patch's context rewrites the patch file "
              "(its header kept), the base and sha256 in the manifest",
              str(rep))
        with open(os.path.join(u.src, "eng", "src", "a.c"), "rb") as f:
            a = f.read()
        check(b"int line20 = 20; /* ours */" in a and b"int line23 = 23;" in a
              and u.check(derive=True) == [],
              "and re-vendors: the new tree is the new base + the series",
              str(u.check(derive=True)))
    finally:
        u.close()


def test_committed_vendored_trees():
    """The real trees under engines/src: each is exactly what the manifest
    records, carries its licences, and holds nothing its rule excludes."""
    m = be.load_yaml(be.MANIFEST)
    engines = m.get("engines") or {}
    vendored = [n for n, e in engines.items() if (e or {}).get("vendor")]
    check(vendored, "the manifest vendors engines", str(vendored))
    live = [n for n, e in engines.items()
            if e.get("kind", "source") == "source" and e.get("config_refs")]
    check(set(live) <= set(vendored),
          "every source engine config.yaml runs (config_refs) is vendored",
          str(sorted(set(live) - set(vendored))))
    for n in vendored:
        rec = engines[n]["vendor"]
        check(rec.get("path") == f"engines/src/{n}",
              f"{n}: vendored at engines/src/{n}")
        check(len(str(rec.get("commit"))) == 40 and rec.get("repo")
              and len(str(rec.get("upstream_tree"))) == 40,
              f"{n}: origin recorded (repo, commit, upstream tree)")
        problems = ev.check(m, n)
        check(not problems, f"{n}: tree hash, patch series, licences and "
              "exclusion rule hold", "; ".join(problems))
    # Once committed, every vendored file must be tracked: upstream's own
    # .gitignore files (and ours) would silently drop some from a plain
    # `git add` (docs/ENGINES.md: `git add -f engines/src`).
    r = ev.git(["ls-files", "engines/src"], ROOT, check=False)
    if r.returncode == 0 and r.stdout.strip():
        r2 = ev.git(["ls-files", "-o", "-i", "--exclude-standard", "engines/src"],
                    ROOT, check=False)
        left = ev.out(r2).splitlines()
        check(not left, "every vendored file is committed (git add -f)",
              f"{len(left)} ignored, e.g. {left[:3]}")
    attr = ev.out(ev.git(["check-attr", "text", "--",
                          "engines/src/x/CMakeLists.txt"], ROOT, check=False))
    check(attr.endswith("unset"), ".gitattributes stores engines/src without "
          "end-of-line conversion (-text)", attr)


def main() -> int:
    for fn in (test_pe_imports_read_both_tables,
               test_closure_is_what_the_exe_loads_beside_it,
               test_config_binaries_expand_macros,
               test_everything_matching_passes,
               test_one_changed_byte_fails,
               test_size_and_missing_fail,
               test_binary_not_in_manifest_fails,
               test_unpinned_loaded_dll_fails,
               test_missing_manifest_or_config_fails,
               test_deploy_check_engines_check,
               test_deploy_check_fails_before_waiting,
               test_out_dir_safety,
               test_pending_patch_refuses_before_anything,
               test_engine_hash_names_the_source_not_the_shipment,
               test_previous_blocks_stay_verifiable,
               test_llama_cpp_builds_without_the_ui,
               test_flag_and_cache_parsing,
               test_committed_manifest_is_well_formed,
               test_tree_hash_is_gits_own,
               test_exclusion_rule,
               test_vendor_then_check_offline,
               test_materialize_checks_out_like_the_originals,
               test_update_rebases_reports_conflicts_and_writes,
               test_committed_vendored_trees):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
