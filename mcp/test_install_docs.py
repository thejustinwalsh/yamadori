#!/usr/bin/env python
"""The install path stays true: config.example.yaml, make_config.py, requirements.txt, scripts/install.ps1, the
launchers' local settings, the README's links.

What it gates (offline; reads files, writes only to a temp directory, never touches a port or the live state):

  config.example.yaml   every @@PLACEHOLDER@@ is one make_config.py knows; it renders to YAML with the models the
                        stack serves; every model file its commands name is a path models/manifest.yaml records
                        (so the example cannot drift from the manifest); no personal path, UUID or address in it
  make_config.py        renders, refuses an unresolved placeholder, never overwrites without --force
  requirements.txt      every third-party import in mcp/ and scripts/ is a package named there, and every pin is the
                        version requirements.lock.txt records
  scripts/install.ps1   its model sets name real manifest artifacts at the manifest's sizes; its engines are real
                        engine entries; it parses (PowerShell's own parser, when powershell is present)
  launchers             start-stack.bat / watchdog.ps1 / install-autostart.ps1 name no home directory, read stack.env,
                        and fall back to the same values they had before it existed
  build_engine.py       --portable waives only the toolchain-identity checks, never a source or configuration check
  fetch_models.py       --models-dir places a file where the manifest's path says
  README.md             every relative link and image resolves

Why: a doc or a script that drifts from the code is how this repo shipped a "128k" label on a 147,456 pool; the install
path is read by people with no one to ask.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import yaml  # noqa: E402

import make_config as mc  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


DUMMY = {
    "MODELS_DIR": "D:/models", "MAIN_MODEL": mc.DEFAULT_MAIN_MODEL,
    "MAIN_GPU_UUID": "GPU-00000000-0000-0000-0000-000000000001",
    "SECOND_GPU_UUID": "GPU-00000000-0000-0000-0000-000000000002",
    "SERVER_NUDGE": "D:/e/a/llama-server.exe", "SERVER_DECIDE": "D:/e/a/llama-server.exe",
    "SERVER_PRISM": "D:/e/a/llama-server.exe", "SERVER_MIRAI_S": "D:/e/m/llama-server.exe",
    "SERVER_UPSTREAM_MOE": "D:/e/f/llama-server.exe", "SD_SERVER": "D:/e/s/sd-server.exe",
}


def test_config_example() -> None:
    text = read("config.example.yaml")
    names = mc.placeholders_in(text)
    check(set(names) == set(mc.PLACEHOLDERS),
          "config.example.yaml: its placeholders are exactly make_config.PLACEHOLDERS",
          f"in the file {names}; known {sorted(mc.PLACEHOLDERS)}")
    rendered = mc.render(text, DUMMY)
    cfg = yaml.safe_load(rendered)
    ids = set((cfg.get("models") or {}))
    want = {"bonsai", "flash-next", "mirai-s", "bonsai-vision", "bonsai-a4000", "imagegen", "imagegen-turbo",
            "embeddings"}
    check(want <= ids, "config.example.yaml: renders to YAML with every model the stack serves",
          f"missing {sorted(want - ids)}")
    check("bonsai-q4kv" not in ids and "critic-disabled" not in ids and "profiles" not in cfg,
          "config.example.yaml: the dead trial entries are not in it")
    groups = ((cfg.get("routing") or {}).get("router") or {}).get("settings", {}).get("groups", {})
    members = {m for g in groups.values() for m in g.get("members", [])}
    check(members <= ids, "config.example.yaml: every routing-group member is a model entry",
          str(sorted(members - ids)))
    check(cfg["models"]["bonsai"].get("aliases") == ["bonsai-agent"],
          "config.example.yaml: `bonsai-agent` is still an alias of bonsai")

    bad = []
    for pat, why in ((r"jwals", "a home directory"), (r"C:[/\\]Users", "an absolute user path"),
                     (r"GPU-[0-9a-f]{8}-[0-9a-f]{4}", "a GPU UUID"), (r"thejustinwalsh", "the operator's domain"),
                     (r"10\.242\.", "a private address")):
        if re.search(pat, text):
            bad.append(why)
    check(not bad, "config.example.yaml: no personal path, UUID, domain or address", ", ".join(bad))

    # every model-file argument resolves to a path the manifest records (verify_artifacts' own reader)
    import verify_artifacts as va
    with tempfile.TemporaryDirectory() as d:
        models = os.path.join(d, "models").replace("\\", "/")
        vals = dict(DUMMY, MODELS_DIR=models)
        p = os.path.join(d, "config.yaml")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(mc.render(text, vals))
        refs = va.config_files(p)
        m = va.load_manifest(va.MANIFEST, None)
        macros = {"models": models}
        recorded = {va.norm(va.resolve(str(a["path"]), macros)) for a in m["artifacts"] if a.get("path")}
        unrecorded = sorted({f"{e}: {os.path.basename(path)}" for e, _flag, path in refs
                             if va.norm(os.path.normpath(path)) not in recorded})
        check(len(refs) >= 12 and not unrecorded,
              "config.example.yaml: every model file its commands name is recorded in models/manifest.yaml",
              f"{len(refs)} references; unrecorded: {unrecorded}")


def test_make_config() -> None:
    text = read("config.example.yaml")
    try:
        mc.render(text, {k: v for k, v in DUMMY.items() if k != "SD_SERVER"})
        check(False, "make_config: an unresolved placeholder is refused")
    except ValueError as e:
        check("SD_SERVER" in str(e), "make_config: an unresolved placeholder is refused", str(e))
    try:
        mc.render(text, dict(DUMMY, NOT_A_PLACEHOLDER="x"))
        check(False, "make_config: a value for an unknown placeholder is refused")
    except ValueError:
        check(True, "make_config: a value for an unknown placeholder is refused")
    d = mc.defaults("D:\\engines", "D:\\m")
    check(d["SERVER_NUDGE"] == "D:/engines/llama-bonsai2-ada/src/build/bin/llama-server.exe"
          and d["SD_SERVER"].endswith("sd-cpp/src/build/bin/sd-server.exe")
          and d["MODELS_DIR"] == "D:/m",
          "make_config: engine paths default to <root>/<engine>/src/build/bin, with forward slashes", str(d))
    gpus = mc.detect_gpus("GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-aaaaaaaa-0000-0000-0000-000000000001)\n"
                          "GPU 1: NVIDIA RTX A4000 (UUID: GPU-bbbbbbbb-0000-0000-0000-000000000002)\n")
    check([g["name"] for g in gpus] == ["NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX A4000"]
          and gpus[1]["uuid"].startswith("GPU-bbbbbbbb"),
          "make_config: nvidia-smi -L is read into (index, name, uuid), in order", str(gpus))
    with tempfile.TemporaryDirectory() as d2:
        out = os.path.join(d2, "config.yaml")
        argv = ["--out", out, "--models-dir", "D:/m", "--engines-root", "D:/e",
                "--main-gpu", DUMMY["MAIN_GPU_UUID"], "--second-gpu", DUMMY["SECOND_GPU_UUID"]]
        rc1 = mc.main(argv)
        first = open(out, encoding="utf-8").read() if os.path.exists(out) else ""
        with open(out, "w", encoding="utf-8") as f:
            f.write("mine")
        rc2 = mc.main(argv)
        kept = open(out, encoding="utf-8").read()
        rc3 = mc.main(argv + ["--force"])
        baks = [n for n in os.listdir(d2) if n.startswith("config.yaml.bak-")]
        check(rc1 == 0 and "@@" not in first and yaml.safe_load(first) is not None,
              "make_config: writes a config with no placeholder left")
        check(rc2 == 2 and kept == "mine", "make_config: never overwrites an existing config without --force")
        check(rc3 == 0 and len(baks) == 1, "make_config: --force keeps a .bak copy", str(baks))


def _scan_imports() -> dict[str, set[str]]:
    std = set(sys.stdlib_module_names)
    local = {f[:-3] for d in ("mcp", "scripts") for f in os.listdir(os.path.join(ROOT, d)) if f.endswith(".py")}
    local |= {"mcp", "scripts", "bench", "tools", "harness_box", "sandbox_net", "typesafe_sdk", "offline_stores"}
    found: dict[str, set[str]] = {}
    for base in ("mcp", "scripts"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, base)):
            dn[:] = [x for x in dn if x not in ("__pycache__", "offline_guard", "fixtures", "jev_sdk")]
            for f in fn:
                if not f.endswith(".py") or f.startswith("test_"):
                    continue
                path = os.path.join(dp, f)
                try:
                    tree = ast.parse(open(path, encoding="utf-8").read())
                except (SyntaxError, UnicodeDecodeError):
                    continue
                for n in ast.walk(tree):
                    mods: list[str] = []
                    if isinstance(n, ast.Import):
                        mods = [a.name.split(".")[0] for a in n.names]
                    elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                        mods = [n.module.split(".")[0]]
                    for m in mods:
                        if m not in std and m not in local:
                            found.setdefault(m, set()).add(os.path.relpath(path, ROOT).replace("\\", "/"))
    return found


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


def test_requirements() -> None:
    req = _pins(read("requirements.txt"))
    lock = _pins(read("requirements.lock.txt"))
    wrong = {k: (v, lock.get(k)) for k, v in req.items() if lock.get(k) != v}
    check(req and not wrong, "requirements.txt: every pin is the version requirements.lock.txt records",
          f"(requirements, lock): {wrong}")
    dist = {"PIL": "pillow", "yaml": "pyyaml", "markdown_it": "markdown-it-py",
            "tree_sitter_language_pack": "tree-sitter-language-pack"}
    missing = {}
    for mod, files in _scan_imports().items():
        if _norm(dist.get(mod, mod)) not in req:
            missing[mod] = sorted(files)[:2]
    check(not missing, "requirements.txt: every third-party import in mcp/ and scripts/ is a package named in it",
          str(missing))
    check({"jinja2", "httpx", "ruff"} <= set(req),
          "requirements.txt: the test tooling (jinja2, httpx, ruff) is in it")


def test_install_script() -> None:
    from_ps = read("scripts/install.ps1")
    import verify_artifacts as va
    m = va.load_manifest(va.MANIFEST, None)
    sizes = {a["id"]: int(a["size"]) / 2 ** 30 for a in m["artifacts"]}
    prov = {a["id"]: (a.get("provenance") or {}).get("status") for a in m["artifacts"]}
    items = re.findall(r"@\{ Id = '([^']+)';\s+GiB = ([0-9.]+);", from_ps)
    bad = [(i, g, round(sizes.get(i, -1), 2)) for i, g in items if abs(float(g) - sizes.get(i, -99)) > 0.006]
    check(len(items) >= 10 and not bad, "install.ps1: every model it names is a manifest artifact at the recorded size",
          f"{len(items)} items; mismatches (id, script GiB, manifest GiB): {bad}")
    check(all(prov.get(i) == "pinned" for i, _ in items),
          "install.ps1: every model it fetches is a pinned download (a recipe cannot be fetched)",
          str([i for i, _ in items if prov.get(i) != "pinned"]))
    em = yaml.safe_load(open(os.path.join(ROOT, "engines", "manifest.yaml"), encoding="utf-8"))
    names = re.findall(r"@\{ Name = '([^']+)';", from_ps)
    check(names and all(n in em["engines"] for n in names),
          "install.ps1: its engines are engine entries in engines/manifest.yaml", str(names))
    eng = {v[0] for v in mc.ENGINE_PATHS.values()}
    check(set(names) <= eng, "install.ps1: every engine it builds is one make_config.py places",
          f"{sorted(set(names) - eng)}")
    check(not re.search(r"[^\x00-\x7f]", from_ps), "install.ps1: ASCII only (Windows PowerShell 5.1 reads it without a BOM)")
    for sw in ("DryRun", "WhatIf", "FetchModels", "BuildEngines"):
        check(sw in from_ps, f"install.ps1: has -{sw}" if sw != "WhatIf" else "install.ps1: supports -WhatIf",
              "")
    ps = _powershell()
    if ps:
        for rel in ("scripts/install.ps1", "scripts/watchdog.ps1", "scripts/install-autostart.ps1",
                    "scripts/wsl_engine.ps1"):
            code = ("$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
                    f"'{os.path.join(ROOT, rel)}',[ref]$null,[ref]$e);$e.Count")
            r = subprocess.run([ps, "-NoProfile", "-Command", code], capture_output=True, text=True, timeout=60)
            check(r.stdout.strip() == "0", f"{rel}: parses (PowerShell's own parser)", r.stdout + r.stderr)
    else:
        check(True, "PowerShell not found: the parse checks are skipped (not a failure on this machine)")


def _powershell() -> str | None:
    for c in ("powershell.exe", "pwsh"):
        try:
            if subprocess.run([c, "-NoProfile", "-Command", "1"], capture_output=True, timeout=30).returncode == 0:
                return c
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def test_launchers() -> None:
    bat = read("scripts/start-stack.bat")
    wd = read("scripts/watchdog.ps1")
    auto = read("scripts/install-autostart.ps1")
    for name, text in (("start-stack.bat", bat), ("watchdog.ps1", wd), ("install-autostart.ps1", auto)):
        check("jwals" not in text and not re.search(r"C:[/\\]Users", text),
              f"{name}: names no home directory", "")
    check("stack.env" in bat and "stack.env" in wd, "launchers: start-stack.bat and watchdog.ps1 read stack.env")
    for var in ("YAMADORI_PYTHON", "YAMADORI_CUDA_BIN", "YAMADORI_SEARXNG_DIR", "YAMADORI_PUBLIC_BASE"):
        check(var in bat, f"start-stack.bat: reads {var}")
    check("%USERPROFILE%\\textgen\\installer_files\\env\\python.exe" in bat
          and "textgen\\installer_files\\env\\python.exe" in wd,
          "launchers: with no stack.env the interpreter is still the textgen env under the home directory")
    check("%USERPROFILE%\\textgen\\installer_files\\cudabuild\\Library\\bin" in bat,
          "start-stack.bat: with no stack.env the CUDA runtime directory is still the cudabuild env")
    check("https://ai.thejustinwalsh.me" in bat and "https://ai.thejustinwalsh.me" in wd,
          "launchers: with no stack.env the public base is still the one they had")
    check("Split-Path -Parent $PSScriptRoot" in auto, "install-autostart.ps1: the repository is where the script is")
    check("DryRun" in auto, "install-autostart.ps1: has -DryRun")
    gi = read(".gitignore")
    check(re.search(r"^config\.yaml\s*$", gi, re.M) and re.search(r"^stack\.env\s*$", gi, re.M)
          and not re.search(r"^config\.example\.yaml\s*$", gi, re.M),
          ".gitignore: config.yaml and stack.env are ignored, config.example.yaml is not")
    # the loader line itself, run for real in cmd (Windows): precedence env > stack.env, spaces, parentheses, '='
    if os.name == "nt":
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "stack.env"), "w", newline="\r\n") as f:
                f.write("# c\nYAMADORI_PYTHON=C:\\Program Files (x86)\\x\\python.exe\nYAMADORI_PUBLIC_BASE=http://a\n"
                        "YAMADORI_SEARXNG_DIR=D:\\a=b\n")
            line = next(ln for ln in bat.splitlines() if ln.startswith('if exist "stack.env" for /f'))
            with open(os.path.join(d, "t.bat"), "w", newline="\r\n") as f:
                f.write('@echo off\nsetlocal\nset "YAMADORI_PUBLIC_BASE=from-env"\n' + line + "\n"
                        "echo [%YAMADORI_PYTHON%][%YAMADORI_PUBLIC_BASE%][%YAMADORI_SEARXNG_DIR%]\n")
            r = subprocess.run(["cmd", "/c", os.path.join(d, "t.bat")], cwd=d, capture_output=True, text=True, timeout=30)
            check(r.stdout.strip() == "[C:\\Program Files (x86)\\x\\python.exe][from-env][D:\\a=b]",
                  "start-stack.bat: the stack.env loader keeps spaces, parentheses and '=', and the environment wins",
                  r.stdout + r.stderr)


def test_engine_portable_and_fetch() -> None:
    import build_engine as be
    b = object.__new__(be.Builder)
    b.report = {"checks": []}
    b.say = lambda msg: None
    b.portable = False
    try:
        b.check(False, "vcvars64 selects the recorded MSVC toolset", "14.44", identity=True)
        check(False, "build_engine: a toolchain-identity check fails the build without --portable")
    except be.BuildError:
        check(True, "build_engine: a toolchain-identity check fails the build without --portable")
    b.portable = True
    try:
        b.check(False, "vcvars64 selects the recorded MSVC toolset", "14.44", identity=True)
        waived = b.report["checks"][-1].get("portable_waived")
    except be.BuildError:
        waived = False
    check(waived is True, "build_engine --portable: a toolchain-identity check is a WARN, recorded as waived")
    try:
        b.check(False, "patch 0042 applied", "", identity=False)
        check(False, "build_engine --portable: a source / configuration check still fails the build")
    except be.BuildError:
        check(True, "build_engine --portable: a source / configuration check still fails the build")
    src = read("scripts/build_engine.py")
    sites = src.count("identity=True")
    check(sites >= 10, "build_engine: the identity checks are marked (MSVC, SDK, PATH tools, DLL copies, native CPU, "
          "original cache)", f"{sites} sites")
    # manifest.local.yaml merges over defaults and toolchains only
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "manifest.yaml"), "w") as f:
            f.write("defaults: {engines_root: A}\ntoolchains: {t: {vcvars: X, cuda: {root: Y, version: V}}}\n"
                    "engines: {e: {k: 1}}\n")
        with open(os.path.join(d, "manifest.local.yaml"), "w") as f:
            f.write("defaults: {engines_root: B}\ntoolchains: {t: {vcvars: Z, cuda: {root: W}}}\nengines: {e: {k: 2}}\n")
        m = be.load_yaml(os.path.join(d, "manifest.yaml"))
        check(m["defaults"]["engines_root"] == "B" and m["toolchains"]["t"]["vcvars"] == "Z"
              and m["toolchains"]["t"]["cuda"] == {"root": "W", "version": "V"} and m["engines"]["e"]["k"] == 1,
              "build_engine: manifest.local.yaml is merged over defaults and toolchains only, deeply", str(m))
    # fetch_models --models-dir
    import fetch_models as fm
    import argparse
    art = {"id": "x", "path": "${models}/qwen-image-2.1/vae/v.safetensors"}
    ns = argparse.Namespace(models_dir="D:/m", dest=None)
    got = fm._dest_file(art, ns).replace("\\", "/")
    check(got == "D:/m/qwen-image-2.1/vae/v.safetensors", "fetch_models --models-dir: the file goes where the "
          "manifest's path says", got)
    ns = argparse.Namespace(models_dir=None, dest="D:/flat")
    check(fm._dest_file(art, ns).replace("\\", "/") == "D:/flat/v.safetensors",
          "fetch_models --dest: unchanged, flat under the local name")


def test_readme_links() -> None:
    for rel in ("README.md", "docs/INSTALL.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            check(False, f"{rel}: exists")
            continue
        text = read(rel)
        missing = []
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if re.match(r"^[a-z]+:", target):
                continue
            base = os.path.dirname(p)
            if not os.path.exists(os.path.normpath(os.path.join(base, target))):
                missing.append(target)
        check(not missing, f"{rel}: every relative link and image resolves", str(missing))
        pri = [w for w in ("jwals", "contact.me@", "C:\\Users\\", "C:/Users/") if w in text]
        check(not pri, f"{rel}: no home directory or e-mail address", str(pri))
    check(os.path.exists(os.path.join(ROOT, "docs", "img", "dashboard.png")),
          "README: docs/img/dashboard.png exists")


def main() -> int:
    for fn in (test_config_example, test_make_config, test_requirements, test_install_script, test_launchers,
               test_engine_portable_and_fetch, test_readme_links):
        try:
            fn()
        except Exception as e:                                                      # noqa: BLE001
            import traceback
            check(False, f"{fn.__name__} raised", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")
    for ok, name, detail in _results:
        print(("  pass  " if ok else "  FAIL  ") + name + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
