#!/usr/bin/env python
"""Install TypeSafe's two SDKs, pinned, in ISOLATED locations -- the test
clients of the Jev API (mcp/jev_api.py; mcp/test_jev_sdk.py;
docs/JEV-CONFORMANCE.md section 6). Never into the stack's Python.

    python tools/typesafe-sdk/install.py          # the plan; nothing is sent
    python tools/typesafe-sdk/install.py --run    # download, verify, install

WHAT, AND WHERE
  Python  typesafe-sdk==0.7.2 into tools/typesafe-sdk/py/.venv (a venv of
          this interpreter). The wheel is downloaded alone first (--no-deps)
          and its sha256 checked against PY_PIN before anything is
          installed; then it is installed from that file with its
          dependencies (binary wheels only), constrained by
          locks/typesafe-sdk.lock.txt when that lock exists. The lock is
          (re)written after the install from the venv's own `pip freeze`,
          in scripts/verify_artifacts.py's format (runtime
          `typesafe-sdk-python` in models/manifest.yaml checks it).
  JS      @typesafe-ai/sdk@0.6.0 into tools/typesafe-sdk/js/node_modules by
          `npm ci --ignore-scripts` from js/package-lock.json, whose
          `integrity` is the registry's sha512 (npm refuses a tarball that
          does not match it); runtime `typesafe-sdk-js` records the two files'
          sha256.

THE PINS: read from the registries' own metadata on 2026-09-29/30
(pypi.org/pypi/typesafe-sdk/0.7.2/json; registry.npmjs.org/@typesafe-ai/sdk/
0.6.0), not from a download. Licences: both MIT (PyPI classifier "License ::
OSI Approved :: MIT License"; npm `license: MIT`). The Python SDK's declared
dependencies (httpx2>=2.0.0, pydantic>=2.12.0, pydantic-core>=2.41.1,
tenacity>=9.0.0, typing-extensions>=4.13.0) and their own resolve at install
time; the lock written after it is the record.

Operator approval for this download: required before --run (a download).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY_DIR = os.path.join(HERE, "py")
VENV = os.path.join(PY_DIR, ".venv")
DIST = os.path.join(PY_DIR, "dist")
JS_DIR = os.path.join(HERE, "js")
LOCK = os.path.join(ROOT, "locks", "typesafe-sdk.lock.txt")

PY_PIN = {
    "name": "typesafe-sdk", "version": "0.7.2",
    "wheel": "typesafe_sdk-0.7.2-py3-none-any.whl",
    "sha256": "0a961148187d52e18276ed7f2d02617cfac48e3b97673cb631a8749397d43d1e",
    "size": 36448, "licence": "MIT", "requires_python": ">=3.10",
    "uploaded": "2026-09-26T21:20:23Z",
    "source": "https://pypi.org/project/typesafe-sdk/0.7.2/",
}
JS_PIN = {
    "name": "@typesafe-ai/sdk", "version": "0.6.0",
    "integrity": "sha512-IddX+Q0XM+VagOUZFeP7wZjaO4SHMdvnh2zEBdrZZnXedWI3BNK1lKhMx3ayrkFWvVLbVcUHJy6AVZlY+e6Jaw==",
    "shasum": "dbba30689e77c317e7619fbee006caa18f37a76a",
    "tarball": "https://registry.npmjs.org/@typesafe-ai/sdk/-/sdk-0.6.0.tgz",
    "unpacked_size": 209203, "licence": "MIT", "node": ">=20",
}


def venv_python() -> str:
    return os.path.join(VENV, "Scripts" if os.name == "nt" else "bin",
                        "python.exe" if os.name == "nt" else "python")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def plan() -> list[str]:
    return [
        f"{sys.executable} -m venv {VENV}",
        f"{venv_python()} -m pip download {PY_PIN['name']}=={PY_PIN['version']} "
        f"--no-deps --only-binary=:all: --dest {DIST}   "
        f"# {PY_PIN['wheel']}, {PY_PIN['size']:,} bytes, from PyPI",
        f"check sha256({PY_PIN['wheel']}) == {PY_PIN['sha256']}",
        f"{venv_python()} -m pip install --only-binary=:all: "
        + (f"-c {LOCK} " if os.path.exists(LOCK) else "")
        + f"{os.path.join(DIST, PY_PIN['wheel'])}   # + its dependencies",
        f"write {LOCK} (pip freeze of the venv)",
        f"npm ci --ignore-scripts --no-audit --no-fund   (in {JS_DIR}; "
        f"{JS_PIN['name']}@{JS_PIN['version']}, {JS_PIN['unpacked_size']:,} "
        f"bytes unpacked, integrity {JS_PIN['integrity'][:24]}...)",
    ]


def install_python() -> None:
    if not os.path.exists(venv_python()):
        run([sys.executable, "-m", "venv", VENV])
    os.makedirs(DIST, exist_ok=True)
    run([venv_python(), "-m", "pip", "download",
         f"{PY_PIN['name']}=={PY_PIN['version']}", "--no-deps",
         "--only-binary=:all:", "--dest", DIST])
    wheel = os.path.join(DIST, PY_PIN["wheel"])
    got = sha256(wheel)
    if got != PY_PIN["sha256"]:
        os.remove(wheel)
        raise SystemExit(f"REFUSED: {PY_PIN['wheel']} sha256 {got} is not the "
                         f"pinned {PY_PIN['sha256']} (removed)")
    print(f"  sha256 ok: {PY_PIN['wheel']}", flush=True)
    cmd = [venv_python(), "-m", "pip", "install", "--only-binary=:all:"]
    if os.path.exists(LOCK):
        cmd += ["-c", LOCK]
    run(cmd + [wheel])
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import verify_artifacts
    n = verify_artifacts.write_lock(LOCK, {
        "id": "typesafe-sdk-python", "interpreter": venv_python(),
        "freeze": "pip", "lock": os.path.relpath(LOCK, ROOT)})
    print(f"  wrote {os.path.relpath(LOCK, ROOT)}: {n} packages", flush=True)


def install_js() -> None:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=JS_DIR)
    pj = os.path.join(JS_DIR, "node_modules", "@typesafe-ai", "sdk",
                      "package.json")
    with open(pj, encoding="utf-8") as f:
        v = json.load(f).get("version")
    if v != JS_PIN["version"]:
        raise SystemExit(f"REFUSED: installed {JS_PIN['name']} is {v}, "
                         f"not {JS_PIN['version']}")
    print(f"  ok: {JS_PIN['name']}@{v}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="download, verify and install (default: the plan)")
    ap.add_argument("--only", choices=("python", "js"))
    a = ap.parse_args(argv)
    if not a.run:
        print("PLAN (nothing sent; --run to do it):")
        for step in plan():
            print("  " + step)
        return 0
    if a.only in (None, "python"):
        install_python()
    if a.only in (None, "js"):
        install_js()
    return 0


if __name__ == "__main__":
    sys.exit(main())
