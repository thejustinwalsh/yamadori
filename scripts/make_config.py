#!/usr/bin/env python
"""Render config.yaml (llama-swap's file: every model server, its card and its flags) from config.example.yaml.

    python scripts/make_config.py --list-gpus
    python scripts/make_config.py --models-dir D:/models --engines-root D:/engines            # GPUs from nvidia-smi
    python scripts/make_config.py --main-gpu GPU-... --second-gpu GPU-... --out config.yaml
    python scripts/make_config.py --stdout --set MAIN_MODEL=Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf

WHY THIS EXISTS. config.yaml is gitignored: it carries a machine's paths and its cards' UUIDs. config.example.yaml is
the committed copy of the stack's real config with every machine-specific value turned into an @@NAME@@ placeholder,
and this script is the one thing that fills them in. Nothing else is changed: the flags, windows and groups in the
example are the ones the stack runs (and were measured on an RTX 5060 Ti 16 GB + RTX A4000 16 GB; see docs/INSTALL.md
for what to re-measure on other cards).

THE PLACEHOLDERS (`PLACEHOLDERS` below: name, meaning, default)

    MODELS_DIR        directory holding the .gguf / .safetensors files (models/manifest.yaml names them)
    SERVER_NUDGE      llama-server of engine llama-bonsai2-ada: `bonsai`, the main model
    SERVER_DECIDE     the same engine, for `bonsai-a4000` (jjava's batched reads, --decide-seqs)
    SERVER_PRISM      llama-server of engine llama-prism: embeddings and `bonsai-vision`
    SERVER_MIRAI_S    llama-server of engine llama-mirai-s: the xhigh tier (optional)
    SERVER_UPSTREAM_MOE  llama-server of engine llama-upstream-flash: Flash-Next, the max tier (optional)
    SD_SERVER         sd-server of engine sd-cpp: image generation (optional)
    MAIN_MODEL        the main model's file name inside MODELS_DIR (the Bonsai trunk with its MTP head grafted on)
    MAIN_GPU_UUID     the card the main model lives alone on (CUDA_VISIBLE_DEVICES, by UUID, never by number)
    SECOND_GPU_UUID   the card everything else lives on

An engine path defaults to <ENGINES_ROOT>/<engine>/src/build/bin/<exe>, which is where
`python scripts/build_engine.py build <engine> --out <ENGINES_ROOT>/<engine>` puts it.

It never overwrites an existing config.yaml without --force, and it never starts anything. Standard library only.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXAMPLE = os.path.join(ROOT, "config.example.yaml")
OUT = os.path.join(ROOT, "config.yaml")

DEFAULT_MAIN_MODEL = "Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf"

# name -> (meaning, engine, exe) for the engine-path placeholders
ENGINE_PATHS = {
    "SERVER_NUDGE": ("llama-bonsai2-ada", "llama-server.exe"),
    "SERVER_DECIDE": ("llama-bonsai2-ada", "llama-server.exe"),
    "SERVER_PRISM": ("llama-prism", "llama-server.exe"),
    "SERVER_MIRAI_S": ("llama-mirai-s", "llama-server.exe"),
    "SERVER_UPSTREAM_MOE": ("llama-upstream-flash", "llama-server.exe"),
    "SD_SERVER": ("sd-cpp", "sd-server.exe"),
}
PLACEHOLDERS = ("MODELS_DIR", "MAIN_MODEL", "MAIN_GPU_UUID", "SECOND_GPU_UUID", *ENGINE_PATHS)

_TOKEN = re.compile(r"@@([A-Z][A-Z0-9_]*)@@")


def fwd(path: str) -> str:
    """llama-swap reads these inside YAML double quotes and passes them to a process: forward slashes."""
    return path.replace("\\", "/")


def engine_path(engines_root: str, engine: str, exe: str) -> str:
    return fwd(os.path.join(engines_root, engine, "src", "build", "bin", exe))


def detect_gpus(text: str | None = None) -> list[dict]:
    """[{index, name, uuid}] from `nvidia-smi -L` (or the text given), in nvidia-smi's own (PCI bus) order."""
    if text is None:
        try:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return []
        text = r.stdout
    gpus = []
    for m in re.finditer(r"^GPU (\d+): (.+?) \(UUID: (GPU-[0-9a-fA-F-]+)\)", text, re.M):
        gpus.append({"index": int(m.group(1)), "name": m.group(2), "uuid": m.group(3)})
    return gpus


def defaults(engines_root: str | None, models_dir: str | None) -> dict:
    v = {"MAIN_MODEL": DEFAULT_MAIN_MODEL}
    if models_dir:
        v["MODELS_DIR"] = fwd(models_dir)
    if engines_root:
        for name, (engine, exe) in ENGINE_PATHS.items():
            v[name] = engine_path(engines_root, engine, exe)
    return v


def placeholders_in(text: str) -> list[str]:
    return sorted(set(_TOKEN.findall(text)))


def render(template: str, values: dict) -> str:
    """The template with every @@NAME@@ replaced. A name with no value, or a value for a name the template lacks
    that is not a known placeholder, is an error: a half-filled config would start and fail at the first request."""
    wanted = placeholders_in(template)
    missing = [n for n in wanted if not values.get(n)]
    if missing:
        raise ValueError("no value for " + ", ".join(missing))
    unknown = [n for n in values if n not in PLACEHOLDERS]
    if unknown:
        raise ValueError("not a placeholder: " + ", ".join(unknown))
    return _TOKEN.sub(lambda m: str(values[m.group(1)]), template)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--example", default=EXAMPLE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stdout", action="store_true", help="print the rendered config instead of writing it")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out (a copy is kept as <out>.bak-<time>)")
    ap.add_argument("--models-dir")
    ap.add_argument("--engines-root")
    ap.add_argument("--main-gpu", help="UUID of the main card (default: the first GPU nvidia-smi lists)")
    ap.add_argument("--second-gpu", help="UUID of the second card (default: the second GPU nvidia-smi lists)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="set any placeholder, e.g. SERVER_NUDGE=D:/x/llama-server.exe (wins over the options above)")
    ap.add_argument("--list-gpus", action="store_true")
    ap.add_argument("--list-placeholders", action="store_true")
    a = ap.parse_args(argv)

    if a.list_gpus:
        gpus = detect_gpus()
        for g in gpus:
            print(f"  GPU {g['index']}  {g['name']:36s}  {g['uuid']}")
        if not gpus:
            print("  nvidia-smi lists no GPU (is the NVIDIA driver installed?)")
        return 0 if gpus else 1
    if a.list_placeholders:
        with open(a.example, encoding="utf-8") as f:
            print("\n".join(placeholders_in(f.read())))
        return 0

    with open(a.example, encoding="utf-8") as f:
        template = f.read()
    values = defaults(a.engines_root, a.models_dir)
    if a.main_gpu:
        values["MAIN_GPU_UUID"] = a.main_gpu
    if a.second_gpu:
        values["SECOND_GPU_UUID"] = a.second_gpu
    if not (values.get("MAIN_GPU_UUID") and values.get("SECOND_GPU_UUID")):
        gpus = detect_gpus()
        if len(gpus) >= 2:
            values.setdefault("MAIN_GPU_UUID", gpus[0]["uuid"])
            values.setdefault("SECOND_GPU_UUID", gpus[1]["uuid"])
            print(f"  GPUs from nvidia-smi: main = {gpus[0]['name']}, second = {gpus[1]['name']}", file=sys.stderr)
    for kv in a.set:
        k, sep, v = kv.partition("=")
        if not sep:
            print(f"  --set wants NAME=VALUE, got {kv!r}", file=sys.stderr)
            return 2
        values[k] = fwd(v) if k in ENGINE_PATHS or k == "MODELS_DIR" else v
    try:
        text = render(template, values)
    except ValueError as e:
        print(f"  REFUSED: {e}. --list-placeholders names them all; --list-gpus finds the UUIDs.", file=sys.stderr)
        return 2
    if a.stdout:
        sys.stdout.buffer.write(text.encode("utf-8"))      # LF, as the file is written (not Windows text mode)
        return 0
    if os.path.exists(a.out):
        if not a.force:
            print(f"  REFUSED: {a.out} exists (--force replaces it, keeping a .bak copy; --stdout prints instead)",
                  file=sys.stderr)
            return 2
        import shutil
        import time
        shutil.copy2(a.out, f"{a.out}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
