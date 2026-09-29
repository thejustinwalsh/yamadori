"""Fetch a SUBSET of a sharded safetensors checkpoint from Hugging Face by HTTP
range requests, and write it as one local safetensors file.

Why: Qwen3.8-Flash-Next's MTP draft block (31 `mtp.*` tensors, 5.2 GB BF16)
is spread over 28 of the checkpoint's 131 shards (360 GB). The GGUF quants
ship no MTP head, so the draft is converted from these tensors -- and only
these bytes need to come down. Each shard starts with an 8-byte header length
and a JSON header naming every tensor's dtype, shape and byte range, so the
header and then exactly the named ranges are read.

    python scripts/fetch_hf_tensors.py --repo Qwen/Qwen3.8-Flash-Next \\
        --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \\
        --prefix mtp. --out <models>/flash-next/mtp-bf16

writes, in --out:
  parts/<tensor>.bin     one raw payload per tensor (resumable: a partial
                         file is continued from its size)
  <name>.safetensors     the subset as ONE safetensors file: the tensors in
                         name order, payloads byte for byte as fetched
  fetch-manifest.json    repo, revision, per tensor: source shard, byte range
                         in that shard, dtype, shape, payload sha256; and the
                         output file's size and sha256

Verification: HF's LFS sha256 (X-Linked-ETag) covers a WHOLE shard, which a
range fetch cannot check. What is checked: every response names the pinned
commit (X-Repo-Commit) and returns exactly the requested length, and every
payload's length equals its header range. The payload sha256s are recorded so
a later fetch (or the full shard, if ever downloaded) can be compared.

Only GETs with a Range header go out; nothing is uploaded. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request

UA = "yamadori-fetch-hf-tensors/1"
CHUNK = 64 << 20


def get(url: str, start: int | None = None, end: int | None = None,
        revision: str | None = None, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            if start is not None:
                req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req, timeout=180) as r:
                commit = r.headers.get("X-Repo-Commit")
                data = r.read()
            if revision and commit and commit != revision:
                raise RuntimeError(f"{url}: X-Repo-Commit {commit} != pinned {revision}")
            if start is not None and len(data) != end - start + 1:
                raise IOError(f"short range read: {len(data)} of {end - start + 1}")
            return data
        except RuntimeError:
            raise
        except Exception as e:                      # network: retried, then surfaced
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"retry in {wait}s: {url} [{start}-{end}]: {e}", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def inventory(base: str, revision: str, prefix: str) -> list[dict]:
    index = json.loads(get(base + "model.safetensors.index.json", revision=revision))
    wanted = {k: v for k, v in index["weight_map"].items() if k.startswith(prefix)}
    if not wanted:
        sys.exit(f"no tensor in the index starts with {prefix!r}")
    rows = []
    for shard in sorted(set(wanted.values())):
        n = struct.unpack("<Q", get(base + shard, 0, 7, revision))[0]
        header = json.loads(get(base + shard, 8, 8 + n - 1, revision))
        for name, meta in header.items():
            if name == "__metadata__" or wanted.get(name) != shard:
                continue
            a, b = meta["data_offsets"]
            rows.append({"name": name, "shard": shard, "dtype": meta["dtype"],
                         "shape": meta["shape"], "start": 8 + n + a,
                         "end": 8 + n + b - 1, "bytes": b - a})
    missing = sorted(set(wanted) - {r["name"] for r in rows})
    if missing:
        sys.exit(f"named in the index but absent from their shard headers: {missing}")
    return sorted(rows, key=lambda r: r["name"])


def fetch(base: str, revision: str, rows: list[dict], parts: str) -> None:
    os.makedirs(parts, exist_ok=True)
    total = sum(r["bytes"] for r in rows)
    done = 0
    for r in rows:
        path = os.path.join(parts, r["name"] + ".bin")
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have > r["bytes"]:
            sys.exit(f"{path}: {have} bytes, more than the {r['bytes']} it should hold")
        with open(path, "ab") as f:
            pos = r["start"] + have
            while pos <= r["end"]:
                end = min(pos + CHUNK - 1, r["end"])
                f.write(get(base + r["shard"], pos, end, revision))
                f.flush()
                pos = end + 1
        done += r["bytes"]
        print(f"{done / total:6.1%}  {r['name']}  {r['bytes']:,} B", file=sys.stderr)
        if os.path.getsize(path) != r["bytes"]:
            sys.exit(f"{path}: size {os.path.getsize(path)} != {r['bytes']}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def assemble(rows: list[dict], parts: str, out_file: str) -> None:
    """One safetensors file: header (padded to 8 bytes with spaces, as the
    format allows), then each payload in name order."""
    header, off = {}, 0
    for r in rows:
        header[r["name"]] = {"dtype": r["dtype"], "shape": r["shape"],
                             "data_offsets": [off, off + r["bytes"]]}
        off += r["bytes"]
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = out_file + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for r in rows:
            h = hashlib.sha256()
            with open(os.path.join(parts, r["name"] + ".bin"), "rb") as p:
                for block in iter(lambda: p.read(1 << 24), b""):
                    h.update(block)
                    f.write(block)
            r["sha256"] = h.hexdigest()
    os.replace(tmp, out_file)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True, help="a full commit sha")
    ap.add_argument("--prefix", required=True, help="tensor-name prefix, e.g. mtp.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--name", default="subset", help="output file stem")
    ap.add_argument("--inventory-only", action="store_true")
    a = ap.parse_args(argv)
    if len(a.revision) != 40:
        sys.exit("--revision must be a full 40-hex commit")
    base = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/"
    os.makedirs(a.out, exist_ok=True)
    rows = inventory(base, a.revision, a.prefix)
    total = sum(r["bytes"] for r in rows)
    print(f"{len(rows)} tensors, {total:,} bytes, in "
          f"{len({r['shard'] for r in rows})} shards", file=sys.stderr)
    if a.inventory_only:
        json.dump(rows, sys.stdout, indent=1)
        return 0
    parts = os.path.join(a.out, "parts")
    fetch(base, a.revision, rows, parts)
    out_file = os.path.join(a.out, f"{a.name}.safetensors")
    assemble(rows, parts, out_file)
    manifest = {"repo": a.repo, "revision": a.revision, "prefix": a.prefix,
                "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "file": os.path.basename(out_file),
                "size": os.path.getsize(out_file), "sha256": sha256_file(out_file),
                "tensor_bytes": total, "tensors": rows}
    with open(os.path.join(a.out, "fetch-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"{out_file}\n  size {manifest['size']:,}\n  sha256 {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
