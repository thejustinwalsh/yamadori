#!/usr/bin/env python
"""Re-obtain a model file from its pinned Hugging Face revision, and check it.

    python scripts/fetch_models.py list
    python scripts/fetch_models.py fetch ID --dest DIR [--dry-run]
    python scripts/fetch_models.py fetch ID --models-dir DIR [--dry-run]   # DIR/<the manifest's own subpath>
    python scripts/fetch_models.py fetch --repo R --revision SHA --filename F
                                         --dest DIR [--sha256 HEX] [--dry-run]
    python scripts/fetch_models.py check-remote [ID ...]
    python scripts/fetch_models.py --verify [--config config.yaml] [--rehash]

fetch      Downloads https://huggingface.co/<repo>/resolve/<revision>/<file>
           -- the revision is a full commit, so the bytes cannot move -- into
           DEST as `<name>.part`, hashing as it streams, and renames it to the
           manifest's local file name only when the size and sha256 are the
           recorded ones. A file already in DEST with the recorded hash is left
           alone. It never overwrites a different file. A derived artifact
           (provenance recipe/reproduced) is not downloadable: fetch prints its
           recipe instead. `--repo/--revision/--filename` fetches any file ad
           hoc; without `--sha256` it checks against what the HF API says the
           file is at that revision (the LFS sha256, or the git blob id for a
           small non-LFS file), which is an independent check.
check-remote
           Asks the HF API, per pinned artifact, whether the file at the
           pinned revision still has the recorded size and sha256 (no
           download). A repo taken down or made gated shows up here first.
--verify   scripts/verify_artifacts.py's verify(): hashes what config.yaml
           points at and compares it with the manifest. Exit 1 on an error.

Standard library + PyYAML. Set HF_TOKEN for a gated repo (sent only to
huggingface.co).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_artifacts as va  # noqa: E402

HF = os.environ.get("YAMADORI_HF_ENDPOINT", "https://huggingface.co")
UA = {"User-Agent": "yamadori-fetch-models/1"}


def _headers() -> dict:
    h = dict(UA)
    tok = os.environ.get("HF_TOKEN")
    if tok and urllib.parse.urlparse(HF).hostname == "huggingface.co":
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get_json(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()),
                                timeout=timeout) as r:
        return json.load(r)


def resolve_url(repo: str, revision: str, filename: str) -> str:
    return (f"{HF}/{repo}/resolve/{revision}/"
            + "/".join(urllib.parse.quote(p) for p in filename.split("/")))


def remote_info(repo: str, revision: str, filename: str) -> dict:
    """{size, sha256 (LFS) or None, blob_id (git sha1), revision (full)}."""
    d = _get_json(f"{HF}/api/models/{repo}/revision/{revision}?blobs=true")
    for s in d.get("siblings") or []:
        if s.get("rfilename") == filename:
            lfs = s.get("lfs") or {}
            return {"size": s.get("size"), "sha256": lfs.get("sha256"),
                    "blob_id": s.get("blobId"), "revision": d.get("sha")}
    raise LookupError(f"{filename} is not in {repo} at {revision}")


def git_blob_sha1(path: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest_file: str, *, size: int | None, sha256: str | None,
             blob_id: str | None = None) -> str:
    """Stream url to dest_file.part, verify, rename. Returns the sha256."""
    if os.path.exists(dest_file):
        got = va.sha256_file(dest_file)
        if sha256 and got == sha256:
            print(f"  already present and verified: {dest_file}")
            return got
        raise SystemExit(f"REFUSED: {dest_file} exists and is not the recorded file "
                         f"(sha256 {got[:16]}...). Move it aside first; this never overwrites.")
    os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
    part = dest_file + ".part"
    h = hashlib.sha256()
    n = 0
    t0 = time.time()
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        for b in iter(lambda: r.read(8 << 20), b""):
            f.write(b)
            h.update(b)
            n += len(b)
            if size and size > 64 << 20:
                print(f"\r  {n / 2**30:.2f}/{size / 2**30:.2f} GiB  "
                      f"{n / 2**20 / max(time.time() - t0, 1e-3):.0f} MiB/s",
                      end="", flush=True)
    if size and size > 64 << 20:
        print()
    got = h.hexdigest()
    bad = []
    if size is not None and n != size:
        bad.append(f"size {n:,} != {size:,}")
    if sha256 and got != sha256:
        bad.append(f"sha256 {got} != {sha256}")
    if blob_id and not sha256 and git_blob_sha1(part) != blob_id:
        bad.append(f"git blob id != {blob_id}")
    if bad:
        raise SystemExit(f"VERIFY FAILED, kept as {part} for inspection: " + "; ".join(bad))
    os.replace(part, dest_file)
    return got


def _artifact(m: dict, aid: str) -> dict:
    for a in m["artifacts"]:
        if a.get("id") == aid:
            return a
    ids = ", ".join(a.get("id", "?") for a in m["artifacts"])
    raise SystemExit(f"no artifact {aid!r} in the manifest. Known: {ids}")


def cmd_list(m: dict) -> int:
    for a in m["artifacts"]:
        p = a.get("provenance") or {}
        src = (f"{p.get('repo')}@{str(p.get('revision'))[:10]}" if p.get("status") == "pinned"
               else "recipe" if p.get("status") in ("reproduced", "recipe") else "hash only")
        print(f"{a['id']:<44} {a['status']:<11} {p.get('status', '?'):<11} "
              f"{int(a['size']) / 2**30:7.2f} GiB  {src}")
    return 0


def _dest_file(art: dict, a: argparse.Namespace) -> str:
    """--dest puts the file flat in DIR under its local name; --models-dir puts it where the manifest's `path` says
    (`${models}/qwen-image-2.1/vae/x.safetensors` -> DIR/qwen-image-2.1/vae/x.safetensors), which is what config.yaml
    expects when its `models` macro is DIR (scripts/install.ps1 uses this)."""
    if a.models_dir:
        path = str(art["path"])
        if not path.startswith("${models}/"):
            raise SystemExit(f"{art['id']}: its path {path!r} is not under ${{models}}; use --dest")
        return os.path.join(a.models_dir, *path[len("${models}/"):].split("/"))
    return os.path.join(a.dest, os.path.basename(str(art["path"])))


def cmd_fetch(m: dict, a: argparse.Namespace) -> int:
    if a.repo:
        if not (a.revision and a.filename):
            raise SystemExit("--repo needs --revision and --filename")
        info = remote_info(a.repo, a.revision, a.filename)
        want = a.sha256 or info["sha256"]
        if not a.dest:
            raise SystemExit("--repo needs --dest")
        dest = os.path.join(a.dest, os.path.basename(a.filename))
        url = resolve_url(a.repo, info["revision"], a.filename)
        print(f"{a.repo}@{info['revision']} {a.filename}: {info['size']:,} bytes, "
              + (f"sha256 {want}" if want else f"git blob {info['blob_id']}"))
        if a.dry_run:
            print(f"  dry run: would GET {url} -> {dest}")
            return 0
        got = download(url, dest, size=info["size"], sha256=want,
                       blob_id=None if want else info["blob_id"])
        print(f"  OK {dest} sha256 {got}")
        return 0
    art = _artifact(m, a.id)
    prov = art.get("provenance") or {}
    if prov.get("status") != "pinned":
        print(f"{art['id']} is not downloadable (provenance {prov.get('status')}).")
        rec = prov.get("recipe")
        if rec:
            print("Rebuild it with this recipe (docs/MODELS.md):")
            print(json.dumps(rec, indent=1))
        return 2
    dest = _dest_file(art, a)
    url = resolve_url(prov["repo"], prov["revision"], prov["filename"])
    print(f"{art['id']}: {prov['repo']}@{prov['revision']} {prov['filename']} "
          f"({int(art['size']):,} bytes, sha256 {art['sha256']})")
    if a.dry_run:
        print(f"  dry run: would GET {url} -> {dest}")
        return 0
    got = download(url, dest, size=int(art["size"]), sha256=art["sha256"])
    print(f"  OK {dest} sha256 {got}")
    return 0


def cmd_check_remote(m: dict, ids: list[str]) -> int:
    bad = 0
    n = 0
    for art in m["artifacts"]:
        prov = art.get("provenance") or {}
        if prov.get("status") != "pinned" or (ids and art["id"] not in ids):
            continue
        n += 1
        try:
            info = remote_info(prov["repo"], prov["revision"], prov["filename"])
        except (urllib.error.URLError, LookupError, OSError, ValueError) as e:
            print(f"FAIL {art['id']}: {e}")
            bad += 1
            continue
        ok = info["sha256"] == art["sha256"] and info["size"] == int(art["size"])
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {art['id']}: {prov['repo']}@"
              f"{prov['revision'][:10]} size {info['size']} sha256 {str(info['sha256'])[:16]}...")
    print(f"\n{n - bad}/{n} pinned artifacts match their pinned revision")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=va.MANIFEST)
    ap.add_argument("--local", default=va.LOCAL_MANIFEST)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config", default=va.DEFAULT_CONFIG)
    ap.add_argument("--rehash", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("id", nargs="?")
    f.add_argument("--dest")
    f.add_argument("--models-dir")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--repo")
    f.add_argument("--revision")
    f.add_argument("--filename")
    f.add_argument("--sha256")
    c = sub.add_parser("check-remote")
    c.add_argument("ids", nargs="*")
    a = ap.parse_args(argv)
    if a.verify:
        return va.main(["--config", a.config, "--manifest", a.manifest,
                        "--local", a.local] + (["--rehash"] if a.rehash else []))
    m = va.load_manifest(a.manifest, a.local)
    if a.cmd == "list":
        return cmd_list(m)
    if a.cmd == "fetch":
        if not a.id and not a.repo:
            raise SystemExit("fetch needs an artifact id or --repo/--revision/--filename")
        if bool(a.dest) == bool(a.models_dir):
            raise SystemExit("fetch needs exactly one of --dest DIR and --models-dir DIR")
        return cmd_fetch(m, a)
    if a.cmd == "check-remote":
        return cmd_check_remote(m, a.ids)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
