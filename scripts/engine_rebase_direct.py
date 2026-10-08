#!/usr/bin/env python
"""Re-base an engine's patch series onto a new upstream commit by applying the patches DIRECTLY to the new tree.

`build_engine.py update` replays the series on the OLD base first, so it cannot take a series whose first patches
were dropped because upstream merged them (the rest then need those patches' context). Here every patch is applied,
in manifest order, to the new base's tree (`git apply --index`, no fuzz, no 3-way); a patch that does not apply stops
the run. A patch whose diff is unchanged keeps its file; one whose offsets moved is rewritten in its own preamble and
tail (the same helpers `update` uses). Nothing is written without --write.

    python scripts/engine_rebase_direct.py llama-bonsai2-ada 4fbda1292562 [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_engine as be  # noqa: E402
import engine_vendor as ev  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("to")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(HERE), "engines", "manifest.yaml"))
    a = ap.parse_args()
    manifest = be.load_yaml(a.manifest)
    e = manifest["engines"][a.engine]
    cache = ev.Cache(ev.cache_path(manifest), ev._git_exe(manifest), print)
    cache.fetch(e["upstream"], a.to)
    to = ev.out(cache.g(["rev-parse", f"{a.to}^{{commit}}"]))
    new_subs: dict = {}
    t_new = cache.flat_tree(e["upstream"], to, None, new_subs)
    work = os.path.join(cache.path, "work", a.engine + "-direct")
    if os.path.exists(work):
        shutil.rmtree(ev._long(work), onerror=be._force_remove)
    os.makedirs(work)

    def w(args, **kw):
        return ev.git(args, work, env=dict(ev._IDENT), exe=cache.exe, **kw)

    try:
        w(["init", "-q", "."])
        w(["config", "core.autocrlf", "false"])
        with open(os.path.join(work, ".git", "objects", "info", "alternates"), "wb") as f:
            f.write(os.path.join(cache.repo, ".git", "objects").replace("\\", "/").encode() + b"\n")
        c_new = ev.out(w(["commit-tree", t_new, "-m", f"base {to}"]))
        w(["checkout", "-q", "-f", "-B", "series", c_new])
        rows = ev.patch_list(manifest, a.engine, be.PATCHES)
        new_files: dict[str, bytes] = {}
        for r in rows:
            args = ["apply", "--index"]
            if r["dir"] not in (".", ""):
                args.append(f"--directory={r['dir']}")
            res = w(args + [r["path"]], check=False)
            if res.returncode:
                print(f"  CONFLICT {r['file']}: {(res.stderr or b'').decode('utf-8', 'replace')[-400:]}")
                return 1
            w(["commit", "-q", "--allow-empty", "-m", r["file"]])
            stat = w(["diff", "--stat=72", "--summary", "HEAD^", "HEAD"]).stdout.decode("utf-8")
            diff = w(["diff", "--no-color", "--no-ext-diff", "HEAD^", "HEAD"]).stdout.decode("utf-8")
            with open(r["path"], encoding="utf-8", newline="") as f:
                orig = f.read()
            head, tail = ev._preamble_and_tail(orig)
            new = (ev._restat(head, stat) + diff + tail)
            same = new == orig
            print(f"  {r['file']:70s} {'unchanged' if same else 'offsets moved'}")
            if not same:
                new_files[r["file"]] = new.encode("utf-8")
        if not a.write:
            print(f"dry run: {len(new_files)} of {len(rows)} patch files would be rewritten; nothing written")
            return 0
        for fname, data in new_files.items():
            p = os.path.join(be.PATCHES, a.engine, fname)
            with open(p, "wb") as f:
                f.write(data)
            ev.set_patch_sha(a.manifest, a.engine, fname, hashlib.sha256(data).hexdigest())
        ev.set_engine_scalar(a.manifest, a.engine, "base_commit", to)
        ev.vendor(a.manifest, a.engine, be.PATCHES, ev.SRC_ROOT, cache, log=print)
        print(f"wrote {len(new_files)} patch files, base_commit {to}, re-vendored")
        return 0
    finally:
        if os.path.exists(work):
            shutil.rmtree(ev._long(work), onerror=be._force_remove)


if __name__ == "__main__":
    sys.exit(main())
