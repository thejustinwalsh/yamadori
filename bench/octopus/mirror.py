#!/usr/bin/env python
"""Read-only mirror of files a run wrote OUTSIDE /workspace, inside its sandbox.

    python bench/octopus/mirror.py v0b-V0-xhigh-1 [--src /root/space-shooter]

Why: in v0b-V0-xhigh-1 the model wrote the game to /root/space-shooter, a
tmpfs in Hermes' docker sandbox that dies with the container (log #32). The
coordinator's decision: do not intervene, grade the mirror.

WHEN it copies (so the final state cannot be lost):
  - after EVERY tool result in Hermes' stream (logs/<run>/hermes.jsonl): a
    write_file / patch / terminal has just run, so the tree changed
  - after every new relay row (a step started or ended)
  - every 60 s as a floor
  - when the stream shows the session ending (a `result` event): one last
    copy at once, retried every 0.5 s until the container is gone
Hermes tears the container down itself at session exit; neither this script
nor run.py controls that, so the last tool result is the last guaranteed copy
point (a final answer writes nothing).

HOW: `docker exec <c> tar -C <parent> -cf - <dir>` (docker cp cannot read a
tmpfs), unpacked into logs/<run>/root_mirror/<dir>.tmp, then swapped in.
Nothing is written into the container or the run folder.

VERIFY (`--verify`, and after every copy into root_mirror/verify.json): every
path under --src that the model's own write_file / patch calls touched
(hermes.jsonl tool_use) exists in the mirror; for paths whose LAST touch was a
write_file whose result reported bytes_written, the size matches.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as runmod  # noqa: E402


def containers(run_id: str) -> list[str]:
    p = subprocess.run(["docker", "ps", "--filter", "label=hermes-agent=1", "--format",
                        '{{.ID}} {{.Label "hermes-profile"}}'],
                       capture_output=True, text=True, timeout=30)
    return [ln.split()[0] for ln in p.stdout.splitlines()
            if len(ln.split()) > 1 and ln.split()[1].startswith(f"octo-{run_id}-")]


def copy(run_id: str, src: str, dst_root: str) -> str | None:
    parent, name = os.path.dirname(src), os.path.basename(src)
    for c in containers(run_id):
        tmp = os.path.join(dst_root, name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        ex = subprocess.Popen(["docker", "exec", c, "tar", "-C", parent, "-cf", "-", name],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        un = subprocess.run(["tar", "-xf", "-", "-C", tmp], stdin=ex.stdout,
                            capture_output=True)
        ex.wait(timeout=120)
        if ex.returncode == 0 and un.returncode == 0 and os.path.isdir(os.path.join(tmp, name)):
            final = os.path.join(dst_root, name)
            shutil.rmtree(final, ignore_errors=True)
            os.replace(os.path.join(tmp, name), final)
            shutil.rmtree(tmp, ignore_errors=True)
            with open(os.path.join(dst_root, "last_copy.txt"), "w") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {c}\n")
            return c
        shutil.rmtree(tmp, ignore_errors=True)
    return None


def verify(run_id: str, src: str, dst_root: str) -> dict:
    log = os.path.join(runmod.LOGS_DIR, run_id, "hermes.jsonl")
    last: dict[str, dict] = {}
    pending: list[tuple[str, str]] = []   # FIFO: parallel calls return in order
    for ln in open(log, encoding="utf-8", errors="replace"):
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("type") == "tool_use" and e.get("name") in ("write_file", "patch"):
            path = (e.get("input") or {}).get("path") or ""
            pending.append((e.get("name"), path))
        elif e.get("type") == "tool_result" and pending and e.get("name") == pending[0][0]:
            name, path = pending.pop(0)
            try:
                out = json.loads(e.get("output") or "{}")
            except ValueError:
                out = {}
            ok = bool(out.get("bytes_written") is not None or out.get("success"))
            if path.startswith(src + "/") and ok:
                last[path] = {"op": name, "bytes": out.get("bytes_written")}
    base = os.path.dirname(src)
    missing, size_mismatch = [], []
    for path, rec in sorted(last.items()):
        local = os.path.join(dst_root, *os.path.relpath(path, base).split("/"))
        if not os.path.isfile(local):
            missing.append(path)
        elif rec["op"] == "write_file" and rec["bytes"] is not None:
            if os.path.getsize(local) != rec["bytes"]:
                size_mismatch.append({"path": path, "wrote": rec["bytes"],
                                      "mirror": os.path.getsize(local)})
    mirrored = sum(len(f) for _d, _s, f in os.walk(os.path.join(dst_root, os.path.basename(src))))
    res = {"t": time.strftime("%H:%M:%S"), "paths_written_by_model": len(last),
           "files_in_mirror": mirrored, "missing": missing, "size_mismatch": size_mismatch,
           "ok": not missing and not size_mismatch}
    with open(os.path.join(dst_root, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--src", default="/root/space-shooter")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    log_dir = os.path.join(runmod.LOGS_DIR, a.run_id)
    dst = os.path.join(log_dir, "root_mirror")
    os.makedirs(dst, exist_ok=True)
    if a.verify:
        print(json.dumps(verify(a.run_id, a.src, dst), indent=1))
        return 0
    hj, rj = os.path.join(log_dir, "hermes.jsonl"), os.path.join(log_dir, "relay.jsonl")
    seen = (-1, -1)
    last_copy = 0.0
    while True:
        try:
            ev = open(hj, encoding="utf-8", errors="replace").read()
            rl = open(rj, encoding="utf-8", errors="replace").read()
        except OSError:
            ev, rl = "", ""
        n_results = ev.count('"type": "tool_result"')
        n_rows = rl.count("\n")
        ended = '"type": "result"' in ev or os.path.isfile(os.path.join(log_dir, "meta.json"))
        if (n_results, n_rows) != seen or time.time() - last_copy > 60 or ended:
            c = copy(a.run_id, a.src, dst)
            if c:
                last_copy = time.time()
                seen = (n_results, n_rows)
                v = verify(a.run_id, a.src, dst)
                print(f"{time.strftime('%H:%M:%S')} copied from {c}: "
                      f"{v['files_in_mirror']} files, model wrote {v['paths_written_by_model']}, "
                      f"ok={v['ok']}", flush=True)
            elif ended:
                print(f"{time.strftime('%H:%M:%S')} session ended and the container is gone: "
                      "final mirror is the last copy above", flush=True)
                return 0
        time.sleep(0.5 if ended else 2)


if __name__ == "__main__":
    sys.exit(main())
