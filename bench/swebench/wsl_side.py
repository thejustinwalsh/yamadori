#!/usr/bin/env python
"""The half of the SWE-bench runner that lives in WSL, next to Docker.

Run by run.py (Windows) through `wsl -d Ubuntu`, with the venv interpreter
~/swebench-yamadori/.venv/bin/python. It can also be run by hand:

    python wsl_side.py check
    python wsl_side.py agent --arm bonsai --instance django__django-15277 \
        --out /mnt/c/.../results/<run-id>/bonsai --key-file /mnt/c/.../dogfood.key
    python wsl_side.py evaluate --out /mnt/c/.../results/<run-id>/bonsai \
        --run-id <run-id>

THE KEY. It is read from --key-file into OPENAI_API_KEY in the environment of
the mini-extra child only -- litellm's `openai/` provider reads it from there.
It is never an argument, never written to the overlay config (mini-swe-agent
dumps model_kwargs into every trajectory), and never printed.

WHAT IS CHANGED FROM THE LEADERBOARD CONFIG. The leaderboard's own
swebench.yaml, as shipped in the pinned mini-swe-agent, is passed FIRST and
unmodified (its sha256 is recorded in each run's run.json). One overlay is
merged on top, and it only points the model at our endpoint. Every key it
sets is listed in OVERLAY_DEVIATIONS and in docs/SWE-BENCH.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arms  # noqa: E402

NATIVE = os.name == "nt"
if NATIVE:
    # Windows-native since 2026-09-23: see dockerfix.py for why (Docker
    # Desktop's WSL integration failed after the crash; its read cache is
    # stale). mini-swe-agent and the harness run through dockerfix.py.
    VENV = os.environ.get("SWEBENCH_VENV",
                          r"C:\Users\jwals\swebench-yamadori\venv-win")
    PY = os.path.join(VENV, "Scripts", "python.exe")
    MINI_CMD = [PY, os.path.join(HERE, "dockerfix.py"), "mini"]
    EVAL_CMD = [PY, os.path.join(HERE, "dockerfix.py"), "eval"]
else:
    VENV = os.path.expanduser("~/swebench-yamadori/.venv")
    PY = os.path.join(VENV, "bin", "python")
    MINI_CMD = [os.path.join(VENV, "bin", "mini-extra"), "swebench"]
    EVAL_CMD = [PY, "-m", "swebench.harness.run_evaluation"]

# Every key the overlay sets, and why it is required. Nothing else in the
# leaderboard config is touched: not the prompts, not step_limit (250), not
# cost_limit ($3), not the observation or format-error templates.
OVERLAY_DEVIATIONS = {
    "model.model_name": "openai/yamadori -- our model, through litellm's "
                        "OpenAI-compatible provider",
    "model.model_kwargs.api_base": "the proxy on :1234 as seen from WSL",
    "model.model_kwargs.extra_headers": "X-Yamadori-Features: the arm",
    "model.model_kwargs.extra_body": "reasoning_effort=max so the tier allows "
                                     "what the header decides (bench/domain)",
    "model.model_kwargs.timeout": "3600 s: litellm's 600 s default is shorter "
                                  "than one deep-thinking turn here, and a "
                                  "timed-out call is RETRIED, doubling GPU load",
    "model.model_kwargs.temperature": "None instead of the yaml's 0.0: all 13 "
                                      "mini v2 submissions on the leaderboard "
                                      "(trajectories on swe-bench-submissions "
                                      "S3) ran with temperature None, i.e. the "
                                      "provider default",
    "model.litellm_model_registry": "cost 0 per token for openai/yamadori",
    "model.cost_tracking": "ignore_errors: mini raises on a cost of 0, which a "
                           "free local model always has. Consequence: the $3 "
                           "cost_limit can never bind; only step_limit does",
}


def _read_key(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        sys.exit(f"key file is empty: {path}")
    return key


def _get(url: str, key: str | None = None, timeout: int = 5) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:                                            # noqa: BLE001
        return 0, b""


def resolve_api_base(explicit: str | None) -> str:
    """The proxy as reachable from WSL. localhost works only under mirrored
    networking; under NAT it is the Windows host, the default gateway."""
    if explicit:
        return explicit.rstrip("/")
    cands = ["127.0.0.1"]
    try:
        out = subprocess.run(["ip", "route"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("default via "):
                cands.append(line.split()[2])
    except OSError:
        pass
    for h in cands:
        if _get(f"http://{h}:1234/health")[0] == 200:
            return f"http://{h}:1234/v1"
    sys.exit(f"the proxy's /health answers on none of {cands} at :1234 -- "
             "is the stack up? (not retryable from here; the operator starts it)")


def default_config() -> str:
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    from minisweagent.run.benchmarks.swebench import DEFAULT_CONFIG_FILE
    return str(DEFAULT_CONFIG_FILE)


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def versions() -> dict:
    import importlib.metadata as md
    return {p: md.version(p) for p in
            ("mini-swe-agent", "swebench", "litellm", "datasets")}


def write_overlay(out: str, arm: str, api_base: str) -> str:
    registry = os.path.join(out, "model_registry.json")
    zero = {"max_tokens": 32768, "max_input_tokens": 163840,
            "max_output_tokens": 32768, "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0, "litellm_provider": "openai",
            "mode": "chat", "supports_function_calling": True}
    with open(registry, "w", encoding="utf-8") as f:
        json.dump({arms.MODEL_NAME: zero, arms.PUBLIC_MODEL: zero}, f, indent=1)
    overlay = {"model": {
        "model_name": arms.MODEL_NAME,
        "model_kwargs": {
            "api_base": api_base,
            "extra_headers": {"X-Yamadori-Features": arms.header(arm)},
            "extra_body": {"reasoning_effort": arms.BODY_EFFORT},
            "timeout": 3600,
            # null, not the shipped 0.0: every v2 leaderboard submission ran
            # with temperature None (the provider's default) -- see
            # OVERLAY_DEVIATIONS. litellm drops a None, so the server's own
            # sampling (--temp 1.0, Qwen's recommendation) applies.
            "temperature": None,
        },
        "litellm_model_registry": registry,
        "cost_tracking": "ignore_errors",
    }}
    # mini-swe-agent's -c only accepts a .yaml suffix; JSON is valid YAML.
    path = os.path.join(out, "config_overlay.yaml")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overlay, f, indent=1)
    return path


def cmd_ids(a) -> int:
    """The instance ids of a subset, as one JSON line."""
    from datasets import load_dataset
    ds = load_dataset(arms.DATASETS[a.subset], split="test")
    print(json.dumps(sorted(ds["instance_id"])))
    return 0


def cmd_check(_a) -> int:
    cfg = default_config()
    print(json.dumps({"versions": versions(), "config": cfg,
                      "config_sha256": sha256(cfg)}))
    return 0


def apply_override(a) -> None:
    """<run-dir>/arm_override.json, e.g. {"yamadori": "yamadori-auto"}, sends
    every job still queued for one arm to another arm, in that arm's own
    directory, without restarting the runner (this file is re-read per job).
    The switch and its reason are the operator's decision, recorded in the
    file itself and in docs/SWE-BENCH.md."""
    run_dir = os.path.dirname(os.path.abspath(a.out))
    p = os.path.join(run_dir, "arm_override.json")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        m = json.load(f).get("map") or {}
    arm = getattr(a, "arm", None) or os.path.basename(os.path.abspath(a.out))
    new = m.get(arm)
    if new and new in arms.ARMS:
        a.arm = new
        a.out = os.path.join(run_dir, new)
        os.makedirs(a.out, exist_ok=True)


def cmd_agent(a) -> int:
    apply_override(a)
    if a.arm not in arms.ARMS:
        sys.exit(f"unknown arm {a.arm!r}; known: {sorted(arms.ARMS)}")
    os.makedirs(a.out, exist_ok=True)
    key = _read_key(a.key_file)
    api_base = resolve_api_base(a.api_base)
    # Aliveness with the key (PROTOCOL rule 1): the proxy must list our model.
    code, body = _get(api_base + "/models", key)
    if code != 200 or arms.PUBLIC_MODEL.encode() not in body:
        sys.exit(f"GET {api_base}/models -> {code}; expected 200 listing "
                 f"{arms.PUBLIC_MODEL!r} (check the key file and the proxy)")
    # Each agent run writes into its OWN directory and is merged into the
    # arm's preds.json / <instance>/ under a lock afterwards: two workers on
    # the same arm would otherwise race on mini's read-modify-write of
    # preds.json (its lock is per process).
    work = os.path.join(a.out, "_work", a.instance)
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    overlay = write_overlay(work, a.arm, api_base)
    cfg = default_config()
    meta_path = os.path.join(a.out, "run.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"arm": a.arm, "features": arms.ARMS[a.arm],
                       "body_reasoning_effort": arms.BODY_EFFORT,
                       "dataset": arms.DATASETS[a.subset], "split": "test",
                       "api_base": api_base, "versions": versions(),
                       "leaderboard_config": cfg,
                       "leaderboard_config_sha256": sha256(cfg),
                       "overlay": overlay, "deviations": OVERLAY_DEVIATIONS,
                       "created": time.time()}, f, indent=1)
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = key
    env.pop("OPENAI_BASE_URL", None)
    env.pop("OPENAI_API_BASE", None)
    env["MSWEA_CONFIGURED"] = "true"    # no first-run setup prompt in batch mode
    # A private, empty global-config dir: nothing in a user's ~/.config .env
    # (a model name, a cost limit) can leak into the benchmark.
    env["MSWEA_GLOBAL_CONFIG_DIR"] = os.path.join(os.path.dirname(VENV), "mswea-global")
    env["MSWEA_SILENT_STARTUP"] = "1"
    env["PYTHONUTF8"] = "1"
    env["SWEBENCH_PROGRESS"] = os.path.join(a.out, "progress.log")
    env["SWEBENCH_INSTANCE"] = a.instance
    # A 429 from the proxy ("all main lanes busy") is admission, not a model
    # result: the request never ran (AGENTS.md: treat a 429 as "not run").
    # mini's default of 10 attempts (~6 min of backoff) is shorter than a
    # shared card's queue, and exhausting it fails the instance on plumbing.
    # Retries happen inside one model query, so they never add a step.
    env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "5000"
    cmd = [*MINI_CMD, "--subset", arms.DATASETS[a.subset],
           "--split", "test", "--filter", f"^{a.instance}$", "-o", work,
           "-w", "1", "-c", cfg, "-c", overlay, "--redo-existing"]
    # Pull the image first, outside the timed agent run: a first-time pull of
    # a multi-GB image is network time, not model time, and mini-swe-agent's
    # own pull_timeout (120 s) would fail the instance on a slow link.
    image = ("docker.io/swebench/sweb.eval.x86_64."
             + a.instance.replace("__", "_1776_") + ":latest").lower()
    tp = time.time()
    have = subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0
    if not have:
        subprocess.run(["docker", "pull", "-q", image], capture_output=True)
    pull_secs = round(time.time() - tp, 1)
    # Per instance (two workers would interleave one shared file); merge()
    # moves it next to the trajectory.
    log = os.path.join(work, "mini_stdout.log")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== {a.instance} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.flush()
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    secs = time.time() - t0
    merge(a.out, work, a.instance)
    traj = os.path.join(a.out, a.instance, f"{a.instance}.traj.json")
    rec = {"instance": a.instance, "arm": a.arm, "rc": rc,
           "seconds": round(secs, 1), "pull_seconds": pull_secs,
           "started": t0, "traj": os.path.exists(traj)}
    if rec["traj"]:
        with open(traj, encoding="utf-8") as f:
            info = json.load(f).get("info") or {}
        rec["exit_status"] = info.get("exit_status")
        rec["api_calls"] = (info.get("model_stats") or {}).get("api_calls")
    with open(os.path.join(a.out, "timings.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return 0 if rc == 0 else rc


class _Lock:
    """A cross-process lock on one arm directory (O_EXCL lock file)."""

    def __init__(self, d: str):
        self.p = os.path.join(d, ".merge.lock")

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - t0 > 120:     # a crashed holder: take it over
                    try:
                        os.remove(self.p)
                    except OSError:
                        pass
                time.sleep(0.2)

    def __exit__(self, *exc):
        os.close(self.fd)
        try:
            os.remove(self.p)
        except OSError:
            pass


def merge(out: str, work: str, iid: str) -> None:
    """Move one finished run from its work dir into the arm directory."""
    with _Lock(out):
        wp = os.path.join(work, "preds.json")
        entry = None
        if os.path.exists(wp):
            with open(wp, encoding="utf-8") as f:
                entry = json.load(f).get(iid)
        pp = os.path.join(out, "preds.json")
        preds = {}
        if os.path.exists(pp):
            with open(pp, encoding="utf-8") as f:
                preds = json.load(f)
        if entry is not None:
            preds[iid] = entry
        else:
            preds.pop(iid, None)
        tmp = pp + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(preds, f, indent=2)
        os.replace(tmp, pp)
        src, dst = os.path.join(work, iid), os.path.join(out, iid)
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)
            shutil.move(src, dst)
            for name in ("minisweagent.log", "mini_stdout.log"):
                log = os.path.join(work, name)
                if os.path.exists(log):
                    shutil.copy(log, os.path.join(dst, name))


def cmd_evaluate(a) -> int:
    apply_override(a)
    preds = os.path.join(a.out, "preds.json")
    if not os.path.exists(preds):
        sys.exit(f"no predictions at {preds}")
    with open(preds, encoding="utf-8") as f:
        ids = sorted(json.load(f))
    if a.instances:
        want = set(a.instances.split(","))
        ids = [i for i in ids if i in want]
    if not ids:
        print(json.dumps({"evaluate_rc": 0, "instances": 0}), flush=True)
        return 0
    cmd = [*EVAL_CMD,
           "--dataset_name", arms.EVAL_DATASETS[a.subset], "--split", "test",
           "--predictions_path", preds, "--instance_ids", *ids,
           "--max_workers", str(a.workers), "--run_id", a.run_id,
           "--report_dir", a.out,
           # Keep the pulled instance images: the other arms need the same
           # ones, and the default (env) would delete them after grading.
           "--cache_level", "instance"]
    log = os.path.join(a.out, "eval_stdout.log")
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== evaluate {a.run_id} {ids} {time.strftime('%H:%M:%S')}\n")
        lf.flush()
        rc = subprocess.run(cmd, cwd=a.out, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            env=dict(os.environ, PYTHONUTF8="1")).returncode
    print(json.dumps({"evaluate_rc": rc, "instances": len(ids), "log": log}),
          flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    i = sub.add_parser("ids")
    i.add_argument("--subset", default="verified", choices=sorted(arms.DATASETS))
    g = sub.add_parser("agent")
    g.add_argument("--arm", required=True)
    g.add_argument("--instance", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--key-file", required=True)
    g.add_argument("--subset", default="verified", choices=sorted(arms.DATASETS))
    g.add_argument("--api-base")
    g.add_argument("--redo", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--out", required=True)
    e.add_argument("--run-id", required=True)
    e.add_argument("--subset", default="verified", choices=sorted(arms.DATASETS))
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--instances", default="", help="comma-separated subset")
    a = ap.parse_args()
    return {"check": cmd_check, "ids": cmd_ids, "agent": cmd_agent,
            "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
