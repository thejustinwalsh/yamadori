"""Drive one LiveBench arm over a fixed question order, one request in flight.

Runs in the WSL venv that has LiveBench installed (see README.md). It calls
LiveBench's own gen_api_answer.py in chunks (one chunk = one question per task,
in the order make_sample.py fixed), so a run stopped by --deadline has answered
a stratified prefix and the other arm can be run on exactly that prefix.

It never talks to the model itself. What it adds around the harness:

* arm -> header. The X-Yamadori-Features JSON and the answer-file display name
  come from ONE table (ARMS), so an arm cannot be run under the other's name.
  Every answer's api_info.x_yamadori is then checked against the arm; a
  mismatch stops the run (the proxy silently drops an unparseable header).
* errors are not answers. A `$ERROR$` row whose error does not name
  finish_reason=length is a transport failure (429, 5xx, timeout): it is
  logged to errors.jsonl, removed from the answer file and retried (max
  --attempts). One that names finish_reason=length is a budget event and is
  kept (it scores 0 in LiveBench, and we report it as a length finish).
* the key is read from --key-file into the child's LIVEBENCH_API_KEY and is
  never printed or written.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanisms  # noqa: E402

LB_ROOT = os.path.expanduser("~/livebench-run/LiveBench/livebench")

# Two kinds of arm:
#  * "features": a benchmark arm forced through the X-Yamadori-Features header.
#  * "tier": a PRODUCT arm -- body reasoning_effort exactly as a client
#    (Hermes) sends it, NO header, so mcp/selection.py decides per request
#    within what the tier allows. The effort itself rides in the model config
#    (patches/yamadori_local.yml api_kwargs) under the same display name.
# Operator decision 2026-09-23 ~07:30: the augmentation toggles ARE the tiers.
ARMS = {
    "bonsai": {  # "bare @ medium": everything forced off, effort medium -- the effort-matched baseline
        "display": "yamadori-bonsai-arm",
        "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1, "effort": "medium"},
    },
    "yamadori": {  # product tier max (proxy sends effort xhigh upstream)
        "display": "yamadori-tier-max",
        "tier": "max", "effort_sent": "xhigh",
        "allowed": {"investigate": True, "fanout": 3},
    },
    "yamadori-xhigh": {  # product tier xhigh: every augmentation at MEDIUM effort -- the all-on arm
        # matched to bonsai "bare @ medium" (operator, 2026-09-23 ~11:10; mcp/tiers.py "xhigh")
        "display": "yamadori-tier-xhigh",
        "tier": "xhigh", "effort_sent": "medium",
        "allowed": {"investigate": True, "fanout": 3},
    },
    "minimal": {  # product tier minimal, REDEFINED ~11:30 2026-09-23: thinking OFF
        # (enable_thinking=false, vendor instruct sampling 0.7/0.80/20/presence 1.5). No
        # minimal answers exist from the old definition (thinking at low). effort_sent is not
        # asserted until a gate answer shows what the new tier records.
        "display": "yamadori-tier-minimal",
        "tier": "minimal", "effort_sent": None,
        "allowed": {"investigate": False, "fanout": 1},
    },
    # check_code / repair (mcp/code_check.py), each paired against `bonsai` on
    # the same questions (--paired-with bonsai). The bonsai header plus the
    # forced key(s), nothing else. Proof on each answer: x_yamadori.check_code
    # .offered is true (tool arms) and x_yamadori.repair.enabled is true
    # (repair arms); a proxy started before 2026-09-23's check_code change
    # DROPS the unknown keys silently and runs these as plain bonsai;
    # check_arm below refuses such an answer.
    "bonsai+check": {  # bare model + the check_code tool + the repair pass
        "display": "yamadori-bonsai-check-arm",
        "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1, "effort": "medium",
                     "check_code": True, "repair": True},
    },
    "bonsai+check-tool": {  # bare model + the check_code tool only
        "display": "yamadori-bonsai-checktool-arm",
        "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1, "effort": "medium",
                     "check_code": True},
    },
    "bonsai+repair": {  # bare model + the repair pass only (no tool offered)
        "display": "yamadori-bonsai-repair-arm",
        "features": {"retrieval": False, "hints": False, "investigate": False, "fanout": 1, "effort": "medium",
                     "repair": True},
    },
    # Retired 07:30: the header-forced full stack. Its answers were produced
    # before the fan-out winner fix (shortest candidate won) and are set aside.
    "yamadori_features_prefix": {
        "display": "yamadori-full-arm",
        "features": {"retrieval": True, "hints": True, "investigate": True, "fanout": 3, "effort": "medium"},
    },
}


def answer_path(category: str, task: str, display: str) -> str:
    return os.path.join(LB_ROOT, "data", "live_bench", category, task, "model_answer", f"{display}.jsonl")


def read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_error(row: dict) -> bool:
    t = row["choices"][0]["turns"]
    return t == "$ERROR$" or (isinstance(t, list) and t and t[0] == "$ERROR$")


def is_incomplete(row: dict) -> bool:
    """The proxy's marker for an upstream connection drop (mcp/proxy.py:
    finish_reason "incomplete", a partial answer prefixed "[the connection to
    the model dropped ...]"). A stack failure, never an answer: re-asked."""
    return (row.get("api_info") or {}).get("finish_reason") == "incomplete"


def is_budget_event(row: dict) -> bool:
    return "finish_reason=length" in (row.get("error_msg") or "")


def check_arm(row: dict, arm: str) -> str | None:
    """None if the proxy's decision record matches the arm, else why not."""
    if is_error(row):
        return None
    xy = (row.get("api_info") or {}).get("x_yamadori")
    if not xy:
        return "answer carries no x_yamadori: arm not proven"
    sel = xy.get("selection") or {}
    forced = set((sel.get("signals") or {}).get("forced") or [])
    spec = ARMS[arm]
    if "tier" in spec:
        # product arm: right tier, right effort upstream, nothing forced, and
        # the tier ALLOWED what it should (selection then decides per request)
        allowed = (sel.get("signals") or {}).get("allowed") or {}
        if xy.get("tier") != spec["tier"]:
            return f"tier {xy.get('tier')!r} != {spec['tier']!r}"
        if spec["effort_sent"] is not None and xy.get("effort_sent") != spec["effort_sent"]:
            return f"effort_sent {xy.get('effort_sent')!r} != {spec['effort_sent']!r}"
        if forced:
            return f"flags forced on a product arm (a header leaked?): {sorted(forced)}"
        if bool(allowed.get("investigate")) != spec["allowed"]["investigate"] \
                or int(allowed.get("fanout") or 1) != spec["allowed"]["fanout"]:
            return f"allowed {allowed} != {spec['allowed']}"
        return None
    want = spec["features"]
    # a proxy that predates a feature key DROPS it silently and runs plain
    # bonsai: prove the check_code tool / repair pass were actually on
    if want.get("check_code") and not (xy.get("check_code") or {}).get("offered"):
        return "check_code forced but x_yamadori.check_code.offered is false (proxy dropped the key?)"
    if want.get("repair") and not (xy.get("repair") or {}).get("enabled"):
        return "repair forced but x_yamadori.repair.enabled is not true (proxy dropped the key?)"
    if not {"hints", "investigate", "fanout"} <= forced:
        return f"flags not forced by header: forced={sorted(forced)}"
    if sel.get("fanout_n") != want["fanout"] or bool(sel.get("investigate")) != want["investigate"] \
            or bool(sel.get("hints")) != want["hints"]:
        return f"selection {sel.get('fanout_n')}/{sel.get('investigate')}/{sel.get('hints')} != arm {arm}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=0, help="prefix of the order to run (0 = all)")
    ap.add_argument("--paired-with", choices=sorted(ARMS), default=None,
                    help="restrict the order to questions this other arm has answered")
    ap.add_argument("--deadline", type=float, default=0, help="epoch seconds; no new chunk starts after it")
    ap.add_argument("--parallel-file", default=None,
                    help="file holding the requests-in-flight for the NEXT chunk (1-4); read every chunk")
    ap.add_argument("--api-base", default="http://172.29.32.1:1234/v1")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--attempts", type=int, default=12,
                    help="transport failures tolerated per question before it is reported not-run")
    ap.add_argument("--backoff", type=float, default=90, help="seconds to wait after a chunk with a transport failure")
    ap.add_argument("--max-tokens", type=int, default=4096, help="LiveBench's default; the proxy adds thinking on top")
    a = ap.parse_args()

    spec = json.load(open(os.path.join(a.run_dir, f"order_{a.category}.json")))
    release, order = spec["release"], spec["order"]
    if a.paired_with:
        # only questions the other arm has a real answer for, in the same order
        other = ARMS[a.paired_with]["display"]
        answered = {r["question_id"] for t in {x["task"] for x in order}
                    for r in read_rows(answer_path(a.category, t, other)) if not is_error(r)}
        order = [r for r in order if r["question_id"] in answered]
        print(f"paired with {a.paired_with}: {len(order)} questions", flush=True)
    if a.n:
        order = order[: a.n]
    tasks = sorted({r["task"] for r in order})
    display = ARMS[a.arm]["display"]
    log_dir = os.path.join(a.run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{a.arm}_{a.category}.log")
    err_path = os.path.join(a.run_dir, "errors.jsonl")

    env = dict(os.environ)
    with open(a.key_file) as f:
        env["LIVEBENCH_API_KEY"] = f.read().strip()
    if "features" in ARMS[a.arm]:
        env["LB_EXTRA_HEADERS"] = json.dumps({"X-Yamadori-Features": json.dumps(ARMS[a.arm]["features"])})
    else:
        env.pop("LB_EXTRA_HEADERS", None)  # product arm: no header at all
    env["LB_NO_SDK_RETRY"] = "1"  # the harness's 429 loop is the only retry policy
    env["LB_TIMEOUT"] = env.get("LB_TIMEOUT", "7200")

    def parallel_now() -> int:
        """Requests in flight for the next chunk: read from --parallel-file each chunk, clamped 1..4."""
        if not a.parallel_file:
            return 1
        try:
            return max(1, min(4, int(open(a.parallel_file).read().strip())))
        except (OSError, ValueError):
            return 1

    attempts: dict[str, int] = {}
    submitted: dict[str, int] = {}
    t0 = time.time()

    def done_ids() -> dict[str, dict]:
        out = {}
        for t in tasks:
            for r in read_rows(answer_path(a.category, t, display)):
                out[r["question_id"]] = r
        return out

    def sweep_errors() -> int:
        """Move transport-error rows out of the answer files; return how many."""
        moved = 0
        for t in tasks:
            p = answer_path(a.category, t, display)
            rows = read_rows(p)
            keep = []
            for r in rows:
                # a mechanism that BROKE (tool error, stack failure) is a
                # stack error: the answer is not scored and is re-asked. The
                # full row is kept in stack_errors.jsonl as evidence.
                stack = [] if is_error(r) else \
                    mechanisms.record((r.get("api_info") or {}).get("x_yamadori")).get("stack_error") or []
                if stack and "no x_yamadori on the answer" in stack:
                    stack = []  # check_arm stops the run on this; it is not a retryable fault
                if not is_error(r) and is_incomplete(r):
                    stack = stack + ["finish_reason incomplete: upstream connection dropped mid-answer"]
                if stack:
                    with open(os.path.join(a.run_dir, "stack_errors.jsonl"), "a") as f:
                        f.write(json.dumps({"arm": a.arm, "category": a.category, "task": t,
                                            "stack_error": stack, "tstamp": time.time(), "row": r}) + "\n")
                if (is_error(r) and not is_budget_event(r)) or stack:
                    moved += 1
                    attempts[r["question_id"]] = attempts.get(r["question_id"], 0) + 1
                    with open(err_path, "a") as f:
                        f.write(json.dumps({"arm": a.arm, "category": a.category, "task": t,
                                            "question_id": r["question_id"],
                                            "error": "stack_error" if stack else r.get("error"),
                                            "error_msg": ("; ".join(stack) if stack else (r.get("error_msg") or ""))[:300],
                                            "attempt": attempts[r["question_id"]], "tstamp": time.time()}) + "\n")
                    if attempts[r["question_id"]] >= a.attempts:
                        keep.append(r)  # give up: it stays an error row, reported as not-run
                else:
                    keep.append(r)
            if len(keep) != len(rows):
                with open(p, "w") as f:
                    for r in keep:
                        f.write(json.dumps(r) + "\n")
        return moved

    chunk = len(tasks)
    i = 0
    while True:
        sweep_errors()
        have = done_ids()
        pending = [r for r in order if r["question_id"] not in have
                   and submitted.get(r["question_id"], 0) <= a.attempts]
        if not pending:
            break
        if a.deadline and time.time() > a.deadline:
            print(f"deadline reached; {len(pending)} of {len(order)} not started", flush=True)
            break
        batch = pending[:chunk]
        ids = [r["question_id"] for r in batch]
        for x in ids:
            submitted[x] = submitted.get(x, 0) + 1
        # "live_bench" + --question-id, never "live_bench/<category>": LiveBench
        # takes split('_')[0] of the category name, so instruction_following
        # and data_analysis would load nonexistent datasets.
        cmd = [sys.executable, "gen_api_answer.py", "--bench-name", "live_bench",
               "--model", display, "--api-base", a.api_base, "--livebench-release-option", release,
               "--question-id", *ids, "--parallel", str(parallel_now()), "--resume", "--no-incremental-grading",
               "--max-tokens", str(a.max_tokens)]
        tb = time.time()
        with open(log_path, "a") as lf:
            lf.write(f"\n=== chunk {i} {time.strftime('%H:%M:%S')} ids={ids}\n")
            lf.flush()
            rc = subprocess.run(cmd, cwd=LB_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        i += 1
        have = done_ids()
        for r in batch:
            row = have.get(r["question_id"])
            if row is None:
                continue
            bad = check_arm(row, a.arm)
            if bad:
                print(f"ARM CHECK FAILED on {r['question_id']}: {bad}. Stopping.", flush=True)
                return 2
        n_done = sum(1 for r in order if r["question_id"] in have and not is_error(have[r["question_id"]]))
        n_err = sum(1 for r in order if r["question_id"] in have and is_error(have[r["question_id"]]))
        msgs = [f"{have[x]['api_info'].get('finish_reason')}" for x in ids if x in have and not is_error(have[x])]
        print(f"[{time.strftime('%H:%M:%S')}] {a.arm}/{a.category} chunk {i} rc={rc} "
              f"{time.time()-tb:.0f}s  answered {n_done}/{len(order)} err_rows {n_err}  finish={msgs}  "
              f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)
        # Back off after ANY transport failure: a 429 (lanes held by another
        # benchmark) or a restart (llama-swap reload takes 1-2 min) would
        # otherwise burn every retry within seconds and mark the question not-run.
        if any(x in have and is_error(have[x]) and not is_budget_event(have[x]) for x in ids):
            time.sleep(a.backoff)
        if rc != 0 and not any(x in have for x in ids):
            print(f"harness exited {rc} and wrote no row for this chunk; stopping (see {log_path})", flush=True)
            return 1
    have = done_ids()
    print(f"FINISHED {a.arm}/{a.category}: {sum(1 for r in order if r['question_id'] in have)}/{len(order)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
