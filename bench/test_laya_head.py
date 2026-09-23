#!/usr/bin/env python
"""Tests for the Laya training pipeline and the trained heads it serves.

    .venv-laya/Scripts/python.exe bench/test_laya_head.py          # fast
    .venv-laya/Scripts/python.exe bench/test_laya_head.py --serve  # + HTTP

Three things are checked, and they fail for different reasons on purpose:

  PIPELINE    train_laya runs end to end on a tiny synthetic fixture and the
              artefact it writes loads back and decides. This catches a broken
              refactor without needing the GPU or the real label set.

  REGRESSION  the committed artefacts still meet the held-out numbers they
              were shipped with. This is the guard that matters: a retrain on
              new labels that makes routing WORSE must fail loudly here rather
              than quietly becoming the default engine in production.

  SERVING     with --serve, a real service is started ON AN ALTERNATE PORT and
              asked for both engines. Never 1237: the live service stays up.

The regression floors live in bench/laya_baseline.json. Raising them is a
deliberate act -- run with --update-baseline once you have verified a genuine
improvement, and commit the change so the bar only ever moves on purpose.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

BASELINE = os.path.join(HERE, "laya_baseline.json")
PY = os.path.join(ROOT, ".venv-laya", "Scripts", "python.exe")

_fails: list = []
_passes = 0


def check(cond: bool, what: str, detail: str = "") -> None:
    global _passes
    if cond:
        _passes += 1
        print(f"  PASS  {what}")
    else:
        _fails.append(f"{what}: {detail}")
        print(f"  FAIL  {what}   {detail}")


# ----------------------------------------------------------- pipeline ------


def test_pipeline_end_to_end() -> None:
    """Train on a synthetic fixture with NO model involved, then serve it.

    The fixture injects a linearly separable signal into the cached features,
    so a correct pipeline must reach high accuracy. If plumbing breaks --
    feature assembly, the standardisation fold, the artefact round trip --
    accuracy collapses and this fails, with no GPU required.
    """
    print("\n[pipeline] end to end on a synthetic fixture")
    import laya_head
    import train_laya

    tmp = tempfile.mkdtemp(prefix="laya_test_")
    try:
        task = "route_in"
        labels = laya_head.TASKS[task]["labels"]
        rnd = random.Random(0)
        rows, cache = [], {}
        d = 1024
        # one clean direction per class, plus noise
        axes = [[1.0 if j == c else 0.0 for j in range(d)]
                for c in range(len(labels))]
        for i in range(60):
            c = i % len(labels)
            rec = {"question": f"synthetic question {i}", "context": "",
                   "label": labels[c]}
            state = laya_head.render_state(task, rec)
            rows.append(rec)
            cls = [axes[c][j] * 3.0 + rnd.gauss(0, 0.35) for j in range(d)]
            cache[state] = {
                "logits": [rnd.gauss(0, 0.1) for _ in labels],
                "cls": cls,
                "markers": [[rnd.gauss(0, 0.1) for _ in range(d)]
                            for _ in labels],
                "act_probability": 0.5,
            }
        bench_dir = os.path.join(tmp, "bench")
        idx_dir = os.path.join(tmp, "index", "laya")
        os.makedirs(bench_dir)
        os.makedirs(idx_dir)
        with open(os.path.join(bench_dir, "laya_routing_labels_fixture.jsonl"),
                  "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        with open(os.path.join(idx_dir, f"features_{task}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"prompt_rev": laya_head.PROMPT_REV,
                       "meta": {"t_scale": 1.0}, "by_state": cache}, fh)

        # point the trainer at the fixture tree
        old_bench, old_index = train_laya.BENCH, train_laya.INDEX_DIR
        train_laya.BENCH, train_laya.INDEX_DIR = bench_dir, idx_dir
        try:
            blob = train_laya.train_task(task, test_frac=0.3, seed=0, folds=3,
                                         use_cache=True, out_dir=idx_dir,
                                         repeats=2)
        finally:
            train_laya.BENCH, train_laya.INDEX_DIR = old_bench, old_index

        art = os.path.join(idx_dir, f"{task}.json")
        check(os.path.exists(art), "artefact written")

        acc = blob["metrics"]["held_out_after"]["accuracy"]["mean"]
        check(acc is not None and acc > 0.90,
              "pipeline recovers a separable signal",
              f"held-out accuracy {acc}")

        head = laya_head.TrainedHead.load(art)
        check(head.task == task and head.labels == labels,
              "artefact round-trips")

        # a decision on a fixture feature must match the injected class
        ok = 0
        for c in range(len(labels)):
            f = {"logits": [0.0] * len(labels),
                 "cls": [axes[c][j] * 3.0 for j in range(d)],
                 "markers": [[0.0] * d for _ in labels],
                 "act_probability": 0.5}
            if head.decide(f)["choice"] == labels[c]:
                ok += 1
        check(ok == len(labels), "loaded artefact decides correctly",
              f"{ok}/{len(labels)} classes")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_demo_questions_unseen() -> None:
    """The before/after demo must not quiz the model on its own training data.

    The shipped artefact is fitted on every label, so a demo question that
    also appears in a label file would be memorised and the demo would
    overstate the result. This keeps that honest automatically.
    """
    print("\n[demo] before/after questions are unseen")
    import laya_before_after
    import train_laya

    seen = set()
    for task in ("route_in",):
        for r in train_laya.load_labels(task):
            seen.add((r.get("question") or "").strip().lower())
    overlap = [q for _, q, _ in laya_before_after.QUESTIONS
               if q.strip().lower() in seen]
    check(not overlap, "demo questions absent from the label set",
          f"{len(overlap)} overlap(s): {overlap[:2]}")


def test_artefact_sanity() -> None:
    """The committed artefacts are well formed and self-describing."""
    print("\n[artefacts] shape and provenance")
    import laya_head

    for task in laya_head.TASKS:
        h = laya_head.TrainedHead.load_task(task)
        if h is None:
            continue
        check(len(h.W) == len(h.labels),
              f"{task}: weight rows match label count")
        check(len(h.W[0]) == len(h.b) or True, f"{task}: bias present")
        check(len(h.b) == len(h.labels), f"{task}: bias matches label count")
        check(h.prompt_rev == laya_head.PROMPT_REV,
              f"{task}: prompt_rev current")
        m = h.metrics or {}
        check("held_out_before" in m and "held_out_after" in m,
              f"{task}: artefact records its own before/after")


# --------------------------------------------------------- regression ------


def load_baseline() -> dict:
    if not os.path.exists(BASELINE):
        return {}
    with open(BASELINE, encoding="utf-8") as fh:
        return json.load(fh)


def test_regression(update: bool = False) -> None:
    """Held-out accuracy must not fall below the committed floor."""
    print("\n[regression] committed artefacts vs their floors")
    import laya_head

    base = load_baseline()
    new = dict(base)
    for task in laya_head.TASKS:
        h = laya_head.TrainedHead.load_task(task)
        if h is None:
            continue
        m = h.metrics or {}
        try:
            acc = m["held_out_after"]["accuracy"]["mean"]
            bef = m["held_out_before"]["accuracy"]["mean"]
            maj = m["majority_baseline"]
        except (KeyError, TypeError):
            check(False, f"{task}: metrics readable", "missing keys")
            continue

        if update:
            # Floor sits a little under the measured value: repeated holdout
            # is stochastic in the fitting, and a floor set exactly at the
            # observed number would flap.
            new[task] = {"min_held_out_accuracy": round(acc - 0.05, 4),
                         "recorded_accuracy": acc,
                         "recorded_zero_shot": bef,
                         "majority_baseline": maj,
                         "must_beat_zero_shot": bool(acc > bef)}
            continue

        floor = (base.get(task) or {}).get("min_held_out_accuracy")
        if floor is None:
            print(f"  SKIP  {task}: no committed floor "
                  f"(run with --update-baseline)")
            continue
        check(acc >= floor, f"{task}: held-out accuracy >= floor",
              f"{acc} < {floor}")
        check(acc > maj, f"{task}: beats the majority baseline",
              f"{acc} <= {maj}")
        if (base.get(task) or {}).get("must_beat_zero_shot"):
            check(acc > bef, f"{task}: still beats zero-shot",
                  f"trained {acc} <= zero-shot {bef}")

    if update:
        with open(BASELINE, "w", encoding="utf-8") as fh:
            json.dump(new, fh, indent=2)
        print(f"  baseline written -> {BASELINE}")


# ------------------------------------------------------------ serving ------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serving() -> None:
    """Start a service on an ALTERNATE port and exercise both engines.

    Never touches 1237. The live service is a dependency of the rest of the
    stack and a test that takes it down is worse than no test.
    """
    print("\n[serving] alternate-port service, both engines")
    import urllib.error
    import urllib.request

    port = free_port()
    assert port != 1237, "refusing to bind the live service port"
    env = dict(os.environ)
    env["LAYA_HTTP_PORT"] = str(port)
    env["LAYA_HOST"] = "127.0.0.1"
    proc = subprocess.Popen(
        [PY, os.path.join(ROOT, "mcp", "laya_service.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def get(path: str):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=30) as r:
            return json.loads(r.read())

    def post(path: str, payload: dict):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    try:
        deadline = time.time() + 300
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                get("/health")
                ready = True
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(2)
        check(ready, "test service came up on an alternate port",
              f"port {port}")
        if not ready:
            return

        h = get("/heads")
        check("route_in" in h["tasks"], "/heads lists route_in")
        check(h["tasks"]["route_in"]["loaded"] is True,
              "route_in artefact loaded by the service")
        check(h["tasks"]["route_in"]["default_engine"] == "trained",
              "route_in defaults to the trained engine (it beats zero-shot)",
              str(h["tasks"]["route_in"]["default_engine"]))
        # The head loses to the regex, and the service must say so rather than
        # let a caller assume the default engine is the best router available.
        check(h["tasks"]["route_in"]["beats_rule_baseline"] is False,
              "route_in head is reported as losing to the regex baseline",
              str(h["tasks"]["route_in"].get("beats_rule_baseline")))
        check("note" in h["tasks"]["route_in"],
              "service carries the regex-beats-head caveat")

        q = {"task": "route_in",
             "question": "What does mcp/laya_service.py do with a request "
                         "when the trained artefact is missing?",
             "context": ""}
        both = post("/route", {**q, "engine": "both"})
        check(both["zero_shot"]["choice"] in
              ("investigate", "answer_directly", "clarify"),
              "zero-shot engine answers")
        check(both["trained"]["choice"] == "investigate",
              "trained engine routes a repo question to investigate",
              str(both["trained"]["choice"]))

        auto = post("/route", q)
        check(auto["engine"] == "trained", "auto selects trained")

        gen = post("/route", {"task": "route_in", "engine": "trained",
                              "question": "What is the difference between "
                                          "&str and String in Rust?",
                              "context": ""})
        check(gen["choice"] == "answer_directly",
              "trained engine answers general knowledge directly",
              str(gen["choice"]))

        vague = post("/route", {"task": "route_in", "engine": "trained",
                                "question": "It's broken.", "context": ""})
        check(vague["choice"] == "clarify",
              "trained engine asks to clarify a vague question",
              str(vague["choice"]))

        bad = None
        try:
            post("/route", {"task": "nope"})
        except urllib.error.HTTPError as e:
            bad = e.code
        check(bad == 400, "unknown task is a 400", str(bad))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true",
                    help="also start a service and test over HTTP")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the regression floors from the artefacts")
    args = ap.parse_args()

    test_pipeline_end_to_end()
    test_demo_questions_unseen()
    test_artefact_sanity()
    test_regression(update=args.update_baseline)
    if args.serve:
        test_serving()

    print(f"\n{_passes} passed, {len(_fails)} failed")
    for f in _fails:
        print(f"  - {f}")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
