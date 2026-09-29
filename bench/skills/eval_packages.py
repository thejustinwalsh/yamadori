#!/usr/bin/env python
"""THE PACKAGE DETECTOR, measured offline (mcp/skill_packages.py).

    python bench/skills/eval_packages.py [--labels] [--stub] [--stack] [--h4] [-v]
    (no flag: all four)

  --labels  bench/skills/package_detect_labels.jsonl: implicit-use prompts
            that never name the package, and near misses. Precision and
            recall per package and overall. IN-SAMPLE (labelled by the agent
            that built the detector).
  --stub    the same prompts through the CURRENT selector (skill_select.
            decide, the evidence stub, deterministic stages only, on a copy
            of the live store: bench/skills/replay_selection.py's harness):
            which rows it fails, i.e. where the detector would add a check
            the daily eval does not yet have.
  --stack   the Octopus V4 and pagoda prompts (test material: the prompt
            text is never printed): the packages detected and by what.
  --h4      pagoda-h4's Hermes export (session.jsonl, the post-compaction
            part of the run) replayed request by request: what the package
            rule injects beside what h4 injected (relay.jsonl,
            x_yamadori.skills.decisions), aligned on the relay's last
            chat requests.

OFFLINE: nothing reaches a model or the GPU; the live store is copied first
(replay_selection does it on import) and only read.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
os.environ.setdefault("YAMADORI_SKILL_DECIDER", "stub")
_argv = sys.argv
sys.argv = [sys.argv[0]]
import replay_selection as R  # noqa: E402  (copies the store to a temp dir)
sys.argv = _argv

import package_registry  # noqa: E402
import skill_packages as SP  # noqa: E402
import skill_select  # noqa: E402

LABELS = package_registry.LABELS
H4 = os.path.join("C:/Users/jwals/octo/logs", "pagoda-h4-pagoda-xhigh-1")
# The selector's area for a package's skills (skill_select.area_of of the
# package's skills in the store), for judging the stub by area: the package
# registry's (its SEED is the map that was written here by hand).
PACKAGE_AREA = package_registry.package_area()
STACK_AREAS = sorted(set(PACKAGE_AREA.values()))


def rows(path=LABELS):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("//"):
                out.append(json.loads(ln))
    return out


def user(text):
    return [{"role": "system", "content": R.DAILY_SYSTEM},
            {"role": "user", "content": text}]


def labels(verbose=False):
    per = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    ex = {"tp": 0, "fp": 0, "fn": 0}
    bad = []
    for r in rows():
        got = SP.detect(user(r["text"]))
        want = set(r["expect"])
        have = set(got)
        for p in have | want:
            k = "tp" if p in have and p in want else "fp" if p in have \
                else "fn"
            per[p][k] += 1
            ex[k] += 1
        if have != want or verbose:
            bad.append((r["id"], sorted(want), {p: [f"{w['how']}:{w['what']}"
                                                   for w in e["why"]]
                                               for p, e in got.items()}))
    n_imp = sum(1 for r in rows() if not r.get("near"))
    n_near = sum(1 for r in rows() if r.get("near"))
    print(f"\n== labels: {n_imp} implicit-use prompts, {n_near} near misses "
          "(IN-SAMPLE)")
    for p, v in sorted(per.items()):
        pr = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else None
        rc = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None
        print(f"  {p:<28} precision {'-' if pr is None else f'{pr:.2f}':>5} "
              f"recall {'-' if rc is None else f'{rc:.2f}':>5}  "
              f"(tp {v['tp']} fp {v['fp']} fn {v['fn']})")
    pr = ex["tp"] / (ex["tp"] + ex["fp"]) if ex["tp"] + ex["fp"] else 0
    rc = ex["tp"] / (ex["tp"] + ex["fn"]) if ex["tp"] + ex["fn"] else 0
    print(f"  ALL (package decisions)      precision {pr:.2f} recall {rc:.2f}"
          f"  (tp {ex['tp']} fp {ex['fp']} fn {ex['fn']})")
    near_fp = sum(1 for r in rows() if r.get("near") and SP.detect(
        user(r["text"])))
    print(f"  near misses with any detection: {near_fp}/{n_near}")
    for rid, want, got in bad:
        print(f"    {rid}: want {want or '-'} got {got or '-'}")
    return per


def stub(verbose=False):
    pool = R.skills.armed()
    area = {s["name"]: skill_select.area_of(s.get("rule") or {})
            for s in pool}
    fails = []
    for r in rows():
        msgs = user(r["text"])
        try:
            rc = R.route.classify(msgs, client_tools=R.HERMES_TOOLS,
                                  util={"utility": False}, gate=None,
                                  dbs={})["class"]
        except Exception:                                        # noqa: BLE001
            rc = None
        _t, rec, _st = skill_select.decide(msgs, rc, pool, R.HERMES_TOOLS,
                                           None, key=f"pk:{r['id']}")
        names = [d["name"] for d in rec.get("decisions") or []]
        areas = {area.get(n) for n in names}
        want_areas = {PACKAGE_AREA.get(p) for p in r["expect"]} - {None}
        if r.get("near"):
            wrong = sorted(areas & set(STACK_AREAS))
            ok = not wrong
            why = f"injected {names}" if wrong else ""
        else:
            ok = bool(want_areas & areas) if want_areas else True
            why = f"injected {names or 'nothing'}"
        det = sorted(SP.detect(msgs))
        want = sorted(r["expect"])
        det_ok = det == want
        if not ok or verbose:
            fails.append((r["id"], ok, det_ok, why, sorted(want_areas)))
    print(f"\n== the current stub on the same prompts ({len(rows())})")
    nf = sum(1 for f in fails if not f[1])
    print(f"  stub fails {nf}; of those the detector gets right "
          f"{sum(1 for f in fails if not f[1] and f[2])}")
    for rid, ok, det_ok, why, wa in fails:
        if not ok:
            print(f"    {rid}: stub {why}; want area {wa or 'none'}; "
                  f"detector {'right' if det_ok else 'WRONG'}")
    return fails


def stack():
    print("\n== stack prompts (text never printed)")
    srcs = {"octopus V4 (exact)": "octopus_v4",
            "octopus original": "octopus",
            "pagoda r3f-stack (exact)": "pagoda_stack",
            "pagoda single-html": "pagoda"}
    out = {}
    for label, src in srcs.items():
        try:
            text = R._source_text(src)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {label}: not available ({type(e).__name__})")
            continue
        got = SP.detect(user(text))
        dec, _ = SP.decide(got, None)
        out[label] = got
        print(f"  {label}: " + ("; ".join(
            f"{p} v{e['version'] or '?'} <- " + ", ".join(
                f"{w['how']}:{w['what']}" for w in e["why"][:5])
            + (f" (+{len(e['why']) - 5})" if len(e["why"]) > 5 else "")
            for p, e in sorted(got.items())) or "nothing"))
        print("      injects: " + (", ".join(f"{d['skill']}({d['form']})"
                                            for d in dec if d["skill"])
                                  or "nothing"))
    return out


def _session_messages(path):
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return R._hermes_messages(s)


def _relay_chat_rows(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("path") not in ("/v1/responses", "/v1/chat/completions"):
                continue
            x = (r.get("response") or {}).get("x_yamadori") or {}
            if x.get("utility") or x.get("utility_kind"):
                continue
            out.append(x)
    return out


def h4(verbose=False):
    msgs = _session_messages(os.path.join(H4, "session.jsonl"))
    relay = _relay_chat_rows(os.path.join(H4, "relay.jsonl"))
    # A request is everything before an assistant message (the response it
    # got); the first is the restated user turn after the compaction.
    points = [i for i, m in enumerate(msgs) if m["role"] == "assistant"
              and i > 0]
    tail = relay[-len(points):] if len(relay) >= len(points) else []
    agree = 0
    for k, i in enumerate(points):
        ends = "user" if msgs[i - 1]["role"] == "user" else "tool"
        rx = tail[k] if k < len(tail) else {}
        sig = ((rx.get("route") or {}).get("signals") or {}).get("ends_on")
        agree += sig == ends
    print(f"\n== pagoda-h4 session.jsonl: {len(msgs)} messages, "
          f"{len(points)} requests; relay chat rows {len(relay)}; "
          f"alignment on the last {len(tail)}: ends_on agrees "
          f"{agree}/{len(points)}")
    state = None
    ours_total = collections.Counter()
    theirs_total = collections.Counter()
    first_seen: dict[str, int] = {}
    lines = []
    for k, i in enumerate(points):
        upto = msgs[:i]
        got = SP.detect(upto)
        dec, state = SP.decide(got, state)
        for p, e in got.items():
            first_seen.setdefault(p, k)
        ours = [f"{d['skill']}({d['form']})" for d in dec
                if d["form"] != "none"]
        rx = tail[k] if k < len(tail) else {}
        sk = rx.get("skills") or {}
        theirs = [f"{d.get('name')}({d.get('form')})"
                  for d in sk.get("decisions") or []]
        for x in ours:
            ours_total[x] += 1
        for x in theirs:
            theirs_total[x] += 1
        if ours or theirs or verbose:
            lines.append((k, i, ours, theirs,
                          {p: [f"{w['how']}:{w['what']}" for w in e["why"]][:3]
                           for p, e in got.items()}))
    for k, i, ours, theirs, why in lines:
        print(f"  req {k:>2} (msg {i:>3}): package rule {ours or '-'} | h4 "
              f"{theirs or '-'}")
        if verbose or ours:
            print(f"        in play: {why}")
    print(f"  package rule injected {sum(ours_total.values())} time(s): "
          f"{dict(ours_total)}")
    print(f"  h4 injected {sum(theirs_total.values())} time(s): "
          f"{dict(theirs_total)}")
    print(f"  first request each package was in play: {first_seen}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for f in ("labels", "stub", "stack", "h4"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args(argv)
    every = not (a.labels or a.stub or a.stack or a.h4)
    v = SP.vocabulary()
    print(f"vocabulary: {len(v['symbols'])} unique symbols over "
          f"{len(v['packages'])} packages")
    if every or a.labels:
        labels(a.v)
    if every or a.stub:
        stub(a.v)
    if every or a.stack:
        stack()
    if every or a.h4:
        h4(a.v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
