#!/usr/bin/env python
"""Offline checks for bench/kv_context. Sends nothing, loads nothing.

    python bench/kv_context/test_kv_context.py

Prints `N/M checks passed` (scripts/run_tests.py convention). What it proves:
  - the VRAM arithmetic reproduces its anchor and its measured inputs;
  - the synthetic filler never contains a needle or decoy name, is
    deterministic, and carries needle-shaped distractors;
  - config.yaml's trial entries are `bonsai` with exactly three flags changed,
    pinned to the same card, in a non-exclusive group, reachable only through
    the two profiles, and no profile is active at startup;
  - run.py's request body is thinking-off, features forced off, through the
    proxy; the ladders fit their pools; the plan runs with no network;
  - analyse.py's gates fail what they should fail and pass what they should
    pass, on fabricated rows (PROTOCOL rule 3: a check that cannot fail is
    not a check).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# bench/longctx has an analyse.py too: this directory must come FIRST.
sys.path.insert(0, os.path.join(ROOT, "bench", "longctx"))
sys.path.insert(0, HERE)

import analyse  # noqa: E402
import haystack as hs  # noqa: E402
import synth_code  # noqa: E402
import vram  # noqa: E402

RESULTS = []


def check(cond, what, detail=""):
    RESULTS.append(bool(cond))
    print(("  ok    " if cond else "  FAIL  ") + what + ("" if cond else f"  [{detail}]"))


def load_run():
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv_run", os.path.join(HERE, "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vram():
    check(vram.kv_kib("q8_0", "q8_0") == 34.0, "q8_0 K+V = the measured 34.0 KiB/token")
    check(vram.kv_kib("q4_0", "q4_0") == 18.0, "q4_0 K+V = 18.0 KiB/token")
    check(vram.slope_kib("q8_0", "q8_0", False) == 44.0,
          "q8_0 + compute + MTP = 44 KiB/token, budget.py's KV_KIB_PER_TOKEN")
    a = vram.project(163_840, "q8_0", "q8_0")
    check(a["used_mib"] == 14_340 and a["free_idle_mib"] == 1_711, "anchor row = anchor", a)
    hi = vram.project(262_144, conservative=True)
    lo = vram.project(262_144, conservative=False)
    check(hi["free_peak_mib"] < lo["free_peak_mib"], "conservative slope is the tighter one")
    check(1_000 < hi["free_peak_mib"] < 1_100, "262,144 conservative peak ~1.04 GB", hi)
    check(vram.largest_c(3_072) == 188_416 and vram.largest_c(3_072, conservative=False) == 196_608,
          "3 GB floor: 188,416 (conservative) .. 196,608 (component)")
    check(vram.project(262_144, "q4_0", "q8_0", conservative=False)["free_idle_mib"] < 0,
          "q4_0 K / q8_0 V at 262,144 does not fit the card")


def test_synth():
    ch = synth_code.chunks()
    corpus = "".join(t for _, t in ch)
    check(len(corpus) >= 1_800_000, "default corpus covers a 262k-token haystack", len(corpus))
    check(synth_code.fingerprint(ch) == synth_code.fingerprint(synth_code.chunks()),
          "generation is deterministic")
    check(synth_code.fingerprint(ch) != synth_code.fingerprint(synth_code.chunks(seed=8)),
          "the seed changes the corpus")
    hits = [x["name"] for i in range(200) for it in [hs.make_item(1, i)]
            for x in it["needles"] + it["decoys"] if x["name"] in corpus]
    check(not hits, "no needle or decoy name occurs in the filler (items 0-199, seed 1)", hits[:5])
    shape = re.findall(r"export function \w+\(\) \{\n  return \d{4};\n\}", corpus)
    check(len(shape) > 1000, "filler carries >1000 needle-shaped distractors", len(shape))
    words = set(synth_code.WORDS) | set(synth_code.VERBS)
    check(not (words & set(hs._SYL)), "vocabulary shares no whole word with the needle syllables")
    h = hs.build_haystack(hs.make_item(1, 4), 600_000, ch)
    off = max(abs(v["actual"] - v["target"]) for v in h["landed"].values())
    check(off < 0.01, "needles land within 1% of their depth in synthetic filler", off)


def test_config():
    try:
        import yaml
    except ImportError:
        check(False, "PyYAML available to read config.yaml")
        return
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    ms = cfg["models"]
    run = load_run()

    def flags(cmd):
        return [ln.strip() for ln in cmd.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    base = flags(ms["bonsai"]["cmd"])
    for c, (mid, prof) in run.CANDIDATES.items():
        m = ms.get(mid)
        check(m is not None, f"{mid} exists")
        if not m:
            continue
        f = flags(m["cmd"])
        extra = [x for x in f if x not in base]
        missing = [x for x in base if x not in f]
        check(sorted(extra) == sorted([f"-c {c}", "--cache-type-k q4_0 --cache-type-v q4_0",
                                       f"--kv-mean-center ${{models}}/{os.path.basename(run.BIAS)}"]),
              f"{mid}: exactly -c, KV types and the bias added", extra)
        check(sorted(missing) == sorted(["-c 163840", "--cache-type-k q8_0 --cache-type-v q8_0"]),
              f"{mid}: exactly bonsai's -c and KV types removed", missing)
        check(m["env"] == ms["bonsai"]["env"], f"{mid}: env and UUID pin identical to bonsai")
        check(m.get("unlisted") is True and int(m.get("ttl", 0)) > 0,
              f"{mid}: unlisted, ttl > 0 (watchdog treats it as on-demand)")
        check(m.get("filters") == ms["bonsai"].get("filters"), f"{mid}: same filters as bonsai")
        pins = cfg["profiles"][prof]["pins"]
        check(pins == {"bonsai": mid, "bonsai-agent": mid}, f"profile {prof} pins bonsai -> {mid}")
    groups = cfg["routing"]["router"]["settings"]["groups"]
    g = groups.get("context-trial") or {}
    check(g.get("exclusive") is False and g.get("swap") is True and not g.get("persistent"),
          "context-trial group: swap, not exclusive, not persistent", g)
    check(groups["primary"]["members"] == ["bonsai"] and groups["primary"]["persistent"],
          "primary group unchanged")
    check("profile" not in (cfg.get("hooks") or {}).get("on_startup", {}),
          "no profile is activated at startup")
    check(re.search(r"GPU-de660e90", " ".join(ms["bonsai-q4kv"]["env"])) is not None
          and run.CARD_UUID.startswith("GPU-de660e90"), "trial and guard read the 5060 Ti by UUID")


def test_run():
    run = load_run()
    for c in run.CANDIDATES:
        lad = [L for L in run.SHARED_LADDER + run.BEYOND_LADDER[c] if run.fits(L, c)]
        check(lad == run.SHARED_LADDER + run.BEYOND_LADDER[c], f"every rung fits the {c} pool", lad)
    check(all(run.fits(L, run.PROD_C) for L in run.SHARED_LADDER),
          "every shared rung fits q8's 163,840 pool")
    check(not run.fits(run.SHARED_LADDER[-1] + 8192, run.PROD_C),
          "the shared ladder ends at the q8 pool")
    p = run.plan(12, 3)
    check(p["arms"]["q4"]["accuracy_requests"] == 9 * 12 * 3 and p["hours_total"] > 0,
          "plan counts requests offline", p["arms"]["q4"]["accuracy_requests"])
    check(all(v is False for k, v in run.FEATURES.items() if k != "fanout")
          and run.FEATURES["fanout"] == 1, "every augmentation forced off")
    sent = {}

    class Fake:
        features = None

        def chat(self, body, stream):
            sent.update(body=body, stream=stream)
            return {"ok": True}
    px = run.Proxy.__new__(run.Proxy)
    px.client, px.run_id, px.session = Fake(), "t", ""
    px.ask([{"role": "user", "content": "x"}], "s", 256)
    b = sent["body"]
    check(b["reasoning_effort"] == "minimal" and b["model"] == "yamadori" and sent["stream"],
          "accuracy/speed body: model yamadori, tier minimal (thinking off), streamed", b)
    check(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", px.session) is not None,
          "session token is valid for X-Yamadori-Session", px.session)
    check(run.to_wsl(r"C:\Users\jwals\k.txt") == "/mnt/c/Users/jwals/k.txt", "WSL path translation")
    check("kv-q8" in run.ARMS_SNIPPET and '"check_code": False' in run.ARMS_SNIPPET,
          "the LiveBench arm snippet forces check_code off")
    check(set(run.GATES) == {"floor_mib", "speed_min_ratio", "acc_max_drop_points",
                             "acc_alpha", "quality_alpha", "quality_max_drop_questions"},
          "gates are the documented six")


def test_analyse():
    check(analyse.mcnemar(0, 6) == 0.03125 and analyse.mcnemar(1, 8) == 0.0390625
          and analyse.mcnemar(0, 0) == 1.0 and analyse.mcnemar(1, 7) == 0.0703125,
          "exact McNemar matches bench/livecodebench.py values")
    fit = [{"c": 262144, "verdict": "pass", "idle": {"min_free": 1500}, "min_free_mib": 1300,
            "stress": [{"ok": True}]},
           {"c": 196608, "verdict": "pass", "idle": {"min_free": 3400}, "min_free_mib": 3200,
            "stress": [{"ok": True}, {"ok": True}]}]
    g = analyse.gate_fit(fit, 3072)
    check(g["adopt"] == 196608 and not g["per_candidate"][262144]["recheck"],
          "fit re-derives the floor: a row marked pass under the floor is not adopted", g["adopt"])
    check(analyse.gate_fit(fit, 1024)["adopt"] == 262144, "a 1 GB floor adopts 262,144")

    def acc_rows(q4_wrong_at: set, q4_L_wrong: set = frozenset()):
        out = []
        for arm, Ls in (("q8", [8192, 65536]), ("q4", [8192, 65536, 196608])):
            for L in Ls:
                for i in range(12):
                    for t in hs.TASKS:
                        ok = not ((arm == "q4" and (L, i) in q4_wrong_at)
                                  or (arm == "q4" and L in q4_L_wrong))
                        out.append({"arm": arm, "L": L, "item": i, "task": t,
                                    "outcome": "correct" if ok else "wrong", "correct": ok})
        return out
    good = analyse.gate_accuracy(acc_rows(set()), 0.05, 5.0)
    check(good["g3a"]["status"] == "pass" and good["g3b"]["status"] == "pass"
          and good["g3b"]["usable_context"] == 196608, "identical arms pass G3a and G3b", good["g3a"])
    bad = analyse.gate_accuracy(acc_rows({(65536, i) for i in range(4)}), 0.05, 5.0)
    check(bad["g3a"]["status"] == "fail" and bad["g3a"]["q8_only"] == 12,
          "q4 losing 12 cells at one shared rung fails G3a", bad["g3a"])
    beyond = analyse.gate_accuracy(acc_rows(set(), {196608}), 0.05, 5.0)
    check(beyond["g3b"]["status"] == "fail" and beyond["g3b"]["usable_context"] == 65536,
          "q4 collapsing past q8's reach fails G3b and caps usable context", beyond["g3b"])
    none = analyse.gate_accuracy([], 0.05, 5.0)
    check(none["g3a"]["status"] == "not run", "no rows is 'not run', never a pass")
    sp = [{"arm": a, "L": L, "outcome": "measured", "cold": True,
           "decode_tps": d, "prefill_tps": 300.0}
          for a, d in (("q8", 50.0), ("q4", 30.0)) for L in (2048, 98304)]
    check(analyse.gate_speed(sp, 0.85)["status"] == "fail", "a 40% decode drop fails G2")
    sp2 = [dict(r, decode_tps=50.0) for r in sp]
    check(analyse.gate_speed(sp2, 0.85)["status"] == "pass", "equal speed passes G2")
    q8 = {str(i): 1.0 for i in range(21)}
    q4 = dict(q8, **{str(i): 0.0 for i in range(7)})
    r = analyse._pair(q8, q4, 0.05, 2)
    check(r["status"] == "fail" and r["q8_only"] == 7, "LiveBench 7-0 against q4 fails G4", r)
    q4b = dict(q8, **{"0": 0.0})
    check(analyse._pair(q8, q4b, 0.05, 2)["status"] == "pass", "one lost question passes G4")
    with tempfile.TemporaryDirectory() as d:
        json.dump({"gates": {"floor_mib": 3072, "speed_min_ratio": 0.85,
                             "acc_max_drop_points": 5.0, "acc_alpha": 0.05, "quality_alpha": 0.05,
                             "quality_max_drop_questions": 2}},
                  open(os.path.join(d, "manifest.json"), "w"))
        v = analyse.analyse(d)
        check(v["verdict"].startswith("INCOMPLETE"), "an empty run is INCOMPLETE, not ADOPT",
              v["verdict"])
        check("G1 fit" in analyse.render(v), "the verdict renders")


if __name__ == "__main__":
    for t in (test_vram, test_synth, test_config, test_run, test_analyse):
        print(f"-- {t.__name__}")
        t()
    print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
