"""PHASE B of the three-slot KV layout (operator, 2026-09-28; docs/ENGINES.md
"KV rank"): every GPU measurement the deploy needs, on the 0041 build, in the
window discipline of bench/engine_corruption.py (quiet stack, the worker's gpu
lane paused, production `bonsai` unloaded, a guard that kills the arm the
moment llama-swap starts anything on the card, production restored in
`finally`). Nothing here runs without --window: the operator's "GPU free".

THE LAYOUT UNDER TEST (operator): always run from VRAM; the main cap near the
measured VRAM line; the child (second brain, decider, side calls, as-sent
compactions) 64k+ and swapped into VRAM while its conversation's main pauses;
3 slots (two conversations + the child); every conversation advertised the
full cap; "the first conversation never spills; if that is not possible,
either spilling under pressure is fine".

The target argv is config.yaml's `bonsai` argv (ProCreations head, --mmproj,
q8_0 K/V) with the candidate binary and:

    -np 3  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0
    --spec-draft-window 16384 (0036)  --spec-draft-n-max 2 --spec-draft-n-max-tail 4
    --kv-vram-cells N  -c 2N + CHILD   (two conversations at the cap + the child)

STEPS (each writes <out>/<step>/...; --skip names any of them):

  line        N, the all-VRAM line: engine_phase2's fit (their demotion margin,
              1,000 MiB with no display on the card, else 1,300; the card's
              LOWEST free VRAM while a warm runs -- an 8k prefill and decode on
              slot 0, 2k on slot 1, and an attached image, so the projector's
              compute buffer is allocated), up to 3 rounds. A cell moved into
              VRAM costs its K/V (34,816 B) and, since -c = 2N + CHILD grows
              with N, the host tail's staging row (2,176 B): 36,992 B. With
              the draft window the MTP context is not tiered (its K/V is 3 x
              ~18.7k cells whatever N is). --line N skips the fit.
  cap         the main conversation ALONE at the cap (slot 0, kv_rank 2):
              prefill + decode at 8k, 64k and N - 1k; its span (the engine's
              find_slot debug line: "seq 0: cells, span") must stay <= N, and
              no `kv promote` line may appear (nothing to move).
  concurrent  a second conversation (slot 1, kv_rank 1) beside the primary
              (slot 0, kv_rank 2):
                spill   A at N - 8k, then B 32k (it lands above A, past N):
                        A decodes with B idle-holding its cells; then A and B
                        decode AT ONCE (their tokens share batches) -- A's and
                        B's tok/s, and A's span (must stay <= N);
                below   the reference: A 32k and B 32k, both below N, at once;
                first   B arrives FIRST (32k, the bottom of the pool), then A
                        grows to N - 8k: the engine must move B out (rank 1 <
                        2), A's span <= N -- the move's cells and ms.
              What placement cannot fix, recorded: one batch carries both, so
              a spilled B's host staging lengthens every step A waits for.
  child       the swap (for each --child-size: 49,152 / 65,536 / 98,304): A at
              N - 4k on slot 0 (kv_rank 2, then idle), the child on slot 2 at
              that size (kv_rank 2: the primary's child), decode; then A
              continues (its prompt + what it generated + 256 new tokens) --
              the child's prefill seconds and decode, the moves (cells, ms),
              A's swap-back prompt ms and its decode after, and both spans
              (<= N). The baseline: A continued with no child in between.
  corruption  bench/engine_corruption.py --vision on the target arm (greedy x3
              reps identical, 16 plain without a symptom, 10 tools; the image
              described).

    python bench/kv_rank.py --window --out bench/results/kv_rank/<date> [--line N] [--binary EXE] [--layout v1]
    python bench/kv_rank.py --summarise DIR

Numbers, the placement checks (spans <= N, a fact of the engine's own log), and
no verdict on speed: the operator judges the output. The deploy is
bench/deploy_kv_rank.py, given the line this measured.

LAYOUT V2 (operator, 2026-09-29; `--layout v2`, THE DEFAULT since then; `--layout v1` is the
2026-09-28 layout above, unchanged):

  (a) "The point is to get more context at speed in vram, so decider was the only thing
      that needed room."  (b) "Vision can go to second card and swap in and out."

  So `bonsai` loses --mmproj (the projector cost ~24.8k cells of the line, the 2026-09-27
  fit, docs/ENGINES.md), and the child slot becomes THE LANE: the decider and small side
  calls only, budget.LANE_TOKENS cells, KEPT between turns (never released), at kv_rank
  slots.RANK_LANE (3, above the primary conversation's 2) so the engine never moves it out
  of VRAM. Its cells come out of the line: the main cap is N - LANE_TOKENS
  (budget.main_cap). The arm: config.yaml's `bonsai` argv WITHOUT --mmproj, and

    -np 3 --kv-unified  -c 2N + LANE_TOKENS  --kv-vram-cells N  (the rest as above)

  line        the same fit, warmed WITHOUT an image (there is no projector) and with the
              lane holding LANE_TOKENS cells at its rank (slot 2), so its cells are in
              VRAM while the free memory is read. Prints the line to use:
              `python bench/deploy_layout_v2.py --line N`.
  cap         the main conversation alone at the main CAP (N - LANE_TOKENS).
  lane        the lane beside a primary at the cap, in both orders (the lane filled
              after the primary, and before it): both spans <= N, nothing moves; the
              primary's decode with the lane holding its cells; a decider-shaped lane
              request after a kept lane (its head cached) vs after a release (the cost
              the release used to add: logs/proxy.log, the last 200 decider turns,
              release median 395 ms, p90 824 ms).
  concurrent  as above, with the primary at the cap.
  child       v1 only by default (the second brain is off in v2); --child-size runs it.
  corruption  bench/engine_corruption.py on the v2 arm, without --vision (DROP:--mmproj).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import engine_corruption as ec                                       # noqa: E402
import engine_phase2 as ep                                           # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import budget                                                        # noqa: E402
import slots                                                         # noqa: E402

PY = sys.executable
LANE_BY = "kv_rank"
# The 0041 build (scripts/build_engine.py llama-bonsai2-ada, 2026-09-28): 0001-0041.
CANDIDATE = "C:/Users/jwals/engines/llama-bonsai2-ada-80d2c60d/src/build/bin/llama-server.exe"
CHILD = 65536                   # budget.CHILD_TOKENS: the operator's "64k+"
CHILD_SIZES = (49152, 65536, 98304)
PER_CELL = ep.CELL_BYTES + ep.DRAFT_CELL_BYTES   # K/V + the staging row (-c = 2N + CHILD)
GEN = 256
REPS = 3
RANK_PRIMARY, RANK_SECOND = 2, 1
# LAYOUT V2: the lane's size and rank are the proxy's own constants (one source).
LANE = budget.LANE_TOKENS
RANK_LANE = slots.RANK_LANE
LAYOUTS = ("v1", "v2")
RE_SPAN = re.compile(r"find_slot: stream\[\d+\], n = +(\d+), used = +(\d+), head = +\d+, size = +\d+, "
                     r"n_swa = +\d+, seq (\d+): cells = (\d+), span = (\d+)")
RE_PROMOTE = re.compile(r"kv promote: seqs (\S+): (\d+) cells, span (\d+) -> (\d+).*?: (\d+) moved, "
                        r"(\d+) exchanged with idle cells, (\d+) idle cells moved out; (\d+) rows "
                        r"\(([\d.]+) MiB\) in (\d+) graph\(s\), ([\d.]+) ms")


def log(msg: str) -> None:
    ec.log(msg)


def target_args(n: int, child: int = CHILD, debug: bool = False, layout: str = "v2") -> dict:
    """The arm's argv overrides. v1: -c 2N + CHILD, --mmproj kept (config.yaml's argv). v2: -c 2N + LANE
    and --mmproj REMOVED (a None value: engine_corruption.arm_command drops the flag and its path)."""
    # --kv-unified: an explicit -np turns llama-server's unified pool OFF (3 streams of -c/3 each), and the
    # tiered cache (0028) and the moves (0040/0041) need ONE stream -- without it the whole pool was device
    # memory, oversubscribed into system memory (2026-09-29 first fit: 21-29 MiB free at N 147k and 119k,
    # decode 3.8 tok/s, prefill ~95 tok/s)
    extra = LANE if layout == "v2" else child
    a = {"-np": "3", "--kv-unified": "", "-c": str(2 * n + extra), "--kv-vram-cells": str(n),
         "--cache-type-k-draft": "q8_0", "--cache-type-v-draft": "q8_0",
         "--spec-draft-window": "16384", "--spec-draft-n-max": "2", "--spec-draft-n-max-tail": "4"}
    if layout == "v2":
        a["--mmproj"] = None
    if debug:
        a["-lv"] = "5"
    return a


def main_cap(n: int, layout: str) -> int:
    """The main conversation's cap on this arm: the line (v1), the line less the lane (v2: the lane's
    cells are kept in VRAM at a rank above the primary's)."""
    return n - LANE if layout == "v2" else n


class Log:
    """The arm's log since the last mark: per-sequence spans and the moves."""

    def __init__(self, path: str):
        self.path, self.pos = path, 0

    def mark(self) -> None:
        self.pos = os.path.getsize(self.path) if os.path.exists(self.path) else 0

    def read(self) -> dict:
        with open(self.path, encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            text = f.read()
        self.pos += len(text.encode("utf-8", "replace"))
        spans: dict[int, int] = {}
        pool_max = 0
        for m in RE_SPAN.finditer(text):
            n, _used, seq, _cells, span = (int(x) for x in m.groups())
            spans[seq] = max(spans.get(seq, 0), span)
            pool_max = max(pool_max, n)
        moves = [{"seqs": m.group(1), "cells": int(m.group(2)), "span_before": int(m.group(3)),
                  "span_after": int(m.group(4)), "moved": int(m.group(5)), "exchanged": int(m.group(6)),
                  "evicted": int(m.group(7)), "rows": int(m.group(8)), "mib": float(m.group(9)),
                  "graphs": int(m.group(10)), "ms": float(m.group(11))} for m in RE_PROMOTE.finditer(text)]
        return {"max_span_by_seq": spans, "pool_max_p1": pool_max or None, "moves": moves,
                "move_ms": round(sum(x["ms"] for x in moves), 1), "move_mib": round(sum(x["mib"] for x in moves), 1)}


def gen(base: str, slot: int, tokens: list[int], n: int, rank: int, ranks: list[int]) -> dict:
    """One greedy /completion on `slot` with the ranks the proxy would send (slots RANKS)."""
    t0 = time.time()
    d = ep.post(base, "/completion", {"prompt": tokens, "n_predict": n, "id_slot": slot, "cache_prompt": True,
                                      "temperature": 0, "top_k": 1, "seed": 0, "ignore_eos": True,
                                      "return_tokens": True, "kv_rank": rank, "kv_ranks": ranks})
    t = d.get("timings") or {}
    out = {k: t.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n",
                                 "predicted_per_second", "draft_n", "draft_n_accepted")}
    out["wall_s"] = round(time.time() - t0, 2)
    out["tokens"] = d.get("tokens") or []
    return out


def clear(base: str) -> None:
    """Empty every slot (slots.release_idle's shrink) with rank 0."""
    for i in range(3):
        ep.post(base, "/completion", {"prompt": f"x{i}", "n_predict": 1, "id_slot": i, "cache_prompt": True,
                                      "kv_rank": 0, "kv_ranks": [0, 0, 0]})


def med(rows: list[dict]) -> float | None:
    v = [r["predicted_per_second"] for r in rows if r.get("predicted_per_second")]
    return round(statistics.median(v), 2) if v else None


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "tokens"}


# ------------------------------------------------------------------ steps --
def line_warm_v2(base: str, rec: dict, guard) -> None:
    """A v2 fit round: a real prefill and decode on the two conversation slots, and the lane (slot 2)
    holding LANE cells at its rank -- no image: v2's `bonsai` has no projector."""
    toks = ep.corpus_tokens(base)
    rec["warm"] = [strip(gen(base, 0, toks[:8192], GEN, 2, [2, 0, RANK_LANE])),
                   strip(gen(base, 1, toks[8192:10240], GEN, 1, [2, 1, RANK_LANE])),
                   strip(gen(base, 2, toks[10240:10240 + LANE - 1], 1, RANK_LANE, [2, 1, RANK_LANE]))]


def line_warm(base: str, rec: dict, guard) -> None:
    """A fit round: a real prefill and decode on two slots, and an attached image (the projector's buffer)."""
    toks = ep.corpus_tokens(base)
    rec["warm"] = [strip(gen(base, 0, toks[:8192], GEN, 2, [2, 0, 0])),
                   strip(gen(base, 1, toks[8192:10240], GEN, 1, [2, 1, 0]))]
    import base64
    png = base64.b64encode(ec.two_colour_png()).decode()
    st, txt = ec.http("POST", base + "/v1/chat/completions", {
        "model": "x", "max_tokens": 64, "id_slot": 2, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Which colour is the left half, which the right?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}]}, timeout=600)
    rec["warm_image"] = {"status": st, "content": (json.loads(txt)["choices"][0]["message"].get("content")
                                                   if st == 200 else txt[:200])}


def find_line(s: dict, out: str, a) -> dict:
    margin, display = ep.margin_mib()
    n = a.start
    rounds = []
    v2 = a.layout == "v2"
    extra = LANE if v2 else a.child
    for r in range(3):
        name = f"line-{r}"
        s["arms"][name] = (a.binary, {}, target_args(n, a.child, layout=a.layout))
        rec, res = ep.in_window(name, s, line_warm_v2 if v2 else line_warm, out, a.attempts, a.wait_quiet)
        free = rec.get("gpu_min_free")
        rounds.append({"n": n, "c": 2 * n + extra, "min_free_mib": free, "error": rec.get("error"),
                       "image": rec.get("warm_image"), "lane_warm": (rec.get("warm") or [])[2:]})
        log(f"[line] round {r}: N={n} -c {2 * n + extra}: min free {free} MiB (margin {margin}, "
            f"display_active {display})")
        if rec.get("error") or free is None:
            break
        step = int((free - margin) * 1048576 / PER_CELL) // 256 * 256
        if step == 0:
            break
        n = max(16384, n + step)
        if res and res.get("proxy_health") != 200:
            break
    ok = [x for x in rounds if not x["error"] and x["min_free_mib"] is not None and x["min_free_mib"] >= margin]
    best = max(ok, key=lambda x: x["n"]) if ok else None
    res = {"layout": a.layout, "margin_mib": margin, "display_active": display, "per_cell_bytes": PER_CELL,
           "rounds": rounds, "n": best["n"] if best else None, "child": a.child}
    if v2:
        res.update(lane=LANE, rank_lane=RANK_LANE, mmproj="removed")
    if res["n"]:
        res["c"] = 2 * res["n"] + extra
        res["main_cap"] = main_cap(res["n"], a.layout)
        res["host_ram_gb"] = round((res["c"] - res["n"]) * ep.CELL_BYTES / 1e9, 2)
        res["free_ram_gb_at_fit"] = ep.host_free_gb()
    json.dump(res, open(os.path.join(out, "line.json"), "w", encoding="utf-8"), indent=1)
    return res


def measure_fn(n: int, steps: set, child_sizes: tuple, layout: str = "v1"):
    cap = main_cap(n, layout)
    lane_rank = RANK_LANE if layout == "v2" else 0

    def fn(base: str, rec: dict, guard) -> None:
        lg = Log(rec["_log"])
        toks = ep.corpus_tokens(base)
        rec["corpus_tokens"] = len(toks)
        need = max(n + max(child_sizes) + 70000, 400000 + 512 + GEN)   # the highest offset used is 400,000
        if len(toks) < need:
            reps = need // len(toks) + 1
            toks = (toks * reps)[:need]          # repeated code: acceptance numbers are then optimistic
            rec["corpus_repeated"] = reps

        def seg(k: int, off: int) -> list[int]:
            return toks[off:off + k]

        def stage(label: str, f):
            lg.mark()
            r = f()
            time.sleep(0.5)
            r["log"] = lg.read()
            rec.setdefault("stages", {})[label] = r
            spans = r["log"]["max_span_by_seq"]
            log(f"  {label}: {json.dumps({k: v for k, v in r.items() if k not in ('log',)})[:300]} "
                f"spans {spans} moves {len(r['log']['moves'])} ({r['log']['move_ms']} ms)")
            if guard.tripped:
                raise RuntimeError(guard.tripped)
            return r

        checks = rec.setdefault("checks", [])

        def check(ok: bool, what: str, detail) -> None:
            checks.append({"ok": bool(ok), "what": what, "detail": detail})
            log(f"  {'pass' if ok else 'FAIL'}  {what}  {detail}")

        if "cap" in steps:
            for d in (8192, 65536, cap - 1024 - GEN):
                def one(d=d):
                    clear(base)
                    cold = gen(base, 0, seg(d, 0), GEN, 2, [2, 0, lane_rank])
                    reps = [gen(base, 0, seg(d, 0), GEN, 2, [2, 0, lane_rank]) for _ in range(REPS)]
                    return {"depth": d, "cold": strip(cold), "reps": [strip(x) for x in reps],
                            "decode_median": med(reps)}
                r = stage(f"cap_{d}", one)
                sp = r["log"]["max_span_by_seq"].get(0)
                check(sp is not None and sp <= n and not r["log"]["moves"],
                      f"alone at {d}: the conversation's span stays at or below the line, nothing moves",
                      {"span": sp, "line": n, "moves": len(r["log"]["moves"])})

        if "lane" in steps and layout == "v2":
            # THE LANE (v2): LANE cells on slot 2 at RANK_LANE beside the primary at the cap. The lane's
            # prompt is decider-shaped: a fixed head (standing in for the decider's system block: 64
            # tokens of a fixed segment), then a state that differs per turn.
            a_depth = cap - GEN - 256
            head = seg(64, 350000)
            ranks = [2, 0, RANK_LANE]

            def lane_prompt(k: int) -> list[int]:
                return head + seg(LANE - 64 - 1, 360000 + k * 4096)

            def order(first: str):
                def run():
                    clear(base)
                    if first == "lane":
                        lane0 = gen(base, 2, lane_prompt(0), 1, RANK_LANE, ranks)
                        gen(base, 0, seg(a_depth, 0), 1, 2, ranks)
                    else:
                        gen(base, 0, seg(a_depth, 0), 1, 2, ranks)
                        lane0 = gen(base, 2, lane_prompt(0), 1, RANK_LANE, ranks)
                    reps = [gen(base, 0, seg(a_depth, 0), GEN, 2, ranks) for _ in range(REPS)]
                    kept = gen(base, 2, lane_prompt(1), 1, RANK_LANE, ranks)
                    # the release the lane no longer gets (slots.release_idle's shrink), then the same kind
                    # of request cold
                    ep.post(base, "/completion", {"prompt": "x2", "n_predict": 1, "id_slot": 2,
                                                  "cache_prompt": True, "kv_rank": RANK_LANE,
                                                  "kv_ranks": ranks})
                    released = gen(base, 2, lane_prompt(2), 1, RANK_LANE, ranks)
                    return {"first": first, "A_depth": a_depth, "lane_cells": LANE, "lane_0": strip(lane0),
                            "A_decode_median": med(reps), "A_reps": [strip(x) for x in reps],
                            "lane_after_kept": strip(kept), "lane_after_release": strip(released)}
                return run
            for first in ("primary", "lane"):
                r = stage(f"lane_{first}_first", order(first))
                sp = r["log"]["max_span_by_seq"]
                check(sp.get(0) is not None and sp[0] <= n and sp.get(2) is not None and sp[2] <= n
                      and not r["log"]["moves"],
                      f"the lane ({LANE} cells, rank {RANK_LANE}) beside the primary at the cap ({cap}), "
                      f"{first} first: both spans at or below the line, nothing moves",
                      {"spans": sp, "line": n, "cap": cap, "moves": len(r["log"]["moves"]),
                       "lane_prompt_ms_kept": r["lane_after_kept"].get("prompt_ms"),
                       "lane_prompt_ms_after_release": r["lane_after_release"].get("prompt_ms")})

        if "concurrent" in steps:
            a_depth, b_depth = cap - 8192, 32768

            def spill():
                clear(base)
                gen(base, 0, seg(a_depth, 0), 1, 2, [2, 0, 0])
                gen(base, 1, seg(b_depth, 200000), 1, 1, [2, 1, 0])
                alone = [gen(base, 0, seg(a_depth, 0), GEN, 2, [2, 1, 0]) for _ in range(REPS)]
                both = []
                for _ in range(REPS):
                    outs: dict = {}
                    th = [threading.Thread(target=lambda s_, p_, r_: outs.__setitem__(s_, gen(base, s_, p_, GEN, r_,
                                                                                              [2, 1, 0])),
                                           args=(s_, p_, r_))
                          for s_, p_, r_ in ((0, seg(a_depth, 0), 2), (1, seg(b_depth, 200000), 1))]
                    for t in th:
                        t.start()
                    for t in th:
                        t.join()
                    both.append({"A": strip(outs[0]), "B": strip(outs[1])})
                return {"A_depth": a_depth, "B_depth": b_depth,
                        "A_alone_B_idle_median": med(alone),
                        "A_concurrent_median": med([x["A"] for x in both]),
                        "B_concurrent_median": med([x["B"] for x in both]), "concurrent": both}
            r = stage("concurrent_spill", spill)
            sp = r["log"]["max_span_by_seq"]
            check(sp.get(0) is not None and sp[0] <= n,
                  "a second conversation beside the primary: the primary's span stays at or below the line",
                  {"spans": sp, "line": n})

            def below():
                clear(base)
                gen(base, 0, seg(32768, 0), 1, 2, [2, 0, 0])
                gen(base, 1, seg(32768, 200000), 1, 1, [2, 1, 0])
                both = []
                for _ in range(REPS):
                    outs: dict = {}
                    th = [threading.Thread(target=lambda s_, p_, r_: outs.__setitem__(s_, gen(base, s_, p_, GEN, r_,
                                                                                              [2, 1, 0])),
                                           args=(s_, p_, r_))
                          for s_, p_, r_ in ((0, seg(32768, 0), 2), (1, seg(32768, 200000), 1))]
                    for t in th:
                        t.start()
                    for t in th:
                        t.join()
                    both.append({"A": strip(outs[0]), "B": strip(outs[1])})
                return {"A_concurrent_median": med([x["A"] for x in both]),
                        "B_concurrent_median": med([x["B"] for x in both]), "concurrent": both}
            stage("concurrent_below", below)

            def first():
                clear(base)
                gen(base, 1, seg(b_depth, 200000), 1, 1, [0, 1, 0])
                t0 = time.time()
                grow = gen(base, 0, seg(a_depth, 0), GEN, 2, [2, 1, 0])
                reps = [gen(base, 0, seg(a_depth, 0), GEN, 2, [2, 1, 0]) for _ in range(REPS)]
                return {"A_prefill": strip(grow), "A_prefill_wall_s": round(time.time() - t0, 2),
                        "A_decode_median": med(reps)}
            r = stage("concurrent_second_arrived_first", first)
            sp = r["log"]["max_span_by_seq"]
            check(sp.get(0) is not None and sp[0] <= n,
                  "the second conversation arrived first: the primary still ends at or below the line "
                  "(the engine moved the lower rank out)",
                  {"spans": sp, "line": n, "moves": len(r["log"]["moves"]), "move_ms": r["log"]["move_ms"]})

        if "child" in steps:
            a_depth = cap - 4096

            def baseline():
                clear(base)
                a0 = gen(base, 0, seg(a_depth, 0), 64, 2, [2, 0, 0])
                cont = seg(a_depth, 0) + a0["tokens"] + seg(256, 400000)
                back = gen(base, 0, cont, GEN, 2, [2, 0, 0])
                return {"A_continue": strip(back)}
            stage("child_baseline_no_child", baseline)
            for size in child_sizes:
                def swap(size=size):
                    clear(base)
                    a0 = gen(base, 0, seg(a_depth, 0), 64, 2, [2, 0, 0])
                    t0 = time.time()
                    c = gen(base, 2, seg(size, 300000), GEN, 2, [2, 0, 2])
                    c_wall = round(time.time() - t0, 2)
                    cont = seg(a_depth, 0) + a0["tokens"] + seg(256, 400000)
                    t1 = time.time()
                    back = gen(base, 0, cont, GEN, 2, [2, 0, 2])
                    return {"size": size, "child": strip(c), "child_wall_s": c_wall, "A_continue": strip(back),
                            "A_continue_wall_s": round(time.time() - t1, 2)}
                r = stage(f"child_{size}", swap)
                sp = r["log"]["max_span_by_seq"]
                check(sp.get(2) is not None and sp[2] <= n and sp.get(0) is not None and sp[0] <= n,
                      f"child {size}: the child ran at or below the line, and so did its main after it",
                      {"spans": sp, "line": n, "moves": len(r["log"]["moves"]), "move_ms": r["log"]["move_ms"]})
    return fn


def summary(out: str) -> str:
    rows = []
    line = ep.load(os.path.join(out, "line.json"))
    if line:
        rows.append(f"line N = {line.get('n')} (layout {line.get('layout', 'v1')}, margin "
                    f"{line.get('margin_mib')} MiB, -c {line.get('c')}, main cap {line.get('main_cap')}, "
                    f"host RAM pinned ~{line.get('host_ram_gb')} GB)")
        if line.get("n") and line.get("layout") == "v2":
            rows.append(f"THE LINE TO USE: python bench/deploy_layout_v2.py --line {line['n']} --preview "
                        "(then without --preview)")
        for r in line.get("rounds") or []:
            rows.append(f"  round N={r['n']} -c {r['c']}: min free {r['min_free_mib']} MiB {r['error'] or ''}")
    m = ep.load(os.path.join(out, "measure", "kvrank.json"))
    if m:
        rows.append("")
        rows.append("| stage | numbers | max span by seq | moves (ms) |")
        rows.append("|---|---|---|---|")
        for k, s in (m.get("stages") or {}).items():
            nums = {x: y for x, y in s.items() if x not in ("log", "concurrent", "reps", "cold")
                    and not isinstance(y, list)}
            rows.append(f"| {k} | {json.dumps(nums)[:220]} | {s['log']['max_span_by_seq']} | "
                        f"{len(s['log']['moves'])} ({s['log']['move_ms']}) |")
        rows.append("")
        for c in m.get("checks") or []:
            rows.append(f"{'pass' if c['ok'] else 'FAIL'}  {c['what']}  {json.dumps(c['detail'])[:200]}")
        if m.get("error"):
            rows.append(f"ERROR {m['error']}")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", action="store_true", help="the operator grants the window (production unloaded)")
    ap.add_argument("--out")
    ap.add_argument("--binary", default=CANDIDATE)
    ap.add_argument("--line", type=int, default=0, help="skip the fit: use this N")
    ap.add_argument("--start", type=int, default=147456, help="the fit's first N")
    ap.add_argument("--child", type=int, default=CHILD)
    ap.add_argument("--child-size", type=int, action="append", default=[])
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--wait-quiet", type=float, default=30.0)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["line", "cap", "lane", "concurrent", "child", "corruption"])
    ap.add_argument("--layout", choices=LAYOUTS, default="v2",
                    help="v2 (default): no --mmproj, -c 2N + the lane; v1: the 2026-09-28 child layout")
    ap.add_argument("--summarise")
    a = ap.parse_args(argv)
    if a.summarise:
        print(summary(a.summarise))
        return 0
    if not a.window or not a.out:
        print("refusing: needs --window (the operator's GPU free; production is unloaded) and --out")
        return 2
    if not os.path.exists(a.binary):
        print(f"refusing: no binary at {a.binary}")
        return 2
    os.makedirs(a.out, exist_ok=True)
    ep.LANE_BY = LANE_BY
    s = ep.setup()
    s["port"], s["arms"] = a.port, {}
    n = a.line
    if not n and "line" not in a.skip:
        os.makedirs(os.path.join(a.out, "line"), exist_ok=True)
        n = (find_line(s, os.path.join(a.out, "line"), a) or {}).get("n") or 0
        if n:
            json.dump(json.load(open(os.path.join(a.out, "line", "line.json"), encoding="utf-8")),
                      open(os.path.join(a.out, "line.json"), "w", encoding="utf-8"), indent=1)
    if not n:
        print("no line: pass --line N or run the fit")
        return 1
    steps = {x for x in ("cap", "lane", "concurrent", "child") if x not in a.skip}
    if a.layout == "v2" and not a.child_size:
        # the second brain is off by default in v2: no 49k-98k child to swap (--child-size runs it)
        steps.discard("child")
    if steps:
        mdir = os.path.join(a.out, "measure")
        os.makedirs(mdir, exist_ok=True)
        args = target_args(n, a.child, debug=True, layout=a.layout)
        s["arms"]["kvrank"] = (a.binary, {"LLAMA_KV_CACHE_DEBUG": "1"}, args)
        fn = measure_fn(n, steps, tuple(a.child_size) or CHILD_SIZES, a.layout)

        def with_log(base, rec, guard):
            rec["_log"] = os.path.join(mdir, "kvrank.log")
            rec["line"] = n
            fn(base, rec, guard)
        ep.in_window("kvrank", s, with_log, mdir, a.attempts, a.wait_quiet)
    if "corruption" not in a.skip:
        targs = target_args(n, a.child, layout=a.layout)
        spec = f"kvrank={a.binary}," + ",".join(f"DROP:{k}" if v is None else f"ARG:{k}={v}"
                                                for k, v in targs.items())
        cmd = ([PY, os.path.join(HERE, "engine_corruption.py"), "--window"]
               + (["--vision"] if a.layout == "v1" else [])
               + ["--out", os.path.join(a.out, "corruption"), "--port", str(a.port), "--attempts",
                  str(a.attempts), "--wait-quiet", str(a.wait_quiet), "--arm", spec])
        log("corruption: " + " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT)
        r = ep.load(os.path.join(a.out, "corruption", "kvrank.json"))
        ok, why = ep.corruption_ok(r)
        json.dump({"ok": ok, "why": why}, open(os.path.join(a.out, "corruption_gate.json"), "w"), indent=1)
        log(f"corruption gate: {'pass' if ok else 'FAIL'} ({why})")
    text = summary(a.out)
    print(text)
    if a.layout == "v2":
        print(f"\nTHE LINE TO USE: {n}  (main cap {main_cap(n, 'v2')}, lane {LANE}, -c {2 * n + LANE})\n"
              f"  python bench/deploy_layout_v2.py --line {n} --preview")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
