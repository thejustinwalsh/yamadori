#!/usr/bin/env python
"""HOW REPEATABLE IS A JJAVA READ? The same questions over the same states,
repeated, in four regimes (GPU; a coordinator-granted window only).

    python bench/decider/determinism.py --dry-run
    python bench/decider/determinism.py [--model bonsai] [--states 10] \\
        [--repeats 5] [--regimes cold,cached,cold_beside,cached_beside] \\
        [--cases inj_all/cases.jsonl] [--decode-slot 0]

WHY (docs/JJAVA.md 8): SGLang's decision models (docs.sglang.io/docs/
supported-models/decision_models, read 2026-09-30) measured, on
Qwen3.8-27B -- the base Bonsai 2's card names -- and Qwen3.5-35B-A3B, that
cold vs prefix-cached requests and the batch's composition move answer
probabilities "by up to several hundredths (0.07 in our checks)" and the
label mass "by up to about 0.14", with the chosen options stable, and that
turning the prefix cache off "gave identical values across sequential
repeats". Our TIE_BAND (0.0034) is one number of that kind, measured once
(20 states x 5 questions, cached vs cache_prompt false, one run). This
measures the whole picture on OUR engine, per model, so the operator can
choose between a cold read that repeats exactly and a cached read that is
faster -- judged against the minutes of generative thinking a question
replaces, so every regime's seconds are reported beside its spread.

THE REGIMES (2 x 2):
  cold            cache_prompt false: every read re-processes the whole
                  prompt; nothing else generating.
  cached          the production path: the state placed once on the lane
                  (a placement read, not counted), each question read after
                  it with the prefix cached; nothing else generating.
  cold_beside     cold, while a main-model decode runs on another slot
                  (--decode-slot) the whole time: the batch holds the
                  decode's tokens beside ours (batch composition).
  cached_beside   cached, beside the same decode.
Each regime reads every question on every state --repeats times, both
orders each (decider_bonsai.read, the typed readout as served).

WHAT IS REPORTED per regime (results/determinism_<model>.json):
  prob     over every (state, question, order, key): |p_i - p_1| for each
           repeat i after the first -- max and mean -- and the share of
           groups whose repeats are identical at the 6 decimals read()
           rounds to; the same over the AVERAGED answer; argmax flips
  label_mass  the same spread over each order's label mass
  seconds  per question (two orders): median, p90, max; processed tokens
  diagnostics  mean label mass, the case-variant masses (decider_bonsai.
           case_variants), how much of the regime the decode overlapped
And ACROSS regimes: the repeat-mean answer of each regime against cold's
(max / mean |diff|, argmax agreement) -- the TIE_BAND quantity, measured
four ways. First of all, template_check(): the serving model's own
/apply-template of jjava's question read back closed (the think-block
guard confirmed live).

THE CHOICE, AND WHAT IS RECORDED (operator, 2026-09-30, through the
coordinator: "20s is not too slow"; "we would normally be paying 10s of
minutes or more without jjava so it's a huge speed up"): when all four
regimes ran, choose() picks the CHEAPEST read mode (cold or cached) that
repeats exactly in BOTH its environments (alone and beside a decode --
jjava cannot choose which it meets), fully cold included, and record()
writes into the model's profile (bench/decider/results/models/<model>.json,
the record decider_bonsai reads): `read_regime` {mode, repeats_exactly,
seconds per question, every mode's numbers} -- decider_bonsai.read_cache()
then reads every question that way -- and `tie_band`, derived IN THAT MODE:
the largest difference one question's answer shows there (within either
environment's repeats, and between the environments' means). No mode exact:
the regime stays cached (nothing written for it) and the band is derived
and written for cached, the path jjava runs. --no-write records nothing.

THE STATES: decide_turn.state_of over the daily eval's points (as
bonsai_decider `batching` picks them: every len/N-th), or --cases (the
injector's labelled cases, bench/skills/inject_labels.py build): the same
STATE_TOKENS cut the lane serves, counted by the model's own /tokenize.
THE QUESTIONS: build intent (a noul), phase (a choice of 4) and the
injector's item score (4 levels; the case's first item, else a fixed fact)
-- one of each type. Only ids and numbers are written, never state text.

ONE GPU CONSUMER: GET /running (never loads a model) must list --model;
decider_target's preflight before every state (no other slot processing,
no hermes.exe). The decode slot's cache IS overwritten: run it on an idle
stack only. Exit 2 busy, 3 not run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)

import decider_bonsai as D  # noqa: E402
import decider_target as DT  # noqa: E402

VERSION = "determinism/1"
REGIMES = ("cold", "cached", "cold_beside", "cached_beside")
# Bench parameters (the measurement's design, not runtime constants): the
# decode only has to keep a main-model generation in the batch while the
# regime reads; rounds repeat until the regime ends, and the report says
# how much of the regime they covered (`decode.overlap`).
DECODE_TOKENS = 256
DECODE_SOURCE_CHARS = 12000
FIXED_FACT = ("DO: pass `ref` to a function component as a regular prop; "
              "forwardRef is not needed in React 19.")


def questions(fact: str) -> list[dict]:
    import decide_turn as T
    import skill_inject as I
    return [D.q_noul("build_intent", T.BUILD_INTENT_Q),
            D.q_choice("phase", T.PHASE_Q[0], T.PHASE_Q[1],
                       keys=list(T.PHASES)),
            D.q_score("skill_item:determinism", I.ITEM_Q.format(fact=fact),
                      I.ITEM_LEVELS)]


def states(n: int, cases: str | None, count) -> list[dict]:
    """[{id, state, tokens, fact}] -- ids only leave this process."""
    import decide_turn as T
    out = []
    if cases:
        with open(cases, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        step = max(len(rows) // n, 1)
        for c in rows[::step][:n]:
            fact = (c.get("items") or [{}])[0].get("fact") or FIXED_FACT
            out.append({"id": c["case"], "state": c["state"],
                        "tokens": (c.get("state_info") or {}).get("tokens"),
                        "fact": fact})
        return out
    import decider_labels as Y
    pts = Y._daily_points()
    step = max(len(pts) // n, 1)
    for p in pts[::step][:n]:
        text, info = T.state_of(p["msgs"], count)
        out.append({"id": p["id"], "state": text,
                    "tokens": info.get("tokens"), "fact": FIXED_FACT})
    return out


# --------------------------------------------------------------- decode ----
class Decode:
    """A main-model generation on `slot`, restarted until stop(): what the
    `_beside` regimes read next to."""

    def __init__(self, target: DT.Target, slot: int):
        self.t, self.slot = target, slot
        self.stop_ev = threading.Event()
        self.rounds, self.busy_s, self.errors = 0, 0.0, []
        src = open(os.path.join(ROOT, "mcp", "decider_bonsai.py"),
                   encoding="utf-8").read()[:DECODE_SOURCE_CHARS]
        self.body = {
            "messages": [{"role": "user", "content": "Describe this Python "
                          "source section by section.\n\n" + src}],
            "max_tokens": DECODE_TOKENS, "ignore_eos": True, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": True, "id_slot": int(slot), "stream": False}
        if target.mode == "one door":
            import model
            self.body["model"] = model.MODEL

    def _run(self):
        while not self.stop_ev.is_set():
            t0 = time.time()
            try:
                if self.t.mode == "one door":
                    import model
                    model.post(dict(self.body))
                else:
                    self.t.post(dict(self.body), 300)
                self.rounds += 1
            except Exception as e:                               # noqa: BLE001
                self.errors.append(f"{type(e).__name__}: {e}"[:160])
                if len(self.errors) > 3:
                    return
            self.busy_s += time.time() - t0

    def __enter__(self):
        self.t0 = time.time()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        time.sleep(2.0)            # the first round's prefill, then decode
        return self

    def __exit__(self, *exc):
        self.stop_ev.set()
        self.th.join(timeout=600)
        self.wall_s = time.time() - self.t0


# ----------------------------------------------------------------- read ----
def _one(state: str, q: dict, slot, cache: bool) -> dict:
    a = D.read(state, q, slot=slot, cache=cache)
    d = a["diagnostics"]
    return {"p": D.answer_probs(a), "argmax": d["argmax"],
            "orders": [{"raw": o["raw"], "label_mass": o["label_mass"],
                        "variant_mass": o.get("variant_mass"),
                        "word_mass": o.get("word_mass"),
                        "processed": o.get("processed_tokens")}
                       for o in d["orders"]],
            "ms": d["ms"], "processed": d["processed_tokens"]}


def run_regime(regime: str, sts: list[dict], repeats: int, slot, t,
               decode_slot, mine) -> dict:
    cache = not regime.startswith("cold")
    beside = regime.endswith("_beside")
    reads: dict = {}
    DT.preflight(t, mine)          # before our own decode starts
    dec = Decode(t, decode_slot) if beside else None
    if dec:
        dec.__enter__()
    try:
        for s in sts:
            if not beside:
                DT.preflight(t, mine)
            qs = questions(s["fact"])
            if cache:              # place the state (not counted)
                _one(s["state"], qs[0], slot, True)
            for r in range(repeats):
                for q in qs:
                    reads.setdefault((s["id"], q["name"]), []).append(
                        _one(s["state"], q, slot, cache))
    finally:
        if dec:
            dec.__exit__(None, None, None)
    out = summarise(reads)
    if dec:
        out["decode"] = {"slot": decode_slot, "rounds": dec.rounds,
                         "busy_s": round(dec.busy_s, 1),
                         "wall_s": round(dec.wall_s, 1),
                         "overlap": round(dec.busy_s / dec.wall_s, 3)
                         if dec.wall_s else None,
                         "errors": dec.errors}
    out["cache"], out["beside"] = cache, beside
    return out, reads


def _spread(groups: list[list[float]]) -> dict:
    diffs = [abs(v - g[0]) for g in groups for v in g[1:]]
    same = sum(1 for g in groups if all(round(v, 6) == round(g[0], 6)
                                        for v in g))
    return {"max": round(max(diffs), 6) if diffs else None,
            "mean": round(statistics.mean(diffs), 6) if diffs else None,
            "identical_groups": same, "groups": len(groups)}


def summarise(reads: dict) -> dict:
    prob, avg, mass, ms, proc, flips = [], [], [], [], [], 0
    lm, vm, wm = [], [], []
    for rs in reads.values():
        keys = list(rs[0]["p"])
        for k in keys:
            avg.append([r["p"][k] for r in rs])
        for oi in range(len(rs[0]["orders"])):
            for k in rs[0]["orders"][oi]["raw"]:
                prob.append([r["orders"][oi]["raw"][k] for r in rs])
            mass.append([r["orders"][oi]["label_mass"] or 0.0 for r in rs])
        flips += len({r["argmax"] for r in rs}) > 1
        for r in rs:
            ms.append(r["ms"])
            proc.append(r["processed"] or 0)
            for o in r["orders"]:
                lm.append(o["label_mass"] or 0.0)
                if o.get("variant_mass") is not None:
                    vm.append(o["variant_mass"])
                if o.get("word_mass"):
                    wm.append(sum(o["word_mass"].values()))
    ms.sort()
    q = lambda xs, f: xs[min(int(f * len(xs)), len(xs) - 1)] if xs else None  # noqa: E731
    return {"prob_per_order": _spread(prob), "prob_answer": _spread(avg),
            "label_mass": _spread(mass), "argmax_flip_groups": flips,
            "groups": len(reads),
            "seconds_per_question": {
                "median": round(q(ms, 0.5) / 1000, 3) if ms else None,
                "p90": round(q(ms, 0.9) / 1000, 3) if ms else None,
                "max": round(ms[-1] / 1000, 3) if ms else None},
            "processed_tokens_mean": round(statistics.mean(proc), 1)
            if proc else None,
            "label_mass_mean": round(statistics.mean(lm), 6) if lm else None,
            "variant_mass_mean": round(statistics.mean(vm), 6) if vm
            else None,
            "word_mass_mean": round(statistics.mean(wm), 6) if wm else None}


def across(all_reads: dict) -> dict:
    """Each regime's repeat-mean answer against cold's."""
    base = all_reads.get("cold")
    out = {}
    if not base:
        return out

    def means(reads):
        return {g: {k: statistics.mean(r["p"][k] for r in rs)
                    for k in rs[0]["p"]} for g, rs in reads.items()}
    b = means(base)
    for name, reads in all_reads.items():
        if name == "cold":
            continue
        m = means(reads)
        diffs, agree = [], 0
        for g in b:
            if g not in m:
                continue
            diffs += [abs(m[g][k] - b[g][k]) for k in b[g]]
            agree += max(m[g], key=m[g].get) == max(b[g], key=b[g].get)
        out[f"{name}_vs_cold"] = {
            "max": round(max(diffs), 6) if diffs else None,
            "mean": round(statistics.mean(diffs), 6) if diffs else None,
            "argmax_agree": agree, "groups": len(b)}
    return out


def _means(reads: dict) -> dict:
    return {g: {k: statistics.mean(r["p"][k] for r in rs)
                for k in rs[0]["p"]} for g, rs in reads.items()}


def _mean_diff(a: dict, b: dict) -> float:
    ma, mb = _means(a), _means(b)
    return max((abs(ma[g][k] - mb[g][k]) for g in ma if g in mb
                for k in ma[g]), default=0.0)


def exact(out: dict) -> bool:
    """Every repeat identical (6 decimals) in every order's probabilities
    and label mass, and no argmax flip."""
    return (out["prob_per_order"]["identical_groups"]
            == out["prob_per_order"]["groups"]
            and out["label_mass"]["identical_groups"]
            == out["label_mass"]["groups"]
            and out["argmax_flip_groups"] == 0)


def choose(outs: dict, all_reads: dict, repeats: int) -> dict:
    """THE OPERATOR'S RULE (2026-09-30, through the coordinator: "20s is not
    too slow"; "we would normally be paying 10s of minutes or more without
    jjava"): the CHEAPEST read mode that repeats exactly becomes jjava's
    default, fully cold included. A mode is cold or cached; jjava cannot
    choose whether a main decode runs beside it, so a mode counts as exact
    only when BOTH its environments (alone, beside a decode) repeat exactly.
    Cost: the two environments' median seconds per question, averaged.
    The TIE_BAND in the chosen mode (none exact: in `cached`, the
    production path, which then stays) is the largest difference the same
    question's answer shows there: within either environment's repeats,
    and between the environments' means (batch composition)."""
    modes = {}
    for m in ("cold", "cached"):
        a, b = outs.get(m), outs.get(m + "_beside")
        if not a or not b:
            continue
        cross = _mean_diff(all_reads[m], all_reads[m + "_beside"])
        band = max(a["prob_answer"]["max"] or 0.0,
                   b["prob_answer"]["max"] or 0.0, cross)
        secs = [x["seconds_per_question"]["median"] for x in (a, b)]
        modes[m] = {"exact": exact(a) and exact(b),
                    "exact_alone": exact(a), "exact_beside": exact(b),
                    "across_environments_max": round(cross, 6),
                    "tie_band": round(band, 6),
                    "seconds_per_question": round(statistics.mean(secs), 3)
                    if None not in secs else None,
                    "n": (a["groups"] + b["groups"]) * repeats}
    exact_modes = sorted((m for m in modes if modes[m]["exact"]),
                         key=lambda m: modes[m]["seconds_per_question"]
                         or float("inf"))
    chosen = exact_modes[0] if exact_modes else None
    band_mode = chosen or ("cached" if "cached" in modes else None)
    return {"modes": modes, "chosen": chosen, "band_mode": band_mode,
            "why": (f"{chosen}: the cheapest mode whose repeats are "
                    "identical alone and beside a decode" if chosen else
                    "no mode repeated exactly in both environments: the "
                    "read regime stays cached; its band is measured")}


def record(model: str, ch: dict, res: dict) -> dict:
    """The chosen regime and its band into the model's profile record
    (measure_model.write_field): `read_regime` only for an exact mode,
    `tie_band` for the band's mode."""
    import measure_model as MM
    date = time.strftime("%Y-%m-%d")
    out = {"written": []}
    m = ch.get("band_mode")
    if m is None:
        return {"written": [], "why": "no complete mode (both environments)"}
    x = ch["modes"][m]
    MM.write_field(model, "tie_band", {
        "value": x["tie_band"], "n": x["n"], "runs": 1, "date": date,
        "script": "bench/decider/determinism.py", "regime": m,
        "how": f"max |p| difference of one question's answer in the {m} "
               "read mode: within repeats alone, within repeats beside a "
               "main decode, and between the two environments' means",
        "states": len(res.get("states") or [])})
    out["written"].append("tie_band")
    if ch.get("chosen"):
        MM.write_field(model, "read_regime", {
            "mode": ch["chosen"], "repeats_exactly": True, "n": x["n"],
            "date": date, "script": "bench/decider/determinism.py",
            "seconds_per_question": x["seconds_per_question"],
            "modes": ch["modes"],
            "rule": "the cheapest mode that repeats exactly (operator, "
                    "2026-09-30)"})
        out["written"].append("read_regime")
    return out


def plan(n_states: int, repeats: int, regimes) -> dict:
    per = n_states * 3 * repeats * 2
    place = n_states * 2
    return {"states": n_states, "questions_per_state": 3,
            "repeats": repeats, "regimes": list(regimes),
            "reads_per_regime": per,
            "reads_total": sum(per + (place if not r.startswith("cold")
                                      else 0) for r in regimes),
            "estimate": "cold reads re-process the whole state (~2k tokens "
                        "each); cached reads only the question -- the run "
                        "reports the seconds it took"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    DT.add_args(ap)
    ap.add_argument("--states", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--cases")
    ap.add_argument("--decode-slot", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-write", action="store_true",
                    help="do not record the chosen regime and band")
    a = ap.parse_args(argv)
    regimes = [r for r in a.regimes.split(",") if r in REGIMES]
    if a.dry_run:
        sts = states(a.states, a.cases, None)
        qs = questions(sts[0]["fact"]) if sts else []
        for q in qs:
            for c in D.rendered_orders(q):
                D.guard_body(D.body("x", c, None, 20, "bonsai"))
        print(json.dumps({"plan": plan(len(sts), a.repeats, regimes),
                          "state_ids": [s["id"] for s in sts],
                          "state_tokens_chars3": [s["tokens"] for s in sts],
                          "questions": [q["name"] for q in qs],
                          "guard": "every body passes guard_body"},
                         indent=1))
        return 0
    try:
        t = DT.from_args(a)
    except (DT.NotRun, ValueError) as e:
        print(f"NOT RUN: {e}")
        return 3
    if t.mode == "one door":
        import gpu_room
        import model
        loaded, why = gpu_room.model_loaded(model.UPSTREAM, model.MODEL)
        if not loaded:
            print(f"NOT RUN: {model.MODEL} is not loaded ({why}); this bench "
                  "never loads a model")
            return 3
    DT.install(t, decisions_dir=os.path.join(HERE, "results",
                                              "determinism_decisions"))
    import cancel
    import max_mode
    import slots
    res = {"version": VERSION, "target": t.describe(),
           "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "plan": plan(a.states, a.repeats, regimes)}
    with cancel.bound(cancel.Token()):
        if t.mode == "one door":
            max_mode.set_current(t.model)
        try:
            DT.preflight(t)
        except DT.Busy as e:
            print(f"BUSY: {e}")
            return 2
        res["template_check"] = D.template_check()
        if res["template_check"].get("checked") and \
                not res["template_check"].get("ok"):
            print("STOP: the served rendering leaves the think block open:",
                  json.dumps(res["template_check"]))
            _write(t.model, res)
            return 1
        count = (lambda s: len(D._upstream("/tokenize", {
            "content": s, "add_special": False})["tokens"]))
        sts = states(a.states, a.cases, count)
        grant = slots.acquire(None, transient=True)
        slot = grant.get("slot")
        dslot = a.decode_slot if a.decode_slot is not None else (
            0 if slot != 0 else 1)
        mine = (slot, dslot)
        res.update(slot=slot, decode_slot=dslot,
                   states=[{"id": s["id"], "tokens": s["tokens"]}
                           for s in sts])
        all_reads, t0 = {}, time.time()
        try:
            for r in regimes:
                t1 = time.time()
                try:
                    out, reads = run_regime(r, sts, a.repeats, slot, t,
                                            dslot, mine)
                except DT.Busy as e:
                    res["stopped"] = f"busy in {r}: {e}"
                    break
                out["wall_s"] = round(time.time() - t1, 1)
                res.setdefault("regimes", {})[r] = out
                all_reads[r] = reads
                print(r, json.dumps({k: out[k] for k in (
                    "prob_per_order", "label_mass", "seconds_per_question")}),
                      flush=True)
        finally:
            slots.release(grant)
        res["across"] = across(all_reads)
        res["wall_s"] = round(time.time() - t0, 1)
        if not res.get("stopped") and set(REGIMES) <= set(all_reads):
            res["choice"] = choose(res["regimes"], all_reads, a.repeats)
            res["record"] = (record(t.model, res["choice"], res)
                             if not a.no_write else
                             {"written": [], "why": "--no-write"})
        else:
            res["record"] = {"written": [], "why": "not every regime ran"}
    path = _write(t.model, res)
    print("written", path)
    return 2 if res.get("stopped") else 0


def _write(model: str, res: dict) -> str:
    path = os.path.join(HERE, "results", f"determinism_{model}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return path


if __name__ == "__main__":
    sys.exit(main())
