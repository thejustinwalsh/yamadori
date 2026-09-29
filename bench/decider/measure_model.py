#!/usr/bin/env python
"""MEASURE ONE MODEL FOR JJAVA: its label tokens, letter priors, label bias,
tie band and the legacy-vs-typed comparison, filed as the model's profile
record -- the file the runtime reads (decider_bonsai THE MODEL PROFILE).

    python bench/decider/measure_model.py --model flash-next \\
        --base-url http://127.0.0.1:18095                  # the plan only
    python bench/decider/measure_model.py --run --model flash-next \\
        --base-url http://127.0.0.1:18095                  # GPU: all parts
    python bench/decider/measure_model.py --run --model flash-next \\
        --only labels,letter_prior,label_bias             # the short parts

2026-09-29: jjava runs against whichever model serves the conversation
(bonsai; flash-next at max; mirai-s at xhigh). "Per-model letter priors and
label bias must be MEASURED, not assumed" (the coordinator's brief). The
engine is bench/decider/decider_target.py's: --base-url for a bare
llama-server on a test port, else llama-swap's /upstream/<model> (which
must already be loaded: /running is checked, nothing is loaded here), else
-- the default model, no --base-url -- the one door, unchanged.

THE PARTS (in this order; --only picks):

  labels         /tokenize only (no generation): which spellings of A..Z
                 (listed "A", answered " A") and of yes/no
                 (decider_bonsai.LABEL_SPELLINGS) are ONE token, and their
                 ids (an id hash compares tokenizers across records).
  letter_prior   THE LETTER BIAS: for every option count 2..26, the
                 content-free read (state decider_bonsai.NEUTRAL_STATE,
                 every option's text NEUTRAL_STATE -- Calibrate Before Use's
                 content-free input; decide_turn._id_prior's blanking) of
                 decide_turn.FIT_Q, in both renderings the decider uses:
                 `typed` (decider_bonsai.messages, "A. N/A") and `choose`
                 (decide_turn._choose_msgs, the options in their own
                 message, "A) N/A"). The two option orders print the SAME
                 prompt here (every option is N/A), so one read per count
                 and rendering. Recorded: the distribution by letter, the
                 label mass, the largest deviation from uniform, the top
                 letter.
  label_bias     each question set's own template on the content-free
                 state, through the typed readout (decider_bonsai.read, two
                 orders): build_intent (noul), phase (choice), choose:stop
                 (judge_stop's choice, the choose rendering). Recorded: the
                 answer, each order's distribution, their disagreement, the
                 label mass.
  tie_band       bench/decider/bonsai_decider.py `batching`, unchanged: 20
                 daily-eval states x 5 questions, prefix-cached vs
                 cache_prompt false; the band is the largest |p| difference
                 (how Bonsai's 0.0034 was measured, n=100). THIS VALUE IS
                 USED by the runtime for this model (decider_bonsai
                 tie_band()).
  legacy_vs_typed  bench/decider/legacy_vs_typed.py run() (intent and
                 daily_choose, both arms; --lvt-parts picks), its full
                 result in results/legacy_vs_typed[.<model>].json, its
                 summary() in the record.

THE RECORD: decider_bonsai.PROFILES_DIR/<model>.json (default
bench/decider/results/models/; --out-dir), version decider_bonsai.
PROFILE_VERSION, one field per part -- each with its date, script, n, the
engine (decider_target.Target.describe: mode, base, the server's model file
and build) -- merged into the existing record (a part not re-run keeps its
old field). Written after every part, so a stopped run keeps what it
finished. A part that fails or is stopped writes nothing for its field.
Every number is ONE run (docs/PROTOCOL.md: repeat before citing a
difference).

ONE GPU CONSUMER: the preflight (decider_target.preflight: no slot of the
target's server processing, no hermes.exe) before each part, and inside
tie_band and legacy_vs_typed every CHECK_EVERY (25) reads or rows as those
benches do; busy -> exit 2 (NOT RUN, never a
result); the target unreachable or not loaded -> exit 3.

Decisions the parts make through decide_turn go to logs/decider_measure/
<model>.decisions.jsonl, never the live decision log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)

import decider_bonsai as D  # noqa: E402
import decider_target as DT  # noqa: E402

MEASURE_VERSION = "decider-measure/1"
PARTS = ("labels", "letter_prior", "label_bias", "tie_band",
         "legacy_vs_typed")
SCRIPT = "bench/decider/measure_model.py"
# 2..26: one letter per option up to decider_bonsai.LETTERS (a choice takes
# at least two options; LETTERS bounds the rest).
COUNTS = tuple(range(2, len(D.LETTERS) + 1))
DECISIONS_DIR = os.path.join(ROOT, "logs", "decider_measure")


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class _Slot:
    """The slot a part reads on: the target's (install() placed it), or --
    the one door -- the stack's decider lane, acquired and released like a
    Turn's (slots.acquire(None, transient=True))."""

    def __init__(self, t: DT.Target):
        self.t, self.grant = t, None

    def __enter__(self):
        if self.t.mode != "one door":
            return self.t.slot
        import slots
        self.grant = slots.acquire(None, transient=True)
        return self.grant.get("slot")

    def __exit__(self, *exc):
        if self.grant is not None:
            import slots
            slots.release(self.grant)
        return False


def _doors(t: DT.Target) -> dict:
    """post / upstream for the decider: the one door's own (None: resolved
    by decider_bonsai at call time) or the target's."""
    if t.mode == "one door":
        return {"post": None, "upstream": None}
    return {"post": t.post, "upstream": t.upstream}


def _field(t: DT.Target, part: str, n: int, **kw) -> dict:
    return {"date": _today(), "script": f"{SCRIPT} --only {part}", "n": n,
            "runs": 1, "engine": t.describe(), "template":
            D.TEMPLATE_VERSION, "readout": D.READOUT_VERSION, **kw}


# ------------------------------------------------------------------ parts --
def part_labels(t: DT.Target) -> dict:
    """Which label spellings are one token on the target's tokenizer."""
    spellings = {lab: (lab, " " + lab) for lab in D.LETTERS}
    spellings.update({k: tuple(v) for k, v in D.LABEL_SPELLINGS.items()})
    ids: dict = {}
    not_one: dict = {}
    n = 0
    for lab, spell in spellings.items():
        for s in spell:
            n += 1
            toks = t.upstream("/tokenize", {"content": s,
                                            "add_special": False})["tokens"]
            if isinstance(toks, list) and len(toks) == 1:
                tk = toks[0]
                ids.setdefault(lab, {})[s] = int(tk["id"] if isinstance(
                    tk, dict) else tk)
            else:
                not_one.setdefault(lab, []).append(s)
    letters_ok = all(" " + lab in ids.get(lab, {}) and lab in ids.get(lab, {})
                     for lab in D.LETTERS)
    yes_no_ok = all(ids.get(k) for k in D.LABEL_SPELLINGS)
    return _field(t, "labels", n,
                  single_letters_one_token=letters_ok,
                  yes_no_one_token=yes_no_ok, not_one_token=not_one,
                  ids=ids, ids_sha=hashlib.sha256(json.dumps(
                      ids, sort_keys=True).encode()).hexdigest()[:16])


def _choose_render(question: str):
    """decide_turn's choose rendering (its _choose_typed with no earlier
    rounds): state / the options message / the question / the prefill."""
    import decide_turn as T

    def render(c: dict, state: str = D.NEUTRAL_STATE) -> list[dict]:
        printed = [{"letter": lab, "text": x}
                   for lab, x in zip(c["labels"], c["options"])]
        return T._choose_msgs(state, T._options_msg(printed), [], question)
    return render


def _dist_summary(p: dict, n: int) -> dict:
    top = max(p, key=p.get) if p else None
    return {"top": top, "top_p": round(p.get(top, 0.0), 6) if top else None,
            "max_dev": round(max(abs(v - 1.0 / n) for v in p.values()), 6)
            if p else None}


def part_letter_prior(t: DT.Target, counts=COUNTS) -> dict:
    """The content-free letter distribution per option count, in the typed
    and the choose renderings."""
    import decide_turn as T
    render = _choose_render(T.FIT_Q)
    out: dict = {"typed": {}, "choose": {}}
    reads = 0
    with _Slot(t) as slot:
        for n in counts:
            q = D.q_choice("letter_prior", T.FIT_Q, [D.NEUTRAL_STATE] * n)
            c = D.rendered_orders(q)[0]
            for name, msgs in (("typed", None), ("choose", render(c))):
                a = D.ask_one(D.NEUTRAL_STATE, c, slot=slot, msgs=msgs,
                              **_doors(t))
                reads += 1
                p = {lab: float(v) for lab, v in a["probs"].items()}
                out[name][str(n)] = dict(
                    p={k: round(v, 6) for k, v in p.items()},
                    label_mass=a.get("label_mass"), exact=a.get("exact"),
                    reads=a.get("reads"), **_dist_summary(p, n))
    summ = {}
    for name, rows in out.items():
        tops = [r["top"] for r in rows.values()]
        summ[name] = {
            "max_dev": max((r["max_dev"] or 0.0) for r in rows.values())
            if rows else None,
            "top_is_first": sum(1 for x in tops if x == "A"),
            "top_is_last": sum(1 for k, r in rows.items()
                               if r["top"] == D.LETTERS[int(k) - 1]),
            "counts": len(rows),
            "label_mass_min": min((r["label_mass"] or 0.0)
                                  for r in rows.values()) if rows else None}
    return _field(t, "letter_prior", reads, question=T.FIT_Q,
                  neutral=D.NEUTRAL_STATE, renderings=out, summary=summ,
                  orders_note="both option orders print the same prompt "
                              "(every option is the neutral text): one read "
                              "per count and rendering")


def part_label_bias(t: DT.Target) -> dict:
    """Each question set's own template, content-free, typed readout."""
    import decide_turn as T
    stop_keys = [k for k, _ in T.STOP_OPTIONS]
    qs = [(D.q_noul("build_intent", T.BUILD_INTENT_Q), None),
          (D.q_choice("phase", T.PHASE_Q[0], T.PHASE_Q[1], keys=T.PHASES),
           None),
          (D.q_choice("choose:stop", T.STOP_Q,
                      [x for _, x in T.STOP_OPTIONS], keys=stop_keys,
                      none=stop_keys.index("other")),
           _choose_render(T.STOP_Q))]
    out: dict = {}
    reads = 0
    for q, render in qs:
        with _Slot(t) as slot:
            a = D.read(D.NEUTRAL_STATE, q, slot=slot, render=render,
                       **_doors(t))
        d = a["diagnostics"]
        reads += d.get("reads") or 0
        out[q["name"]] = {
            "type": a["type"],
            **({"noul": a["noul"]} if a["type"] == "noul" else
               {"choice": a.get("choice"),
                "probabilities": a.get("probabilities"),
                "confidence": a.get("confidence")}),
            "orders": [{"printed": o["printed"], "p": o["probs"],
                        "label_mass": o.get("label_mass")}
                       for o in d.get("orders") or []],
            "disagreement": d.get("disagreement"),
            "argmax_agree": d.get("argmax_agree"),
            "label_mass_min": d.get("label_mass_min")}
    return _field(t, "label_bias", reads, neutral=D.NEUTRAL_STATE,
                  questions=out)


def tie_band_of_batching(res: dict) -> dict:
    """The tie band from bonsai_decider.part_batching's result: the largest
    |p| difference over every (state, question, label)."""
    rows = res.get("rows") or []
    if not rows:
        raise ValueError("the batching part returned no rows")
    return {"value": round(max(float(r["max_prob_diff"]) for r in rows), 6),
            "n": int(res.get("n_states") or len(rows))
            * int(res.get("questions_per_state") or 0),
            "states": len(rows),
            "same_answers": sum(1 for r in rows if r.get("same_answers")),
            "sum_processed": res.get("sum_processed"),
            "sum_batch_ms": res.get("sum_batch_ms")}


def part_tie_band(t: DT.Target) -> dict:
    import bonsai_decider as B
    B.TARGET = t
    res = B.part_batching()
    tb = tie_band_of_batching(res)
    return _field(t, "tie_band", tb.pop("n"), **tb,
                  how="bench/decider/bonsai_decider.py batching: max "
                      "|p_label| difference between a prefix-cached and a "
                      "cache_prompt-false read of the same question")


def part_legacy_vs_typed(t: DT.Target, parts=("intent", "daily_choose")
                         ) -> dict:
    import legacy_vs_typed as L
    res, code = L.run(list(L.ARMS), list(parts), t)
    path = L.out_path(t.model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if code == 2:
        raise DT.Busy(res["status"])
    if code == 3:
        raise DT.NotRun(res["status"])
    if code:
        raise RuntimeError(res["status"])
    s = L.summary(res)
    n = sum(v.get("n") or 0 for k, v in (s.get("intent") or {}).items()
            if k in L.ARMS)
    n += sum(v.get("total") or 0 for v in (s.get("daily_choose")
                                          or {}).values())
    return _field(t, "legacy_vs_typed", n, parts=list(parts),
                  result=os.path.relpath(path, ROOT).replace("\\", "/"),
                  **s)


# ----------------------------------------------------------------- record --
def record_path(model: str, out_dir: str | None = None) -> str:
    return os.path.join(out_dir or D.PROFILES_DIR,
                        f"{D.canonical_model(model)}.json")


def write_field(model: str, field: str, value: dict, *,
                out_dir: str | None = None, log: dict | None = None) -> str:
    """Merge one measured field into the model's record (atomic)."""
    path = record_path(model, out_dir)
    rec: dict = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = {}
    if rec.get("version") not in (None, D.PROFILE_VERSION):
        rec = {}
    rec.update(version=D.PROFILE_VERSION, model=D.canonical_model(model),
               measure=MEASURE_VERSION, updated=time.strftime(
                   "%Y-%m-%dT%H:%M:%S"))
    rec[field] = value
    if log:
        rec.setdefault("log", []).append(log)
        rec["log"] = rec["log"][-50:]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def plan(t: DT.Target, only, lvt_parts) -> str:
    import decide_turn as T
    n_daily = n_intent = "?"
    try:
        import legacy_vs_typed as L
        n_intent = len(L._rows(L.INTENT))
        n_daily = len(L._rows(L.DAILY))
    except Exception:                                            # noqa: BLE001
        pass
    est = {"labels": f"{2 * len(D.LETTERS) + sum(len(v) for v in D.LABEL_SPELLINGS.values())} /tokenize calls, no generation",
           "letter_prior": f"{2 * len(COUNTS)} one-token reads of ~60-200 "
                           "prompt tokens (a read is repeated with K x10 "
                           "only when a letter is outside the top 20)",
           "label_bias": "6 one-token reads (3 questions x 2 orders)",
           "tie_band": "200 one-token reads (20 states x 5 questions x "
                       "cached/uncached); ~16k processed tokens cached, "
                       "~71k uncached, one state 11.4k tokens (Bonsai took "
                       "169 s: bench/decider/results/bonsai.json batching)",
           "legacy_vs_typed": f"intent {n_intent} rows x 2 arms x 2 reads; "
                              f"daily_choose {n_daily} rows x 2 arms x "
                              "(facts + the selector's rounds) -- "
                              + ",".join(lvt_parts)}
    lines = ["THE PLAN (nothing is sent without --run):",
             f"  target: {t.model} ({t.mode}, {t.base}, slot {t.slot})",
             f"  record: {os.path.relpath(record_path(t.model), ROOT)} "
             f"({D.PROFILE_VERSION}; merged, written after each part)",
             f"  decisions: logs/decider_measure/{t.model}.decisions.jsonl"]
    for p in only:
        lines.append(f"  {p:<16} {est[p]}")
    if D.canonical_model(t.model) in D.BUILTIN_PROFILES and "tie_band" in only:
        lines.append(f"  NOTE: a measured tie_band REPLACES {t.model}'s "
                     f"built-in value ({D.BUILTIN_PROFILES[D.canonical_model(t.model)]['tie_band']['value']}) "
                     "at runtime once the record is written")
    lines.append(f"  question for the letter prior: {T.FIT_Q!r}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="run it (GPU); without it the plan is printed")
    ap.add_argument("--only", default=",".join(PARTS),
                    help=f"parts, in order: {','.join(PARTS)}")
    ap.add_argument("--lvt-parts", default="intent,daily_choose",
                    help="legacy_vs_typed's parts")
    ap.add_argument("--out-dir", help="where the record goes (default "
                    "decider_bonsai.PROFILES_DIR)")
    DT.add_args(ap)
    a = ap.parse_args(argv)
    only = [p for p in a.only.split(",") if p]
    lvt = [p for p in a.lvt_parts.split(",") if p]
    if set(only) - set(PARTS) or set(lvt) - {"intent", "daily_choose"}:
        print(f"parts are {PARTS}; --lvt-parts from intent,daily_choose")
        return 2
    if not a.run:
        # The plan only: nothing is asked of any engine.
        import model as M
        name = a.model or M.MODEL
        mode = ("base-url" if a.base_url else "one door"
                if D.canonical_model(name) == D.canonical_model(M.MODEL)
                else "llama-swap")
        shown = DT.Target(name, mode, (a.base_url or "").rstrip("/")
                          or f"{M.UPSTREAM}/upstream/{name}", a.slot)
        print(plan(shown, only, lvt))
        return 0
    if a.out_dir:
        D.PROFILES_DIR = os.path.abspath(a.out_dir)
    try:
        t = DT.from_args(a)
    except (DT.NotRun, ValueError) as e:
        print(f"NOT RUN: {e}")
        return 3
    print(plan(t, only, lvt), flush=True)
    DT.install(t, decisions_dir=DECISIONS_DIR)
    fns = {"labels": part_labels, "letter_prior": part_letter_prior,
           "label_bias": part_label_bias, "tie_band": part_tie_band,
           "legacy_vs_typed": lambda tt: part_legacy_vs_typed(tt, lvt)}
    code = 0
    for part in only:
        t0 = time.time()
        entry = {"part": part, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            DT.preflight(t)
            value = fns[part](t)
        except DT.Busy as e:
            print(f"[{part}] STOPPED (not run): {e}", flush=True)
            code = 2
            break
        except DT.NotRun as e:
            print(f"[{part}] NOT RUN: {e}", flush=True)
            code = 3
            break
        except Exception as e:                                   # noqa: BLE001
            print(f"[{part}] FAILED: {type(e).__name__}: {e}"[:400],
                  flush=True)
            code = 1
            continue
        entry["seconds"] = round(time.time() - t0, 1)
        path = write_field(t.model, part, value, out_dir=a.out_dir,
                           log=entry)
        print(f"[{part}] done in {entry['seconds']} s -> {path}", flush=True)
    st = D.profile_status(t.model)
    print("profile now:", json.dumps({k: st[k] for k in (
        "model", "measured", "unmeasured", "tie_band")}, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
