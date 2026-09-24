"""Build the leaderboard comparison for a run.

    python compare.py results/<run-id> <leaderboard.json> <table_2024_11_25.csv>

Three tiers, never mixed in one number:

A. SAME QUESTIONS. LiveBench publishes per-question judgments for its
   2024-11-25 leaderboard models (HF livebench/model_judgment, split
   "leaderboard"). Each reference model is re-scored on exactly the questions
   our arm scored, with the same LiveBench aggregation. This is the only
   apples-to-apples row. Before it is trusted, the same code recomputes each
   reference model on the FULL release population and compares it with the
   published table_2024_11_25.csv (reported as `reproduction`).
B. SAME RELEASE, FULL POPULATION: the published 2024-11-25 table itself.
C. CURRENT LEADERBOARD (2026-06-25): different, unpublished questions and
   different task mixes. Context only; it includes Qwen3.8-27B, our base.

Writes comparison.json and comparison.md into the run dir.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize as S  # noqa: E402

CAT_LABEL = {"coding": "Coding", "reasoning": "Reasoning", "instruction_following": "IF",
             "math": "Mathematics", "data_analysis": "Data Analysis", "language": "Language"}

# (label shown, name in model_judgment, name in table_2024_11_25.csv)
REFERENCE = [
    ("Claude 3.7 Sonnet thinking (64k)", "claude-3-7-sonnet-20250219-thinking-64k", "claude-3-7-sonnet-thinking"),
    ("Claude 3.7 Sonnet", "claude-3-7-sonnet-20250219-base", "claude-3-7-sonnet"),
    ("Claude 3.5 Sonnet (2024-10-22)", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
    ("Claude 3.5 Haiku", "claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"),
    ("o3-mini high", "o3-mini-2025-01-31-high", "o3-mini-2025-01-31-high"),
    ("o1 high", "o1-2024-12-17-high", "o1-2024-12-17-high"),
    ("GPT-4.5 preview", "gpt-4.5-preview-2025-02-27", "gpt-4.5-preview"),
    ("GPT-4o (2024-11-20)", "gpt-4o-2024-11-20", "gpt-4o-2024-11-20"),
    ("GPT-4o mini", "gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"),
    ("Gemini 2.5 Pro exp", "gemini-2.5-pro-exp-03-25", "gemini-2.5-pro-exp-03-25"),
    ("DeepSeek R1", "deepseek-r1", "deepseek-r1"),
    ("QwQ-32B", "qwq-32b", "qwq-32b"),
    ("Qwen2.5-Max", "qwen2.5-max", "qwen2.5-max"),
    ("Qwen2.5-72B Instruct", "Qwen2.5-72B-Instruct-Turbo", "qwen2.5-72b-instruct-turbo"),
    ("Qwen2.5-Coder-32B Instruct", "Qwen2.5-Coder-32B-Instruct", "qwen2.5-coder-32b-instruct"),
    ("R1-Distill-Qwen-32B", "deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-qwen-32b"),
    ("Gemma 3 27B", "gemma-3-27b-it", "gemma-3-27b-it"),
    ("Mistral Small 3.1 (2503)", "mistral-small-2503", "mistral-small-2503"),
]

CURRENT = ["Claude Fable 5.1 Max Effort", "claude-opus-5-5-max-effort", "Claude Sonnet 5 xHigh Effort",
           "GPT-6 Astra Max Effort", "GPT-6 Sol Max Effort", "GPT-6 Luna Max Effort", "gpt-5.4-mini-xhigh",
           "Qwen 3.8 Max", "Qwen3.8 27B", "Smaug Mini", "Qwen 3.8 Flash Next", "Qwen 3.6 27B",
           "GLM-5.3 Flash", "Gemma 4 31B", "QwQ 32B"]


def load_judgments():
    from datasets import load_dataset
    ds = load_dataset("livebench/model_judgment", split="leaderboard")
    wanted = {r[1] for r in REFERENCE}
    by = defaultdict(dict)  # model -> qid -> (task, category, score)
    for r in ds:
        if r["model"] in wanted:
            prev = by[r["model"]].get(r["question_id"])
            if prev is None or r["tstamp"] >= prev[3]:
                by[r["model"]][r["question_id"]] = (r["task"], r["category"], float(r["score"]), r["tstamp"])
    return by


def score_on(judg: dict, qids: dict[str, tuple[str, str]], complete_only: bool = False) -> tuple[dict, int, int]:
    """LiveBench aggregation of a reference model restricted to qids {qid: (category, task)}.

    With complete_only, a category is dropped unless the reference has a judgment
    for EVERY one of our questions in it, and the overall is None unless every
    category survived: the HF judgments cover only coding, IF/paraphrase and
    language, and a partial category would compare different task mixes."""
    by = defaultdict(lambda: defaultdict(list))
    missing = 0
    miss_cat = defaultdict(int)
    for q, (c, t) in qids.items():
        j = judg.get(q)
        if j is None:
            missing += 1
            miss_cat[c] += 1
            continue
        by[c][t].append(j[2])
    if complete_only:
        cats_all = {c for c, _ in qids.values()}
        by = {c: v for c, v in by.items() if miss_cat[c] == 0}
        agg = S.aggregate(by)
        if set(agg["categories"]) != cats_all:
            agg["overall"] = None
        return agg, len(qids) - missing, missing
    return S.aggregate(by), len(qids) - missing, missing


def main() -> int:
    run_dir, lb_json, table_csv = sys.argv[1:4]
    summary = S.build(run_dir, B=S.B_DEFAULT)  # same B and seed as summary.json, so the CIs are identical
    rows = S.load_rows(run_dir)
    orders = {}
    for c in CAT_LABEL:
        p = os.path.join(run_dir, f"order_{c}.json")
        if os.path.exists(p):
            orders[c] = json.load(open(p))["order"]

    table = {r["model"]: r for r in csv.DictReader(open(table_csv))}
    cats_2411 = {"Reasoning": ["web_of_lies_v2", "zebra_puzzle", "spatial"], "Coding": ["LCB_generation", "coding_completion"],
                 "Mathematics": ["AMPS_Hard", "math_comp", "olympiad"], "Data Analysis": ["cta", "tablejoin", "tablereformat"],
                 "Language": ["connections", "plot_unscrambling", "typos"],
                 "IF": ["paraphrase", "simplify", "story_generation", "summarize"]}
    judg = load_judgments()

    out = {"run_id": summary["run_id"], "release": summary["release"], "arms": {}, "reproduction": {},
           "same_questions": {}, "published_2024_11_25": {}, "current_leaderboard": {}}

    # --- reproduction: full population vs the published table -------------------------------
    full = {q["question_id"]: (c, q["task"]) for c, o in orders.items() for q in o}
    for label, jname, tname in REFERENCE:
        if jname not in judg or tname not in table:
            out["reproduction"][label] = {"status": "missing", "judgment": jname in judg, "table": tname in table}
            continue
        agg, n, miss = score_on(judg[jname], full)
        diffs = {}
        for c, v in agg["categories"].items():
            for t, s in v["tasks"].items():
                pub = table[tname].get(t)
                if pub not in (None, ""):
                    diffs[t] = round(s - float(pub), 3)
        out["reproduction"][label] = {"n": n, "missing": miss,
                                      "max_abs_task_diff": max((abs(x) for x in diffs.values()), default=None),
                                      "task_diffs": diffs}

    # --- A: same questions as each arm --------------------------------------------------------
    for arm, v in summary["arms"].items():
        cats = v["overall"]["categories_included"]
        out["arms"][arm] = {"overall": v["overall"], "categories": {c: {"score": v["categories"][c]["score"],
                            "ci95": v["categories"][c]["ci95"], "n": v["categories"][c]["n"]} for c in cats}}
        qids = {r["id"]: (r["category"], r["task"]) for r in rows if r["arm"] == arm and r["status"] in S.SCORED}
        ref = {}
        for label, jname, _ in REFERENCE:
            if jname not in judg:
                continue
            agg, n, miss = score_on(judg[jname], qids, complete_only=True)
            ref[label] = {"overall": agg["overall"], "n": n, "missing": miss,
                          "categories": {c: x["score"] for c, x in agg["categories"].items()}}
        out["same_questions"][arm] = ref

    # --- B: published 2024-11-25 table (full population) ---------------------------------------
    for label, _, tname in REFERENCE:
        r = table.get(tname)
        if not r:
            continue
        cats = {}
        for c, ts in cats_2411.items():
            vals = [float(r[t]) for t in ts if r.get(t) not in (None, "")]
            if vals:
                cats[c] = sum(vals) / len(vals)
        out["published_2024_11_25"][label] = {"categories": cats,
                                              "global": sum(cats.values()) / len(cats) if cats else None}

    # --- C: current leaderboard ------------------------------------------------------------------
    lb = json.load(open(lb_json))
    out["current_leaderboard"] = {"leaderboard_date": lb.get("leaderboard_date"), "release": lb.get("release"),
                                  "sources": lb.get("sources"), "models": {}}
    for m in lb["models"]:
        if m["name"] in CURRENT and m.get("release") in ("2026-06-25",) or (m["name"] == "Gemma 4 31B" or m["name"] == "QwQ 32B"):
            key = f"{m['name']} [{m['release']}]"
            out["current_leaderboard"]["models"][key] = {"global": m["global"], "categories": m["categories"],
                                                        "release": m["release"], "source": m.get("source")}

    with open(os.path.join(run_dir, "comparison.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_md(run_dir, out, summary)
    print(f"-> {os.path.join(run_dir, 'comparison.json')} and comparison.md")
    return 0


def fmt(x, ci=None):
    if x is None:
        return "--"
    return f"{x:.1f}" + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")


def pval(p):
    return "--" if p is None else f"{p:.3f}"


def write_md(run_dir, out, summary):
    arms = list(out["arms"])
    cats = sorted({c for a in out["arms"].values() for c in a["categories"]}, key=list(CAT_LABEL).index)
    L = []
    L.append(f"# LiveBench comparison -- run {out['run_id']}\n")
    L.append("**Read this first.** Our numbers are a local, ternary-quantised (1.75 bpw) derivative of "
             "Qwen3.8-27B served on one consumer GPU, plus our tool stack in the `yamadori` arm. LiveBench's "
             "leaderboard numbers come from the providers' own APIs at full precision. The question set is the "
             f"latest PUBLIC release, {', '.join(out['release'])}; the current leaderboard (2026-06-25) uses "
             "newer questions that LiveBench has not published, so tier C below is context, not a comparison. "
             "The 2024-11-25 questions have been public since April 2025 and may be in any 2026 model's training "
             "data, ours included: treat every number here as possibly contaminated.\n")
    L.append("## A. Same questions (apples to apples)\n")
    L.append("Reference models re-scored from LiveBench's published per-question judgments on exactly the "
             "questions our arm scored; same aggregation (task mean -> category mean of tasks -> overall mean of "
             "categories). 95% interval: bootstrap over questions, ours only (reference rows are fixed "
             "published judgments on the same sample).\n")
    for arm in arms:
        a = out["arms"][arm]
        L.append(f"### arm `{arm}` (n per category: " + ", ".join(f"{c} {a['categories'][c]['n']}" for c in a["categories"]) + ")\n")
        L.append("| model | overall | " + " | ".join(CAT_LABEL[c] for c in cats if c in a["categories"]) + " |")
        L.append("|---|---|" + "---|" * len([c for c in cats if c in a["categories"]]))
        L.append(f"| **ours: {arm}** | **{fmt(a['overall']['score'], a['overall']['ci95'])}** | " +
                 " | ".join(fmt(a["categories"][c]["score"], a["categories"][c]["ci95"]) for c in cats if c in a["categories"]) + " |")
        ref = out["same_questions"][arm]
        for label, v in sorted(ref.items(), key=lambda kv: -(kv[1]["overall"] or 0)):
            L.append(f"| {label} | {fmt(v['overall'])} | " + " | ".join(fmt(v["categories"].get(c)) for c in cats if c in a["categories"]) + " |")
        L.append("\n`--` = LiveBench publishes no per-question judgments for that category on this release "
                 "(the HF set covers coding, IF/paraphrase and language only); use table B for it.\n")
        L.append("")
    labels = summary.get("arm_labels", {})
    if labels:
        L.append("### Arms\n")
        for arm, lab in labels.items():
            L.append(f"- `{arm}`: {lab}")
        L.append("\nTier `xhigh` sends MEDIUM effort upstream, so `yamadori-xhigh` vs `bonsai` (bare @ medium) is "
                 "effort-matched: all augmentation allowed vs all forced off. Tier `max` sends xhigh effort, so "
                 "`yamadori` (tier max, a partial arm dropped by the operator) vs `bonsai` mixes effort and tools.\n")
    L.append("### Mechanism health per arm\n")
    L.append("Counts over each arm's answers, read from `x_yamadori` on every response (bench/livebench/mechanisms.py). "
             "allowed = the tier or header permitted it; decided = selection chose it for that request; ran = it "
             "executed; produced = it returned data. A broken mechanism is a stack_error: that answer is not scored "
             "and is re-asked. Retrieval is read from `x_yamadori.tools`; for answers from before the proxy emitted it only "
             "`hops` is known, those count under `answers_without_tool_record`, and `produced` is `--` when no answer "
             "had the record.\n")
    L.append("| arm | answers | stack errors | mechanism | allowed | decided | ran | produced | detail |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for arm, v in summary["arms"].items():
        mh = v.get("mechanism_health")
        if not mh:
            continue
        first = True
        for m, c in mh["mechanisms"].items():
            detail = ", ".join(f"{k} {x}" for k, x in c.items() if k not in ("allowed", "decided", "ran", "produced"))
            L.append(f"| {arm if first else ''} | {mh['answers'] if first else ''} | {mh['stack_errors'] if first else ''} | "
                     f"{m} | {c.get('allowed', '--')} | {c.get('decided', '--')} | {c.get('ran', '--')} | "
                     f"{'--' if c.get('produced') is None else c.get('produced')} | {detail} |")
            first = False
    L.append("")
    for key in sorted(k for k in summary if k.startswith("paired_")):
        p = summary[key]
        if p["diff_overall"] is None:
            continue
        arm = key[len("paired_"):-len("_minus_bonsai")]
        L.append(f"### Paired difference, {arm} minus bonsai (same questions)\n")
        L.append(f"| category | n pairs | bonsai | {arm} | diff [95% CI] | exact McNemar p |")
        L.append("|---|---|---|---|---|---|")
        for c, v in p["categories"].items():
            L.append(f"| {CAT_LABEL.get(c, c)} | {v['n_pairs']} | {v['a_score']:.1f} | {v['b_score']:.1f} | "
                     f"{v['diff']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] | "
                     f"{pval(v['mcnemar_p'])} |")
        L.append(f"| overall | {p['n_pairs']} | | | {p['diff_overall']:+.1f} [{p['ci95_overall'][0]:+.1f}, {p['ci95_overall'][1]:+.1f}] | |\n")
    L.append("### Reproduction check\n")
    L.append("Each reference model recomputed on the FULL release from per-question judgments, against the "
             "published table_2024_11_25.csv. Max absolute per-task difference (points):\n")
    L.append("| model | n | max abs task diff |")
    L.append("|---|---|---|")
    for label, v in out["reproduction"].items():
        L.append(f"| {label} | {v.get('n', '--')} | {v.get('max_abs_task_diff', v.get('status'))} |")
    L.append("")
    L.append("## B. Published 2024-11-25 leaderboard (full release, all questions)\n")
    ran = [CAT_LABEL[c] for c in cats]
    L.append(f"Same release and question pool; ours is a stratified random sample of it (95% CI from the "
             f"bootstrap), theirs is every question. `mean of ours` averages only the categories we ran "
             f"({', '.join(ran)}) so it lines up with our overall.\n")
    six = ["Reasoning", "Coding", "IF", "Mathematics", "Data Analysis", "Language"]
    L.append("| model | mean of ours | global (6 cat.) | " + " | ".join(six) + " |")
    L.append("|---|---|---|" + "---|" * len(six))
    for arm in arms:
        a = out["arms"][arm]
        byl = {CAT_LABEL[c]: a["categories"][c] for c in a["categories"]}
        L.append(f"| **ours: {arm}** | **{fmt(a['overall']['score'], a['overall']['ci95'])}** | -- | " +
                 " | ".join(fmt(byl[k]["score"], byl[k]["ci95"]) if k in byl else "--" for k in six) + " |")
    for label, v in sorted(out["published_2024_11_25"].items(), key=lambda kv: -(kv[1]["global"] or 0)):
        c = v["categories"]
        sub = [c[k] for k in ran if k in c]
        m = sum(sub) / len(sub) if sub and len(sub) == len(ran) else None
        L.append(f"| {label} | {fmt(m)} | {fmt(v['global'])} | " + " | ".join(fmt(c.get(k)) for k in six) + " |")
    L.append("")
    cl = out["current_leaderboard"]
    L.append(f"## C. Current leaderboard, release {cl['leaderboard_date']} -- NOT comparable\n")
    L.append("Different (unpublished) questions, different tasks per category (Coding = code_generation + "
             "code_completion of 2025-04-25; Reasoning adds theory_of_mind and logic_with_navigation; IF was "
             "rebuilt 2025-11-25), and an Agentic Coding category we did not run. Shown so the audience can place "
             "the base model: **Qwen3.8 27B** is the model ours is quantised from.\n")
    L.append("| model (release) | global | Reasoning | Coding | IF | Agentic Coding | Mathematics | Data Analysis | Language |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(cl["models"].items(), key=lambda kv: -(kv[1]["global"] or 0)):
        c = v["categories"]
        L.append(f"| {k} | {fmt(v['global'])} | " + " | ".join(fmt(c.get(x)) for x in ["Reasoning", "Coding", "IF", "Agentic Coding", "Mathematics", "Data Analysis", "Language"]) + " |")
    L.append(f"\nSources: {', '.join(cl['sources'] or [])}; https://livebench.ai/table_2024_11_25.csv; "
             "HF dataset livebench/model_judgment (split leaderboard).\n")
    with open(os.path.join(run_dir, "comparison.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
