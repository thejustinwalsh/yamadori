#!/usr/bin/env python
"""Generate grounded/fabricated pairs that CARRY the source they talk about.

    .venv-laya/Scripts/python.exe bench/make_grounded_excerpts.py

WHY THIS FILE EXISTS
--------------------
bench/laya_grounded_diagnostic.py shows that telling a true file-specific
finding from a fabricated one is at chance when the state names a file but
never shows it. That is not a model limitation; the deciding information is
absent from the input. "MAX_TRACES is 256" and "MAX_TRACES is 1024" are the
same sentence to any classifier that cannot see shomen.py.

So this builds the same task as an ENTAILMENT problem: each example carries a
real excerpt, extracted live from a real file in this repo, and the finding is
either supported by that excerpt or contradicted by it. Now the answer is a
function of the input and a head can actually learn it.

Excerpts are pulled from source at generation time rather than pasted, so they
cannot drift out of sync with the files. If a function is renamed the
generator fails loudly instead of emitting a stale excerpt.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "laya_grounded_excerpt_labels.jsonl")

VENV_LAYA = os.path.join(ROOT, ".venv-laya", "Lib", "site-packages", "laya")

# (path, function name, grounded finding, fabricated finding)
CASES: List[Tuple[str, str, str, str]] = [
    (
        os.path.join(ROOT, "mcp", "laya_service.py"), "agent",
        "The model is lazily constructed behind a double-checked lock, so the "
        "first /decide call pays the load cost and later calls do not.",
        "The model is eagerly constructed at import time and the lock is only "
        "held for the duration of a prediction.",
    ),
    (
        os.path.join(ROOT, "mcp", "laya_service.py"), "decide",
        "decide() raises ValueError when 'state' is missing or 'questions' is "
        "not a non-empty dict, and it holds a lock across the predict call.",
        "decide() silently substitutes an empty string when 'state' is missing "
        "and runs predictions without any locking.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "render_options",
        "For a noul question the options are always rendered in the order "
        "false then true, with defaults used when no criteria are supplied.",
        "For a noul question the options are rendered true then false, and a "
        "missing criteria dict raises a KeyError.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "build_sequence",
        "Each option is prefixed with the tokenizer's mask token id and its "
        "own token ids are truncated to at most 48 tokens.",
        "Each option is prefixed with the CLS token id and options are never "
        "truncated, which is why long rubrics are safe.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "ece_score",
        "ece_score buckets by confidence into 15 bins by default and weights "
        "each bin's gap by the fraction of samples that fall in it.",
        "ece_score uses 10 equal-count bins by default and returns the maximum "
        "bin gap rather than a weighted mean.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "confidence_from_probs",
        "Confidence is one minus the Shannon entropy normalised by log(k), and "
        "it returns exactly 1.0 when k is below 2.",
        "Confidence is the gap between the top two probabilities, and it "
        "returns 0.0 when k is below 2.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "temp_bucket",
        "The bucket name joins the question type with an option-count band of "
        "2, 3-5, 6-10 or 11+.",
        "The bucket name joins the question type with the exact option count, "
        "so every distinct k gets its own temperature.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "serialize_state",
        "A string state is returned unchanged; anything else is dumped to JSON "
        "with ensure_ascii disabled.",
        "Every state is dumped to JSON, so a plain string comes back wrapped "
        "in quotes.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "collate_items",
        "Sequences are padded to the longest item in the batch with pad_id and "
        "a boolean marker_mask marks the real marker slots.",
        "Sequences are padded to a fixed 512 length and the marker mask is an "
        "integer count rather than a boolean tensor.",
    ),
    (
        os.path.join(VENV_LAYA, "agent.py"), "_fix_tokenizer_config",
        "It rewrites tokenizer_class to PreTrainedTokenizerFast and converts a "
        "list-valued extra_special_tokens into a mapping, swallowing errors.",
        "It deletes tokenizer_config.json entirely and forces a re-download of "
        "the tokenizer from the hub.",
    ),
    (
        os.path.join(VENV_LAYA, "agent.py"), "_verify_compatibility",
        "It requires the keys 'encoder' and 'head_layers' in the config and "
        "checks that encoder., type_emb., scorer. and act_head. prefixes exist.",
        "It only checks that the safetensors file is readable and logs a "
        "warning when parameter shapes disagree.",
    ),
    (
        os.path.join(ROOT, "scripts", "calibrate_laya.py"), "apply_temp",
        "The probability is clamped away from 0 and 1, converted to a logit, "
        "divided by the temperature, then passed back through a sigmoid.",
        "The probability is multiplied by the temperature directly and then "
        "renormalised, with no logit transform involved.",
    ),
    (
        os.path.join(ROOT, "scripts", "calibrate_laya.py"), "nll",
        "It returns the mean negative log likelihood, applying the temperature "
        "to each probability before clamping it.",
        "It returns the summed squared error between probability and label, "
        "without applying any temperature.",
    ),
    (
        os.path.join(ROOT, "scripts", "train_laya.py"), "best_gate",
        "The gate is swept from 0 up to 0.75 in steps of 0.05 and scored by a "
        "utility of plus one for correct and minus one for wrong answers.",
        "The gate is found by binary search on the ROC curve and chosen to hold "
        "the false positive rate below five percent.",
    ),
    (
        os.path.join(ROOT, "scripts", "train_laya.py"), "stratified_split",
        "Indices are grouped by class, shuffled with a seeded Random, and at "
        "least one example per class always goes to the test side.",
        "The split is a plain random 70/30 cut over all rows, so a rare class "
        "can end up entirely in train.",
    ),
    (
        os.path.join(ROOT, "mcp", "laya_head.py"), "build_vector",
        "The markers spec flattens the per-option hidden states into one long "
        "vector, and cls_logits concatenates the pooled state with the logits.",
        "The markers spec averages the per-option hidden states into a single "
        "1024-d vector, and cls_logits multiplies them elementwise.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "render_criterion",
        "A string criterion is returned unchanged, and anything else is dumped "
        "to compact JSON with a default= fallback for unserialisable values.",
        "Every criterion is coerced with str(), which is why dict-valued "
        "criteria leak a Python repr into the prompt.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "amp_dtype",
        "It returns bfloat16 only when the name is exactly 'bf16', and float16 "
        "for everything else including None.",
        "It inspects the current CUDA device capability and returns bfloat16 on "
        "Ampere or newer, float16 otherwise.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "proper_reward",
        "The reward adds a weighted spherical score to the log score, and the "
        "ranked probability term is subtracted only for score-type questions.",
        "The reward is a plain cross entropy, and the ranked probability score "
        "is applied uniformly to all three question types.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "td_lambda_targets",
        "If the batch has no 'ep_group' key the targets are returned unchanged, "
        "so single-turn data passes straight through.",
        "If the batch has no 'ep_group' key it raises a KeyError, so every "
        "batch must carry episode grouping.",
    ),
    (
        os.path.join(VENV_LAYA, "common.py"), "build_model",
        "When an encoder directory exists the encoder is built from config "
        "alone, avoiding a download of pretrained weights that are about to be "
        "overwritten by the checkpoint.",
        "The encoder is always downloaded with from_pretrained, and the "
        "encoder_dir argument only selects a cache location.",
    ),
    (
        os.path.join(ROOT, "mcp", "laya_service.py"), "health",
        "The health payload reports the model id and whether the agent has "
        "been loaded yet, plus the websocket URL.",
        "The health endpoint performs a live one-token inference and reports "
        "the measured latency in milliseconds.",
    ),
    (
        os.path.join(ROOT, "mcp", "laya_service.py"), "decide_route",
        "The blocking decide() call is pushed onto a thread, a ValueError "
        "becomes a 400 carrying the schema, and anything else becomes a 500.",
        "The decide() call runs inline on the event loop and every failure is "
        "returned as a 200 with an 'error' field.",
    ),
    (
        os.path.join(ROOT, "mcp", "laya_head.py"), "_softmax",
        "The maximum is subtracted before exponentiating, which is the standard "
        "guard against overflow.",
        "The values are exponentiated directly and then divided by the count "
        "rather than the sum.",
    ),
    (
        os.path.join(ROOT, "scripts", "train_laya.py"), "quantiles",
        "It returns min, p25, median, p75, max and mean, and an empty input "
        "yields an empty dict rather than raising.",
        "It returns only the mean and standard deviation, and raises "
        "ZeroDivisionError on an empty input.",
    ),
    (
        os.path.join(ROOT, "scripts", "train_laya.py"), "predict_probs",
        "Each row computes a dot product plus bias, then a max-subtracted "
        "softmax, returning one probability list per input vector.",
        "It returns raw logits without any softmax, leaving normalisation to "
        "the caller.",
    ),
    (
        os.path.join(ROOT, "scripts", "train_laya.py"), "_fit_centroid",
        "Class centroids become the weight rows divided by the scale, and the "
        "bias is minus half the squared centroid norm over the same scale.",
        "The centroids are L2-normalised and the bias is fixed at zero, making "
        "the head a pure cosine similarity.",
    ),
    (
        os.path.join(ROOT, "scripts", "train_laya.py"), "evaluate",
        "Margin is the top probability minus the runner-up, and utility counts "
        "plus one for each correct answered case and minus one for each wrong "
        "one.",
        "Margin is the entropy of the distribution, and utility is the mean "
        "log likelihood over answered cases.",
    ),
    (
        os.path.join(ROOT, "scripts", "calibrate_laya.py"), "load_pairs",
        "Rows are grouped by their 'pair' key and only groups of exactly two "
        "members are returned.",
        "Rows are grouped by file path and any group with at least two members "
        "is returned.",
    ),
    (
        os.path.join(HERE, "make_grounded_excerpts.py"), "extract",
        "The function is located by a regex allowing an optional async prefix, "
        "and its body ends at the first non-blank line that is not indented "
        "past the def.",
        "The function is located by parsing the file with ast and its body is "
        "recovered from the node's end_lineno.",
    ),
]


def extract(path: str, name: str) -> str:
    """Pull one top-level def out of a file, by indentation."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    start = None
    pat = re.compile(r"^(\s*)(?:async\s+)?def\s+" + re.escape(name) + r"\s*\(")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            start, indent = i, len(m.group(1))
            break
    if start is None:
        raise SystemExit(f"{path}: no def {name}()")
    body = [lines[start]]
    for ln in lines[start + 1:]:
        if ln.strip() and not ln.startswith(" " * (indent + 1)):
            break
        body.append(ln)
    text = "".join(body).rstrip()
    # Keep the excerpt inside the model's 512-token window; the instruction
    # head and options already consume head_max_len.
    return text[:1100]


def main() -> None:
    rows = []
    for path, name, good, bad in CASES:
        src = extract(path, name)
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        # Both members carry the SAME excerpt, so they must never straddle a
        # train/test split -- a head that memorised one would have effectively
        # seen the other. `pair` is what the trainer groups on.
        rows.append({"pair": name, "finding": good, "cites": rel,
                     "excerpt": src, "label": "grounded",
                     "why": f"Supported by the body of {name}()."})
        rows.append({"pair": name, "finding": bad, "cites": rel,
                     "excerpt": src, "label": "generic",
                     "why": f"Contradicted by the body of {name}()."})
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows ({len(CASES)} pairs) -> {OUT}")


if __name__ == "__main__":
    main()
