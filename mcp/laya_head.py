#!/usr/bin/env python
"""Trained decision heads for Laya, and the frozen features they run on.

WHAT LAYA IS, WHICH DETERMINES WHAT CAN BE TRAINED
--------------------------------------------------
Read `laya/common.py::DecisionModel` before changing anything here. Laya is NOT
a sentence encoder with a similarity head, and it is NOT a fixed-class
classifier. It is an OPTION-SCORING CROSS-ENCODER:

    [CLS] <type> question: <instructions> [SEP]
    [MASK] opt0 [MASK] opt1 ... [SEP]
    <state> [SEP]

Every candidate answer is rendered into the input as text and given its own
[MASK] marker. `scorer` (LayerNorm -> Linear -> GELU -> Linear(d,1)) emits ONE
SCALAR per marker, and a softmax over markers is the answer distribution. The
label space is therefore open by construction -- that is exactly why it can be
prompted zero-shot with arbitrary options, and also why it has no per-task
decision boundary of its own to be good at.

That single fact sets the menu of things that can be trained:

  * The marker logits `z` are a k-vector per (state, question). An affine map
    on `z` (vector scaling) is a legitimate trained head and costs k*k+k
    parameters. It can fix a systematic preference for one option, which a
    scalar temperature CANNOT.
  * The post-head [CLS] state and the per-marker hidden states are frozen
    1024-d features. A linear head on those is a real classifier that can
    learn a boundary the prompt never expressed.
  * A scalar temperature only rescales confidence. It is monotonic, so it can
    move ECE but can NEVER move accuracy. Prior work in this repo stopped
    here, which is why accuracy never moved.

FEATURES ARE EXTRACTED AT BATCH SIZE 1, DELIBERATELY
----------------------------------------------------
The encoder runs under fp16 autocast. Padding a batch changes the fp16
reduction order, which perturbs `z` and the hidden states in the last decimal
places. That is harmless for a zero-shot argmax and NOT harmless for a trained
linear head, whose weights were fitted on one of the two versions. Extracting
at batch size 1 in both the trainer and the server makes train-time and
serve-time features bit-comparable. n is in the hundreds; the cost is seconds.

ARTEFACT FORMAT IS JSON, DELIBERATELY
-------------------------------------
Weights are small (at most k*1024 floats) and JSON is diffable, reviewable in a
PR, and carries no pickle execution risk. `index/laya/<task>.json` holds the
weights, the feature spec, the gate, the version and the held-out metrics it
achieved, so an artefact can always answer "how good was this when it shipped".
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

ARTEFACT_VERSION = 1

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "..", "index", "laya")


# --------------------------------------------------------------- tasks -----
#
# The canonical prompts. These are part of the artefact contract: a head is
# fitted on features produced by EXACTLY these prompts, so changing the wording
# invalidates every trained artefact. If you change one, bump PROMPT_REV and
# retrain -- the server refuses to load an artefact whose prompt_rev differs.

PROMPT_REV = 1

TASKS: Dict[str, Dict[str, Any]] = {
    "route_in": {
        "labels": ["investigate", "answer_directly", "clarify"],
        "question": {
            "type": "choice",
            "instructions": (
                "A developer asked this question of an assistant that has the "
                "repository's source code available to read. Decide how the "
                "question should be handled."
            ),
            "criteria": {
                "investigate": (
                    "answering requires reading this specific repository's own "
                    "source files, because the answer is a fact about this "
                    "codebase and exists nowhere else"
                ),
                "answer_directly": (
                    "this is general knowledge about a language, library, API or "
                    "technique and can be answered correctly without opening any "
                    "file in this repository"
                ),
                "clarify": (
                    "the question is too vague or underspecified to act on -- it "
                    "does not say what 'it' is, what is wrong, or what outcome is "
                    "wanted -- and needs a clarifying question first"
                ),
            },
        },
    },
    # The shippable grounding gate. Identical decision to "grounded", except
    # the state CARRIES the source it is judging, which turns an impossible
    # recall problem into a decidable entailment one. See
    # bench/laya_grounded_diagnostic.py for the measurement that forced this.
    "grounded_excerpt": {
        "labels": ["grounded", "generic"],
        "question": {
            "type": "choice",
            "instructions": (
                "Below is a finding about a source file, followed by the actual "
                "excerpt of that file. Decide whether the excerpt supports the "
                "finding or contradicts it."
            ),
            "criteria": {
                "grounded": (
                    "the excerpt supports the finding -- the identifiers, "
                    "values, control flow or structure the finding describes "
                    "are really present in the excerpt"
                ),
                "generic": (
                    "the excerpt does not support the finding -- the finding "
                    "asserts specifics that the excerpt contradicts, or says "
                    "only generic things that the excerpt cannot confirm"
                ),
            },
        },
    },
    "grounded": {
        "labels": ["grounded", "generic"],
        "question": {
            "type": "choice",
            "instructions": (
                "A finding claims to describe specific source files that were "
                "read. Decide whether the finding is actually grounded in those "
                "files or is general knowledge dressed up to look specific."
            ),
            "criteria": {
                "grounded": (
                    "the finding states concrete particulars that could only come "
                    "from reading the cited files -- real identifiers, values, "
                    "control flow or structure that belong to those files"
                ),
                "generic": (
                    "the finding is generic advice, restates what the library "
                    "does in general, or asserts file specifics that are "
                    "fabricated and do not follow from the cited files"
                ),
            },
        },
    },
}


def render_state(task: str, rec: Dict[str, Any]) -> str:
    """Turn a label record into the exact state string the model sees.

    Deterministic and shared by trainer and server: a head fitted on one
    rendering and served another would silently degrade.
    """
    if task == "route_in":
        q = (rec.get("question") or "").strip()
        ctx = (rec.get("context") or "").strip()
        return f"question: {q}\ncontext: {ctx}" if ctx else f"question: {q}"
    if task in ("grounded", "grounded_excerpt"):
        # Finding first, excerpt last: build_sequence truncates the state from
        # the right, so an over-long excerpt loses its tail rather than the
        # claim being judged.
        finding = (rec.get("finding") or rec.get("question") or "").strip()
        cites = rec.get("cites") or rec.get("context") or ""
        if isinstance(cites, (list, tuple)):
            cites = ", ".join(str(c) for c in cites)
        cites = str(cites).strip()
        excerpt = (rec.get("excerpt") or "").strip()
        parts = [f"finding: {finding}"]
        if cites:
            parts.append(f"cites: {cites}")
        if excerpt:
            parts.append(f"file excerpt:\n{excerpt}")
        return "\n".join(parts)
    raise ValueError(f"unknown task {task!r}")


def render_base_state(task: str, rec: Dict[str, Any]) -> Optional[str]:
    """The 'shared component' state to contrast against, or None.

    For grounding, the excerpt is long and is shared by a supported and a
    contradicted finding, so it dominates the pooled representation: measured,
    the two members of a pair sit at standardised cosine 0.73 while unrelated
    examples sit at 0.00. Running the excerpt ALONE and subtracting cancels
    that shared component and leaves what the finding contributed, which is
    where the label actually lives. See bench/laya_grounded_contrast.py.
    """
    if task == "grounded_excerpt":
        return render_state(task, {**rec, "finding": ""})
    return None


# ------------------------------------------------------ feature extract -----


class FeatureExtractor:
    """Runs the frozen Laya stack and returns every signal a head might use.

    Reimplements `DecisionModel.forward` rather than calling it, because the
    packaged forward returns only (logits, act_logits) and throws away the
    hidden states -- which are the features that actually let a head learn
    something the prompt did not already say.
    """

    def __init__(self, agent):
        self.agent = agent

    def _question(self, task: str):
        from laya.agent import Agent

        return Agent._to_internal(TASKS[task]["question"])

    def features(self, task: str, states: Sequence[str]) -> List[Dict[str, Any]]:
        import torch
        from laya.common import QTYPES, build_sequence, collate_items, render_options

        agent = self.agent
        model = agent.model
        q = self._question(task)
        k_expect = len(render_options(q))
        max_len = agent.cfg.get("max_len", 512)
        head_max_len = agent.cfg.get("head_max_len", 192)
        use_amp = agent.device.type == "cuda"

        out: List[Dict[str, Any]] = []
        for st in states:
            seq, markers = build_sequence(agent.tok, st, q, max_len, head_max_len)
            if len(markers) != k_expect:
                raise ValueError(
                    f"task {task!r}: options exceed head_max_len={head_max_len}"
                )
            item = {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}
            # batch size 1 on purpose -- see module docstring
            b = collate_items([[item]], agent.tok.pad_token_id)
            with torch.no_grad():
                with torch.autocast(
                    device_type=agent.device.type, dtype=agent.dtype, enabled=use_amp
                ):
                    ids = b["input_ids"].to(agent.device)
                    att = b["attention_mask"].to(agent.device)
                    mpos = b["marker_pos"].to(agent.device)
                    mmask = b["marker_mask"].to(agent.device)
                    qtype = b["qtype"].to(agent.device)

                    h = model.encoder(
                        input_ids=ids, attention_mask=att
                    ).last_hidden_state
                    h = h + model.type_emb(qtype)[:, None, :]
                    if model.head is not None:
                        pad = ~att.bool()
                        for layer in model.head.layers:
                            h = layer(h, src_key_padding_mask=pad)
                    idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
                    m = torch.gather(h, 1, idx)
                    logits = model.scorer(m).squeeze(-1).float()
                    logits = logits.masked_fill(~mmask, -1e4)

                    p = torch.softmax(logits.detach(), -1)
                    kk = mmask.sum(-1).clamp(min=2).float()
                    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(kk)
                    top2 = p.topk(2, -1).values
                    feats = torch.stack(
                        [top2[:, 0], top2[:, 0] - top2[:, 1], ent, kk / 255.0], -1
                    )
                    pooled = h[:, 0].float()
                    act_logits = model.act_head(torch.cat([pooled, feats], -1))

            act = torch.softmax(act_logits.float(), -1)
            out.append(
                {
                    "logits": logits[0, :k_expect].float().cpu().tolist(),
                    "cls": pooled[0].float().cpu().tolist(),
                    "markers": m[0, :k_expect].float().cpu().tolist(),
                    "act_probability": float(act[0, 0]),
                }
            )
        return out

    def record(self, task: str, rec: Dict[str, Any],
               contrast: bool = True) -> Dict[str, Any]:
        """One complete feature record for one labelled/unlabelled example.

        Shared by the trainer and the server so a head can never be fitted on
        one rendering and served another. `contrast` is skipped when the head
        does not need it, saving the second forward pass.
        """
        state = render_state(task, rec)
        base = render_base_state(task, rec) if contrast else None
        if base is None:
            return self.features(task, [state])[0]
        f, g = self.features(task, [state, base])
        f["cls_base"] = g["cls"]
        f["logits_base"] = g["logits"]
        return f


# ------------------------------------------------------------- heads --------
#
# Pure python/torch-free maths at inference time. A head is a small affine map
# plus a softmax, so numpy is not even needed to serve it; that keeps the
# server's import surface identical to what it is today.


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


FEATURE_SPECS = ("logits", "cls", "cls_logits", "markers",
                 "delta_cls", "delta_cls_logits")

# Specs needing a contrast pass. A head using one of these costs two forward
# passes per decision instead of one.
CONTRAST_SPECS = ("delta_cls", "delta_cls_logits")


def build_vector(spec: str, f: Dict[str, Any]) -> List[float]:
    """Assemble the input vector for a head from a cached feature record."""
    if spec == "logits":
        return list(f["logits"])
    if spec == "cls":
        return list(f["cls"])
    if spec == "cls_logits":
        return list(f["cls"]) + list(f["logits"])
    if spec == "markers":
        v: List[float] = []
        for row in f["markers"]:
            v.extend(row)
        return v
    if spec in CONTRAST_SPECS:
        if "cls_base" not in f:
            raise ValueError(
                f"spec {spec!r} needs a contrast pass; feature record has no "
                f"'cls_base'. Extract with render_base_state()."
            )
        d = [a - b for a, b in zip(f["cls"], f["cls_base"])]
        if spec == "delta_cls":
            return d
        return d + [a - b for a, b in zip(f["logits"], f["logits_base"])]
    raise ValueError(f"unknown feature spec {spec!r}")


class TrainedHead:
    """A fitted linear head: probs = softmax(W x + b), plus an abstain gate."""

    def __init__(self, blob: Dict[str, Any]):
        self.blob = blob
        self.task: str = blob["task"]
        self.labels: List[str] = blob["labels"]
        self.feature_spec: str = blob["feature_spec"]
        self.kind: str = blob.get("kind", "linear")
        self.W: List[List[float]] = blob["W"]
        self.b: List[float] = blob["b"]
        self.gate: float = float(blob.get("gate", 0.0))
        self.version: int = int(blob.get("artefact_version", ARTEFACT_VERSION))
        self.prompt_rev: int = int(blob.get("prompt_rev", PROMPT_REV))
        self.metrics: Dict[str, Any] = blob.get("metrics", {})
        if self.prompt_rev != PROMPT_REV:
            raise ValueError(
                f"artefact for {self.task!r} was fitted on prompt_rev "
                f"{self.prompt_rev}, server is at {PROMPT_REV}. Retrain: "
                f"scripts/train_laya.py --task {self.task}"
            )

    # -- persistence ------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "TrainedHead":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def load_task(cls, task: str, index_dir: Optional[str] = None
                  ) -> Optional["TrainedHead"]:
        d = index_dir or INDEX_DIR
        p = os.path.join(d, f"{task}.json")
        if not os.path.exists(p):
            return None
        return cls.load(p)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.blob, fh, indent=2)

    # -- inference --------------------------------------------------------
    def probabilities(self, f: Dict[str, Any]) -> Dict[str, float]:
        x = build_vector(self.feature_spec, f)
        if len(x) != len(self.W[0]):
            raise ValueError(
                f"feature width {len(x)} != head width {len(self.W[0])}"
            )
        z = [sum(w_i * x_i for w_i, x_i in zip(row, x)) + bb
             for row, bb in zip(self.W, self.b)]
        p = _softmax(z)
        return {lab: round(v, 6) for lab, v in zip(self.labels, p)}

    def decide(self, f: Dict[str, Any]) -> Dict[str, Any]:
        probs = self.probabilities(f)
        order = sorted(probs.items(), key=lambda kv: -kv[1])
        top, second = order[0], (order[1] if len(order) > 1 else (None, 0.0))
        margin = top[1] - second[1]
        return {
            "type": "choice",
            "choice": top[0],
            "probabilities": probs,
            "margin": round(margin, 6),
            "abstain": bool(margin < self.gate),
            "gate": self.gate,
            "engine": "trained",
            "artefact_version": self.version,
            "trained_metrics": self.metrics,
        }
