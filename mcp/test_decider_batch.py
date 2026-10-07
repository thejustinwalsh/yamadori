#!/usr/bin/env python
"""Batched jjava reads (mcp/decider_batch.py; docs/DECIDE-BATCH.md section 3).
No GPU, no network, no stack port: the model server is a FAKE passed in
(post=, upstream=) or patched behind decider_bonsai's two doors.

    python mcp/test_decider_batch.py      -> "N/M checks passed"

THE FAKE ENGINE implements what the batch path reaches: /apply-template (a
small Qwen-like template, and the REAL served template through jinja2 on
mcp/fixtures/bonsai_chat_template.jinja), /tokenize (a toy tokenizer that
splits the special tokens and then words, so the boundary check means
something, plus a variant that MERGES across the boundary), the sequential
/v1/chat/completions route (next-token log-probs as a deterministic function
of the FULL rendered prompt text and the token id, a proper softmax over a
vocabulary; a checkpoint at the last user message for the cache, as the
hybrid model keeps it), and POST /decide-batch (the same function over
prefix + block; groups, token_ids, top_logprobs, keep_prefix, release,
verify_tokenization; every request recorded).

WHAT THIS GATES
  1. THE DERIVATION: wrapper pieces derived once per role structure through
     /apply-template with sentinels, the prefix cut at a message boundary,
     verified three ways; the substituted rendering equals the served
     template's byte for byte (small template, the real one, a choose-shaped
     structure with a mid-conversation assistant turn); no per-question
     /apply-template call; a template that is not wrapper text, one that
     ignores add_generation_prompt, a tokenizer that merges across the
     boundary and an engine that reports a bad boundary all DISABLE batching
     for that structure (cached) and the answers still come, per read.
  2. IDENTITY: the batched answers equal the per-read answers (noul, choice,
     score, digits, 26 options; one and many questions; a custom render;
     exclusions; a content-free prior question falling back alone), in ONE
     /decide-batch request and zero per-read requests; the spelling mass the
     per-read path leaves unread (below its top K) is within its
     `unread_bound`.
  3. THE SWITCH: off, the requests, bodies and answers are the per-read
     path's, with zero /decide-batch and /apply-template calls.
  4. JEV AND THE TURN: a Jev call's single plans in one request, a two-stage
     plan's chunks in one and its final read in a second on the kept prefix,
     then released; usage equals the per-read path's (cold, warm); the Turn
     keeps the prefix across its decide() calls and frees it at close.
  5. FALLBACK: 404 / 501 / 503 / 400, a timeout, an unreachable server,
     garbage JSON, a body that is not the contract's shape, a missing label
     logprob -- each answered per read with the same answers, a cached
     negative for 60 s on an injectable clock, one retry after it; release
     on an exception; the think-block guard unchanged.
  6. THE MEASUREMENT DRIVER's plan builder (bench/decider/batch_ab.py).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import socket
import sys
import traceback
import urllib.error
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_decider_batch_")
os.environ["YAMADORI_ACCOUNTS_DIR"] = os.path.join(_TMP, "accounts")
os.environ["YAMADORI_DECIDER_MODELS_DIR"] = os.path.join(_TMP, "decider_models")
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ["YAMADORI_SLOTS"] = "3"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
for _v in ("YAMADORI_TIER_MODELS", "YAMADORI_MAX_MODEL", "YAMADORI_DECIDER",
           "YAMADORI_DECIDER_BATCH"):
    os.environ.pop(_v, None)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "decider"))

import budget  # noqa: E402
import decide_turn as T  # noqa: E402
import decider_batch as B  # noqa: E402
import decider_bonsai as D  # noqa: E402
import jev_api as J  # noqa: E402
import model as M  # noqa: E402

budget._POOL = 131072          # never /props
CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail=None) -> bool:
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:600])
    return bool(ok)


# The release door: a fake that records the slot, never a model server.
RELEASED: list = []
D.release = lambda slot, why="": RELEASED.append(slot) or {      # noqa: E731
    "released": True, "slot": slot}
M.release_slot = lambda slot, model=None, timeout=None: {         # noqa: E731
    "ok": True, "cells_before": 0, "ms": 0, "method": "fake"}


def batch(on: bool) -> None:
    if on:
        os.environ["YAMADORI_DECIDER_BATCH"] = "1"
    else:
        os.environ.pop("YAMADORI_DECIDER_BATCH", None)


# ================================================================ THE FAKE ====
SPECIAL = {"<|im_start|>": 5, "<|im_end|>": 6}
_SPLIT = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")
_MERGE_SPLIT = re.compile(r"(\n<\|im_start\|>|<\|im_start\|>|<\|im_end\|>)")
_WORD = re.compile(r"\s?\S+|\s+")
BOS = 1
LABEL_IDS: dict[str, int] = {}
for _i, _L in enumerate(D.LETTERS):
    LABEL_IDS[_L] = 100 + _i
    LABEL_IDS[" " + _L] = 200 + _i
for _d in "0123456789":
    LABEL_IDS[_d] = 300 + int(_d)
    LABEL_IDS[" " + _d] = 310 + int(_d)
ID_TOKEN = {v: k for k, v in LABEL_IDS.items()}
FILL = [2000 + i for i in range(100)]
for _f in FILL:
    ID_TOKEN[_f] = f"<f{_f}>"


def _u(text: str, tid: int) -> float:
    h = hashlib.sha256(f"{text}\0{tid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def tid_of(word: str) -> int:
    return LABEL_IDS.get(word) or 10000 + zlib.crc32(word.encode()) % 50000


def _served_template():
    import jinja2
    env = jinja2.Environment()
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    with open(os.path.join(HERE, "fixtures", "bonsai_chat_template.jinja"),
              encoding="utf-8") as f:
        return env.from_string(f.read())


def served_continuation(tpl, msgs: list[dict], **kw) -> str:
    """llama-server's rendering of a body that ends on an assistant message
    (mcp/test_decider_bonsai.py served_continuation, the same derivation)."""
    head = tpl.render(messages=msgs[:-1], add_generation_prompt=False, **kw)
    gen = tpl.render(messages=msgs[:-1], add_generation_prompt=True,
                     **kw)[len(head):]
    probe = tpl.render(messages=[{"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "CCC",
                                  "reasoning_content": "RRR"}],
                       add_generation_prompt=False)
    i = probe.rindex("<think>")
    start = probe[i:probe.index("RRR", i)]
    end = probe[probe.index("RRR", i) + 3:probe.index("CCC", i)]
    last = msgs[-1]
    reasoning = last.get("reasoning_content") or ""
    content = last.get("content") or ""
    mode = "reasoning" if reasoning and not content else "content"
    out = head + gen[:gen.find(D.THINK_START)] + start + reasoning
    if mode == "content":
        out += end + content
    return out


_REAL = None


def _real_template():
    global _REAL
    if _REAL is None:
        _REAL = _served_template()
    return _REAL


def http_error(code: int, msg: str = "x") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://fake/decide-batch", code, msg, {},
        io.BytesIO(json.dumps({"error": {"message": msg, "code": code}})
                   .encode()))


class Engine:
    """The model server behind decider_bonsai's two doors (post, upstream)."""

    def __init__(self, template: str = "small", *, sparse: bool = False,
                 bare_zero: bool = False, merge: bool = False,
                 share: bool = True, bad_apply: str | None = None,
                 engine_tok_bad: bool = False):
        self.template, self.sparse, self.merge = template, sparse, merge
        self.bare_zero = bare_zero
        self.share, self.bad_apply = share, bad_apply
        self.engine_tok_bad = engine_tok_bad
        self.batch_mode = None          # failure injection for /decide-batch
        self.paths: dict[str, int] = {}
        self.chat_bodies: list[dict] = []
        self.batch_requests: list[dict] = []
        self.apply_payloads: list[dict] = []
        self.slot_prev: dict = {}
        self.resident: str | None = None
        self.releases = 0

    # ---- the template
    def render(self, msgs: list[dict], gen: bool = True) -> str:
        if self.template == "real":
            tpl = _real_template()
            if msgs[-1]["role"] == "assistant":
                return served_continuation(tpl, msgs, enable_thinking=False)
            return tpl.render(messages=msgs, add_generation_prompt=gen,
                              enable_thinking=False)
        cont = msgs[-1]["role"] == "assistant"
        body = msgs[:-1] if cont else msgs
        out = ""
        for m in body:
            c = (m.get("content") or "").strip(" \t\n\v\f\r")
            if self.bad_apply == "length" and m["role"] == "user":
                c += f"[{len(c)}]"
            out += f"<|im_start|>{m['role']}\n{c}<|im_end|>\n"
        if cont:
            out += ("<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    + msgs[-1]["content"])
        elif gen or self.bad_apply == "gen":
            out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return out

    # ---- the tokenizer
    def tok(self, text: str, special: bool) -> list[int]:
        out = [BOS] if special else []
        pat = _MERGE_SPLIT if self.merge else _SPLIT
        for part in pat.split(text):
            if not part:
                continue
            if part in SPECIAL:
                out.append(SPECIAL[part])
            elif part == "\n<|im_start|>":
                out.append(7)
            else:
                out += [tid_of(w) for w in _WORD.findall(part)]
        return out

    # ---- the model
    def dist(self, text: str) -> dict[int, float]:
        """The next-token distribution after `text`: a softmax over the
        vocabulary, a function of the whole text and the token id. The
        printed option labels' spellings dominate (the bare spellings too,
        unless `sparse`)."""
        block = ""
        for m in re.finditer(r"(?:OPTIONS|LEVELS):\n(.*?)(?:\n\n|$)", text,
                             re.S):
            block = m.group(1)
        printed = set(re.findall(r"^([A-Z0-9]+)\. ", block, re.M))
        logits: dict[int, float] = {}
        for tid in FILL:
            logits[tid] = -3.0 * _u(text, tid)
        for sp, tid in LABEL_IDS.items():
            lab = sp.strip()
            spaced = sp.startswith(" ")
            boosted = lab in printed and (spaced or not (self.sparse
                                                         or self.bare_zero))
            if boosted:
                logits[tid] = 8.0 + _u(text, tid)
            elif self.bare_zero and not spaced:
                logits[tid] = -60.0
            else:
                logits[tid] = -3.0 * _u(text, tid)
        m = max(logits.values())
        z = sum(math.exp(v - m) for v in logits.values())
        return {t: v - m - math.log(z) for t, v in logits.items()}

    @staticmethod
    def top(d: dict, k: int) -> list[dict]:
        rows = sorted(d.items(), key=lambda kv: -kv[1])[:k]
        return [{"id": t, "token": ID_TOKEN.get(t, "?"), "logprob": lp}
                for t, lp in rows]

    # ---- the routes
    def upstream(self, path, payload=None, timeout=30):
        self.paths[path] = self.paths.get(path, 0) + 1
        if path == "/tokenize":
            return {"tokens": self.tok(payload["content"],
                                       bool(payload.get("add_special")))}
        if path == "/apply-template":
            self.apply_payloads.append(payload)
            gen = payload.get("add_generation_prompt", True)
            return {"prompt": self.render(payload["messages"], gen)}
        if path == "/slots":
            return []
        if path == "/decide-batch":
            return self.decide_batch(payload)
        raise AssertionError(path)

    def post(self, body, timeout):
        self.paths["/v1/chat/completions"] = self.paths.get(
            "/v1/chat/completions", 0) + 1
        self.chat_bodies.append(body)
        prompt = self.render(body["messages"])
        ids = self.tok(prompt, True)
        reuse = 0
        if body.get("cache_prompt", True):
            segs = prompt.split("<|im_start|>")
            prev = self.slot_prev.get(body.get("id_slot"))
            users = [i for i, s in enumerate(segs) if s.startswith("user\n")]
            limit = users[-1] if users else len(segs)
            n = 0
            while prev and n < min(limit, len(prev)) and segs[n] == prev[n]:
                n += 1
            if n:
                reuse = len(self.tok("<|im_start|>".join(segs[:n]), True))
            self.slot_prev[body.get("id_slot")] = segs
        d = self.dist(prompt)
        top = self.top(d, int(body["top_logprobs"]))
        return {"choices": [{"logprobs": {"content": [
            {"id": top[0]["id"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}}],
                "usage": {"prompt_tokens": len(ids)},
                "timings": {"prompt_n": len(ids) - reuse, "cache_n": reuse,
                            "prompt_ms": 1.0}}

    def decide_batch(self, p: dict):
        self.batch_requests.append(p)
        mode = self.batch_mode
        if isinstance(mode, tuple) and mode[0] == "http":
            raise http_error(mode[1])
        if mode == "timeout":
            raise socket.timeout("timed out")
        if mode == "unreachable":
            raise urllib.error.URLError(ConnectionRefusedError("refused"))
        if mode == "garbage":
            raise json.JSONDecodeError("Expecting value", "<html>", 0)
        if mode == "badshape":
            return {"nope": 1}
        if p.get("release"):
            self.resident = None
            self.releases += 1
            return {"released": True}
        prefix = p["prefix"]
        ptoks = self.tok(prefix, True)
        reused = len(ptoks) if self.resident == prefix else 0
        out_groups, total, saved = [], len(ptoks) - reused, 0
        for blocks in p["groups"]:
            toks = [self.tok(b, False) for b in blocks]
            head = 0
            if self.share and len(blocks) > 1:
                while all(len(t) > head for t in toks) and len(
                        {t[head] for t in toks}) == 1:
                    head += 1
            total += head
            saved += head * (len(blocks) - 1)
            g = []
            for bi, (b, t) in enumerate(zip(blocks, toks)):
                d = self.dist(prefix + b)
                blk = {"tokens": len(t), "shared": head,
                       "processed": len(t) - head + (head if bi == 0 else 0),  # as the engine: head charged to the first block
                       "top_logprobs": self.top(d, int(p["top_logprobs"])),
                       "logprobs": {str(i): d[i] for i in p["token_ids"]}}
                if mode == "missing_logprob" and blk["logprobs"]:
                    # drop one requested id that the top list does not carry
                    top_ids = {e["id"] for e in blk["top_logprobs"]}
                    for i in list(blk["logprobs"]):
                        if int(i) not in top_ids:
                            del blk["logprobs"][i]
                if p.get("verify_tokenization"):
                    blk["tokenization_ok"] = (not self.engine_tok_bad) and (
                        ptoks + t == self.tok(prefix + b, True))
                total += len(t) - head
                g.append(blk)
            out_groups.append(g)
        keep = bool(p.get("keep_prefix"))
        self.resident = prefix if keep else None
        return {"model": "fake",
                "prefix": {"tokens": len(ptoks), "reused": reused,
                           "processed": len(ptoks) - reused,
                           "resident": keep},
                "groups": out_groups,
                "timings": {"total_ms": 1.0, "prefix_ms": 0.5,
                            "shared_ms": 0.1, "blocks_ms": 0.4, "waves": 1,
                            "decode_calls": 1, "processed_tokens": total,
                            "shared_tokens_saved": saved},
                "seqs": {"first": 0, "forks": 2}}


# ============================================================ the questions ===
STATE = ("Customer: my payout failed three times this week.\n"
         "Support: have you tried a different card?\n"
         "Customer: yes, same error.   \n")


def questions() -> list[dict]:
    return [D.q_noul("n1", "Is the material about payments?"),
            D.q_noul("n2", "Does it convey urgency?",
                     criteria={"true": "time-sensitive",
                               "false": "no urgency"}),
            D.q_choice("c1", "Which team owns it?",
                       ["Payments", "Tech", "Sales", "Other"],
                       keys=["pay", "tech", "sales", "other"], none=3),
            D.q_score("s1", "How angry is the customer?",
                      ["Calm", "Annoyed", "Frustrated", "Furious", "Livid"]),
            D.q_score("s2", "How resolved is it?", ["no", "half", "yes"],
                      label_kind="digits")]


VOLATILE_DIAG = {"ms", "read_path", "batch", "batch_fallback", "read_regime",
                 "prompt_tokens", "processed_tokens"}
VOLATILE_ORDER = {"ms", "prompt_tokens", "processed_tokens", "cached_tokens",
                  "http_reads", "processed_all"}


def norm(a: dict) -> str:
    """An answer without what legitimately differs between the paths (the
    clock, the path, the token accounting)."""
    a = json.loads(json.dumps(a))
    a.pop("decision_id", None)
    dg = a["diagnostics"]
    for k in VOLATILE_DIAG:
        dg.pop(k, None)
    dg["orders"] = [{k: v for k, v in o.items() if k not in VOLATILE_ORDER}
                    for o in dg["orders"]]
    return json.dumps(a, sort_keys=True)


def fresh(**kw) -> Engine:
    D._SPELL_IDS.clear()
    B.reset()
    return Engine(**kw)


def seq_answers(eng: Engine, qs, state=STATE, slot=3, **kw) -> list[dict]:
    return [D.read(state, q, slot=slot, post=eng.post, upstream=eng.upstream,
                   **kw) for q in qs]


def batch_answers(eng: Engine, qs, state=STATE, **kw) -> list[dict]:
    batch(True)
    try:
        return B.read_many(state, qs, slot=3, post=eng.post,
                           upstream=eng.upstream, **kw)
    finally:
        batch(False)


def same(a: list[dict], b: list[dict]) -> bool:
    return len(a) == len(b) and all(norm(x) == norm(y) for x, y in zip(a, b))


# ================================================================ derivation ===
def derivation() -> None:
    for tname in ("small", "real"):
        eng = fresh(template=tname)
        qs = questions()
        c = D.rendered_orders(qs[2])[0]
        msgs = D.messages(STATE, c)
        st = B.structure(msgs, eng.upstream, "bonsai")
        contents = [m["content"] for m in msgs]
        want = eng.render(msgs)
        check(f"[derive:{tname}] the substituted rendering equals the "
              "served template's byte for byte (prefix + block)",
              st.prefix(contents) + st.block(contents) == want,
              (st.prefix(contents)[-40:], want[-60:]))
        check(f"[derive:{tname}] the prefix is the system and state messages "
              "cut at the message boundary (ends on the end token and the "
              "newline), the block starts at the next message's start token",
              st.prefix(contents).endswith("<|im_end|>\n")
              and st.block(contents).startswith("<|im_start|>user\n")
              and st.block(contents).endswith("Answer:"),
              (st.prefix(contents)[-20:], st.block(contents)[:30]))
        n_apply = eng.paths.get("/apply-template", 0)
        n_tok = eng.paths.get("/tokenize", 0)
        check(f"[derive:{tname}] derived with 3 /apply-template requests and "
              "3 /tokenize requests, once for the structure",
              n_apply == 3 and n_tok == 3, (n_apply, n_tok))
        for q in qs:
            for o in D.rendered_orders(q):
                m2 = D.messages(STATE, o)
                s2 = B.structure(m2, eng.upstream, "bonsai")
                cs = [m["content"] for m in m2]
                assert s2.prefix(cs) + s2.block(cs) == eng.render(m2)
        check(f"[derive:{tname}] ten more question renderings: no further "
              "/apply-template or /tokenize request for the structure",
              eng.paths.get("/apply-template", 0) == n_apply
              and eng.paths.get("/tokenize", 0) == n_tok,
              eng.paths)
        # the choose-shaped structure: a custom message list with a
        # mid-conversation assistant turn
        hist = [{"role": "system", "content": D.SYSTEM},
                {"role": "user", "content": D.STATE_HEAD + STATE},
                {"role": "user", "content": "OPTIONS:\nA. x\nB. y"},
                {"role": "user", "content": "QUESTION: first?\n\nAnswer"},
                {"role": "assistant", "content": "Answer: B"},
                {"role": "user", "content": "QUESTION: second?\n\nAnswer"},
                {"role": "assistant", "content": D.ANSWER_LEAD}]
        s3 = B.structure(hist, eng.upstream, "bonsai")
        cs = [m["content"] for m in hist]
        check(f"[derive:{tname}] a choose-shaped structure (a mid-"
              "conversation assistant turn, three user messages) derives "
              "and equals the served rendering byte for byte",
              s3.prefix(cs) + s3.block(cs) == eng.render(hist))
        # edge whitespace the template trims
        m4 = D.messages("  lead and trail \n\n", c)
        m4[1]["content"] = D.STATE_HEAD + "  lead and trail \n\n"
        cs = [m["content"] for m in m4]
        check(f"[derive:{tname}] a content with edge whitespace is trimmed "
              "as the template trims it",
              st.prefix(cs) + st.block(cs) == eng.render(m4))

    # ---- disabled structures (cached, recorded), the call goes per read
    cases = [("bad_apply=length", {"bad_apply": "length"}, "second message "
              "set"),
             ("bad_apply=gen", {"bad_apply": "gen"}, "not the start"),
             ("merging tokenizer", {"merge": True}, "merges across")]
    for label, kw, frag in cases:
        qs = questions()[:2]
        ref = seq_answers(fresh(**kw), qs)
        eng = fresh(**kw)
        batch(True)
        try:
            got = B.read_many(STATE, qs, slot=3, post=eng.post,
                              upstream=eng.upstream)
        finally:
            batch(False)
        st = B.stats()
        why = " ".join(v.get("why", "") for v in st["structures"].values())
        check(f"[disabled:{label}] batching is disabled for the structure "
              "(recorded with why) and no /decide-batch request is made",
              any(not v["ok"] for v in st["structures"].values())
              and frag in why and not eng.batch_requests, (why, st))
        check(f"[disabled:{label}] the answers still come, per read, equal "
              "to the per-read path's",
              same(got, ref) and all(
                  a["diagnostics"]["read_path"] == "sequential"
                  for a in got)
              and len(eng.chat_bodies) == 4, len(eng.chat_bodies))
        n = eng.paths.get("/apply-template", 0)
        batch(True)
        try:
            B.read_many(STATE, qs, slot=3, post=eng.post,
                        upstream=eng.upstream)
        finally:
            batch(False)
        check(f"[disabled:{label}] the negative is cached: a second call "
              "derives nothing again", eng.paths.get("/apply-template", 0)
              == n, eng.paths)

    # the engine's own boundary check (verify_tokenization) disables too
    qs = questions()[:2]
    ref = seq_answers(fresh(), qs)
    eng = fresh(engine_tok_bad=True)
    got = batch_answers(eng, qs)
    check("[disabled:engine boundary] an engine that reports "
          "tokenization_ok false: the call is answered per read (same "
          "answers), the structure is disabled",
          same(got, ref) and len(eng.batch_requests) == 1
          and any(not v["ok"] for v in B.stats()["structures"].values()),
          B.stats()["structures"])
    check("[disabled:engine boundary] the first request asked for "
          "verify_tokenization", eng.batch_requests[0]["verify_tokenization"]
          is True)
    before = len(eng.batch_requests)
    batch_answers(eng, qs)
    check("[disabled:engine boundary] and never asks the engine again for "
          "that structure", len(eng.batch_requests) == before)
    # a verified structure stops asking for the engine's check
    eng = fresh()
    batch(True)
    try:
        B.read_many(STATE, qs, slot=3, post=eng.post, upstream=eng.upstream)
        B.read_many(STATE, qs, slot=3, post=eng.post, upstream=eng.upstream)
    finally:
        batch(False)
    check("[verify] verify_tokenization is asked on the first call of a "
          "structure only", [r["verify_tokenization"]
                            for r in eng.batch_requests] == [True, False],
          [r["verify_tokenization"] for r in eng.batch_requests])


# ================================================================== identity ===
def identity() -> None:
    for tname in ("small", "real"):
        qs = questions()
        for share in (True, False):
            eng_b = fresh(template=tname, share=share)
            got = batch_answers(eng_b, qs)
            ref = seq_answers(fresh(template=tname), qs)
            check(f"[identity:{tname},share={share}] noul (plain and with "
                  "criteria), choice with none, score, digit score: the "
                  "batched answers equal the per-read answers",
                  same(got, ref),
                  [a["name"] for a, b in zip(got, ref)
                   if norm(a) != norm(b)])
            check(f"[identity:{tname},share={share}] ONE /decide-batch "
                  "request, zero per-read requests, five groups of two blocks",
                  len(eng_b.batch_requests) == 1
                  and not eng_b.chat_bodies
                  and [len(g) for g in eng_b.batch_requests[0]["groups"]]
                  == [2] * 5, (len(eng_b.batch_requests), eng_b.paths))
            check(f"[identity:{tname},share={share}] every answer says it "
                  "was batched, with the one call's note",
                  all(a["diagnostics"]["read_path"] == "batch"
                      and a["diagnostics"]["batch"]["requests"] == 1
                      and a["diagnostics"]["batch"]["blocks"] == 10
                      and a["diagnostics"]["read_regime"] == "batch"
                      for a in got),
                  got[0]["diagnostics"].get("batch"))
        # one question
        eng_b = fresh(template=tname)
        got = batch_answers(eng_b, qs[2:3])
        ref = seq_answers(fresh(template=tname), qs[2:3])
        check(f"[identity:{tname}] one question: equal, one request",
              same(got, ref) and len(eng_b.batch_requests) == 1)

    # request content: the label ids, top 20, the prefix once
    eng = fresh()
    batch_answers(eng, questions())
    r = eng.batch_requests[0]
    want_ids = sorted({tid for q in questions()
                       for o in D.rendered_orders(q) for lab in o["labels"]
                       for tid in D.spelling_ids(lab, eng.upstream).values()})
    check("[request] token_ids are the union of every label spelling id, "
          "top_logprobs 20, keep_prefix false, the prefix is the state "
          "rendered through the system message once",
          r["token_ids"] == want_ids and r["top_logprobs"] == 20
          and r["keep_prefix"] is False and r["prefix"].startswith(
              "<|im_start|>system\n") and r["prefix"].endswith("<|im_end|>\n")
          and r["prefix"].count("MATERIAL:") == 1
          and all(not b.startswith("<|im_start|>system") for g in r["groups"]
                  for b in g), (r["token_ids"][:5], r["prefix"][:60]))
    check("[request] each group is the two orders of one question (blocks "
          "start at the question message, end on the answer lead)",
          all(len(g) == 2 and all(b.startswith("<|im_start|>user\nQUESTION:")
                                  and b.endswith("Answer:") for b in g)
              and g[0] != g[1] for g in r["groups"]))

    # 26 options (52 spellings > the top 20): equal answers, the per-read
    # path may re-read, the batch never does
    big = D.q_choice("big", "Which?", [f"option {i}" for i in range(26)])
    eng_b = fresh(bare_zero=True)
    got = batch_answers(eng_b, [big])
    eng_s = fresh(bare_zero=True)
    ref = seq_answers(eng_s, [big])
    check("[identity] 26 options (26 spaced spellings: six past the top 20, "
          "so the per-read path re-reads at K 200): the distribution equals "
          "the per-read path's, in one batched request, each order read once",
          got[0]["probabilities"] == ref[0]["probabilities"]
          and got[0]["choice"] == ref[0]["choice"]
          and got[0]["diagnostics"]["orders"][0]["exact"] is True
          and len(eng_b.batch_requests) == 1 and not eng_b.chat_bodies
          and ref[0]["diagnostics"]["orders"][0]["http_reads"] >= 2,
          (ref[0]["diagnostics"]["orders"][0]["http_reads"],
           got[0]["probabilities"] == ref[0]["probabilities"]))

    # a custom render (decide_turn.choose puts the options in their own
    # message): the same structure derivation, the same answer
    q = D.q_choice("rc", "Which fact fits?", ["alpha", "beta", "gamma"],
                   keys=["a", "b", "c"], none=2)

    def render(c):
        opts = "\n".join(f"{lab}. {t}" for lab, t in zip(c["labels"],
                                                         c["options"]))
        return [{"role": "system", "content": D.SYSTEM},
                {"role": "user", "content": D.STATE_HEAD + STATE.strip()},
                {"role": "user", "content": "OPTIONS:\n" + opts},
                {"role": "user", "content": "QUESTION: " + c["question"]
                 + "\n\nAnswer with the letter of one option."},
                {"role": "assistant", "content": D.ANSWER_LEAD}]
    for tname in ("small", "real"):
        eng_b = fresh(template=tname)
        got = batch_answers(eng_b, [q], render=render)
        ref = seq_answers(fresh(template=tname), [q], render=render)
        check(f"[identity:{tname}] a custom render (the options in their own "
              "message): equal answers, one request",
              same(got, ref) and len(eng_b.batch_requests) == 1
              and not eng_b.chat_bodies
              and got[0]["diagnostics"]["read_path"] == "batch",
              got[0]["diagnostics"].get("batch_fallback"))
    # a render that does not start with the same two messages: that
    # question goes per read, the rest batch
    other = [questions()[0], q]
    eng_b = fresh()

    def render2(c):
        m = render(c)
        if c["question"] == "Which fact fits?":
            m[1] = {"role": "user", "content": "MATERIAL:\n\nsomething else"}
        return m
    batch(True)
    try:
        got = B.read_many(STATE, other, slot=3, post=eng_b.post,
                          upstream=eng_b.upstream, render=render2)
    finally:
        batch(False)
    check("[mixed] a question whose prefix differs from the call's goes per "
          "read (said why), the other is batched in the one request",
          got[0]["diagnostics"]["read_path"] == "batch"
          and got[1]["diagnostics"]["read_path"] == "sequential"
          and "prefix differs" in got[1]["diagnostics"]["batch_fallback"]
          and len(eng_b.batch_requests) == 1 and len(eng_b.chat_bodies) == 2)

    # exclusion: a tuple for all, a map by name
    eng_b = fresh()
    qs = [questions()[2], D.q_choice("c2", "Pick?", ["a", "b", "c"],
                                     keys=["x", "y", "z"])]
    got = batch_answers(eng_b, qs, exclude={"c1": ("pay",), "c2": ("y",)})
    eng_s = fresh()
    ref = [D.read(STATE, qs[0], slot=3, post=eng_s.post, upstream=eng_s.upstream,
                  exclude=("pay",)),
           D.read(STATE, qs[1], slot=3, post=eng_s.post, upstream=eng_s.upstream,
                  exclude=("y",))]
    check("[identity] exclusions (a map by question name): equal answers, "
          "the excluded read 0", same(got, ref)
          and got[0]["probabilities"]["pay"] == 0.0
          and got[1]["probabilities"]["y"] == 0.0)
    eng_b = fresh()
    got = batch_answers(eng_b, qs[:1], exclude=("pay", "tech"))
    ref = seq_answers(fresh(), qs[:1], exclude=("pay", "tech"))
    check("[identity] exclusions (a tuple for every question): equal",
          same(got, ref))
    eng_b = fresh()
    batch(True)
    try:
        raised = None
        try:
            B.read_many(STATE, qs[:1], slot=3, post=eng_b.post,
                        upstream=eng_b.upstream,
                        exclude=("pay", "tech", "sales", "other"))
        except D.DeciderUnavailable as e:
            raised = e.code
    finally:
        batch(False)
    check("[guards] every option excluded: ALL_EXCLUDED before anything is "
          "sent", raised == "ALL_EXCLUDED" and not eng_b.batch_requests
          and not eng_b.chat_bodies, raised)

    # a content-free prior question falls back alone
    pq = D.q_noul("pk", "Is X used?", prior="content_free")
    eng_b = fresh()
    cf_calls = []

    def prior_for(c):
        cf_calls.append(c["labels"])
        return {k: 0.5 for k in c["meaning"].values()}
    batch(True)
    try:
        got = B.read_many(STATE, [questions()[0], pq], slot=3, post=eng_b.post,
                          upstream=eng_b.upstream, prior_for=prior_for)
    finally:
        batch(False)
    eng_s = fresh()
    ref = [D.read(STATE, questions()[0], slot=3, post=eng_s.post,
                  upstream=eng_s.upstream),
           D.read(STATE, pq, slot=3, post=eng_s.post, upstream=eng_s.upstream,
                  prior_for=prior_for)]
    check("[mixed] a question built with prior=content_free (prior_for "
          "given) is read per read, the other batched, both answers equal "
          "the per-read path's", same(got, ref)
          and got[0]["diagnostics"]["read_path"] == "batch"
          and got[1]["diagnostics"]["read_path"] == "sequential"
          and "content-free" in got[1]["diagnostics"]["batch_fallback"],
          got[1]["diagnostics"].get("batch_fallback"))

    # the think-block guard still refuses a bad rendering, before a request
    def bad_render(c):
        return D.messages(STATE, c)[:-1]
    eng_b = fresh()
    batch(True)
    try:
        try:
            B.read_many(STATE, qs[:1], slot=3, post=eng_b.post,
                        upstream=eng_b.upstream, render=bad_render)
            code = None
        except D.DeciderUnavailable as e:
            code = e.code
    finally:
        batch(False)
    check("[guards] a render with no answer prefill is refused by the "
          "think-block guard, nothing sent", code == "NO_ANSWER_PREFILL"
          and not eng_b.batch_requests and not eng_b.chat_bodies, code)


# ===================================================== sub-top-20 spellings ===
def unread_bound() -> None:
    qs = [D.q_choice("sp", "Which?", ["a", "b", "c", "d"]),
          D.q_noul("spn", "Is it?")]
    eng_b = fresh(sparse=True)
    got = batch_answers(eng_b, qs)
    eng_s = fresh(sparse=True)
    seen_missing = 0
    ok = True
    worst = 0.0
    for a, q in zip(got, qs):
        for oi, c in enumerate(D.rendered_orders(q)):
            ref = D.ask_one(STATE, c, slot=3, post=eng_s.post,
                            upstream=eng_s.upstream)
            o = a["diagnostics"]["orders"][oi]
            diff = o["label_mass"] - ref["label_mass"]
            seen_missing += len(ref["unread"])
            worst = max(worst, diff)
            # the batch reads exactly; the per-read path left `unread`
            # spellings below its top K, bounded by `unread_bound`
            ok &= o["exact"] is True and -1e-9 <= diff <= ref["unread_bound"] \
                + 1e-6
            if ref["k"] == 20:
                ok &= ref["exact"] == (not ref["unread"])
    check("[unread] the per-read path left spellings unread here (the test "
          "is meaningful)", seen_missing > 0, seen_missing)
    check("[unread] the batched label mass exceeds the per-read one by the "
          "spellings the per-read path left unread, and by no more than "
          "its unread_bound; the batched read is exact", ok, worst)
    check("[unread] the batch made no K re-read (every order one read)",
          all(o["http_reads"] == 1 for a in got
              for o in a["diagnostics"]["orders"])
          and len(eng_b.batch_requests) == 1)


# ================================================================== the switch ===
def switch_off() -> None:
    qs = questions()
    eng_a = fresh()
    ref = seq_answers(eng_a, qs)
    eng_b = fresh()
    got = B.read_many(STATE, qs, slot=3, post=eng_b.post,
                      upstream=eng_b.upstream)
    check("[switch] off (the default): read_many is [read(...)] -- the "
          "same answers key for key, no read_path, no batch note",
          same(got, ref) and all("read_path" not in a["diagnostics"]
                                 and "batch" not in a["diagnostics"]
                                 for a in got))
    check("[switch] off: the very same per-read requests (bodies byte for "
          "byte), zero /decide-batch and zero /apply-template requests",
          [json.dumps(b, sort_keys=True) for b in eng_b.chat_bodies]
          == [json.dumps(b, sort_keys=True) for b in eng_a.chat_bodies]
          and not eng_b.batch_requests
          and "/apply-template" not in eng_b.paths, eng_b.paths)
    seen = {}
    for v in ("0", "off", "1", "on", "ON", "yes", ""):
        os.environ["YAMADORI_DECIDER_BATCH"] = v
        seen[v] = D.batch_on()
    os.environ.pop("YAMADORI_DECIDER_BATCH", None)
    check("[switch] the default is off, and '1' / 'on' are the only values "
          "that turn it on", not D.batch_on() and seen == {
              "0": False, "off": False, "1": True, "on": True, "ON": True,
              "yes": False, "": False}, seen)
    os.environ.pop("YAMADORI_DECIDER_BATCH", None)
    # D.decide off: unchanged, no read_path in its result
    eng_c = fresh()
    out = D.decide(STATE, questions(), post=eng_c.post,
                   upstream=eng_c.upstream, slot=3, keep_slot=True)
    check("[switch] D.decide off: no batch fields in its result, only "
          "per-read requests", "read_path" not in out and len(
              eng_c.chat_bodies) == 10 and not eng_c.batch_requests)
    # on: D.decide goes through one request and says so
    eng_d = fresh()
    batch(True)
    try:
        out2 = D.decide(STATE, questions(), post=eng_d.post,
                        upstream=eng_d.upstream, slot=3, keep_slot=True)
    finally:
        batch(False)
    check("[decide] D.decide on: every question in ONE request, its result "
          "says read_path batch, the answers equal the per-read run's",
          out2["read_path"] == ["batch"] and len(eng_d.batch_requests) == 1
          and not eng_d.chat_bodies and same(
              [out2["answers"][q["name"]] for q in questions()],
              [out["answers"][q["name"]] for q in questions()]),
          out2.get("read_path"))
    check("[decide] usage = the reads' prompt tokens as before: output_tokens"
          " = 10 (five questions, two orders), input_tokens counted per order",
          out2["usage"]["output_tokens"] == 10 == out["usage"]["output_tokens"]
          and out2["usage"]["input_tokens"] > 0)


# ====================================================================== usage ===
def usage() -> None:
    qs = [questions()[0], questions()[2]]

    def orders_of(answers):
        return [o for a in answers for o in a["diagnostics"]["orders"]]

    for tname in ("small", "real"):
        # COLD: no sharing in the fake, so the batch decodes exactly the
        # tokens the per-read path decodes
        eng_s = fresh(template=tname, share=False)
        ref = seq_answers(eng_s, qs)
        eng_b = fresh(template=tname, share=False)
        got = batch_answers(eng_b, qs)
        us, ub = J.usage_of(orders_of(ref)), J.usage_of(orders_of(got))
        check(f"[usage:{tname}] cold prefix: input_tokens and output_tokens "
              "equal the per-read path's (no double count of the prefix)",
              us["input_tokens"] == ub["input_tokens"]
              and us["output_tokens"] == ub["output_tokens"] == 4,
              (us, ub))
        # the engine's own accounting agrees with ours
        t = eng_b.batch_requests and got[0]["diagnostics"]["batch"]
        check(f"[usage:{tname}] the accounted processed tokens equal the "
              "engine's timings.processed_tokens",
              t["accounted_processed"] == t["timings"]["processed_tokens"],
              (t["accounted_processed"], t["timings"]))
        # WARM: the second call of a burst reads on the resident prefix; the
        # per-read path's second call finds the slot holding it
        eng_s = fresh(template=tname, share=False)
        r1 = seq_answers(eng_s, qs[:1])
        r2 = seq_answers(eng_s, qs[1:])
        eng_b = fresh(template=tname, share=False)
        batch(True)
        try:
            with B.burst(STATE, upstream=eng_b.upstream) as bu:
                g1 = bu.read_many(STATE, qs[:1], slot=3, post=eng_b.post,
                                  upstream=eng_b.upstream, keep_prefix=True)
                g2 = bu.read_many(STATE, qs[1:], slot=3, post=eng_b.post,
                                  upstream=eng_b.upstream, keep_prefix=False)
        finally:
            batch(False)
        ub1, ub2 = J.usage_of(orders_of(g1)), J.usage_of(orders_of(g2))
        us1, us2 = J.usage_of(orders_of(r1)), J.usage_of(orders_of(r2))
        pre = g2[0]["diagnostics"]["batch"]["prefix"]
        check(f"[usage:{tname}] warm prefix (the second call of a burst, "
              "the slot holding it for the per-read path): input_tokens "
              "equal the per-read path's, the prefix counted as cached",
              ub1["input_tokens"] == us1["input_tokens"]
              and ub2["input_tokens"] == us2["input_tokens"]
              and ub2["output_tokens"] == us2["output_tokens"]
              and pre["reused"] == pre["tokens"] and pre["processed"] == 0,
              (ub1, us1, ub2, us2, pre))
        check(f"[usage:{tname}] the second call read on the RESIDENT prefix "
              "(reused all of it), and the burst ended with no further "
              "release request (the last call freed it)",
              pre["reused"] > 0 and [r["keep_prefix"] for r in
                                     eng_b.batch_requests] == [True, False]
              and eng_b.releases == 0, eng_b.batch_requests and
              [r.get("keep_prefix") for r in eng_b.batch_requests])
    # with sharing the batch decodes fewer tokens: exactly by the shared head
    eng_s = fresh(share=False)
    eng_b = fresh(share=True)
    batch_answers(eng_s, qs[:1])
    got = batch_answers(eng_b, qs[:1])
    ns = eng_s.batch_requests[0]
    saved = got[0]["diagnostics"]["batch"]["timings"]["shared_tokens_saved"]
    nb = eng_b.batch_requests[0]
    check("[usage] with the shared question head decoded once, the batch's "
          "input is lower than the per-read path's by exactly the head "
          "(shared_tokens_saved), the requests otherwise equal",
          saved > 0 and ns["groups"] == nb["groups"]
          and ns["prefix"] == nb["prefix"], saved)


# ================================================================= JEV =========
class JevBox:
    """Jev's systemone with the fake behind decider_bonsai's doors."""

    def __init__(self, eng: Engine):
        self.eng = eng
        self.saved = (D._post_default, D._upstream)

    def __enter__(self):
        D._post_default, D._upstream = self.eng.post, self.eng.upstream
        return self

    def __exit__(self, *exc):
        D._post_default, D._upstream = self.saved
        return False


def jev_body(rounds: bool = False) -> dict:
    body = {"model": "jjava-latest", "state": STATE,
            "questions": {
                "urgent": {"type": "noul",
                           "instructions": "Does this convey urgency?"},
                "team": {"type": "choice", "instructions": "Team?",
                         "criteria": {"billing": "Payments", "tech": "Tech",
                                      "sales": "Sales"}},
                "anger": {"type": "score", "instructions": "Anger?",
                          "criteria": ["Calm", "Annoyed", "Furious"]}}}
    if rounds:
        body["questions"]["many"] = {
            "type": "choice", "instructions": "Which of these?",
            "criteria": {f"k{i:02d}": f"option number {i}"
                         for i in range(60)}}
    return body


def jev() -> None:
    for rounds in (False, True):
        tag = "two-stage" if rounds else "single"
        eng_s = fresh(share=False, bare_zero=True)
        with JevBox(eng_s):
            batch(False)
            st_s, out_s, _ = J.systemone(jev_body(rounds), None)
        eng_b = fresh(share=False, bare_zero=True)
        with JevBox(eng_b):
            batch(True)
            try:
                st_b, out_b, _ = J.systemone(jev_body(rounds), None)
            finally:
                batch(False)
        ok = st_s == 200 and st_b == 200
        check(f"[jev:{tag}] both paths answer 200", ok, (st_s, st_b,
                                                          str(out_b)[:200]))
        if not ok:
            continue
        check(f"[jev:{tag}] Jev's answers (fields and values) are "
              "identical to the per-read path's",
              json.dumps(out_s["answers"], sort_keys=True)
              == json.dumps(out_b["answers"], sort_keys=True),
              (out_s["answers"], out_b["answers"]))
        check(f"[jev:{tag}] usage equals the per-read path's (cold prefix)",
              out_s["usage"] == out_b["usage"], (out_s["usage"],
                                                 out_b["usage"]))
        reqs = eng_b.batch_requests
        if not rounds:
            check("[jev:single] every question in ONE request (3 groups of "
                  "2), zero per-read requests, the prefix freed with it",
                  len(reqs) == 1 and [len(g) for g in reqs[0]["groups"]]
                  == [2, 2, 2] and not eng_b.chat_bodies
                  and reqs[0]["keep_prefix"] is False and not eng_b.releases,
                  [len(r.get("groups") or []) for r in reqs])
        else:
            check("[jev:two-stage] stage 1 (the three single plans and the "
                  "60-option question's three chunks) in ONE request with "
                  "the prefix kept, the final read in a second request that "
                  "frees it; zero per-read requests",
                  len(reqs) == 2 and len(reqs[0]["groups"]) == 3 + 3
                  and reqs[0]["keep_prefix"] is True
                  and len(reqs[1]["groups"]) == 1
                  and reqs[1]["keep_prefix"] is False
                  and reqs[0]["prefix"] == reqs[1]["prefix"]
                  and not eng_b.chat_bodies and not eng_b.releases,
                  [(len(r.get("groups") or []), r.get("keep_prefix"))
                   for r in reqs])
            r2 = out_b["x_yamadori"]["answers"]["many"]["rounds"][-1]
            check("[jev:two-stage] the final read used the resident prefix "
                  "(its note: reused == tokens)",
                  r2["diagnostics"]["batch"]["prefix"]["reused"]
                  == r2["diagnostics"]["batch"]["prefix"]["tokens"] > 0,
                  r2["diagnostics"]["batch"]["prefix"])
        xb = out_b["x_yamadori"]
        check(f"[jev:{tag}] x_yamadori says the reads were batched",
              xb["batch"]["read_paths"] == ["batch"] and xb["parallel"] is True
              and "batched engine" in xb["note"], xb.get("batch"))
        check(f"[jev:{tag}] switch off: x_yamadori carries no batch field "
              "and says the questions ran one after another",
              "batch" not in out_s["x_yamadori"]
              and out_s["x_yamadori"]["parallel"] is False)
    # a failing batch: the Jev call is still answered, per read, identically
    eng_f = fresh(share=False, bare_zero=True)
    eng_f.batch_mode = ("http", 501)
    with JevBox(eng_f):
        batch(True)
        try:
            st_f, out_f, _ = J.systemone(jev_body(True), None)
        finally:
            batch(False)
    eng_r = fresh(share=False, bare_zero=True)
    with JevBox(eng_r):
        batch(False)
        st_r, out_r, _ = J.systemone(jev_body(True), None)
    check("[jev:fallback] a 501 from the engine: the two-stage call is "
          "answered per read with the same answers and usage, one /decide-"
          "batch request tried and the negative cached",
          st_f == 200 and json.dumps(out_f["answers"], sort_keys=True)
          == json.dumps(out_r["answers"], sort_keys=True)
          and out_f["usage"] == out_r["usage"]
          and len(eng_f.batch_requests) == 1
          and out_f["x_yamadori"]["batch"]["read_paths"] == ["sequential"],
          (st_f, len(eng_f.batch_requests)))


# ================================================================== the Turn ====
def turn() -> None:
    qs = questions()

    def run_turn(on: bool, eng: Engine, calls):
        batch(on)
        try:
            RELEASED.clear()
            t = T.Turn([], state=STATE, state_info={"kind": "step"}, on=True,
                       key="batch-test", request="r1", post=eng.post,
                       upstream=eng.upstream)
            with t:
                outs = [t.decide(c) for c in calls]
            return outs, t
        finally:
            batch(False)
    eng_s = fresh()
    ref, _ = run_turn(False, eng_s, [qs[:3], qs[3:4]])
    eng_b = fresh()
    got, t = run_turn(True, eng_b, [qs[:3], qs[3:4]])
    check("[turn] Turn.decide on: a multi-question list in ONE request, the "
          "Turn's second decide() in a second request on the resident "
          "prefix, then ONE release at close; zero per-read requests",
          [r.get("keep_prefix") for r in eng_b.batch_requests if "groups"
           in r] == [True, True] and eng_b.releases == 1
          and len(eng_b.batch_requests) == 3
          and not eng_b.chat_bodies and t.batch_release["released"] is True,
          [(r.get("keep_prefix"), r.get("release")) for r in
           eng_b.batch_requests])
    check("[turn] the answers (and tiers, decision rows) equal the per-read "
          "Turn's",
          all(same(a, b) for a, b in zip(got, ref))
          and [a.get("tier") for o in got for a in o]
          == [a.get("tier") for o in ref for a in o])
    check("[turn] the second decide() reused the whole resident prefix",
          got[1][0]["diagnostics"]["batch"]["prefix"]["reused"]
          == got[1][0]["diagnostics"]["batch"]["prefix"]["tokens"] > 0)
    check("[turn] the Turn's lane slot is still released as before "
          "(decider release once, or the lane kept)",
          len(RELEASED) == (0 if __import__("slots").lane_kept() else 1),
          RELEASED)
    # off: the Turn is untouched
    eng_o = fresh()
    _o, t_o = run_turn(False, eng_o, [qs[:3]])
    check("[turn] off: per-read requests only, no burst, no release request",
          not eng_o.batch_requests and not hasattr(t_o, "batch_release")
          and len(eng_o.chat_bodies) == 6)
    # a Turn question that needs the content-free prior keeps the old loop
    eng_p = fresh()
    pq = D.q_noul("pk", "Is X used?", prior="content_free")
    saved = T.CONTEXTUAL
    T.CONTEXTUAL = True
    try:
        run_turn(True, eng_p, [[qs[0], pq]])
    finally:
        T.CONTEXTUAL = saved
    check("[turn] a call with a content-free-prior question keeps the "
          "per-read loop whole (no batch request for it)",
          not eng_p.batch_requests and len(eng_p.chat_bodies) >= 4)

    # craft_query.ask: choice, then the gate on its best craft
    try:
        import craft_query as C
        top = [{"id": f"s{i}", "name": f"craft-{i}", "version": 1,
                "trigger": f"t{i}"} for i in range(3)]
        import package_skills as PS
        PS.trigger_text = lambda s: f"when {s['name']} applies"

        def factory(messages, state, info, ctx):
            return T.Turn(messages, state=state, state_info=info, on=True,
                          post=eng_c.post, upstream=eng_c.upstream)
        ctx = {"messages": [{"role": "user", "content": "how do I x?"}],
               "turn_factory": factory, "model": "bonsai"}
        eng_c = fresh()
        batch(False)
        ref_c = C.ask("how do I x?", top, ctx)
        eng_c = fresh()
        batch(True)
        try:
            got_c = C.ask("how do I x?", top, ctx)
        finally:
            batch(False)
        check("[craft] craft_query.ask on: the choice and the gate are two "
              "batched requests on one resident prefix, then one release; "
              "the same craft and probabilities as the per-read path",
              [r.get("keep_prefix") for r in eng_c.batch_requests
               if "groups" in r] == [True, True] and eng_c.releases == 1
              and not eng_c.chat_bodies and not got_c.get("failure")
              and got_c["best"]["id"] == ref_c["best"]["id"]
              and got_c["p"] == ref_c["p"]
              and got_c["gate_p"] == ref_c["gate_p"],
              (got_c.get("failure"), len(eng_c.batch_requests)))
    except ImportError as e:
        check(f"[craft] craft_query importable ({e})", False)


# ==================================================================== fallback ===
def fallback() -> None:
    qs = questions()[:3]
    clock = [1000.0]
    B.CLOCK = lambda: clock[0]
    modes = [("http 404", ("http", 404)), ("http 501", ("http", 501)),
             ("http 503", ("http", 503)), ("http 400", ("http", 400)),
             ("timeout", "timeout"), ("unreachable", "unreachable"),
             ("garbage json", "garbage"), ("bad shape", "badshape"),
             ("http 500", ("http", 500)),
             ("missing logprob", "missing_logprob")]
    try:
        for label, mode in modes:
            clock[0] += 10_000
            kw = {"sparse": True} if mode == "missing_logprob" else {}
            ref = seq_answers(fresh(**kw), qs)
            eng = fresh(**kw)
            eng.batch_mode = mode
            batch(True)
            try:
                got = B.read_many(STATE, qs, slot=3, post=eng.post,
                                  upstream=eng.upstream)
                n1 = len(eng.batch_requests)
                got2 = B.read_many(STATE, qs, slot=3, post=eng.post,
                                   upstream=eng.upstream)
                n2 = len(eng.batch_requests)
                clock[0] += 59.0
                B.read_many(STATE, qs, slot=3, post=eng.post,
                            upstream=eng.upstream)
                n3 = len(eng.batch_requests)
                clock[0] += 2.0
                eng.batch_mode = None
                got4 = B.read_many(STATE, qs, slot=3, post=eng.post,
                                   upstream=eng.upstream)
                n4 = len(eng.batch_requests)
            finally:
                batch(False)
            check(f"[fallback:{label}] answered per read with the same "
                  "answers, never lost",
                  same(got, ref) and same(got2, ref) and all(
                      a["diagnostics"]["read_path"] == "sequential"
                      for a in got), label)
            check(f"[fallback:{label}] one request tried, then a cached "
                  "negative: no engine request for 59 s",
                  n1 == 1 and n2 == 1 and n3 == 1, (n1, n2, n3))
            check(f"[fallback:{label}] after 60 s the engine is tried again "
                  "and (now serving) answers as a batch",
                  n4 == 2 and (same(got4, ref) or mode == "missing_logprob")
                  and got4[0]["diagnostics"]["read_path"] == "batch",
                  (n4, got4[0]["diagnostics"].get("batch_fallback")))
            st = B.stats()
            check(f"[fallback:{label}] counted: a failure, a fallback call, "
                  "a negative cached", st["failures"] == 1
                  and st["fallback_calls"] >= 2 and st["negative_cached"] == 1
                  and st["last_error"] is not None, st)
    finally:
        B.CLOCK = __import__("time").monotonic

    # release on an exception, and after a failed keep
    eng = fresh()
    batch(True)
    try:
        try:
            with B.burst(STATE, upstream=eng.upstream) as bu:
                bu.read_many(STATE, qs[:1], slot=3, post=eng.post,
                             upstream=eng.upstream, keep_prefix=True)
                raise RuntimeError("the caller failed")
        except RuntimeError:
            pass
    finally:
        batch(False)
    check("[release] an exception inside a burst frees the resident prefix "
          "(one release request after the kept call)",
          eng.releases == 1 and eng.resident is None
          and eng.batch_requests[-1].get("release") is True,
          [r.get("release") for r in eng.batch_requests])
    eng = fresh()
    eng.batch_mode = ("http", 500)
    batch(True)
    try:
        with B.burst(STATE, upstream=eng.upstream) as bu:
            got = bu.read_many(STATE, qs[:1], slot=3, post=eng.post,
                               upstream=eng.upstream, keep_prefix=True)
    finally:
        batch(False)
    check("[release] a 500 on a kept call (the engine may hold the prefix): "
          "answered per read, and the burst's close sends the release",
          got[0]["diagnostics"]["read_path"] == "sequential"
          and eng.batch_requests[-1].get("release") is True
          and len(eng.batch_requests) == 2, len(eng.batch_requests))
    eng = fresh()
    eng.batch_mode = "timeout"
    batch(True)
    try:
        with B.burst(STATE, upstream=eng.upstream) as bu:
            bu.read_many(STATE, qs[:1], slot=3, post=eng.post,
                         upstream=eng.upstream, keep_prefix=True)
    finally:
        batch(False)
    check("[release] after a timeout no release is attempted (the server "
          "cannot be asked again)", len(eng.batch_requests) == 1)
    eng = fresh()
    eng.batch_mode = ("http", 501)
    batch(True)
    try:
        with B.burst(STATE, upstream=eng.upstream) as bu:
            bu.read_many(STATE, qs[:1], slot=3, post=eng.post,
                         upstream=eng.upstream, keep_prefix=True)
    finally:
        batch(False)
    check("[release] after a 501 nothing is held, nothing is released",
          len(eng.batch_requests) == 1)
    # a release that fails never raises
    def dead(path, payload=None, timeout=30):
        raise urllib.error.URLError("gone")
    r = B.release(dead)
    check("[release] a failed release is reported, never raised",
          r["released"] is False and "URLError" in r["error"], r)
    # a DeciderUnavailable from the engine door's guards is not swallowed
    class Cap(Exception):
        pass
    eng = fresh()
    qs2 = questions()[:1]
    batch(True)
    try:
        import max_mode

        def capacity(path, payload=None, timeout=30):
            if path == "/decide-batch":
                raise max_mode.ModelAtCapacity("flash-next", 7, "busy") \
                    if _takes3(max_mode.ModelAtCapacity) else \
                    max_mode.ModelAtCapacity("busy")
            return eng.upstream(path, payload, timeout)
        got = None
        try:
            got = B.read_many(STATE, qs2, slot=3, post=eng.post,
                              upstream=capacity)
            err = None
        except max_mode.ModelAtCapacity as e:
            err = e
    finally:
        batch(False)
    check("[guards] max_mode's own refusal (ModelAtCapacity) from the "
          "engine door propagates, as it does from the per-read door",
          err is not None and got is None, err)


def _takes3(cls) -> bool:
    import inspect
    try:
        return len(inspect.signature(cls.__init__).parameters) >= 4
    except (TypeError, ValueError):
        return False


# ================================================================ the driver ====
def driver() -> None:
    import batch_ab as AB
    pool = []
    for i in range(6):
        pool.append({"id": f"it{i}", "state": "state number %d. " % i * (3 + i),
                     "question": (D.q_noul("decision", f"Is {i} even?") if i % 2
                                  else D.q_choice("decision", "Which?",
                                                  ["a", "b", "c"]))})
    cases = [{"case": f"c{i}", "state": "case state %d" % i, "kind": "step",
              "items": [{"key": f"k{i}{j}", "fact": f"fact {i}{j}"}
                        for j in range(3)]} for i in range(4)]
    plan = AB.build_plan(
        pool=pool, cases=cases, ks=(1, 2, 5),
        questions_fn=lambda v, items: [D.q_noul(f"it:{it['key']}", it["fact"])
                                       for it in items],
        stage3_fn=lambda v, items: D.q_noul("stage3", "any?"))
    cnt = AB.request_counts(plan)
    check("[driver] the plan has the JevBench set, K sets and the skill set",
          set(plan["sets"]) == {"jevbench", "k1", "k2", "k5", "skill"}
          and plan["sets"]["jevbench"]["calls"] == 6
          and plan["sets"]["skill"]["calls"] == 4, plan["sets"])
    check("[driver] K-question calls: K questions renamed q0.., the next "
          "items' questions after the item's own (wrapping)",
          [q["name"] for q in plan["calls"][6 + 6 + 1]["questions"]]
          == ["q0", "q1"] and plan["calls"][6 + 6 + 1]["item"] == "it1")
    check("[driver] exact request counts: two per-read requests a question "
          "against one /decide-batch a call (a Turn call: stage 3 and one "
          "release more)",
          cnt["k5"]["sequential_reads"] == 2 * 5 * 6
          and cnt["k5"]["batch_requests"] == 6
          and cnt["skill"]["sequential_reads"] == 2 * 4 * (3 + 1)
          and cnt["skill"]["batch_requests"] == 4 * 3, cnt)
    check("[driver] a call over the 64k-token request limit is left out and "
          "counted", AB.build_plan(pool=[dict(pool[0], state="x" * 400_000)],
                                   ks=(1,))["skipped"] == {"k1": 1})
    # the identity comparison
    a = {"type": "noul", "noul": 0.80, "diagnostics": {"tie": False}}
    b = {"type": "noul", "noul": 0.795, "diagnostics": {"tie": True}}
    c = AB.compare_answers(a, b)
    check("[driver] compare_answers: argmax, max |p|, tie flips",
          c["argmax_differs"] is False and abs(c["max_abs_p"] - 0.005) < 1e-9
          and c["tie_flip"] is True
          and AB.compare_answers(a, {"type": "noul", "noul": 0.4,
                                     "diagnostics": {}})["argmax_differs"],
          c)
    rows = [{"set": "k2", "k": 2, "item": "x", "ok": True, "questions": 2,
             "seq": {"wall_s": 4.0}, "batch": {"wall_s": 1.0,
                                                "read_paths": ["batch"]},
             "compare": {"questions": 2, "argmax_differs": 0,
                         "max_abs_p": 0.001, "tie_flips": 0}},
            {"set": "k2", "k": 2, "item": "y", "ok": True, "questions": 2,
             "seq": {"wall_s": 6.0}, "batch": {"wall_s": 2.0,
                                               "read_paths": ["batch"]},
             "compare": {"questions": 2, "argmax_differs": 1,
                         "max_abs_p": 0.01, "tie_flips": 1}}]
    s = AB.summarize(rows)
    k2 = s["sets"]["k2"]
    check("[driver] summarize: speedup, per-question wall, identity counts",
          abs(k2["speedup_total"] - 10.0 / 3.0) < 1e-9
          and k2["argmax_differs"] == 1 and k2["tie_flips"] == 1
          and abs(k2["max_abs_p"] - 0.01) < 1e-9
          and abs(k2["seq_s_per_question"] - 2.5) < 1e-9
          and "k2" in AB.table(s), k2)
    # the gates: --run is refused without --gpu-go, and does not touch a model
    import io as _io
    from contextlib import redirect_stdout
    buf = _io.StringIO()
    with redirect_stdout(buf):
        rc = AB.main(["--run", "--sets", "skill", "--cases",
                      os.path.join(_TMP, "nope.jsonl")])
    check("[driver] --run without --gpu-go is REFUSED (exit 2), nothing sent",
          rc == 2 and "REFUSED" in buf.getvalue(), buf.getvalue()[:200])
    buf = _io.StringIO()
    with redirect_stdout(buf):
        rc = AB.main(["--sets", "skill", "--cases",
                      os.path.join(_TMP, "nope.jsonl")])
    check("[driver] the default is a dry run that prints the plan (exit 0, "
          "no network)", rc == 0 and "THE PLAN" in buf.getvalue()
          and "NOT LOADABLE" in buf.getvalue(), buf.getvalue()[:300])


# ======================================================================= main ===
def main() -> int:
    try:
        derivation()
        identity()
        unread_bound()
        switch_off()
        usage()
        jev()
        turn()
        fallback()
        driver()
    finally:
        batch(False)
    ok = sum(1 for _, o in CHECKS if o)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
