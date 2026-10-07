#!/usr/bin/env python
"""BATCHED READS: every question of a jjava call in ONE engine request
(2026-10-07; design and the wire contract: docs/DECIDE-BATCH.md section 3).

    import decider_batch as B
    with B.burst(state) as b:
        a1 = b.read_many(state, [q1, q2, q3], keep_prefix=True)  # one request
        a2 = b.read_many(state, [q4], keep_prefix=False)         # last: freed
    # or, for one call:  answers = B.read_many(state, questions)

WHAT IT REPLACES. decider_bonsai.read() answers a question with TWO HTTP
reads (the two option orders, ask_one() each: a /v1/chat/completions request,
max_tokens 1, top_logprobs, on the same state prefix) and a call asks its
questions one after another. Measured (the brief of 2026-10-07): ~0.26 s of
fixed overhead per read plus ~2.3 ms per processed token. The engine's
POST /decide-batch (engines/patches/llama-bonsai2-ada/0042) reads ALL the
blocks of a call in one request: the shared state prefix decoded once, each
block on its own sequence, the answer at each block's last token. Questions
never see each other, so an answer is the one the per-read path gives, up to
the engine's batch noise (bench/decider/batch_ab.py measures it).

THE SWITCH. YAMADORI_DECIDER_BATCH=1|on turns it on; DEFAULT OFF, and off,
no code path of decider_bonsai, jev_api or decide_turn changes (the tests
prove the requests and the answers are the same bytes).

RENDERING WITHOUT A /apply-template CALL PER QUESTION. The served chat
template renders a question's messages (decider_bonsai.messages(): system,
user STATE_HEAD + state, user question, assistant ANSWER_LEAD) as WRAPPER
text between the messages' contents. The wrapper is the template's, not the
question's, so it is DERIVED ONCE per (serving model, role structure,
template flags), through the model server's own /apply-template (the door
/tokenize uses):

  1. the structure's messages are rendered with unique SENTINEL contents; the
     rendering split at the sentinels is W0 c0 W1 c1 ... c(n-1) Wn;
  2. the SHARED PREFIX is the first two messages (system + state), cut at a
     message boundary: the first two messages are rendered ALONE with
     add_generation_prompt false, and the full rendering must start with
     exactly that text (so the block, what follows, starts at the next
     message's own start token). Anything else: the structure is DISABLED;
  3. VERIFIED once per structure, three ways, before it is ever used:
     (a) a second, DIFFERENT message set (other contents, edge whitespace the
         template trims, a real ANSWER_LEAD prefill) is rendered by
         /apply-template and must equal the substituted rendering BYTE FOR
         BYTE;
     (b) that rendering passes decider_bonsai.prompt_guard (the think block is
         closed, the prompt ends on the lead);
     (c) /tokenize: tokens(prefix, add_special) + tokens(block, no special)
         must equal tokens(prefix + block, add_special) -- the boundary
         check the engine's verify_tokenization flag repeats server-side on
         the first call of each structure.
     A mismatch disables batching for that structure (cached for the process,
     recorded in stats()) and the caller falls back to the per-read path. A
     transport failure while deriving caches nothing and falls back for the
     call.

  The contents are used exactly as the sequential path sends them, trimmed
  as the template trims (|trim: space, tab, newline, VT, FF, CR). A content
  that is empty after the trim, or still has unicode whitespace at an edge,
  is not batched (the template's trim and ours could differ): that question
  goes the per-read way.

WHAT FALLS BACK, per question (the per-read path answers it, the answer is
never lost): a custom render that does not start with the same two prefix
messages; a question built with prior="content_free" when the caller passes
prior_for; a structure that is disabled; a prefix that differs from the
call's. Per CALL: a cached negative (below); any failure of the engine call.

CAPABILITY. The first /decide-batch call per serving model is the probe. HTTP
404 / 501 (this build, or --decide-seqs unset) / 503 (the pool has no free
cells) -- and 400 (a malformed body or a token id outside the vocabulary:
deterministic for this request shape; a group of more blocks than forks is
no longer an error, the engine chunks it) -- cache a NEGATIVE for
NEGATIVE_S (60 s: the brief of 2026-10-07; the clock is injectable: CLOCK);
every other failure (a timeout, a reset, a body that is not JSON or not the
contract's shape, a missing label logprob) also falls back for that call, is
counted, and ALSO caches the negative: an engine that just failed is not
asked again for NEGATIVE_S, because a timeout would otherwise be paid in full
on every call. The call is answered by the per-read path either way.

THE READ. The request's `token_ids` are the union of every label spelling's
id (decider_bonsai.spelling_ids), `top_logprobs` 20 (the server's own default,
decider_bonsai.FIRST_K). Each block's top list handed to read_labels /
case_variants is the engine's top entries plus the requested ids' exact
log-probabilities (a requested id outside the top gets its spelling as its
token text), so every label spelling is READ EXACTLY: `exact` is true and
there is no K re-read. The one difference from the per-read path is that
spellings the per-read path leaves unread beyond its top K (bounded by its
`unread_bound`) are read here.

USAGE (jev_api.usage_of). Each read reports prompt_tokens = prefix tokens +
block tokens; cached_tokens = the prefix's resident tokens (response.prefix.
reused for the call's first read, prefix.tokens for every other); processed =
the block's `processed` (the engine charges a group's shared head, decoded once, to the
group's FIRST block: docs/DECIDE-BATCH.md section 3), plus prefix.processed on the call's
first read. So input_tokens = the first read's reused prefix + every read's processed
tokens, nothing counted twice, and output_tokens = the reads. `timings.processed_tokens`
is recorded beside the accounted total (`batch.accounted_processed`).

BURSTS. A burst is the questions asked on one state: its calls keep the
prefix resident (keep_prefix true) and the last one, or the burst's close,
frees it ({"release": true}); a failure releases too. The engine holds the
prefix for the burst, so a second stage (Jev's two-stage rounds) reads on the
resident prefix.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decider_bonsai as D  # noqa: E402

ENV = "YAMADORI_DECIDER_BATCH"
PATH = "/decide-batch"
TOP_LOGPROBS = D.FIRST_K
PREFIX_MESSAGES = 2
MAX_TOKEN_IDS = 256                 # the contract: token_ids <= 256
# The brief of 2026-10-07 (operator's coordinator): "cache a negative for 60 s
# (retry after)". Every failure of an engine call sets it (module doc).
NEGATIVE_S = 60.0
NEGATIVE_STATUS = (400, 404, 501, 503)
CLOCK = time.monotonic              # injectable: the tests move it
_TRIM = " \t\n\v\f\r"               # what the served template's |trim strips

_LOCK = threading.RLock()
_STRUCTS: dict = {}                 # structure key -> structure | disabled
_NEGATIVE: dict = {}                # model -> {until, why, status}
_STATS: dict = {}
_LOCAL = threading.local()
_NONCE = hashlib.sha1(f"{os.getpid()}:{time.time_ns()}".encode()
                      ).hexdigest()[:10]


def enabled() -> bool:
    return D.batch_on()


def _fresh_stats() -> dict:
    return {"calls": 0, "engine_requests": 0, "batched_calls": 0,
            "batched_questions": 0, "batched_reads": 0,
            "sequential_questions": 0, "fallback_calls": 0,
            "failures": 0, "negative_cached": 0, "releases": 0,
            "release_failures": 0, "fallback_reasons": {},
            "structures": {}, "last_error": None}


_STATS = _fresh_stats()


def stats() -> dict:
    """A copy of this process's batching record: calls, engine requests,
    batched / sequential questions, fallback reasons, the structures derived
    (ok / why disabled), the last error, the negatives now cached."""
    with _LOCK:
        out = json.loads(json.dumps(_STATS))
        now = CLOCK()
        out["negative"] = {m: {"remaining_s": round(max(v["until"] - now, 0.0), 3),
                               "why": v["why"], "status": v.get("status")}
                           for m, v in _NEGATIVE.items() if v["until"] > now}
        return out


def reset() -> None:
    """Forget every derived structure, negative and counter (tests; a model
    swap does not need it: everything is keyed by the serving model)."""
    global _STATS
    with _LOCK:
        _STRUCTS.clear()
        _NEGATIVE.clear()
        _STATS = _fresh_stats()


def last_call() -> dict | None:
    """The note of this thread's last engine call (None when none)."""
    return getattr(_LOCAL, "last", None)


def _count(key: str, n: int = 1) -> None:
    with _LOCK:
        _STATS[key] = _STATS.get(key, 0) + n


def _why(why: str) -> None:
    with _LOCK:
        r = _STATS["fallback_reasons"]
        r[why[:120]] = r.get(why[:120], 0) + 1


class _Unbatchable(Exception):
    """This question (or structure) cannot go in the batch: the per-read
    path answers it."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


class _Transient(Exception):
    """The derivation could not reach the model server: nothing is cached,
    the call goes the per-read way."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


class _CallFailed(Exception):
    """The engine call failed (kind: not_supported | failed)."""

    def __init__(self, kind: str, why: str, status: int | None = None,
                 maybe_resident: bool = True):
        super().__init__(why)
        self.kind, self.why, self.status = kind, why, status
        self.maybe_resident = maybe_resident


# ============================================================ the rendering ==
def _trim(s: str) -> str:
    return s.strip(_TRIM)


def _flags() -> dict:
    return {"enable_thinking": False}


def _guard_passes(e: Exception) -> None:
    """max_mode's own refusals (a model that must not be loaded now) and the
    decider's are the caller's, not a transport failure: re-raised."""
    import max_mode
    if isinstance(e, (max_mode.ModelAtCapacity, D.DeciderUnavailable)):
        raise e


def _apply(upstream, msgs: list[dict], *, generation_prompt: bool | None = None
          ) -> str:
    """The served template's rendering of `msgs` (/apply-template): thinking
    off as decider_bonsai.body() sends it. `generation_prompt` False renders
    a list that ends on a user message with no generation prompt."""
    payload = {"messages": msgs, "chat_template_kwargs": _flags(),
               "enable_thinking": False}
    if generation_prompt is False:
        payload["add_generation_prompt"] = False
    try:
        d = upstream("/apply-template", payload)
    except urllib.error.HTTPError as e:
        raise _Unbatchable(f"/apply-template answered {e.code}") from e
    except Exception as e:                                       # noqa: BLE001
        _guard_passes(e)
        raise _Transient(f"/apply-template: {type(e).__name__}: {e}"[:200]
                         ) from e
    p = d.get("prompt") if isinstance(d, dict) else None
    if not isinstance(p, str):
        raise _Unbatchable("/apply-template returned no prompt")
    return p


def _tokens(upstream, text: str, special: bool) -> list:
    try:
        d = upstream("/tokenize", {"content": text, "add_special": special,
                                   "parse_special": True})
    except urllib.error.HTTPError as e:
        raise _Unbatchable(f"/tokenize answered {e.code}") from e
    except Exception as e:                                       # noqa: BLE001
        _guard_passes(e)
        raise _Transient(f"/tokenize: {type(e).__name__}: {e}"[:200]) from e
    toks = d.get("tokens") if isinstance(d, dict) else None
    if not isinstance(toks, list):
        raise _Unbatchable("/tokenize returned no tokens")
    return [t["id"] if isinstance(t, dict) else t for t in toks]


def _verify_contents(roles: tuple) -> list[str]:
    """A message set that is NOT the sentinel set and NOT a question the
    decider asks: other text, edge whitespace the template trims, a real
    ANSWER_LEAD as the prefill."""
    out = []
    for i, r in enumerate(roles):
        if r == "assistant":
            out.append(D.ANSWER_LEAD if i == len(roles) - 1
                       else f"Answer: V{i}")
        elif r == "system":
            out.append("  Verify system éè {{x}} <b>\n")
        else:
            out.append(f"\n  Verify {i} — {{%x%}} <|not_a_token|> line"
                       f"\n\nsecond line  \n")
    return out


class Structure:
    """What the served template does around a message set of one role
    structure: the wrapper pieces and the cut between prefix and block."""

    def __init__(self, roles: tuple, k: int, pre_w: list, pre_tail: str,
                 blk_head: str, blk_w: list):
        self.roles, self.k = roles, k
        self.pre_w, self.pre_tail = pre_w, pre_tail
        self.blk_head, self.blk_w = blk_head, blk_w
        self.engine_verified = False

    def prefix(self, contents: list[str]) -> str:
        return "".join(w + _trim(c) for w, c in
                       zip(self.pre_w, contents[:self.k])) + self.pre_tail

    def block(self, contents: list[str]) -> str:
        return self.blk_head + "".join(
            _trim(c) + w for c, w in zip(contents[self.k:], self.blk_w))


def _derive(roles: tuple, upstream) -> Structure:
    """Derive and verify the structure (module doc). Raises _Unbatchable
    (cached by the caller) or _Transient."""
    n = len(roles)
    if n < 3 or roles[0] != "system" or roles[1] != "user" \
            or roles[-1] != "assistant":
        raise _Unbatchable(f"roles {list(roles)}: batching needs system, "
                           "user (the state) ... assistant (the prefill)")
    k = PREFIX_MESSAGES
    sent = [f"JJB{_NONCE}S{i}E" for i in range(n)]
    msgs = [{"role": r, "content": s} for r, s in zip(roles, sent)]
    full = _apply(upstream, msgs)
    alone = _apply(upstream, msgs[:k], generation_prompt=False)
    idx = []
    for s in sent:
        if full.count(s) != 1:
            raise _Unbatchable(f"a sentinel occurs {full.count(s)} times in "
                               "the rendering (the template drops or repeats "
                               "a message)")
        idx.append(full.index(s))
    if idx != sorted(idx):
        raise _Unbatchable("the template reorders the messages")
    end = [i + len(s) for i, s in zip(idx, sent)]
    if not full.startswith(alone):
        raise _Unbatchable("the first two messages rendered alone are not "
                           "the start of the full rendering (the prefix "
                           "cannot be cut at a message boundary)")
    if not (end[k - 1] <= len(alone) <= idx[k]):
        raise _Unbatchable("the alone rendering does not end between the "
                           "state message and the next message")
    w = [full[:idx[0]]]
    for i in range(1, n):
        w.append(full[end[i - 1]:idx[i]])
    w.append(full[end[n - 1]:])
    st = Structure(roles, k, w[:k], full[end[k - 1]:len(alone)],
                   full[len(alone):idx[k]], w[k + 1:])
    # --- (a) a second, different message set, byte for byte
    vc = _verify_contents(roles)
    vm = [{"role": r, "content": c} for r, c in zip(roles, vc)]
    want = _apply(upstream, vm)
    p_text, b_text = st.prefix(vc), st.block(vc)
    if p_text + b_text != want:
        raise _Unbatchable("the substituted rendering of a second message "
                           "set differs from /apply-template's (the "
                           "template is not wrapper text around contents)")
    # --- (b) the think block is closed and the prompt ends on the lead
    last_user = max(i for i, r in enumerate(roles) if r == "user")
    g = D.prompt_guard(want, after=_trim(vc[last_user]))
    if not g["ok"]:
        raise _Unbatchable(f"prompt guard: {g['why']}")
    # --- (c) tokenization at the boundary
    tp = _tokens(upstream, p_text, True)
    tb = _tokens(upstream, b_text, False)
    tw = _tokens(upstream, p_text + b_text, True)
    if tp + tb != tw:
        raise _Unbatchable("tokens(prefix) + tokens(block) != tokens(prefix "
                           "+ block): the tokenizer merges across the "
                           "boundary")
    return st


def structure(msgs: list[dict], upstream, model: str) -> Structure:
    """The (cached) structure for `msgs`' roles on `model`; raises
    _Unbatchable for a disabled one (cached) or _Transient."""
    for m in msgs:
        if not isinstance(m, dict) or set(m) - {"role", "content"} \
                or not isinstance(m.get("content"), str):
            raise _Unbatchable("a message carries more than role and a "
                               "string content")
    roles = tuple(m["role"] for m in msgs)
    key = f"{model}|{'/'.join(roles)}|{json.dumps(_flags(), sort_keys=True)}"
    with _LOCK:
        hit = _STRUCTS.get(key)
    if hit is None:
        try:
            hit = _derive(roles, upstream)
        except _Unbatchable as e:
            hit = {"disabled": e.why}
            with _LOCK:
                _STRUCTS[key] = hit
                _STATS["structures"][key] = {"ok": False, "why": e.why}
            raise
        with _LOCK:
            _STRUCTS[key] = hit
            _STATS["structures"][key] = {"ok": True}
    if isinstance(hit, dict):
        raise _Unbatchable(hit["disabled"])
    hit.key = key                                        # for the engine check
    return hit


def _disable(key: str, why: str) -> None:
    with _LOCK:
        _STRUCTS[key] = {"disabled": why}
        _STATS["structures"][key] = {"ok": False, "why": why}


# ============================================================== the plan ======
def _exclusion(q: dict, exclude) -> list:
    if isinstance(exclude, dict):
        exclude = exclude.get(q["name"]) or ()
    return sorted({str(x) for x in exclude or ()} & set(q["keys"]))


def _plan(state: str, q: dict, ex: list, *, render, upstream, model: str
          ) -> dict:
    """One typed question's orders as blocks. Raises _Unbatchable (this
    question goes the per-read way), _Transient, or DeciderUnavailable (the
    same refusals ask_one makes: the think-block guard, a label with no
    single-token spelling)."""
    allowed = [x for x in q["keys"] if x not in ex]
    if not allowed:
        raise D.DeciderUnavailable(
            "ALL_EXCLUDED", f"every option of {q['name']!r} is excluded",
            False, "the caller: exclude fewer options")
    orders = []
    for c in D.rendered_orders(q):
        ids = {lab: D.spelling_ids(lab, upstream) for lab in c["labels"]}
        msgs = (render(c) if render else None) or D.messages(state, c)
        D.guard_body(D.body(state, c, None, TOP_LOGPROBS, model, True, msgs))
        st = structure(msgs, upstream, model)
        contents = [m["content"] for m in msgs]
        for t in map(_trim, contents):
            if not t or t != t.strip():
                raise _Unbatchable("a content is empty or has unicode "
                                   "whitespace at an edge after the trim")
        orders.append({"c": c, "ids": ids, "struct": st, "contents": contents,
                       "prefix": st.prefix(contents),
                       "block": st.block(contents)})
    return {"q": q, "ex": ex, "allowed": allowed, "orders": orders}


# ============================================================== the call ======
def _negative_now(model: str) -> dict | None:
    with _LOCK:
        v = _NEGATIVE.get(model)
        if v and v["until"] > CLOCK():
            return dict(v, remaining_s=round(v["until"] - CLOCK(), 3))
        if v:
            del _NEGATIVE[model]
    return None


def _set_negative(model: str, why: str, status: int | None) -> None:
    with _LOCK:
        _NEGATIVE[model] = {"until": CLOCK() + NEGATIVE_S, "why": why[:200],
                            "status": status}
    _count("negative_cached")


def _post(upstream, payload: dict, timeout: float) -> dict:
    """One /decide-batch request through the one door; a _CallFailed for
    anything that is not the contract's 200."""
    _count("engine_requests")
    try:
        d = upstream(PATH, payload, timeout)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:                                        # noqa: BLE001
            pass
        kind = "not_supported" if e.code in NEGATIVE_STATUS else "failed"
        # nothing is held after a refusal the server made before doing any
        # work (400 / 404 / 501 / 503); another 5xx may have left the prefix
        raise _CallFailed(kind, f"HTTP {e.code} {detail}".strip(), e.code,
                          maybe_resident=e.code >= 500
                          and e.code not in NEGATIVE_STATUS) from e
    except (socket.timeout, TimeoutError) as e:
        # no release is attempted after a timeout or an unreachable server
        # (maybe_resident False): it could not be answered, and a release
        # would wait on it again; the next call's prefix replaces a stale one
        raise _CallFailed("failed", f"timeout after {timeout} s: {e}",
                          maybe_resident=False) from e
    except urllib.error.URLError as e:
        raise _CallFailed("failed", f"unreachable: {e.reason}",
                          maybe_resident=False) from e
    except ValueError as e:                    # json.JSONDecodeError
        raise _CallFailed("failed", f"the body is not JSON: {e}"[:200]) from e
    except Exception as e:                                       # noqa: BLE001
        # max_mode's guards (ModelAtCapacity ...) are the caller's refusals,
        # not the engine's: let them through
        _guard_passes(e)
        raise _CallFailed("failed", f"{type(e).__name__}: {e}"[:200]) from e
    return d


def _num(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise _CallFailed("failed", f"{what} is not a number: {x!r}"[:120])
    return float(x)


def _check_response(d, groups: list, verify: bool) -> None:
    if not isinstance(d, dict):
        raise _CallFailed("failed", "the response is not an object")
    p = d.get("prefix")
    if not isinstance(p, dict) or any(
            not isinstance(p.get(f), int) or isinstance(p.get(f), bool)
            for f in ("tokens", "reused", "processed")):
        raise _CallFailed("failed", "no prefix {tokens, reused, processed}")
    gs = d.get("groups")
    if not isinstance(gs, list) or len(gs) != len(groups):
        raise _CallFailed("failed", "groups do not match the request")
    for g, want in zip(gs, groups):
        if not isinstance(g, list) or len(g) != len(want):
            raise _CallFailed("failed", "a group does not match the request")
        for b in g:
            if not isinstance(b, dict) \
                    or not isinstance(b.get("tokens"), int) \
                    or not isinstance(b.get("processed"), int) \
                    or not isinstance(b.get("top_logprobs"), list) \
                    or not isinstance(b.get("logprobs"), dict):
                raise _CallFailed("failed", "a block lacks tokens, "
                                  "processed, top_logprobs or logprobs")


def _top_for(block: dict, ids: dict) -> list:
    """The block's top list for read_labels / case_variants: the engine's
    top entries plus every requested id of THIS question (a spelling outside
    the top gets its spelling as its token text)."""
    top, seen = [], set()
    for t in block["top_logprobs"]:
        if not isinstance(t, dict) or "id" not in t or "logprob" not in t:
            raise _CallFailed("failed", "a top entry lacks id or logprob")
        _num(t["logprob"], "a top logprob")
        top.append(t)
        seen.add(int(t["id"]))
    lp = block["logprobs"]
    for spell in ids.values():
        for sp, tid in spell.items():
            if tid in seen:
                continue
            v = lp.get(str(tid))
            if v is None:
                raise _CallFailed("failed", f"no logprob for requested id "
                                  f"{tid}")
            top.append({"id": tid, "token": sp, "logprob": _num(v, "logprob")})
            seen.add(tid)
    return top


def _answers(plans: list, resp: dict, wall_ms: float, t0: float,
             note: dict) -> list:
    """The per-question answers (read()'s shape) from the engine response."""
    prefix = resp["prefix"]
    n_reads = sum(len(p["orders"]) for p in plans)
    out, first = [], True
    accounted = int(prefix["processed"])
    for plan, group in zip(plans, resp["groups"]):
        q, ex, allowed = plan["q"], plan["ex"], plan["allowed"]
        keys = q["keys"]
        temp = D.temperature_of(q["name"])
        per = []
        for oi, (o, blk) in enumerate(zip(plan["orders"], group)):
            top = _top_for(blk, o["ids"])
            r = D.read_labels(top, o["ids"])
            # the engine charges a group's shared head to its FIRST block's `processed` (docs/DECIDE-BATCH.md 3)
            own = int(blk["processed"])
            proc = own + (int(prefix["processed"]) if first else 0)
            cached = int(prefix["reused"]) if first else int(prefix["tokens"])
            accounted += own
            first = False
            read = {"k": TOP_LOGPROBS, "ms": None, "prompt_n": proc,
                    "cache_n": cached, "prompt_ms": None}
            a = D._ask_result(
                o["c"], o["ids"], r, top, [read], TOP_LOGPROBS,
                {"prompt_tokens": int(prefix["tokens"]) + int(blk["tokens"])},
                t0)
            a["total_ms"] = round(wall_ms / n_reads, 1)
            per.append(D._order_record(q, o["c"], a, keys=keys,
                                       allowed=allowed, temp=temp,
                                       prior_for=None, use_prior=False))
        ans = D._assemble(q, per, keys=keys, ex=ex, allowed=allowed,
                          temp=temp, use_prior=False, regime="batch", t0=t0)
        ans["diagnostics"]["ms"] = round(wall_ms * len(plan["orders"])
                                         / n_reads, 1)
        ans["diagnostics"]["read_path"] = "batch"
        ans["diagnostics"]["batch"] = note
        out.append(ans)
    note["accounted_processed"] = accounted
    return out


def read_many(state: str, questions: list[dict], *, slot: int | None = None,
              post=None, upstream=None, timeout: float = 120,
              cache: bool | None = None, render=None, exclude=(),
              prior_for=None, keep_prefix: bool = False,
              burst: "Burst | None" = None) -> list[dict]:
    """Every question over `state`, answered EXACTLY as [D.read(state, q,
    ...) for q in questions] would (same Jev-shaped answers and diagnostics
    keys), through ONE /decide-batch request where the engine allows, plus
    diagnostics.read_path "batch" | "sequential" and, batched,
    diagnostics.batch (the single engine call's note); a question that
    fell back says why in diagnostics.batch_fallback.

    exclude: keys that are not answers (applied to every question), or a
      {question name: keys} map. render / prior_for as D.read().
    keep_prefix: more questions follow on this state (the prefix stays
      resident for them); the last call of a burst passes False.
    burst: the Burst this call belongs to (it tracks the resident prefix
      and releases it on close).
    Switch off: this is [D.read(...) for q in questions], nothing more."""
    post, upstream = post or D._post_default, upstream or D._upstream
    qs = [D.typed(q) for q in questions]

    def seq(i: int, why: str) -> dict:
        q = qs[i]
        a = D.read(state, q, slot=slot, post=post, upstream=upstream,
                   timeout=timeout, cache=cache, render=render,
                   exclude=_exclusion(q, exclude), prior_for=prior_for)
        if enabled():
            a["diagnostics"]["read_path"] = "sequential"
            a["diagnostics"]["batch_fallback"] = why
            _count("sequential_questions")
            _why(why)
        return a

    if not qs:
        return []
    if not enabled():
        return [seq(i, "") for i in range(len(qs))]
    _count("calls")
    model = D.model_name()
    neg = _negative_now(model)
    if neg:
        _count("fallback_calls")
        return [seq(i, f"engine batch off for {neg['remaining_s']} s more: "
                    f"{neg['why']}") for i in range(len(qs))]
    # ---- plan: every question's blocks, before anything is sent
    plans: dict = {}
    why: dict = {}
    try:
        for i, q in enumerate(qs):
            ex = _exclusion(q, exclude)
            if q.get("prior") == "content_free" and prior_for is not None:
                why[i] = "the question needs the content-free prior"
                continue
            try:
                plans[i] = _plan(state, q, ex, render=render,
                                 upstream=upstream, model=model)
            except _Unbatchable as e:
                why[i] = e.why
    except _Transient as e:
        _count("fallback_calls")
        return [seq(i, f"derivation: {e.why}") for i in range(len(qs))]
    # one prefix per call: a question whose prefix differs goes its own way
    prefix_text = next((p["orders"][0]["prefix"] for p in plans.values()), None)
    for i in list(plans):
        if any(o["prefix"] != prefix_text for o in plans[i]["orders"]):
            why[i] = "its prefix differs from the call's"
            del plans[i]
    if not plans:
        _count("fallback_calls")
        return [seq(i, why[i]) for i in range(len(qs))]
    idx = sorted(plans)
    groups = [[o["block"] for o in plans[i]["orders"]] for i in idx]
    token_ids = sorted({tid for i in idx for o in plans[i]["orders"]
                        for spell in o["ids"].values()
                        for tid in spell.values()})
    structs = {o["struct"].key: o["struct"] for i in idx
               for o in plans[i]["orders"]}
    verify = any(not s.engine_verified for s in structs.values())
    if len(token_ids) > MAX_TOKEN_IDS:
        _count("fallback_calls")
        return [seq(i, why.get(i) or f"{len(token_ids)} label ids exceed "
                    f"the contract's {MAX_TOKEN_IDS}") for i in range(len(qs))]
    payload = {"prefix": prefix_text, "groups": groups,
               "token_ids": token_ids, "top_logprobs": TOP_LOGPROBS,
               "keep_prefix": bool(keep_prefix),
               "verify_tokenization": bool(verify)}
    t0, w0 = time.time(), time.perf_counter()
    answers: dict = {}
    try:
        resp = _post(upstream, payload, timeout)
        _check_response(resp, groups, verify)
        if verify:
            bad = [(gi, bi) for gi, g in enumerate(resp["groups"])
                   for bi, b in enumerate(g) if b.get("tokenization_ok")
                   is not True]
            if bad:
                for gi, bi in bad:
                    s = plans[idx[gi]]["orders"][bi]["struct"]
                    _disable(s.key, "the engine's tokenization check failed "
                             "(prefix + block != whole)")
                raise _CallFailed("tokenization", "the engine's tokenization "
                                  "check failed", maybe_resident=True)
            for s in structs.values():
                s.engine_verified = True
        wall_ms = (time.perf_counter() - w0) * 1000
        note = {"requests": 1, "groups": len(groups),
                "blocks": sum(len(g) for g in groups),
                "questions": [qs[i]["name"] for i in idx],
                "token_ids": len(token_ids), "keep_prefix": bool(keep_prefix),
                "verify_tokenization": bool(verify),
                "prefix": resp["prefix"], "timings": resp.get("timings"),
                "seqs": resp.get("seqs"), "ms": round(wall_ms, 1)}
        got = _answers([plans[i] for i in idx], resp, wall_ms, t0, note)
    except _CallFailed as e:
        _count("failures")
        _count("fallback_calls")
        with _LOCK:
            _STATS["last_error"] = {"kind": e.kind, "why": e.why,
                                    "status": e.status}
        if e.kind != "tokenization":
            _set_negative(model, e.why, e.status)
        if burst is not None:
            burst._after_call(upstream, resident=False,
                              maybe=e.maybe_resident)
        return [seq(i, why.get(i) or f"the engine call failed ({e.kind}): "
                    f"{e.why}"[:200]) for i in range(len(qs))]
    except BaseException:
        if burst is not None:
            burst._after_call(upstream, resident=False, maybe=True)
        raise
    for i, a in zip(idx, got):
        answers[i] = a
    _count("batched_calls")
    _count("batched_questions", len(idx))
    _count("batched_reads", note["blocks"])
    _LOCAL.last = note
    if burst is not None:
        burst._after_call(upstream, resident=bool(keep_prefix), maybe=False,
                          note=note)
    # the questions that were not batched, the per-read way
    for i in range(len(qs)):
        if i not in answers:
            answers[i] = seq(i, why.get(i, "not batched"))
    return [answers[i] for i in range(len(qs))]


def release(upstream=None, timeout: float = 30) -> dict:
    """Free the engine's resident prefix ({"release": true}). Never raises."""
    upstream = upstream or D._upstream
    _count("releases")
    try:
        d = upstream(PATH, {"release": True}, timeout)
        return {"released": True, "response": d if isinstance(d, dict)
                else None}
    except Exception as e:                                       # noqa: BLE001
        _count("release_failures")
        return {"released": False,
                "error": f"{type(e).__name__}: {e}"[:200]}


class Burst:
    """The calls made on one state: each keeps the prefix resident for the
    next, the last (or the close) frees it; any failure frees it too."""

    def __init__(self, state: str | None = None, *, upstream=None,
                 timeout: float = 30):
        self.state, self.upstream, self.timeout = state, upstream, timeout
        self.resident = False
        self.maybe = False
        self.calls: list[dict] = []
        self.release_note: dict | None = None

    def read_many(self, state: str, questions: list[dict], *,
                  keep_prefix: bool = True, **kw) -> list[dict]:
        """B.read_many(..., burst=self); keep_prefix defaults True (more may
        follow): the last call passes False, or the close frees it."""
        return read_many(state, questions, keep_prefix=keep_prefix,
                         burst=self, **kw)

    def _after_call(self, upstream, *, resident: bool, maybe: bool,
                    note: dict | None = None) -> None:
        self.upstream = self.upstream or upstream
        self.resident = resident
        self.maybe = maybe
        if note is not None:
            self.calls.append(note)

    def release(self) -> dict | None:
        """Free the prefix if a call may have left it. Never raises."""
        if not (self.resident or self.maybe):
            return None
        self.resident = self.maybe = False
        self.release_note = release(self.upstream, self.timeout)
        return self.release_note

    def __enter__(self) -> "Burst":
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False


def burst(state: str | None = None, *, upstream=None, timeout: float = 30
          ) -> Burst:
    """`with burst(state) as b:` -- see Burst."""
    return Burst(state, upstream=upstream, timeout=timeout)
