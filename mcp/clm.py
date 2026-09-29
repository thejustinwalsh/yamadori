#!/usr/bin/env python
"""CLM, the contrastive decision model: a state and a closed set of options
in, one probability per option out. The client every caller uses.

    import clm
    p = clm.decide(state_text, ["option text", ...])            # list[float]
    ps = clm.decide_many(state_text, [[...], [...]])            # one state encoding
    d = clm.decide_detail(state_text, options, instructions=Q)  # + logits, timings

WHAT IT IS (docs/CLM.md; the assessment is docs/CLM-EVAL.md)

CLM-v0.1-8B = a frozen Qwen3-8B used as an embedder (the LAST token's final
hidden state, L2-normalised) + two small MLP heads (mcp/clm_heads.py). The
state and every option are embedded SEPARATELY; an option's score is
`scale * cos(state_head(state), action_head(option))` and the answer is the
softmax over the options given. It cannot generate, and its probabilities
are relative to THIS option set: never threshold them across different
states or option sets (the rule docs/CLM-EVAL.md section 3 carries over from
Laya).

WHERE IT RUNS

The encoder is llama-swap's `clm-encoder` (config.yaml): Qwen3-8B, our own
GGUF (models/manifest.yaml `qwen3-8b-clm-*`), `--embeddings --pooling last`,
`-c 2048` (CLM's window), on the A4000 by UUID. Every request goes through
`gpu_room.use("clm-encoder")` -- the A4000's room: a loaded encoder takes a
lease, an unloaded one gets room made first or NoRoom -- exactly as
code_search's embeddings do. Inside a chat turn, `gpu_room.fail_fast()`
bounds the wait (the proxy sets it). The heads run here, in numpy, on the
CPU (microseconds; no torch on the request path).

THE ONE DOOR. Generation goes through mcp/model.py; this model does not
generate, and its requests go to llama-swap's /v1/embeddings the way
code_search's do -- one function here (`HttpEncoder._post`), never :1234 (no
admission lane: nothing here generates, and a lane taken inside a request
that already holds one deadlocks -- model.py explains). Lane-safe: one
encoder request at a time per process (`_LANE`, the server has one slot),
bounded by LANE_WAIT_S; a busy lane is ClmUnavailable(retryable).

INPUTS, EXACTLY AS THE HEADS WERE TRAINED (Contrastive-LM/CLM @ bb42c6c5,
src/clm/schema.py, train/embed_utils.py):

  * the state head sees `state + "\\n\\n" + instructions` -- context first,
    the question last (schema.state_text). `instructions` is optional here:
    a caller whose state text already ends with its question passes none.
  * each option is embedded as its own text, nothing prefixed.
  * tokenised with the pinned Qwen3-8B tokenizer.json, add_special_tokens
    False (no BOS/EOS -- G2 in docs/CLM.md: identical to llama-server's own
    tokenisation of every fidelity text), and sent AS TOKEN IDS so the
    truncation below is exact.
  * at most MAX_TOKENS - 1 = 2047 tokens (Recipe.cap). A longer state is cut
    WITHOUT reordering the caller's evidence: given a list of evidence
    pieces, whole pieces are dropped from the front (keep="tail", CLM's own
    left truncation -- the question at the end survives) or from the back
    (keep="head", for a list ordered most important first), then the
    boundary piece is cut at token level; the instructions are never cut.

CACHING. Option texts are embedded once: `index/clm_actions.npz` maps
sha256(text) -> the encoder's 4096-d vector, under the encoder fingerprint
(the GGUF's sha256 from models/manifest.yaml; another encoder starts an
empty cache). A skill's trigger text changes with its revision, so its key
does too: the cache is per skill revision without knowing about skills.
Projections through the heads are memoised per process. The state is
encoded once per call, whatever the number of questions (decide_many).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import clm_heads  # noqa: E402

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("YAMADORI_CLM_MODEL", "clm-encoder")
# A measurement run's own llama-server (bench/clm/standalone.py), instead of
# llama-swap: its caller holds the A4000's room, so no gpu_room lease here.
URL = os.environ.get("YAMADORI_CLM_URL") or None
MODELS_DIR = os.environ.get("YAMADORI_MODELS_DIR",
                            "C:/Users/jwals/textgen/user_data/models")
TOKENIZER_PATH = os.environ.get(
    "YAMADORI_CLM_TOKENIZER", os.path.join(MODELS_DIR, "Qwen3-8B",
                                           "tokenizer.json"))
ACTIONS_PATH = os.environ.get("YAMADORI_CLM_ACTIONS",
                              os.path.join(ROOT, "index", "clm_actions.npz"))
# The served GGUF's identity (models/manifest.yaml). Cached vectors are only
# valid for the encoder that made them.
ENCODER_ID = os.environ.get(
    "YAMADORI_CLM_ENCODER_ID",
    "Qwen3-8B-Q8_0.gguf@1633fc3e60add8eebb2aede44d8ba6a82872c339c673da6dc6efaa6a86ec860a")

MAX_TOKENS = 2048          # CLM's window (vLLM --max-model-len, -c here)
CAP = MAX_TOKENS - 1       # train/embed_utils.Recipe: max_len - 1
BATCH = 16                 # inputs per /v1/embeddings request (a choice)
TIMEOUT = float(os.environ.get("YAMADORI_CLM_TIMEOUT", "120"))
LANE_WAIT_S = float(os.environ.get("YAMADORI_CLM_LANE_WAIT", "30"))
CACHE_MAX = int(os.environ.get("YAMADORI_CLM_CACHE_MAX", "50000"))
HIDDEN = clm_heads.HIDDEN

# The skill-selection question and the "no skill" option, as used by the
# fidelity set (bench/clm/fidelity.py) and the fine-tune data
# (bench/clm/finetune_data.py). Wording is a choice, not a measurement.
INSTRUCTIONS = "Which skill, if any, should be loaded to handle this request?"
NONE_OPTION = ("None of these skills applies; handle the request without "
               "loading a skill.")


class ClmUnavailable(RuntimeError):
    """The encoder could not be asked. Carries the situation, whether a retry
    can help (a fact), and a remedy with an owner (AGENTS.md "Failure
    returns carry the next step")."""

    def __init__(self, code: str, situation: str, retryable: bool,
                 remedy: str):
        super().__init__(f"{code}: {situation}")
        self.code, self.situation = code, situation
        self.retryable, self.remedy = retryable, remedy

    def facts(self) -> dict:
        return {"code": self.code, "situation": self.situation,
                "retryable": self.retryable, "remedy": self.remedy}


# ------------------------------------------------------------ tokenizer --
class Tokenizer:
    """The pinned Qwen3-8B tokenizer.json, add_special_tokens=False."""

    def __init__(self, path: str = TOKENIZER_PATH):
        from tokenizers import Tokenizer as _T
        self.path = path
        self._t = _T.from_file(path)
        self._space = self._t.encode(" ", add_special_tokens=False).ids

    def ids(self, text: str) -> list[int]:
        x = self._t.encode(text or "", add_special_tokens=False).ids
        return list(x) if x else list(self._space)

    def count(self, text: str) -> int:
        return len(self.ids(text))


_tok: Tokenizer | None = None
_tok_lock = threading.Lock()


def tokenizer() -> Tokenizer:
    global _tok
    with _tok_lock:
        if _tok is None:
            if not os.path.exists(TOKENIZER_PATH):
                raise ClmUnavailable(
                    "CLM_NO_TOKENIZER",
                    f"the Qwen3-8B tokenizer is not at {TOKENIZER_PATH}",
                    False, "operator: fetch models/manifest.yaml "
                           "`qwen3-8b-hf` (scripts/fetch_models.py) or set "
                           "YAMADORI_CLM_TOKENIZER")
            _tok = Tokenizer(TOKENIZER_PATH)
        return _tok


# ------------------------------------------------------------- the state --
def state_text(state, instructions: str | None = None) -> str:
    """schema.state_text for text: context first, the question last."""
    s = "\n\n".join(p.strip() for p in _pieces(state))
    i = (instructions or "").strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def _pieces(state) -> list[str]:
    if state is None:
        return []
    if isinstance(state, str):
        return [state] if state.strip() else []
    return [p for p in state if isinstance(p, str) and p.strip()]


def fit_state(state, instructions: str | None = None, *, keep: str = "tail",
              tok: Tokenizer | None = None, cap: int = CAP
              ) -> tuple[list[int], dict]:
    """(token ids of the state text, what was cut). `state` is a string or a
    list of evidence pieces in the caller's order, never reordered."""
    if keep not in ("tail", "head"):
        raise ValueError("keep is 'tail' (drop from the front, CLM's own "
                         "truncation) or 'head' (drop from the back)")
    tok = tok or tokenizer()
    pieces = [p.strip() for p in _pieces(state)]
    instr = (instructions or "").strip()
    ids = tok.ids(state_text(pieces, instr))
    info = {"tokens": len(ids), "pieces": len(pieces), "truncated": False}
    if len(ids) <= cap:
        return ids, info
    info.update(truncated=True, tokens_before=len(ids))
    tail = tok.ids("\n\n" + instr) if instr else []
    if len(tail) >= cap:
        raise ValueError(f"the instructions alone are {len(tail)} tokens; "
                         f"the window is {cap}")
    budget = cap - len(tail)
    dropped = 0
    # Whole pieces first, from the end the caller marked least important.
    while len(pieces) > 1 and tok.count("\n\n".join(pieces)) > budget:
        pieces = pieces[1:] if keep == "tail" else pieces[:-1]
        dropped += 1
    ev = tok.ids("\n\n".join(pieces)) if pieces else []
    cut = max(0, len(ev) - budget)
    ev = ev[cut:] if keep == "tail" else ev[:budget]
    ids = ev + tail
    info.update(tokens=len(ids), dropped_pieces=dropped, cut_tokens=cut,
                keep=keep)
    return ids, info


# --------------------------------------------------------------- encoder --
_LANE = threading.Lock()


class HttpEncoder:
    """llama-swap's `clm-encoder` (or any llama-server with its flags):
    token-id lists -> [n, 4096] unit vectors."""

    def __init__(self, url: str | None = None, model: str | None = MODEL,
                 gpu_room_model: str | None = MODEL,
                 upstream: str | None = None, timeout: float = TIMEOUT):
        self.upstream = upstream or UPSTREAM
        self.url = url or f"{self.upstream.rstrip('/')}/v1/embeddings"
        self.model, self.gpu_room_model, self.timeout = (model, gpu_room_model,
                                                         timeout)

    def _post(self, payload: dict) -> dict:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            import cancel
            cancel.check()
        except ImportError:
            pass
        if self.gpu_room_model:
            import gpu_room
            try:
                with gpu_room.use(self.gpu_room_model, upstream=self.upstream):
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        return json.loads(r.read())
            except gpu_room.NoRoom as e:
                f = e.facts() if hasattr(e, "facts") else {}
                raise ClmUnavailable(
                    f.get("code", "A4000_NO_ROOM"), str(e), bool(
                        f.get("retryable", True)),
                    "agent: retry later, or decide without CLM; operator: "
                    "free the A4000 (mcp/gpu_room.py)") from None
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def embed_ids(self, id_lists: Sequence[Sequence[int]]) -> np.ndarray:
        import urllib.error
        out: list[np.ndarray] = []
        if not _LANE.acquire(timeout=LANE_WAIT_S):
            raise ClmUnavailable(
                "CLM_BUSY", f"another CLM encoder request held this "
                f"process's lane for {LANE_WAIT_S:.0f}s", True,
                "agent: retry, or decide without CLM")
        try:
            for i in range(0, len(id_lists), BATCH):
                chunk = [list(map(int, x)) for x in id_lists[i:i + BATCH]]
                body: dict = {"input": chunk, "encoding_format": "float"}
                if self.model:
                    body["model"] = self.model
                try:
                    d = self._post(body)
                except urllib.error.HTTPError as e:
                    msg = e.read().decode("utf-8", "replace")[:300]
                    raise ClmUnavailable(
                        "CLM_HTTP_ERROR", f"{self.url} answered {e.code}: {msg}",
                        e.code >= 500 or e.code == 429,
                        "operator: check llama-swap's `clm-encoder` log "
                        "(config.yaml)") from None
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    raise ClmUnavailable(
                        "CLM_UNREACHABLE", f"{self.url}: {e}", True,
                        "operator: is llama-swap up on :11434 with the "
                        "`clm-encoder` entry (restart after a config change)?"
                    ) from None
                got: list = [None] * len(chunk)
                for row in d.get("data") or []:
                    got[int(row["index"])] = np.asarray(row["embedding"],
                                                        dtype=np.float32)
                if any(g is None or g.shape != (HIDDEN,) for g in got):
                    raise ClmUnavailable(
                        "CLM_BAD_RESPONSE", f"{self.url} returned "
                        f"{len(d.get('data') or [])} rows for {len(chunk)} "
                        f"inputs, or a width other than {HIDDEN}", False,
                        "operator: the entry must serve Qwen3-8B with "
                        "--embeddings --pooling last")
                out.extend(got)
        finally:
            _LANE.release()
        v = clm_heads.l2(np.stack(out))
        # PROTOCOL rule 1: a dead encoder must not look like a live one.
        if not np.isfinite(v).all():
            raise ClmUnavailable("CLM_BAD_RESPONSE", "non-finite embedding",
                                 False, "operator: check the encoder")
        return v

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        tok = tokenizer()
        return self.embed_ids([tok.ids(t)[-CAP:] for t in texts])


_encoder = None
_enc_lock = threading.Lock()


def encoder():
    global _encoder
    with _enc_lock:
        if _encoder is None:
            _encoder = (HttpEncoder(url=URL.rstrip("/") + "/v1/embeddings",
                                    model=None, gpu_room_model=None)
                        if URL else HttpEncoder())
        return _encoder


def set_encoder(enc) -> None:
    """Tests (and the fidelity script) swap the encoder: anything with
    embed_ids(list[list[int]]) -> [n, 4096]."""
    global _encoder
    with _enc_lock:
        _encoder = enc
    _proj.clear()


# ------------------------------------------------------ the action cache --
def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ActionCache:
    """sha256(option text) -> encoder vector, persisted to ACTIONS_PATH under
    ENCODER_ID. Reads merge the file; writes merge again and replace it
    atomically (a lost race costs a re-embed, never a wrong vector)."""

    def __init__(self, path: str = ACTIONS_PATH, encoder_id: str = ENCODER_ID):
        self.path, self.encoder_id = path, encoder_id
        self.mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._mtime = None
        self._read()

    def _read(self) -> None:
        try:
            st = os.stat(self.path)
        except OSError:
            return
        if st.st_mtime == self._mtime:
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                if str(z["encoder"]) != self.encoder_id:
                    self._mtime = st.st_mtime
                    return
                for k, v in zip(z["keys"].tolist(), z["vecs"]):
                    self.mem.setdefault(k, np.asarray(v, dtype=np.float32))
        except Exception:                                        # noqa: BLE001
            return
        self._mtime = st.st_mtime

    def get(self, keys: Iterable[str]) -> dict[str, np.ndarray]:
        with self._lock:
            want = list(keys)
            if any(k not in self.mem for k in want):
                self._read()
            return {k: self.mem[k] for k in want if k in self.mem}

    def put(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        with self._lock:
            self._read()
            for k, v in items.items():
                self.mem[k] = np.asarray(v, dtype=np.float32)
                self.mem.move_to_end(k)
            while len(self.mem) > CACHE_MAX:
                self.mem.popitem(last=False)
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = f"{self.path}.{os.getpid()}.tmp.npz"
                ks = list(self.mem)
                np.savez(tmp, encoder=np.array(self.encoder_id),
                         keys=np.array(ks),
                         vecs=np.stack([self.mem[k] for k in ks]))
                os.replace(tmp, self.path)
                self._mtime = os.stat(self.path).st_mtime
            except OSError:
                pass            # the memory copy still serves this process

    def __len__(self) -> int:
        return len(self.mem)


_cache: ActionCache | None = None
_cache_lock = threading.Lock()


def action_cache() -> ActionCache:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = ActionCache()
        return _cache


def set_action_cache(c: ActionCache | None) -> None:
    global _cache
    with _cache_lock:
        _cache = c
    _proj.clear()


# Projections of option vectors through the action head, per heads identity.
_proj: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
_proj_lock = threading.Lock()


def _option_projections(heads, texts: list[str]) -> tuple[np.ndarray, dict]:
    keys = [text_key(t) for t in texts]
    uniq = list(dict.fromkeys(keys))
    by_key = dict(zip(keys, texts))
    with _proj_lock:
        have = {k: _proj[(heads.namespace, k)] for k in uniq
                if (heads.namespace, k) in _proj}
    need = [k for k in uniq if k not in have]
    info = {"options": len(texts), "unique": len(uniq),
            "projection_hits": len(have), "cache_hits": 0, "encoded": 0,
            "option_encode_ms": 0.0}
    if need:
        cache = action_cache()
        vecs = cache.get(need)
        info["cache_hits"] = len(vecs)
        miss = [k for k in need if k not in vecs]
        if miss:
            t0 = time.perf_counter()
            tok = tokenizer()
            got = encoder().embed_ids([tok.ids(by_key[k])[-CAP:] for k in miss])
            info["option_encode_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            info["encoded"] = len(miss)
            new = dict(zip(miss, got))
            cache.put(new)
            vecs.update(new)
        z = heads.project_actions(np.stack([vecs[k] for k in need]))
        with _proj_lock:
            for k, row in zip(need, z):
                _proj[(heads.namespace, k)] = row
                have[k] = row
            while len(_proj) > CACHE_MAX:
                _proj.popitem(last=False)
    return np.stack([have[k] for k in keys]), info


# -------------------------------------------------------------- decisions --
def decide_detail_many(state, option_lists: Sequence[Sequence[str]], *,
                       instructions: str | None = None, keep: str = "tail",
                       temperature: float = 1.0, heads=None) -> dict:
    """One state encoding, one softmax per option list. Returns
    {probabilities: [[...]], logits: [[...]], state: {tokens, truncated, ...},
     options: {...}, timing: {state_encode_ms, option_encode_ms, heads_ms,
     total_ms}, encoder: ENCODER_ID, heads: namespace}."""
    t_all = time.perf_counter()
    lists = [list(o) for o in option_lists]
    if not lists or any(len(o) < 1 for o in lists):
        raise ValueError("every question needs at least one option")
    if not (0 < temperature <= 100):
        raise ValueError("temperature must be in (0, 100]")
    heads = heads or clm_heads.load()
    ids, sinfo = fit_state(state, instructions, keep=keep)
    t0 = time.perf_counter()
    sv = encoder().embed_ids([ids])
    state_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    zs = heads.project_states(sv)[0]
    flat = [t for o in lists for t in o]
    za, oinfo = _option_projections(heads, flat)
    probs, logits, k = [], [], 0
    for o in lists:
        lg = heads.logits(zs, za[k:k + len(o)], temperature)
        k += len(o)
        logits.append([float(x) for x in lg])
        probs.append([float(x) for x in clm_heads.softmax(lg)])
    heads_ms = (time.perf_counter() - t1) * 1000 - oinfo["option_encode_ms"]
    return {"probabilities": probs, "logits": logits, "state": sinfo,
            "options": oinfo,
            "timing": {"state_encode_ms": round(state_ms, 2),
                       "option_encode_ms": oinfo["option_encode_ms"],
                       "heads_ms": round(max(heads_ms, 0.0), 2),
                       "total_ms": round((time.perf_counter() - t_all) * 1000,
                                         2)},
            "encoder": ENCODER_ID, "heads": heads.namespace,
            "scale": heads.scale}


def decide_detail(state, options: Sequence[str], **kw) -> dict:
    d = decide_detail_many(state, [options], **kw)
    d["probabilities"], d["logits"] = d["probabilities"][0], d["logits"][0]
    return d


def decide_many(state, option_lists: Sequence[Sequence[str]], **kw
                ) -> list[list[float]]:
    """One probability list per question, the state encoded once."""
    return decide_detail_many(state, option_lists, **kw)["probabilities"]


def decide(state_text: str, options: Sequence[str], **kw) -> list[float]:
    """The probability of each option, in order (softmax over this set)."""
    return decide_detail(state_text, options, **kw)["probabilities"]


def warm_options(texts: Iterable[str]) -> dict:
    """Embed option texts ahead of use (e.g. every armed skill's trigger at
    a revision), so a request pays only for its state."""
    heads = clm_heads.load()
    _z, info = _option_projections(heads, list(dict.fromkeys(texts)))
    return info


def status() -> dict:
    """What is present, without touching the GPU."""
    out = {"model": MODEL, "upstream": UPSTREAM, "encoder": ENCODER_ID,
           "tokenizer": TOKENIZER_PATH if os.path.exists(TOKENIZER_PATH)
           else None, "heads": None, "actions_cached": None}
    try:
        h = clm_heads.load()
        out["heads"] = {"namespace": h.namespace, "scale": h.scale,
                        "path": h.path}
    except Exception as e:                                       # noqa: BLE001
        out["heads_error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        out["actions_cached"] = len(action_cache())
    except Exception:                                            # noqa: BLE001
        pass
    return out


if __name__ == "__main__":
    print(json.dumps(status(), indent=1))
