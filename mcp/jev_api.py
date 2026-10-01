#!/usr/bin/env python
"""THE JEV API: jjava behind TypeSafe's public HTTP API, exactly.

Operator, 2026-09-29: "Also want the jjava api exact public api endpoints
that match Jev exposed through our proxy." jjava is our Jev (docs/JJAVA.md):
the typed decider mcp/decider_bonsai.py (READOUT typed/2) reading whichever
main model holds the card. This module is a TRANSLATION, like the Responses
and Messages APIs: Jev's request in, decider_bonsai.read() per question on
the decider lane, Jev's response out. mcp/proxy.py is not involved -- a
Jev call is not a chat turn.

SOURCES (docs.typesafe.ai, read 2026-09-29; the vendor's public docs):
  [api]     /api.md                    endpoint, request, answers, errors
  [models]  /models.md                 ids, aliases, limits, GET /v1/models
  [py]      /sdk/python/api/types/responses.md   ListModelsResponse,
            ModelMetadata, SystemOneResponse (pydantic, extra="ignore",
            strict=True); /sdk/python/api/exceptions.md; constants.md
  [js]      /sdk/javascript/api/interfaces/Models.md, ModelCard.md,
            Usage.md, classes/APIError.md, RateLimitError.md,
            UnprocessableEntityError.md (SDK 0.6.0)
  [js-src]  github.com/typesafe-ai/typesafe-sdk-js @ v0.6.0 src/errors.ts
            (read as a web page, not installed): extractMessage reads a
            body's `error`, `message`, then `detail` (a string, an object
            with `message`, or an ARRAY of {loc, msg} validation entries,
            `loc` joined with "." after dropping "body").
  docs/JEV-CONFORMANCE.md is the field-by-field table.

ENDPOINTS
  POST /jev/v1/systemone   [api]; base_url = "<public base>/jev" in a
  GET  /jev/v1/models      TypeSafe SDK (our GET /v1/models is OpenAI's list
                           and does not change)
  POST /v1/systemone       the root alias (it collides with nothing)

MODELS
  jjava-latest   jjava on whichever main model holds the card: max_mode's
                 side-call rule (decide(tier=None, utility=True): the holder,
                 else the model loaded, else the default) -- never a swap.
  jjava-<model>  (jjava-bonsai, jjava-mirai-s, jjava-flash-next): only while
                 that model is the one jjava-latest resolves to; otherwise
                 529 with Retry-After. A model this stack does not configure
                 is 422.
  jev-latest, jev-preview, jev-1.13.0   Jev's own ids [models], accepted as
                 aliases of jjava-latest so an unmodified Jev client works;
                 recorded (x_yamadori.jjava.alias). The response's `model` is
                 the versioned id that answered, as Jev's is [models]:
                 "jjava-<model>".

SEMANTICS (Jev's, mapped onto decider_bonsai's typed API)
  noul    -> q_noul; criteria {true, false} printed after "yes: " / "no: ".
  choice  -> q_choice; each option printed "key: description", or the key
             alone for a null description ([api] "use null when an option
             needs no extra detail"; the Choice page: "The option names and
             their descriptions are both sent to the model"). Up to 255
             options [api]. More than 26 (one single-token letter each, the
             decider's own limit: decider_bonsai LETTERS) go in TWO STAGES,
             never refused: the options in positional chunks of at most 26
             (balanced, so every chunk has >= 2), one read per chunk, then one
             read over the chunk winners (the rounds method of
             verify_moment CHUNK, git e360d37, and docs/JJAVA.md 4.3 "More:
             two stages (5.1) or rounds"). The answer's distribution over ALL
             options is P(option) = P_final(its chunk's winner) x
             P_chunk(option): it sums to 1, and `choice` is its argmax
             (Jev: "The option with the highest probability"). Reported in
             x_yamadori.answers.<id>.rounds.
             One option: probability 1 by definition; nothing is read.
  score   -> q_score; 2 to 10 levels ([api]: "at least two levels; the API
             accepts up to 10"; decider_bonsai allows 26, clamped here to
             Jev's contract). `legend` maps each level number to the
             criteria entry AS GIVEN (a string or the object; [py] Legend
             values are string | object | array).
  structured `instructions`, criteria values and `state` (object / array)
  are printed as JSON (an option's on one line, so its letter line holds).

  Answers carry ONLY Jev's fields; jjava's diagnostics (both orders, their
  disagreement, label mass, tie, rounds, the model profile) are in a
  top-level `x_yamadori`, which Jev clients ignore ([py]: extra="ignore").
  Jev evaluates questions IN PARALLEL [models]; jjava reads them one after
  another on one slot with the state cached (decider_bonsai: each question
  costs its own suffix, two orders).

USAGE, counted from the reads made (never estimated):
  input_tokens  = the first read's whole prompt (the state as placed,
                  whatever part the lane already held) + every later read's
                  processed tokens (llama-server `timings.prompt_n`): the
                  state once and each question's suffix per order and per
                  re-read, the way Jev "ingests the state once" [models].
  output_tokens = the reads: each one generates exactly one token
                  (max_tokens 1).
  When a read's timings are absent, input_tokens falls back to the sum of
  the reads' whole prompts (an upper bound) and x_yamadori.usage says so.

STATE AND CAPACITY
  Jev: "32k tokens for `state` plus the longest question" and "64k tokens
  per request" [models]. "32k" / "64k" are read as 32,768 / 65,536 (the
  binary k of context lengths; the docs print no exact figure). The limit
  here is min(that, the model's window: budget.budgets(model)["window"]),
  counted by the model server's own /tokenize. Over it: 422 on `state`.
  The questions run on THE LANE (slots: the child slot, rank 3,
  budget.LANE_TOKENS = 3,072 cells kept in VRAM). A state + question that
  does not fit the lane runs anyway, past the VRAM line (layout v2: "What
  does not fit is not refused ... past the line if need be"); recorded in
  x_yamadori.lane.
  ONE Jev call at a time (the lane is one slot): a second concurrent call is
  429 at once with Retry-After (Jev documents 429 for too many requests
  [api], and its SDKs retry with backoff); never a hang. Retry-After 1 s:
  the header's smallest unit -- a question's two reads take p50 496 ms,
  p90 577 ms (decide_turn's latency table, bench/decider/bonsai_decider.py).

ERRORS (Jev's statuses [api]; the body is the shape the SDK reads [js-src])
  401 {"detail": "..."}                       missing / invalid key
  422 {"detail": [{"loc": [...], "msg", "type"}, ...]}   validation, one
      entry per offending field, `loc` rooted at "body" (FastAPI's
      RequestValidationError form, which extractMessage parses)
  429 {"detail": "..."} + Retry-After         the lane is busy
  529 {"detail": "..."} + Retry-After         the model is not on the card /
      the model server is unreachable or answered 5xx / 429
  500 {"detail": "..."}                       our own failure
  503 {"detail": "..."}                       jjava off (YAMADORI_DECIDER=0)
  Every response carries `x-typesafe-request-id` (the SDKs' request id).

RECORDS: every call is one corpus event of its own kind, `jev_call` (never a
`turn`), with the account and its traffic class (corpus.account_traffic), so
test traffic never trains anything; the state and instructions are not kept
(a sha1 of the state is).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decider_bonsai as D  # noqa: E402

# ------------------------------------------------------------- constants ----
LATEST = "jjava-latest"
# [models] "Current models" (jev-1.13.0) and "Aliases" (jev-latest,
# jev-preview): Jev's own ids, accepted as aliases of jjava-latest.
JEV_ALIASES = ("jev-latest", "jev-preview", "jev-1.13.0")
# The named ids the operator's brief lists (2026-09-29); any other main model
# the tier table configures is named the same way (jjava-<model>).
NAMED = {"jjava-bonsai": "bonsai", "jjava-mirai-s": "mirai-s",
         "jjava-flash-next": "flash-next"}
# When jjava's typed readout (typed/2, Jev's API shape) shipped: docs/JJAVA.md
# ("operator, 2026-09-29: look to the Jev api for the answer").
RELEASE_DATE = "2026-09-29"
TYPES = ("noul", "choice", "score")
MAX_CHOICE_OPTIONS = 255            # [api] "a maximum of 255 options"
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10   # [api] "at least two ... up to 10"
STATE_LIMIT_TOKENS = 32768          # [models] "32k tokens for state plus the longest question"
REQUEST_LIMIT_TOKENS = 65536        # [models] "64k tokens per request"
ROUND_SIZE = len(D.LETTERS)         # 26: one single-token letter per option
RETRY_AFTER_BUSY_S = 1              # see the docstring (the header's unit)
REQUEST_ID_HEADER = "x-typesafe-request-id"   # [js] APIError.requestId, [py] request_id

_LANE = threading.Lock()            # ONE Jev call on the lane at a time


class JevError(Exception):
    """A refusal in Jev's shape: `status`, `detail` (a string, or a list of
    validation entries) and headers."""

    def __init__(self, status: int, detail, headers: dict | None = None,
                 code: str = ""):
        super().__init__(detail if isinstance(detail, str) else
                         "; ".join(str(d.get("msg")) for d in detail))
        self.status, self.detail = int(status), detail
        self.headers = dict(headers or {})
        self.code = code

    def body(self) -> dict:
        return {"detail": self.detail}


def _v(loc: list, msg: str, typ: str) -> dict:
    return {"loc": ["body", *loc], "msg": msg, "type": typ}


def invalid(errors: list[dict]) -> JevError:
    return JevError(422, errors, code="validation")


def json_invalid(e: Exception) -> JevError:
    pos = getattr(e, "pos", 0)
    return invalid([{"loc": ["body", int(pos or 0)],
                     "msg": f"JSON decode error: {getattr(e, 'msg', e)}",
                     "type": "json_invalid"}])


def unauthorised(why: str) -> JevError:
    return JevError(401, f"{why}. Check the Authorization header "
                         "(Authorization: Bearer <API_KEY>).",
                    {"WWW-Authenticate": "Bearer"}, code="unauthorised")


# ------------------------------------------------------------ rendering ----
def as_text(v, *, one_line: bool = False) -> str:
    """A Jev value (string | object | array) as the text the model reads:
    a string as is, anything else as JSON (one line for an option)."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    if one_line:
        return json.dumps(v, ensure_ascii=False)
    return json.dumps(v, ensure_ascii=False, indent=2)


def option_text(key: str, desc) -> str:
    """How a Choice option is printed: "key: description", or the key alone
    for a null description (the Choice page: the option names and their
    descriptions are both sent to the model)."""
    if desc is None:
        return str(key)
    return f"{key}: {as_text(desc, one_line=True)}"


def chunks(n: int, size: int = ROUND_SIZE) -> list[tuple[int, int]]:
    """[lo, hi) spans of n options in ceil(n / size) near-equal positional
    chunks (each >= 2 when n > size)."""
    k = max(1, math.ceil(n / size))
    base, extra = divmod(n, k)
    out, lo = [], 0
    for j in range(k):
        hi = lo + base + (1 if j < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


# ----------------------------------------------------------- validation ----
_TEXTY = (str, dict, list)


def _type_name(v) -> str:
    return {str: "string", dict: "object", list: "array", bool: "boolean",
            int: "number", float: "number", type(None): "null"}.get(
                type(v), type(v).__name__)


def validate(body) -> tuple[str, object, list[dict]]:
    """(model, state, specs) or raise a 422 listing EVERY offending field.
    A spec: {id, type, instructions, text, criteria?, keys?, descs?,
    levels?}."""
    errs: list[dict] = []
    if not isinstance(body, dict):
        raise invalid([_v([], "Input should be a valid dictionary or object "
                              "to extract fields from", "model_attributes_type")])
    for f in ("state", "model", "questions"):
        if f not in body:
            errs.append(_v([f], "Field required", "missing"))
    model = body.get("model")
    if "model" in body and not isinstance(model, str):
        errs.append(_v(["model"], "Input should be a valid string",
                       "string_type"))
    state = body.get("state")
    # [api] string | object | array; the JS SDK's EntryType also allows null
    if "state" in body and state is not None and not isinstance(state, _TEXTY):
        errs.append(_v(["state"], "Input should be a string, an object or an "
                                  f"array (got {_type_name(state)})",
                       "state_type"))
    qs = body.get("questions")
    specs: list[dict] = []
    if "questions" in body:
        if not isinstance(qs, dict):
            errs.append(_v(["questions"], "Input should be a valid dictionary",
                           "dict_type"))
        elif not qs:
            # [js] SystemOneRequestPayload: "Nonempty questions"
            errs.append(_v(["questions"], "Dictionary should have at least 1 "
                                          "item after validation, not 0",
                           "too_short"))
        else:
            for qid, q in qs.items():
                spec = _validate_question(str(qid), q, errs)
                if spec is not None:
                    specs.append(spec)
    if errs:
        raise invalid(errs)
    return model, state, specs


def _validate_question(qid: str, q, errs: list[dict]) -> dict | None:
    at = ["questions", qid]
    if not isinstance(q, dict):
        errs.append(_v(at, "Input should be a valid dictionary or object",
                       "model_attributes_type"))
        return None
    n0 = len(errs)
    t = q.get("type")
    if "type" not in q:
        errs.append(_v(at + ["type"], "Field required", "missing"))
    elif t not in TYPES:
        errs.append(_v(at + ["type"], "Input tag "
                       f"{json.dumps(t)} found using 'type' does not match any "
                       "of the expected tags: 'noul', 'choice', 'score'",
                       "union_tag_invalid"))
    ins = q.get("instructions")
    if "instructions" not in q:
        errs.append(_v(at + ["instructions"], "Field required", "missing"))
    elif not isinstance(ins, _TEXTY):
        errs.append(_v(at + ["instructions"], "Input should be a string, an "
                       f"object or an array (got {_type_name(ins)})",
                       "instructions_type"))
    crit = q.get("criteria")
    spec: dict = {"id": qid, "type": t, "instructions": ins,
                  "text": as_text(ins) if isinstance(ins, _TEXTY) else ""}
    if t == "noul":
        if crit is not None:
            if not isinstance(crit, dict):
                errs.append(_v(at + ["criteria"], "Input should be a valid "
                               "dictionary with keys 'true' and 'false'",
                               "dict_type"))
            else:
                for k in crit:
                    if k not in ("true", "false"):
                        errs.append(_v(at + ["criteria", k], "Extra inputs "
                                       "are not permitted (a Noul's criteria "
                                       "are 'true' and 'false')",
                                       "extra_forbidden"))
                for k in ("true", "false"):
                    v = crit.get(k)
                    if v is not None and not isinstance(v, _TEXTY):
                        errs.append(_v(at + ["criteria", k], "Input should "
                                       "be a string, an object or an array",
                                       "criteria_type"))
                spec["criteria"] = {k: crit[k] for k in ("true", "false")
                                    if crit.get(k) is not None}
    elif t == "choice":
        if "criteria" not in q:
            errs.append(_v(at + ["criteria"], "Field required", "missing"))
        elif not isinstance(crit, dict):
            errs.append(_v(at + ["criteria"], "Input should be a valid "
                           "dictionary of option -> description",
                           "dict_type"))
        elif not crit:
            errs.append(_v(at + ["criteria"], "Dictionary should have at "
                           "least 1 item after validation, not 0",
                           "too_short"))
        elif len(crit) > MAX_CHOICE_OPTIONS:
            errs.append(_v(at + ["criteria"], "A Choice accepts at most "
                           f"{MAX_CHOICE_OPTIONS} options; got {len(crit)}",
                           "too_long"))
        else:
            for k, v in crit.items():
                if v is not None and not isinstance(v, _TEXTY):
                    errs.append(_v(at + ["criteria", k], "Input should be a "
                                   "string, an object, an array or null",
                                   "criteria_type"))
            spec["keys"] = [str(k) for k in crit]
            spec["descs"] = list(crit.values())
    elif t == "score":
        if "criteria" not in q:
            errs.append(_v(at + ["criteria"], "Field required", "missing"))
        elif not isinstance(crit, list):
            errs.append(_v(at + ["criteria"], "Input should be a valid list "
                           "of level descriptions, lowest first", "list_type"))
        elif len(crit) < MIN_SCORE_LEVELS:
            errs.append(_v(at + ["criteria"], "A Score should have at least "
                           f"{MIN_SCORE_LEVELS} levels; got {len(crit)}",
                           "too_short"))
        elif len(crit) > MAX_SCORE_LEVELS:
            errs.append(_v(at + ["criteria"], "A Score accepts up to "
                           f"{MAX_SCORE_LEVELS} levels; got {len(crit)}",
                           "too_long"))
        else:
            for i, v in enumerate(crit):
                if not isinstance(v, _TEXTY):
                    errs.append(_v(at + ["criteria", i], "Input should be a "
                                   "string, an object or an array",
                                   "criteria_type"))
            spec["levels"] = list(crit)
    return spec if len(errs) == n0 else None


# ---------------------------------------------------------------- models ----
def _max_mode():
    import max_mode
    return max_mode


def configured() -> list[str]:
    """The main models this stack serves (the tier table's, else the one)."""
    mm = _max_mode()
    return list(mm.MODELS) if mm.ENABLED else [mm.MAIN]


def card() -> dict:
    """{model, why, holder}: the model jjava-latest reads NOW -- max_mode's
    rule for a call that names no conversation (a side call): the holder,
    else the loaded main model, else the default. Never a swap."""
    mm = _max_mode()
    d = mm.decide(None, True)
    return {"model": d.model, "why": d.why, "holder": d.holder,
            "enabled": mm.ENABLED}


def named_ids() -> dict[str, str]:
    """Every jjava-<model> id this stack recognises -> its model."""
    out = dict(NAMED)
    for m in configured():
        out.setdefault(f"jjava-{m}", m)
    return out


def resolve(name: str) -> dict:
    """{model, requested, alias_of?, card} for a request's `model`, or raise:
    422 for a name no jjava model carries, 529 for a named model that is not
    the one on the card."""
    c = card()
    rec = {"requested": name, "card": c}
    if name in JEV_ALIASES:
        return dict(rec, model=c["model"], alias_of=LATEST,
                    alias_note="Jev's own id, served as jjava-latest")
    if name == LATEST:
        return dict(rec, model=c["model"])
    ids = named_ids()
    if name in ids:
        m = ids[name]
        if m not in configured():
            raise invalid([_v(["model"], f"{name!r} is not served by this "
                              "stack (its tier table does not configure "
                              f"{m}); accepted: {', '.join(accepted())}",
                              "model_not_found")])
        if m != c["model"]:
            mm = _max_mode()
            ra = int(mm.retry_after())
            raise JevError(529, f"{name} is not available now: the card holds "
                           f"{c['model']} and a jjava call never swaps it "
                           f"({c['why']}). Retry in about {ra} s, or send "
                           f"{LATEST}.", {"Retry-After": str(ra)},
                           code="model_not_on_card")
        return dict(rec, model=m)
    raise invalid([_v(["model"], f"Unknown model {name!r}; accepted: "
                      f"{', '.join(accepted())}", "model_not_found")])


def accepted() -> list[str]:
    return [LATEST, *[f"jjava-{m}" for m in configured()], *JEV_ALIASES]


def public_id(model: str) -> str:
    """The versioned id that answered (Jev reports the versioned id, never
    the alias [models])."""
    return f"jjava-{D.canonical_model(model) or model}"


def _profile(model: str) -> dict:
    try:
        st = D.profile_status(model)
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    measured = set(st.get("measured") or [])
    return {"record": st.get("record"),
            "priors_measured": {"letter_prior": "letter_prior" in measured,
                                "label_bias": "label_bias" in measured},
            "tie_band": st.get("tie_band"),
            "measured": sorted(measured),
            "unmeasured": st.get("unmeasured") or []}


def models_list() -> dict:
    """GET /v1/models in Jev's shape ([py] ListModelsResponse: {models:
    [{name, description, release_date}]}), jjava's facts per model in
    x_yamadori."""
    c = card()
    rows = [{"name": LATEST,
             "description": "jjava, Yamadori's decider in Jev's API, on "
                            "whichever main model holds the card; the "
                            "response's model field names the one that "
                            "answered.",
             "release_date": RELEASE_DATE}]
    facts = {LATEST: {"model": c["model"], "available": True,
                      "why": c["why"], **_profile(c["model"])}}
    for m in configured():
        name = f"jjava-{m}"
        rows.append({"name": name,
                     "description": f"jjava reading {m} (only while {m} holds "
                                    "the card; otherwise 529).",
                     "release_date": RELEASE_DATE})
        on = m == c["model"]
        facts[name] = {"model": m, "available": on,
                       "why": ("on the card" if on else
                               f"not on the card (the card: {c['model']}); "
                               "a jjava call never swaps it"),
                       **_profile(m)}
    return {"models": rows,
            "x_yamadori": {"models": facts,
                           "aliases": {a: LATEST for a in JEV_ALIASES},
                           "parallel": False,
                           "note": "questions are read one after another on "
                                   "one cached state (Jev: in parallel)"}}


# ----------------------------------------------------------------- limits ---
def count(text: str) -> int:
    """The model server's own token count (/tokenize on the bound model)."""
    toks = D._upstream("/tokenize", {"content": text, "add_special": False},
                       timeout=30)["tokens"]
    return len(toks)


def window_of(model: str) -> tuple[int | None, str]:
    """The most one request on `model` may occupy (budget: the main cap, or
    the table's window), or (None, why)."""
    try:
        import budget
        b = budget.budgets(model=model)
        return int(b["window"]), f"budget.budgets({model!r})['window'] ({b.get('cap_source')})"
    except Exception as e:                                       # noqa: BLE001
        return None, f"window unreadable: {type(e).__name__}: {e}"[:200]


def lane_cells() -> int:
    try:
        import budget
        return int(budget.LANE_TOKENS)
    except Exception:                                            # noqa: BLE001
        return 3072


# ---------------------------------------------------------------- reading ---
def build(spec: dict) -> dict:
    """The decider questions one spec needs: {kind: single | rounds |
    trivial, q | stage1 [q] (+ spans)}."""
    t, qid, text = spec["type"], spec["id"], spec["text"]
    if t == "noul":
        crit = {k: as_text(v, one_line=True)
                for k, v in (spec.get("criteria") or {}).items()}
        return {"kind": "single", "q": D.q_noul(qid, text, criteria=crit or None)}
    if t == "score":
        return {"kind": "single",
                "q": D.q_score(qid, text, [as_text(v, one_line=True)
                                           for v in spec["levels"]])}
    keys, descs = spec["keys"], spec["descs"]
    texts = [option_text(k, d) for k, d in zip(keys, descs)]
    if len(keys) == 1:
        return {"kind": "trivial"}
    if len(keys) <= ROUND_SIZE:
        return {"kind": "single", "q": D.q_choice(qid, text, texts, keys=keys)}
    spans = chunks(len(keys))
    return {"kind": "rounds", "spans": spans, "texts": texts,
            "stage1": [D.q_choice(f"{qid}#{j}", text, texts[lo:hi],
                                  keys=keys[lo:hi])
                       for j, (lo, hi) in enumerate(spans)]}


def question_texts(plan: dict) -> list[str]:
    """What a question puts after the state, per read (its longest)."""
    if plan["kind"] == "single":
        return [D.question_text(D.rendered_orders(plan["q"])[0])]
    if plan["kind"] == "rounds":
        return [D.question_text(D.rendered_orders(q)[0])
                for q in plan["stage1"]]
    return []


def _jev_answer(spec: dict, a: dict) -> dict:
    """decider_bonsai's answer reduced to Jev's fields, in Jev's order."""
    t = spec["type"]
    if t == "noul":
        return {"type": "noul", "noul": a["noul"]}
    if t == "choice":
        return {"type": "choice", "choice": a["choice"],
                "probabilities": a["probabilities"],
                "confidence": a["confidence"]}
    return {"type": "score", "score": a["score"],
            "legend": {str(i): lvl for i, lvl in enumerate(spec["levels"])},
            "probabilities": a["probabilities"],
            "confidence": a["confidence"]}


def _orders(a: dict) -> list[dict]:
    return list((a.get("diagnostics") or {}).get("orders") or [])


def answer(spec: dict, plan: dict, state: str, slot) -> tuple[dict, dict,
                                                              list[dict]]:
    """(Jev's answer, its x_yamadori diagnostics, the reads' order records)."""
    if plan["kind"] == "trivial":
        k = spec["keys"][0]
        return ({"type": "choice", "choice": k, "probabilities": {k: 1.0},
                 "confidence": 1.0},
                {"reads": 0, "why": "one option: probability 1 by "
                                    "definition, nothing read"}, [])
    if plan["kind"] == "single":
        a = D.read(state, plan["q"], slot=slot)
        return _jev_answer(spec, a), a["diagnostics"], _orders(a)
    keys, spans = spec["keys"], plan["spans"]
    stage1, orders, rounds = [], [], []
    for j, q in enumerate(plan["stage1"]):
        a = D.read(state, q, slot=slot)
        stage1.append(a)
        orders += _orders(a)
        rounds.append({"stage": 1, "chunk": j, "options": len(q["keys"]),
                       "span": list(spans[j]), "winner": a["choice"],
                       "confidence": a["confidence"],
                       "diagnostics": a["diagnostics"]})
    winners = [a["choice"] for a in stage1]
    qf = D.q_choice(f"{spec['id']}#final", spec["text"],
                    [plan["texts"][keys.index(w)] for w in winners],
                    keys=winners)
    af = D.read(state, qf, slot=slot)
    orders += _orders(af)
    rounds.append({"stage": 2, "options": len(winners), "winners": winners,
                   "choice": af["choice"], "confidence": af["confidence"],
                   "diagnostics": af["diagnostics"]})
    comp: dict[str, float] = {}
    for j, (lo, hi) in enumerate(spans):
        pw = float(af["probabilities"].get(winners[j], 0.0))
        for k in keys[lo:hi]:
            comp[k] = pw * float(stage1[j]["probabilities"].get(k, 0.0))
    z = sum(comp.values()) or 1.0
    comp = {k: v / z for k, v in comp.items()}
    top = max(keys, key=lambda k: comp[k])        # first in given order
    ans = {"type": "choice", "choice": top,
           "probabilities": {k: round(comp[k], 6) for k in keys},
           "confidence": round(D.confidence(comp), 6)}
    diag = {"rounds": rounds, "stages": 2, "chunks": len(spans),
            "round_size": ROUND_SIZE,
            "method": "two stages: positional chunks of <= 26 options, one "
                      "read each, then one read over the chunk winners; "
                      "P(option) = P_final(its chunk's winner) x "
                      "P_chunk(option); choice = its argmax"}
    return ans, diag, orders


def usage_of(orders: list[dict]) -> dict:
    """USAGE (docstring): Jev's {input_tokens, output_tokens} from the reads
    made, and how it was counted."""
    if not orders:
        return {"input_tokens": 0, "output_tokens": 0,
                "x": {"reads": 0, "method": "no read was made"}}
    reads = sum(int(o.get("http_reads") or 1) for o in orders)
    exact = all(o.get("processed_all") is not None for o in orders)
    if exact:
        first = orders[0]
        cached_first = int(first.get("cached_tokens") or 0)
        processed = sum(int(o["processed_all"]) for o in orders)
        inp = cached_first + processed
        how = ("the first read's cached prefix + every read's processed "
               "tokens (timings.prompt_n)")
    else:
        inp = sum(int(o.get("prompt_tokens") or 0) for o in orders)
        processed = None
        how = ("timings absent: the sum of each order's whole prompt (an "
               "upper bound)")
    return {"input_tokens": int(inp), "output_tokens": int(reads),
            "x": {"reads": reads, "processed": processed, "method": how}}


# ---------------------------------------------------------------- serving ---
def enabled() -> tuple[bool, str]:
    try:
        import decide_turn
        return bool(decide_turn.ON), "YAMADORI_DECIDER"
    except Exception as e:                                       # noqa: BLE001
        return False, f"decide_turn unavailable: {type(e).__name__}"


def _record(rec: dict) -> None:
    """One corpus event of kind `jev_call` (never a chat `turn`)."""
    try:
        import corpus
        corpus.log(rec.get("request_id") or uuid.uuid4().hex[:16], "jev_call",
                   name=rec.get("route"), payload=rec)
    except Exception:                                            # noqa: BLE001
        pass


def _traffic(account: str | None) -> str:
    try:
        import corpus
        return corpus.account_traffic(account)
    except Exception:                                            # noqa: BLE001
        return "test"          # corpus.account_traffic fails closed too


def new_request_id() -> str:
    return "jjava-" + uuid.uuid4().hex[:16]


def refused(route: str, rid: str, e: JevError,
            account: str | None = None) -> JevError:
    """A refusal made before systemone() runs (server.py: a missing or bad
    key, a body that is not JSON), recorded like every call -- the JJAVA
    page's endpoint visibility counts 401 and 422 too (2026-09-30). Returns
    `e` for the caller to send. Never raises."""
    try:
        _record({"route": route, "request_id": rid,
                 "account": (account or "")[:16] or None,
                 "traffic": _traffic(account) if account else "unknown",
                 "status": e.status, "ms": 0.0,
                 "error": {"status": e.status, "code": e.code,
                           "detail": e.detail if isinstance(e.detail, str)
                           else [d.get("msg") for d in e.detail][:20]}})
    except Exception:                                            # noqa: BLE001
        pass
    return e


def models_call(account: str | None, rid: str,
                route: str = "/jev/v1/models") -> dict:
    """GET /v1/models, recorded (a corpus `jev_call` event, status 200)."""
    t0 = time.time()
    out = models_list()
    _record({"route": route, "request_id": rid,
             "account": (account or "")[:16] or None,
             "traffic": _traffic(account), "status": 200,
             "ms": round((time.time() - t0) * 1000, 1),
             "models": len(out.get("models") or [])})
    return out


def systemone(body, account: str | None, *, route: str = "/jev/v1/systemone",
              request_id: str | None = None) -> tuple[int, dict, dict]:
    """POST /v1/systemone: (status, JSON body, headers). Runs in a worker
    thread (server.py: run_in_threadpool). Never raises."""
    rid = request_id or new_request_id()
    t0 = time.time()
    rec: dict = {"route": route, "request_id": rid,
                 "account": (account or "")[:16] or None,
                 "traffic": _traffic(account)}
    try:
        status, out = _systemone(body, rec)
        headers = {}
    except JevError as e:
        status, out, headers = e.status, e.body(), e.headers
        rec.update(error={"status": e.status, "code": e.code,
                          "detail": e.detail if isinstance(e.detail, str)
                          else [d.get("msg") for d in e.detail][:20]})
    except Exception as e:                                       # noqa: BLE001
        status, out, headers = 500, {"detail": "jjava failed: "
                                     f"{type(e).__name__}: {e}"[:400]}, {}
        rec.update(error={"status": 500, "code": "internal",
                          "detail": out["detail"]})
    rec.update(status=status, ms=round((time.time() - t0) * 1000, 1))
    _record(rec)
    headers = dict(headers, **{REQUEST_ID_HEADER: rid})
    return status, out, headers


def _systemone(body, rec: dict) -> tuple[int, dict]:
    import cancel
    on, src = enabled()
    if not on:
        raise JevError(503, f"jjava is off in this process ({src}=0).",
                       code="decider_off")
    requested, state, specs = validate(body)
    res = resolve(requested)
    model = res["model"]
    rec.update(requested=requested, model=model,
               alias_of=res.get("alias_of"),
               questions={s["id"]: {"type": s["type"],
                                    "options": len(s.get("keys") or
                                                   s.get("levels") or [])}
                          for s in specs})
    state_text = as_text(state)
    rec["state_sha1"] = hashlib.sha1(state_text.encode("utf-8")).hexdigest()[:16]
    plans = [build(s) for s in specs]
    if not _LANE.acquire(blocking=False):
        raise JevError(429, "The decider lane is busy with another Jev call "
                            "(one at a time: the lane is one slot). Retry "
                            "shortly.",
                       {"Retry-After": str(RETRY_AFTER_BUSY_S)},
                       code="lane_busy")
    mm = _max_mode()
    lease = None
    try:
        if mm.ENABLED:
            lease = mm.Lease(mm.Decision(model, None, True, why="a Jev call "
                                         "(jjava) on the model on the card"))
        tok = cancel.Token()
        with cancel.bound(tok):
            mm.set_current(model)
            return _run(state_text, specs, plans, model, res, rec)
    finally:
        if lease is not None:
            lease.release()
        _LANE.release()


def _unavailable(e: Exception) -> JevError:
    """A model-side failure in Jev's statuses: retryable -> 529 overloaded
    with Retry-After; not retryable -> 500."""
    mm = _max_mode()
    if isinstance(e, mm.ModelAtCapacity):
        ra = int(e.retry_after)
        return JevError(529, f"jjava's model is not available now: {e}",
                        {"Retry-After": str(ra)}, code="model_at_capacity")
    if isinstance(e, D.DeciderUnavailable):
        if e.retryable:
            ra = int(mm.RETRY_AFTER_UNKNOWN)
            return JevError(529, f"jjava is overloaded or its model server "
                            f"is unavailable ({e.code}: {e.situation}). "
                            f"Retry in about {ra} s.",
                            {"Retry-After": str(ra)}, code=e.code)
        return JevError(500, f"jjava could not answer ({e.code}): "
                        f"{e.situation}", code=e.code)
    return JevError(529, f"jjava's model server did not answer "
                    f"({type(e).__name__}: {e})"[:400],
                    {"Retry-After": str(int(mm.RETRY_AFTER_UNKNOWN))},
                    code="model_unreachable")


def _run(state_text: str, specs: list[dict], plans: list[dict], model: str,
         res: dict, rec: dict) -> tuple[int, dict]:
    import slots
    mm = _max_mode()
    # --- the limits, counted by the model server
    try:
        st_tokens = count(D.STATE_HEAD + state_text)
        q_tokens = {s["id"]: max([count(t) for t in question_texts(p)] or [0])
                    for s, p in zip(specs, plans)}
        overhead = count(D.SYSTEM) + count(D.ANSWER_LEAD)
    except (D.DeciderUnavailable, mm.ModelAtCapacity) as e:
        raise _unavailable(e) from e
    except Exception as e:                                       # noqa: BLE001
        raise _unavailable(e) from e
    longest = max(q_tokens.values() or [0])
    window, wsrc = window_of(model)
    limit = STATE_LIMIT_TOKENS if window is None else min(STATE_LIMIT_TOKENS,
                                                          window - overhead)
    limits = {"state_tokens": st_tokens, "longest_question_tokens": longest,
              "questions_tokens": sum(q_tokens.values()),
              "overhead": overhead, "limit": limit, "request_limit": REQUEST_LIMIT_TOKENS,
              "window": window, "window_source": wsrc,
              "limit_source": "min(Jev's 32k for state + the longest question "
                              "(docs.typesafe.ai/models), the model's window "
                              "less the template)"}
    rec["limits"] = limits
    if st_tokens + longest > limit:
        raise invalid([_v(["state"], f"state plus the longest question is "
                          f"{st_tokens + longest} tokens; the limit is "
                          f"{limit} (32k tokens for state plus the longest "
                          "question)", "too_long")])
    if st_tokens + limits["questions_tokens"] > REQUEST_LIMIT_TOKENS:
        raise invalid([_v(["questions"], "state plus all questions is "
                          f"{st_tokens + limits['questions_tokens']} tokens; "
                          f"the limit is {REQUEST_LIMIT_TOKENS} (64k tokens "
                          "per request)", "too_long")])
    needed = overhead + st_tokens + longest + 1
    lane = {"cells": lane_cells(), "needed": needed,
            "fits": needed <= lane_cells(),
            "note": ("fits the lane's VRAM cells" if needed <= lane_cells()
                     else "larger than the lane: runs past the VRAM line, "
                          "slower (layout v2)")}
    # --- the reads, one slot, in order
    t0 = time.time()
    grant = slots.acquire(None, transient=True)
    slot = grant.get("slot")
    lane.update(slot=slot, how=grant.get("how"))
    answers: dict = {}
    diags: dict = {}
    orders: list[dict] = []
    try:
        for spec, plan in zip(specs, plans):
            try:
                a, dg, od = answer(spec, plan, state_text, slot)
            except (D.DeciderUnavailable, mm.ModelAtCapacity) as e:
                raise _unavailable(e) from e
            answers[spec["id"]] = a
            diags[spec["id"]] = dg
            orders += od
    finally:
        slots.release(grant)
        lane["release"] = (slots.lane_kept_note(slot, "jev call", log=[])
                           if slots.lane_kept() else D.release(slot, "jev call"))
    u = usage_of(orders)
    rounds_used = [i for i, p in zip(answers, plans) if p["kind"] == "rounds"]
    rec.update(usage={k: u[k] for k in ("input_tokens", "output_tokens")},
               rounds=rounds_used, lane={k: lane[k] for k in
                                         ("cells", "needed", "fits", "slot")},
               answers={k: {kk: v for kk, v in a.items() if kk != "legend"}
                        for k, a in answers.items()})
    out = {"model": public_id(model), "answers": answers,
           "usage": {"input_tokens": u["input_tokens"],
                     "output_tokens": u["output_tokens"]},
           "x_yamadori": {
               "jjava": {"readout": D.READOUT_VERSION,
                         "template": D.TEMPLATE_VERSION,
                         "model": model, "requested": res["requested"],
                         **({"alias_of": res["alias_of"],
                             "alias_note": res.get("alias_note")}
                            if res.get("alias_of") else {}),
                         "card": res["card"],
                         "model_profile": _profile(model)},
               "answers": diags,
               "rounds": {"used": bool(rounds_used), "questions": rounds_used},
               "lane": lane, "limits": limits, "usage": u["x"],
               "parallel": False,
               "note": "questions were read one after another on one cached "
                       "state (Jev evaluates them in parallel)",
               "ms": round((time.time() - t0) * 1000, 1)}}
    return 200, out
