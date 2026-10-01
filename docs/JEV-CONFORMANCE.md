# The Jev API: conformance

jjava (our decider, docs/JJAVA.md) served behind TypeSafe's public Jev API
(operator, 2026-09-29: "Also want the jjava api exact public api endpoints
that match Jev exposed through our proxy."). Code: `mcp/jev_api.py`, routes
in `mcp/server.py`; gate: `mcp/test_jev_api.py` (offline, 190 checks, a fake
model server behind `decider_bonsai`'s doors, so the real readout runs).
**Offline only: no live run and no TypeSafe SDK run yet** (section 6).

Sources, read 2026-09-29 (vendor docs):

- **[api]** docs.typesafe.ai/api.md: endpoint, request, answer types, errors.
- **[models]** docs.typesafe.ai/models.md: ids, aliases, limits, `GET /v1/models`.
- **[py]** docs.typesafe.ai/sdk/python/api/types/responses.md
  (`SystemOneResponse`, `ListModelsResponse`, `ModelMetadata`; pydantic,
  `extra="ignore"`, `strict=True`), exceptions.md, constants.md, retries.md;
  changelog: SDK v0.7.2 (2026-09-26).
- **[js]** docs.typesafe.ai/sdk/javascript/api/: `Models`, `ModelCard`,
  `Usage`, `APIError`, `RateLimitError`, `UnprocessableEntityError`,
  `SystemOneRequestPayload`; `VERSION` 0.6.0.
- **[js-src]** github.com/typesafe-ai/typesafe-sdk-js at tag v0.6.0,
  `src/errors.ts` (read as a web page; nothing installed):
  `extractMessage` reads a body's `error`, then `message`, then `detail` (a
  string, an object with `message`, or an array of `{loc, msg}` entries,
  `loc` joined with "." after dropping "body").
- **[examples]** every distinct API response example in docs.typesafe.ai's
  llms-full.txt: 13, kept verbatim with their requests in
  `mcp/fixtures/jev_doc_examples.json` (each request's provenance is
  recorded: 5 verbatim JSON, 4 from the pages' Python SDK calls, 4
  reconstructed where the page's request is a widget that llms-full.txt
  does not contain).

## 1. Endpoints

| Jev | ours | difference, and why |
|---|---|---|
| `POST https://api.typesafe.ai/v1/systemone` [api] | `POST <base>/jev/v1/systemone` and `POST <base>/v1/systemone` | our `GET /v1/models` is OpenAI's list and does not change, so the Jev API gets its own base path; a TypeSafe SDK works with `base_url = "<base>/jev"` (`TYPESAFE_BASE_URL`). The root POST collides with nothing, so it is accepted too. |
| `GET https://api.typesafe.ai/v1/models` [models] | `GET <base>/jev/v1/models` | the root path is OpenAI's (above). |
| `Authorization: Bearer <API_KEY>` [api] | the same, with an account key of ours (`accounts.identify`) | none. Single-user mode (no key registry) admits every caller, as every route does. |
| request id header `x-typesafe-request-id` ([js] `APIError.requestId`, [py] `request_id`) | sent on every response, `jjava-<16 hex>` | the SDKs read it; the corpus row carries the same id. |
| unknown path / wrong method | `404 {"detail": "Not Found"}`, `405 {"detail": "Method Not Allowed"}` under `/jev` | not documented by Jev; FastAPI's default bodies, the form [js-src] reads. |

## 2. Request

| Jev field / limit | ours | difference, and why |
|---|---|---|
| `state`: string, object or array, required [api]; `null` allowed by [js] `EntryType` | the same; an object or array is printed as JSON (indent 2) as the MATERIAL | Jev does not say how it renders structure; JSON keeps every name (docs.typesafe.ai/concepts/state: "Use an object ... so each part of the state has a descriptive name"). |
| `model`: required [api] | `jjava-latest`, `jjava-<model>` (`jjava-bonsai`, `jjava-mirai-s`, `jjava-flash-next`), and Jev's own `jev-latest`, `jev-preview`, `jev-1.13.0` as aliases of `jjava-latest` | see section 4. An unknown name is 422 on `model` (Jev does not document this case). |
| `questions`: map of id -> question, nonempty ([api], [js] "Nonempty questions") | the same; answers keyed by the same ids, in the request's order | the id is never sent to the model (as [api] says of Jev). |
| `type`: `noul` / `choice` / `score` [api] | the same | none |
| `instructions`: string, object or array [api] | the same; an object or array is printed as JSON (indent 2) as the QUESTION | rendering as above. |
| noul `criteria`: optional `{true, false}`, each string / object / array [api] | the same; printed as the lettered pair "yes: <true>" / "no: <false>" (decider_bonsai `q_noul`) | that rendering is **unmeasured on Bonsai** (docs/JJAVA.md 2). Any other key is 422. |
| choice `criteria`: map option -> string / object / array / null, **at most 255** [api] | the same; each option printed "key: description", or the key alone for `null` | the Choice page: "The option names and their descriptions are both sent to the model". **More than 26 options run in two stages** (section 3); never refused. 256 or more is 422, as Jev's limit. Zero options is 422; **one option** answers probability 1 without a read (Jev does not document either case). |
| score `criteria`: ordered array, "at least two levels; the API accepts up to 10" [api] | 2 to 10 enforced (422 outside) | decider_bonsai accepts up to 26; clamped to Jev's contract. |
| "32k tokens for `state` plus the longest question"; "64k tokens per request" [models] | limit = min(32,768, the model's window less the template: `budget.budgets(model)["window"]`), counted by the model server's own `/tokenize`; state + all questions <= 65,536. Over either: 422 on `state` / `questions` | "32k" and "64k" are read as 32,768 and 65,536 (the docs print no exact figure; the binary k of context lengths). The window never binds today (every main model's window is > 100k). What Jev returns past its limit is not documented; 422 names the field. |

## 3. Answers

Every answer carries **only** Jev's fields, in Jev's order (gated per type);
jjava's diagnostics are in a separate top-level `x_yamadori`, which Jev's
clients ignore ([py]: `extra="ignore"`).

| Jev answer [api] | ours | how it is computed (decider_bonsai typed/2) |
|---|---|---|
| noul `{type, noul}` | the same | the averaged probability of "yes" over the two option orders |
| choice `{type, choice, probabilities, confidence}` | the same | `probabilities` averaged over two orders (sum 1); `choice` its argmax; `confidence` = clamp((n x p_max - 1) / (n - 1), 0, 1), the n-option form derived in docs/JJAVA.md 2 (agrees with all 17 distinct doc examples within their 2-decimal rounding) |
| score `{type, score, legend, probabilities, confidence}` | the same | levels numbered 0..k; `score` = sum of level x probability; `legend` = each level number -> the criteria entry **as given** (a string, or the object: [py] Legend values are string / object / array, and the Score page's example shows objects) |
| (Jev: up to 255 options in one evaluation) | **> 26 options: two stages**: positional chunks of at most 26 (balanced, so each has >= 2), one read per chunk, then one read over the chunk winners; P(option) = P_final(its chunk's winner) x P_chunk(option), which sums to 1; `choice` = its argmax; `confidence` over all n options | one single-token letter per option is the decider's own limit (`decider_bonsai.LETTERS`, 26; the served Bonsai does not answer double letters, skill_match LABELS 2026-09-27). The method is the rounds of `verify_moment` CHUNK (git e360d37) and docs/JJAVA.md 4.3. **Unmeasured**: how the composite's accuracy and calibration compare with a single read. `x_yamadori.rounds` says when it was used; `x_yamadori.answers.<id>.rounds` has each stage. |
| numbers | floats rounded to 6 decimals | Jev's docs print 2 decimals; the wire precision is not documented. Every number is a float and the usage counts integers, as [py]'s strict models require. |

Reproduced: all 13 [examples], field for field (`mcp/test_jev_api.py`
`test_doc_examples`: the fake reads each example's probabilities; the answer
must have exactly the example's fields, its noul / choice / probabilities /
legend exactly, its confidence and score within the rounding of the two
decimals the docs print). The two illustrative widget values on the Score
page that no formula fits (docs/JJAVA.md 2) are not API response examples
and are not in the set.

## 4. Response, models and usage

| Jev | ours | difference, and why |
|---|---|---|
| `model`: "The model that performed the evaluation"; an alias reports the versioned id [models] | `jjava-<model>` of the model that read it (`jjava-bonsai`, ...) | an alias (`jjava-latest`, `jev-*`) is resolved to the versioned id the same way. The alias used is in `x_yamadori.jjava` {requested, alias_of}. |
| `jev-latest` / `jev-preview` -> `jev-1.13.0` [models] | all three -> `jjava-latest` | so an unmodified Jev client works; recorded. |
| (one model serves every id) | `jjava-latest` = whichever main model holds the card (`max_mode.decide(None, utility=True)`: the holder, else the loaded one, else the default); `jjava-<model>` only while that model is the one on the card, else **529** with Retry-After | a decider call never swaps the card (layout v2, max_mode). |
| `usage` `{input_tokens, output_tokens}` [api] | counted from the reads made: `input_tokens` = the first read's cached prefix + every read's processed tokens (`timings.prompt_n`, re-reads included); `output_tokens` = the reads (one generated token each) | Jev "ingests the state once" [models]; ours counts the state once and each question's suffix per order and per re-read. Timings absent: the reads' whole prompts (an upper bound), said in `x_yamadori.usage.method`. Jev's own numbers are not comparable (its example: 20 output tokens for one noul). |
| questions "evaluated in parallel" [models, fan-out] | read one after another on one slot, the state cached | llama-server; each question costs its own suffix in two orders (decide_turn latency table: p50 496 ms, p90 577 ms per question). `x_yamadori.parallel` is `false`. |
| `GET /v1/models` -> `{models: [{name, description, release_date}]}` ([py] ListModelsResponse / ModelMetadata; [js] ModelCard) | exactly that; `jjava-latest` and `jjava-<model>` for each main model this stack configures; `release_date` 2026-09-29 (typed/2, the readout in Jev's API shape) | availability (on the card or not) and whether the model's priors are measured (`letter_prior`, `label_bias` in `bench/decider/results/models/<model>.json`; none is today) are in `x_yamadori.models`; Jev's ids are listed as `x_yamadori.aliases`, not as models (Jev also accepts ids its list does not show [models]). |

## 5. Errors

| Jev status [api] | ours | body |
|---|---|---|
| 401 missing or invalid key | the same (+ `WWW-Authenticate: Bearer`) | `{"detail": "<why>. Check the Authorization header ..."}` |
| 422 validation; "the body details the offending field" | the same; **every** offending field at once; malformed JSON is 422 `json_invalid` | `{"detail": [{"loc": ["body", ...], "msg": ..., "type": ...}]}`: FastAPI's validation form, which [js-src] parses into "questions.q.criteria: ..." (gated with a port of `extractMessage`). Jev's exact body is not published; this is the form its SDK reads. |
| 429 too many requests | a second Jev call while one is on the lane: 429 **at once**, `Retry-After: 1` | `{"detail": ...}`. The lane is one slot; a refusal, never a hang. 1 s is the header's smallest unit (a question takes p50 496 ms). |
| 529 overloaded | a named model not on the card (Retry-After from `max_mode.retry_after()`); the model server unreachable, or its 5xx / 429 (Retry-After 30, `max_mode.RETRY_AFTER_UNKNOWN`) | `{"detail": ...}` |
| (5xx: [py] `TypeSafeInternalServerError`) | 500 our own failure; 503 jjava switched off (`YAMADORI_DECIDER=0`) | `{"detail": ...}` |

Retries: [api] says the SDKs' default retry policy handles 429 and 529 with
backoff, and [models] that they honour `retry-after`; both of ours carry
it.

## 6. Capacity, records, and what is not proven

- **The lane.** Questions run on the child slot (`slots.acquire(None,
  transient=True)`), rank 3, `budget.LANE_TOKENS` = 3,072 cells kept in
  VRAM (layout v2). A state + question larger than the lane runs anyway,
  past the VRAM line, slower; `x_yamadori.lane` {cells, needed, fits, slot,
  release}. The lane is kept after the call, as for every decider turn.
- **The card.** A call on a model takes a `max_mode.Lease` while it runs, so
  a higher tier cannot swap the model out under it.
- **Records.** One corpus event per call, kind `jev_call` (never `turn`),
  with the account and its traffic class (`corpus.account_traffic`: the live
  suite's accounts are `test`), the status, usage, the lane, the answers and
  a sha1 of the state -- never the state or the instructions.
- **The SDK test, built (2026-09-30), not yet run with the SDKs:**
  `mcp/test_jev_sdk.py` runs each SDK (`mcp/jev_sdk/driver.py`,
  `driver.mjs`) against a local server of `server.app` with the stubbed
  decider (offline), or `--live` against the proxy's `/jev` routes (NOT RUN
  until a deploy has them). The pins, from the registries' metadata:
  `typesafe-sdk` 0.7.2 wheel sha256 `0a961148...97d43d1e` (36,448 bytes,
  MIT) and `@typesafe-ai/sdk` 0.6.0 integrity `sha512-IddX+Q0X...e6Jaw==`
  (MIT, no dependencies), recorded in models/manifest.yaml runtimes
  `typesafe-sdk-python` / `typesafe-sdk-js`; `tools/typesafe-sdk/install.py
  --run` installs them into `tools/typesafe-sdk/py/.venv` and
  `tools/typesafe-sdk/js/node_modules` (never the stack's Python) and writes
  `locks/typesafe-sdk.lock.txt`. `scripts/run_tests.py` skips the suite, with
  a note, until both exist (REQUIRES).
- **Not proven:** no live run (the routes reach the proxy at the next
  deploy); no TypeSafe SDK has been pointed at it yet. The plan: pin
  `typesafe-sdk==0.7.2` (Python, the version in its changelog, 2026-09-26)
  and `@typesafe-ai/sdk@0.6.0` (JS, its `VERSION`), set `base_url` to
  `<base>/jev`, and check: `system_one` returns a `SystemOneResponse` under
  the SDK's strict models for each of the 13 doc examples; `.nouls`,
  `.choices`, `.scores` split them; `models.list()` parses; a 422 raises
  `TypeSafeUnprocessableEntityError` / `UnprocessableEntityError` whose
  message names the field; a 429 raises the rate-limit error with
  `retry_after_ms` / `retryAfterMs` = 1000 and the default policy retries
  it; a 529 is retried; `request_id` is the header's value.
- **Not measured:** the two-stage composite's accuracy against one read, the
  noul-criteria rendering on Bonsai, anything on a model other than Bonsai
  (docs/JJAVA.md 7).
