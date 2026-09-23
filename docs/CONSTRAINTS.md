# Constraints on generation — inventory and verdicts

Every limit applied to model generation on the serving path (llama-swap,
llama-server, proxy, tools, worker, second context, Caddy, watchdog), with the
evidence each one cites, whether that evidence holds, whether the limit is
actually enforced, and a verdict.

- **KEEP**: measured, and it protects a real resource.
- **FIX**: the idea is right and the number is wrong. The correct value is derived below.
- **REMOVE**: unmeasured, or it treats a symptom.

Written 2026-09-22. This was a read-only pass plus live probes. No code was
changed. `mcp/proxy.py` was being edited by another process while this was
written: its line numbers moved twice in an hour. Proxy lines are given as
`function (line at time of writing)`.

**The one-paragraph version.** The main cause of false "the model is broken"
conclusions is the thinking cap, not a hop count. `--reasoning-budget 1000` cuts
the model's thinking off in the middle. At least once in two probes the model
then carried on thinking in the *answer* channel until `max_tokens` ran out.
Every `max_tokens` floor in the repo was sized against a 1000-token thought
followed by a short answer. Measured on three prompts, the model's natural
thinking length is 682 to 2,826 tokens, and it sometimes does not stop at all
(≥20,000 tokens, 2 of 3 runs on one prompt). The second cause is a tier that
cannot work: `reasoning_effort: "high"` is sent to a chat template that accepts
only `low | medium | xhigh`, so every `high`-tier request fails with HTTP 500
upstream and 502 at the proxy.

---

## 1. Live measurements this session

Direct to llama-swap `:11434`, model `bonsai`, temperature 0.3, one request at a
time. Another client held one or two other slots during most runs, so the tok/s
figures are shared-GPU numbers (13.7 to 43.8 tok/s). Reasoning and content were
counted with the server's own `/tokenize`. **n=1 per cell.** These are facts
about what happens, not distributions.

### 1a. What the server actually does

| fact | how it was established |
|---|---|
| `--reasoning-budget 1000` is a hard token cap. Reasoning stopped at **999** tokens (and at 1,017 with a budget message). | `ts_code`, effort low and medium, default budget |
| After the forced close, the model **can keep deliberating in `content`**. One run: 999 reasoning tokens, then **5,000 content tokens** of continued thinking ("`? If A=any, keyof A = string? B maybe. Fine.` …"), then `finish_reason: length` at 6,000. No answer was delivered. | `ts_code`, effort low, max_tokens 6000, 158.9 s |
| This is stochastic. The same prompt at effort medium closed at 999 and gave a clean 310-token answer. | 1 of 2 runs without a budget message spilled over |
| With `reasoning_budget_message` set, the answer was clean in **2 of 2** runs: budget 1000 gave a 301-token answer, and budget 4000 gave a 430-token answer. | same prompt |
| **A per-request override exists and works.** `reasoning_budget_tokens` (alias `thinking_budget_tokens`) is honoured. With 20000 set, reasoning ran to 19,987 and 19,998 tokens. | `llamacpp-prism-official/tools/server/server-common.cpp:1354-1355` (commit 9a9394a, the running build `b1-9a9394a`) |
| `-1` per request means "use the server's value". A request cannot ask for *unlimited*; it can ask for a large number. | same file, `:1356-1357` |
| The template accepts **only** `low`, `medium`, `xhigh`. `high` and `minimal` return HTTP 500 (`raise_exception('Unexpected reasoning effort …')`, template line 48-49). The template's own default is `xhigh`. | `/upstream/bonsai/props` chat_template, plus a direct 1-token request per value |
| Four slots, unified KV, and **each slot reports n_ctx 147,456** (not 131,072 as the budget/admission docstrings say). | `/upstream/bonsai/slots`, `/props` |
| llama-server's own HTTP read/write timeout is 3600 s (default `-to`). | `llama-server --help` |

### 1b. Natural thinking length (budget lifted to 20,000 per request)

| effort | `short_fact` (4-sentence Rust answer) | `ts_code` (typed deepMerge, TS) | `algo` (subarray-sum mod k + Python) |
|---|---|---|---|
| low | R 1,063 / A 108 | **R ≥19,987 (hit the 20k cap)** / A 440, 519 s | R 2,826 / A 283 |
| medium | R 818 / A 134 | R 2,509 / A 236 | R 1,138 / A 597 |
| high | **HTTP 500, template rejects it** | 500 | 500 |
| xhigh | R 682 / A 89 | **R ≥19,998 (hit the 20k cap)** / A 331, 514 s | R 1,597 / A 263 |

R = reasoning tokens, A = answer (content) tokens.

As deployed (budget 1000), effort low: `short_fact` completed twice (R 722 and
811). `ts_code` spilled over as described in 1a.

What this supports, and what it does not:

1. **1000 is below the natural length of an ordinary task.** 5 of the 7 runs
   that finished on their own used more than 1000 reasoning tokens, including
   the trivial 4-sentence question once (1,063).
2. **Effort does not order thinking length at this n.** xhigh used fewer tokens
   than low on two of three prompts. The template's effort sentence is a style
   instruction, not a budget. Nothing here justifies per-tier *token* numbers
   derived from effort.
3. **Some prompts do not converge.** `ts_code` ran to the 20k cap twice. That is
   rumination, not depth. A cap has to exist as a *runaway breaker*. It cannot
   be removed.
4. **A budget message is what makes a cap safe.** Without one, a forced close can
   hand the rest of the thinking to the answer channel. With one, 2 of 2 runs
   closed cleanly. n=2, so this needs repeating, but it is the documented
   mechanism for exactly this failure and it costs nothing.

### 1c. Corroborating record already in the repo

- **LiveCodeBench.** 24 rows across `bench/lcb_*.jsonl` ended
  `finish: "length"` with **14,113 to 26,945 characters of non-empty content**.
  12 of them were scored "no code in the reply". A reply that long which reaches
  `length` without a code block matches spilled-over reasoning. The content was
  not stored, so this is circumstantial, and it is labelled as such. These rows
  were recorded as model failures.
- **`arc193_a`**, the case `config.yaml` and `docs/KNOWN-ISSUES.md` cite as a
  "compute-buffer spike" (the reason given for backing `-c` out of 196,608):
  every recorded attempt is either an HTTP 502 / connection refused with
  `ms: 0`, or (the one completed run, `lcb_results_restart_contaminated.jsonl`)
  `finish: length` with 26,572 characters and "no code in the reply". **No row
  shows an OOM.** The `-c 147456` decision still stands on its own VRAM
  measurement (3,931 MiB free, warm), but the arc193_a attribution is
  unsupported.
- **Context-economy run** (`bench/context_economy_results.jsonl`, n=26 second-context
  rows). Helper tokens *summed across hops* have median 15,798 and max 64,460.
  The helper's *peak window* has median 5,344 and max 9,400. Findings: 2 of 26
  are exactly 1,400 characters, which is the cap.
- **Proxy corpus** (`index/corpus.sqlite3`). 25 of 99 tool-using turns reached
  hop index 11, which is the `MAX_TOOL_HOPS` landing. All 25 fall between
  2026-09-21 11:07 and 2026-09-22 01:50. **Zero have occurred since**, across
  traffic up to 17:26 today. None of the 25 logged an answer.

---

## 2. Inventory, sorted by severity

Severity measures how likely the limit was to produce a false "system is dead /
model can't do X" conclusion.

| # | constraint | where | value | limits | enforced? | verdict |
|---|---|---|---|---|---|---|
| 1 | Server thinking cap | `config.yaml:192` | `--reasoning-budget 1000` | reasoning tokens per request, all clients | **yes** (live: stops at 999) | **FIX** → 8192 + `--reasoning-budget-message` |
| 2 | Tier `high` effort string | `mcp/tiers.py:171`, `ACCEPTED_EFFORTS` `:206` | `"high"` (also `"minimal"` passes `safe_effort`) | which template branch runs | **yes: HTTP 500 upstream, 502 to the caller** | **FIX** → map `high`→`xhigh`, drop `minimal`/`high` from `ACCEPTED_EFFORTS` |
| 3 | `summarize_text` budget | `mcp/code_search.py:1421` | `max(256, 3*max_words)` = **900** | whole reply (reasoning + answer), bypasses the proxy floor | yes (verified this session: 0 chars, `finish=length`) | **FIX** → derived rule, §3 |
| 4 | `summarize_text` failure return | `mcp/code_search.py:1429-1437` | `retryable=True`, "a shorter prompt usually completes" | what the agent does next | yes | **FIX** → state `finish_reason=length`, `retryable=false`, remedy owner = operator |
| 5 | Tier `max_tokens` floors | `mcp/tiers.py:155-177` | 1200 / 1200 / 1500 / 3000 / 6000 | total tokens per upstream call | yes, `tiers.apply` `:253` | **FIX** → one rule (§3); the per-tier numbers go |
| 6 | Proxy floor | `mcp/proxy.py:84`, applied in `prepare` (`:1032`) | `MIN_BUDGET=1500` | same | yes, and it overrides the 1200 tier floors, so minimal/low are really 1500 | **FIX** → becomes the rule's answer allowance |
| 7 | Helper "KV budget" | `mcp/shomen.py:473,503` | cumulative `total_tokens` ≥ 36,864 | ends an investigation early | yes, **on the wrong quantity** | **FIX** → compare the last hop's `prompt+completion` |
| 8 | Per-hop budget in the second context | `mcp/shomen.py:467` | `max_tokens: 2500` | each helper hop | yes | **FIX** → rule (§3); 2500 only fits the 1000 cap |
| 9 | Finding cap | `mcp/shomen.py:126`, `:595` | `MAX_FINDING_CHARS=1400` | what crosses back | yes, silently (2/26 truncated) | **FIX** → 6000 as a breaker, with a truncation marker |
| 10 | Streamed tool loop: no landing, answer regenerated | `mcp/proxy.py` `stream_body` (`:1610`, `:1646-1649`) | `MAX_TOOL_HOPS` | streamed requests | yes | **FIX** (defect, not a number): see item detail |
| 11 | Reasoning dumped as the answer | `mcp/proxy.py` `complete` (`:1229-1230`) | `[:8000]` chars | non-streamed truncation handling | yes | **FIX** → report `finish_reason=length`, never ship raw reasoning as content |
| 12 | Tool-result truncation | `mcp/proxy.py` (`:1280`, `:1642`), `shomen.py:551`, `fanout.py:166` | `out[:6000]` chars | what the model reads from a tool | yes, **silently** (3/651 corpus results ≥6000) | **FIX** → keep as a breaker, append a marker naming the cut and the next call |
| 13 | Fan-out timeout | `mcp/fanout.py:187` | 900 s, non-streamed, 3 concurrent | high/max tiers | yes (time to first byte = whole generation) | **FIX** → stream, or ≥1800 once floors rise |
| 14 | Caddy upstream timeouts | `caddy/Caddyfile:56-58` | read/write 1800 s | public non-streamed requests | yes | **FIX** → 3600 s, or require streaming for `max` (measured 1,423–1,752 s) |
| 15 | Proxy hop breaker | `mcp/proxy.py:69` | `MAX_TOOL_HOPS=12` | tool hops per request | yes | **KEEP** as an unadvertised breaker |
| 16 | Helper hop breaker | `mcp/shomen.py:100` | `MAX_HOPS=16` | helper tool hops | yes | **KEEP** as a breaker |
| 17 | Main lanes / admission wait | `mcp/admission.py:51,55` | 2 lanes, 20 s, then 429 + `Retry-After: 30` (`server.py:110`) | concurrent main requests | yes | **KEEP** (harnesses must treat 429 as not-run) |
| 18 | Helper lane | `mcp/admission.py:52`, `:131-169` | 1 lane, 20 s | second contexts | yes | **KEEP** |
| 19 | Main / helper KV shares | `mcp/budget.py:80-85` | 60% / 25% / floor 50% | nominal split of the pool | **main: never read by anything that gates.** helper: yes (item 7) | **REMOVE** the main figure as a claim, or wire it; fix the 131072 fallback (`:121`) to fail loudly |
| 20 | Context, KV, batch | `config.yaml:173-177` | `-c 147456`, q8_0 K/V, `-b 1024 -ub 512`, 4 auto slots | pool size | yes | **KEEP** (VRAM-measured). Drop the arc193_a rationale. |
| 21 | `--no-context-shift` | `config.yaml:201` | on | fail instead of silently dropping context | yes | **KEEP** |
| 22 | Default effort / temperature | `config.yaml:118,122` | `reasoning_effort?: medium`, `temperature?: 0.3` | defaults only | yes | **KEEP** (measured: temp 0.3 about 15% fewer tokens) |
| 23 | Proxy upstream socket timeout | `mcp/proxy.py:138,221`; `streaming.py:101,197` | 1800 s per read (streamed, so it measures the gap between tokens) | stalled upstream | yes | **KEEP** |
| 24 | Proxy retry on empty drop | `mcp/proxy.py:273-277` | 1 retry | transport | yes | **KEEP** |
| 25 | Helper non-streamed timeout | `mcp/shomen.py:203` | 900 s | one helper hop | yes | **KEEP** after the rule (8192+1500 tokens ≈ 240–700 s); revisit if it trips |
| 26 | Extraction worker | `mcp/worker.py:88-91` | 8192 tokens, 12k-char chunks, 60 chunks, 1800 s | recipe extraction | yes | **KEEP**, raise to R_cap+2048 if the cap rises above 6k |
| 27 | Watchdog | `scripts/watchdog.ps1:49-50` | `/health` only, 15 s, 10-min cooldown | restarts | yes | **KEEP** (the rewrite is doing its job; no llama-swap restarts in the current log) |
| 28 | uvicorn | `mcp/server.py:316,319` | keep-alive 300 s, graceful 120 s | idle sockets / shutdown | yes | **KEEP** |
| 29 | llama-swap | `config.yaml:19` | `healthCheckTimeout: 900` | model load | yes | **KEEP** |
| 30 | Laya / router timeouts | `shomen.py:235`, `fanout.py:242`, `laya_router.py:103` | 20 / 30 / 120 s | decision service | yes | **KEEP** (nothing gates on Laya) |
| 31 | Routing evals' budget | `scripts/eval_prompt_style.py:134`, `scripts/eval_routing.py:111` | `max_tokens: 400` on a thinking model | first-call measurements | yes | **FIX** before re-running; the AGENTS.md routing numbers are at risk (§4) |
| 32 | Retrieval caps | `code_search.py:53-91`, `hints.py:126,140`, find_* result caps | top-k 5, candidates 40, rerank ≤2, hint floor 0.55, 30/40-row caps | tool output size | yes | out of scope; not generation constraints (the reranker is covered in `FINDINGS.md` #20) |

---

## 3. The recommended single rule for token budgets

> **Superseded 2026-09-22 (operator decision).** The fixed `R_CAP = 8192`
> breaker below no longer exists. Thinking is now derived from the context
> split in `mcp/budget.py` -- main 5/8 of the pool, one helper 3/8, so at
> `-c 147456` main 92,160 and helper 55,296, and at the `-c 163840`
> `config.yaml` launches with now main 102,400 and helper 61,440, reserve 0
> (see the 2026-09-22 note under item 19 for why it is no longer 1/2 +
> 2 x 1/4):
>
> ```
> answer   = max(client.max_tokens, A_MIN)                 # A_MIN = 2048
> thinking = max(share[role] // share_n - prompt - answer, 1024)
> upstream.max_tokens              = thinking + answer
> upstream.reasoning_budget_tokens = thinking
> ```
>
> `role` is `helper` for deep thinking (shomen), `main` otherwise; a fan-out
> of n samples uses `share_n = n`; a per-request `reasoning_cap` overrides
> `thinking`, clamped to [1024, pool]. `A_MIN` and "add, not max()" below
> still hold. The split is a decision, not a measurement: it replaced
> 60/25/15, which item 19 found had nothing behind it. The text below is
> left as written.

**The client's `max_tokens` is an answer allowance. The server adds the thinking
allowance on top. Thinking is capped by a breaker, not by a quality target.**

```
R_CAP   = 8192        # thinking breaker. Server --reasoning-budget, and sent
                      # per request as reasoning_budget_tokens, with
                      # reasoning_budget_message so a forced close ends cleanly.
A_MIN   = 2048        # smallest answer allowance ever sent

upstream.max_tokens        = R_CAP + max(client.max_tokens or 0, A_MIN)
upstream.reasoning_budget_tokens = R_CAP
upstream.reasoning_budget_message =
    "\n\nThinking budget reached. I will stop deliberating and write the final answer now.\n"
```

How each number is derived:

- **R_CAP = 8192.** The longest natural completion measured was 2,826 reasoning
  tokens (effort low, `algo`). The 1000 cap was below 5 of the 7 natural
  completions. 8192 is about 2.9× the largest natural completion. At the
  measured 40–44 tok/s, it bounds rumination at about 190–205 s. The two
  rumination runs were forcibly closed at 20k after about 515 s, and 4000 plus a
  message had already produced a clean answer. So 8192 is a breaker a healthy
  request should not reach. The value 4096 would leave only 1.45× headroom over
  a single observation. **n is small (7 natural completions across 3 prompts),
  so this is a breaker value, not a fitted p95.** Re-measure at n≥30 per effort
  on the LCB and api_truth prompts before tightening it. Uniform across efforts,
  because effort does not order thinking length in these data.
- **A_MIN = 2048.** Measured answers ran 89–597 tokens on these prompts. LCB
  completions (thinking included) have a median of 1,124–1,700 and run to 4,595
  when they finish. 2048 covers a full code answer. It is a floor under what the
  client asked for, never a ceiling on it. A client that asks for 16k gets
  R_CAP + 16k.
- **Why add instead of max().** `max(client, floor)` gives a client that sent
  4000, expecting a 4000-token answer, a 4000-token *total*, and the thinking
  eats most of it. Adding makes the answer allowance mean what it says whatever
  the thinking length.

Applied to each call site:

| call site | today | under the rule |
|---|---|---|
| proxy, every tier (`MIN_BUDGET`, `tiers.floor`) | 1500–6000 total | 8192 + max(client, 2048). The `floor` key and `MIN_BUDGET` go. `MIN_BUDGET` becomes `A_MIN`. |
| `summarize_text` | 900 total | 8192 + ceil(1.6 × max_words) (1.6 tokens/word covers identifiers and paths) → 8,672 at the 300-word default. Also send `reasoning_effort: "low"`. |
| `shomen` per hop | 2500 | 8192 + 1500 (a tool call, or a finding of ≤200 words ≈ 300 tokens, with headroom) |
| `worker` extraction | 8192 | 8192 + 4096 (a JSON array of recipes from a 3k-token chunk) |
| `fanout` variants | inherits payload | inherits the rule. The timeout becomes streaming or ≥1800 (item 13). |

**Also required for the rule to hold.** Detect `finish_reason == "length"`
everywhere and report it as a budget event, both in the proxy's streamed and
non-streamed paths and in every tool that calls the model. A truncated answer
must never read as the model's answer or as "the model returned nothing".

---

## 4. Per-item detail

### 1. `--reasoning-budget 1000` (config.yaml:192) — FIX

**Cited rationale** (config.yaml:180-192): "at 300 tokens this model returns 0
characters … This is what makes xhigh safe … With the proxy flooring max_tokens
at 1500, there is always room left for an answer after a 1000-token thought."
The source is the quant card's "a 256 token cap leaves most answers empty" and
commit fdc9067 (max_tokens 300 → 0 chars, 1200 → complete). That is one task,
n≈2.

**What that evidence shows.** It shows that `max_tokens` must exceed the
thinking. It does not show that the thinking should be *cut* at 1000; no
measurement in the repo sized the 1000 itself.

**Live.** Natural thinking is 682–2,826 tokens, and 5 of 7 natural completions
exceed 1000. A forced close without a message produced 5,000 tokens of
continued deliberation in `content` and no answer, in 1 of 2 runs. "Makes xhigh
safe" is false twice over. Effort does not change the length at this n. And the
cap turns every effort level into the same 1000-token thought followed by
possible spillover.

**Fix.** Set `--reasoning-budget 8192` and `--reasoning-budget-message "…"`.
Also send `reasoning_budget_tokens` per request from the proxy, so the value is
in one place in code rather than relying on the launch flag. §3 has the
derivation.

### 2. `reasoning_effort: "high"` (tiers.py:171, :206) — FIX

`tiers.ACCEPTED_EFFORTS` claims to be "empirical, not copied from a spec". It
lists `minimal` and `high`, and the running template rejects both with HTTP 500
(template lines 47-49, reproduced this session). The `high` tier sets
`effort: "high"`. `safe_effort("high")` passes it through. `_post` gets a 500,
retries once and raises, and the ASGI server returns **502**. Any client that
sends the OpenAI-standard `reasoning_effort: "high"` gets a 502. This is a whole
tier that cannot answer, with an error that reads as a transport fault.
`config.yaml:36-37` already said "valid values are low/medium/xhigh only". The
two files disagree, and the config file is right.

**Fix.** Set `ACCEPTED_EFFORTS = {"low", "medium", "xhigh"}`. `safe_effort` maps
`high`/`max`→`xhigh` and `minimal`/`none`→`low`. Add a startup assertion that
sends one 1-token request per accepted value, which would have caught this.

### 3–4. `summarize_text` (code_search.py:1407-1450) — FIX

The call is `max_tokens = max(256, max_words*3)`, which is 900 at the default,
sent to llama-swap directly. It never passes through the proxy floor. It has no
`reasoning_effort`, so it runs at the llama-swap default `medium`. The operator
verified on the real GPU that 900 tokens went entirely to thinking and 0
characters of answer came back. The live data explains why: the trivial
`short_fact` alone used 682–1,063 thinking tokens.

The failure return is worse than the budget. It says `retryable: True` and
"a shorter prompt usually completes". Both are false. The same budget fails the
same way, and shortening the input does not shorten the thinking. By this
repo's own rule ("failure returns carry the next step"), this is a return that
causes a retry loop.

**Fix.** Apply the §3 budget. Check `finish_reason`, and on `length` return
`BUDGET_EXHAUSTED`, `retryable: false`, `fixable_by: operator`.

### 5–6. Tier floors and `MIN_BUDGET` (tiers.py:155-177, proxy.py:84) — FIX

**Cited:** "at max_tokens=300 … 0 characters … At 1200 both return a complete
answer" (n=2 prompts, commit fdc9067). No measurement justifies 3000 or 6000;
they are guesses that scale with the tier. Because `MIN_BUDGET` (1500) is
applied after `tiers.apply`, the 1200 floors of `minimal` and `low` are dead
code. The real floors are 1500 / 1500 / 1500 / 3000 / 6000.

Under a 1000 cap, 1500 leaves 500 answer tokens. That is enough for a
4-sentence answer. It is not enough for code: LCB answers that finished run to
4,595 completion tokens. It also does nothing about spillover. **Fix:** replace
with the §3 rule.

### 7. Helper KV budget (shomen.py:473, :503) — FIX

`spent += usage.total_tokens` on every hop. `prompt_tokens` is the *whole* prompt
each hop, so this sums the conversation over and over and grows quadratically.
What the helper actually holds in KV at hop h is `prompt_h + completion_h`. In
the n=26 context-economy run, the sum had median 15,798 (max 64,460) against a
peak window of median 5,344 (max 9,400): a 3.1× median overcount. The check
would have ended 4 of 26 investigations "at budget" while their real footprint
never exceeded 9,400 of the 36,864 allocation (25%).

`docs/FINDINGS.md` #13's "spent 51,206 tokens against a helper allocation of
36,864 — 39% over" is almost certainly this same sum, not a KV overrun.

**Fix.** Compare `usage.prompt_tokens + usage.completion_tokens` of the latest
hop against `_helper_budget()`. Keep the sum as telemetry.

### 8. shomen `max_tokens: 2500` (shomen.py:467) — FIX

2500 is the only derived number in the repo (1000 thinking + 1500 answer). It
is correct only while the cap is 1000. Under §3 it becomes 8192 + 1500.
`bench/context_economy.py:302` hardcodes the same 2500 and must follow it, or
the benchmark will not measure production.

### 9. `MAX_FINDING_CHARS = 1400` (shomen.py:126) — FIX

**Cited:** "the finding must be small or the whole point is lost". The prompt
asks for "under 200 words". Measured: 2 of 26 findings were cut at exactly 1400
characters, with no marker. The citation check runs on the untruncated text, so
the receipts are right while the text the caller reads is cut mid-sentence. The
measured saving (531 tokens median crossing, n=26) does not depend on this cap.
**Fix:** 6000 characters as a breaker, plus a visible `[finding truncated at N
chars; full text under trace handle X]`.

### 10. Streamed tool loop (proxy.py `stream_body`) — FIX

These three are not numbers, but they are how a constraint turns into a
misleading result:

- (a) When the loop reaches `MAX_TOOL_HOPS` there is **no landing**. `complete()`
  withdraws the tools and asks for the answer, but `stream_body` streams the
  final call with the tools still attached. The model can then call one of
  *our* tools, and that call is forwarded to a client that does not have it.
- (b) Every streamed answer is **generated twice**. The loop's last non-streamed
  response, which has no tool calls, is discarded and regenerated through
  `stream_upstream`. That doubles wall clock on every streamed turn. Because
  sampling runs at temperature 0.3, the regeneration can also differ, or call
  a tool.
- (c) `corpus.log_answer(..., MAX_TOOL_HOPS, ...)` logs 12 as the hop count for
  every streamed answer. The corpus shows 17 answers at "hops 12", and 10 of
  them made zero tool calls.

### 11. Truncated-reasoning fallback (proxy.py `complete`) — FIX

When content is empty and reasoning exists, the proxy returns up to 8,000
characters of raw reasoning as the answer, prefixed "raise max_tokens". Two
problems. It exposes chain-of-thought as the reply. And it misses the spillover
case entirely, where content is *non-empty* and is itself reasoning. **Fix:**
key on `finish_reason == "length"`, return a short structured notice with the
budget that was used, and never return reasoning text as `content`.

### 12. Tool results cut at 6000 characters — FIX

This is applied in four places with no marker. `read_file_range` returns 120
lines by default, and at about 50 characters per line plus 7 of line-number
gutter it can exceed 6000. The model then gets a file range whose header
promises lines the body does not contain. It is rare (3 of 651 corpus results),
so the severity is low. Keep 6000 as a breaker and append
`[truncated at 6000 of N chars; request lines X-Y with read_file_range]`.

### 13–14. Fan-out 900 s and Caddy 1800 s — FIX

`fanout._tool_loop` is non-streamed. The 900 s timeout therefore covers the
whole generation, for 3 variants in parallel on a GPU that also serves the main
lanes. At the current 3000/6000 floors that is tight. Under §3 it is not
survivable, since 3 × (8192 + 2048) tokens on shared decode can exceed 900 s.
Caddy's 1800 s read timeout covers a non-streamed public request of the `max`
tier, which was measured at 1,423–1,752 s. The Caddyfile comment also still
says `:1234` is llama-swap; it is the proxy.

### 15–16. `MAX_TOOL_HOPS = 12`, `MAX_HOPS = 16` — KEEP as breakers

`FINDINGS.md` #27 already reframed these, and the corpus supports it. 25 trips,
all before the tool-result fix; zero since. Nothing tells the model about them.
They protect a real resource: a loop holds one of two main lanes, or the single
helper lane. Two changes: log a trip explicitly (for example a
`breaker_tripped` event), and fix 10a so the streamed path lands too.

### 17–18. Admission lanes — KEEP

The rationale is real: three concurrent benchmark processes degraded each other
into 502s. A 429 with `Retry-After` is honest. Two notes. The docstring's "the
number comes from the KV pool" leans on a main budget (88,473) that nothing
enforces (item 19). And `fanout` (3 generations), the extraction worker and
`summarize_text` all call llama-swap without taking a lane, so the lane count
under-states real concurrency. Benchmarks must record a 429 as "not run", never
as a failure.

### 19. `budget.py` shares — REMOVE the unenforced claim

`budget.cap_for` and `budget.fits` have **no callers**. The main-conversation
budget of 88,473 is shown on the vitals page and quoted in docstrings, and
nothing gates on it. The helper share is enforced, on the wrong quantity (item
7). The 131,072 fallback (`:121`) silently mis-sizes when the server cannot be
reached, and its own docstring flags it. Either wire `main` to something real
or stop presenting it as a budget. The fallback should fail loudly instead of
returning a guess.

> **Update 2026-09-22.** The operator replaced the 60/25/15 split with main
> 1/2 and 1/4 for each of up to two helpers (`HELPERS = 2`,
> `admission.HELPER_LANES = 2`): at 147,456 that is main 73,728, helper
> 36,864 x2, reserve 0. `main` is now wired to something real -- every
> request's thinking budget is derived from its role's share
> (`tiers.budget`, §3 note). The 88,473 and 22,119 above are the retired
> split. The 131,072 fallback is unchanged.

> **Update 2026-09-22, later the same day: 5/8 + 1 x 3/8.** The operator
> replaced 1/2 + 2 x 1/4 with main 5/8 and ONE helper at 3/8 (`HELPERS = 1`,
> `admission.HELPER_LANES = 1`): at 147,456 that is main 92,160 + helper
> 55,296, reserve 0; at the 163,840 `config.yaml` now launches with, 102,400 +
> 61,440. Why: two concurrent helpers required two conversations each
> investigating at the same moment -- within one conversation the tool calls
> run one at a time under `admission.helper_lane()`, so the second helper's
> quarter was held for a context that never existed, and that held quarter
> capped main's window at half the pool. Unified KV allocates cells on demand,
> so the change costs no VRAM: a stress test at `-c 163840` with main and the
> helpers filled measured idle 13,868 MiB, peak 13,905 MiB, min free 2,406
> MiB. A second concurrent investigation now waits for the lane and is
> refused with `HELPER_BUSY` if it cannot get it. The 73,728 / 36,864 x2
> figures above are the retired split.

### 20–22. Server launch flags — KEEP

`-c 147456` is measured (12,122 MiB used, 3,931 free, warm). Keep it on that
evidence alone. The arc193_a "compute-buffer spike" sentence should be removed
from `config.yaml` and `KNOWN-ISSUES.md`, because the only completed arc193_a
row died of `max_tokens`. q8_0 KV, `-b 1024 -ub 512`, 4 auto slots with unified
KV, and `--no-context-shift` are VRAM or correctness measures and harmless. The
docstrings in `budget.py`, `admission.py` and `config.yaml:154-157` say slots
report n_ctx 131,072. Live, they report 147,456.

### 23–30. Timeouts — KEEP

The proxy's 1800 s is a *per-read* socket timeout on a streamed upstream, so it
measures the gap between tokens, not the length of the answer. Fine.

The `_post` docstring's "reaper" explanation for the 104 s / 215 s drops is
contradicted by `PROTOCOL.md` rule 13 (the drops correlated with watchdog
kills). Streaming does no harm, and that docstring should say its own
conclusion was reversed.

The watchdog is doing its job. The recent log shows no llama-swap restarts.
Its 15 s timeout is on `/health` only, and llama-server answers that from its
HTTP thread even during generation.

### 31. Routing evals at `max_tokens: 400` — FIX before citing again

`AGENTS.md` quotes first-call routing of 10.7/14 → 11.7/14 and a prohibition
curve (10.7 / 10.0 / 9.3). The scripts that produce first-call routing send
`max_tokens: 400` to a thinking model. On `short_fact` alone the model thinks
for 682–1,063 tokens. A request that runs out while thinking returns no tool
call and is scored as "no call", i.e. a mis-route. These scripts also record
tokens but not `finish_reason`.

The historical numbers cannot be re-checked from the repo, because there is no
results file. Their validity depends on thinking having stayed under 400 tokens
for every task, which is not what the model does today.

The scripts now also point at `:1234`, which is the proxy. The proxy executes
the stack's own tools server-side, so the first call the scripts try to observe
never reaches them. **Before re-citing those numbers**, re-run with
`max_tokens` from §3, directly against `:11434`, and record `finish_reason`.

---

## 5. Breakers that must stay, and what goes

**These stay, as unadvertised runaway breakers** (never stated to the model,
never reached by a healthy request, a trip is a defect report):

- the thinking cap: `R_CAP = 8192`, with a budget message
- `MAX_TOOL_HOPS = 12` (proxy), `MAX_HOPS = 16` (shomen)
- tool-result cut at 6000 characters and finding cut at 6000 characters, both
  with visible markers
- admission lanes (2 main, 1 helper), 20 s wait, then 429
  *(2026-09-22: briefly 2 helper lanes, then back to 1 the same day with a
  3/8 share (item 19); the thinking cap above is derived from the context
  split rather than fixed at 8192 -- see the §3 note)*
- `--no-context-shift`, `-c 147456`
- transport timeouts: per-read 1800 s on streams; 3600 s at llama-server;
  1800–3600 s at Caddy

**These are removed or replaced:**

- the per-tier `floor` values and `MIN_BUDGET` as a total. Both are replaced by
  the §3 rule (`MIN_BUDGET` becomes `A_MIN = 2048`, an answer allowance).
- `--reasoning-budget 1000` as a quality cap. It is replaced by the breaker value.
- `summarize_text`'s `max(256, 3*max_words)`, and its false retry advice.
- the cumulative-token helper budget check. It is replaced by a peak-window check.
- `ACCEPTED_EFFORTS` entries `minimal` and `high`.
- `budget.py`'s main-conversation budget as a stated constraint, until
  something enforces it.
- the `[:8000]` raw-reasoning fallback.

**Guardrails for the next change.** Every limit that ends a generation must
surface `finish_reason: length` (or the breaker that tripped) to the caller and
to the corpus. The failures in this document were recorded as "the model could
not do it" when the cause was a limit, and that cannot be seen unless the limit
reports itself.

---

## Probe provenance

The probe script and raw rows are in this session's scratchpad (`probe.py`,
`sweep.jsonl`, `natural.jsonl`, `followup.jsonl`). The numbers are reproduced in
§1 because the scratchpad is not kept. To reproduce: POST
`/v1/chat/completions` to `:11434` with model `bonsai`, temperature 0.3,
`reasoning_effort` ∈ {low, medium, xhigh}, and optionally
`reasoning_budget_tokens` / `reasoning_budget_message`. Count
`reasoning_content` and `content` with `/upstream/bonsai/tokenize`.
