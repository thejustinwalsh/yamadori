# Compaction: should the model think, and what makes a summary good (2026-09-24)

This note is research and an eval design. Nothing here was run, nothing touched
the GPU, and no code was changed.

Every claim carries one of these labels:

| label | meaning |
|---|---|
| **[paper]** | published with evidence (a paper or a technical report with numbers) |
| **[vendor-eval]** | a vendor's own measurement, published but not independently replicated |
| **[vendor-doc]** | documented vendor practice: official docs, or source code read at the cited path |
| **[repo]** | this repository: code read today, or a measurement in `docs/` with its n |
| **[opinion]** | my inference. It is not measured. |

Sources are numbered and listed at the end.

---

## 0. The short version

- **Thinking during compaction: there is no published evidence about agent
  compaction specifically, in either direction.** No vendor or paper reports
  thinking-on vs thinking-off for a compaction summary scored by whether the
  task continues correctly. I searched for one and found none.
- **The nearest evidence is conversation summarization on the same Qwen3
  hybrid checkpoints** (OmniCSEval [1]). Qwen3-8B and Qwen3-14B with thinking
  on (budget 8,192) against thinking off gave **+4.8 to +5.9 points of
  completeness**, +4.0 to +4.7 of conciseness, and **−0.3 to −0.4 of
  faithfulness**. Two other results push the other way. More reasoning budget
  "does not reliably improve summarization and can even reduce factual
  consistency" [2]. Qwen3's own report found that thinking slightly *degrades*
  long-context retrieval on RULER (Qwen3-32B: 93.7 off, 91.0 on) [3]. **[paper]**
- **Vendor practice is close to unanimous: the summarizer thinks at the
  conversation's own setting.** Anthropic's compaction runs with the request's
  thinking settings [4]. Codex compacts at the effort pinned for the context
  window, and its source says why: cache preservation [5]. OpenCode passes the
  conversation's model variant [6]. Claude Code's prompt asks for an
  `<analysis>` scratchpad before the `<summary>` [7]. Cursor gives its trained
  summarizer "scratch space to think" [8]. Cline allows reasoning and records
  the failure where reasoning ate the output budget [9]. Hermes makes it a
  per-task config and raises its timeout floor to 300 s for reasoning
  compressors [10]. **[vendor-doc]**
- **What actually goes wrong in compaction is not what thinking fixes.** Every
  measured compactor is weakest on the **artifact trail**: which files were
  touched (Factory: 2.19–2.45 out of 5 for all three methods [11]).
  Compactors also drop **user constraints**: 17% retained on average [12], and
  Claude Code's `/compact` kept 53% of safety rules after one round and 10%
  after five [13]. Summaries also **lengthen the remaining run**, by 13–15%,
  because they "smooth over" signals that the agent should stop [14]. The
  fixes vendors ship for these are **mechanical, not deliberative**:
  - Hermes' "Anchor Index (mechanically extracted, exact)" [10]
  - Cline's extracted Read/Modified file lists [9]
  - Claude Code re-reading the five most recently accessed files [15]
  - Cursor's searchable history file [16]
  - a constraint extractor, which took retention from 17% to over 90% [12]
- **Recommendation until the eval runs** (§5): thinking **on** at the
  conversation's own effort, **at every effort, medium included**. Raise the
  thinking budget from 1,024 to **2,048**. Raise the answer cap to honour
  Hermes' stated target. Thinking on is cache-free at every effort. Thinking
  off is cache-free only at medium. At medium today, thinking off also brings
  instruct sampling with `presence_penalty 1.5`, which is an unmeasured risk to
  verbatim identifiers.
- **The eval** (§4) is paired and continuation-based, and runs through `:1234`.
  Probes are mechanical, drawn from the span that was dropped: files touched,
  exact error strings, an injected user constraint, and the recorded next
  action. The primary comparison is OFF vs ON-2048 at effort medium, n = 100
  compaction points from about 50 trajectories. It needs two small code
  changes first: a header to force the arm, and a full-text capture of
  compaction requests. It costs about 15 GPU-hours per pass plus a repeat pass.
  **We do not have the data yet.** The corpus cannot supply items. Only 2 Bonsai
  agent trajectories exist today. The Hermes dogfood runs are planned, not run.

---

## 1. What the stack does today (verified in code)

`mcp/compaction.py` (`prefix_fields`, `renders_effort_line`),
`mcp/proxy.py` `_serve_compaction` (lines ~1906–2015) and `prepare` (~1761,
~1870–1898), `mcp/tiers.py` `compaction_budget` (~470–545). **[repo]**

| path | when | thinking | answer allowance |
|---|---|---|---|
| in place, **spliced** onto the stored prompt | Codex / Claude-Code shape; the history matches the proxy's stored conversation | kept, with `reasoning_budget_tokens = MIN_THINKING = 1024`, **only if** the stored prompt renders an effort line (thinking on AND effort `xhigh` or `low`); otherwise **off** | `compaction_budget`: `max(client, 5120)`, capped at 10,240 |
| in place, **as sent** | no stored entry (restart, eviction) | the same rule, applied to the request as the conversation's tier shaped it | same |
| flattened, **rewritten** | Hermes shape, mapped onto a stored conversation | the same rule, applied to the stored conversation | Hermes' "Target ~N tokens" is read as the client value |
| flattened, **as sent** | Hermes shape, nothing mapped | **off**. A utility call is overridden to tier `minimal` | same |

The operator's premise needs one correction. "It keeps the conversation's
thinking setting" is true only where the effort line renders. **At effort
`medium`, the `config.yaml` default, compaction runs with thinking OFF today**,
because `renders_effort_line` is false there. A medium-effort Hermes
conversation therefore compacts without thinking.

Consequences of the current rule. **[repo]**, with **[opinion]** where marked:

1. **Cache.** The effort line sits at the top of the system block, so turning
   thinking off at `xhigh`/`low` changes the rendered prefix and forces a full
   re-prefill. At about 490 tok/s (`docs/KNOWN-ISSUES.md`, short prompt;
   long-prompt prefill is unmeasured), a 40k-token conversation costs about
   80 s. At `medium`, on and off differ only in the generation prompt at the
   very end, so **both are cache-free at medium, and only "on" is cache-free
   at xhigh/low.**
2. **Sampling changes with thinking.** `tiers.enforce_sampling(body, thinks)`
   sends the vendor's instruct values when thinking is off:
   `temperature 0.7, top_p 0.8, presence_penalty 1.5`. So "thinking off" is
   not a single variable. It is thinking off plus a different sampler.
   **[opinion]** A presence penalty of 1.5 penalizes tokens that recently
   appeared. A summary is exactly the text that must repeat path prefixes
   (`js/`, `.js`, `mcp/`) and identifiers verbatim. llama-server's default
   penalty window is 64 tokens, so the effect is local, but it is still a
   plausible mechanism for corrupted identifiers. It is unmeasured.
3. **The 1,024 thinking budget sits below the model's natural thinking
   length.** `docs/CONSTRAINTS.md` §1b measured 7 finished runs at 682–2,826
   reasoning tokens, and 2 of 9 runs ran to the 20k cap. Only 2 of the 7
   finished within 1,024, and 5 of 7 within 2,048. These were coding and fact
   prompts, not summaries, at n = 1 per cell. §1a records that after a forced
   close the model once kept deliberating in `content` (1 of 2 runs without a
   budget message; 0 of 2 with one; compaction sends the message). **[opinion]**
   A spill-over in a compaction writes deliberation into the summary the
   client stores permanently.
4. **Hermes' target and our cap disagree.** **[repo, derived from source;
   not observed live]** Hermes computes the summary budget as 20% of the
   compacted content, clamped to [2,000, min(5% of the window, 10,000)]. In
   its default lean mode it adds about 4,000 tokens for a "Detailed Session
   Log", and the prompt states the sum as "Target ~N tokens" [10]. We
   advertise the main share, and `config.yaml` `bonsai` runs `-c 262144`, so
   5/8 = 163,840. 5% of that is 8,192, so the stated target reaches **12,192**.
   Our cap is **10,240**. A summary cut at 10,240 returns
   `finish_reason: length`. Hermes treats that as a truncated summary: it
   raises `RuntimeError`, marked with `_TRUNCATED_SUMMARY_MARKER =
   "finish_reason=length"`, and retries on the main model. The retry is a
   whole second compaction. This is `docs/PROTOCOL.md` rule 15 (instruct and
   constrain in opposite directions).
5. **Stale comment.** The comment above `COMPACTION_BUDGET` in `mcp/tiers.py`
   (~line 470) and AGENTS.md ("runs at `minimal` on the main model") describe
   the older rule. Today only the flattened as-sent path is `minimal`.

**The corpus** (`index/corpus.sqlite3`, 1,025 turns). There are 12 compaction
turns. Each request is stored truncated at 2,000 characters, and each answer
only as `{chars, cited_paths, hops, ms}`, with **no text**. Answers ran
7,733–41,702 characters (median about 20,700), and all 12 cite the same
project (`octopus-invaders`). They are therefore one or two sessions, **not
12 independent samples**, and they cannot be replayed or scored. **[repo]**

---

## 2. What the industry does

| system | who writes the summary | thinking during compaction | budget | recent turns verbatim | prompt structure | label |
|---|---|---|---|---|---|---|
| **Anthropic API compaction** (on demand `compact-2026-09-04`; threshold `compact_20260112`, default trigger 150k, minimum 50k) | the request's own model | **the request's thinking settings**. `max_tokens` "caps the whole call, including any thinking". The response carries no thinking. | the caller's `max_tokens`; the docs say "allow several thousand tokens" | optional ("compaction that keeps recent turns") | default prompt: state, next steps, learnings; `instructions` replaces it entirely | [vendor-doc] [4] |
| **Claude Code** `/compact` and auto-compact | the main model | an explicit **`<analysis>` block before `<summary>`**: a visible scratchpad | not stated in the prompt | the five most recently accessed files are re-attached | 9 sections: Primary Request and Intent; Key Technical Concepts; Files and Code Sections; Errors and fixes; Problem Solving; **All user messages**; Pending Tasks; Current Work; Optional Next Step, which quotes the latest work verbatim "to ensure there's no drift" | [vendor-doc] [7][15] (prompt from reverse-engineered extraction [7], v2.1.280) |
| **OpenAI Codex CLI**, local path | the session's model | **the effort pinned for this context window**. `reasoning_effort.rs`: "Sampling and compaction share the original request effort for this context window"; module doc: "Cache-preserving effort updates" | not capped in the prompt | **up to 20,000 tokens of the most recent user messages** (`COMPACT_USER_MESSAGE_MAX_TOKENS`); assistant turns and tool output are dropped | `prompt.md`: "CONTEXT CHECKPOINT COMPACTION ... handoff summary for another LLM": progress and decisions, constraints and preferences, what remains, critical data and references. Reinjected under a prefix telling the next model to "build on the work" | [vendor-doc] [5] |
| **OpenAI Responses API** `/responses/compact` and `compact_threshold` | server-side | not documented. The item is **encrypted and opaque**; it "carries forward key prior state and reasoning" | – | recent user messages retained | not visible | [vendor-doc] [17] |
| **OpenCode** | a `compaction` agent, defaulting to the conversation's model **and its `variant`** | follows the conversation's variant (effort) | core `SUMMARY_OUTPUT_TOKENS = 4096` | preserve 25% of the usable window, clamped to 2k–15k tokens (core default keep 8k). Old tool outputs are pruned first (protect 40k, prune at 20k) | Objective; Important Details; Work State (Completed / Active / Blocked); Next Move; Relevant Files. "Preserve exact file paths, symbols, commands, error strings". Previous summary merged, with "the conversation wins" | [vendor-doc] [6] |
| **Hermes Agent** | an auxiliary model (configurable; falls back to main) | a per-task `reasoning_effort`. A "fast lane" only when reasoning is explicitly disabled. Timeout floor of 300 s because "reasoning compression models can exceed" 120 s | **no `max_tokens`**, because "thinking models burn it on reasoning". Target 20% of the content, clamped to [2,000, min(5% of window, 10,000)], +4,000 for the session log | lean tail: 2.5% of the window, clamped to 10k–25k and ≤ 20% of the window; **user messages verbatim** (24,000 characters, newest first) | Historical Task Snapshot; Goal; Constraints & Preferences; Completed Actions (numbered: ACTION target → outcome [tool]); Active State; Blocked; Key Decisions; Errors & Fixes (quote the user's corrections); Resolved Questions; Relevant Files; Critical Context; Detailed Session Log; plus a **mechanically extracted Anchor Index** (PRs, SHAs, branches, files, errors, URLs). Iterative update of the previous summary. A temporal-anchoring rule against finished work phrased as a to-do | [vendor-doc] [10] |
| **Cline** (SDK) | the active model, or a configured summarizer | allowed. It records `reasoningChars`, and "output_budget_consumed_by_reasoning" as a failure class | `DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS = 8192` | `DEFAULT_PRESERVE_RECENT_TOKENS = 20000`; trigger at 0.9 of the window | Goal; State; Highlights; Next; **Files (Read / Modified, extracted mechanically, not written by the model)**. The docs say it reuses the prompt cache | [vendor-doc] [9] |
| **Roo Code** | "always uses your active conversation provider/model" (to avoid quality loss from switching) | not documented | not documented | not guaranteed; originals kept for checkpoint rewind | custom prompt allowed | [vendor-doc] [18] |
| **Aider** | the **weak model** | no | `max_chat_history_tokens`; recursive halving, up to depth 3 | the newest half is kept | "*Briefly* summarize"; **must** include function names, libraries and filenames; written in the first person as the user ("I asked you...") | [vendor-doc] [19] |
| **Cursor** | Composer trained with RL to self-summarize; earlier, a prompted summary | **yes, "scratch space to think"** before writing | trained summaries average **~1,000 tokens**, against 5,000+ for its tuned prompted baseline | the chat history is written to a file the agent can search after summarizing | goals, solutions, errors and fixes, remaining work, file paths | [vendor-eval] [8], [vendor-doc] [16] |
| **Letta / MemGPT** | the agent's model | – | evict about 50–70% of messages | the FIFO tail is kept; evicted messages stay searchable in recall storage | recursive summary in the first slot | [paper] [20], [vendor-doc] |
| **Manus** | – | – | – | prefers restorable compression: keep the URL or path, drop the content | KV-cache hit rate is "the single most important metric"; keep errors in context; recite a todo list | [vendor-doc] [21] |

Patterns across vendors, all **[opinion]** drawn from the table:

1. **Nobody switches thinking off to compact.** The two that say why (Codex in
   source, Roo in docs) cite cache or quality loss from changing settings
   mid-conversation. Hermes allows either, and its reason is latency, not
   quality.
2. **Everyone keeps something verbatim.** It may be recent turns (Codex 20k of
   user text, Cline 20k, OpenCode 2k–15k, Hermes 10k–25k), every user message
   (Claude Code, Hermes), or a re-read of recent files (Claude Code).
3. **The strongest harnesses stopped trusting the model with identifiers.**
   Hermes' Anchor Index and Cline's file lists are extracted by code. Cursor
   and Letta keep the raw history recoverable.
4. **Budgets run from about 4k to 10k tokens.** Only Cursor's *trained*
   summarizer is short (about 1k).

---

## 3. Evidence on quality

### 3.1 Compaction and context management for agents

| study | setup | result | label |
|---|---|---|---|
| **The Complexity Trap** (JetBrains, NeurIPS 2025 DL4Code) [14] | SWE-bench Verified, SWE-agent/OpenHands; Qwen3-Coder 480B, Gemini 2.5 Flash and others, 32B–480B, **thinking and non-thinking**; masking keeps 10 turns; summarization folds 21 turns and keeps 10 | observation masking **halved cost** and matched, sometimes exceeded, summarization's solve rate. Summarization made runs **13–15% longer** (Gemini 2.5 Flash: 52 turns on average), because summaries "smooth over" stop signals. Summary calls were more than 7% of cost. A hybrid saved 7% over masking and 11% over summarization | [paper] |
| **Factory: Evaluating context compression** (Dec 2025) [11] | 36,611 production messages; probes of four kinds: **recall, artifact, continuation, decision**; GPT-5.2 judge, 0–5 scale, 6 dimensions | overall Factory 3.70, Anthropic 3.44, OpenAI 3.35. **Artifact trail 2.19–2.45 for all three**, "an unsolved problem". Compression ratios 98.6–99.3%. Their method is **anchored iterative**: only the newly dropped span is summarized and merged into persistent sections | [vendor-eval] |
| **ACON** (Kang et al., 2025) [22] | AppWorld, OfficeBench, multi-objective QA | compression guidelines optimized **from failure analysis** (contrast a run that failed after compression with one that succeeded without it); peak tokens −26–54% with task success preserved or improved; distilled into small compressors; small LMs up to +46% | [paper] |
| **Lost in Compaction** (2026) [12] | COMPINT: session constraints injected in chat, agent and research settings | current compactors retain **17%** of injected constraints on average, and most do worse than no compaction. A plug-in constraint extractor reaches **over 90%** | [paper] |
| **The Compaction Cliff** (2026) [13] | safety rules under repeated compaction | Claude Code's `/compact` on Sonnet 4.6 keeps **53% of rules after one round, 10% after five**. Type-aware compaction keeps 2–4x more (96% recall over five rounds) | [paper] |
| **What does context compression cost an agent?** (2026) [23] | planning task and ALFWorld; 3 models | completion is unchanged, but agents **reacquire dropped state**. GPT-5.5 retrieval calls went from 21.0 to 63.9 (p = .002) while completion went from 80% to 85% (n.s.). Task-completion metrics hide this cost | [paper] |
| **Slipstream** (2026) [24] | SWE-bench Verified, BrowseComp | compaction validated **against the agent's own continued trajectory**: a judge checks that the summary supports what the agent did next. Up to +8.8 points of accuracy and −39.7% latency | [paper] |
| **CompactionRL** (2026) [25]; **Cursor self-summarization** [8]; **ReSum** [26] | compaction trained jointly with the task | trained compaction beats prompted: +5.5–7.0 points on SWE-bench Verified (GLM); Cursor 50% fewer compaction errors at a fifth of the tokens; ReSum +4.5% training-free, +8.2% with RL | [paper] / [vendor-eval] |
| **Context Rot** (Chroma, Jul 2025) [27] | 18 models including Qwen3 | performance falls as input grows, even on simple tasks. On LongMemEval, focused prompts beat full ones, and **thinking raised both but did not close the gap** | [vendor-eval] (a vendor's research report) |

**Longer vs shorter.**
- For *prompted* summarizers there is no controlled result on summary length
  and continuation. [opinion]
- Anthropic's own guidance is to tune the prompt for recall first, then
  precision [15]. [vendor-doc]
- The only length-vs-quality result is Cursor's: its trained 1k summaries
  beat its prompted 5k+ baseline [8]. That is a claim about training. It says
  nothing about truncating an untrained model's summary. [vendor-eval]
- Factory's winner was not the shortest: its ratio was 98.6%, against OpenAI's
  99.3% [11]. [vendor-eval]
- Here, truncation is not neutral. A cut summary ends in `finish_reason:
  length`, and Hermes discards it and retries (§1.4). [repo]

**What fails in practice**, in order of how well the evidence supports it:

1. Lost artifact trail: file paths, and what changed where [11].
2. Lost user constraints, which gets worse with each round of compaction
   [12][13].
3. Reacquisition: the agent re-reads what the summary dropped, which costs
   turns without changing the success rate [23].
4. Elongated runs and missed stop signals [14].
5. Drift across repeated compactions. This is why Factory, Hermes and
   OpenCode merge into a previous summary instead of regenerating it
   [6][10][11].
6. Hallucinated or mis-tensed progress. Hermes' temporal-anchoring rule
   exists to stop finished actions being phrased as open, and open ones as
   finished [10]. That is [vendor-doc] evidence that the failure was seen;
   Hermes publishes no rate.
7. Repetition. One of our 12 corpus summaries (35,425 characters) was
   repetitive. [repo, n = 1]

### 3.2 Thinking, specifically

| evidence | what it says | applies to us? | label |
|---|---|---|---|
| **OmniCSEval** [1]: 28 models, 1,800 conversations of 128–32k tokens; Qwen3-8B and Qwen3-14B **as the same hybrid checkpoint**, thinking budget 8,192 vs thinking disabled (their Table 8) | overall completeness / conciseness / faithfulness. 14B: **60.0 / 81.6 / 93.8** on, **55.2 / 77.6 / 94.2** off. 8B: **56.8 / 81.3 / 92.6** on, **50.9 / 76.6 / 92.9** off. For reasoning models in general: "no gain, or even regress, in faithfulness" | the closest analogue we have: a Qwen3 hybrid, thinking on vs off. But it is conversation summarization, not agent traces; there is no continuation measure; significance is not reported for these pairs; the budget is 8,192, not 1–2k | [paper] |
| **Understanding LLM Reasoning for Abstractive Summarization** [2]: 8 strategies, 3 reasoning models, 8 datasets | "increasing an LRM's internal reasoning budget does not reliably improve summarization and can even reduce factual consistency"; "more reasoning is not always better" | argues for a *bounded* budget, not for off | [paper] |
| **Qwen3 Technical Report** [3], Table 23 (RULER, thinking budget 8,192) | thinking mode "slightly degrades" retrieval: Qwen3-32B 93.7 → 91.0 on average; 235B 95.0 → 92.2. The authors' hypothesis is that thinking "may instead interfere with the retrieval process". Their thinking-budget scaling curves cover only maths, coding and STEM | compaction is partly retrieval (copying exact identifiers). A small penalty is plausible | [paper] |
| **The Coupling Tax** [28] (Qwen3) | when thinking and answer **share** one `max_tokens`, long traces crowd out the answer; non-thinking matched or beat thinking on GSM8K and MATH-500 at budgets up to 2,048. Split budgets fix it | our budgets are already split (`max_tokens = thinking + answer`, `reasoning_budget_tokens` separate), so the tax as defined does not apply. The forced close at 1,024 is a separate risk (§1.3) | [paper] |
| our own thinking-length data (`docs/CONSTRAINTS.md` §1b) | natural thinking 682–2,826 tokens on 7 finished runs; 2 of 9 ran away | a 1,024 cap usually cuts mid-thought | [repo, n = 1 per cell] |

**Plain statement:**
- **No published result measures thinking-on vs thinking-off for agent
  compaction by continuation.** The same holds for budget-limited thinking at
  1–4k tokens on a summarization task.
- The same-checkpoint Qwen3 evidence favours thinking on **completeness**,
  which is the dimension compaction loses, at a faithfulness cost under half a
  point.
- The retrieval evidence and the "more budget can hurt" evidence argue for
  capping the budget, not for switching thinking off.

That is weak, indirect evidence. It is exactly why the eval in §4 exists.

---

## 4. The eval (design only; not run)

### 4.1 Question and arms

**Primary question.** At a fixed conversation effort, does thinking during
compaction change whether the model can continue the task? Arms:

| arm | thinking | budget | role |
|---|---|---|---|
| **OFF** | off (instruct sampling, as served today at medium) | – | primary |
| **ON-2048** | on (thinking sampling) | 2,048 | primary (the recommended interim default) |
| ON-1024 | on | 1,024 | dose arm: today's xhigh/low behaviour |
| ON-4096 | on | 4,096 | dose arm |
| OFF-thinksampling | off, but thinking-mode sampling (no presence penalty) | – | separates "thinking" from "sampler" (§1.2) |

- **Effort:** `medium` for the primary comparison. It is the default, and the
  only effort where both arms are cache-free and so both are real options.
- **Second stratum:** `xhigh`, dose arms only, to check the budget there.
- **Secondary factor:** answer allowance 5,120 vs 12,288, on the winning
  thinking arm. This is the "longer vs shorter" question.

### 4.2 Prerequisites (code changes, not made here)

1. **A way to force the arm through `:1234`.** Add a new `X-Yamadori-Features`
   field read by `proxy._serve_compaction`, for example
   `{"compaction_thinking": "off" | <budget>, "compaction_sampling":
   "thinking" | "instruct", "compaction_answer": N}`. Without it, the only
   lever is effort, and effort confounds the comparison. The eval must not
   call `:11434` (AGENTS.md, "Never test around the stack").
2. **Full-text capture of compaction requests and of the conversation they
   compact.** The corpus truncates requests at 2,000 characters and stores no
   answer text. `compaction._store` holds the full conversation, but only in
   memory. A durable, account-scoped JSONL under `logs/` would make every
   Hermes dogfood compaction an eval item, captured from the real producer
   (PROTOCOL rule 7).

### 4.3 Items: where the data comes from

| source | what we have | usable for |
|---|---|---|
| `index/corpus.sqlite3` | 12 compaction turns, truncated, **no answer text**, one project | nothing but the length distribution |
| `bench/swebench/results/mini50-20260923/bonsai/*.traj.json` | **2** Bonsai trajectories through `:1234`, both resolved: django-11999 (80 steps, final prompt 27,774 tokens) and sphinx-8269 (66 steps, 39,336 tokens), with full messages and reasoning | **the pilot only**: harness check and a variance estimate |
| more `bench/swebench` runs (`--subset verified-mini --all`) | 50 instances; at the 6,000 s per instance observed with two runs sharing the card, about 2–3 days | primary items (real producer, our model) |
| Hermes dogfood runs (`docs/SELF-IMPROVEMENT-PLAN.md` Phase 1) | **planned, not run** | primary items in the flattened shape, once capture (§4.2.2) exists |
| `nebius/SWE-agent-trajectories` (80,036 trajectories, CC-BY-4.0) [29] | public; another producer's models | volume for the probe-only part, **labelled foreign-producer** and reported separately |

**A compaction point** is a pair (trajectory, cut). The cut falls at 50% and
at 75% of a trajectory's tokens, so each trajectory gives 2 items. The
compacted span is everything before a protected tail, and the tail follows the
harness being imitated. There are two shapes, balanced across items:

- **Hermes-flattened.** The prompt is built by the pinned Hermes version's own
  `_build_summary_prompt`, and the tail follows Hermes' lean rule.
- **Codex in-place.** The literal `prompt.md`, with the next context built as
  the summary plus up to 20k tokens of recent user messages.

Before compacting, each item is **primed** as a normal turn, so the proxy
holds the stored prompt and the spliced or rewritten path runs. That is the
production path. `x_yamadori.compaction.mode` must read `spliced` or
`rewritten`, never `as_sent`, or the item is void (PROTOCOL rule 2).

### 4.4 Probes (ground truth extracted mechanically from the dropped span)

After compaction, the model gets the post-compaction context the harness would
build (summary plus tail) and one batched probe turn, answered in JSON. The
probe turn runs at the **same fixed settings in every arm** (medium, thinking
on, budget 1,024), so only the summary differs.

| probe | ground truth | score |
|---|---|---|
| **artifact**: "list every file you modified; list every file you read" | write/edit tool calls and shell writes (`sed -i`, `cat >`, `apply_patch`) in the span, parsed. Paths come from call arguments, not regex over prose | set F1 on exact paths |
| **recall**: the last error or exception in the span; the last failing test; one command that was run | tool outputs and calls in the span | exact substring match |
| **constraint**: in half the items, one user constraint injected early in the span ("do not modify anything under `tests/`", "keep the public signature of `X`") | the injected text | retained in the summary (exact or near-exact), and **respected** by the next action (mechanical) |
| **continuation** | the recorded trajectory's next action after the cut | same tool family and same target path or command |
| **redo** | files already read, or commands already run, in the span | whether the next action re-reads or re-runs them (reacquisition [23]) |
| **false progress** | the span's actual modified files and final test state | the summary claims a file modified that was not, or tests passing where the last run failed |
| decision (secondary) | – | needs a judge. A stronger model offline, or the operator on a 20% sample. Never primary |

- **Primary endpoint:** the per-item probe score, the mean of artifact F1,
  recall, constraint retention and continuation. It is 0–1 and paired by item.
- **Secondary endpoints:** continuation match (binary); false-progress rate;
  `finish_reason: length` rate; spill-over rate (thinking hit its budget and
  the content does not start with the requested structure); summary tokens;
  wall time; `x_yamadori.cache` reused vs processed.

### 4.5 Statistics and power

- **Primary test:** a paired permutation test with sign flips **at the
  trajectory level**. Items from one trajectory are not independent. Report
  the mean difference with a cluster-bootstrap 95% CI.
- **Binary secondaries:** McNemar exact on discordant pairs (PROTOCOL rule
  6), with Holm correction across the dose arms.
- **Power (PROTOCOL rule 4), stated before running.** The SD of the paired
  difference is unknown. Assuming 0.20, which the pilot replaces, n = 100
  items gives a minimum detectable difference of about 2.8 × 0.20 / √100 ≈
  **0.056**, before the clustering penalty. With 2 items per trajectory and
  an ICC up to 0.5, the effective n is 67–100, so the MDE is about
  **0.06–0.07**. For continuation match (binary) with about 25% discordant
  pairs, n = 100 detects about a 12–15 point difference. **n = 50 (the dose
  arms) detects only about 0.08–0.10**, so the dose arms can rank budgets
  only if the differences are large. Say so in the report.
- **Repeat (PROTOCOL rule 10).** Sampling is stochastic (temperature 1.0 or
  0.7). Run the full primary pass twice with different seeds. A result counts
  only if both passes agree in sign and the pooled CI excludes 0.
- **Pre-registered decision rule:**
  - ON-2048 better, CI excluding 0 in both passes: thinking on at every
    effort.
  - OFF better: off at medium, and at xhigh/low weigh the measured
    re-prefill cost.
  - |difference| < 0.03 with the CI inside ±0.06: decide on latency. That
    means off at medium and on at xhigh/low, which is today's behaviour apart
    from the budget.
  - Budget: the smallest budget within 0.03 of the best.

### 4.6 Time cost at our speeds

Planning speeds:
- **Prefill:** about 490 tok/s (`docs/KNOWN-ISSUES.md`, short prompts; the
  operator quotes 490–580; long-prompt prefill is unmeasured).
- **Decode:** 25–45 tok/s. That covers 46.06 single-stream without MTP
  (`KNOWN-ISSUES.md`), about 54 with the MTP head (`docs/MTP-STAGING.md`), and
  13.7–43.8 on a shared card (`CONSTRAINTS.md` §1).

| step | tokens | time |
|---|---|---|
| prime the item (cold prefill of the span, once per item; arms reuse the slot) | 15–30k | 30–60 s |
| compaction summary | 3–6k answer | 70–240 s (midpoint about 130 s) |
| + thinking | 0 / 1,024 / 2,048 / 4,096 (at most) | 0 / 23–41 / 45–82 / 91–164 s |
| post-compaction prefill (summary + tail) | 5–15k | 10–30 s |
| batched probe answer (thinking ≤ 1,024 + about 300) | ≤ 1.3k | 30–50 s |
| next-action turn (shares the prefix) | ≤ 1.3k | 30–50 s |
| **per item per arm** | | **about 230 s (OFF), 270 s (ON-2048)** |

| run | n | arms | GPU time |
|---|---|---|---|
| pilot: 2 Bonsai SWE trajectories × 3 cuts | 6 | OFF, ON-2048 | about 1 h |
| primary pass | 100 | OFF, ON-2048 | about 14–15 h |
| repeat pass (required) | 100 | OFF, ON-2048 | about 14–15 h |
| dose and sampler arms | 50 | ON-1024, ON-4096, OFF-thinksampling | about 11 h |
| answer-length factor | 50 | 5,120 vs 12,288 | about 8 h |

- Everything goes through `bench/queue_runner.py`, one GPU consumer at a
  time. A 429 counts as "not run".
- The optional end-to-end confirmation is Hermes dogfood tasks with a low
  compaction threshold, paired by task, scored on the acceptance tests with
  McNemar. At n = 40 tasks it costs 40–120 GPU-hours and can detect only
  differences of about 20 points or more. Run it only if the probe eval finds
  a difference worth confirming (PROTOCOL rule 5: the cheapest layer that can
  answer the question).

---

## 5. Recommendation until the eval runs

**Thinking on, at the conversation's own effort, at every effort, medium
included, with a 2,048-token thinking budget. The answer cap should honour a
client's stated target.** Concretely, in `compaction.prefix_fields`:

1. **Keep thinking on at medium too, instead of off.** At medium this is
   cache-free (§1.1).
2. **Raise `MIN_THINKING` to 2,048 for compaction only.** Leave the global
   floor alone and add a compaction-specific constant. **Where the conversation
   itself thinks off** (tier `minimal`), stay off. That is what the cache
   wants, and it is what the client chose.
3. **Raise the answer cap** from 2 × 5,120 to at least Hermes' stated target
   (12,192 at the current pool). For example, `cap = max(2 × budget,
   stated_target)`, bounded by the window check that already exists.

Why, labelled:

- **[vendor-doc]** Every harness whose behaviour is documented compacts at the
  conversation's own thinking or effort setting, or leaves it to
  configuration: Anthropic, Codex, OpenCode, Roo, Cline, Claude Code's
  `<analysis>`, Cursor's scratch space. None switches thinking off for
  quality. Codex's stated reason is ours: keep the cache.
- **[paper]** The only same-checkpoint Qwen3 evidence gives thinking about +5
  points of completeness for under half a point of faithfulness [1].
  Completeness is the dimension compaction loses [11][12][13].
- **[repo]** "On" is cache-free at every effort; "off" costs a full re-prefill
  at xhigh and low (about 80 s for 40k tokens). At medium, "off" also means
  `presence_penalty 1.5` on a text that must repeat identifiers. That concern
  is **[opinion]** and unmeasured, and the OFF-thinksampling arm tests it.
- **[repo]** 1,024 lets 2 of the 7 finished natural-thinking runs end on
  their own, and 2,048 lets 5 of 7. Those were not summaries, and n = 1 per
  cell. A forced close has spilled deliberation into `content` before (1 of 2
  runs without the budget message). **[paper]** A bounded budget rather than
  the share-derived one, because more thinking budget can hurt factual
  consistency [2] and thinking slightly hurts retrieval [3].
- **[repo, derived from source]** A cap below the client's own target turns a
  long summary into a discarded summary and a second compaction.

**The cost of this default.** It adds at most 45–82 s of thinking per
compaction at 25–45 tok/s, on top of a summary that already takes about 2–4
minutes. **It is an interim choice, not a finding.** It rests on vendor
practice and indirect evidence, and §4 exists to overturn it.

**What would do more for quality than thinking.** These are **[opinion]**,
each drawn from the evidence above and each testable as an arm in §4:

1. **Append a mechanically extracted ledger to in-place compactions.** The
   proxy holds the whole stored conversation, so it can list files read and
   modified (from tool-call arguments), the last error, and the last test
   state, as Cline and Hermes do. This targets the artifact trail, the one
   dimension every compactor fails [11].
2. **Carry user constraints verbatim.** Extract imperative user instructions
   from the compacted span and restate them, as the constraint extractor does
   (17% → >90% [12]).
3. **Leave the harness's tail and iterative update alone.** Both are
   harness-side. The proxy already references Hermes' previous summary in
   place instead of re-prefilling it.
4. **Keep the dropped span recoverable.** `mcp/rings.py` is the stack's work
   log that survives compaction. Whether a pointer into it (Cursor's history
   file, Letta's recall storage) helps a client harness is untested.

---

## Sources

1. OmniCSEval: "A Large-Scale Multi-Dimensional Empirical Study of LLMs for Conversation Summarization", arXiv 2606.15974. https://arxiv.org/abs/2606.15974 (Qwen3 rows and Table 8 read from https://arxiv.org/html/2606.15974)
2. "Understanding LLM Reasoning for Abstractive Summarization", arXiv 2512.03503. https://arxiv.org/abs/2512.03503
3. Qwen Team, "Qwen3 Technical Report", arXiv 2505.09388, Appendix A.1.1, Table 23. https://arxiv.org/abs/2505.09388
4. Anthropic docs: Compaction overview, Compaction on demand, Compaction at a token threshold. https://platform.claude.com/docs/en/build-with-claude/compaction · https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand · https://platform.claude.com/docs/en/build-with-claude/compaction-threshold
5. OpenAI Codex source: `codex-rs/core/src/session/reasoning_effort.rs`, `codex-rs/core/src/compact.rs` (`COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000`), `codex-rs/prompts/templates/compact/prompt.md`, `summary_prefix.md`. https://github.com/openai/codex
6. OpenCode source: `packages/opencode/src/session/compaction.ts`, `packages/opencode/src/agent/prompt/compaction.txt`, `packages/core/src/session/compaction.ts` (branch `dev`). https://github.com/sst/opencode
7. Piebald-AI, claude-code-system-prompts (Claude Code v2.1.280): `agent-prompt-conversation-summarization-with-additional-instructions.md`, `agent-prompt-recent-message-summarization.md`. https://github.com/Piebald-AI/claude-code-system-prompts (reverse-engineered, not Anthropic-published)
8. Cursor, "Training Composer for longer horizons" (2026-03-17). https://cursor.com/blog/self-summarization
9. Cline: Auto Compact docs, https://docs.cline.bot/features/auto-compact ; SDK source `sdk/packages/core/src/extensions/context/agentic-compaction.ts`, `compaction-shared.ts`. https://github.com/cline/cline
10. Hermes Agent source: `agent/context_compressor.py`, `agent/auxiliary_client.py` (main branch, read 2026-09-24). https://github.com/NousResearch/hermes-agent
11. Factory, "Evaluating Context Compression for AI Agents" (2025-12-16). https://factory.ai/news/evaluating-compression (now at factory.com)
12. "Lost in Compaction: Evaluating Side-Constraint Loss under Context Compaction", arXiv 2608.11242. https://arxiv.org/abs/2608.11242
13. "The Compaction Cliff in Long-Running AI Agent Memory", arXiv 2608.22752. https://arxiv.org/abs/2608.22752
14. Lindenbauer et al., "The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management", arXiv 2508.21433 (NeurIPS 2025 DL4Code). https://arxiv.org/abs/2508.21433 ; JetBrains Research blog, https://blog.jetbrains.com/research/2025/12/efficient-context-management/
15. Anthropic Engineering, "Effective context engineering for AI agents" (2025-09-29). https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
16. Cursor, "Dynamic context discovery" (2026-01-06), https://cursor.com/blog/dynamic-context-discovery ; Cursor docs, Summarization, https://docs.cursor.com/en/agent/chat/summarization
17. OpenAI API docs, Compaction. https://developers.openai.com/api/docs/guides/compaction
18. Roo Code docs, Intelligent Context Condensing. https://roocodeinc.github.io/Roo-Code/features/intelligent-context-condensing
19. Aider source: `aider/prompts.py` (`summarize`), `aider/history.py` (`ChatSummary`). https://github.com/Aider-AI/aider ; options, https://aider.chat/docs/config/options.html
20. Packer et al., "MemGPT: Towards LLMs as Operating Systems", arXiv 2310.08560. https://arxiv.org/abs/2310.08560 ; Letta docs, https://docs.letta.com/guides/legacy/memgpt_agents_legacy
21. Manus, "Context Engineering for AI Agents: Lessons from Building Manus" (2025-07-18). https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
22. Kang et al., "ACON: Optimizing Context Compression for Long-horizon LLM Agents", arXiv 2510.00615. https://arxiv.org/abs/2510.00615
23. "What Does Context Compression Cost an Agent? Interaction Costs Unrevealed by Task-Completion Metrics", arXiv 2608.16370. https://arxiv.org/abs/2608.16370
24. "Slipstream: Trajectory-Grounded Compaction Validation for Long-Horizon Agents", arXiv 2605.08580. https://arxiv.org/abs/2605.08580
25. Li et al., "CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents", arXiv 2607.05378. https://arxiv.org/abs/2607.05378
26. "ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization", arXiv 2509.13313. https://arxiv.org/abs/2509.13313
27. Chroma, "Context Rot: How Increasing Input Tokens Impacts LLM Performance" (Jul 2025). https://www.trychroma.com/research/context-rot
28. "The Coupling Tax: How Shared Token Budgets Undermine Visible Chain-of-Thought Under Fixed Output Limits", arXiv 2605.07686. https://arxiv.org/abs/2605.07686
29. nebius/SWE-agent-trajectories (Hugging Face, CC-BY-4.0). https://huggingface.co/datasets/nebius/SWE-agent-trajectories
