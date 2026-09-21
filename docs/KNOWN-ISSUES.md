# Known issues

## OPEN — tool calling is unreliable on the ternary model

**Status: unresolved. Do not rely on this stack for agentic tool use yet.**

The `bonsai` ternary model degenerates into runs of `/` when asked to emit a
tool call. Sometimes it produces garbage arguments, sometimes it crashes the
sampler outright:

```
"symbol":"parse_tool_call/////////////////////////////
got exception: Unexpected empty grammar stack after accepting piece: / (14)
```

### What was measured

Same prompt (`"Where is parse_tool_call defined?"`), same single tool
definition, counting only responses with a correctly-named call and correct
arguments:

| path | clean |
|---|---|
| ternary PTQ1_0 + llama.cpp `tools` parameter | 0/10 |
| Qwen3.8-27B Q4_K_M + llama.cpp `tools` parameter | 0/10 |
| Qwen3.8-27B Q4_K_M + **stock** llama.cpp build | 0/10 |
| shim (tools in prompt), hand-written preamble, isolated | 5/5 |
| shim, same preamble, 10-run | 4/10 |
| shim, preamble copied verbatim from the chat template | 0/12 |

### What was ruled out

- **Not the ternary quantization.** Q4_K_M fails identically.
- **Not the custom fork.** Stock llama.cpp fails identically with the same
  model, so it is not a `pr-ptq1-mmv` regression.
- **Not llama-swap.** Failures reproduce hitting `llama-server` directly.
- **Not sampling.** temp 0.2–1.0, `repeat_penalty` 1.0–1.1, `presence_penalty`
  0 and 1.5, `top_p` 0.8–0.95 were all tried. The model card's instruct-mode
  profile (temp 0.7 / top_p 0.80 / presence 1.5) scored *worse* (1/9) than the
  thinking-mode profile (3/9). Nothing moved it into a usable range.
- **Not thinking mode.** `enable_thinking: false` is forced and confirmed; the
  failures are `finish_reason: length` with `/` runs, not `<think>` overrun.
  (Thinking on is a *separate*, real problem — see below.)

### What is still suspected

llama.cpp reports `Chat format: peg-native` and applies a PEG/grammar to
constrain tool output. That grammar appears to fight this model family's XML
tool syntax. The `mcp/tool_shim.py` approach — strip `tools`, render them into
the prompt, parse the XML back — removes the grammar and *did* work in one
isolated 5/5 run, but has not been reproduced. The difference between that run
and later 0/10 runs has not been identified. Treat the 5/5 as unexplained,
not as a result.

### Confirmed and fixed: thinking must be off for tool turns

Independent of the above, with thinking enabled the model spends its entire
budget reasoning and never emits a call:

| config | result |
|---|---|
| thinking on, `max_tokens` 2000 | 2000 tokens of `<think>`, no call |
| `reasoning_effort: low` | 2000 tokens of `<think>`, no call |
| `enable_thinking: false` | call emitted in 29 tokens |

This is why the `bonsai-agent` alias exists and forces
`chat_template_kwargs.enable_thinking = false` server-side.

### Recommended path

Until this is solved, **the ternary stack is for generation, not agentic tool
loops.** For tool-driven work, the previously verified configuration was
Qwen3.8-27B Q4_K_M under **textgen**, which renders tools into the prompt and
parses them with its own `_parse_xml_param_tool_calls` — that combination was
observed returning `finish_reason: tool_calls` with correct arguments
repeatedly. The textgen autostart task still exists (disabled):

```powershell
Enable-ScheduledTask -TaskName text-generation-webui
```

### Next things to try

1. Diff textgen's exact rendered prompt against the shim's, byte for byte.
   textgen demonstrably worked; the shim is an approximation of it.
2. Check whether `logit_bias` on the `/` token stops the degeneration.
3. Test a non-hybrid model on the same stack to confirm the failure is
   specific to the `qwen3_5` XML tool format rather than the stack.

---

## Not fitted: adversarial critic

Qwen3.8-27B Q4_K_M as a second-opinion reviewer needs ~19.7 GiB on one card
(15.41 weights + 1.06 q8 KV @32k + ~2.0 compute + ~1.2 MTP context) and the
A4000 has 16. It only ran earlier by splitting across both GPUs, which is no
longer possible now the primary owns the 5060 Ti. A ~12 GiB quant
(Q3_K_XL / IQ4_XS) would fit. The entry is present but disabled in
`config.yaml`.

## VRAM headroom

240k context loads on the 5060 Ti but leaves ~2% free and then dies when a
long prefill allocates its compute buffer. 208k is the largest size measured
stable. Similarly, three models on the A4000 pinned it at 98%, so the vision
variant was moved to on-demand.
