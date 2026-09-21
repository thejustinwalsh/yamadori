# Known issues

## RESOLVED — degenerate output and broken tool calls

**Cause: the wrong llama.cpp build. Fixed by using the official PrismML fork.**

For several hours this stack produced garbage: generation collapsed into runs
of `/` mid-sentence, and tool calls never completed. It was blamed in turn on
the ternary quantization, llama.cpp's tool grammar, sampler settings and the
chat template. All of those were wrong.

The build was `sudoingX/llama.cpp` branch `pr-ptq1-mmv` — a contributor's
experimental PR branch carrying an optimised PTQ1_0 decode kernel. Its tags
only reach `prism-b9601`; the model requires `prism-b10685` or newer. The
kernel is genuinely faster and numerically broken.

| | `pr-ptq1-mmv` | **official `prism`** |
|---|---|---|
| decode (5060 Ti) | **55.33 tok/s** | 46.06 tok/s |
| prefill | 490.8 tok/s | 490.5 tok/s |
| plain generation | 3/5 degenerate | **5/5 clean** |
| tool calls (native param) | 0/10 | **9/9** |
| tool calls (through the stack) | 0/10 | **9/9** |

The speed was real. So was the corruption. **Do not switch back.**

Correct source, from [PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo):

```
repo   https://github.com/PrismML-Eng/llama.cpp
branch prism
pinned prism-b10709-9a9394a
```

### Why it took so long to find

The failure looked like a tool-calling problem because tool calls are long and
structured, so they hit the degeneration almost every time, while short replies
often finished before it appeared. Early successful tests were short.

The decisive test was trivial and should have come first: **ask for plain prose
with no tools.** It degenerated immediately, which ruled out everything
tool-related in one step. Several hours of grammar, sampler and template
debugging came before that.

Lesson for the next odd failure here: check whether the *simplest* path is
also broken before investigating the complicated one.

### Consequences

- `mcp/tool_shim.py` is **no longer needed**. It was built to work around
  llama.cpp's tool grammar, which was never the problem. It is kept in the
  repo, unused, purely as a record; nothing starts it.
- Sampler settings were never the issue. The config uses the model card's
  thinking-mode values.

### Still true: thinking must be off for tool turns

Independent of the build, with thinking enabled the model spends its whole
budget reasoning and never emits a call:

| config | result |
|---|---|
| thinking on, `max_tokens` 2000 | 2000 tokens of `<think>`, no call |
| `reasoning_effort: low` | 2000 tokens of `<think>`, no call |
| `enable_thinking: false` | call emitted in 29 tokens |

This is why the `bonsai-agent` alias forces
`chat_template_kwargs.enable_thinking = false` server-side. Point agents at
`bonsai-agent`, not `bonsai`.

---

## Not fitted: adversarial critic

Qwen3.8-27B Q4_K_M as a second-opinion reviewer needs ~19.7 GiB on one card
(15.41 weights + 1.06 q8 KV @32k + ~2.0 compute + ~1.2 MTP context); the A4000
has 16. It only ran earlier by splitting across both GPUs, which is no longer
possible now the primary owns the 5060 Ti. A ~12 GiB quant (Q3_K_XL / IQ4_XS)
would fit. The entry is present but disabled in `config.yaml`.

## VRAM headroom

240k context loads on the 5060 Ti but leaves ~2% free and then dies when a
long prefill allocates its compute buffer. 208k is the largest size measured
stable. Three models on the A4000 pinned it at 98%, so the vision variant was
moved to on-demand.
