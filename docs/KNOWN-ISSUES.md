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

### NOT still true: "thinking must be off for tool turns"

**REVERSED. Left in place because the reversal is the useful part.** This
section used to say that with thinking enabled the model spends its whole
budget reasoning and never emits a call (2000 tokens of `<think>`, zero calls;
a call in 29 tokens with `enable_thinking: false`), and that the
`bonsai-agent` alias existed to force it off server-side.

Every one of those observations was made on the **corrupt `pr-ptq1-mmv`
build documented above**. They do not survive the fork fix. Re-measured on the
official build at temp 0.3, 5 tasks x 3 reps, first call:

| config | result |
|---|---|
| thinking OFF | 13/15 correct, **0/15 failures to call** |
| thinking ON | 12/15 correct, **0/15 failures to call** |

Equivalent within noise, zero failures to call either way. `config.yaml` now
sets `chat_template_kwargs.enable_thinking = true` on **both** ids, so
`bonsai-agent` is an alias with identical settings and exists only so older
clients keep working. Either id is fine.

Thinking is ON for a different reason — prompt injection, where thinking OFF
leaked 5/5 against one naive payload and thinking ON at the shipped temp 0.3
leaked 1/5. **1/5 is a reduction, not a defence**, on an abliterated model; the
real controls are the architectural ones listed in `config.yaml`.

This is the same class of error as the fork bug itself: a downstream symptom
measured while an upstream component was broken, then written down as a
property of the model. See `docs/PROTOCOL.md` rules 1 and 13.

---

## Not fitted: adversarial critic

Qwen3.8-27B Q4_K_M as a second-opinion reviewer needs ~19.7 GiB on one card
(15.41 weights + 1.06 q8 KV @32k + ~2.0 compute + ~1.2 MTP context); the A4000
has 16. It only ran earlier by splitting across both GPUs, which is no longer
possible now the primary owns the 5060 Ti. A ~12 GiB quant (Q3_K_XL / IQ4_XS)
would fit. The entry is present but disabled in `config.yaml`.

## VRAM headroom

240k context loads on the 5060 Ti but leaves ~2% free and then dies when a
long prefill allocates its compute buffer. 208k was recorded here as "the
largest size measured stable"; **that is no longer what ships and no longer
the largest known-good figure.** 196,608 was tried after it and backed out —
it fits on paper (~8.1 GB of KV plus 5.95 GB of weights inside 16.3 GB) but
`arc193_a` then failed on all three benchmark arms while 193-second
generations succeeded elsewhere, which is a compute-buffer spike, not a
timeout. **`config.yaml` launches at `-c 147456`**, warm-measured at 12,122 MiB
used and 3,931 free, i.e. the ~3 GB of headroom that absorbs those spikes.
`mcp/budget.py` divides whatever the server reports (5/8 main, 3/8 for one
helper, reserve 0 -- at 147,456 that is 92,160 / 55,296), so `-c` is the only
place the number lives.

Three models on the A4000 pinned it at 98%, so the vision variant was moved to
on-demand.
