# PartiPrompts sample, 2026-09-23

One prompt from each of the 8 largest PartiPrompts categories (Google's Parti
evaluation set, https://github.com/google-research/parti), drawn with
`random.Random(20260923)`. Nobody picked the prompts, and every result is
kept.

- Model: Qwen-Image-2.1 Q5_K_M + Heretic Qwen3-VL-8B encoder + bf16 VAE
  (docs/IMAGEGEN.md), via the live proxy's POST /v1/images/generations.
- 1024x1024, 20 steps, cfg 6.0. 105-111 s per image on the A4000.
- `manifest.json`: prompt, category, challenge, seconds, file.
- Verdicts are by eye (the coordinator's judgment, not a benchmark score):
  5 clean, 2 partial (the milk fountain reads as water; the Spam label
  garbled), 1 miss (the shoe rack "without any shoes" is full of shoes, a
  negation failure).
- `run.py` reproduces the draw. The dogfood key path is a local file and is
  never stored here.
