# Sources and Repository Assessment

Companion to `design_style.jsonl`. Compiled 2026-09-21.

All star counts below were read from the GitHub REST API (`api.github.com/repos/{owner}/{repo}`)
on **2026-09-21**. They are a snapshot, not a stable figure. License is the SPDX id GitHub
reports; `NONE` means no license file, `NOASSERTION` means GitHub could not classify it.

---

## Part 1 — Primary sources actually fetched for the recipes

These were read directly (full page or raw file), not inferred from search snippets.

| Source | URL | What it gave |
|---|---|---|
| Richard Fabian, *Data-Oriented Design* (dodbook) | https://www.dataorienteddesign.com/dodmain/ and `/dodbook/node3.html` | Chapter list, existence-based processing, condition tables, relational/normalization thinking, the structure trade-off statement |
| Data-oriented design (Wikipedia) | https://en.wikipedia.org/wiki/Data-oriented_design | AoS vs SoA framing, cache/locality rationale, history |
| Steven Gong's notes on Acton's talk | https://stevengong.co/notes/Data-Oriented-Design | The Three Big Lies, "purpose of all programs is to transform data", "where there is one there are many", compiler-solves-10% |
| Sander Mertens, ECS FAQ | https://github.com/SanderMertens/ecs-faq | ECS definition, archetype vs sparse-set vs bitset vs reactive storage trade-offs |
| bevy_ecs docs | https://docs.rs/bevy_ecs/latest/bevy_ecs/ | Table vs SparseSet component storage, change detection, archetypes |
| EnTT entity docs | https://github.com/skypjack/entt/blob/master/docs/md/entity.md | Sparse sets, paged storage, views vs owning groups, pointer stability, entity versions |
| Casey Muratori, "Semantic Compression" | https://caseymuratori.com/blog_0015 | Write usage code first; compress after two instances; usable before reusable |
| Jane Street, "Core Principles: uniformity of interface" | https://blog.janestreet.com/core-principles-uniformity-of-interface/ | module-per-type with `t`, `t`-first argument order, `of_`/`to_`, `_exn`, option-by-default |
| Jane Street, "Effective ML Revisited" | https://blog.janestreet.com/effective-ml-revisited/ | Make illegal states unrepresentable, exhaustive matching, uniform interfaces, open few modules |
| Jane Street, "How to fail — introducing Or_error.t" | https://blog.janestreet.com/how-to-fail-introducing-or-error-dot-t/ | option vs result vs `Or_error.t`, when each applies, consistency argument |
| janestreet/base source | `src/list.mli`, `src/int_intf.ml` (raw) | Verified in code: `*_exn` family, `of_string_opt`, `of_`/`to_` pairs, `~f`/`~init` labelled args |

### Sources I could NOT fully verify (recipes marked `medium`)

- **Mike Acton, CppCon 2014.** The canonical artifact is a `.pptx` in
  https://github.com/CppCon/CppCon2014 and the video at
  https://www.youtube.com/watch?v=rX0ItVEVjHc. I did not parse the slides or the video;
  the Acton recipes come from a secondary notes page plus Wikipedia corroboration. The
  specific claims used (Three Big Lies, purpose-of-programs, compiler ~10%) are widely
  and consistently reported, so they are marked `high`; nothing narrower was taken.
- **Andrew Kelley, "Practical Data Oriented Design"** (Handmade Seattle 2021,
  https://vimeo.com/649009599). No transcript or detailed writeup was reachable — the one
  writeup found (dcreager.net) is a two-sentence stub. The three Kelley recipes cover only
  what multiple secondary sources agree the talk covers (struct layout/padding, shrinking
  integer types, MultiArrayList struct-of-arrays) and are marked `medium`.
  **I deliberately omitted recipes about his "encodings" / variable-size-node technique
  because I could not verify the details.**
- **`opensource.janestreet.com/standards/`** now 301-redirects to `github.com/janestreet`.
  There is no public, complete Jane Street style guide document. Everything in area 2 is
  drawn from their blog plus the actual `Base` source. The "no abbreviations" and
  "verb-first" recipes are observed practice in the source, not quoted policy — marked
  `medium`. `web.archive.org` is blocked in this environment, so the old page could not be
  retrieved.

---

## Part 2 — Repository assessment: coding rules for humans and for AI assistants

Content was inspected directly (READMEs plus sample rule/skill files via raw content and
the GitHub trees API), not judged from descriptions.

### Summary table

| Repo | Stars | License | Maintained | Honest quality judgement |
|---|---:|---|---|---|
| [airbnb/javascript](https://github.com/airbnb/javascript) | 148,256 | MIT | Yes (2026-04) | *Metadata only — content not inspected in this pass.* The most-starred style guide on GitHub. Historically influential and linter-backed; I did not sample its rules here, so treat the quality claim as unverified. |
| [google/styleguide](https://github.com/google/styleguide) | 39,623 | NOASSERTION | Yes (2026-09) | **Genuinely excellent.** 14 HTML + 18 Markdown guides; `cppguide.html` 242 KB, `pyguide.md` 118 KB, Go `best-practices.md` 139 KB. Numbered, addressable rules with rationale and explicit exceptions. Ships IDE formatter configs. Caveat: HTML guides are awkward to feed an agent; the Markdown subset (Python, Go, Shell, Obj-C) is the usable part. |
| [google/eng-practices](https://github.com/google/eng-practices) | 23,299 | NOASSERTION | **Archived** (2024-09) | **High signal, small, frozen.** 14 files / ~78 KB of code-review doctrine. Argues *why*, not just what — e.g. reasons from concurrency risk to a reviewability conclusion. `small-cls.md` and `standard.md` are the most-cited code review texts in the industry. Read as a finished document. |
| [uber-go/guide](https://github.com/uber-go/guide) | 17,716 | Apache-2.0 | Yes (2026-04) | **Best signal-per-byte on this list.** `style.md` is 87 KB / 4,107 lines, ~40 rules in Bad/Good tables. Every rule is falsifiable and Go-specific ("Zero-value Mutexes are Valid", "Start Enums at One", "Copy Slices and Maps at Boundaries"). Zero "write clean code" filler. |
| [rust-lang/rust-clippy](https://github.com/rust-lang/rust-clippy) | 13,520 | Apache-2.0 | Yes (2026-09) | **Substantive, executable.** Not prose: hundreds of lints (360 source modules under `clippy_lints/src`), each documented with a rationale and known-problems section. The rationale text is the useful artifact for prompting; the lint is the ground truth. *Exact lint count not verified — the published `lints.json` endpoint 404'd.* |
| [rust-lang/api-guidelines](https://github.com/rust-lang/api-guidelines) | 1,350 | Apache-2.0 | **Stale** (2025-07) | **Most underrated here, and the best-shaped for agent use.** 15 chapters plus `checklist.md`: an auditable checkbox list with stable ids (`[C-CASE]`, `[C-CONV]`, `[C-ITER-TY]`, `[C-SEND-SYNC]`) each linking to its rationale. Machine-checkable, no padding. Only concern is ~14 months untouched. |
| [rubocop/ruby-style-guide](https://github.com/rubocop/ruby-style-guide) | 16,549 | NONE | Yes (2026-07) | *Metadata only — content not inspected in this pass.* Note the redirect: `bbatsov/ruby-style-guide` now resolves here. Paired to an enforcing linter, which structurally keeps rules honest, but I did not sample it. |
| [obra/superpowers](https://github.com/obra/superpowers) | 289,699 | MIT | Yes (2026-09, v6.4.1) | **Real, and the most interesting AI-targeted repo found.** 15 `SKILL.md` files (3–32 KB) plus 66 actual tests. It encodes *process discipline* rather than tech-stack trivia: `verification-before-completion` maps each completion claim to the evidence that proves it and the evidence that does not; includes a rationalization-prevention table anticipating the excuses a model generates to skip the rule. Caveats: sales-inflected README, a commercial-services section, a 17-platform install matrix that is partly distribution strategy, and unverified productivity claims. The skill content itself holds up. |
| [anthropics/skills](https://github.com/anthropics/skills) | 177,467 | NONE (per-skill terms) | Yes (2026-09) | **Substantive but small, and not about coding style.** Only ~20 skills, but each is a working package (the `pdf` skill ships 8 executable Python scripts; `mcp-builder` ships four 25–28 KB reference docs and an eval harness). Uses progressive disclosure properly: `SKILL.md` routes, detail loads on demand. Stars measure vendor gravity, not corpus size. 1,264 open issues. |
| [github/awesome-copilot](https://github.com/github/awesome-copilot) | 39,244 | MIT | Yes (daily) | **Best large collection, by a wide margin.** 194 `*.instructions.md` (median ~7.5 KB, only 15 under 1.5 KB), plus 1,232 skills / 222 agents. Content is technically real — the Go MCP instructions contain correct, compiling API usage against the actual SDK. `applyTo` globs are genuinely per-file. 437 contributors and ~50 CI workflows including duplicate detection and quality gates. Weakness: the 54–64 KB files drift into LLM padding ("## Your Mission / As GitHub Copilot, you are an expert in…"), and AI-generated submissions are getting merged — just with gates around them. |
| [ciembor/agent-rules-books](https://github.com/ciembor/agent-rules-books) | 2,838 | MIT | Yes (2026-09) | **Impressive process, questionable foundation.** 14 classic books distilled at three compression levels (full ~14 KB / mini ~4 KB / nano ~1.2 KB). `PROCESS.md` sets a sharp bar — compressed rules should be "decision-equivalent to the full source" — and `CRITICISM.md` openly concedes the biggest objection is unsolved (no benchmarks, self-scored ~2/10). But the traceability files trace back to *this repo's own* generated `full.md`, never to book pages; and the book-compatibility matrix emits precise-looking percentages with no stated methodology. Content is competent LLM paraphrase of books frontier models already know well, so marginal value is unproven. Also: systematic derivative distillation of 14 in-copyright books, MIT-licensed, with legal review listed as unresolved. |
| [steipete/agent-rules](https://github.com/steipete/agent-rules) | 5,684 | MIT | **Archived** (2026-05) | **Two-tier, honest, small.** 22 `project-rules/*.mdc`, median ~1.4 KB. The short ones are near-filler (`clean.mdc` is `black .` / `isort .` / `flake8`). The long ones carry real expertise in the author's actual specialty — `screenshot-automation` (9 KB), `safari-automation`, `mcp-inspector-debugging`, `modern-swift` (takes real positions on `@Observable` vs `@ObservableObject`). **Technical defect: the sampled `.mdc` files have no YAML frontmatter, so Cursor auto-attach cannot work on them.** |
| [sanjeed5/awesome-cursor-rules-mdc](https://github.com/sanjeed5/awesome-cursor-rules-mdc) | 3,571 | CC0-1.0 | **No** (2025-12) | **Openly, entirely LLM-generated bulk.** 1,121 `.mdc` files — but 882 sit in `rules-v0-deprecated/`; only 243 are live. The README states it generates content with Exa search + Gemini; a recent commit is literally "Regenerate all MDC rules with fresh Exa data". To its credit the output is better than awesome-cursorrules (real Bad/Good pairs, proper scoped globs). But nobody reviewed 243 files across 243 libraries; it is unverifiable synthesis, and the LLM register is everywhere ("the definitive framework for…"). Star-history chart embedded in the README. Abandoned 9 months. |
| [PatrickJS/awesome-cursorrules](https://github.com/PatrickJS/awesome-cursorrules) | 40,811 | CC0-1.0 | Repo yes, content no | **Star-farmed. The worst offender relative to popularity.** 257 rule files, median 2,775 bytes; **67 (26%) under 1.5 KB, 7 under 400 bytes, and at least one with no body at all** (`go-temporal-dsl-prompt-file.mdc`, 118 bytes = frontmatter only). Every file carries `globs: **/*` and `alwaysApply: false` — no scoping was ever considered. Verbatim duplication: the same Next.js persona prompt appears across many files. Truncated/corrupt imports (filenames cut at 42 chars; one file ends mid-sentence). Typical content is a restatement of framework docs ("Use hx-get for GET requests") or pure persona flattery with zero domain content. Format chaos — JS comments, Markdown, pseudo-arrays, prose, all in one corpus. Three paid sponsor lockups with UTM tracking above the fold. Recent commits are CI polish on top of a low-quality corpus. |
| [agentsmd/agents.md](https://github.com/agentsmd/agents.md) | 24,534 | MIT | Yes | **Not a rules repository at all.** The tree is a Next.js marketing site: 19 `.tsx` components, 27 SVGs, a lockfile, a governance PDF. **There is no spec file anywhere.** The entire "standard" is one fenced code block in a 2 KB README: name a Markdown file `AGENTS.md`, put it at the repo root. No schema, no required sections, no precedence rules, no conformance language. The stars measure genuine ecosystem coordination value (a Schelling point for a filename), not content. As a source of coding rules there is nothing here. |
| [airbnb/ruby](https://github.com/airbnb/ruby) | 3,892 | MIT | Stale (2025-12) | *Metadata only — content not inspected.* Listed for completeness; largely superseded in practice by the rubocop guide. |
| [dbartolini/data-oriented-design](https://github.com/dbartolini/data-oriented-design) | 4,479 | NONE | Yes (2026-09) | **Useful as an index, not as content.** A curated link list of DOD talks, papers and articles — no original material. Listed because it is the fastest route to the primary DOD sources; link quality not exhaustively checked. |
| [SanderMertens/ecs-faq](https://github.com/SanderMertens/ecs-faq) | 2,759 | NONE | Yes (2026-07) | **Genuinely authoritative for its size.** Written by the author of flecs. Concrete, comparative, willing to state trade-offs rather than hedge. Used as a primary source above. |

### Patterns worth remembering

1. **Star count is nearly uncorrelated with content quality in this space.**
   `rust-lang/api-guidelines` (1,350 stars) beats `PatrickJS/awesome-cursorrules`
   (40,811 stars) on density, specificity, internal consistency and absence of duplication.
2. **Uniform frontmatter is the reliable low-substance tell.** Every file carrying
   `globs: **/*` means nobody thought about scoping even once. Compare awesome-copilot's
   genuine per-file `applyTo` globs.
3. **The size distribution exposes bulk.** A corpus where a quarter of files are under
   1.5 KB and one is empty was assembled by a script, not curated by a person.
4. **Persona prompts are the filler signature.** A file that opens "You are an expert
   in…" and never names a real API is prompt-theater. A file that opens with a function
   signature or a Bad/Good pair is engineering.
5. **The best AI-targeted content is process discipline, not tech-stack lists.**
   Framework trivia is exactly what models already know and what rots fastest;
   verification gates and debugging protocol are what they actually fail at.
6. **Several of the best sources are frozen.** `google/eng-practices` and
   `steipete/agent-rules` are archived; `rust-lang/api-guidelines` is 14 months stale.
