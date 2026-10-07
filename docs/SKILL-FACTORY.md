# The skill factory (NAEDOKO)

Skills are the one knowledge system of the stack (operator, 2026-09-25/26:
"It's time to retire hints and build the skills system ... because skills
are proven in the greater ecosystem." "We don't have rules to fix bugs, we
have skills."). This document is the CONTRACT between the backend
(`mcp/skills.py`, `mcp/skill_pipeline.py`, `mcp/dash_skills.py`) and the
dashboard surface that drives it. `mcp/test_skill_factory.py` holds the
backend to it. The React surface is a follow-up task built against this
document; nothing here depends on it.

## What a skill is

**The definition (operator, 2026-09-28, verbatim):** "All of our skills
increase confidence and improve correctness; if it can't, then the line
doesn't need to exist. That is what a skill is about: facts, proven
patterns, helpful knowledge you can act on immediately."

And on negatives: "Do not cast with `as any`" is actionable -- a concrete
pitfall that names the wrong move and the right one is GOOD content. What is
banned is self-verification ("verify the thing I just looked at", "check
before relying on"), doubt, uncertainty and version history. Written after
pagoda-h6, where the migrated r3f skills (compiled from the r3f v10
migration guide and alpha changelog: recipe rows whose schema carried
`replaces`, the old code, and `applies_to_version`) read "v10 alpha.3
deletes Canvas props ...", "state.clock is gone in v10", "Verify each
WebGPU component before relying on it", the kickoff plan's RISKS said
"Check @react-three/fiber README for the exact prop API before writing
main.tsx", and main probed node_modules for hours ("This model overthinks,
don't give it reason to!").

THE ASSURED VOICE (`skill_limits.doubt`, one rule, used everywhere a line
is written for the model): a line is REJECTED, with its reason recorded,
when it is

| reason | the shape |
|---|---|
| `verify_first` | an imperative to verify / check / confirm / look up an API, a prop, a README, a version or node_modules, or anything "before writing / using / relying on" it |
| `instability` | an API, a version or an alpha that may change, differ or break; "is unstable / incomplete"; "subject to change"; "provisional" |
| `version_history` | deprecated, "is gone", removed / renamed in, migrate from / to, upgrading to a version, a release that does something to the API ("v10 replaces ..."), or two releases compared in one line |
| `caveat` | be careful / mindful / wary, beware, watch out, keep in mind, contested |

Running the result (the build, the tests, the page) is an action and
passes; a version named as the version a line is FOR passes. Where it
applies: `skill_builder.validate` drops the item (every origin: distilled,
decomposed, compiled, authored, an edit); the review stage's rewrite must
pass it; `skills._row_of` leaves such a line out of what an ARMED skill
serves (the stored version untouched, `doubt_dropped` on the row;
`YAMADORI_SKILL_DOUBT_FILTER=0` serves the stored lines) and a skill whose
every line is doubt is not served; the planner's and deep thinking's
hand-offs drop such a line (`shomen.plan_handoff`, `shomen.handoff`,
`doubt_dropped` in their stats). Tests: `mcp/test_skill_factory.py`
[assured], [review], [serve]; `mcp/test_deep.py`
`test_the_plan_states_decisions_never_doubt`.

An Agent Skills folder, portable to a harness's skills directory (Hermes'
`~/.hermes/skills/<category>/`):

```
index/skills/library/<name>/
  SKILL.md      YAML frontmatter + a short markdown body
  tests.json    activation tests and behaviour checks (a harness ignores it)
```

```markdown
---
name: browser-app-entry-point          # [a-z0-9-], <= 64, = the folder name
description: Use when building, debugging or verifying a browser app ...
version: 1.0.0                         # semver (Hermes' linter expects it)
author: Yamadori
license: MIT                           # SPDX, from a verbatim quote
metadata:
  hermes: {tags: [JavaScript, code, implement, ...], related_skills: []}
  yamadori:
    id: 508dc52b9091                   # the store's id
    revision: 1                        # the store's version number
    state: armed                       # armed | quarantined | draft | disabled | archived
    category: {artifact: [code], language: [javascript], framework: [],
               phase: [implement, debug, verify], domain: [web-frontend]}
    applies_when: {artifacts, languages, frameworks, phases, situations,
                   all_of, topics, triggers, text}
    escalate: false
    provenance: {kind, source, url, parent, section, rows, files,
                 licence: {spdx, quote, where}, fetched_at, sha256}
    tests: {activation: {passed, score, n}, behaviour: 1}
    items: [{line, quote, ref}]        # each item's verbatim source quote / row ref
---
# Browser App Entry Point

## When to use

<the description again, for a harness that shows only the body>

## Guidance

- WHEN <situation>: <what to do>
- DO: <practice>
- DO NOT: <one observed failure>       # at most 2 prohibitions per skill
```

What reaches the model at request time is the title line and the items --
never the frontmatter, never "When to use" (`skill_md.injection`).
Hermes' own linter (`tools/skill_linter.py`) finds no error in this form
(checked in `mcp/test_skill_factory.py` where Hermes is installed).

## The taxonomy (`skill_classify.taxonomy()`; grows with the package registry)

Fixed by hand until 2026-09-27; since then a PACKAGE ONBOARDING whose
vocabulary is promoted adds ONE framework term for its package (operator
decision 7: "The taxonomy grows automatically from package names plus the
aliases the operator types"; `mcp/package_registry.py`,
docs/PACKAGE-ONBOARDING.md 5). The term's words are the exact npm name and
the aliases the operator typed -- never a model's proposal; a one-word npm
name that is an English word matches only forms that cannot mean anything
else (`name@<digit>`, a subpath, `npm install name`), as `pmndrs_math` does.
The hand-written terms below are `skill_classify.HAND_VOCAB`; a promoted
term is picked up on the next `skill_classify.refresh()` (one stat of
`packages.json`), with no restart. The tag stage records the catalogue it
rendered: `classify._tag.prompt` is `tag/6+<sha8 of the catalogue>`
(`skill_prompts.tag_version()`), and the prompt pin pins the TEMPLATE with
the catalogue as a placeholder, so adding a term never breaks the pin.

| axis | values |
|---|---|
| artifact | code, tests, docs, slides, ui_design, data, config, prose |
| language | typescript, javascript, rust, c, cpp, zig, wgsl, glsl, python, sql, css, html |
| framework | react, r3f, threejs, koota, pmndrs_math, typegpu, webgpu, wasm_bindgen, emscripten, webassembly, cbindgen, tailwind, vitest, jest, pytest |
| phase | plan, implement, debug, verify, refactor, review |

Finer gates a rule MAY carry: `situations` (error_output, call_mismatch,
file_line, multi_file, screenshot_made -- facts about the conversation),
the CLIENT'S TOOLS (`tools_any`, `tools_all`, `tools_none`: tool names the
request offers, or that its history shows the client calling -- how a skill
is keyed on the harness; Hermes offers `vision_analyze` and `terminal`,
OpenCode `read`/`bash`, Codex `view_image`; a rule keyed only on tools
matches as a fact), `all_of` (terms that
must all be present, e.g. html for a browser app), and `topics` (API names
and terms; a code-shaped topic such as `useActionState` is a fact on its
own (a platform or core API such as `localStorage` or `useEffect` only
inside a specific framework's rule: `skill_classify.COMMON_API`) unless the request names other languages instead of the rule's; plain
words need the primary key AND two of them, and then only ASK -- the
embedding stage or the fallback confirms). A rule with no phases applies
in every phase. Every migrated skill is gated by its topics, never by its
language alone.

`pmndrs_math` is the npm package `math` (pmndrs/math): its words are only
forms that cannot mean anything else (`pmndrs/math`, a subpath such as
`math/noise`, `math@0.1.0`, "npm math"), and its import counts only from a
JavaScript/TypeScript grammar (Python's `import math` is the standard
library; `discover` judges stdlib names per grammar).

### What selection reads (2026-09-26, from `bench/skills/replay_selection.py --stack`)

| signal | strength | why |
|---|---|---|
| a PINNED dependency: a manifest line (`"koota": "0.6.6"`) or a spec's pin list (`koota 0.6.6, math 0.1.0`) | fact | a spec pins the packages it is built on; they were words before |
| an import in the last 8 messages' TOOL RESULTS, or in the code a tool call WROTE | fact | an agent step that read an R3F component carried no R3F fact |
| the code a tool call wrote | topic text | a terrain file written with `simplex2d.create` names the noise skill |
| `e.g`, a scoped package name (`@types/react`), a project file's name (`tsconfig.json`) as a topic | prose (confirms, 0.25) | they are in every manifest, pin list or file list |
| a call topic (`select()`, `Fn()`) | needs the call | the bare word matched "no-select" in a CSS rule |
| a code-shaped topic / a plain one | ranks by the number of code-shaped (strong) topics, after strength (no score weights since 2026-09-27: `confidence()` is lexicographic -- strength ordinal fact 3 / phrase 2 / word 1, then strong-topic count, then cosine) | plain words ("render", "imports") outranked every R3F skill on a long spec |
| a topic in the NEWEST evidence (the window since the last user turn, `window_start`) | the newest evidence opens its area (no score bonus since 2026-09-27) | a turn about a failing koota query lost to what the opening spec named |
| a build spec that says "outline" or "spec" | implement as well as plan | code work that is not debugging, verifying, refactoring or reviewing is being implemented |
| the EMBEDDING query: the last user text + a digest of the newest evidence (the last two tool results and writes in the last 8 messages: error lines incl. `error TS2339:`, packages imported and what they are, API names, source files) | stage 2's input (2026-09-27, `skill_classify.embedding_query`) | a short follow-up embedded alone ("the bullets never move ...": koota-queries-and-systems 0.35, fact -> ask) |
| ... joined only when the turn is ABOUT the evidence: an agent step, a failure report / fix ask (the debug phrases), or a user turn naming a word of it (file, API name, identifier part) | else withheld | "make the background dark blue" after a koota error embedded near the koota skills with the digest (0.34 -> 0.55) |
| ... user text first; the digest capped at 800 characters beside a short user text, 300 beside a substantive one (>= 400 characters); 2,000 in all | dominance | a spec's own words still decide its query |

The embedding query changes only stage 2 (and the sticky key, which now
carries it): `query` -- what the fallback's durable record, idle-time
learning, E1 and Laya read -- stays the last user text. The record,
`x_yamadori.skills.embedding.query`, says what the query was built from in
counts and kinds (`digest` joined / withheld / none and `because`,
`sources`, `errors`, `frameworks`, `imports`, `names`, `files`,
`user_chars`, `digest_chars`), never the text; the fallback record's
`features` never carry the request's text (`skill_classify.TEXT_FIELDS`;
`fresh_text` rode along in them from 2026-09-26 to 2026-09-27). Every cap
and the join rule are CHOICES; the replay below is in-sample and n=1 per
scenario.

**Durable records hold no text** (2026-09-27). Every writer of a fallback
record or a router label goes through `skill_learn.durable()`: the query is
the user's own words (`skill_classify.prose_excerpt`: fenced blocks and
code- or log-shaped lines dropped, 600 characters), the signals are
`skill_classify.durable_signals` -- term/artifact ids with strength and
evidence KINDS, phase and situation NAMES (a `file_line` situation's
path:line and an `error_output` match never reach the record), domain ids,
the client's tool names, the embedding query's counts. `python
mcp/skill_learn.py --scrub` rewrote the rows written before it in place
(33 fallback rows and 66 label lines, 2026-09-27; backup in
`index/_archive/skill-records-before-scrub-*`; VACUUM dropped the freed
pages). `mcp/test_skills.py` scans a written row and a scrubbed legacy row.

**Error output** (`_ERROR_OUT`, 2026-09-27) also reads compiler and linter
formats: tsc `error TS2339:`, C# `error CS0103:`, MSVC `error C2065:`,
rustc `error[E0425]:`, ESLint's stylish lines and summary, esbuild/Vite
`[ERROR]`; a coded error pasted into a user turn is the debug phase. A tsc
failure in a tool result now puts the request in `debug` (the V4 TSL step:
implement -> debug; its injected skills unchanged, offline and with the
embedder).

| replay (`--stack --embed`, 2026-09-27; `--user-only` = before) | before | after |
|---|---|---|
| V4 turn 1, pagoda single-html, pagoda r3f-stack (no tool evidence) | unchanged | unchanged (the same query, byte for byte) |
| user turn 2 "the bullets never move" (koota) | injects r3f-frame-loop-10, koota-frame-loop-systems; koota-queries-and-systems ask @0.35 | injects koota-queries-and-systems @0.47, r3f-frame-loop-10 |
| agent steps (R3F read, TSL tsc error, koota throw) | koota-queries-and-systems ask @0.43 | inject @0.46 (the slots unchanged: koota-react-integration, r3f-v10-setup-21) |
| terrain written with math/noise | math-noise-and-seeded-random inject @0.50 | @0.55 |
| Octopus v0e-V0 user turns | p1: browser-app-entry-point + module-exports-match-callers; p2: the same, fix-located-defect-first ask | p1 unchanged; p2 adds fix-located-defect-first |

THE SLOTS before 2026-09-27: the best skill of each AREA first, then the
rest, three in all -- replaced by the section below.

### The explicit ask, precision and the slots (2026-09-27)

Operator: "ensure the engine isn't just injecting 'use this shit' blindly,
it's more about I asked for it, reinforce it." Measured by the DAILY-WORK
EVAL (below); every rule and number here is a CHOICE.

| rule | where | why (the replay that found it) |
|---|---|---|
| EXPLICIT ASK: a framework, library or package the user names in their OWN words in the current request (`skill_classify.asked_terms`: the prose of the last user turn -- fenced blocks, code/log lines and pasted manifest, lockfile and listing lines dropped; a name behind "without", "not", "instead of" asks for nothing; a version right after it, "r3f v10" or "r3f (react-three-fiber) v10", is kept) | `request_signals` `asked`; a SPECIFIC framework becomes a FACT (`asked ...`), a language or a host framework (React) stays a WORD | "Build a small particle demo in React with @react-three/fiber v10 (webgpu) and koota ...; use TSL" injected NOTHING (a scoped package counted only as confirming prose); "pmndrs math" was not a term at all |
| "my React page", "this Rust error": a CONTEXT mention, not a choice | `asked[..].mode` | the project it is, not what to build it with: its area gets a skill only when the skill is about the request |
| PLATFORM APIS are no subject: `localStorage`, `fetch`, `useEffect`, `setTimeout`, React's hooks, the language's core objects, and the topics the store files under 3+ areas (`skill_classify.COMMON_API`) count as PLAIN topics -- except inside a SPECIFIC framework's rule (koota, R3F, math), where they are its subject | `strong_topic(t, rule)` | "Add a dark-mode toggle ... remembers the choice in localStorage" injected an auth-token skill (topics localStorage, sessionStorage) as a FACT |
| SUBJECT RELEVANCE: the request's own words against the skill's trigger text (name, title, description, trigger lines, topics -- never its body), each shared word weighted by its rarity in the store (IDF); an identifier stays one word | `skill_select.relevance` | ranks an asked area's skills; the asking SENTENCE's words first ("build it with r3f v10" -> the v10 setup skill, not the one that shares "detail" with a long spec), then deterministic evidence, then the request's words, then the more general skill (fewer gates); a skill whose NAME carries a major other than the one asked (v9 beside v10) is skipped (a structural filter since 2026-09-27; it was a rank penalty) |
| a HOST area (a language, React) or a CONTEXT mention gets a skill only when two shared words of weight >= 3.5 (in <= ~3% of the store) sum to 7.0 | `relevant_enough` | the dark-mode request shares toggle + mode with the application-state skill ("color mode toggles"): in; a shadow-acne question shares shadow + scene (4% of the store) with a TSL skill: out |
| a host area's CONFIDENCE extra whose area evidence is prose only needs a subject topic (its trigger text names the topic) or subject relevance | `subject_topics` | an ungated React skill ("render a component within another") fired on every React request |
| a HOST language's skill filed under several languages serves each (embind under C and C++); a framework's does not | `area_pick` | an ask for C++ found no C++ skill; an R3F skill filed r3f + webgpu is not the WebGPU skill of a plain three.js ask |

THE SLOTS (operator, 2026-09-27: "one slot per EXPLICITLY asked area, then
the implied areas ..., then confidence-ranked extras only when strongly
supported"; `skill_select.select`, each recorded as `matched[].slot`):

1. **asked** -- every area the user asked for gets its best skill
   (`area_pick`); a specific framework is reinforced even before any of its
   API names appear, a host area or a context mention only when the skill
   is about the request. An asked area an earlier pick already names is
   covered (the R3F v10 setup skill, filed r3f + webgpu, is WebGPU's too).
2. **implied** -- `skill_select.IMPLIES`, applied top to bottom, each row
   citing the package's own docs (the held READMEs,
   `index/packages/_src/<pkg>/`, or the skill's own source):

   | asked | implies | from |
   |---|---|---|
   | r3f | react-dev-learn-referencing-values-with-refs-ref-vs-state-2 (area react) | @react-three/fiber readme: "a React renderer for threejs" that "must pair with a major version of React"; its per-frame rule is refs over setState |
   | r3f v10+ | r3f-tsl-hooks-18, threejs-llms-full-tsl-e-g (area threejs) | @react-three/fiber 10 readme: "significant work in v10 WebGPU support is first class. We support all ThreeJS WebGPU features/Nodes" |
   | webgpu + threejs | threejs-llms-full-tsl-e-g | three.js readme: WebGL and WebGPU renderers; WebGPURenderer materials are TSL |
   | koota + react | koota-react-integration | koota readme: `import { WorldProvider, useQuery, useTrait } from 'koota/react'` |
   | koota + r3f | koota-with-react-three-fiber | koota's own references/react-patterns.md |
   | pmndrs_math + threejs (or r3f) | math-with-three-js | math readme: "interops with WebGL, WebGPU, ... your favourite renderer" |
   | typegpu + threejs | typegpu-three-tsl-integration-2 | @typegpu/three readme: "A helper library for using TypeGPU with Three.js" |

   A named skill, not an area's best: React holds 225 skills and three.js
   26 TSL ones; the one an R3F project needs from each is known from the
   docs. A row whose skill is not armed is skipped.
3. **confidence** -- the remaining candidates only when STRONGLY supported
   (fact or phrase strength, the embedding stage's high cosine, or a model
   stage's confirmation), ONE per area not already covered, a host area
   after the areas it hosts.

No per-turn token budget and no small count cap: the skills are budgeted
at input (size targets below); a sanity CEILING of 8 per decision
(`skill_limits.MAX_SKILLS_PER_TURN`, a CHOICE) guards a pathological
request, and when it binds, the asked slots come first. THE OPERATOR'S
SENTENCE ("Build it with r3f (react-three-fiber) v10 and Koota and pmndrs
math.") appended to the pagoda prompt injects 8 bodies: r3f-v10-setup-21,
koota-with-react-three-fiber, math-data-oriented-functions (asked);
react-dev-learn-referencing-values-with-refs-ref-vs-state-2, r3f-tsl-hooks-18,
threejs-llms-full-tsl-e-g, koota-react-integration, math-with-three-js
(implied) -- ~2,290 tokens (estimated at 3 chars/token), 1.73% of main's
132,096-token window; with the craft index (12 entries, ~620 tokens) and
yama_recall_craft's description (~210) ~3,100 tokens, 2.4%. On the Octopus spec
the same sentence fills the ceiling with browser-app-entry-point first (the
spec asks for an HTML page) and drops koota-with-react-three-fiber and
math-with-three-js (~2,250 tokens, 1.70%).

### Per-turn, evidence-triggered injection (2026-09-27)

Operator: "a really smart skill selector based on inputs where it matches
and knows a skill would really reinforce here at this turn" -- and, from
their own practice, "remember we don't do x, we do y, we need to be mindful
of performance here ... it works". `skill_select.decide` runs on EVERY
request, agent steps included:

| trigger | fires when | first time | later |
|---|---|---|---|
| asked | the user names the stack in this turn (asked / implied slots) | body | recall |
| new_area | its area or API names appear in the NEWEST evidence -- the user turn, or what the last step read or wrote (the first koota query, the first TSL node material, the first math/noise) -- and, on a step, a topic the skill is ABOUT (`subject_topics`) | body | nothing (unless phase / error / asked) |
| error | the newest tool result shows an error the skill matches (a koota query throws) | body | recall |
| phase | the work changed phase (implement -> debug -> verify) and the skill is gated on the new one | body | recall |

No fade and no cooldown (2026-09-27, docs/CONSTANTS-AUDIT.md
"FADE_TOKENS", "RECALL_EVERY_STEPS": both were ours): a given skill comes
back only on its own EVENT -- asked, an error, a phase entered.

THE TWO FORMS: the first time a skill is needed, its body (under the craft
header); after that a RECALL LINE -- the skill's FIRST DO (or WHEN) item and
its FIRST DO NOT item, no scoring (2026-09-27: the item weights were ours),
worded `Remember (craft <name>): <DO
item>; not <DO NOT item>.` or `Remember (craft <name>), be mindful when
<its WHEN situation>: <item>` (skill_prompts CRAFT_RECALL_*, pinned). A DO
NOT part comes only from the skill's own DO NOT items. At most one recall
per area per turn, never the identical line twice in a row; a need that
comes back with nothing new (same phase, no error, not asked) is not
repeated. An agent step has no body cap of its own (`STEP_MAX_BODIES` is
gone); the per-decision sanity ceiling applies.

PLACEMENT: the end of what the model reads next -- the user turn's tail,
or APPENDED TO THE TOOL RESULT that raised the need (`proxy._skills_step`,
the `skills` part after library use and the situations), decided once
under that message's ledger key and replayed byte for byte, so the next
request extends the slot (`mcp/test_skill_turns.py` [served], through the
served template).

THE STATE (one ledger row per conversation, kind `skills`): what was given
and at what conversation size, the last recall per area, the phase, the
request count, the craft offer and the tools withheld. Idempotent per
message key (a retry gets the same text and changes nothing); a
COMPACTION -- the proxy's own count (`progress` "compactions", passed as
`compactions` by `proxy._skills_tail` / `_skills_step`; the old 0.7 shrink
ratio is gone) moving -- forgets what was given; a conversation the state
has never seen treats everything as new. A HARNESS NOTICE (a user turn
that opens with a bracket tag, `skill_classify.is_harness_notice`: Hermes'
`[CONTEXT COMPACTION ...]` summary, `[IMPORTANT: Background process ...]`)
asks for nothing: it opens no asked slot (daily_eval
h4-compaction-summary: the summary's boilerplate had picked
typescript-6-0-release-notes-type-ordering in the pagoda-h4 replay).

THE WINDOW OF EVIDENCE (`skill_classify.window_start`; 2026-09-27, it
replaced `WORKED_MESSAGES` / `WORKED_CHARS`): the messages since the last
user turn, or since the one before it when the request ends on a user
turn. The embedding query is the user text first, then a FIXED digest of
that window (every error, import, name and file line; no counts), always
joined, the whole cut at `EMBED_QUERY_CHARS`.

`x_yamadori.skills` per decision: `kind` (user / step), `decisions[]`
{id, name, slot, trigger, form (body / recall), why, items (a recall's item
indexes), tokens}, `skipped[]` with the reason, `phase`, `phase_changed`.

### The work's own evidence (2026-09-27, pagoda-h4)

The pagoda-h4 run (C:/Users/jwals/octo/logs/pagoda-h4-pagoda-xhigh-1,
relay.jsonl `x_yamadori.skills.matched`, n=1 run) injected craft the work
never needed. Each case traced to evidence that was not the project's, and
each is fixed at its cause, by structure (no threshold):

| what fired | why it fired | the fix |
|---|---|---|
| sudheerj onClick, React 19 upgrade (dom migration), you-might-not-need-an-effect, forwardRef, patterns-by-usecase (new_area bodies) | the step READ a dependency's installed files: koota's README (`<button onClick>`), fiber's `index.d.ts` exports (`unmountComponentAtNode`, `createRoot`, `act`), koota's types d.ts (`onChange`) | `evidence_view`: a tool result whose call reads inside `node_modules/<pkg>/` (`site-packages/<pkg>/`) is not evidence; `x_yamadori.skills.set_aside.dependency_reads` |
| ts-react-cheatsheet-reactnode-async-children (error recall on nearly every write) | Hermes' write_file lint listed tsc errors located in `node_modules/@types/react` ("Type alias 'ReactNode' circularly references itself", from a `lib` setting) | lines whose first located path is inside an installed dependency are dropped from the evidence (the error SITUATION and phase still read what was sent) |
| react-dev-rules component-usage, bulletproof URL state (new_area bodies) | writing package.json / vite.config.ts: filed React + TypeScript, NO topic, so the file type was all the evidence | on a step (and a later user turn) a craft needs what it is ABOUT in the newest evidence: a subject topic, or a gate of its own (situation, all_of) -- never its area's file type alone |
| typescript-6-0-release-notes-es2025-target-2 (new_area body) | writing / reading / patching tsconfig.json: `moduleResolution`, `allowSyntheticDefaultImports` are every tsconfig's keys | a manifest or config file's content (package.json, tsconfig*.json, *.config.*, pyproject.toml ...) is not topic evidence; its pins still are (read from the whole conversation) |
| typegpu-pipelines-pipeline-unwrap-4, typegpu-pipelines-command-encoders-1, web-gpu-typed-arrays-21, web-gpu-use-transition-24 (error bodies, relay 206/210) | the proxy selected on the LEDGER-RESTORED messages: math's noise body (injected at 121) says "draw"; the unwrap body injected at 206 says `root.unwrap` and TypeGPU, which picked web-gpu-typed-arrays-21 at 210. `root.unwrap` occurs nowhere in the transcript | `proxy._skills_tail` selects on the client's messages (`raw`); `evidence_view` strips a craft block of ours handed back; an ERROR brings a craft only when the error's own lines name it (`error_names`); a craft about a framework the conversation does not use -- every framework it is filed under or its description names, closed under IMPLIES and `BUILT_ON` (typegpu -> webgpu, r3f -> react, threejs, each from the package's readme) -- stays out on evidence (`subject_unused`) |

`bench/skills/replay_selection.py --transcript pagoda-h4-pagoda-xhigh-1
--no-server-tools` (the Hermes stream, 266 messages, state carried): 29
bodies and 40 recalls before, 16 bodies and 25 recalls after; every case in
the table gone, the TS moduleResolution craft now arriving on the tsc error
that names `moduleResolution` (TS5110). The daily eval keeps its 464 and
gains 34 MUST-NOT checks from h4 (498/498); a mutant with the four checks
turned off fails 8 of them.

### Server-tool recall (2026-09-27) -- RETIRED the same day

**Retired** (operator, after pagoda-h5: three recall lines went out and
yama_think_deeply was never called; "this model isn't taking gentle hints,
we need to tell it what to do ... at the right time"). The detection below
survives as `skill_select.server_tool_triggers`; each trigger now RUNS its
job (deep.decide kind `auto`, the result a hidden hop, then a directive line
in the model's voice) -- AGENTS.md "Heavy-handed, at the right time". The
`TOOL_RECALL_*` texts and their pins are gone; nothing below goes out any
more. Kept as the record of what was built and why.

Operator: "The goal is to influence research and planning on the turns that
happen right before model is told to go implement the thing." Through the
per-turn channel above -- a line at the end of what the model reads next,
decided once under the message's ledger key and replayed -- a RECALL line
that names one of our server tools, only where main has it (the proxy's
`_server_tools`: `deep.think_tool_offered`, not withheld for a conflict):

| trigger | fires when | line (skill_prompts, craft/3) |
|---|---|---|
| probe | a call reads, greps, cats or lists inside an installed package (a package.json version grep is not a probe) | `TOOL_RECALL_THINK`: "Remember (server tool yama_think_deeply): it reads <pkg>'s own source on the Yamadori server and answers how <pkg> works in one call, each fact with its file and line." |
| scratch | a call writes a throwaway probe: a temp directory, `_tmp_*` / `tmp_*` / `*.tmp.*`, progress.py's scratch names (`cd DIR && cat > x` resolves to DIR) | the same line naming the package the file imports, else `TOOL_RECALL_THINK_ANY` |
| next_piece | a project write opens a directory, under the plan's tree, that the plan's FILES never named | `TOOL_RECALL_PLAN_NEXT` ("<dir>/ is a new piece of the work; it plans the next piece ...") |
| plan_done | every file the plan's FILES lists has been written | `TOOL_RECALL_PLAN_DONE` |
| implement | a user turn after an answer asks for something to be made or changed (`route.work_intent`) | `TOOL_RECALL_PLAN_USER` |

Once per KEY per conversation (think:<pkg> for a probe and a scratch file
alike, scratch:<dir>, piece:<dir>, plan_done:<plan sha>, implement:<n>),
kept in the skill state (`tool_recall`) with the plan's files (read from
the ledger-restored yama_plan result, `plan_files_of`, and kept: a
harness's compaction drops the hidden hop). `x_yamadori.skills.tool_recall`
[{name, trigger, key, evidence: {package, paths | path, why | dir}}] --
paths and packages, never text. UNMEASURED WORDING.

`route.work_intent` reuses the router's own verb sets (`_WRITE`,
selection's act verbs less `run`, the edit verbs) with the pro-verbs that
carry a proposal forward ("do it", "go ahead", "ship it", "carry on") in a
clause-head grammar: an imperative, a request lead ("can you"), a desire
("I want a leaderboard"), "let's", or a sequencing adverb heading a noun
phrase ("now the HUD part"); a question, a negation or an information verb
is not one. The embedder and E1 could not be used (no head for it, and no
GPU for this pass). On `bench/skills/work_intent.jsonl` (written before the
detector; IN-SAMPLE): first pass 46/47 positives, 47/47 negatives (the miss:
"I'd like you to ..."); after the fix and 5 hard rows added, 99/99
(precision 1.00, recall 1.00, 49 positive / 50 negative).

The pagoda-h4 stream through the replay (`--transcript`, both tools
offered, the kickoff plan's 20 FILES read read-only from the ledger):
probe for koota @9, **math @21 (its first probe) and never again**
(/tmp/t.ts, `_tmp_test.ts` and `/tmp/pagoda/_tmp_test.ts` import math: the
same key), @react-three/fiber @37, three @43, @react-three/drei @55; scratch
`/workspace/pagoda/tsconfig.tmp.json` @129; next_piece `pagoda/src/layout`
@111 (builder.ts). The run has no user turn after an answer that asks for
work. Gates: `mcp/test_tool_recall.py`.

### The craft index and yama_recall_craft (progressive disclosure, 2026-09-27)

**Names (2026-09-27, operator).** Every tool the proxy adds to main is `yama_*`, a name no harness offers, and its description says it is a server tool (it runs on the Yamadori server and does not touch the workspace): `recall_craft` is `yama_recall_craft` (craft/2 in `mcp/fixtures/craft_prompt_pins.json`). The old names are still read in stored ledger rows and records, and a call by an old name runs as the new tool (`proxy.LEGACY_TOOL_NAMES`, AGENTS.md "The surface").

Operator: "we can lead with you have these skills available, if you need
more, with a sound way for the model to ask". On a conversation's FIRST
request (`skill_select.craft_offer`, kept in its state like yama_think_deeply's
offer: the system block and the tool list never change mid-conversation):

- an INDEX of the craft relevant to it -- the chosen skills, the asked and
  implied areas' next best, the `ask` candidates -- at the END of the
  system text, after the addendum: `## Craft you can recall` and one
  `- name: when it applies` line each, at most `INDEX_MAX` (12) lines of
  140 characters;
- ONE tool of ours on main, `yama_recall_craft` (`skill_prompts.CRAFT_TOOL_NAME`,
  the one constant; argument `name_or_topic`), run by the proxy as a HIDDEN
  HOP like yama_think_deeply: the craft in full as the tool result, ledger-
  replayed. It reads by NAME only (craft/4, 2026-09-27: the topic-match
  threshold was ours); a topic or an unknown name returns `NO_SUCH_CRAFT`
  (retryable, a remedy, and `near`: every name sharing a word, closest
  first). A
  craft read this way counts as GIVEN. `x_yamadori.craft` {tool, why, kept,
  listed, index_chars, reads[] {found, name, version, how, also, tokens}}.

Offered only when the index has entries (a CHOICE: Shi et al. 2023, "Large
Language Models Can Be Easily Distracted by Irrelevant Context" -- a
conversation nothing matched gets neither). A conversation that continues
without a decision gets none (adding one then would change the cached
system block).

THE WORD: everything the model reads says CRAFT, never "skill" (operator,
2026-09-27: "not a skill, it is a craft, mastery, occupation, or some other
name the model attaches meaning to"): Hermes offers skills_list /
skill_view / skill_manage and OpenCode a `skill` tool for the HARNESS's own
skills. Internal names stay skill_*. The texts (block header, recall
wording, index head, the tool's description written as trigger conditions,
its results) are versioned (`craft/4` since reading by name only, 2026-09-27) in skill_prompts and pinned by
`mcp/test_skill_turns.py` against `mcp/fixtures/craft_prompt_pins.json`.

NO CONFLICT WITH THE HARNESS'S TOOLS (`proxy.tool_conflicts`, every tool of
ours on main): withheld on the same name normalised (case, `-`/`_`, a
plural s) or a DECLARED OVERLAP (`proxy.TOOL_OVERLAPS`): yama_describe_image
beside Hermes' vision_analyze; yama_generate_image beside a client image
generator (Codex's hosted image_generation); the delegate arm beside
OpenCode's `task` / Codex's multi_agent_v1. Decided NOT to overlap:
yama_describe_image vs Codex view_image (view_image attaches; this text model
sees it only through yama_describe_image), yama_recall_craft vs the harness's skill
tools (different stores, different names), yama_think_deeply vs a sub-agent
tool. The withheld set is kept for the conversation;
`x_yamadori.tools_withheld` [{ours, because, client_tool}].
`mcp/test_harness_decisions.py` column `tools`: every fixture of every
harness (Hermes, OpenCode, Pi, Codex; chat and Responses) served at `max`
with an image server configured, no duplicate or overlapping names, Hermes
withholds yama_describe_image; a mutant that adds ours regardless is caught.

### The daily-work eval (2026-09-27; the permanent gate)

`bench/skills/daily_eval.jsonl`: 70 rows, 85 judged requests -- R3F v10 +
koota + math (the exact Octopus V4 prompt, whose first line is now the
operator's "build a space shooter game with r3f (react-three-fiber) v10 and
Koota and pmndrs math", the exact pagoda r3f-stack prompt, the sentence on
the original prompts and their no-sentence TWINS), three.js TSL / TypeGPU / WGSL, TypeScript and
React apps, Rust / WASM / C ABIs / emscripten / Zig, Python scripts,
shell and devops, SQL, docs and prose, questions with no code, bug reports
with stack traces, agent steps with tool results, and SEQUENCES (a koota
area first gets the body, a later koota error step a recall line naming
updateEach, an unrelated step nothing (no fade since 2026-09-27); an unrelated user turn after a stack kickoff gets nothing; a
"still broken" user turn brings the koota queries skill back). Labels come
from the request's INTENT: `must` (a skill, or `area:<id>`), `must_not_areas`,
`must_not`, `nothing`, `allow_areas` (may go in, not counted against
precision), `forms`, `recall_has`. Runner: `python
bench/skills/replay_selection.py --daily [-v]` (a copy of the live store,
deterministic stages; `--embed` adds the embedding stage when nothing else
uses the A4000); gate: `bench/skills/test_daily_eval.py` in
`scripts/run_tests.py`, failing on any MUST, MUST-NOT or NOTHING miss.

| daily eval (n=1 each, in-sample: the rules were adjusted while reading these misses) | before | after |
|---|---|---|
| checks passed (68 rows; the 2 exact-prompt rows were added after, 22 checks, all pass) | 396/442 | 442/442 (464/464 with them) |
| MUST-NOT violations | 3 (dark mode: the auth-token and module-init skills; a React form: web-gpu-precision-14) | 0 |
| MUST misses | 43 | 0 |
| recall, asked areas (r3f / koota / pmndrs_math / threejs / typegpu) | 0.00 / 0.19 / 0.00 / 0.17 / 0.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| precision, react / threejs / koota | 1.00 / 0.50 / 1.00 | 1.00 / 1.00 / 0.94 |

BEFORE = the pre-change classifier and slot rule (3 skills, 900 / 1,500
tokens), the pre-retag store backup, injection on user turns only (the old
proxy never injected at a step). The one precision miss left: a "still
broken" turn's TypeError (`Cannot read properties of undefined`) matches
`module-exports-match-callers`' call-mismatch situation, a broader regex
than the skill's subject.

RETAG (through the edit path; `skills/retag/2026-09-27-daily-precision.json`;
backup `index/_archive/skills-before-daily-precision-20260927-052115`): the
11 emscripten skills gained the `emscripten` framework (an ask for
emscripten found no area; the `e.g` topic dropped; one --
emscripten-interacting-with-code-js-c-interop-3 -- was restored to v1: the
edit's re-screen flagged a `<script>` element in its body, a pre-existing
finding the old version was armed with); web-gpu-precision-14 lost its React
19 topics and filing (a mismatched source had given a three.js precision
skill `useActionState`, `startTransition`, ...); the auth-token skill
gained JWT, HttpOnly and RBAC topics. Activation tests: 535/535 armed skills
pass under the new rules (koota-react-entity-lifetime's useEffect / useState
cases are why common APIs stay strong inside a specific framework). The
retag changed 12 trigger texts, so the live trigger vectors are rebuilt on
the next skill request (in the background, the request path's rule).

### The selector: rounds, questions, a decider, a statechart (2026-09-27)

Operator: "we can have a handful of options because we already filter down
by language first; do successive rounds of category filtering with vector
search or just regular pattern matching; then we will be left with a
handful of real questions that we ask and get options from." Built as that,
on top of everything above (the rows, the ask, the implication table, the
slots, body vs recall, placement and ledger replay are unchanged):

**Round 1 -- successive filters** (`skill_select.run_rounds`), each logging
its survivors in `x_yamadori.skills.rounds[] {name, survivors, how, why}`.
PATTERN first (skill_classify's extraction: fences, imports, pins, paths,
the explicit ask, the implied areas). A round whose patterns find nothing
is `how: silent` and admits nothing on a cosine (2026-09-27: `EMB_HIGH` /
`EMB_LOW` were absolute cuts across queries, ours); the cosine only RANKS
options inside a question.

AN AREA IS OPEN (`skill_select.open_areas` / `open_artifacts`; coordinator,
2026-09-27) only when the user ASKED for it, when code shows it (a fence, an
import, a pin, a file path or extension in the request or a tool call, the
router's code class), or when an error line names it. A bare word does not
open one (React as the host of R3F in a long spec, a language in the
harness's system prompt), nor does the derivation "a language is named, so
code is involved", nor an IMPLIED area: the IMPLIES table contributes only
the skills its rows NAME, which pass the category rounds by name. Each
round's `areas` / `why` lists what is open and why, and what is closed
(`react(implied)`, `python(word)`, `code(derived)`).

| round | a skill survives when |
|---|---|
| language | its filing fits the OPEN languages and artifacts (a slides skill needs slides; a framework skill filed under languages needs one of them or one of its frameworks) |
| framework | a framework skill has one of its frameworks OPEN, or is filed under an ASKED language and is not a host framework's (embind under C++ yes; a React skill filed under TypeScript does not ride a TypeScript ask) |
| phase | its phase gate meets the phases the work is in |
| situation | it is a candidate row (all its gates held), or it serves an asked area / is a named implication, its other gates holding (topics relaxed); an asked HOST area's (a language, React) or a context mention's skill must share at least one rare word (weight >= `REL_WORD_MIN`) with the request -- a superset of what its question could ever pick (two such words) |
| not_given_or_faded | the statechart makes it legal now (below; the name predates the fade's removal) |

A row matched by FACT or PHRASE carries its own proof of its category and
never falls to a category round; a bare WORD does not (so "without React",
now `signals.negated`, drops React's word-matched skills and no fallback is
asked about them).

**Round 2 -- one typed question per area** (`skill_deciders.Question`):
STATE = the turn's evidence (the embedding query: the user's words + the
digest of the newest tool results and writes); OPTIONS = the survivors'
trigger conditions (their descriptions), pre-ranked by the evidence and
capped at 5, plus an explicit NONE (`MAX_OPTIONS` = 6, a CHOICE). In order:
the ASKED areas (`asked:<area>`, the asking sentence's words first as
before); the IMPLIED skills (the IMPLIES table: facts, not questions); then
the EVIDENCE areas left open (`evidence:<area>`, host areas after the areas
they host). An asked area whose candidates the evidence cannot pick joins
that area's evidence question instead of being asked twice; an asked area
answered NONE is closed. When no pattern places the turn in any area,
nothing is asked (the CATEGORY question went with the vector admission,
2026-09-27). An agent step asks `step:<area>` questions over its evidence rows.

**The decider** (`skill_deciders`): `decide(state, options) ->
probabilities`; the pick is the argmax when it beats NONE; probabilities are
relative to the set, never compared across questions or to a number.
`YAMADORI_SKILL_DECIDER` names the first; one that cannot answer abstains
and the next is tried:

| decider | reads | status |
|---|---|---|
| `stub` (default) | each option's EVIDENCE: eligible = fact/phrase strength (or confirmed); logit = the rounds' rank; NONE one below the weakest eligible | the gate: the daily eval 464/464, every decision identical to the pre-question selector (86/86 requests, same forms, triggers and slots) |
| `lexical` (TEST-ONLY since 2026-09-27: not in `skill_deciders.SERVING`) | the TEXT only (state vs option words, IDF within the set) | a stand-in for a text decider: 455/464 on the daily eval, 0 MUST-NOT violations; its misses are r3f-v10-setup vs r3f-v9-setup by word overlap and a koota step it answers NONE -- the questions a text model must get right |
| `clm` | the TEXT only, through `mcp/clm.py` (Qwen3-8B `clm-encoder` on the A4000 behind gpu_room, the heads in numpy, option vectors cached per text = per skill revision) | wired, **not run live**: all of a turn's skill questions share `clm.INSTRUCTIONS`, so the state is encoded ONCE (`decide_detail_many`); NONE is `clm.NONE_OPTION`; `ClmUnavailable` -> abstain (recorded in `questions[].abstained`) -> the stub |
| fallback | the model, one call, thinking off (FALLBACK_SYSTEM) | the LAST RESORT: options the evidence cannot settle (`ask` tier: word strength or an embedding-only match), asked ONCE across the turn's open questions, before the stub answers (E1 / Laya as before when switched on) |

**The statechart** (`mcp/skill_chart.py`; XState semantics: a transition
whose guard returns None is an illegal event). Top: the FINE phases
(`implement`, `debug`, `verify`, ...; PHASE_TOP's coarse mapping is gone,
2026-09-27; several phases at once read `a+b`), FOCUS when only the area
changes. Per area: `unseen --BODY--> given`; `given --ASKED|ERROR|PHASE-->
recall_eligible --RECALLED--> given`; `COMPACTED -> unseen`. No faded state
and no cooling guard (2026-09-27). These are the rules `decide` applied
AFTER selection before, moved IN FRONT of the questions: round 5 drops
every option the chart does not allow. Persisted in the existing ledger row (kind
`skills`) as `chart {v, top, areas, skill_area, last}`; the sticky cache
keys on the chart's signature.

Per-round narrowing (offline, deterministic stages, the live store's 535
armed skills; n=1 each, in-sample):

| request | language | framework | phase | situation | legal | questions (stub) |
|---|---|---|---|---|---|---|
| Octopus V4 (exact) | 84 | 81 | 81 | 56 | 56 | asked r3f, koota, pmndrs_math; evidence ui_design (none), javascript |
| pagoda r3f-stack (exact) | 47 | 44 | 44 | 44 | 44 | asked r3f, koota, pmndrs_math |
| pagoda, no stack sentence | 41 | 0 | 0 | 0 | 0 | none |
| "Start a shooter in TypeScript and use koota" | 356 | 98 | 98 | 5 | 5 | asked koota; typescript (none) |
| koota bug report (stack trace) | 358 | 68 | 68 | 3 | 3 | evidence koota, javascript, code |
| a koota error step after the body | 1 (vector not run on a step) | 1 | 1 | 1 | 1 | step koota -> recall |
| "Add a dark-mode toggle to my React settings page" | 267 | 227 | 227 | 13 | 13 | asked react |
| "a particle demo in React with @react-three/fiber v10 ... koota ... TSL" | 353 | 349 | 348 | 73 | 73 | asked r3f, koota, threejs |

Before the opening rule the same requests read V4 382 -> 381 -> 380 -> 60,
pagoda 345 -> 344 -> 343 -> 44, TypeScript+koota 405 -> 370 -> 369 -> 62,
dark mode 316 -> 276 -> 275 -> 225, particles 353 -> 349 -> 348 -> 289: the
implied React and three.js areas and every framework word kept their
hundreds of skills. After the framework round the pagoda holds exactly the
asked areas' skills (r3f 32, koota 6, math 4) and the two named
implications; V4 also keeps the ui_design and docs areas its spec OPENS by
naming a `.css` file and README.md (answered none). A request that ASKS for
a host (TypeScript, React) keeps that host's area open through the framework
round -- the user asked -- and the situation round keeps only its skills
that share a rare word with the request. Every decision is unchanged:
daily eval 464/464, 86/86 requests identical, the stack scenarios inject
the same skills; no MUST needed a generic React or three.js skill (the
React and three.js skills injected on the stack prompts are the IMPLIES
table's named ones). `python bench/skills/replay_selection.py
--daily --show <id,...>` prints each request's rounds, questions (options,
probabilities, pick) and chart; `--decider lexical|clm` swaps the first
decider (the gate pins `stub`). Tests: `mcp/test_skill_questions.py` (the
rounds, questions, deciders incl. CLM through a fake and through the real
client with a synthetic checkpoint, the transition table and legality, and
the served template: every request extends the slot while the chart moves
given -> recall_eligible -> given).

**What CLM needs before `YAMADORI_SKILL_DECIDER=clm` is claimed**
(docs/CLM-EVAL.md's rules; nothing here ran on the GPU):

1. The encoder served: llama-swap `clm-encoder` loaded on the A4000
   (`gpu_room` SIZES row `clm-encoder`, 9,400 MiB ESTIMATE for Q8_0: it
   does not co-reside with image generation), and `python mcp/clm.py`
   showing the heads and tokenizer (both present 2026-09-27).
2. Option vectors warmed for the armed store (`clm.warm_options` over every
   armed skill's `option_text`, plus `clm.NONE_OPTION`), so a request
   encodes only its state.
3. A live check through :1234 that a turn's questions cost ONE state
   encoding and the record's `questions[].clm {state_tokens, truncated,
   ms}` is sane, then the daily eval with `--decider clm` on an idle card
   (a GPU consumer: queue it), reported as MUST / MUST-NOT beside the stub's
   464/464 and lexical's 455/464, and the stack scenarios. CLM replaces the
   stub only if it matches the stub's MUST-NOT count (0) and loses no MUST;
   its value is where the stub must ASK (the ask tier the fallback settles
   today) -- measure the fallback-call rate it removes.

## Size targets (`skill_limits.py`; CHOICES, not measurements)

| limit | value | why |
|---|---|---|
| skill body | aim 100-300 tokens, hard 450 | THE BUDGET LIVES HERE (operator, 2026-09-27: "the input skills need budgeted"): enforced at authoring and ingest, never by truncating a skill live |
| items per skill | 6 | one situation, a handful of moves |
| characters per item | 300 | the corpus runs p50 166 / p90 233 |
| prohibitions per skill | 2 | AGENTS.md "Prompting this model"; over it FAILS validation (flagged, not rewritten) |
| per decision | no token budget; a sanity ceiling of 8 skills (`MAX_SKILLS_PER_TURN`) | selection gives each asked area a slot, then implied areas, then strong extras; `TURN_TOKENS_*` and the dashboard meter are gone (2026-09-27) |
| recall | no fade, no cooldown; one recall per area per turn, never the identical line twice in a row; 2 items (first DO/WHEN + first DO NOT) | structural (2026-09-27) |
| craft index | 12 lines of 140 characters (`INDEX_MAX`, `INDEX_LINE_CHARS`) | a list to choose from, not the craft |
| name / description | 64 / 1,024 characters | Agent Skills and Hermes caps (the 300 aim is gone, 2026-09-27) |

THE STORE AGAINST THESE CAPS (audit 2026-09-27, 535 armed, 3 chars/token):
body p50 293, p90 403, max 442 tokens; 262 (49%) over the 300 aim, 82
between 250 and 300, 0 over the 450 hard cap, 39 under 100; 0 over 6 items
or 2 prohibitions. By kind: migration n=522 p50 298 / p90 404, 259 over the
aim; authored 3, decomposed 6 and url 4 skills 1 each. PROPOSAL (not
applied: the operator decides): tighten the distil and decompose prompts
from "2 to 6 items, each under 300 characters" to "2 to 4 items, each under
220 characters" (p90 of the corpus is 233) and make the validator FLAG a
body over the 300-token aim for review rather than pass it silently;
re-distil the 259 migrated skills over the aim through the pipeline when
the model is free. DECOMPOSE_SYSTEM still says the compact model "reads at
most three skills at a time", no longer true since the ceiling rose; its
wording is pinned, so it changes with a version bump.

Tokens are estimated at 3 characters per token (deliberately high).

## One pipeline

| kind | stages |
|---|---|
| URL | fetch -> screen -> screen_model -> licence -> distil -> review -> classify -> tests -> validate -> prove -> arm |
| pasted text (+ optional goal) | screen -> screen_model -> licence -> distil -> review -> classify -> tests -> validate -> prove -> arm |
| frontier SKILL.md (URL ending `/SKILL.md`, a GitHub `tree/` folder, or a pasted SKILL.md) | [fetch ->] screen -> screen_model -> licence -> decompose; each ATOMIC child: review -> classify -> tests -> validate -> prove -> arm |
| compiled (migration, authored, a dataset's rows) | screen -> review -> classify -> tests -> validate -> prove -> arm (installed INLINE with no model: review and prove record "not run") |
| edit | screen -> screen_model -> review -> classify -> tests -> validate -> prove -> arm (the skill serves nothing meanwhile) |
| rebuild (`mcp/skill_rebuild.py`, 2026-09-28) | review -> classify -> tests -> validate -> prove -> arm: a served skill's own items as the draft, its source and quotes kept; the served version serves until this one arms |
| watch (scheduled re-fetch) | a changed source is a new version from screen; the served one serves until it arms or quarantines. A changed FRONTIER source takes `watch_frontier` (screen -> screen_model -> licence -> decompose, since 2026-09-27; it was distilled into one skill): a re-proposed child (same name) becomes a new version of itself, a child whose section is gone is archived |

States: a skill is `pipeline | armed | quarantined | failed | disabled |
archived | decomposed`; a version `running | armed | superseded |
quarantined | failed | decomposed`. No HUMAN review stage: a version that
passes validate and prove ARMS. A failed screen, failed activation tests or
a probe the skill makes worse QUARANTINE.

THE REVIEW (`review`, operator 2026-09-28: "have an auto review agent in a
second pass to ensure it doesn't happen again, and we get the best skill
reduction without [losing] actionable information"). After distil or
decompose and before tagging, the model reads the draft against the
definition (`review/1`) and gives each item keep, rewrite (the most compact
actionable form, the current way) or drop (a reason from
`skill_prompts.REVIEW_DROP_REASONS`: no_action, verify, instability,
history, other_stack, duplicate). The code decides what stands
(`skill_pipeline.review_items`): a rewrite must parse as one item, pass
`skill_limits.doubt`, be no longer than the item, name no code its item and
quote do not, and keep one of the item's code names, and it keeps the
item's quote; a drop must give a listed reason, and a `no_action` drop of an
item that names code and passes the doubt rule is REFUSED (it carries an
action by construction); an item the reply leaves out is kept. The record --
before, after, every verdict and refusal, the tokens before and after -- is
the version's `review` column. A model that cannot answer raises
(retryable); a reply that is not the JSON asked for is recorded and the
draft goes on (validate's floor applies). `YAMADORI_SKILL_MODEL_REVIEW=0`
switches it off, recorded. It runs as the helper job `skill.review` under
the helper thinking cap.

THE PROOF (`prove`, operator 2026-09-28: "each skill needs proof that it
works ... a mini A/B test to ensure it doesn't make a task worse, generated
in the skill pipeline"; `mcp/skill_prove.py`). After validate, 2-3 probe
tasks from the skill's own should-cases, checked by CODE (every code block
parses; a TYPE CHECK -- `mcp/typecheck.py`: tsc 5.9.3 / pyright 1.1.414 from
the harness box's tools volume `yamadori-typecheck-tools1`, mounted
read-only in a throwaway `--rm` container of the box image, against the HELD
package version (the skill's `package_version`, else the held major its name
names, else the newest held) and its held `@types/*`, npm-installed once per
package set into a named cache volume; "not run" only when the package ships
no types or the checker cannot run, said why; a Python probe naming a
package is "not run": the box has no pip; a DO's code present, a DO NOT's
absent; a model judge only when no code check decides, labelled `judge:
model`); the served model answers each WITHOUT and WITH the skill injected
exactly as the selector injects it, the same seed and sampling per pair. A
check that passed without and failed with is FLAGGED, and a flag is one
sample: the probe is run again on a NEW seed (the same on both sides) and the
check is worse only if worse in a strict majority of its runs (THE REPEAT RULE,
2026-10-06, `skill_prove.REPEATS` = 1, so two runs and a majority of two; the
derivation is in that constant's comment). A confirmed worse QUARANTINES the
version with the evidence and the runs; an unconfirmed flag is recorded in the
record's `unconfirmed`; a tie arms with "no measurable gain"; no threshold. A proof in
which no check decided (every probe on the token limit, or nothing
checkable, or no probe derivable) is retried ONCE -- each probe again with
the job's full answer room and a second seed, the same on both sides -- and
still undecided is UNPROVEN: QUARANTINED, never armed (operator,
2026-09-28). (tiers.budget already gives a helper request the whole share
less its prompt as max_tokens, so the retry's room equals the first
attempt's; what differs is the sample.) A backlog proof of a skill that is
already serving records `unproven` and leaves it serving. Templates `probe/1`, `prove_answer/1`,
`prove_judge/1` (registered and pinned with the others). GPU work: every
prove job waits for an idle stack (`skills.enqueue`), and the worker keeps
one idle-gated backlog proof queued for the armed library, the newest
versions first (`skill_prove.schedule_backlog`; `--estimate` reports the
seconds per skill once proofs have run). Not yet run live.

CHANNEL-SCOPED ELIGIBILITY (coordinator, 2026-10-07; operator: "Lets cook the
features ... gather evidence"). The activation tests test the TEXT MATCHER
(skill_classify / skill_select), which serves the per-turn selector and the
matcher-based triggers; the package skills channel (mcp/package_skills.py)
picks by the exact package and major the model looked up and never uses it.
So a skill that only the activation tests object to -- the screen, licence,
quote and faithfulness gates having passed, and the skill not operator-written
or a lead -- is NOT quarantined when it has a package channel to ride (its
area is a registry package's: koota, r3f, threejs, pmndrs_math, typegpu; a
skill about React or TypeScript in general has none): validate records
`validate.channels = ["package"]` with the failing cases and why
(`validate.package_only`) and PROVE decides, under the repeat rule. One that
passes is armed PACKAGE-ONLY: `skills.armed()` (every text-matched path) never
lists it; `skills.armed(include_package_only=True)` (the package section, its
between-turn triggers, yama_recall_craft by name or question) does. One PROVE
finds worse stays quarantined. `skill_pipeline.admit_package_only(sid)` moves
an already-quarantined one back to PROVE; `YAMADORI_SKILL_PACKAGE_ONLY=0`
restores the old rule. `python mcp/skill_prove.py --promote-package-only`
runs the activation tests again for every package-only skill and makes each
one that passes a full skill (`skill_pipeline.promote_package_only`).

THE MATCHER'S VERSION AND NEGATION GATES (mcp/skill_classify.py, 2026-10-07,
from the activation cases of the gap-fill skills): a major version the
request states ("React Three Fiber v9", "React 17", a pinned dependency)
different from the one the skill is about (the one major its applies-when and
trigger text name right after a framework) excludes it; a name right behind a
negation ("no React", "without R3F", "no React, no React Three Fiber") is no
evidence for it and rules out the skills gated on it, as does a "vanilla" /
"plain" / "standalone" stack for the React and R3F layers; a request naming
R3F meets a "needs React" gate. `mcp/test_matcher_gates.py`.

THE LICENCE NEVER BLOCKS (operator, 2026-10-07, verbatim: "WE DONT NEED TO FUCKING LICENSE TEXT THAT WE INJECT IT IS FAIR USE WE ARE NOT DISTRUBITING IT ANYTHING HERE WE ARE DOING IS FINE"). The licence stage RECORDS what it finds, from
a verbatim quote only (a SKILL.md's own `license:` line, a licence line in the
source, the repository's own licence file at the pinned ref -- LICENSE-DOCS*
for a documentation page, then LICENSE*, LICENCE*, COPYING* -- or the
operator's statement, `POST /dash/api/skill/licence`) and otherwise records
`not established`; it ALWAYS passes. No quarantine, no failure, no operator
remedy, no clarify question: a missing, restricted or no-derivatives licence
is provenance (a no-derivatives one carries a note). A licence is still never
guessed and written as fact, and a CC BY licence's attribution block is
optional provenance. A pasted text with no licence line is recorded as
`operator-supplied`.
The ~39 skills MIGRATED from the recipe corpus with an `unknown` or absent
licence STAY ARMED (operator, 2026-09-26: "Keep the skills armed"): they
were already served as hints, and the licence is still never guessed --
the library's licence filter shows them, and recording one through
`POST /dash/api/skill/licence` fills it in. The licence stage above binds
new submissions only.
Every model stage (screen_model, distil, decompose, tag, tests, faithful) runs
as the named helper job `skill.<purpose>` (`skill_pipeline.shaped_body`):
role helper, so `tiers.budget` caps its thinking at `HELPER_THINKING`
(6,144; no pipeline job has a `JOB_THINKING` row). Until 2026-09-27 they ran
as role main with no job: thinking was the main share less the prompt.

The screen STRIPS zero-width typography (ZWSP, ZWNJ, ZWJ, word joiner, an
inner BOM) and records it as a note; the stripped text is what is screened,
read by the model and quote-checked (`skill_screen.strip_typography`,
`skill_pipeline.source_texts`). Unicode tag characters and bidi
overrides/isolates still quarantine (2026-09-27: three.js r185's
`docs/llms-full.txt` was quarantined for two U+200B on line 457).

TOPICS are API names and symbols (distil/5, decompose/4, tag/6): a proposed topic is kept only when it is code-shaped (`skill_classify.code_shaped`) and named by both the skill (its items or their quotes) and the source it was made from; each refusal is recorded (`classify._tag.topics_dropped`). The koota live run of 2026-09-27 (0 of 13 armed) is why: plain topics, short quotes and should-cases that missed their own gates.

BOUNDARIES are recorded, never gating (`mcp/skill_boundaries.py`; research
Part 4.4): validate records whether the description states a "Not for"
boundary (`validate.boundary`) and runs every SIBLING's own should-cases
-- armed skills keyed on the same framework, else language -- against the
new skill (`validate.siblings`: siblings, cases, fires, the cases that
fired). `python mcp/skill_boundaries.py black-holes` lists every armed
skill that is a candidate for other AREAS' should-cases, most areas
first: black-hole candidates for the operator, never retired
automatically.

Scripts are data: a frontier skill's `scripts/` are never fetched, and a
command line in an item is dropped by the item screen.

### Rebuilding the library to the definition (`mcp/skill_rebuild.py`, 2026-09-28)

Operator: "You can't just archive skills, you need to build new skills ...
Archive the doubt-skills only when their replacements are ready to arm, or
in the same step." The plan is `skills/replacements/2026-09-28-assured-voice.json`:

| batch | what | replaces (archived on settle) |
|---|---|---|
| `r3f-v10-docs` | 42 new URL skills: @react-three/fiber's own docs at v10.0.0-alpha.5 (commit 27df622f; 26 pages -- Canvas, objects, events, hooks, TypeScript, testing, pitfalls, scaling, the frame loop, the scene pages, the tutorials, the six WebGPU pages) and drei v11.0.0-alpha.7's (commit fee88b6c; 16 pages); the migration guides are left out | on settle, the 17 armed skills compiled from `bench/recipes/r3f.jsonl` that carry a doubt line (r3f-alpha-3, r3f-v10-setup-21, r3f-frame-loop-10, r3f-useframe-1/2, r3f-peer-deps-13, r3f-drei-pairing-8, ...); the 11 CLEAN how-to packs (r3f-events-9, r3f-instancing-11, r3f-html-2, r3f-useloader, r3f-usethree, r3f-tsl-hooks-18, r3f-bounds-1, r3f-lod-3, r3f-softshadows-4, r3f-concurrency-7, r3f-usegltf-20) each only when EVERY page `replaces_when_proved` maps to it has an armed replacement whose proof is better or tie (a clean pack the map does not list is kept); the four `r3f-v10-*` skills from the v10 docs recipes stay |
| `pmndrs-three-upstream` | koota v0.6.6's and pmndrs/math 0.1.0's SKILL.md and references re-run through the fixed templates as new versions of the skills that hold them; three.js r185's `docs/llms.txt` and `docs/llms-full.txt` (the full TSL reference; the screen strips its zero-width characters) distilled new | nothing (a new version supersedes its own skill) |
| `rebuild` | every other armed skill with a doubt line (54 on the 2026-09-28 store): a new version of itself through the review | a rebuild that keeps no item archives its skill |

`--plan` lists it read-only; `--queue` creates and queues it (GPU work: every
gpu stage of a rebuild or replacement waits for an idle stack); the worker
SETTLES a batch after each of its skill jobs (`skill_pipeline.advance_after`
-> `skill_rebuild.settle_after`), and `--settle` does it by hand: when no
member is still in the pipeline and at least one armed, the replaced skills'
SKILL.md folders are copied under `index/_archive/` and the skills ARCHIVED
(`skills.archive`, nothing deleted) with the reason citing the operator; a
batch that armed nothing archives nothing. Until then the serving-time floor
(THE ASSURED VOICE, above) keeps their doubt lines out of every request.
`bench/skills/daily_eval.jsonl` rows that name `r3f-v10-setup-21` carry
`pending_reingest: {"r3f-v10-setup-21": "area:r3f"}`: while it is armed the
named check applies; once archived, the r3f area stands in until a row
names the replacement. Tests: `mcp/test_skill_rebuild.py`. Not yet run
(needs the GPU).

### The offline path (`mcp/skill_offline.py`): the pipeline with no model run

The operator's rule while fixes are in flight is "no model runs"; a source
still goes through the ONE pipeline. A spec (JSON beside hand-written
replies) names each source URL, its kind (`frontier` -> decompose, `url` ->
distil) and the replies the model stages would have given (decompose or
distil, tag, tests). The code checks everything it always checks: robots
and the size cap, the deterministic screen, the licence from a verbatim
quote (the repo's LICENSE), every quote found in the source, the caps, the
taxonomy, the activation tests, arming. The model screen is SKIPPED and
recorded so. The faithfulness check is MANDATORY (2026-09-27): a spec
with no `"faithful": {"by": ...}` verdict arms nothing, and a hand
verdict is recorded with who wrote it. Nothing is queued
(`skills.create(..., enqueue_first=False)`, `run_inline`), so the worker
never claims a stage. A spec's `rules` key a skill the pipeline filed
elsewhere (the tag stage adds at most two terms and no `all_of`): an
operator EDIT, version 2, through the edit path; version 1 keeps the
pipeline's filing. `--retag FILE` edits armed skills' filing and trigger
text the same way (items untouched); an edit that does not arm is undone
(`skills.reinstate`: the previous version serves again). An edit keeps its
description sentences as triggers and its source's licence; an item carried
over unchanged keeps its verified quote.

| spec | what |
|---|---|
| `skills/ingested/pmndrs/spec.json` | koota v0.6.6's and pmndrs/math@0.1.0's own SKILL.md (frontier), koota's react-patterns.md and runtime.md and math's API.md (distil): 10 skills |
| `skills/retag/2026-09-26-r3f-tsl.json` | 54 migrated skills refiled: TSL under three.js, drei under R3F, typegpu-three under TypeGPU + three.js, R3F descriptions rewritten, `e.g` topics dropped |
| `skills/retag/2026-09-27-daily-precision.json` | the daily eval's precision fixes: 11 emscripten skills under the `emscripten` framework (one restored: its re-screen flagged a `<script>`), web-gpu-precision-14's React topics and filing dropped, the auth-token skill's JWT / HttpOnly / RBAC topics |

`python mcp/skill_offline.py SPEC --temp --fixtures bench/skills/fixtures/pmndrs`
replays the ingestion offline (`mcp/test_pmndrs_stack.py` does);
`--temp --copy-live --retag FILE` tries a retag on a copy of the live store.

### The prompt templates (`mcp/skill_prompts.py`)

| template | version | stage | job |
|---|---|---|---|
| distil | distil/6 | distil | distil/6 (2026-09-28): the DEFINITION above opens the prompt, and the item table routes what changed between versions to "the current way only, as a DO for the version the skill is for", and caveats, hedges, "verify / check before using" advice, advice with no action and advice for another stack to "leave it out"; a DO NOT names the wrong move and the right one. Before: the table had no row for history or verification, and the r3f recipes it was never given were written as migration diffs. A source becomes ONE atomic skill (name, description, phases, topics, items with quotes); the description may add a sentence opening "Not for" that names the nearby situation needing other advice (a BOUNDARY: recorded by validate, kept out of the trigger text by `skill_classify.strip_boundary`) |
| decompose | decompose/5 | decompose | decompose/5 (2026-09-28): the DEFINITION and distil/6's item rows; "what changed between versions" is left out, the current way kept. A frontier SKILL.md becomes ONE compact LEAD skill for its package first (`lead: <package>`: what the package is and its core pattern from its opening sections, the size of any skill -- SkillsBench 2602.12670: compact +19.0 vs comprehensive +0.7; the first part's lead only; the package named in the source or declared as the source's `meta.package`; its SKILL.md carries `metadata.yamadori.lead_for` and its served row `lead_for`, which mcp/skill_packages.py reads to PUSH it; the matcher excludes it) and then atomic skills, each with the distil/4 boundary sentence: keep advice about the work, drop steps for tools our harnesses lack, rationale and scripts; keep a DON'T that names a real failure (<= 2) |
| tag | tag/6 | classify | files the skill on the taxonomy; every value is verified (a term the source names, a topic the skill contains). `all_of` (tag/5): terms that must ALSO be in use (koota WITH React), kept only when the text names them (`skill_classify.verified_all_of`), moved out of the OR list, refused when nothing would be left to key on. When the tag keys a skill on a framework and lists no language, a language the text merely names is its host and leaves the OR list (`_tag.host_languages`) |
| tests | tests/3 | tests | activation tests (should / near-miss should_not) and behaviour checks; since tests/3 the prompt carries the skill's own GATES (libraries and all_of, topics, phases, situations: `skill_pipeline.gates_text`) and each should-case must satisfy them. A phase or situation gate that refuses one of the skill's own should-cases is DROPPED and recorded (`tests.gates_dropped`). A LEAD's tests are written from its package alone and run through `skill_packages.detect` (`_lead_activation`: the package in play for each should-case, not for each near miss), since a lead is pushed, never matched |
| faithful | faithful/1 | validate | does each item say what its quote says; unfaithful items are dropped. MANDATORY for every model-written skill (2026-09-27; research Part 4.5: SkillsBench self-generated -8.1 to -11.5, ASI verification +4.2): switched off -> the version FAILS; the check errors -> retryable, the version waits at validate; an item with no verdict is dropped. Every verdict records its caveat (`skill_pipeline.FAITHFUL_JUDGE`): the judge is the model that wrote the items |
| review | review/1 | review | the second pass (THE REVIEW above): keep / rewrite / drop per item against the definition, verified by code |
| probe, prove_answer, prove_judge | probe/1, prove_answer/1, prove_judge/1 | prove | THE PROOF above (`mcp/skill_prove.py`) |
| quote_repair | quote_repair/1 | validate | ONE repair round before an item is dropped for its quote: the item, its exact failing quote and the source section (the child's own headings, or the chunk sharing most words); a new quote is kept only when it verifies exactly (in the source, at least MIN_QUOTE_CHARS). `validate.quote_repair` {asked, repaired, items} |
| screen | screen/1 | screen_model | `skill_screen.SCREEN_SYSTEM` |

Each carries the data-not-instructions paragraph and at most two
prohibitions; each text is pinned to its version
(`mcp/fixtures/skill_prompt_pins.json`): change the text, bump the version.
The model PROPOSES; the code verifies every proposal.

### Tests stored with each skill (`tests.json`, `mcp/skill_tests.py`)

```json
{"activation": {
   "should": [{"text": "The page loads but nothing draws.",
               "files": ["index.html", "js/game.js"],
               "tool_output": "Uncaught TypeError: x is not a function",
               "route_class": "agent_step"}],
   "should_not": [{"text": "Write a Node.js script that sums a CSV column.",
                   "files": ["scripts/sum.js"]}]},
 "behaviour": [{"prompt": "...", "check": {"kind": "regex", "pattern": "..."},
                "why": "..."}]}
```

A case becomes a small conversation and runs through the selector's
deterministic stages. ARMS only when every `should` case is a candidate and
no `should_not` case is (a choice). The score and, against the armed pool,
each should-case's rank are recorded. Behaviour checks are stored, not run:
they are the live A/B (with vs without the skill).

## The API

All paths are under `/dash/api/`, behind the same key check as every other
dashboard API (`accounts.identify` in `mcp/server.py`). JSON in, JSON out.
Errors: `{"ok": false, "error": "..."}` with 400 (bad input) or 404 (no such
skill). No response carries a filesystem path.

### GET /dash/api/skills

```json
{"skills": [{"id": "508dc52b9091", "name": "browser-app-entry-point",
             "title": "Browser App Entry Point", "status": "armed",
             "reason": null, "enabled": true, "served_version": 1,
             "latest_version": 1, "source_kind": "authored",
             "description": "Use when building, ...", "tags": ["JavaScript"],
             "category": {"artifact": ["code"], "language": ["javascript"],
                          "framework": [], "phase": ["implement", "debug",
                          "verify"], "domain": ["web-frontend"]},
             "gates": {"phases": [...], "situations": [], "all_of": ["html"],
                       "topics": []},
             "applies_when": "code is being written or edited (JavaScript); with HTML; while implementing, debugging or verifying",
             "activation": {"passed": true, "score": 1.0, "n": 7,
                            "failures": []},
             "provenance": {"kind": "authored", "licence": {"spdx": "MIT", "quote": "MIT License"}},
             "folder": "browser-app-entry-point",
             "meta": {}, "created": 1790400000.0, "updated": 1790400000.0}],
 "counts": {"pipeline": 0, "armed": 526, "quarantined": 0, "failed": 0,
            "disabled": 0, "archived": 0, "decomposed": 0},
 "recall": "skills",
 "stages": ["fetch", "screen", ...], "paths": {"url": [...], ...},
 "taxonomy": {"artifact": [...], "language": [...], "framework": [...],
              "phase": [...], "situation": [...]},
 "taxonomy_counts": {"phase": {"debug": 2}, "language": {"typescript": 120}},
 "limits": {...}, "thresholds": {...},
 "prompts": [{"name": "distil", "version": "distil/2", "stage": "distil",
              "chars": 3100, "sha256": "cb7311f79147f542"}],
 "learning": {...}, "queue": {...}}
```

Browsing by taxonomy, tags or state is client-side over this list.

Every row (and the detail view) carries `size` (`dash_skills.size_of`):
`{tokens, chars, items, prohibitions, longest_item_chars,
description_chars}` of the INJECTED body (title + items; tokens at
`limits.chars_per_token`), so the library can show each skill against the
caps; null for a skill with no text yet.

### GET /dash/api/skills/{id}

`{"skill": {...everything above..., "text" (= skill_md, the name the
current SKILLS tab reads), "skill_md": "---
name: ...", "versions": [...], "jobs": [...],
"tests": {...tests.json...}, "children": [{id, name, status}],
"selections": [{ts, skill, version, route_class, decided_by, strength,
request}], "learned_triggers": [...]}}`. Each version carries every
stage's output: `fetched`, `screen` (deterministic + model), `licence`,
`distil` (prompt version, counts), `classify` (the rule; `_tag` = the tag
template's proposal), `tests`, `validate` (items kept and dropped with
reasons, `activation` with every case), `text` (the SKILL.md), `reason`.
Jobs carry `queue, state, stage, attempts, error, progress, result`.

### GET /dash/api/skills/{id}/skill.md

The SKILL.md itself, `text/markdown`.

### GET /dash/api/skill-factory/prompts

`{"prompts": [{name, version, stage, chars, sha256, text}]}`.

### GET /dash/api/skill-factory/selections

`{"selections": [...]}`: the 200 most recent selections across skills (one
row per injected skill per DECISION; replays and sticky hits are not
selections). `request` is a hash, never text.

### GET /dash/api/skill-factory/recent

`{"requests": [{at, utility, skills: {...}}], "keep": 50, "limits":
{...}}`: the proxy's last requests' `x_yamadori.skills` (newest first),
kept in the proxy process's memory by `recent_turns` (gone on a restart):
`on, route_class, ids, versions, names, chars, tokens, why, candidates,
armed, cache, replayed, matched[{id, version, name, title, decided_by,
strength, cosine, confidence, why}], dropped[{id, why}], fallback{ran, ok},
embedding{ok, why}`. Never `signals` (terms read from the request) or the
fallback's free-text reason.

### POST /dash/api/skill -- submit

```json
{"url": "https://raw.githubusercontent.com/o/r/main/skills/pdf/SKILL.md"}
{"text": "Prefer small pure functions ...", "goal": "a skill for TS style"}
{"text": "---\nname: x\ndescription: Use when ...\n---\n# ...", "frontier": true}
{"urls": ["https://.../a/SKILL.md", "https://.../b/SKILL.md"]}   // 1-25
```

Optional: `name`, `goal` (steers distil/decompose), `frontier` (forces the
decompose path; default: detected), `watch_hours` (URL; 0 stops).
Response: `{"ok": true, "skill": {...}}` (or `"skills": [...]` for `urls`),
status `pipeline`; the first stage's job is queued.

### POST /dash/api/skill/edit

`{"id", "text", "tests"?}`: `text` is a whole SKILL.md (its frontmatter may
change the name, description and applies_when) or just the body. A new
version; the skill is disarmed until it re-passes the screen, its tests and
validation. Tests carry over only while the rule is unchanged.

### POST /dash/api/skill/tests

`{"id", "tests": {"activation": {"should": [...], "should_not": [...]},
"behaviour": [...]}}`: new tests as an edit of the current text.

### POST /dash/api/skill/activation

`{"id"}`: run the activation tests now against the armed pool. No state
change. `{"ok": true, "id": ..., "activation": {"passed", "score", "n",
"failures", "cases": [...], "pool": {"size", "should_within_cap",
"should"}}}`.

### POST /dash/api/skill/{disable|enable|archive|quarantine}

`{"id", "reason"?}`. `archive` takes a skill out of service and out of the
default view; `enable` brings back a disabled or archived skill (its served
version serves again at once). `quarantine` is an operator's quarantine:
the latest version is marked and the skill disarmed.

### POST /dash/api/skill/rerun

`{"id", "stage"}`: run the latest version again from `stage` (must be on
its path; refused for an armed, superseded or decomposed version -- edit
instead). For a licence supplied after the fact, edited tests, or a failure
that has passed.

### POST /dash/api/skill/reprove

`{"limit"?, "include_unproven"?, "dry_run"?}`: enqueue an idle-gated gpu-lane
prove job (payload `reprove`) for every skill whose latest version PROVE
quarantined as `worse` under the one-sample rule (a record with no `rule`;
not the activation-test quarantines). Each job proves it again under THE
REPEAT RULE and serves it again unless the worse result repeats. `dry_run`
lists and enqueues nothing. Same as `python mcp/skill_prove.py --reprove
[--limit N] [--include-unproven] [--dry-run]`. Returns `{enqueued |
would_enqueue, skipped, estimate, rule, repeats}`.

### POST /dash/api/skill/licence

`{"id", "licence": "MIT", "quote": "MIT License ..."}`: the operator's
licence and the text that grants it; then rerun `licence`.

### POST /dash/api/skill/{watch|refetch}

`{"id", "hours"}` / `{"id"}`: as before.

### Package onboarding (docs/PACKAGE-ONBOARDING.md 8.1; `mcp/onboarding.py`)

`POST /dash/api/skill` with a `prompt` is a PROMPT WITH LINKS: `{"prompt":
"Add koota 0.6.6 https://github.com/pmndrs/koota", "links": [...]?,
"aliases": [...] | "a, b"?, "replaces": "replace" | "alongside"?}` ->
`{"ok": true, "onboarding": {"id", "stage": "resolve", "packages": []}}`.
An alias written `alias=package` binds to that package; an unbound alias
applies only when the prompt names one package. With no `replaces`, the
same major replaces and a new major sits alongside (operator decision 2).
A prompt with no link is a 400. The `url` / `text` / `urls` shapes are
unchanged.

| call | answer |
|---|---|
| `GET /dash/api/skill-factory/onboarding` | `{"onboardings": [{id, group, package, version, stage, state, updated, created, counts, jobs}]}` |
| `GET /dash/api/skill-factory/onboarding/<id>` | `{"onboarding": {...the summary, prompt, stages, joins, job_rows (with result, progress, not_before), blockers, missing, warnings, next_stage, field_states, held, waiting, notes, resolution, licence, index, vocab, examples, knn, sources, skills, skill_counts, retire, eval, reviews}}`; 404 for an unknown id |
| `POST /dash/api/skill-factory/onboarding/review` | `{id, stage, note}`: an after-the-fact note (no review gate) |
| `POST /dash/api/skill-factory/onboarding/promote` | `{id}`: force a HELD vocabulary, recorded with the author; 400 when it is not held |
| `POST /dash/api/skill-factory/onboarding/tier3` | `{id}`: ingest the recorded llms.txt pages |

The licence answer and a re-run of an errored stage job are the datasets
API's (`POST /dash/api/dataset/answer`, `/dash/api/dataset/rerun`); a clarify
is never held for a licence (provenance only,
2026-10-07); `/dash/api/dataset/advance` advances one that waits for a
locator. No answer carries a filesystem path.

## x_yamadori.skills (every response)

`{on, route_class, ids, versions, names, chars, tokens, why, matched:
[{id, version, name, title, decided_by, strength, cosine, confidence,
why}], candidates, armed, embedding, laya | e1, fallback, cache, dropped}`;
on a ledger replay `{ids, versions, names, chars, replayed: true, why}`.
Since 2026-09-27 also `rounds[] {name, survivors, how, why}`, `questions[]
{qid, kind, area, options (skill ids + none), tiers, probabilities, pick,
decider, state_chars, last_resort?, abstained?, clm?}`, `decider`,
`not_legal[] {id, why}` and `chart {state, area, areas, transitions,
events}`; `matched[].question` names the question that picked it. Never the
request's text (a question records its state's length only).
The `hints` / `suppressed_hints` aliases are gone (2026-09-26).

## Operating it

- `python mcp/skill_migrate.py --dry-run | --apply | --authored | --ingest
  DIR` (the migration, the authored skills, a local checkout of a skills
  collection on the frontier path).
- `python mcp/skill_select.py --refresh-triggers` builds the embedding
  stage's trigger vectors for everything armed (~3,000 texts for the
  migrated store). The request path never embeds more than
  `YAMADORI_SKILL_TRIGGER_BUILD_MAX` (128) texts itself: past that it starts
  a background build and runs without the embedding stage, recorded
  retryable, until the vectors exist.
- `python bench/skills/replay_selection.py` replays selection offline over
  an Octopus run and a corpus sample; `--stack` replays the V4 stack's
  prompts (Octopus V4, both pagoda prompts) and hand-written agent steps
  (an R3F component read, a TSL build error, a koota query failing, a
  terrain written with math/noise); `YAMADORI_REPLAY_DB` /
  `YAMADORI_REPLAY_SKILLS_DIR` replay a temp store instead of the live one.
  `python bench/skills/frontier_example.py` shows the frontier path on a
  local fixture.
- `skills.export_armed(dest)` copies every armed skill folder into a
  harness's skills directory.

## Follow-ups (not in this pass)

- A factory view of TOOLS (operator idea, 2026-09-26, design note): read
  the corpus for the tools each harness OFFERED against the ones the model
  USED, and for improvised equivalents (a Playwright script run through
  `terminal` where `browser_*` tools were offered), and recommend a
  harness CONFIGURATION change to the operator -- never to the model. The
  selection side already exists (`tools_any` / `tools_all` / `tools_none`);
  a future skill such as "a web app you built: when browser_* tools are
  offered, load the page and read the console before saying it works"
  waits for evidence.

- The React surface is BUILT (2026-09-26; the SKILLS tab, `/skills`,
  `/skills/create`, `/skills/selections`, `/skills/prompts`,
  `/skills/<id>`: `web/src/screens/Skills.tsx`, `SkillDetail.tsx`,
  `skills/parts.tsx`, helpers in `web/src/api/skills.ts`). Smoke-checked
  against a mocked API only; not yet against the live proxy, whose
  process predates `size` and `/skill-factory/recent` until it restarts.
- Remote repo listing for frontier collections (the GitHub tree API):
  BUILT for a package onboarding (2026-09-27, `package_sources.tree`: one
  recursive tree at the resolved commit, a truncated one walked below the
  skills, docs and examples directories); a collection that is not a
  package is still submitted as `urls`, or ingested from a local checkout.
- The live A/B of behaviour checks, and the embedding stage's thresholds
  calibrated from the fallback records.
