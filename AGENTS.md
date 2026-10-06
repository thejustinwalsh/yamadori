# Repository guidance

**Yamadori** serves a local 27B (Bonsai 2, ternary) with a code-intelligence
tool layer, for SWE work in TypeScript, Rust/WASM across C ABIs, and
three.js TSL / TypeGPU.

Clients reach it at `:1234`, the proxy (`mcp/server.py`, logic in
`mcp/proxy.py`). The proxy advertises one model, `yamadori`, and any
unknown name resolves to it (`mcp/catalog.py`). It serves `/v1` -- chat
completions AND the Responses API (`/v1/responses`, the primary path for
harnesses since 2026-09-26; "The Responses API" below) AND the Anthropic
Messages API (`/v1/messages`, for Claude Code, 2026-09-29; "The Messages
API" below) -- and jjava, our decider, behind TypeSafe's Jev API
(`/jev/v1/systemone`, `/jev/v1/models`, and `/v1/systemone`, 2026-09-29;
"The Jev API" below) -- and the
dashboard at `/` (a browser asking for HTML gets the React
SPA; any other client gets the JSON service descriptor, and old `/dash`
links redirect). llama-swap sits behind it on loopback `:11434`, where
`bonsai` and `bonsai-agent` are the same process: the alias exists only for
old clients; after the layout-v2 deploy `bonsai-vision` (below) is served
too. The tools API is `:1235` and SearXNG (`docs/SEARCH.md`) is loopback
`:8888`; nothing in the proxy searches the web since deep thinking was
removed (2026-09-29), and retiring the service is a deploy step. The
watchdog supervises llama-swap, proxy, tools-api, worker and searxng (Laya,
`:1237`, is retired: docs/REMOVED.md).

**Removed 2026-09-29** (operator: "these old things are failed decisions
... It should be in GitHub if we want to go back, clean it all up."): deep
thinking and everything that fired it, fan-out, the fix-up repair and the
code-check notes, the addendum, the delegate arm, Laya and CLM,
selection's legacy path, the library-definitions injection and LIBRARY
USE. `docs/REMOVED.md` lists what went and why; the way back is commit
`e360d37`. Text below that still names them is history.

## Layout v3: one conversation + the jjava lane (operator, 2026-09-29; built offline 2026-09-29, deployed 2026-10-01/02)

The operator, verbatim: "we should be doing that for all models, we should not have a second conversation at all,
it is too slow, we have a second gpu if we want a second conversation, that is how it has to play out, the jjava
engine should be the only other thing we need ready to go, and if that also makes it dog shit slow, then we just
use jjava bansai and put it on the other gpu, done and done." And: "We clear jjava lane too after it is done right,
not slow down slop." Evidence: Flash-Next decoded an 8K conversation at 8.6 tok/s beside an idle 64K slot -- the
unified pool's top-cell cost (#59, as on Bonsai). It builds on layout v2 (operator, the same day: "The point is to
get more context at speed in vram, so decider was the only thing that needed room"; "Vision can go to second card
and swap in and out"), whose vision move stands. Where this section and older text below disagree, this section is
what the code does (or does after the deploy, where it says so). Details: docs/ENGINES.md "Layout v3".

- **Bonsai and Flash-Next are LOCKED at `-np 1`** (operator, 2026-09-30: Flash-Next "stays locked in once it is
  swapped"; Bonsai "2. Yes"): the ONE conversation alone on the card, no lane; their jjava and side calls run on
  `bonsai-a4000` (a second Bonsai on the A4000, `-np 2`, ctx 141,312 measured by `bench/a4000_fit.py`, n=1 fit;
  REQUIRED for both; `mcp/tier_models.yaml` `locked`, `helpers`). No OTHER work touches a locked card -- no
  decider, side call, other conversation or release (`max_mode.touch_allowed`); the conversation's OWN work does
  (`max_mode.OWN_WORK`, coordinator 2026-09-30): its COMPACTIONS (below), its WARMS (its own slot extended with the
  turn as delivered), and a `/slots` read of the loaded model (the dashboard's KV panel and pulse; never
  `/upstream/<model>/`, which would load an unloaded one). Every INTERNAL generation (`mcp/model.py` post: summarize_text,
  the worker, the skill pipeline, skill_prove, the questions bank, skill_select's fallback decider) goes to the
  locked model's helper, bonsai-a4000 (`max_mode.internal_model`), never onto the locked card, BUDGETED AGAINST
  ITS OWN WINDOW (the table's fitted 141,312, cap 138,240: `tiers.window_model_of`) and refused before anything is
  sent when it cannot fit (`model.WindowExceeded`, with its sizes; the worker fails it, not retryable). A request
  bound for the helper server gets that server's lane slot, 1 (`slots._helper_server_grant`; slot 0 there is the
  other card's conversation). THE WORKER'S GPU LANE PER CARD (`mcp/jobs.py` GPU_SCOPES, coordinator 2026-09-30):
  `gpu` the main card, `gpu_a4000` the A4000 -- the embedder's queues always, a model stage's while the main card
  is locked (decided at claim, recorded on the row as `card`); one running job per card, each paused on its own
  (`jobs.pause("gpu")` is a 5060 Ti window and no longer holds the A4000's jobs; `hold_lane(..., lane=)` in the
  gates, `bench/a4000_fit.py` holds `gpu_a4000`); a job claimed for the A4000 never touches the main card
  (`max_mode.check_scope`: it waits); anything unrouted stays `gpu`. Evidence for Bonsai: the lane's arms (`bench/kv_rank.py --layout v3` lane3, n=3
  each) -- a jjava burst on the lane cut main's decode ~30% (8K 73.8 -> 51.3 tok/s, 32K 72.7 -> 52.4, 64K 73.0 ->
  51.0). Bonsai: `-c` = N, every cell in VRAM, main cap = N (`YAMADORI_LANE_TOKENS` 0); N = 209,920 measured at
  `-np 2` (one fit), safe at `-np 1` (derived: the unified pool is `-c` cells either way; `/props` checked after
  the restart); KV stays q8_0 (operator, 2026-09-30, "Kv8": q4_0 at 262K measured by `bench/kv_q4.py`, n=1, NOT
  deployed). ORDER: `bench/deploy_tier_models.py` first (it brings `bonsai-a4000` up), then
  `bench/deploy_layout_v3.py` (default `--np 1`, refused until the tier deploy ran). Mirai S is LOCKED at `-np 1` too
  (its gate's lane step, `bench/mirai_s_gate.py`, `bench/results/mirai_s/gate-20261001-merged/gate.json`, 2026-10-01,
  n=3 per arm): main decode with the lane ACTIVE was 29.3 / 26.8 / 23.8 tok/s at 8K / 32K / 64K against 33.7 / 31.2 /
  28.2 KEPT and 34.7 / 31.4 / 28.3 CLEARED -- 13-15% below KEPT, 15-16% below CLEARED -- and the gate's rule (ACTIVE
  below CLEARED's minimum at every size) called the lane material: `-np 1`, no lane, `-c 125952` (the fit's MTP window,
  45,568 B/cell and 9,482 MiB fixed from two 8K-warmed launches, less 2,048 cells to keep the 1,000 MiB margin at the
  deepest measured prompt, 126,740 tokens, 951 MiB free), and its jjava and side calls on `bonsai-a4000`
  (`mcp/tier_models.yaml` `locked`, `helpers`). (The lane's size for a model that runs one stays in `budget.LANE_TOKENS`
  = 3,072 = STATE_TOKENS 2,048 + the decider's measured non-state maximum, 999 cells over 200 turns, rounded up to
  256-cell blocks; rank `slots.RANK_LANE` = 3. No main-card model runs a lane now.)
- **One conversation per card; a second one gets THE OTHER CARD** (`mcp/slots.py` ONE CONVERSATION / THE OTHER
  CARD; in force when the served `/props` says 1 or 2 slots). The operator, 2026-09-30, verbatim: "If anything it
  would be ensuring that a second message that came in out of order got the other card, like a 60s timeout before
  you can assign a new conversation to it makes sense, if it doesn't match the current id". The owner is the
  conversation on slot 0 while it has a request in flight or was active (its last request's end, or its latest
  ARRIVAL -- so its queued follow-ups keep the card) less than `slots.PRIMARY_HOLD_S` (60 s; bounded below by the
  relay's longest in-task gap, 38.2 s, n=482) ago. A request whose conversation id is not the owner's, inside that
  hold, is ROUTED to the other card -- the table row's `other_card` (bonsai: `bonsai-a4000`, its slot 0; jjava and
  side calls there use slot 1) -- and KEEPS it for its life (`x_yamadori.slots.routed` {routed: other_card, model,
  why, owner, owner_busy, owner_idle_s}; the proxy sends that conversation's requests to that model, budgeted
  against its window, 138,240). A newcomer whose tier's model cannot run there (flash-next at max, mirai-s at
  xhigh: no `other_card`) gets 503 `conversation_at_capacity` with Retry-After = the rest of the hold (30 s while
  the owner is mid-request) -- NEVER silently downgraded; the one switch `YAMADORI_OTHER_CARD_DOWNGRADE=1` would
  serve it on the default model's other card, recorded. Both cards held: 503. After the hold a new id TAKES the
  main card (`x_yamadori.slots.switch` {from, owner_idle_s, saved}): the previous owner's cells go to llama-server's
  host-RAM prompt cache (`--cache-ram`) when the new prompt diverges from them, and come back if it returns (its
  `x_yamadori.slots.resumed_cold` {how: restored | reprocessed, prompt_ms}). A request in flight is never
  evicted; compactions and warms stay with their own conversation's card. Owner first: when the card frees, the
  owner's request is served first (only the owner is granted the main card inside its hold). The same-account
  immediate takeover of earlier the same day was REMOVED (replaced by this rule). `mcp/test_one_conversation.py`
  (`test_other_card_routing`, `test_owner_first_race`); live: `mcp/test_live_stack.py` `slots`.
  **`--cache-ram` (set in config.yaml and deployed 2026-10-01/02: bonsai 16384, bonsai-a4000 5120; derived)**: bonsai `16384` MiB = two full-context entries (36,992
  B/cell, `bench/results/kv_rank/20260929-v2/line.json`, x 209,920 cells = 7.23 GiB + ~150 MiB recurrent state,
  docs/ENGINES.md = ~7,560 MiB each); bonsai-a4000 `5120` MiB = one entry (35,840 B/cell, `bench/a4000_fit.py`, x
  138,240 = 4.61 GiB + ~150 MiB = ~4,870 MiB); flash-next `8192` (its default, stated). The worst case on the 64 GB
  box is Flash-Next loaded (~38 GB, the coordinator's figure) + 8 + 5 GiB of caches = ~52 GB, ~12 GB for
  everything else; with Bonsai loaded (weights in VRAM) 21 GiB of caches. The caps fill only on switches (a
  ceiling, not a reservation); the rest of the box's RAM use is not measured here.
- **Compactions are never refused for being compactions**: the owner's (its key; in place, or mapped) runs on
  slot 0; one sent up as is with no identity runs on slot 0 once no request is in flight there, waiting at most
  admission's `WAIT_SECONDS` (20 s) and then the Retry-After path; a different conversation's follows the
  second-conversation rule. Never the lane. **A compaction stays on the card of the conversation it summarises**
  (operator, 2026-09-30, verbatim: "it compacts on the same card it came from right? To get cache gains."): it is
  NOT a side call for routing (`max_mode.decide(..., kind="compaction")`, `max_mode.utility_kind`), never goes to
  `bonsai-a4000`, even at max or while the card is locked -- the one exception to "locked"
  (`touch_allowed("compaction")`). `mcp/test_max_mode.py` `test_compaction_stays_on_its_card`.
- **The lane is cleared after each burst** (`slots.LANE_KEEP` False): when decide_turn's Turn closes and after a
  side call, the lane is released (the erase, else the one-token prompt), recorded in `x_yamadori.slots.released`
  with why `lane burst ended` (or `side call ended`) and its ms. Within a burst (one state, many questions) the
  cache is untouched. Measured per model by the lane's three arms (KEPT / ACTIVE / CLEARED, main decode tok/s at
  8K/32K/64K, n=3, and the next burst's re-prefill ms: `bench/kv_rank.py --layout v3`, `mirai_s_gate.py` step
  `lane`). If the lane materially slows main, jjava moves to `bonsai-a4000` (the row's `helpers`) and the main
  card runs `-np 1` -- measured and decided for Bonsai and Mirai S (above), locked for Flash-Next by the operator;
  no main-card model runs a lane now, and the lane's code remains for one that does.
- **Vision off the main card** (layout v2, deployed): `bonsai` runs without `--mmproj`; `bonsai-vision` (the 27B +
  mmproj on the A4000, on demand, ttl 300, `ondemand` group) sees for it (`vision._vision_model`); a tier model
  whose table row says `vision: false` does the same.


## Naming

**Products, routes, groups and classifications take bonsai terms.** They are
the names a person reads.

| name | what it is | in code today |
|---|---|---|
| `canopy` | serving group for the main model | no. `config.yaml` says `primary` |
| `rootstock` | embeddings, always resident (the reranker was removed 2026-10-01) | no. `config.yaml` says `retrieval` |
| `graft` | on-demand models, swapped in and evicted | no. `config.yaml` says `ondemand` |
| `taproot` | result tier: a declaration with this name is here | yes, `find_by_meaning` output |
| `branch` | result tier: both retrievers agree | yes |
| `shoot` | result tier: one retriever only, unproven | yes |
| `rings` | the durable work log that survives compaction | yes, `mcp/rings.py` |

Whether the group names or `config.yaml` should change is an open operator
decision. Do not "fix" either side on your own.

**Tool and MCP call names take Jane Street conventions.** They are the names a
machine reads, and they are part of the prompt.

- `snake_case`, verb first, spelled out; no abbreviations
- `_opt` when the answer may legitimately be absent (`find_definition_opt`)
- `_exn` only if it raises, which none of these do
- `of_` / `to_` for conversions

**The surface** (one model, one cache: operator, 2026-09-24):

| who sees it | tools |
|---|---|
| MCP clients (`code_search.TOOLS`, `:1235`, unchanged; since 2026-09-26 an account key on every request but `/health`, and a browser Origin not on `YAMADORI_TOOLS_ALLOWED_ORIGINS` is 403 -- `docs/TOOLS-API.md`) | `find_by_meaning`, `find_definition_opt`, `find_references`, `find_by_pattern`, `read_file_range`, `summarize_text`, `describe_index`, `run_check` |
| main -- the model the client talks to (`proxy.main_tools`) | the CLIENT's tools, untouched and first, plus only `yama_generate_image` and `yama_describe_image` where offered (rows below), `yama_recall_craft` where the conversation's craft offer says so (row below), and the MCP host's package lookups where its offer says so (row below; 2026-09-28). EVERY TOOL OF OURS ON MAIN IS `yama_*` (operator, 2026-09-27): a name no harness offers, one constant per name (`images.TOOL_NAME`, `images.DESCRIBE_TOOL_NAME` = `vision.TOOL_NAME`, `skill_prompts.CRAFT_TOOL_NAME`, the MCP host's rows), each description saying it is a server tool (runs on the Yamadori server, does not touch the workspace). The old names (`generate_image`, `describe_image`, `recall_craft`: `proxy.LEGACY_TOOL_NAMES`) are still READ -- stored ledger hops replay byte for byte, a withheld list in the craft state maps to the new name, a call by an old name runs as the new tool (`proxy.canonical_tool_name`, `proxy.is_ours`), and a client tool under an old name is a declared overlap. No code tool, no work-log tool, no `check_code`. EVERY tool of ours is compared with the client's first (`proxy.tool_conflicts`, operator 2026-09-27): withheld on the same name normalised (case, `-`/`_`, a plural s) or a declared overlap (`proxy.TOOL_OVERLAPS`: yama_describe_image beside Hermes' `vision_analyze` or a client `describe_image`; yama_generate_image beside a client image generator, `generate_image` included), kept for the conversation, recorded in `x_yamadori.tools_withheld`; gated on every harness fixture by `mcp/test_harness_decisions.py` column `tools`. |
| main, where the craft offer holds (skill_select PROGRESSIVE DISCLOSURE; operator-sanctioned exception to "no tool of ours on main", 2026-09-27) | `yama_recall_craft` (`skill_prompts.CRAFT_TOOL_NAME`, the one constant; argument `name_or_topic`). Its description is written as trigger conditions: it answers "how is this done well?" for one craft from this service's library -- call it INSTEAD of guessing a library's pattern; when about to write code with a library the craft list names, when a line said "Remember (craft ...)", when unsure of a library's pattern or API, when an error comes from a library a craft covers. The proxy runs it as a HIDDEN HOP (the craft in full as the tool result, ledger-replayed); an unknown name returns `NO_SUCH_CRAFT` with its near misses. Offered with the craft INDEX at the end of the system text, both decided on the conversation's first request and kept, only when the index has entries. Everything the model reads says CRAFT, never "skill" (the harness's own skill tools are another store). `x_yamadori.craft`. Skills are off at every tier today (below), so it is offered only where a header turns skills on. |
| main, where the MCP host serves (operator, 2026-09-28: "the proxy provides MCP servers to the model zero-config for every harness, starting with PackageLens"; switch `mcp_tools`, tiers `medium` and up, header / `YAMADORI_MCP_TOOLS`) | `yama_find_package` (words -> the registry's exact names), `yama_list_package_versions` (every version, its date, the dist-tags, prereleases included, each npm version's peer dependencies), `yama_read_package_readme` (the registry's README, else the repository's) -- 3 of PackageLens 0.1.11's 8 tools (`mcp/mcp_config.py` says why each other one is out; `ecosystem` required on all three); renamed verb first 2026-09-29 (they were `yama_package_versions` / `yama_package_readme`: each row's `legacy`, read as the new tool by `mcp_config.tool` / `canonical`, `mcp_host.definitions`, `line_for`, `run_tool`, which records `called_as`); and `yama_resolve_packages` (2026-09-29, operator: "packagelens should be able to resolve peer dep ranges ... recommend not only the core package names but the dependency exact names and version too ... All mechanically from package json info"; `mcp/npm_resolve.py`, a config row with `runner: npm_resolve` in place of an upstream): npm 11.19.0's OWN resolver (`npm install --package-lock-only --ignore-scripts`, `mcp_servers/packagelens/resolve_driver.js` on stdin) in a throwaway container of the same pinned image on the same gated network -- every package together, then the exact set again; an ERESOLVE moves one package within what was asked; the peers npm installed, the optional peers it left out with their ranges, DefinitelyTyped's `@types/*` by major.minor, all verified by npm as one install line plus a dev line; a prerelease only with `allow_prerelease` or a named one, and then the version a dist-tag names; `ecosystem` "npm" required; bound npm's own one-fetch worst case, 970 s; `mcp/test_npm_resolve.py` (the driver against a fake npm with npm's own output shapes), live smoke 2026-09-29 (the sets installed with a real `npm install --ignore-scripts`, no ERESOLVE); **the model's use is not yet probed** (THE TOOL RECIPE rule 7). The PROXY IS THE MCP CLIENT (`mcp/mcp_host.py`, our own stdio JSON-RPC client): each enabled server of `index/mcp/servers.json` (absent -> the built-in default; `YAMADORI_MCP_SERVERS`) runs ONCE, long-lived, started by server.py (never on import), on Docker Engine in the Ubuntu WSL distro (since 2026-10-06: the `wsl-ubuntu` docker context, mutual TLS on 127.0.0.1:2376 only, kept up by `scripts/wsl_engine.ps1`, WSL capped at 5 GB / 4 CPUs; docs/DOCKER-WSL.md; Docker Desktop stays installed, not started), in its pinned image (`mcp_servers/packagelens/`, `models/manifest.yaml` `mcp-packagelens`: the image id is checked before every start) on its own `--internal` network through the egress gate (`bench/sandbox/sandbox_net.py`, NODE_USE_ENV_PROXY=1), uid 1000, every capability dropped, read-only root, no credential; restarted on the next call after a crash; a call bounded by the server's own `call_timeout_s` (PackageLens: 48.6 s, derived from its http.js). Run like the image tools: a HIDDEN HOP (`proxy._run_our_tool`), ledger-replayed; the result rendered to text (a README's images as their alt text), screened by `skill_screen.screen_fetched` and framed as data. The README fallback and the npm packument read go through `mcp/pinned_fetch.py` (the pinned GET: every hop resolved once and `ip.is_global`, GET only, fixed headers, a byte cap; what is left of the removed research tools). Offered on a conversation's first request and KEPT by name (`proxy._mcp_offer`, the skill state's `mcp`), with ONE system line where offered (`mcp_config` `line`: "To find a package, its versions and its README, and the exact versions that install together, call yama_find_package, yama_list_package_versions, yama_read_package_readme and yama_resolve_packages before you write package.json or install it."; 2026-09-29; UNMEASURED WORDING). Withheld beside a client tool that answers the same question (each tool's `overlaps`, read by `tool_conflicts`). `x_yamadori.mcp` {switch, offered, on_main, kept, why, line_chars, calls[{tool, server, upstream, ms, ok, bytes, error, screen, args (120 chars), names}]}; `GET /dash/api/mcp` (read-only; `mcp_config.save` / `set_enabled` are the dashboard page's writers, not routed). GAPS: PackageLens returns `dependencies` only (the peers come from the proxy's own reads since 2026-09-29: `mcp_host.npm_peers`, `yama_resolve_packages`); the npm registry serves an EMPTY README for many packages (math, koota, @react-three/fiber, react, zustand, express, axios, typescript, zod, date-fns, react-router, tailwindcss; 2026-09-28) -- the result then says the README is in the installed package. `mcp/test_mcp_host.py`; live smoke `python mcp/mcp_host.py smoke packagelens` (no model; run 2026-09-28); **the model's use is not yet measured** (`bench/mcp/lookup_probe.py`). |
| main, only when `YAMADORI_IMAGEGEN_URL` is set | `yama_generate_image` (`mcp/images.py`). Every tier -- a capability, not a gate (operator, 2026-09-23). Two image models, `base` (20 steps) and `turbo` (Viggle 4-step): the caller's account preference picks (dashboard SETTINGS, `PUT /dash/api/settings/image`), else `YAMADORI_IMAGEGEN_DEFAULT`, else `turbo` -- the built-in default since 2026-09-24 (operator; the launch scripts also set it; 23.6 s vs ~108 s; quality vs base not yet compared by `bench/imagegen/compare_turbo.py`). The model has no say; `x_yamadori.images` records the model, steps and where the choice came from. **On main the proxy SHOWS the picture the moment it exists** (operator, 2026-09-25): streamed, its markdown line goes out as CONTENT when the tool returns -- before the next generation is requested -- and the rest of the turn's reasoning goes out as heartbeats (CHANNEL ORDER); blocking, the line opens the answer (the same content). Main is offered `images.MAIN_TOOL` (its description says the picture is shown when made), and the tool result says the picture is already shown above the reply (`images.shown_on_main`: the situation, no prohibition); a copy the model writes anyway -- the image with that url, any alt, or the bare url on its own line -- is removed from the CLIENT's copy only (`proxy._ImageDedup`). The line is content the slot did not generate: the ledger keys the turn by the client's copy and renders the slot's text (#10), like a compaction summary's session line, so the next request extends the slot. `x_yamadori.images[]` adds `emitted` {ms since the request started, after_hop, before_hop, at: stream | answer_start, order} and `duplicates_stripped`. `mcp/test_image_emit.py` (through the served template); **not yet run live**. See `docs/IMAGEGEN.md`. On `/v1/responses` a request's hosted `image_generation` tool IS this tool, at the tool's size, and each image also comes back as an `image_generation_call` item ("The Responses API" below). |
| main, wherever `yama_generate_image` is offered, and on every tier when the request carries an attached image; off with `YAMADORI_VISION=0` | `yama_describe_image` (`mcp/vision.py`): sends one image and a question to `bonsai-vision` (the 27B + mmproj on the A4000) through `mcp/model.py`, in the image lane, and returns the answer as text. It reads only this request's attached images (by id `image-<10 hex>`) and our media store (by sha, with a verified signed link or an image made in this request). It never fetches a URL and never opens a path. `proxy.prepare` replaces each attached image part with a placeholder naming its id; an image part carrying OUR signed /media link (signature and expiry verified, host `YAMADORI_PUBLIC_BASE`'s, the request's own, or loopback) is read from the media store as an attachment -- how Hermes' `vision_analyze` looks at what we drew (2026-09-24; offline tests only, the live check in `mcp/test_live_stack.py` images group has not run). `x_yamadori.vision` and `x_yamadori.attachments` record calls and attachments. Streamed, a look the proxy ran on main leaves ONE reasoning line, `` `looked at image-<id>: <the description's first 100 characters>` `` (`proxy._describe_line`, operator 2026-09-25), screened like any tool text shown to the user (markup, links, the template's markers and invisible characters out; a credential, AI-directed text or exfiltration withholds the excerpt: `mcp/skill_screen.py`); once content has started it is a heartbeat. Not delivered on the blocking path (reasoning is not). **Run live once, on an attached image** (2026-09-24, `mcp/test_live_stack.py --only images`, n=1: a correct answer, 5.9 s, 168 prompt tokens for a 256x256 PNG). The draw-then-look check in `docs/IMAGEGEN.md` "Seeing: `yama_describe_image`" has not run. The A4000 VRAM concern there was measured in the same test: 308 MiB free at the peak. |

Deleted 2026-09-24: `bind_project_context` (it pinned versions for tools main
no longer has) and the `check_code` TOOL. The proxy writes the work log
itself (`proxy._log_turn`: the client calls the model made) and re-injects
it on the first user turn after a compaction -- since 2026-09-26 also when
the conversation keeps its session key across the compaction (#41 made the
old new-key link unreachable for Hermes: `progress.note_compaction` in
`_compaction_done`, switch `work_log_reinject`). `proxy.OUR_NAMES` is what the
proxy executes itself: the image tools, the craft tool and the MCP host's
tools. REMOVED 2026-09-29 (docs/REMOVED.md): the second brain's tools (the
code tools, the work-log tools, `find_in_knowledge_base`, `read_web_page`,
`search_web`), `yama_think_deeply`, `yama_plan` and the
`delegate_investigation` arm; their names stay in `proxy._LEGACY_NAMES`, so
a corpus replay never counts them as a client's.

**A client's own side calls get the bare model** (`selection.utility_call`):
no client tools, one exchange, and a reply fixed by a contract (a title
call -- checked first: OpenCode's title generator says it is "summarizing"
"this conversation" and read as a compaction --, a closed
one-word/one-of/JSON/yes-no form addressed to the reply, `response_format`
JSON, or "summarise this conversation", including Pi's and OpenCode's own
summarisers, `compaction.harness_of`). `x_yamadori.utility_kind`:
`compaction` | `title` | `classifier` | `structured` | `other` (2026-09-26:
Hermes' title namer is `title` too, no longer `structured`). The proxy
overrides the tier to `minimal` and adds nothing; the call has no session
(`x_yamadori.utility`, `tier_overridden`). Replayed by `mcp/test_utility.py`:
72/72 side calls in the corpus (Hermes, OpenCode's titles, Pi's summaries;
2026-09-26), 0 of 1,511 task turns and 0 of 4,317 benchmark prompts
misclassified. A header `{"utility": true|false}` forces it -- except that a request carrying an image is never a utility call (the bare text model cannot see; 2026-09-24) -- unless it is a tool-less, single-exchange TITLE or SUMMARY side call quoting the user's image (OpenCode's title call): that stays a utility call, its image becomes the placeholder, nothing is described, and it never becomes the conversation's first request (operator, 2026-09-26; `mcp/test_harness_decisions.py` opencode/chat/title-with-image). Each conversation
is pinned to a llama-server slot (`mcp/slots.py`, `id_slot`); the THREE-SLOT
layout (operator, 2026-09-28; `-np 3` lands with PHASE B's deploy,
`bench/deploy_kv_rank.py`): conversations on slots 0..n-2, and ONE CHILD
slot, `slots.child_slot()` = n-1, the same number in every process, never a
conversation's, shared by the decider, side calls and helper-role requests
(`slots.helper_slot()`; the second brain's jobs ran there until they were
removed, 2026-09-29) -- they queue on it at the server (n-2 of four until
2026-09-28). Every
request names the slots' KV ranks (`kv_rank` / `kv_ranks`, slots RANKS: the
primary conversation 2, another live one 1, the child at its conversation's)
for engine patch 0041, which keeps the primary's cells below the tiered
cache's VRAM line (docs/ENGINES.md "KV rank"); `mcp/budget.py` THE CAP
LAYOUT: main = the line (advertised to every conversation), the child 65,536
of its own. LAYOUT V2 (2026-09-29, "Layout v2" above): the child slot is THE
LANE -- the decider and small side calls (titles), kept, rank 3, 3,072 cells
taken off the cap (`budget.budgets()["child"]`: role "decider lane"); an
as-sent compaction goes to the least recently used conversation slot. `x_yamadori.cache` reports prompt tokens reused vs
processed; `x_yamadori.warm_before` reports the previous turn's warm. A conversation that a harness compacts keeps its work log,
tools and slot (`proxy._continue_after_compaction`).

**An idle slot's cache is not free, so the slots nobody reuses are
released** (2026-09-26; #58 in `docs/SELF-IMPROVEMENT-LOG.md`). With the
unified KV pool every decode step attends over the pool up to its HIGHEST
used cell (`get_n_kv` pads `used_max_p1`; this build's FA vec kernel reads
every cell below it, masked or empty -- only a masked TAIL is ever trimmed,
and only for >= 1,024 queries), idle slots' included: what costs is how
high the top cell sits, not how many cells are held (#59, 2026-09-27).
Measured on the running stack (the
kv_share_probe: bonsai, `-c 181248`, q8_0 KV, 4 slots; **n=2 per cell**),
decode tok/s on slot 0 with the other three slots shrunk to ~0 vs holding
~40k tokens each: 8k context 60.7/47.3 vs 20.1/18.2; 32k 42.8/43.2 vs
15.0/17.8; 64k 26.4/31.5 vs 14.3/15.9 (64k + 3 x 40k exceeds the pool, so
the server may have purged part of that row's fill). A linear fit of those
twelve runs (inferred, not a separate measurement): ~16 ms a token plus
~0.28 ms per 1,000 occupied cells. So the proxy -- and only the proxy
(`slots.enable_release()`, server.py; the worker and the tools API cannot
see a proxy job between hops) -- empties (`slots.release_idle`,
`model.release_slot`) the transient slot after each side call or as-sent
compaction (nothing reuses it: compaction affinity only ever considers
PINNED slots, and a continuation opens with the tools and the summary).
Until 2026-09-29 it also emptied the second brain's slot at the end of
each run (the helper lane, `admission.helper_lane`, which nothing takes
now). Never a pinned slot
(an affinity-placed compaction's included), never a slot with a request of
this process in flight (a new request waits for a release in flight),
never the helper's slot while a run holds a lane, never one another process
is generating on (`/slots` `is_processing`), and only a slot this process
filled. HOW: llama-server's erase (`POST /slots/<id>?action=erase`) where
the server allows it -- it answers 501 without `--slot-save-path`, which
config.yaml does not set -- else a one-token prompt of its own on that slot
(`/completion`, `id_slot`, `n_predict` 0: diverges at position 0). The
switch: `X-Yamadori-Features {"slot_release": false}` or
`YAMADORI_SLOT_RELEASE=0` (default on; `tiers.BEHAVIOURS`);
`x_yamadori.slots` {release: {on, source}, released: [{slot, why,
released, cells_before, ms, method, skipped?}]} and one log line each. What
it costs (CHOICES, unmeasured): a retried as-sent compaction re-prefills
its transcript; each release is one `/slots` read plus one tiny request on
the path of the request that ends the run. Offline: `mcp/test_slots.py`
(release), `mcp/test_utility.py` (side call). **Not yet run live.**

**An idle CONVERSATION's slot is cleared for an active one** (operator,
2026-09-26: "If we get big token boost then clear the idle, no brainer";
#59; this reverses 83831e5's "keep idle slots cached" for slots idle past
the threshold). When a request is about to generate (`slots.clear_idle`,
from `proxy._post_events` and `model.post`, after the slot is granted), every
OTHER conversation's pinned slot idle longer than `slots.IDLE_CLEAR_S` is
cleared the same way. Idle: no request or warm of this process in flight on
it and no warm pending for it (`proxy._idle_guard`), measured from the end
of its last request or warm and from its conversation's last arrival. Never
the requesting conversation's own slot -- a lone conversation never clears
itself, and a side call working for a conversation never clears that
one -- never while a compaction sent up as is is in
flight (it names no conversation; an in-place or mapped compaction runs on
its conversation's slot, which is then busy), never a slot another process
is generating on. The PIN AND THE LEDGER ARE KEPT: the conversation keeps
its slot and replays byte for byte; only the cells go, and its next request
gets its prompt back one of two ways and says which
(`x_yamadori.slots.resumed_cold` {slot, cleared_at, idle_s, cells_cleared,
how, reused, processed, prompt_ms}, also on its `x_yamadori.cache` record;
`slots.resumed_how`): `restored` -- the release's one-token prompt diverges
at position 0, so llama-server SAVES the slot's state to its host-RAM prompt
cache before clearing it (`get_available_slot`, f_keep < 0.5; `--cache-ram`,
default 8,192 MiB, not set in config.yaml) and LOADS it back for the
resumed request (41k tokens: 516 ms live, 1,141-1,214 ms in the probe) --
or `reprocessed` when that entry was evicted. The clearing request lists
`x_yamadori.slots.cleared_idle`.
`IDLE_CLEAR_S` = 600 s is a CHOICE: in the Octopus relay logs (v0b-v0f
V0-xhigh-1, a conversation's request `t_end` to its next `t0`) the 482 gaps
with no compaction between them run median 0.48 s, p90 1.73, p99 17.1, max
38.2 s (a `terminal` step), and every gap over 60 s (11, 81-7,984 s) spans
the conversation's own compaction; one harness, one task family, no human
think time and no cold install in the sample, so the data is thin and the
10-minute floor applies (~16x the longest tool step seen;
`YAMADORI_IDLE_CLEAR_S`). The switch: `X-Yamadori-Features {"idle_clear":
false}` or `YAMADORI_IDLE_CLEAR=0` (default ON, so the live measurement can
confirm the boost). **The boost depends on PLACEMENT** (2026-09-27, #59;
probe on the running engine, one idle 41k-token slot B, active A, decode
tok/s, n=3 each, two runs): clearing B helps only when B's cells lie above
A's highest cell -- 8k: kept 37.1, cleared 62.4 (alone 62.0); 32k: 29.7 /
44.6 (alone 44.4); 64k: 23.3 / 31.3. With B BELOW A it barely helps -- 8k
36.9 / 40.4, 32k 29.7 / 32.4, 64k 23.4 / 24.8 -- and a restored B lands
wherever the free cells are, often above A again. The live test saw exactly
that: kept [29.3, 29.06, 28.05], cleared [45.36, 29.88, 27.51]. Placement
is the engine's (`find_slot` from the head); the proxy cannot see or set
it. The engine fix is `engines/patches/llama-bonsai2/0003` (docs/ENGINES.md,
"Masked KV cells"), built, NOT deployed. A test account may override the threshold for one request
(`X-Yamadori-Features {"idle_clear_s": N}`, `corpus.account_traffic` ==
"test"; refused and recorded otherwise), and then only its own
conversations are candidates. Offline: `mcp/test_slots.py` (idle clear),
`mcp/test_utility.py` (two conversations through `complete()`). Live:
`mcp/test_live_stack.py --only slots` (both mechanisms: released slots read
back ~0 through `/dash/api/vitals/pulse`; decode tok/s kept vs cleared,
n=3 each -- pass: cleared >= 95% of kept, since clearing never slows the
active conversation, the gain printed as evidence; `resumed_cold` with
`how` agreeing with its counts, and its ms; `x_yamadori.cache` carries
`prompt_ms` and `decode_tps`). First run 2026-09-27 (numbers above); the
checks were corrected after it.

**A conversation is named by an explicit session id, never inferred**
(operator, 2026-09-25; #41 in `docs/SELF-IMPROVEMENT-LOG.md`;
`mcp/session_id.py`, `proxy.session_identity`). The first source present:
`prompt_cache_key` in the body, else the `X-Yamadori-Session` header (the key
as before: `nebari.key_of` with the token), else OUR ID, else a NEW
conversation (minted: 12 hex digits; a retried opening mints another,
nothing is resumed). **No visible line in answers** (operator, 2026-09-25:
the first carrier, `yamadori session <id>` on the first answer, broke
exact-output callers live -- "Reply with exactly: ok" came back
"yamadori session dda0ff896a3c\n\nok"). Our id rides INSIDE THE TOOL-CALL
IDS the proxy returns (`proxy._carry_session`, `session_id.carry`): every
client call the model makes goes out as `call_<id>_<8 hex>` (26 chars,
unique per call), streamed from its first delta; a client with its own id
gets its ids untouched. Every client must echo `tool_calls[].id` and
`tool_call_id` verbatim, and the served template renders neither (gated:
the prompt with our ids is byte-identical to the prompt with
llama-server's), so the model never sees them and the ids pass upstream as
the client sent them, every pair consistent (llama-server pairs by id only
for DeepSeek V4). Where it is read back, first found wins, in this order:
`tool_call_id` (ANY assistant `tool_calls[].id` or tool `tool_call_id` of
our form in the history; ids we did not mint name nothing; Hermes' `_d<n>`
duplicate suffix is tolerated), `summary_line` (the one visible marker
left: a compaction summary the proxy writes for a conversation with our id
-- in place, or flattened and mapped by `_serve_compaction` -- opens with
`yamadori session <id>`, so a continuation that keeps only the summary keeps
the conversation), `answer_record` (an answer of ours that carried nothing
-- no client call -- is recorded with the id under the unsalted chain key
before it plus its text as the client stores it, `_record_answer_session`;
our own generated bytes, never the opening alone). **The answer record is a
deviation from the brief, for the operator to confirm**: the brief accepted
that a text-first conversation's next request is a new session; built
literally, that broke the slot's cache on every text-only turn (the new
session's chain salt loses turn 1's recorded injection -- the skills and
the concept seed -- and the tool offers are decided afresh, so the tool list
changes). KEPT at
the 2026-09-27 audit (docs/CONSTANTS-AUDIT.md lists it D, "remove"): it is
not a number but the only carrier a text-only conversation has, and
removing it breaks "one model, one cache" (operator, 2026-09-24) for every
client that sends no key or header -- each such request would be a new
conversation, on a new slot pin. The operator decides. The key is
`session_id.conversation_key` (account, source, id -- never the messages;
all our sources key alike), and EVERYTHING keyed by the conversation uses
it: nebari state, lineage (work log, the concept seed, slot pin,
compaction store), and the ledger's chain keys, which are salted with
it (`chain_keys(salt=)`: two conversations that open alike no longer share
their opening's decisions or seeds). The summary line never reaches the
model (`ledger_restore` strips it from anything rendered; the #10 path
renders the slot's text for the summary turn). `_continue_after_compaction`
stays as the fallback for a continuation with no id, and links ONLY a
compaction this proxy MAPPED to a conversation whose continuation CARRIES
its summary: the recency guess (`COMPACTION_LINK_SECONDS`, 1,800 s) that
also linked an unmapped compaction, or a continuation that did not carry
the summary, to the account's last conversation was REMOVED 2026-09-27
(docs/CONSTANTS-AUDIT.md; operator 2026-09-25: "we can't assume a new
session is a resumable one"). The old fallback --
`nebari.key_of` on the first two messages, `source: none` -- is no longer
reached. `x_yamadori.session` {id, source: `prompt_cache_key` | `header` |
`tool_call_id` | `summary_line` | `answer_record` | `minted` |
`compaction_map` (a flattened compaction mapped to a conversation), why,
carried (client calls whose ids carry it this turn), line (this answer is a
summary that opens with it)}. Hermes sends no per-conversation
`prompt_cache_key` to a custom endpoint (agent/transports/chat_completions.py
`_add_prompt_cache_key`, only for a provider marked
`supports_prompt_cache_key`); a client that sends a CONTENT-addressed one
(the same value for every conversation with one system prompt) would merge
them -- honoured as sent, by rule. Gated offline by `mcp/test_sessions.py`
(43 checks, through the served template). **Not yet run live**: that Hermes
echoes our ids (read from its source: agent/message_sanitization.py) and
keeps the line in its compaction summary is not observed.
**A client's key that changes mid-conversation** (coordinator's decision,
2026-09-26; `session_id.py`): Hermes' Responses `prompt_cache_key` hashes
its session, instructions and tools, so it changes at a compaction. A
conversation named by a key now ALSO carries an id of ours in its call ids
(`session_id.carrier_of(key)`, recorded as an alias of the key; a client's
call ids are no longer returned untouched), and its compaction summaries
open with that carrier's line (in place, or a flattened compaction
`_serve_compaction` maps). A key NEW to us, with no session header, joins
an existing conversation ONLY on COMPACTION evidence -- our summary line in
its history -- and becomes its alias (`x_yamadori.session.aliased_to`).
Our id in tool-call ids alone is a FORK (a fork keeps the full history;
Codex's `fork` sends only a new thread key): a NEW conversation, recorded
with `forked_from` <id> (the slot may still reuse the prefix by affinity).
A key seen before keeps the conversation it named first; an unknown key
with no carried id is new; a session header (OpenCode) or a harness's own
key (Pi) keeps its own. Gated in `mcp/test_responses_api.py` and
`mcp/test_sessions.py`.

## One model, one cache (operator, 2026-09-24)

A client -- Hermes, OpenCode, Pi, anything -- sees ONE standard model. What
the proxy adds never breaks the pinned slot's prompt cache, and main's
context holds only the conversation, what the ledger puts back and main's
own generations. The mechanics this rests on were measured before anything was built
(`docs/SELF-IMPROVEMENT-PLAN.md` Phase 0.5, STEP 0; mechanism probes on the
transient slot, each run twice): after an assistant PREFILL the next request
reused 976 of 995 prompt tokens; after a zero-token WARM it processed exactly
its 23-token tail (93 without the warm); and a request that diverges
anywhere before the end of the slot's sequence falls back to a context
checkpoint (448 tokens in, wherever the edit was -- inferred as
checkpoint restore, not confirmed from the log). So a request must EXTEND
the slot's last sequence -- and since 2026-09-27 it does again, past
reasoning included: the ledger restores the slot's own reasoning for every
past turn (see "Past reasoning is restored" below). From 2026-09-24 to
2026-09-27 past reasoning passed through, and each request diverged at the
previous assistant turn's think block.

**What main's context holds:** the client's messages, what the LEDGER puts
back, and main's own generations (our tools' hidden hops included). Nothing
else -- no turn the client never received. `proxy._run_turn` is the one
turn implementation both paths run (`complete()` drains it;
`stream_body()` streams its events).

**The ledger** (`proxy.ledger_restore` / `ledger_record_turn`; storage in
`nebari.py`, table `additions` in `index/nebari.sqlite3`). Everything the
proxy added, per message, re-added byte for byte on every request:

| addition | keyed by | when decided |
|---|---|---|
| a user turn's injection: skills, the CONCEPT SEED (the conversation's first user turn only, below), the work log after a compaction (a turn whose content is a LIST of text parts -- Pi, Responses `input_text` -- gets it as one more text part, `message_text`; before 2026-09-26 only a string turn got any) | a hash of the conversation up to and including that user turn | ONCE, on the request whose last message is that turn (the first request after a compaction may add the work log); replayed ever after |
| a tool result's injection: skills on an agent step (and, replayed byte for byte, what an older build recorded there: LIBRARY USE #19, the SITUATIONS #50/#54) | a hash of the conversation up to and including that tool result | ONCE, on the request that ends on it |
| a call turn's delivered content, where it differs from what the slot rendered | its tool-call ids | when delivered |
| a past assistant turn's REASONING, the slot's own (a prefill's line included), put back when the client sent the turn without reasoning; an echo is kept as sent (switch `restore_reasoning`, default on, 2026-09-27; below) | the turn's first tool-call id, else its text (like its content) | when delivered |
| hidden internal hops (our tools the proxy ran inside the turn: the image tools, the MCP lookups, the craft tool), with their own reasoning while past reasoning is restored; with switch `restore_reasoning` off, their reasoning emptied -- EXCEPT for a client that echoes: when the turn's reasoning is exactly what the client was shown (kind `hop_echo`: a hash of it and the slot's own reasoning for the turn), the hops keep their reasoning and the turn gets the slot's, so the next request extends the slot (2026-09-26, docs/HARNESS-PI.md gap 5: Pi's request after a `yama_describe_image` turn reused 1,829 of 3,663; `mcp/test_harness_forms.py`, not yet run live). Hops stored by removed features (`yama_think_deeply`, `yama_plan`) replay as stored | the final visible turn, keyed by the content the client STORES (every byte streamed, #10) | when delivered; expanded back in place on replay |
| the conversation's concept seed ({word, token_id, u32}) | the conversation's lineage (`proxy.SEED_KEY`) | with the first user turn's injection; `x_yamadori.session.seed` on every later request, a compaction included |

**Past reasoning is restored** (operator, 2026-09-27, reversing the
2026-09-24 pass-through: "Keeping thinking across turns seems useful, fuck
Hermes, Hermes can do whatever it wants."). The ledger records the slot's
own reasoning for every assistant turn it delivers (kind `reasoning`, under
the turn's first key, keyed by what the client stores, #10; a prefill's
line is part of it) and `proxy.ledger_restore` puts it back into each
past turn the client sent WITHOUT reasoning; hidden hops keep theirs. A
client that echoes keeps its echo, unchanged (a template marker inside an
echo is counted in `x_yamadori.ledger.restored`, not scrubbed). Hermes
strips `reasoning_content` from every replayed assistant turn
(`agent/message_sanitization.py` `apply_reasoning_content_policy`) and
gets it back anyway. Evidence behind the decision: Bonsai 2's card names
its base as Qwen3.8-27B, and the Qwen3.8-27B card says `preserve_thinking`
is "enabled by default for all workloads", citing consistency and
"improved KV cache utilization" (docs/research/SKILLS-RESEARCH.md 1.1,
finding 8). **The served template already renders every past turn's think
block** (the GGUF's `tokenizer.chat_template`, byte-identical to
`mcp/fixtures/bonsai_chat_template.jinja`, line 116: `{%- if
preserve_thinking is undefined or preserve_thinking is true or
loop.index0 > ns.last_query_index %}` renders `<think>\n{reasoning|trim}\n
</think>\n\n` + content), and the proxy never sets `preserve_thinking`, so
nothing is passed: a stripped turn rendered an EMPTY think block, a restored
one renders what the slot generated. So each request EXTENDS the slot again
-- no divergence at the previous think block, and a PREFILLED turn (a
struggle hand-off) needs no warm when nothing else in it changed (from
2026-09-24 to 2026-09-27 its warm re-read back to the server's nearest
surviving checkpoint: 3,512 tokens after the step-1 kickoff, 8,793 after
the v0f step-25 struggle hand-off, #37). THE COST IS CONTEXT: every past
turn's reasoning rides on every request (V0 pilot: 6-10k tokens a step).
Estimated from the pagoda-h4 relay (Hermes, xhigh, n=1 run, 122 main
requests, `completion_tokens_details.reasoning_tokens`, ~3.8 characters a
token): a step's reasoning median 128 tokens, mean 663-1,559 per
conversation segment, p90 3,737, max 4,634; restored reasoning would have
reached 32-41k tokens by each of the three compactions, and the prompt
size each compaction came at would have been reached at request 36 of 54,
32 of 44 and 14 of 24 -- compactions 27-42% sooner (inferred from the
relay, not measured). The client's window check counts it
(`check_client_prompt` measures the request as it goes up, the ledger's
additions included), so a request that no longer fits is refused 400
`context_length_exceeded` and Hermes compacts. The switch:
`restore_reasoning` (`tiers.BEHAVIOURS`, default ON; header
`{"restore_reasoning": false}` or `YAMADORI_RESTORE_REASONING=0` gives the
2026-09-24 pass-through, whose prefilled-turn warm and checkpoint rules
remain in `proxy` A PREFILL MID-CONVERSATION). `x_yamadori.ledger`
{`restore_reasoning` {on, source}, `reasoning_recorded` (chars),
`restored` {`reasoning`, `reasoning_chars`, `reasoning_missing` -- a past
turn with no record: generated with the switch off, pruned, another
server's}}. `mcp/test_ledger.py` gates it offline through the served
template: every request of every session extends what the slot holds (its
previous prompt plus the generation, reasoning included), the simulated
server resumes every request of a mid-conversation prefill session at what
it held (no checkpoint restore), the window check refuses a request its
restored reasoning pushes past the window, and the pass-through arm keeps
its own tests. **Not yet run live.**

**Stray template markers in the answer are the client's to never see**
(operator, 2026-09-25, reversing #12's "delivered as written"). The model
sometimes writes a second literal `</think>` in its ANSWER and repeats
itself (`'Done.\n</think>\n\nDone.'`; 7 of 40 replays of one fixed context
with reasoning echoed, 3 of 40 stripped). `proxy._StrayMarkers` removes a
stray `</think>`, `<think>`, `<|im_end|>` or `<|im_start|>` (with its role
line) from the client's copy, and the text after it while that repeats what
was already sent (a whitespace-normalised prefix: all of it, or whole lines
of it); new text after it goes through, one blank line at most between. A
marker inside code (a fence, or inside backticks) is literal and stays.
Streamed, only the tail that could still be a marker or a repeat is held;
fed a character at a time it gives what `complete()` gives. It runs after
the image-duplicate filter (a repeat is judged against what the client was
sent) and never sees the proxy's own lines (the session line, image lines).
The slot's text is untouched: the ledger keys the turn by the client's copy
and renders the slot's (#10), so the next request extends the slot.
`x_yamadori.template_markers` records `stripped` {marker: n},
`repeat_chars_dropped`, `new_text_after` beside `in_content` (markers left
in code) and the model/ours attribution. `mcp/test_stray_markers.py`.

Only what the proxy or the model produced is stored; the caller's messages
are only hashed (nebari's standing rule: "Not kept: the caller's code"),
EXCEPT a Responses request that did not say `store: false`: its input and
output items are kept under its response id in `mcp/response_store.py`, a
separate store (operator, 2026-10-06; "The Responses API" below), never in
this table.
Every query names the account. Memory is a read-through LRU in front of the
table (`YAMADORI_LEDGER_MEM_MB`, 64). Reasoning and hops follow
`YAMADORI_LEDGER_PERSIST_REASONING` (default on; off, they are kept in
memory only and lost on a restart). Reasoning is now the ledger's largest
kind: pagoda-h4's 2.8 h run would have written ~0.45 MB of it (117,550
reasoning tokens at ~3.8 characters), one row at most ~18k characters, far
inside the caps below and the 64 MB memory layer. Eviction is by SIZE, not
nebari's 36 h TTL: whole sessions, least recently seen first, past
`YAMADORI_LEDGER_ACCOUNT_MB` (256) per account, then
`YAMADORI_LEDGER_TOTAL_MB` (2048) in all, and anything older than
`YAMADORI_LEDGER_MAX_DAYS` (30) -- all three CHOICES (operator, 2026-09-24),
not measurements; pruning runs in a background thread at most once a minute.
`mcp/test_ledger.py` prints the bytes each turn adds (10-370 bytes a turn on
its fixture, whose reasoning is a few words; real reasoning is larger) and
gates prefix stability with the served template itself.

**One system message.** The client's system text may arrive as a
`developer` message (Pi and any OpenAI-SDK client with a reasoning model):
`system_roles.one_system` (the one helper both wires use) maps it to
`system` on the way in (`_run_turn`, `prepare`), before anything reads the
messages -- the leading system/developer messages join into one, a later
one becomes a user turn -- because the template renders only `system`, and
only first (2026-09-26, docs/HARNESS-PI.md gap 1: a system message of ours
in front of Pi's `developer` message was a deterministic 502;
`x_yamadori.roles`; `mcp/test_harness_forms.py`, not yet run live). What the
proxy adds to the system text -- the MCP host's line, the craft index -- goes
at its END (`proxy.add_system_tail`).

**The concept seed** (operator, 2026-09-29). One word drawn from the 27B's
own token embeddings (`mcp/concept_seed.py`, `concept_seed.seed_for` away
from the user's words; `proxy._draw_seed`), ONCE per conversation, appended
to its FIRST user turn -- the first request (no answer in it), not a
compaction's continuation, not a side call -- as
`concept_seed.USER_TURN_LINE`, with the rest of that turn's injection
(`proxy.INJECT_PARTS` = skills, seed, work log), so the ledger replays it
byte for byte and a compaction served on the ledger's rendering summarises
it with the rest. Tiers `high`, `xhigh`, `max` (the tier flag `seed`; the
tiers where every second-brain job carried one until those jobs were
removed); `X-Yamadori-Features {"seed": false}` turns it off. Recorded by
the conversation's lineage and reported as `x_yamadori.session.seed`
{word, token_id, u32}. The line's wording is the research jobs' (#52: a
word drawn at random, not a clue, left out of what is written); UNMEASURED
on main. `mcp/test_ledger.py` `test_the_concept_seed_rides_the_first_user_turn`.
**Not yet run live.**

## Removed 2026-09-29: the second brain and its triggers

Deleted, not switched off (operator, 2026-09-29: "these old things are
failed decisions, they are documented, they are not kept around for later.
It should be in GitHub if we want to go back, clean it all up."). What went,
why, and what each cost is in `docs/REMOVED.md`; the code is at commit
`e360d37`:

- **Deep thinking**: the second brain (`mcp/shomen.py`: the `investigate`
  and `plan` jobs, the hand-off and plan formats, THE EVIDENCE),
  `yama_think_deeply`, `yama_plan` and the kickoff plan hop, the triggers
  (struggle, known-hard area, kickoff, and the server-tool triggers probe /
  scratch / next_piece / plan_done / implement: `mcp/deep.py`), the verify
  directive (`mcp/verify_moment.py`), continue_stated_step, the
  `deep_decisions` rows and their learner (`mcp/deep_learn.py`,
  `/dash/api/deep`), and the research tools only it used
  (`find_in_knowledge_base`, `read_web_page`, `search_web`;
  `mcp/research_tools.py` -- its pinned GET lives on as
  `mcp/pinned_fetch.py` for the MCP host). The evidence: the model never
  called `yama_think_deeply` on its own (0 voluntary calls over whole runs,
  `bench/mcp/lookup_probe.jsonl`), and on the pagoda runs the pushed
  triggers made things worse (hours of probe runs, echoed hand-offs;
  memory "pagoda evidence 2026-09-28").
- **Fan-out** (`mcp/fanout.py`): its only measurement was a null (7/8 vs
  7/8, zero discordant pairs, 3.2x wall clock, n=8).
- **The fix-up repair and the code-check notes** ("Verified" / "Repaired" /
  "Checked", `tool_code`'s check, the answer repair,
  `code_check.review_answer`, the formatters): the model imitated the notes
  (#36) and nothing measured that the repair helped. The IMAGE GUARD stays.
- **The addendum** (`proxy.ADDENDUM`, `ADDENDUM_THINK_ROW`,
  `ADDENDUM_TOOLS`), **the delegate arm** (`delegate_investigation`),
  **Laya and CLM** (services, heads, the `laya` dataset kind), **selection's
  legacy regex/Laya path**, **the library-definitions injection** and
  **LIBRARY USE** (#19): the MCP host's package lookups answer on the
  model's call instead (THE TOOL RECIPE).

**The directive prefill is KEPT** (operator, 2026-09-29: "2 is the right
answer here"): `proxy.directive_prefill(line)` -- a line as the opening of
main's REASONING with the think block left open (reasoning_content the
line, content empty) -- opens hop 0 when a payload carries `_prefill`, and
the ledger restores that reasoning with the turn (restore_reasoning), so no
warm is needed. Nothing sets it today; it is the channel the skills
renderer measures against injected text per model. That llama-server leaves
the block open for a reasoning-only prefill is INFERRED from STEP 0's
/apply-template probe, not measured: confirm it on the live
/apply-template before a result that uses it counts. `mcp/test_ledger.py`
gates its warm, checkpoint and restore rules with an injected prefill.

## The project paths

`mcp/progress.py` keeps which written paths are the project's
(`progress.project_of` / `is_project`: not a temp directory, a dot-file, a
scratch-harness name or outside the working directory the conversation
states, unless the task names the file) and the command readers
(`command_of`, `inspect_only`), for the skills' evidence (skill_select), and
the compaction count for the work log's re-injection (`note_compaction`,
`reinject_due`, `mark_reinjected`). State only, no text the model reads.
The per-step learning of the project (`learn`, an inferred working
directory) served deep thinking's label rule 2 and went with it.

**Every switch** (`tiers.BEHAVIOURS`, default ON): X-Yamadori-Features
`{"<name>": false}` or its environment variable `=0` turns one off --
`work_log_reinject` (`YAMADORI_WORK_LOG_REINJECT`), `step_nudge`
(`YAMADORI_AGENT_STEP_NUDGE`), `slot_release` (`YAMADORI_SLOT_RELEASE`),
`idle_clear` (`YAMADORI_IDLE_CLEAR`), `restore_reasoning`
(`YAMADORI_RESTORE_REASONING`; "Past reasoning is restored", above),
`mcp_tools` (`YAMADORI_MCP_TOOLS`; the MCP host's tools, medium and up, the
surface row "main, where the MCP host serves"), `preread` and
`preread_overlap` (`YAMADORI_PREREAD`, `YAMADORI_PREREAD_OVERLAP`; Flash-Next's
expert-file pre-read, "Flash-Next's first prompt" below). Removed with the features
they switched: `step_thinking`, `fixup_project_only`, `plan_tools`,
`plan_budget` (2026-09-27, docs/CONSTANTS-AUDIT.md); `seed_frame`,
`helped_needs_change`, `plan_prompt`, `deep_tool_hop`,
`continue_stated_step`, `auto_triggers`, `verify_directive` (2026-09-29).
`x_yamadori.progress` records the agent step's thinking cap and nudge and
each switch with its source (header / env / tier / default).

## The code-work router

**One class per request** (`mcp/route.py`), decided once in `proxy.prepare`
and recorded as `x_yamadori.route {class, because, signals}`. First rule that
fits wins: `utility` (`selection.utility_call`) → `agent_step` (ends on a
client tool result; a harness notice or bare "continue"; `acts_locally`) →
`code_edit` (the prompt carries code that PARSES, and the instruction asks
for work on it) → `code_generation` (a write verb + code noun, "export
`name`", "write a Rust ...") → `library_question` (a question, and the gate
offered for a reason about it, or a held source defines a name it uses) →
`prose`. The skills and the agent step's thinking cap read the class
(fan-out and the repair pass, which read the code classes, and the
definitions injection, which read `library_question`, were removed
2026-09-29). A header that forces one still forces it.

Measured by `bench/route/eval_route.py` on `bench/route/labels.jsonl`:
1,016 corpus turns (196 Hermes, 200 SWE-agent, 612 benchmark/probe, 8
other), 189 unique user-speaking requests labelled by hand against the
rubric in that script, 341 agent steps whose tool-result ending is
**reconstructed** (the corpus did not record it; rows from 2026-09-24 do). A
blind second pass on a seeded 20% (203 turns) agreed on 202. **In-sample**:
the rules were adjusted while reading these misses.

| truth (n) | recall | precision |
|---|---|---|
| utility (43) | 1.000 | 1.000 |
| agent_step (371) | 0.995 | 1.000 |
| code_edit (160) | 0.812 | 0.963 |
| code_generation (223) | 0.978 | 0.879 |
| library_question (60) | 0.900 | 1.000 |
| prose (159) | 1.000 | 0.952 |

**Misroutes (utility/agent_step → a code class): 0.** Code vs not: 383/383
code turns routed to code, 0 non-code turns sent there. The code_edit ↔
code_generation confusion is mostly LiveBench starter code the corpus cut
off at 2,000 characters; both classes run the same pipelines. Benchmark
sets: LiveCodeBench 342/342, `bench/domain` 100/100, react 40/40, type
challenges 183/183 route to code. None of the 196 Hermes turns is a code
request: Hermes writes code through tool calls.

**Tool-call repair was REMOVED 2026-09-29** (docs/REMOVED.md): the check of
client writes before they were forwarded, the fix-up that repaired them and
the "Verified / Repaired / Checked" notes. The client receives the calls
exactly as the model wrote them. `mcp/tool_code.py` keeps the DETECTION the
skills, the no-progress guard and result compression read (`detect`,
`read_target`, the known-names table) and the image guard.
**The image guard** (`tool_code` IMAGE GUARD, #46; operator 2026-09-26: image
tools only, no generic size or alphabet heuristic): an IMAGE argument --
known by name (Hermes' `vision_analyze.image_url`, our `yama_describe_image.image`)
or by the request schema's shape -- whose value opens as image data
(`data:image/`, any `data:...;base64,`, base64 of a PNG/JPEG/GIF/WebP header)
is caught at its first characters while the call streams
(`proxy._post_events_raw`), the upstream connection is closed at once, the
call is not forwarded, and the model gets it back NOT EXECUTED in a hidden
hop (the ledger replays it; it counts in `tool_turns`) with the remedy: the
image file's path, or yama_generate_image's link; after `IMAGE_REGENERATIONS` (2,
a choice) the turn lands. No other argument is read.
`x_yamadori.image_guard`; `mcp/test_image_guard.py`. **Not yet run live.**

## Tool descriptions are prompts

Rewriting one description moved first-call routing from 10.7/14 to 11.7/14.
`find_references` failed in every prompt variant while its description said
what the tool *did*. It won once the description did three things: led with
the question it answers, contrasted itself with the tool it was losing to,
and listed the phrasings that should trigger it. **These numbers are at
risk.** The eval sent `max_tokens: 400` to a thinking model and did not
record `finish_reason`. See `docs/CONSTRAINTS.md` #31, and re-run before
citing them.

Write descriptions as trigger conditions. When a tool is mis-selected, fix its
description before touching the system prompt.

**THE TOOL RECIPE (operator, 2026-09-29: "lock it in").** Evidence: the
PackageLens lookup probe (`bench/mcp/lookup_probe.py`,
`bench/mcp/results/lookup_probe.jsonl`, Pi, Bonsai at medium, the vague pagoda
prompt; n=4 valid trials, one prompt, no ablation): the model called our lookup
before installing in 4/4 and found the right names; `yama_think_deeply`,
offered for whole runs, was called 0 times voluntarily. Every server tool the
model sees follows it, and none is offered by default until its probe passes:
1. **Few tools, each answering one concrete question that moves the task
   forward** (which package is this / which versions exist / how is it used).
   Never an abstract "think harder" tool.
2. **The description is the prompt:** it opens with the question it answers
   ("Answers '...'"), then when to call it in the model's own situation, then
   "A server tool: it runs on the Yamadori server and does not touch your
   workspace." A vendor's own description is replaced, never passed through.
3. **Arguments that remove ambiguity** (PackageLens: `ecosystem` required).
4. **One positive system line** naming the tools and the moment ("... call X
   before you install or import it."). No prohibitions, no CRITICAL.
5. **A clean context:** the tool is judged with nothing else of ours
   injected.
6. **Useful, immediate results:** run as a hidden hop, plain text, screened
   as fetched data, the best answer first; a failure says what to call
   instead.
7. **A probe before it ships:** the call rate at the moment it is for, on
   this model, recorded like `lookup_probe.jsonl`.

## Failure returns carry the next step

A tool that fails without saying why causes retry loops. `find_by_pattern`
returned "no matches in 0 files" for an unindexed directory, and that
produced 14 consecutive retries with permuted arguments. Every failure path
returns three things: the situation, whether it is retryable (as a fact),
and a remedy with an owner. It names what IS available: near-miss symbols,
indexed roots, top-level directories.

Never add a prompt rule to suppress a loop a tool return is causing.

Each tool loop is capped per context at the vendor's 10 tool turns (PrismML
Bonsai-demo `agenticMaxTurns`), 20 at tier `max` (`tiers.tool_turn_limit`,
used by `proxy.py`; operator decision 2026-09-23). The main loop runs only
for our tools on main (the image tools, the MCP lookups, the craft tool).
Neither number is measured. At the cap the loop **lands**:
tools withdrawn, answer asked for, never a tool request returned as the
answer. **Landing is for OUR hops only** (hop 1+, after one of our tools
ran; `proxy.context_full`): the client's own request is never landed --
see "The client's own window" below. With the tools withdrawn the model
may write its next call AS TEXT (`<tool_call>`, `<function=...>`): it
reaches the client as written (the `_ToolMarkup` filter and its "[no
answer: ...]" notice were removed 2026-09-27, docs/CONSTANTS-AUDIT.md: built
from one run). A call of OURS asked for at the landing is withheld and
nothing is written in its place: empty content, finish_reason stop.
`x_yamadori.tool_turns` records `{limit, turns, hit}`. A hit alongside repeated
empty or error calls is a tool defect to fix, not a budget spent.

## One door to the model, one budget rule

- **Client requests** are shaped by `tiers.apply()` in `proxy.prepare`.
  **Internal generation** (`summarize_text`, the skills pipeline, the
  decider, the worker) goes through
  `mcp/model.py`, which calls the same `tiers.apply()`. Nothing else talks to
  llama-swap. Internal callers do not call `:1234`, because that would
  re-enter admission, pollute the corpus and need a key. `model.py` explains
  why.
- **Tokens** come from `tiers.budget()`. The client's `max_tokens` is an
  *answer* allowance, `answer = max(client, A_MIN)` with `A_MIN=2048`.
  Thinking is derived from the request's share of the KV pool
  (`mcp/budget.py`: one helper of a FIXED `HELPER_TOKENS` = 49,152 and main
  gets the rest of the pool -- at `-c 181248` q8_0, main 132,096; operator
  2026-09-25, "~48k second brain is the constraint", "the main model gets
  the share of the new tokens", a choice; `YAMADORI_HELPER_TOKENS=0` falls
  back to the 0.70/0.30 fractions; layout v2's cap layout is in
  `mcp/budget.py`):
  `thinking = max(share[role] // share_n - prompt - answer, 1024)`, then
  CAPPED (operator 2026-09-25, "this model overthinks"; choices, unmeasured
  for quality): a request routed `agent_step` thinks at most
  `tiers.AGENT_STEP_THINKING` = 6,144 (the by-result caps of #53 -- 2,048
  after a read, 6,144 after an error -- were REMOVED 2026-09-27,
  docs/CONSTANTS-AUDIT.md: chosen from one run), any other turn
  `tiers.USER_TURN_THINKING`; and an agent step's nudge is
  `tiers.AGENT_STEP_NUDGE_MESSAGE`, the operator's voice naming the action
  ("... Let me make the call I've already worked out -- the edit itself, if
  I know the fix -- and only keep thinking if that call would be
  misleading."; approved by the operator as written, AskUserQuestion
  2026-09-26, UNMEASURED WORDING; switch `step_nudge`; the general
  `NUDGE_MESSAGE` and `BUDGET_MESSAGE` are unchanged). A helper-role
  request (the skills pipeline's model stages) thinks at most
  `HELPER_THINKING` = 6,144 (1.5x its first value, operator 2026-09-25:
  "1.5 those numbers above and dial it in"). With the tier -> model table on
  (`mcp/tier_models.py`) the MODEL's token profile supplies the caps for the
  route the request names (`tiers.apply(..., turn=)`: "user_turn" /
  "agent_step"; `x_yamadori.progress.step.thinking_cap` is the cap applied).
  The cap is per REQUEST (`reasoning_budget_tokens`), not per slot.
  Evidence for capping: Octopus v0b-V0 (n=1 run) -- reasoning 96% of
  output, agent steps median 730 / p90 3,713 / max 24,620 completion
  tokens. A benchmark's `reasoning_cap` header still wins over every cap.
  Upstream gets `max_tokens = thinking + answer`,
  `reasoning_budget_tokens = thinking` and a `reasoning_budget_message`.
  The 0.70/0.30 split replaced the operator's 5/8 + 3/8 of 2026-09-22, not
  a measurement either, which replaced a 1/2 + 2 x 1/4 split the same day,
  which had replaced a fixed `R_CAP=8192` breaker and a 60/25/15 split that
  had no measurement behind it (`docs/CONSTRAINTS.md` item 19). Natural
  thinking ran 682–2,826 tokens at n=7 (§1), so no measured result says a
  shorter thought is better. `config.yaml` launches with
  `--reasoning-budget 32768` and the same message, the server default for a
  request that sends no budget of its own. (The second brain's job caps
  `JOB_THINKING`, its research nudge and its share of the helper lane went
  with it, 2026-09-29.)
- **A client's compaction is part of the conversation it summarises**
  (`mcp/compaction.py`, `proxy._serve_compaction`; `x_yamadori.utility_kind`
  and `x_yamadori.compaction`). Two shapes. *In place* (the history resent
  plus a summarise turn, as Claude Code and Codex send it: not a utility
  call, so it keeps its session, tools and slot) is served on the LEDGER's
  rendering of the resent history, which puts back everything the proxy
  added, and is checked against the stored prompt (the last prompt the proxy
  sent for that conversation, and the turn it delivered); only when they
  differ is the resent history spliced onto the stored prompt (mode
  `ledger` / `spliced`). *Flattened* (Hermes' one user message, "TURNS TO
  SUMMARIZE: [USER]: ..."; since 2026-09-26 also Pi's `<conversation>` /
  `# Conversation` summaries and OpenCode's "Here is the conversation so
  far: <conversation>", each read from the harness's source,
  `compaction.parse_transcript`; `x_yamadori.compaction.harness`) has its
  records mapped onto the stored prompt by
  text and its transcript replaced by a reference to that span (the
  iterative form's previous summary too; Pi's own summariser system prompt
  opens the instruction). No match means the request goes up
  as sent, with the reason recorded; a flattened one then falls back to the
  transient slot by prefix affinity (`slots.acquire` `prefix`). A compaction
  THINKS at the conversation's own effort -- at every effort, medium
  included -- with the conversation's own sampling; thinking is off only
  where the conversation itself runs with it off; one that maps onto
  nothing stored thinks at the effort the CLIENT sent (Pi sends its
  thinking level; it ran with thinking off before, recorded as "the
  conversation itself runs without it"), and off only when the client sent
  none. So the effort line is
  always the conversation's (interim defaults, operator 2026-09-24, until
  `docs/COMPACTION-RESEARCH.md`'s eval runs). Its thinking budget is what
  its own window leaves -- window - prompt - answer, never below
  `MIN_THINKING` (the one budget rule; `tiers.compaction_budget`
  `thinking_tokens`). `COMPACTION_THINKING` = 2,048 ("5 of the 7
  self-finished thinking runs", coding prompts, not a compaction
  measurement) was REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md).
  `tool_choice: "none"`, which llama-server
  does not render. The answer allowance is at least `COMPACTION_BUDGET`
  (5,120, the operator's figure; `YAMADORI_COMPACTION_BUDGET`) and otherwise
  the client's own target -- `max_tokens`, or Hermes' "Target ~N tokens" (up
  to ~12,192 at this pool; a cut summary makes Hermes discard it and compact
  again) -- bounded only by the pool (`tiers.compaction_budget`; the fixed 2x
  ceiling is gone). It never waits for the helper lane: its window is the
  pool less a running second brain's share (`admission.helper_active`),
  falling back to the main share when it does not fit. The advertised window
  stays the main share. **Not yet run live.** The mapping cannot be replayed
  from the corpus, which keeps 2,000 characters of each request (11 of its 12
  Hermes compactions are the iterative form, all preamble in that head).
- **A `finish_reason: length` is a budget event, never an answer.**
  `model.BudgetEvent`. To the client it is `finish_reason: "length"` with
  the content that was generated -- the "[no answer: ... token limit ...]"
  / "[answer cut off ...]" notice was REMOVED 2026-09-27 (it entered the
  client's stored history), and so was the "[no answer: the model stopped
  ...]" notice on a stop with nothing written (blank content, finish
  `stop`).
- **A model server that dies mid-generation is an ERROR, never an answer**
  (operator, 2026-10-05; `proxy._post_events_raw`, `_run_turn`). llama-swap
  answers 200 and closes the body; a stream with no finish chunk and nothing
  in hand is retried once, then 503; with reasoning or text in hand the turn
  RAISES (HTTP 503 `model_unavailable`, Retry-After 30, before the first
  byte; ONE `data: {"error": ...}` event and `[DONE]` after it) -- it was
  delivered as content plus a note with finish_reason "incomplete", which a
  harness read as the model's reply. Nothing is recorded as delivered. The
  request AFTER a kill: llama-swap answers 502 (connection refused) until it
  notices the exit (live 2026-10-05: two 502s 4 ms apart, then "upstream
  process exited unexpectedly"), so the retry waits for it
  (`proxy._await_swap_notice`: GET /running no longer lists the model ready,
  then llama-swap loads it) and a 502/504 that survives is a retryable 503
  `model_unavailable`, never a 502 `server_error`. `mcp/test_stream.py`;
  live: `bench/harness_soak.py --only k`.
- **The client's own window** (C1, `docs/OPENAI-CONFORMANCE.md`,
  2026-09-25). `/v1/models` `context_length` is the main share and it is THE
  LIMIT ENFORCED (`proxy.window_limit`; a compaction's is its budget
  window). Before anything runs, `proxy.check_client_prompt` measures the
  request as it would go up (the client's messages plus what the ledger and
  prepare add, and the tools): a high estimate first (chars / 3 plus half
  the extra UTF-8 bytes), and near the edge the model server's own count
  (`/apply-template` + `/tokenize`). A prompt that leaves less than the
  generation floor (`A_MIN` + `MIN_THINKING` when thinking: 3,072; a CHOICE)
  is refused: **400 `context_length_exceeded`** in OpenAI's wording ("This
  model's maximum context length is M tokens. However, your messages
  resulted in N tokens ..."), which Hermes matches and compacts on. The
  client's tools are never withdrawn and no user message is ever appended to
  its request; hop 0's `max_tokens` is cut to what the window leaves
  (`proxy.fit_window`). `x_yamadori.context` is the record.
- **One error path** (E1/E2, `mcp/api_errors.py`). Every failure is
  `{"error": {message, type, param, code}}`: 400 for bad JSON, a bad body and
  the fields we must refuse (`api_errors.validate_chat`: `n` > 1, `logprobs`,
  audio, `input_audio`/`file` parts, non-function tool forms -- the fields we
  ignore are listed there too), an upstream 4xx stays a 4xx (never 502: SDKs
  retry 5xx), a chat-template refusal (llama-server's 500 carrying "Jinja
  Exception: ...") is a 400 `invalid_prompt` with the template's own
  sentence (2026-09-26; `api_errors.template_refusal`: Pi retried the
  deterministic 502 three times), 404 for an unknown `/v1` route, 503 for an
  unreachable or loading model server. STREAMED: `server.chat` pulls the stream's first
  chunk before it commits the 200 (`proxy.stream_body` yields nothing until
  the turn's first event -- the first upstream generation answered, or deep
  thinking's first heartbeat), so a failure before it is a real HTTP error;
  after it, ONE `data: {"error": ...}` event and `[DONE]` -- never assistant
  content. A client that hangs up during the wait cancels the token.
- **The proxy's three reliability fixes of 2026-10-02** (commit 6d8007f; from the operator's live session of
  2026-10-01 16:53-20:25 ET, VS Copilot agents at max; offline only -- test_stream 112, test_ledger 144,
  test_responses_api 102, test_messages_api 88, test_one_conversation 55, test_slots 123 -- **not yet run live**).
  THE WAIT IS HEARD (`proxy._TurnPump`): the stream said nothing until the turn's first event, and a max request's
  first event waits for the Flash-Next load (131-272 s measured) and then its prompt (a cold 30K prompt: 135-321
  s); the client hung up at exactly 180.0 s with no byte received (17:28, again 20:25) and each hang-up cancelled
  the load or prefill it waited on. The turn now runs on a thread of its own (it advances only when the stream
  pulls, so the ledger's ordering holds) and whenever it has produced nothing for `HEARTBEAT` (5 s, the proxy's
  existing beat, under the shortest client silence limit known: Hermes' codex transport, 12 s) the stream sends
  a heartbeat -- the empty delta, `response.in_progress` on Responses, `ping` on Messages. This amends "nothing
  until the turn's first event" above: E1 keeps its promise for fast failures -- a refusal inside the first
  `HEARTBEAT` seconds (validation, the window check, 503 at capacity: all decided before any model work) is
  still a real HTTP status; one after the first heartbeat is the committed stream's one error event.
  AN UNANSWERED ATTEMPT HOLDS NOTHING (`mcp/slots.py`): the first request's client hung up 180 s into the load,
  its retries arrived as NEW conversation ids (a retried opening mints another) and were refused 503
  `conversation_at_capacity` for 60 s, held off by the cancelled attempt's own pin ("its latest activity was 4 s
  ago"). A conversation becomes the card's owner when a generation of it FINISHES (`slots.remember`); a pin made
  by an attempt that ended with none -- cancelled, refused, failed -- is dropped at release with its activity
  stamps (the grant says `unanswered`; the slot's end is not stamped), so the retry finds the card free. A conversation answered before
  (or restored from a persisted pin) keeps its pin whatever happens to a later request. `budget.pool_size` NEVER
  CACHES ITS FALLBACK: a proxy that started before the model server answered kept the 131,072 fallback and no
  slot count for its whole life (the deploy check of 2026-10-01: `/v1/models` said 131072 and requests were
  ranked for 3 slots while bonsai served 209,920 cells on 1 slot, so ONE CONVERSATION was off). The fallback is
  returned uncached, and the next call asks again, at most once per `budget.RETRY_S` (5 s, the heartbeat) while
  the server is away, so a starting server is noticed within one beat.
- **Flash-Next's first prompt: the expert file is PRE-READ into the OS file cache** (2026-10-06; `mcp/preread.py`,
  wired in `max_mode.wait_ready`, tier-table row key `preread` on flash-next; `mcp/test_preread.py` 50 checks, offline
  offline; run live through :1234 on 2026-10-06 in two rounds, results below, the no-room skip offline only). Evidence (`bench/fn_first_prompt.py`, `bench/results/fn_first_prompt/20261006-a`, n=3 per
  arm, n=1 for the idle arm, requests sent to llama-swap :11434, ~24.6K-token fresh prompts): the experts (IQ2_XS shard 1,
  39.2 GB, mmapped, unpinned: `LLAMA_PIN_EXPERTS=0`) are demand-paged from disk during the first 8,192-token batch of the
  first prompt after a load -- from a cold file cache 31.5-31.9 GB read at ~0.4 GB/s, batch 1 75-87 s against 16.2-16.9 s
  warm, ttft 201-211 s. A sequential read of shard 1 with 16 MiB cached reads before the load took 17.0-17.4 s (2.3 GB/s),
  left 42.5-44.3 GB available (evictable cache) and cut the prefill's disk reads to 1.6-3.8 GB and ttft to 83-103 s.
  Shard 2 (26.8 GB, the n-gram table) is NOT read: both together (63.4 GB) exceed what the cache holds and push shard 1
  out. An idle loaded server (~4 h, n=1) had its working set trimmed to 67 MB and its next prompt read 24.6 GB from disk
  (ttft 99.6 s): the penalty also follows a long idle. WHAT THE PROXY DOES: (1) on a flash-next SWAP, a background thread
  reads the file named by the `-m` of flash-next's llama-swap entry (config.yaml, macros expanded; never a path in code)
  starting as the swap request goes out (`preread_overlap`, default; off = read, then load), and the request's prefill
  waits for it under `card_wait`, so the stream's heartbeats keep flowing (`_TurnPump`); whether the overlap helps or
  contends for the disk with the load is UNMEASURED, the live check measures both; (2) on a request for the LOADED
  model, when the llama-server process's working set (psutil rss = Windows WorkingSetSize, the process found by the
  model file in its command line) is below HALF the file's size: the measured states fall in two groups with a wide
  gap -- trimmed 67 MB (4 h idle, n=1) and ~900 MB (an idle server, `preread.py status`), warm 39.1-44.4 GB after a
  prompt (fn_first_prompt 20261006-a) -- and half the file (~19.6 GB) sits inside that gap. "Below the file's size"
  (the first build) put the warm 39.1 GB on the trimmed side and would have re-read 39 GB on ordinary agent turns
  (coordinator, 2026-10-06). A pure continuation cannot be told from a first prompt before it is sent. One read at a time (a lock); a read in flight is
  joined, never restarted; bounded by 900 s (`max_mode.LOAD_TIMEOUT_S`, llama-swap's health-check timeout), then the
  request goes on (`timed_out`). THE READ RUNS AT NORMAL MEMORY AND I/O PRIORITY (found 2026-10-06, `bench/results/fn_first_prompt/20261006-proxy`, n=3 per
  arm through :1234 at tier max): the stack's processes run at memory priority 2 / low I/O priority (the bench's shell at 5 /
  normal), a read thread inherited that, and the load's ~40 GB of private memory consumed exactly the pages it had cached --
  the prefill still read 31-33 GB in 6 of 6 swaps (ttft 165-174 s, the same as no read, 163-167 s). Setting the thread's
  memory priority to 5 and the file's I/O hint to normal (`raise_cache_priority`) gave prefill disk reads of 1.7-1.8 GB, batch 1
  24-26 s and ttft 91-99 s (`20261006-proxy-b`, 48 GB available, n=2; the cause is inferred from the priorities and this
  result). THE CACHE MUST HAVE ROOM FOR THE WHOLE FILE (`decide_room`): with ~38 GB available (another model's host memory,
  imagegen-turbo, was resident) a read evicts its own head -- read-then-load ttft 191-205 s, overlapped 240 s, trimmed-server
  reads (33-35 GB available) 26 s for nothing, against 163-167 s with no read -- so the read is skipped, with
  `available_mb` recorded, when the memory available (free + standby) is below the file's size; the margin between 39 and 47 GB
  is unmeasured. Windows' background thread mode is not used: it would lower the cached pages' memory priority. A missing file or
  an unreadable working set records its reason and the request goes on. Recorded: `x_yamadori.preread` {why swap |
  trimmed, model, file, file_bytes, bytes, ms, gb_per_s, overlapped, working_set_mb, joined, waited_ms, ok, timed_out,
  error | skipped} and one log line. Switches (default ON): `YAMADORI_PREREAD` / X-Yamadori-Features `{"preread":
  false}` and `YAMADORI_PREREAD_OVERLAP` / `{"preread_overlap": false}`. `python mcp/preread.py status` is a read-only
  look at the file, its size and the server's working set.
- **Usage** (U1/U2). `usage` is the FINAL main generation's (the last hop,
  or a fold-back continuation): `prompt_tokens` its prompt -- the context the
  conversation occupies, which Hermes compacts on; it was the SUM over hops --
  `completion_tokens` what it generated, `prompt_tokens_details.cached_tokens`
  and `completion_tokens_details.reasoning_tokens` (llama-server does not
  report reasoning tokens: counted with its `/tokenize`, absent when that
  fails). Hidden hops have no spec field; their numbers and the old sums are
  in `x_yamadori.usage` (`usage.hops` moved there). Streamed, usage is sent
  only with `stream_options.include_usage`, as its own last chunk with
  `choices: []`.

## The Responses API (operator, 2026-09-26)

`POST /v1/responses` is a TRANSLATION over the one turn, never a second
pipeline (`mcp/responses_api.py`; decisions and evidence in
`docs/OPENAI-CONFORMANCE.md` "Status: R1"). `to_chat` turns the request
into the chat body `_run_turn` takes; `server._serve_turn` serves both
routes (admission, supersede, the E1 commit point, cancel on disconnect);
`of_chat` / `Stream` turn the result back (the Stream reads
`proxy.stream_body`'s own chunks, so every stream promise above holds).
Anything built for chat reaches Responses by construction; a feature that
touches the request or the wire must be checked in both
(`mcp/test_responses_api.py` replays a Codex-shaped client through the
served template). The rules that are Responses-only:

- **Stored responses** (operator, 2026-10-06, AskUserQuestion, chosen "Store,
  local, capped": "Save each response's full input and output under its id,
  per account, on this machine only. Same size and age limits as the ledger
  (256 MB per account, 2 GB total, 30 days). previous_response_id rebuilds the
  conversation from it, and GET and DELETE /v1/responses/{id} work like
  OpenAI's. Applies when the client asks to store (OpenAI's default). This ends
  the rule that client messages are only hashed, for Responses requests.").
  Before it the route was stateless and refused `previous_response_id`, a
  production trap: OpenAI's `store` defaults to true and a client may send only
  the new input plus `previous_response_id`. `mcp/response_store.py` (sqlite,
  `index/responses.sqlite3`, `YAMADORI_RESPONSES_DB`; schema in its module
  doc): per response id and account, the request's own input items, the output
  items, the Response object less `output` and `x_yamadori`, the id it chained
  from and its chain root, and its size; no key is stored. `store` absent or
  true stores the response when it ends (blocking: before the body returns;
  streamed: before the terminal event, so a client that chains the moment it
  sees `response.completed` finds it; a stream the client abandons is not
  stored); `store: false` stores nothing, as before; the object's `store`
  field reports the truth (a response larger than the per-account cap is not
  stored and says false). `previous_response_id` rebuilds the history -- the
  chain's input then output items, root first, then this request's `input` --
  and runs the one translation on it, so the chat messages, the ledger's keys,
  the session (a continuation that sends no `prompt_cache_key` keeps the
  chain's) and the slot's prompt cache are those of a client that resent
  everything (`mcp/test_responses_api.py` compares the messages and runs the
  served template). `instructions` and `tools` are NOT carried over (OpenAI:
  "When using along with previous_response_id, the instructions from a
  previous response will not be carried over to the next response"); an id
  that is another account's, stored with `store: false`, deleted, evicted or
  aged out is 400 `previous_response_not_found` (param `previous_response_id`,
  "Previous response with id '<id>' not found."; OpenAI's status, type, code
  and message as its clients report them, github.com/dotnet/extensions/issues/7704,
  github.com/microsoft/semantic-kernel/issues/13128; the reference does not
  print the body). `GET /v1/responses/{id}` returns the stored object (an
  image's base64 is put back from the media store while it is there),
  `DELETE` answers `{id, object: "response.deleted", deleted: true}` (the
  response only; a chain through it is then "not found"),
  `GET /v1/responses/{id}/input_items` lists that request's own input items
  (`order` default desc, `limit` 1-100 default 20, `after`); another
  account's id is 404 on all three, exactly as an unknown one
  (`Response with id '<id>' not found.`), `?stream=true` on GET is 400. The
  caps are the ledger's, separately metered (`YAMADORI_RESPONSES_ACCOUNT_MB`
  256, `_TOTAL_MB` 2048, `_MAX_DAYS` 30, defaulting to `nebari.LEDGER_*`):
  whole CHAINS evicted least recently used first, then by age, by a
  background thread at most once a minute. `YAMADORI_RESPONSE_STORE=0` turns
  the store off (every response `store: false`, `previous_response_id` 400).
  Still 400 `unsupported_parameter`: `conversation`, `prompt`, `background:
  true`, `item_reference` items (they would need an index of every item id; no
  harness we serve sends one). Records: `x_yamadori.responses.stored` {id,
  chained_from, items, bytes} and `previous_response` {id, chain, items}, a
  log line per store and delete. Privacy: local, the account's own
  conversation; `corpus.py` and the learning paths never import the store
  (gated by `test_nothing_that_learns_reads_the_store`). The session is
  `prompt_cache_key` (Codex always sends one), else the headers, else our id
  in the `call_id`s. **Live: written 2026-10-06, not yet run**
  (`test_live_stack.py --only responses_features`, `test_responses_stored_state`).
- **Items -> messages, deterministically**: ONE system message
  (instructions + leading system/developer items; the template refuses a
  later one -- those become user messages); one assistant message per turn
  (reasoning, text and calls of a turn merge; ignored hosted-call records do
  not split it); `function_call_output` arrays keep their images for the
  chat image path (`chat_part_of` is the one Responses-part mapping).
- **Reasoning**: shown as a `summary` (`YAMADORI_RESPONSES_REASONING`:
  `summary` | `content` | `off`); on input only `content[].reasoning_text`
  reaches the model -- summaries and `encrypted_content` never do.
- **Tools**: function and custom tools served; the hosted
  `image_generation` tool IS our `yama_generate_image` (its size wins; the image
  comes back as an `image_generation_call` item, streamed the moment it
  exists, no invented partials); other hosted tools ignored and recorded;
  client-run built-ins we cannot render are 400.
- **Keep-alive is a real event** (`response.in_progress`, compact, at most
  1/s): Hermes' codex transport times out after 12 s without a parsed
  event, and SSE comments never reach its timer.
- **Errors**: before the first byte, the HTTP status and the error object
  (as chat); after it, `response.failed` with Codex's codes. `length` is
  `response.incomplete` (`max_output_tokens`).
- **Coverage: every feature of the chat wire runs through Responses, offline
  and live** (operator, 2026-10-06, his VS Code agents use this wire, often at
  max: "ensure we have sound responses api coverage and our image generation,
  vision, and all other api's work with responses api too"). Offline
  `mcp/test_responses_api.py` (175 checks, 2026-10-06): the wait heard
  (`response.in_progress` beats through a swap and a cold prefill), a killed
  model server (`response.failed`, or 503 before the first byte), the window
  (HTTP 400 before any byte), titles and classifiers, both compaction shapes,
  the concept seed, the one-conversation rule and the other card (503 +
  Retry-After; `x_yamadori.slots.routed`), every effort a tier and a refused
  tier a 503, the package tools, `yama_describe_image` as a hidden hop (an
  `input_image` and a tool output that carries one), drawing without the
  hosted tool, closing the stream cancelling the turn, reasoning `off`. Live
  (`mcp/test_live_stack.py` groups `responses`, `responses_features`,
  `responses_compaction`, `responses_slots`, `responses_tiers`, 2026-10-06, all
  passing): the same on the real models, the tier walk with its swaps heard,
  a lower tier served on bonsai-a4000 while max works, xhigh refused 503; and
  `bench/harness_soak.py --api responses` (scenario i, and `--only k`: a killed
  model is `response.failed`). Not covered live: the package tools while
  Docker is down (NOT APPLICABLE, recorded), `length` ->
  `response.incomplete` and the `content` reasoning mode (offline only).
  Rows R1-R17 of docs/LIVE-COVERAGE.md.

## The Messages API (operator, 2026-09-29)

"Yes add anthropic api endpoints for claude". `POST /v1/messages` and `POST
/v1/messages/count_tokens` are a TRANSLATION over the one turn, like
Responses (`mcp/messages_api.py`: `to_chat` -> `server._serve_turn` ->
`of_chat` / `Stream`; the mapping tables are its module doc; decisions and
evidence in `docs/ANTHROPIC-CONFORMANCE.md`). Claude Code reaches it with
`ANTHROPIC_BASE_URL` = the public base (it posts `/v1/messages?beta=true`);
the harness kit's Claude Code entries export the setup. **Offline only**
(`mcp/test_messages_api.py`, through the served template); **no real Claude
Code run yet** -- every claim about Claude Code's behaviour is read from its
docs (code.claude.com, 2026-09-29), unmeasured. The rules that are
Messages-only:

- **Auth**: an account key as `Authorization: Bearer`
  (`ANTHROPIC_AUTH_TOKEN`) or `x-api-key` (`ANTHROPIC_API_KEY`)
  (`server._identify_anthropic`). `anthropic-version` / `anthropic-beta` are
  accepted and recorded (`x_yamadori.messages`).
- **Effort -> tier, deterministic**: `output_config.effort` names the tier
  (low/medium/high/xhigh/max: the same scale); else `thinking` disabled or
  between_tools -> `minimal`; else the default (or a `yamadori-*` name's
  tier). A thinking budget picks nothing (the model's profile owns it). The
  tier picks the model (`mcp/tier_models.yaml`). Recorded with its rule in
  `x_yamadori.messages.effort` {tier, serves}.
- **Sessions**: Claude Code's `x-claude-code-session-id` (+
  `x-claude-code-agent-id` for a subagent) is sent as `prompt_cache_key`;
  our `X-Yamadori-Session` wins; else `metadata.user_id`'s session; else our
  id in the `tool_use` ids (kept verbatim, as chat's call ids).
- **Thinking**: returned as `thinking` blocks signed with an HMAC of the
  account and the text (a key derived from the media-link secret); an echoed
  block whose signature verifies is the turn's reasoning (an echo is kept as
  sent); one that does not is DROPPED and the ledger restores the slot's own.
- **Side calls**: with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` (the kit sets
  it) an `x-claude-code-request-class: auxiliary` request with no tools is a
  utility call (X-Yamadori-Features `utility`, the caller's header wins).
- **tool_choice** in what llama-server serves (auto/none/required): `any`
  -> required, a named `tool` -> the tools narrowed to it + required.
  Server tools (web_search_*, advisor_*, ...) ignored and recorded;
  Anthropic-schema client tools (bash_*, text_editor_*, ...) 400.
- **Usage**: `input_tokens` = the prompt LESS the reused cache,
  `cache_read_input_tokens` = chat's `cached_tokens`, `output_tokens` =
  `completion_tokens` (the final generation's, U1). `stop_sequence` is
  never claimed: llama-server's chat route does not say which stop fired.
- **The stream**: Anthropic's grammar (message_start, content_block_*,
  thinking/signature/text/input_json deltas, message_delta, message_stop);
  a `ping` per chat heartbeat (at most 1/s) and, from a pump thread, one
  whenever the turn has sent nothing for `proxy.HEARTBEAT` s (Claude Code
  aborts a stream silent for 300 s). Nothing before the turn's first chunk
  (E1).
- **Errors**: `{type: "error", error: {type, message}, request_id}`; capacity
  (429 server_busy, 503 model_at_capacity / model_unavailable) is
  `overloaded_error` with Retry-After; a prompt past the window is 400
  "prompt is too long: N tokens > M maximum" (M = the window less the
  generation floor: `api_errors.context_length_exceeded` now carries
  `floor`), the wording Claude Code compacts on. After the first byte: one
  `error` event.
- **count_tokens**: the model server's own count of the client's messages
  and tools (`proxy.count_prompt_tokens`) on the tier's model when it is on
  the card; else the window check's high estimate, labelled. Nothing is
  decided or recorded.

## The Jev API (operator, 2026-09-29)

"Also want the jjava api exact public api endpoints that match Jev exposed
through our proxy." jjava (docs/JJAVA.md: `mcp/decider_bonsai.py` typed/2)
answers TypeSafe's public Jev API exactly (`mcp/jev_api.py`, routes in
`mcp/server.py`; every field, limit and difference with its source in
docs/JEV-CONFORMANCE.md). A TRANSLATION like the two above, but NOT over the
turn: `mcp/proxy.py` is not involved -- a Jev call is typed questions on the
decider lane, not a chat turn. **Offline only** (`mcp/test_jev_api.py`, 190
checks: all 13 API response examples in docs.typesafe.ai reproduced field
for field through a fake model server behind decider_bonsai's doors); **no
live run, no TypeSafe SDK run yet**. The rules:

- **Paths**: `POST /jev/v1/systemone` and `GET /jev/v1/models` (a TypeSafe
  SDK takes `base_url = "<public base>/jev"`: our `GET /v1/models` stays
  OpenAI's), and `POST /v1/systemone` at the root. Auth: an account key as
  `Authorization: Bearer`. Every response carries `x-typesafe-request-id`.
- **Models**: `jjava-latest` = whichever main model holds the card
  (`max_mode.decide(None, utility=True)`, never a swap); `jjava-bonsai`,
  `jjava-mirai-s`, `jjava-flash-next` only while that model is on the card,
  else 529 with Retry-After; one this stack does not configure is 422.
  Jev's `jev-latest`, `jev-preview`, `jev-1.13.0` are aliases of
  `jjava-latest` (recorded). The response's `model` is the one that read it.
  The models list is TypeSafe's `{models: [{name, description,
  release_date}]}`; availability and whether the model's priors are
  measured (`bench/decider/results/models/<model>.json`) in `x_yamadori`.
- **Answers carry ONLY Jev's fields** (noul `{type, noul}`; choice `{type,
  choice, probabilities, confidence}`; score `{type, score, legend,
  probabilities, confidence}`, the legend the criteria as given); jjava's
  diagnostics per question in a top-level `x_yamadori` (clients ignore it).
- **Jev's limits, enforced**: Choice up to 255 options -- more than 26 (one
  single-token letter each, the decider's own limit) in TWO STAGES, never
  refused: positional chunks of <= 26, one read each, then one over the
  winners; P(option) = P_final(its chunk's winner) x P_chunk(option);
  `x_yamadori.rounds` says so (the composite's accuracy is UNMEASURED).
  Score 2-10 levels (decider_bonsai allows 26; clamped). State plus the
  longest question <= min(32,768, the model's window) tokens and 64k per
  request, counted by the model server's `/tokenize` ("32k"/"64k" read as
  binary k); past it, 422 on the field.
- **Capacity**: the questions run on THE LANE (the child slot, rank 3,
  `budget.LANE_TOKENS` 3,072 cells), one after another on the cached state
  (Jev: in parallel; `x_yamadori.parallel` false); a state larger than the
  lane runs past the VRAM line, recorded in `x_yamadori.lane`. ONE Jev call
  runs at a time and the others QUEUE in arrival order (THE QUEUE, `mcp/jev_api.py`; operator, 2026-10-01: "Can we
  queue jev calls, don't they only take a few seconds, like queue a set number, and send retry estimate if busy?"): a
  call is admitted while its estimated finish (the lane work ahead plus its own) is inside `CLIENT_TIMEOUT_S` = 10 s,
  the TypeSafe SDKs' own request timeout (typesafe-sdk 0.7.2 `DEFAULT_TIMEOUT`; derived, not chosen), else 429 at
  once with Retry-After = the estimated seconds of work ahead (ceil, at least 1) -- never a hang. The estimate is
  the median seconds per question of this process's last 50 calls, seeded with decide_turn's measured p90 for a
  question's two reads, 577 ms (`bench/decider/bonsai_decider.py`); `x_yamadori.queue` records the wait. Offline
  only (`mcp/test_jev_api.py` 191 checks, `mcp/test_jev_sdk.py` 106 with both SDKs, 2026-10-02); not run live. A
  call holds a `max_mode.Lease` on its model while it runs.
- **Usage**, counted from the reads (decider_bonsai's per-read timings):
  `input_tokens` = the first read's cached prefix + every read's processed
  tokens; `output_tokens` = the reads (one token each).
- **Errors** in Jev's statuses, the body its SDK reads (`{"detail": ...}`,
  a 422's detail a list of `{loc, msg, type}` rooted at "body", every
  offending field at once): 401, 422, 429, 529 (a model off the card, the
  model server down: Retry-After), 500 ours, 503 jjava off
  (`YAMADORI_DECIDER=0`).
- **Records**: one corpus event per call of kind `jev_call` (never `turn`),
  with the account and its traffic class, so live-suite calls never train
  anything; a sha1 of the state, never its text.
- **SDK conformance, planned (a download: the operator's approval)**:
  `typesafe-sdk==0.7.2` and `@typesafe-ai/sdk@0.6.0` pointed at
  `<base>/jev` (docs/JEV-CONFORMANCE.md section 6).

## The A4000's room

`mcp/gpu_room.py` decides what is loaded on the A4000 (operator,
2026-09-24: "if it fits with headroom fine, if it doesn't drop them and load
in what you need on use"). Every caller about to make llama-swap load an
A4000 model wraps that request in `gpu_room.use(model, upstream=...)`:
`images.generate`, `model.post` (vision), `code_search._post` (embeddings),
`scripts/index_code.py`. It reads `/running` and nvidia-smi by
UUID, sizes the model from `SIZES` (each row names its source; only
`imagegen` is measured), and if free − need < 1,331 MiB it unloads other
A4000 models least recently used first through `/api/models/unload`, or
refuses with `A4000_NO_ROOM` / `A4000_BUSY`. The main model is never
touched; a model in use (a lease) is never unloaded. A file lock
serialises it across the proxy, tools API and worker. `x_yamadori.gpu_room`
records every decision. llama-swap's groups remain the backstop. A new A4000
model needs a `SIZES` row (`mcp/test_gpu_room.py` checks the table against
config.yaml). Offline suites run with `YAMADORI_GPU_ROOM=0`. **Not yet run
live** (SELF-IMPROVEMENT-LOG #16).

## Prompting this model

- A decision-router table beats prose. Measured 10.7 vs 10.0.
- Prohibitions degrade monotonically: 0 `never` (10.7) > 2 (10.0) > 6 (9.3).
  A negative instruction fires attention on the thing it forbids. Keep at most
  two, each naming a specific observed failure. (Same eval, same caveat as
  above.)
- **The one sanctioned prohibition: "do not repeat"** (operator, 2026-09-29:
  "do not repeat is a good exemption to add a rule for that helps steer a
  session"). The no-progress guard (`mcp/progress_guard.py`, from Atomic
  Agent / OpenClaw; docs/research/ATOMIC-AGENT.md) may tell the model, on the
  tool result of an identical repeated call, not to repeat it, plus what to do
  instead. It names an observed failure: 8 repeated-outcome episodes in our
  logs (e.g. `search_files` returning 0 results 9 times; `vision_analyze`
  failing identically 5 times; `bench/atomic/replay_logs.py`, 1,564 calls, 29
  runs). No other prohibition is added with it.
- `reasoning_effort` values are **read from the served chat template**
  (`tiers.accepted_efforts()`). Today they are `low`, `medium` and `xhigh`.
  Any other value makes the template return an HTTP 500. `safe_effort` rounds
  up: `high` becomes `xhigh`, `minimal` becomes `low`, and `max` becomes
  `xhigh`. The default is `medium` (`config.yaml`).
- Tier ladder (`tiers.TIERS`): the client's `reasoning_effort` picks a tier,
  and each tier sets what is *allowed* (the selection engine still decides per
  request whether skills run). The same table is
  in `README.md` under "Effort tiers". Both are generated from
  `tiers.features` (the one feature matrix, which the dashboard's tier
  ladder also reads); `mcp/test_tier_docs.py` fails when a copy disagrees.
  SKILLS ARE OFF AT EVERY TIER (operator, 2026-09-29: "Stop skills until
  we have a good skill injector." -- "Skills are still valuable we just
  haven't found the unlock yet. TBD."): `tiers.TIERS` `skills` is False
  everywhere, so no skill body, recall line, craft index or
  `yama_recall_craft` reaches the model; the skills code (pipeline and
  serving) stays, and a header forcing `{"skills": true}` still reaches
  it.

  | `reasoning_effort` | thinking sent | skills | MCP tools | images | concept seed | adds |
  |---|---|---|---|---|---|---|
  | `minimal` | off | – | – | yes | – | thinking off, the vendor's instruct sampling, nothing of ours (fastest; least injection-resistant) |
  | `low` | medium | – | – | yes | – | nothing: the model as it ships, the benchmark baseline |
  | `medium` | medium | – | yes | yes | – | the MCP host's package lookups (skills: off at every tier, 2026-09-29) |
  | `high` | medium | – | yes | yes | yes | a concept seed on the conversation's first user turn |
  | `xhigh` | medium | – | yes | yes | yes | everything `max` has, at medium thinking: the effort-matched pair to `max` |
  | `max` | xhigh | – | yes | yes | yes | everything, at xhigh thinking |
  `MCP tools`: the MCP host's package lookups (switch `mcp_tools`).
  `images`: `yama_generate_image` / `yama_describe_image` are offered on
  every tier where `YAMADORI_IMAGEGEN_URL` is set. `concept seed`: drawn
  once and appended to the conversation's first user turn (the tier flag
  `seed`). Library help, the code check, fan-out, deep thinking and the
  addendum -- the columns this table had until 2026-09-29 -- were removed
  (docs/REMOVED.md); `high` and `xhigh` now differ from `medium` by the seed
  only, and `xhigh` from `max` by the effort sent. Because the tiers also
  raise thinking effort, a `minimal` vs `max` comparison mixes effort with
  what we add; benchmarks report an effort-matched pair (everything forced
  off vs on at the same effort) beside the ladder.
- Effort does not order thinking length at the measured n
  (`docs/CONSTRAINTS.md` §1b). Do not derive token numbers from effort.

## Claims carry their evidence

Every threshold, default and cut names the script that justifies it and the n
it was measured at. A number from `n=11` is labelled as such. Three.js results
are labelled contaminated: it is in every training set.

**A number or rule exists only if it is (A) the operator's decision, quoted
with its date, (B) derived from a real constraint (VRAM, the window, an
engine's, protocol's or harness's limit, security, correctness) with the
derivation beside it, or (C) measured (script + n).** "A CHOICE,
unmeasured" is not a fourth class (operator, 2026-09-27: "You made that
limit up, unfounded."). `docs/CONSTANTS-AUDIT.md` is the inventory: every
constant in `mcp/`, its class and evidence, what was removed, and what is
pending for whom.

If a measurement does not survive a repeat, it is not a result. This repo has a
history of single-run conclusions that reversed.

**`docs/PROTOCOL.md` enforces this.** It has sixteen rules, and each one is
attached to a specific failure in this repo. Read it before changing a
default, cutting a component, or reporting a result.

## Before you claim anything works

1. `python scripts/run_tests.py`. This runs every offline suite plus
   `ruff --select=E9,F`. Use the stack interpreter,
   `C:\Users\jwals\textgen\installer_files\env\python.exe`. A suite that
   prints no count is a failure. An offline suite that reaches a stack port
   OR writes the live state (anything under index/ or logs/; a write
   statement on a live sqlite database) fails, with the frames that asked
   (scripts/offline_guard/sitecustomize.py); a before/after snapshot of the
   live files backs it up for non-Python writers (run_tests.py "OFFLINE
   SUITES NEVER WRITE THE LIVE STATE"). A new suite points every store it
   touches at a temp path BEFORE importing (mcp/offline_stores.py).
2. Then run the live suite **through `:1234`**:
   `python scripts/run_tests.py --live --key-file PATH` (or
   `YAMADORI_TEST_KEY=...`). This covers `mcp/test_live_stack.py` (every
   claimed feature; the map is `docs/LIVE-COVERAGE.md`),
   `mcp/test_tools_live.py` and the
   bench suites that take `--live`. It judges the model's actual output
   through the door users use, prints every check's evidence, and treats a
   429 as NOT RUN: exit 1 on any failure, 3 when nothing failed but
   something did not run, 0 only when everything ran and passed. **If we
   don't have tests that exercise the real models, we don't have tests**
   (operator, 2026-09-24): a feature with no live test is unproven, however
   green the offline suites are.
   **A deploy is not good until `python scripts/deploy_check.py --key-file
   PATH` exits 0 after the restart.** It waits for every service and an idle
   card, runs the live suites (`run_tests.py --live --live-only`), and
   appends the verdict to `logs/deploy_check.jsonl`. Exit 3 (a 429) is not
   good either: run it again on an idle stack. The intrusive live tests
   (they restart the proxy) run only with `--maintenance`.
3. **Never test around the stack.** Stubs and direct calls to `:11434`
   have declared whole systems dead when the real cause was a setting (a
   budget, a hop cap, a timeout).
4. **One GPU consumer at a time.** Two consumers on one card degrade each
   other into 429s and 502s, and the loser looks like the one with the bug.
   Queue benchmark runs through `bench/queue_runner.py`. Treat a 429 as
   "not run", never as a failure.
5. The watchdog (`scripts/watchdog.ps1`) restarts a service only on the
   **second consecutive** failed `/health`, 5 minutes apart. Before it
   restarts llama-swap, it reads the slot counters twice, 10 s apart. If
   they are moving, the stack is generating and is not restarted. A slow
   answer is not an outage.

## What has been cut, and why

- **2026-10-01: the reranker** (operator: "Remove reranker"): the
  `reranker` model, its `retrieval` group membership, `code_search.rerank`
  and its bench arms. `docs/REMOVED.md` "Removed 2026-10-01"; the way back
  is commit `e360d37`.
- **2026-09-29: the second brain and everything around it** -- deep
  thinking, fan-out, the fix-up repair and its notes, the addendum, the
  delegate arm, Laya, CLM, selection's legacy path, library definitions and
  LIBRARY USE. `docs/REMOVED.md` has each one's evidence; the way back is
  commit `e360d37`.
- `judge` (Laya as a truth judge) scored 6/10 against a 5/10 coin flip, with
  systematic false positives on negative cases. It scores what a passage is
  *about*, not whether a proposition holds. (Its gate, `scripts/eval_judge.py`,
  went with Laya, 2026-09-29.)
- `apply_edit` was cut because the harness owns writing. There is one write
  path per repo.

**The reranker was REMOVED 2026-10-01** (operator: "Remove reranker";
`docs/REMOVED.md` "Removed 2026-10-01"; the way back is commit `e360d37`).
It had been held "not cut, not trusted, and not used" since `docs/FINDINGS.md`
#20 found `/v1/rerank` scores depend on batch composition (15–16/89 batched
vs 68/89 one at a time), which voided every reranker number in the repo.
Semantic search returns embedding (cosine) order; the A4000 gets its ~2.9 GB
back.

## E1

**E1 replaced Laya** (`mcp/e1.py`, `docs/E1.md`): logistic heads on the
resident embedder's vector of the request. route_in v1 scored 104/120 on
held-out set 1 vs Laya's 80 (p=0.0001) and 110/141 on set 2 vs 86
(p=0.003); no Laya combination beat E1 alone. Laya -- its service, heads,
trainer, dataset kind and harnesses -- was REMOVED 2026-09-29
(docs/REMOVED.md; the rules for reading its scores are in git at
`e360d37`). E1 is HELD (operator, 2026-09-29: until the llama-based
"jjava" decider rewrite gets a full chance): `YAMADORI_E1=1`, off by
default; its `skill_applies` head serves the skills; `route_in` has no
caller since selection's legacy path went, and the `escalate` head and the
deep_decisions label source went with deep thinking. Its idle-time
learning was scheduled by deep_learn, which is gone.

## Where things live

- **Datasets.** `mcp/datasets.py` holds the stages and `mcp/jobs.py` is the
  durable queue. The lane limit is global across processes
  (`{gpu: 1, cpu: 4, net: 4}`). `mcp/worker.py` claims and runs the jobs.
  There is no review stage: review is optional and happens after the fact on
  the dashboard. The `clarify` stage is model-assisted. A licence is filled only
  from a verified verbatim quote, and a guessed licence is never proposed.
  A job may WAIT without spending an attempt (`jobs.defer`: back to
  `queued` with `not_before`; `claim()` skips it until then; 2026-09-27): a
  job whose payload says `"idle": true` is checked by `worker.run_one`
  BEFORE its handler against `mcp/idle.py`'s one definition of an idle
  stack -- no client request for `skill_learn.IDLE_MINUTES` (15; operator,
  2026-09-27: keep ours, no external evidence), no other gpu-lane job
  running, no main-model slot generating
  (`/slots`, read only when llama-swap's `/running` lists the model) -- and
  deferred to `until` (derived: the last request + IDLE_MINUTES, or the
  worker's next tick) when it is not; a handler raises `worker.Deferred` for
  a server's own rate-limit reset. Idle-gated: the package onboarding's gpu
  stages, its skills' model stages (`skills.enqueue` sets the flag for a
  skill with `meta.onboarding`), and the skill document index rebuild
  (`skill_match.schedule`, operator decision 5, 2026-09-27: a newly armed
  skill is out of retrieval until the stack is idle; `deploy_check.py`
  still builds it directly).
- **Package onboarding** (`mcp/onboarding.py`; docs/PACKAGE-ONBOARDING.md;
  operator, 2026-09-27: "We should be able to automate all of this from a
  prompt with some links ... we already have the durable machine too"). A
  prompt with links (`POST /dash/api/skill {prompt, links?, aliases?,
  replaces?}`, the Skills page's PROMPT + LINKS mode) becomes a dataset of
  kind `package` whose stages are jobs on the same queue: `resolve` (npm /
  PyPI / GitHub links -> package@version at a commit, each choice with its
  rule; the licence from verbatim quotes: `mcp/package_resolve.py`, GET only
  through `mcp/package_net.py`, unauthenticated GitHub with the server's own
  rate-limit reset honoured) -> `clarify` (a person only when no quote was
  found, or a restricted licence) -> `index` (gpu, idle: the tarball
  verified against `dist.integrity` and `dist.unpackedSize`,
  `deps.fetch_verified`) -> `vocab` (`mcp/package_registry.py`: the
  candidate vocabulary is PROMOTED only if the standing detection labels do
  not get worse -- operator decision 4 -- else HELD with the reason; the
  taxonomy grows a framework term per package from its npm name and the
  aliases the operator typed, decision 7) -> `examples` / `knn`
  (`mcp/package_examples.py`, `mcp/example_knn.py`: human-written example
  code labelled by its own imports; k by leave-one-GROUP-out, decision 1)
  -> `sources` (`mcp/package_sources.py`: every SKILL.md and its references
  at the commit through the ONE skills pipeline; llms.txt pages only when
  the package ships no SKILL.md, decision 3; a new version of the same
  major REPLACES the old skills, a new major sits ALONGSIDE, decision 2)
  -> `skills` (a JOIN) -> `retire` -> `rebuild` (a JOIN on both indexes) ->
  `evaluate` (`mcp/package_eval.py`, leave-one-group-out). The hand-written
  package tables (TERM_PACKAGE, CANONICAL, PACKAGE_AREA, AREA_PACKAGE,
  BUILT_ON) are `package_registry.SEED`; `packages.json`, `held.json` and
  `vocabulary.json` beside the indexes extend them and are re-read on a
  stat, so a promoted package needs no restart. The kNN vote reaches the
  selector as a WEAK area and a rank tier only (`skill_match.plan_turn`
  `knn=`; with the stub decider nothing more is delivered;
  `x_yamadori.skills.knn`). The evaluation's significance level
  `package_eval.ALPHA` is 0.05, the convention the repo's p-values are read
  against (operator, 2026-09-27: keep ours). Tests:
  `mcp/test_onboarding.py`, `mcp/test_onboarding_registry.py`,
  `mcp/test_package_registry.py`, `mcp/test_example_knn.py`,
  `mcp/test_knn_selection.py`, `mcp/test_package_eval.py`. **Not yet run
  live** (`mcp/test_live_stack.py --only onboarding --onboard PROMPT
  --onboard-wait S`, opt-in: it writes live state).
- **Skills -- the one knowledge system** (operator, 2026-09-25/26: "It's
  time to retire hints and build the skills system ... because skills are
  proven in the greater ecosystem." "We don't have rules to fix bugs, we
  have skills."). The hints path, its vector cache and its switch are GONE;
  the recipe corpus (bench/recipes, kept as the migration's source) became
  522 atomic skills (`mcp/skill_migrate.py --apply`, 2026-09-26; the corpus
  and the old cache are archived under index/_archive/). A skill is an Agent
  Skills folder, `index/skills/library/<name>/SKILL.md` + `tests.json`
  (`mcp/skill_md.py`: `name`, a `description` that IS the trigger
  condition, Hermes' `metadata.hermes.tags`, ours under
  `metadata.yamadori` -- id, revision, state, category, applies_when,
  provenance with the verbatim licence quote, each item's source quote);
  harness-portable (Hermes' own linter passes it). WHAT A SKILL IS
  (operator, 2026-09-28, verbatim): "All of our skills increase confidence
  and improve correctness; if it can't, then the line doesn't need to
  exist. That is what a skill is about: facts, proven patterns, helpful
  knowledge you can act on immediately." A concrete pitfall ("Do not cast
  with as any") is good content; self-verification, doubt, uncertainty and
  version history are not (`skill_limits.doubt`, the ASSURED VOICE: the
  validator drops such an item, the serving path leaves it out of an armed
  skill, `YAMADORI_SKILL_DOUBT_FILTER=0` serves stored lines;
  docs/SKILL-FACTORY.md "What a skill is"). ONE pipeline
  (`mcp/skill_pipeline.py`): fetch → screen → screen_model → licence →
  distil (one atomic skill) or decompose (a frontier SKILL.md into several)
  → review → classify → tests → validate → prove → arm, plus watch;
  compiled skills (migration, authored, a dataset's rows) run screen →
  review → classify → tests → validate → prove → arm, inline with no model
  (review and prove record "not run"). REVIEW (`review/1`, 2026-09-28): the
  model's second pass over the draft -- keep, rewrite to the most compact
  actionable form, or drop with a listed reason -- verified by code
  (`skill_pipeline.review_items`; a code-naming item is never dropped as
  "no action"). PROVE (`mcp/skill_prove.py`, 2026-09-28): 2-3 probes from
  the skill's should-cases, answered WITHOUT and WITH the skill (the same
  seed), checked by code -- parse, DO present / DO NOT absent, and a type
  check (`mcp/typecheck.py`: the harness box's pinned tsc / pyright from
  volume yamadori-typecheck-tools1, read-only, in a throwaway container,
  against the held package version) -- worse on any check QUARANTINES, a tie
  arms "no measurable gain", and a proof no check decided, after one retry,
  is UNPROVEN and quarantined; idle-gated gpu jobs, and a backlog for the
  armed library. The library's rebuild to the definition is
  `mcp/skill_rebuild.py` with `skills/replacements/2026-09-28-assured-voice.json`
  (the r3f recipe set replaced by R3F v10 / drei v11 docs: its doubt-
  bearing skills archived only when the replacement batch has armed, its
  clean how-to packs each only when its replacement pages passed PROVE;
  three's llms.txt and llms-full.txt; every other doubt-bearing skill
  rebuilt through the review); built and tested offline, **not yet run**
  (GPU). Every model-assisted stage's prompt is a versioned
  template in `mcp/skill_prompts.py`, pinned by a test; the model proposes,
  the code verifies. Every model stage runs as the helper job
  `skill.<purpose>`, thinking capped at `HELPER_THINKING` (2026-09-27; it
  ran uncapped as role main). decompose/3 adds ONE compact LEAD skill
  per upstream SKILL.md (`metadata.yamadori.lead_for`, and `lead_for` on
  the served row: pushed by skill_packages, never matched); the
  faithfulness check is MANDATORY for a model-written skill (off -> it
  fails; an error -> it waits at validate; no verdict -> the item is
  dropped); "Not for" boundaries and sibling near misses are recorded,
  not gated (`mcp/skill_boundaries.py`, which also lists black-hole
  candidates for the operator). tag/5 files `all_of` (koota WITH React), verified against the text;
  a changed frontier source re-decomposes (`watch_frontier`), superseding
  its children; the screen strips zero-width typography and records it
  (tags and bidi overrides still quarantine). The taxonomy (`mcp/skill_classify.py`): artifact,
  language, framework, PHASE (plan/implement/debug/verify/refactor/review),
  plus situations, all_of and topics. The size caps (`mcp/skill_limits.py`,
  CHOICES): body hard 450 tokens, 6 items, 2 prohibitions (over it FAILS,
  flagged not rewritten) -- the budget lives in the skills: no per-turn
  token budget, a sanity ceiling of 8 per decision (2026-09-27). Each skill
  carries ACTIVATION TESTS (`mcp/skill_tests.py`): a skill whose near miss
  selects it is QUARANTINED; behaviour checks are stored for a live A/B.
  `mcp/skill_select.py` selects at tiers medium and up (the tier flag
  `skills`; X-Yamadori-Features reads `hints` as its alias for ONE release,
  because benchmarks still send it) as SUCCESSIVE FILTER ROUNDS (language,
  framework, phase, situation, legal-now; a round whose patterns are silent
  admits nothing, a cosine only ranks inside a question -- no absolute
  cosine cut since 2026-09-27), then ONE typed QUESTION per area answered
  by a decider (`mcp/skill_deciders.py`: the evidence stub by default --
  the CLM decider was removed 2026-09-29 -- the fallback model as last
  resort), only the questions the conversation's STATECHART
  (`mcp/skill_chart.py`, in the ledger row) makes legal (2026-09-27,
  docs/SKILL-FACTORY.md); the chosen skills' bodies are a
  user-turn tail injection recorded in the ledger;
  `x_yamadori.skills` {ids, versions, names, chars, why, matched,
  embedding.query}. The embedding stage embeds the last user text, then a
  fixed digest (every error, import, name and file line) of the messages
  since the last user turn (`skill_classify.window_start`), always joined,
  the whole cut at `EMBED_QUERY_CHARS` (`skill_classify.embedding_query`,
  2026-09-27; the record carries counts, never the text). SLOTS
  (2026-09-27): every area the user ASKS for by name gets its best skill
  (`skill_classify.asked_terms`: the user's own prose, not a pasted
  manifest; a specific framework is a fact, a language or React a word),
  then the areas it implies (`skill_select.IMPLIES`, each row citing the
  package's docs: r3f -> React refs, r3f v10 -> TSL, koota + React ->
  koota/react, ...), then strongly supported extras, one per area; a
  platform API (`localStorage`, `useEffect`) is no fact outside a specific
  framework (`COMMON_API`). PER-TURN (`skill_select.decide`, every request,
  agent steps included): a skill goes in only when this turn brings new
  evidence (asked / new_area / error / phase; no fade and no cooldown since
  2026-09-27, never the identical line twice in a row), the first time as
  its body, later as a RECALL line -- its first DO (or WHEN) item and first
  DO NOT ("Remember (craft <name>): ...; not ...") -- at the end of what
  the model reads next
  -- the user turn's tail or APPENDED TO THE TOOL RESULT (the `skills` part,
  ledger-replayed); the per-conversation state is a ledger row (kind
  `skills`), which forgets what was given when the proxy's own compaction
  count moves (`progress` "compactions", passed on the skills paths). A
  HARNESS NOTICE -- a user turn opening with a bracket tag, Hermes'
  `[CONTEXT COMPACTION ...]` summary included (`skill_classify.
  is_harness_notice`) -- asks for nothing and opens no asked slot. The
  skills owner's constants-audit items were applied 2026-09-27 (60 done,
  21 kept with the check each removal fails, 7 for the operator:
  docs/CONSTANTS-AUDIT.md). The gate: `bench/skills/test_daily_eval.py`
  (`bench/skills/daily_eval.jsonl`, everyday work labelled by intent;
  docs/SKILL-FACTORY.md). **Not yet run live.**
  **The work's own evidence** (2026-09-27; pagoda-h4, relay.jsonl, n=1
  run: every blind injection traced to evidence that was not the
  project's): selection reads the CLIENT's messages -- `proxy._skills_tail`
  hands `raw` over (`_SKILL_CTX`), never the ledger-restored ones, whose
  earlier craft bodies had picked the next craft (math's noise body said
  "draw", then a TypeGPU body's `root.unwrap` picked another TypeGPU craft
  in an r3f project); `skill_select.evidence_view` sets aside a tool result
  whose call read an installed dependency (node_modules/<pkg>/,
  site-packages/<pkg>/), the lines of a result located inside one (tsc's
  node_modules/@types/react errors), a manifest/config file's content
  (package.json, tsconfig*.json, *.config.*) and any craft block of ours
  handed back; an ERROR brings a craft only when the error's own lines name
  it (`error_names`: a code-shaped topic, its package or framework, or it is
  an error craft); a craft about a framework the conversation does not use
  (a fact, the user's own prose, IMPLIES and `BUILT_ON` closing it) stays out
  on evidence (`subject_unused`; typegpu-pipelines-* are filed under
  TypeScript and are TypeGPU's); and a later turn or a step needs what the
  craft is ABOUT in its evidence (a subject topic or a gate of its own), never
  its area's file type alone. `x_yamadori.skills.set_aside`; skipped entries
  say why. **SERVER-TOOL RECALL** (craft/3, "Remember (server tool
  yama_think_deeply): ...") was RETIRED the same day (operator, after
  pagoda-h5, where three such lines went out and the tool was never
  called); the triggers that then ran the job for the model were removed
  with deep thinking (2026-09-29). The detection is still
  `skill_select.server_tool_triggers`; gates `mcp/test_tool_recall.py`.
  `mcp/skill_learn.py` the fallback records, idle-time learning and the
  per-skill selection log. Like `deep_decisions`, every record, learned
  trigger, selection row and label carries its TRAFFIC class
  (`skill_learn.traffic_of`: `test` for the live suite's accounts via
  `corpus.account_traffic`, failing closed; `client`; `unknown` with no
  account -- offline suites, replays) and ONLY `client` rows are learned
  from or served as learned triggers (2026-09-27: the one learned trigger
  came from a live-suite prompt, and an offline suite had written 21
  fallback records; the trigger was removed, backup
  index/_archive/skill-traffic-before-20260927-022536). A skill ARMS without review. The skill factory
  API and its contract: `mcp/dash_skills.py`, docs/SKILL-FACTORY.md (the
  React NAEDOKO surface is a follow-up). The skills are the service's
  WHOLE knowledge base (operator, 2026-09-25) -- never this repo's docs
  (`find_in_knowledge_base`, the second brain's reader of them, went with
  deep thinking, 2026-09-29). Three skills are AUTHORED from our own evidence
  (skills/authored/: browser-app-entry-point, fix-located-defect-first,
  module-exports-match-callers; `--authored` installs them). **Our skills
  are about the WORK, never about driving a harness** (operator,
  2026-09-26: "We shouldn't carry harness specific skills, that is the job
  of the harness"): a harness's own usage lives in ITS skills folder, as
  config -- the Octopus Hermes profile's screenshot skill is
  bench/octopus/hermes_skills/, installed by make_profile.py; the server
  copy is archived. Harness skills and our auto-selected skills coexist.
  The client-tools gate (`tools_any`/`tools_all`/`tools_none`) stays, for
  a work skill that needs a kind of tool to apply. Tests:
  `mcp/test_skill_factory.py`, `mcp/test_skills.py`.
  **The V4 stack's knowledge (2026-09-26).** pmndrs ships its own Agent
  Skills (koota `skills/koota/SKILL.md`, pmndrs/math `skills/math/SKILL.md`);
  they went through the ONE pipeline at the commits of the versions V4 pinned
  (koota v0.6.6, math@0.1.0; since 2026-09-27 the Octopus V4 and pagoda
  r3f-stack prompts name the stack in ONE sentence -- "Build it with r3f
  (react-three-fiber) v10 and Koota and pmndrs math." -- with no pins,
  layout or rules, and grader v4 judges V4 by behaviour: bench/octopus/
  variants.py, grade.py GRADER_CHANGES[4]), with koota's reference docs and math's API.md
  distilled for what the SKILL.md files lack: 10 skills (koota-traits-and-
  entities, koota-queries-and-systems, koota-react-integration,
  koota-with-react-three-fiber, koota-react-entity-lifetime,
  koota-frame-loop-systems; math-data-oriented-functions,
  math-hot-path-pitfalls, math-with-three-js, math-noise-and-seeded-random).
  No model run: `mcp/skill_offline.py` runs the pipeline with the model
  stages' proposals written by hand (skills/ingested/pmndrs/), every check
  still made; the model screen and the faithfulness check are recorded
  SKIPPED. `koota` and `pmndrs_math` are taxonomy terms (the tag template
  is tag/3). The migrated TSL, R3F and drei skills were refiled
  (skills/retag/2026-09-26-r3f-tsl.json: TSL under three.js, not TypeScript
  alone; R3F descriptions that read "Use when working with general (...)"
  rewritten), and selection reads pinned dependencies, imports in tool
  results and the code a tool call wrote as facts, spreads the three slots
  over areas (host areas -- React, a language -- after the areas they
  host), and ranks the newest evidence higher (docs/SKILL-FACTORY.md
  "Selection"). Replay: `bench/skills/replay_selection.py --stack`;
  tests: `mcp/test_pmndrs_stack.py`.
- **Engines.** `engines/manifest.yaml` pins every binary the stack runs
  (llama.cpp x3, sd.cpp, llama-swap): base SHA, patches in
  `engines/patches/<engine>/`, toolchain, flags, and the SHA-256 of what we
  ship. `scripts/build_engine.py <engine>` rebuilds one into a new
  directory; `--verify-only` checks what `config.yaml` points at, and
  `scripts/deploy_check.py` fails a deploy that runs an unpinned binary.
  Source-reproducible, not bit-for-bit: `docs/ENGINES.md`.
- **Packages.** `mcp/deps.py index name@version [--embed]` indexes a
  package, and `mcp/deps.py health name@version` checks one (it reports
  `zero_vectors`: an index built without `--embed` is all zeros by design). They are stored in
  `index/packages/*.sqlite3`. `describe_index` does *not* list them; it
  covers only the bound repository. Version is part of identity. Held for
  the V4 stack (2026-09-26): `math@0.1.0` (pmndrs/math, embedded),
  `koota@0.6.6`, `@react-three/fiber@10.0.0-alpha.5` (still the newest
  alpha; only 10.0.0-canary builds are newer).
- **Concept seeds.** `mcp/concept_seed.py` draws from the 27B's own token
  embeddings, `index/token_embd.npz` (one word per conversation, on its
  first user turn: "The concept seed" above). Extract them once with
  `scripts/extract_token_embd.py`.
- **Models.** `models/manifest.yaml` records every model file config.yaml
  loads and every derived artifact code loads (size, sha256, the HF repo at
  a full commit, or the recipe), plus the runtimes and their locks
  (`requirements.lock.txt`, `locks/`). `scripts/fetch_models.py` re-obtains a
  pinned file; `scripts/verify_artifacts.py` `verify(config_path)` proves at
  deploy that what is loaded is what is recorded. A model whose name is not
  public goes in the gitignored `models/manifest.local.yaml`. Changing a
  model means changing its entry: `docs/MODELS.md`. Binaries are
  `engines/manifest.yaml`'s.
- **Dashboard.** The React app is in `web/`, and the committed `web/dist` is
  served at `/` (a browser asking for HTML; old `/dash` links redirect,
  mcp/server.py). The Python pages are at `/dash/classic*`. `/dash/api/*`
  requires a key. The KV panel's child-slot label comes from
  `budget.budgets()["child"]` (layout v2: the decider lane). A VIEW NEVER
  LOADS A MODEL (2026-09-30): every `/dash/api/*` read goes through
  llama-swap's `GET /running` first and reads a model only when it is
  loaded, from its own port (`mcp/test_dash_no_load.py`: zero `/upstream`
  requests with the model off the card). JJAVA (`/jjava`,
  `/dash/api/jjava`, `mcp/dash_jjava.py`: the decider's use, latency,
  answers, thresholds, the lane, the Jev API's endpoints, the injector) and
  SOKUDO (`/performance`, `/dash/api/perf`, `mcp/dash_perf.py`: tok/s per
  model, both GPUs, the gates, the swaps; it replaced the benchmark page;
  since 2026-10-06 also TOK/S BY EFFORT TIER, `by_tier`: decode and prefill
  tok/s per tier x model x role and context bucket, warm and cold apart,
  traffic `client` | `test` | `all` at `/dash/api/perf/<window>/<traffic>`,
  from `stats.sqlite3` generation rows that now carry tier, traffic, corpus
  turn id and a cold flag -- rows before the deploy carry none (tier null,
  in `all` only; operator: "we consider different qwen variants the same
  model behind our proxy");
  NEBARI was retired the same day and folded into the Skills page's LIBRARY
  view: served skills by area, held packages and what reads them,
  `/dash/api/skill-factory/library`; `/nebari` redirects to `/skills`)
  read `index/stats.sqlite3` (`mcp/stats_store.py`, recorded off the
  response path where the token ledger records) and the logs already
  written. A new bundle is never built in place (the live proxy serves
  web/dist from disk): `npm run build:stage` -> web/dist-next,
  `mcp/test_dash_static.py --staged`, and the deploy's
  `scripts/swap_dash_dist.py`. docs/DASHBOARD.md is the inventory. The Skills page's CREATE has a PROMPT + LINKS mode (a
  package onboarding), WATCHING shows an onboarding card per onboarding,
  and `/skills/onboarding/<id>` is one onboarding's stages and evaluation
  (`web/src/screens/Onboarding.tsx`, `web/src/api/onboarding.ts`;
  2026-09-27, checked against a fixture record, not yet against the live
  proxy).
- **Design.** `design/` holds the design source. `docs/SELECTION-BUILD.md`
  is the selection plan, and `docs/HANDOFF.md` is the latest state.

## Checks

`run_check` executes project commands by LABEL from a whitelist
(`code_search.VERIFY_CHECKS`), never a command line, so text arriving from a
source file or a model cannot reach a shell. Adding a project means adding a
row.
