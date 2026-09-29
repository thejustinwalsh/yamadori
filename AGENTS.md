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
API" below) -- and the
dashboard at `/` (a browser asking for HTML gets the React
SPA; any other client gets the JSON service descriptor, and old `/dash`
links redirect). llama-swap sits behind it on loopback `:11434`, where
`bonsai` and `bonsai-agent` are the same process: the alias exists only for
old clients; `clm-encoder` (the CLM decider's Qwen3-8B Q8_0 encoder,
`docs/CLM.md`) is swapped onto the A4000 on demand by gpu_room and is served
only after llama-swap next restarts, and after the layout-v2 deploy so is
`bonsai-vision` (below). The tools API is `:1235`, Laya is `:1237` and SearXNG (web search,
`docs/SEARCH.md`) is loopback `:8888`. The watchdog supervises six services:
llama-swap, proxy, tools-api, laya, worker and searxng.

## Layout v2 (operator, 2026-09-29; PREPARED OFFLINE, not deployed)

The operator: (a) "The point is to get more context at speed in vram, so
decider was the only thing that needed room." (b) "Vision can go to second
card and swap in and out." (The operator's single-model decision -- the
second brain removed -- is a separate removal, not part of this layout.)
Where this section and older text below disagree, this section is what the
code does (or does after the deploy, where it says so). Details:
docs/ENGINES.md "Layout v2".

- **Vision off the main card** (after `bench/deploy_layout_v2.py`): `bonsai`
  runs without `--mmproj`; `bonsai-vision` (the 27B + mmproj on the A4000,
  on demand, ttl 300, `ondemand` group; gpu_room SIZES row restored) is
  back, and `mcp/vision.py` routes attached images (placeholders) and
  `yama_describe_image` to it whenever the main model's `/props` has no
  `modalities.vision` -- the pre-fold route, restored
  (`vision._vision_model`, `model.VISION_MODEL` = `bonsai-vision`). The
  image rules (docs/IMAGEGEN.md, docs/VISION.md) are unchanged.
- **The lane** (`mcp/slots.py` THE LANE, `mcp/budget.py`): the child slot
  serves the decider and small side calls only; it is NOT released after a
  decider turn or a side call, and it is
  ranked `slots.RANK_LANE` = 3, above the primary conversation, so the
  engine (0041) keeps it in VRAM (no switch back: git has the 2026-09-28
  rule). Its size
  `budget.LANE_TOKENS` = 3,072 = STATE_TOKENS 2,048 + the decider's measured
  non-state maximum (999 cells, 200 turns of logs/proxy.log) rounded up to
  256-cell blocks; it costs the main line those 3,072 cells (main cap = the
  line less the lane). `-c` = 2N + lane. An as-sent compaction goes to the
  least recently used CONVERSATION slot (coordinator: a cache miss beats a
  failed compaction; recorded in the grant, `displaced`, and the displaced
  conversation's `resumed_cold`), emptied after it. A pin persisted from the
  four-slot era onto the child slot is dropped (`stray_pins_dropped`).
  `budget.budgets()["child"]` names the slot's role for the dashboard's KV
  panel: {role: "decider lane", tokens: 3,072, serves: the decider and small
  side calls, kept, ranked, rank, slot} -- no longer "deep thinking".


## Naming

**Products, routes, groups and classifications take bonsai terms.** They are
the names a person reads.

| name | what it is | in code today |
|---|---|---|
| `canopy` | serving group for the main model | no. `config.yaml` says `primary` |
| `rootstock` | embeddings + reranker, always resident | no. `config.yaml` says `retrieval` |
| `graft` | on-demand models, swapped in and evicted | no. `config.yaml` says `ondemand` |
| `taproot` | result tier: a declaration with this name is here | yes, `find_by_meaning` output |
| `branch` | result tier: both retrievers agree | yes |
| `shoot` | result tier: one retriever only, unproven | yes |
| `rings` | the durable work log that survives compaction | yes, `mcp/rings.py` |
| `shomen` | deep thinking, the second context | yes, `mcp/shomen.py` |

Whether the group names or `config.yaml` should change is an open operator
decision. Do not "fix" either side on your own.

In anything a user sees, the second context is called **deep thinking**.

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
| main -- the model the client talks to (`proxy.main_tools`) | the CLIENT's tools, untouched and first, plus only `yama_generate_image` and `yama_describe_image` where offered (rows below), `yama_think_deeply` and `yama_plan` at `xhigh` and `max` (below), and `yama_recall_craft` where the conversation's craft offer says so (row below), and the MCP host's package lookups where its offer says so (row below; 2026-09-28). EVERY TOOL OF OURS ON MAIN IS `yama_*` (operator, 2026-09-27): a name no harness offers, one constant per name (`deep.TOOL_NAME`, `deep.PLAN_TOOL_NAME`, `images.TOOL_NAME`, `images.DESCRIBE_TOOL_NAME` = `vision.TOOL_NAME`, `skill_prompts.CRAFT_TOOL_NAME`), each description saying it is a server tool (runs on the Yamadori server, does not touch the workspace). The old names (`think_deeply`, `generate_image`, `describe_image`, `recall_craft`: `proxy.LEGACY_TOOL_NAMES`) are still READ -- stored ledger hops replay byte for byte, a withheld list in the craft state maps to the new name, a call by an old name runs as the new tool (`proxy.canonical_tool_name`, `proxy.is_ours`), and a client tool under an old name is a declared overlap. No code tool, no work-log tool, no `check_code`. EVERY tool of ours is compared with the client's first (`proxy.tool_conflicts`, operator 2026-09-27): withheld on the same name normalised (case, `-`/`_`, a plural s) or a declared overlap (`proxy.TOOL_OVERLAPS`: yama_describe_image beside Hermes' `vision_analyze` or a client `describe_image`; yama_generate_image beside a client image generator, `generate_image` included; the delegate arm beside OpenCode's `task` / Codex's `multi_agent_v1`), kept for the conversation, recorded in `x_yamadori.tools_withheld`; gated on every harness fixture by `mcp/test_harness_decisions.py` column `tools`. |
| main, where the craft offer holds (skill_select PROGRESSIVE DISCLOSURE; operator-sanctioned exception to "no tool of ours on main", 2026-09-27) | `yama_recall_craft` (`skill_prompts.CRAFT_TOOL_NAME`, the one constant; argument `name_or_topic`). Its description is written as trigger conditions: it answers "how is this done well?" for one craft from this service's library -- call it INSTEAD of guessing a library's pattern; when about to write code with a library the craft list names, when a line said "Remember (craft ...)", when unsure of a library's pattern or API, when an error comes from a library a craft covers. The proxy runs it as a HIDDEN HOP like yama_think_deeply (the craft in full as the tool result, ledger-replayed); an unknown name returns `NO_SUCH_CRAFT` with its near misses. Offered with the craft INDEX at the end of the system text, both decided on the conversation's first request and kept, only when the index has entries. Everything the model reads says CRAFT, never "skill" (the harness's own skill tools are another store). `x_yamadori.craft`. |
| main, at tiers `xhigh` and `max` (`deep.think_tool_offered`; Phase 0.6) | `yama_think_deeply` (`deep.THINK_TOOL`): with `yama_plan` the non-image tools of ours on main (one offer, `deep.think_tool_offered`), the model-chosen deep-thinking trigger. Arguments: the question, and what was tried. The proxy runs it as a hidden hop (below, "Deep thinking's triggers"). Offered by the tier's own allowance and not forced off, decided on a conversation's first request and KEPT for the conversation (a tool list that changes between turns changes the cached system block; a first request is always decided afresh, so benchmark arms that share opening messages do not inherit each other's list); a call on a request that forbids deep thinking gets `DEEP_THINKING_OFF`. Never on a header-forced arm at a lower tier. KNOWN and kept (pre-deploy review, 2026-09-24): the offer flips from no to yes when a conversation's tier rises to `xhigh`/`max` mid-way, which changes the system block once and costs that request its prefix. |
| main, at tiers `xhigh` and `max`, with `yama_think_deeply` (operator, 2026-09-27; pagoda-h2) | `yama_plan` (`deep.PLAN_TOOL`; argument `task`): the plan job as a server tool. Its description is written as trigger conditions: before a significantly long implementation task -- a new app, a multi-file feature, a migration, a rewrite -- call it for the files, the order, the key decisions and the risks. The proxy runs it as a HIDDEN HOP exactly like `yama_think_deeply` (`proxy._plan_task`: `shomen.run("plan")`, its own budget #60): the plan is the TOOL RESULT (`proxy.PLAN_RESULT_HEAD` + the four sections + `proxy.PLAN_RESULT_TAIL`), ledger-replayed, and NOTHING is prefilled after it -- main goes on acting. `PLAN_CALLS_PER_REQUEST` = 1; not after a run before main in the same request (`ALREADY_PLANNED`). THE INITIAL PROMPT IS ALWAYS PLANNED THE SAME WAY: on the conversation's first user turn (`deep.kickoff` `new_task`) the proxy inserts a synthetic `yama_plan` call (`deep.KICKOFF_PLAN_ARGS`, a pointer to the user's message, not a copy) and its result before main's first generation (`proxy.synthetic_hop`, `_run_turn` `pre_hops`), so main continues from a tool result: no visible prefill, no "Today I was inspired by" / "After thinking deeply," opening, no `deep.PLAN_HEAD` reasoning. Before 2026-09-27 the plan was prefilled under that opening, and on pagoda-h2 main continued it as prose about the second model and stopped with no tool call (finish=stop, 71 tokens); the harness ended the run. `x_yamadori.investigate` {into: `tool_result`, tool}, `x_yamadori.deep.plan_tool` {offered, calls}. `mcp/test_ledger.py` `[kickoff]`, `[yama_plan]` (the served template); **not yet run live**. |
| the second brain (`proxy.deep_thinking_tools`; `shomen.run`) | the eight above, plus `record_step` and `read_rings` (`INTERNAL_TOOLS`, scoped to the conversation's lineage), `yama_generate_image` and `yama_describe_image`, and (Phase 0.6, `mcp/research_tools.py`) `find_in_knowledge_base` (the service's knowledge base: the skills pipeline, nowhere else -- operator, 2026-09-25/26. The ARMED skills, `skills.armed()` -- the state the request-time path serves; the recipe corpus hints used to serve was migrated into skills on 2026-09-26; each item re-screened on read by `skill_screen.screen_fetched`; cite `skill:<id>`. It replaced `find_skills`, and no developer doc -- docs/, AGENTS.md, README.md -- is readable by any model: docs/SELF-IMPROVEMENT-LOG.md is our own test tasks' answer key, #42. The work log is `read_rings`), `read_web_page` (a pinned GET: every hop -- robots.txt and each redirect -- resolved once and connected to at that address, refused unless `ip.is_global`, one deadline and a byte cap; widened 2026-09-25 (operator: "let it search lots of stuff ... be careful of request params and headers, and keep it on get requests"): GET only (`_send`), and exactly the fixed `REQUEST_HEADERS` -- a plain `yamadori-research/2` User-Agent, Accept, `Accept-Language: en`, `Accept-Encoding: identity`; never Cookie (Set-Cookie ignored), Authorization or Referer. A SEEN URL -- a `search_web` result this run, the user's own, or a link of a page read this run -- is fetched with its query EXACTLY as seen (a redirect's Location as the server sent it); any other URL (the model's memory) by its PATH only, query and fragment stripped, and the result says so. For every provenance the whole URL passes a guard: length caps (256-char path for a memory URL), no secret-shaped or high-entropy segment, no code in the query, no 24+ character run of the conversation's text (except in a search result's URL). The result lists up to 20 of the page's visible links, same site first, screened and guarded, to browse on. At most `READS_PER_RUN` = 20 reads; a 429 or CAPTCHA page pauses that host (`RATE_LIMITED`, retryable, Retry-After). `x_yamadori.deep.screen[].fetches` records each fetch: host, provenance `search`/`user`/`link`/`memory`, `query_stripped`, status, bytes, refusals with reasons -- never the path. Cite the URL, labelled `(web)`; `mcp/test_web_access.py`) and `search_web` (the local SearXNG, `YAMADORI_SEARCH_URL`, loopback only; `categories=it` for code; at most `SEARCHES_PER_RUN` = 10 per run (3 until 2026-09-25; Brave's ~10 per 2 min, a choice); SearXNG's 429, or no results with every silent engine CAPTCHA'd or rate-limited, pauses search `SEARCH_BACKOFF_S` = 180 s (Brave's suspension) -> `SEARCH_RATE_LIMITED`, retryable; a query carrying code or a secret is refused before it leaves; down -> `SEARCH_UNAVAILABLE`, retryable, the remedy). Never `yama_think_deeply` (it would recurse). Never the client's tools: what the harness read is in the conversation already, and a file it would need becomes the hand-off's NEXT STEP, for main to read with the harness's own tool. |
| off by default | `delegate_investigation`. Only with `YAMADORI_DELEGATE_TOOL=1` or a tier's `delegate` flag. It is a benchmark arm; main runs it as the runner's `investigate` job. |
| main, where the MCP host serves (operator, 2026-09-28: "the proxy provides MCP servers to the model zero-config for every harness, starting with PackageLens"; switch `mcp_tools`, tiers `medium` and up, header / `YAMADORI_MCP_TOOLS`) | `yama_find_package` (words -> the registry's exact names), `yama_list_package_versions` (every version, its date, the dist-tags, prereleases included, each npm version's peer dependencies), `yama_read_package_readme` (the registry's README, else the repository's) -- 3 of PackageLens 0.1.11's 8 tools (`mcp/mcp_config.py` says why each other one is out; `ecosystem` required on all three); renamed verb first 2026-09-29 (they were `yama_package_versions` / `yama_package_readme`: each row's `legacy`, read as the new tool by `mcp_config.tool` / `canonical`, `mcp_host.definitions`, `line_for`, `run_tool`, which records `called_as`); and `yama_resolve_packages` (2026-09-29, operator: "packagelens should be able to resolve peer dep ranges ... recommend not only the core package names but the dependency exact names and version too ... All mechanically from package json info"; `mcp/npm_resolve.py`, a config row with `runner: npm_resolve` in place of an upstream): npm 11.19.0's OWN resolver (`npm install --package-lock-only --ignore-scripts`, `mcp_servers/packagelens/resolve_driver.js` on stdin) in a throwaway container of the same pinned image on the same gated network -- every package together, then the exact set again; an ERESOLVE moves one package within what was asked; the peers npm installed, the optional peers it left out with their ranges, DefinitelyTyped's `@types/*` by major.minor, all verified by npm as one install line plus a dev line; a prerelease only with `allow_prerelease` or a named one, and then the version a dist-tag names; `ecosystem` "npm" required; bound npm's own one-fetch worst case, 970 s; `mcp/test_npm_resolve.py` (the driver against a fake npm with npm's own output shapes), live smoke 2026-09-29 (the sets installed with a real `npm install --ignore-scripts`, no ERESOLVE); **the model's use is not yet probed** (THE TOOL RECIPE rule 7). The PROXY IS THE MCP CLIENT (`mcp/mcp_host.py`, our own stdio JSON-RPC client): each enabled server of `index/mcp/servers.json` (absent -> the built-in default; `YAMADORI_MCP_SERVERS`) runs ONCE, long-lived, started by server.py (never on import), in its pinned image (`mcp_servers/packagelens/`, `models/manifest.yaml` `mcp-packagelens`: the image id is checked before every start) on its own `--internal` network through the egress gate (`bench/sandbox/sandbox_net.py`, NODE_USE_ENV_PROXY=1), uid 1000, every capability dropped, read-only root, no credential; restarted on the next call after a crash; a call bounded by the server's own `call_timeout_s` (PackageLens: 48.6 s, derived from its http.js). Run like the image tools: a HIDDEN HOP (`proxy._run_our_tool`), ledger-replayed; the result rendered to text (a README's images as their alt text), screened by `skill_screen.screen_fetched` and framed as data. Offered on a conversation's first request and KEPT by name (`proxy._mcp_offer`, the skill state's `mcp`), with ONE system line where offered (`mcp_config` `line`: "To find a package, its versions and its README, and the exact versions that install together, call yama_find_package, yama_list_package_versions, yama_read_package_readme and yama_resolve_packages before you write package.json or install it."; 2026-09-29; UNMEASURED WORDING). Withheld beside a client tool that answers the same question (each tool's `overlaps`, read by `tool_conflicts`). `x_yamadori.mcp` {switch, offered, on_main, kept, why, line_chars, calls[{tool, server, upstream, ms, ok, bytes, error, screen, args (120 chars), names}]}; `GET /dash/api/mcp` (read-only; `mcp_config.save` / `set_enabled` are the dashboard page's writers, not routed). GAPS: PackageLens returns `dependencies` only (the peers come from the proxy's own reads since 2026-09-29: `mcp_host.npm_peers`, `yama_resolve_packages`); the npm registry serves an EMPTY README for many packages (math, koota, @react-three/fiber, react, zustand, express, axios, typescript, zod, date-fns, react-router, tailwindcss; 2026-09-28) -- the result then says the README is in the installed package. `mcp/test_mcp_host.py`; live smoke `python mcp/mcp_host.py smoke packagelens` (no model; run 2026-09-28); **the model's use is not yet measured** (`bench/mcp/lookup_probe.py`). |
| main and the second brain, only when `YAMADORI_IMAGEGEN_URL` is set | `yama_generate_image` (`mcp/images.py`). Every tier -- a capability, not a gate (operator, 2026-09-23) -- even when the code tools are withheld, and to deep thinking for mockups and designs (its markdown crosses back in the finding; `yama_describe_image` below is how the text-only model looks at what it drew). Two image models, `base` (20 steps) and `turbo` (Viggle 4-step): the caller's account preference picks (dashboard SETTINGS, `PUT /dash/api/settings/image`), else `YAMADORI_IMAGEGEN_DEFAULT`, else `turbo` -- the built-in default since 2026-09-24 (operator; the launch scripts also set it; 23.6 s vs ~108 s; quality vs base not yet compared by `bench/imagegen/compare_turbo.py`). The model has no say; `x_yamadori.images` records the model, steps and where the choice came from. **On main the proxy SHOWS the picture the moment it exists** (operator, 2026-09-25): streamed, its markdown line goes out as CONTENT when the tool returns -- before the next generation is requested -- and the rest of the turn's reasoning goes out as heartbeats (CHANNEL ORDER); blocking, the line opens the answer (the same content). Main is offered `images.MAIN_TOOL` (its description says the picture is shown when made; deep thinking keeps `images.TOOL`, whose markdown crosses in the hand-off), and the tool result says the picture is already shown above the reply (`images.shown_on_main`: the situation, no prohibition); a copy the model writes anyway -- the image with that url, any alt, or the bare url on its own line -- is removed from the CLIENT's copy only (`proxy._ImageDedup`). The line is content the slot did not generate: the ledger keys the turn by the client's copy and renders the slot's text (#10), like a compaction summary's session line, so the next request extends the slot. `x_yamadori.images[]` adds `emitted` {ms since the request started, after_hop, before_hop, at: stream | answer_start, order} and `duplicates_stripped`. `mcp/test_image_emit.py` (through the served template); **not yet run live**. See `docs/IMAGEGEN.md`. On `/v1/responses` a request's hosted `image_generation` tool IS this tool, at the tool's size, and each image also comes back as an `image_generation_call` item ("The Responses API" below). |
| main and the second brain, wherever `yama_generate_image` is offered (deep thinking included), and on every tier when the request carries an attached image; off with `YAMADORI_VISION=0` | `yama_describe_image` (`mcp/vision.py`): sends one image and a question to `bonsai-vision` (the 27B + mmproj on the A4000) through `mcp/model.py`, in the image lane, and returns the answer as text. It reads only this request's attached images (by id `image-<10 hex>`) and our media store (by sha, with a verified signed link or an image made in this request). It never fetches a URL and never opens a path. `proxy.prepare` replaces each attached image part with a placeholder naming its id; an image part carrying OUR signed /media link (signature and expiry verified, host `YAMADORI_PUBLIC_BASE`'s, the request's own, or loopback) is read from the media store as an attachment -- how Hermes' `vision_analyze` looks at what we drew (2026-09-24; offline tests only, the live check in `mcp/test_live_stack.py` images group has not run). `x_yamadori.vision` and `x_yamadori.attachments` record calls and attachments. Streamed, a look the proxy ran on main leaves ONE reasoning line, `` `looked at image-<id>: <the description's first 100 characters>` `` (`proxy._describe_line`, operator 2026-09-25), screened like any tool text shown to the user (markup, links, the template's markers and invisible characters out; a credential, AI-directed text or exfiltration withholds the excerpt: `mcp/skill_screen.py`); once content has started it is a heartbeat. Not delivered on the blocking path (reasoning is not). **Run live once, on an attached image** (2026-09-24, `mcp/test_live_stack.py --only images`, n=1: a correct answer, 5.9 s, 168 prompt tokens for a 256x256 PNG). The draw-then-look check in `docs/IMAGEGEN.md` "Seeing: `yama_describe_image`" has not run. The A4000 VRAM concern there was measured in the same test: 308 MiB free at the peak. |

Deleted 2026-09-24: `bind_project_context` (it pinned versions for tools main
no longer has) and the `check_code` TOOL (the proxy checks code itself --
`mcp/tool_code.py` for client writes, `code_check.review_answer` for an
answer's code). The proxy writes the work log itself (`proxy._log_turn`: the
client calls the model made, what the checks found, what deep thinking and
fan-out handed back) and re-injects it on the first user turn after a
compaction -- since 2026-09-26 also when the conversation keeps its session
key across the compaction (#41 made the old new-key link unreachable for
Hermes: `progress.note_compaction` in `_compaction_done`, switch
`work_log_reinject`). `proxy.OUR_NAMES` is what the proxy executes itself (22 names:
17 since `find_skills` folded into `find_in_knowledge_base`, 2026-09-25,
`yama_recall_craft` and `yama_plan`, 2026-09-27, the MCP host's three
package lookups, 2026-09-28).

Fan-out is **not a tool**: `mcp/selection.py` turns it on per request, and
the tier (`mcp/tiers.py`) only says what is *allowed*. **Deep thinking runs
on four triggers** (Phase 0.6, operator 2026-09-24; section "Deep thinking's
triggers" below), at `xhigh` and `max`: the model's own `yama_think_deeply` call,
struggle the proxy detects, a known-hard area, a new task's kickoff. It is no
longer limited to `library_question`, and Laya is NOT consulted for it (a
separate evaluation decides Laya vs Tev1). The regex + symbol lookup + Laya
path survives only as selection's LEGACY path (no route, no trigger), for
the offline evaluators that replay it. Every decision -- and every
non-decision -- rides on the response as `x_yamadori` (`selection`, `deep`).

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
conversation's, shared by the second brain (deep thinking, fan-out's B/C,
fix-ups, summaries: `slots.helper_slot()`), the decider, side calls and an
as-sent compaction -- they queue on it at the server (2026-09-24 #10/#11 made
the second brain's slot reserved; n-2 of four until 2026-09-28). Every
request names the slots' KV ranks (`kv_rank` / `kv_ranks`, slots RANKS: the
primary conversation 2, another live one 1, the child at its conversation's)
for engine patch 0041, which keeps the primary's cells below the tiered
cache's VRAM line (docs/ENGINES.md "KV rank"); `mcp/budget.py` THE CAP
LAYOUT: main = the line (advertised to every conversation), the child 65,536
of its own. LAYOUT V2 (2026-09-29, "Layout v2" above): the child slot is THE
LANE -- the decider and small side calls (titles), kept, rank 3, 3,072 cells
taken off the cap (`budget.budgets()["child"]`: role "decider lane"); a
second-brain job, while that machinery exists, still runs there at its
conversation's rank; an as-sent compaction goes to the least recently used
conversation slot. `x_yamadori.cache` reports prompt tokens reused vs
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
`model.release_slot`): the second brain's slot at the END of each run, when
the helper lane is let go (`admission.helper_lane`: one shomen.run job, or a
fan-out's B and C together -- never between a job's hops, and never
between B and C, since C's prompt is B's with the last user turn replaced
and reuses B's history from the server's checkpoint there), and the
transient slot after each side call or as-sent compaction (nothing reuses
it: compaction affinity only ever considers PINNED slots, and a
continuation opens with the tools and the summary). Never a pinned slot
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
it costs (CHOICES, unmeasured): the next second-brain run re-prefills the
job's shared head (system + tools), a code chat's next fan-out B the
history its last B held, and a retried as-sent compaction its transcript;
each release is one `/slots` read plus one tiny request on the path of the
request that ends the run. Offline: `mcp/test_slots.py` (release),
`mcp/test_utility.py` (side call), `mcp/test_ledger.py` [release] (a fix-up
through the real doors). **Not yet run live.**

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
itself, and a second-brain run or side call working for a conversation
never clears that one -- never while a compaction sent up as is is in
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
session's chain salt loses turn 1's recorded injection; at xhigh
`yama_think_deeply`'s offer is decided afresh and the tool list changes; area
coverage and the decision rows reset). KEPT at
the 2026-09-27 audit (docs/CONSTANTS-AUDIT.md lists it D, "remove"): it is
not a number but the only carrier a text-only conversation has, and
removing it breaks "one model, one cache" (operator, 2026-09-24) for every
client that sends no key or header -- each such request would be a new
conversation, on a new slot pin. The operator decides. The key is
`session_id.conversation_key` (account, source, id -- never the messages;
all our sources key alike), and EVERYTHING keyed by the conversation uses
it: nebari state, lineage (work log, `deep:<lineage>`, library use, slot
pin, compaction store), and the ledger's chain keys, which are salted with
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
the proxy adds never breaks the pinned slot's prompt cache; the second brain
does the heavy work and folds a short result back; main's context stays
small. The mechanics this rests on were measured before anything was built
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
back, the static addendum, and main's own generations. Nothing else -- no
repair turn, no tool-call round, no weigh turn, no hand-off as a user turn.
`proxy._run_turn` is the one turn implementation both paths run
(`complete()` drains it; `stream_body()` streams its events).

**The ledger** (`proxy.ledger_restore` / `ledger_record_turn` /
`ledger_seed`; storage in `nebari.py`, table `additions` in
`index/nebari.sqlite3`). Everything the proxy added, per message, re-added
byte for byte on every request:

| addition | keyed by | when decided |
|---|---|---|
| a user turn's injection: skills, library definitions, the work log after a compaction (a turn whose content is a LIST of text parts -- Pi, Responses `input_text` -- gets it as one more text part, `message_text`; before 2026-09-26 only a string turn got any) | a hash of the conversation up to and including that user turn | ONCE, on the request whose last message is that turn (the first request after a compaction may add the work log); replayed ever after |
| a tool result's injection: library use (the held packages the conversation uses, #19), then the SITUATIONS (the progress line and unchanged-read lines, `mcp/progress.py`; #50, #54) | a hash of the conversation up to and including that tool result | ONCE, on the request that ends on it |
| a call turn's delivered content (the check note) | its tool-call ids | when delivered |
| a past assistant turn's REASONING, the slot's own (a prefilled hand-off included), put back when the client sent the turn without reasoning; an echo is kept as sent (switch `restore_reasoning`, default on, 2026-09-27; below) | the turn's first tool-call id, else its text (like its content) | when delivered |
| hidden internal hops (image tools, `yama_think_deeply` and `yama_plan` the proxy ran inside the turn, the hand-off or plan as the tool result; and the call + result the proxy INSERTS before main's first generation -- the initial prompt's `yama_plan`, and behind switch `deep_tool_hop` a pre-main `yama_think_deeply`), with their own reasoning while past reasoning is restored; with switch `restore_reasoning` off, their reasoning emptied -- EXCEPT for a client that echoes: when the turn's reasoning is exactly what the client was shown (kind `hop_echo`: a hash of it and the slot's own reasoning for the turn), the hops keep their reasoning and the turn gets the slot's, so the next request extends the slot (2026-09-26, docs/HARNESS-PI.md gap 5: Pi's request after a `yama_describe_image` turn reused 1,829 of 3,663; `mcp/test_harness_forms.py`, not yet run live) | the final visible turn, keyed by the content the client STORES (every byte streamed, #10) | when delivered; expanded back in place on replay |
| a second-brain job's concept seed | the request (hash of its last message) and the job | on the job's first run; a retry or re-render reuses it |

**Past reasoning is restored** (operator, 2026-09-27, reversing the
2026-09-24 pass-through: "Keeping thinking across turns seems useful, fuck
Hermes, Hermes can do whatever it wants."). The ledger records the slot's
own reasoning for every assistant turn it delivers (kind `reasoning`, under
the turn's first key, keyed by what the client stores, #10; a prefilled
hand-off is part of it) and `proxy.ledger_restore` puts it back into each
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

**The model's copies of our notes are delivered as written** (2026-09-27,
docs/CONSTANTS-AUDIT.md). The model writes tool_code's note itself, as the
last line of its content before its calls (#36: Octopus v0f 13 lines over
11 requests; v0b 22, v0e 27). `proxy._ImitatedNotes`, which removed those
lines from the client's copy, and `x_yamadori.imitated_notes` were REMOVED:
the code claimed an operator decision that no operator message holds, and
it altered model output. Our own note still goes past the marker filter
(`_Out.note`; the blocking path splits it off the same way).

Only what the proxy or the model produced is stored; the caller's messages
are only hashed (nebari's standing rule: "Not kept: the caller's code").
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

**The static addendum** (`proxy.ADDENDUM`): one short, fixed decision table
at the end of the client's system text -- what the second model does and the
phrases it uses -- added only where every row is true (where the fixup runs:
`high`, `xhigh`, `max`). The client's system text may arrive as a
`developer` message (Pi and any OpenAI-SDK client with a reasoning model):
`system_roles.one_system` (the one helper both wires use) maps it to
`system` on the way in (`_run_turn`, `prepare`), before
anything reads the messages -- the leading system/developer messages join
into one, a later one becomes a user turn --
because the template renders only `system`, and only first (2026-09-26,
docs/HARNESS-PI.md gap 1: the addendum in front of Pi's `developer` message
was a deterministic 502 at `high` and up; `x_yamadori.roles`;
`mcp/test_harness_forms.py`, not yet run live). Where main has `yama_think_deeply` it carries one more
row (`proxy.ADDENDUM_THINK_ROW`, `proxy.addendum_text`), kept for the
conversation like the tool. No prohibition. Its wording is a choice.

**Library definitions** (`proxy._library_definitions`; operator decision
2026-09-24, an UNMEASURED choice): a user turn classified
`library_question`, at a tier with library help, where the gate says
something held can answer and deep thinking does not run, gets the
definitions of the names it uses that a held source defines (the router's
own lookup, then `find_definition_opt`), capped at 3 names and 3,000
characters (choices), as a tail injection recorded in the ledger.

## The second brain

ONE runner, `shomen.run(job, ...)`, on the one helper lane
(`admission.helper_lane`, `HELPER_LANES = 1`). It replaced three paths:
`shomen.investigate`, fan-out's B/C generation, and the repair loops that
ran on main. Our tools live only here.

| job | what it gets | what comes back |
|---|---|---|
| `investigate` | the question and context, our tools | the four-section hand-off (FACTS / SEARCHED, FOUND NOTHING / OPEN QUESTIONS / NEXT STEP). handoff/2 (operator, 2026-09-28: "facts and the next concrete action, no doubt"): a point the sources did not settle is DECIDED in NEXT STEP, OPEN QUESTIONS holds a gap stated as a fact (never "what would settle it"), NEXT STEP is one concrete action, and `shomen.handoff` drops a line of verification homework, instability or version history (`skill_limits.doubt`; `doubt_dropped` in its stats) |
| `plan` (Phase 0.6, task kickoff) | the task's spec and earlier user turns; our tools and investigate's tool-turn limit (10, 20 at max), thinking `tiers.JOB_THINKING['plan']` 12,288 (the operator's 1.5x); `shomen.PLAN_SYSTEM_V2` (switch `plan_prompt`: the engineer has the task text, no word target, the tools line matches what is offered; off: `PLAN_SYSTEM`); `run_check` is dropped when no repository is bound. #60's plan budget -- tools only on evidence (`plan_tools`), 2 tool turns and a 150 s landing (`plan_budget`, `PLAN_TOOL_TURNS` / `PLAN_SECONDS`), a 4,096 thinking cap -- and the plan prompts' word targets and style sentences were REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md: chosen from one v0f run, n=1). `res.generations` / `x_yamadori.investigate.handoff.plan.run` record each generation's seconds and tokens; its cited/unsupported count only KEY DECISIONS | a four-section plan (FILES / ORDER / KEY DECISIONS / CONSTRAINTS; `shomen.plan_handoff`). plan/3 (`shomen.PLAN_PROMPT_VERSION`; operator, 2026-09-28, after pagoda-h6: "plans don't sow doubt"): RISKS -- whose row read "the risk and how to check for it" and produced "Check @react-three/fiber README for the exact prop API before writing main.tsx" -- is gone; CONSTRAINTS holds properties the code must have, stated as facts; an unknown is DECIDED in KEY DECISIONS; ORDER's last step runs the result. `plan_handoff` drops any line of verification homework, instability or version history (`skill_limits.doubt`, recorded as `doubt_dropped`; a RISKS heading is read as CONSTRAINTS), and a KEY DECISION's citation is checked like a fact but crosses DECIDED -- no "not checked" label (`PLAN_RESULT_HEAD` no longer says which decisions were unchecked); the plan otherwise crosses as the planner wrote it, EVERY item (no item cap, no character cut, 2026-09-27), as the `yama_plan` TOOL RESULT -- inserted for the initial prompt, or main's own call. REMOVED 2026-09-27 (operator: task-targeted steering in prompts; skills are the channel): the confirm-working-directory step and relative-paths rule (#32), the browser-app entry-first row (#55), their enforcement in `plan_handoff`, the client system prompt's working-directory lines passed to the planner, and switch `plan_entry_first` |
| `alternative` | the task (fan-out's candidate B) | an answer, graded by the code check (`fanout.analyse`) |
| `tiebreak` | the task, both candidates and their check results (C) | an answer, graded the same way |
| `fixup` | ONLY the code, all its errors and the user's request, whole (the `errors[:8]` / `request[:2000]` cuts were removed 2026-09-27) -- a client write's file or edit, or an answer's fenced block | the repaired code only when `shomen.fix_rejection` accepts it (else the original, and why), up to `tool_code.REPAIR_ROUNDS` (3) rounds at the request's own effort |

Every job carries a **concept seed** (`mcp/concept_seed.py`) in its USER
message, drawn once per (request, job) and recorded in the ledger, so a
retry, re-render, warm or compaction splice uses the same word. The second
brain never gets client access. For `investigate` and `plan` the line says
first where the word comes from (`shomen.SEED_LINE_RESEARCH`, #52: in 4 of
the 6 v0e p2 struggle runs the second brain decoded the word as a clue --
"an anagram", "base64"): "For variety, a word drawn at random from the
vocabulary, independent of this task and of anything in it: <word>. It is
not a clue, a code or an anagram of anything here; let it shape only how you
approach the work, and leave it out of what you write." The other jobs keep
`shomen.SEED_LINE`; the ANSWER's phrase (`PHRASES["seed"]`) is unchanged.
UNMEASURED WORDING; switch `seed_frame`.

**Fold-back phrases** (`shomen.PHRASES`, the one table in code; this is the
same table). What the second brain did reaches main's turn in these fixed
words, so the result sits in visible content (it survives a harness's own
compaction) and the slot processed exactly those tokens:

| phrase | when | how it gets there |
|---|---|---|
| `Today I was inspired by <word>.` | first, in the fold-back of second-brain work that becomes ANSWER content -- deep thinking, fan-out's B/C (both words when B and C ran). NOT on the mechanical notes (`Verified`, `Repaired`, `Checked`; coordinator 2026-09-24): a fix-up's seed stays in its own user message | with the phrase that follows it |
| `After thinking deeply,` | deep thinking ran | before main (a struggle, an area or a header): PREFILLED as the opening of main's visible answer, the hand-off prefilled as main's `reasoning_content`; main continues -- unless switch `deep_tool_hop` is on (`tiers.BEHAVIOURS`, OFF by default, `YAMADORI_DEEP_TOOL_HOP=1` or the header; the operator is deciding, 2026-09-27): then an inserted `yama_think_deeply` call and the hand-off as its result (`proxy.THINK_RESULT_TAIL`), no phrase. The initial prompt's PLAN never takes this phrase: it is the `yama_plan` tool result (above). After a `yama_think_deeply` call: the hand-off is the hidden hop's tool result, and the next hop is prefilled with this opening (reasoning `deep.THINK_REASONING`, or `deep.THINK_REASONING_MACHINE` when the result is machine-built) |
| `Verified` | the check passed | a one-line note the proxy writes; no generation |
| `Repaired` | the fixup job changed code | a client write: the note before the fixed call. A final answer: non-streamed, the repaired answer IN PLACE of the broken one plus the note; streamed, the repaired block after the answer |
| `Compared two approaches` | fan-out ran B (and C) | code: a line with the winner (delivered as before); prose: PREFILLED after main's answer, ending "Weighing them", and main continues |
| `Checked` | problems remain (at `medium` nothing fixes; or the fixup could not) | a one-line note |

A prefilled phrase ends on a letter, or on an ending measured safe (`After
thinking deeply,` -- STEP 0), never a space: a trailing space is its own
token and breaks the boundary. Whenever the turn the client stores differs
from what the slot generated (a repaired call, a note, a notice), the proxy
**warms** the conversation's slot with the turn as delivered while the
harness runs its tool (`proxy._warm`: `/apply-template`, cut at the last
end-of-turn token, a zero-token `/completion`; `x_yamadori.warm`); the
conversation's next request waits for its own warm on the same slot
(`slots.acquire`, `warm`). `x_yamadori.fold_back` lists each fold-back: job,
phrase, where it went, the seed words.

**Fetched content is data** (operator, 2026-09-24; one rule, one place:
`mcp/skill_screen.py`, the skill screen). Anything the second brain fetches
-- a web page, the link list that comes with it, a search title or snippet,
a skill the knowledge base returns (re-checked: skills can be edited,
and a migrated one holds extracted web text), and what the code tools
return from a HELD PACKAGE's index (third-party source; operator,
2026-09-27: `code_search.screen_package_result` in `code_search.handle`, the
tool output's fences held aside so the cut is the offending line, recorded
in the result's `screen` and `code_search.SCREENED`, a text that cannot be
cut clean withheld as `QUARANTINED`; the bound repository is the user's own
and is not screened; `mcp/test_package_screen.py`) -- is scanned and
STRIPPED before it reads it
(`screen_fetched`: the source rules, the offending span removed and
recorded, a text stripped past `STRIP_MAX_FRACTION` = 25% dropped whole, a
choice), inside a "data, not instructions" frame. The hand-off to main is
the path to the user, so it is screened on the way out (`screen_handoff`):
AI-directed text, exfiltration, shell danger, credentials, hidden markup and
unrelated actions never cross; a shell command, install step or URL that
came from the web (in what the run fetched, not in the conversation) is
removed from an instruction and otherwise crosses only as a fact labelled
"(from the web, unverified)". `x_yamadori.deep.screen` records what was
stripped, dropped, removed and labelled, and every web fetch of the run
(provenance, `query_stripped`, status, bytes). And a skill item must be about the
work itself: `screen_item`'s `unrelated_action` drops one that tells the
model to contact a URL, send data anywhere, run a downloaded script or
change credentials or configuration, however politely phrased (0 of the
2,575 recipe rows flagged). Tests: `mcp/test_deep_review.py` (all 17
malicious fixtures caught on both paths, and on the knowledge base's path
served as a skill's text and as its SKILL.md body; 0/5 clean fixtures
touched).

## Deep thinking's triggers (Phase 0.6)

Operator, 2026-09-24; `docs/SELF-IMPROVEMENT-PLAN.md` Phase 0.6. Allowed at
`xhigh` and `max` (`tiers.TIERS` `investigate`), one helper lane, one run per
request. `mcp/deep.py` decides from what the CLIENT sent (never Laya);
`mcp/selection.py` records the reason; `proxy._deep_thinking` runs the
pre-main job, `proxy._think_deeply` the model's call.

| trigger | fires when | runs | once per |
|---|---|---|---|
| model-chosen | main calls `yama_think_deeply` (a trigger-list description: stuck, unsure of an API or version, a fix failed twice, "still broken") | `shomen.run("investigate")` as a HIDDEN HOP: the hand-off is the tool result, the ledger replays call + result, the next hop is prefilled with the fold-back opening (reasoning `deep.think_reasoning(concluded)`) | `THINK_CALLS_PER_REQUEST` = 1 per request; not after a pre-main run in the same request. A refused call gets `ALREADY_THOUGHT` (`proxy.already_thought`): the facts -- that deep thinking already ran for this reply, when, whether it concluded, how many searches, where its hand-off or plan is (this reply's thinking, a tool result, or that call's result), that nothing more runs for this reply -- not retryable, and a remedy with an owner that points at that result. The steering it carried from 2026-09-26 ("answer now" / "make the next call now", "cite each by its path:line", excerpt labels and files read) was REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md: unmeasured wording, n=1). Whether a second run is allowed after a FAILED one is an open operator decision; it is refused today |
| struggle | at least `STRUGGLE_THRESHOLD` (3) signals in the current episode (every message after the boundary: the last run, or a compaction): the same tool failing again with the same error -- its first error line, RAW (`deep.error_signature`; records keep a `sha1:` digest of it, never the text) -- a file written or patched again when its OWN previous write/patch failed (an error result, the harness's structured `no_change` flag, or our note saying the fix-up left problems) or when the identical edit repeats -- never a clean patch after clean patches (#33: Octopus v0b-V0 sent 3 clean patches of one file to 756 s of deep thinking), a failing command re-run that fails the same way (a re-run that passes is the fix working), our fix-up's "still there after N rounds of repair". A failure is read ONLY from STRUCTURED fields -- an exit status, a harness's `exit_code_meaning`, an ok/success flag, an error field -- never from plain text. ONE EVENT, ONE SIGNAL: a re-run is `failing_command_rerun`, not also `tool_error_repeat`; a rewrite's own failing result is not also a repeat. Every failure counts. REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md: word lists, windows and tables chosen from single Octopus runs): `STRUGGLE_WINDOW` (40 messages), the error-line normaliser, the plain-text error word list (`_ERR_TEXT`) and the "not applied" phrases, the user's "still broken" phrases (`STILL_BROKEN`, signal `user_still_broken`), and the `ENVIRONMENT` table of harness refusals that did not count. CONSEQUENCE: a harness whose tool results are plain text (Codex, OpenCode, Pi; `mcp/test_harness_decisions.py` marks their struggle fixtures KNOWN) gives no failure signal; Hermes (JSON with `exit_code`) does. Reading Codex's "Process exited with code N" and Pi's "Command exited with code N" (each harness's own exit-status line) is an open decision | `investigate` BEFORE main, with the task and the failing output (marked as data); the hand-off prefilled as main's reasoning | episode: a run moves the boundary. `COOLDOWN_REQUESTS` (3) and the same-pattern suppression ("one incident, one run") were REMOVED 2026-09-27: struggle may fire again on the next request with a threshold of new signals, a recurring pattern included; `x_yamadori.deep.last_run` {requests_since_run, boundary} |
| known-hard area | the conversation USES (`proxy.library_uses`) a held package that is unseen (`deep.unseen`: the PACKAGE was first published after `YAMADORI_MODEL_CUTOFF` = 2025-12-31; or the used version is a new major against the last release before the cutoff -- a prerelease is judged by the same rules as any version (the "any prerelease is unseen" rule was removed 2026-09-27; the registry history stores stable releases only, so a prerelease's own date is not known); or `YAMADORI_UNSEEN_PACKAGES`) -- a minor or patch of a long-lived package is not; or an injected skill declares `escalate: true` in its frontmatter | `investigate` before main, on the package's API as the task uses it | package (or skill) per conversation |
| task kickoff | the conversation's INITIAL prompt -- its first user turn only (operator, 2026-09-27: a follow-up after a finished answer is not planned); not a harness notice; a harness's synthetic tool-media turn and its CONTEXT turn (Codex's `<environment_context>`, `message_text.HARNESS_CONTEXT_TURNS`, 2026-09-26) are not user turns; nor is a client side call (utility) or a compaction continuation -- WHATEVER ITS LENGTH (operator, 2026-09-27: "Doesn't matter the length of the prompt, we should always give it a planning turn"; the `KICKOFF_TOKENS` threshold of 1,500, its env pin, its learner and E1's `kickoff` head are gone) | the `plan` job before main (our tools, investigate's tool-turn limit; #60's no-tools / 2-turn / 150 s budget was removed 2026-09-27); an inserted `yama_plan` call and the plan as its TOOL RESULT, a hidden hop before main's first generation (no prefill, no fold-back phrase); main starts acting | conversation |

Priority: a header forcing it off > forcing it on > struggle > kickoff >
area > a server-tool trigger ("Heavy-handed, at the right time", below).
A trigger that finds the helper lane BUSY waits like every helper job
(`admission.WAIT_SECONDS`) and, still busy, is skipped for that request
only (`DEFER_REQUESTS` and the one-wait-per-episode rule were removed
2026-09-27; nothing is carried to the next request). A compaction starts a
new EPOCH of message indexes and keeps the episode and coverage; a
compaction continuation is never a kickoff. A tool result with an exit
status decides by it (0 is success; exit 1 with no output is grep's "no
match"; a harness's own `exit_code_meaning` "... (not an error)" wins --
Hermes marks grep's no-match so, v0b). The state is changed under a
per-conversation lock.
**Every number above is a CHOICE, unmeasured.** Unseen packages
(coordinator's revision, 2026-09-24: a version's own date flagged every
three.js conversation): each package's registry history -- its first
publish and its stable releases' dates -- is stored PER PACKAGE in
`index/packages/registry_history.json` (`deps.record_history`;
`deps.index_package` records it, `deps.py published name@version` backfills).
Backfilled 2026-09-24 for every held package: unseen are `@pmndrs/glyph`
(first published 2026-08-15), `three-flatland` (2026-02-25), and, as new
majors, `@react-three/fiber@10.0.0-alpha.5` (10 against 9.5.0) and
`@react-three/drei@11.0.0-alpha.7` (11 against 10.7.7);
`three@0.185.1` (0.182.0 before the cutoff), `koota@0.6.6` (first published
2024-10-15) and the rest are not. A 0.x minor is not a new major. A REUSED
NAME is a new package (2026-09-26, `deps.history_of_packument`): npm `math`
was js-math (kaleb/js-math, 2011) before pmndrs published a different
project under it, so its history starts at the current repository's first
version -- `math@0.1.0` is unseen (first published 2026-08-16), the 2011
versions recorded under `name_reused`; a repository move with no year-long
gap (@react-three/fiber's drcmda -> pmndrs) and versions with no repository
(koota's first 37) stay in the lineage (`REUSE_GAP_DAYS` = 365, a CHOICE).
`discover` judges a standard-library name by the grammar that parsed it:
`import { vec3 } from 'math'` names the npm package, Python's `import math`
nothing (one shared list had hidden pmndrs/math from library use). The per-conversation state (episode boundary,
areas and kickoffs done, the `yama_think_deeply` offer) is in the ledger
(`deep:<lineage>`), so a compaction-linked continuation shares it.

**The self-improvement loop.** Every request of a conversation leaves ONE
row in `deep_decisions` (the corpus database): the trigger or `none`, the
signals (names and counts, never text), the thresholds in force and their
source, whether deep thinking ran, the hand-off's size and the names it gave
main. Each later request of the conversation OBSERVES the open rows against
what the client sent back (did the struggle stop, did the check pass, was a
file rewritten again, was the hand-off used) and, after
`OUTCOME_REQUESTS` = 5 observations or at once on a clear outcome, labels
them: `fine` / `missed` (a struggle under way, no deep thinking, then more
signals) / `escalated_later` for a non-decision; `skipped` when the helper
lane was busy (`missed_same_pattern` and `in_cooldown` went with their rules,
2026-09-27; old rows keep them). A RUN is judged
against the PATTERN it ran on (#45 (e), `deep._recurrence`), not the episode
a run resets: one recurrence -- a signal with the pattern's key, or ONE
failure with one of its (tool, signature) pairs -- is `not_helped` at
once; `helped` / `wasted` (a hand-off none of whose names
main then used) need `HELPED_WINDOW` = 20 observations (a CHOICE) without
one. **Label rule 2** (`deep.LABEL_RULE`, #52, 2026-09-26): a run is
`helped` only if a PROJECT file (`progress.is_project`) was written after it
-- a write or patch whose result did not fail -- else `no_effect` (v0e p2:
six runs, no product write after any, five labelled `helped`). The rule
rides with the label (`deep_decisions.label_rule`); rows labelled before it
are not relabelled, and `deep.learn` and E1 ignore RUN rows labelled under
the old rule; `no_effect` counts with `wasted` (a CHOICE). Checked
2026-09-26 on the corpus: `deep_adjustments` was empty and E1's escalate
and kickoff heads untrained, so nothing had been learned from the old
labels (the kickoff head was removed 2026-09-27). Switch `helped_needs_change` (the row then labels under rule 1). Before, `helped` was automatic: a run reset the episode and the
cooldown held, so "no struggle within 5 requests" was true by construction
(Octopus v0e-V0 prompt 2: 5 of 6 runs `helped`, the next run 3-50 minutes
later each time). A run row whose conversation stops before 20 requests is
closed `unobserved` after 24 h. ONE INCIDENT, ONE
LABEL: a non-decision label counts once per episode (the other rows of the
episode are `same_episode`). Test traffic is recorded and never learned
from; `corpus.account_traffic` fails CLOSED (an unreadable registry means
test), and `hermes-dogfood` is client traffic. The writes (observe +
record, one transaction) run on one background thread through a bounded
queue, never on the response path. The idle-time job `deep.learn`
(`mcp/deep_learn.py`, cpu lane, scheduled by the worker like `skill.learn`)
moves `struggle_threshold` by 1 (bounds 2-6) when at least `LEARN_MIN_N` =
5 of the deciding label accumulated since its last change (`kickoff_tokens`
is retired, 2026-09-27: its old adjustment rows stay listed, set nothing,
and cannot be reverted); each adjustment records its
old and new value and its n, and is reversible (`POST
/dash/api/deep/revert {id}`); an environment variable pins a parameter.
It PROPOSES `yama_think_deeply` description variants from missed rows' phrasings
and never arms one (a description is a prompt; PROTOCOL) -- inert for new
rows since the "still broken" phrases were removed. `x_yamadori.deep`
carries the decision, the thresholds (default / env / learned with n),
`last_run`, `yama_think_deeply`'s calls and the row id; `GET /dash/api/deep` the
thresholds, adjustments, proposals, runs per day by trigger and the labels.
The dashboard panel is to follow. Tests: `mcp/test_deep.py` (the triggers,
the records, the learner, the tools), `mcp/test_ledger.py` `[think]`,
`[struggle]`, `[kickoff]` (end to end through the served template: each
request after one extends the slot). **Not yet run live.**

## Heavy-handed, at the right time (operator, 2026-09-27)

Operator, after watching pagoda-h5 (the model probed node_modules for 15
minutes; three "Remember (server tool yama_think_deeply): ..." recall lines
went out at requests 7, 10 and 12 and it never called the tool): "it
probably doesn't know what a server tool is ... we don't need to nudge it to
think deeply if it isn't going to, we should decide when to fire it, and be
more heavy handed, this model isn't taking gentle hints, we need to tell it
what to do ... at the right time". The research agrees that pushed beats
model-pulled (SRA 2604.24594: model-pulled 55.3 vs 62.4/67.2; Skills in the
Wild: force-loaded 41.2 vs chosen 31.6; docs/research/SKILLS-RESEARCH.md).
Three mechanisms, each a line in the MODEL'S OWN VOICE, on the request
where it applies only, generic (never task-specific), never "server tool",
ending on a letter (the prefill rule). The trigger and verify DIRECTIVES are
prefilled as the LAST LINE OF MAIN'S REASONING with the think block LEFT
OPEN (coordinator, 2026-09-27: "so the model keeps thinking toward the
action and then acts, instead of writing code with zero reasoning on that
step"; `proxy.directive_prefill`: reasoning_content the line, content
empty). That llama-server leaves the block open for a reasoning-only
prefill is INFERRED from STEP 0's /apply-template probe (a trailing
assistant message with reasoning and tool calls rendered as `<think>` + the
reasoning and stopped), not measured: confirm on the live /apply-template.
The client is streamed the line as reasoning (llama-server re-sends a
prefill first); it is the slot's own reasoning, which the ledger restores
(restore_reasoning), so no warm. The continuation line stays visible
content. Each has a switch (below). **Offline only; not yet run
live.** Every wording is the operator's example or ours, UNMEASURED.

**The server-tool triggers** (`tiers.BEHAVIOURS` `auto_triggers`; deep.decide
kind `auto`, after struggle, kickoff and area: one run per request, the
helper lane as for every trigger). The moments the retired recall lines
named (`skill_select.server_tool_triggers`) now RUN the job, by the proxy,
before main, delivered as an inserted `yama_think_deeply` / `yama_plan` call
and its result -- the hidden hop `deep_tool_hop` built -- whatever that
switch says:

| trigger | fires when | job | directive (`proxy.AUTO_DIRECTIVE_*`) |
|---|---|---|---|
| probe | a step reads inside `node_modules/<P>/` (site-packages) -- not a package.json version grep | investigate P's API as the task uses it | "The findings above answer this, so I'll use them and write the code now" (`AUTO_DIRECTIVE_THINK`) |
| scratch | a throwaway write (temp dir, `_tmp_*`, `*.tmp.*`) importing P (or no package the work names) | investigate P (or the library it tests) | the same |
| next_piece | a project write in a directory the newest plan's FILES never named | plan the next piece | "Next I'll do the first step of this plan" (`AUTO_DIRECTIVE_PLAN`) |
| plan_done | every file of the newest plan written | plan what comes next | the same |
| implement | a user turn after an answer with BUILD INTENT (the decider's, `decide_turn.build_intent`; `route.work_intent` only when the decider cannot answer) | plan it | the same |

The directives were REWORDED 2026-09-28 (operator, after pagoda-h6: "don't
say 'I have it from source' or anything that confuses the model into
overthinking its way out of using the findings"): the 2026-09-27 lines ("I
have P's API from its own source now, so I'll stop reading its files ...")
named where the findings came from and what to stop doing, and after every
one the model kept probing node_modules and writing scratch scripts. A line
now says only that the result just given answers the question, to use it,
and the next concrete action -- no provenance, no package name, no
prohibition; the operator's candidates as given. UNMEASURED.

Once per key per conversation (think:<P>, scratch:<dir>, piece:<dir>,
plan_done:<plan>, implement:<n>), marked when the job RAN (a busy lane
skips the request, not the key); a package an area run covered is covered.
**Once per key survives a compaction and a fork** (2026-09-28, pagoda-h6):
the state is the conversation's deep state (`deep:<lineage>`), which a
compaction that keeps the key, our summary line, or a linked continuation
already share (a new EPOCH, the keys kept). On h6 Hermes' summariser call
hung up after 600 s, Hermes compacted on its own, its Responses
`prompt_cache_key` changed and the kept tail carried our id only in its
tool-call ids -- `session_identity` named it a FORK (unchanged), a new
lineage with an EMPTY deep state, and the probes for koota, math, three and
@react-three/fiber all fired again (14-18 minutes each). A fork shares its
history with the conversation it came from, so on its first request it
INHERITS that conversation's once-per-key state (`deep.inherit_state`,
`deep.INHERITED`: the fired trigger keys and plan tracking, the areas and
escalating skills researched, the verify moments passed, the
yama_think_deeply offer; never the message-index state), once
(`proxy._inherit_once_per_key`; `x_yamadori.deep.inherited_from`).
KNOWN: `implement:<n>` counts user turns, which a compaction renumbers.
Test: `mcp/test_ledger.py` `[refire]` (a compaction that keeps the key, then
the h6 shape: no second fire for a key; a new package still fires once).
Only where main has the tool (`think_tool_offered`, not withheld).
`x_yamadori.deep.auto` {trigger, key, job, tool, evidence, seconds,
delivered_as, directive}; `x_yamadori.deep.signals.auto` the candidates and
the keys fired before. Tests: `mcp/test_tool_recall.py` (detection),
`mcp/test_ledger.py` `[auto]`, `[auto plan]` (each fires once, hidden hop +
directive, the next request extends the slot, streamed == blocking, no
recall line goes out).

**The verify directive** (`verify_directive`; `mcp/verify_moment.py`; operator:
"if it's a web page the browser works, if it's a native app, then it needs
something else ... How we going to get that to land?"). NOT a grader: the
model runs its own output with its own harness tools; nothing of ours
judges it. The ARTIFACT KIND is mechanical (`verify_moment.KINDS`, each row
naming its file signal: an Expo / React Native manifest, an .html page, a
Vite or Next config, Cargo.toml, go.mod, pyproject / a .py file); the
MOMENT is mechanical, once per moment per conversation: the kind's ENTRY
first written (index.html / src/main.*, App.*, src/main.rs, main.go,
main.py) or every file of the current plan written. The TOOL is the Bonsai
decider's (`Turn.choose`, rotated, label prior divided out): "Which of these
tools can run this <kind> and show whether it works?" over the CLIENT's own
tools (name + the first sentence of its description) and "None of these
fits." -- no per-harness table; more tools than the 25 single-letter
labels "None" leaves go in rounds, then the winners. The line
is NAMED ("I'll open it with <tool> and check what it shows before going
further"; "run it" / "launch it" by kind) or GENERIC, the operator's words
("I need to build and run the app to capture its output and check my
progress") when the decider picks none, ties, cannot answer, the kind is
unknown, or an earlier named line did not land (the turn that opened with
it called something else). On a request where a server-tool trigger also
fired, the verify line replaces its directive (the job's result is still
inserted); deep thinking's visible opening is never replaced. The armed
craft for checking that kind of work (phase verify) rides in the same
injection; where none is armed the record names the gap (every kind today).
`x_yamadori.deep.auto.verify` {moment, kind, signal, options, pick, tie,
line, form, why, delivered, skill}. Tests: `mcp/test_verify_moment.py`
(Hermes-, OpenCode- and Pi-shaped tool lists, a fake decider behind the real
Turn.choose), `mcp/test_ledger.py` `[verify]`.

**Continue a stated step** (`continue_stated_step`; operator-approved fix,
2026-09-27, "planning without action"). An agent step ended finish=stop with
NO tool call and text that only states its next action ("Now let me
understand the task...") -- deploy_check's agent loop [echo] step 3,
pagoda-h3 -- and the harness, seeing an answer, ended the run. TRIGGER
(structural): an agent step (route `agent_step`, or the conversation carries
client tool calls), finish `stop`, no tool call, visible text, the client
offered tools; not at `minimal`/`low`, not on a landing. JUDGE: the decider,
one choice over the text and its reasoning's tail (`decide_turn.judge_stop`:
finished and reporting / says what it is about to do next / asks the user /
none of these; the state bounded like every decider state), on the
transient slot, released. CONTINUE when it said what it is about to do:
main's own turn is PREFILLED -- its text, a blank line, `proxy.CONTINUE_LINE`
= "Let me do that now" (a letter ending, the prefill rule; a period was
never measured; the voice of the operator's approved agent-step nudge) --
and generated again with the client's tools and the same budget rule. ONCE
per request: a continuation that stops again is delivered as is. The client
receives ONE turn (the text, the line, the continuation -- normally a call);
nothing is hidden or rewritten, so the slot holds exactly the turn delivered
and the ledger records it like any turn (no warm unless something else
changed it). Streamed: the text went out as generated; the continuation's
re-sent prefill (reasoning and text) is skipped (`proxy._Resent`), so
streamed == blocking. `x_yamadori.continued` {judged, distribution,
raw_pick, pick, continued, line, call_followed, finish, ms, why}; None when
the trigger did not hold. Cost, UNMEASURED: the continuation diverges from
the slot at the stated step's end-of-turn token, so the server restores its
checkpoint near the previous prompt's end and re-reads that turn. Tests:
`mcp/test_ledger.py` `[stated]`, `mcp/test_responses_api.py` (the
translation), `mcp/test_decide_turn.py` (judge_stop).

## The project: which writes are the project's

`mcp/progress.py` keeps, per conversation, which files are the project's --
state only, no text the model reads -- for deep thinking's label rule 2
(#52, switch `helped_needs_change`). The fix-up's scope (#56,
`tool_code.scope`, switch `fixup_project_only`) and the agent step's
thinking cap by what the step answers (#53, `progress.step_kind`, switch
`step_thinking`) were REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md: their
tables and numbers were chosen from single Octopus runs). The project
inference below stays because label rule 2 reads it; removing it with
`helped_needs_change` is the operator's call (the audit's "unassigned").

- **The project** (`progress.is_project`): a written path is the project's
  unless it is in a temp directory, a dot-file, a scratch-harness name
  (`progress.SCRATCH_NAMES`, each row seen in an Octopus run) or outside a
  known working directory -- a file the task names always is. The task's
  files: relative paths in its text, and a project TREE's files qualified
  by their folders (`progress.tree_paths`). The working directory: one the
  conversation states, else inferred from the first write of a file the
  task names (the most qualified name decides), widened to the deepest
  common directory when a later named file lands outside it (v0f-V0: a bare
  `config.js` made it `.../space-shooter/js`). `progress.learn` reads each
  tool result's successful writes into it; a write is found by
  `tool_code.detect(whole_pages=True)`: an `.html` page is one file.
- REMOVED 2026-09-27 (operator: "keep our system prompts clean; fixes go
  through skills" -- task-targeted steering in prompts; skills are the
  channel): the tool-result SITUATIONS -- the progress line (#50, "No
  project file has changed in the last N steps ...", switch
  `progress_note`), the unchanged-read line (#54, "Unchanged since step N
  ...", switch `unchanged_read`), the work log's files-read listing -- and
  the tier flag `situations`. A tool result that already carries one in the
  ledger replays it byte for byte (`ledger_restore` replays stored text;
  nothing regenerates it), so those conversations' slots keep their
  prefix. The skill `fix-located-defect-first` (armed) remains the #50
  channel: `bench/harness_shapes/hermes/chat__skills-located-defects.json`
  gates that it is selected on the v0e prompt-2 shape.

**Every overthinking behaviour has a switch** (`tiers.BEHAVIOURS`, default
ON except `tiers.OFF_BY_DEFAULT`: `deep_tool_hop`): X-Yamadori-Features
`{"<name>": false}` or its environment variable `=0` turns one off --
`work_log_reinject` (`YAMADORI_WORK_LOG_REINJECT`), `step_nudge`
(`YAMADORI_AGENT_STEP_NUDGE`), `seed_frame` (`YAMADORI_SEED_FRAME`),
`helped_needs_change` (`YAMADORI_DEEP_HELPED_NEEDS_CHANGE`),
`restore_reasoning` (`YAMADORI_RESTORE_REASONING`; "Past reasoning is
restored", above), `continue_stated_step` (`YAMADORI_CONTINUE_STATED_STEP`),
`auto_triggers` (`YAMADORI_AUTO_TRIGGERS`), `verify_directive`
(`YAMADORI_VERIFY_DIRECTIVE`) ("Heavy-handed, at the right time", above),
`mcp_tools` (`YAMADORI_MCP_TOOLS`; the MCP host's tools, medium and up, the
surface row "main, where the MCP host serves"). Removed 2026-09-27 with the code they switched (docs/CONSTANTS-AUDIT.md):
`step_thinking`, `fixup_project_only`, `plan_tools`, `plan_budget`.
`x_yamadori.progress` records each switch and its source (header / env /
tier / default), the agent step's thinking cap and nudge, and the project
(root, how it was found, the files the task names, this request's project
and scratch writes). Tests: `mcp/test_progress.py`, `mcp/test_ledger.py`
`[situations]` (none is added; an old row replays), `[removed]` (no
removed text in any request the suite sends), `[step]`, `[work log]`,
`[fixup]`, `[seed]`. **Built, not yet run live.**

## The code-work router and tool-call repair

**One class per request** (`mcp/route.py`), decided once in `proxy.prepare`
and recorded as `x_yamadori.route {class, because, signals}`. First rule that
fits wins: `utility` (`selection.utility_call`) → `agent_step` (ends on a
client tool result; a harness notice or bare "continue"; `acts_locally`) →
`code_edit` (the prompt carries code that PARSES, and the instruction asks
for work on it) → `code_generation` (a write verb + code noun, "export
`name`", "write a Rust ...") → `library_question` (a question, and the gate
offered for a reason about it, or a held source defines a name it uses) →
`prose`. Features read the class: **fan-out and the repair pass on the code
classes only; the definitions injection on `library_question` only.** Deep
thinking no longer reads it (Phase 0.6: triggers, on any class). A header
that forces one still forces it.

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
challenges 183/183 route to code. **What changed because of it** (tier max,
Laya absent): deep thinking unforced on the domain bench went from 80/100,
9/40 and 166/183 tasks to 0 (the header arms that force it are unchanged);
context-economy 25/26 and held-out 91/120 are unchanged; design questions
(prose) no longer fan out. None of the 196 Hermes turns is a code request:
Hermes writes code through tool calls, which is what the next paragraph is
for.

**Tool-call repair** (`mcp/tool_code.py`; never for a utility call). When a
generation ends with CLIENT calls that write or patch a file, the code is
checked before anything is forwarded: whole files by the parser, ruff
`E9,F63,F7,F82` for Python, and the formatter's own parser; edits (old/new
strings, diff hunks, SEARCH/REPLACE) block only when the text they replace
parses and the replacement does not, otherwise they are flagged -- in the
record only: a flagged fragment gets NO note (2026-09-26: "Checked" is for
problems that remain, and Pi's model imitated the fragment note, #36). An
`.html` write is checked through its inline scripts (tree-sitter's HTML
grammar; no `src`, a JavaScript type or `module` -- a shader or import map
is not), each as a JavaScript file with the same rule, a repair spliced
back into that script's span only (`tool_code.script_units`; Pi's
canvas.html went out unchecked). Detection: a
known-names table (Hermes, MCP filesystem, Anthropic's editor, Claude Code,
Gemini CLI, Cline/Roo, OpenCode, Codex, Continue, Pi -- `edit(path,
edits[{oldText,newText}])`, its string and legacy forms, and `write`,
VERIFIED from pi-coding-agent 0.87.1 `dist/core/tools/edit.js:10-21`,
`write.js:8-11`; the table marks which
entries were verified from source), then argument shape; an unrecognised
call with a code-sized string is logged in `x_yamadori.tool_code.unknown`.
At `medium` (`tiers.check_code_offered`) what the check found is only
NOTED. Every blocking unit is repaired, whatever its path (the "project
files only" scope, `tool_code.scope` / `fixup_project_only`, #56, was
REMOVED 2026-09-27, docs/CONSTANTS-AUDIT.md: its scratch-name table named
one run's files). At `high` and up (`tiers.repair_on`) each blocking unit goes to the
second brain's **fixup** job with ONLY its code, its errors and the user's
request, at the request's own effort, up to `REPAIR_ROUNDS = 3` (not
measured; the final-answer repair is the same job). The repaired version is
written back into the call ONLY when it came from a closed fence, the reply
was not cut off (`finish_reason: length`) and it has no errors left
(`shomen.FIXUP_MIN_KEEP`, "at least 1/2 of the original's non-blank
characters", was removed 2026-09-27: a choice, not measured); otherwise the
model's own content goes untouched and the note says the repair was not
used, and why (`shomen.fix_rejection`). Main generates the call once and never sees a
failed attempt. A unit inside a diff or a multi-file patch has nowhere to
write a fix and is only noted. **Change what the model wrote only when it
is broken** (2026-09-24, #24 in `docs/SELF-IMPROVEMENT-LOG.md`): the client
receives the calls exactly as written, except the lines a repair of real
errors required (the fixup is told to change only those and keep every other
line, formatting included). Formatter output is never applied -- it rewrote
every whole-file write (5-65 lines each, V0 pilot), the model then patched
with its own pre-format text and the patch failed -- and is recorded in
`x_yamadori.tool_code.files[].format_would_change`, never in the note
(#34: "fixed in 1 round; prettier would change 69 lines; sent as written"
read as a contradiction, and the model reads these notes). The note before
the calls is mechanical, in the fold-back phrases and with NO concept-seed
line (the seed stays in the fixup job's own user message), and says plainly
what was sent: `Repaired js/game.js (javascript): 2 syntax errors fixed in
1 round (the repaired file was sent; formatting left as written).`,
`Verified app.py (python): parses, lint clean; sent as written.`, "Checked
...: N problems ...; sent as written" when problems remain. The model
imitates these notes in its own content (Octopus v0b-V0: six "Repaired"
lines no tool_code record backs, #36); those lines reach the client as the
model wrote them (the `_ImitatedNotes` filter was removed 2026-09-27);
`x_yamadori.tool_code` is the record, never the transcript's text.
Nothing is changed after the client has the call, and the note claims syntax
and lint of each file on its own, nothing more. The proxy then warms the slot with the turn as delivered.
`mcp/test_tool_code.py`, `mcp/test_stream.py` and `mcp/test_route.py` hold
the checks. **Not yet run live.**
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
used by `proxy.py` and `shomen.py`; operator decision 2026-09-23). The main
loop -- which since 2026-09-24 runs only for our tools on main (the image
tools, the delegate arm) -- and every deep-thinking run each count their own;
fan-out candidates and fixup rounds make no tool calls. Neither number is measured. At the cap the loop **lands**:
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
  **Internal generation** (`summarize_text`, shomen, the worker) goes through
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
  back to the 0.70/0.30 fractions):
  `thinking = max(share[role] // share_n - prompt - answer, 1024)`, then
  CAPPED (operator 2026-09-25, "this model overthinks"; choices, unmeasured
  for quality): a request routed `agent_step` thinks at most
  `tiers.AGENT_STEP_THINKING` = 6,144 (the by-result caps of #53 -- 2,048
  after a read, 6,144 after an error -- were REMOVED 2026-09-27,
  docs/CONSTANTS-AUDIT.md: chosen from one run);
  and an agent step's nudge is
  `tiers.AGENT_STEP_NUDGE_MESSAGE`, the operator's voice naming the action
  ("... Let me make the call I've already worked out -- the edit itself, if
  I know the fix -- and only keep thinking if that call would be
  misleading."; approved by the operator as written, AskUserQuestion
  2026-09-26, UNMEASURED WORDING; switch `step_nudge`; the general `NUDGE_MESSAGE` and
  `BUDGET_MESSAGE` are unchanged); a RESEARCH hop of the second brain
  (investigate, plan) is nudged with `tiers.HELPER_NUDGE_MESSAGE` ("... Let
  me make the search I've already worked out, or write the hand-off now, and
  only keep thinking if that would be misleading."; operator 2026-09-26,
  because "go with the best answer" inside a search loop plausibly made
  handle d28941fb stop with no call -- n=1, a hypothesis; UNMEASURED
  WORDING; off with `YAMADORI_HELPER_NUDGE=0`; fixup, alternative and
  tiebreak end in an answer and keep `NUDGE_MESSAGE`); each
  second-brain hop at most `tiers.JOB_THINKING[job]` (fixup 3,072,
  investigate 6,144, plan 12,288, alternative/tiebreak 6,144; `shomen.run`
  names the job on its thread), any other helper request
  `HELPER_THINKING` = 6,144 (all 1.5x their first values, operator
  2026-09-25: "1.5 those numbers above and dial it in" -- the direction is
  the operator's, the first values were ours; the plan's #60 cut to 4,096
  was reverted 2026-09-27, docs/CONSTANTS-AUDIT.md; `YAMADORI_PLAN_THINKING`
  pins it). The cap is per
  REQUEST (`reasoning_budget_tokens`), not per slot. Evidence for capping:
  Octopus v0b-V0 (n=1 run) -- reasoning 96% of output, agent steps median
  730 / p90 3,713 / max 24,620 completion tokens; after midnight the second
  brain wrote 83,031 tokens in 35 generations vs main's 107,935. A
  benchmark's `reasoning_cap` header still wins over every cap.
  Upstream gets `max_tokens = thinking + answer`,
  `reasoning_budget_tokens = thinking` and a `reasoning_budget_message`.
  Deep thinking (`shomen`) and fan-out both draw from the helper's share, one
  second-brain job at a time (`admission.HELPER_LANES = 1`; in one request
  they run in sequence). Fan-out is sequential (`mcp/fanout.py`, operator
  decision 2026-09-23): the original answer is candidate A, the second brain
  writes B with the helper's share, the code check grades the two, and
  only when that does not separate them does it write a tie-breaker C from
  both candidates and their check results. At most two contexts are live.
  Every second-brain run carries a fresh concept seed in its user turn, and
  its work always crosses back (next bullet). The 0.70/0.30 split replaced
  the operator's 5/8 + 3/8 of 2026-09-22, not a measurement either, which
  replaced a 1/2 + 2 x 1/4 split the same
  day (a second concurrent helper needed two conversations investigating at
  once), which had replaced a fixed `R_CAP=8192` breaker and a 60/25/15 split
  that had no measurement behind it (`docs/CONSTRAINTS.md` item 19). Natural thinking ran 682–2,826 tokens
  at n=7 (§1), so no measured result says a shorter thought is better.
  `config.yaml` launches with `--reasoning-budget 32768` and the same
  message, the server default for a request that sends no budget of its own.
- **The second brain's work is never thrown away** (operator decisions
  2026-09-23 and 2026-09-24). "Only the conclusion crosses" still holds for
  summarization and lookup. Work it DID crosses back as a distillation plus
  facts, folded into main's own turn (see "The second brain" above):
  - *Deep thinking* always hands off, in four fixed sections (FACTS, each
    ending `path:line` or labelled `(reasoning, not checked against
    source)`; SEARCHED, FOUND NOTHING; OPEN QUESTIONS; NEXT STEP),
    PREFILLED as main's `reasoning_content` for this answer; the visible
    answer opens with the seed line and "After thinking deeply,". Citations
    are still checked against what was retrieved; an unchecked fact is
    labelled, not dropped. A verified fact carries the lines it cites
    INLINE, read by the verifier from the held index and labelled
    `pkg@version path:a-b` (main and the user cannot open our paths); a
    citation the verifier cannot read is REMOVED and its fact labelled
    reasoning (shomen "THE EVIDENCE"; caps are choices). The hand-off ends
    by telling main that the user sees only its answer, which states the
    findings in full (`proxy.FOLD_BACK_TAIL`; live gate 2026-09-24: an
    answer pointed at "the hand-off above"). No search: it crosses under "reasoning, no
    sources checked". Turn cap, helper-budget stop, or a hop that ENDS IN
    ITS REASONING (no tool call, no text; since 2026-09-26 -- until then it
    failed at once as "ran out of budget while reasoning", whatever ended
    it: deploy check, handle d28941fb, 15 searches with no budget near its
    end): the run LANDS once, tools withdrawn, and the landing prompt
    requires the hand-off. Each such hop's end is recorded in the trace and
    the failure reason (`finish_reason`, sizes, a tool call left inside the
    reasoning); "token limit" is said only on `length`. Nothing written: the
    proxy builds one from the trace, labelled machine-built
    (`shomen.machine_handoff`). It carries every path retrieved and one
    search-log line per distinct call with its count, whole (the MACHINE
    EVIDENCE excerpts -- `MACHINE_EXCERPTS` = 4 passages picked by the
    question's words -- and the `paths[:12]` / `asked[:200]` / `why[:360]`
    cuts were REMOVED 2026-09-27, docs/CONSTANTS-AUDIT.md: CHOICES from one
    handle, n=1); OPEN QUESTIONS keeps the user's question (after a
    trigger's `QUESTION:`); NEXT STEP is one main can take without a file
    tool. Its tail is `proxy.FOLD_BACK_TAIL_MACHINE` (no conclusion was
    written; work it out from what is known, say what was not checked
    against the source), and after a machine-built `yama_think_deeply` result the
    next hop's reasoning is `deep.THINK_REASONING_MACHINE`
    (`deep.think_reasoning(concluded)`). The hand-off screen masks the tool
    name of our own "(search log)" lines (it had removed the first one of
    every hand-off as `ai_directed`); their queries are still screened.
    `x_yamadori.investigate.why` says why a run wrote no conclusion. All
    wording is a CHOICE, unmeasured. NOTHING CUTS A HAND-OFF OR A PLAN
    (operator, 2026-09-27: "You made that limit up, unfounded."):
    `MAX_FINDING_CHARS` (6000) and its "[hand-off cut ...]" marker are gone,
    and so are the per-section item caps and their "(+N more)" line --
    pagoda-h2's 8,958-character kickoff plan was cut mid-sentence, main
    echoed the marker and stopped. The bound is the job's own generation
    budget (`tiers.JOB_THINKING` plus its answer allowance); on main a
    `yama_think_deeply` / `yama_plan` result is exempt from
    `repeats.RESULT_CAP`. One that does not fit main's window is the window
    check's: hop 0's `max_tokens` is cut to what is left (`proxy.fit_window`),
    a `length` finish is a budget event, a later hop lands (`context_full`),
    and the client's next request is refused `context_length_exceeded`
    (`check_client_prompt`), which a harness compacts on. `mcp/test_stream.py`
    `test_nothing_cuts_a_hand_off_or_a_plan`; `mcp/test_ledger.py` scans every
    request for "cut at" / "continues below". `x_yamadori.investigate.handoff`
    has the counts.
  - *Fan-out, prose*: B's differing points are PREFILLED after main's own
    answer, in its own turn ("Compared two approaches: ... Weighing them"),
    and main continues -- it weighs them itself. The weigh turn (a user
    turn in main's context the client never saw) is gone (operator,
    2026-09-24). A continuation that fails leaves the points as a note.
    *Code*: the winner is delivered (in place on the blocking path, after
    the answer on a stream) with one line: why, and the APIs the losers used
    that it does not. `x_yamadori.fanout.handback`.
  The context-economy numbers in `mcp/shomen.py` (main context 8,326 ->
  2,410 tokens, n=26) were measured under the old conclusion-only rule and
  **must be re-measured** (`bench/context_economy.py`) before they are cited.
  The "material difference" thresholds in `fanout.handback` are choices, not
  measurements.
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
  `stop`). Neither gets the code check or fan-out.
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

- **Stateless.** `store: false` only; `previous_response_id`,
  `conversation`, `background`, `prompt` and `item_reference` are 400
  `unsupported_parameter`. The session is `prompt_cache_key` (Codex always
  sends one), else the headers, else our id in the `call_id`s.
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

## The A4000's room

`mcp/gpu_room.py` decides what is loaded on the A4000 (operator,
2026-09-24: "if it fits with headroom fine, if it doesn't drop them and load
in what you need on use"). Every caller about to make llama-swap load an
A4000 model wraps that request in `gpu_room.use(model, upstream=...)`:
`images.generate`, `model.post` (vision), `code_search._post` (embeddings,
reranker), `scripts/index_code.py`. It reads `/running` and nvidia-smi by
UUID, sizes the model from `SIZES` (each row names its source; only
`imagegen` is measured), and if free − need < 1,331 MiB it unloads other
A4000 models least recently used first through `/api/models/unload`, or
refuses with `A4000_NO_ROOM` / `A4000_BUSY`. The main model and Laya are
never touched; a model in use (a lease) is never unloaded. A file lock
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
  request whether fan-out and deep thinking actually run). The same table is
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

  | `reasoning_effort` | thinking sent | library help | skills | code check | fan-out | deep thinking | addendum | images | concept seed | adds |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `minimal` | off | – | – | – | 1 | – | – | yes | – | thinking off, the vendor's instruct sampling, nothing of ours (fastest; least injection-resistant) |
  | `low` | medium | – | – | – | 1 | – | – | yes | – | nothing: the model as it ships, the benchmark baseline |
  | `medium` | medium | definitions | – | note | 1 | – | – | yes | – | library definitions for a library question, a note when a client write does not parse (skills: off at every tier, 2026-09-29) |
  | `high` | medium | definitions | – | repair | up to 3 | – | yes | yes | yes | the second brain: code that does not parse is repaired, a second approach is compared, and the addendum says so |
  | `xhigh` | medium | definitions | – | repair | up to 3 | allowed | yes | yes | yes | everything `max` has, at medium thinking: the effort-matched pair to `max` |
  | `max` | xhigh | definitions | – | repair | up to 3 | allowed | yes | yes | yes | everything, at xhigh thinking |
  `code check`: syntax and lint; the model's code is changed only where a
  repair of real errors needs it -- formatter output is never applied, only
  reported (#24, 2026-09-24). `library help` also covers, on every route
  class from `medium` up, the held packages a conversation USES (imports in
  tool results and written files, manifests): LIBRARY USE (#19), an
  unmeasured choice; each imported name's definition whole, bounded only by
  what the window leaves (`proxy._injection_room`; the `USE_MAX_*` caps and
  the package overview were removed 2026-09-27, docs/CONSTANTS-AUDIT.md).
  `images`: `yama_generate_image` / `yama_describe_image` are offered on every tier
  where `YAMADORI_IMAGEGEN_URL` is set. `library help` is the definitions
  injection (a choice, unmeasured); where deep thinking runs, it does the
  reading instead. `concept seed`: every second-brain job carries one. Because the tiers also raise thinking effort, a `minimal` vs `max`
  comparison mixes effort with tools; benchmarks report an effort-matched pair
  (everything forced off vs on at the same effort) beside the ladder.
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
   `mcp/test_tools_live.py`, `bench/test_laya_head.py --serve` and the
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

- `judge` (Laya as a truth judge) scored 6/10 against a 5/10 coin flip, with
  systematic false positives on negative cases. It scores what a passage is
  *about*, not whether a proposition holds. `scripts/eval_judge.py` is the
  gate to restore it.
- `apply_edit` was cut because the harness owns writing. There is one write
  path per repo.

**The reranker is not cut, not trusted, and not used.** `docs/FINDINGS.md`
#20 found that `/v1/rerank` scores depend on batch composition: the same 89
documents scored 15–16/89 batched against 68/89 one at a time. That voids
every reranker number in the repo, in both directions:

- the old "no gain above `top_k=2`" cut (11 queries, against an index of
  zero vectors)
- `bench/retrieval_results.jsonl` (n=356, rerank hit@1 48/356 vs embedding
  264/356)

The resolution is in `docs/PLAN.md` under "Cut criteria". Do not cut or
defend the reranker until the rank path is fixed and re-measured.

## Laya

**Being replaced by E1** (`mcp/e1.py`, `docs/E1.md`): logistic heads on the
resident embedder's vector of the request. route_in v1 scores 104/120 on
held-out set 1 vs Laya's 80 (p=0.0001) and 110/141 on the new set 2 vs 86
(p=0.003). No Laya combination beat E1 alone, so the verdict is to retire
Laya. `YAMADORI_E1=1` puts E1 in Laya's place and takes Laya off the request
path. It is **off by default** until `mcp/test_live_stack.py --only e1`
passes; the retirement checklist is docs/E1.md §9. Until then, the rules
below still hold for Laya.

**Use Laya only where the state is FIXED and the options vary**, as a
`choice` over a small closed set. Its scores are not comparable across
different passages:

- Holding a passage fixed and varying the query separates real from nonsense
  by 0.497. **Unverified:** no script in the repo produces this number
  (`docs/LAYA-FACTCHECK.md`).
- Varying both collapses the gap to zero.

Never threshold a Laya score across queries.

It also silently drops the tail of a long `state`. The window is about 512
tokens, cut from the end: a question after ~620 tokens of preamble is
discarded, and every input then scores identically
(`bench/laya_factcheck/tail_probe.py`). Put the thing being judged first.

**`route_in` head.** It was retrained on 2026-09-22 on 289 labels
(`index/laya/route_in.json`; the previous head is in `index/laya/_backup_*`).
It was scored by `bench/eval_route_heldout.py` on 120 held-out
package-domain labels, which no training file matches:

| condition | investigate-vs-not |
|---|---|
| new head | 80/120 |
| old head | 64/120 |
| `selection.decide` | 89/120 |
| bare regex, no symbol lookup | 73/120 |
| rule + head live, disagreement escalates | 91/120 (p=0.79 vs 89, not significant) |

The head's features are Laya's CLS vector plus its three option logits. It
beats the bare regex (80 vs 73, not significant) but not the rule with its
symbol lookup, so it never decides alone. The live row is
`bench/laya_factcheck/live_route_check.py`; old vs new head is p=0.011.
Since Phase 0.6 (2026-09-24) the proxy's deep-thinking decision does not
consult it at all (triggers, above); these rows describe selection's legacy
path, which the evaluators still replay.
Deep thinking still injects findings that carry no citation
(`docs/LAYA-FACTCHECK.md`); since 2026-09-23 each such fact crosses labelled
as unchecked reasoning rather than passing as read.

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
  by a decider (`mcp/skill_deciders.py`: the evidence stub by default, CLM
  with `YAMADORI_SKILL_DECIDER=clm`, not yet run live; the fallback model
  as last resort), only the questions the conversation's STATECHART
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
  called): its triggers now RUN the job -- "Heavy-handed, at the right
  time" below. The detection is `skill_select.server_tool_triggers`; gates
  `mcp/test_tool_recall.py`.
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
  WHOLE knowledge base (operator, 2026-09-25): the second brain searches
  them through `find_in_knowledge_base` and nothing else -- never this
  repo's docs. Three skills are AUTHORED from our own evidence
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
- **CLM, the decision model** (`docs/CLM.md`). `mcp/clm.py` is the client
  (`decide(state, options) -> probabilities`, `decide_many` encodes the
  state once): Qwen3-8B Q8_0 last-token embeddings from llama-swap's
  `clm-encoder` behind gpu_room, the CLM-v0.1-8B heads in numpy
  (`mcp/clm_heads.py`, `index/clm/heads/`), option vectors cached per text
  in `index/clm_actions.npz`. The selector reaches it as
  `skill_deciders.ClmDecider` (`YAMADORI_SKILL_DECIDER=clm`; the stub stays
  the default). Measured: 40/40 decisions equal an fp32 HF reference; on the
  daily eval 445/464 against the in-sample stub's 464/464 (MUST 41/54,
  MUST-NOT 385/385), the same with the skills-fine-tuned heads (n=1 each).
  Bench: `bench/clm/` (fidelity, tier0, swap, finetune_data).
- **Concept seeds.** `mcp/concept_seed.py` draws from the 27B's own token
  embeddings, `index/token_embd.npz`. Extract them once with
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
  `budget.budgets()["child"]` (layout v2: the decider lane). The Skills page's CREATE has a PROMPT + LINKS mode (a
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
