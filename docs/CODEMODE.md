# The server toolbox: `yama_compose_tools` (codemode, server side)

Operator, 2026-10-07 (verbatim): "pi now has codemode https://pi.dev/docs/latest/codemode we could easily
structure our proxy to allow the server to facilitate this feature built into the proxy. It even gives an api for
image gen, jjava etc... we should make codemode in our server. QuickJS sandbox is a good idea, I love this."

And, the same day, on the first name: "yama_run_code is a terrible name, a part of code mode are the calls to find and
describe tools, so let's be thoughtful of the feature and not just think it's running a JS script in the server." And:
"We can use yama_prefix for server tool identity here too." And on the prompt: "how does the system prompt change
without getting too big?"

**The feature is DISCOVERY + COMPOSITION.** The model finds out what the server can do (`tools.search`,
`tools.describe`, `tools.namespaces`, `tools.all`) and gets several facts from it in one go (`tools.yama_*`, called in
parallel, filtered in code). The JavaScript runner is only the mechanism. The tool is named for what the model does:
**`yama_compose_tools`** (AGENTS.md "Naming": `yama_` prefix, verb first, spelled out, snake_case; the object is the
server's tools). Alternatives considered: `yama_use_toolbox` (the verb says nothing about composing),
`yama_call_tools` (loses discovery), `yama_run_code` (rejected by the operator). Files: `mcp/codemode.py` (library:
registry, BM25 search, declarations, the bridge to the sandbox, the result format), `mcp/codemode_sandbox.py` (the
sandbox: QuickJS on WASI under wasmtime), the bridge in `mcp/proxy.py` (`run_compose_tools`, `_codemode_apply`),
`mcp/tiers.py` (the switch).

Source of the design: https://pi.dev/docs/latest/codemode (read 2026-10-07). What we take, what we change:

| Pi | here |
|---|---|
| ONE tool `codemode` in the model's list; its description carries the TypeScript declarations under `codemode.inlineBudget` (default 3000 tokens) | ONE tool `yama_compose_tools`; description = trigger conditions + the declarations of the functions most relevant to the conversation, under a budget DERIVED from the prompt: what the tools it covers cost today (their served definitions + the MCP system line, the proxy's own estimate), at most Pi's 3,000. Measured 2026-10-07: our additions on main are about 9,200 characters (~2,400 tokens): 4 package tools 4,768, 2 image tools 2,827, `yama_recall_craft` ~1,100, the MCP line ~250. The covered set is the MCP lookups (+ craft where `only` hides it): ~1,585 tokens on the default config, so the toolbox is size-neutral and a new MCP server adds nothing up front (it is found with `tools.search`) |
| discovery INSIDE the script: `searchTools(query, {limit?, namespace?})` (BM25, default limit 8), `describeTool(name)`, `describeNamespace()`, `ALL_TOOLS` | `tools.search(query, {limit?, namespace?})` (BM25 over each function's name + description + trigger text; no embedder, deterministic; limit 8; a `hint` property when nothing matches), `tools.describe(name)` (full or short name; an unknown one returns the nearest and the next step), `tools.namespaces(name?)`, `tools.all()`. A script may search and call in one run, and the description's example does |
| `tools.<name>(args)`, promises, parallel | `tools.yama_<verb>_<object>(args)` returns a Promise; the callable name IS the tool's own `yama_*` name, the arguments are the tool's own; every pending call runs in parallel, **at most four at once** (Pi: "At most four such calls run at once per script; more calls wait for a free slot"). Namespaces only GROUP (`tools.namespaces()`, `tools.search(q, {namespace})`, the declarations' headings); they are not part of the call |
| QuickJS sandbox, 256 MB, no Node, fs, network or timers | QuickJS compiled to WASI, run under wasmtime in-process: a linear-memory maximum of 256 MiB for the whole sandbox (Pi's), no files / network / environment, the real clock, an epoch interrupt for time. See "The runner" |
| `text()`, `image()`, `console.*`, `return`, `exit()`, `store/load`, `models.classify/generateImages` | `text()`, `console.*`, `return`, `exit()`. No `image()` (an image made by `yama_generate_image` inside a script is shown to the user by the proxy, as the direct tool's is), no `store/load` (one call, one script); jjava and the image tools are ordinary functions (`yama_decide_*`, `yama_generate_image`) |
| result starts `Script completed` / `Script failed`, wall time, `==> text N/M <==` headers, one `<console_output>` block | the same shapes |
| `max_output_tokens` (10000), `timeout_ms` (unset) | output cap = the hidden hop's own size cap, `repeats.RESULT_CAP` 6,000 characters; `timeout_ms` argument, can only lower the derived ceiling |
| config `codemode.mode` | switch `codemode` + `codemode_mode`: `on` (the individual lookups stay offered) or `only` (the ones it replaces are hidden and reached only through the toolbox) |

## What the model sees

One tool. Its description (the model's prompt; AGENTS.md "Tool descriptions are prompts") opens "Answers 'how do I
get several facts from the server in one go?'", says when to call it (two or more lookups at once; only part of a long
result; not knowing which function answers: `tools.search("what you need")` ranks them, `tools.describe(name)` shows
one), that calls are Promises that run in parallel, that output is what the script returns / `text()` / `console.log`
writes and is cut at 6,000 characters, that the sandbox has no network, files or timers, that **bash, read and write are
the client's tools and cannot be called from it**, and the server-tool line. Then the declarations, then an example that
searches and calls in one run:

```ts
type ToolResult = { ok: boolean; text: string; data?: any; error?: { code: string; reason: string; retryable: boolean; next: string } };
declare const tools: {
  search(query: string, opts?: { limit?: number; namespace?: string }): { name; namespace; description; score }[];
  describe(name: string): {...};  namespaces(name?: string): any;  all(): {...}[];
  // packages: Package registries ...
  /** which package is this, and what is its exact name in the registry? */
  yama_find_package(args: { query: string; ecosystem: "npm" | "pypi" | ...; }): Promise<ToolResult>;
  ...
};
// Not declared here, found with tools.search / tools.describe: yama_decide_choice, ...
```

Plus ONE system line where it is offered, positive and procedural (THE TOOL RECIPE rule 4; unmeasured wording): "To get
several facts from the server at once, write a short script for yama_compose_tools; tools.search lists what the server
can do." In mode `only` it replaces the MCP line.

### Namespaces and functions (the registry, `codemode.build_toolbox`)

Built from what the conversation was actually OFFERED (a lookup withheld for a client tool is not in the toolbox
either).

**One source per tool.** A function's declaration and description are read from the tool's own definition (the OpenAI
function the proxy already sends: `mcp_config.definition`, `images.MAIN_TOOL`, `vision.TOOL`, `skill_select.READ_TOOL`,
and the decide tools below) -- no second copy of any description. The same `yama_*` name appears in
`x_yamadori.codemode.calls[].tool`, `x_yamadori.mcp.calls`, `x_yamadori.tools`, whether the tool was called directly or
from a script.

| namespace | functions | backed by |
|---|---|---|
| one per enabled MCP server (`packages` for PackageLens: the config row's `namespace`, else its `id`; `namespace_description`) | `yama_find_package`, `yama_list_package_versions`, `yama_read_package_readme`, `yama_resolve_packages` -- every tool the server's config row lists | `mcp_host.run_tool`; a server added through the dashboard (`mcp_config`) shows up in `tools.namespaces()` and `tools.search` with no prompt change, its description and parameters read from its config row |
| `craft` | `yama_recall_craft({name_or_topic})` | the craft tool, where the craft offer holds (the per-request cap of 3 and the do-not-repeat line apply to every call) |
| `decide` | `yama_decide_noul`, `yama_decide_choice`, `yama_decide_score` ({state, question, ...}; one per question type of jjava, each with the arguments its type needs) | jjava through the Jev API's own entry (`jev_api.systemone`, route `/codemode/decide`: its limits, its queue, batched when `YAMADORI_DECIDER_BATCH` is on), where the decider is on |
| `images` | `yama_generate_image`, `yama_describe_image` | the image tools, where those are offered |

Every function resolves to one `ToolResult`: `{ok, text, data?, error?}`. **A function never throws**: a failure is
`{ok:false, error:{code, reason, retryable, next}}` and `text` carries the same sentence ("Failure returns carry the
next step"). `data` is the structured part (package names found; a decider's probabilities; an image's url). A bare
string is the argument of a function whose only required parameter is a string. Four of the functions' arguments are
the tool's own, so a model that knows `yama_find_package` calls it the same way inside a script.

## The runner (the mechanism)

**QuickJS on WASI, inside a WebAssembly runtime** (operator, 2026-10-08: "Yes download the wasi and quickjs stuff we
need to build our codemode sandbox"; this replaced a first build of 2026-10-07 that ran the native `quickjs-ng`
binding in a child process). Two pinned downloads, both recorded in `models/manifest.yaml` runtimes `codemode-wasi`
and `requirements.lock.txt`:

| piece | what | pin |
|---|---|---|
| the module | quickjs-ng's OWN WASI build of `qjs`: release v0.17.0 asset `qjs-wasi.wasm` (1,569,050 bytes), unmodified | sha256 `42a732a676ec2d93488c19411e0fad283bf72658fdad746f089914b523c783b1` (also the digest GitHub lists for the asset, checked 2026-10-08); URL `https://github.com/quickjs-ng/quickjs/releases/download/v0.17.0/qjs-wasi.wasm`; lives in `bin/` (not committed); `python mcp/codemode_sandbox.py fetch` re-obtains it and refuses any other sha256; verified before every first use |
| the runtime | `wasmtime` (wasmtime-py) 49.0.0, wheel `wasmtime-49.0.0-py3-none-win_amd64.whl` | sha256 `46a4e4220fd1bee45ad9c29a73cadad54373dad840eff104726490527bc7ef89`; `requirements.txt` / lock |

The official artifact exists, so no WASI SDK was needed. The release's other WASI artifact, `qjs-wasi-reactor.wasm`
(sha256 `1a4e0bce...`), exports only `qjs_init`, `qjs_init_argv`, `qjs_get_context`, `qjs_destroy` and allocation: no
eval, so a host cannot run a script in it. The command module is driven instead. The native `quickjs-ng` binding is
removed from the lock (nothing else imports it). The earlier engine comparison stands for the record: the archived
`quickjs` (PetterS) has no CPython 3.13 Windows wheel; `mini-racer` (V8) is a 30+ MB dependency.

**What makes it a sandbox: the module's imports are the whole world.** `qjs-wasi.wasm` imports exactly 23 functions,
all `wasi_snapshot_preview1`. Nothing of wasmtime's own WASI is used: `mcp/codemode_sandbox.py` defines those 23
itself, as a shim that grants nothing:

| capability | the shim |
|---|---|
| files, directories | **no preopened directory**: `fd_prestat_get` / `fd_prestat_dir_name` answer EBADF, every `path_*` answers ENOTCAPABLE, `fd_readdir` and `fd_seek` refuse. `os.open`, `std.loadFile`, `os.readdir`, `os.mkdir` all fail (tested) |
| network | none: WASI preview1 has no sockets, and the shim defines none |
| environment | empty (`environ_sizes_get` = 0); arguments are ours only (`qjs -e <bootstrap>`) |
| stdin / stdout / stderr | IN-MEMORY pipes to the host (the line protocol below); no other fd |
| clock | REAL (operator, 2026-10-08: "why do we fake this, this should be reliable"): realtime / monotonic / cpu clocks read the host's; `poll_oneoff` sleeps for a timer, in slices that see the wall limit and cancel; `Math.random()` is QuickJS's own, seeded from the clock. The script gets `sleep(ms)` (a host-settled wait: other calls keep running meanwhile; clamped to the wall limit) and `setTimeout` / `clearTimeout` built on it, so a script can poll |
| exit | `proc_exit` unwinds the module; the host ignores the code and reads the `done` message |

Every capability the module asked for and was refused is recorded (`Run.denied`). A module that imports anything
else is refused before it is instantiated.

**Limits.** *Memory*: `Store.set_limits(memory_size=256 MiB)` is a hard maximum of the module's whole linear memory
(Pi's figure; the box has about 12 GB spare with Flash-Next loaded and at most two conversations run a script at
once). This closes the process-cap item of the first build: it is the whole sandbox, the heap, QuickJS's tables and
the module's stack included. A memory bomb is QuickJS's own "out of memory" InternalError once growth is refused.
*Time*: wasmtime EPOCH INTERRUPTION. A thread of the run's own ticks the engine every 50 ms (how late an interrupt is
seen, not a limit); the deadline is the wall limit, so an infinite loop traps at it, and a cancelled request makes the
ticker overshoot the deadline so the module traps at its next loop header or call. The wall limit is derived as before
(table below). One `Engine` per run: engines are cheap, the COMPILED module is cached as wasmtime's serialised bytes
(3.6 MB) and deserialised per run, so one run's tick or cancel never reaches another run's deadline (tested with a
spinning script and a working one at once). *Stack*: QuickJS cannot see wasm's call stack, so its own recursion check
never fires; runaway recursion ends as wasmtime's stack-exhausted trap (`stack_overflow`: "RangeError: Maximum call
stack size exceeded (the sandbox ended the script)", the output so far kept). The difference from a native engine: a
script cannot catch that RangeError and carry on.

**Startup, measured 2026-10-08** (`mcp/codemode_sandbox.py`, this machine, n=20 for the per-run figures):
compile + serialise once per process 121 ms (`codemode_sandbox.warm()` does it in a background thread when the
toolbox is offered), sha256 check of the module once per path 88 ms, **a whole trivial script median 3.4 ms (p90 5.4),
of which engine + deserialise + instantiate median 2.1 ms (max 3.7)**; a 200,000-element loop and reduce 23 ms. The
module is cached across calls; each call only instantiates.

**The tool bridge is a line protocol over the module's stdin / stdout** (JSON, one per line; the module reads the
host's replies, the host runs the calls). The first message is `init` {script, names}; then
`{"op":"out"}` (streamed as the script writes, so a killed script keeps its partial output), `{"op":"call"}` (the host
runs up to FOUR at once, Pi's bound, and answers each with `{"op":"result"}` as it finishes), `{"op":"q"}` (the
synchronous discovery calls `tools.search` / `describe` / `namespaces` / `all`, answered at once with `{"op":"r"}`),
and `{"op":"done"}`. The module's driver is JavaScript (`codemode_sandbox.BOOT`) running inside QuickJS: a script's
Promises are queued as it runs, the requests go out as they are issued (so awaited calls run in parallel), and the
driver blocks for the next result only when QuickJS's microtask queue is EMPTY (a 0 ms timer turn of its own event
loop), so every continuation a result enables has run first. A promise nothing can settle is `never_settles`.
It runs in-process on the calling thread (the loop's `_in_thread`, heartbeats and all); the four calls run on four
others. JS reaches nothing but its prelude: `tools`, `text`, `console`, `exit`; `print`, `scriptArgs`, `execArgv`,
`argv0`, `navigator` and `gc` are deleted. `import('qjs:os')` and `import('qjs:std')` still resolve (they are the
module's own), and so a script can write protocol lines itself: the host validates every request against the toolbox
it built (a name not in it is refused, a call past the budget is refused), so forging a message gains nothing
(tested). Every other capability behind them is the shim's.

Limits, each with its source:

| limit | value | source |
|---|---|---|
| memory | 256 MiB, the module's whole linear memory | Pi's (docs); box headroom above |
| concurrent inner calls | 4 | Pi's |
| output items / characters collected | 100,000 calls, 16,777,216 characters | Pi's ("Scripts fail when output exceeds ...") |
| output returned to the model | `repeats.RESULT_CAP` = 6,000 characters, cut at a line with a marker that says "print less: filter in the script" | the hidden hop's own breaker (every tool result is cut there) |
| wall limit | the slowest exposed function's own bound: PackageLens 48.6 s (`mcp_config.PL_CALL_TIMEOUT_S`), `yama_resolve_packages` 970 s (`npm_resolve.FETCH_WORST_S`), images and vision 900 s (`images.timeout()`, `vision.timeout()`), craft 240 s and decide 120 s (the decider's per-read timeout `decider_bonsai.read`/`ask_one` 120 s: craft's question path makes two reads, a decide call one); script ceiling = the maximum over the functions exposed. `timeout_ms` may only lower it. A spinning script traps at this limit: with the resolver exposed that is long, which is why the description invites a short `timeout_ms` | derived |
| inner server calls per request | `tiers.tool_turn_limit(tier)` (10; 20 at `max`) across all of the request's scripts | the loop would allow one lookup per tool turn; a script may not exceed what the loop would have allowed in total. Searching, describing and listing are in the sandbox and do not count |

A script that awaits a promise no pending call can settle fails at once ("never settles"), as Pi's does.

## Accounting, caps, skills, the ledger

* **One `yama_compose_tools` call is one hidden hop** toward `tool_turn_limit` however many inner calls it makes. It
  always counts as a tool turn (it is not a craft read). Calls past the tool-turn limit land with the client's tools
  kept (the 2026-10-07 landing rule).
* **The craft rules apply inside scripts**: `yama_recall_craft` runs `_run_our_tool`'s craft branch, so the per-request
  cap of `skill_match.BODIES_PER_DECISION` (3) and the do-not-repeat line hold, and `x_yamadori.craft.capped` /
  `.repeat` and the landing rule count them. Craft calls are serialised (the cap reads the request's reads).
* **Package skills sections surface in the script result, never in the JS values.** An inner package lookup's section
  (`package_skills.decide`) is computed AFTER the script ends, in the order the calls were ISSUED (deterministic;
  parallel completion order is not), and appended to the result after the output cap, exactly where an individual
  package tool's goes, so a long output never cuts it. The same package under a different major gets its own section,
  as it does when the lookups are made one by one.
* **Screening**: external text (a README) is screened inside `mcp_host.run_tool` as for the individual tool; a script
  that called a package function has its final output screened again (`skill_screen.screen_fetched`: the model may
  recombine fetched text), quarantined as a unit if it fails.
* **The client's tools are not callable**: `bash`, `read`, `write` run on the client's machine. The description says
  so. Pi's own codemode covers that side.
* **A harness with its own code mode never gets this one** (operator, 2026-10-08, verbatim: "if we detect pi or other
  harnesses codemode in the tools list we do not turn on internal codemode, it is meant as a fallback for other
  harnesses that don't already have harness level codemode."). `codemode.HARNESS_TOOLS` is the ONE list of such tool
  names, starting with Pi's `codemode` (pi.dev/docs/latest/codemode: the model's tool list holds one tool named
  `codemode`); a name is added only from evidence (a harness fixture or a recorded request), never a guess. It is a
  declared overlap in `proxy.TOOL_OVERLAPS`, read by `proxy.tool_conflicts` (the comparison every tool of ours gets) when
  the offer is decided on the conversation's first request, and KEPT like every offer. The toolbox is withheld, recorded
  in `x_yamadori.tools_withheld` ({ours, because, client_tool}) and `x_yamadori.codemode` {offered: false, why: "the
  client brings its own code mode (codemode)"}; in mode `only` nothing is hidden, so the individual MCP lookups stay
  offered as direct tools (and the MCP system line stays). Gated by `mcp/test_codemode_turn.py` (a Pi-shaped request with a
  `codemode` tool in both modes, a Hermes-shaped one without it) and `mcp/test_harness_decisions.py` (fixtures
  `pi/chat/codemode-tool`, `hermes/chat/codemode-fallback`, the `tools` column).
* **Ledger**: the call and its result are an ordinary hidden hop (`hops_added`), replayed byte for byte.
  **The kept definition is stored as TEXT**: the conversation's skill state is re-serialised with sorted keys, which
  reordered the tool's keys (and its parameters') in the rendered prompt, so the first version of this feature broke
  the slot's prefix on the second request; `mcp/test_codemode_turn.py` compares `json.dumps` of the tool list
  (order-sensitive) and that the next request extends the slot through the served template.
* **Record**: `x_yamadori.codemode` {switch, offered, mode, kept, why, declared, left_out, budget_tokens, tokens,
  hidden, ours {tools, tool_tokens, system_tokens, tokens}, runs, calls[{tool, ms, ok}], wall_ms, output_chars, failed,
  reasons, heads, searches, describes, discovery}. `ours` is the cost of OUR additions to the prompt, recorded with the
  toolbox on or off (the probe's figure).

## The switch, the modes, the default

* `codemode` (tiers.BEHAVIOURS, `YAMADORI_CODEMODE`; allowed where the tier's `mcp_tools` flag is set, medium and up;
  **OFF BY DEFAULT** until its probe passes, THE TOOL RECIPE rule 7: `tiers.OFF_BY_DEFAULT`). The header
  `{"codemode": true}` forces it at any tier.
* `codemode_mode` (header or `YAMADORI_CODEMODE_MODE`): `on` (default) or `only`.
  * `on`: the toolbox is one more tool; the individual lookups stay offered.
  * `only`: the MCP-backed lookups are NOT in the tool list (nor their system line); they are reached through the
    toolbox. The craft tool stays top-level when the craft index or the package skills channel (on by default) is
    active, because sections and the index tell the model to call it by name. The image tools stay top-level (their
    pictures stream to the chat as they are made, and attached-image ids are request state). A call to a hidden tool by
    its name still runs (`ours` keeps the names).
* Decided on the conversation's FIRST request and kept (the tool list and system block never change mid-conversation,
  the MCP offer's rule); a conversation that began without a decision gets none.

## The probe (written, not run)

`bench/mcp/codemode_probe.py`, on the pagoda prompt's first `--steps` requests, **Pi and Hermes** (`--harness`), **three
arms**: `off` (today's tools), `on` (toolbox beside the lookups), `only` (toolbox instead of the MCP lookups), interleaved.
Per arm it records: prompt tokens our additions cost (`codemode.ours`, first request), hops, tool-turn cap hits, context
added by our tool results (`x_yamadori.tools[].chars`), whether the model calls `yama_compose_tools` and what it writes,
script failure rate, and **discovery**: did it call `tools.search` / `tools.describe`, did a function it then called
come from the search's hits (`discovery_hit`), describe misses, empty searches, the first function called; plus the
pitfall rules on the files it wrote. Hermes has no setting for `X-Yamadori-Features`: the recording relay adds the
header (`bench/octopus/relay.py --inject-header`). Commands are in the script's plan output (`python
bench/mcp/codemode_probe.py --trials N --steps S`). The reader is checked offline against the record the proxy writes
(`mcp/test_codemode.py` `test_the_probe_reads_what_the_proxy_records`).

## Not measured / open

* The model's use of the toolbox (the probe). The description's and the system line's wording.
* Output text collected by the host (up to Pi's 16 MiB) is Python memory outside the 256 MiB module maximum.
* A native engine's catchable RangeError on deep recursion (wasmtime's stack trap ends the run instead).
* `image()` output and `store/load` are not provided.
* The full text of a script is in the corpus tool_call rows; `x_yamadori.codemode.heads` keeps its first 120 characters.
