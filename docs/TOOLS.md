# The tool audit

**Benchmarks are gated on `mcp/test_tools.py`.** A benchmark measures a model
*through* the tools; if the tools are broken the number describes the tools,
and it does so without saying that is what it is measuring. A night was already
spent measuring a model that was being handed empty strings.

| | |
|---|---|
| gate | `python mcp/test_tools.py` — **356 checks, ~2s, no GPU** (re-run 2026-09-22: `356/356 checks passed`; this table said 349 until the suite grew) |
| opt-in | `python mcp/test_tools_live.py --live` — needs the card, **not run yet** |
| lint | `python -m ruff check mcp scripts --select=E9,F` — clean. `bench/` is another workstream's and drifts; see finding **R6** |

The contract every tool result is held to is stated in the docstring of
`mcp/test_tools.py`. Seven points; the load-bearing ones are **never empty**,
**never a raw exception**, **the situation named**, and **`retryable` is a
fact, not a suggestion**.

---

## 1. Coverage: every tool × every state

`test_every_tool_has_a_test` enumerates `proxy.OUR_NAMES` and fails if any tool
has no case, so a thirteenth tool cannot be added and silently go untested.
The set it checks against is filled by `call()` itself, so it records what the
run actually did rather than a hand-maintained list. This table is the
human-readable version of the same thing — it shows the *gaps*, which the
assertion cannot.

Legend: **Y** covered · **n/a** the state does not exist for that tool ·
**live** covered only in `test_tools_live.py` · **—** not covered, with the
reason under the table.

| tool | no index | hit | miss | retriever outage | bad args | absurd values | traversal / safety | live |
|---|---|---|---|---|---|---|---|---|
| `find_by_meaning`      | Y | Y | Y | Y (total + partial) | Y | Y | n/a | n/a |
| `find_by_pattern`      | Y | Y | Y (+ bad-glob) | n/a¹ | Y | Y | n/a | n/a |
| `find_definition_opt`  | Y | Y | Y (near-miss) | n/a¹ | Y | n/a | n/a | n/a |
| `find_references`      | Y | Y | Y (dead code) | n/a¹ | Y | n/a | n/a | n/a |
| `read_file_range`      | Y | Y | Y (did-you-mean) | n/a¹ | Y | Y | Y | n/a |
| `describe_index`       | Y | Y | n/a | n/a¹ | Y (ignores extras) | n/a | n/a | n/a |
| `run_check`            | Y (NO_REPOSITORY) | Y (listing) | Y (unknown check) | Y (NO_PROJECT_ROOT) | Y | n/a | Y (label ≠ command) | —² |
| `summarize_text`       | n/a | live | n/a | Y (MODEL_UNAVAILABLE) | Y | Y | n/a | **live** |
| `record_step`          | Y | Y | n/a | n/a | Y | Y (20k entry) | Y (session scoping) | n/a |
| `read_rings`           | Y | Y | Y (empty log) | n/a | Y | Y | Y (session scoping) | n/a |
| `bind_project_context` | Y | Y | Y (unindexed pkg) | n/a | Y | n/a | n/a | n/a |
| `delegate_investigation` | n/a | Y (stubbed) | n/a | Y (HELPER_BUSY) | —³ | n/a | Y (no recursion) | **live** |

¹ These read sqlite directly and have no retriever layer. Their equivalent
outage — a database missing a table — is covered by
`test_roots_table_is_part_of_the_schema`.

² **Not covered: `run_check` actually executing a check.** It shells out to
`npm run lint-core` / `test-unit` / `build` with a 900 s timeout. Running one
here would make the gate depend on node, a network and a project's
`package.json`, and could take fifteen minutes. What *is* covered is
everything around it: the allow-list (an arbitrary label is never a command
line), the listing, the unknown-check answer, the missing-repository answer and
the missing-project-root answer.

³ **Not covered: `delegate_investigation` argument validation.** See finding
**R2** — it has no argument gate, so a call with no question starts a real
investigation on the GPU. Testing it would need the card. The stubbed tests
cover everything the proxy is responsible for around it.

**Also not covered anywhere:** `judge` (finding **R3**), and the semantic
retriever (`CODE_SEARCH_SEMANTIC=1`), which needs the embeddings model. The
suite asserts that it reports itself as *disabled by configuration* rather than
broken, which is the state it is actually in.

---

## 2. Defects found, with evidence

### Fixed in `mcp/code_search.py`

---

#### F1 — `OperationalError: no such table: roots`, through a second door

**What it did.** `_db()` created `chunks`, `defs` and `refs` and never `roots`.
Three tools read `SELECT path FROM roots`: `find_by_pattern`,
`read_file_range` and `run_check`. Against any database *this module* created
rather than `scripts/index_code.py` — a half-built index, a dependency index,
or the proxy's own `index/_no_repository_bound.sqlite3`, which `_db()` creates
itself — all three failed.

**Evidence** (index with chunks, no `roots`, repository bound):

```json
{"tool": "find_by_pattern", "ok": false, "error": "TOOL_RAISED",
 "reason": "OperationalError: no such table: roots", "retryable": false,
 "remedies": [{"fixable_by": "operator", ...}]}
```

Identical for `read_file_range` and `run_check`. Confirmed present on the
live `index/_no_repository_bound.sqlite3`, whose tables were
`['chunks', 'defs', 'refs']`.

**Why it mattered.** This is the *same* failure the tool audit was opened for.
It had been fixed for `db is None` by gating `run_check` behind a bound
repository — the fix closed one door. Contract 2 (never a raw exception),
4 (`retryable` false is a lie: the agent cannot fix it, but neither can
retrying) and 5 (the operator is named, but the schema is the bug).

**Fix.** `_db()` now creates `roots` empty alongside the other tables. All
three call sites already handle zero roots as "nothing is indexed", which is
the truth about such a database.

**Guarded by.** `test_roots_table_is_part_of_the_schema`. Reverting the one
line turns 7 checks red.

---

#### F2 — every argument mistake blamed the operator and said retrying was futile

**What it did.** No tool validated its arguments. Every mistake fell through to
the generic handler at the bottom of `handle()`.

**Evidence.** Measured outputs, one per shape:

| call | `reason` returned |
|---|---|
| `find_by_pattern {}` | `KeyError: 'pattern'` |
| `find_by_meaning {"query":"x","top_k":"lots"}` | `ValueError: invalid literal for int() with base 10: 'lots'` |
| `read_file_range {"path": 7}` | `AttributeError: 'int' object has no attribute 'replace'` |
| `find_definition_opt {"symbol": ["a"]}` | `ProgrammingError: Error binding parameter 1: type 'list' is not supported` |
| `run_check {"check": 5}` | `AttributeError: 'int' object has no attribute 'strip'` |
| `record_step {"kind":"did","summary":5}` | `AttributeError: 'int' object has no attribute 'strip'` |
| `read_rings {"limit":"sixty"}` | `ValueError: invalid literal for int() with base 10: 'sixty'` |

Every one of them carried `"error": "TOOL_RAISED"`, `"retryable": false` and
`"fixable_by": "operator"`.

**Why it mattered.** Four things wrong at once, all pointing the same way —
*stop working*:

- the `reason` is a raw exception (contract 2), and one of them names the
  storage layer in a user's conversation;
- `retryable: false` means, per contract 4, *no arguments can ever succeed
  here*. A model that forgot `pattern` was told that supplying it could not
  help;
- `fixable_by: operator` names the one party who cannot see the call;
- `TOOL_RAISED` is indistinguishable from the server actually breaking.

**Fix.** `validate_args()` + `bad_arguments_error()`, called once at the top of
`tools/call` before the dispatch. Argument spec declared as data in
`_ARGSPEC` / `_BOOL_ARGS`. The answer is now `"error": "BAD_ARGUMENTS"`,
`"retryable": true`, `"fixable_by": "agent"`, with the correcting action
spelled out.

**Guarded by.** `test_bad_arguments_blame_the_caller_and_stay_retryable` —
`BAD_ARGUMENT_CASES`, four assertions each. Removing the required-arg check
turns 28 checks red; removing the type check, 28; flipping `retryable`, 26;
flipping the owner, 26; removing the gate entirely, 70.

---

#### F3 — a bad argument was reported as a service outage that had not happened

**What it did.** `find_by_meaning {"query": 12345}` returned:

```json
{"tool": "find_by_meaning", "ok": false, "error": "RETRIEVERS_UNAVAILABLE",
 "reason": "The index exists but no retriever completed, so nothing was
            searched.", "retryable": false}
```

**Why it mattered.** Nothing was down. A non-string query raised inside *both*
retrievers' `except` blocks in `search_fused`, so the diagnostic recorded them
as failed and the tool reported an outage — for a mistake the caller had made
and could have fixed in one call. An agent doing the right thing here reports a
broken service to its user.

**Fix.** Covered by F2: validation runs before `search_fused`, so the type
error never reaches the retrievers.

**Guarded by.** `find_by_meaning / query is a number` in the bad-argument
table.

---

#### F4 — `find_by_pattern` returned a **false negative** for a pattern that matches

**What it did.** `max_results` below 1 made `len(hits) >= limit` true before
the first file was read.

**Evidence.** `find_by_pattern {"pattern": "pool", "max_results": -1}` against
an index where `pool` matches three lines in two files:

```json
{"tool": "find_by_pattern", "ok": true, "matches": 0, "files_scanned": 1,
 "retryable": true,
 "reason": "The index is populated and the regex matched no line in any
            scanned file."}
```

**Why it mattered.** This is the worst of the defects here, because nothing
about it looks like a failure. `ok: true`. A confident, well-structured,
*wrong* answer, which the agent has no reason to doubt.

**Fix.** Values below 1 for `top_k`, `max_results`, `max_words` and `limit` are
refused as `BAD_ARGUMENTS`. `start` is exempt and still clamps to 1 — line 0
and line −5 both unambiguously mean "the beginning". **Enormous values are
deliberately still accepted**: `top_k`, `max_results` and `limit` of 10⁹ were
all measured to answer correctly and fast, because they are slice and `LIMIT`
bounds. Refusing them would be inventing a failure.

**Guarded by.** `test_absurd_but_legal_values_are_answered`.

---

#### F5 — a partial retriever outage claimed the search was complete

**What it did.** With the lexical retriever down and the symbol table finding
nothing, `find_by_meaning` returned:

```json
{"ok": true, "matches": 0, "retryable": true,
 "retrievers": {"lexical": {"ran": false, "reason": "failed",
                            "error": "RuntimeError: ..."},
                "symbols": {"ran": true, "hits": 0}},
 "reason": "Every available retriever ran and no chunk scored above threshold
            for this query."}
```

**Why it mattered.** The prose contradicts the data shipped beside it, and the
prose is what a model reads. A negative produced by half the index was
presented as a clean one; the agent concludes the code does not exist.

**Fix.** The reason is now conditional on `diag["retrievers_failed"]`, and the
result carries `retrievers_failed` and `complete` as machine-readable fields:

> The retrievers that ran found nothing, but lexical failed, so part of the
> index was never searched. This is a weaker negative than a clean miss: do not
> conclude the code is absent, and report the failure.

**Guarded by.** `test_partial_outage_says_the_search_was_incomplete` and
`test_a_real_miss_is_a_real_miss` — which assert the *opposite* wordings, so
both directions of the mutation are caught.

*The total-outage path was already correct and now has a regression test:
`test_total_outage_is_an_error_not_a_miss` confirms `RETRIEVERS_UNAVAILABLE`,
not a zero-match result.*

---

#### F6 — an empty line range came back as an empty code fence

**Evidence.** `read_file_range {"path": "src/pool.ts", "start": 3, "end": 1}`:

```
src/pool.ts:3-1  (4 lines total)
```
```

and `{"start": 1000000000}` produced the same body under the header
`src/pool.ts:1000000000-4`.

**Why it mattered.** Contract 1 evaded by punctuation. It is not an empty
string, so it passes a character count, and it is meaningless to a reader: a
file with no such lines is indistinguishable from a reader that broke. The
header also reports a range (`3-1`, `1000000000-4`) that cannot exist.

**Fix.** A structured `EMPTY_RANGE` error carrying `lines_in_file` and the
range as requested, `retryable: true`, `fixable_by: agent` — because one more
call fixes it.

**Guarded by.** `test_absurd_but_legal_values_are_answered`.

---

#### F7 — `run_check` would have run a build in the server's own directory

**What it did.** `cwd = os.path.dirname(roots[0]...) if roots else os.getcwd()`.
With an empty `roots` table — which, before F1, was *any* database this module
created — `npm run build` would have run in the `llama-stack` checkout. The
listing branch also printed the server's absolute path into the conversation as
`project root:`.

**Fix.** No fallback. With no indexed root the listing says so, and running a
check returns `NO_PROJECT_ROOT` with the user named as the fixer.

**Guarded by.** `test_roots_table_is_part_of_the_schema`.

---

#### F8 — `summarize_text` handed the agent a raw exception, and an empty completion straight through

**Evidence.** With the stack unreachable, the entire tool result was the bare
string:

```
compaction failed: URLError: <urlopen error [WinError 10061] No connection
could be made because the target machine actively refused it>
```

**Why it mattered.** Contract 2 in one line, with nothing in it to act on and
no statement of whether retrying could help — and it *could*, since the text
was never sent. The agent also cannot tell from that string whether its own
context survived.

**Fix.** `MODEL_UNAVAILABLE`, `retryable: true`, the underlying error moved
into a `detail` field for the operator, and two remedies: the agent continues
without compacting, the operator checks `LLAMA_STACK_URL`. Separately, an empty
completion now returns `MODEL_RETURNED_NOTHING` rather than an empty string.

**Guarded by.** `test_summarize_text_names_an_unreachable_model` — which runs
on the gate, with no GPU, by pointing `cs.STACK` at a closed port.

---

#### F9 — `calls_only` inverted itself

**What it did.** `bool(args.get("calls_only", False))`. `bool("false")`,
`bool("no")` and `bool("0")` are all `True`. A caller asking for the wider
result set got the narrower one, silently.

**Fix.** `_BOOL_ARGS` — a non-boolean is refused, with the message naming what
it *would* have been read as. There is no safe coercion: `"false"` cannot be
told from a typo.

**Guarded by.** `find_references / calls_only is the STRING false`.

---

### Reported, not fixed — files owned by other workstreams

---

#### R1 — `bind_project_context` reports success having bound nothing

**File.** `mcp/proxy.py`, `bind_project_context()`. *Not mine to edit.*

**Evidence.** `bind_project_context {"versions": {"three": 185}}` returns, in
full:

```
bound for this session:
```

**Cause.** `if not isinstance(raw, str): continue` silently drops every
non-string version, and the header is printed unconditionally. A model that
reads a manifest and passes a numeric version is told its versions are bound.
Every subsequent answer comes from the wrong source, or from no source, with
nothing anywhere saying so.

**Required fix.** In `mcp/proxy.py`, in the loop over `versions.items()`,
replace the silent `continue` with collection into a `dropped` list, and gate
the header:

```python
dropped = []
for name, raw in list(versions.items())[:40]:
    if not isinstance(raw, str):
        dropped.append(f"{name}={raw!r} ({type(raw).__name__})")
        continue
    ...
if not lines and not unknown:
    return ("bind_project_context bound nothing: "
            + ", ".join(dropped) + " — versions must be strings, e.g. "
            '{"three": "0.185.1"}.')
...
if dropped:
    out.append("  NOT bound (version was not a string): " + ", ".join(dropped))
```

The test that will cover it is already written and currently passes through the
prose branch: `bind_project_context` in `BAD_ARGUMENT_CASES`. Tighten it to
assert the non-string case names what was dropped once the fix lands.

---

#### R2 — `delegate_investigation` has no argument gate, and an empty question starts a GPU run

**File.** `mcp/proxy.py`, `run_our_tool()`. *Not mine to edit.*

**Cause.** `shomen.investigate(args.get("question", ""), ...)`. The schema
marks `question` as required and nothing enforces it. A call with no question
acquires the helper lane and starts a real second context on the card with an
empty prompt.

**Found how.** This suite tried to add `("delegate_investigation", {},
"missing question")` to the bad-argument table and it began a live
investigation. The case was removed, with a comment, rather than left in — a
gate that needs the GPU is not a gate.

**Required fix.** In `mcp/proxy.py`, immediately before the
`admission.helper_lane()` block:

```python
q = args.get("question")
if not isinstance(q, str) or not q.strip():
    return cs.bad_arguments_error(
        "delegate_investigation",
        "No question was supplied, so no investigation was started.",
        "call delegate_investigation again with a self-contained `question`")
```

`cs.bad_arguments_error` already exists and produces the right envelope.
Once it lands, add the case back to `BAD_ARGUMENT_CASES` in
`mcp/test_tools.py` — it is safe there the moment the gate exists, and unsafe
until then.

---

#### R3 — `judge` is advertised nowhere and *was* callable by name — FIXED

**File.** `mcp/code_search.py` `handle()` — the dispatch is mine, but the
decision to expose it is not.

**State.** `judge` is implemented and wired to the Laya service on port 1237.
It is in `TOOLS_DISABLED_JUDGE`, a **string constant**, and therefore in no
tool list: not `TOOLS`, not `INTERNAL_TOOLS`, not `ALL_TOOLS`, not
`proxy.OUR_NAMES`. The model is never offered it.

**The gap, as originally written.** The dispatch did not check membership. An
MCP client that sent `tools/call` with `name: "judge"` got it, even though
`tools/list` never mentioned it. The gate was a comment, not a mechanism.

**Closed.** `handle()` now checks the name against `ALL_TOOLS` before
dispatching, exactly as recommended below, and `mcp/test_tools.py` asserts all
three halves of it — `judge is advertised nowhere`, `an unadvertised name does
not dispatch`, and `no result is produced for it`. Verified in the
2026-09-22 run (356/356). Re-exposing `judge` now requires editing a tool list
and breaking a test, which is the point.

**Why it is disabled** (from the comment, preserved here because it is the
actual measurement): `scripts/eval_judge.py`, unambiguous yes/no engineering
questions, **6/10 against a 5/10 coin flip**, with systematic errors — three of
four misses were false positives on the negative cases, all at 0.67–0.69. It
scores what a passage is *about*, not whether the proposition holds.

**Recommendation.** Leave it disabled; make the gate real. In `handle()`,
before the dispatch:

```python
if name not in {t["name"] for t in ALL_TOOLS}:
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"unknown tool {name}"}}
```

Was "not applied here because it changes the *server's* surface"; it has since
been applied, and `test_every_tool_has_a_test` asserts it, so re-exposing
`judge` has to be a decision rather than an accident.

---

#### R4 — the dependency fallback can no longer see a miss — FIXED SINCE

**File.** `mcp/packages.py`, `_is_empty()`. *Not mine to edit.*

**State.** The required fix below has landed: `_is_empty()` now parses the JSON
envelope first and falls back to the prose phrases, with its own docstring
carrying this finding's reasoning. The report is kept because the failure shape
— a predicate that silently stopped firing when a tool's return type changed —
is the useful part.

**Cause.** `_is_empty` decides whether to consult dependency indexes by
substring-matching the first 120 characters for `"no matches"`, `"not found"`,
`"no results"` and so on. Zero-match results are now **JSON envelopes**:

```json
{"tool": "find_by_meaning", "ok": true, "matches": 0, "query": "...", ...}
```

None of those substrings appear. So a structured miss from `find_by_meaning`
or `find_by_pattern` never triggers `packages.search` or
`search_discovered` — the dependency fallback is dead for exactly the two
tools most likely to need it. The prose misses (`find_definition_opt`,
`find_references`, `read_file_range`) still work.

**Required fix.** In `mcp/packages.py`:

```python
def _is_empty(text: str) -> bool:
    if not text:
        return True
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        if isinstance(d, dict) and "tool" in d:
            # A structured result: zero matches, or any failure, is empty.
            return d.get("matches") == 0 or d.get("ok") is False
    head = text[:120].lower()
    return ("no matches" in head or "not found" in head
            or "no definition" in head or "no references" in head
            or "matched 0 of" in head or "no results" in head)
```

Note the `ok is False` branch also makes `NO_INDEX` count as empty, which is
what the caller wants — that is precisely the case where a dependency index is
the only hope.

**Not covered by a test**, because the fallback needs real package indexes on
disk under `index/packages/`. Worth a fixture of its own once R4 is fixed.

---

#### R5 — `record_step` silently relabels a null `kind`

**File.** `mcp/rings.py`, `record()`. *Mine to edit, deliberately not edited.*

**Evidence.** `record_step {"kind": null, "summary": "y"}` →
`recorded #2 as entry 2 of session 'p2'.` — filed as `did`. Meanwhile
`{"kind": "nonsense"}` is correctly refused with
`unknown kind 'nonsense'. Use one of: did, learned, check, decided.`

**Assessment.** Low severity, and defaulting is defensible — but the two paths
disagree, and the mislabelled entry is what a later session reads back as
evidence of what happened. `kind` is `required` in the schema.

**Left alone** because changing it would refuse calls that currently succeed,
and that is a behaviour change with no measured failure behind it. If it is
wanted: in `rings.record()`, replace `kind = (kind or "did").strip().lower()`
with an explicit check that `None`/empty is refused the same way an unknown
kind is.

---

#### R6 — lint failures in `bench/`, which is being written right now

`bench/` is untracked and another workstream is actively editing it. Two
states were seen:

- **`bench/laya_calibration.py`, lines 638 and 671** — `ruff F541`, stray `f`
  prefixes on strings with no placeholders. Pre-existing; they made the stated
  lint gate red before this work started. **Fixed**, `print(f"..." + ...)` →
  `print("..." + ...)`, semantically identical.
- **`bench/laya_grounded_contrast.py`, line 45** — `ruff F401`, `best_gate`
  and `kfold` imported from `train_laya` and unused. This file appeared
  *during* this session; `ruff check mcp bench scripts --select=E9,F` passed
  minutes earlier. **Not fixed** — the file is in flux and belongs to whoever
  is writing it. The fix is to drop the two names from the import list, or add
  them to `__all__` if they are about to be used.

`mcp` and `scripts` are clean on their own:

```
python -m ruff check mcp scripts --select=E9,F     # All checks passed!
```

---

## 3. What could not be tested, and why

| | why |
|---|---|
| `run_check` actually running `lint`/`test`/`build` | needs node, a network and a project `package.json`; up to a 900 s timeout. Everything around it is covered — see note ² above. |
| `summarize_text` / `delegate_investigation` live | need the GPU. Written in `mcp/test_tools_live.py`, **not run** — another workstream holds ports 11434/10001. Dry-run against stubs: 24/24, so the code paths are sound; the assertions have not met a real model. |
| `delegate_investigation` argument validation | would start a GPU run. Finding **R2**. |
| `judge` | not advertised, deliberately disabled. Finding **R3**. |
| the semantic retriever | needs the embeddings model. The suite asserts it reports itself *disabled by configuration* rather than broken, which is the state it is in. |
| `packages` dependency fallback | needs real package indexes under `index/packages/`. Finding **R4**. |
| concurrent `HELPER_BUSY` | the refusal path is covered with the lane stubbed; real contention needs two requests in flight. |

## 4. How to trust this suite

Every fix above was **mutation-tested**: the fix was reverted and the suite
re-run, to prove the checks can fail rather than merely pass.

| finding | reverted | checks that went red |
|---|---|---|
| F1 | `roots` table creation | 7 |
| F2 | argument validation as a whole | 70 |
| F2 | required-argument check only | 28 |
| F2 | string type check only | 28 |
| F2 | `retryable: true` on bad arguments | 26 |
| F2 | `fixable_by: agent` on bad arguments | 26 |
| F4 | integer floor (the false negative) | 2 |
| F5 | partial-outage wording, in *both* directions | 2 and 1 |
| F6 | `EMPTY_RANGE` | 2 |
| F7 | `run_check` cwd fallback | 1 |
| F8 | `MODEL_UNAVAILABLE` envelope | 2 |
| F9 | boolean check on `calls_only` | 2 |

Two of these started at 0 — the boolean check and the `MODEL_UNAVAILABLE`
envelope both fell through to a loose "it answered something" branch, which is
a test that cannot fail. Both were tightened until reverting the fix turns them
red. The dry run of `test_tools_live.py` against stubs caught a third such flaw
in its own fixture: the summary it asks the model to preserve contained
`OperationalError`, which the contract-2 leak detector cannot tell from a real
leak. The token was changed to `ENOSPC`, which keeps both checks meaningful.

Nothing here writes to `index/code.sqlite3`, `index/corpus.sqlite3` or the real
rings database: `CODE_INDEX_DB`, `YAMADORI_CORPUS_DB` and `RINGS_DB` are all
pointed at temporary files before `code_search` is imported, and
`test_the_fixture_is_not_the_real_index` asserts it.
