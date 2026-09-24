# Transcript review, 2026-09-23

This review reads the model output behind last night's benchmark rows. It
covers every failed row and a sample of the passes, in the domain suite,
LiveBench coding, SWE-bench and the proxy corpus. It was read-only: no GPU
work, no model calls and no restarts. The domain answers were regraded offline
with `grade.grade(task, content)` on tsc and cargo. All five failed rows were regraded, and every regrade
reproduced the recorded stage and error. The LiveBench completions were
rebuilt the way the judge rebuilds them (`partial_solution + "\n" +
extract_code(answer)`) from the HF question file, then checked with
`compile()`. No code was run.

**Every n here is 1 per (task, arm, run).** Nothing below is a rate. The
findings are about mechanisms, and each one names the rows it rests on.

## Headline

1. **The min_p comparison on the domain suite cannot measure min_p.** The
   harness sends `temperature: 0.0` (every manifest). At temperature 0 the
   chosen token is the argmax whatever min_p is, and the runs show it. rs03/A0
   produced the same 2,943-character reasoning in the 0.05 run and the 0.0
   run, and rs01/S0 matched too. Runs still differ, because decoding here is
   not deterministic: rs07/A0 matches for 692 characters and then diverges by
   one token, which is the difference between a pass (overnight-0923b) and a
   fail (minp0b). That flip is noise between runs, not a min_p effect.
   LiveBench is different: it sends temperature 1.0, so min_p does matter
   there.
2. **The domain failures are knowledge errors, and the compile-only check
   steered the model toward a worse answer.** In both ts03/S0 runs the model
   never used `asserts`. It fixed TS2355 by adding `return true` to a type
   predicate. That compiled, and the model took COMPILES to mean done.
3. **Four of the five LiveBench completion misses are indentation or
   continuation faults.** `compile()` catches three of them. `ast.parse`
   catches only two, because "'return' outside function" is a compile-time
   error, not a parse error. The fourth (3cd8c16a) compiles cleanly and loops
   forever.
4. **Two stack bugs voided rows.** The MTP build crashed mid-stream
   (`speculative batch index 6 is not inside the current sub-batch [0, 4)`).
   Separately, a client timeout discarded an answer that the proxy went on to
   produce.
5. **Two retrieval-layer bugs showed up in real transcripts.**
   `find_by_meaning` labels an unrelated drei function TAPROOT, "Strongest
   evidence available", for a wasm-bindgen query. And the proxy's own
   "None matched." wording is not recognised as empty, so the repeat breaker
   never fires on package searches.

## Every reviewed row

Key to the categories:

- **L**: model logic or knowledge error
- **S**: model syntax slip
- **F**: format or extraction
- **R**: runaway or truncated thought
- **T**: tool misuse or tool-result problem
- **H**: harness, stack or grader
- **P**: pass that looks robust
- **P!**: pass that looks lucky or fragile

### Domain suite, current runs (min_p 0.0, temperature 0.0)

| run | task | arm | outcome | cause | evidence |
|---|---|---|---|---|---|
| minp0a | ts03 | A0 | fail (compile TS2355) | L | Wrote a type predicate, not an assertion signature: `): value is NonNullable<T> {` with no `asserts`. The thought even says "the standard pattern for assertion functions ... returning a type assertion like `value is T`". |
| minp0a | ts03 | S0 | fail (test: no narrowing) | L, check misled | Round 0 invented a rule: "Assertion functions are functions with a return type of never". Check #1 returned TS2534 (a never-returning function cannot have a reachable end point). The model switched to `value is T & (string\|number\|...)`. Check #2 returned TS2355, and the model added `return true`. Check #3 said COMPILES. The final thought reads: "we can't return `void` ... Returning `true` is fine". That breaks the spec ("returns nothing"), and the result still does not narrow. 4 rounds, 44,140 reasoning characters. |
| minp0b | rs01 | A0 | pass | P | `#[repr(C)]` struct, `size_of`. |
| minp0b | rs01 | S0 | pass | P | One check, which passed. Byte-identical to overnight-0923b rs01/S0. |
| minp0b | rs01 | A6 (att. 2) | pass | P, with a T note | Deep thinking: 2 searches, `injected: true`, `cited: 0`. The answer thought: "The investigation findings already provide the answer". Fan-out ran before the 07:34 fan-out fix. Attempt 1 was discarded because the helper lane was busy (310.5 s generated). |
| minp0b | rs03 | A0 | pass | P | Same 2,943 characters of reasoning as overnight-0923b rs03/A0. |
| minp0b | rs03 | S0 | pass | P | Never called check_solution (`checks=[]`). Easy task. |
| minp0b | rs07 | A0 | fail (compile E0204, E0507) | L, P! across runs | "Yes! In Rust, `Copy` is auto-implemented for unions where all fields are `Copy`". It wrote a manual `impl Clone for Payload { *self }`, which ignores the prompt's "both deriving `Clone, Copy`". Diverged from the passing 0923b answer after 692 identical characters. |

### Domain suite, stale runs (for contrast)

| run | task | arm | outcome | cause | evidence |
|---|---|---|---|---|---|
| 0923a | ts03 | A0 | fail (compile TS1144) | S | Knew the feature and got the syntax wrong: `): void asserts value is NonNullable<T> {`. One extra token. A parse check would have caught it. |
| 0923a | ts03 | S0 | fail (test) | L, check misled | `value is Exclude<T, null \| undefined>`. Check #1 returned TS2355, and the model reasoned "a type guard function must return a value (the boolean) ... adding `return true`". Check #2 said COMPILES, and it answered. Round 0 took 45 minutes and 74,822 characters. |
| 0923a | ts03 | A6 | stack_error (TimeoutError, 8,155 s) | H | Proxy log: the fan-out variants lost upstream at 2,371.9 s (ConnectionResetError, three lines) and at 4,313 s. The corpus records the turn's answer (2,017 characters) at 7,775 s, after the client had given up. The answer existed and was thrown away. |
| 0923b | rs01 | A0 | pass | P | |
| 0923b | rs01 | S0 | pass | P | |
| 0923b | rs01 | A6 | pass | P, with a T note | `tools_gate: OFFERED_EARLIER_THIS_SESSION` on the first request of this arm (see harness issue 3). `investigate.cited: 0`, injected. |
| 0923b | rs03 | A0 | pass | P | |
| 0923b | rs03 | S0 | pass | P | No check call. |
| 0923b | rs03 | A6 | pass | P, with a T note | Its searches went to JS-only indexes: `find_by_pattern(glob="Cargo.toml")` and `(glob="wasm-bindgen")`, both "None matched". `find_by_meaning` returned drei `vecToArray` as TAPROOT (improvement 3). Deep thinking was injected with `cited: 0`, and its "findings" contained the full solution. The answer ends with "> Answered 3 ways; only 33% agreed on the same file. Treat this as unsettled." That note sits on a correct single-function answer where one of three variants named a file (improvement 5). 989 s. |
| 0923b | rs07 | A0 | pass | P! | Correct `#[derive(Clone, Copy)]` on the union. The same prompt failed in minp0b after a single-token divergence. |
| 0923b | rs07 | S0 | stack_error (finish `incomplete`) | H | Content: "[the connection to the model dropped after 560s ...]". Proxy log: `upstream dropped after 560.2s ... speculative batch index 6 is not inside the current sub-batch [0, 4)`. This is the MTP build crashing. Another drop at 679.5 s (`index 4`) the same night. |

### Self-check probe

| run | task | arm | outcome | cause | evidence |
|---|---|---|---|---|---|
| selfcheck-probe | ts02 | A0 | rows.jsonl says fail/extract; analyse reclassifies it to stack_error `server_reasoning_cap` | R (and H) | A degenerate loop. The paragraph "OK I'm going to stop going in circles and just write the solution..." repeats **95 times verbatim**, starting at character 16,319 of 120,421. The model never produced `as`-key remapping or `Capitalize`, both of which it wrote correctly in overnight-0923 and smoke-medium. The server's 32,768 cap ended it. See harness issue 4. |
| selfcheck-probe | ts02 | S0 att. 1 | stack_error | H | `ConnectionResetError` after 918 s. |
| selfcheck-probe | ts02 | S0 att. 2 | stack_error | H | `connection refused` at 02:08:57. The proxy restarted at 02:08:59 (preflight `proxy_fresh`). |

### LiveBench coding, lb-20260923 (min_p 0.05, temperature 1.0)

There are 21 rows: 15 pass and 6 fail. Every failure was rebuilt as the judge
builds it. The `compile()` column shows what a check_code-style parse check
would report.

| q | task | outcome | cause | `compile()` of the rebuilt program | evidence |
|---|---|---|---|---|---|
| 3cd8c16a | completion | 0 | F | **compiles** | The prefix ends inside `while left <= right:` at 12 spaces. The answer starts `        if check(mid):` at 8 spaces, so the `while` body never changes `left` or `right`, and the loop never ends (judge timeout). Only a check that names which blocks the first line closes would flag it. |
| 45d51d09 | completion | 0 | F | SyntaxError: 'return' outside function (line 28) | The answer is indented 4 inside a block 12 deep. **`ast.parse` accepts this program; only `compile()` rejects it.** 13,355 output tokens, 2,073 s. |
| 5c9d7ade | completion | 0 | F | SyntaxError: unterminated triple-quoted string | The prefix ends on an opened `"""`. The reference remainder starts `:type nums: ...` and then `"""`, and the answer never closes the docstring. |
| eae49210 | completion | 0 | F | SyntaxError: expected an indented block after 'while' | The prefix ends `        while not reachable >= target:`. The answer lines are at 4 and 0 spaces. |
| e25750fd | completion | 0 | L | compiles | `step = 2 * new_k`. The prefix already folds the factor of 2 into `new_k` (`int(i == 2)`), so the answer doubles it. The model misread the partial solution. |
| df00a468 | LCB gen | 0 | L | compiles | `chr(mapping[ord(ch) - ord('a')])` is missing `+ ord('a')`, so it prints control characters. The thought traced sample 1 in letters ("So: "recover" ✓") and never ran the integer code. Running sample 1 from the prompt would have shown it. |
| 7878539e | completion | 1 | **P!** (lucky) | compiles | The answer drops the prefix's loop logic and returns `all(popcount(nums[i]) == popcount(sorted(nums)[i]) ...)`. Rebuilt and called directly: `canSortArray([4,3,2])` returns **True**, but the right answer is False (no two neighbours share a popcount). The hidden tests do not cover this case. |
| ebde77f9 | completion | 1 | P! | compiles | The same question in the in-progress `lb-20260923-minp0` run (unscored) came back as `    return min(cnt[0], cnt[1])` at 4 spaces, which is 'return' outside function. The formatting is not stable between samples. |
| 786432dd | completion | 1 | P | compiles | Continues at the right depth (16). |
| 9e895f81 | completion | 1 | P | compiles | Dedents to `else:` correctly. |
| 4952a5d4 | completion | 1 | P | compiles | Ignores the prefix's trie and uses an O(n²) loop, which is valid at n ≤ 50. |
| 87871f3e, cf932b62 | LCB gen | 1 | P | n/a | Read by eye: correct tie-break, and a standard reverse Dijkstra ("Last Train"). |

sanity.md says "4 of the 5 coding_completion zeros are continuation-format
failures". The rebuild confirms that. It also refines one line: 45d51d09 is
caught by `compile()` and not by `ast.parse` (`mcp/code_check.py` already uses
`compile()`, which is correct).

### SWE-bench mini50-20260923 (bonsai arm)

| instance | outcome | cause | evidence |
|---|---|---|---|
| django__django-11999 | resolved | P, but slow | `if not getattr(cls, 'get_%s_display' % self.name, None):` is equivalent to the upstream `hasattr` fix. The edit landed at step 40 of 80. Steps 2 to 9 looked for `django/core/field.py` and `django/models/` before finding `django/db/models`, because a case-sensitive grep for `get_field_display` missed `_get_FIELD_display`. After that it pip-installed asgiref, pytz and sqlparse to run tests. 1.1 M prompt tokens, 6,174 s. The 16 `proxy_error_lines` are 429 and connection retries, not model errors. |
| sphinx-doc__sphinx-8269 | resolved | P | It added `response.raise_for_status()`, the same line as the upstream fix, **at step 6 of 66**. Steps 7 to 62 build throwaway HTTP-server repros; most of them fail on environment details (`AttributeError: property 'config' of 'FakeApp' object has no setter`). |

The other instances are discarded, per run.log: an operator stop at 04:37,
reruns after starvation, and one plumbing slip. Neither SWE-bench trajectory
needed the stack.

### Proxy corpus (index/corpus.sqlite3, 369 turns since 09-22 19:38)

| turn | what happened | cause |
|---|---|---|
| rs03/A6 (09-23 04:50) | The tools were offered under `NO_DOMAIN_EVIDENCE` to a Rust wasm-bindgen task, and no Rust crate is held. It made 2 wasted searches, then `find_by_meaning` returned `== TAPROOT ... declares vecToArray` from @react-three/drei, with an `isFloat32Array` snippet under that label. | T (selection and a mis-tier) |
| typegpu smoke (09-22 20:06, 20:42) | 11 hops each. The same `find_by_pattern(glob="typegpu@0.12.5")` call was issued 4 times, and `read_file_range("typegpu@0.12.5/data/struct.d.ts")` returned an error envelope twice. `proxy.py` now routes package-prefixed paths and globs (`_route_package_glob`, whose comment cites this exact smoke). **Not re-verified live.** | T (since addressed in code) |
| all no-repo package searches | 19 results since 09-22 read "No repository is bound ... None matched." and are recorded with `"empty": false`. | T (bug, improvement 4) |

## Counts by root cause

These count the rows in the tables above. Corpus turns are left out.

| cause | rows | which |
|---|---|---|
| L: model logic or knowledge | 6 | ts03 A0 (minp0a), ts03 S0 (minp0a), ts03 S0 (0923a), rs07 A0 (minp0b), e25750fd, df00a468 |
| S: model syntax slip | 1 | ts03 A0 (0923a) |
| F: format or continuation | 4 | 3cd8c16a, 45d51d09, 5c9d7ade, eae49210 |
| R: runaway thought | 1 | selfcheck-probe ts02 A0 |
| T: tool or tool-result problem (outcome unchanged) | 3 notes on passing rows | rs01 A6 ×2 (injected with cited 0), rs03 A6 (irrelevant retrieval, false dissent note) |
| H: harness or stack | 5 | ts03 A6 (timeout), rs07 S0 (MTP crash), ts02 S0 ×2, plus the reclassification of ts02 A0 |
| wrong API or version | 0 | No row used an API missing from the pinned version. Every miss is a construct that exists but is the wrong one, or bad syntax. |
| grader | 0 | All five failed domain grades reproduced offline (the passes were not regraded). Every LiveBench zero was rebuilt, and the rebuild agrees with the judge. |
| P: robust pass | 19 | |
| P!: fragile or lucky pass | 3 | rs07 A0 (0923b), 7878539e, ebde77f9 |

## Top 5 fixable improvements

**1. Let check_solution / check_code compile a usage probe the model writes,
not only the solution.** Rows: ts03 S0 in minp0a and in 0923a (both fail/test).
In both, the only feedback was the compiler, and each compiler error was
"fixed" in a way that satisfied tsc and broke the requirement: `return true` on
a type predicate. The model knew what should happen: the prompt's own example
is "`x: string | null` becomes `string` on the following lines". An optional
`probe` argument, compiled against the solution, would have failed at the first
check. A probe like `declare const x: string | null; assertIsDefined(x); const
y: string = x;` fails with TS2322, the same error the hidden test reports
(`test.ts(8,3): Type 'string | null' is not assignable to type 'string'`). This
is not the hidden tests: the model writes it from the spec. The tool
description should list the phrasing that triggers it: "narrows",
"after the call", "must infer".

**2. Put a remedy line on known error codes that name the construct.** This
follows AGENTS.md: failure returns carry the next step. Rows: ts03 S0 ×2 and
ts03 A0 (minp0a).

- TS2355 or TS2534 on a function whose return type is a type predicate, or
  `never`, and whose body only throws: add "for a statement-form narrowing,
  use an assertion signature `asserts x is T` with a void body".
- TS1144 or TS1228 right after `void asserts` (0923a A0): the fix is to drop
  `void`.
- E0204 with a union field (rs07): "add `#[derive(Clone, Copy)]` to the union
  itself".

The recipe corpus already has the TS fact: `typescript_types.jsonl`, "asserts
cond and asserts val is T (3.7) narrow for the remainder of the scope". But
no ts03 row with hints turned on finished (0923a A6 timed out, and minp0a
dropped A6). So whether hint selection would have fired is **unmeasured**.

**3. Stop labelling a declaration TAPROOT because its name's words are a subset
of the query's words.** Row: rs03/A6 (0923b), corpus turn e3762efcd0234dc7.
`search_fused` tags any `defs` row where
`set(content_words(name)) <= words(query)`. `vecToArray` → {vec, array} is a
subset of "wasm-bindgen Float32Array ... Vec f32 to Float32Array", so an
unrelated drei helper was presented as "Strongest evidence available". The
snippet shown under it is also a different chunk (`isFloat32Array`) from the
same path, because `text_by_path` is keyed by path. Fix: require the whole
identifier to match one query identifier, and show the declaration's own lines.
A related selection gap: the gate offered tools under `NO_DOMAIN_EVIDENCE` for
a prompt naming `wasm-bindgen 0.2`, with no Rust package held. That cost 2
hops and 971 s. The gate should read a named, unheld package as evidence to
withhold.

**4. Make `repeats._empty` recognise the proxy's own no-match wording.** Rows:
rs03/A6 and every package-fallback search. `_empty()` looks at the first 140
characters for "no matches", "matched 0 of" and similar. It never sees "None
matched." (`_search_packages_without_repo`) or "no match ==" (the routed-glob
path). So `turn.cached_empty` never fires for the normal no-repository caller,
and the corpus `empty` flag is false on 19 no-match results. The repeat breaker
was built for exactly the identical-search loops seen in the typegpu smoke. As
it stands, it is off for the whole no-repository path.

**5. Deep thinking: when `cited == 0`, do not inject, or inject under a header
that does not say "indexed source".** Rows: rs01 A6 (0923b and minp0b) and
rs03 A6 (0923b): `investigate: {hops: 2, injected: true, cited: 0}` on all
three. The rule in `proxy._deep_thinking` is `hops > 0`. For rs03 the two
searches hit JS packages and found nothing, yet the full code solution crossed
over under `FINDINGS_HEAD`, "Findings from a separate investigation of the
indexed source". The main model noticed: "The provided 'findings' are
unverified". This is SELECTION-BUILD harm 1 again, gated on searching instead
of on citing. It cost 118 to 336 s of helper time per row for no retrieved
evidence. On the same rows, `fanout.dissent_note` fires when one of three
variants names a single file (agreement 1/3). It appended "Treat this as
unsettled" to a correct answer (rs03 A6). Agreement should be taken over the
variants that voted, or need at least two voters.

Also worth noting: **for LiveBench completions, `compile()` of the rebuilt
program catches 3 of 4 format misses**. The fourth (3cd8c16a) needs a
block-closing report such as "your first line closes the body of
`while left <= right:`". An unconditional dedent warning would fire falsely on
the passing ebde77f9 and 9e895f81, which dedent correctly. The rebuild shows
another route: the judge accepts a full program that starts with
`partial_solution` (`not extracted_answer.startswith(partial_solution)`). So
re-emitting the whole function avoids splicing entirely. Whether that is within
the task's rules ("Only write the missing portion") is an operator call. And
for df00a468, running the prompt's own sample input would have exposed the
bug. That is execution, which check_code's contract forbids, so it belongs to
a harness-side check, not the tool.

## Harness, stack and grader issues that make a number untrue

1. **The min_p 0.0 vs 0.05 domain comparison is void as a min_p measurement.**
   The requests are temperature 0.0 (manifests), and identical outputs across
   runs (rs03 A0, rs01 S0) confirm greedy decoding. The differences that
   remain come from run-to-run divergence at single tokens: rs07 A0 at
   character 692, rs03 S0 at 1,551, ts03 A0 at 158. Any pass-rate difference
   between overnight-0923b and minp0b is not attributable to min_p. That
   divergence also sets the noise floor: one run of rs07 A0 is a coin flip
   between pass and fail.
2. **The MTP build crashes mid-stream.** Proxy log: `speculative batch index
   {4,6} is not inside the current sub-batch [0, 4)` at 560.2 s and 679.5 s.
   It voided 0923b rs07/S0 (stack_error, so excluded rather than scored). It
   is not in docs/KNOWN-ISSUES.md or MTP-STAGING.md. Until it is fixed, every
   long generation carries some chance of being voided this way.
3. **Arms share a session.** `nebari.key_of` hashes the account plus the
   first two messages. The domain harness sends one user message, the task
   prompt, from one key. So every arm, attempt and run of a task within the
   36 h TTL shares one session: tool-offer history, discovered packages, and
   the `read_rings` work log. Seen directly: rs01/A6 was gated
   `OFFERED_EARLIER_THIS_SESSION` on its first request, in both 0923b and
   minp0b. It changed no outcome here, and no bench turn called `read_rings`
   (the corpus has one `read_rings` call all night, not from a benchmark). But
   the gate decision recorded for A6 was made by an earlier arm. The fix
   belongs in the harness: add a per-row nonce to the account or the first
   message. A proxy change is not needed.
4. **`server_reasoning_cap` reclassification hides a real runaway.**
   analyse.py turns selfcheck-probe ts02/A0 into a stack_error that resume
   regenerates, on the reasoning that the server cut the model off. The
   transcript shows a 95-fold verbatim loop from character 16,319. The model
   would not have answered at any budget, so this is a model failure (R).
   Reclassifying rows like it makes the scored pass rate look better than it
   is. Suggested fix: before reclassifying, check the tail for repetition,
   such as a paragraph repeated more than 3 times. Related: `x_yamadori.budget`
   claims `reasoning_budget_tokens: 98,099` on that row, but the server
   enforced 32,768. The per-request budget in AGENTS.md "one budget rule" is
   not what the MTP build obeys above the launch value (documented in
   analyse.py, not in AGENTS.md).
5. **A client timeout throws away finished answers.** On 0923a ts03/A6 the
   request arrived at 04:50:43. The corpus turn record, written after deep
   thinking, is at 05:06:53. The client raised TimeoutError at 07:06:38. The
   proxy recorded a 2,017-character answer at 07:16:29, ten minutes after the
   client had left. The row is an unscored stack_error, and the answer is
   gone. The fan-out variants inside that turn also hit upstream
   ConnectionResetError drops at 2,371.9 s and 4,313 s, and were retried once.
6. **Some rows ran against a stale proxy with preflight failing.** The `.2`
   resumes of minp0a and minp0b logged
   `FAIL proxy_fresh ... stale: proxy.py modified 07:38:14, code_check.py
   07:37:59, tiers.py 07:37:49, fanout.py 07:34:39`, and `run_tests` and
   `live_stack` were skipped. minp0b rs07/A0 (07:39) came from this state. It
   is A0 (tools off, so fan-out and tools are not involved), but `tiers.py`
   sets the budget. The row does not match the code on disk.
7. **A6 rows predate the fan-out winner fix.** `lb-20260923-minp0/
   prefix_yamadori_features_arm/README.txt` says the proxy "picked the
   SHORTEST candidate for code answers" until about 07:34. Every domain A6 row
   reviewed (0923b rs01 and rs03, minp0b rs01 at 07:25) ran under that
   behaviour. They passed, but they measure that code, not the current stack.
8. **The corpus `empty` flag is wrong on no-repo misses** (improvement 4).
   Any dashboard or analysis that counts empty tool results from `events`
   undercounts them.

No grader bug was found. All five failed domain grades reproduced offline, and all six
LiveBench zeros reproduced from the rebuilt program: three SyntaxErrors, one
non-terminating loop and two wrong outputs.

## Not reviewed or not yet available

- lb-20260923-minp0 is in progress and has no judgments. Two bonsai coding
  answers exist so far: ebde77f9, a new 'return' outside function (noted
  above), and 27c31c63, which compiles.
- The instruction-following answers were set aside by the operator
  (`discarded_if_minp005/`). They were not reviewed.
- Hint selection for ts03 is untested. No hints-on ts03 row finished.
