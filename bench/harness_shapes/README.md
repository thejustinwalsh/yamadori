# Harness request shapes

Real request shapes from each harness, replayed offline through every
decision point by `mcp/test_harness_decisions.py` (run by
`scripts/run_tests.py`). One directory per harness; one JSON file per
request.

```
{
  "id": "<harness>/<wire>/<name>",        unique
  "harness": "pi", "harness_version": "...",
  "wire": "chat" | "responses",
  "source": "which capture the shape came from, or FROM SOURCE",
  "headers": {...},                        lower-case names; session headers
                                           are read by server.session_of_headers
  "body": {...},                           the request body as sent
  "expect": {<decision point>: <expected decision>},
  "why": {<decision point>: "why, from the harness's source or intent"}
}
```

Decision points and the forms `expect` takes (the test's docstring says
what each one calls):

| point | expect |
|---|---|
| `accepted` | `true`: validate_chat / responses_api.to_chat serve it |
| `route` | the route class (`agent_step`, `prose`, `utility`, ...) |
| `utility` | `{"utility": bool, "kind": "title" \| "classifier" \| "compaction" \| ...}` |
| `compaction` | `{"none": true}`, `{"in_place": true}`, or `{"harness": "pi", "records_min": 3}` |
| `session` | `{"source": "prompt_cache_key" \| "header" \| "tool_call_id" \| "minted" \| null, "header": "x-session-id"}` |
| `deep` | any of `kickoff`, `speaking`, `still_broken` (bool), `struggle_min`, `struggle_max` |
| `roles` | `{"one_system_first": true, "addendum": bool}` |
| `image` | `{"forms": ["tool_media_turn"]}` (the register's forms) |
| `tool_code` | a list: `{"call": [assistant-with-calls index, call index], "detected": "table" \| "shape", "kind": "file" \| "edit", "language": ..., "blocking": bool}` |
| `skills` | `{"includes": [...], "excludes": [...]}` (authored skills) |
| `project` | any of `root`, `project_writes_seen` (over every prefix); always that each tool result goes up as the harness sent it (the situation lines were removed 2026-09-27): the conversation replayed prefix by prefix (each request ending on tool results) through the wire and `proxy.prepare` |

Expectations are labelled from the HARNESS's source and intent, never from
what the code does today. A disagreement is marked in the test's `KNOWN`
table with its reason, not by changing the expectation.

**Adding a harness:** a new directory, its fixtures, and a `README.md` that
names the harness version and where each shape came from. The matrix the
test prints grows a row.

**Privacy (the repository is public):** no key, no operator path or user
name, no operator image or prompt, no Octopus task-spec text (skills must
never see answer keys). Content is synthetic, or a trimmed head of the
harness's own prompt text. Images are 8x8 solid-colour PNGs drawn for these
fixtures.
