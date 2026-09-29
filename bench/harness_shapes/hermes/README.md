# Hermes Agent shapes

**Version:** hermes-agent commit `ee5ee84` (0.21.5), the Windows install
`bench/octopus/run.py` drives.

**Sources**

- Chat main turns, tool calls and results: the Octopus runs' relay logs
  (`bench/octopus/relay.py`: request summaries) and Hermes' own
  `hermes.jsonl` event logs (the exact argument keys: `write_file(path,
  content)`, `patch(path, old_string, new_string)` / `patch(mode="patch",
  patch)`, `terminal(command, workdir)`; results as JSON strings
  `{output, exit_code, error}` / `{bytes_written, ...}`).
- Side calls: `index/corpus.sqlite3` (read-only) turn events -- the title
  namer (turns 3711, 3785), the security reviewer (3703, 3857), the
  flattened compaction (3907, itself a synthetic test compaction), the
  resume notice (304) and the compaction note (3642). The title namer's
  full text is from the Responses capture below.
- Vision: `tools/vision_tools.py` in the install -- `_media_messages` (:674,
  the aux call: one user message, text + one `image_url`) and
  `_build_native_vision_tool_result` (:490, a tool result whose content is
  `[text, image_url]`).
- Responses wire: the recording-stub capture of 2026-09-26
  (`docs/HARNESS-RESPONSES.md` s2; session scratchpad `hcap/cap.jsonl`
  rows 6 and 7): `instructions` + a string user item, flat tools with
  `strict: false`, a content-hash `prompt_cache_key`.

- Responses tool loop (`responses__reread-unchanged.json`): the item shapes
  Hermes replays -- `function_call {call_id, name, arguments}`,
  `function_call_output {call_id, output}` (a string), assistant `message`
  items with `output_text` and `status` -- from
  `agent/codex_responses_adapter.py` (`_replay_tool_call_items`,
  `_tool_output_items`, `_assistant_message_item`), checked against the
  Octopus v0f-V0 run's session (its read_file / write_file result keys).

**Changed from the captures:** the system prompt trimmed to its head; every
task, file and output is a stand-in; tool descriptions trimmed (names and
parameter keys kept); ids in our carrier form are synthetic.
