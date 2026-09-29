# Pi shapes

**Version:** @earendil-works/pi-coding-agent **0.87.1** (pi-ai
openai-completions / openai-responses).

**Sources**

- Chat: the hands-on test of 2026-09-26 (`docs/HARNESS-PI.md`), recording
  relay log `logs/relay.jsonl` in the Pi test directory (rows `#0`-`#43`):
  the `developer` instruction role with `reasoning: true` (#1-#5, the 502
  at `high`), user turns as lists of text parts (every row), the affinity
  headers (#8), `prompt_cache_key` + `prompt_cache_retention` (#11),
  `write(path, content)` (#16), `edit(path, edits[{oldText, newText}])`
  (#17), `bash(command)`, the `read` of a png, its tool text `Read image
  file [image/png]` and the synthetic user turn `Attached image(s) from tool
  result:` (#22), the `canvas.html` write (#23), the user-attached image with
  its `<file name=...>` part (#20), the compaction calls (#34 `<conversation>`
  records, #35 `# Conversation` / `# Instructions`) and the turn after
  (#36).
- The edit argument forms not captured (edits as a JSON string, the legacy
  top-level oldText/newText): `dist/core/tools/edit.js:43-73`
  `prepareEditArguments`.
- The summariser's system prompt and the compaction prompts' heads:
  `dist/core/compaction/utils.js:139`, `compaction.js:400` and `:684`.
- Responses wire: the recording-stub capture (`picap/cap.jsonl` #0): a
  leading `developer` item, `prompt_cache_key` = the session id,
  `session_id` header, tools without `strict`.

**Changed from the captures:** paths neutralised; the test image replaced by
an 8x8 PNG drawn for these fixtures; content synthetic; the system prompt
trimmed.
