# OpenCode shapes

**Version:** opencode-ai **1.18.32** (a bun-compiled binary; AI SDK
`@ai-sdk/openai-compatible` for chat, bundled `@ai-sdk/openai` 3.0.84 for
Responses).

**Sources**

- Chat: the hands-on test of 2026-09-26 (`docs/HARNESS-OPENCODE.md`),
  recording relay logs `relay.1.jsonl` / `relay.2.jsonl` in the OpenCode
  test directory: the title call (relay.1 #0) with its two user messages,
  main turns with `X-Session-Id` / `x-session-affinity`, `write(filePath,
  content)` and `bash(command, workdir)` calls with our ids echoed, the
  user-attached image (relay.2 #1: two text parts, an `image_url`, the
  question), the `read` of a png and its synthetic user turn
  `Attached media from tool result:` (relay.2 #6), the fork's new session id
  over the original's history (relay.2 #13).
- `edit(filePath, oldString, newString)`: the binary's tool registry
  (@101022464); not called in the capture.
- The title generator's text: the Responses stub capture (`occap/cap.jsonl`
  #0), task and rules kept, examples trimmed -- its rule "the user message
  you are summarizing" is what the old compaction rule misread.
- **Compaction: FROM SOURCE, not captured live** (OpenCode compacts only
  with `limit.context` set): session/compaction `buildPrompt` read from the
  binary -- "Here is the conversation so far:" + `<conversation>` records +
  "Create a new anchored summary from the conversation history in the
  <conversation> tags above ...".
- Responses wire: the recording-stub capture (`occap/cap.jsonl` #0, #1):
  a leading `developer` item, `prompt_cache_key` = the `ses_` id,
  `max_output_tokens`.

**Changed from the captures:** paths neutralised to `C:\work\project`;
session ids synthetic (`ses_` + 26 characters, the real form); content and
images synthetic; the system prompt trimmed.
