# Codex CLI shapes

**Version:** @openai/codex **0.157.1** (native `codex.exe`); one fixture
from **0.133.0**. Codex speaks only the Responses API.

**Sources**

- The capture-stub bodies of 2026-09-26 (`docs/HARNESS-CODEX.md` s3-s4;
  session scratchpad `final.body.json`, `vis.body.json`,
  `patch.body.json`): `instructions`, a `developer` message (skills and
  permissions), the `<environment_context>` user message, then the user's
  message; `-i square.png` as input_text / input_image / input_text;
  `view_image` and its `function_call_output` array of `input_image`;
  `apply_patch` as a `custom_tool_call` whose `input` is a V4A patch;
  `exec_command(cmd, workdir)` and its output string ("Process exited with
  code N").
- 0.133.0 (`namespace-and-hosted-tools`): the WSL capture
  (`cxcap/cap.jsonl` #0, `docs/HARNESS-RESPONSES.md` s5): the
  `multi_agent_v1` namespace tool and hosted `web_search` on every request.
- **In-place compaction: FROM SOURCE, not captured** -- the head of
  `codex-rs/prompts/templates/compact/prompt.md` as quoted in
  `mcp/compaction.py`. What 0.157.1 sends when it compacts
  (`x-codex-beta-features: remote_compaction_v2`) is unknown
  (docs/HARNESS-CODEX.md live plan item 9).

**Changed from the captures:** the 17,174-character instructions and the
developer text trimmed to their heads; the environment context's cwd set to
`/work/project`; installation, thread and window ids replaced by synthetic
UUIDs; content and images synthetic.
