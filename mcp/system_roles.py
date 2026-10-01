#!/usr/bin/env python
"""ONE system message, first: the shape the served chat template accepts.

WHY. The served template (mcp/fixtures/bonsai_chat_template.jinja) renders
`messages[0]` as the system block and raises "System message must be at the
beginning." for a system message anywhere else; llama-server hands it a
`developer` message as `system`. Clients send both roles freely: Codex sends
`instructions` AND a leading developer message (then more developer
messages mid-session), Pi and OpenCode a leading developer item, chat
clients a `developer` first message (docs/HARNESS-RESPONSES.md). And the
proxy appends to the system text (proxy.add_system_tail: the MCP host's
line, the craft index), which must find it.

THE RULE (one helper for both wires: /v1/chat/completions and
mcp/responses_api.py):

  - the leading run of system/developer messages -- every one before the
    first message of another role -- becomes ONE `system` message, their
    text joined by a blank line, in order;
  - a system/developer message after that becomes a `user` message with the
    same text (its words reach the model where the client put them; the
    template cannot take a second system block);
  - a list that already has at most one leading `system` message with
    string content and no other system/developer message is returned AS IS
    (the same object): a chat client's prompt stays byte-identical.

`one_system(messages) -> (messages, record)`; the record counts what moved
({"merged": n, "developer_as_user": n}), for x_yamadori.
"""
from __future__ import annotations

SYSTEM_ROLES = ("system", "developer")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text") or "" for p in content
                       if isinstance(p, dict))
    return "" if content is None else str(content)


def one_system(messages: list) -> tuple[list, dict]:
    msgs = messages or []
    lead = 0
    while lead < len(msgs) and isinstance(msgs[lead], dict) and \
            msgs[lead].get("role") in SYSTEM_ROLES:
        lead += 1
    later = [i for i in range(lead, len(msgs)) if isinstance(msgs[i], dict)
             and msgs[i].get("role") in SYSTEM_ROLES]
    if not later and (lead == 0 or (
            lead == 1 and msgs[0].get("role") == "system"
            and isinstance(msgs[0].get("content"), str))):
        return messages, {}
    rec: dict = {}
    out: list = []
    if lead:
        texts = [_text(m.get("content")) for m in msgs[:lead]]
        out.append({"role": "system",
                    "content": "\n\n".join(t for t in texts if t)})
        if lead > 1 or msgs[0].get("role") != "system":
            rec["merged"] = lead
    for m in msgs[lead:]:
        if isinstance(m, dict) and m.get("role") in SYSTEM_ROLES:
            out.append({"role": "user", "content": m.get("content")
                        if m.get("content") is not None else ""})
        else:
            out.append(m)
    if later:
        rec["developer_as_user"] = len(later)
    return out, rec
