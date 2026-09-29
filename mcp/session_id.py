#!/usr/bin/env python
"""A conversation's identity: an EXPLICIT session id, never inferred.

THE DEFECT (#41, Octopus v0d-V0-xhigh-1, 2026-09-25). A conversation was
keyed by a hash of its account and its first two messages (nebari.key_of).
Every Octopus V0 run sends the same Hermes system prompt and the same task
from the same account, so run v0d's first request was filed as a
continuation of v0b and v0c: its deep-thinking state said the kickoff was
done (epoch 2, episode 5, in cooldown) and the kickoff plan never ran. Anyone
who opens a new conversation the way an old one opened inherited the old
one's deep state, ledger decisions, work log and slot pin.

THE RULE (operator, 2026-09-25): identity comes from the first of
  1. `prompt_cache_key` in the request body (OpenAI's chat-completions field;
     Codex/OpenCode-style clients send one per conversation),
  2. the `X-Yamadori-Session` header,
  3. OUR ID, CARRIED IN THE TOOL-CALL IDS the proxy returns (below), read
     from any assistant `tool_calls[].id` or tool message `tool_call_id` in
     the history that has our form -- the first found wins,
  4. OUR SUMMARY LINE: a compaction summary the proxy writes carries the id
     on a line of its own (`yamadori session <id>`), so a continuation that
     keeps only the summary keeps the conversation,
and otherwise a request is a NEW conversation: the proxy mints an id --
with or without earlier answers in it. Nothing is guessed, so the old
fallback (`none`, nebari.key_of on the first two messages) is no longer
reached by a conversation request.

THE CARRIER (operator, 2026-09-25: "no visible line in answers"). The first
design put `yamadori session <id>` on the first line of a new conversation's
answer; it broke exact-output callers live ("Reply with exactly: ok" ->
"yamadori session dda0ff896a3c\\n\\nok"). The id now rides inside every tool
call id the proxy returns:

    call_3f9a2c7e41b0_9c01d2e7

`call_`, the 12 hex digits of the session id, `_`, 8 random hex digits (the
call's own uniqueness: the ledger keys a call turn by its ids, so two calls
of one conversation must never share one). 26 characters. Why this is safe:
  - every OpenAI-compatible client must echo `tool_calls[].id` and the tool
    result's `tool_call_id` verbatim, or its tool results would not pair;
  - Hermes (agent/message_sanitization.py) only coalesces, splits on `|` and
    de-duplicates with a `_d<n>` suffix -- none of which a unique id without
    `|` triggers; the suffix is tolerated here anyway;
  - the served chat template never renders a tool-call id (mcp/test_sessions
    renders it both ways), so the model never sees one and the slot's prompt
    is byte-identical whatever the ids are;
  - llama-server generates 32 alphanumeric characters with no `_`
    (tools/server/server-common.cpp gen_tool_call_id), so an id it made can
    never match; it pairs results to calls by id only for DeepSeek V4
    (common/chat.cpp deepseek_v4_sort_tool_results), and the client echoes
    our ids on both sides anyway, so every pair stays consistent upstream.
A conversation whose answers make no client tool call has no carrier: its
next request is a new conversation (AGENTS.md, "A conversation is named by
an explicit session id").

THE SUMMARY LINE, the one visible marker left, only in compaction summaries:

    yamadori session 3f9a2c7e41b0

twelve lowercase hex digits, nothing else on the line. Hermes' compaction
redaction leaves it alone: no `=` or `:` after a `session`/`token` word (its
key=value rules), and under 16 hex digits (its bare-hex credential rule,
agent/redact.py _looks_like_opaque_credential). The ledger strips it from
anything it renders (proxy.ledger_restore), so the model never reads it.
"""
from __future__ import annotations

import hashlib
import re
import secrets

PREFIX = "yamadori session "
ID_HEX = 12
LINE = re.compile(r"(?m)^[ \t]*yamadori session ([0-9a-f]{12})[ \t]*$")
# Our tool-call ids. The optional `_d<n>` is Hermes' duplicate repair
# (uniquify_tool_call_ids), which keeps the id it extends.
CALL_PREFIX = "call_"
CALL_SUFFIX_HEX = 8
CALL_ID = re.compile(r"^call_([0-9a-f]{12})_[0-9a-f]{8}(?:_d[0-9]+)?$")
# A prompt_cache_key is the client's own string; bounded, never stored raw.
MAX_CACHE_KEY = 1024

SOURCES = ("prompt_cache_key", "header", "tool_call_id", "summary_line",
           "answer_record", "minted",
           # Recorded on a flattened compaction (a utility call, no session
           # of its own) that proxy._serve_compaction mapped to a
           # conversation with our id: the id its summary line carries.
           "compaction_map")
# The sources that are OUR id: its tool calls carry it.
OURS = ("minted", "tool_call_id", "summary_line", "answer_record")


def mint() -> str:
    return secrets.token_hex(ID_HEX // 2)


def call_id(sid: str) -> str:
    """A fresh tool-call id carrying `sid`."""
    return f"{CALL_PREFIX}{sid}_{secrets.token_hex(CALL_SUFFIX_HEX // 2)}"


def of_call_id(value) -> str | None:
    """The session id a tool-call id carries, or None when we did not mint
    it (llama-server's, another server's, a client's own)."""
    m = CALL_ID.match(value) if isinstance(value, str) else None
    return m.group(1) if m else None


def carry(calls: list[dict], sid: str) -> tuple[list[dict], dict]:
    """(calls with ids that carry `sid`, {original id: new id}). A call that
    already carries `sid` keeps its id; every other call -- the model's, as
    llama-server named it -- gets a fresh one. New dicts; `calls` is not
    changed."""
    out, ids = [], {}
    for c in calls or []:
        if not isinstance(c, dict):
            out.append(c)
            continue
        old = c.get("id")
        new = old if of_call_id(old) == sid else call_id(sid)
        while new != old and new in ids.values():
            new = call_id(sid)
        if old:
            ids[old] = new
        out.append(dict(c, id=new))
    return out, ids


def line(sid: str) -> str:
    """The text put in front of a compaction summary: the line and a blank
    line, so a markdown client shows it as a paragraph of its own."""
    return f"{PREFIX}{sid}\n\n"


def _text(m: dict) -> str:
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, list):
        c = "\n".join(p.get("text") or "" for p in c
                      if isinstance(p, dict) and p.get("type") == "text")
    return c if isinstance(c, str) else ""


def find(text: str) -> str | None:
    m = LINE.search(text or "")
    return m.group(1) if m else None


def strip(text: str) -> str:
    """`text` without our line (and the blank line after it)."""
    if not text or PREFIX not in text:
        return text
    return re.sub(r"(?m)^[ \t]*yamadori session [0-9a-f]{12}[ \t]*(?:\n\n?|$)",
                  "", text)


def from_messages(messages: list[dict]) -> tuple[str | None, str | None]:
    """(id, source): the first tool-call id of ours in the history -- an
    assistant's `tool_calls[].id` or a tool message's `tool_call_id` --
    else the first summary line in any message's text. (None, None) when
    the history carries neither."""
    msgs = [m for m in messages or [] if isinstance(m, dict)]
    for m in msgs:
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                sid = of_call_id(c.get("id")) if isinstance(c, dict) else None
                if sid:
                    return sid, "tool_call_id"
        elif m.get("role") == "tool":
            sid = of_call_id(m.get("tool_call_id"))
            if sid:
                return sid, "tool_call_id"
    for m in msgs:
        sid = find(_text(m))
        if sid:
            return sid, "summary_line"
    return None, None


def summary_of(messages: list[dict]) -> str | None:
    """The id on OUR summary line in the history (a compaction summary the
    proxy wrote), or None. The one COMPACTION evidence: a harness that
    compacted keeps the summary, a fork keeps the full history instead."""
    for m in messages or []:
        sid = find(_text(m)) if isinstance(m, dict) else None
        if sid:
            return sid
    return None


def cache_key_of(body: dict) -> str:
    """The body's prompt_cache_key when it is a usable string, else "".
    `promptCacheKey` is accepted as an alias: the Vercel AI SDK's
    openai-compatible provider (OpenCode's `setCacheKey: true`) sends the
    camelCase key (docs/HARNESS-OPENCODE.md, 2026-09-26)."""
    b = body or {}
    v = b.get("prompt_cache_key")
    if v is None:
        v = b.get("promptCacheKey")
    if not isinstance(v, str):
        return ""
    v = v.strip()
    return v if 0 < len(v) <= MAX_CACHE_KEY else ""


def conversation_key(account: str, source: str, value: str) -> str:
    """The conversation key for an explicit id: account, source and id only
    -- never the messages, so a compacted conversation keeps it. Every
    source of OUR id uses "token", so a conversation keeps its key whichever
    carrier brought the id back."""
    h = hashlib.sha256()
    h.update(("account\x00" + (account or "") + "\x00" + source + "\x00"
              + value).encode("utf-8", "replace"))
    return h.hexdigest()[:20]


# A CLIENT'S KEY THAT CHANGES MID-CONVERSATION (coordinator's decision,
# 2026-09-26; docs/HARNESS-RESPONSES.md gap 4). Hermes' Responses transport
# sends `prompt_cache_key` = a hash of (its session, instructions, tools), so
# the key CHANGES when Hermes rebuilds its system prompt at a compaction --
# and a new key was a new conversation (a new slot, deep state, ledger).
# Only explicit ids are used to join them:
#   - a conversation named by prompt_cache_key also carries an id of ours in
#     the tool-call ids it returns: `carrier_of(its key)`, recorded as an
#     alias of that key (`carrier_ref`), so the id names the conversation;
#   - a prompt_cache_key NEW to us (no alias row), with no session header,
#     whose history carries OUR SUMMARY LINE -- COMPACTION evidence: the
#     line opens a summary the proxy wrote, in place or for a flattened
#     compaction _serve_compaction mapped (a key-named conversation's line
#     names its carrier) -- CONTINUES that conversation; the new key is
#     recorded as its alias (`cache_key_ref`);
#   - a new key whose history carries our id ONLY in tool-call ids is a
#     FORK (coordinator, 2026-09-26: a fork keeps the full history; Codex's
#     `fork` sends only a new thread key): a NEW conversation, recorded with
#     `forked_from` <id>. The slot may still reuse the prefix by affinity.
#   - an unknown key with no carried id is a new conversation, as before; a
#     key seen before keeps the conversation it named first.
ALIAS_KIND = "session_alias"


def carrier_of(conversation: str) -> str:
    """The id of ours a conversation named by the client carries in its
    tool-call ids: 12 hex digits derived from its key."""
    return hashlib.sha256(("carrier\x00" + (conversation or "")).encode()
                          ).hexdigest()[:ID_HEX]


def carrier_ref(sid: str) -> str:
    return "sid:" + sid


def cache_key_ref(cache_key: str) -> str:
    return "pck:" + hashlib.sha256((cache_key or "").encode(
        "utf-8", "replace")).hexdigest()[:32]


def alias_get(account: str, ref: str) -> str | None:
    import nebari
    v = nebari.ledger_get(account, ref, ALIAS_KIND)
    return v if isinstance(v, str) and v else None


def alias_put(account: str, conversation: str, ref: str) -> None:
    import nebari
    nebari.ledger_put(account, conversation, ref, ALIAS_KIND, conversation)


def conversation_of_id(account: str, sid: str) -> str:
    """The conversation an id of ours names: the client-named conversation
    it is the carrier of, else the id's own conversation."""
    return alias_get(account, carrier_ref(sid)) or \
        conversation_key(account, "token", sid)


def short(value: str) -> str:
    """What x_yamadori shows of a client's own id: a hash prefix."""
    return hashlib.sha256((value or "").encode("utf-8", "replace")
                          ).hexdigest()[:8]
