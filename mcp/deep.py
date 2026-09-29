#!/usr/bin/env python
"""Deep thinking with triggers (Phase 0.6; operator, 2026-09-24).

WHAT CHANGED. Deep thinking -- the second brain taking a hard problem off
main (mcp/shomen.py) -- used to run on ONE class of request: a
`library_question` (mcp/route.py) where the regex, the held-symbol lookup and
Laya's route_in head agreed (mcp/selection.py). Harness traffic never got it:
every Hermes request routes `agent_step` (#19, docs/SELF-IMPROVEMENT-LOG.md).
The operator replaced that gate with FOUR TRIGGERS, allowed at tiers `xhigh`
and `max` (tiers.TIERS `investigate`), on the one helper lane:

  1. MODEL-CHOSEN   main calls `yama_think_deeply` (THINK_TOOL below),
                    with `yama_plan` the non-image tools this service adds
                    to main. The proxy runs
                    it through shomen.run("investigate") as a hidden hop the
                    ledger replays (proxy._run_turn), and the answer opens
                    with the fold-back (seed line + "After thinking deeply,").
  2. STRUGGLE       from what the client sends back, deterministically
                    (struggle_scan), from the STRUCTURED fields of tool
                    results only (an exit status, the harness's error field,
                    its exit_code_meaning): the same tool failing with the
                    same first error line again, the same file rewritten
                    after a failure, a failing command re-run failing the
                    same way, our fix-up at its round cap -- one signal per
                    event, every failure counted. At STRUGGLE_THRESHOLD
                    signals since the episode boundary (the last run, or a
                    compaction), deep thinking runs BEFORE main generates
                    and its hand-off is prefilled as main's reasoning. A run
                    moves the boundary, so its signals are not counted
                    again; a pattern that recurs fires again at the
                    threshold. #45, docs/SELF-IMPROVEMENT-LOG.md. Removed
                    2026-09-27 (docs/CONSTANTS-AUDIT.md, invented
                    constants): the 40-message window, the cooldown, the
                    same-pattern suppression, the lane deferral, the error
                    word list, the "still broken" phrases, the error-line
                    normaliser and the ENVIRONMENT table.
  3. KNOWN-HARD     the conversation USES a held package the model cannot
     AREA           have seen (unseen(): the package first published after
                    MODEL_CUTOFF, or a new major, or the operator's list),
                    or a skill that applies declares `escalate`. Once per
                    package (or skill) per conversation.
  4. KICKOFF        the conversation's INITIAL prompt -- its first user turn
                    only (operator, 2026-09-27: a follow-up after a finished
                    answer is not planned) -- whatever its length ("Doesn't
                    matter the length of the prompt, we should always give
                    it a planning turn"; the KICKOFF_TOKENS size threshold,
                    1,500, and its learner are gone): the PLANNING goes to
                    the second brain (shomen's `plan` job, with its own
                    budget, #60), the plan comes back as the result of an
                    inserted `yama_plan` call (proxy.synthetic_hop), and main
                    starts acting. Evidence: Octopus V0 steps 1-3 spent
                    12-14k reasoning tokens each planning on main (#18).

Priority when several apply to one request (one run per request, one helper
lane): a header that forces deep thinking > struggle > kickoff > area. A
header that forces it OFF wins over everything. Laya is NOT consulted.

E1 (mcp/e1.py; YAMADORI_E1=1, off by default). The triggers consult E1's
heads on ONE embedding of the request when a head is trained, and the rules
otherwise: `escalate` decides a struggle in place of the threshold (once at
least one signal exists), and `route_in` adds a
known-hard area: a
library_question the head says needs source read first (104/120 and 110/141
on the two held-out sets, docs/E1.md). The rule's unseen-package area still
fires on its own. The embedded state is stored with the row (`e1_state`) so
the idle learner (e1.learn) can retrain on the labels these rows earn.

EVERY NUMBER HERE IS A CHOICE, NONE MEASURED. STRUGGLE_THRESHOLD 3 is a
starting point: an environment variable pins it, and otherwise the idle-time
learner (mcp/deep_learn.py) moves it within BOUNDS from labelled outcomes,
recording each change with its n.

THE SELF-IMPROVEMENT LOOP. Every request of a conversation leaves ONE row in
`deep_decisions` (the corpus database, next to the events it summarises):
the trigger or NONE, the signals, the thresholds in force, whether deep
thinking ran and how big its hand-off was. On each later request of the same
conversation the open rows are OBSERVED (observe()): did the struggle stop,
did the check pass, was the same file rewritten again, was the hand-off
used. After OUTCOME_REQUESTS observations (or at once on a clear outcome) a
non-decision gets its LABEL: fine / missed / escalated_later. A RUN is
judged against the pattern it ran on: one recurrence is not_helped at once;
helped / wasted need HELPED_WINDOW
observations without one (#45: with 5, and the episode a run resets,
`helped` was automatic). Test traffic (corpus.account_traffic) is recorded
and never learned from.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import queue
import re
import sqlite3
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------- numbers --
# Each is a CHOICE (see the docstring). BOUNDS limit what the learner may set.
# `kickoff_tokens` (1,500, bounds 500-8,000, YAMADORI_KICKOFF_TOKENS) was
# retired 2026-09-27 (operator: every new task is planned); its old
# deep_adjustments rows stay readable and no longer set anything.
STRUGGLE_THRESHOLD_DEFAULT = 3
BOUNDS = {"struggle_threshold": (2, 6)}
ENV = {"struggle_threshold": "YAMADORI_STRUGGLE_THRESHOLD"}
DEFAULTS = {"struggle_threshold": STRUGGLE_THRESHOLD_DEFAULT}
# Struggle is counted from the episode boundary (the last run, or a
# compaction) to the end: STRUGGLE_WINDOW (40 messages) and the cooldown
# after a run (COOLDOWN_REQUESTS, 3) were invented and are gone
# (docs/CONSTANTS-AUDIT.md, 2026-09-27).
# Later requests a row is observed over before it is labelled.
OUTCOME_REQUESTS = int(os.environ.get("YAMADORI_DEEP_OUTCOME_REQUESTS", "5"))
# A RUN's row is labelled `helped` only after this many later requests with
# no recurrence of the pattern it ran on (#45 (e); a CHOICE). With 5 it was
# automatic: a run restarted the episode and the (since removed) cooldown
# held, so "no struggle in the next 5 requests" held by construction
# (Octopus v0e-V0 prompt 2: 5 of 6 runs `helped`, the next run 3-50 minutes
# later each time).
HELPED_WINDOW = int(os.environ.get("YAMADORI_DEEP_HELPED_WINDOW", "20"))
# THE LABEL RULE (#52 in docs/SELF-IMPROVEMENT-LOG.md, remedy 7 of
# docs/research/OVERTHINKING.md; 2026-09-26). Octopus v0e p2: six struggle
# runs, 6,536 s, and no project file changed after any of them -- five were
# labelled `helped`; with v0b, 10 of the 11 client runs. From rule 2 a RUN is
# `helped` only if a PROJECT file (mcp/progress.is_project) was written after
# it within HELPED_WINDOW observations; with none it is `no_effect`. The rule
# a label was made under is stored with it (deep_decisions.label_rule); rows
# labelled before rule 2 (label_rule NULL) are not relabelled, and the
# learner (deep_learn) and E1 (e1._deep_label) ignore RUN rows labelled
# under the old rule. A request with X-Yamadori-Features
# {"helped_needs_change": false} or YAMADORI_DEEP_HELPED_NEEDS_CHANGE=0 is
# labelled under rule 1 (the row's signals.helped_rule).
LABEL_RULE = 2
# A trigger waits for the helper lane like every helper job
# (admission.WAIT_SECONDS); a lane still busy then skips that request's
# trigger (recorded, label `skipped`). The deferral counter (DEFER_REQUESTS,
# 3) and the one-wait-per-episode rule were invented and are gone
# (docs/CONSTANTS-AUDIT.md, 2026-09-27).
# The spec-size estimate: ~4 characters a token (English and code; a choice).
CHARS_PER_TOKEN = 4
# The model's training cutoff for the known-hard-area rule. The model card
# states none; Bonsai 2 derives from a 2025 Qwen base, so 2025-12-31 is a
# CHOICE. Compared with a package's FIRST publish and its releases' dates
# (deps.package_history), never a version's own date.
MODEL_CUTOFF = os.environ.get("YAMADORI_MODEL_CUTOFF", "2025-12-31")
# What the second brain gets of a task, a spec, the context and the failing
# output is NOT cut (docs/CONSTANTS-AUDIT.md, 2026-09-27): QUESTION_CHARS
# 6000, CONTEXT_CHARS 1500, the task's [:1500], the error excerpts (300 +
# 400 characters of 3 errors) and the event / path / phrase counts were
# invented, the same class as MAX_FINDING_CHARS. A question too big for the
# helper's window meets the window mechanism (tiers.budget; a budget event),
# never a silent cut.
#
# E1 is the one exception, a real constraint (B): its heads were trained on
# states whose question was cut to 2,000 characters and context to 1,500
# (bench/e1/build_heldout2.py:193-194), and e1.render caps the state at
# e1.STATE_CHARS inside the embedder's window (config.yaml `embeddings`,
# -c 8192). Serving a longer state than the heads were fitted on is
# off-distribution, so the E1 input keeps the training cut.
E1_QUESTION_CHARS = 2000
E1_CONTEXT_CHARS = 1500

# THE TOOLS OURS ON MAIN ARE `yama_*` (operator, 2026-09-27): a name no
# harness offers, so a clash with the client's tools cannot happen, and one
# that reads as this service's. The old names are LEGACY_TOOL_NAMES in
# proxy: stored ledger rows and records written before the rename still
# carry them, and they are read as the new ones (proxy.canonical_tool_name).
TOOL_NAME = "yama_think_deeply"
LEGACY_TOOL_NAME = "think_deeply"
# THE DESCRIPTION IS A PROMPT (AGENTS.md "Tool descriptions are prompts"): it
# leads with the question it answers, contrasts itself with guessing and with
# repeating a failing edit, and lists the phrasings that should trigger it.
# Its wording is a CHOICE; the learner proposes variants, it never arms one.
# It says where it runs (operator, 2026-09-27): on the Yamadori server, not
# in the workspace -- it cannot read or change the user's files.
THINK_DESCRIPTION = (
    "Answers 'what is actually going on here, and what should I do next?' "
    "when you are stuck or unsure: a second model researches the question in "
    "library source, skills, this service's notes and the web, then hands "
    "back facts with citations, what it searched and found nothing for, open "
    "questions and a next step. Call it INSTEAD of guessing an API or "
    "repeating an edit that already failed. Call it when: a fix failed "
    "twice and the error persists; you are unsure of a library's API or "
    "version; you are about to read a package's installed files (its dist "
    "or .d.ts under node_modules) or write scratch files to find out how it "
    "works -- it reads the library's own source and answers in one call; "
    "the answer needs many files or docs read; the user says it is "
    "still broken, didn't work, or shows the same error. A server tool: it "
    "runs on the Yamadori server and does not touch your workspace. It takes "
    "minutes; the result arrives in your context, and your answer continues "
    "from it.")
THINK_TOOL = {"type": "function", "function": {
    "name": TOOL_NAME,
    "description": THINK_DESCRIPTION,
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": (
            "One self-contained question. The researcher cannot see this "
            "conversation: name the files, symbols, versions and the exact "
            "error.")},
        "tried": {"type": "string", "description": (
            "What was tried already and how it failed, so it is not "
            "repeated.")}},
        "required": ["question"]}}}

# YAMA_PLAN (operator, 2026-09-27; pagoda-h2): the plan job (shomen.run
# "plan", its budget #60) as a server tool on main, wherever
# yama_think_deeply is offered (think_tool_offered). The proxy runs it as a
# HIDDEN HOP exactly like yama_think_deeply: the plan is the TOOL RESULT,
# the ledger replays the call and the result, and main goes on acting from
# it -- no visible opening, no prefilled reasoning. One per request, and not
# after a run before main. The conversation's INITIAL prompt is always
# planned the same way: the proxy inserts this call and its result before
# main's first generation (proxy._deep_thinking, KICKOFF_PLAN_ARGS). The
# description is a prompt, its wording a CHOICE, unmeasured.
PLAN_TOOL_NAME = "yama_plan"
PLAN_DESCRIPTION = (
    "Answers 'how should I build this, and in what order?' before a "
    "significantly long implementation task: a second model plans it and "
    "hands back the files to create or change, the order of the steps, the "
    "key decisions, and the constraints the code holds. Call it BEFORE you "
    "start a new app, a multi-file feature, a migration or a rewrite, "
    "instead of planning it all in your head. Call it AGAIN mid-task when "
    "the plan you have is carried out or no longer fits, or before a large "
    "piece of the work it does not cover (a new subsystem, the next set of "
    "files); say what is built so far. A one-file fix or a question "
    "needs no plan. A server tool: it runs on the Yamadori server and does "
    "not touch your workspace, so you carry out the plan with your own "
    "tools. It takes minutes; the plan arrives as its result.")
PLAN_TOOL = {"type": "function", "function": {
    "name": PLAN_TOOL_NAME,
    "description": PLAN_DESCRIPTION,
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string", "description": (
            "The task to plan, self-contained: what to build or change, the "
            "libraries and versions it names, where it goes, and what is "
            "already built. The planner sees this and the conversation's "
            "recent turns.")}},
        "required": ["task"]}}}
# The synthetic call the proxy inserts for the initial prompt's plan: a
# pointer, not a copy of the task -- the task is the user message right
# before it, and a copy would sit in main's context for the whole
# conversation (a CHOICE).
KICKOFF_PLAN_ARGS = {"task": "The task in the user's message above."}


def synthetic_args(trig: dict | None, job: str) -> dict:
    """The arguments of the call the proxy inserts on main's behalf for a
    run before main (proxy.synthetic_hop): KICKOFF_PLAN_ARGS for the plan;
    for yama_think_deeply (switch deep_tool_hop) a short question naming
    what the trigger saw -- the trigger's own question, with the failing
    output, went to the second brain, and main's context already holds it.
    Pointers, not copies (a CHOICE). A server-tool trigger (kind "auto")
    carries its own (deep._auto_question)."""
    kind = (trig or {}).get("kind")
    if kind == "auto" and (trig or {}).get("synthetic_args"):
        return dict(trig["synthetic_args"])
    if job == "plan":
        return dict(KICKOFF_PLAN_ARGS)
    if kind == "struggle":
        return {"question": "What keeps failing in the steps above, and "
                            "what should I do next?"}
    if kind == "area":
        pk = ", ".join((trig or {}).get("packages") or []) or "the library"
        return {"question": f"How does {pk} work as this task uses it?"}
    return {"question": "The question in the user's message above."}
# The reasoning prefilled on the hop after a yama_think_deeply result. The
# user sees neither the result (a hidden hop) nor this thinking, so it says
# the answer carries the findings itself (live gate 2026-09-24: an answer
# that pointed at "the hand-off above", which the user never saw). Ends on a
# letter or a period (prefill rule 4, docs/SELF-IMPROVEMENT-PLAN.md).
THINK_REASONING = ("I thought this through deeply; the hand-off came back "
                   f"as the {TOOL_NAME} result. The user sees neither that "
                   "result nor this thinking. So my answer stands on its "
                   "own: it acts on the hand-off's NEXT STEP, and states the "
                   "findings it relies on in full, each with its path:line "
                   "and the source lines that show it.")
# The same line after a MACHINE-BUILT result (deep thinking wrote no
# conclusion; 2026-09-26, the deploy check's handle d28941fb): THINK_REASONING
# asked for "the findings ... in full" when there were none, and the pre-main
# path's answer then told the user the pass was "cut off before finishing".
# Mirrors proxy.FOLD_BACK_TAIL_MACHINE: what IS in the result -- source lines
# it read, or only a search log -- and what the answer does with it. No
# prohibition. UNMEASURED WORDING.
THINK_REASONING_MACHINE = (
    f"I thought this through deeply; the {TOOL_NAME} result came back with "
    "no conclusion written: the paths it retrieved and a log of its "
    "searches. The user sees neither that result nor this thinking. So my "
    "answer works the question out itself from what I know and acts on "
    "it.")
# (2026-09-27: "the source lines it read that contain the question's words"
# dropped -- the machine-built result carries none since shomen's MACHINE
# EVIDENCE excerpts were removed, docs/CONSTANTS-AUDIT.md.)


def think_reasoning(concluded: bool) -> str:
    """The reasoning prefilled on the hop after a yama_think_deeply result:
    THINK_REASONING when the hand-off concluded, else
    THINK_REASONING_MACHINE."""
    return THINK_REASONING if concluded else THINK_REASONING_MACHINE


PLAN_HEAD = ("I planned this task before acting: the plan below was written "
             "by a second model, which read what it cites. I carry it out "
             "step by step.\n\n")
TRIGGERS = ("model", "struggle", "kickoff", "area", "auto", "forced")

# ============================================================ struggle =====
# A tool result is a failure ONLY by its structured fields: an exit status,
# the harness's own error field or ok/success flag, its exit_code_meaning.
# The word list that read plain text (_ERR_TEXT: "error:", "Traceback",
# "npm ERR!", "N failed", ...) and the user's "still broken" phrases
# (STILL_BROKEN, the user_still_broken signal) were invented and are gone
# (docs/CONSTANTS-AUDIT.md, 2026-09-27): a plain-text result never signals.
# Exit-status keys. A present integer one DECIDES: 0 is success, full stop.
_EXIT_KEYS = ("exit_code", "exitCode", "returncode", "return_code",
              "exit_status", "exitStatus")
_COMMAND_KEYS = ("command", "cmd", "script", "commands")
# tool_code's note when the fix-up did not clear the errors (tool_code.finish:
# "...; still there after N rounds of repair; sent as written").
FIXUP_CAPPED = re.compile(r"still there after \d+ rounds? of repair")


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = "\n".join(p.get("text") or "" for p in c
                      if isinstance(p, dict) and p.get("type") == "text")
    return c if isinstance(c, str) else ""


def _args(call: dict) -> dict:
    raw = ((call or {}).get("function") or {}).get("arguments")
    try:
        a = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


def is_error(text: str) -> bool:
    """Does a tool result say, in its STRUCTURED fields, that it failed?

    A JSON result with an exit status decides by it: 0 is success whatever
    the output says, and exit 1 with no output at all is grep's (or diff's,
    test's) "no match", not a failure; the harness's exit_code_meaning
    "... (not an error)" wins. Otherwise the envelope: ok/success false, or
    an error field. A plain-text result, or a JSON one with none of these,
    is not a failure (no word list reads the text)."""
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith("{"):
        try:
            d = json.loads(t)
        except ValueError:
            d = None
        if isinstance(d, dict):
            inner = " ".join(str(d.get(k) or "") for k in
                             ("output", "stdout", "stderr", "result"))
            # The harness's own reading of the status wins: Hermes marks
            # grep's exit 1 "No matches found (not an error)" (v0b-V0).
            if re.search(r"(?i)\bnot an error\b",
                         str(d.get("exit_code_meaning") or "")):
                return False
            for k in _EXIT_KEYS:
                v = d.get(k)
                if isinstance(v, int) and not isinstance(v, bool):
                    if v == 0:
                        return False
                    return not (v == 1 and not inner.strip()
                                and not d.get("error"))
            if d.get("ok") is False or d.get("success") is False:
                return True
            err = d.get("error")
            if isinstance(err, (str, dict)) and err:
                return True
    return False


def _command(args: dict) -> str | None:
    for k in _COMMAND_KEYS:
        v = args.get(k)
        if isinstance(v, list):
            v = " && ".join(str(x) for x in v)
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())[:500]
    return None


def _write_edits(call: dict) -> dict[str, set[str]]:
    """path -> the fingerprints of the edits this call makes to it (a whole
    file's content, or an edit's old and new text), via tool_code.detect.
    An .html page is one file here (whole_pages): a page with only `src`
    scripts has no script unit, and its rewrites were invisible."""
    try:
        import tool_code
        units = tool_code.detect(call, whole_pages=True).get("units") or []
    except Exception:                                            # noqa: BLE001
        units = []
    out: dict[str, set[str]] = {}
    for u in units:
        if not u.get("path"):
            continue
        key = json.dumps([u.get("kind"), u.get("old"), u.get("code")],
                         ensure_ascii=False)
        out.setdefault(u["path"], set()).add(key)
    return out


# A write or patch that reported success but did not change the file: the
# harness's own `no_change` flag (Hermes: "File already contains the target
# text ... No write performed"; #33, docs/SELF-IMPROVEMENT-LOG.md). The
# errors proper are is_error's. The phrase list that read plain text
# (_NOT_APPLIED: "not applied", "hunk failed", ...) was invented and is gone
# (docs/CONSTANTS-AUDIT.md, 2026-09-27).


def not_applied(text: str) -> bool:
    """Did a write/patch tool result say, by the harness's structured
    `no_change` flag, that nothing was changed? Plain text never does."""
    t = (text or "").strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
        except ValueError:
            d = None
        if isinstance(d, dict):
            return d.get("no_change") is True
    return False


def _note_left_broken(text: str, path: str) -> bool:
    """Does tool_code's note in this assistant turn say `path` went out with
    problems our fix-up did not clear? ("Checked <path> (<lang>): N
    problems ..." -- tool_code.finish; a clean or repaired file never says
    'problem')."""
    if not text or path not in text:
        return False
    return bool(re.search(re.escape(path) + r" \([^)]*\): \d+ problems?\b",
                          text))


# ------------------------------------------------ error signature (#45) ----
# tool_error_repeat counts the SAME tool failing with the SAME error, not any
# error from a tool that failed before: two different failing `terminal`
# commands are debugging, not a loop (Octopus v0e-V0 prompt 2: 16
# tool_error_repeat signals, most of them different errors from one tool).
# The signature is the FIRST LINE of the result's error message, raw and
# compared exactly. The line picker (_SIG_SKIP / _SIG_NAMED) and the
# normaliser that stripped paths, numbers, quotes and URLs (a 160-character
# cut) were invented and are gone, and so is the ENVIRONMENT table that kept
# harness refusals and a tool's own input errors out of the count: every
# failure counts (docs/CONSTANTS-AUDIT.md, 2026-09-27).


def _error_message(text: str) -> tuple[str, int | None]:
    """(the text an error signature is read from, the exit status). A JSON
    result: its `error` field when it has one (the harness or tool's own
    message), else its output streams. Otherwise the text."""
    t = (text or "").strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
        except ValueError:
            d = None
        if isinstance(d, dict):
            code = next((d[k] for k in _EXIT_KEYS if isinstance(d.get(k), int)
                         and not isinstance(d.get(k), bool)), None)
            err = d.get("error")
            if isinstance(err, dict):
                err = err.get("message") or json.dumps(err, sort_keys=True)
            if isinstance(err, str) and err.strip():
                return err, code
            inner = "\n".join(str(d.get(k) or "") for k in
                              ("stderr", "output", "stdout", "result"))
            return inner, code
    return t, None


def error_signature(text: str) -> str:
    """The first non-blank line of a failing tool result's error message,
    exactly as written (stripped of surrounding spaces); "exit N" when there
    is no text at all."""
    msg, code = _error_message(text)
    first = next((ln.strip() for ln in msg.splitlines() if ln.strip()), "")
    if not first:
        return f"exit {code}" if code is not None else ""
    return first


def struggle_scan(messages: list[dict], start: int = 0) -> dict:
    """{events, errors} for messages[start:].

    `events` are the struggle signals, in order (struggle_events); `errors`
    every failing tool result (is_error): {tool, signature, at}. Each event
    is {kind, at, ...} with names only (a tool, a path, a signature), never
    tool output. Kinds:

      tool_error_repeat      a tool fails with the SAME error signature it
                             failed with before since `start` (#45: any
                             error from a tool that had failed counted)
      failing_command_rerun  the SAME command, run again, fails again with
                             the same signature -- counted at its result,
                             and INSTEAD of tool_error_repeat (#45: one
                             event, one signal; a re-run that passes is the
                             fix working, no signal)
      file_rewritten         see below; the rewrite's own failing result is
                             not also a tool_error_repeat
      fixup_capped           our fix-up at its round cap (our own note)

    file_rewritten (Octopus v0b-V0, #33): a write or patch of a file counts
    ONLY when the PREVIOUS write/patch of that same file failed -- its tool
    result says so in its structured fields (is_error, or the harness's
    no_change flag, not_applied), or our fix-up could not repair it
    (tool_code's "N problems" note for that path) -- or when it repeats an
    edit already sent (identical old/new text, or identical whole-file
    content). Building a file in clean patches is how agents work: the old
    rule counted every write after the second and sent 3 clean patches of
    enemies.js to 756 s of deep thinking. A failure elsewhere (a failing
    command, another file's error) is not this file's failure;
    tool_error_repeat and failing_command_rerun count those. `why` on the
    event says which: after_failure or same_edit."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    start = max(0, min(start, len(msgs)))
    calls: dict[str, dict] = {}
    errored: set[tuple[str, str]] = set()          # (tool, signature)
    failed_cmds: set[tuple[str, str]] = set()      # (command, signature)
    # path -> did its LAST write/patch fail? (None: no result seen yet)
    last_failed: dict[str, bool | None] = {}
    sent: dict[str, set[str]] = {}
    out: list[dict] = []
    errors: list[dict] = []
    for i in range(start, len(msgs)):
        m = msgs[i]
        role = m.get("role")
        if role == "assistant":
            text = _text(m)
            if FIXUP_CAPPED.search(text):
                out.append({"kind": "fixup_capped", "at": i})
            for c in m.get("tool_calls") or []:
                if not isinstance(c, dict):
                    continue
                a = _args(c)
                name = (c.get("function") or {}).get("name") or ""
                edits = _write_edits(c)
                rec = {"name": name, "cmd": _command(a),
                       "paths": list(edits), "signalled": False}
                calls[c.get("id") or f"_{i}"] = rec
                for p, keys in edits.items():
                    why = ("after_failure" if last_failed.get(p) else
                           "same_edit" if keys & sent.get(p, set()) else None)
                    if why:
                        out.append({"kind": "file_rewritten", "at": i,
                                    "path": p[-120:], "why": why})
                        rec["signalled"] = True
                    sent.setdefault(p, set()).update(keys)
                    # Went out with problems our fix-up left: failed already.
                    last_failed[p] = True if _note_left_broken(text, p) \
                        else None
        elif role in ("tool", "function"):
            c = calls.get(m.get("tool_call_id") or "") or {}
            name = c.get("name") or m.get("name") or "?"
            body = _text(m)
            if c.get("paths"):
                bad = is_error(body) or not_applied(body)
                for p in c["paths"]:
                    # Our note's verdict stands unless the result is worse.
                    last_failed[p] = bool(last_failed.get(p)) or bad
            if not is_error(body):
                continue
            sig = error_signature(body)
            errors.append({"tool": name, "signature": sig, "at": i})
            cmd = c.get("cmd")
            if c.get("signalled"):
                pass            # the rewrite already counted this event
            elif cmd and (cmd, sig) in failed_cmds:
                out.append({"kind": "failing_command_rerun", "at": i,
                            "tool": name, "signature": sig})
            elif (name, sig) in errored:
                out.append({"kind": "tool_error_repeat", "at": i,
                            "tool": name, "signature": sig})
            errored.add((name, sig))
            if cmd:
                failed_cmds.add((cmd, sig))
    return {"events": out, "errors": errors}


def struggle_events(messages: list[dict], start: int = 0) -> list[dict]:
    """The struggle signals that COUNT in messages[start:], in order
    (struggle_scan's `events`)."""
    return struggle_scan(messages, start)["events"]


def sig_digest(sig: str | None) -> str:
    """An error signature as it is STORED: its SHA-1, never the line. The
    signature is now the raw first line of a tool's error (2026-09-27), and
    it can carry the user's paths and names; rows keep names, counts and
    labels, never message text. Equal lines, equal digests: the exact
    comparison is unchanged."""
    if not sig:
        return ""
    return "sha1:" + hashlib.sha1(sig.encode("utf-8", "replace")).hexdigest()


def pattern_of(events: list[dict]) -> list[str]:
    """A struggle's PATTERN: the set of its signals as `kind|subject|
    signature` (subject: the tool or the path; signature: the error
    signature's digest, sig_digest, or why a file was rewritten). A run's
    row keeps it, and the run is judged by its recurrence (_recurrence,
    #45 (e))."""
    return sorted({f"{e['kind']}|{e.get('tool') or e.get('path') or ''}|"
                   + (sig_digest(e["signature"]) if e.get("signature")
                      else (e.get("why") or ""))
                   for e in events})


def _recurrence(pattern: list[str], scan: dict) -> list[str]:
    """What in `scan` (the messages after a run) repeats the run's pattern:
    a counted signal with a key in it, or ONE occurrence of an error whose
    (tool, signature) it held -- a single failure again is enough; the
    episode reset after the run does not hide it (#45 (e))."""
    keys = set(pattern or [])
    if not keys:
        return []
    errs = set()
    for k in keys:
        kind, subj, sig = (k.split("|", 2) + ["", ""])[:3]
        if kind in ("tool_error_repeat", "failing_command_rerun"):
            errs.add((subj, sig))
    hits = [k for k in pattern_of(scan.get("events") or []) if k in keys]
    for e in scan.get("errors") or []:
        d = sig_digest(e.get("signature"))
        if (e.get("tool"), d) in errs:
            hits.append(f"error|{e['tool']}|{d}")
    return sorted(set(hits))


def _counts(events: list[dict]) -> dict:
    out: dict[str, int] = {}
    for e in events:
        out[e["kind"]] = out.get(e["kind"], 0) + 1
    return out


# ============================================================ kickoff ======
_COMPACTION_NOTE = re.compile(r"^\W*\[?\s*context compaction", re.I)


def _tool_media(msgs: list[dict], i: int) -> bool:
    """Is msgs[i] a harness's synthetic tool-media turn (OpenCode's
    "Attached media from tool result:", Pi's "Attached image(s) from tool
    result:", Cline's image-only turn; image_input.TOOL_MEDIA_TURNS)? It
    carries a tool's image: never a kickoff or the user speaking
    (docs/VISION.md 5a). The same for a harness's CONTEXT turn
    (Codex's <environment_context>, message_text.HARNESS_CONTEXT_TURNS,
    2026-09-26): the harness writes it before the user's own message."""
    import image_input
    import message_text
    return image_input.is_tool_media(msgs, i) or (
        0 <= i < len(msgs) and message_text.harness_context(msgs[i])
        is not None)


def kickoff(messages: list[dict], continues: bool = False) -> dict:
    """{new_task, tokens, why}: is the request a new task's first user turn?
    `tokens` is the spec's size, recorded only (every new task is planned).
    `continues`: the session continues a compacted one
    (proxy._continue_after_compaction). A compaction continuation is never a
    kickoff (pre-deploy review, 2026-09-24): its first user turn is the
    summary of a task in flight."""
    import route
    import selection
    msgs = [m for m in (messages or []) if isinstance(m, dict)
            and m.get("role") not in ("system", "developer")]
    if not msgs or msgs[-1].get("role") != "user":
        return {"new_task": False, "tokens": 0,
                "why": "the request does not end on a user turn"}
    if _tool_media(msgs, len(msgs) - 1):
        return {"new_task": False, "tokens": 0,
                "why": "the last user turn is the harness's synthetic turn "
                       "carrying a tool result's media: the task in flight "
                       "goes on"}
    # Looking back, a synthetic tool-media turn is the tool result it
    # carries, not a user turn (OpenCode puts it after the assistant's
    # closing text, so a real turn after it follows that answer).
    sent = msgs
    msgs = [m for j, m in enumerate(sent) if not _tool_media(sent, j)]
    whole = _text(msgs[-1])
    instr, _att = selection.instruction_of(whole)
    tokens = len(whole) // CHARS_PER_TOKEN
    answered = any(m.get("role") == "assistant" for m in msgs)
    if not answered and (continues or any(
            m.get("role") == "user" and _COMPACTION_NOTE.match(_text(m))
            for m in msgs)):
        return {"new_task": False, "tokens": tokens,
                "why": "the first request after a compaction continues the "
                       "task in flight"}
    if route.harness_notice(instr):
        return {"new_task": False, "tokens": tokens,
                "why": "a harness notice or a bare 'continue': the task in "
                       "flight goes on"}
    prev = msgs[-2] if len(msgs) >= 2 else None
    if prev is None:
        return {"new_task": True, "tokens": tokens,
                "why": "the first user turn of the conversation"}
    # ONLY THE INITIAL PROMPT IS PLANNED (operator, 2026-09-27: "I thought it
    # was only the initial prompt that got planning?"). A user turn after a
    # finished answer -- a follow-up, a "Thanks." -- goes on without a plan.
    if prev.get("role") == "assistant" and not prev.get("tool_calls"):
        return {"new_task": False, "tokens": tokens,
                "why": "a follow-up after a finished answer: only the "
                       "conversation's initial prompt is planned"}
    return {"new_task": False, "tokens": tokens,
            "why": "a user turn in the middle of a tool loop"}


# ========================================================== known-hard =====
def _names(env: str) -> set[str]:
    return {x.strip() for x in os.environ.get(env, "").split(",") if x.strip()}


def _major(version: str) -> int | None:
    m = re.match(r"\s*v?(\d+)", version or "")
    return int(m.group(1)) if m else None


def _history(pkg: str) -> dict | None:
    """The package's registry history (deps.package_history). A seam."""
    try:
        import deps
        return deps.package_history(pkg)
    except Exception:                                            # noqa: BLE001
        return None


def unseen(pkg: str, version: str, db: str | None = None) -> str | None:
    """Why the model cannot have seen this held package version, or None.

    THE RULE (a CHOICE; the coordinator's revision of 2026-09-24, after the
    version-date rule flagged every three.js conversation -- r185 is newer
    than the cutoff, three.js is not):
      - the operator names it: YAMADORI_UNSEEN_PACKAGES (YAMADORI_SEEN_PACKAGES
        exempts one);
      - the PACKAGE was first published after MODEL_CUTOFF (its earliest
        version on the registry; deps.package_history, stored per package):
        the model cannot know a library that did not exist;
      - the used version is a new MAJOR relative to the latest release
        published before the cutoff (r3f 10 against 9.5.0). A minor or patch
        release of a long-lived package is not unseen, and neither is a 0.x
        minor (three 0.185 against 0.182: the first number is the major,
        semver's 0.x caveat notwithstanding -- three.js has used 0.x for
        thirteen years).
    A PRERELEASE is judged by these same rules, like any version
    (2026-09-27, docs/CONSTANTS-AUDIT.md: "any prerelease is unseen" was
    invented; the audit's rule -- unseen when its own release date is after
    the cutoff -- needs per-version dates, and the recorded history keeps
    stable releases only, deps.history_of_packument). r3f 10.0.0-alpha.5
    and drei 11.0.0-alpha.7 stay unseen as new majors.
    Without a recorded history (`deps.py published name@version` records it;
    deps.index_package does from now on) only the lists apply. `db` is kept
    for callers; the version's own date is no longer read."""
    if pkg in _names("YAMADORI_SEEN_PACKAGES"):
        return None
    if pkg in _names("YAMADORI_UNSEEN_PACKAGES"):
        return "named in YAMADORI_UNSEEN_PACKAGES"
    hist = _history(pkg) or {}
    first = str(hist.get("first_published") or "")[:10]
    if first and MODEL_CUTOFF and first > MODEL_CUTOFF:
        return (f"the package was first published {first}, after the "
                f"model's cutoff ({MODEL_CUTOFF})")
    before =[(d, v) for v, d in (hist.get("releases") or {}).items()
              if d and d <= MODEL_CUTOFF and _major(v) is not None]
    if before:
        last_d, last_v = max(before)
        used = _major(version)
        if used is not None and used > _major(last_v):
            return (f"a new major ({version}) after the latest release "
                    f"before the cutoff ({last_v}, {last_d})")
    return None


# ============================================================ thresholds ===
_THR_CACHE: dict = {"at": 0.0, "db": None, "val": None}
_THR_TTL = 30.0


def thresholds(fresh: bool = False) -> dict:
    """{param: {value, source, adjustment?, n?}}. An environment variable
    pins a value (source "env"); else the learner's latest standing
    adjustment (source "learned", with its n); else the default ("default:
    a choice, unmeasured")."""
    db = os.path.abspath(_db_path())
    now = time.time()
    if (not fresh and _THR_CACHE["val"] is not None
            and _THR_CACHE["db"] == db and now - _THR_CACHE["at"] < _THR_TTL):
        return json.loads(json.dumps(_THR_CACHE["val"]))
    out: dict = {}
    for p, default in DEFAULTS.items():
        env = os.environ.get(ENV[p])
        if env:
            try:
                out[p] = {"value": int(env), "source": "env",
                          "bounds": list(BOUNDS[p])}
                continue
            except ValueError:
                pass
        rec = {"value": default, "source": "default: a choice, unmeasured",
               "bounds": list(BOUNDS[p])}
        try:
            con = _db()
            try:
                row = con.execute(
                    "SELECT id, new, n FROM deep_adjustments WHERE param=? "
                    "AND reverted_at IS NULL ORDER BY created DESC LIMIT 1",
                    (p,)).fetchone()
            finally:
                con.close()
        except sqlite3.Error:
            row = None
        if row:
            lo, hi = BOUNDS[p]
            rec = {"value": int(min(max(row[1], lo), hi)), "source": "learned",
                   "adjustment": row[0], "n": row[2], "bounds": [lo, hi]}
        out[p] = rec
    _THR_CACHE.update(at=now, db=db, val=out)
    return json.loads(json.dumps(out))


# ================================================== conversation state =====
def _state_key(lineage: str) -> str:
    return "deep:" + (lineage or "")


_STATE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_STATE_LOCKS_GUARD = threading.Lock()


def state_lock(account: str, lineage: str) -> threading.Lock:
    """One lock per conversation around every load-modify-save of its state
    (pre-deploy review, 2026-09-24: decide and mark_ran raced). In-process:
    the proxy is the one writer."""
    with _STATE_LOCKS_GUARD:
        if len(_STATE_LOCKS) > 4096:
            _STATE_LOCKS.clear()
        return _STATE_LOCKS.setdefault((account or "", lineage or ""),
                                       threading.Lock())


def load_state(account: str, lineage: str) -> dict:
    if not lineage:
        return {}
    import nebari
    try:
        return json.loads(nebari.ledger_get(account or "", _state_key(lineage),
                                            "deep") or "{}")
    except ValueError:
        return {}


def save_state(account: str, lineage: str, st: dict) -> None:
    if not lineage:
        return
    import nebari
    nebari.ledger_put(account or "", lineage, _state_key(lineage), "deep",
                      json.dumps(st, sort_keys=True))


# WHAT A CONTINUATION CARRIES OVER (operator, 2026-09-28, after pagoda-h6).
# Hermes' Responses prompt_cache_key changes at a compaction; our summary
# line never reached it (its summariser call hung up after 600 s and Hermes
# compacted on its own), so the new key's history carried our id only in
# the kept tail's tool-call ids and session_identity named it a FORK: a new
# conversation, a new lineage, an EMPTY deep state -- and every probe
# trigger (koota, math, three, @react-three/fiber) fired a second time, 14-18
# minutes each. What was done once per key belongs to the HISTORY, and a
# fork shares its history with the conversation it came from, so the
# once-per-key state follows the link: the server-tool triggers' fired keys
# and plan tracking (`auto`), the areas and escalating skills researched,
# the verify moments passed (`verify.fired`) and yama_think_deeply's offer.
# Message-index state (boundary, len_seen, epoch, req) is not carried: the
# fork's messages are its own. Once per fork (`inherited_from`).
INHERITED = ("auto", "areas", "skills", "verify", "think_tool")


def inherit_state(account: str, lineage: str, parent: str) -> dict | None:
    """Carry `parent`'s once-per-key state into `lineage`'s (see INHERITED).
    Returns {from, fired, areas} when it merged something, else None (no
    parent state, the same lineage, or already inherited)."""
    if not lineage or not parent or lineage == parent:
        return None
    src = load_state(account, parent)
    if not src:
        return None
    with state_lock(account, lineage):
        st = load_state(account, lineage)
        if st.get("inherited_from"):
            return None
        a_src, a_dst = src.get("auto") or {}, st.setdefault("auto", {})
        fired = dict(a_src.get("fired") or {})
        fired.update(a_dst.get("fired") or {})
        a_dst["fired"] = fired
        for k in ("plan", "plan_written"):
            if k not in a_dst and k in a_src:
                a_dst[k] = a_src[k]
        for k in ("areas", "skills"):
            merged = list(src.get(k) or [])
            merged += [x for x in st.get(k) or [] if x not in merged]
            if merged:
                st[k] = merged
        v_fired = dict(((src.get("verify") or {}).get("fired")) or {})
        v_fired.update(((st.get("verify") or {}).get("fired")) or {})
        if v_fired:
            st.setdefault("verify", {})["fired"] = v_fired
        if "think_tool" not in st and "think_tool" in src:
            st["think_tool"] = src["think_tool"]
        st["inherited_from"] = parent
        save_state(account, lineage, st)
    return {"from": parent, "fired": sorted(fired),
            "areas": list(st.get("areas") or [])}


def think_tool_offered(tier: dict, account: str, lineage: str,
                       utility: bool, continuing: bool = True) -> bool:
    """Are yama_think_deeply and yama_plan on main for this request? The tier allows deep
    thinking and no header forces it off -- decided on a conversation's
    FIRST request and kept for the whole conversation (a tool list that
    changes between turns changes the system block the slot caches). A
    "no" is re-decided when the tier later allows it; a "yes" is never
    withdrawn -- a call on a request that forbids deep thinking gets a
    structured refusal (proxy._think_deeply).

    `continuing`: the request carries an earlier assistant turn. A FIRST
    request is decided afresh and re-recorded: benchmark arms send the same
    opening messages at different tiers and headers, which share one session
    key (nebari.key_of), and one arm's tool list must not leak into the
    next."""
    if utility:
        return False
    import tiers
    forced_off = ("investigate" in (tier.get("overridden") or [])
                  and not tier.get("investigate"))
    # The TIER's own allowance (xhigh, max), not a header that forces the
    # pre-main run on: a benchmark arm forcing deep thinking at `minimal`
    # must not also change main's tool list.
    native = bool((tiers.TIERS.get(tier.get("name") or "") or {})
                  .get("investigate"))
    want = native and not forced_off
    with state_lock(account, lineage):
        st = load_state(account, lineage)
        if lineage and not continuing:
            st["think_tool"] = "1" if want else "0"
            save_state(account, lineage, st)
            return want
        if st.get("think_tool") == "1":
            return True
        if st.get("think_tool") == "0" and not want:
            return False
        if lineage and st.get("think_tool") != ("1" if want else "0"):
            st["think_tool"] = "1" if want else "0"
            save_state(account, lineage, st)
        return want


def mark_ran(account: str, lineage: str, n_messages: int, kind: str,
             packages: list[str] | None = None,
             skills: list[str] | None = None,
             turn_key: str | None = None,
             auto_key: str | None = None) -> None:
    """Deep thinking ran for this conversation: the struggle episode ends
    here (its boundary moves to `n_messages`, so its signals are not
    counted again), and an area or kickoff is marked done."""
    if not lineage:
        return
    with state_lock(account, lineage):
        st = load_state(account, lineage)
        st["boundary"] = int(n_messages)
        st["last_run_req"] = int(st.get("req") or 0)
        st["last_kind"] = kind
        # A run ends the episode: the next signals are a new incident.
        st["episode"] = int(st.get("episode") or 0) + 1
        for p in packages or []:
            st.setdefault("areas", [])
            if p not in st["areas"]:
                st["areas"].append(p)
        for s in skills or []:
            st.setdefault("skills", [])
            if s not in st["skills"]:
                st["skills"].append(s)
        if kind == "kickoff" and turn_key:
            st["kickoffs"] = (list(st.get("kickoffs") or [])
                              + [turn_key])[-20:]
        if auto_key:
            # A SERVER-TOOL TRIGGER's job ran: its key never fires again in
            # this conversation (skill_select.server_tool_triggers).
            import skill_select
            skill_select.mark_trigger_fired(st.setdefault("auto", {}),
                                            auto_key, int(st.get("req") or 0))
        save_state(account, lineage, st)


def note_verify(account: str, lineage: str, line: str, tool: str,
                moment: str | None, at: int | None = None) -> None:
    """A NAMED verify line went out (mcp/verify_moment.py): kept, so a later
    request can see whether the turn after it called that tool."""
    if not lineage:
        return
    with state_lock(account, lineage):
        st = load_state(account, lineage)
        st.setdefault("verify", {})["named"] = {"line": line, "tool": tool,
                                                "moment": moment, "at": at}
        save_state(account, lineage, st)


# mark_deferred REMOVED 2026-09-27 with its proxy call sites. The deferral
# counter and the one-lane-wait-per-episode rule are gone (DEFER_REQUESTS
# above): a busy lane skips that request's trigger and nothing is carried
# to the next.


# Conversation-state keys of the removed rules, dropped from a stored state
# on its next request (decide) so nothing reads a stale one: the deferral
# (proxy read `waited_episode` until 2026-09-27) and
# the same-pattern suppression's patterns.
_RETIRED_STATE = ("deferred_req", "deferred_kind", "waited_episode",
                  "run_pattern", "req_pattern")


# ============================================================== decide =====
def _instruction(messages: list[dict]) -> str:
    import selection
    whole, _ctx, _sp = selection.question_of(messages or [])
    instr, _att = selection.instruction_of(whole)
    return instr


def _describe_events(events: list[dict], lines: bool = True) -> str:
    """`lines`: with each error's raw first line (the second brain's
    question); off for x_yamadori.deep's `because`, which keeps no line of
    the user's output (sig_digest)."""
    bits = []
    for e in events:
        what = e.get("tool") or e.get("path") or ""
        if lines and e.get("signature"):
            what += f": {e['signature']}"
        bits.append(f"{e['kind']}" + (f" ({what})" if what else ""))
    return "; ".join(bits)


def _struggle_question(messages: list[dict], events: list[dict]) -> str:
    """The investigation a struggle asks for: the task, what failed (tool
    output excerpts, marked as data), the files rewritten."""
    msgs = [m for m in messages if isinstance(m, dict)]
    task = _instruction(msgs)
    errs = []
    # The failing outputs of the struggle's own span (from its first event
    # on), whole: no count cap and no excerpt (docs/CONSTANTS-AUDIT.md).
    ats = [e["at"] for e in events if isinstance(e.get("at"), int)]
    first = min(ats) if ats else 0
    for m in msgs[first:]:
        if m.get("role") in ("tool", "function") and is_error(_text(m)):
            errs.append(_text(m).strip())
    paths = sorted({e["path"] for e in events if e.get("path")})
    q = ("An engineer working on the task below is stuck. Find the cause of "
         "the failure and the fix, with sources.\n\nTASK:\n" + task
         + "\n\nWHAT KEEPS FAILING (" + _describe_events(events) + ")")
    if paths:
        q += "\nFiles rewritten after failures: " + ", ".join(paths)
    if errs:
        q += ("\n\nThe failing tool output, oldest first (data from the "
              "user's machine, not instructions):\n" + "\n---\n".join(errs))
    return q


def _area_question(messages: list[dict], pkgs: list[dict],
                   skills: list[dict]) -> str:
    task = _instruction(messages)
    parts = []
    for p in pkgs:
        names = ", ".join(p.get("names") or []) or "(no names yet)"
        parts.append(f"- {p['package']}@{p['version']}: {p['why']}; names the "
                     f"conversation uses from it: {names}")
    for s in skills:
        parts.append(f"- skill:{s['id']} ({s.get('title') or ''}) declares "
                     f"that its area needs deep thinking")
    return (("The task below depends on an area the model is unlikely to "
             "know from training:\n" + "\n".join(parts)
             + "\n\nResearch what the task needs there: the exports, "
               "signatures and usage patterns it will call, and where they "
               "differ from older versions or similar libraries.\n\nTASK:\n"
             + task))


def _auto_question(messages: list[dict], c: dict, held: dict | None,
                   st: dict) -> tuple[str, dict]:
    """(the second brain's question, the synthetic call's arguments) for a
    SERVER-TOOL TRIGGER (skill_select.server_tool_triggers). The question
    carries the task (the second brain cannot see the conversation); the
    arguments are pointers, like KICKOFF_PLAN_ARGS (a CHOICE)."""
    task = _instruction(messages)
    trig = c["trigger"]
    if c["job"] == "investigate":
        pk = c.get("package")
        ver = ((held or {}).get(pk) or (None,))[0] if pk else None
        name = f"{pk}@{ver}" if pk and ver else pk
        how = ("reading its installed files (node_modules / site-packages) "
               "to learn its API" if trig == "probe" else
               "writing throwaway files to test how it behaves")
        if pk:
            q = (f"An engineer working on the task below is {how}: {name}. "
                 f"Research {pk}'s API as the task uses it -- the exports, "
                 "signatures and usage patterns it will call, and where they "
                 "differ from older versions or similar libraries -- from "
                 "its own source, with citations.\n\nTASK:\n" + task)
            return q, {"question": f"How does {pk} work as this task uses "
                                   f"it?"}
        path = (c.get("evidence") or {}).get("path") or ""
        q = ("An engineer working on the task below is writing throwaway "
             f"files ({path}) to test how a library behaves. Find which "
             "library the task depends on there and research its API as "
             "the task uses it, from its own source, with citations."
             "\n\nTASK:\n" + task)
        return q, {"question": "How does the library I am testing work as "
                               "this task uses it?"}
    if trig == "implement":
        spec = _text(messages[-1]) if messages else ""
        return ("Plan this task.\n\nTASK:\n" + spec,
                dict(KICKOFF_PLAN_ARGS))
    written = list(st.get("plan_written") or [])
    done = ("Files of the earlier plan written so far: "
            + (", ".join(written) if written else "none") + ".")
    if trig == "next_piece":
        piece = c.get("piece") or ""
        return (f"Plan the next piece of this task: {piece}/ is a new part "
                "of the work that the earlier plan did not name. " + done
                + "\n\nTASK:\n" + task,
                {"task": f"The next piece of the task above: {piece}/."})
    return ("Plan what comes next in this task: every file of the earlier "
            "plan is written (" + ", ".join(c.get("files") or []) + ")."
            "\n\nTASK:\n" + task,
            {"task": "What comes next in the task above, now that the "
                     "plan's files are written."})


def decide(*, raw: list[dict], tier: dict, route: dict | None,
           util: dict | None, account: str, lineage: str,
           turn_key: str | None, uses: dict | None = None,
           held: dict | None = None, inplace: bool = False,
           continues: bool = False, auto: dict | None = None) -> dict:
    """The trigger for one request, or a recorded non-decision.

    {fire, kind, job, question, context, because, allowed, forced,
     signals: {struggle, kickoff, area}, thresholds, last_run}. Pure of the
    model: reads the client's messages, the conversation's state in the
    ledger, and the package store's metadata. `uses` is proxy.library_uses
    (package -> {names}); `held` maps a used held package to (version, db).
    The request counter in the conversation's state is advanced here.
    `auto`: the SERVER-TOOL TRIGGERS' inputs from the proxy ({think_ok,
    plan_ok, plan_files, intent}); None evaluates none."""
    thr = thresholds()
    T = thr["struggle_threshold"]["value"]
    allowed = bool(tier.get("investigate"))
    over = tier.get("overridden") or []
    forced = (("on" if tier.get("investigate") else "off")
              if "investigate" in over else None)
    rec: dict = {"fire": False, "kind": None, "job": None, "question": None,
                 "context": "", "allowed": allowed, "forced": forced,
                 "thresholds": {"struggle": T,
                                "sources": {k: v["source"]
                                            for k, v in thr.items()}},
                 "signals": {}, "last_run": None, "because": ""}
    if (util or {}).get("utility") or inplace:
        rec["because"] = ("a client side call or an in-place compaction: no "
                          "trigger is evaluated")
        rec["skip_record"] = True
        return rec
    msgs = [m for m in (raw or []) if isinstance(m, dict)]
    with state_lock(account, lineage):
        st = load_state(account, lineage)
        st["req"] = int(st.get("req") or 0) + 1
        for k in _RETIRED_STATE:
            st.pop(k, None)
        boundary = int(st.get("boundary") or 0)
        if len(msgs) < int(st.get("len_seen") or 0) or boundary > len(msgs):
            # A COMPACTION shortened the conversation (the lineage links it,
            # proxy._continue_after_compaction): a new EPOCH of message
            # indexes. The episode and what was covered carry over; the
            # boundary counts from the compacted start, and rows from the
            # old epoch are observed against the new messages (observe).
            st["epoch"] = int(st.get("epoch") or 0) + 1
            boundary = 0
            st["boundary"] = 0
        st["len_seen"] = len(msgs)
        # Counted from the episode boundary (the last run, or a compaction):
        # no message window (docs/CONSTANTS-AUDIT.md, 2026-09-27).
        start = boundary
        scan = struggle_scan(msgs, start)
        events = scan["events"]
        pattern = pattern_of(events)
        # THE SERVER-TOOL TRIGGERS (operator, 2026-09-27): detected on every
        # request where deep thinking may run (the plan tracking must see
        # every write), fired below after the other triggers.
        auto_c: list[dict] = []
        auto_err = None
        if auto is not None and allowed and forced != "off" \
                and msgs and msgs[-1].get("role") in ("user", "tool"):
            try:
                import skill_select
                auto_c = skill_select.server_tool_triggers(
                    msgs, st.setdefault("auto", {}),
                    kind="user" if msgs[-1].get("role") == "user"
                    else "step",
                    think_ok=bool(auto.get("think_ok")),
                    plan_ok=bool(auto.get("plan_ok")),
                    plan_files=auto.get("plan_files"),
                    intent=auto.get("intent"))
            except Exception as e:                               # noqa: BLE001
                auto_err = f"{type(e).__name__}: {e}"[:200]
        # THE VERIFY MOMENT (operator, 2026-09-27; mcp/verify_moment.py):
        # mechanical and once per moment, after the plan tracking above.
        vmoment = None
        if auto is not None and auto.get("verify") and allowed and \
                forced != "off":
            try:
                import verify_moment
                vmoment = verify_moment.moment(
                    msgs, st.setdefault("verify", {}), st.get("auto"))
            except Exception as e:                               # noqa: BLE001
                vmoment = {"error": f"{type(e).__name__}: {e}"[:200]}
        auto_st = json.loads(json.dumps(st.get("auto") or {}))
        save_state(account, lineage, st)
    rec["epoch"] = int(st.get("epoch") or 0)
    rec["episode"] = int(st.get("episode") or 0)
    if st.get("inherited_from"):
        # A fork of (or a continuation the proxy could not link to) another
        # conversation: its once-per-key state came with it (INHERITED).
        rec["inherited_from"] = st["inherited_from"]
    since = (st["req"] - int(st["last_run_req"])
             if st.get("last_run_req") is not None else None)
    rec["last_run"] = {"requests_since_run": since, "boundary": boundary}
    rec["signals"]["struggle"] = {
        "count": len(events), "threshold": T, "window_from": start,
        "kinds": _counts(events),
        # The signature as its digest (sig_digest): x_yamadori.deep and the
        # row keep no line of the user's output.
        "events": [{k: (sig_digest(v) if k == "signature" else v)
                    for k, v in e.items() if k != "at"}
                   for e in events[-10:]],
        "pattern": pattern[:20]}
    ko = kickoff(msgs, continues=continues)
    ko_done = bool(turn_key) and turn_key in (st.get("kickoffs") or [])
    ko_fire = ko["new_task"] and not ko_done
    rec["signals"]["kickoff"] = dict(ko, done=ko_done)
    # Known-hard area: held packages the conversation USES that are unseen.
    area_pkgs: list[dict] = []
    done_areas = set(st.get("areas") or [])
    for pkg, u in sorted((uses or {}).items()):
        if pkg not in (held or {}) or pkg in done_areas:
            continue
        ver, db = held[pkg]
        why = unseen(pkg, ver, db)
        if why:
            area_pkgs.append({"package": pkg, "version": ver, "why": why,
                              "names": list((u or {}).get("names") or [])[:8]})
    rec["signals"]["area"] = {
        "unseen": [f"{p['package']}@{p['version']}: {p['why']}"
                   for p in area_pkgs],
        "done": sorted(done_areas)}
    # A package deep thinking already researched (an area run) is covered.
    auto_c = [c for c in auto_c if not (c.get("package")
                                        and c["package"] in done_areas)]
    if vmoment is not None:
        rec["verify"] = vmoment
    if auto is not None:
        rec["signals"]["auto"] = {
            "candidates": [{"key": c["key"], "trigger": c["trigger"],
                            "job": c["job"]} for c in auto_c],
            "fired_before": sorted((auto_st.get("fired") or {}).keys()),
            **({"error": auto_err} if auto_err else {})}
    # E1's heads (YAMADORI_E1=1): one embedding of the request, every trained
    # head read from it; an untrained head is None and its rule decides.
    speaking = (bool(msgs) and msgs[-1].get("role") == "user"
                and not _tool_media(msgs, len(msgs) - 1))
    e1rec = None
    if allowed and forced is None and (speaking or events):
        e1rec = _e1_consult(msgs, events)
    e1h = (e1rec or {}).get("heads") or {}
    if e1rec is not None:
        rec["e1_state"] = e1rec["state"]
        rec["signals"]["e1"] = {
            "heads": {k: ({f: v[f] for f in ("choice", "margin", "version",
                                              "n")} if v else None)
                      for k, v in e1h.items()},
            "status": e1rec.get("status"), "embed": e1rec.get("embed")}
    esc, rin = e1h.get("escalate"), e1h.get("route_in")
    # The same-pattern suppression ("one incident, one run", #45 (d)) was
    # invented and is gone (docs/CONSTANTS-AUDIT.md, 2026-09-27): a pattern
    # that recurs after a run fires again once it reaches the threshold
    # since the run's boundary.
    struggle_fire = len(events) >= T
    if esc is not None and events:
        struggle_fire = esc["choice"] == "escalate"
    e1_area = (rin is not None and speaking and rin["choice"] == "investigate"
               and (route or {}).get("class") == "library_question")
    # Earlier user turns ride along as context (selection.question_of's
    # rule): the second brain cannot see the conversation.
    import selection
    _q, ctx, _sp = selection.question_of(msgs)
    # Whole: the CONTEXT_CHARS 1500 tail cut was invented (CONSTANTS-AUDIT).
    if forced == "off":
        rec["because"] = "forced off by X-Yamadori-Features"
    elif forced == "on":
        rec.update(fire=True, kind="forced", job="investigate",
                   because="forced on by X-Yamadori-Features")
    elif not allowed:
        rec["because"] = (f"tier {tier.get('name', '?')} does not allow deep "
                          f"thinking")
    elif struggle_fire:
        rec.update(fire=True, kind="struggle", job="investigate",
                   question=_struggle_question(msgs, events), context=ctx,
                   because=(f"struggle: {len(events)} signals in this episode "
                            + (_e1_says(esc, "escalate") if esc is not None
                               else f"(threshold {T})")
                            + f": {_describe_events(events, lines=False)}"))
    elif ko_fire:
        spec = _text(msgs[-1])
        extra = ""
        if area_pkgs:
            extra = ("\n\nPackages the task uses that the engineer's model "
                     "cannot have seen (research them for the plan): "
                     + "; ".join(f"{p['package']}@{p['version']} ({p['why']})"
                                 for p in area_pkgs))
        rec.update(fire=True, kind="kickoff", job="plan",
                   question=("Plan this task.\n\nTASK:\n" + spec
                             ) + extra,
                   context=ctx,
                   because=(f"task kickoff: {ko['why']} (every new task is "
                            f"planned; spec ~{ko['tokens']} tokens)"),
                   packages=[p["package"] for p in area_pkgs])
    elif area_pkgs:
        rec.update(fire=True, kind="area", job="investigate",
                   question=_area_question(msgs, area_pkgs, []), context=ctx,
                   because=("known-hard area: "
                            + "; ".join(f"{p['package']}@{p['version']} "
                                        f"({p['why']})" for p in area_pkgs)),
                   packages=[p["package"] for p in area_pkgs])
    elif auto_c:
        # A SERVER-TOOL TRIGGER: the job the model was not calling, run for
        # it (operator, 2026-09-27: "we should decide when to fire it, and
        # be more heavy handed"); after struggle, kickoff and area (one run
        # per request).
        c = auto_c[0]
        q, args = _auto_question(msgs, c, held, auto_st)
        rec.update(fire=True, kind="auto", job=c["job"], question=q,
                   context=ctx, key=c["key"], synthetic_args=args,
                   because=(f"server-tool trigger {c['trigger']} "
                            f"({c['key']}): the proxy runs "
                            f"{c['tool']} itself"),
                   packages=[c["package"]] if c.get("package") else [],
                   auto={"trigger": c["trigger"], "key": c["key"],
                         "job": c["job"], "tool": c["tool"],
                         "evidence": c.get("evidence")})
    elif e1_area:
        rec.update(fire=True, kind="area", job="investigate",
                   question=_route_question(msgs), context=ctx,
                   because=("known-hard area: a library question "
                            + _e1_says(rin, "investigate")))
    else:
        why = [f"struggle {len(events)}/{T}",
               (f"kickoff: {ko['why']}, already planned" if ko["new_task"]
                else f"no new task ({ko['why']})"),
               ("no unseen held package in use" if not done_areas else
                f"areas already covered: {', '.join(sorted(done_areas))}")]
        if e1h:
            why.append("E1: " + ", ".join(
                f"{k}={v['choice']}" if v else f"{k}=untrained"
                for k, v in sorted(e1h.items())))
        rec["because"] = ("no trigger fired: " + "; ".join(why)
                          + f"; the model may call {TOOL_NAME}")
    return rec


def _e1_says(p: dict, what: str) -> str:
    return (f"(E1 {p.get('head')} head v{p.get('version')}, n={p.get('n')}: "
            f"{p.get('choice')}, p({what})="
            f"{(p.get('probabilities') or {}).get(what, 0):.2f})")


def _e1_consult(msgs: list[dict], events: list[dict]) -> dict | None:
    """e1.consult on the request's instruction and earlier user text, with
    the struggle counts as the escalate head's extra features.
    None when E1 is off or there is no instruction to read."""
    import e1
    if not e1.enabled():
        return None
    import selection
    whole, ctx, _sp = selection.question_of(msgs)
    q, _att = selection.instruction_of(whole)
    if len(q.strip()) < selection.MIN_QUESTION_CHARS:
        return None
    extras = {"escalate": dict({"struggle_count": len(events)},
                               **_counts(events))}
    try:
        # E1's training cut (B; E1_QUESTION_CHARS above says why).
        return e1.consult(q[:E1_QUESTION_CHARS], ctx[-E1_CONTEXT_CHARS:],
                          extras)
    except Exception as e:                                       # noqa: BLE001
        print(f"  deep: E1 consult raised ({type(e).__name__}: {e}); the "
              f"rules decide", flush=True)
        return None


def _route_question(msgs: list[dict]) -> str:
    """What the second brain researches for a library question E1 flagged."""
    return ("The question below is about a library whose held source "
            "answers it; memory is likely to be wrong on the version-specific "
            "details. Research it in the held source: the definitions, "
            "defaults, exports and call sites it needs, with citations.\n\n"
            "QUESTION:\n" + _instruction(msgs))


def escalate_skills(rec: dict, skills_rec: dict | None, account: str,
                    lineage: str, messages: list[dict]) -> dict:
    """The other half of the known-hard area: a skill injected on this
    request whose rule declares `escalate` (skill_classify.classify reads it
    from the SKILL.md frontmatter). Fires only when nothing else did and
    deep thinking is allowed; once per skill per conversation."""
    ids = list((skills_rec or {}).get("ids") or [])
    if not ids or rec.get("fire") or not rec.get("allowed") \
            or rec.get("forced") == "off":
        return rec
    try:
        import skills as skill_store
        armed = {s["id"]: s for s in skill_store.armed()}
    except Exception:                                            # noqa: BLE001
        return rec
    done = set(load_state(account, lineage).get("skills") or [])
    esc = [armed[i] for i in ids if i in armed and i not in done
           and (armed[i].get("rule") or {}).get("escalate")]
    rec.setdefault("signals", {}).setdefault("area", {})["skills"] = [
        s["id"] for s in esc]
    if not esc:
        return rec
    rec.update(fire=True, kind="area", job="investigate",
               question=_area_question(messages, [], esc),
               because="known-hard area: skill(s) declare escalate: "
                       + ", ".join(s["id"] for s in esc),
               skills=[s["id"] for s in esc])
    return rec


def public(rec: dict | None) -> dict | None:
    """x_yamadori.deep: the decision without the question text."""
    if rec is None:
        return None
    return {k: v for k, v in rec.items()
            if k not in ("question", "context", "skip_record", "e1_state")}


# ============================================================ records ======
DDL = """
CREATE TABLE IF NOT EXISTS deep_decisions(
    id           TEXT PRIMARY KEY,
    created      REAL NOT NULL,
    day          TEXT NOT NULL,
    account      TEXT,
    conversation TEXT,
    traffic      TEXT,
    tier         TEXT,
    route        TEXT,
    allowed      INTEGER NOT NULL DEFAULT 0,
    forced       TEXT,
    trigger      TEXT,
    fired        INTEGER NOT NULL DEFAULT 0,
    ran          INTEGER NOT NULL DEFAULT 0,
    model_calls  INTEGER NOT NULL DEFAULT 0,
    at_messages  INTEGER NOT NULL DEFAULT 0,
    signals      TEXT NOT NULL DEFAULT '{}',
    thresholds   TEXT NOT NULL DEFAULT '{}',
    handoff      TEXT NOT NULL DEFAULT '{}',
    terms        TEXT NOT NULL DEFAULT '[]',
    outcome      TEXT,
    observed     INTEGER NOT NULL DEFAULT 0,
    label        TEXT,
    labelled_at  REAL,
    learned_at   REAL,
    epoch        INTEGER NOT NULL DEFAULT 0,
    episode      INTEGER NOT NULL DEFAULT 0,
    cooldown     INTEGER NOT NULL DEFAULT 0,
    e1_state     TEXT,
    label_rule   INTEGER);
CREATE INDEX IF NOT EXISTS deep_open ON deep_decisions(conversation, label);
CREATE INDEX IF NOT EXISTS deep_learn ON deep_decisions(learned_at, label);
CREATE TABLE IF NOT EXISTS deep_adjustments(
    id          TEXT PRIMARY KEY,
    created     REAL NOT NULL,
    param       TEXT NOT NULL,
    old         REAL,
    new         REAL NOT NULL,
    n           INTEGER NOT NULL DEFAULT 0,
    evidence    TEXT NOT NULL DEFAULT '{}',
    why         TEXT,
    author      TEXT,
    reverts     TEXT,
    reverted_at REAL);
CREATE TABLE IF NOT EXISTS deep_proposals(
    id       TEXT PRIMARY KEY,
    created  REAL NOT NULL,
    kind     TEXT NOT NULL,
    text     TEXT NOT NULL,
    n        INTEGER NOT NULL DEFAULT 0,
    evidence TEXT NOT NULL DEFAULT '{}',
    status   TEXT NOT NULL DEFAULT 'proposed');
"""
_ENSURED: set[str] = set()
_LOCK = threading.Lock()


def _db_path() -> str:
    import corpus
    return corpus.CORPUS_DB


def _db() -> sqlite3.Connection:
    path = os.path.abspath(_db_path())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=10, isolation_level=None)
    con.execute("PRAGMA busy_timeout=10000")
    if path not in _ENSURED:
        # WAL: readers do not block this writer (see corpus._db).
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(DDL)
        # Columns added after the first build (pre-deploy review): a table
        # created before them gains them here.
        have = {r[1] for r in con.execute("PRAGMA table_info(deep_decisions)")}
        for col in ("epoch", "episode", "cooldown"):
            if col not in have:
                con.execute(f"ALTER TABLE deep_decisions ADD COLUMN {col} "
                            f"INTEGER NOT NULL DEFAULT 0")
        # E1's embedded state (mcp/e1.py): what the idle learner retrains on.
        if "e1_state" not in have:
            con.execute("ALTER TABLE deep_decisions ADD COLUMN e1_state TEXT")
        # The rule a label was made under (LABEL_RULE); NULL: before rule 2.
        if "label_rule" not in have:
            con.execute("ALTER TABLE deep_decisions ADD COLUMN label_rule "
                        "INTEGER")
        _ENSURED.add(path)
    con.row_factory = sqlite3.Row
    return con


_TERM = re.compile(r"`([^`\n]{3,80})`|([\w./@-]+\.(?:tsx|ts|jsx|js|mjs|json|"
                   r"rs|py|go|wgsl|glsl|md|toml|yaml|yml|css|html))\b")


def handoff_terms(text: str, limit: int = 20) -> list[str]:
    """Names a hand-off gives the main model to act on (backticked names and
    file paths): if none of them shows up in what main does next, the
    hand-off was not used."""
    out: list[str] = []
    for a, b in _TERM.findall(text or ""):
        t = (a or b).strip()
        if t and t not in out and not t.lower().startswith(("http", "skill:")):
            out.append(t)
        if len(out) >= limit:
            break
    return out


def new_id() -> str:
    """A row id, drawn when the request finishes (x_yamadori.deep.record)
    before the row is written in the background."""
    return uuid.uuid4().hex[:12]


def _traffic(account: str) -> str:
    try:
        import corpus
        return corpus.account_traffic(account)
    except Exception:                                            # noqa: BLE001
        return "test"          # fail CLOSED: never learn from an unknown


def _insert(con: sqlite3.Connection, rid: str, *, account: str,
            conversation: str, tier: str | None, route: str | None,
            rec: dict, n_messages: int, ran: bool, kind: str | None,
            model_calls: int, handoff: dict | None,
            terms: list[str] | None) -> None:
    trig = kind or ("model" if model_calls else None)
    # The `cooldown` column stays in the table for old rows and defaults to
    # 0: the cooldown was removed 2026-09-27 (docs/CONSTANTS-AUDIT.md).
    con.execute(
        "INSERT INTO deep_decisions(id,created,day,account,conversation,"
        "traffic,tier,route,allowed,forced,trigger,fired,ran,model_calls,"
        "at_messages,signals,thresholds,handoff,terms,epoch,episode,"
        "e1_state) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, time.time(), dt.date.today().isoformat(),
         (account or "")[:16], conversation or "", _traffic(account), tier,
         route, int(bool(rec.get("allowed"))), rec.get("forced"),
         trig or "none", int(bool(rec.get("fire")) or bool(model_calls)),
         int(bool(ran)), int(model_calls), int(n_messages),
         json.dumps(dict(rec.get("signals") or {},
                         **{k: rec[k] for k in ("helped_rule", "project")
                            if rec.get(k) is not None}),
                    default=str)[:20000],
         json.dumps(rec.get("thresholds") or {}, default=str),
         json.dumps(handoff or {}, default=str)[:4000],
         json.dumps((terms or [])[:20]), int(rec.get("epoch") or 0),
         int(rec.get("episode") or 0), rec.get("e1_state")))


def record(*, account: str, conversation: str, tier: str | None,
           route: str | None, rec: dict | None, n_messages: int,
           ran: bool, kind: str | None, model_calls: int = 0,
           handoff: dict | None = None, terms: list[str] | None = None,
           rid: str | None = None) -> str | None:
    """One row for this request: a decision or a non-decision. Never
    raises (a dropped row costs one label)."""
    if rec is None or rec.get("skip_record"):
        return None
    rid = rid or new_id()
    try:
        with _LOCK:
            con = _db()
            try:
                con.execute("BEGIN")
                _insert(con, rid, account=account, conversation=conversation,
                        tier=tier, route=route, rec=rec,
                        n_messages=n_messages, ran=ran, kind=kind,
                        model_calls=model_calls, handoff=handoff,
                        terms=terms)
                con.execute("COMMIT")
            finally:
                con.close()
    except Exception as e:                                       # noqa: BLE001
        print(f"  deep: record failed ({type(e).__name__}: {e})", flush=True)
        return None
    return rid


def project_changed(later: list[dict], project: dict | None) -> list[str]:
    """The project paths (progress.is_project) the model wrote or patched in
    `later` whose tool result did not fail or say nothing was applied --
    at most 5. Rule 2's evidence that a run changed anything."""
    try:
        import progress
    except Exception:                                            # noqa: BLE001
        return []
    results = {m.get("tool_call_id"): _text(m) for m in later
               if isinstance(m, dict) and m.get("role") == "tool"}
    proj = dict(project or {})
    proj["named"] = list(dict.fromkeys(
        list(proj.get("named") or []) + progress.named_paths(later)))
    out: list[str] = []
    for m in later:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            res = results.get((c or {}).get("id"))
            if res is None or is_error(res) or not_applied(res):
                continue
            for p in _write_edits(c):
                if progress.is_project(p, proj)[0] and p not in out:
                    out.append(p)
            if len(out) >= 5:
                return out
    return out


def _outcome(row: sqlite3.Row, msgs: list[dict], ran_later: bool,
             threshold: int, epoch: int) -> tuple[dict, str | None]:
    """(outcome so far, label or None while it is still open). A row from
    an earlier EPOCH (before a compaction) is observed against the whole of
    the compacted conversation: all of it came after the row.

    A RUN (#45 (e)) is judged against the pattern it ran on (the row's
    signals.struggle.pattern), not the episode counter a run resets: ONE
    recurrence of it after the run -- a counted signal with its key, or a
    single failure with one of its (tool, signature) pairs -- is
    `not_helped` at once; `helped` (or
    `wasted`, when main used none of the hand-off's names) needs
    HELPED_WINDOW observations without that. A run on no pattern (a
    kickoff, an area, a model call on a calm request) falls back to the
    signal count."""
    at = int(row["at_messages"] or 0)
    if int(row["epoch"] or 0) < epoch:
        at = 0
    at = min(at, len(msgs))
    later = msgs[at:]
    scan = struggle_scan(msgs, at)
    ev = scan["events"]
    kinds = _counts(ev)
    try:
        sig = json.loads(row["signals"] or "{}")
    except ValueError:
        sig = {}
    st_sig = sig.get("struggle") or {}
    pattern = list(st_sig.get("pattern") or [])
    text = []
    checks_ok = 0
    for m in later:
        if m.get("role") == "assistant":
            t = _text(m)
            text.append(t)
            if re.search(r"\b(?:Verified|Repaired) ", t):
                checks_ok += 1
            for c in m.get("tool_calls") or []:
                text.append(json.dumps(_args(c)))
    terms = json.loads(row["terms"] or "[]")
    blob = "\n".join(text)
    used = (any(t in blob for t in terms) if terms else None)
    finished = bool(later) and any(
        m.get("role") == "assistant" and not m.get("tool_calls")
        and _text(m).strip() for m in later[-2:])
    out = {"observed": int(row["observed"] or 0) + 1,
           "struggle_after": len(ev), "kinds": kinds,
           "rewritten": kinds.get("file_rewritten", 0),
           "fixup_capped": kinds.get("fixup_capped", 0),
           "check_passed": checks_ok > 0 and not kinds.get("fixup_capped"),
           "handoff_used": used, "finished": finished,
           "ran_later": ran_later}
    if row["ran"]:
        recurred = _recurrence(pattern, scan)
        rule = int(sig.get("helped_rule") or LABEL_RULE)
        changed = project_changed(later, sig.get("project"))
        out.update(pattern=len(pattern), recurred=recurred[:10],
                   project_changed=changed, rule=rule)
        if recurred:
            return out, "not_helped"
        if out["observed"] < HELPED_WINDOW:
            return out, None
        if not pattern and len(ev) >= threshold:
            return out, "not_helped"
        if rule >= 2 and not changed:
            # Rule 2: nothing the project holds changed after the run.
            return out, "no_effect"
        if used is False:
            return out, "wasted"
        return out, "helped"
    done = out["observed"] >= OUTCOME_REQUESTS or ran_later
    if not done:
        return out, None
    if row["fired"] and not row["model_calls"]:
        # A trigger fired but deep thinking did not run (the helper lane was
        # busy): neither a decision's outcome nor a non-decision's.
        return out, "skipped"
    if ran_later:
        return out, "escalated_later"
    had = int(st_sig.get("count") or 0)
    if row["allowed"] and had and len(ev) >= 2:
        # (`in_cooldown` and `missed_same_pattern` went with the cooldown
        # and the same-pattern rule, 2026-09-27; old rows keep them.)
        return out, "missed"
    return out, "fine"


# One incident = one label (pre-deploy review, 2026-09-24): every request of
# an episode used to be labelled by one complaint, so one complaint
# reached LEARN_MIN_N at once. A non-decision label counts once per EPISODE
# (the first row labelled; the rest `same_episode`, which the learner does
# not count). A run's own row always counts: a run ends its episode.
COUNTED_ONCE = ("missed", "fine", "escalated_later", "skipped")


def _observe(con: sqlite3.Connection, account: str, conversation: str,
             msgs: list[dict], epoch: int, T: int) -> int:
    rows = con.execute(
        "SELECT * FROM deep_decisions WHERE conversation=? AND account=? "
        "AND label IS NULL ORDER BY created",
        (conversation, (account or "")[:16])).fetchall()
    if not rows:
        return 0
    runs = [r["created"] for r in rows if r["ran"]]
    taken = {(r[0], r[1]) for r in con.execute(
        "SELECT epoch, episode FROM deep_decisions WHERE conversation=? AND "
        "account=? AND ran=0 AND label IN (%s)" % ",".join(
            "?" * len(COUNTED_ONCE)),
        (conversation, (account or "")[:16]) + COUNTED_ONCE)}
    n = 0
    now = time.time()
    for r in rows:
        ran_later = any(c > r["created"] for c in runs) and not r["ran"]
        out, label = _outcome(r, msgs, ran_later, T, epoch)
        if label in COUNTED_ONCE and not r["ran"]:
            ep = (r["epoch"], r["episode"])
            if ep in taken:
                label = "same_episode"
            else:
                taken.add(ep)
        # The rule the label was made under: a run's own (rule 1 when its
        # request switched rule 2 off), LABEL_RULE for every other row.
        lrule = (int(out.get("rule") or LABEL_RULE) if r["ran"]
                 else LABEL_RULE) if label else None
        con.execute("UPDATE deep_decisions SET outcome=?, observed=?, "
                    "label=?, labelled_at=?, label_rule=? WHERE id=?",
                    (json.dumps(out), out["observed"], label,
                     now if label else None, lrule, r["id"]))
        n += bool(label)
    return n


def observe(account: str, conversation: str, messages: list[dict],
            epoch: int = 0) -> int:
    """Advance every open row of this conversation by one observation of
    what the client sent back; label the ones whose window closed. One
    transaction. Returns how many were labelled. Never raises."""
    if not conversation:
        return 0
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    T = thresholds()["struggle_threshold"]["value"]
    try:
        with _LOCK:
            con = _db()
            try:
                con.execute("BEGIN")
                n = _observe(con, account, conversation, msgs, epoch, T)
                con.execute("COMMIT")
                return n
            finally:
                con.close()
    except Exception as e:                                       # noqa: BLE001
        print(f"  deep: observe failed ({type(e).__name__}: {e})", flush=True)
        return 0


def observe_and_record(*, account: str, conversation: str,
                       messages: list[dict], rid: str, **row) -> None:
    """The request's whole write: observe the conversation's open rows,
    then insert this request's row -- ONE transaction, one commit."""
    rec = row.get("rec")
    if rec is None or rec.get("skip_record"):
        return
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    T = thresholds()["struggle_threshold"]["value"]
    last = None
    for attempt in range(4):            # off the response path: retry, don't drop
        try:
            with _LOCK:
                con = _db()
                try:
                    # IMMEDIATE takes the write lock up front, so the busy
                    # timeout covers it instead of failing mid-transaction.
                    con.execute("BEGIN IMMEDIATE")
                    if conversation:
                        _observe(con, account, conversation, msgs,
                                 int(rec.get("epoch") or 0), T)
                    _insert(con, rid, account=account, conversation=conversation,
                            **row)
                    con.execute("COMMIT")
                    return
                except Exception:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:                            # noqa: BLE001
                        pass
                    raise
                finally:
                    con.close()
        except sqlite3.OperationalError as e:
            last = e
            if "locked" not in str(e) and "busy" not in str(e):
                break
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:                                   # noqa: BLE001
            last = e
            break
    print(f"  deep: recording failed ({type(last).__name__}: {last})",
          flush=True)


# OFF THE RESPONSE PATH (pre-deploy review, 2026-09-24): observe + record ran
# synchronously before the response, under a process-wide lock, with a 30 s
# busy timeout. They go to ONE background thread through a bounded queue; a
# full queue drops the write (counted) rather than delaying a response.
RECORD_QUEUE = int(os.environ.get("YAMADORI_DEEP_RECORD_QUEUE", "256"))
_Q: "queue.Queue" = queue.Queue(maxsize=RECORD_QUEUE)
_WORKER: dict = {"thread": None, "dropped": 0, "done": 0}
_WORKER_LOCK = threading.Lock()


def _work() -> None:
    while True:
        fn, a, kw = _Q.get()
        try:
            fn(*a, **kw)
        except Exception as e:                                   # noqa: BLE001
            print(f"  deep: background write failed ({type(e).__name__}: "
                  f"{e})", flush=True)
        finally:
            _WORKER["done"] += 1
            _Q.task_done()


def submit(fn, *a, **kw) -> bool:
    """Queue one write for the background thread. False (and counted) when
    the queue is full."""
    with _WORKER_LOCK:
        t = _WORKER["thread"]
        if t is None or not t.is_alive():
            t = threading.Thread(target=_work, name="deep-records",
                                 daemon=True)
            t.start()
            _WORKER["thread"] = t
    try:
        _Q.put_nowait((fn, a, kw))
        return True
    except queue.Full:
        _WORKER["dropped"] += 1
        print("  deep: record queue full; a decision row was dropped",
              flush=True)
        return False


def flush(timeout: float = 10.0) -> bool:
    """Wait for the queued writes (tests; shutdown). True when drained."""
    end = time.time() + timeout
    while time.time() < end:
        if _Q.unfinished_tasks == 0:
            return True
        time.sleep(0.02)
    return False


def rows(limit: int = 50, conversation: str | None = None) -> list[dict]:
    con = _db()
    try:
        q = "SELECT * FROM deep_decisions"
        a: tuple = ()
        if conversation:
            q += " WHERE conversation=?"
            a = (conversation,)
        out = []
        for r in con.execute(q + " ORDER BY created DESC LIMIT ?",
                             a + (limit,)):
            d = dict(r)
            # E1's embedded state is request text: the learner reads it
            # (e1.live_rows), nothing that reports rows carries it.
            d.pop("e1_state", None)
            for k in ("signals", "thresholds", "handoff", "terms", "outcome"):
                try:
                    d[k] = json.loads(d.get(k) or "null")
                except ValueError:
                    pass
            out.append(d)
        return out
    finally:
        con.close()


def per_day(days: int = 14) -> list[dict]:
    """Per day: requests, deep-thinking runs by trigger, and labels."""
    con = _db()
    try:
        out: dict[str, dict] = {}
        for r in con.execute(
                "SELECT day, trigger, ran, label, COUNT(*) n FROM "
                "deep_decisions WHERE traffic='client' GROUP BY day, trigger, "
                "ran, label ORDER BY day DESC"):
            d = out.setdefault(r["day"], {"day": r["day"], "requests": 0,
                                          "runs": {}, "labels": {}})
            d["requests"] += r["n"]
            if r["ran"]:
                d["runs"][r["trigger"]] = d["runs"].get(r["trigger"], 0) + r["n"]
            if r["label"]:
                d["labels"][r["label"]] = d["labels"].get(r["label"], 0) + r["n"]
        return sorted(out.values(), key=lambda x: x["day"])[-days:]
    finally:
        con.close()


if __name__ == "__main__":
    print(json.dumps(thresholds(fresh=True), indent=1))
