#!/usr/bin/env python
"""THE INJECTOR'S LABEL SET: real turns, their stage-1 candidate items, and a
truth level per (turn, item) against a written rubric (RUBRIC below).

    python bench/skills/inject_labels.py build --out DIR [--per 40] [--seed 7]
    python bench/skills/inject_labels.py sheets --out DIR --batches 4 \\
        [--second 0.2]                      # labeller sheets; a blind sample
    python bench/skills/inject_labels.py ingest --out DIR --answers FILE \\
        --labeller A|B                      # appends to LABELS
    python bench/skills/inject_labels.py agree  # pass A vs the blind pass B
    python bench/skills/inject_labels.py stats

The operator (2026-09-29): thresholds "TUNED ON LABELS PER MODEL with bench/
decider/tune.py. Build the label set from real turns: the pagoda relays,
Hermes/Pi logs and corpus turns, labelled against a written rubric of when a
skill item would have increased correctness. Use a blind second pass on a
sample for agreement, and report n."

WHERE THE TURNS COME FROM (read only)

  hermes    C:/Users/jwals/octo/logs/<run>/hermes.jsonl + prompt.md -- the
            run's whole Hermes stream (bench/skills/replay_selection.
            _stream_messages): the pagoda runs h1, h3-h6 and the Octopus
            V0 runs (v0b-v0f, pilot).
  pi        <run>/pi.jsonl -- Pi's message_end events (pi_messages): the
            pagoda runs p1-p4 and the PackageLens lookup probes.
  corpus    index/corpus.sqlite3 (read only), TEST traffic only (corpus.
            account_traffic == "test": our own suites and benchmarks, never
            a client's private work): the request head the corpus keeps, as
            one user turn.

STAGE 1, REPLAYED: every request point of a transcript (a user turn, or the
last tool result of a step) goes through skill_select.decide in order with
the conversation's skill state carried, OFFLINE (no embedder, no fallback:
replay_selection's patches; no decider Turn, so the per-skill path runs and
its candidates are captured at the point the injector takes them:
skill_select._decide_injected's `cands`, bodies and recalls; the injector
answered by the stub decider, pass_all). NOTE: the 2026-09-29 case set was
built before the per-skill path was removed, when a skill offered was a
skill given; with pass_all the composer's caps (3 skills, 6 items) can
leave a later skill not given, so a REBUILD can differ in a few later
points -- the labels are keyed to the built cases, not to a rebuild. A point with at
least one candidate item is a CASE: its state exactly as the decider reads
it (decide_turn.state_of, chars/3 as the count) and its candidate items
(skill_inject.candidates: at most MAX_SKILLS skills, no doubt-bearing item,
no repeat; the given-before filter is NOT applied, so every item the live
injector could ask about is labelled).

SAMPLING: `--per` cases per (source group, kind) stratum, seeded; the
strata are hermes/pi/corpus x user/step.

NOTHING HERE STORES CONVERSATION TEXT IN THE REPO: cases (with their state
text) go to --out (a scratch directory); LABELS holds ids, hashes and the
truth only (the corpus's rule: "Not kept: the caller's code").
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))

LOGS = os.environ.get("YAMADORI_OCTO_LOGS", "C:/Users/jwals/octo/logs")
LABELS = os.environ.get("YAMADORI_INJECT_LABELS") or os.path.join(
    HERE, "inject", "labels.jsonl")
LABEL_VERSION = "inject-labels/1"
HERMES_RUNS = ("pagoda-h1-pagoda-xhigh-1", "pagoda-h3-pagoda-xhigh-1",
               "pagoda-h4-pagoda-xhigh-1", "pagoda-h5-pagoda-xhigh-1",
               "pagoda-h6-pagoda-xhigh-1", "v0b-V0-xhigh-1",
               "v0c-V0-xhigh-1", "v0d-V0-xhigh-1", "v0e-V0-xhigh-1",
               "v0f-V0-xhigh-1", "pilot-V0-xhigh-1")
PI_RUNS = ("pagoda-p1-pagoda-pi-max-1", "pagoda-p1-pagoda-pi-xhigh-1",
           "pagoda-p2-pagoda-pi-low-1", "pagoda-p3-pagoda-pi-low-1",
           "pagoda-p4-pagoda-pi-medium-1", "probe-lookup-pi-medium-1",
           "probe-lookup-pi-medium-2", "probe2-lookup-pi-medium-1",
           "probe2-lookup-pi-medium-2", "probe2-lookup-pi-medium-3")
PI_TOOLS = ["read", "bash", "edit", "write", "grep", "find", "ls"]
CORPUS_ROUTES = ("code_generation", "code_edit", "library_question", "prose")
TASK_HEAD = 600

# THE RUBRIC (written 2026-09-29 for this label set; what a labeller reads).
RUBRIC = """\
You label whether a skill FACT would have made a coding assistant's NEXT
action more correct, at one moment of a real session.

MATERIAL is exactly what the assistant had at that moment: either the
user's newest message (with the tail of the assistant's previous reply), or
the assistant's newest step (the tool calls it made and their results).
TASK is the session's opening request, for context only.

Give each FACT one level:

0 OFF     About a library, API, language or task the material does not
          involve. (A WebGPU compute fact while the step installs npm
          packages for a React app; a React-compiler fact in a Rust build.)
1 AREA    The material involves the fact's library or area, but the fact
          is about something other than the code, command or error the
          assistant is working on at this moment. (A koota query pattern
          while the step fixes a tsconfig error in a koota project.)
2 DONE    About the code, command or error at hand, but the material
          already shows it done the way the fact says, or the fact restates
          what the assistant plainly already does: reading it would not
          change the next action.
3 NEEDED  About the code, command or error at hand, and the material does
          not show it done that way yet: the material shows the mistake the
          fact prevents or fixes, or the assistant is about to write or run
          exactly what the fact covers with no sign it knows the pattern.
          Reading it would plausibly make the next code or command more
          correct.

Rules:
- Judge the NEXT action only, not the whole session.
- A generic good-practice fact that would not change the next code is at
  most 1, unless the material shows the specific mistake it addresses.
- A fact that is wrong for the material's library, version or API is 0: it
  cannot make anything more correct.
- When the step only reads, lists or installs, a fact about code the
  assistant will write LATER is 1, not 3. When the user asks for code to be
  written, or the step is about to write it, facts about that code can be 3.
- Shared keywords are not relevance. Ask: would the next code or command be
  more correct with this fact in front of the assistant?
"""


# ------------------------------------------------------------- sources -----
def pi_messages(path: str) -> list[dict]:
    """Pi's session as chat messages, from its message_end events."""
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("type") != "message_end":
                continue
            m = d.get("message") or {}
            role = m.get("role")
            c = m.get("content")
            parts = c if isinstance(c, list) else [{"type": "text",
                                                    "text": c or ""}]
            text = "\n".join(str(p.get("text") or "") for p in parts
                             if isinstance(p, dict) and p.get("type") ==
                             "text")
            if role == "system":
                sec = m.get("sections") or {}
                out.append({"role": "system", "content": "\n\n".join(
                    str(v) for v in sec.values()) or text})
            elif role == "user":
                out.append({"role": "user", "content": text})
            elif role == "assistant":
                calls = [{"id": p.get("id"), "type": "function",
                          "function": {"name": p.get("name"),
                                       "arguments": json.dumps(
                                           p.get("arguments") or {})}}
                         for p in parts if isinstance(p, dict)
                         and p.get("type") == "toolCall"]
                mm = {"role": "assistant", "content": text}
                if calls:
                    mm["tool_calls"] = calls
                out.append(mm)
            elif role == "toolResult":
                out.append({"role": "tool",
                            "tool_call_id": m.get("toolCallId"),
                            "content": text})
    return out


def corpus_turns() -> list[dict]:
    import corpus
    db = os.environ.get("YAMADORI_CORPUS_DB_READ") or os.path.join(
        ROOT, "index", "corpus.sqlite3")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = [json.loads(p) for (p,) in con.execute(
            "SELECT payload FROM events WHERE kind='turn' ORDER BY id")]
    finally:
        con.close()
    out, seen = [], set()
    for r in rows:
        route = r.get("route")
        route = route.get("class") if isinstance(route, dict) else route
        req = str(r.get("request") or "")
        if route not in CORPUS_ROUTES or r.get("ends_on") != "user" \
                or len(req) < 40 or req in seen:
            continue
        if (r.get("traffic") or corpus.account_traffic(r.get("account"))) \
                != "test":
            continue
        seen.add(req)
        out.append({"route": route, "messages": [
            {"role": "system", "content": r.get("system_head") or ""},
            {"role": "user", "content": req}]})
    return out


# ------------------------------------------------------------- stage 1 -----
def replay(messages: list[dict], tools: list[str], name: str):
    """Yield (index, kind, candidate rows) for every request point of a
    transcript, stage 1 replayed in order with the skill state carried.
    The injector runs under the STUB decider (mcp/decider_stub.py,
    pass_all: jjava transparent, every candidate item within the caps goes
    in), so the conversation's state moves as it would with everything
    stage 1 offered accepted; its candidates are captured where the
    injector takes them (skill_select._decide_injected's `cands`)."""
    import replay_selection as R          # isolates the stores (its import)
    import decider_stub
    import route
    import skill_select as S
    captured: list = []
    orig = S._decide_injected

    def capture(cands, *a, **kw):
        captured[:] = [c for c in cands if c.get("form") in ("body",
                                                             "recall")]
        return orig(cands, *a, **kw)
    S._decide_injected = capture
    pool = R.skills.armed()
    state = None
    try:
        with decider_stub.installed():
            for i in range(2, len(messages) + 1):
                last = messages[i - 1]
                if last["role"] == "assistant" or (
                        last["role"] == "tool" and i < len(messages)
                        and messages[i]["role"] == "tool") or                         last["role"] == "system":
                    continue
                upto = messages[:i]
                try:
                    rc = route.classify(upto, client_tools=tools,
                                        util={"utility": False}, gate=None,
                                        dbs={})["class"]
                except Exception:                                # noqa: BLE001
                    rc = None
                if rc == "utility":
                    continue
                captured[:] = []
                try:
                    _t, rec, state = S.decide(upto, rc, pool, tools, state,
                                              key=f"{name}:{i}")
                except Exception as e:                           # noqa: BLE001
                    print(f"  {name}:{i} replay raised {type(e).__name__}:"
                          f" {e}", flush=True)
                    continue
                kind = "user" if last["role"] == "user" else "step"
                yield i, kind, list(captured)
    finally:
        S._decide_injected = orig


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def case_of(source: str, group: str, messages: list[dict], i: int,
            kind: str, cands: list[dict]) -> dict | None:
    import decide_turn
    import skill_inject
    skills = [c["skill"] for c in cands]
    events = {c["skill"]["id"]: c.get("trigger") for c in cands}
    items, _left = skill_inject.candidates(skills, None, events)
    if not items:
        return None
    upto = messages[:i]
    state, info = decide_turn.state_of(upto)
    task = next((m.get("content") for m in upto if m.get("role") == "user"),
                "") or ""
    return {"case": _sha(source + ":" + str(i) + ":" + state), "source":
            source, "group": group, "index": i, "kind": kind,
            "state_sha": _sha(state), "state": state,
            "state_info": info, "task": str(task)[:TASK_HEAD],
            "skills": [{"id": s["id"], "name": s.get("name"),
                        "trigger": events.get(s["id"])} for s in skills],
            "items": [{"key": it["key"], "sha": _sha(skill_inject.fact_text(
                it)), "skill": it["name"], "fact": skill_inject.fact_text(it)}
                for it in items]}


def build(out: str, per: int, seed: int, limit_runs: int | None = None
          ) -> dict:
    os.makedirs(out, exist_ok=True)
    allc: list[dict] = []
    srcs = []
    for run in HERMES_RUNS[:limit_runs]:
        d = os.path.join(LOGS, run)
        if os.path.exists(os.path.join(d, "hermes.jsonl")):
            import replay_selection as R
            srcs.append((run, "hermes", R._stream_messages(d),
                         R.HERMES_TOOLS))
    for run in PI_RUNS[:limit_runs]:
        p = os.path.join(LOGS, run, "pi.jsonl")
        if os.path.exists(p):
            srcs.append((run, "pi", pi_messages(p), PI_TOOLS))
    stats = collections.Counter()
    for name, group, msgs, tools in srcs:
        n0 = len(allc)
        for i, kind, cands in replay(msgs, tools, name):
            stats[(group, kind, "points")] += 1
            if not cands:
                continue
            c = case_of(name, group, msgs, i, kind, cands)
            if c:
                allc.append(c)
        print(f"  {name}: {len(msgs)} messages, {len(allc) - n0} cases",
              flush=True)
    for k, t in enumerate(corpus_turns()):
        for i, kind, cands in replay(t["messages"], [], f"corpus{k}"):
            stats[("corpus", kind, "points")] += 1
            if cands:
                c = case_of(f"corpus{k}", "corpus", t["messages"], i, kind,
                            cands)
                if c:
                    allc.append(c)
    print(f"  corpus: {sum(1 for c in allc if c['group'] == 'corpus')} "
          "cases", flush=True)
    for c in allc:
        stats[(c["group"], c["kind"], "cases")] += 1
    rnd = random.Random(seed)
    strata = collections.defaultdict(list)
    for c in allc:
        strata[(c["group"], c["kind"])].append(c)
    picked = []
    for key in sorted(strata):
        xs = strata[key]
        rnd.shuffle(xs)
        picked += xs[:per]
    picked.sort(key=lambda c: (c["group"], c["source"], c["index"]))
    with open(os.path.join(out, "cases.jsonl"), "w", encoding="utf-8") as f:
        for c in picked:
            f.write(json.dumps(c) + "\n")
    rep = {"version": LABEL_VERSION, "cases_all": len(allc),
           "picked": len(picked),
           "items": sum(len(c["items"]) for c in picked),
           "strata": {f"{g}/{k}": len(v) for (g, k), v in
                      sorted(strata.items())},
           "points": {"/".join(k[:2]): v for k, v in sorted(stats.items())
                      if k[2] == "points"}, "seed": seed, "per": per}
    with open(os.path.join(out, "build.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


# ------------------------------------------------------------- hindsight ---
# HINDSIGHT LABELS (coordinator, 2026-09-29: "Label a turn mechanically from
# what happened later in the same transcript ... Write each rule down, with
# its K and its exact test. Hand labels ... go only to the judgment calls
# hindsight can't settle."). ONE rule settles an item:
#
#   H1  THE ITEM'S API, USED NEXT. An item's CODE TOKENS are the words of
#       its text that skill_classify.code_shaped accepts (a call, a dotted
#       name, a camelCase or snake_case identifier; TOKEN below), each
#       matched as skill_classify._topic_rx matches a topic (a call stays a
#       call, word boundaries). An item with none is NOT settled by H1: it
#       goes to the hand labels.
#       NEXT = the arguments of every tool call the assistant makes in the
#       next K assistant turns after the case's point, stopping at the next
#       user message (a new request). Three horizons, each taken from
#       structure, none a chosen number:
#         K=write    through the first assistant turn that WRITES a file
#                    (skill_select._write_targets: a file tool or a shell
#                    redirect) -- the question's own "what the assistant
#                    WRITES or runs next" (skill_inject.ITEM_Q): THE LABEL
#                    TUNED ON (`truth`)
#         K=1        the very next assistant turn only (runs next)
#         K=episode  every turn to the next user message (in an agent run
#                    with one user turn: the rest of the run)
#       K=1 and K=episode are reported beside it as the sensitivity.
#       MATERIAL = the case's state exactly as the decider reads it.
#         a token used NEXT and absent from the MATERIAL  -> level 3 NEEDED
#           (the assistant went on to write exactly what the item covers,
#           and nothing before showed it: a DO NOT item used next is the
#           mistake it prevents -- NEEDED too)
#         a token used NEXT and present in the MATERIAL   -> level 2 DONE
#         no token used NEXT                              -> level 1 if a
#           token is in the MATERIAL (its area, not the next action),
#           else level 0
#       Corpus cases have no "next" (the corpus keeps one turn): hand only.
#
# What H1 cannot see, and the blind hand pass measures: whether the item
# would have made the next code MORE CORRECT (the model may use an API
# wrongly, or rightly without help). `agree` reports H1 against the rubric
# on the items both labelled.
TOKEN = re.compile(r"[A-Za-z_$@][\w$./@-]*(?:\(\))?")


def code_tokens(fact: str) -> list[str]:
    import skill_classify as C
    toks = {t.rstrip(".,;:-/") for t in TOKEN.findall(fact or "")}
    return sorted(t for t in toks if t and C.code_shaped(t))


def _calls_text(m: dict) -> str:
    return "\n".join(str(((tc or {}).get("function") or {}).get(
        "arguments") or "") for tc in m.get("tool_calls") or [])


def _writes(m: dict) -> bool:
    import skill_select
    return any(skill_select._write_targets(tc) for tc in
               m.get("tool_calls") or [])


def next_calls(messages: list[dict], i: int, k) -> str:
    """The tool-call arguments of the next k assistant turns after point i:
    k an int; "write" -- through the first turn that WRITES a file (a
    file-writing tool, or a shell redirect: skill_select._write_targets);
    None -- to the next user message."""
    out, n = [], 0
    for m in messages[i:]:
        if m.get("role") == "user":
            break
        if m.get("role") == "assistant":
            n += 1
            out.append(_calls_text(m))
            if k == "write" and _writes(m):
                break
            if isinstance(k, int) and n >= k:
                break
    return "\n".join(out)


def h1_level(fact: str, material: str, nxt: str) -> int | None:
    import skill_classify as C
    toks = code_tokens(fact)
    if not toks:
        return None
    rx = [C._topic_rx(t) for t in toks]
    used = any(r.search(nxt) for r in rx)
    seen = any(r.search(material) for r in rx)
    if used:
        return 2 if seen else 3
    return 1 if seen else 0


def _source_messages(name: str) -> list[dict] | None:
    d = os.path.join(LOGS, name)
    if os.path.exists(os.path.join(d, "hermes.jsonl")):
        import replay_selection as R
        return R._stream_messages(d)
    if os.path.exists(os.path.join(d, "pi.jsonl")):
        return pi_messages(os.path.join(d, "pi.jsonl"))
    return None


def hindsight(out: str) -> dict:
    """H1 labels for every case with a transcript; labellers H1@write
    (tuned on), H1@1 and H1@episode (sensitivity). Writes LABELS rows; returns counts."""
    cases = load_cases(out)
    msgs_by: dict[str, list] = {}
    n = collections.Counter()
    hand = []
    with open(LABELS, "a", encoding="utf-8") as f:
        for c in cases:
            if c["source"] not in msgs_by:
                msgs_by[c["source"]] = _source_messages(c["source"])
            msgs = msgs_by[c["source"]]
            for it in c["items"]:
                if msgs is None or not code_tokens(it["fact"]):
                    hand.append((c["case"], it["key"]))
                    n["hand"] += 1
                    continue
                for lab, k in (("H1@1", 1), ("H1@write", "write"),
                               ("H1@episode", None)):
                    lv = h1_level(it["fact"], c["state"],
                                  next_calls(msgs, c["index"], k))
                    n[(lab, lv)] += 1
                    f.write(json.dumps({
                        "row": "label", "v": LABEL_VERSION, "case": c["case"],
                        "source": c["source"], "group": c["group"],
                        "kind": c["kind"], "state_sha": c["state_sha"],
                        "item": it["key"], "item_sha": it["sha"],
                        "level": lv, "labeller": lab,
                        "rule": "H1", "k": k if k is not None else
                        "episode"}) + "\n")
    with open(os.path.join(out, "hand_items.json"), "w",
              encoding="utf-8") as f:
        json.dump(hand, f)
    return {"hand": n["hand"], **{f"{a}:{b}": v for (a, b), v in
                                  sorted((k, v) for k, v in n.items()
                                         if isinstance(k, tuple))}}


# ------------------------------------------------------- stage 1 recall ---
# STAGE 1'S RECALL (coordinator, 2026-09-29: "report how many NEEDED facts
# had no candidate at all (items the model needed that the library has but
# stage 1 never offered)"). The rubric labels only what stage 1 offered, so
# recall is measured by H1's test over the WHOLE library at every request
# point: a library item (armed, not doubt-bearing) whose code token appears
# in the next code written (K=write) and not in the material is "used next"
# -- and it was OFFERED if it was among the point's stage-1 candidate items.
# Two token sets, both structural: any code token, and only a call or a
# dotted name (a token with "(" or "."), which a common word cannot be.
# CAVEAT, measured: H1 does not agree with the rubric on the blind sample
# (`agree`), so "used next" is what the assistant DID, not what it needed;
# this is the library's coverage of the code actually written, not a rubric
# recall.
def recall(out: str) -> dict:
    import replay_selection as R
    import skill_classify as C
    import skill_inject
    pool = R.skills.armed()
    lib = []
    for s in pool:
        for it in skill_inject.skill_items(s):
            if it["doubt"]:
                continue
            toks = code_tokens(skill_inject.fact_text(it))
            if toks:
                lib.append((it["key"], toks, [C._topic_rx(t) for t in toks]))
    ident = re.compile(r"[A-Za-z_$][\w$]*")
    per_run = collections.defaultdict(collections.Counter)
    srcs = []
    for run in HERMES_RUNS:
        d = os.path.join(LOGS, run)
        if os.path.exists(os.path.join(d, "hermes.jsonl")):
            srcs.append((run, R._stream_messages(d), R.HERMES_TOOLS))
    for run in PI_RUNS:
        p = os.path.join(LOGS, run, "pi.jsonl")
        if os.path.exists(p):
            srcs.append((run, pi_messages(p), PI_TOOLS))
    import decide_turn
    for name, msgs, tools in srcs:
        c = per_run[name]
        for i, kind, cands in replay(msgs, tools, name):
            c["points"] += 1
            offered = set()
            for x in cands:
                offered |= {it["key"] for it in skill_inject.skill_items(
                    x["skill"])}
            nxt = next_calls(msgs, i, "write")
            if not nxt.strip():
                continue
            state, _ = decide_turn.state_of(msgs[:i])
            words = set(ident.findall(nxt))
            used_any, used_call = set(), set()
            for key, toks, rxs in lib:
                if not any(t.split(".")[0].split("(")[0] in words
                           for t in toks):
                    continue
                hit = [t for t, r in zip(toks, rxs) if r.search(nxt)
                       and not r.search(state)]
                if hit:
                    used_any.add(key)
                    if any("(" in t or "." in t for t in hit):
                        used_call.add(key)
            c["points_with_write"] += 1
            for tag, used in (("any", used_any), ("call", used_call)):
                c[f"used_{tag}"] += len(used)
                c[f"used_{tag}_offered"] += len(used & offered)
                c[f"points_used_{tag}"] += int(bool(used))
                c[f"points_used_{tag}_none_offered"] += int(
                    bool(used) and not (used & offered))
    tot = collections.Counter()
    for c in per_run.values():
        tot.update(c)
    rep = {"library_items_with_tokens": len(lib),
           "total": dict(tot), "per_run": {k: dict(v) for k, v in
                                           sorted(per_run.items())}}
    with open(os.path.join(out, "recall.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


# ------------------------------------------------------------- labelling ---
def load_cases(out: str) -> list[dict]:
    with open(os.path.join(out, "cases.jsonl"), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def sheet(cases: list[dict]) -> str:
    """One labeller's sheet: the rubric, then each case's task, material
    and numbered facts, and the answer form."""
    parts = ["# Labelling sheet", "", RUBRIC, "",
             "Answer with ONE JSON object and nothing else: "
             '{"<case id>": {"<fact key>": <level 0-3>, ...}, ...} -- every '
             "fact of every case below.", ""]
    for c in cases:
        parts += [f"## case {c['case']} ({c['kind']})", "",
                  "TASK (context):", "```", c["task"], "```", "",
                  "MATERIAL:", "````", c["state"], "````", "", "FACTS:"]
        parts += [f"- `{it['key']}` [{it['skill']}] {it['fact']}"
                  for it in c["items"]]
        parts.append("")
    return "\n".join(parts)


def sheets(out: str, batches: int, second: float, seed: int,
           all_items: bool = False) -> dict:
    """Pass A: only the items hindsight cannot settle (hand_items.json,
    written by `hindsight`), batched. Pass B: a BLIND sample of `second` of
    all cases with EVERY item, for A-vs-B and H1-vs-rubric agreement. An
    item pass A already labelled, and a case pass B already read, is not
    sheeted again (a grown case set labels only what is new)."""
    cases = load_cases(out)
    done_a = set(labels("A"))
    done_b = {k[0] for k in labels("B")}
    if all_items:
        # THE RUBRIC OVER EVERY ITEM: H1 turned out not to agree with the
        # rubric (`agree`: kappa on NEEDED ~0 on the blind sample), so the
        # items it settled are hand-labelled too; an item pass B labelled
        # is not sheeted again for A.
        done_a |= set(labels("B"))
    try:
        with open(os.path.join(out, "hand_items.json"), encoding="utf-8") as f:
            hand = {tuple(x) for x in json.load(f)}
    except OSError:
        hand = None
    if hand is None or all_items:
        hand = {(c["case"], it["key"]) for c in cases for it in c["items"]}
    rnd = random.Random(seed + 1)
    order = [c for c in cases if c["case"] not in done_b]
    rnd.shuffle(order)
    n2 = max(1, int(round(len(order) * second))) if second else 0
    blind = order[:n2]
    a_cases = []
    for c in cases:
        its = [it for it in c["items"] if (c["case"], it["key"]) in hand
               and (c["case"], it["key"]) not in done_a]
        if its:
            a_cases.append(dict(c, items=its))
    for b in range(batches):
        with open(os.path.join(out, f"sheet_A{b + 1}.md"), "w",
                  encoding="utf-8") as f:
            f.write(sheet(a_cases[b::batches]))
    if blind:
        with open(os.path.join(out, "sheet_B.md"), "w",
                  encoding="utf-8") as f:
            f.write(sheet(blind))
    return {"cases": len(cases), "a_cases": len(a_cases),
            "a_items": sum(len(c["items"]) for c in a_cases),
            "batches": batches, "blind_sample": n2,
            "blind_items": sum(len(c["items"]) for c in blind)}


def ingest(out: str, answers: str, labeller: str) -> dict:
    cases = {c["case"]: c for c in load_cases(out)}
    with open(answers, encoding="utf-8") as f:
        raw = f.read()
    s, e = raw.find("{"), raw.rfind("}")
    got = json.loads(raw[s:e + 1])
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    n, bad = 0, []
    with open(LABELS, "a", encoding="utf-8") as f:
        for cid, items in got.items():
            c = cases.get(cid)
            if not c or not isinstance(items, dict):
                bad.append(cid)
                continue
            by = {it["key"]: it for it in c["items"]}
            for key, lvl in items.items():
                if key not in by or str(lvl) not in ("0", "1", "2", "3"):
                    bad.append(f"{cid}/{key}")
                    continue
                f.write(json.dumps({
                    "row": "label", "v": LABEL_VERSION, "case": cid,
                    "source": c["source"], "group": c["group"],
                    "kind": c["kind"], "state_sha": c["state_sha"],
                    "item": key, "item_sha": by[key]["sha"],
                    "level": int(lvl), "labeller": labeller}) + "\n")
                n += 1
    return {"written": n, "refused": bad[:20], "refused_n": len(bad)}


def labels(labeller: str | None = None) -> dict:
    """{(case, item): level} -- the latest label of each, by labeller."""
    out = {}
    if not os.path.exists(LABELS):
        return out
    with open(LABELS, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("row") != "label":
                continue
            if labeller and r.get("labeller") != labeller:
                continue
            out[(r["case"], r["item"])] = r["level"]
    return out


def kappa(pairs: list[tuple], cats) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] / n * cb[k] / n for k in cats)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def truth(kind: str = "hindsight") -> dict:
    """{(case, item): level}.
    hindsight  H1 at K=write where it settles the item, else the hand label
               of pass A (the coordinator's method: hindsight for volume,
               hand labels for the judgment calls)
    rubric     the hand labels only: pass A (the items H1 cannot settle)
               and the blind pass B (every item of its sample cases; where
               A and B both labelled an item, A)
    Both are reported because H1 and the rubric measure different things:
    H1 sees what the assistant DID next; the rubric what would have made it
    more correct -- including a pattern it should have used and did not
    (`agree` gives their agreement)."""
    if kind == "rubric":
        t = labels("B")
        t.update(labels("A"))
        return t
    t = labels("A")
    t.update({k: v for k, v in labels("H1@write").items() if v is not None})
    return t


def _pair(x: dict, y: dict) -> dict:
    both = sorted(set(x) & set(y))
    lv = [(x[k], y[k]) for k in both if x[k] is not None and y[k] is not None]
    bi = [(a == 3, b == 3) for a, b in lv]
    conf = collections.Counter(lv)
    return {"n_items": len(lv), "n_cases": len({k[0] for k in both}),
            "exact": round(sum(a == b for a, b in lv) / len(lv), 4)
            if lv else None,
            "kappa_4level": kappa(lv, range(4)),
            "needed_agreement": round(sum(a == b for a, b in bi) / len(bi), 4)
            if bi else None,
            "kappa_needed": kappa(bi, (True, False)),
            "confusion": {f"{a}->{b}": n for (a, b), n in
                          sorted(conf.items())}}


def agree() -> dict:
    """Pass A vs the blind pass B on hand items; H1 (K=1, K=episode) vs
    pass B on the items H1 settles."""
    b = labels("B")
    c = labels("C")
    h = labels("H1@write")
    return {"A_vs_B": _pair(labels("A"), b),
            "A_vs_C": _pair(labels("A"), c),
            "B_vs_C": _pair(b, c),
            "C_vs_H1@write": _pair(c, h),
            "H1@1_vs_B": _pair(labels("H1@1"), b),
            "H1@write_vs_B": _pair(labels("H1@write"), b),
            "H1@episode_vs_B": _pair(labels("H1@episode"), b),
            "H1@1_vs_H1@episode": _pair(labels("H1@1"),
                                        labels("H1@episode"))}


def export_kit(out: str, dest: str) -> dict:
    """The Codex pass-C bundle: the labelling script and its README, the
    cases, the rubric verbatim, and pass B's blind case ids -- one folder,
    portable (no path of this machine inside it)."""
    import shutil
    os.makedirs(dest, exist_ok=True)
    kit = os.path.join(HERE, "codex_label")
    for fn in ("label_codex.py", "README.md"):
        shutil.copyfile(os.path.join(kit, fn), os.path.join(dest, fn))
    shutil.copyfile(os.path.join(out, "cases.jsonl"),
                    os.path.join(dest, "cases.jsonl"))
    with open(os.path.join(dest, "rubric.txt"), "w", encoding="utf-8") as f:
        f.write(RUBRIC)
    blind = sorted({k[0] for k in labels("B")})
    with open(os.path.join(dest, "blind_cases.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(blind) + "\n")
    cases = load_cases(out)
    return {"dest": dest, "cases": len(cases),
            "facts": sum(len(c["items"]) for c in cases),
            "blind_cases": len(blind)}


def stats() -> dict:
    """Levels and the NEEDED base rate under each truth, overall and per
    run (the run = the label rows' `source`; the corpus's turns as one)."""
    src = {}
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                src[r["case"]] = r["source"] if r.get("group") != "corpus" \
                    else "corpus"
    out = {}
    for kind in ("hindsight", "rubric"):
        t = truth(kind)
        per = collections.defaultdict(collections.Counter)
        for (case, _item), v in t.items():
            if v is None:
                continue
            c = per[src.get(case, "?")]
            c["items"] += 1
            c["needed"] += int(v == 3)
        by = collections.Counter(v for v in t.values() if v is not None)
        n = sum(by.values())
        out[kind] = {"items": n, "cases": len({k[0] for k in t}),
                     "levels": dict(sorted(by.items())),
                     "needed_rate": round(by[3] / n, 4) if n else None,
                     "cases_with_a_needed_item": len(
                         {k[0] for k, v in t.items() if v == 3}),
                     "per_run": {r: f"{c['needed']}/{c['items']}"
                                 for r, c in sorted(per.items())}}
    out["from_H1"] = sum(1 for v in labels("H1@write").values()
                         if v is not None)
    out["hand_A"] = len(labels("A"))
    out["blind_B"] = len(labels("B"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--per", type=int, default=40)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--limit-runs", type=int)
    s = sub.add_parser("sheets")
    s.add_argument("--out", required=True)
    s.add_argument("--batches", type=int, default=4)
    s.add_argument("--second", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--all-items", action="store_true",
                   help="sheet every item not yet hand-labelled (the H1-"
                   "settled ones too); no new blind sample")
    g = sub.add_parser("ingest")
    g.add_argument("--out", required=True)
    g.add_argument("--answers", required=True)
    g.add_argument("--labeller", required=True, choices=("A", "B", "C"))
    k = sub.add_parser("export-kit", help="the Codex pass-C bundle "
                       "(bench/skills/codex_label/README.md)")
    k.add_argument("--out", required=True, help="the case directory "
                   "(cases.jsonl)")
    k.add_argument("--dest", required=True, help="the bundle folder to "
                   "write (outside the repo)")
    h = sub.add_parser("hindsight")
    h.add_argument("--out", required=True)
    rc = sub.add_parser("recall")
    rc.add_argument("--out", required=True)
    sub.add_parser("agree")
    sub.add_parser("stats")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        rep = build(a.out, a.per, a.seed, a.limit_runs)
    elif a.cmd == "sheets":
        rep = sheets(a.out, a.batches, 0.0 if a.all_items else a.second,
                     a.seed, all_items=a.all_items)
    elif a.cmd == "ingest":
        rep = ingest(a.out, a.answers, a.labeller)
    elif a.cmd == "hindsight":
        rep = hindsight(a.out)
    elif a.cmd == "recall":
        rep = recall(a.out)
    elif a.cmd == "export-kit":
        rep = export_kit(a.out, a.dest)
    elif a.cmd == "agree":
        rep = agree()
    else:
        rep = stats()
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
