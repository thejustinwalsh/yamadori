#!/usr/bin/env python
"""THE VERIFY DIRECTIVE: at the moment the work can first be run, the model
is told -- in its own voice -- to run it with one of its harness's tools.

Operator, 2026-09-27: "I'll open it in the browser likely needs to see what
tools the harness provides and make a selection for the tool that will help
it build, if it's a web page the browser works, if it's a native app, then
it needs something else ... How we going to get that to land?" And, on the
fallback: when the tool choice does not land, a GENERIC line that names no
tool ("I need to build and run the app to capture its output and check my
progress.").

This is NOT a grader: the model runs its own output with its own harness
tools; nothing of ours judges the result or feeds it back.

Four parts, each where it belongs:

  1. ARTIFACT KIND -- mechanical, from the files the conversation has
     written (tool calls whose result did not fail) and the newest plan's
     FILES: KINDS, one row per file signal, the first row that matches wins
     (a React Native app has a package.json too, so it is read first).
  2. THE MOMENT -- mechanical, once per moment per conversation (deep's
     ledger row, `verify`): the kind's ENTRY file first written (the web
     page's index.html or src/main.*, the app's entry), and every file of
     the current plan written (the plan tracking skill_select.
     server_tool_triggers keeps).
  3. THE TOOL -- the Bonsai decider (decide_turn.Turn.choose: a typed
     CHOICE read in two option orders, positional letters, no label prior
     divided out, Jev's confidence recorded -- decide_turn READOUT; the
     2026-09-27 rotation plus label prior under YAMADORI_DECIDER_READOUT=
     legacy until its removal; each round's decision id
     is in `decision_ids`, the join key into logs/decider_decisions.jsonl): "Which
     of these tools can run this <kind> and show whether it works?" over the
     CLIENT's own tools (name and the first sentence of its own
     description), plus "None of these fits." No per-harness table. More
     tools than single-letter labels: rounds of CHUNK tools, then the
     winners.
  4. THE LINE -- proxy prefills it as main's turn opening, like the
     server-tool triggers' directive (proxy.auto_directive): NAMED when the
     decider picked a tool; GENERIC when it picked none, tied, the kind is
     unknown, the decider could not answer, or an earlier named line did
     not land (the model's turn after it called something else).

A per-artifact "check your work" craft rides with it where one is armed
(phase verify, a language or framework of the kind's; skills stay the
knowledge channel); where none is, the record says so (a coverage gap).
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Kind:
    kind: str           # the artifact kind's id
    label: str          # what the question and the line call it
    verb: str           # the named line's verb phrase
    signal: str         # the file signal this row reads (the row's citation)
    path: str = ""      # a regex over a written or planned file's path
    manifest: str = ""  # a regex over a package.json the work wrote


# The first row that matches wins. Each row names the file signal it reads.
KINDS: tuple[Kind, ...] = (
    Kind("native_app", "native app", "launch it",
         "a package.json that declares react-native or expo "
         "(React Native's and Expo's own install instructions)",
         manifest=r'"(?:react-native|expo)"\s*:'),
    Kind("web_page", "web page", "open it",
         "an .html page (a browser loads it)", path=r"\.html?$"),
    Kind("web_page", "web page", "open it",
         "a Vite config (vite.config.*: Vite serves a web page)",
         path=r"(?:^|/)vite\.config\.[cm]?[jt]s$"),
    Kind("web_page", "web page", "open it",
         "a Next.js config (next.config.*)",
         path=r"(?:^|/)next\.config\.[cm]?[jt]s$"),
    Kind("rust", "Rust program", "run it",
         "a Cargo.toml (cargo's manifest)", path=r"(?:^|/)Cargo\.toml$"),
    Kind("go", "Go program", "run it", "a go.mod (the Go module file)",
         path=r"(?:^|/)go\.mod$"),
    Kind("python", "Python program", "run it",
         "a pyproject.toml or setup.py (Python packaging)",
         path=r"(?:^|/)(?:pyproject\.toml|setup\.py)$"),
    Kind("python", "Python program", "run it", "a .py file",
         path=r"\.py$"),
)
# The kind's ENTRY: the file whose first write is the moment it can run.
ENTRY = {
    "web_page": r"(?:^|/)index\.html?$|(?:^|/)src/main\.[cm]?[jt]sx?$",
    "native_app": r"(?:^|/)App\.[jt]sx?$",
    "rust": r"(?:^|/)src/main\.rs$",
    "go": r"(?:^|/)main\.go$",
    "python": r"(?:^|/)(?:main|app|__main__)\.py$",
}
# A verify craft's languages / frameworks per kind (skill_classify's ids).
SKILL_TERMS = {
    "web_page": {"html", "javascript", "typescript", "css", "react", "r3f",
                 "threejs"},
    "native_app": set(),     # the taxonomy has no React Native term yet
    "rust": {"rust"},
    "go": set(),
    "python": {"python", "pytest"},
}

QUESTION = "Which of these tools can run this {label} and show whether it works?"
NONE_OPTION = "None of these fits."
# One single-token label per option (decider_bonsai.LETTERS, A..Z): a round
# holds every letter but the one "None of these" takes -- the decider's own
# limit (TOO_MANY_OPTIONS), not a chosen number.
CHUNK = 25
# A tool's option text is its name and the FIRST SENTENCE of its own
# description (the sentence a description leads with says what the tool
# does; no character cut is chosen).
# THE LINES (proxy prefills them; the prefill rule: end on a letter).
NAMED = "I'll {verb} with {tool} and check what it shows before going further"
# The operator's wording (2026-09-27), its final period dropped by the
# prefill rule.
GENERIC = ("I need to build and run the app to capture its output and check "
           "my progress")


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(str(p.get("text") or "") for p in c
                         if isinstance(p, dict))
    return str(c or "")


def written(msgs: list[dict]) -> list[dict]:
    """[{path, body, at}] of every file a tool call wrote whose result did
    not fail (skill_select's reading of a write), oldest first; `at` is the
    index of the assistant message that made the call."""
    import skill_select
    out: list[dict] = []
    res = {id(c): _text(msgs[j])
           for j, c in skill_select._pair_results(msgs).items()}
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for c in m.get("tool_calls") or []:
            if not isinstance(c, dict) or id(c) not in res:
                continue
            if skill_select._write_failed(res[id(c)]):
                continue
            for path, body in skill_select._write_targets(c):
                if skill_select.scratch_why(path):
                    continue
                out.append({"path": path.replace("\\", "/"), "body": body,
                            "at": i})
    return out


def artifact_kind(files: list[dict], plan_files: list[str] | None = None
                  ) -> dict:
    """{kind, label, verb, signal, path} of the first KINDS row a written or
    planned file matches; {kind: None} when none does."""
    paths = [(f["path"], f.get("body") or "") for f in files] + [
        (p, "") for p in plan_files or []]
    for k in KINDS:
        for path, body in paths:
            if k.manifest:
                if path.endswith("package.json") and re.search(k.manifest,
                                                               body):
                    return {"kind": k.kind, "label": k.label, "verb": k.verb,
                            "signal": k.signal, "path": path}
            elif k.path and re.search(k.path, path):
                return {"kind": k.kind, "label": k.label, "verb": k.verb,
                        "signal": k.signal, "path": path}
    return {"kind": None, "why": "no written or planned file matches a "
                                 "KINDS row"}


def observe(msgs: list[dict], vst: dict) -> None:
    """Did the last NAMED line land? The turn that opened with it (the
    line opens its REASONING, which a client may drop) is the assistant
    message at the position the request had (`at`: that request's message
    count), else the one whose reasoning or text opens with the line;
    landed when that turn called the named tool."""
    nm = vst.get("named")
    if not nm or "landed" in nm:
        return
    at = nm.get("at")
    cands = ([msgs[at]] if isinstance(at, int) and at < len(msgs) else [])
    cands += [m for m in msgs if str(m.get("reasoning_content") or "")
              .startswith(nm["line"]) or _text(m).startswith(nm["line"])]
    for m in cands:
        if m.get("role") == "assistant":
            used = [((c or {}).get("function") or {}).get("name")
                    for c in m.get("tool_calls") or []]
            nm["landed"] = nm["tool"] in used
            nm["called"] = used
            if not nm["landed"]:
                vst["named_failed"] = {"moment": nm.get("moment"),
                                       "tool": nm["tool"], "called": used}
            return


def moment(msgs: list[dict], vst: dict, auto_st: dict | None
           ) -> dict | None:
    """The verify moment this request is, or None: {moment, key, kind...}.
    Marks its key (once per moment per conversation). `auto_st`: the
    server-tool triggers' state (its plan tracking: plan {sha, files},
    plan_written)."""
    plan_st = dict(((auto_st or {}).get("plan") or {}),
                   plan_written=(auto_st or {}).get("plan_written") or [])
    if not msgs or msgs[-1].get("role") != "tool":
        return None
    observe(msgs, vst)
    fired = vst.setdefault("fired", {})
    files = written(msgs)
    pfiles = list((plan_st or {}).get("files") or [])
    kind = artifact_kind(files, pfiles)
    ai = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"
              and m.get("tool_calls")), default=-1)
    newest = [f for f in files if f["at"] == ai]
    out = None
    if kind.get("kind") and "verify:entry" not in fired:
        pat = ENTRY.get(kind["kind"])
        hit = next((f for f in newest if pat and re.search(pat, f["path"])),
                   None)
        if hit:
            out = {"moment": "entry_written", "key": "verify:entry",
                   "path": hit["path"]}
    if out is None and pfiles:
        sha = (plan_st or {}).get("sha")
        key = f"verify:piece:{sha}"
        wrote = list((plan_st or {}).get("plan_written") or [])
        if key not in fired and len(wrote) >= len(pfiles) and any(
                f["path"].endswith(p) for f in newest for p in pfiles):
            out = {"moment": "piece_done", "key": key, "files": pfiles}
    if out is None:
        return None
    fired[out["key"]] = {"at": len(msgs)}
    out["kind"] = kind
    return out


def tool_options(client_tools: list | None) -> list[dict]:
    """[{name, text}] for the client's own tools: its name and the first
    sentence of its own description (whitespace collapsed: an option is
    one line)."""
    out = []
    for t in client_tools or []:
        f = (t or {}).get("function") or t or {}
        name = f.get("name")
        if not name:
            continue
        d = re.split(r"(?<=[.!?])\s", (f.get("description") or "").strip(),
                     maxsplit=1)[0]
        d = " ".join(d.split())
        out.append({"name": name, "text": f"{name}: {d}" if d else name})
    return out


def choose_tool(messages: list[dict], label: str, options: list[dict], *,
                account: str = "", key: str | None = None,
                request: str | None = None, post=None, upstream=None,
                count=None, on: bool | None = None) -> dict:
    """The decider's pick among `options` (tool_options) for running this
    `label`: {judged, pick (a tool name, or None for none / a tie),
    tie, rounds, distribution, raw_pick, ms, failure?}. Never raises."""
    import decide_turn as T
    import decider_bonsai as D
    t0 = time.time()
    rec: dict = {"judged": False, "pick": None, "tie": False, "rounds": []}
    if not options:
        rec["why"] = "the client offered no tools"
        return rec
    q = QUESTION.format(label=label)
    turn = T.Turn(messages, key=key, account=account, request=request,
                  post=post, upstream=upstream, count=count, on=on)
    if not turn.on:
        rec["why"] = "the decider is off in this process"
        return rec

    def ask(t, opts: list[dict], area: str) -> tuple[str | None, dict]:
        lab = [{"letter": D.LETTERS[i], "id": o["name"], "name": o["name"],
                "text": o["text"]} for i, o in enumerate(opts)]
        lab.append({"letter": D.LETTERS[len(opts)], "id": None,
                    "name": None, "text": NONE_OPTION})
        got, dist = t.choose(area, lab, {"kind": "verify", "question": q})
        byname = {(o["name"] or "none"): dist.get(o["letter"], 0.0)
                  for o in lab}
        a = next((x for x in reversed(t.answers)
                  if x.get("name") == f"choose:{area}"), {})
        by_letter = {o["letter"]: (o["name"] or "none") for o in lab}
        rnd = {"options": [o["name"] for o in opts], "distribution": byname,
               "pick": by_letter.get(got) if got else None,
               "tie": got is None, "raw_pick": by_letter.get(
                   a.get("raw_pick")),
               "decision_id": a.get("decision_id"),
               **({"confidence": a.get("confidence"),
                   "tier": a.get("tier"),
                   "disagreement": a.get("disagreement")}
                  if a.get("readout") else {})}
        rec["rounds"].append(rnd)
        if a.get("decision_id"):
            rec.setdefault("decision_ids", []).append(a["decision_id"])
        return (by_letter.get(got) if got else None), byname

    try:
        with turn:
            chunks = [options[i:i + CHUNK]
                      for i in range(0, len(options), CHUNK)]
            if len(chunks) == 1:
                pick, dist = ask(turn, chunks[0], "verify")
            else:
                winners = []
                for n, ch in enumerate(chunks):
                    p, _d = ask(turn, ch, f"verify:{n}")
                    if p and p != "none":
                        winners.append(next(o for o in ch
                                            if o["name"] == p))
                pick, dist = (ask(turn, winners, "verify:final")
                              if winners else (None, {}))
                if not winners:
                    rec["rounds"].append({"final": "no chunk picked a tool"})
        last = rec["rounds"][-1] if rec["rounds"] else {}
        rec.update(judged=True, pick=None if pick in (None, "none") else pick,
                   none=pick == "none", tie=bool(last.get("tie")),
                   distribution=dist, raw_pick=last.get("raw_pick"))
    except D.DeciderUnavailable as e:
        rec.update(why="the decider could not answer", failure=e.facts())
    except Exception as e:                                       # noqa: BLE001
        rec.update(why="the decider raised",
                   failure={"code": type(e).__name__,
                            "situation": str(e)[:200], "retryable": False})
    rec.update(slot=turn.slot, release=turn.release_rec,
               ms=round((time.time() - t0) * 1000, 1))
    return rec


def directive(kind: dict, choice: dict, vst: dict) -> tuple[str, str, str]:
    """(line, form named|generic, why)."""
    if not kind.get("kind"):
        return GENERIC, "generic", "the artifact kind is unknown"
    if vst.get("named_failed"):
        nf = vst["named_failed"]
        return (GENERIC, "generic",
                f"the named line at {nf.get('moment')} did not land (the "
                f"turn after it called {nf.get('called') or 'nothing'}, not "
                f"{nf.get('tool')})")
    if not choice.get("judged"):
        return (GENERIC, "generic",
                choice.get("why") or "the decider did not judge")
    if choice.get("tie"):
        return GENERIC, "generic", "the decider's pick was a tie"
    if not choice.get("pick"):
        return GENERIC, "generic", "the decider picked none of the tools"
    return (NAMED.format(verb=kind["verb"], tool=choice["pick"]), "named",
            f"the decider picked {choice['pick']}")


def verify_skill(kind: str | None, pool: list[dict] | None = None
                 ) -> dict | None:
    """The armed craft for checking this kind of work (phase verify, one of
    SKILL_TERMS[kind] among its languages / frameworks), or None."""
    if not kind:
        return None
    terms = SKILL_TERMS.get(kind) or set()
    if not terms:
        return None
    if pool is None:
        import skills
        pool = skills.armed()

    def ids(v):
        return {x if isinstance(x, str) else (x or {}).get("id")
                for x in (v or [])}
    for s in pool:
        rule = s.get("rule") or {}
        if "verify" in ids(rule.get("phase")) and (
                ids(rule.get("language")) | ids(rule.get("framework"))
        ) & terms:
            return s
    return None
