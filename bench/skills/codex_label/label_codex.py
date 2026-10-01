#!/usr/bin/env python3
"""PASS C: an independent, NON-CLAUDE labelling pass by Codex, over the
injector's label cases, against the same WRITTEN RUBRIC as passes A and B.

    python label_codex.py --bundle DIR [--model MODEL] [--sample blind]
                          [--first-batch] [--batch-cases 5]

Self-contained: Python 3 standard library and the `codex` CLI on PATH, on
Windows or Linux. DIR is the bundle `bench/skills/inject_labels.py
export-kit` writes (cases.jsonl, rubric.txt, blind_cases.txt, this script,
README.md). Output goes INTO the bundle, so the one folder comes back:

  pass_C.jsonl      one row per labelled item (resumable: a case already
                    in it is skipped on the next run)
  answers_C.json    the same labels in pass B's exact schema
                    {"<case id>": {"<fact key>": <level 0-3>}}, rewritten
                    after every batch -- what inject_labels.py ingests
  codex_log.jsonl   one row per call: the batch's case ids, seconds, exit
                    code, and why a batch was refused (never the prompt)

EACH CALL: `codex exec` in its SAFEST form -- `--sandbox read-only`,
`--ephemeral`, `--skip-git-repo-check`, run in an EMPTY temporary directory
(`--cd`), `--output-schema` forcing the JSON shape, the answer read from
`--output-last-message`; the prompt (the rubric verbatim + the batch) on
stdin and a line telling it to answer from the text alone and run nothing.
No --full-auto, no approval bypass, no dangerous flag. This script never
opens, reads or prints a credential or a config file; Codex uses its own
login as it normally does.

A reply that is not valid JSON, misses a fact key or gives a level outside
0-3 is recorded as refused for that batch (the log says why) and the batch
is retried once; items still missing stay unlabelled and a later run picks
them up.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["labels"],
    "properties": {"labels": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["case", "key", "level"],
        "properties": {"case": {"type": "string"},
                       "key": {"type": "string"},
                       "level": {"type": "integer", "minimum": 0,
                                 "maximum": 3}}}}}}

HEAD = ("You are a careful data labeller. Answer ONLY from the text below. "
        "Do not run any command, open any file or use any tool: everything "
        "you need is in this message.\n\n")
TAIL = ("\n\nReply with ONE JSON object and nothing else: {\"labels\": "
        "[{\"case\": \"<case id>\", \"key\": \"<fact key>\", \"level\": "
        "<0-3>}, ...]} -- one entry for EVERY fact of EVERY case above.")


def load(bundle: str):
    with open(os.path.join(bundle, "rubric.txt"), encoding="utf-8") as f:
        rubric = f.read()
    with open(os.path.join(bundle, "cases.jsonl"), encoding="utf-8") as f:
        cases = [json.loads(ln) for ln in f if ln.strip()]
    blind = set()
    p = os.path.join(bundle, "blind_cases.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            blind = {ln.strip() for ln in f if ln.strip()}
    return rubric, cases, blind


def block(c: dict) -> str:
    facts = "\n".join(f"- `{it['key']}` [{it['skill']}] {it['fact']}"
                      for it in c["items"])
    return (f"## case {c['case']} ({c['kind']})\n\nTASK (context):\n```\n"
            f"{c['task']}\n```\n\nMATERIAL:\n````\n{c['state']}\n````\n\n"
            f"FACTS:\n{facts}\n")


def done_cases(out: str) -> dict:
    got: dict = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                got.setdefault(r["case"], {})[r["key"]] = r["level"]
    return got


def ask(prompt: str, model: str | None, timeout: int) -> tuple[str | None,
                                                               dict]:
    work = tempfile.mkdtemp(prefix="codex_label_")
    try:
        schema = os.path.join(work, "schema.json")
        last = os.path.join(work, "last.txt")
        with open(schema, "w", encoding="utf-8") as f:
            json.dump(SCHEMA, f)
        cmd = [shutil.which("codex") or "codex", "exec", "--sandbox",
               "read-only", "--ephemeral", "--skip-git-repo-check",
               "--cd", work, "--color", "never", "--output-schema", schema,
               "--output-last-message", last]
        if model:
            cmd += ["--model", model]
        cmd.append("-")
        t0 = time.time()
        try:
            p = subprocess.run(cmd, input=prompt.encode("utf-8"),
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, {"why": f"timed out after {timeout} s",
                          "s": round(time.time() - t0, 1)}
        info = {"exit": p.returncode, "s": round(time.time() - t0, 1)}
        if not os.path.exists(last):
            info["why"] = ("no last message written; stderr tail: "
                           + p.stderr.decode("utf-8", "replace")[-300:])
            return None, info
        with open(last, encoding="utf-8") as f:
            return f.read(), info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse(reply: str, batch: list[dict]) -> tuple[dict, str | None]:
    s, e = reply.find("{"), reply.rfind("}")
    try:
        d = json.loads(reply[s:e + 1])
    except ValueError as ex:
        return {}, f"not JSON: {ex}"
    want = {(c["case"], it["key"]) for c in batch for it in c["items"]}
    got: dict = {}
    for r in d.get("labels") or []:
        k = (str(r.get("case")), str(r.get("key")))
        lv = r.get("level")
        if k in want and isinstance(lv, int) and 0 <= lv <= 3:
            got.setdefault(k[0], {})[k[1]] = lv
    n = sum(len(v) for v in got.values())
    return got, (None if n == len(want) else
                 f"{len(want) - n} of {len(want)} facts missing or invalid")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", help="the Codex model (e.g. the operator's "
                    "\"Sol\"); default: Codex's own configured model")
    ap.add_argument("--sample", choices=("all", "blind"), default="all",
                    help="all cases, or only pass B's blind sample")
    ap.add_argument("--batch-cases", type=int, default=5)
    ap.add_argument("--first-batch", action="store_true",
                    help="run ONE batch and stop, to check the output")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true",
                    help="count the calls; run nothing")
    a = ap.parse_args(argv)
    rubric, cases, blind = load(a.bundle)
    if a.sample == "blind":
        cases = [c for c in cases if c["case"] in blind]
    out = os.path.join(a.bundle, "pass_C.jsonl")
    answers = os.path.join(a.bundle, "answers_C.json")
    log = os.path.join(a.bundle, "codex_log.jsonl")
    done = done_cases(out)
    todo = [c for c in cases if set(it["key"] for it in c["items"])
            - set(done.get(c["case"]) or {})]
    batches = [todo[i:i + a.batch_cases]
               for i in range(0, len(todo), a.batch_cases)]
    print(f"{len(cases)} cases, {sum(len(c['items']) for c in cases)} facts;"
          f" {len(todo)} cases to label in {len(batches)} codex call(s)"
          + (" -- first batch only" if a.first_batch else ""), flush=True)
    if a.dry_run:
        return 0
    if not shutil.which("codex"):
        print("the `codex` CLI is not on PATH", file=sys.stderr)
        return 2
    for bi, batch in enumerate(batches[:1] if a.first_batch else batches):
        prompt = (HEAD + rubric.strip() + "\n\n" + "\n".join(
            block(c) for c in batch) + TAIL)
        got, why = {}, "not run"
        for attempt in (1, 2):
            reply, info = ask(prompt, a.model, a.timeout)
            if reply is None:
                why = info.get("why")
            else:
                got, why = parse(reply, batch)
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"batch": bi, "attempt": attempt,
                                    "cases": [c["case"] for c in batch],
                                    **info, "refused": why}) + "\n")
            if not why:
                break
        with open(out, "a", encoding="utf-8") as f:
            for cid, items in got.items():
                for key, lv in items.items():
                    if key not in (done.get(cid) or {}):
                        f.write(json.dumps({"case": cid, "key": key,
                                            "level": lv, "labeller": "C",
                                            "model": a.model or "codex "
                                            "default"}) + "\n")
                        done.setdefault(cid, {})[key] = lv
        with open(answers, "w", encoding="utf-8") as f:
            json.dump(done, f, indent=1)
        print(f"  batch {bi + 1}/{len(batches)}: "
              f"{sum(len(v) for v in got.values())} facts"
              + (f" ({why})" if why else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
