#!/usr/bin/env python
"""THE QUESTION BANK: every armed skill indexed by the questions it answers.

SHELVED (operator, 2026-09-27): the question pipeline is not on any path;
the package rule (mcp/skill_packages.py) is the selection path. Kept, with
its test (mcp/test_skill_questions_bank.py), unused.

    python mcp/skill_questions_bank.py build [--limit N] [--ids a,b] [--pick-spread N]
    python mcp/skill_questions_bank.py embed
    python mcp/skill_questions_bank.py status

Operator, 2026-09-27: "It's kind of an inversion of skills then selection
... these skills answer these questions and this prompt asks these
questions." The skill side of the question match (the prompt side is
mcp/skill_question_match.py). Established pattern: doc2query (Nogueira et
al. 2019, a document indexed by the questions it answers) and FAQ
question-to-question retrieval.

WHAT A BANK ENTRY IS. For one armed skill at one revision, the questions its
items and its description answer (question_prompts.BANK_SYSTEM: one per
item, one for the whole skill), each tagged with

    kind    how | choose | error | check  (the model's, from the table)
    item    the item it came from (0: the skill as a whole)
    area    the skill's area (skill_select.area_of, read only)
    phases  the skill's phases (its rule, skill_classify's taxonomy)

THE MODEL PROPOSES, THE CODE VERIFIES (verify()). A question is rejected,
and the rejection recorded with its reason, when it

    - is not one line ending in "?" or is over QUESTION_CHARS characters
    - names an item the skill does not have, or a kind outside KINDS
    - names an API, package or term the skill never mentions (a code-shaped
      token -- a call, a dotted or camelCase name, a scoped package, a
      version -- that is not in the skill's name, title, description,
      topics or items)
    - repeats a question already accepted for the skill

WHERE IT LIVES. One JSON file per (skill id, revision, template version):
<skills store>/questions/<id>/v<rev>__<qbank-N>.json. A file that exists is
done (resume = run again); a skill edit (a new revision) or a template bump
re-distils only what changed. Vectors: <store>/questions/_vectors.npz,
keyed by sha1 of the question text under the embedder's model name; the
questions are embedded PLAIN (Qwen3-Embedding: documents get no
instruction).

ONE DOOR, ONE CONSUMER. Generation goes through mcp/model.py (tiers.apply,
the main model), effort "medium" and temperature 0.2 as the skill
pipeline's own model stages (skill_pipeline.ask_model). Before each skill
the batch reads llama-swap's /slots for the main model and waits while any
slot is processing (another consumer is generating; AGENTS.md "One GPU
consumer at a time"). Embedding goes through code_search.embed (gpu_room,
the resident embedder), as skill_select's trigger vectors do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import question_prompts as QP  # noqa: E402

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MAIN_MODEL = os.environ.get("YAMADORI_MODEL", "bonsai")
# The answer allowance the bank call asks for; tiers.budget raises it to its
# A_MIN floor (2,048) anyway, so this is that floor, named.
ANSWER_TOKENS = 2048
TEMPERATURE = 0.2          # skill_pipeline.ask_model's default
EFFORT = "medium"          # skill_pipeline.ask_model's effort
# While another consumer generates, the batch re-reads /slots this often.
# Not a quality number: how long a paused batch sleeps between looks.
BUSY_POLL_S = 15


def area_of(rule: dict) -> str:
    """skill_select.area_of, restated (that module imports the prompt
    templates, which are edited separately; mcp/test_skill_questions_bank.py
    checks the two agree on every armed skill): the rule's first framework,
    else its first language, else its first non-code artifact, else
    "code"."""
    a = (rule or {}).get("applies_to") or {}
    for key in ("frameworks", "languages"):
        if a.get(key):
            return str(a[key][0])
    arts = [x for x in a.get("artifacts") or [] if x != "code"]
    return str(arts[0]) if arts else "code"


def store_dir() -> str:
    import skills
    return os.path.join(os.path.abspath(skills.STORE), "questions")


def template_tag() -> str:
    return QP.BANK_VERSION.replace("/", "-")


def entry_path(sid: str, revision: int) -> str:
    return os.path.join(store_dir(), sid, f"v{int(revision)}__"
                        f"{template_tag()}.json")


def vectors_path() -> str:
    return os.path.join(store_dir(), "_vectors.npz")


# ---------------------------------------------------------------------------
# Parse and verify.
# ---------------------------------------------------------------------------
_LINE = re.compile(r"^\s*Q(\d+)\s*\[\s*([a-z]+)\s*\]\s*:\s*(.+?)\s*$")


def parse_reply(reply: str) -> tuple[list[dict], list[dict]]:
    """([{item, kind, text}], [{line, why}]) from the model's reply. A line
    that is not in the reply shape is a rejection, never guessed at."""
    reply = re.sub(r"(?s)<think>.*?</think>", "", reply or "")
    got, bad = [], []
    for line in reply.splitlines():
        if not line.strip():
            continue
        m = _LINE.match(line)
        if not m:
            bad.append({"line": line.strip()[:240], "why": "not in the reply "
                        "shape `Qn [kind]: question?`"})
            continue
        got.append({"item": int(m.group(1)), "kind": m.group(2),
                    "text": " ".join(m.group(3).split())})
    return got, bad


# A code-shaped token: a call `name(`, a dotted name `a.b`, a scoped or
# slashed package `@a/b` `a/b`, a camelCase or PascalCase-with-inner-capital
# name, a snake_case name, a backticked span, a version `v10` / `0.185`.
_CODE = re.compile(
    r"`([^`]+)`"
    r"|(@?[A-Za-z_][\w-]*/[\w./-]+)"
    r"|([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+)"
    r"|([A-Za-z_$][\w$]*)\("
    r"|\b([a-z]+[A-Z][\w$]*|[A-Z][a-z0-9]+[A-Z][\w$]*)\b"
    r"|\b([a-z][a-z0-9]*_[a-z0-9_]+)\b"
    r"|\b(v\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*)\b")


def code_tokens(text: str) -> list[str]:
    out = []
    for m in _CODE.finditer(text or ""):
        tok = next(g for g in m.groups() if g)
        tok = tok.strip().rstrip(".?,;:")
        if tok and tok not in out:
            out.append(tok)
    return out


def skill_text(skill: dict) -> str:
    """Everything a question may name: the skill's name, title,
    description, topics and items (their situations too)."""
    import skill_md
    parts = [skill.get("name") or "", skill.get("title") or "",
             skill.get("description") or "", skill.get("body") or ""]
    parts += [str(t) for t in (skill.get("rule") or {}).get("topics") or []]
    parts += [skill_md.item_line(it) for it in skill.get("items") or []]
    return "\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def unknown_names(question: str, skill: dict) -> list[str]:
    """The code-shaped tokens of `question` the skill never mentions."""
    have = skill_text(skill).lower()
    have_n = " " + _norm(have) + " "
    out = []
    for tok in code_tokens(question):
        t = tok.lower()
        if t in have:
            continue
        # `Position` for `Position` inside `world.query(Position, ...)`:
        # the token's own words, in order, present in the skill.
        if _norm(t) and (" " + _norm(t) + " ") in have_n:
            continue
        out.append(tok)
    return out


def verify(q: dict, skill: dict, n_items: int, seen: set) -> str | None:
    """None when the question is kept, else why it is rejected."""
    text = q.get("text") or ""
    if "\n" in text or not text.endswith("?"):
        return "not one line ending in '?'"
    if len(text) > QP.QUESTION_CHARS:
        return f"over {QP.QUESTION_CHARS} characters ({len(text)})"
    if q.get("kind") not in QP.KINDS:
        return f"kind {q.get('kind')!r} is not one of {'/'.join(QP.KINDS)}"
    if not 0 <= int(q.get("item", -1)) <= n_items:
        return f"item {q.get('item')} is not one of the skill's 0..{n_items}"
    bad = unknown_names(text, skill)
    if bad:
        return "names what the skill never mentions: " + ", ".join(bad[:4])
    key = _norm(text)
    if key in seen:
        return "repeats a question already kept for this skill"
    return None


# ---------------------------------------------------------------------------
# One skill.
# ---------------------------------------------------------------------------
def slots_busy(model: str = MAIN_MODEL) -> list[int] | None:
    """The main model's slots that are processing (another consumer is
    generating), [] when none, None when /slots cannot be read."""
    import max_mode
    if max_mode.blocks(model):
        return None        # MAX MODE: the model is off the card; asking would load it (mcp/max_mode.py)
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/upstream/{model}/slots",
                                    timeout=10) as r:
            table = json.loads(r.read().decode("utf-8"))
    except Exception:                                            # noqa: BLE001
        return None
    return [s.get("id") for s in table if isinstance(s, dict)
            and s.get("is_processing")]


def wait_idle(log=print) -> dict:
    """Wait until no slot of the main model is processing. Returns
    {waited_s, looks}. /slots unreadable counts as busy (the model may be
    loading): the batch never generates blind."""
    t0, looks = time.time(), 0
    while True:
        busy = slots_busy()
        looks += 1
        if busy == []:
            return {"waited_s": round(time.time() - t0, 1), "looks": looks}
        if looks == 1 or looks % 20 == 0:
            log(f"  paused: main model slots {busy if busy is not None else '(unreadable)'} "
                "are processing; waiting")
        time.sleep(BUSY_POLL_S)


def generate(skill: dict) -> dict:
    """One bank generation through the one door. Returns {reply, usage,
    timings, finish, reasoning_chars, seconds}."""
    import model
    msgs = [{"role": "system", "content": QP.BANK_SYSTEM},
            {"role": "user", "content": QP.bank_user(skill)}]
    t0 = time.time()
    d = model.chat(msgs, effort=EFFORT, max_tokens=ANSWER_TOKENS,
                   temperature=TEMPERATURE)
    secs = time.time() - t0
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {"reply": msg.get("content") or "", "usage": d.get("usage") or {},
            "timings": d.get("timings") or {},
            "finish": ch.get("finish_reason"),
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "seconds": round(secs, 2)}


def distil(skill: dict, gen=generate) -> dict:
    """The bank entry for one skill (not written). `gen(skill)` is the
    generation (replaced in tests)."""
    items = skill.get("items") or []
    rule = skill.get("rule") or {}
    g = gen(skill)
    got, bad = parse_reply(g["reply"])
    kept, rejected, seen = [], list(bad), set()
    for q in got:
        why = verify(q, skill, len(items), seen)
        if why:
            rejected.append({"text": q["text"], "item": q["item"],
                             "kind": q["kind"], "why": why})
            continue
        seen.add(_norm(q["text"]))
        kept.append({"text": q["text"], "kind": q["kind"],
                     "item": q["item"]})
    covered = sorted({q["item"] for q in kept})
    return {"skill": skill["id"], "name": skill["name"],
            "revision": int(skill["version"]), "template": QP.BANK_VERSION,
            "area": area_of(rule),
            "phases": [p for p in rule.get("phases") or []],
            "items": len(items), "questions": kept, "rejected": rejected,
            "items_without_question": [n for n in range(len(items) + 1)
                                       if n not in covered],
            "finish": g["finish"],
            "cost": {"seconds": g["seconds"],
                     "prompt_tokens": g["usage"].get("prompt_tokens"),
                     "completion_tokens": g["usage"].get("completion_tokens"),
                     "reasoning_chars": g["reasoning_chars"],
                     "predicted_per_second": (g["timings"] or {}).get(
                         "predicted_per_second")},
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def write_entry(entry: dict) -> str:
    p = entry_path(entry["skill"], entry["revision"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return p


def read_entry(sid: str, revision: int) -> dict | None:
    try:
        with open(entry_path(sid, revision), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The batch.
# ---------------------------------------------------------------------------
def todo(pool: list[dict]) -> list[dict]:
    """The armed skills with no entry at their revision and this template."""
    return [s for s in pool if read_entry(s["id"], s["version"]) is None]


def spread(pool: list[dict], n: int) -> list[dict]:
    """`n` skills spread over the areas (round robin over areas, in id
    order): a cost sample that is not all one area."""
    by: dict[str, list[dict]] = {}
    for s in sorted(pool, key=lambda s: s["id"]):
        by.setdefault(area_of(s.get("rule") or {}), []).append(s)
    out, i = [], 0
    areas = sorted(by)
    while len(out) < n and any(by.values()):
        a = areas[i % len(areas)]
        if by[a]:
            out.append(by[a].pop(0))
        i += 1
    return out


def build(skills_list: list[dict], log=print) -> list[dict]:
    """Distil every skill in `skills_list` that has no entry yet, waiting for
    an idle main model before each one. Returns the entries written."""
    out = []
    for n, s in enumerate(skills_list, 1):
        if read_entry(s["id"], s["version"]) is not None:
            continue
        w = wait_idle(log)
        try:
            e = distil(s)
        except Exception as ex:                                  # noqa: BLE001
            log(f"  [{n}/{len(skills_list)}] {s['name']}: FAILED "
                f"{type(ex).__name__}: {ex}"[:300])
            continue
        e["cost"]["waited_s"] = w["waited_s"]
        write_entry(e)
        out.append(e)
        c = e["cost"]
        log(f"  [{n}/{len(skills_list)}] {s['name']:<44} {c['seconds']:>6.1f}s "
            f"{c['completion_tokens']} tok  kept {len(e['questions'])} "
            f"rejected {len(e['rejected'])}  finish {e['finish']}")
    return out


def entries(pool: list[dict]) -> list[dict]:
    """The bank entries of the armed skills at their served revision."""
    out = []
    for s in pool:
        e = read_entry(s["id"], s["version"])
        if e is not None:
            out.append(e)
    return out


def question_rows(pool: list[dict]) -> list[dict]:
    """One row per accepted question of the armed skills: {skill, name,
    area, kind, item, text, phases}."""
    rows = []
    for e in entries(pool):
        for q in e["questions"]:
            rows.append({"skill": e["skill"], "name": e["name"],
                         "area": e["area"], "kind": q["kind"],
                         "item": q["item"], "text": q["text"],
                         "phases": e.get("phases") or []})
    return rows


# ---------------------------------------------------------------------------
# Vectors.
# ---------------------------------------------------------------------------
def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _embed_model() -> str:
    import code_search
    return str(code_search.EMBED_MODEL)


def load_vectors() -> dict:
    import numpy as np
    p = vectors_path()
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=False)
    if str(z["model"]) != _embed_model():
        return {}
    return dict(zip([str(k) for k in z["keys"]], z["mat"]))


def embed_bank(rows: list[dict], batch: int = 64, log=print) -> dict:
    """Embed every accepted question not yet in the vector file (plain: no
    instruction on the document side). Returns {have, added}."""
    import numpy as np
    import code_search
    have = load_vectors()
    need = list(dict.fromkeys(r["text"] for r in rows
                              if text_key(r["text"]) not in have))
    for i in range(0, len(need), batch):
        chunk = need[i:i + batch]
        mat = code_search.embed(chunk, is_query=False)
        norms = np.linalg.norm(mat, axis=1)
        if float(norms.min()) < 0.5:
            raise RuntimeError("the embedder returned zero vectors")
        for t, v in zip(chunk, mat):
            have[text_key(t)] = v.astype(np.float32)
    if need:
        keys = sorted(have)
        os.makedirs(store_dir(), exist_ok=True)
        tmp = vectors_path() + ".tmp.npz"
        np.savez(tmp, keys=np.array(keys), model=_embed_model(),
                 mat=np.stack([have[k] for k in keys]))
        os.replace(tmp, vectors_path())
    log(f"  vectors: {len(have)} held, {len(need)} added")
    return {"have": len(have), "added": len(need)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "embed", "status"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--pick-spread", type=int, default=0,
                    help="distil N skills spread over the areas (a cost "
                         "sample)")
    a = ap.parse_args(argv)
    import skills
    pool = skills.armed()
    if a.cmd == "status":
        es = entries(pool)
        nq = sum(len(e["questions"]) for e in es)
        nr = sum(len(e["rejected"]) for e in es)
        print(f"armed {len(pool)}; bank entries {len(es)} ({QP.BANK_VERSION});"
              f" questions {nq}, rejected {nr}; vectors "
              f"{len(load_vectors())}")
        return 0
    if a.cmd == "embed":
        embed_bank(question_rows(pool))
        return 0
    if a.ids:
        want = set(a.ids.split(","))
        sel = [s for s in pool if s["id"] in want or s["name"] in want]
    elif a.pick_spread:
        sel = spread(todo(pool), a.pick_spread)
    else:
        sel = todo(pool)
    if a.limit:
        sel = sel[:a.limit]
    print(f"bank {QP.BANK_VERSION}: {len(sel)} skill(s) to distil "
          f"({len(todo(pool))} of {len(pool)} armed have no entry)")
    t0 = time.time()
    got = build(sel)
    print(f"done: {len(got)} entries in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
