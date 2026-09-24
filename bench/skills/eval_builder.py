#!/usr/bin/env python
"""Live eval of the skill builder: distil 5 real public docs, score the format.

    python bench/skills/eval_builder.py            print the plan, run nothing
    python bench/skills/eval_builder.py --live     fetch, screen, classify,
                                                   distil and validate each
                                                   source through the model

STATUS: WRITTEN, NOT RUN. It sends generations to llama-swap through
mcp/model.py (the one door), so it is a GPU consumer: queue it through
bench/queue_runner.py, never beside a benchmark (AGENTS.md "One GPU consumer
at a time"). A 429 or a transport error is recorded as NOT RUN for that
source, never as a format failure.

WHAT IT MEASURES -- format compliance of BUILDER_SYSTEM on this model, per
source, and nothing about whether the skills help:

  finished       the call returned an answer (a length finish is a budget
                 event, recorded separately -- model.BudgetEvent)
  title          a title line was found
  applies_when   the reply carried an applies-when line (it is replaced by
                 the classified one either way)
  items_proposed item lines in the reply
  well_formed    items in one of the three forms (DO / WHEN x / DO NOT)
  quoted         items that carried a source line
  verified       items whose quote is in the source character for character
  do_not         DO NOT items proposed, and how many were justified
  kept           items that survived validate(), and the drop reasons
  tokens         the validated skill's estimated tokens (skill_limits)
  stray          lines that fit no part of the format

n = 5 sources. That is a smoke test of the format, not a result (PROTOCOL
rule 4): it can show the builder cannot produce the format, not that it
reliably can. Repeat it before quoting any rate (rule 10).

THE SOURCES were each fetched successfully by this repo's recipe pipeline
(their rows are in bench/recipes/ with these source_url values and
licences); one is a reference file inside a real agent-skill package
(`.agent/skills/impeccable`). SKILL.md files proper (obra/superpowers,
anthropics/skills -- bench/recipes/SOURCES.md Part 2) belong here once their
exact paths are confirmed; none is listed because no path has been checked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

SOURCES = [
    ("impeccable craft floor (agent skill reference)", "Apache-2.0",
     "https://github.com/pbakaus/impeccable/blob/main/.agent/skills/"
     "impeccable/reference/craft-floor.md"),
    ("react.dev useActionState", "CC-BY-4.0",
     "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/"
     "reference/react/useActionState.md"),
    ("TypeGPU pipelines", "MIT",
     "https://raw.githubusercontent.com/software-mansion/TypeGPU/main/apps/"
     "typegpu-docs/src/content/docs/apis/pipelines.mdx"),
    ("WebGPU fundamentals: bind group layouts", "BSD-3-Clause",
     "https://raw.githubusercontent.com/webgpu/webgpufundamentals/main/webgpu/"
     "lessons/webgpu-bind-group-layouts.md"),
    ("Rustonomicon: FFI", "MIT OR Apache-2.0",
     "https://doc.rust-lang.org/nomicon/ffi.html"),
]
OUT = os.path.join(HERE, "results")


def one(name: str, url: str, model_screen: bool) -> dict:
    import model
    import skill_builder
    import skill_classify
    import skill_limits
    import skill_pipeline
    import skill_screen
    import skills

    rec: dict = {"name": name, "url": url, "at": time.time()}
    try:
        raw_b, meta = skill_pipeline.fetch_source(skills.normalise_url(url))
    except skill_pipeline.Refused as e:
        return dict(rec, status="refused", why=str(e))
    except RuntimeError as e:
        return dict(rec, status="not_run", why=f"fetch: {e}")
    raw = raw_b.decode(meta.get("charset") or "utf-8", "replace")
    html = "html" in (meta.get("content_type") or "")
    text = skill_pipeline.page_text(raw) if html else raw
    rec.update(bytes=len(raw_b), chars=len(text),
               robots=meta["robots"]["why"])
    scr = skill_screen.screen(raw, text, "html" if html else "markdown")
    rec["screen"] = {"ok": scr["ok"], "why": skill_screen.summary(scr),
                     "notes": len(scr["notes"])}
    if model_screen:
        verdicts = []
        for part in skill_pipeline.chunks_of(text):
            try:
                reply = skill_pipeline.ask_model(
                    skill_screen.SCREEN_SYSTEM,
                    skill_screen.screen_user(part, name),
                    max_tokens=skill_pipeline.SCREEN_MAX_TOKENS,
                    temperature=0.0, purpose="screen")
                verdicts.append(skill_screen.read_verdict(
                    skill_pipeline.parse_object(reply),
                    skill_builder._norm(text))["verdict"])
            except Exception as e:                               # noqa: BLE001
                verdicts.append(f"error: {type(e).__name__}: {e}"[:160])
        rec["model_screen"] = verdicts
    rule = skill_classify.classify(text)
    rec["applies_when"] = rule.get("text")
    rec["triggers"] = len(rule.get("triggers") or [])
    parts = skill_pipeline.chunks_of(text)
    rec["chunks"] = len(parts)
    parsed, finished, budget_events = [], 0, 0
    for i, part in enumerate(parts[:skill_pipeline.MAX_CHUNKS]):
        try:
            reply = skill_pipeline.ask_model(
                skill_builder.BUILDER_SYSTEM,
                skill_builder.builder_user(
                    part, applies_when=rule.get("text") or "", name=name,
                    part=f"{i + 1} of {len(parts)}" if len(parts) > 1 else ""),
                max_tokens=skill_pipeline.DISTIL_MAX_TOKENS, purpose="distil")
            finished += 1
            parsed.append(skill_builder.parse(reply))
        except RuntimeError as e:
            if "token limit" in str(e):
                budget_events += 1
            else:
                rec.setdefault("errors", []).append(str(e)[:200])
        except model.BudgetEvent:
            budget_events += 1
    rec.update(finished=finished, budget_events=budget_events)
    if not parsed:
        return dict(rec, status="not_run" if rec.get("errors") else "no_reply")
    merged = skill_builder.merge(parsed)
    items = merged["items"]
    norm = skill_builder._norm(text)
    rec.update(
        title=bool(merged["title"]), applies_line=bool(merged["applies_when"]),
        items_proposed=len(items),
        quoted=sum(1 for it in items if it.get("quote")),
        verified=sum(1 for it in items if it.get("quote")
                     and skill_builder._norm(it["quote"]) in norm),
        do_not_proposed=sum(1 for it in items if it["form"] == "DO NOT"),
        do_not_justified=sum(1 for it in items if it["form"] == "DO NOT"
                             and skill_builder.ABSOLUTE.search(
                                 it.get("quote") or "")),
        stray=len(merged["stray"]))
    res = skill_builder.validate(merged, source=text,
                                 applies_when=rule.get("text") or "")
    drops: dict[str, int] = {}
    for d in res["dropped"]:
        key = d["why"].split(":")[0][:60]
        drops[key] = drops.get(key, 0) + 1
    rec.update(valid=res["ok"], kept=len(res["items"]), drop_reasons=drops,
               tokens=skill_limits.tokens(res["text"]) if res["ok"] else None,
               skill=res["text"] or None, why=res["why"] or None,
               status="scored")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="actually fetch and call the model (GPU)")
    ap.add_argument("--model-screen", action="store_true",
                    help="also run the model-assisted screen on each source")
    a = ap.parse_args(argv)
    if not a.live:
        print("  plan (nothing run; pass --live):")
        for name, lic, url in SOURCES:
            print(f"    {name}  [{lic}]\n      {url}")
        return 0
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"eval_builder_{time.strftime('%Y%m%d_%H%M%S')}"
                             ".jsonl")
    rows = []
    for name, lic, url in SOURCES:
        rec = dict(one(name, url, a.model_screen), licence=lic)
        rows.append(rec)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {rec.get('status'):<8} {name[:40]:<40} items "
              f"{rec.get('kept', '-')}/{rec.get('items_proposed', '-')} "
              f"verified {rec.get('verified', '-')} valid {rec.get('valid')}")
    scored = [r for r in rows if r.get("status") == "scored"]
    print(f"\n  scored {len(scored)}/{len(rows)}; valid "
          f"{sum(1 for r in scored if r['valid'])}/{len(scored)}; "
          f"items kept {sum(r['kept'] for r in scored)}/"
          f"{sum(r['items_proposed'] for r in scored)}; quotes verified "
          f"{sum(r['verified'] for r in scored)}/"
          f"{sum(r['quoted'] for r in scored)}   (n=5: a smoke test)")
    print(f"  results: {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
