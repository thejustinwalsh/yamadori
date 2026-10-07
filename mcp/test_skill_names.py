#!/usr/bin/env python
"""A skill built for a name keeps it (mcp/skills.py create / find / armed,
mcp/skill_pipeline.py _name_for). No GPU, no network, no live store.

    python mcp/test_skill_names.py      -> "N/M checks passed"

Found 2026-10-06/07 (gap fill): `koota-react-reactive-hooks` became
`koota_react_reactive_hooks` (the store's display slug) and, once distilled,
whatever name the model wrote (`r3f-v10-webgpu-canvas` became
`r3f-v10-webgpu-canvas-setup`), so the pitfall cases that name a skill, and
any list of names, could not find it. What this gates:

  1. create(name=...) records the requested name in the SKILL.md form
     (meta.name_requested); a skill created with no name records none.
  2. the pipeline's final name is the requested one (_name_for), over the
     model's, the store's and the title; with none requested the order is
     what it was.
  3. find() resolves a name in every spelling the store has used: the name
     column, the hyphenated form, the requested name.
  4. an armed skill whose name differs from the requested one serves a row
     with `alias` = the requested name, and the pitfall harness resolves a
     case's skill by name, alias or id.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_skill_names_")
os.environ["YAMADORI_SKILL_CACHE_SECONDS"] = "0"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import skill_classify as C  # noqa: E402
import skill_md  # noqa: E402
import skill_pipeline as P  # noqa: E402
import skills  # noqa: E402

RESULTS: list[tuple[bool, str, object]] = []


def check(ok: bool, what: str, detail=None) -> None:
    RESULTS.append((bool(ok), what, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}"
          + ("" if ok or detail is None else f"\n        {detail!r}"[:600]),
          flush=True)


def reset() -> None:
    con = skills._db()
    try:
        con.execute("DELETE FROM skills")
        con.execute("DELETE FROM skill_versions")
        con.execute("DELETE FROM jobs")
    finally:
        con.close()
    skills._invalidate()


URL = "https://raw.githubusercontent.com/o/r/" + "a" * 40 + "/README.md"


def test_create_records_the_requested_name() -> None:
    reset()
    a = skills.create(url=URL, name="koota-react-reactive-hooks",
                      author="t", enqueue_first=False)
    check(a["meta"]["name_requested"] == "koota-react-reactive-hooks"
          and a["name"] == "koota_react_reactive_hooks",
          "[create] the requested name is kept in the SKILL.md form while "
          "the store's display slug stays underscored", (a["name"], a["meta"]))
    b = skills.create(url=URL, author="t", enqueue_first=False)
    check("name_requested" not in (b["meta"] or {}),
          "[create] no name given, none recorded", b["meta"])


def test_the_pipeline_keeps_it() -> None:
    s = {"name": "koota_react_reactive_hooks",
         "meta": {"name_requested": "koota-react-reactive-hooks"}}
    got = P._name_for(s, {"name": "koota-reactive-hooks-in-react"},
                      {"name": "x"}, "Koota Reactive Hooks")
    check(got == "koota-react-reactive-hooks",
          "[name] the requested name wins over the model's", got)
    got2 = P._name_for({"name": "store-name", "meta": {}},
                       {"name": "model-name"}, {"name": "d"}, "Title")
    got3 = P._name_for({"name": "store-name", "meta": {}}, {},
                       {"name": "distil-name"}, "Title")
    got4 = P._name_for({"name": "", "meta": {}}, {}, {}, "A Title")
    check((got2, got3, got4) == ("model-name", "distil-name", "A Title"),
          "[name] none requested: the model's, then the distil stage's, "
          "then the title, as before", (got2, got3, got4))
    reset()
    a = skills.create(url=URL, name="koota-react-actions", author="t",
                      enqueue_first=False)
    skills.set_name(a["id"], P._name_for(skills.get(a["id"]),
                                         {"name": "model-picked"}, {}, "T"))
    check(skills.get(a["id"])["name"] == "koota-react-actions",
          "[name] set_name with it stores the requested name", skills.get(
              a["id"])["name"])


def test_find_in_every_spelling() -> None:
    reset()
    a = skills.create(url=URL, name="r3f-v10-webgpu-canvas", author="t",
                      enqueue_first=False)
    skills.set_name(a["id"], "r3f-v10-webgpu-canvas-setup")   # an old build
    ids = {skills.find(n)["id"] if skills.find(n) else None for n in (
        "r3f-v10-webgpu-canvas-setup", "r3f_v10_webgpu_canvas_setup",
        "r3f-v10-webgpu-canvas")}
    check(ids == {a["id"]}, "[find] the final name, its underscore form and "
          "the requested name all find the skill", ids)
    check(skills.find("no-such-skill") is None, "[find] an unknown name "
          "finds nothing")


def _arm(sid: str, name: str, *, requested: str | None) -> None:
    rule = C.rule_from_metadata({"languages": ["typescript"]})
    sk = {"name": name, "description": "Use when testing names.",
          "title": "T", "items": [{"form": "DO", "situation": "",
                                   "text": "call `thing()` first.",
                                   "quote": "", "ref": ""}],
          "yamadori": {"id": sid}}
    text = skill_md.render(sk)
    meta = {"name_requested": requested} if requested else {}
    skills._insert(sid, name, "authored", "authored",
                   skills.PATHS["authored"], url=None, author="t",
                   meta=meta, watch=None)
    skills.update_version(sid, 1, stage="validate", text=text,
                          text_sha256=skill_md.sha256(text), classify=rule,
                          validate={"ok": True, "title": "T",
                                    "items": sk["items"]},
                          tests={"activation": {"should": [],
                                                "should_not": []}})
    skills.arm(sid, 1)
    skills._invalidate()


def test_the_served_row_and_the_harness() -> None:
    reset()
    _arm("aaaaaaaaaaaa", "r3f-v10-webgpu-canvas-setup",
         requested="r3f-v10-webgpu-canvas")
    _arm("bbbbbbbbbbbb", "plain-skill", requested=None)
    rows = {r["id"]: r for r in skills.armed()}
    check(rows["aaaaaaaaaaaa"].get("alias") == "r3f-v10-webgpu-canvas"
          and "alias" not in rows["bbbbbbbbbbbb"],
          "[row] a renamed skill serves its requested name as `alias`; one "
          "with its requested name has none", rows["aaaaaaaaaaaa"].get("alias"))
    import pitfall_harness as H
    got, missing = H.library_skills(["r3f-v10-webgpu-canvas"])
    check(len(got) == 1 and got[0]["id"] == "aaaaaaaaaaaa" and not missing,
          "[harness] a case naming the requested name finds the skill",
          (missing, [g["id"] for g in got]))
    got2, miss2 = H.library_skills(["r3f-v10-webgpu-canvas-setup",
                                    "bbbbbbbbbbbb", "nothing-here"])
    check([g["id"] for g in got2] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
          and miss2 == ["nothing-here"],
          "[harness] by the final name or by id too; an unknown name is "
          "still reported missing", (miss2, [g["id"] for g in got2]))


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        print(f"\n{name}", flush=True)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{name} raised", traceback.format_exc()[-1500:])
    n_ok = sum(1 for ok, _w, _d in RESULTS if ok)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed", flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
