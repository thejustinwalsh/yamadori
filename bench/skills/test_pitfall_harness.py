#!/usr/bin/env python
"""The pitfall harness's rules and scoring (bench/skills/pitfall_harness.py),
offline: tree-sitter only, no model, no store.

    python bench/skills/test_pitfall_harness.py   -> "N/M checks passed"

GATES: every case's rules separate its own pitfall example from its good
example; every case cites a react.dev page with a verbatim quote and a
licence and never names its good pattern's API in the prompt; the answer's
blocks are read with their file names; the paired score counts better and
worse by seed; leave-one-case-out picks a rendering without the held-out
case.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_pitfall_harness_")
sys.path.insert(0, HERE)

import pitfall_harness as H  # noqa: E402

CHECKS: list = []
GOOD_API = {"uses_effect_event": "useEffectEvent", "use_with_suspense": "use(",
            "action_state": "useActionState", "uses_optimistic":
            "useOptimistic", "transition": "useTransition",
            "compiler_configured": "babel-plugin-react-compiler",
            "use_cached_promise": "use(", "open_as_prop": "isOpen"}


def check(name, ok, detail=None):
    CHECKS.append((name, bool(ok)))
    if not ok:
        print("FAIL", name, "" if detail is None else str(detail)[:400])


def tool_result_checks() -> None:
    """THE TOOL-RESULT RENDERINGS (operator, 2026-10-06): the same prompt with
    the skill delivered in a package tool's result, inject and router; a fake
    armed library and a fake model, offline."""
    import json
    import skills
    import package_skills as PS
    import model as M

    def mk(name, area, items, desc):
        its = [{"form": f, "text": t, "situation": "", "quote": "q" * 30}
               for f, t in items]
        return {"id": "id-" + name, "name": name, "version": 1,
                "description": desc, "title": name, "items": its,
                "body": "# " + name + "\n" + "\n".join(
                    f"- {f}: {t}" for f, t in items), "text": "",
                "rule": {"applies_to": {"frameworks": [area]}, "topics": []},
                "lead_for": None, "package": None, "tags": []}
    lead = mk("koota-traits-and-entities", "koota", [
        ("DO", "Signal an in-place array mutation with player.changed(Inventory).")],
        "Use when writing koota traits.")
    other = mk("koota-react-integration", "koota", [
        ("DO", "Read traits in React with useTrait.")],
        "Use when reading koota traits in React components.")
    pool = [lead, other]
    saved = (skills.armed, PS.PROVE_OF, M.post, M.shape,
             H.skill_packages_held if hasattr(H, "skill_packages_held")
             else None)
    skills.armed = lambda *a, **k: list(pool)
    PS.PROVE_OF = lambda s: None
    case = {"id": "koota-x", "area": "koota", "kind": "pitfall",
            "prompt": "Write pickUp.", "packages": ["koota"],
            "skills": [{"name": "koota-traits-and-entities"}],
            "pitfall": ["koota_mutates_trait_unflagged"],
            "good": ["koota_change_flagged"]}
    sent: list[dict] = []
    script: list[dict] = []

    def fake_shape(body, **k):
        return dict(body)

    def fake_post(b):
        sent.append(json.loads(json.dumps(b)))
        return script.pop(0)

    def resp(content="", calls=None):
        m = {"content": content, "reasoning_content": "think"}
        if calls:
            m["tool_calls"] = calls
        return {"choices": [{"message": m, "finish_reason": "tool_calls"
                             if calls else "stop"}],
                "usage": {"completion_tokens": 5}}
    M.shape, M.post = fake_shape, fake_post
    try:
        check("the two renderings are accepted and are not in the default "
              "variants", set(H.TOOL_RESULT_VARIANTS)
              == {"tool_result_inject", "tool_result_router"}
              and not set(H.TOOL_RESULT_VARIANTS) & set(H.VARIANTS))
        di = H.tool_result_delivery(case, "tool_result_inject", "bonsai")
        check("inject: the lookup result carries the lead skill's item, the "
              "case's own skill is among the crafts delivered",
              "player.changed(Inventory)" in di["result"]
              and "koota-traits-and-entities"
              in di["case_skills_delivered"]
              and di["result"].startswith("SOURCE: the npm registry"),
              di["result"][-300:])
        dr = H.tool_result_delivery(case, "tool_result_router", "bonsai")
        check("router: the result carries a table and NO skill body",
              "yama_recall_craft" in dr["result"]
              and "player.changed" not in dr["result"]
              and "Read traits in React" not in dr["result"], dr["result"][-300:])
        ans = "```ts\nplayer.changed(Inventory)\n```"
        script[:] = [resp(ans)]
        a1, g1 = H.tool_arm_generate(case, "tool_result_inject", "m",
                                     "medium", 1, di)
        tools1 = [t["function"]["name"] for t in sent[-1]["tools"]]
        check("inject arm: one generation, the find_package call and its "
              "result are in the conversation, no craft tool",
              a1 == ans and g1["hops"] == 1 and g1["recalls"] == []
              and tools1 == ["yama_find_package"]
              and sent[-1]["messages"][-1]["role"] == "tool"
              and sent[-1]["messages"][2]["tool_calls"][0]["function"][
                  "name"] == "yama_find_package", tools1)
        call = {"id": "c1", "type": "function", "function": {
            "name": "yama_recall_craft",
            "arguments": json.dumps({"name_or_topic":
                                     "koota-react-integration"})}}
        call2 = {"id": "c2", "type": "function", "function": {
            "name": "yama_recall_craft",
            "arguments": json.dumps({"name_or_topic": "how do I read a "
                                     "trait in react?"})}}
        script[:] = [resp("", [call]), resp("", [call2]), resp(ans)]
        n0 = len(sent)
        a2, g2 = H.tool_arm_generate(case, "tool_result_router", "m",
                                     "medium", 1, dr)
        tools2 = [t["function"]["name"] for t in sent[-1]["tools"]]
        check("router arm: the craft tool is offered, a call by name returns "
              "the craft in full, a question is answered by the nearest "
              "craft (no jjava) and said so, the answer is the last turn",
              a2 == ans and g2["hops"] == 3 and len(sent) - n0 == 3
              and "yama_recall_craft" in tools2
              and [r["form"] for r in g2["recalls"]] == ["name", "question"]
              and g2["recalls"][0]["craft"] == "koota-react-integration"
              and g2["recalls"][1].get("resolved_by")
              and "Craft koota-react-integration" in sent[n0 + 1]["messages"][-1][
                  "content"], json.dumps(g2["recalls"]))
        arm = {"sides": {"without": {"repeats": [{
            "answered": True, "pitfall": True, "good": False,
            "types": None}]},
            "tool_result_router": {"repeats": [{
                "answered": True, "pitfall": False, "good": True,
                "types": None, "recalls": 2, "gen": {}}],
                "case_skills_delivered": ["koota-traits-and-entities"]}},
            "case": "k", "skills_missing": []}
        rep = H.report([arm])["cases"]["k"]["tool_result_router"]
        check("the report carries recalls and delivery for the tool-result "
              "arms", rep["recalls_mean"] == 2.0
              and rep["called_recall_in"] == 1
              and rep["case_skills_delivered"]
              == ["koota-traits-and-entities"], rep)
        pool[:] = []
        out = H.run_tool_arm(case, "tool_result_inject", "m", "medium", 1)
        check("no armed craft for the case's packages: not run, said so (a "
              "GAP)", "not_run" in out, out)
    finally:
        skills.armed, PS.PROVE_OF, M.post, M.shape = saved[:4]


def main() -> int:
    st = H.self_test()
    check("every case's rules separate its pitfall example from its good "
          "example", st["cases"] >= 15 and st["ok"] == st["cases"],
          st["failed"])
    for c in H.load_cases():
        src = c.get("source") or {}
        if c["area"] == "react":
            check(f"{c['id']}: cites a react.dev page, a verbatim quote and "
                  "the licence", src.get("url", "").startswith(
                      "https://react.dev/")
                  and len(src.get("quote") or "") > 10
                  and "CC-BY-4.0" in (src.get("licence") or ""))
        else:
            # every area past React (2026-09-30): a PINNED copy of the page
            # (its MANIFEST: commit or fetch date, sha256, licence), the
            # quote verbatim in it, the licence named on the case
            check(f"{c['id']}: cites a page, its pinned copy, a verbatim "
                  "quote and the licence",
                  src.get("url", "").startswith("https://")
                  and bool(src.get("file"))
                  and len(src.get("quote") or "") > 10
                  and len(src.get("licence") or "") > 3, src)
        for r in c["good"]:
            api = GOOD_API.get(r)
            if api and c["kind"] == "pitfall":
                check(f"{c['id']}: the prompt never names the good pattern "
                      f"({api})", api not in c["prompt"])
        for api in c.get("never_in_prompt") or []:
            check(f"{c['id']}: the prompt never names the good pattern "
                  f"({api})", api not in c["prompt"])
        if c["area"] != "react":
            check(f"{c['id']}: names what the prompt must never say "
                  "(never_in_prompt)", bool(c.get("never_in_prompt")))
        if src.get("file"):
            path = os.path.join(H.CASES_DIR, "sources", src["file"])
            with open(path, encoding="utf-8") as f:
                page = f.read()
            check(f"{c['id']}: the quote is VERBATIM in the pinned page "
                  f"({src['file']})", src["quote"] in page)
        check(f"{c['id']}: its rules exist",
              all(r in H.RULES for r in c["pitfall"] + c["good"]))
    b = H.blocks("vite.config.ts\n```ts\nexport default 1\n```\n"
                 "package.json\n```json\n{}\n```")
    check("blocks: each fenced block with the file name before it",
          [x["name"] for x in b] == ["vite.config.ts", "package.json"])
    rec = {"sides": {"without": {"repeats": [
        {"answered": True, "pitfall": True, "good": False, "types": None},
        {"answered": True, "pitfall": False, "good": True, "types": None}]},
        "list": {"repeats": [
            {"answered": True, "pitfall": False, "good": True,
             "types": None},
            {"answered": True, "pitfall": True, "good": False,
             "types": None}]}}}
    p = H.paired(rec, "list")
    check("paired score: better and worse counted by seed, per check",
          p["better"] == 2 and p["worse"] == 2, p)
    recs = []
    for i, gain in enumerate([1, 1, 1]):
        recs.append({"case": f"c{i}", "skills_missing": [], "sides": {
            "without": {"repeats": [{"answered": True, "pitfall": True,
                                     "good": False, "types": None}]},
            "list": {"repeats": [{"answered": True, "pitfall": True,
                                  "good": False, "types": None}]},
            "table": {"repeats": [{"answered": True, "pitfall": False,
                                   "good": True, "types": None}]}}})
    rep = H.report(recs)
    check("leave-one-case-out picks the rendering that wins on the other "
          "cases and scores it on the held-out one",
          rep["held_out_rendering"]["picks"] == {"table": 3}
          and rep["held_out_rendering"]["better"] == 6, rep)
    check("React's cases are all there (20)",
          sum(c["area"] == "react" for c in H.load_cases()) == 20)
    import hashlib
    import json
    man_ok = []
    for d in sorted(os.listdir(os.path.join(H.CASES_DIR, "sources"))):
        mp = os.path.join(H.CASES_DIR, "sources", d, "MANIFEST.json")
        if not os.path.isfile(mp):
            man_ok.append((d, "no MANIFEST.json"))
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if not (m.get("commit") or m.get("fetched")) or not (
                m.get("licence") or {}).get("spdx"):
            man_ok.append((d, "no commit/fetch date or licence"))
        for x in m.get("files") or []:
            fp = os.path.join(H.CASES_DIR, "sources", d, x["file"])
            with open(fp, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != x["sha256"]:
                    man_ok.append((d, x["file"]))
    check("every pinned source: a MANIFEST with commit or fetch date and "
          "licence, and each file's sha256 as recorded", not man_ok, man_ok)
    import pitfall_gap_fill as G
    gl = G.check(G.load())
    check("every gap entry (pitfalls/gaps/*.json): a pinned URL, its pinned "
          "copy's sha256, the cases it serves exist", all(r["ok"] for r in gl),
          [r for r in gl if not r["ok"]])
    rc = H.Code("\n".join([
        "lib.rs", "```rust", "fn main() {}", "```",
        "Cargo.toml", "```toml", "[package]", "name = 'x'", "```",
        "tsconfig.json", "```json", "{ // c",
        ' "compilerOptions": {"strict": true,},', "}", "```"]))
    check("Code reads Rust (tree-sitter), TOML and a commented tsconfig",
          len(rc.rust) == 1 and rc.toml_of("Cargo.toml")[0]["package"][
              "name"] == "x" and rc.json_of("tsconfig.json")[0][
                  "compilerOptions"]["strict"] is True)
    # --- THE TYPE-CHECK FIXES (coordinator, 2026-09-30; found by the area
    # agents): each skewed the with/without comparison
    import io
    import subprocess
    import tarfile
    import typecheck as TC
    check("[tsc 1] a global-only type package (@webgpu/types) is referenced "
          "by the probe; other packages are not",
          TC.global_type_refs(["typegpu", "@webgpu/types"])
          == '/// <reference types="@webgpu/types" />\n'
          and TC.global_type_refs(["react", "three"]) == "")
    sent = {}
    saved = (TC.available, TC._docker, TC._volume_for, TC._box_image)

    def fake_docker(args, stdin=None, timeout=120):
        if stdin:
            with tarfile.open(fileobj=io.BytesIO(stdin)) as t:
                for m in t.getmembers():
                    sent[m.name] = t.extractfile(m).read().decode()
        return subprocess.CompletedProcess(args, 0, (
            b"@@INSTALL ok\n@@TYPES {\"typegpu\": true}\n@@CHECK\n@@RC 0\n"),
            b"")
    TC.available = lambda: (True, "")
    TC._docker = fake_docker
    TC._volume_for = lambda specs: None
    TC._box_image = lambda: "box"
    try:
        TC.check("const d: GPUDevice | null = null;", "tsx",
                 ["typegpu", "@webgpu/types"])
    finally:
        TC.available, TC._docker, TC._volume_for, TC._box_image = saved
    check("[tsc 1] ... and typecheck.check puts the reference at the top of "
          "the probe it sends", sent.get("probe.tsx", "").startswith(
              '/// <reference types="@webgpu/types" />\n'), sent.keys())
    got = {}
    saved_check = TC.check
    TC.check = lambda code, lang, packages, **k: got.update(
        packages=packages) or {"ran": True, "ok": True}
    try:
        H.types_ok({"kind": "pitfall", "packages": []},
                   "```ts\nconst x: number = 1;\n```")
        plain = got.get("packages")
        H.types_ok({"kind": "pitfall"}, "```tsx\nconst x = 1;\n```")
        react = got.get("packages")
    finally:
        TC.check = saved_check
    check("[tsc 2] an explicit empty `packages` stays empty (plain "
          "TypeScript); absent means React", plain == [] and react == ["react"],
          [plain, react])
    probe = H.ts_probe("\n".join([
        "store.ts", "```ts", "import { useState } from 'react';",
        "import type { Item } from './types';",
        "import { api, type Cfg } from './api';",
        "export default function useStore(c: Cfg): Item[] {",
        "  const [x] = useState<Item[]>([]); api(c); return x; }", "```",
        "App.tsx", "```tsx", "import { useState, useEffect } from 'react';",
        "import useStore from './store';",
        "export default function App() { useEffect(() => {}, []);",
        "  useStore({} as any); return null; }", "```"]))
    check("[tsc 3] a local TYPE import is declared as a type only; a local "
          "value import as a value AND a type; a name the answer defines "
          "itself is not declared",
          "type Item = any;" in probe and "declare const Item" not in probe
          and "declare const api: any; type api = any;" in probe
          and "type Cfg = any;" in probe and "useStore: any" not in probe,
          probe)
    check("[tsc 3] several files joined: package imports merged (one react "
          "import), one default export",
          probe.count("from 'react'") == 1
          and "import { useState, useEffect } from 'react';" in probe
          and probe.count("export default") == 1
          and "\nfunction App()" in probe, probe)
    check("the harness system text asks for named files (setup cases)",
          re.search(r"file's name", H.SYSTEM) is not None)
    tool_result_checks()
    ok = sum(1 for _n, v in CHECKS if v)
    print(f"{ok}/{len(CHECKS)} checks passed")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
