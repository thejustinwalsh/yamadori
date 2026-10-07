#!/usr/bin/env python
"""The pipeline's licence detector (mcp/skill_pipeline.py licence_of): a URL
is an address, never a licence statement. No GPU, no network, no live store.

    python mcp/test_skill_licence.py      -> "N/M checks passed"

Found 2026-10-07: react.dev's ref-as-prop page carries an HTML example that
plays `https://interactive-examples.mdn.mozilla.net/media/cc0-videos/
flower.mp4`; the detector read `cc0` out of the URL and recorded the skill as
CC0-1.0 with that line as its quote. Every URL and every bare host/path is
now blanked from a line before the line is read; a licence stated in words
beside a URL, in the frontmatter or on its own line is found as before.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_skill_licence_")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import skill_pipeline as P  # noqa: E402

RESULTS: list[tuple[bool, str, object]] = []


def check(ok: bool, what: str, detail=None) -> None:
    RESULTS.append((bool(ok), what, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}"
          + ("" if ok or detail is None else f"\n        {detail!r}"[:600]),
          flush=True)


def test_a_url_is_not_a_licence() -> None:
    cases = [
        '<video src="https://interactive-examples.mdn.mozilla.net/media/'
        'cc0-videos/flower.mp4" controls />',
        "![cover](https://example.com/images/mit-license-badge.svg)",
        "Read more at https://creativecommons.org/publicdomain/zero/1.0/",
        "See creativecommons.org/licenses/by-sa/4.0 for the terms.",
        "[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)",
        "Fetch `https://cdn.example.com/gpl/spdx/lib.js` first.",
    ]
    got = [P.licence_of(c) for c in cases]
    check(all(g is None for g in got),
          "[url] a licence name inside a URL, a bare host/path or a link "
          "target is not a licence statement", list(zip(cases, got)))


def test_a_statement_in_words_is_found_beside_a_url() -> None:
    for text, spdx in (
            ("Licensed under MIT. See https://opensource.org/licenses/MIT",
             "MIT"),
            ("Content is CC BY 4.0 (https://creativecommons.org/licenses/"
             "by/4.0/)", "CC-BY-4.0"),
            ("Copyright (c) 2024 Meta. Released under the MIT License.",
             "MIT"),
            ("# Docs\n\nThis page is in the public domain dedication.",
             "CC0-1.0")):
        got = P.licence_of(text)
        check(got and got["spdx"] == spdx and got["quote"] in text,
              f"[words] {spdx} stated in words is found, the quote is the "
              "line as written", got)


def test_the_frontmatter_still_wins() -> None:
    got = P.licence_of("---\nname: x\nlicense: Apache-2.0\n---\n"
                       "![v](https://x.io/cc0-videos/a.mp4)\n")
    check(got and got["spdx"] == "Apache-2.0"
          and got["where"] == "frontmatter",
          "[frontmatter] a `license:` key is read first, as before", got)


def test_the_mdn_page_shape() -> None:
    page = ("# ref as a prop\n\nIn React 19, ref is a prop.\n\n"
            '```html\n<video src="https://interactive-examples.mdn.mozilla.'
            'net/media/cc0-videos/flower.mp4" />\n```\n\nMore text here.\n')
    check(P.licence_of(page) is None,
          "[page] the react.dev page shape yields no licence (the stage then "
          "looks beside the source and fails with the operator's remedy)")


# ===========================================================================
# The repository's own licence files (2026-10-07): LICENSE-DOCS for docs pages
# ===========================================================================
import skills  # noqa: E402

PIN = "75ef18a9172e4c100b5d5650ae14c395c5c8ef42"
BASE = f"https://raw.githubusercontent.com/reactjs/react.dev/{PIN}/"
DOCS_PAGE = BASE + "src/content/reference/react/forwardRef.md"
CODE_PAGE = BASE + "src/utils/app.js"
LICENSE_DOCS = ("Attribution 4.0 International\n\n=======================\n\n"
                "Creative Commons Corporation (\"Creative Commons\") is not a "
                "law firm and does not provide legal services.\n\n"
                "Creative Commons Attribution 4.0 International Public "
                "License\n\nBy exercising the Licensed Rights, You accept.\n")


class Ctx:
    def beat(self, progress=None):
        pass


def _reset() -> None:
    con = skills._db()
    try:
        con.execute("DELETE FROM skills")
        con.execute("DELETE FROM skill_versions")
        con.execute("DELETE FROM jobs")
    finally:
        con.close()
    skills._invalidate()


def _licence_run(url: str, files: dict) -> tuple[dict, list, str]:
    """(job result, the URLs asked for, the skill id) with a fake fetch that
    serves `files` ({url: text}) and 404s the rest."""
    _reset()
    rec = skills.create(url=url, name="licence-test", author="t",
                        enqueue_first=False)
    sid = rec["id"]
    skills.store_source(sid, 1, b"# A page\n\nNo licence line in here.\n",
                        {"kind": "url", "content_type": "text/markdown",
                         "charset": "utf-8"})
    skills.update_version(sid, 1, stage="licence")
    asked: list = []

    def fake(u, beat=None):
        asked.append(u)
        if u in files:
            return files[u].encode("utf-8"), {"content_type": "text/plain",
                                              "charset": "utf-8"}
        raise P.Refused(f"GET {u} answered HTTP 404")
    saved = P.fetch_source
    P.fetch_source = fake
    try:
        out = P.handle_licence({"payload": {"skill": sid, "version": 1,
                                            "stage": "licence"}}, Ctx())
    finally:
        P.fetch_source = saved
    return out, asked, sid


def test_the_docs_rule() -> None:
    check(P.is_docs_path("src/content/reference/react/useRef.md")
          and P.is_docs_path("docs/guide/intro.mdx")
          and P.is_docs_path("website/blog/post.html"),
          "[rule] a markup file under a docs/content/website directory is a "
          "documentation page")
    check(not P.is_docs_path("README.md") and not P.is_docs_path(
        "src/utils/app.js") and not P.is_docs_path("docs/build.py")
        and not P.is_docs_path("src/content/data.json") and
        not P.is_docs_path(""),
          "[rule] a root README, source code and data are not, even under a "
          "docs directory when the file is code")
    docs = [c["url"].rsplit("/", 1)[1] for c in P.licence_files_for(
        DOCS_PAGE)]
    code = [c["url"].rsplit("/", 1)[1] for c in P.licence_files_for(
        CODE_PAGE)]
    check(docs[0] == "LICENSE-DOCS.md" and "LICENSE" in docs
          and docs.index("LICENSE-DOCS.md") < docs.index("LICENSE")
          and all(c["url"].startswith(BASE) for c in P.licence_files_for(
              DOCS_PAGE)),
          "[rule] a docs page reads the docs licence file first, then the "
          "general files, all at the pinned ref", docs)
    check(not any("DOCS" in n for n in code) and code[0] == "LICENSE"
          and "LICENCE" in code and "COPYING" in code,
          "[rule] a code path is never offered the docs licence file; "
          "LICENSE*, LICENCE* and COPYING* are", code)
    other = P.licence_files_for("https://example.org/docs/page.html")
    check(other and all(c["kind"] == "general" for c in other),
          "[rule] a source off GitHub keeps worker.licence_candidates",
          other[:2])


def test_a_repo_with_only_license_docs_and_docs_pages() -> None:
    out, asked, sid = _licence_run(
        DOCS_PAGE, {BASE + "LICENSE-DOCS.md": LICENSE_DOCS})
    ver = skills.version(sid, 1)
    lic = ver["licence"]
    check(out.get("licence") == "CC-BY-4.0" and lic["where"] ==
          BASE + "LICENSE-DOCS.md" and lic["file_kind"] == "docs"
          and "documentation page" in lic["file_rule"]
          and lic["quote"] == "Creative Commons Attribution 4.0 "
          "International Public License"
          and lic["quote"] in LICENSE_DOCS,
          "[docs] a docs page of a repo with only LICENSE-DOCS.md is "
          "CC-BY-4.0, from that file's own name sentence, with its URL at "
          "the commit and the rule that tied the file to the path", lic)
    check(asked == [BASE + "LICENSE-DOCS.md"],
          "[docs] the first candidate was enough", asked)
    s = skills.get(sid)
    prov = P._provenance(sid, ver, s)
    att = prov.get("attribution") or {}
    check(att.get("credit") == "reactjs/react.dev"
          and att.get("commit") == PIN
          and att.get("licence_url") == "https://creativecommons.org/"
          "licenses/by/4.0/"
          and att.get("licence_file") == BASE + "LICENSE-DOCS.md"
          and att.get("changes") and att.get("source_url") == DOCS_PAGE,
          "[docs] the provenance carries what CC BY asks: the credit, the "
          "commit, the licence and its link, where it was read and that the "
          "text was changed", att)
    check(prov["licence"]["spdx"] == "CC-BY-4.0",
          "[docs] and the licence beside it")


def test_license_docs_does_not_cover_a_code_path() -> None:
    out, asked, sid = _licence_run(
        CODE_PAGE, {BASE + "LICENSE-DOCS.md": LICENSE_DOCS})
    ver = skills.version(sid, 1)
    check(out.get("licence") == P.NOT_ESTABLISHED
          and ver["state"] == "running"
          and ver["licence"]["status"] == P.NOT_ESTABLISHED
          and not any("DOCS" in u for u in asked),
          "[code] a code path of the same repo does NOT take the docs "
          "licence: LICENSE-DOCS.md is never asked for, the licence is "
          "recorded as not established, and the stage PASSES (it never "
          "blocks)", (out, asked[:3]))
    prov = P._provenance(sid, ver, skills.get(sid))
    check(prov["licence"] == {"status": P.NOT_ESTABLISHED}
          and "attribution" not in prov,
          "[code] the provenance says not established", prov.get("licence"))
    check(any(u.endswith("/COPYING") for u in asked)
          and any(u.endswith("/LICENCE") for u in asked),
          "[code] LICENCE* and COPYING* were looked for", asked)


def test_a_general_file_and_never_a_guess() -> None:
    out, asked, sid = _licence_run(
        DOCS_PAGE, {BASE + "LICENCE.md": "MIT License\n\nCopyright (c) x\n"})
    lic = skills.version(sid, 1)["licence"]
    check(out.get("licence") == "MIT" and lic["file_kind"] == "general"
          and lic["quote"] == "MIT License"
          and lic["where"] == BASE + "LICENCE.md",
          "[general] a LICENCE.md is read when the docs file is absent", lic)
    out2, asked2, sid2 = _licence_run(
        DOCS_PAGE, {BASE + "LICENSE-DOCS.md": "Our docs are made by a team "
                    "and may be read by anyone.\n"})
    check(out2.get("licence") == P.NOT_ESTABLISHED
          and skills.version(sid2, 1)["state"] == "running"
          and not skills.version(sid2, 1)["licence"].get("spdx"),
          "[guess] a licence file with no licence line fills nothing: the "
          "licence is never guessed, and the stage still passes", out2)


def test_the_licence_never_blocks() -> None:
    """Operator, 2026-10-07: "WE DONT NEED TO FUCKING LICENSE TEXT THAT WE
    INJECT IT IS FAIR USE WE ARE NOT DISTRUBITING IT ANYTHING HERE WE ARE
    DOING IS FINE"."""
    out, _asked, sid = _licence_run(
        DOCS_PAGE, {BASE + "LICENSE-DOCS.md": "Attribution-NoDerivatives "
                    "4.0 International\n\nCreative Commons "
                    "Attribution-NoDerivatives 4.0 International Public "
                    "License\n"})
    ver = skills.version(sid, 1)
    check(out.get("licence") == "CC-BY-ND-4.0" and ver["state"] == "running"
          and "restricts derivatives" in ver["licence"].get("note", ""),
          "[never] a no-derivatives licence is RECORDED with a note and "
          "passes: it no longer fails the version", (out, ver["state"],
                                                     ver["licence"]))
    out2, _a2, sid2 = _licence_run(DOCS_PAGE, {})
    v2 = skills.version(sid2, 1)
    check(out2.get("licence") == P.NOT_ESTABLISHED and v2["state"] ==
          "running" and not v2.get("reason"),
          "[never] no licence anywhere: recorded as not established, the "
          "version keeps running (no failure, no remedy)", (out2, v2[
              "state"], v2.get("reason")))
    out3, _a3, sid3 = _licence_run(
        DOCS_PAGE, {BASE + "LICENSE": "Copyright (c) 2026 Acme. All rights "
                    "reserved.\n"})
    v3 = skills.version(sid3, 1)
    check(v3["state"] == "running", "[never] \"all rights reserved\" "
          "passes too", (out3, v3["state"]))


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
