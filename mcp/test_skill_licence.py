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
