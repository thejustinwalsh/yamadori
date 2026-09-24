#!/usr/bin/env python
"""Every copy of the tier feature matrix matches tiers.TIERS.

The matrix -- what RUNS at each tier: thinking, library help, skills, the
code check, fan-out, deep thinking, the addendum, images, the concept seed
(operator, 2026-09-24) -- is derived in ONE place, `tiers.features`. Its
copies are checked against it instead of trusted:

  README.md      "Effort tiers" (thinking on/off only: no effort mapping on
                 show, operator 2026-09-23)
  AGENTS.md      "Prompting this model" (the effort actually sent)
  /dash/api/tiers  every tier's `features` and the `feature_columns`
  web/src        the dashboard's tier ladder and cockpit panel render the
                 API's `features`; a hardcoded column of their own would
                 drift, so none may remain

Documentation that drifts from the code is how this repo shipped a "128k"
label on a 147,456 pool.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiers  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
_results: list[tuple[bool, str, str]] = []
tiers._accepted = ("low", "medium", "xhigh")


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def table(text: str) -> tuple[list[str] | None, dict[str, list[str]]]:
    """(the header cells after `reasoning_effort`, {tier: cells})."""
    head, rows = None, {}
    for line in text.splitlines():
        if re.match(r"\s*\|\s*`reasoning_effort`\s*\|", line):
            head = [c.strip() for c in line.strip().strip("|").split("|")][1:]
            continue
        m = re.match(r"\s*\|\s*`(" + "|".join(tiers.ORDER) + r")`\s*\|(.*)\|\s*$",
                     line)
        if m:
            rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    return head, rows


def expected_head(doc: str) -> list[str]:
    return [("thinking sent" if c == "thinking" and doc == "AGENTS.md" else c)
            for c in tiers.FEATURE_COLUMNS]


def test_documents() -> None:
    n = len(tiers.FEATURE_COLUMNS)
    for doc in ("README.md", "AGENTS.md"):
        with open(os.path.join(ROOT, doc), encoding="utf-8") as f:
            head, rows = table(f.read())
        check(head is not None and head[:n] == expected_head(doc),
              f"{doc}: the columns are tiers.FEATURE_COLUMNS, in order",
              f"doc {head} vs code {expected_head(doc)}")
        check(set(rows) == set(tiers.TIERS),
              f"{doc}: the tier table lists exactly the tiers in tiers.TIERS",
              f"doc {sorted(rows)} vs code {sorted(tiers.TIERS)}")
        for name in tiers.TIERS:
            if name not in rows:
                continue
            got, want = rows[name][:n], tiers.feature_row(name, doc)
            check(got == want, f"{doc}: `{name}` row matches the code",
                  f"doc {got} vs code {want}")


def test_the_dashboard_api() -> None:
    import dashboard
    hit = dashboard.handle_get("/dash/api/tiers")
    body = json.loads(hit[2]) if hit else {}
    check(body.get("feature_columns") == list(tiers.FEATURE_COLUMNS),
          "/dash/api/tiers: feature_columns is tiers.FEATURE_COLUMNS",
          str(body.get("feature_columns")))
    for name in tiers.ORDER:
        got = ((body.get("tiers") or {}).get(name) or {}).get("features")
        check(got == tiers.features(name, "README.md"),
              f"/dash/api/tiers: `{name}` features match the code", str(got))


def test_the_web_source() -> None:
    for rel in ("web/src/screens/sentei/Other.tsx",
                "web/src/screens/Cockpit.tsx"):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            src = f.read()
        check("features" in src,
              f"{rel}: renders the API's `features`", rel)
        stale = [w for w in ("head: 'retrieval'", "head: 'check_code'",
                             "x.retrieval && 'search'")
                 if w in src]
        check(not stale, f"{rel}: no hardcoded column of its own", str(stale))
    with open(os.path.join(ROOT, "web/src/api/types.ts"),
              encoding="utf-8") as f:
        types = f.read()
    check("feature_columns" in types and "features?" in types,
          "web/src/api/types.ts: the tiers type carries features")


def main() -> int:
    for fn in (test_documents, test_the_dashboard_api, test_the_web_source):
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{fn.__name__} raised", f"{type(e).__name__}: {e}")
    for ok, name, detail in _results:
        print(("  pass  " if ok else "  FAIL  ") + name
              + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
