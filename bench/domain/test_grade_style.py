#!/usr/bin/env python
"""Proof that grade_style (the separate style / modernness score) measures
what it says, and never touches pass/fail.

    python bench/domain/test_grade_style.py

  - the lint sandbox holds exactly the pinned versions
  - a reference answer (ts, typegpu, react, tc) is scored with no scorer error
  - a deliberately sloppy answer (`any`, unused vars) scores MORE lint errors
    than the reference for the same task (the lint can fail)
  - a hook called conditionally is reported by react-hooks/rules-of-hooks
  - each React 19 flag fires on its idiom and stays silent on the reference;
    an error boundary class is exempt; a credit counts only when the prompt
    names the API
  - the same answer scores the same twice (deterministic)
  - a Rust task is skipped, prose is skipped, neither is an error
"""
from __future__ import annotations

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grade  # noqa: E402
import grade_style as S  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def read(p: str) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read()


def fence(code: str, tag: str = "ts") -> str:
    return f"Here it is.\n\n```{tag}\n{code.rstrip()}\n```\n"


def tasks() -> dict:
    ts = grade.load_tasks() + grade.load_tasks(os.path.join(HERE, "tasks_react.jsonl"))
    if os.path.isfile(grade.TC_JSONL):
        ts += grade.load_tasks(grade.TC_JSONL)
    return {t["id"]: t for t in ts}


T = tasks()


def test_sandbox():
    check(S.ready() is None, "lint sandbox holds the pinned versions", str(S.ready()))


def test_references():
    for tid, tag in (("ts01", "ts"), ("tg01", "ts"), ("react-01", "tsx"),
                     ("tc-00002-return-type", "ts")):
        t = T.get(tid)
        if not t:
            continue
        r = S.style(t, fence(read(grade._p(t["reference"])), tag))
        check(r["style_error"] is None and r["skipped"] is None
              and isinstance(r.get("lint_errors"), int) and not r.get("lint_fatal"),
              f"{tid}: reference scored, no scorer error",
              str({k: r.get(k) for k in ("style_error", "skipped", "lint_errors",
                                         "lint_rules")})[:300])


def test_lint_can_fail():
    t = T["ts01"]
    ref = S.style(t, fence(read(grade._p(t["reference"]))))
    sloppy = read(grade._p(t["reference"])) + (
        "\nexport function sloppy(x: any): any {\n  const unused = 1;\n"
        "  return x.whatever;\n}\n")
    bad = S.style(t, fence(sloppy))
    check(bad["lint_errors"] > ref["lint_errors"]
          and "@typescript-eslint/no-explicit-any" in bad["lint_rules"],
          "`any` and an unused variable score more lint errors than the reference",
          f"ref={ref['lint_errors']} bad={bad['lint_errors']} {bad['lint_rules']}")
    again = S.style(t, fence(sloppy))
    check((again["lint_errors"], again["lint_warnings"], again["lint_rules"])
          == (bad["lint_errors"], bad["lint_warnings"], bad["lint_rules"]),
          "the same answer scores the same twice")


def test_hooks_rule():
    t = T["react-01"]
    code = ("import { useState } from 'react';\n"
            "export function Bad({ on }: { on: boolean }) {\n"
            "  if (on) { const [x] = useState(0); return <p>{x}</p>; }\n"
            "  return null;\n}\n")
    r = S.style(t, fence(code, "tsx"))
    check("react-hooks/rules-of-hooks" in (r.get("lint_rules") or {}),
          "a conditional hook is caught by react-hooks/rules-of-hooks",
          str(r.get("lint_rules")))


LEGACY = """\
import React, { forwardRef, Component, useEffect, useState } from 'react';
import ReactDOM from 'react-dom';

export const Input = forwardRef<HTMLInputElement, { v: string }>((p, ref) => <input ref={ref} value={p.v} />);

export class Old extends React.Component<{ n: number }> {
  render() { return <p>{this.props.n}</p>; }
}

export function Label({ text }: { text?: string }) { return <span>{text}</span>; }
Label.defaultProps = { text: 'x' };

export function Full({ first, last }: { first: string; last: string }) {
  const [full, setFull] = useState('');
  useEffect(() => { setFull(first + ' ' + last); }, [first, last]);
  return <p>{full}</p>;
}

ReactDOM.render(<Old n={1} />, document.getElementById('root'));
"""

MODERN = """\
import { use, useActionState, useOptimistic, Component } from 'react';

export class Boundary extends Component<{ children: React.ReactNode }, { e: boolean }> {
  state = { e: false };
  static getDerivedStateFromError() { return { e: true }; }
  render() { return this.state.e ? null : this.props.children; }
}

export function Form({ p }: { p: Promise<string> }) {
  const v = use(p);
  const [s, act] = useActionState(async (_: string, f: FormData) => String(f.get('x')), v);
  const [o] = useOptimistic(s);
  return <form action={act}>{o}</form>;
}
"""


def test_react19_rules():
    got = S.react19(LEGACY, "")["modern_flags"]
    want = ["class_component", "defaultProps", "effect_derived_state",
            "forwardRef", "legacy_render"]
    check(sorted(set(got)) == want, "each legacy idiom is flagged", str(got))
    m = S.react19(MODERN, "Use `useActionState` and `useOptimistic` for the form.")
    check(m["modern_flags"] == [], "an error boundary class is exempt; nothing "
          "modern is flagged", str(m["modern_flags"]))
    check(m["modern_used"] == ["use", "useActionState", "useOptimistic"]
          and m["modern_credits"] == ["useActionState", "useOptimistic"],
          "credits count only APIs the prompt names", str(m))
    fetchy = ("import { useEffect, useState } from 'react';\n"
              "export function F() { const [d, setD] = useState('');\n"
              "  useEffect(() => { fetch('/x').then(r => r.text()).then(setD); }, []);\n"
              "  return <p>{d}</p>; }\n")
    check("effect_derived_state" not in S.react19(fetchy, "")["modern_flags"],
          "an effect that fetches is not flagged as derived state")
    for tid in ("react-01", "react-05"):
        t = T.get(tid)
        if t:
            r = S.react19(read(grade._p(t["reference"])), t["prompt"])
            check(r["modern_flags"] == [], f"{tid} reference: no legacy flag",
                  str(r["modern_flags"]))


def test_skips():
    r = S.style(T["rs01"], fence("pub fn x() {}", "rust"))
    check(r["skipped"] and r["style_error"] is None, "a Rust task is skipped")
    r = S.style(T["ts01"], "I would write a function.")
    check(r["skipped"] == "no code extracted" and r["style_error"] is None,
          "prose is skipped, not an error")


def main() -> int:
    t0 = time.perf_counter()
    for fn in (test_sandbox, test_references, test_lint_can_fail, test_hooks_rule,
               test_react19_rules, test_skips):
        print(f"\n--- {fn.__name__} ---", flush=True)
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, nm, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + nm
                  + (f"\n        <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{passed}/{len(_results)} checks passed "
          f"in {time.perf_counter() - t0:.1f}s")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
