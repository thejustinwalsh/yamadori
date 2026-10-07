#!/usr/bin/env python
"""The text matcher's VERSION gate and NEGATION gate (mcp/skill_classify.py
request_signals / match). No GPU, no network, no live store.

    python mcp/test_matcher_gates.py      -> "N/M checks passed"

Coordinator, 2026-10-07 (operator: "Lets cook the features ... gather
evidence"): the activation tests of the gap-fill skills failed on cases the
matcher could not tell apart -- R3F v9 against a v10 skill, "plain three.js
(no React)" against a React skill. What this gates, from those cases:

  1. VERSION: a stated major different from the one the skill is about
     excludes it ("React Three Fiber v9" / a v10 skill, "React 17" / a
     React 19 skill, a pinned dependency); the same major, no stated
     version, and a skill that names two majors of a name are untouched.
  2. NEGATION: a name right behind a negation ("no React", "without R3F",
     "no React, no React Three Fiber") is no evidence FOR it and rules out
     the skills gated on it; so does a "vanilla" / "plain" / "standalone"
     stack for the framework layers (React, R3F). "not" far from the name
     in the same sentence is NOT a negation of it.
  3. IMPLICATION: a request naming React Three Fiber meets a "needs React"
     gate.
  4. NOTHING WORSE: every activation case of every skill in the checked-in
     fixtures still passes (the daily eval, test_skills and the whole
     library's activation tests are run beside this suite).
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_matcher_gates_")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import skill_classify as C  # noqa: E402

RESULTS: list[tuple[bool, str, object]] = []


def check(ok: bool, what: str, detail=None) -> None:
    RESULTS.append((bool(ok), what, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}"
          + ("" if ok or detail is None else f"\n        {detail!r}"[:600]),
          flush=True)


def verdict(rule: dict, text: str, route="code_generation") -> dict:
    sig = C.request_signals([{"role": "user", "content": text}], route)
    return C.match(rule, sig)


def rule(frameworks, languages=("typescript",), text="", all_of=(),
         topics=()) -> dict:
    r = C.rule_from_metadata({"frameworks": list(frameworks),
                              "languages": list(languages),
                              "all_of": list(all_of),
                              "topics": list(topics)})
    r["text"] = text
    return r


R3F_V10 = rule(["r3f"], text="Writing React Three Fiber v10 useFrame "
               "callbacks, or migrating v9 FrameState.clock usage",
               all_of=["react"], topics=["useFrame", "state.clock"])
R3F_ANY = rule(["r3f"], text="Writing React Three Fiber useFrame callbacks",
               all_of=["react"], topics=["useFrame", "state.clock"])
REACT_19 = rule(["react"], ("javascript",), text="Using React 19 ref as a "
                "prop in function components", topics=["forwardRef"])
RT = ("\nI use useFrame and state.clock.getDelta() in my callback "
      "(state.clock.elapsedTime too).")


def test_the_version_gate() -> None:
    check(C.rule_versions(R3F_V10) == {"r3f": 10}
          and C.rule_versions(R3F_ANY) == {}
          and C.rule_versions(REACT_19) == {"react": 19},
          "[version] the major a skill is about is the one its text names "
          "right after a framework", [C.rule_versions(r) for r in (
              R3F_V10, R3F_ANY, REACT_19)])
    both = rule(["r3f"], text="React Three Fiber v9 to React Three Fiber "
                "v10 migration", all_of=["react"], topics=["useFrame"])
    check(C.rule_versions(both) == {},
          "[version] a skill that names two majors of one name is about the "
          "move: no version gate", C.rule_versions(both))
    v9 = verdict(R3F_V10, "I'm on React Three Fiber v9 and need to debug a "
                 "useFrame callback." + RT)
    check(v9["strength"] is None and "v9" not in "" and "10" in (
        v9.get("gate") or "") and "9" in (v9.get("gate") or ""),
          "[version] a stated v9 excludes the v10 skill, the gate says "
          "which", v9)
    v10 = verdict(R3F_V10, "I'm on React Three Fiber v10 and need to debug "
                  "a useFrame callback." + RT)
    none = verdict(R3F_V10, "I need to debug a React Three Fiber useFrame "
                   "callback." + RT)
    check(v10["strength"] is not None and none["strength"] is not None,
          "[version] the same major, or none stated, is untouched",
          (v10, none))
    any9 = verdict(R3F_ANY, "I'm on React Three Fiber v9 and need to debug "
                   "a useFrame callback." + RT)
    check(any9["strength"] is not None,
          "[version] a skill with no major takes any stated one", any9)
    r17 = verdict(REACT_19, "I maintain a legacy React 17 app and wonder "
                  "about forwardRef.")
    r19 = verdict(REACT_19, "In React 19 I wonder about forwardRef.")
    check(r17["strength"] is None and r19["strength"] is not None,
          "[version] React 17 does not meet a React 19 skill; 19 does",
          (r17, r19))
    pinned = ('Here is package.json:\n{"dependencies": '
              '{"@react-three/fiber": "^9.8.0"}}\nAnd now useFrame '
              'state.clock in a TypeScript component, import { useFrame } '
              'from "@react-three/fiber";')
    vp = verdict(R3F_V10, pinned)
    check(vp["strength"] is None and "10" in (vp.get("gate") or ""),
          "[version] a pinned dependency states the major too", vp)


def test_negation() -> None:
    plain = verdict(R3F_ANY, "I'm building a standalone Three.js web page "
                    "in plain HTML and JavaScript (no React, no React Three "
                    "Fiber). I want to read useFrame-like deltas." + RT)
    check(plain["strength"] is None and "rules out" in (
        plain.get("gate") or ""),
          "[negation] \"no React, no React Three Fiber\" rules out the R3F "
          "skill", plain)
    sig = C.request_signals([{"role": "user", "content": "A plain three.js "
                              "scene (no React, no R3F)."}], None)
    check("react" not in sig["terms"] and "r3f" not in sig["terms"]
          and "threejs" in sig["terms"]
          and {"react", "r3f"} <= set(sig["ruled_out"]),
          "[negation] the negated names are no evidence for themselves "
          "(three.js still is) and are recorded as ruled out", (
              sorted(sig["terms"]), sig["ruled_out"]))
    wo = C.ruled_out("Build it without R3F or koota.")
    check({"r3f", "koota"} <= wo,
          "[negation] \"without R3F or koota\" rules out both", wo)
    vanilla = C.ruled_out("I wrote a vanilla JavaScript widget for a custom "
                          "element.")
    standalone = C.ruled_out("A standalone Three.js page that spins a cube.")
    check({"react", "r3f"} <= vanilla and {"react", "r3f"} <= standalone,
          "[negation] a vanilla / plain / standalone stack rules out the "
          "React and R3F layers", (vanilla, standalone))
    named = C.ruled_out("A vanilla JavaScript page, and I want to add React "
                        "to it.")
    check("react" not in named, "[negation] a layer the request plainly "
          "asks for is not ruled out", named)
    far = verdict(R3F_ANY, "A memoised function is not hitting the cache "
                  "despite being called with the same arguments, in my "
                  "React Three Fiber scene." + RT)
    check(far["strength"] is not None,
          "[negation] a \"not\" earlier in the sentence, far from the name, "
          "is no negation of it", far)
    react_skill = rule(["react"], ("javascript",), text="Using React hooks")
    nr = verdict(react_skill, "A vanilla JavaScript widget with no React.")
    check(nr["strength"] is None,
          "[negation] \"no React\" rules out a React skill", nr)


def test_implication() -> None:
    needs_react = rule(["threejs"], ("javascript",), text="three.js scenes "
                       "in React", all_of=["react"], topics=["useFrame"])
    r = verdict(needs_react, "I'm planning a React Three Fiber landscape "
                "with three.js and a useFrame loop." + RT)
    check(r["strength"] is not None,
          "[implication] a request naming React Three Fiber meets a "
          "\"needs React\" gate", r)
    r2 = verdict(needs_react, "A three.js scene with a useFrame loop in "
                 "scene.js." + RT)
    check(r2["strength"] is None and "needs React" in (r2.get("gate") or ""),
          "[implication] a three.js request without React still does not",
          r2)


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
