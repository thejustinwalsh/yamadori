#!/usr/bin/env python
"""THE PITFALL HARNESS: does a skill steer a model off its default pitfall and
onto the current best practice? Per model, per rendering, WITHOUT and WITH
the skill, n repeats, checked BY CODE.

    python bench/skills/pitfall_harness.py --list [--area react]
    python bench/skills/pitfall_harness.py --self-test          # offline
    python bench/skills/pitfall_harness.py --model bonsai --effort medium \\
        [--area react] [--repeats 3] [--variants list,table,...] [--dry-run]
    python bench/skills/pitfall_harness.py --model bonsai --report

THE OPERATOR (2026-09-29, verbatim): "skills are meant to improve the style,
accuracy, and provide knowledge that takes time to search for and is
learned through repeated failure. So you can not decide from our limited
set of data the value of a skill. If you want to understand the value of a
skill you need a test harness of pitfalls, good patterns, then if the skill
avoids the pitfall and uses the good pattern. The react docs site is the
best use of something like this right now." And: "Even something as simple
as recommending the react compiler, and where to get the info for setting
it up is a skill, we should be steering toward the react compiler by
default." "I think if we think of skills as best-practices and
auto steering we will build something good."

A CASE (bench/skills/pitfalls/<area>.jsonl, one per line):
  id, area, kind       "pitfall" (the default completion falls into it) or
                       "setup" (the project should end up on the best
                       practice: the React Compiler configured per the docs)
  prompt               a realistic task whose DEFAULT completion falls into
                       the pitfall; it never names the good pattern
  pitfall / good       rule names (RULES below), each a check BY CODE on the
                       answer's code blocks, parsed with tree-sitter (tsx);
                       config files (package.json, vite.config.*) by their
                       parser (json) or the same tsx tree
  skills               the library skills that teach the good pattern, by
                       NAME, in the order their items are composed; a case
                       whose skills are all missing from the armed library
                       is a GAP (reported: to be filled through the ONE
                       skills pipeline from the cited source, never by hand)
  source               {url, quote, licence}: the docs page and the VERBATIM
                       sentence the case comes from (react.dev: text CC BY
                       4.0, code MIT); the quotes are the ones the recipe
                       corpus already holds (bench/recipes/react_dev_*.jsonl
                       `evidence`, fetched from reactjs/react.dev)
  examples             {pitfall: code, good: code}: the rules' own test -- the
                       pitfall example must trip the pitfall rule and not the
                       good one, and the reverse (--self-test)

A RUN, per case: the same prompt WITHOUT and WITH the skill's items rendered
by mcp/skill_inject.render in each VARIANT (the per-model renderer's
choices: list, table, first_person as text, first_person_prefill as the
model's reasoning line), `repeats` seeds, the SAME seed on every side of a
repeat (skill_prove's pairing). Generations through mcp/model.py, thinking
capped at tiers.HELPER_THINKING (one bound for every model), answer
allowance tiers.A_MIN; each generation's finish_reason and reasoning size
are recorded (a rendering that only wins by thinking longer is visible).

THE SCORE, per case and side: pitfall avoided, good pattern used, and the
code still type-checks (mcp/typecheck.py against the held React version;
"not run" when no React types are held -- said so, never a pass). A skill
MOVES a case for a model when, paired by seed, WITH avoids the pitfall or
uses the good pattern on more repeats than WITHOUT and never type-checks
worse. A skill that does not move its case is REWRITTEN (per-model rendering
first) and re-measured -- never dropped (operator).

THE RENDERING CHOICE per model: leave-one-case-out -- the variant is picked
on the other cases (largest paired net gain) and scored on the held-out
case; pooled, the share of changed checks that got better, with its Wilson
95% interval; it SHIPS when the lower end is above 0.5.

ONE GPU CONSUMER AT A TIME: refuses unless --model is loaded (GET /running).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CASES_DIR = os.path.join(HERE, "pitfalls")
RESULTS = os.path.join(HERE, "inject", "results")
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)

HARNESS_VERSION = "pitfall-harness/1"
VARIANTS = ("list", "table", "first_person", "first_person_prefill")
# The system text every side runs under: multi-file answers, each file named
# (setup cases need package.json and vite.config.ts). Only the injection
# differs between sides. UNMEASURED WORDING (it is the same on every side).
SYSTEM = ("You are a coding assistant. Answer with the code. Put each file "
          "in its own fenced code block tagged with its language, with the "
          "file's name on the line just before the block, then at most a "
          "few sentences.")


# ------------------------------------------------------------------ code ---
def blocks(answer: str) -> list[dict]:
    """The answer's fenced blocks: {lang, code, name} (name: the file name
    on the line before the fence, when there is one)."""
    out = []
    lines = (answer or "").split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*```\s*([\w+#.-]*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        prev = lines[i - 1].strip() if i else ""
        nm = re.search(r"([\w./-]+\.(?:tsx?|jsx?|json|mjs|cjs))", prev)
        j = i + 1
        body = []
        while j < len(lines) and not re.match(r"^\s*```\s*$", lines[j]):
            body.append(lines[j])
            j += 1
        out.append({"lang": m.group(1).lower(), "code": "\n".join(body),
                    "name": nm.group(1) if nm else ""})
        i = j + 1
    return out


class Code:
    """The answer's code as rules read it: every JS/TS block parsed as tsx
    (tree-sitter), and the JSON blocks parsed as JSON."""

    def __init__(self, answer: str):
        from tree_sitter_language_pack import get_parser
        self.parser = get_parser("tsx")
        self.blocks = blocks(answer)
        self.trees = []
        self.json = []
        for b in self.blocks:
            if b["lang"] == "json" or b["name"].endswith(".json"):
                try:
                    self.json.append((b, json.loads(b["code"])))
                except ValueError:
                    pass
                continue
            if b["lang"] in ("", "ts", "tsx", "typescript", "js", "jsx",
                             "javascript", "mjs") or re.search(
                    r"\.(tsx?|jsx?|mjs|cjs)$", b["name"]):
                src = b["code"].encode("utf-8")
                self.trees.append((b, src, self.parser.parse(src)))

    # -- walking
    def nodes(self, types=None):
        for b, src, t in self.trees:
            stack = [t.root_node]
            while stack:
                n = stack.pop()
                if types is None or n.type in types:
                    yield b, src, n
                stack.extend(reversed(n.children))

    @staticmethod
    def text(src: bytes, n) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def callee(self, src, call) -> str:
        f = call.child_by_field_name("function")
        return self.text(src, f) if f is not None else ""

    def calls(self, name: str):
        """call_expression nodes whose callee is `name` or `X.name`."""
        for b, src, n in self.nodes({"call_expression"}):
            c = self.callee(src, n)
            if c == name or c.endswith("." + name):
                yield b, src, n

    def called(self, name: str) -> bool:
        return any(True for _ in self.calls(name))

    def effect_bodies(self):
        """(src, callback node) of every useEffect / useLayoutEffect."""
        for name in ("useEffect", "useLayoutEffect"):
            for b, src, n in self.calls(name):
                args = n.child_by_field_name("arguments")
                cb = args.named_children[0] if args is not None and \
                    args.named_children else None
                if cb is not None:
                    yield src, cb, n

    @staticmethod
    def within(src, node, types) -> list:
        out, stack = [], [node]
        while stack:
            x = stack.pop()
            if x.type in types:
                out.append(x)
            stack.extend(x.children)
        return out

    def jsx_names(self) -> list[str]:
        out = []
        for b, src, n in self.nodes({"jsx_opening_element",
                                     "jsx_self_closing_element"}):
            nm = n.child_by_field_name("name")
            if nm is not None:
                out.append(self.text(src, nm))
        return out

    def jsx_attrs(self, element: str | None = None) -> list[tuple]:
        """(element name, attribute name, value text)."""
        out = []
        for b, src, n in self.nodes({"jsx_opening_element",
                                     "jsx_self_closing_element"}):
            nm = n.child_by_field_name("name")
            en = self.text(src, nm) if nm is not None else ""
            if element and en != element:
                continue
            for a in n.children:
                if a.type == "jsx_attribute" and a.named_children:
                    an = self.text(src, a.named_children[0])
                    val = self.text(src, a.named_children[1]) \
                        if len(a.named_children) > 1 else ""
                    out.append((en, an, val))
        return out

    def all_code(self) -> str:
        return "\n".join(self.text(src, t.root_node)
                         for _b, src, t in self.trees)


# ----------------------------------------------------------------- rules ---
# Each rule: Code -> bool. Named, so a case file names its checks.
def r_ref_current_assigned(c: Code) -> bool:
    """A `.current` member assigned a value -- the latest-value ref habit
    (a callback or prop mirrored into a ref)."""
    for b, src, n in c.nodes({"assignment_expression"}):
        left = n.child_by_field_name("left")
        if left is not None and left.type == "member_expression" and \
                c.text(src, left).endswith(".current"):
            right = n.child_by_field_name("right")
            rt = c.text(src, right) if right is not None else ""
            if not re.match(r"^(null|undefined|true|false|\d|['\"`])", rt):
                return True
    return False


def r_ref_callback_called(c: Code) -> bool:
    """A ref's `.current` called as a function (`x.current(...)`,
    `x.current?.(...)`)."""
    for b, src, n in c.nodes({"call_expression"}):
        cal = c.callee(src, n)
        if re.search(r"\.current\??\.?$", cal) or cal.endswith(".current"):
            return True
    return False


def r_latest_ref(c: Code) -> bool:
    return r_ref_current_assigned(c) and r_ref_callback_called(c)


def r_exhaustive_deps_suppressed(c: Code) -> bool:
    for b, src, n in c.nodes({"comment"}):
        if "exhaustive-deps" in c.text(src, n):
            return True
    return False


def r_latest_ref_or_suppressed(c: Code) -> bool:
    return r_latest_ref(c) or r_exhaustive_deps_suppressed(c)


def r_uses_effect_event(c: Code) -> bool:
    return c.called("useEffectEvent")


_SIDE = ("fetch", "addEventListener", "subscribe", "setInterval",
         "setTimeout", "connect", "createConnection", "observe", "open")


def r_effect_only_sets_state(c: Code) -> bool:
    """An Effect whose body calls a state setter and nothing that talks to
    the outside world -- state derived or reset in an Effect ("You Might Not
    Need an Effect")."""
    for src, cb, _n in c.effect_bodies():
        calls = c.within(src, cb, {"call_expression"})
        names = [c.callee(src, x) for x in calls]
        sets = [x for x in names if re.match(r"^set[A-Z]", x)]
        side = [x for x in names if x.split(".")[-1] in _SIDE]
        if sets and not side:
            return True
    return False


def r_no_effect(c: Code) -> bool:
    return not any(True for _ in c.effect_bodies())


def r_fetch_in_effect(c: Code) -> bool:
    for src, cb, _n in c.effect_bodies():
        if any(c.callee(src, x).split(".")[-1] in ("fetch", "post", "send",
                                                   "sendMessage")
               for x in c.within(src, cb, {"call_expression"})):
            return True
    return False


def r_fetch_in_effect_no_cleanup(c: Code) -> bool:
    """A fetch in an Effect whose callback returns no cleanup function."""
    for src, cb, _n in c.effect_bodies():
        calls = [c.callee(src, x) for x in c.within(src, cb,
                                                    {"call_expression"})]
        if not any(x.split(".")[-1] == "fetch" for x in calls):
            continue
        body = cb.child_by_field_name("body")
        rets = c.within(src, body, {"return_statement"}) if body is not None \
            else []
        cleanup = any(r.named_children and r.named_children[0].type in (
            "arrow_function", "function_expression", "function", "identifier")
            for r in rets)
        if not cleanup:
            return True
    return False


def r_effect_cleanup_or_use(c: Code) -> bool:
    return not r_fetch_in_effect_no_cleanup(c) and (
        r_fetch_in_effect(c) or c.called("use"))


def r_key_reset(c: Code) -> bool:
    """A `key` attribute on a component element (a reset by identity)."""
    return any(a == "key" and en[:1].isupper() for en, a, _v in c.jsx_attrs())


def r_uses_use_hook(c: Code) -> bool:
    return any(True for _b, src, n in c.calls("use")
               if c.callee(src, n) in ("use", "React.use"))


def r_use_of_render_promise(c: Code) -> bool:
    """`use(f(...))`: a promise created in render and handed to `use`."""
    for b, src, n in c.calls("use"):
        if c.callee(src, n) not in ("use", "React.use"):
            continue
        args = n.child_by_field_name("arguments")
        a0 = args.named_children[0] if args is not None and \
            args.named_children else None
        if a0 is not None and a0.type == "call_expression":
            return True
    return False


def r_effect_fetch_state(c: Code) -> bool:
    return r_fetch_in_effect(c) or r_use_of_render_promise(c)


def r_use_with_suspense(c: Code) -> bool:
    return r_uses_use_hook(c) and not r_use_of_render_promise(c) and \
        "Suspense" in c.jsx_names()


def r_forward_ref(c: Code) -> bool:
    return c.called("forwardRef")


def r_ref_as_prop(c: Code) -> bool:
    if r_forward_ref(c):
        return False
    for b, src, n in c.nodes({"object_pattern"}):
        t = c.text(src, n)
        if re.search(r"(^|[{,\s])ref\s*[,}:=]", t):
            return True
    return any(a == "ref" for _e, a, _v in c.jsx_attrs())


def r_context_provider(c: Code) -> bool:
    return any(n.endswith(".Provider") for n in c.jsx_names())


def r_context_as_provider(c: Code) -> bool:
    names = c.jsx_names()
    return not r_context_provider(c) and any(
        n.endswith("Context") and n[:1].isupper() for n in names)


def r_manual_submit(c: Code) -> bool:
    """onSubmit + preventDefault + a pending/loading useState."""
    code = c.all_code()
    return c.called("preventDefault") and bool(re.search(
        r"useState[^\n]*\n?|\[\s*\w*(?:[Pp]ending|[Ll]oading|[Ss]ubmitting)",
        code)) and bool(re.search(r"\[\s*\w*(?:[Pp]ending|[Ll]oading|"
                                  r"[Ss]ubmitting)\w*\s*,\s*set", code))


def r_action_state(c: Code) -> bool:
    return c.called("useActionState") or any(
        en == "form" and a == "action" for en, a, _v in c.jsx_attrs())


def r_manual_rollback(c: Code) -> bool:
    """A like/count updated first and undone in a catch: a hand-rolled
    optimistic update."""
    for b, src, n in c.nodes({"catch_clause"}):
        if any(re.match(r"^set[A-Z]", c.callee(src, x)) for x in
               c.within(src, n, {"call_expression"})):
            return True
    return False


def r_uses_optimistic(c: Code) -> bool:
    return c.called("useOptimistic")


def r_no_transition(c: Code) -> bool:
    return not (c.called("useTransition") or c.called("startTransition")
                or c.called("useDeferredValue"))


def r_transition(c: Code) -> bool:
    return not r_no_transition(c)


def r_manual_memo(c: Code) -> bool:
    return c.called("useMemo") or c.called("useCallback") or \
        c.called("memo")


def r_no_manual_memo(c: Code) -> bool:
    return not r_manual_memo(c)


def _vite_trees(c: Code):
    for b, src, t in c.trees:
        if "vite.config" in b["name"] or re.search(
                r"\bdefineConfig\b", c.text(src, t.root_node)):
            yield b, src, t


def r_compiler_configured(c: Code) -> bool:
    """The React Compiler on per the docs: the Vite config passes
    babel-plugin-react-compiler to @vitejs/plugin-react's babel plugins, and
    package.json lists the plugin."""
    in_vite = False
    for b, src, t in _vite_trees(c):
        for n in Code.within(src, t.root_node, {"string"}):
            if "babel-plugin-react-compiler" in Code.text(src, n):
                in_vite = True
    in_pkg = any("babel-plugin-react-compiler" in json.dumps(j)
                 for _b, j in c.json)
    return in_vite and in_pkg


def r_compiler_missing(c: Code) -> bool:
    return not r_compiler_configured(c)


RULES = {k[2:]: v for k, v in dict(globals()).items()
         if k.startswith("r_") and callable(v)}


# ----------------------------------------------------------------- cases ---
def load_cases(area: str | None = None) -> list[dict]:
    out = []
    for fn in sorted(os.listdir(CASES_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(CASES_DIR, fn), encoding="utf-8") as f:
            for ln in f:
                if ln.strip() and not ln.lstrip().startswith("//"):
                    c = json.loads(ln)
                    if area is None or c["area"] == area:
                        out.append(c)
    return out


def check_answer(case: dict, answer: str) -> dict:
    code = Code(answer or "")
    return {"blocks": len(code.blocks),
            "pitfall": any(RULES[r](code) for r in case["pitfall"]),
            "good": all(RULES[r](code) for r in case["good"])}


def self_test() -> dict:
    """Every case's rules against its own examples: the pitfall example
    trips a pitfall rule and not every good rule; the good example trips no
    pitfall rule and every good rule."""
    res = []
    for case in load_cases():
        ex = case.get("examples") or {}
        bad = check_answer(case, "```tsx\n" + ex.get("pitfall", "") + "\n```"
                           if not ex.get("pitfall", "").startswith("```")
                           else ex["pitfall"])
        good = check_answer(case, "```tsx\n" + ex.get("good", "") + "\n```"
                            if not ex.get("good", "").startswith("```")
                            else ex["good"])
        ok = bad["pitfall"] and not bad["good"] and not good["pitfall"] \
            and good["good"]
        res.append({"id": case["id"], "ok": ok, "pitfall_example": bad,
                    "good_example": good})
    return {"cases": len(res), "ok": sum(r["ok"] for r in res),
            "failed": [r for r in res if not r["ok"]]}


def _skill_ref(x) -> tuple[str, list[int] | None]:
    """A case's skill entry: a name (every item) or {name, items: [index]}
    (only those items: a skill whose other lines are about something else
    injects only the lines the case is about)."""
    if isinstance(x, dict):
        return x["name"], list(x.get("items") or []) or None
    return str(x), None


def library_skills(refs: list) -> tuple[list[dict], list[str]]:
    """(armed skill rows, in order, each cut to the items the case names;
    the names not armed)."""
    import skills
    by = {s.get("name"): s for s in skills.armed()}
    got, missing = [], []
    for x in refs:
        name, idx = _skill_ref(x)
        s = by.get(name)
        if s is None:
            missing.append(name)
            continue
        if idx is not None:
            import skill_select
            its = skill_select._items(s)
            s = dict(s, items=[its[i] for i in idx if i < len(its)])
        got.append(s)
    return got, missing


def rendering(case: dict, variant: str, model: str) -> dict:
    import skill_inject as I
    rows, _missing = library_skills(case.get("skills") or [])
    items, _left = I.candidates(rows)
    prof = dict(I.profile_for(model))
    prof.update(format="table" if variant == "table" else "list",
                voice={"first_person": "first_person",
                       "first_person_prefill": "first_person_prefill"}.get(
                           variant, "note"))
    chosen = I.compose(items, [it["key"] for it in items], prof)
    r = I.render(chosen, prof)
    r["items"] = [it["key"] for it in chosen]
    return r


# ----------------------------------------------------------------- runs ----
def seed_of(case_id: str, k: int) -> int:
    return int(hashlib.sha256(f"{case_id}#{k}".encode()).hexdigest()[:8],
               16) & 0x7FFFFFFF


def body(case: dict, r: dict | None, model: str, effort: str,
         seed: int) -> dict:
    import model as M
    import tiers
    user = case["prompt"] + ((r or {}).get("text") or "")
    b = M.shape({"messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                 "max_tokens": tiers.A_MIN}, effort=effort, role="helper",
                step_cap=tiers.HELPER_THINKING)
    b.pop("_share", None)
    b["model"] = model
    b["seed"] = seed
    if r and r.get("prefill"):
        b["messages"].append({"role": "assistant", "content": "",
                              "reasoning_content": r["prefill"]})
    return b


def generate(b: dict) -> tuple[str | None, dict]:
    import model as M
    t0 = time.time()
    try:
        d = M.post(b)
    except Exception as e:                                       # noqa: BLE001
        return None, {"why": f"not run: {type(e).__name__}: {e}"[:200],
                      "s": round(time.time() - t0, 2)}
    ch = ((d.get("choices") or [{}])[0]) if isinstance(d, dict) else {}
    msg = ch.get("message") or {}
    usage = d.get("usage") or {}
    gen = {"finish_reason": ch.get("finish_reason"),
           "completion_tokens": usage.get("completion_tokens"),
           "reasoning_chars": len(msg.get("reasoning_content") or ""),
           "s": round(time.time() - t0, 2)}
    try:
        return M.answer(d), gen
    except M.BudgetEvent as e:
        gen["why"] = f"budget event: {e}"[:200]
        return None, gen


def types_ok(case: dict, answer: str) -> bool | None:
    import skill_prove as SP
    ok, _why = SP.run_check({"kind": "types"}, answer, "tsx",
                            case.get("packages") or ["react"])
    return ok


def run_case(case: dict, model: str, effort: str, repeats: int,
             variants) -> dict:
    rec = {"v": HARNESS_VERSION, "model": model, "effort": effort,
           "case": case["id"], "area": case["area"], "kind": case["kind"],
           "sides": {}}
    _rows, missing = library_skills(case.get("skills") or [])
    rec["skills_missing"] = missing
    for side in ("without",) + tuple(variants):
        r = None if side == "without" else rendering(case, side, model)
        if r is not None and not (r.get("text") or r.get("prefill")):
            rec["sides"][side] = {"not_run": "no armed skill item to "
                                  "inject (a GAP)"}
            continue
        reps = []
        for k in range(repeats):
            seed = seed_of(case["id"], k)
            ans, gen = generate(body(case, r, model, effort, seed))
            chk = check_answer(case, ans) if ans is not None else None
            reps.append({"seed": seed, "answered": ans is not None,
                         "gen": gen, **(chk or {}),
                         "types": types_ok(case, ans) if ans else None,
                         "answer_sha": hashlib.sha1((ans or "").encode(
                             "utf-8")).hexdigest()[:12]})
        rec["sides"][side] = {"repeats": reps,
                              "items": (r or {}).get("items"),
                              "tokens": (r or {}).get("tokens")}
    return rec


def _wilson(k: int, n: int, z: float = 1.959964) -> list:
    if not n:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    mid = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((mid - half) / den, 4), round((mid + half) / den, 4)]


def paired(rec: dict, variant: str) -> collections.Counter:
    """better / worse / same over the case's repeats, paired by seed, on
    each of pitfall-avoided, good-used and types."""
    c = collections.Counter()
    w = (rec["sides"].get("without") or {}).get("repeats") or []
    v = (rec["sides"].get(variant) or {}).get("repeats") or []
    for a, b in zip(w, v):
        if not (a.get("answered") and b.get("answered")):
            c["not_run"] += 1
            continue
        for key, good_is in (("pitfall", False), ("good", True),
                             ("types", True)):
            x, y = a.get(key), b.get(key)
            if x is None or y is None:
                continue
            gx, gy = (x == good_is), (y == good_is)
            c["better" if gy and not gx else "worse" if gx and not gy
              else "same"] += 1
    return c


def report(recs: list[dict]) -> dict:
    variants = sorted({v for r in recs for v in r["sides"]
                       if v != "without"})
    cases = {}
    for r in recs:
        w = (r["sides"].get("without") or {}).get("repeats") or []
        row = {"skills_missing": r.get("skills_missing"),
               "without": {"pitfall": sum(bool(x.get("pitfall")) for x in w),
                           "good": sum(bool(x.get("good")) for x in w),
                           "n": len(w)}}
        for v in variants:
            s = r["sides"].get(v) or {}
            reps = s.get("repeats") or []
            p = paired(r, v)
            row[v] = {"pitfall": sum(bool(x.get("pitfall")) for x in reps),
                      "good": sum(bool(x.get("good")) for x in reps),
                      "n": len(reps), "better": p["better"],
                      "worse": p["worse"],
                      "moved": p["better"] > 0 and p["worse"] == 0,
                      "reasoning_chars_mean": round(sum(
                          (x.get("gen") or {}).get("reasoning_chars") or 0
                          for x in reps) / max(len(reps), 1))}
            if s.get("not_run"):
                row[v]["not_run"] = s["not_run"]
        cases[r["case"]] = row
    # leave-one-case-out rendering choice
    better = worse = 0
    picks = collections.Counter()
    for held in recs:
        tally = collections.Counter()
        for r in recs:
            if r is held:
                continue
            for v in variants:
                p = paired(r, v)
                tally[v] += p["better"] - p["worse"]
        pick = max(variants, key=lambda v: (tally[v], v == "list")) \
            if variants else None
        picks[str(pick)] += 1
        if pick:
            p = paired(held, pick)
            better += p["better"]
            worse += p["worse"]
    ci = _wilson(better, better + worse)
    return {"cases": cases, "gaps": sorted(k for k, r in cases.items()
                                           if r.get("skills_missing")),
            "held_out_rendering": {"picks": dict(picks), "better": better,
                                   "worse": worse,
                                   "share_better_ci95": ci,
                                   "ships": ci[0] is not None
                                   and ci[0] > 0.5}}


def running(upstream: str) -> list[str]:
    with urllib.request.urlopen(f"{upstream}/running", timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = d.get("running") if isinstance(d, dict) else d
    return [str((x or {}).get("model")) for x in rows or []
            if isinstance(x, dict)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--area")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    cases = load_cases(a.area)
    if a.self_test:
        print(json.dumps(self_test(), indent=1))
        return 0
    if a.list:
        import replay_selection  # noqa: F401  (store isolation)
        for c in cases:
            got, missing = library_skills(c.get("skills") or [])
            print(f"{c['id']:<34} {c['kind']:<8} skills {len(got)}"
                  + (f"  MISSING {missing}" if missing else ""))
        return 0
    if not a.model:
        ap.error("--model is required to run or report")
    out = os.path.join(RESULTS, f"pitfall_{a.model}.jsonl")
    if a.report:
        with open(out, encoding="utf-8") as f:
            recs = [json.loads(ln) for ln in f if ln.strip()]
        print(json.dumps(report(recs), indent=1))
        return 0
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    if a.dry_run:
        print(json.dumps({"cases": len(cases), "variants": variants,
                          "repeats": a.repeats,
                          "generations": len(cases) * a.repeats
                          * (1 + len(variants))}))
        return 0
    import replay_selection  # noqa: F401  (store isolation)
    import model as M
    have = running(M.UPSTREAM)
    if a.model not in have:
        print(f"REFUSED: {a.model} is not loaded (GET /running: {have})")
        return 2
    import cancel
    import max_mode
    os.makedirs(RESULTS, exist_ok=True)
    with cancel.bound(cancel.Token()):
        max_mode.set_current(a.model)
        for c in cases:
            rec = run_case(c, a.model, a.effort, a.repeats, variants)
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {c['id']}: done", flush=True)
    with open(out, encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    print(json.dumps(report(recs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
