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
                       sentence the case comes from (react.dev: CC BY 4.0,
                       its LICENSE-DOCS.md); the quotes are the ones the recipe
                       corpus already holds (bench/recipes/react_dev_*.jsonl
                       `evidence`, fetched from reactjs/react.dev)
  examples             {pitfall: code, good: code}: the rules' own test -- the
                       pitfall example must trip the pitfall rule and not the
                       good one, and the reverse (--self-test)
  THE AREAS PAST REACT (operator, 2026-09-30: "Yes to everything pending,
  queue it up lets go"; r3f v10 + three, TSL, TypeGPU, koota, pmndrs math,
  TypeScript, Rust/WASM) add:
  never_in_prompt      the good pattern's names; the test asserts none is in
                       the prompt
  source.file          the PINNED copy (pitfalls/sources/<slug>/, its
                       MANIFEST.json: commit or fetch date, sha256, licence);
                       the quote must be a verbatim substring of it
  packages             the held npm packages tsc installs ([] = plain TS)
  typecheck "none"     + typecheck_why, where tsc cannot apply (Rust)
  example_lang         the fence a bare example is wrapped in (default tsx)
  Their rules live in bench/skills/pitfall_rules/<area>.py (loaded into
  RULES; a duplicate name is an error); their gaps in pitfalls/gaps/<area>
  .json, filled through the ONE pipeline by bench/skills/pitfall_gap_fill.py
  in a GPU window. Code reads Rust (tree-sitter rust), TOML, C headers and
  WGSL/HTML/shell text besides TS/JSON.

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
        nm = re.search(r"([\w./-]+\.(?:json|tsx|ts|jsx|js|mjs|cjs|rs|toml"
                       r"|wgsl|html|h|hpp|c))\b", prev)
        j = i + 1
        body = []
        while j < len(lines) and not re.match(r"^\s*```\s*$", lines[j]):
            body.append(lines[j])
            j += 1
        out.append({"lang": m.group(1).lower(), "code": "\n".join(body),
                    "name": nm.group(1) if nm else ""})
        i = j + 1
    return out


RUST_LANGS = ("rust", "rs")
TOML_LANGS = ("toml",)
C_LANGS = ("c", "h", "cpp", "hpp", "c++")
RAW_LANGS = ("wgsl", "html", "sh", "bash", "shell", "console")


class Code:
    """The answer's code as rules read it: every JS/TS block parsed as tsx
    (tree-sitter), the JSON blocks parsed as JSON; since 2026-09-30 (the
    pitfall areas past React) Rust blocks parsed with tree-sitter's rust
    grammar (`rust`), TOML blocks with tomllib (`toml`: Cargo.toml,
    cbindgen.toml), C headers with tree-sitter's c grammar (`c`), and WGSL,
    HTML and shell blocks kept as text (`raw`)."""

    def __init__(self, answer: str):
        from tree_sitter_language_pack import get_parser
        self.parser = get_parser("tsx")
        self.blocks = blocks(answer)
        self.trees = []
        self.json = []
        self.rust = []
        self.toml = []
        self.c = []
        self.raw = []
        for b in self.blocks:
            if b["lang"] == "json" or b["name"].endswith(".json"):
                try:
                    self.json.append((b, json.loads(b["code"])))
                except ValueError:
                    pass
                continue
            if b["lang"] in RUST_LANGS or b["name"].endswith(".rs"):
                src = b["code"].encode("utf-8")
                self.rust.append((b, src, get_parser("rust").parse(src)))
                continue
            if b["lang"] in TOML_LANGS or b["name"].endswith(".toml"):
                import tomllib
                try:
                    self.toml.append((b, tomllib.loads(b["code"])))
                except (ValueError, tomllib.TOMLDecodeError):
                    pass
                continue
            if b["lang"] in C_LANGS or re.search(r"\.(h|hpp|c)$", b["name"]):
                src = b["code"].encode("utf-8")
                self.c.append((b, src, get_parser("c").parse(src)))
                continue
            if b["lang"] in RAW_LANGS or re.search(r"\.(wgsl|html)$",
                                                   b["name"]):
                self.raw.append(b)
                continue
            if b["lang"] in ("", "ts", "tsx", "typescript", "js", "jsx",
                             "javascript", "mjs") or re.search(
                    r"\.(tsx?|jsx?|mjs|cjs)$", b["name"]):
                src = b["code"].encode("utf-8")
                self.trees.append((b, src, self.parser.parse(src)))

    # -- walking
    def nodes(self, types=None, lang: str = "ts"):
        """Every node of the TS/JS trees (lang "ts"), the Rust trees
        ("rust") or the C trees ("c"), optionally of the given types."""
        trees = {"ts": self.trees, "rust": self.rust, "c": self.c}[lang]
        for b, src, t in trees:
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

    def all_code(self, lang: str = "ts") -> str:
        trees = {"ts": self.trees, "rust": self.rust, "c": self.c}[lang]
        return "\n".join(self.text(src, t.root_node)
                         for _b, src, t in trees)

    def files(self, suffix: str) -> list[dict]:
        """The blocks whose file name ends with `suffix` (e.g. "Cargo.toml",
        "tsconfig.json", ".wgsl")."""
        return [b for b in self.blocks if b["name"].endswith(suffix)]

    def toml_of(self, suffix: str) -> list[dict]:
        """Parsed TOML of the blocks named ...`suffix` (e.g. "Cargo.toml");
        an unnamed TOML block counts when it is the only one."""
        named = [d for b, d in self.toml if b["name"].endswith(suffix)]
        if named:
            return named
        unnamed = [d for b, d in self.toml if not b["name"]]
        return unnamed if len(self.toml) == 1 else []

    def json_of(self, suffix: str) -> list:
        """Parsed JSON of the blocks named ...`suffix` (e.g.
        "tsconfig.json"); tsconfig allows comments and trailing commas, so a
        block that JSON rejects is read again with those removed."""
        out = [d for b, d in self.json if b["name"].endswith(suffix)]
        for b in self.blocks:
            if b["name"].endswith(suffix) and not any(
                    b is x for x, _d in self.json):
                t = re.sub(r"//[^\n]*|/\*.*?\*/", "", b["code"], flags=re.S)
                t = re.sub(r",\s*([}\]])", r"\1", t)
                try:
                    out.append(json.loads(t))
                except ValueError:
                    pass
        return out


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


_IO_CALL = re.compile(r"^(fetch|load|post|send|sendMessage|axios)\w*$")


def _io_calls(c: Code, src, node) -> bool:
    """A call in `node` whose callee has a segment naming network IO
    (fetch..., load..., post, send...)."""
    for x in c.within(src, node, {"call_expression"}):
        segs = re.split(r"[.(]", c.callee(src, x))
        if any(_IO_CALL.match(g) for g in segs if g):
            return True
    return False


def r_fetch_in_effect(c: Code) -> bool:
    return any(_io_calls(c, src, cb) for src, cb, _n in c.effect_bodies())


def r_fetch_in_effect_no_cleanup(c: Code) -> bool:
    """A fetch in an Effect whose callback returns no cleanup function."""
    for src, cb, _n in c.effect_bodies():
        if not _io_calls(c, src, cb):
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
    """The React Compiler on per react.dev's installation page (pinned
    75ef18a, learn/react-compiler/installation.md): package.json lists
    `babel-plugin-react-compiler` ("Install React Compiler as a
    devDependency") AND the Vite config turns it on in one of the page's
    three forms -- `reactCompilerPreset()` from @vitejs/plugin-react >= 6
    with @rolldown/plugin-babel, the Babel plugin by name in
    @rolldown/plugin-babel's plugins, or (plugin-react < 6) the inline
    `babel: { plugins: ['babel-plugin-react-compiler'] }`."""
    in_vite = False
    for b, src, t in _vite_trees(c):
        for n in Code.within(src, t.root_node, {"string"}):
            if "babel-plugin-react-compiler" in Code.text(src, n):
                in_vite = True
        for n in Code.within(src, t.root_node, {"call_expression"}):
            f = n.child_by_field_name("function")
            if f is not None and Code.text(src, f) == "reactCompilerPreset":
                in_vite = True
    in_pkg = any("babel-plugin-react-compiler" in json.dumps(j)
                 for _b, j in c.json)
    return in_vite and in_pkg


def r_compiler_missing(c: Code) -> bool:
    return not r_compiler_configured(c)


_FUNCS = {"function_declaration", "function_expression", "arrow_function",
          "function", "method_definition"}


def _components(c: Code):
    """(src, body node) of every component: a function declaration, or an
    arrow / function assigned to a const, whose name starts with a capital
    letter."""
    for b, src, n in c.nodes({"function_declaration"}):
        nm = n.child_by_field_name("name")
        body = n.child_by_field_name("body")
        if nm is not None and body is not None and \
                c.text(src, nm)[:1].isupper():
            yield src, body
    for b, src, n in c.nodes({"variable_declarator"}):
        nm = n.child_by_field_name("name")
        val = n.child_by_field_name("value")
        if nm is not None and val is not None and val.type in (
                "arrow_function", "function_expression") and \
                c.text(src, nm)[:1].isupper():
            body = val.child_by_field_name("body")
            if body is not None:
                yield src, body


def _top_level(node):
    """The nodes of a function body outside any nested function."""
    out, stack = [], list(node.children)
    while stack:
        x = stack.pop()
        if x.type in _FUNCS:
            continue
        out.append(x)
        stack.extend(x.children)
    return out


def r_ref_current_in_render(c: Code) -> bool:
    """`.current` read or written in a component's render body (outside any
    nested function), except the docs' one-time initialisation idiom
    `if (!ref.current) ref.current = ...`."""
    for src, body in _components(c):
        lazy = set()
        for x in _top_level(body):
            if x.type == "if_statement":
                cond = x.child_by_field_name("condition")
                if cond is not None and re.search(
                        r"!\s*[\w$.]+\.current\b|[\w$.]+\.current\s*===?\s*"
                        r"null", c.text(src, cond)):
                    for y in Code.within(src, x, {"member_expression"}):
                        lazy.add(y.id)
        for x in _top_level(body):
            if x.type == "member_expression" and x.id not in lazy and \
                    c.text(src, x).endswith(".current"):
                return True
    return False


def r_no_ref_current_in_render(c: Code) -> bool:
    return bool(c.trees) and not r_ref_current_in_render(c)


def _effect_cleanup(c: Code, src, cb) -> bool:
    body = cb.child_by_field_name("body")
    if body is None:
        return False
    if body.type != "statement_block":
        return body.type in _FUNCS
    for r in _top_level(body):
        if r.type == "return_statement" and r.named_children and \
                r.named_children[0].type in ("arrow_function",
                                             "function_expression",
                                             "identifier"):
            return True
    return False


def r_ref_guard_in_effect(c: Code) -> bool:
    """An Effect guarded by a ref so it "runs once" (`if (!x.current)` /
    `if (x.current) return`) -- the docs' "Don't use refs to prevent
    Effects from firing"."""
    for src, cb, _n in c.effect_bodies():
        for x in c.within(src, cb, {"if_statement"}):
            cond = x.child_by_field_name("condition")
            if cond is not None and ".current" in c.text(src, cond):
                return True
    return False


def r_effect_cleanup_no_guard(c: Code) -> bool:
    bodies = list(c.effect_bodies())
    return bool(bodies) and not r_ref_guard_in_effect(c) and all(
        _effect_cleanup(c, src, cb) for src, cb, _n in bodies
        if _io_calls(c, src, cb) or "connect" in c.text(src, cb))


def r_use_cached_promise(c: Code) -> bool:
    return r_uses_use_hook(c) and not r_use_of_render_promise(c)


def r_imperative_handle(c: Code) -> bool:
    return c.called("useImperativeHandle")


def r_open_as_prop(c: Code) -> bool:
    """The modal's visibility is a prop (`isOpen` / `open`), not an
    imperative handle."""
    if r_imperative_handle(c):
        return False
    if any(a in ("isOpen", "open") and en[:1].isupper()
           for en, a, _v in c.jsx_attrs()):
        return True
    for b, src, n in c.nodes({"object_pattern"}):
        if re.search(r"(^|[{,\s])(isOpen|open)\s*[,}:=]", c.text(src, n)):
            return True
    return False


def _is_compiler_entry(src, k) -> bool:
    """A Babel plugins-array entry that IS the compiler: its name as a
    string, or `[name, options]`."""
    name = "babel-plugin-react-compiler"
    if k.type == "string":
        return name in Code.text(src, k)
    if k.type == "array" and k.named_children:
        return k.named_children[0].type == "string" and \
            name in Code.text(src, k.named_children[0])
    return False


def _compiler_plugins_array(c: Code):
    """The Babel plugins array the compiler is listed in (the innermost
    array with the compiler as a direct entry)."""
    for b, src, t in _vite_trees(c):
        for n in Code.within(src, t.root_node, {"array"}):
            kids = n.named_children
            if any(_is_compiler_entry(src, k) for k in kids):
                return src, kids
    return None, None


def r_compiler_first(c: Code) -> bool:
    src, kids = _compiler_plugins_array(c)
    return bool(kids) and _is_compiler_entry(src, kids[0])


def r_compiler_not_first(c: Code) -> bool:
    return not r_compiler_first(c)


RULES = {k[2:]: v for k, v in dict(globals()).items()
         if k.startswith("r_") and callable(v)}
# THE OTHER AREAS' RULES (2026-09-30): bench/skills/pitfall_rules/<area>.py,
# each a module of `r_<name>(c: Code) -> bool` functions (they import
# pitfall_harness for Code). A name defined twice is an error, never a
# silent override.
RULES_DIR = os.path.join(HERE, "pitfall_rules")


def _load_rule_modules() -> None:
    import importlib.util
    if not os.path.isdir(RULES_DIR):
        return
    for fn in sorted(os.listdir(RULES_DIR)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(
            f"pitfall_rules_{fn[:-3]}", os.path.join(RULES_DIR, fn))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k, v in vars(mod).items():
            if k.startswith("r_") and callable(v) \
                    and getattr(v, "__module__", "") == mod.__name__:
                if k[2:] in RULES:
                    raise ValueError(f"pitfall rule {k[2:]!r} defined twice "
                                     f"({fn})")
                RULES[k[2:]] = v


if __name__ == "__main__":
    sys.modules.setdefault("pitfall_harness", sys.modules["__main__"])
_load_rule_modules()


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
        def wrap(x: str, lang=case.get("example_lang", "tsx")) -> str:
            return x if "```" in x else f"```{lang}\n" + x + "\n```"
        bad = check_answer(case, wrap(ex.get("pitfall", "")))
        good = check_answer(case, wrap(ex.get("good", "")))
        ok = bad["pitfall"] and not bad["good"] and not good["pitfall"] \
            and good["good"]
        res.append({"id": case["id"], "ok": ok, "pitfall_example": bad,
                    "good_example": good})
    return {"cases": len(res), "ok": sum(r["ok"] for r in res),
            "failed": [r for r in res if not r["ok"]]}


def _skill_ref(x) -> tuple[str, list[str] | None]:
    """A case's skill entry: a name (every item) or {name, match: [text]}
    (only the items whose text contains one of the given pieces: a skill
    whose other lines are about something else injects only the lines the
    case is about; matched by text, not by position, because the served row
    leaves doubt-bearing lines out)."""
    if isinstance(x, dict):
        return x["name"], list(x.get("match") or []) or None
    return str(x), None


def library_skills(refs: list) -> tuple[list[dict], list[str]]:
    """(armed skill rows, in order, each cut to the items the case names;
    the names not armed, and a name whose `match` found no item)."""
    import skills
    import skill_select
    by: dict = {}
    for s in skills.armed():
        # a skill is found by its name, the name its creator asked for (a
        # served row's `alias`: the pipeline now keeps it, but a skill built
        # before that carries the model's) or its id
        for key in (s.get("name"), s.get("alias"), s.get("id")):
            if key:
                by.setdefault(key, s)
    got, missing = [], []
    for x in refs:
        name, match = _skill_ref(x)
        s = by.get(name)
        if s is None:
            missing.append(name)
            continue
        if match is not None:
            its = [it for it in skill_select._items(s)
                   if any(m.lower() in str(it.get("text") or "").lower()
                          for m in match)]
            if not its:
                missing.append(f"{name} (no item matches {match})")
                continue
            s = dict(s, items=its)
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


# ----------------------------------------------- THE TOOL-RESULT RENDERINGS --
# Operator, 2026-10-06: "If the model uses some of our other tools or mcps
# this may be a good point to skill up", and "if a model was told the tool to
# call and given a few router like decisions maybe it would be inclined to
# call the craft tool for more of the skill". The same prompt WITHOUT and
# WITH the skill delivered the way the proxy delivers it in a package tool's
# result (mcp/package_skills.py), two designs of the same channel -- a
# comparison the operator asked for, not an on/off arm:
#   tool_result_inject  the model "called" yama_find_package for the case's
#                       packages; its result carries the package's lead skill
#                       and strongest items (package_skills.decide, inject)
#   tool_result_router  the result carries NO skill body: a decision-router
#                       table, one row per armed, proven craft of the
#                       package, and yama_recall_craft is on the tool list;
#                       the model may call it (up to tiers.TOOL_TURNS hops,
#                       the vendor demo's cap) and gets the craft in full
# What rides is what the CHANNEL picks over the armed library (the packages
# of the case, their lead and strongest crafts), not the case's own skill
# list: each repeat records whether a case skill was among the crafts
# delivered (`case_skills_delivered`) and which crafts the model recalled, so
# a result is read conditional on delivery. A question to yama_recall_craft
# is answered here by the nearest craft (the embedder's shortlist, no jjava:
# the harness runs without the serving process's decider; the probe's
# question arm measures the real path through the proxy), and said so.
TOOL_RESULT_VARIANTS = ("tool_result_inject", "tool_result_router")
_MODE_OF = {"tool_result_inject": "inject", "tool_result_router": "router"}


def _found_for(case: dict) -> list[dict]:
    import skill_packages
    out = []
    for pkg in case.get("packages") or []:
        v = skill_packages.held_versions(pkg)
        out.append({"name": pkg, "version": v[-1] if v else None})
    return out


def tool_result_delivery(case: dict, variant: str, model: str) -> dict:
    """What the channel delivers for the case's packages: {section, result
    (the lookup's result as the model reads it, the section appended),
    found, delivered (craft names), case_skills_delivered, records}."""
    import mcp_host
    import package_skills as PS
    import skills
    found = _found_for(case)
    mode = _MODE_OF[variant]
    pool = skills.armed()
    section, recs, _st = PS.decide({}, found, {
        "mode": mode, "serving": model, "craft_tool": mode == "router",
        "tool": "yama_find_package", "asked": {}, "key": None}, pool)
    d = {"query": ", ".join(f["name"] for f in found),
         "searchedEcosystems": ["npm"],
         "results": [{"ecosystem": "npm", "total": len(found), "results": [
             {"name": f["name"], "version": f["version"],
              "description": ""} for f in found]}]}
    body_text, _names = mcp_host.render_search(d, {"ecosystem": "npm"})
    result = (f"SOURCE: the npm registry, through PackageLens (registry "
              f"data)\n{mcp_host.DATA_NOTE}\n\n{body_text}")
    delivered = [x["name"] for r in recs for x in r.get("skills") or []]
    case_names = {_skill_ref(x)[0] for x in case.get("skills") or []}
    return {"section": section, "result": result + section, "found": found,
            "delivered": delivered,
            "case_skills_delivered": sorted(case_names & set(delivered)),
            "records": recs}


def _tool_defs(router: bool) -> list[dict]:
    import mcp_config
    import skill_select
    t = next(x for x in mcp_config.PACKAGELENS["tools"]
             if x["name"] == "yama_find_package")
    return [mcp_config.definition(t)] + (
        [skill_select.READ_TOOL] if router else [])


def _recall(args: dict, pool: list[dict], pkg: str | None) -> tuple[str, dict]:
    """A yama_recall_craft call in the harness: a name is that craft
    (skill_select.read_craft); a question is answered by the nearest craft
    of the embedder's shortlist (no jjava here)."""
    import craft_query
    import skill_select
    q = str((args or {}).get("name_or_topic") or "").strip()
    text, rec = skill_select.read_craft(args, pool)
    if rec.get("found"):
        return text, {"arg": q[:200], "form": "name",
                      "craft": rec.get("name")}
    cand = craft_query.proven(pool)
    top, _n = craft_query.shortlist(q, cand, {"package": pkg}) \
        if cand else ([], {})
    if not top:
        return text, {"arg": q[:200], "form": "question", "craft": None}
    s = top[0]
    return (skill_select_text(s), {"arg": q[:200], "form": "question",
                                   "craft": s.get("name"),
                                   "resolved_by": "nearest, no jjava"})


def skill_select_text(s: dict) -> str:
    import skill_prompts as P
    import skill_select
    return (P.CRAFT_RESULT_HEAD.format(name=s.get("name")) + "\n\n"
            + skill_select.injected_text(s))


def tool_arm_generate(case: dict, variant: str, model: str, effort: str,
                      seed: int, dl: dict) -> tuple[str | None, dict]:
    """One repeat of a tool-result arm: the conversation (user prompt, the
    model's find_package call, its result with the section), then the
    model's turns until it answers -- a call of yama_recall_craft is run and
    its result handed back. {finish_reason, completion_tokens,
    reasoning_chars, s, hops, recalls}."""
    import model as M
    import skills
    import tiers
    router = variant == "tool_result_router"
    pool = skills.armed()
    call = {"id": "call_h1", "type": "function", "function": {
        "name": "yama_find_package", "arguments": json.dumps({
            "query": ", ".join(f["name"] for f in dl["found"]),
            "ecosystem": "npm"})}}
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": case["prompt"]},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_h1",
             "content": dl["result"]}]
    gen = {"hops": 0, "recalls": [], "completion_tokens": 0,
           "reasoning_chars": 0, "s": 0.0}
    pkg = (dl["found"][0]["name"] if dl["found"] else None)
    for hop in range(max(int(tiers.TOOL_TURNS), 1)):
        b = M.shape({"messages": msgs, "tools": _tool_defs(router),
                     "max_tokens": tiers.A_MIN}, effort=effort,
                    role="helper", step_cap=tiers.HELPER_THINKING)
        b.pop("_share", None)
        b["model"], b["seed"] = model, seed
        t0 = time.time()
        try:
            d = M.post(b)
        except Exception as e:                                   # noqa: BLE001
            gen["why"] = f"not run: {type(e).__name__}: {e}"[:200]
            return None, gen
        gen["s"] = round(gen["s"] + time.time() - t0, 2)
        gen["hops"] = hop + 1
        ch = ((d.get("choices") or [{}])[0]) if isinstance(d, dict) else {}
        msg = ch.get("message") or {}
        gen["finish_reason"] = ch.get("finish_reason")
        gen["completion_tokens"] += int((d.get("usage") or {}).get(
            "completion_tokens") or 0)
        gen["reasoning_chars"] += len(msg.get("reasoning_content") or "")
        calls = msg.get("tool_calls") or []
        if not calls:
            if ch.get("finish_reason") == "length":
                gen["why"] = "budget event: length"
                return None, gen
            return msg.get("content") or "", gen
        msgs.append({"role": "assistant", "content": msg.get("content") or "",
                     "reasoning_content": msg.get("reasoning_content") or "",
                     "tool_calls": calls})
        for c in calls:
            fn = (c.get("function") or {}).get("name")
            try:
                args = json.loads((c.get("function") or {}).get(
                    "arguments") or "{}")
            except ValueError:
                args = {}
            if fn == "yama_recall_craft" and router:
                text, rec = _recall(args, pool, pkg)
                gen["recalls"].append(dict(rec, hop=hop + 1))
            else:
                text = ("This harness runs no other tool: answer with the "
                        "code.")
            msgs.append({"role": "tool", "tool_call_id": c.get("id"),
                         "content": text})
    gen["why"] = "the hop cap (tiers.TOOL_TURNS) ended the run"
    return None, gen


def run_tool_arm(case: dict, variant: str, model: str, effort: str,
                 repeats: int) -> dict:
    dl = tool_result_delivery(case, variant, model)
    if not dl["section"]:
        return {"not_run": "the channel delivers nothing for this case's "
                "packages (no armed, proven craft: a GAP)"}
    reps = []
    for k in range(repeats):
        seed = seed_of(case["id"], k)
        ans, gen = tool_arm_generate(case, variant, model, effort, seed, dl)
        chk = check_answer(case, ans) if ans is not None else None
        reps.append({"seed": seed, "answered": ans is not None, "gen": gen,
                     **(chk or {}),
                     "types": types_ok(case, ans) if ans else None,
                     "recalls": len(gen.get("recalls") or []),
                     "answer_sha": hashlib.sha1((ans or "").encode(
                         "utf-8")).hexdigest()[:12]})
    return {"repeats": reps, "section_chars": len(dl["section"]),
            "delivered": dl["delivered"],
            "case_skills_delivered": dl["case_skills_delivered"],
            "records": [{k: r.get(k) for k in ("package", "major", "mode",
                                               "skills", "chars", "why")}
                        for r in dl["records"]]}


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


_IMPORT = re.compile(
    r"^[ \t]*import\s+(?P<clause>[^;]*?)\s*from\s*(?P<q>['\"])(?P<src>[^'\"]+)"
    r"(?P=q)[ \t]*;?[ \t]*$|^[ \t]*import\s*(?P<q2>['\"])(?P<side>[^'\"]+)"
    r"(?P=q2)[ \t]*;?[ \t]*$", re.M | re.S)
_TOP_DECL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
    r"(?:function\s*\*?|class|const|let|var|interface|type|enum|"
    r"abstract\s+class)\s+([A-Za-z_$][\w$]*)", re.M)
_ID = r"[A-Za-z_$][\w$]*"


def _clause(clause: str) -> dict:
    """An import clause read: {type_only, default, namespace, named:
    [(imported, local, is_type)]}."""
    c = " ".join(clause.split())
    type_only = bool(re.match(r"^type\s+(?!from\b)", c))
    if type_only:
        c = c[5:].strip()
    out = {"type_only": type_only, "default": None, "namespace": None,
           "named": []}
    m = re.search(r"\{(.*)\}", c, re.S)
    if m:
        for part in m.group(1).split(","):
            p = part.strip()
            if not p:
                continue
            is_type = p.startswith("type ")
            p = p[5:].strip() if is_type else p
            mm = re.match(rf"({_ID})(?:\s+as\s+({_ID}))?$", p)
            if mm:
                out["named"].append((mm.group(1), mm.group(2) or mm.group(1),
                                     is_type or type_only))
        c = c[:m.start()] + c[m.end():]
    ns = re.search(rf"\*\s*as\s+({_ID})", c)
    if ns:
        out["namespace"] = ns.group(1)
        c = c[:ns.start()] + c[ns.end():]
    d = re.match(rf"^\s*({_ID})\s*,?", c)
    if d and d.group(1) not in ("type",):
        out["default"] = d.group(1)
    return out


def ts_probe(answer: str) -> str:
    """ONE type-check probe from an answer's TS/JS blocks (2026-09-30, the
    coordinator's harness fixes; the TypeScript area found both bugs):
      - the task's OWN modules ('./chat', '../api') exist only in the
        prompt, or as another block of the answer: a name imported from one
        is declared here UNLESS the answer itself defines it -- a TYPE-only
        import as `type X = any`, a value import as BOTH `declare const X:
        any` and `type X = any` (legal: separate declaration spaces), so a
        later type use of X no longer fails ("refers to a value");
      - package imports are MERGED per module across blocks (a name
        imported by two files is imported once: no duplicate identifier);
      - only the first block keeps an `export default` (a second one is a
        plain declaration; `export default <name>;` lines go), so several
        files joined are one valid module.
    The type check judges the package API use, not a missing file."""
    codes = [b["code"] for b in blocks(answer)
             if b["lang"] in ("", "ts", "tsx", "typescript", "js", "jsx",
                              "javascript")]
    bodies, locals_, pkgs, side = [], {}, {}, []
    seen_default = False
    for code in codes:
        def take(m):
            if m.group("side"):
                if m.group("side") not in side:
                    side.append(m.group("side"))
                return ""
            src, cl = m.group("src"), _clause(m.group("clause"))
            if src.startswith("."):
                for n, t in ([(cl["default"], cl["type_only"])]
                             if cl["default"] else []) + \
                        ([(cl["namespace"], cl["type_only"])]
                         if cl["namespace"] else []) + \
                        [(loc, t) for _i, loc, t in cl["named"]]:
                    locals_[n] = locals_.get(n, True) and t
                return ""
            e = pkgs.setdefault(src, {"default": None, "namespace": [],
                                      "named": {}, "type_only": True})
            e["type_only"] = e["type_only"] and cl["type_only"] and \
                not cl["default"] and not cl["namespace"]
            if cl["default"]:
                e["default"] = e["default"] or cl["default"]
            if cl["namespace"] and cl["namespace"] not in e["namespace"]:
                e["namespace"].append(cl["namespace"])
            for imp, loc, t in cl["named"]:
                prev = e["named"].get(loc)
                e["named"][loc] = (imp, (prev[1] if prev else True) and t)
            return ""
        body = _IMPORT.sub(take, code)
        if re.search(r"^\s*export\s+default\b", body, re.M):
            if seen_default:
                body = re.sub(r"^(\s*)export\s+default\s+(?=(?:async\s+)?"
                              r"(?:function|class)\b)", r"\1", body,
                              flags=re.M)
                body = re.sub(rf"^\s*export\s+default\s+{_ID}\s*;?\s*$", "",
                              body, flags=re.M)
            seen_default = True
        bodies.append(body)
    joined = "\n".join(bodies)
    defined = set(_TOP_DECL.findall(joined))
    head = [f"import '{s}';" for s in side]
    for src, e in pkgs.items():
        named = ", ".join(
            ("type " if t and not e["type_only"] else "")
            + (imp if imp == loc else f"{imp} as {loc}")
            for loc, (imp, t) in e["named"].items())
        parts = [p for p in (e["default"], "{ " + named + " }" if named
                             else None) if p]
        kw = "import type" if e["type_only"] and not e["default"] else \
            "import"
        if parts:
            head.append(f"{kw} {', '.join(parts)} from '{src}';")
        for ns in e["namespace"]:
            head.append(f"import * as {ns} from '{src}';")
    for n, type_only in locals_.items():
        if n in defined:
            continue
        head.append(f"type {n} = any;" if type_only
                    else f"declare const {n}: any; type {n} = any;")
    return "\n".join(head) + ("\n" if head else "") + joined


def types_ok(case: dict, answer: str) -> bool | None:
    """tsc (mcp/typecheck.py, the held React 19.2.8 + @types/react 19.2.18)
    over ONE probe of the answer's TS/JS blocks (ts_probe: the task's
    own modules declared `any`, package imports merged, one default
    export); a setup case (config files) is not type-checked."""
    if case["kind"] != "pitfall" or case.get("typecheck") == "none":
        return None
    import typecheck
    if not any(b["code"].strip() for b in blocks(answer)
               if b["lang"] in ("", "ts", "tsx", "typescript", "js", "jsx",
                                "javascript")):
        return None
    code = ts_probe(answer)
    # `packages` absent: React (the first area); present and empty: plain
    # TypeScript, no package installed (the typescript area)
    r = typecheck.check(code, "tsx", case["packages"]
                        if "packages" in case else ["react"])
    return bool(r.get("ok")) if r.get("ran") else None


def run_case(case: dict, model: str, effort: str, repeats: int,
             variants) -> dict:
    rec = {"v": HARNESS_VERSION, "model": model, "effort": effort,
           "case": case["id"], "area": case["area"], "kind": case["kind"],
           "sides": {}}
    _rows, missing = library_skills(case.get("skills") or [])
    rec["skills_missing"] = missing
    for side in ("without",) + tuple(variants):
        if side in TOOL_RESULT_VARIANTS:
            rec["sides"][side] = run_tool_arm(case, side, model, effort,
                                              repeats)
            continue
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
            if v in TOOL_RESULT_VARIANTS and reps:
                row[v]["recalls_mean"] = round(sum(
                    x.get("recalls") or 0 for x in reps) / len(reps), 2)
                row[v]["called_recall_in"] = sum(
                    1 for x in reps if x.get("recalls"))
                row[v]["case_skills_delivered"] = s.get(
                    "case_skills_delivered")
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
    ap.add_argument("--cases", help="only these case ids (comma separated)")
    ap.add_argument("--only-ready", action="store_true",
                    help="only cases whose every named skill is armed NOW "
                    "(a case with a skill still being re-proved or still a "
                    "gap is left for a later run)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip case ids already in the results file (the "
                    "library is read when the process starts, so a later "
                    "run covers the cases a skill had not armed for)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="renderings to run beside `without`; the package-"
                    "tool channel's two designs are tool_result_inject and "
                    "tool_result_router (not in the default list)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    cases = load_cases(a.area)
    if a.cases:
        want = {x.strip() for x in a.cases.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in want]
    if a.only_ready:
        import replay_selection  # noqa: F401  (store isolation)
        cases = [c for c in cases if c.get("skills")
                 and not library_skills(c["skills"])[1]]
    if a.skip_done and a.model:
        try:
            with open(os.path.join(RESULTS, f"pitfall_{a.model}.jsonl"),
                      encoding="utf-8") as f:
                done = {json.loads(ln)["case"] for ln in f if ln.strip()}
        except OSError:
            done = set()
        cases = [c for c in cases if c["id"] not in done]
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
    variants = [v for v in a.variants.split(",")
                if v in VARIANTS + TOOL_RESULT_VARIANTS]
    # THE PREFILL ARM needs the engine check (coordinator, 2026-09-29: "if
    # prefill_check fails (the think block doesn't stay open), drop the
    # first_person_prefill arm from the harness run and record why"):
    # bench/skills/prefill_check.py's verdict for this model, read here.
    if "first_person_prefill" in variants:
        pc = os.path.join(RESULTS, f"prefill_check_{a.model}.json")
        try:
            with open(pc, encoding="utf-8") as f:
                pcr = json.load(f)
        except (OSError, ValueError):
            pcr = None
        if not (pcr or {}).get("confirmed"):
            variants.remove("first_person_prefill")
            why = ("prefill_check has not run for this model" if pcr is None
                   else "prefill_check did not confirm the think block stays "
                   "open and the generation continues it: " + json.dumps(
                       {"template": (pcr.get("template") or {}).get(
                           "confirmed"), "generation": (pcr.get(
                               "generation") or {}).get("confirmed")}))
            if not a.dry_run:
                os.makedirs(RESULTS, exist_ok=True)
                with open(os.path.join(RESULTS,
                                       f"pitfall_{a.model}_notes.jsonl"),
                          "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.time(), "dropped":
                                        "first_person_prefill", "why": why})
                            + "\n")
            print(f"  first_person_prefill arm DROPPED: {why}", flush=True)
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
