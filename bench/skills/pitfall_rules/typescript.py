"""THE TYPESCRIPT PITFALL RULES (bench/skills/pitfall_harness.py loads this
module; the cases are bench/skills/pitfalls/typescript.jsonl, each citing a
page of microsoft/TypeScript-Website pinned under
pitfalls/sources/TypeScript-Website@6556b08756b7/).

Each rule: `r_typescript_<name>(c) -> bool`, `c` the harness's Code (the
answer's blocks, TS/JS parsed with tree-sitter tsx, tsconfig.json read with
comments and trailing commas tolerated). Syntax-tree checks; a tsconfig is
read as parsed JSON. Plain module: nothing is imported from the harness.
"""
from __future__ import annotations


# --------------------------------------------------------------- helpers ---
def _parent_chain(n):
    p = n.parent
    while p is not None:
        yield p
        p = p.parent


def _is_unknown_type(c, src, t) -> bool:
    return t is not None and c.text(src, t).strip().lstrip(":").strip() \
        == "unknown"


def _function_returns_unknown(c, src, fn) -> bool:
    rt = fn.child_by_field_name("return_type")
    return rt is not None and _is_unknown_type(c, src, rt)


def _param_is_unknown(c, callee: str, index: int) -> bool:
    """A function declared in the answer, named `callee`, whose parameter
    at `index` is typed `unknown`."""
    for _b, src, fn in c.nodes({"function_declaration"}):
        nm = fn.child_by_field_name("name")
        if nm is None or c.text(src, nm) != callee:
            continue
        ps = fn.child_by_field_name("parameters")
        params = [x for x in (ps.named_children if ps is not None else [])
                  if x.type in ("required_parameter", "optional_parameter")]
        if index < len(params):
            t = params[index].child_by_field_name("type")
            if _is_unknown_type(c, src, t):
                return True
    return False


def _parse_lands_in_unknown(c, src, call) -> bool:
    """A JSON.parse(...) whose `any` result goes straight into an `unknown`
    slot: `const x: unknown = JSON.parse(s)`, `JSON.parse(s) as unknown`,
    `return JSON.parse(s)` from a function declared `: unknown`, or an
    argument to a function (declared in the answer) whose parameter is
    `unknown`."""
    node = call
    for p in _parent_chain(call):
        if p.type == "parenthesized_expression" or p.type == "await_expression":
            node = p
            continue
        if p.type == "as_expression":
            return c.text(src, p).rstrip().endswith("unknown")
        if p.type == "variable_declarator":
            return _is_unknown_type(c, src, p.child_by_field_name("type"))
        if p.type == "return_statement":
            for q in _parent_chain(p):
                if q.type in ("function_declaration", "arrow_function",
                              "function_expression", "method_definition"):
                    return _function_returns_unknown(c, src, q)
            return False
        if p.type == "arrow_function":
            # `(s: string): unknown => JSON.parse(s)`
            return _function_returns_unknown(c, src, p)
        if p.type == "arguments":
            outer = p.parent
            if outer is None or outer.type != "call_expression":
                return False
            args = [x for x in p.named_children]
            idx = next((i for i, x in enumerate(args) if x.id == node.id), -1)
            return idx >= 0 and _param_is_unknown(
                c, c.callee(src, outer), idx)
        return False
    return False


# ----------------------------------------------------- any vs unknown ------
def r_typescript_any_or_unchecked_parse(c) -> bool:
    """PITFALL: an explicit `any` anywhere, or JSON.parse's `any` result
    used without first landing in `unknown` (Do's and Don'ts: "Don't use
    `any` as a type unless you are in the process of migrating a JavaScript
    project to TypeScript")."""
    for _b, src, n in c.nodes({"predefined_type"}):
        if c.text(src, n) == "any":
            return True
    for _b, src, call in c.calls("parse"):
        if c.callee(src, call) != "JSON.parse":
            continue
        if not _parse_lands_in_unknown(c, src, call):
            return True
    return False


def _narrows(c) -> bool:
    """A runtime narrowing construct: `typeof x ===`, `'k' in x`,
    `x instanceof C`, `Array.isArray(x)`, or a type predicate (`x is T`)."""
    for _b, src, n in c.nodes({"binary_expression"}):
        op = n.child_by_field_name("operator")
        opt = c.text(src, op) if op is not None else ""
        if opt in ("in", "instanceof"):
            return True
        if opt in ("===", "!==", "==", "!="):
            left = n.child_by_field_name("left")
            right = n.child_by_field_name("right")
            if any(x is not None and x.type == "unary_expression"
                   and c.text(src, x).startswith("typeof")
                   for x in (left, right)):
                return True
    if c.called("isArray"):
        return True
    return any(True for _ in c.nodes({"type_predicate"}))


def r_typescript_unknown_narrowed(c) -> bool:
    """GOOD: an `unknown` type is used and narrowed before use."""
    has_unknown = any(c.text(src, n) == "unknown"
                      for _b, src, n in c.nodes({"predefined_type"}))
    return has_unknown and _narrows(c)


# ------------------------------------------------ enum vs union/as const ---
def r_typescript_enum_declared(c) -> bool:
    """PITFALL: an `enum` (or `const enum`) declaration."""
    return any(True for _ in c.nodes({"enum_declaration"}))


def r_typescript_union_or_const_object(c) -> bool:
    """GOOD: the set of values is a union of literal types
    (`type S = 'a' | 'b'`) or an object literal `as const` (Enums: "In
    modern TypeScript, you may not need an enum when an object with
    `as const` could suffice")."""
    for _b, src, n in c.nodes({"type_alias_declaration"}):
        v = n.child_by_field_name("value")
        if v is None:
            continue
        if v.type == "union_type":
            lits = c.within(src, v, {"literal_type"})
            if len(lits) >= 2:
                return True
        if v.type == "lookup_type" and "keyof typeof" in c.text(src, v):
            return True
        if v.type == "index_type_query":
            # `type S = keyof typeof STATUS`
            return True
    for _b, src, n in c.nodes({"as_expression"}):
        if c.text(src, n).rstrip().endswith("as const") and any(
                x.type in ("object", "array") for x in n.named_children):
            return True
    return False


# ------------------------------------------------- namespace vs modules ----
def _namespaces(c):
    for _b, src, n in c.nodes({"internal_module", "module"}):
        if not n.is_named:
            continue            # the `module` keyword token itself
        nm = n.child_by_field_name("name")
        if nm is not None and nm.type == "string":
            continue            # `declare module "pkg"`: an ambient module
        yield src, n


def r_typescript_namespace_declared(c) -> bool:
    """PITFALL: a `namespace X {}` (or legacy `module X {}`) block."""
    return any(True for _ in _namespaces(c))


def r_typescript_module_exports(c) -> bool:
    """GOOD: the group is an ES module -- functions exported at the top
    level of a file, no namespace (Modules: "we recommend you use that to
    align with JavaScript's direction")."""
    if r_typescript_namespace_declared(c):
        return False
    for _b, src, t in c.trees:
        for st in t.root_node.named_children:
            if st.type != "export_statement":
                continue
            d = st.child_by_field_name("declaration")
            if d is None:
                continue
            if d.type == "function_declaration":
                return True
            if d.type == "lexical_declaration" and any(
                    x.type in ("arrow_function", "function_expression")
                    for x in c.within(src, d, {"arrow_function",
                                               "function_expression"})):
                return True
    return False


# --------------------------------------------------- type-only imports -----
def _named_imports(c):
    """(src, import statement, specifier, local name, marked type)."""
    for _b, src, st in c.nodes({"import_statement"}):
        whole_type = c.text(src, st).lstrip().startswith("import type")
        for sp in c.within(src, st, {"import_specifier"}):
            alias = sp.child_by_field_name("alias")
            name = sp.child_by_field_name("name")
            local = alias if alias is not None else name
            if local is None:
                continue
            marked = whole_type or c.text(src, sp).lstrip().startswith(
                "type ")
            yield src, st, sp, c.text(src, local), marked


def _uses(c, src, name: str, skip) -> tuple[int, int]:
    """(value uses, type uses) of `name` in one block, outside `skip`."""
    val = typ = 0
    for _b, s2, n in c.nodes({"identifier", "type_identifier",
                              "shorthand_property_identifier"}):
        if s2 is not src or c.text(s2, n) != name:
            continue
        if skip.start_byte <= n.start_byte < skip.end_byte:
            continue
        if n.type == "type_identifier":
            typ += 1
        else:
            val += 1
    return val, typ


def r_typescript_type_import_unmarked(c) -> bool:
    """PITFALL: a named import used only in type positions but imported
    without `type` (verbatimModuleSyntax: "any imports or exports without a
    `type` modifier are left around")."""
    for src, st, _sp, name, marked in _named_imports(c):
        if marked:
            continue
        val, typ = _uses(c, src, name, st)
        if typ and not val:
            return True
    return False


def r_typescript_type_import_marked(c) -> bool:
    """GOOD: type-only imports are written `import type { T }` or
    `import { type T }`, and none is left unmarked."""
    if r_typescript_type_import_unmarked(c):
        return False
    return any(marked for _s, _st, _sp, _n, marked in _named_imports(c))


# ------------------------------------------------------------ satisfies ----
def r_typescript_widening_annotation_or_cast(c) -> bool:
    """PITFALL: an object literal whose declared type widens it (a
    `Record<...>` or index-signature annotation), or a member cast back with
    `as` (not `as const`) -- the 4.9 notes' "we'd lose the information about
    each property"."""
    for _b, src, n in c.nodes({"variable_declarator"}):
        t = n.child_by_field_name("type")
        v = n.child_by_field_name("value")
        if t is None or v is None or v.type != "object":
            continue
        if "Record<" in c.text(src, t) or c.within(src, t,
                                                   {"index_signature"}):
            return True
    for _b, src, n in c.nodes({"as_expression"}):
        if c.text(src, n).rstrip().endswith("as const"):
            continue
        inner = n.named_children[0] if n.named_children else None
        if inner is not None and inner.type in ("member_expression",
                                                "subscript_expression"):
            return True
    return False


def r_typescript_uses_satisfies(c) -> bool:
    """GOOD: the literal is checked with `satisfies`, keeping its own
    type."""
    return any(True for _ in c.nodes({"satisfies_expression"}))


# -------------------------------------------------------------- tsconfig ---
_LEGACY_RESOLUTION = {"node", "node10", "classic"}
_NODE_MODULES = {"node16", "node18", "node20", "nodenext"}


def _compiler_options(c) -> list[dict]:
    out = []
    for d in c.json_of("tsconfig.json"):
        if isinstance(d, dict) and isinstance(d.get("compilerOptions"),
                                              dict):
            out.append({k.lower(): v for k, v in
                        d["compilerOptions"].items()})
    return out


def _low(v) -> str:
    return str(v).lower() if isinstance(v, str) else ""


def r_typescript_legacy_module_resolution(c) -> bool:
    """PITFALL: tsconfig.json resolves modules the Node 10 / classic way --
    `moduleResolution` node, node10 or classic, or no `moduleResolution`
    with a `module` whose default is one of them (commonjs -> node10;
    es2015..esnext, amd, umd, system -> classic; TS 5.x's defaults, Theory
    "Module resolution")."""
    for co in _compiler_options(c):
        mr = _low(co.get("moduleresolution"))
        if mr in _LEGACY_RESOLUTION:
            return True
        if not mr:
            mod = _low(co.get("module"))
            if mod and mod not in _NODE_MODULES and mod != "preserve":
                return True
    return False


def _strict_on(co: dict) -> bool:
    return co.get("strict") is True


def r_typescript_bundler_tsconfig(c) -> bool:
    """GOOD (bundled app): `moduleResolution: "bundler"` (or `module:
    "preserve"`, which implies it), `verbatimModuleSyntax` or
    `isolatedModules` on, and `strict` on."""
    for co in _compiler_options(c):
        mr = _low(co.get("moduleresolution"))
        mod = _low(co.get("module"))
        bundler = mr == "bundler" or (not mr and mod == "preserve")
        per_file = co.get("verbatimmodulesyntax") is True \
            or co.get("isolatedmodules") is True
        if bundler and per_file and _strict_on(co):
            return True
    return False


def r_typescript_node_tsconfig(c) -> bool:
    """GOOD (Node.js app compiled by tsc): `module` nodenext (or node16 /
    node18 / node20), `moduleResolution` absent or the matching Node mode,
    `verbatimModuleSyntax` on, and `strict` on."""
    for co in _compiler_options(c):
        mod = _low(co.get("module"))
        mr = _low(co.get("moduleresolution"))
        if mod in _NODE_MODULES and (not mr or mr in _NODE_MODULES) \
                and co.get("verbatimmodulesyntax") is True \
                and _strict_on(co):
            return True
    return False
