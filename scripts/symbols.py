"""Symbol and call-site index built with tree-sitter.

This is deliberately NOT a full AST dump. Storing whole syntax trees is large,
slow to query and mostly useless to a model. What an agent actually asks is:

    "where is `parse_hunks` defined?"      -> definitions
    "who calls `parse_hunks`?"             -> call sites
    "where else is `Buffer` mentioned?"    -> references

so we extract exactly those three relations and index them in sqlite. Lookups
are exact-match and need no GPU, which makes them complementary to the vector
search: embeddings find things you cannot name, symbols find things you can.

Definition kinds are normalised across grammars (function/class/struct/...)
so a Rust `function_item`, a C++ `function_definition` and a TS
`function_declaration` all report as "function".
"""
from __future__ import annotations

import os
import sqlite3

from ast_chunker import LANG_BY_EXT, _parser

# node type fragment -> normalised kind, first match wins
KIND_MAP = (
    ("method", "method"),
    ("constructor", "constructor"),
    ("function", "function"),
    ("fn_proto", "function"), ("fnproto", "function"),
    ("class", "class"),
    ("struct", "struct"),
    ("interface", "interface"),
    ("protocol", "interface"),
    ("trait", "trait"),
    ("enum", "enum"),
    ("union", "union"),
    ("type_alias", "type"), ("type_definition", "type"), ("type_declaration", "type"),
    ("namespace", "namespace"),
    ("module", "module"),
    ("impl", "impl"),
)

NAME_NODE_TYPES = ("identifier", "type_identifier", "field_identifier",
                   "property_identifier", "name", "namespace_identifier")


# A HERITAGE CLAUSE NAMES A TYPE; IT DOES NOT DEFINE ONE. `class_heritage`
# contains "class", `implements_clause` contains "impl", `superclass` and
# `base_class_clause` contain "class", `trait_bounds` contains "trait" -- so
# each matched KIND_MAP and the parent type was recorded as DEFINED in every
# file that extends it. Measured in three@0.185.1: `AnalyticLightNode` had 8
# definitions, one real (src/nodes/lighting/AnalyticLightNode.js) and seven
# `class X extends AnalyticLightNode`. find_definition_opt returned all 8.
# The identifiers inside are still walked, so the parent is recorded as a
# reference from each subclass, which is what that line is.
_HERITAGE = ("heritage", "superclass", "super_interface", "base_class",
             "extends", "implements", "trait_bound", "inheritance",
             "delegation_specifier", "supertype")


def _kind(node_type: str) -> str | None:
    tl = node_type.lower()
    if tl.endswith(("_list", "_body", "_block")) or tl in NAME_NODE_TYPES:
        return None
    if any(h in tl for h in _HERITAGE):
        return None
    # `Buffer { .. }` is a struct EXPRESSION, not a definition. Without this
    # every construction site is recorded as if it declared the type.
    if "expression" in tl or tl.endswith(("_literal", "_type", "_pattern")):
        return None
    for frag, kind in KIND_MAP:
        if frag in tl:
            return kind
    return None


def _name_of(node) -> str | None:
    """Best-effort name for a definition node.

    Prefers an explicit `name` field where the grammar provides one, since
    that is unambiguous; otherwise takes the first identifier-ish child.

    A Rust `impl Display for Circle` has no name field, and its first
    identifier is the TRAIT -- so every impl of Display was recorded as a
    definition of Display. The block defines Circle's methods, so it is
    named by its `type` field (the heritage fix above, for Rust).
    """
    if node.type == "impl_item":
        try:
            t = node.child_by_field_name("type")
            if t is not None:
                return t.text.decode("utf-8", "replace")
        except Exception:
            pass
    try:
        f = node.child_by_field_name("name")
        if f is not None:
            return f.text.decode("utf-8", "replace")
    except Exception:
        pass
    for c in node.named_children:
        if c.type in NAME_NODE_TYPES:
            return c.text.decode("utf-8", "replace")
    return None


def _callee(node) -> str | None:
    """Extract the called symbol from a call-ish node."""
    try:
        f = node.child_by_field_name("function")
    except Exception:
        f = None
    target = f if f is not None else (node.named_children[0] if node.named_children else None)
    if target is None:
        return None
    # a.b.c() / a::b() -> take the final segment, that is the symbol a user means
    txt = target.text.decode("utf-8", "replace")
    for sep in ("::", "."):
        if sep in txt:
            txt = txt.split(sep)[-1]
    txt = txt.strip()
    return txt if txt.isidentifier() else None


# An exported binding is API. `export const positionLocal = ...` declares a
# symbol every caller imports by name, and tree-sitter files it under
# lexical_declaration, which matches none of the KIND_MAP markers above.
#
# The consequence was total for node-graph libraries: three.js builds nearly
# the whole TSL surface out of exported consts, so `positionLocal`, `vec3`,
# `float`, `uniform`, `mix` and `screenUV` had ZERO definitions in the index
# while carrying 57, 835, 1916, 823, 420 and 32 references respectively.
# find_definition_opt answered "no definition found" for the most-used API in
# the corpus, and any check of "does this symbol exist" would have called a
# perfectly real import a hallucination.
#
# Only EXPORTED bindings are taken. Every local `const` would bury the API
# surface under loop counters, which is the opposite of the point.
_VALUE_DECLS = ("lexical_declaration", "variable_declaration")


def _exported_bindings(node) -> list[tuple[str, str]]:
    """[(name, kind)] for `export const x = ...`, including destructuring."""
    out: list[tuple[str, str]] = []
    for decl in node.named_children:
        if decl.type not in _VALUE_DECLS:
            continue
        kw = decl.text[:5].decode("utf-8", "replace")
        kind = "const" if kw.startswith("const") else "binding"
        for d in decl.named_children:
            if d.type != "variable_declarator":
                continue
            target = d.child_by_field_name("name") or (
                d.named_children[0] if d.named_children else None)
            if target is None:
                continue
            if target.type in NAME_NODE_TYPES:
                out.append((target.text.decode("utf-8", "replace"), kind))
            else:
                # `export const { a, b } = obj` -- each bound name is API too.
                # Destructuring binds through object_pattern, whose children
                # are shorthand_property_identifier_pattern rather than plain
                # identifiers, so the NAME_NODE_TYPES list does not cover them.
                # Matching on "identifier" catches every pattern variant
                # without enumerating each grammar's spelling.
                stack = list(target.named_children)
                while stack:
                    c = stack.pop()
                    if "identifier" in c.type:
                        out.append((c.text.decode("utf-8", "replace"), kind))
                    else:
                        stack.extend(c.named_children)
    return out


def extract(path: str, rel: str) -> tuple[list[tuple], list[tuple]]:
    """Return (definitions, references) rows for one file."""
    lang = LANG_BY_EXT.get(os.path.splitext(path)[1].lower())
    if not lang:
        return [], []
    try:
        with open(path, "rb") as f:
            src = f.read()
        tree = _parser(lang).parse(src)
    except Exception:
        return [], []

    lines = src.decode("utf-8", "replace").splitlines()
    defs: list[tuple] = []
    refs: list[tuple] = []
    seen_def_names: set[tuple[str, int]] = set()

    def line_text(i: int) -> str:
        return lines[i].strip()[:200] if 0 <= i < len(lines) else ""

    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        t = n.type
        tl = t.lower()

        if t == "export_statement":
            for nm, bkind in _exported_bindings(n):
                if nm and (nm, n.start_point[0]) not in seen_def_names:
                    seen_def_names.add((nm, n.start_point[0]))
                    defs.append((nm, bkind, rel, n.start_point[0] + 1,
                                 n.end_point[0] + 1,
                                 line_text(n.start_point[0])))

        kind = _kind(t)
        if kind:
            nm = _name_of(n)
            if nm and (nm, n.start_point[0]) not in seen_def_names:
                seen_def_names.add((nm, n.start_point[0]))
                defs.append((nm, kind, rel, n.start_point[0] + 1,
                             n.end_point[0] + 1, line_text(n.start_point[0])))

        elif "call" in tl and "callable" not in tl:
            nm = _callee(n)
            if nm:
                refs.append((nm, "call", rel, n.start_point[0] + 1,
                             line_text(n.start_point[0])))

        elif t in NAME_NODE_TYPES:
            nm = n.text.decode("utf-8", "replace")
            if nm.isidentifier() and len(nm) > 2:
                refs.append((nm, "ref", rel, n.start_point[0] + 1,
                             line_text(n.start_point[0])))

        stack.extend(n.named_children)

    return defs, refs


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS defs(
        name TEXT, kind TEXT, path TEXT, start INT, end INT, line TEXT);
    CREATE TABLE IF NOT EXISTS refs(
        name TEXT, kind TEXT, path TEXT, line_no INT, line TEXT);
    CREATE INDEX IF NOT EXISTS idx_defs_name ON defs(name);
    CREATE INDEX IF NOT EXISTS idx_refs_name ON refs(name);
    CREATE INDEX IF NOT EXISTS idx_refs_kind ON refs(name, kind);
    """)


def find_definition(con: sqlite3.Connection, symbol: str, limit: int = 20):
    return con.execute(
        "SELECT name,kind,path,start,end,line FROM defs WHERE name=? "
        "ORDER BY CASE kind WHEN 'function' THEN 0 WHEN 'method' THEN 1 ELSE 2 END, path "
        "LIMIT ?", (symbol, limit)).fetchall()


def find_references(con: sqlite3.Connection, symbol: str, calls_only: bool = False,
                    limit: int = 60):
    q = ("SELECT name,kind,path,line_no,line FROM refs WHERE name=? "
         + ("AND kind='call' " if calls_only else "")
         + "ORDER BY path, line_no LIMIT ?")
    return con.execute(q, (symbol, limit)).fetchall()
