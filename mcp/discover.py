#!/usr/bin/env python
"""Work out what a caller is using from the code they send, not from where they are.

WHY NOT DETECT A DIRECTORY

The first version of this asked "which repository is the caller in?" and looked
for a working directory in the conversation. That question only has an answer
when the caller runs on the same machine as the server. Yamadori is a remote
service. It never sees the caller's filesystem, it is never handed a path it
can open, and a fallback that trusted `127.0.0.1` described a deployment that
does not exist.

WHAT ACTUALLY ARRIVES

Code. Any request that needs help with code contains code, and code states its
own dependencies at the top of the file:

    import { createWorld } from 'koota'
    import { vec3, Fn } from 'three/tsl'
    import tgpu from 'typegpu'

That is a dependency manifest the caller sends for free, in every message,
with no configuration and no question asked.

WHY TREE-SITTER AND NOT A REGEX

The first version matched imports with regular expressions and was wrong
within one sample: the Python rule fired on the JavaScript line
`import tgpu from 'typegpu'` and invented a package called `tgpu`. A lookahead
fixed that case and would have been wrong again on the next one, because a
regex cannot see that a line sits inside a string or a comment, and cannot
tell a JavaScript `import` from a Python one at all.

A parser knows. `import_statement` is a node type, its specifier is a field on
that node, and a string that merely contains the word "import" is a `string`
node that never matches. The indexer in `scripts/symbols.py` has parsed this
way from the start; this module was the odd one out.

Pasted code is usually a fragment, which tree-sitter handles: it parses what it
can and marks the rest ERROR, and imports at the top of a fragment parse
cleanly whatever follows them.

WHAT IT STILL CANNOT SAY

The version. `import 'three'` is true of every three.js release ever made, and
0.150 to 0.185 is the difference between a working answer about TSL and a
confident wrong one. Versions are read when a caller pastes a manifest, and
otherwise asked for -- see `nebari.wants_version`.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

# Node types that carry a module specifier, per grammar. Taken from the
# grammars rather than guessed.
_IMPORT_NODES = {
    "typescript": ("import_statement", "export_statement", "call_expression"),
    "tsx": ("import_statement", "export_statement", "call_expression"),
    "javascript": ("import_statement", "export_statement", "call_expression"),
    "python": ("import_statement", "import_from_statement"),
    "rust": ("use_declaration", "extern_crate_declaration"),
}

# Fence info string -> grammar. A caller writes ```ts, or ```tsx, or nothing.
_FENCE_LANG = {
    "ts": "typescript", "typescript": "typescript", "mts": "typescript",
    "tsx": "tsx", "jsx": "tsx",
    "js": "javascript", "javascript": "javascript", "mjs": "javascript",
    "py": "python", "python": "python",
    "rs": "rust", "rust": "rust",
}

_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.S)

# Standard libraries are not dependencies worth indexing, and including them
# would put `fs` and `path` at the top of every discovery.
_BUILTIN = {
    "fs", "path", "os", "url", "util", "events", "stream", "buffer", "crypto",
    "http", "https", "net", "zlib", "child_process", "worker_threads", "assert",
    "process", "readline", "tty", "dns", "cluster", "perf_hooks", "timers",
    "sys", "re", "json", "time", "math", "typing", "dataclasses", "itertools",
    "collections", "functools", "subprocess", "std", "core", "alloc", "self",
    "crate", "super",
}

# A version written next to a package name, in the forms a caller pastes:
# package.json, a lockfile line, Cargo.toml, pip freeze. This one stays a
# regex on purpose -- it reads fragments of half a dozen manifest formats,
# usually partial and quoted into prose, where there is no tree to parse.
_VERSION = re.compile(
    r"""['"]?([@\w][\w.@/-]*?)['"]?\s*[:=]\s*['"]\^?~?>?=?\s*v?(\d+\.\d+\.\d+[\w.-]*)['"]"""
    r"""|^([\w-]+)==(\d+\.\d+(?:\.\d+)?[\w.-]*)""",
    re.MULTILINE)


def package_of(spec: str) -> str | None:
    """The installable package a specifier belongs to.

    `three/tsl` and `three/webgpu` are both `three`; `@pmndrs/koota/react` is
    `@pmndrs/koota`. Subpath exports are how these libraries are normally
    consumed, so collapsing them is most of the work.
    """
    spec = (spec or "").strip().strip("'\"")
    if not spec or spec.startswith((".", "/", "#")) or ":" in spec:
        return None                      # relative, absolute, or node:/http:
    parts = spec.split("/")
    name = "/".join(parts[:2]) if spec.startswith("@") else parts[0]
    return None if name.lower() in _BUILTIN else name


def _parse(source: str, lang: str):
    try:
        from ast_chunker import _parser
        return _parser(lang).parse(source.encode("utf-8", "replace")).root_node
    except Exception:                                            # noqa: BLE001
        return None                      # grammar missing: skip, never crash


def _text(node) -> str:
    return node.text.decode("utf-8", "replace")


def _specifiers(node, lang: str, out: list[str]) -> None:
    """Walk the tree collecting module specifiers.

    Only node types that actually carry one are inspected, so a string that
    happens to contain a package name is never mistaken for an import -- the
    failure a regex cannot avoid.
    """
    if node.type in _IMPORT_NODES.get(lang, ()):
        if lang in ("typescript", "tsx", "javascript"):
            if node.type == "call_expression":
                fn = node.child_by_field_name("function")
                if fn is not None and _text(fn) in ("require", "import"):
                    args = node.child_by_field_name("arguments")
                    for c in (args.named_children if args is not None else []):
                        if c.type == "string":
                            out.append(_text(c))
            else:
                field = node.child_by_field_name("source")
                if field is not None:
                    out.append(_text(field))
        elif lang == "python":
            mod = node.child_by_field_name("module_name")
            if mod is not None:
                out.append(_text(mod).split(".")[0])
            else:                                     # plain `import a, b`
                for c in node.named_children:
                    if c.type in ("dotted_name", "aliased_import"):
                        t = c.child_by_field_name("name") or c
                        out.append(_text(t).split(".")[0])
        elif lang == "rust":
            arg = node.child_by_field_name("argument") or (
                node.named_children[0] if node.named_children else None)
            if arg is not None:
                out.append(_text(arg).split("::")[0])
    for c in node.named_children:
        _specifiers(c, lang, out)


def _blocks(text: str) -> list[tuple[str, str]]:
    """(grammar, source) for each fenced block, or the whole text if unfenced.

    A caller who pastes code with no fence is still parsed. An unfenced blob
    is tried under each grammar, and a wrong guess costs nothing: a failed
    parse yields no imports rather than wrong ones, which is the property the
    regex version did not have.
    """
    out = []
    for info, body in _FENCE.findall(text):
        out.append((_FENCE_LANG.get(info.lower(), "typescript"), body))
    if not out and ("import " in text or "use " in text or "require(" in text):
        out = [("typescript", text), ("python", text), ("rust", text)]
    return out


def imports(text: str) -> list[str]:
    """Every package named by an import, once per occurrence, in order.

    Repeats are kept deliberately. A file pulling from `three/webgpu`,
    `three/tsl` and `three/addons` is more about three.js than one importing
    it once, and de-duplicating threw that away -- an early sample scored four
    libraries at exactly 2 apiece, so the ranking carried no information.
    """
    found: list[str] = []
    seen_blocks: set[str] = set()
    for lang, body in _blocks(text):
        root = _parse(body, lang)
        if root is None:
            continue
        specs: list[str] = []
        _specifiers(root, lang, specs)
        # An unfenced blob is offered to several grammars, and more than one
        # may succeed. Count each specifier once per block, not once per
        # grammar that happened to recognise it.
        for spec in specs:
            pkg = package_of(spec)
            if not pkg:
                continue
            tag = f"{id(body)}\x00{spec}\x00{specs.index(spec)}"
            if tag in seen_blocks:
                continue
            seen_blocks.add(tag)
            found.append(pkg)
    return found


def _names_in(node, lang: str, out: dict[str, list[str]]) -> None:
    """Walk a tree collecting {package: [imported names]} (see
    imported_names)."""
    if lang in ("typescript", "tsx", "javascript") and \
            node.type in ("import_statement", "export_statement"):
        src = node.child_by_field_name("source")
        pkg = package_of(_text(src)) if src is not None else None
        if pkg:
            names = out.setdefault(pkg, [])
            stack = list(node.named_children)
            while stack:
                c = stack.pop(0)
                if c.type in ("import_specifier", "export_specifier"):
                    n = c.child_by_field_name("name")
                    if n is not None and _text(n) not in names:
                        names.append(_text(n))
                elif c.type in ("import_clause", "named_imports",
                                "export_clause"):
                    stack[:0] = list(c.named_children)
            return
    if lang in ("typescript", "tsx", "javascript") and \
            node.type == "variable_declarator":
        # const { a, b } = require('pkg')
        val = node.child_by_field_name("value")
        pat = node.child_by_field_name("name")
        if val is not None and val.type == "call_expression" and \
                pat is not None and pat.type == "object_pattern":
            fn = val.child_by_field_name("function")
            args = val.child_by_field_name("arguments")
            strs = [a for a in (args.named_children if args is not None
                                else []) if a.type == "string"]
            if fn is not None and _text(fn) == "require" and strs:
                pkg = package_of(_text(strs[0]))
                if pkg:
                    names = out.setdefault(pkg, [])
                    for c in pat.named_children:
                        if c.type == "shorthand_property_identifier_pattern" \
                                and _text(c) not in names:
                            names.append(_text(c))
    if lang == "python" and node.type == "import_from_statement":
        mod = node.child_by_field_name("module_name")
        pkg = package_of(_text(mod).split(".")[0]) if mod is not None else None
        if pkg:
            names = out.setdefault(pkg, [])
            for c in node.children_by_field_name("name"):
                t = c.child_by_field_name("name") or c
                if _text(t) not in names:
                    names.append(_text(t))
        return
    for c in node.named_children:
        _names_in(c, lang, out)


def imported_names(text: str) -> dict[str, list[str]]:
    """{package: [the names imported from it, in order]} for every import in
    `text` -- `import { Text, Font } from '@pmndrs/glyph'` gives
    {"@pmndrs/glyph": ["Text", "Font"]}. A default or namespace import names
    the package with no names. Parsed like imports() (tree-sitter, fenced
    blocks by their tag, an unfenced blob by each grammar); a failed parse
    yields nothing rather than a guess. Used by the proxy's library-use
    injection (#19, docs/SELF-IMPROVEMENT-LOG.md)."""
    out: dict[str, list[str]] = {}
    for lang, body in _blocks(text or ""):
        root = _parse(body, lang)
        if root is None:
            continue
        found: dict[str, list[str]] = {}
        _names_in(root, lang, found)
        for pkg, names in found.items():
            cur = out.setdefault(pkg, [])
            cur.extend(n for n in names if n not in cur)
    return out


def _text_of(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ""


def versions(text: str) -> dict[str, str]:
    """Versions stated anywhere in the text, keyed by package."""
    out: dict[str, str] = {}
    for m in _VERSION.finditer(text):
        name = m.group(1) or m.group(3)
        ver = m.group(2) or m.group(4)
        if name and ver and package_of(name):
            out.setdefault(name, ver)
    return out


def scan(messages: list[dict]) -> dict:
    """Everything the conversation reveals about the caller's dependencies.

    Weighted by recency: what the caller pasted in their latest message
    describes what they are working on better than something twenty turns ago.
    """
    counts: dict[str, int] = {}
    vers: dict[str, str] = {}
    local = False
    for i, m in enumerate(messages):
        if m.get("role") not in ("user", "system", "tool"):
            continue
        text = _text_of(m)
        if not text:
            continue
        weight = 1 + (i >= len(messages) - 3)      # the last few turns count double
        for pkg in imports(text):
            counts[pkg] = counts.get(pkg, 0) + weight
        vers.update(versions(text))
        if re.search(r"""['"]\.\.?/""", text):
            local = True                            # relative imports: a real project
    ranked = sorted(counts, key=lambda p: (-counts[p], p))
    return {"packages": ranked, "counts": counts, "versions": vers,
            "has_project": local}


if __name__ == "__main__":
    import json
    print(json.dumps(scan([{"role": "user", "content": sys.stdin.read()}]), indent=2))
