"""Tree-sitter chunking: split source on real syntax nodes, not line windows.

Why this matters more than any other retrieval knob: an embedding of half a
function matches poorly against a question about what that function does.
Character-window chunking routinely cuts mid-body, and every downstream stage
(vector search, rerank, the model reading the snippet) inherits that damage.

Rather than hardcode node types per grammar, we accept any named node whose
type looks like a definition. Grammars are remarkably consistent about this:
  rust  function_item, struct_item, impl_item, trait_item
  cpp   function_definition, class_specifier, struct_specifier
  ts    function_declaration, class_declaration, interface_declaration
  zig   FnProto / variable declarations holding struct/enum
  wgsl  function_declaration, struct_declaration
so a suffix match generalises without a per-language table.

Container nodes (impl blocks, classes, namespaces) are recursed into so that
individual methods become their own chunks, each still carrying its parent's
signature line as context.
"""
from __future__ import annotations

import os

# extension -> tree-sitter language name
LANG_BY_EXT = {
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".hh": "cpp", ".cxx": "cpp",
    ".zig": "zig",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".py": "python",
    ".go": "go",
    ".swift": "swift",
    ".kt": "kotlin",
    ".java": "java",
    ".cs": "csharp",
    ".m": "objc", ".mm": "objc",
    ".wgsl": "wgsl",
    ".glsl": "glsl", ".vert": "glsl", ".frag": "glsl", ".comp": "glsl",
    ".hlsl": "hlsl",
}

# A node is a "definition" if its type ends with / contains one of these.
DEF_MARKERS = (
    "function", "method", "constructor",
    "class", "struct", "enum", "union", "interface", "protocol",
    "impl", "trait", "namespace", "module",
    "type_alias", "type_definition", "type_declaration",
    "fnproto", "fn_proto",          # zig
    "decl",                          # wgsl / glsl style *_declaration
)

# Nodes we descend into looking for nested definitions.
# "export" matters for TS/JS: `export function f()` parses as an
# export_statement wrapping the declaration, which matches neither the
# definition markers nor the structural ones, so the whole file would be
# skipped without it.
CONTAINER_MARKERS = (
    "impl", "class", "namespace", "module", "trait", "interface",
    "declaration_list", "field_declaration_list", "body", "block",
    "source_file", "translation_unit", "program", "export",
)

# Types that are BOTH a definition and a container. These must recurse first:
# emitting a whole C++ namespace or Rust impl block as one chunk buries the
# individual methods that a search is usually looking for. We only emit the
# node itself when it turns out to hold no nested definitions.
RECURSE_FIRST = ("impl", "namespace", "module", "class", "trait", "interface")

MAX_CHUNK_LINES = 120
MIN_CHUNK_CHARS = 24

_parsers: dict[str, object] = {}


def _parser(lang: str):
    if lang not in _parsers:
        from tree_sitter_language_pack import get_parser
        _parsers[lang] = get_parser(lang)
    return _parsers[lang]


def _is_structural(t: str) -> bool:
    """Pure container nodes that hold declarations but are not one themselves.

    Checked BEFORE _is_def, because several of these would otherwise match a
    definition marker and be emitted whole: C++ `declaration_list` (the body of
    a namespace) contains "decl", so without this the entire namespace becomes
    one chunk and the classes inside it are never individually searchable.
    """
    tl = t.lower()
    return (tl.endswith(("_list", "_body", "_block", "_identifier", "_name"))
            or tl in ("identifier", "type_identifier", "field_identifier")
            or tl in ("body", "block", "source_file", "translation_unit",
                      "program", "compilation_unit")
            or tl.startswith("export"))


def _is_def(t: str) -> bool:
    if _is_structural(t):
        return False
    tl = t.lower()
    return any(m in tl for m in DEF_MARKERS)


def _is_container(t: str) -> bool:
    if _is_structural(t):
        return True
    tl = t.lower()
    return any(m in tl for m in CONTAINER_MARKERS)


def _collect(node, src: bytes, out: list, depth: int = 0) -> None:
    """Walk the tree, emitting (start_line, end_line) for definition nodes."""
    if depth > 6:
        return
    for child in node.named_children:
        t = child.type
        tl = t.lower()
        s, e = child.start_point[0], child.end_point[0]
        n_lines = e - s + 1

        if _is_def(t):
            # Container-definitions (impl/class/namespace/...) recurse first so
            # their members become individually searchable.
            if any(m in tl for m in RECURSE_FIRST):
                before = len(out)
                _collect(child, src, out, depth + 1)
                if len(out) == before:
                    out.append((s, min(e, s + MAX_CHUNK_LINES - 1)))
                continue

            if n_lines <= MAX_CHUNK_LINES:
                out.append((s, e))
                continue

            # Oversized plain definition: prefer its nested defs over one blob.
            before = len(out)
            _collect(child, src, out, depth + 1)
            if len(out) == before:
                for w in range(s, e + 1, MAX_CHUNK_LINES):
                    out.append((w, min(w + MAX_CHUNK_LINES - 1, e)))
        elif _is_container(t):
            _collect(child, src, out, depth + 1)


def chunk_with_ast(path: str) -> list[tuple[int, int, str]] | None:
    """Return [(start_line_1indexed, end_line, text)] or None if unsupported."""
    lang = LANG_BY_EXT.get(os.path.splitext(path)[1].lower())
    if not lang:
        return None
    try:
        with open(path, "rb") as f:
            src = f.read()
        tree = _parser(lang).parse(src)
    except Exception:
        return None

    if tree.root_node.has_error and not tree.root_node.named_children:
        return None

    spans: list[tuple[int, int]] = []
    _collect(tree.root_node, src, spans)
    if not spans:
        return None

    lines = src.decode("utf-8", errors="replace").splitlines()
    spans.sort()

    # Attach leading comments/attributes to the definition that follows them:
    # a doc comment is often the most semantically useful text in the chunk.
    out: list[tuple[int, int, str]] = []
    for s, e in spans:
        start = s
        j = s - 1
        while j >= 0:
            stripped = lines[j].strip()
            if stripped.startswith(("//", "#", "/*", "*", "///", "--")) or \
               stripped.startswith(("#[", "@", "[[")):
                start = j
                j -= 1
            elif not stripped and start != s:
                j -= 1        # allow a blank line inside a comment block
            else:
                break
        text = "\n".join(lines[start:e + 1]).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            out.append((start + 1, e + 1, text))

    # de-duplicate identical spans produced by overlapping container walks
    seen, uniq = set(), []
    for s, e, t in out:
        if (s, e) not in seen:
            seen.add((s, e))
            uniq.append((s, e, t))
    return uniq or None
