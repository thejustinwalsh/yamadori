"""THE RUSTWASM PITFALL RULES (bench/skills/pitfalls/rustwasm.jsonl): Rust ->
WebAssembly with wasm-bindgen, and Rust exposing a C ABI with a
cbindgen-generated header. Loaded BY bench/skills/pitfall_harness.py
(`_load_rule_modules`); each `r_rustwasm_<name>(c) -> bool` reads a
pitfall_harness.Code: Rust blocks by tree-sitter's rust grammar
(`c.nodes(types, lang="rust")`), Cargo.toml by tomllib (`c.toml_of`).

What each rule checks is what the pinned docs state:
  wasm-bindgen guide, wasm-bindgen/wasm-bindgen@165586f854c8 guide/src/**
  cbindgen docs.md,   mozilla/cbindgen@2b757a22bab7
(bench/skills/pitfalls/sources/<slug>/). No Rust toolchain runs here: the
checks are syntax-tree checks, never a compile.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------- helpers ---
_ATTR_SKIP = ("attribute_item", "line_comment", "block_comment")


def _txt(src: bytes, n) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "replace")


def _walk(n, types):
    out, stack = [], [n]
    while stack:
        x = stack.pop()
        if x.type in types:
            out.append(x)
        stack.extend(x.children)
    return out


def _attrs(src: bytes, item) -> list[str]:
    """The attribute texts written on `item` (`#[...]` lines before it,
    doc comments between them skipped): e.g. "wasm_bindgen(catch)",
    "repr(C)", "unsafe(no_mangle)"."""
    out = []
    p = item.prev_sibling
    while p is not None and p.type in _ATTR_SKIP:
        if p.type == "attribute_item":
            for a in p.named_children:
                if a.type == "attribute":
                    out.append(re.sub(r"\s+", "", _txt(src, a)))
        p = p.prev_sibling
    return out


def _is_wb(attr: str) -> bool:
    return attr == "wasm_bindgen" or attr.startswith("wasm_bindgen(") or \
        attr.startswith("wasm_bindgen::prelude::wasm_bindgen")


def _wb_exports(c):
    """(src, function_item) of every function exported with
    #[wasm_bindgen]: on the fn itself, or a `pub fn` of an impl block
    carrying #[wasm_bindgen]."""
    for _b, src, n in c.nodes({"function_item"}, lang="rust"):
        par = n.parent
        if any(_is_wb(a) for a in _attrs(src, n)) and not (
                par is not None and par.type == "declaration_list"
                and par.parent is not None
                and par.parent.type == "foreign_mod_item"):
            yield src, n
            continue
        if par is not None and par.type == "declaration_list" and \
                par.parent is not None and par.parent.type == "impl_item" \
                and any(_is_wb(a) for a in _attrs(src, par.parent)) and \
                any(k.type == "visibility_modifier" for k in n.children):
            yield src, n


def _wb_imports(c):
    """(src, function_signature_item, its attribute texts) of every JS
    import: a fn declared in a #[wasm_bindgen] extern "C" { ... } block."""
    for _b, src, n in c.nodes({"foreign_mod_item"}, lang="rust"):
        if not any(_is_wb(a) for a in _attrs(src, n)):
            continue
        body = n.child_by_field_name("body")
        if body is None:
            continue
        for f in body.named_children:
            if f.type == "function_signature_item":
                yield src, f, _attrs(src, f)


def _ret(src, fn) -> str:
    r = fn.child_by_field_name("return_type")
    return re.sub(r"\s+", "", _txt(src, r)) if r is not None else ""


def _is_result(ret: str) -> bool:
    # `Result<T, E>`, a path to it, or a crate alias named ...Result<T>
    return bool(re.match(r"^([\w:]*::)?\w*Result(<|$)", ret))


# ------------------------------------ 1. exported fn panics, no Result ---
# catch-unwind.md: "By default, when a Rust function exported to
# JavaScript panics, Rust will abort and any allocated resources will be
# leaked."; types/result.md: an exported fn returning Result throws its Err
# as a JS exception. The panicking forms a model writes: panic!-family
# macros, .unwrap()/.expect(), and wasm_bindgen::throw_str/throw_val (a
# throw that skips Rust's destructors, catch.md).
_PANIC_MACROS = {"panic", "unreachable", "todo", "unimplemented", "assert",
                 "assert_eq", "assert_ne"}


def _panics_in(src, body) -> bool:
    for m in _walk(body, {"macro_invocation"}):
        name = m.child_by_field_name("macro")
        if name is not None and _txt(src, name).split("::")[-1] in \
                _PANIC_MACROS:
            return True
    for call in _walk(body, {"call_expression"}):
        f = call.child_by_field_name("function")
        if f is None:
            continue
        if f.type == "field_expression":
            fld = f.child_by_field_name("field")
            if fld is not None and _txt(src, fld) in ("unwrap", "expect"):
                return True
        if _txt(src, f).split("::")[-1] in ("throw_str", "throw_val"):
            return True
    return False


def r_rustwasm_export_panics(c) -> bool:
    """A #[wasm_bindgen] export that does not return Result and panics
    (or throws past Rust's destructors) on bad input."""
    for src, fn in _wb_exports(c):
        body = fn.child_by_field_name("body")
        if body is not None and not _is_result(_ret(src, fn)) and \
                _panics_in(src, body):
            return True
    return False


def r_rustwasm_export_returns_result(c) -> bool:
    """A #[wasm_bindgen] export returns Result<T, E> (Err thrown in JS)."""
    return any(_is_result(_ret(src, fn)) for src, fn in _wb_exports(c))


# ------------------------------- 2. throwing JS import without catch ---
# attributes/on-js-imports/catch.md; types/result.md: "if you import a JS
# function with `Result` you need `#[wasm_bindgen(catch)]`". The case's
# import is `loadSettings` (the prompt names it); its Rust name may be an
# alias with js_name.
def _names_load_settings(src, f, attrs) -> bool:
    nm = f.child_by_field_name("name")
    names = [_txt(src, nm) if nm is not None else ""] + attrs
    return any("loadsettings" in x.lower().replace("_", "") for x in names)


def _has_catch(attrs) -> bool:
    return any(_is_wb(a) and re.search(r"[(,]catch[,)]", a) for a in attrs)


def r_rustwasm_throwing_import_no_catch(c) -> bool:
    """`loadSettings` imported without #[wasm_bindgen(catch)] (a JS throw
    then crosses Rust uncaught)."""
    return any(_names_load_settings(src, f, a) and not _has_catch(a)
               for src, f, a in _wb_imports(c))


def r_rustwasm_throwing_import_catch(c) -> bool:
    """`loadSettings` imported with #[wasm_bindgen(catch)] returning
    Result<_, JsValue>."""
    return any(_names_load_settings(src, f, a) and _has_catch(a)
               and _is_result(_ret(src, f)) for src, f, a in _wb_imports(c))


# ------------------------------------ 3. serde into JsValue ---------------
# reference/arbitrary-data-with-serde.md: serde-wasm-bindgen's
# to_value/from_value; the JSON-based JsValue::from_serde/into_serde were
# moved to gloo-utils and "the originals were deprecated". The deprecated
# path: from_serde/into_serde with no gloo_utils import, or wasm-bindgen's
# old `serde-serialize` feature.
def _deps_tables(t: dict):
    for k in ("dependencies", "dev-dependencies", "build-dependencies"):
        if isinstance(t.get(k), dict):
            yield k, t[k]
    for tv in (t.get("target") or {}).values():
        if isinstance(tv, dict):
            for k in ("dependencies", "build-dependencies"):
                if isinstance(tv.get(k), dict):
                    yield k, tv[k]


def _dep(c, name: str, kind: str = "dependencies"):
    """The Cargo.toml entry of dependency `name` ("" when absent; a string
    version or a table)."""
    for t in c.toml_of("Cargo.toml"):
        for k, d in _deps_tables(t):
            if k == kind and name in d:
                return d[name]
    return None


def _features(dep) -> list[str]:
    return list(dep.get("features") or []) if isinstance(dep, dict) else []


def r_rustwasm_serde_json_jsvalue(c) -> bool:
    code = c.all_code("rust")
    old = re.search(r"\bfrom_serde\s*\(|\.into_serde\s*(::<[^>]*>)?\s*\(",
                    code) and "gloo_utils" not in code
    wb = _dep(c, "wasm-bindgen")
    return bool(old) or "serde-serialize" in _features(wb)


def r_rustwasm_serde_wasm_bindgen(c) -> bool:
    """serde_wasm_bindgen::to_value / from_value (a path call, or the
    function imported from serde_wasm_bindgen)."""
    code = c.all_code("rust")
    if "serde_wasm_bindgen" not in code:
        return False
    for _b, src, n in c.nodes({"call_expression"}, lang="rust"):
        f = n.child_by_field_name("function")
        if f is not None and re.search(
                r"(^|::)(to_value|from_value)$", _txt(src, f)) and (
                "serde_wasm_bindgen::" in _txt(src, f) or re.search(
                    r"use\s+serde_wasm_bindgen::\{?[^;]*\b(to_value|"
                    r"from_value)\b", code)):
            return True
    return False


# ------------------------------------------ 4. web-sys cargo features ---
# web-sys/cargo-features.md: "there is a cargo feature for every type
# defined in `web-sys`. To access that type, you must enable its feature."
# The types a Rust block names from web_sys (paths web_sys::X, `use
# web_sys::{X, ...}`), plus the ones the calls the case needs imply, each
# listed in the docs' per-method feature list: web_sys::window() -> Window,
# .document() -> Document, .get_element_by_id() -> Element.
_IMPLIED = {"window": "Window", "document": "Document",
            "get_element_by_id": "Element", "create_element": "Element"}


def _web_sys_types(c) -> set[str]:
    """The web-sys features the Rust needs: every interface type it names
    from web_sys, the `console` namespace (feature "console"), and the
    types the calls in _IMPLIED return."""
    code = c.all_code("rust")
    need = set(re.findall(r"\bweb_sys::([A-Z]\w*)", code))
    if re.search(r"\bweb_sys::console\b", code):
        need.add("console")
    for m in re.finditer(r"use\s+web_sys::\{([^}]*)\}", code):
        need |= {x.strip().split(" ")[0] for x in m.group(1).split(",")
                 if re.match(r"\s*([A-Z]|console\b)", x)}
    for _b, src, n in c.nodes({"call_expression"}, lang="rust"):
        f = n.child_by_field_name("function")
        if f is None:
            continue
        t = _txt(src, f)
        last = t.split("::")[-1].split(".")[-1]
        if f.type == "field_expression":
            fld = f.child_by_field_name("field")
            last = _txt(src, fld) if fld is not None else last
            if last in _IMPLIED and last != "window":
                need.add(_IMPLIED[last])
        elif t.endswith("web_sys::window") or t == "window" and \
                "web_sys::window" in code:
            need.add("Window")
    return need


def r_rustwasm_web_sys_features_missing(c) -> bool:
    """The Rust uses web-sys types whose cargo features Cargo.toml does not
    enable (or web-sys is not a dependency at all)."""
    need = _web_sys_types(c)
    if not need:
        return False
    dep = _dep(c, "web-sys")
    return not need <= set(_features(dep))


def r_rustwasm_web_sys_features_enabled(c) -> bool:
    need = _web_sys_types(c)
    dep = _dep(c, "web-sys")
    return bool(need) and dep is not None and need <= set(_features(dep))


# ---------------------------------------- 5. repr(C) across the C ABI ---
# cbindgen docs.md "Supported Types": "Most things in Rust don't have a
# guaranteed layout by default" ... "`#[repr(C)]`: give this
# struct/union/enum the same layout and ABI C would".
def _c_exports(c):
    """(src, function_item) of every `extern "C"` / `extern` fn."""
    for _b, src, n in c.nodes({"function_item"}, lang="rust"):
        mods = [k for k in n.children if k.type == "function_modifiers"]
        if not mods:
            continue
        m = re.search(r'\bextern\b\s*("([^"]*)")?', _txt(src, mods[0]))
        if m and (m.group(2) is None or m.group(2).startswith("C")):
            yield src, n


def _by_value_types(src, fn) -> set[str]:
    """Type names a C export takes or returns BY VALUE (not behind a
    pointer or reference)."""
    out = set()
    ps = fn.child_by_field_name("parameters")
    tys = [p.child_by_field_name("type") for p in (
        ps.named_children if ps is not None else []) if p.type == "parameter"]
    tys.append(fn.child_by_field_name("return_type"))
    for t in tys:
        if t is not None and t.type == "type_identifier":
            out.add(_txt(src, t))
    return out


def _structs(c) -> dict:
    """{name: [attribute texts]} of every struct / enum / union."""
    out = {}
    for _b, src, n in c.nodes({"struct_item", "enum_item", "union_item"},
                              lang="rust"):
        nm = n.child_by_field_name("name")
        if nm is not None:
            out[_txt(src, nm)] = _attrs(src, n)
    return out


def _guaranteed(attrs) -> bool:
    return any(re.match(r"^repr\((C|transparent|u\d+|i\d+)\b", a)
               for a in attrs)


def _ffi_by_value(c):
    st = _structs(c)
    for src, fn in _c_exports(c):
        for t in _by_value_types(src, fn):
            if t in st:
                yield t, st[t]


def r_rustwasm_ffi_struct_no_repr(c) -> bool:
    """A struct passed by value across an extern "C" fn has no guaranteed
    layout (#[repr(C)] / transparent / an integer repr for an enum)."""
    return any(not _guaranteed(a) for _t, a in _ffi_by_value(c))


def r_rustwasm_ffi_struct_repr_c(c) -> bool:
    got = list(_ffi_by_value(c))
    return bool(got) and all(_guaranteed(a) for _t, a in got)


# ------------------------------------ 6. SETUP: a wasm-bindgen crate ---
# examples/hello-world.md: "The `Cargo.toml` lists the `wasm-bindgen` crate
# as a dependency. Also of note is the `crate-type = ["cdylib"]` which is
# largely used for wasm final artifacts today."
def _crate_types(c) -> list[str]:
    out = []
    for t in c.toml_of("Cargo.toml"):
        lib = t.get("lib") if isinstance(t.get("lib"), dict) else {}
        out += list(lib.get("crate-type") or lib.get("crate_type") or [])
    return out


def r_rustwasm_no_cdylib(c) -> bool:
    return "cdylib" not in _crate_types(c)


def r_rustwasm_wasm_crate_configured(c) -> bool:
    """Cargo.toml: [lib] crate-type with "cdylib" and wasm-bindgen as a
    dependency; the Rust exports a fn with #[wasm_bindgen]."""
    return "cdylib" in _crate_types(c) and \
        _dep(c, "wasm-bindgen") is not None and \
        any(True for _ in _wb_exports(c))


# --------------------------- 7. SETUP: a C ABI with a generated header ---
# cbindgen docs.md: the header by hand is "much more likely to be
# error-prone than machine-generated headers"; the build.rs form
# (cbindgen::Builder::new().with_crate(..).generate()..write_to_file(..),
# or cbindgen::generate(..)), with cbindgen under [build-dependencies]; and
# cbindgen walks the crate for "`#[no_mangle] pub extern fn`".
def _build_rs(c):
    for b, src, t in c.rust:
        if b["name"].endswith("build.rs") or (
                not b["name"] and "cbindgen::" in _txt(src, t.root_node)
                and re.search(r"\bfn\s+main\s*\(", _txt(src, t.root_node))):
            yield b, src, t


def r_rustwasm_cbindgen_configured(c) -> bool:
    if _dep(c, "cbindgen", "build-dependencies") is None:
        return False
    for _b, src, t in _build_rs(c):
        code = _txt(src, t.root_node)
        builder = re.search(r"\bBuilder::new\s*\(", code) and \
            re.search(r"\.generate\s*\(", code) and \
            re.search(r"\.write_to_file\s*\(", code) and "cbindgen" in code
        # cbindgen::generate(dir) (docs.md, with a cbindgen.toml) and its
        # explicit-config form generate_with_config(dir, config)
        gen = re.search(r"\bcbindgen::generate(_with_config)?\s*\(",
                        code) and \
            re.search(r"\.write_to_file\s*\(", code)
        if builder or gen:
            return True
    return False


def r_rustwasm_header_by_hand(c) -> bool:
    return not r_rustwasm_cbindgen_configured(c)


def _no_mangle(attrs) -> bool:
    return any(re.match(r"^(unsafe\()?no_mangle\)?$", a) for a in attrs)


def r_rustwasm_c_exports_no_mangle(c) -> bool:
    """At least one `pub extern "C" fn`, and every one is #[no_mangle]
    (or #[unsafe(no_mangle)]), outside build.rs."""
    got = [(src, fn) for src, fn in _c_exports(c)
           if any(k.type == "visibility_modifier" for k in fn.children)]
    return bool(got) and all(_no_mangle(_attrs(src, fn)) for src, fn in got)
