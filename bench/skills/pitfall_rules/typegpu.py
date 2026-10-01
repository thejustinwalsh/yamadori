"""THE TYPEGPU PITFALL RULES (bench/skills/pitfalls/typegpu.jsonl), loaded by
bench/skills/pitfall_harness.py. Each `r_typegpu_<name>(c) -> bool` reads
the answer's code (`c`, a pitfall_harness.Code) by syntax tree where it can.

Every pattern comes from the TypeGPU docs pinned at the v0.12.5 release tag
(bench/skills/pitfalls/sources/typegpu@57d442ad1a8e, MIT): schemas and
root.createBuffer instead of byte-sized raw buffers (fundamentals/your-first-
gpu-program), buffer.write with plain values instead of hand-packed typed
arrays (apis/buffers), one root for the program (apis/roots),
root.configureContext (apis/roots), 'use gpu' functions and root pipelines
instead of WGSL strings (apis/functions), and the project setup
(getting-started, tooling/unplugin-typegpu). The name lists below (which
calls count as dispatch work, which typed arrays count as hand packing) are
UNMEASURED choices of what a model plausibly writes.
"""
from __future__ import annotations

import re

_FN_TYPES = {"function_declaration", "function_expression", "arrow_function",
             "method_definition", "generator_function_declaration",
             "function"}
_LOOP_TYPES = {"for_statement", "for_in_statement", "while_statement",
               "do_statement"}
# UNMEASURED: the typed-array / byte-buffer constructors a hand packer uses.
_BYTE_CTORS = {"Float32Array", "Float64Array", "Uint32Array", "Int32Array",
               "Uint16Array", "Int16Array", "Uint8Array", "Int8Array",
               "ArrayBuffer", "DataView"}
# UNMEASURED: the calls that submit GPU work in TypeGPU and raw WebGPU.
_WORK = {"dispatchThreads", "dispatchWorkgroups", "dispatchWorkgroupsIndirect",
         "draw", "drawIndexed", "drawIndirect", "drawIndexedIndirect",
         "submit"}


# ------------------------------------------------------------- helpers ---
def _last(callee: str) -> str:
    return re.split(r"[.?]", callee)[-1] if callee else ""


def _args(call):
    a = call.child_by_field_name("arguments")
    return list(a.named_children) if a is not None else []


def _obj_keys(c, src, obj) -> set:
    """The keys of an object literal's own pairs (not nested ones)."""
    out = set()
    if obj is None or obj.type != "object":
        return out
    for p in obj.named_children:
        if p.type == "pair":
            k = p.child_by_field_name("key")
            if k is not None:
                out.add(c.text(src, k).strip("'\""))
        elif p.type == "shorthand_property_identifier":
            out.add(c.text(src, p))
    return out


def _enclosing_fn(node):
    p = node.parent
    while p is not None:
        if p.type in _FN_TYPES:
            return p
        p = p.parent
    return None


def _new_ctor(c, src, node) -> str:
    """`new X(...)` -> X, else ""."""
    if node is not None and node.type == "new_expression":
        k = node.child_by_field_name("constructor")
        return c.text(src, k) if k is not None else ""
    return ""


def _byte_ids(c) -> set:
    """Names bound to `new Float32Array(...)` / ArrayBuffer / DataView (a
    declarator or an assignment), and `x.buffer` of one."""
    ids = set()
    for _b, src, n in c.nodes({"variable_declarator"}):
        nm = n.child_by_field_name("name")
        v = n.child_by_field_name("value")
        if nm is not None and _new_ctor(c, src, v) in _BYTE_CTORS:
            ids.add(c.text(src, nm))
    for _b, src, n in c.nodes({"assignment_expression"}):
        left = n.child_by_field_name("left")
        right = n.child_by_field_name("right")
        if left is not None and _new_ctor(c, src, right) in _BYTE_CTORS:
            ids.add(c.text(src, left))
    return ids


def _is_bytes(c, src, arg, ids) -> bool:
    if arg is None:
        return False
    if _new_ctor(c, src, arg) in _BYTE_CTORS:
        return True
    t = c.text(src, arg)
    return t in ids or (t.endswith(".buffer") and t[:-7] in ids)


def _init_calls(c):
    """tgpu.init / tgpu.initFromDevice (any alias ending so) and raw
    adapter.requestDevice calls."""
    for b, src, n in c.nodes({"call_expression"}):
        cal = c.callee(src, n)
        if re.search(r"(^|\.)(init|initFromDevice)$", cal) and \
                re.search(r"tgpu|typegpu", cal, re.I) or \
                cal.endswith(".requestDevice"):
            yield b, src, n


# ------------------------------------------- schemas and typed buffers ---
def r_typegpu_raw_buffer(c) -> bool:
    """A raw WebGPU buffer: `device.createBuffer({ size, usage })` (a byte
    size by hand) or `queue.writeBuffer(...)`."""
    for _b, src, n in c.calls("createBuffer"):
        a = _args(n)
        if a and a[0].type == "object" and "size" in _obj_keys(c, src, a[0]):
            return True
    return c.called("writeBuffer")


def r_typegpu_struct_schema(c) -> bool:
    """A TypeGPU struct schema: `d.struct({...})` (or `struct` imported from
    typegpu/data)."""
    for _b, src, n in c.nodes({"call_expression"}):
        cal = c.callee(src, n)
        if cal == "struct" or cal.endswith(".struct"):
            a = _args(n)
            if a and a[0].type == "object":
                return True
    return False


def r_typegpu_typed_buffer(c) -> bool:
    """A buffer made from a schema: `root.createBuffer(schema, ...)` (the
    first argument is not a raw descriptor object) or
    root.createMutable / createUniform / createReadonly."""
    for _b, src, n in c.nodes({"call_expression"}):
        last = _last(c.callee(src, n))
        a = _args(n)
        if last in ("createMutable", "createUniform", "createReadonly") \
                and a:
            return True
        if last == "createBuffer" and a and a[0].type != "object":
            return True
    return False


def r_typegpu_manual_byte_packing(c) -> bool:
    """Bytes packed by hand: a DataView, `set<Type>(offset, ...)` calls, a
    raw `writeBuffer`, or a typed array / ArrayBuffer handed to a TypeGPU
    `.write` / `.patch` (copied as-is, padding and all)."""
    if c.called("writeBuffer"):
        return True
    for _b, src, n in c.nodes({"new_expression"}):
        if _new_ctor(c, src, n) == "DataView":
            return True
    for _b, src, n in c.nodes({"call_expression"}):
        if re.search(r"\.set(Float32|Uint32|Int32|Float64|Uint16|Int16"
                     r"|Uint8|Int8)$", c.callee(src, n)):
            return True
    ids = _byte_ids(c)
    for _b, src, n in c.nodes({"call_expression"}):
        if _last(c.callee(src, n)) in ("write", "patch"):
            a = _args(n)
            if a and _is_bytes(c, src, a[0], ids):
                return True
    return False


def r_typegpu_typed_write(c) -> bool:
    """`buffer.write(value)` / `buffer.patch(value)` with a plain object,
    array, schema instance or other value TypeGPU serializes itself (not a
    typed array or ArrayBuffer)."""
    ids = _byte_ids(c)
    for _b, src, n in c.nodes({"call_expression"}):
        cal = c.callee(src, n)
        if _last(cal) not in ("write", "patch") or "." not in cal:
            continue
        a = _args(n)
        if not a or _is_bytes(c, src, a[0], ids):
            continue
        if a[0].type in ("object", "array", "call_expression", "identifier",
                         "member_expression", "number", "spread_element"):
            return True
    return False


# ------------------------------------------------------------- roots ---
def r_typegpu_root_per_call(c) -> bool:
    """A root / device made where the work is done: tgpu.init (or
    requestDevice) in the same function that dispatches or draws, inside a
    loop, or inside a requestAnimationFrame callback -- a new device every
    call."""
    for _b, src, n in _init_calls(c):
        fn = _enclosing_fn(n)
        p = n.parent
        while p is not None and p is not fn:
            if p.type in _LOOP_TYPES:
                return True
            p = p.parent
        q = fn
        while q is not None:
            call = q.parent.parent if q.parent is not None else None
            if call is not None and call.type == "call_expression" and \
                    _last(c.callee(src, call)) == "requestAnimationFrame":
                return True
            q = _enclosing_fn(q)
        if fn is None:
            continue
        name = _fn_name(c, src, fn)
        for w in c.within(src, fn, {"call_expression"}):
            if _enclosing_fn(w) != fn:
                continue
            last = _last(c.callee(src, w))
            if last in _WORK:
                return True
            # the function schedules itself: a per-frame callback
            a = _args(w)
            if last == "requestAnimationFrame" and name and a and \
                    c.text(src, a[0]) == name:
                return True
    return False


def _fn_name(c, src, fn) -> str:
    nm = fn.child_by_field_name("name")
    if nm is not None:
        return c.text(src, nm)
    p = fn.parent
    if p is not None and p.type == "variable_declarator":
        nm = p.child_by_field_name("name")
        return c.text(src, nm) if nm is not None else ""
    return ""


def r_typegpu_root_once(c) -> bool:
    """A TypeGPU root is made (tgpu.init / initFromDevice) and never in the
    function that does the per-call work."""
    has = any(not c.callee(src, n).endswith(".requestDevice")
              for _b, src, n in _init_calls(c))
    return has and not r_typegpu_root_per_call(c)


def r_typegpu_manual_context_configure(c) -> bool:
    """`context.configure({ device, format, ... })` by hand."""
    for _b, src, n in c.calls("configure"):
        a = _args(n)
        if a and "device" in _obj_keys(c, src, a[0]):
            return True
    return False


def r_typegpu_configure_context(c) -> bool:
    return c.called("configureContext")


# ---------------------------------------------------------- functions ---
_WGSL_MARK = re.compile(r"@(compute|vertex|fragment|workgroup_size)\b"
                        r"|var<(storage|uniform)")


def r_typegpu_wgsl_string_shader(c) -> bool:
    """The shader as a WGSL string: `createShaderModule`, a string or
    template holding an entry point / binding (`@compute`,
    `@workgroup_size`, `var<storage>`), or a WGSL block with one."""
    if c.called("createShaderModule"):
        return True
    for _b, src, n in c.nodes({"template_string", "string"}):
        if _WGSL_MARK.search(c.text(src, n)):
            return True
    return any(b["lang"] == "wgsl" or b["name"].endswith(".wgsl")
               for b in c.raw if _WGSL_MARK.search(b["code"]))


def _use_gpu_body(c, src, fn) -> bool:
    body = fn.child_by_field_name("body")
    if body is None or body.type != "statement_block" or \
            not body.named_children:
        return False
    first = body.named_children[0]
    if first.type != "expression_statement":
        return False
    return c.text(src, first).rstrip(";").strip() in ("'use gpu'",
                                                       '"use gpu"')


def r_typegpu_tgsl_function(c) -> bool:
    """Shader logic in TypeScript: a function whose body opens with the
    'use gpu' directive, or a function (not a WGSL string) given to a
    tgpu.fn / computeFn / vertexFn / fragmentFn shell."""
    for _b, src, n in c.nodes(_FN_TYPES):
        if _use_gpu_body(c, src, n):
            return True
    for _b, src, n in c.nodes({"call_expression"}):
        f = n.child_by_field_name("function")
        if f is None or f.type != "call_expression":
            continue
        if re.search(r"(^|\.)(fn|computeFn|vertexFn|fragmentFn)$",
                     c.callee(src, f)):
            a = _args(n)
            if a and a[0].type in ("arrow_function", "function_expression",
                                   "function"):
                return True
    return False


def r_typegpu_typed_pipeline(c) -> bool:
    """A TypeGPU pipeline: root.createGuardedComputePipeline, or
    createComputePipeline / createRenderPipeline without a raw descriptor
    (no `layout`, no shader `module`)."""
    for _b, src, n in c.nodes({"call_expression"}):
        last = _last(c.callee(src, n))
        if last == "createGuardedComputePipeline":
            return True
        if last in ("createComputePipeline", "createRenderPipeline"):
            a = _args(n)
            if not a or a[0].type != "object":
                continue
            keys = set()
            for x in c.within(src, a[0], {"object"}):
                keys |= _obj_keys(c, src, x)
            if not keys & {"layout", "module"}:
                return True
    return False


# ------------------------------------------------------------- setup ---
def _deps(c) -> dict:
    out = {}
    for d in c.json_of("package.json"):
        if isinstance(d, dict):
            for k in ("dependencies", "devDependencies"):
                if isinstance(d.get(k), dict):
                    out.update(d[k])
    return out


def _vite_trees(c):
    for b, src, t in c.trees:
        if "vite.config" in b["name"] or re.search(
                r"\bdefineConfig\b", c.text(src, t.root_node)):
            yield b, src, t


def r_typegpu_build_plugin(c) -> bool:
    """unplugin-typegpu per the docs (tooling/unplugin-typegpu): listed in
    package.json AND the Vite config imports its plugin from
    'unplugin-typegpu/vite' (or /rollup) and calls it."""
    if "unplugin-typegpu" not in _deps(c):
        return False
    for _b, src, t in _vite_trees(c):
        stack, names = [t.root_node], []
        while stack:
            x = stack.pop()
            stack.extend(x.children)
            if x.type != "import_statement":
                continue
            s = x.child_by_field_name("source")
            if s is None or not re.fullmatch(
                    r"['\"]unplugin-typegpu(/(vite|rollup|rolldown))?['\"]",
                    c.text(src, s)):
                continue
            for i in c.within(src, x, {"identifier"}):
                names.append(c.text(src, i))
        for x in c.within(src, t.root_node, {"call_expression"}):
            cal = c.callee(src, x)
            if cal in names or cal.split(".")[0] in names:
                return True
    return False


def r_typegpu_build_plugin_missing(c) -> bool:
    return not r_typegpu_build_plugin(c)


def r_typegpu_webgpu_types(c) -> bool:
    """@webgpu/types per the docs (getting-started): in package.json AND
    named in tsconfig's compilerOptions.types (or typeRoots), or referenced
    with a triple-slash directive."""
    if "@webgpu/types" not in _deps(c):
        return False
    for d in c.json_of("tsconfig.json"):
        co = (d or {}).get("compilerOptions") or {}
        if "@webgpu/types" in (co.get("types") or []) or any(
                "@webgpu/types" in str(x) for x in co.get("typeRoots") or []):
            return True
    return bool(re.search(r"///\s*<reference\s+types=['\"]@webgpu/types",
                          c.all_code()))


def r_typegpu_webgpu_types_missing(c) -> bool:
    return not r_typegpu_webgpu_types(c)


def r_typegpu_root_init(c) -> bool:
    return any(not c.callee(src, n).endswith(".requestDevice")
               for _b, src, n in _init_calls(c))


_CLI = re.compile(r"(?m)^\s*(?:\$\s*)?(npx|bunx|pnpx|pnpm\s+dlx|yarn\s+dlx)"
                  r"\s+typegpu(@[\w.-]+)?(\s|$)")
_HAND = re.compile(r"(?m)(create[\s-]vite|npm\s+init\s+vite|"
                   r"(npm\s+(i|install|add)|pnpm\s+add|yarn\s+add|bun\s+add)"
                   r"\b[^\n]*\btypegpu\b)")


def r_typegpu_cli_scaffold(c) -> bool:
    """The project started with the TypeGPU CLI (getting-started:
    `npx typegpu@latest`, or its pnpm/yarn/bun forms)."""
    return any(_CLI.search(b["code"]) for b in c.blocks)


def r_typegpu_hand_scaffold(c) -> bool:
    """A project scaffolded by hand: create-vite and/or a manual
    `npm install typegpu`, with no TypeGPU CLI."""
    return not r_typegpu_cli_scaffold(c) and any(
        _HAND.search(b["code"]) for b in c.blocks)
