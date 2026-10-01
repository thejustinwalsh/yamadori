"""THE TSL (three.js Shading Language) PITFALL RULES
(bench/skills/pitfalls/tsl.jsonl; loaded by bench/skills/pitfall_harness.py).

Each rule: `r_tsl_<name>(c) -> bool`, `c` a pitfall_harness.Code. The good
patterns come from the pinned three.js r186 docs
(bench/skills/pitfalls/sources/three.js@9b4a2ac29c63: manual/pages/
webgpurenderer.html and tsl/content/Guide.md, MIT). Syntax-tree checks
(tree-sitter tsx) over the answer's TS/JS blocks AND the module scripts of
its HTML blocks (a single-file page keeps its code in
<script type="module">); regexes only for import specifiers, the import
map's JSON and GLSL markers inside strings.

Every heuristic threshold or word list below that is not from the docs is
UNMEASURED (a reading of what a model plausibly writes, not a measurement).
"""
from __future__ import annotations

import json
import re

# ------------------------------------------------------------- helpers ---


def _html_module_trees(c) -> list:
    """(src, tree) of every <script type="module"> in the HTML blocks."""
    got = getattr(c, "_tsl_html_trees", None)
    if got is not None:
        return got
    got = []
    for b in c.raw:
        if b["lang"] != "html" and not b["name"].endswith(".html"):
            continue
        for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", b["code"],
                             re.S | re.I):
            if re.search(r"type\s*=\s*['\"]?module", m.group(1), re.I):
                src = m.group(2).encode("utf-8")
                got.append((src, c.parser.parse(src)))
    c._tsl_html_trees = got
    return got


def _trees(c) -> list:
    return [(src, t) for _b, src, t in c.trees] + _html_module_trees(c)


def _nodes(c, types):
    for src, t in _trees(c):
        stack = [t.root_node]
        while stack:
            n = stack.pop()
            if n.type in types:
                yield src, n
            stack.extend(reversed(n.children))


def _txt(src, n) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "replace") \
        if n is not None else ""


def _sub(node, types) -> list:
    out, stack = [], [node]
    while stack:
        x = stack.pop()
        if x.type in types:
            out.append(x)
        stack.extend(x.children)
    return out


def _callee(src, call) -> str:
    return _txt(src, call.child_by_field_name("function"))


def _called(c, *names) -> bool:
    for src, n in _nodes(c, {"call_expression"}):
        cal = _callee(src, n)
        if any(cal == x or cal.endswith("." + x) for x in names):
            return True
    return False


def _imports(c):
    """(specifier, import_statement node, src) for every static import."""
    for src, n in _nodes(c, {"import_statement"}):
        s = n.child_by_field_name("source")
        spec = _txt(src, s)[1:-1] if s is not None else ""
        yield spec, n, src


def _importmaps(c) -> list[dict]:
    """The `imports` object of every <script type="importmap"> (trailing
    commas tolerated)."""
    out = []
    for b in c.raw:
        for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", b["code"],
                             re.S | re.I):
            if not re.search(r"type\s*=\s*['\"]?importmap", m.group(1),
                             re.I):
                continue
            t = re.sub(r",\s*([}\]])", r"\1", m.group(2))
            try:
                d = json.loads(t)
            except ValueError:
                continue
            if isinstance(d, dict) and isinstance(d.get("imports"), dict):
                out.append(d["imports"])
    return out


def _new_of(c, suffix: str) -> bool:
    """A `new X(...)` whose constructor is `suffix` or `NS.suffix`."""
    for src, n in _nodes(c, {"new_expression"}):
        k = _txt(src, n.child_by_field_name("constructor"))
        if k == suffix or k.endswith("." + suffix):
            return True
    return False


def _inside(node, ranges) -> bool:
    return any(a <= node.start_byte and node.end_byte <= b
               for a, b in ranges)


def _callback_ranges(c, src, root_pred) -> list:
    """Byte ranges of the function arguments of calls whose callee
    satisfies `root_pred` (in the tree `src` belongs to)."""
    out = []
    for s, n in _nodes(c, {"call_expression"}):
        if s is not src or not root_pred(_callee(s, n)):
            continue
        args = n.child_by_field_name("arguments")
        for a in (args.named_children if args is not None else []):
            if a.type in ("arrow_function", "function_expression",
                          "function"):
                out.append((a.start_byte, a.end_byte))
    return out


_FUNCS = {"function_declaration", "function_expression", "arrow_function",
          "method_definition", "function", "generator_function_declaration"}


def _innermost_function(node):
    p = node.parent
    while p is not None and p.type not in _FUNCS:
        p = p.parent
    return p


# --------------------------------------------------- imports and setup ---
# Legacy entries for the node system / WebGPU renderer (pre-'three/tsl'
# releases: 'three/nodes', examples/jsm/nodes/Nodes.js, the addon renderer).
# The pinned manual names the current entries ('three/webgpu', 'three/tsl').
_LEGACY_SPEC = re.compile(
    r"(^three/nodes(?:/|$)|^three/src/nodes/|/examples/jsm/nodes/"
    r"|^three/examples/jsm/nodes|^three/addons/nodes/"
    r"|/jsm/renderers/webgpu/|^three/(?:addons|examples/jsm)/renderers/"
    r"webgpu/)")
# The classes that live only in the WebGPU build (@types/three@0.186.0:
# src/Three.d.ts exports none of them; src/Three.WebGPU.d.ts does).
_WEBGPU_ONLY = re.compile(r"^(?:\w*NodeMaterial|WebGPURenderer)$")


def _three_maps_to_webgpu(c) -> bool:
    """The answer's import map points bare 'three' at the WebGPU build (the
    manual's recommended map does), so `from 'three'` IS the WebGPU
    build."""
    return any(str(m.get("three", "")).endswith("three.webgpu.js")
               for m in _importmaps(c))


def r_tsl_legacy_node_import(c) -> bool:
    """TSL / node classes / the WebGPU renderer imported from a legacy
    entry ('three/nodes', 'three/examples/jsm/nodes/...', the addon
    WebGPURenderer), or a WebGPU-build-only class (a *NodeMaterial,
    WebGPURenderer) taken from core 'three'."""
    core_ns = set()
    for spec, n, src in _imports(c):
        if _LEGACY_SPEC.search(spec):
            return True
        if spec != "three" or _three_maps_to_webgpu(c):
            continue
        for s in _sub(n, {"import_specifier"}):
            nm = _txt(src, s.child_by_field_name("name"))
            if _WEBGPU_ONLY.match(nm):
                return True
        for s in _sub(n, {"namespace_import"}):
            ids = _sub(s, {"identifier"})
            if ids:
                core_ns.add(_txt(src, ids[0]))
    if core_ns:
        for src, n in _nodes(c, {"member_expression"}):
            obj = _txt(src, n.child_by_field_name("object"))
            prop = _txt(src, n.child_by_field_name("property"))
            if obj in core_ns and _WEBGPU_ONLY.match(prop):
                return True
    return False


def _spec_is(spec: str, entry: str, build_file: str) -> bool:
    return spec == entry or spec.endswith("/" + build_file)


def r_tsl_webgpu_entry_imports(c) -> bool:
    """Classes from 'three/webgpu' and TSL functions from 'three/tsl' (or
    the same build files by URL), as the pinned manual's import map
    names them."""
    specs = [s for s, _n, _src in _imports(c)]
    return any(_spec_is(s, "three/webgpu", "three.webgpu.js") for s in specs) \
        and any(_spec_is(s, "three/tsl", "three.tsl.js") for s in specs)


def r_tsl_imports_tsl(c) -> bool:
    return any(_spec_is(s, "three/tsl", "three.tsl.js")
               for s, _n, _src in _imports(c))


def r_tsl_webgl_renderer(c) -> bool:
    return _new_of(c, "WebGLRenderer")


def _renderer_started(c) -> bool:
    """`<x>.setAnimationLoop(...)` or `<renderer>.init()` (the object of
    `init` named like a renderer: UNMEASURED, a model's naming habit)."""
    if _called(c, "setAnimationLoop"):
        return True
    for src, n in _nodes(c, {"call_expression"}):
        f = n.child_by_field_name("function")
        if f is not None and f.type == "member_expression" and _txt(
                src, f.child_by_field_name("property")) == "init" and                 re.search(r"renderer|^gl$|^r$", _txt(
                    src, f.child_by_field_name("object")), re.I):
            return True
    return False


def r_tsl_webgpu_renderer_ready(c) -> bool:
    """A WebGPURenderer driven the manual's way: `setAnimationLoop()`
    ("will automatically ensure the renderer is initialized when rendering
    the first frame") or `init()` before rendering from its own loop."""
    return _new_of(c, "WebGPURenderer") and (
        _renderer_started(c))


def r_tsl_webgpu_render_before_init(c) -> bool:
    """A WebGPURenderer rendered from the app's own loop with neither
    `setAnimationLoop()` nor `init()`: WebGPU initializes asynchronously."""
    return _new_of(c, "WebGPURenderer") and not (
        _renderer_started(c))


def r_tsl_importmap_missing_webgpu_build(c) -> bool:
    """An import map without the 'three/webgpu' and 'three/tsl' entries."""
    return any("three/webgpu" not in m or "three/tsl" not in m
               for m in _importmaps(c))


def r_tsl_importmap_webgpu_build(c) -> bool:
    """An import map mapping 'three/webgpu' to three.webgpu.js and
    'three/tsl' to three.tsl.js (the pinned manual's map)."""
    return any(str(m.get("three/webgpu", "")).endswith("three.webgpu.js")
               and str(m.get("three/tsl", "")).endswith("three.tsl.js")
               for m in _importmaps(c))


# ------------------------------------------- custom shading: GLSL hacks ---
_GLSL = re.compile(r"void\s+main\s*\(|gl_FragColor|gl_Position\s*=|#include\s*<"
                   r"|texture2D\s*\(|\bvarying\s+vec|\buniform\s+(?:sampler|vec"
                   r"|float|mat)")
_GLSL_LANGS = ("glsl", "frag", "vert", "vs", "fs")


def r_tsl_glsl_material_hack(c) -> bool:
    """Custom shading the WebGL way: `onBeforeCompile`, a ShaderMaterial /
    RawShaderMaterial, or GLSL source in strings or GLSL blocks -- which the
    pinned manual says WebGPURenderer does not support ("This part of your
    application must be ported to node materials and TSL.")."""
    if any(b["lang"] in _GLSL_LANGS for b in c.blocks):
        return True
    for src, n in _nodes(c, {"property_identifier"}):
        if _txt(src, n) == "onBeforeCompile":
            return True
    if _new_of(c, "ShaderMaterial") or _new_of(c, "RawShaderMaterial"):
        return True
    for src, n in _nodes(c, {"template_string", "string"}):
        if _GLSL.search(_txt(src, n)):
            return True
    return False


def _node_slot(prop: str) -> bool:
    return bool(re.match(r"^[a-z]\w*Node$", prop))


def r_tsl_material_node_assigned(c) -> bool:
    """A node assigned to a material's input slot (`material.colorNode =
    ...`, or `colorNode: ...` in a `new XMaterial({...})`)."""
    for src, n in _nodes(c, {"assignment_expression"}):
        left = n.child_by_field_name("left")
        if left is None or left.type != "member_expression":
            continue
        obj = _txt(src, left.child_by_field_name("object"))
        prop = _txt(src, left.child_by_field_name("property"))
        if _node_slot(prop) and not re.search(
                r"scene|pipeline|postprocessing", obj, re.I):
            return True
    for src, n in _nodes(c, {"new_expression"}):
        k = _txt(src, n.child_by_field_name("constructor"))
        if not k.endswith("Material"):
            continue
        for p in _sub(n, {"pair"}):
            if _node_slot(_txt(src, p.child_by_field_name("key"))):
                return True
    return False


# ------------------------------------------------ branching on the GPU ---
_TSL_CMP = re.compile(r"\.(?:lessThan|greaterThan|lessThanEqual|"
                      r"greaterThanEqual|equal|notEqual)\s*\(")
# TSL constructors whose appearance in a JS branch means the branch picks
# between NODES (UNMEASURED list: the constructors a colour/value branch
# plausibly returns).
_TSL_CTORS = {"color", "vec2", "vec3", "vec4", "float", "int", "uint",
              "bool", "texture", "mix"}
_CPU_CALLBACK = re.compile(r"(?:^|\.)(?:On[A-Z]\w*|on[A-Z]\w*Update)$")


def r_tsl_js_branch_on_node(c) -> bool:
    """A JavaScript `if` / `?:` / `while` deciding a shader value: its
    condition is a TSL comparison (`x.lessThan(...)`), or it sits inside a
    `Fn()` body (outside a CPU event callback such as OnMaterialUpdate), or
    a relational `?:` / `if` picks between TSL constructor nodes. The
    pinned guide: JavaScript `if` statements "only run once on the CPU
    during the shader construction phase"."""
    for src, t in _trees(c):
        fn_r = _callback_ranges(c, src, lambda k: k == "Fn"
                                or k.endswith(".Fn"))
        cpu_r = _callback_ranges(c, src, lambda k: bool(
            _CPU_CALLBACK.search(k)))
        stack = [t.root_node]
        while stack:
            n = stack.pop()
            stack.extend(n.children)
            if n.type not in ("if_statement", "ternary_expression",
                              "while_statement"):
                continue
            cond = n.child_by_field_name("condition")
            ct = _txt(src, cond)
            if _TSL_CMP.search(ct):
                return True
            if "builder" in ct:
                continue   # a build-time branch on the NodeBuilder
            if _inside(n, fn_r) and not _inside(n, cpu_r):
                return True
            if re.search(r"[<>]=?", ct):
                arms = [n.child_by_field_name(f) for f in (
                    "consequence", "alternative")]
                for a in arms:
                    if a is None:
                        continue
                    for call in _sub(a, {"call_expression"}):
                        k = _callee(src, call).split(".")[-1]
                        if k in _TSL_CTORS:
                            return True
    return False


def r_tsl_gpu_branch(c) -> bool:
    """TSL's own branching: `If()` (with ElseIf/Else), `select()` or
    `Switch()`."""
    return _called(c, "If", "select", "Switch")


# ------------------------------------------------------------ uniforms ---
def _is_material(obj: str) -> bool:
    """`material`, `mat`, `shieldMaterial`, `this.material`, `mesh.material`
    (the last segment), never `mesh.instanceMatrix`."""
    last = obj.split(".")[-1]
    return bool(re.match(r"(?i)^(?:\w*material|mat)$", last))


def r_tsl_node_rebuilt_at_runtime(c) -> bool:
    """Changing a value by rebuilding the shader: a material's
    `needsUpdate = true`, a node slot (`xNode = ...`) re-assigned inside a
    function that does not create the material (a setter / per-frame
    function), or a material re-created into an existing binding inside a
    function."""
    for src, n in _nodes(c, {"assignment_expression"}):
        left = n.child_by_field_name("left")
        right = n.child_by_field_name("right")
        if left is None or left.type != "member_expression":
            if left is not None and right is not None and \
                    right.type == "new_expression" and _txt(
                        src, right.child_by_field_name("constructor")
                    ).endswith("Material") and _innermost_function(n):
                return True
            continue
        obj = _txt(src, left.child_by_field_name("object"))
        prop = _txt(src, left.child_by_field_name("property"))
        if prop == "needsUpdate" and _is_material(obj) and \
                _txt(src, right) == "true":
            return True
        if right is not None and right.type == "new_expression" and _txt(
                src, right.child_by_field_name("constructor")).endswith(
                    "Material") and _innermost_function(n) is not None:
            f = _innermost_function(n)
            if f.type == "method_definition" and _txt(
                    src, f.child_by_field_name("name")) == "constructor":
                continue
            return True
        if _node_slot(prop):
            f = _innermost_function(n)
            if f is None:
                continue
            fr = _callback_ranges(c, src, lambda k: k == "Fn"
                                  or k.endswith(".Fn"))
            if _inside(n, fr):
                continue
            makes = any(_txt(src, x.child_by_field_name("constructor"))
                        .endswith("Material")
                        for x in _sub(f, {"new_expression"}))
            if not makes:
                return True
    return False


def r_tsl_uniform_value_update(c) -> bool:
    """A `uniform()` node whose value changes in place: `.value` assigned
    or mutated (`u.value = x`, `u.value.setHex(x)`), or an update event
    (`onFrameUpdate` / `onRenderUpdate` / `onObjectUpdate`)."""
    if not _called(c, "uniform"):
        return False
    if _called(c, "onFrameUpdate", "onRenderUpdate", "onObjectUpdate"):
        return True
    for src, n in _nodes(c, {"assignment_expression",
                             "augmented_assignment_expression"}):
        left = n.child_by_field_name("left")
        if left is not None and left.type == "member_expression" and _txt(
                src, left.child_by_field_name("property")) == "value":
            return True
    for src, n in _nodes(c, {"call_expression"}):
        if ".value." in _callee(src, n):
            return True
    return False


# ------------------------------------------------------------- compute ---
_CPU_SET = re.compile(r"\.(?:set(?:X|Y|Z|W|XY|XYZ|XYZW|MatrixAt|ColorAt))$")
_LOOPS = {"for_statement", "for_in_statement", "while_statement",
          "do_statement"}


def r_tsl_cpu_buffer_loop(c) -> bool:
    """Per-element work on the CPU: GPUComputationRenderer (the WebGL
    GPGPU helper, built on ShaderMaterial), or a JS loop writing buffer
    elements (`a[i] = ...`, `attr.setXYZ(...)`, `setMatrixAt`) with an
    attribute re-uploaded (`needsUpdate = true` on a non-material)."""
    for src, n in _nodes(c, {"identifier", "type_identifier"}):
        if _txt(src, n) == "GPUComputationRenderer":
            return True
    reupload = False
    for src, n in _nodes(c, {"assignment_expression"}):
        left = n.child_by_field_name("left")
        if left is not None and left.type == "member_expression" and _txt(
                src, left.child_by_field_name("property")) == "needsUpdate" \
                and not _is_material(_txt(
                    src, left.child_by_field_name("object"))):
            reupload = True
    if not reupload:
        return False
    for src, n in _nodes(c, _LOOPS | {"call_expression"}):
        if n.type == "call_expression":
            if not _callee(src, n).endswith(".forEach"):
                continue
        for x in _sub(n, {"assignment_expression",
                          "augmented_assignment_expression"}):
            left = x.child_by_field_name("left")
            if left is not None and left.type == "subscript_expression":
                return True
        for x in _sub(n, {"call_expression"}):
            if _CPU_SET.search(_callee(src, x)):
                return True
    return False


def r_tsl_compute_dispatched(c) -> bool:
    """A TSL compute node (`Fn(...)().compute(count)` or `compute(node,
    count)`) dispatched with `renderer.compute(...)` /
    `renderer.computeAsync(...)`, as the pinned guide's examples do.
    UNMEASURED: the dispatching object is told apart by its name
    containing "renderer"."""
    made = dispatched = False
    for src, n in _nodes(c, {"call_expression"}):
        f = n.child_by_field_name("function")
        if f is None:
            continue
        if f.type == "identifier" and _txt(src, f) == "compute":
            made = True
            continue
        if f.type != "member_expression":
            continue
        prop = _txt(src, f.child_by_field_name("property"))
        if prop not in ("compute", "computeAsync"):
            continue
        obj = f.child_by_field_name("object")
        if obj is not None and obj.type != "call_expression" and re.search(
                r"renderer", _txt(src, obj), re.I):
            dispatched = True
        elif prop == "compute":
            made = True
    return made and dispatched
