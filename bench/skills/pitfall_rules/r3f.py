"""THE r3f PITFALL RULES (react-three-fiber v10 + three.js), loaded by
bench/skills/pitfall_harness.py (`_load_rule_modules`). Each rule is
`r_r3f_<name>(c) -> bool` over a pitfall_harness.Code; the case file
(bench/skills/pitfalls/r3f.jsonl) names them without the `r_`.

The patterns come from the pinned docs,
bench/skills/pitfalls/sources/react-three-fiber@27df622f4990/ (the tag
v10.0.0-alpha.5, the held @react-three/fiber): advanced/pitfalls.mdx
("Performance pitfalls"), migration/v10.mdx, webgpu/overview.mdx. The lists
of three.js method names and loader/instancing element names below are ours,
from three.js's and drei's public API as a model would write it -- UNMEASURED
(no recorded answers were read to choose them)."""
from __future__ import annotations

import re

_FUNCS = {"function_declaration", "function_expression", "arrow_function",
          "function", "method_definition"}


def _text(src, n) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "replace")


def _within(node, types) -> list:
    out, stack = [], [node]
    while stack:
        x = stack.pop()
        if x.type in types:
            out.append(x)
        stack.extend(x.children)
    return out


def _callee(src, call) -> str:
    f = call.child_by_field_name("function")
    return _text(src, f) if f is not None else ""


def _first_fn_arg(call):
    args = call.child_by_field_name("arguments")
    if args is None or not args.named_children:
        return None
    a0 = args.named_children[0]
    return a0 if a0.type in _FUNCS else None


def _frame_callbacks(c):
    """(src, callback) of every `useFrame(cb, ...)`."""
    for _b, src, n in c.calls("useFrame"):
        cb = _first_fn_arg(n)
        if cb is not None:
            yield src, cb


# A bare React state setter: `setX(...)` (never a method: `vec.set(...)`).
_SETTER = re.compile(r"^set[A-Z]\w*$")


def _setter_calls(src, node) -> bool:
    return any(_SETTER.match(_callee(src, x))
               for x in _within(node, {"call_expression"}))


# ------------------------------------------------ setState in the frame loop
def r_r3f_setstate_in_frame_loop(c) -> bool:
    """A React state setter called from the frame loop, a timer loop or a
    fast pointer event -- the pitfalls page's "Avoid setState in loops":
    inside a `useFrame` callback, a `setInterval` / `requestAnimationFrame`
    callback, or an `onPointerMove` handler."""
    for src, cb in _frame_callbacks(c):
        if _setter_calls(src, cb):
            return True
    for name in ("setInterval", "requestAnimationFrame"):
        for _b, src, n in c.calls(name):
            cb = _first_fn_arg(n)
            if cb is not None and _setter_calls(src, cb):
                return True
    for _b, src, n in c.nodes({"jsx_attribute"}):
        if n.named_children and _text(src, n.named_children[0]) == \
                "onPointerMove" and _setter_calls(src, n):
            return True
    return False


# three.js mutators a frame callback calls on an object it holds (UNMEASURED
# list: the Vector3 / Euler / Quaternion / Object3D / InstancedMesh methods
# that change the receiver).
_MUTATORS = {"set", "copy", "lerp", "lerpVectors", "add", "sub",
             "addScaledVector", "multiplyScalar", "setScalar",
             "applyAxisAngle", "applyQuaternion", "applyEuler", "rotateX",
             "rotateY", "rotateZ", "rotateOnAxis", "translateX",
             "translateY", "translateZ", "lookAt", "slerp", "setFromEuler",
             "setFromAxisAngle", "setRotationFromAxisAngle", "setMatrixAt",
             "setColorAt", "updateMatrix"}


def _mutates(src, node) -> bool:
    for x in _within(node, {"assignment_expression",
                            "augmented_assignment_expression"}):
        left = x.child_by_field_name("left")
        if left is not None and left.type in ("member_expression",
                                              "subscript_expression"):
            return True
    for x in _within(node, {"call_expression"}):
        cal = _callee(src, x)
        if "." in cal and cal.rsplit(".", 1)[-1] in _MUTATORS:
            return True
    return False


def r_r3f_mutates_in_frame(c) -> bool:
    """A `useFrame` callback that mutates a three.js object (a property
    write or a mutating method call) -- "Fast updates are carried out in
    `useFrame` by mutation"."""
    return any(_mutates(src, cb) for src, cb in _frame_callbacks(c))


# ---------------------------------------------- allocation in the frame loop
def _allocates(src, node) -> bool:
    if _within(node, {"new_expression"}):
        return True
    return any(_callee(src, x).endswith(".clone")
               for x in _within(node, {"call_expression"}))


def r_r3f_alloc_in_frame(c) -> bool:
    """A `useFrame` callback that allocates every frame: a `new ...`
    (`new THREE.Vector3(...)`) or a `.clone()` -- "Don't re-create objects
    in loops"."""
    return any(_allocates(src, cb) for src, cb in _frame_callbacks(c))


def r_r3f_frame_no_alloc(c) -> bool:
    """At least one `useFrame` callback, and none allocates (the objects it
    needs are created once, outside the loop, and re-used)."""
    cbs = list(_frame_callbacks(c))
    return bool(cbs) and not any(_allocates(src, cb) for src, cb in cbs)


# ------------------------------------------- many objects of a similar type
def _jsx_name(src, el) -> str:
    op = el if el.type == "jsx_self_closing_element" else next(
        (k for k in el.children if k.type == "jsx_opening_element"), None)
    if op is None:
        return ""
    nm = op.child_by_field_name("name")
    return _text(src, nm) if nm is not None else ""


def _attrs(src, el) -> set:
    op = el if el.type == "jsx_self_closing_element" else next(
        (k for k in el.children if k.type == "jsx_opening_element"), None)
    if op is None:
        return set()
    return {_text(src, a.named_children[0]) for a in op.children
            if a.type == "jsx_attribute" and a.named_children}


_ELEMENTS = {"jsx_element", "jsx_self_closing_element"}


def _inline_resources(src, mesh) -> bool:
    """A mesh element with its own geometry AND material child elements
    (`<boxGeometry />` + `<meshStandardMaterial />`): created per mesh."""
    names = [_jsx_name(src, x) for x in _within(mesh, _ELEMENTS)
             if x is not mesh]
    return any(n.endswith("Geometry") and n[:1].islower() for n in names) \
        and any(n.endswith("Material") and n[:1].islower() for n in names)


def _mapped_callbacks(c):
    """(src, callback) of every `xs.map(cb)` / `Array.from(xs, cb)`."""
    for _b, src, n in c.nodes({"call_expression"}):
        cal = _callee(src, n)
        if cal.endswith(".map") or cal == "Array.from":
            args = n.child_by_field_name("arguments")
            for a in (args.named_children if args is not None else []):
                if a.type in _FUNCS:
                    yield src, a


def _component_bodies(c) -> dict:
    """{component name: [(src, function node)]}: capitalised function
    declarations and capitalised consts holding an arrow/function."""
    out: dict = {}
    for _b, src, n in c.nodes({"function_declaration"}):
        nm = n.child_by_field_name("name")
        if nm is not None and _text(src, nm)[:1].isupper():
            out.setdefault(_text(src, nm), []).append((src, n))
    for _b, src, n in c.nodes({"variable_declarator"}):
        nm = n.child_by_field_name("name")
        val = n.child_by_field_name("value")
        if nm is not None and val is not None and val.type in _FUNCS and \
                _text(src, nm)[:1].isupper():
            out.setdefault(_text(src, nm), []).append((src, val))
    return out


def _per_item_meshes(c):
    """(src, mesh element) of every mesh drawn once per item: a `mesh`
    inside a map/Array.from callback, or inside a component that such a
    callback renders."""
    comps = _component_bodies(c)
    for src, cb in _mapped_callbacks(c):
        for el in _within(cb, _ELEMENTS):
            nm = _jsx_name(src, el)
            if nm == "mesh":
                yield src, el
            elif nm[:1].isupper():
                for csrc, fn in comps.get(nm, []):
                    for el2 in _within(fn, _ELEMENTS):
                        if _jsx_name(csrc, el2) == "mesh":
                            yield csrc, el2


def r_r3f_mesh_per_item(c) -> bool:
    """Many similar objects drawn as one `<mesh>` per item, each with its own
    geometry and material elements -- the pitfalls page: every geometry is
    processed and every material compiled; share them, or instance."""
    return any(_inline_resources(src, m) for src, m in _per_item_meshes(c))


# Instancing as a model writes it: three's InstancedMesh (JSX or `new`),
# drei's Instances / Merged (UNMEASURED list).
_INSTANCED = {"instancedMesh", "Instances", "Merged"}


def r_r3f_instanced_or_shared(c) -> bool:
    """The many objects are instanced (`<instancedMesh>`, drei `<Instances>`
    / `<Merged>`, `new InstancedMesh`), or every per-item mesh takes a
    shared `geometry=` and `material=`."""
    if _INSTANCED & set(c.jsx_names()):
        return True
    for _b, src, n in c.nodes({"new_expression"}):
        k = n.child_by_field_name("constructor")
        if k is not None and _text(src, k).split(".")[-1] == \
                "InstancedMesh":
            return True
    meshes = list(_per_item_meshes(c))
    return bool(meshes) and all(
        {"geometry", "material"} <= _attrs(src, m) for src, m in meshes)


# ------------------------------------------------------- plain three loaders
def r_r3f_plain_loader(c) -> bool:
    """A three.js loader driven by hand: `.load(...)` / `.loadAsync(...)`
    on a `new XLoader()` (inline, or when the code constructs a loader) --
    "No re-use is bad for perf: This re-fetches, re-parses for every
    component instance"."""
    code = c.all_code()
    constructs = re.search(r"\bnew\s+(?:[\w$]+\.)*\w*Loader\s*\(", code)
    for _b, src, n in c.nodes({"call_expression"}):
        cal = _callee(src, n)
        if re.search(r"\.(load|loadAsync)$", cal) and (
                "Loader" in cal or constructs):
            return True
    return False


def r_r3f_uses_loader_hook(c) -> bool:
    """`useLoader(...)` (drei's `useTexture`, useLoader + TextureLoader,
    counts) -- "Instead use useLoader, which caches assets"."""
    return c.called("useLoader") or c.called("useTexture")


# ------------------------------------------------------ v10: frame timing
def _pattern_keys(src, node) -> set:
    keys = set()
    for p in _within(node, {"object_pattern"}):
        for k in p.named_children:
            if k.type == "shorthand_property_identifier_pattern":
                keys.add(_text(src, k))
            elif k.type in ("pair_pattern", "object_assignment_pattern"):
                kk = k.child_by_field_name("key") or \
                    k.child_by_field_name("left") or \
                    (k.named_children[0] if k.named_children else None)
                if kk is not None:
                    keys.add(_text(src, kk))
    return keys


def _member_props(c) -> set:
    out = set()
    for _b, src, n in c.nodes({"member_expression"}):
        p = n.child_by_field_name("property")
        if p is not None:
            out.add(_text(src, p))
    return out


def _all_pattern_keys(c) -> set:
    keys = set()
    for _b, src, t in c.trees:
        keys |= _pattern_keys(src, t.root_node)
    return keys


def r_r3f_v10_frame_clock(c) -> bool:
    """The v9 frame clock read off R3F state: `state.clock` / `s.clock` / a
    `clock` destructured from useFrame's or useThree's state -- "The
    `state.clock` (THREE.Clock) has been removed from FrameState". (A clock
    the code makes itself, `new THREE.Clock()`, is not this.)"""
    return "clock" in _member_props(c) or "clock" in _all_pattern_keys(c)


def r_r3f_v10_frame_elapsed(c) -> bool:
    """v10's timing read off the frame state: `state.elapsed` or a
    destructured `elapsed`."""
    return "elapsed" in _member_props(c) or "elapsed" in _all_pattern_keys(c)


# ------------------------------------------------- v10: state.gl -> renderer
def r_r3f_v10_state_gl(c) -> bool:
    """The v9 name for the renderer in R3F state: `state.gl`, `s => s.gl`,
    or `gl` destructured from useThree()/useFrame state -- "The `gl`
    property in state has been renamed to `renderer`"."""
    return "gl" in _member_props(c) or "gl" in _all_pattern_keys(c)


def r_r3f_v10_state_renderer(c) -> bool:
    """The renderer read as `renderer` from R3F state."""
    return "renderer" in _member_props(c) or \
        "renderer" in _all_pattern_keys(c)


# ------------------------------------------------ v10: the Canvas background
def r_r3f_color_attach_background(c) -> bool:
    """The scene background set the v9 way: `<color attach="background" />`
    or an assignment to `scene.background` -- "deprecated in favor of the
    new `background` prop on Canvas"."""
    if any(en == "color" and a == "attach" and "background" in v
           for en, a, v in c.jsx_attrs()):
        return True
    for _b, src, n in c.nodes({"assignment_expression"}):
        left = n.child_by_field_name("left")
        if left is not None and left.type == "member_expression":
            p = left.child_by_field_name("property")
            if p is not None and _text(src, p) == "background":
                return True
    return False


def r_r3f_v10_canvas_background(c) -> bool:
    """`<Canvas background=...>`."""
    return any(a == "background" for _e, a, _v in c.jsx_attrs("Canvas"))


# ----------------------------------------------- SETUP: v10 WebGPU project
def r_r3f_v10_canvas_renderer(c) -> bool:
    """`<Canvas renderer>` (bare, a parameter bag, or an instance) -- "Opt
    into WebGPU by adding the `renderer` prop to `Canvas`"."""
    return any(a == "renderer" for _e, a, _v in c.jsx_attrs("Canvas"))


def r_r3f_v9_webgpu_gl_factory(c) -> bool:
    """The v9 WebGPU recipe: a Canvas `gl` prop that builds a
    WebGPURenderer and/or awaits its `init()` -- v10: "No manual renderer
    initialization is required"."""
    return any(a == "gl" and ("WebGPURenderer" in v or "init(" in v)
               for _e, a, v in c.jsx_attrs("Canvas"))


def _import_sources(c) -> list[str]:
    out = []
    for _b, src, n in c.nodes({"import_statement", "export_statement"}):
        s = n.child_by_field_name("source")
        if s is not None:
            out.append(_text(src, s).strip("'\"`"))
    return out


def r_r3f_v10_webgpu_entry(c) -> bool:
    """R3F imported from its WebGPU entry, `@react-three/fiber/webgpu` (the
    entry that auto-extends the node materials and carries the TSL hooks)."""
    return "@react-three/fiber/webgpu" in _import_sources(c)


def _deps(c) -> dict:
    out: dict = {}
    for j in c.json_of("package.json"):
        if isinstance(j, dict):
            for k in ("dependencies", "devDependencies", "peerDependencies"):
                if isinstance(j.get(k), dict):
                    out.update(j[k])
    return out


def _floor(spec) -> tuple | None:
    """The lowest version a semver range admits, as (major, minor, patch):
    the first x.y.z of the spec (`^0.185.0`, `>=0.185`, `~0.186.1`,
    `10.0.0-alpha.5`); None for a tag (`latest`) or `*`."""
    m = re.search(r"(\d+)(?:\.(\d+|x|\*))?(?:\.(\d+|x|\*))?", str(spec))
    if not m:
        return None
    return tuple(int(g) if g and g.isdigit() else 0 for g in m.groups())


# v10's peer floor for three, from migration/v10.mdx ("| `three` | `>=0.156`
# | `>=0.185.0` |"): a docs number, not ours.
_THREE_FLOOR = (0, 185, 0)


def r_r3f_v10_deps_too_old(c) -> bool:
    """package.json pins R3F below v10, or three below v10's peer floor
    (>=0.185.0) -- "If you are on r181–r184 you cannot install v10"."""
    d = _deps(c)
    f = _floor(d["@react-three/fiber"]) if "@react-three/fiber" in d \
        else None
    t = _floor(d["three"]) if "three" in d else None
    return (f is not None and f[0] < 10) or \
        (t is not None and t < _THREE_FLOOR)


def r_r3f_v10_deps(c) -> bool:
    """package.json lists @react-three/fiber at major 10 and three at or
    above v10's peer floor (0.185.0)."""
    d = _deps(c)
    f = _floor(d.get("@react-three/fiber", ""))
    t = _floor(d.get("three", ""))
    return f is not None and f[0] == 10 and t is not None and \
        t >= _THREE_FLOOR
