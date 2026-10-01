"""THE PMNDRS MATH PITFALL RULES (the npm package `math`, pmndrs/math;
bench/skills/pitfall_harness.py loads this module; the cases are
bench/skills/pitfalls/pmath.jsonl). Each rule reads the answer's code as a
pitfall_harness.Code (`c`) and returns a bool; the case file names a rule
without its `r_` prefix.

The APIs come from the pinned API.md of math@0.1.0
(bench/skills/pitfalls/sources/pmath@c6713e38dd86; MIT). Which call shapes
count as the pitfall and as the good pattern, and every name list below,
are our reading of that reference: UNMEASURED (no labelled answers yet),
written for the variants a model plausibly writes (a namespace import, an
alias, a const arrow or a function declaration).
"""
from __future__ import annotations

import re

_LOOPS = {"for_statement", "for_in_statement", "while_statement",
          "do_statement"}
# Array methods whose callback runs once per element (a loop in effect).
_ITERATORS = {"forEach", "map", "reduce", "for", "every", "some"}
# The package's value namespaces (API.md "Modules"): a call
# `<ns>.create|clone|fromValues(...)` allocates a new tuple.
_NS = ("vec2", "vec3", "vec4", "quat", "quat2", "euler", "mat2", "mat2d",
       "mat3", "mat4", "spherical", "polar")
_ALLOC_FNS = {"create", "clone", "fromValues"}
# three.js value classes a model reaches for (a `new` of one allocates).
_THREE_VALUES = {"Vector2", "Vector3", "Vector4", "Quaternion", "Matrix3",
                 "Matrix4", "Euler", "Spherical"}
# Constants of the PRNGs a model hand-rolls (mulberry32, the LCGs of
# Numerical Recipes and Park-Miller, 2^31-1). UNMEASURED list.
_PRNG_CONSTANTS = re.compile(
    r"0x6D2B79F5|1831565813|\b1664525\b|\b1013904223\b|\b16807\b|"
    r"\b48271\b|\b2147483647\b|\b4294967296\b", re.I)


def _args(n) -> list:
    a = n.child_by_field_name("arguments")
    return list(a.named_children) if a is not None else []


def _ns_fn(c, src, call) -> tuple[str, str]:
    """`vec3.scaleAndAdd(...)` -> ('vec3', 'scaleAndAdd');
    `math.vec3.add` -> ('vec3', 'add'); a bare call -> ('', name)."""
    parts = c.callee(src, call).split(".")
    return (parts[-2] if len(parts) > 1 else "", parts[-1])


def _in_loop(c, src, n) -> bool:
    """Inside a loop statement, or inside the callback of forEach / map /
    reduce (called once per element)."""
    p = n.parent
    while p is not None:
        if p.type in _LOOPS:
            body = p.child_by_field_name("body")
            if body is not None and body.start_byte <= n.start_byte \
                    and n.end_byte <= body.end_byte:
                return True
        if p.type == "call_expression" and \
                _ns_fn(c, src, p)[1] in _ITERATORS:
            a = p.child_by_field_name("arguments")
            if a is not None and a.start_byte <= n.start_byte \
                    and n.end_byte <= a.end_byte:
                return True
        p = p.parent
    return False


def _imports_from(c, pattern: str) -> bool:
    for _b, src, n in c.nodes({"import_statement"}):
        s = n.child_by_field_name("source")
        if s is not None and re.fullmatch(pattern,
                                          c.text(src, s).strip("'\"`")):
            return True
    return False


def _in_type(n) -> bool:
    p = n.parent
    while p is not None:
        if p.type in ("type_annotation", "type_alias_declaration",
                      "tuple_type", "interface_declaration",
                      "type_arguments"):
            return True
        p = p.parent
    return False


# ------------------------------------------------- allocation per step ---
def r_pmath_allocates_in_loop(c) -> bool:
    """A new vector per element per frame: `vec3.create()` / `clone` /
    `fromValues`, a `new THREE.Vector3()`, or a 2-4 element array literal
    (a fresh tuple), inside the per-particle loop."""
    for _b, src, n in c.nodes({"call_expression"}):
        ns, fn = _ns_fn(c, src, n)
        if ns in _NS and fn in _ALLOC_FNS and _in_loop(c, src, n):
            return True
    for _b, src, n in c.nodes({"new_expression"}):
        k = n.child_by_field_name("constructor")
        if k is not None and c.text(src, k).split(".")[-1] in \
                _THREE_VALUES and _in_loop(c, src, n):
            return True
    for _b, src, n in c.nodes({"array"}):
        if 2 <= len(n.named_children) <= 4 and not _in_type(n) and \
                _in_loop(c, src, n):
            return True
    return False


def r_pmath_writes_through_out(c) -> bool:
    """A package function writes its result into an existing tuple: the
    out argument (first) is also one of its inputs -- `vec3.scaleAndAdd(
    v, v, g, dt)`, `vec3.add(p, p, d)` -- the tuple updated in place."""
    for _b, src, n in c.nodes({"call_expression"}):
        ns, _fn = _ns_fn(c, src, n)
        if ns not in _NS:
            continue
        a = [c.text(src, x) for x in _args(n)]
        if len(a) >= 2 and a[0] in a[1:]:
            return True
    return False


# ------------------------------------------------------ orbit position ---
def r_pmath_hand_rolled_or_three_spherical(c) -> bool:
    """The orbit position computed by hand (Math.sin and Math.cos) or with
    three.js' own helpers (`setFromSpherical`, `setFromSphericalCoords`, a
    `new Spherical` / `new Vector3`)."""
    code = c.all_code()
    if re.search(r"\bMath\.sin\s*\(", code) and \
            re.search(r"\bMath\.cos\s*\(", code):
        return True
    if c.called("setFromSpherical") or c.called("setFromSphericalCoords"):
        return True
    for _b, src, n in c.nodes({"new_expression"}):
        k = n.child_by_field_name("constructor")
        if k is not None and c.text(src, k).split(".")[-1] in (
                "Spherical", "Vector3"):
            return True
    return False


def r_pmath_spherical_to_vec3(c) -> bool:
    """The package's spherical coordinates turned into the position:
    `spherical.toVec3(out, s)`."""
    for _b, src, n in c.nodes({"call_expression"}):
        if _ns_fn(c, src, n) == ("spherical", "toVec3"):
            return True
    return False


# ------------------------------------------------------ seeded random ---
def r_pmath_unseeded_or_hand_rolled_random(c) -> bool:
    """Math.random() (no seed), or a PRNG written by hand (Math.imul, a
    known PRNG constant, an xorshift step)."""
    code = c.all_code()
    return bool(re.search(r"\bMath\.random\s*\(", code)
                or re.search(r"\bMath\.imul\s*\(", code)
                or _PRNG_CONSTANTS.search(code)
                or re.search(r"\^=\s*\w+\s*(<<|>>>?)", code))


def r_pmath_seeded_generator(c) -> bool:
    """A seeded generator from 'math/random' (`mulberry32.create(seed)`,
    `isaac32` / `isaac64`), drawn with its `sample` or through the `random`
    helpers."""
    if not _imports_from(c, r"math/random"):
        return False
    created = drawn = False
    for _b, src, n in c.nodes({"call_expression"}):
        ns, fn = _ns_fn(c, src, n)
        if ns in ("mulberry32", "isaac32", "isaac64") and fn == "create" \
                and _args(n):
            created = True
        if (ns in ("mulberry32", "isaac32", "isaac64") and fn in
                ("sample", "next")) or ns == "random":
            drawn = True
    return created and drawn


# ------------------------------------------------------------- noise ---
_NOISE_GENS = ("perlin2d", "perlin3d", "simplex2d", "simplex3d",
               "simplex4d", "worley2d", "worley3d")


def r_pmath_hand_rolled_or_foreign_noise(c) -> bool:
    """Noise from another package ('simplex-noise' ...), a noise function
    written by hand (a function named perlin / simplex / noise / grad /
    fade that does not call a generator's `sample`), or the octaves summed
    by a hand loop (frequency / amplitude multiplied inside a loop)."""
    for _b, src, n in c.nodes({"import_statement"}):
        s = n.child_by_field_name("source")
        if s is not None:
            spec = c.text(src, s).strip("'\"`")
            if re.search(r"noise|perlin|simplex", spec, re.I) and \
                    spec != "math/noise":
                return True
    for _b, src, n in c.nodes({"function_declaration",
                               "variable_declarator"}):
        nm = n.child_by_field_name("name")
        if nm is None:
            continue
        name = c.text(src, nm)
        val = n if n.type == "function_declaration" else \
            n.child_by_field_name("value")
        if val is None or val.type not in (
                "function_declaration", "arrow_function",
                "function_expression", "function"):
            continue
        if re.search(r"perlin|simplex|noise|^grad|^fade", name, re.I) and \
                not re.search(r"\.sample\s*\(", c.text(src, val)):
            return True
    for _b, src, n in c.nodes({"augmented_assignment_expression"}):
        op = n.child_by_field_name("operator")
        left = n.child_by_field_name("left")
        if op is not None and c.text(src, op) == "*=" and left is not None \
                and re.search(r"freq|amp", c.text(src, left), re.I):
            p = n.parent
            while p is not None and p.type not in (
                    "for_statement", "while_statement", "do_statement",
                    "for_in_statement"):
                p = p.parent
            if p is not None:
                return True
    return False


def r_pmath_noise_generator(c) -> bool:
    """A seeded generator from 'math/noise' (`simplex2d.create(seed)`, or
    perlin / worley) sampled with its `sample`."""
    if not _imports_from(c, r"math/noise"):
        return False
    created = sampled = False
    for _b, src, n in c.nodes({"call_expression"}):
        ns, fn = _ns_fn(c, src, n)
        if ns in _NOISE_GENS and fn == "create" and _args(n):
            created = True
        if ns in _NOISE_GENS and fn == "sample":
            sampled = True
    return created and sampled


def r_pmath_fractal_octaves(c) -> bool:
    """The octaves summed by the package's fractal helper: `fbm(sample,
    octaves, lacunarity, gain)` (or ridged / billow)."""
    for _b, src, n in c.nodes({"call_expression"}):
        if _ns_fn(c, src, n)[1] in ("fbm", "ridged", "billow") and \
                len(_args(n)) >= 2:
            return True
    return False
