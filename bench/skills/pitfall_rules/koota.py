"""THE KOOTA PITFALL RULES (bench/skills/pitfall_harness.py loads this
module; the cases are bench/skills/pitfalls/koota.jsonl). Each rule reads
the answer's code as a pitfall_harness.Code (`c`) and returns a bool; the
case file names a rule without its `r_` prefix.

The patterns come from the pinned koota docs, v0.6.6
(bench/skills/pitfalls/sources/koota@7d1329aa82e3: README.md,
skills/koota/SKILL.md; ISC). Which call shapes count as the pitfall and as
the good pattern, and every name list below, are our reading of those
docs: UNMEASURED (no labelled answers yet), written for the variants a
model plausibly writes (aliases, a const arrow or a function declaration,
a trait read into a local first).
"""
from __future__ import annotations

import re

_FUNCS = {"function_declaration", "function_expression", "arrow_function",
          "function", "method_definition", "generator_function_declaration"}
# Array methods that change the array they are called on. UNMEASURED list
# (the README's own example is `inventory.items.push(item)`).
_MUTATORS = {"push", "unshift", "splice", "pop", "shift", "sort", "reverse",
             "fill", "copyWithin"}
# The React effects the README shows direct world use inside ("Or access
# world directly and use it", README "Modify Koota state safely with
# actions").
_EFFECTS = {"useEffect", "useLayoutEffect"}


# ------------------------------------------------------------- helpers ---
def _args(n) -> list:
    a = n.child_by_field_name("arguments")
    return list(a.named_children) if a is not None else []


def _prop(c, src, call) -> str:
    """The method name of a call `x.y(...)` ('y'), else the callee text."""
    return c.callee(src, call).rsplit(".", 1)[-1]


def _receiver(call):
    """The object a method is called on: `a.b.push(x)` -> `a.b`."""
    f = call.child_by_field_name("function")
    if f is not None and f.type == "member_expression":
        return f.child_by_field_name("object")
    return None


def _fn_name(c, src, fn) -> str:
    """A function node's own name, or the name it is assigned to."""
    nm = fn.child_by_field_name("name")
    if nm is not None:
        return c.text(src, nm)
    p = fn.parent
    if p is not None and p.type == "variable_declarator":
        nm = p.child_by_field_name("name")
        if nm is not None:
            return c.text(src, nm)
    return ""


def _enclosing_fns(n) -> list:
    out, p = [], n.parent
    while p is not None:
        if p.type in _FUNCS:
            out.append(p)
        p = p.parent
    return out


def _in_component(c, src, n) -> bool:
    """Inside a function whose name starts with a capital letter (a React
    component, the harness's own reading of a component)."""
    return any(_fn_name(c, src, f)[:1].isupper()
               for f in _enclosing_fns(n))


def _inside_call(c, src, n, names: set) -> bool:
    """`n` lies inside the arguments of a call whose callee (or its last
    member) is one of `names`."""
    p = n.parent
    while p is not None:
        if p.type == "call_expression":
            cal = c.callee(src, p)
            if cal in names or cal.rsplit(".", 1)[-1] in names:
                a = p.child_by_field_name("arguments")
                if a is not None and a.start_byte <= n.start_byte \
                        and n.end_byte <= a.end_byte:
                    return True
        p = p.parent
    return False


def _in_event_handler(c, src, n) -> bool:
    """Inside a JSX attribute named on* (onClick, onPointerDown ...)."""
    p = n.parent
    while p is not None:
        if p.type == "jsx_attribute" and p.named_children:
            if re.match(r"on[A-Z]", c.text(src, p.named_children[0])):
                return True
        p = p.parent
    return False


def _capitalised_first_arg(c, src, call) -> bool:
    a = _args(call)
    return bool(a) and c.text(src, a[0])[:1].isupper()


def _decl_values(c) -> dict:
    """name -> the text of the value each const/let declares it with."""
    out: dict = {}
    for _b, src, n in c.nodes({"variable_declarator"}):
        nm = n.child_by_field_name("name")
        val = n.child_by_field_name("value")
        if nm is None or val is None:
            continue
        names = [nm] if nm.type == "identifier" else c.within(
            src, nm, {"identifier", "shorthand_property_identifier_pattern"})
        for x in names:
            out.setdefault(c.text(src, x), []).append(c.text(src, val))
    return out


def _imports_from(c, pattern: str) -> bool:
    for _b, src, n in c.nodes({"import_statement"}):
        s = n.child_by_field_name("source")
        if s is not None and re.fullmatch(pattern,
                                          c.text(src, s).strip("'\"`")):
            return True
    return False


# ----------------------------------------- change detection (inventory) ---
def _trait_derived(c, src, recv, decls) -> bool:
    """The array a mutator is called on is trait data: a member of an
    object (`inv.items`, `entity.get(Inventory)!.items`), or a local that
    was read from a trait (`const items = player.get(Inventory)!.items`)."""
    if recv is None:
        return False
    t = c.text(src, recv)
    if recv.type in ("member_expression", "subscript_expression",
                     "non_null_expression"):
        return True
    if recv.type == "identifier":
        return any(re.search(r"\.get\(|useTrait\(|\.items\b", v)
                   for v in decls.get(t, []))
    return ".get(" in t


def r_koota_change_flagged(c) -> bool:
    """The change is signalled the way the docs say: `entity.changed(Trait)`
    after mutating in place, or the trait written with `entity.set(Trait,
    ...)` / `world.set(Trait, ...)` (both trigger change events)."""
    for _b, src, n in c.nodes({"call_expression"}):
        p = _prop(c, src, n)
        if p == "changed" and _args(n) and _receiver(n) is not None:
            return True
        if p == "set" and _receiver(n) is not None and \
                _capitalised_first_arg(c, src, n):
            return True
    return False


def r_koota_mutates_trait_unflagged(c) -> bool:
    """An array held in trait data is mutated in place (push, splice ...)
    and nothing flags the change: a hook such as useTrait never sees it
    (README: "This change will not be detected since the array is mutated
    and will pass the comparison")."""
    decls = _decl_values(c)
    mutated = False
    for _b, src, n in c.nodes({"call_expression"}):
        if _prop(c, src, n) in _MUTATORS and _trait_derived(
                c, src, _receiver(n), decls):
            mutated = True
            break
    return mutated and not r_koota_change_flagged(c)


# ------------------------------------------------- classes vs traits ---
def r_koota_class_holds_state(c) -> bool:
    """A class of our own (no `extends`: an external library's class is the
    docs' exception) whose methods write its own fields: data and behaviour
    in one object (SKILL.md "Prefer traits + actions over classes")."""
    for _b, src, n in c.nodes({"class_declaration", "class"}):
        if any(ch.type == "class_heritage" for ch in n.children):
            continue
        body = n.child_by_field_name("body")
        if body is None:
            continue
        for m in body.named_children:
            if m.type != "method_definition":
                continue
            nm = m.child_by_field_name("name")
            if nm is not None and c.text(src, nm) == "constructor":
                continue
            for a in c.within(src, m, {"assignment_expression",
                                       "augmented_assignment_expression",
                                       "update_expression"}):
                tgt = a.child_by_field_name("left") or \
                    a.child_by_field_name("argument")
                if tgt is not None and c.text(src, tgt).startswith("this."):
                    return True
    return False


def r_koota_schema_traits_query_system(c) -> bool:
    """Data as schema traits (`trait({ ... })`) and the per-frame work as a
    query system (`world.query(...).updateEach(...)`)."""
    schema = False
    for _b, src, n in c.calls("trait"):
        a = _args(n)
        if a and a[0].type == "object":
            schema = True
    return schema and c.called("updateEach")


# ----------------------------------------------- for...of + entity.get ---
def _from_query(c, src, node, decls) -> bool:
    t = c.text(src, node)
    if re.search(r"\.query(First)?\(|\bquery\(|useQuery\(", t):
        return True
    return node.type == "identifier" and any(
        re.search(r"\.query\(|\bquery\(|useQuery\(", v)
        for v in decls.get(t, []))


def _has_get_or_set(c, src, node) -> bool:
    for x in c.within(src, node, {"call_expression"}):
        if _prop(c, src, x) in ("get", "set") and \
                _capitalised_first_arg(c, src, x):
            return True
    return False


def r_koota_query_loop_get_set(c) -> bool:
    """A query's entities walked one by one, each trait read with
    `entity.get(Trait)` / written with `entity.set(Trait, ...)` (a for...of
    or `.forEach` over `world.query(...)`), where the docs batch the data:
    "Prefer `updateEach`/`readEach` over `for...of` + `entity.get()`"."""
    decls = _decl_values(c)
    for _b, src, n in c.nodes({"for_in_statement"}):
        right = n.child_by_field_name("right")
        body = n.child_by_field_name("body")
        if right is not None and body is not None and \
                _from_query(c, src, right, decls) and \
                _has_get_or_set(c, src, body):
            return True
    for _b, src, n in c.nodes({"call_expression"}):
        if _prop(c, src, n) not in ("forEach", "map"):
            continue
        recv = _receiver(n)
        a = _args(n)
        if recv is not None and a and _from_query(c, src, recv, decls) \
                and _has_get_or_set(c, src, a[0]):
            return True
    return False


def r_koota_batch_update_each(c) -> bool:
    """The query's data read and written in one batch: updateEach /
    readEach (or useStores, the README's direct-store loop)."""
    return any(c.called(x) for x in ("updateEach", "readEach", "useStores"))


# -------------------------------------------------- reads during render ---
def r_koota_reads_world_in_render(c) -> bool:
    """A component reads koota state without a hook: `world.query(...)` /
    `queryFirst` or `entity.get(Trait)` in its body (render, useMemo, an
    effect or a timer) -- a snapshot that does not rerender on change. Event
    handlers (on* props) are not render reads."""
    for _b, src, n in c.nodes({"call_expression"}):
        p = _prop(c, src, n)
        if _receiver(n) is None:
            continue
        is_read = p in ("query", "queryFirst") or (
            p == "get" and _capitalised_first_arg(c, src, n))
        if is_read and _in_component(c, src, n) and \
                not _in_event_handler(c, src, n):
            return True
    return False


def r_koota_query_hook(c) -> bool:
    """The component's entity list comes from `useQuery` / `useQueryFirst`
    (koota/react), which rerender when the query's entities change."""
    return c.called("useQuery") or c.called("useQueryFirst")


def r_koota_trait_hook(c) -> bool:
    """A trait's value is read with `useTrait(entity, Trait)` (koota/react),
    which rerenders when it is added, removed or changes value."""
    return c.called("useTrait")


# ------------------------------------------- mutating from the view ---
def r_koota_view_mutates_world(c) -> bool:
    """A component spawns or destroys entities itself (in a click handler or
    a helper it declares) instead of through actions; the README's
    in-effect `world.spawn` / `entity.destroy()` cleanup is not counted."""
    for _b, src, n in c.nodes({"call_expression"}):
        if _prop(c, src, n) not in ("spawn", "destroy") or \
                _receiver(n) is None:
            continue
        if not _in_component(c, src, n):
            continue
        if _inside_call(c, src, n, {"createActions"} | _EFFECTS):
            continue
        return True
    return False


def r_koota_uses_actions(c) -> bool:
    """Mutations defined with `createActions((world) => ({ ... }))` and
    bound in the component with `useActions(actions)`."""
    return c.called("createActions") and c.called("useActions")


# ------------------------------------------------------------- setup ---
def r_koota_world_provider(c) -> bool:
    """The app is wrapped in `<WorldProvider world={...}>`."""
    return any(a == "world" for _e, a, _v in c.jsx_attrs("WorldProvider"))


def _create_world_calls(c):
    for _b, src, n in c.calls("createWorld"):
        yield src, n


def r_koota_world_module_scope(c) -> bool:
    """The world is created once, at module scope (never per render)."""
    calls = list(_create_world_calls(c))
    return bool(calls) and all(not _enclosing_fns(n) for _src, n in calls)


def r_koota_imports_koota_react(c) -> bool:
    """The React bindings come from 'koota/react'."""
    return _imports_from(c, r"koota/react")


def r_koota_world_setup_missing(c) -> bool:
    """No WorldProvider in the answer, or a world created inside a
    component (a new world on each mount or render)."""
    if not r_koota_world_provider(c):
        return True
    return any(_in_component(c, src, n) for src, n in _create_world_calls(c))
