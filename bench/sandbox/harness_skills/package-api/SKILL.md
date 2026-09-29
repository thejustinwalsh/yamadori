---
name: package-api
description: "Look up how to call an installed package -- the functions, components, hooks and types it exports, with their exact signatures -- before writing code that uses it, or when a call to it does not type-check."
license: MIT
metadata:
  yamadori:
    source: bench/sandbox/harness_skills (harness box default loadout, docs/HARNESSES.md)
---

# Look up an installed package's API

One command prints how a package's exports are called, from the package's
own type declarations for the version installed -- what an editor shows on
hover. Run it from the project root, with the module name your code
imports:

| to see | command |
|---|---|
| names in full: every overload, a type's members, the doc comment, file:line | `node ~/.agents/skills/package-api/api.cjs koota createWorld World` |
| one member of a type or namespace | `node ~/.agents/skills/package-api/api.cjs koota World.query` |
| the exports whose names match | `node ~/.agents/skills/package-api/api.cjs three 'Instanced*'` |
| a sub-entry of the package | `node ~/.agents/skills/package-api/api.cjs koota/react useQuery` |
| every export, one line each, then the package's other entry points | `node ~/.agents/skills/package-api/api.cjs koota` |

A name the package root does not export is looked up in its other entry
points and inside its namespaces (`api.cjs math fromEuler` answers with
`quat.fromEuler`), and the answer says where it was found. A large
package's full list is long (three has about 680 exports); a name or a
pattern prints only what you ask for.

Write the code from those signatures, then run the type check (skill
`type-check`): its errors name the argument or property that does not fit.
