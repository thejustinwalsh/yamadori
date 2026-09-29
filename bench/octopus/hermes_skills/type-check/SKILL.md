---
name: type-check
description: "Check TypeScript or Python for type errors after writing or changing it in the docker sandbox, before calling the work done or running it."
version: 1.0.0
author: Yamadori Octopus runner
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [typescript, python, types, tsc, pyright, verify, docker, sandbox]
    related_skills: []
---

# Type-check what you changed

Use this after you write or patch `.ts`, `.tsx` or `.py` files in the docker
sandbox (a /workspace project). The file tools' own lint does not type-check
a TypeScript project that has a `tsconfig.json`, so run the checker yourself
with `terminal` from the project root and read every error it prints (file,
line, column).

| project | command |
|---|---|
| TypeScript with a `tsconfig.json` and its own `typescript` installed | `npx --no-install tsc --noEmit -p .` |
| TypeScript, no compiler installed in the project | `/opt/yamadori-tools/bin/tsc --noEmit -p .` |
| TypeScript files, no `tsconfig.json` | `/opt/yamadori-tools/bin/tsc --noEmit --strict src/file.ts` |
| Python | `/opt/yamadori-tools/bin/pyright path/to/file.py` |

`/opt/yamadori-tools` holds TypeScript 5.9.3 and pyright 1.1.414, read-only,
and needs no network. Exit status 0 means no errors.

Fix the first error, then run the check again: later errors are often caused
by earlier ones.
