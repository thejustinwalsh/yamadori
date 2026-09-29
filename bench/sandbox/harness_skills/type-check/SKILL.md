---
name: type-check
description: "Check TypeScript or Python for type errors after writing or changing it, before calling the work done or running it."
license: MIT
metadata:
  yamadori:
    source: bench/sandbox/harness_skills (harness box default loadout, docs/HARNESSES.md)
---

# Type-check what you changed

Run the language's own checker from the project root and read every error it
prints. It reports the file, line and column.

| project | command |
|---|---|
| TypeScript with a `tsconfig.json` | `npx --no-install tsc --noEmit -p .` (the project's own compiler); if the project has none installed, `tsc --noEmit -p .` |
| TypeScript files, no `tsconfig.json` | `tsc --noEmit --strict path/to/file.ts` |
| Python | `pyright path/to/file.py` (or `pyright` for the whole project) |

`tsc` (TypeScript 5.9.3) and `pyright` (1.1.414) are installed in this
environment and need no network. Exit status 0 means no errors.

Fix the first error, then run the check again: later errors are often caused
by earlier ones.
