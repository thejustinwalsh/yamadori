---
name: module-exports-match-callers
description: >-
  Use when writing or fixing a call from one JavaScript or TypeScript module to
  a function another module defines, or when an error says a function is not a function
  or not defined.
version: 1.0.0
author: Yamadori
license: MIT
metadata:
  hermes:
    tags: []
    related_skills:
    - browser-app-entry-point
    - fix-located-defect-first
  yamadori:
    state: draft
    applies_when:
      artifacts:
      - code
      languages:
      - javascript
      - typescript
      phases:
      - implement
      - debug
      situations:
      - multi_file
      - call_mismatch
      triggers:
      - a module object has no such method
      - wire calls between several script modules
    provenance:
      kind: authored
      source: >-
        Octopus v0e-V0-xhigh-1 prompt 2 (a call to a method the callee module never
        exported: `is not a function` in the browser console)
      log_rows: []
      licence:
        spdx: MIT
        quote: MIT License
        where: LICENSE (this repository)
    items:
    - line: >-
        - WHEN a module calls a function another module defines: open the callee's
        file and read what it exports (the returned object, its export statements
        or module.exports) before writing the call.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
    - line: >-
        - DO: call a function by the exact name and shape the callee exports; when
        it exists but is not exported, add it to the callee's exports.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
    - line: >-
        - WHEN an error says `X.y is not a function` or `y is not defined`: search
        the project for where the function is defined and where the object it is
        called on is built, and fix whichever side is wrong.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
    - line: >-
        - DO: after adding or renaming an exported function, search the project
        for every caller and update them in the same change.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
    - line: >-
        - DO NOT: invent a method name on a module object; if its export list lacks
        the method, add it there first.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
---
# Module Exports Match Their Callers

## When to use

Use when writing or fixing a call from one JavaScript or TypeScript module to a function another module defines, or when an error says a function is not a function or not defined.

## Guidance

- WHEN a module calls a function another module defines: open the callee's file and read what it exports (the returned object, its export statements or module.exports) before writing the call.
- DO: call a function by the exact name and shape the callee exports; when it exists but is not exported, add it to the callee's exports.
- WHEN an error says `X.y is not a function` or `y is not defined`: search the project for where the function is defined and where the object it is called on is built, and fix whichever side is wrong.
- DO: after adding or renaming an exported function, search the project for every caller and update them in the same change.
- DO NOT: invent a method name on a module object; if its export list lacks the method, add it there first.
