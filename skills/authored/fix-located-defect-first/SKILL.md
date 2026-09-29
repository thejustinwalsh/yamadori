---
name: fix-located-defect-first
description: >-
  Use when debugging and a defect has been located to a specific file and line or
  a named missing function: apply the fix before building any new diagnostics.
version: 1.0.0
author: Yamadori
license: MIT
metadata:
  hermes:
    tags: []
    related_skills:
    - browser-app-entry-point
    - module-exports-match-callers
  yamadori:
    state: draft
    applies_when:
      artifacts:
      - code
      phases:
      - debug
      situations:
      - error_output
      - call_mismatch
      - file_line
      triggers:
      - the error names the missing function and the file it is called from
      - fix the reported bugs in the project
    provenance:
      kind: authored
      source: >-
        Octopus v0e-V0-xhigh-1 prompt 2 (the model located a missing function at
        a named line, then spent 5.4 h building diagnostics and wrote no fix) and
        docs/SELF-IMPROVEMENT-LOG.md #40
      log_rows:
      - '#40'
      licence:
        spdx: MIT
        quote: MIT License
        where: LICENSE (this repository)
    items:
    - line: >-
        - WHEN a defect is located to a file and line: apply the smallest change
        that fixes it first, then re-run the check that exposed it.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
    - line: >-
        - DO: confirm a fix with the check that already exists (the test, the console
        error, the reported failure); build new test harnesses or diagnostic tools
        only when no existing check can observe the defect.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
    - line: >-
        - WHEN several defects are reported together: fix them one at a time in
        the order given, re-run the check after each, and say which are resolved.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
    - line: >-
        - DO: write the fix into the project's own files; a scratch script that
        reproduces the bug changes nothing the user runs.
      ref: docs/SELF-IMPROVEMENT-LOG.md#40
    - line: >-
        - DO NOT: start a new diagnostic tool, screenshot pipeline or test framework
        while a located defect is still unfixed.
      ref: Octopus v0e-V0-xhigh-1 prompt 2
---
# Fix a Located Defect First

## When to use

Use when debugging and a defect has been located to a specific file and line or a named missing function: apply the fix before building any new diagnostics.

## Guidance

- WHEN a defect is located to a file and line: apply the smallest change that fixes it first, then re-run the check that exposed it.
- DO: confirm a fix with the check that already exists (the test, the console error, the reported failure); build new test harnesses or diagnostic tools only when no existing check can observe the defect.
- WHEN several defects are reported together: fix them one at a time in the order given, re-run the check after each, and say which are resolved.
- DO: write the fix into the project's own files; a scratch script that reproduces the bug changes nothing the user runs.
- DO NOT: start a new diagnostic tool, screenshot pipeline or test framework while a located defect is still unfixed.
