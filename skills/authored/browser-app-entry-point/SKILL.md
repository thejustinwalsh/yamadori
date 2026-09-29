---
name: browser-app-entry-point
description: >-
  Use when building, debugging or verifying a browser app whose JavaScript is split
  across several files loaded by one HTML page, especially when the page loads without
  errors but nothing runs or draws.
version: 1.0.0
author: Yamadori
license: MIT
metadata:
  hermes:
    tags: []
    related_skills:
    - fix-located-defect-first
    - module-exports-match-callers
  yamadori:
    state: draft
    applies_when:
      artifacts:
      - code
      languages:
      - javascript
      all_of:
      - html
      phases:
      - implement
      - debug
      - verify
      triggers:
      - a multi-file browser app loads but nothing runs or draws
      - wire the entry point of a browser game or app split into several scripts
    provenance:
      kind: authored
      source: >-
        docs/SELF-IMPROVEMENT-LOG.md rows #35 and #40 (Octopus v0b-V0-xhigh-1: a
        finished multi-file app with no call to its init; a harness that called
        init itself)
      log_rows:
      - '#35'
      - '#40'
      licence:
        spdx: MIT
        quote: MIT License
        where: LICENSE (this repository)
    items:
    - line: >-
        - WHEN the app is split into modules that each define an init or start function:
        make exactly one entry point call the app's init on page load: a DOMContentLoaded
        or load listener, or a final bootstrap line in the last script the page
        loads.
      ref: docs/SELF-IMPROVEMENT-LOG.md#35
    - line: >-
        - DO: load the file that holds the entry-point call last, after every module
        it uses, or import the modules from one script type="module" so the order
        is explicit.
      ref: docs/SELF-IMPROVEMENT-LOG.md#35
    - line: >-
        - WHEN the page loads with no console errors but nothing runs or draws:
        search the project for the call to the app's init (for example `Game.init(`)
        before anything else; a defined init that nothing calls is the usual cause.
      ref: docs/SELF-IMPROVEMENT-LOG.md#35
    - line: >-
        - DO: verify by loading the page itself (index.html in a browser, or a headless
        browser that only navigates to it) and checking that the first frame draws.
      ref: docs/SELF-IMPROVEMENT-LOG.md#40
    - line: >-
        - DO NOT: verify with a harness that calls init() itself: it passes while
        the real page still never boots.
      ref: docs/SELF-IMPROVEMENT-LOG.md#40
---
# Browser App Entry Point

## When to use

Use when building, debugging or verifying a browser app whose JavaScript is split across several files loaded by one HTML page, especially when the page loads without errors but nothing runs or draws.

## Guidance

- WHEN the app is split into modules that each define an init or start function: make exactly one entry point call the app's init on page load: a DOMContentLoaded or load listener, or a final bootstrap line in the last script the page loads.
- DO: load the file that holds the entry-point call last, after every module it uses, or import the modules from one script type="module" so the order is explicit.
- WHEN the page loads with no console errors but nothing runs or draws: search the project for the call to the app's init (for example `Game.init(`) before anything else; a defined init that nothing calls is the usual cause.
- DO: verify by loading the page itself (index.html in a browser, or a headless browser that only navigates to it) and checking that the first frame draws.
- DO NOT: verify with a harness that calls init() itself: it passes while the real page still never boots.
