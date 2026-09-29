---
name: page-check
description: "Open a web page you are building in a real browser: see its console errors and uncaught exceptions, read what it shows, click and type, take a screenshot."
license: MIT
metadata:
  yamadori:
    source: bench/sandbox/harness_skills (harness box default loadout, docs/HARNESSES.md)
---

# Check a page in the browser

A Chromium browser is already running beside this environment and shares its
`localhost`. Drive it with the `agent-browser` command through the shell,
always with `--cdp 9322`:

```
agent-browser --cdp 9322 open http://localhost:5173/
agent-browser --cdp 9322 errors --json # uncaught exceptions of the page
agent-browser --cdp 9322 console       # console messages (errors included)
agent-browser --cdp 9322 snapshot -i   # interactive elements, with @refs to click
agent-browser --cdp 9322 click @e2
agent-browser --cdp 9322 press Space
agent-browser --cdp 9322 screenshot /tmp/page.png
```

Start your dev server in the background first (for example
`npm run dev -- --host 127.0.0.1 --port 5173 > /tmp/dev.log 2>&1 &`), then open
its address. After a change, reload (`agent-browser --cdp 9322 reload`) and
read `errors --json` and `console` again (plain `errors` prints only a marker per error in 0.26.0; `--json` has the text): an error that shows only while the page
runs never appears in the build output.

To see the page, take the screenshot and open the PNG with the `read` tool
(`read /tmp/page.png`): it comes back as an image you can look at.

This browser has no GPU: it renders WebGL in software. A heavy 3D scene can
take seconds per frame, and while a frame renders, `open`, `reload`,
`screenshot` and `eval` can answer `CDP command timed out`. That is the page
being slow, not broken. Read `errors --json` and take one screenshot; when
there are no errors, the page is working.

`agent-browser skills get core` prints the full command guide.
