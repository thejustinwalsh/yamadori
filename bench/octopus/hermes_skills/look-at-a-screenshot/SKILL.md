---
name: look-at-a-screenshot
description: "Look at a screenshot or image you made inside the docker sandbox with vision_analyze."
version: 1.1.0
author: Yamadori Octopus runner
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [screenshot, vision, image, docker, sandbox, playwright, verify]
    related_skills: []
---

# Look at a Screenshot You Made

Use this when your terminal runs in the docker sandbox (a /workspace project)
and you want to look at a screenshot or image you produced there, e.g. a
Playwright capture of the page you built.

In this profile, /root/.hermes/cache/screenshots/octo/ is a folder that both
the sandbox and vision_analyze can read (bench/octopus/run.py mounts it).
Any other container path is read on the host, where it does not exist, and
the call fails without saying why.

- Save the image, or `cp` it from /workspace or /tmp, to
  /root/.hermes/cache/screenshots/octo/<name>.png.
- Call vision_analyze with that exact path as image_url.
- Ask one concrete question (is the canvas blank, where is the ship, which
  colour is the HUD) rather than a general description.
- Pass the file path, not base64 or a data: URL.

## A slow page in the sandbox browser

The sandbox browser has no GPU: it renders WebGL in software. A heavy 3D
scene can take seconds per frame, and while a frame renders the browser's
commands (navigate, screenshot, evaluate) can time out. That is the page
being slow, not broken. Read the console errors and take one screenshot;
when there are no errors, the page is working.
