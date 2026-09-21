#!/usr/bin/env python
"""Repo detection against the shapes a real client actually sends."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
from detect import detect_repo  # noqa: E402

G = "C:/Users/jwals/gauntlet"

CASES = [
    ("harness states cwd in system prompt", [
        {"role": "system", "content": f"You are a coding agent.\nWorking directory: {G}/glyph\nPlatform: win32"},
        {"role": "user", "content": "add a dashOffset property"}], "glyph"),

    ("only a pasted stack trace", [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": f"TypeError at {G}/koota/packages/core/src/world.ts:42"}], "koota"),

    ("tool result echoes a windows path", [
        {"role": "user", "content": "why is this failing"},
        {"role": "tool", "content": "read " + f"{G}/deherm/packages/abi/src/index.ts".replace("/", "\\") + " ok"}], "deherm"),

    ("conversation moves repos, recency should win", [
        {"role": "user", "content": f"earlier we looked at {G}/glyph/packages/glyph/src/index.ts"},
        {"role": "user", "content": f"now look at {G}/koota/packages/core/src/entity.ts"},
        {"role": "user", "content": f"and {G}/koota/packages/core/src/trait.ts"}], "koota"),

    ("no paths at all", [
        {"role": "user", "content": "what is a monad"}], None),

    ("node_modules must not become the root", [
        {"role": "user", "content": f"error in {G}/glyph/node_modules/three/build/three.module.js:100"}], "glyph"),

    ("path that does not exist on disk is ignored", [
        {"role": "user", "content": "see /home/someone/not-real/src/main.rs"}], None),

    ("multimodal content blocks", [
        {"role": "user", "content": [{"type": "text", "text": f"look at {G}/deherm/packages/cli/src/main.ts"}]}], "deherm"),
]


def main() -> None:
    ok = 0
    for name, msgs, want in CASES:
        root, ev = detect_repo(msgs)
        got = os.path.basename(root) if root else None
        hit = got == want
        ok += hit
        note = ev.get("reason") or f"margin={ev.get('margin')}"
        print(f"  {'ok  ' if hit else 'MISS'} {name:<44} -> {str(got):<8} want={want}   {note}")
    print(f"\n  {ok}/{len(CASES)} correct")


if __name__ == "__main__":
    main()
