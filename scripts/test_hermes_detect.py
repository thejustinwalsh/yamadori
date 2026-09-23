#!/usr/bin/env python
"""Does repo detection work on what Hermes ACTUALLY sends?

The synthetic prompts this was first tested against were written by the same
person who wrote the detector, which is not a test. Hermes emits, from
prompt_builder.py line 995:

    host_lines.append(f"Current working directory: {resolve_agent_cwd()}")

On the first real run against Hermes, detection returned None and every tool
call searched an empty index. This checks the real string, in every form
Windows might render it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import detect  # noqa: E402

REPO = os.path.join("C:", os.sep, "Users", "jwals", "gauntlet", "koota")
FORMS = {
    "windows backslash": REPO,
    "forward slash": REPO.replace(os.sep, "/"),
    "quoted": f'"{REPO}"',
    "trailing slash": REPO.replace(os.sep, "/") + "/",
}


def main() -> None:
    print(f"  repo exists on disk: {os.path.isdir(REPO)}")
    print(f"  has .git           : {os.path.exists(os.path.join(REPO, '.git'))}\n")
    for label, form in FORMS.items():
        line = f"Current working directory: {form}"
        msgs = [{"role": "system",
                 "content": f"You are Hermes.\n{line}\nPlatform: win32"},
                {"role": "user", "content": "Where is Trait defined?"}]
        root, _ = detect.detect_repo(msgs, trusted_only=True)
        got = os.path.basename(root) if root else None
        print(f"  {'ok  ' if got == 'koota' else 'MISS'} {label:<20} -> {got}")
        if got is None:
            print(f"       paths extracted: {detect.extract_paths(line)}")


if __name__ == "__main__":
    main()
