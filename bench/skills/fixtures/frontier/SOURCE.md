# Frontier-skill fixture: systematic-debugging

`systematic-debugging/SKILL.md` is a verbatim copy of Hermes Agent's bundled
skill `skills/software-development/systematic-debugging/SKILL.md`
(hermes-agent commit ee5ee84a345204a3b1d6ef6ba1ab747e602867b9, 2026-09-24;
sha256 bd480f51a2e9969c88544003c30b7539b7df0b7358703f99b0495514c2dd878a),
itself "adapted from obra/superpowers". Hermes Agent is MIT-licensed:

    MIT License

    Copyright (c) 2025 Nous Research

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

It is a LOCAL fixture: the frontier path was exercised offline (no network,
no live model). `decompose_reply.txt` is a HAND-WRITTEN stand-in for the
model's `decompose/1` reply, written by reading the skill -- it is NOT model
output. It exists so `bench/skills/frontier_example.py` can show the rest of
the pipeline (screen, licence from the frontmatter line, quote verification,
tagging, activation tests, arming) on a real frontier skill. The live
decompose on this fixture has not run.
