#!/usr/bin/env python
"""Every size limit on skills, in one place.

THESE ARE FUZZY BEST-PRACTICE TARGETS, NOT MEASUREMENTS.

Nothing in this repo has measured how long a skill should be for this model.
The numbers follow published practice and the operator's targets
(2026-09-24):

  - Anthropic Agent Skills: a SKILL.md `description` is at most 1,024
    characters, and the body is kept short, with detail disclosed
    progressively (docs/DECISION-TREES-AND-SKILLS.md §2.2, [R] S23-S25).
  - SkillsBench v4 reports compact skills ahead of detailed ones and long
    documentation near zero ([R] S34, §2.3); instruction-following falls as
    instructions pile up ([R] S35).
  - The operator's targets for our compressed form: aim ~300-500 tokens per
    skill, hard cap ~800; per turn aim <= ~1,500 injected tokens, hard cap
    ~2,500, dropping the lowest-confidence skill first. A raw long document
    is never injected.

`AIM` values steer (the builder is told them; the selector stops adding at
them). `HARD` values are enforced: the validator drops trailing items until a
skill fits, and the selector never exceeds the per-turn cap.

Tokens are ESTIMATED at 3 characters per token -- tiers.estimate_prompt_tokens'
deliberately high rate, so a cap errs toward injecting less.
"""
from __future__ import annotations

import math
import os


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


CHARS_PER_TOKEN = 3.0

# Per skill (the compressed form: title + applies-when + items).
SKILL_TOKENS_AIM = (300, 500)
SKILL_TOKENS_HARD = _i("YAMADORI_SKILL_TOKENS_HARD", 800)
MAX_ITEMS = _i("YAMADORI_SKILL_MAX_ITEMS", 10)
MAX_ITEM_CHARS = _i("YAMADORI_SKILL_MAX_ITEM_CHARS", 240)
MAX_TITLE_CHARS = 80
# AGENTS.md "Prompting this model": prohibitions degrade the model
# monotonically (0 > 2 > 6, at-risk numbers); keep at most two.
MAX_DO_NOT = 2

# Per turn (everything the selector injects into one request).
TURN_TOKENS_AIM = _i("YAMADORI_SKILL_TURN_TOKENS_AIM", 1500)
TURN_TOKENS_HARD = _i("YAMADORI_SKILL_TURN_TOKENS_HARD", 2500)
MAX_SKILLS_PER_TURN = _i("YAMADORI_SKILL_K", 3)

# Triggers: the Anthropic `description` cap, and how many trigger lines a
# skill keeps (from its source, plus what the fallback teaches it).
DESCRIPTION_CHARS = 1024
MAX_TRIGGERS = 8
TRIGGER_CHARS = 200
MAX_LEARNED_TRIGGERS = _i("YAMADORI_SKILL_MAX_LEARNED_TRIGGERS", 40)


def tokens(text: str) -> int:
    """Estimated tokens: high on purpose."""
    return int(math.ceil(len(text or "") / CHARS_PER_TOKEN))


def summary() -> dict:
    """For the dashboard: every limit, and the label that goes with them."""
    return {"label": "fuzzy best-practice targets, not measurements",
            "chars_per_token": CHARS_PER_TOKEN,
            "skill_tokens_aim": list(SKILL_TOKENS_AIM),
            "skill_tokens_hard": SKILL_TOKENS_HARD,
            "max_items": MAX_ITEMS, "max_item_chars": MAX_ITEM_CHARS,
            "max_do_not": MAX_DO_NOT,
            "turn_tokens_aim": TURN_TOKENS_AIM,
            "turn_tokens_hard": TURN_TOKENS_HARD,
            "max_skills_per_turn": MAX_SKILLS_PER_TURN,
            "description_chars": DESCRIPTION_CHARS,
            "max_triggers": MAX_TRIGGERS,
            "max_learned_triggers": MAX_LEARNED_TRIGGERS}
