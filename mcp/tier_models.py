#!/usr/bin/env python
"""ONE MODEL PER EFFORT TIER, and each model's ONE token profile (the table's one place: mcp/tier_models.yaml).

THE OPERATOR (2026-09-29)

  "If we get clean swaps and a decider on each model that actually works, we should make mirai xhigh flash-next max,
  and bonsai everything else. These models seem to do better on one reasoning level, so we pick the best reasoning
  level and capability is now the model swap."

  "Token settings, thinking efforts, etc. Are all going to need to be really dialed in from the research we have ...
  it is mostly about tuning how big of a turn we allow, and what the thinking budget should be. This is stuff the
  user and the harness will just get wrong, the proxy makes this atomic agent style, no thinking just solid work
  coming out of the box."

WHAT THIS MODULE IS

  The loader and the questions every other module asks of the table:

    model_for(tier)      the upstream model a tier is served by (every tier the table does not name: the default)
    rank(model)          the highest tier a model serves (mcp/max_mode.py: a higher tier's model waits for a lower
                         one's work and then takes the card; a lower tier is refused while a higher one holds it)
    profile(model)       the model's ONE token profile -- the effort sent to the template, the thinking budget and
                         nudge per route (user turn / agent step), the answer allowance, the turn floor, the
                         force-close message, sampling with thinking on and off. tiers.apply() and tiers.budget()
                         read it, so blocking and streaming, chat and Responses agree (one door).
    window(model)        the model's own KV pool and main cap (mcp/budget.py: the advertised and enforced window
                         is per model)
    vision(model)        whether the model's llama-swap entry carries its own projector (mcp/vision.py)

  Every profile value carries its CLASS and SOURCE (AGENTS.md "Claims carry their evidence": an operator decision
  quoted with its date, a derivation from a real constraint, or a measurement with its script and n). A value the
  table does not give falls back to today's constant in mcp/tiers.py with that constant's own source.

WHERE THE TABLE COMES FROM (env/config, never hardcoded here)

  YAMADORI_TIER_MODELS   a path to the table (YAML or JSON; relative paths from the repo root) -- the deploy
                         (bench/deploy_tier_models.py) sets it to mcp/tier_models.yaml for the proxy, the tools API
                         and the worker -- or an inline form for experiments: "xhigh=mirai-s,max=flash-next".
  YAMADORI_MAX_MODEL     the 2026-09-28 max-mode switch, still read: {max: <it>} when YAMADORI_TIER_MODELS is unset.
  neither                OFF: one model (YAMADORI_MODEL), every function answers exactly what the stack did before.

  The table is read once per process (reload() re-reads it: the tests, and a deploy restarts the services).
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORDER = ["minimal", "low", "medium", "high", "xhigh", "max"]
DEFAULT_MODEL = os.environ.get("YAMADORI_MODEL", "bonsai")

# The profile's keys and, for each, where today's value lives when the table gives none (mcp/tiers.py).
PROFILE_KEYS = ("effort", "user_turn_thinking", "agent_step_thinking", "nudge_at", "nudge_user_turn",
                "nudge_agent_step", "answer_tokens", "turn_min_tokens", "force_close", "sampling_thinking",
                "sampling_instruct", "job_caps")


class Table:
    def __init__(self, spec: dict | None, source: str, full: bool = False):
        spec = spec or {}
        self.source = source
        # FULL: the table came from a FILE (YAMADORI_TIER_MODELS=<path>) -- the token profiles, the explicit timed
        # swap, the per-model windows and the general blocking apply. The older forms (YAMADORI_MAX_MODEL, the inline
        # tier=model list) keep the 2026-09-28 max-mode behaviour exactly: routing only (mcp/test_max_mode.py
        # test_legacy_is_inert gates it).
        self.full = bool(full)
        self.default = str(spec.get("default") or DEFAULT_MODEL)
        self.tiers: dict[str, str] = {}
        for t, m in (spec.get("tiers") or {}).items():
            t = str(t).strip().lower()
            if t not in ORDER:
                raise ValueError(f"tier_models: unknown tier {t!r} (tiers are {ORDER})")
            if m:
                self.tiers[t] = str(m).strip()
        self.models: dict[str, dict] = {}
        for m, row in (spec.get("models") or {}).items():
            self.models[str(m)] = dict(row or {})
        for m in set(self.tiers.values()) | {self.default}:
            self.models.setdefault(m, {})

    # ------------------------------------------------------------ routing --
    @property
    def profiles_on(self) -> bool:
        return self.enabled and self.full

    @property
    def enabled(self) -> bool:
        """More than one main model: something other than the default serves a tier."""
        return any(m != self.default for m in self.tiers.values())

    def model_for(self, tier: str | None) -> str:
        if not self.enabled:
            return self.default
        return self.tiers.get(str(tier or "").lower(), self.default)

    def main_models(self) -> list[str]:
        """Every main model (the ones that hold the 5060 Ti), in rank order, the default first."""
        seen = [self.default] + [m for m in self.tiers.values() if m != self.default]
        out: list[str] = []
        for m in seen:
            if m not in out:
                out.append(m)
        return sorted(out, key=self.rank)

    def tiers_of(self, model: str) -> list[str]:
        return [t for t in ORDER if self.model_for(t) == model]

    def rank(self, model: str | None) -> int:
        """The index (in ORDER) of the highest tier the model serves; -1 for a model that serves none."""
        ts = self.tiers_of(model or "")
        return max(ORDER.index(t) for t in ts) if ts else -1

    # ------------------------------------------------------------ profile --
    def row(self, model: str | None) -> dict:
        return self.models.get(model or self.default) or {}

    def window(self, model: str | None) -> dict | None:
        """{ctx, slots, main_cap?} the table declares for a model, or None (the default model's pool is read from its
        /props by mcp/budget.py, as before)."""
        w = self.row(model).get("window")
        if not isinstance(w, dict) or not w.get("ctx"):
            return None
        out = {"ctx": int(w["ctx"])}
        for k in ("slots", "main_cap"):
            if w.get(k) is not None:
                out[k] = int(w[k])
        out["source"] = str(w.get("source") or "")
        return out

    def helpers(self, model: str | None) -> dict:
        """Where this model's jjava (the decider, the skills injector, the Jev API) and the client's side calls
        run: {decider, side_calls}, each a llama-swap model id, or "self" (the main card's lane). Only a table FILE
        routes helpers (the older forms keep today's behaviour)."""
        h = self.row(model).get("helpers") if self.full else None
        return dict(h) if isinstance(h, dict) else {}

    def locked(self, model: str | None) -> bool:
        """The model runs ALONE on the card: one conversation slot, no lane, and no OTHER work touches it -- no
        decider, no side calls, no other conversation, no release (operator, 2026-09-30, for Flash-Next, then Bonsai).
        Its own conversation's work is allowed: its compactions, its warms, and a /slots read of the loaded model
        (max_mode.OWN_WORK; coordinator 2026-09-30)."""
        return bool(self.full and self.row(model).get("locked"))

    def vision(self, model: str | None) -> bool | None:
        v = self.row(model).get("vision")
        return None if v is None else bool(v)

    def profile(self, model: str | None) -> dict:
        """{key: {value, class, source}} for every PROFILE_KEYS entry: the table's value, else today's constant."""
        raw = self.row(model).get("profile") or {}
        out = {}
        base = defaults()
        for k in PROFILE_KEYS:
            if k in raw:
                v = raw[k]
                out[k] = dict(v) if isinstance(v, dict) and "value" in v else {"value": v}
                out[k].setdefault("class", "unlabelled")
                out[k].setdefault("source", f"{self.source}: models.{model}.profile.{k}")
            else:
                out[k] = dict(base[k])
        return out


def defaults() -> dict:
    """Today's constants (mcp/tiers.py), as profile entries with their own sources: what a model gets for a key the
    table does not set."""
    import tiers
    return {
        "effort": {"value": None, "class": "tier",
                   "source": "the tier's own effort (tiers.TIERS): the table sets none for this model"},
        "user_turn_thinking": {"value": tiers.USER_TURN_THINKING, "class": "operator",
                               "source": "tiers.USER_TURN_THINKING (operator 2026-09-27, bonsai-ada-surgery's "
                                         "20,480)"},
        "agent_step_thinking": {"value": tiers.AGENT_STEP_THINKING, "class": "operator",
                                "source": "tiers.AGENT_STEP_THINKING (operator 2026-09-25)"},
        "nudge_at": {"value": tiers.NUDGE_AT, "class": "operator", "source": "tiers.NUDGE_AT (operator 2026-09-25)"},
        "nudge_user_turn": {"value": tiers.NUDGE_MESSAGE, "class": "operator", "source": "tiers.NUDGE_MESSAGE"},
        "nudge_agent_step": {"value": tiers.AGENT_STEP_NUDGE_MESSAGE, "class": "operator",
                             "source": "tiers.AGENT_STEP_NUDGE_MESSAGE (approved 2026-09-26)"},
        "answer_tokens": {"value": tiers.A_MIN, "class": "operator", "source": "tiers.A_MIN"},
        "turn_min_tokens": {"value": None, "class": "none", "source": "no turn floor beyond the answer allowance"},
        "force_close": {"value": tiers.BUDGET_MESSAGE, "class": "operator",
                        "source": "tiers.BUDGET_MESSAGE (operator 2026-09-25: the original line)"},
        "sampling_thinking": {"value": dict(tiers.VENDOR_SAMPLING), "class": "vendor",
                              "source": "tiers.VENDOR_SAMPLING (PrismML's card = Qwen's thinking values)"},
        "sampling_instruct": {"value": dict(tiers.VENDOR_SAMPLING_INSTRUCT), "class": "vendor",
                              "source": "tiers.VENDOR_SAMPLING_INSTRUCT"},
        "job_caps": {"value": True, "class": "operator",
                     "source": "tiers.JOB_THINKING / HELPER_THINKING (derived on Bonsai)"},
    }


# ------------------------------------------------------------------ load --
def _inline(s: str) -> dict:
    """"xhigh=mirai-s,max=flash-next" (an optional ":effort" after a model sets that model's effort)."""
    spec: dict = {"tiers": {}, "models": {}}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"tier_models: {part!r} is not tier=model")
        t, m = (x.strip() for x in part.split("=", 1))
        if ":" in m:
            m, eff = (x.strip() for x in m.split(":", 1))
            spec["models"].setdefault(m, {}).setdefault("profile", {})["effort"] = {
                "value": eff, "class": "inline", "source": "YAMADORI_TIER_MODELS inline"}
        spec["tiers"][t] = m
    return spec


def _read_file(path: str) -> dict:
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(p, encoding="utf-8") as f:
        text = f.read()
    if p.endswith(".json"):
        return json.loads(text)
    import yaml
    return yaml.safe_load(text) or {}


def load(env: dict | None = None) -> Table:
    env = os.environ if env is None else env
    spec_s = (env.get("YAMADORI_TIER_MODELS") or "").strip()
    if spec_s:
        if spec_s.endswith((".yaml", ".yml", ".json")) or os.path.sep in spec_s or "/" in spec_s:
            return Table(_read_file(spec_s), spec_s, full=True)
        return Table(_inline(spec_s), "YAMADORI_TIER_MODELS (inline)")
    mx = (env.get("YAMADORI_MAX_MODEL") or "").strip()
    if mx:
        return Table({"tiers": {"max": mx}}, "YAMADORI_MAX_MODEL")
    return Table({}, "off (YAMADORI_TIER_MODELS and YAMADORI_MAX_MODEL unset)")


_TABLE: Table | None = None


def table() -> Table:
    global _TABLE
    if _TABLE is None:
        try:
            _TABLE = load()
        except Exception as e:                                       # noqa: BLE001
            # A table that cannot be read must not take the stack down: one model, and the reason kept.
            print(f"  tier_models: the table could not be read ({type(e).__name__}: {e}); one model", flush=True)
            _TABLE = Table({}, f"unreadable: {type(e).__name__}: {e}"[:200])
    return _TABLE


def reload(env: dict | None = None) -> Table:
    global _TABLE
    _TABLE = load(env)
    return _TABLE


def enabled() -> bool:
    return table().enabled


def profiles_on() -> bool:
    """The token profiles apply (a table FILE is configured); False keeps every tiers.py constant as before."""
    return table().profiles_on


def model_for(tier: str | None) -> str:
    return table().model_for(tier)


def profile(model: str | None) -> dict:
    return table().profile(model)


def value(model: str | None, key: str):
    return (profile(model).get(key) or {}).get("value")


def describe() -> dict:
    """The table as the dashboard and x_yamadori show it: tier -> model, each model's rank, window and profile
    values with their classes (texts as their length)."""
    t = table()
    out = {"enabled": t.enabled, "profiles": t.profiles_on, "source": t.source, "default": t.default,
           "tiers": {tier: t.model_for(tier) for tier in ORDER}, "models": {}}
    for m in t.main_models():
        prof = {}
        for k, v in t.profile(m).items():
            val = v.get("value")
            prof[k] = {"value": (f"<{len(val)} chars>" if isinstance(val, str) and len(val) > 40 else val),
                       "class": v.get("class")}
        out["models"][m] = {"rank": t.rank(m), "tiers": t.tiers_of(m), "window": t.window(m),
                            "vision": t.vision(m), "locked": t.locked(m), "helpers": t.helpers(m),
                            # THE OTHER CARD (mcp/slots.py): where a second conversation of this model's tiers runs
                            "other_card": t.row(m).get("other_card"), "profile": prof}
    return out


if __name__ == "__main__":
    print(json.dumps(describe(), indent=1))
