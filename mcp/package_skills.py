#!/usr/bin/env python
"""THE PACKAGE SKILLS CHANNEL: proven skills ride in the results of OUR tool
calls the model already wants to make, and on evidence between turns.

    package_skills.decide(st, packages, ctx, pool=...)  -> (text, records, st)
    package_skills.step_triggers(messages, st, ctx)     -> [trigger, ...]
    package_skills.switch(tier)                         -> {on, source, mode, mode_source}

Operator, 2026-10-06 (verbatim): "Yes skills can not be selected from the
first prompt alone, if nothing else I was thinking our skills need a way to
trigger between turns, or our agent needs an mcp or tool call to get skill
guidance from so the model is compelled to seal it out. If the model uses some
of our other tools or mcps this may be a good point to skill up. Like didn't we
add the package mcp that helps if find package versions and info, this is where
skills probably hit harder, structured into a tool call our model wants to
make." And: "Right package can even lean into discovery like it can tell the
model you can call this tool for more expert advice on using the package, etc.
research showed injection did work better than discovery, but I don't know that
research measured discovery in the way I am proposing it, if a model was told
the tool to call and given a few router like decisions maybe it would be
inclined to call the craft tool for more of the skill from a more simple
instruction about the skills available bound to the package fetch."

WHY IT FITS THE EVIDENCE. bench/mcp/results/lookup_probe.jsonl (Pi, Bonsai, the
vague pagoda prompt, n=4 valid trials): the model called our package lookup
before installing in 4/4. The jjava injector (bonsai-a4000, 193 labelled cases,
docs/JJAVA.md 9) separates on-topic from off-topic (AUROC 0.92) and not NEEDED
from same-AREA (0.63): picking among ~500 skills from a first prompt fails. An
exact package name out of a tool call is a FACT, so no decider is asked here:
the package decides, the registry maps it to its skills, and the code ranks
them by what was proven. Everything below is deterministic and runs without the
GPU.

THE TWO RENDERINGS (`package_skills_mode`; X-Yamadori-Features, else
YAMADORI_PACKAGE_SKILLS_MODE, else `inject`). A comparison of two delivery
designs the operator asked to test, not an on/off arm:
  inject  the package's LEAD skill and the strongest items of its other
          skills, in the serving model's profile (skill_inject.render), then
          the craft names and a pointer: yama_recall_craft returns any in full
  router  NO skill body: one decision-router row per armed, proven craft of the
          package and major version -- "when <the craft's trigger> -> call
          yama_recall_craft {"name_or_topic": "<craft>"}" -- under one positive
          line naming the tool and the moment (THE TOOL RECIPE rule 4)
  both    the lead skill's body (inject) and the router table for the rest
router and both need yama_recall_craft on main (it is offered with the MCP
tools whenever this channel is on: proxy._package_craft_offer); without it the
section is the inject form and says so in its record.

WHERE IT GOES. A package tool's section is appended to that tool result inside
the proxy's hidden hop (proxy._run_turn), so the ledger replays it byte for
byte with the hop and the slot's cache extends. The between-turn triggers go on
the client's tool result the request ends on, as the existing `skills` part of
the ledger's `inject` row. Our text is never evidence: skill_select's
evidence_view strips our blocks.

WHAT RIDES. Only ARMED skills (a quarantined or superseded skill is not in
skills.armed()); a skill whose PROVE verdict is `worse`, `unproven` or
`not_run` is left out; ranking: the verdict (better, then tie, then no record),
then the pitfall cases that name it (bench/skills/pitfalls/*.jsonl), then name.
A skill that names a major version other than the package's is left out
(skill_select.skill_majors). The package's skills are the lead (skill_packages.
canonical_skill: lead_for, else the registry's canonical row by major), the
skills whose metadata names the package, and -- for the package that names
the area (package_registry.area_package) -- the area's skills. Every item goes
through skill_limits.doubt (the ASSURED VOICE): a doubt-bearing line never
rides.

SIZES, all existing constants and none new: at most skill_inject.MAX_SKILLS (3,
SkillsBench) skills and the profile's max_items items per package, under
skill_limits.SKILL_TOKENS_HARD (one skill body's hard cap) tokens; the router
table at most skill_limits.INDEX_MAX rows of skill_limits.INDEX_LINE_CHARS
characters under the same token cap; at most skill_limits.MAX_SKILLS_PER_TURN
skills in one call's section.

ONCE. Each skill is given once per conversation per major version (state:
`pkg` in the ledger's `skills` row, beside the selector's `given`, which it
also marks); the same package again later gets a recall line (the craft's own
first DO and first DO NOT, skill_prompts.CRAFT_RECALL_*), never the identical
text twice in a row; the router table is not repeated. A compaction resets
what was given (the selector's rule).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import package_registry  # noqa: E402
import skill_limits as L  # noqa: E402

MODES = ("inject", "router", "both")
# "both" (the lead skill's body + the router table for the rest): the default the operator turned on with the
# channel (2026-10-06, "turn them on and run the job you need to gather evidence"); the probe compares the three.
DEFAULT_MODE = "both"
MODE_ENV = "YAMADORI_PACKAGE_SKILLS_MODE"
# The tools whose result carries a section: the MCP package tools.
TOOLS = ("yama_find_package", "yama_list_package_versions",
         "yama_read_package_readme", "yama_resolve_packages")
CRAFT_TOOL = "yama_recall_craft"
CRAFT_ARG = "name_or_topic"
# A PROVE verdict that keeps a skill off this channel, and the rank of the
# others (better first). No record: an armed skill from before PROVE existed.
PROVE_EXCLUDED = ("worse", "unproven", "not_run")
PROVE_RANK = {"better": 0, "tie": 1}
NO_RECORD_RANK = 2

# ---- the texts (UNMEASURED WORDING until bench/mcp/lookup_probe.py's
# package-* arms have run; each is positive and states a moment, THE TOOL
# RECIPE rule 4) ---------------------------------------------------------
ROUTER_HEAD = ("Craft for {package}{major} from this service's library: when "
               "the work reaches one of these, call {tool} with the craft's "
               "name to read it in full; one or two crafts are usually "
               "enough for a step.")
ROUTER_COLUMNS = "| when you are about to | call " + CRAFT_TOOL + " with |\n|---|---|"
POINTER = ("Crafts for {package}{major}: {names}. {tool} returns any of them "
           "in full.")
SECTION_RULE = "\n---\n"


# ------------------------------------------------------------------ switch --
def mode_source(tier: dict | None) -> tuple[str, str]:
    """(mode, source): the header's `package_skills_mode` (tier overridden),
    else YAMADORI_PACKAGE_SKILLS_MODE, else inject. A value outside MODES is
    ignored (and the source says so)."""
    tier = tier or {}
    if "package_skills_mode" in (tier.get("overridden") or []):
        v = str(tier.get("package_skills_mode") or "").strip().lower()
        if v in MODES:
            return v, "header"
        return DEFAULT_MODE, f"header (unknown mode {v!r})"
    v = os.environ.get(MODE_ENV, "").strip().lower()
    if v in MODES:
        return v, "env"
    return DEFAULT_MODE, "default"


def switch(tier: dict | None) -> dict:
    """{on, source, mode, mode_source}: tiers.BEHAVIOURS `package_skills`
    (off by default, medium and up where on) and its rendering."""
    import tiers
    on, src = tiers.behaviour_source(tier, "package_skills")
    mode, msrc = mode_source(tier)
    return {"on": bool(on), "source": src, "mode": mode, "mode_source": msrc}


# --------------------------------------------------------------- evidence ---
_EVID: dict = {"key": None, "map": {}}
_EVID_LOCK = threading.Lock()


def _pitfall_dir() -> str:
    return os.environ.get("YAMADORI_PITFALLS_DIR") or os.path.join(
        ROOT, "bench", "skills", "pitfalls")


def pitfall_evidence() -> dict[str, dict]:
    """{skill name: {cases: n, matches: [strings]}} from the pitfall case
    files (bench/skills/pitfalls/*.jsonl): a case names the skills whose
    items answer its pitfall, each with the strings an item carries
    (`match`). Re-read when a file changes; {} when the directory is
    absent."""
    d = _pitfall_dir()
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".jsonl"))
        key = (d, tuple((n, os.stat(os.path.join(d, n)).st_mtime_ns)
                        for n in names))
    except OSError:
        return {}
    with _EVID_LOCK:
        if _EVID["key"] == key:
            return _EVID["map"]
    out: dict[str, dict] = {}
    for n in names:
        try:
            with open(os.path.join(d, n), encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("//"):
                continue
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            for sk in row.get("skills") or []:
                if not isinstance(sk, dict) or not sk.get("name"):
                    continue
                e = out.setdefault(str(sk["name"]), {"cases": 0,
                                                     "matches": []})
                e["cases"] += 1
                for m in sk.get("match") or []:
                    if str(m) not in e["matches"]:
                        e["matches"].append(str(m))
    with _EVID_LOCK:
        _EVID.update(key=key, map=out)
    return out


def _prove_default(s: dict) -> str | None:
    """The skill's PROVE verdict (skill_prove's record on its served
    version), None when there is no record or the store cannot be read."""
    try:
        import skills
        import skill_prove
        con = skills._db()
        r = con.execute("SELECT * FROM skill_versions WHERE skill=? AND "
                        "version=?", (s["id"], s.get("version"))).fetchone()
        if r is None:
            return None
        ver = dict(r)
        for k in ("prove", "validate"):
            if isinstance(ver.get(k), str):
                try:
                    ver[k] = json.loads(ver[k] or "{}")
                except ValueError:
                    ver[k] = {}
        return skill_prove.record_of(ver).get("verdict") or None
    except Exception:                                            # noqa: BLE001
        return None


# A suite replaces this ({skill dict} -> verdict | None).
PROVE_OF = _prove_default


# ------------------------------------------------------------ the skills ----
def _major_of(v) -> int | None:
    m = re.match(r"v?(\d+)", str(v or "").strip())
    return int(m.group(1)) if m else None


def _items_of(s: dict) -> list[dict]:
    import skill_inject
    return [it for it in skill_inject.skill_items(s) if not it["doubt"]]


def eligible(pkg: str, major: int | None, pool: list[dict]) -> dict:
    """{lead: skill | None, others: [skills, ranked], why: [...]} for one
    package at one major version, over the ARMED pool (see the module
    docstring for what counts as the package's and what is left out)."""
    import skill_packages
    import skill_select
    area = package_registry.package_area().get(pkg)
    names_area = bool(area) and package_registry.area_package().get(
        area) == pkg
    lead_name = skill_packages.canonical_skill(
        pkg, str(major) if major is not None else None, pool)
    ev = pitfall_evidence()
    why: list[dict] = []
    found: list[tuple] = []
    for s in pool:
        is_lead = s.get("name") == lead_name
        lf = s.get("lead_for")
        lf = [lf] if isinstance(lf, str) else list(lf or [])
        if lf and pkg not in lf:
            continue                       # another package's lead skill
        member = s.get("package") == pkg or (
            names_area and skill_select.area_of(s.get("rule") or {}) == area)
        if not (is_lead or member):
            continue
        majors = skill_select.skill_majors(s)
        if major is not None and majors and major not in majors:
            why.append({"skill": s["name"], "left_out": f"about v"
                        f"{sorted(majors)}, not v{major}"})
            continue
        verdict = PROVE_OF(s)
        if verdict in PROVE_EXCLUDED:
            why.append({"skill": s["name"], "left_out": f"prove: {verdict}"})
            continue
        if not _items_of(s):
            why.append({"skill": s["name"], "left_out": "no item survives "
                        "the assured voice"})
            continue
        rank = PROVE_RANK.get(verdict, NO_RECORD_RANK)
        cases = int((ev.get(s["name"]) or {}).get("cases") or 0)
        found.append((not is_lead, rank, -cases, s["name"], s, verdict))
    found.sort(key=lambda t: t[:4])
    lead = next((t[4] for t in found if not t[0]), None)
    others = [t[4] for t in found if t[0]]
    return {"lead": lead, "others": others, "why": why}


def ranked_items(s: dict) -> list[dict]:
    """The skill's items, the ones a pitfall case's evidence names first
    (an item that carries a case's `match` string), else in the skill's own
    order."""
    items = _items_of(s)
    ms = [m.lower() for m in
          (pitfall_evidence().get(s.get("name")) or {}).get("matches") or []]
    if not ms:
        return items
    return sorted(items, key=lambda it: (
        0 if any(m in (it["text"] + " " + it["situation"]).lower()
                 for m in ms) else 1, it["i"]))


# --------------------------------------------------------------- renderers --
def _profile(serving: str | None) -> dict:
    import skill_inject
    prof = dict(skill_inject.profile_for(serving))
    if prof.get("voice") == "first_person_prefill":
        # a tool result cannot carry a reasoning prefill
        prof["voice"] = "note"
    return prof


def trigger_text(s: dict) -> str:
    """When the craft applies, from its description (the craft index's own
    rule): "Use when" and the "Not for" sentence cut, at most
    skill_limits.INDEX_LINE_CHARS characters."""
    d = " ".join(str(s.get("description") or s.get("title")
                     or s.get("name") or "").split())
    d = re.sub(r"^Use when\s+", "", d, flags=re.I)
    d = re.split(r"\s*\bNot for\b", d, maxsplit=1)[0]
    d = re.sub(r"\s*\([^)]*\)\.?$", "", d).rstrip(" .;")
    d = d[:1].lower() + d[1:] if d[:2] != d[:2].upper() else d
    if len(d) > L.INDEX_LINE_CHARS:
        d = d[:L.INDEX_LINE_CHARS - 3].rsplit(" ", 1)[0] + "..."
    return d.replace("|", "/")


def router_table(pkg: str, major: int | None, skills: list[dict],
                 craft_tool: str = CRAFT_TOOL) -> tuple[str, list[dict]]:
    """(the router text, the skills with a row): one row per craft, best
    first, at most skill_limits.INDEX_MAX rows and skill_limits.
    SKILL_TOKENS_HARD tokens. Positive: the table's head names the tool and
    the moment."""
    head = ROUTER_HEAD.format(package=pkg, tool=craft_tool,
                              major=f" {major}" if major is not None else "")
    rows: list[str] = []
    used: list[dict] = []
    for s in skills:
        if len(rows) >= L.INDEX_MAX:
            break
        row = (f"| {trigger_text(s)} | "
               f"{json.dumps({CRAFT_ARG: s['name']})} |")
        text = (SECTION_RULE + head + "\n\n" + ROUTER_COLUMNS + "\n"
                + "\n".join(rows + [row]))
        if rows and L.tokens(text) > L.SKILL_TOKENS_HARD:
            break
        rows.append(row)
        used.append(s)
    if not rows:
        return "", []
    return (SECTION_RULE + head + "\n\n" + ROUTER_COLUMNS + "\n"
            + "\n".join(rows)), used


def inject_block(pkg: str, major: int | None, lead: dict | None,
                 others: list[dict], prof: dict, names: list[str],
                 craft_tool: bool) -> tuple[str, list[dict], list[str]]:
    """(the text, the skills whose items are in it, the item keys): the
    lead's items first, then the others' strongest items, at most the
    profile's max_items under SKILL_TOKENS_HARD tokens, at most
    skill_inject.MAX_SKILLS skills; then the pointer to the craft names."""
    import skill_inject
    order: list[dict] = []
    skills = ([lead] if lead else []) + list(others)
    for k, s in enumerate(skills[:skill_inject.MAX_SKILLS]):
        order += ranked_items(s) if k else _items_of(s)
    cap = max(int(prof.get("max_items") or skill_inject.DEFAULT_MAX_ITEMS), 1)
    use = order[:cap]
    while use:
        out = skill_inject.render(use, prof)
        text = out["text"]
        if L.tokens(text) <= L.SKILL_TOKENS_HARD or len(use) == 1:
            break
        use = use[:-1]
    if not use:
        return "", [], []
    body = skill_inject.render(use, prof)["text"]
    if craft_tool and names:
        body += "\n\n" + POINTER.format(
            package=pkg, tool=CRAFT_TOOL, names=", ".join(names),
            major=f" {major}" if major is not None else "")
    by = {s["id"]: s for s in skills}
    sk = [by[i] for i in dict.fromkeys(it["skill"] for it in use)
          if i in by]
    return body, sk, [it["key"] for it in use]


def recall_line(s: dict) -> str:
    """The craft's own first DO (or WHEN) and first DO NOT, in the recall
    line's form (skill_prompts.CRAFT_RECALL_*)."""
    import skill_prompts as P
    import skill_select
    items = _items_of(s)
    do = next((it for it in items if it["form"] != "DO NOT"), None)
    no = next((it for it in items if it["form"] == "DO NOT"), None)
    if do is None and no is None:
        return ""
    name = s.get("name") or s["id"]
    if do is None:
        return P.CRAFT_RECALL_DO.format(
            name=name, text=skill_select._clean(no["text"]))
    if do["form"] == "WHEN" and do["situation"]:
        line = P.CRAFT_RECALL_WHEN.format(
            name=name, situation=do["situation"].rstrip(" ."),
            text=skill_select._clean(do["text"]))
    else:
        line = P.CRAFT_RECALL_DO.format(
            name=name, text=skill_select._clean(do["text"]))
    if no is not None:
        line += P.CRAFT_RECALL_NOT.format(text=skill_select._clean(no["text"]))
    return line


# ------------------------------------------------------------------ state ---
def _new_pkg() -> dict:
    return {"v": 1, "tick": 0, "tick_key": None, "compactions_seen": 0,
            "given": {}, "last": {}, "fired": {}, "sections": {},
            "router_shown": {}, "recalled": {}, "turn_key": None, "outs": {}}


def pkg_state(st: dict) -> dict:
    pk = st.get("pkg")
    if not isinstance(pk, dict) or pk.get("v") != 1:
        pk = _new_pkg()
        st["pkg"] = pk
    for k, v in _new_pkg().items():
        pk.setdefault(k, v)
    return pk


def begin_request(st: dict, key: str | None,
                  compactions: int | None = None) -> dict:
    """Count this request once (`tick`: the unit "steps later" is in) and
    apply a compaction's reset (the selector's rule: what was given is
    gone). Idempotent for one key."""
    pk = pkg_state(st)
    if compactions is not None and int(compactions) > int(
            pk.get("compactions_seen") or 0):
        for k in ("given", "last", "fired", "sections", "router_shown",
                  "recalled"):
            pk[k] = {}
        pk["compactions_seen"] = int(compactions)
    if key is None or pk.get("tick_key") != key:
        pk["tick"] = int(pk.get("tick") or 0) + 1
        pk["tick_key"] = key
    return pk


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _short(x, n: int = 120) -> str:
    return " ".join(str(x).split())[:n]


# ----------------------------------------------------------------- decide ---
def major_for(tool: str | None, pkg: str, version, asked: dict | None,
              args: dict | None) -> tuple[int | None, str]:
    """(the major version this package is read at, where it came from):
    a resolved version (yama_resolve_packages: what installs); else the
    version the conversation asked for or pinned (`asked`:
    skill_packages.detect's); else a README's asked version; else the
    registry's latest. None: no version known (skills that name a major are
    then left out unless the package has only that one)."""
    if tool == "yama_resolve_packages" and _major_of(version) is not None:
        return _major_of(version), "resolved"
    a = _major_of((asked or {}).get(pkg))
    if a is not None:
        return a, "asked"
    ra = _major_of((args or {}).get("version"))
    if ra is not None and tool == "yama_read_package_readme":
        return ra, "readme version"
    v = _major_of(version)
    if v is not None:
        return v, "registry"
    return None, "unknown"


def decide(st: dict, packages: list[dict], ctx: dict, pool: list[dict]
           ) -> tuple[str, list[dict], dict]:
    """(the text to append, one record per package, the new state).

    `packages`: [{name, version?}] best first. `ctx`: mode, serving (the
    model that reads the text), craft_tool (is yama_recall_craft on main),
    tool (a package tool's name) or trigger + evidence, asked {package:
    version}, args, key ("<turn key>|<hop>": a request that runs again
    gets the text it got), compactions, fired_key_of (trigger kinds).
    Pure over `st` (a copy is returned)."""
    st = json.loads(json.dumps(st)) if st else {}
    pk = begin_request(st, None, ctx.get("compactions"))
    key = ctx.get("key")
    if key:
        tk, _, _rest = str(key).partition("|")
        if pk.get("turn_key") != tk:
            pk["turn_key"], pk["outs"] = tk, {}
        got = pk["outs"].get(key)
        if got is not None:
            return got["text"], [dict(r, replayed=True)
                                 for r in got["recs"]], st
    mode = ctx.get("mode") or DEFAULT_MODE
    craft_tool = bool(ctx.get("craft_tool"))
    eff = mode if craft_tool or mode == "inject" else "inject"
    prof = _profile(ctx.get("serving"))
    trigger = ctx.get("trigger")
    texts: list[str] = []
    recs: list[dict] = []
    budget = L.MAX_SKILLS_PER_TURN
    seen_pkg: set[str] = set()
    for p in packages:
        pkg = str(p.get("name") or "")
        if not pkg or pkg in seen_pkg:
            continue
        seen_pkg.add(pkg)
        major, mfrom = major_for(ctx.get("tool"), pkg, p.get("version"),
                                 ctx.get("asked"), ctx.get("args"))
        rec: dict = {"package": pkg, "version": p.get("version"),
                     "major": major, "major_from": mfrom, "mode": eff,
                     "skills": [], "chars": 0}
        if ctx.get("tool"):
            rec["tool"] = ctx["tool"]
        if trigger:
            rec["trigger"] = trigger
            rec["evidence"] = _short(ctx.get("evidence") or "")
        if eff != mode:
            rec["mode_requested"] = mode
            rec["mode_note"] = (f"{CRAFT_TOOL} is not on main: the inject "
                                "form")
        if package_registry.package_area().get(pkg) is None \
                and pkg not in package_registry.canonical():
            # a package the registry has no skill area for (react, a type
            # package, a search hit): nothing to say, nothing recorded
            continue
        el = eligible(pkg, major, pool)
        if el["why"]:
            rec["left_out"] = el["why"][:6]
        every = ([el["lead"]] if el["lead"] else []) + el["others"]
        if not every:
            rec["why"] = "no armed, proven skill for this package"
            recs.append(rec)
            continue
        kp = f"{pkg}@{major}"
        names = [s["name"] for s in every][:L.INDEX_MAX]
        first = not any(f"{s['id']}@{major}" in pk["given"] for s in every)
        fk = (f"{trigger}:{kp}" if trigger else None)
        if fk and fk in pk["fired"]:
            rec["why"] = "this trigger already fired for the package"
            recs.append(rec)
            continue
        if budget <= 0:
            rec["why"] = (f"past {L.MAX_SKILLS_PER_TURN} skills in one "
                          "section")
            recs.append(rec)
            continue
        text, used, form = "", [], "body"
        if first:
            if eff == "router":
                text, used = router_table(pkg, major, every)
                form = "router"
            elif eff == "both":
                body, sk, _k = inject_block(
                    pkg, major, el["lead"], [], prof, [], False)
                rest = [s for s in every if s is not el["lead"]]
                tbl, rows = router_table(pkg, major, rest)
                text = body + tbl
                used = sk + rows
            else:
                text, used, _k = inject_block(
                    pkg, major, el["lead"], el["others"], prof, names,
                    craft_tool)
        else:
            form = "recall"
            if trigger and trigger != "error":
                # the selector's chart: a given skill comes back on an
                # ERROR (or an ask or a phase change), never on an install,
                # an import or a first write of a package already given
                rec["why"] = ("given before; only an error brings a "
                              "package's skills back (the chart's recall "
                              "events)")
                recs.append(rec)
                continue
            if eff == "router":
                rec["why"] = "the router table was given for this package"
                recs.append(rec)
                continue
            lead = el["lead"] or every[0]
            line = recall_line(lead)
            if line:
                text, used = SECTION_RULE + line, [lead]
        if not text:
            rec["why"] = rec.get("why") or "nothing to give"
            recs.append(rec)
            continue
        last = pk["last"].get(kp) or {}
        if form == "recall" and last.get("sha") == _sha(text) and not (
                trigger == "error" and last.get("trigger") != "error"):
            rec["why"] = "the same line was the last one given"
            recs.append(rec)
            continue
        budget -= len(used)
        for s in used:
            sform = ("router" if first and (eff == "router" or (
                eff == "both" and s is not el["lead"])) else form)
            rec["skills"].append({"id": s["id"], "name": s["name"],
                                  "form": sform})
            pk["given"][f"{s['id']}@{major}"] = {"tick": pk["tick"],
                                                 "form": sform}
            st.setdefault("given", {}).setdefault(s["id"], {
                "req": int(st.get("req") or 0),
                "chars": int(st.get("chars") or 0), "via": "package"})
        if first and eff in ("router", "both"):
            for s in used:
                if s is el["lead"] and eff == "both":
                    continue
                pk["router_shown"][s["name"]] = {
                    "tick": pk["tick"], "package": pkg, "mode": eff,
                    "major": major}
        pk["last"][kp] = {"sha": _sha(text), "trigger": trigger or ctx.get(
            "tool")}
        if fk:
            pk["fired"][fk] = pk["tick"]
        pk["sections"][kp] = {"tick": pk["tick"], "mode": eff,
                              "package": pkg, "major": major}
        rec["chars"] = len(text)
        rec["listed"] = names
        rec["why"] = ("first appearance in this conversation" if first
                      else "the package came up again: a recall line")
        texts.append(text)
        recs.append(rec)
    out = "".join(texts)
    if key:
        pk["outs"][key] = {"text": out, "recs": recs}
    return out, recs, st


def note_craft_call(st: dict, craft: str | None, found: bool) -> tuple[dict | None, dict]:
    """The record of a yama_recall_craft call that follows a package section
    in this conversation (x_yamadori.skills.package, event
    recall_after_section): the craft, whether a router listed it, the
    section's mode and how many requests later it came. None when no section
    has been given. Returns (record, new state)."""
    st = json.loads(json.dumps(st)) if st else {}
    pk = pkg_state(st)
    if not pk["sections"]:
        return None, st
    shown = pk["router_shown"].get(craft or "")
    last = max(pk["sections"].values(), key=lambda s: s["tick"])
    ref = shown or {"tick": last["tick"], "mode": last["mode"],
                    "package": None}
    rec = {"event": "recall_after_section", "craft": craft, "found": found,
           "listed_by_router": shown is not None, "mode": ref["mode"],
           "package": ref.get("package"),
           "steps_later": int(pk["tick"]) - int(ref["tick"])}
    if found and craft:
        pk["recalled"][craft] = pk["tick"]
    return rec, st


# ----------------------------------------------------------------- pool -----
def user_named(messages: list[dict]) -> list[tuple[str, int | None]]:
    """[(package, major)] the USER named in their own words, in the order
    they first appear across the conversation's user turns: the registry's
    terms (skill_classify.asked_terms: "r3f v10", "Koota", "pmndrs math",
    "@react-three/drei"), never a pasted manifest or a fenced block
    (user_prose), never a harness notice, never a name behind "without" or a
    context mention ("my React page"). Our own injected text is stripped
    first (skill_select._strip_ours). The version written after the name is
    the major. Used to boost a craft question's shortlist before any package
    lookup (craft_query.shortlist)."""
    import skill_classify
    import skill_packages
    import skill_select
    term_pkg = dict(skill_packages.TERM_PACKAGE)
    out: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        text = skill_select._strip_ours(skill_select._text(m))
        if not text.strip():
            continue
        for term, e in skill_classify.asked_terms(text).items():
            if e.get("mode") != "choice":
                continue
            pkg = term_pkg.get(term)
            form = str(e.get("form") or "")
            if term == "threejs" and skill_packages._TSL_FORM.match(form):
                pkg = skill_packages.TSL
            if term == "r3f" and form.lower().startswith("@react-three/"):
                pkg = skill_packages.package_of_specifier(form) or pkg
            if pkg and pkg not in seen:
                seen.add(pkg)
                out.append((pkg, _major_of(e.get("version"))))
    return out


def package_pool(pool: list[dict]) -> bool:
    """Does the armed library hold any skill of a package the registry
    knows? (the craft tool is offered with the MCP tools only then)"""
    import skill_select
    areas = set(package_registry.package_area().values())
    names = {r.get("skill") for r in package_registry.canonical().values()}
    return any(skill_select.area_of(s.get("rule") or {}) in areas
               or s.get("name") in names or s.get("lead_for")
               or s.get("package") for s in pool or [])


# --------------------------------------------------------------- triggers ---
# A result line that ECHOES the command (a shell prompt first): a README's
# install line that a `cat` printed is not the model's action.
_INSTALL_ECHO = re.compile(
    r"^\s*[$>]\s*(?:npm\s+(?:i|install|add)|pnpm\s+(?:add|i|install)"
    r"|yarn\s+add|bun\s+add|pip3?\s+install|uv\s+(?:add|pip\s+install))\s+"
    r"([^\n;&|]+)", re.M)
_INSTALL_CMD = re.compile(
    r"(?:npm\s+(?:i|install|add)|pnpm\s+(?:add|i|install)|yarn\s+add|"
    r"bun\s+add|pip3?\s+install|uv\s+(?:add|pip\s+install))\s+([^\n;&|]+)")


def _registry_package(name: str) -> str | None:
    import skill_packages
    n = name.strip().strip("'\"`")
    if not n or n.startswith("-"):
        return None
    return skill_packages.package_of_specifier(n) or (
        n if package_registry.entry(n) is not None else None)


def _install_packages(text: str, rx) -> list[tuple[str, str | None, str]]:
    out = []
    for line in rx.findall(text or ""):
        for tok in line.split():
            if tok.startswith("-"):
                continue
            at = tok.rfind("@")
            name, ver = (tok[:at], tok[at + 1:]) if at > 0 else (tok, None)
            pkg = _registry_package(name)
            if pkg:
                out.append((pkg, ver, tok))
    return out


def _unescape(text: str) -> str:
    return text.replace("\\n", "\n").replace('\\"', '"')


def step_triggers(msgs: list[dict], pool: list[dict] | None = None
                  ) -> list[dict]:
    """The package events in the NEWEST evidence of an agent step, from the
    CLIENT's own messages (never ours): [{trigger: install | import | error
    | new_area, package, version, evidence}] in order, one per package and
    kind, an install before an import before a new area.

      install   an npm / pnpm / yarn / bun / pip install line the model ran
                (its command, or a line of the result that echoes one), or a
                dependency the model wrote into package.json
      import    an import or require of a registry package in code the model
                wrote (a write call; never a file it only read)
      new_area  a unique exported symbol of a package in code the model wrote
      error     an error whose own lines name the package: a quoted module of
                the registry, or a topic of its skills
                (skill_select.error_names)

    NEVER on a manifest's content read: package.json the model read is a
    tool result and a tool result's pins are not looked at -- only what the
    model's own calls say and its errors. Reuses skill_select's evidence
    machinery (fresh_messages, evidence_view, error_text, error_names) and
    skill_packages' detector; nothing new matches."""
    import skill_packages
    import skill_select
    fresh = skill_select.fresh_messages(msgs)
    if not fresh or not any(m.get("role") == "tool" for m in fresh):
        return []
    found: dict[tuple, dict] = {}

    def add(kind, pkg, ver, ev):
        found.setdefault((kind, pkg), {"trigger": kind, "package": pkg,
                                       "version": ver, "evidence": ev})

    asst = [m for m in fresh if m.get("role") == "assistant"]
    # --- what the model's own calls say: installs, imports, symbols --------
    import progress
    for m in asst:
        for c in m.get("tool_calls") or []:
            cmd = progress.command_of(c)
            if not cmd:
                continue
            for pkg, ver, tok in _install_packages(_unescape(cmd),
                                                   _INSTALL_CMD):
                add("install", pkg, ver, f"install line: {tok}")
    try:
        in_play = skill_packages.detect(asst) if asst else {}
    except Exception:                                            # noqa: BLE001
        in_play = {}
    for pkg, e in in_play.items():
        for w in e.get("why") or []:
            if w["how"] == "pin" and w.get("where") in ("config", "code"):
                add("install", pkg, e.get("version"),
                    f"dependency written: {w['what']}")
            elif w["how"] == "import":
                add("import", pkg, e.get("version"), f"import {w['what']}")
            elif w["how"] == "symbol" and w.get("strength") != "weak":
                add("new_area", pkg, e.get("version"), f"symbol {w['what']}")
    # --- result lines that echo an install command (a shell's echo) --------
    for m in fresh:
        if m.get("role") == "tool":
            for pkg, ver, tok in _install_packages(
                    skill_select._text(m), _INSTALL_ECHO):
                add("install", pkg, ver, f"install line: {tok}")
    # --- an error whose own lines name the package ------------------------
    view, _info = skill_select.evidence_view(fresh)
    etext = skill_select.error_text(view)
    if etext:
        for q in skill_select._QUOTED_MODULE.findall(etext):
            pkg = _registry_package(q)
            if pkg:
                add("error", pkg, None, f"error names module {q}")
        pool = pool if pool is not None else _armed()
        for pkg in sorted(package_registry.load()):
            if ("error", pkg) in found:
                continue
            el = eligible(pkg, None, pool)
            for s in ([el["lead"]] if el["lead"] else []) + el["others"]:
                names = skill_select.error_names(s, etext)
                if names:
                    add("error", pkg, None, f"error: {names[0]}")
                    break
    order = {"install": 0, "import": 1, "new_area": 2, "error": 3}
    # one event per package: the strongest kind, errors always
    best: dict[str, dict] = {}
    out: list[dict] = []
    for (kind, pkg), row in sorted(found.items(),
                                   key=lambda kv: (order[kv[0][0]], kv[0][1])):
        if kind == "error":
            out.append(row)
        elif pkg not in best:
            best[pkg] = row
            out.append(row)
    return out


def _armed() -> list[dict]:
    import skills
    return skills.armed(include_package_only=True)
