#!/usr/bin/env python
"""The harness kit: what we recommend putting INTO a harness (Hermes, Pi,
OpenCode, Codex, Claude Code, or any harness) -- prompts, skill folders,
downloads, MCP servers, config snippets and notes -- each with its evidence,
and a per-harness export for the operator's own machine.

Operator, 2026-09-29, verbatim: "Harness tools should be something we can
add information to on the dashboard, prompts to put in the harness or what
to download and install to make the harness better, with a harness dropdown
to select from our research as well as general tool recommendations, etc."
And: "We are going to run the next waves on my remote machine with a
monitor and gpu attached, the way I would really do the work. We can still
use the local harnesses for testing things that don't need access to a
browser with a GPU to test the product."

THE STORE. One sqlite file, YAMADORI_HARNESS_KIT_DB (default
index/harness_kit.sqlite3; mcp/offline_stores.py moves it for offline
suites). Written ONLY through the keyed dashboard API (mcp/dash_harness.py),
which seeds it on first use from mcp/harness_kit_seed.py. Two tables:

  entries   the current version of each entry (soft-deleted rows stay,
            `deleted` 1, with who, when and why)
  history   every version ever written, whole, with its action (seed,
            create, edit, delete, restore, reseed) and author

An edit names the version it was made against (`base_version`); a stale one
is refused (409), so two editors never overwrite each other silently.

NO SECRET IS STORED. Every string of an entry is scanned (SECRET_PATTERNS:
this stack's own `ym-` keys, OpenAI/Anthropic/GitHub/Slack/AWS token shapes,
a literal Bearer credential, a PEM private key) and a match refuses the
write. The export is scanned again after templating (defence in depth).

THE EXPORT (`export`): per harness, a zip for the operator to unpack and
follow on the machine the next waves run on. It carries the harness's config
pointing at the proxy's public base, the recommended skill folders, the
prompt snippets, the install commands of the RECOMMENDED downloads each with
its exact version and hash, and a README with the steps per OS where the
harness's docs say how. It never contains a key: every place a key goes
reads an environment variable or holds KEY_PLACEHOLDER, and the README says
where to put the real one. It never downloads anything: it is text.

Entries apply to a machine (`where`): `remote` (the operator's machine with
a monitor and a GPU), `box` (the local harness box / Octopus sandbox, whose
browser renders WebGL in software), or `both`. The export is for the remote
machine, so it takes `remote` and `both`.
"""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import threading
import time
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB_PATH = os.environ.get("YAMADORI_HARNESS_KIT_DB",
                         os.path.join(ROOT, "index", "harness_kit.sqlite3"))

# (id, label). The operator's list, 2026-09-29, plus "any harness" for the
# general recommendations.
HARNESSES = (("hermes", "Hermes"), ("pi", "Pi"), ("opencode", "OpenCode"),
             ("codex", "Codex"), ("claude-code", "Claude Code"),
             ("any", "any harness"))
HARNESS_IDS = tuple(h for h, _ in HARNESSES)
# The operator's kinds, 2026-09-29.
KINDS = (("prompt", "prompt / system prompt / AGENTS.md text"),
         ("skill", "skill folder to install"),
         ("download", "tool or package to download and install"),
         ("mcp_server", "MCP server"),
         ("config", "config snippet"),
         ("note", "note"))
KIND_IDS = tuple(k for k, _ in KINDS)
STATUSES = ("recommended", "trying", "rejected")
# What the entry's claim rests on -- AGENTS.md "Claims carry their evidence":
# (A) an operator decision, (C) measured (a script or log, with its n), or
# verified from source / in Docker with no model; anything else is
# unmeasured and says so.
PROOFS = (("operator", "an operator decision, quoted with its date"),
          ("measured", "measured: a run or script, with its n"),
          ("verified", "verified: read from source, or checked in Docker with no model"),
          ("unmeasured", "unmeasured: no run behind it"))
PROOF_IDS = tuple(p for p, _ in PROOFS)
WHERES = (("remote", "the operator's machine (monitor, GPU browser)"),
          ("box", "the local harness box / Octopus sandbox (software WebGL)"),
          ("both", "both"))
WHERE_IDS = tuple(w for w, _ in WHERES)

# A skill folder an entry may point at: only these roots of this repo, so an
# entry can never make the export read an arbitrary file (a key file above
# all). Each is where a harness skill of ours lives today.
SKILL_ROOTS = ("bench/sandbox/harness_skills", "bench/octopus/hermes_skills")

# Tokens a body may carry; the export replaces them (templated_text).
KEY_PLACEHOLDER = "<PASTE-YOUR-YAMADORI-KEY-HERE>"
TOKENS = ("{{PUBLIC_BASE}}", "{{API_BASE}}", "{{TOOLS_MCP_URL}}",
          "{{CONTEXT_WINDOW}}", "{{MAX_OUTPUT}}", "{{KEY_PLACEHOLDER}}")

SECRET_PATTERNS = (
    ("yamadori key", re.compile(r"\bym-[A-Za-z0-9_\-]{20,}")),     # accounts.create
    ("OpenAI/Anthropic-style key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}")),
    ("Slack token", re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # a literal credential after Bearer; a reference ($VAR, ${VAR}, {env:X},
    # <PLACEHOLDER>) is not one
    ("literal Bearer credential", re.compile(r"(?i)\bbearer\s+(?![$<{])[A-Za-z0-9._~+/=\-]{20,}")),
)

EXACT_VERSION = re.compile(r"^v?\d+(?:\.\d+){0,3}(?:[-+][0-9A-Za-z.\-]+)?$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
SRI512 = re.compile(r"^sha512-[A-Za-z0-9+/]{86}==$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
OSES = ("windows", "unix", "any")

_LOCK = threading.Lock()


class Refused(Exception):
    """A write the store will not make: (code, message, reasons)."""

    def __init__(self, code: int, message: str, reasons: list | None = None):
        super().__init__(message)
        self.code, self.message, self.reasons = code, message, reasons or []


# ------------------------------------------------------------------ database

def _connect(db: str | None = None) -> sqlite3.Connection:
    path = db or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS entries(
        id TEXT PRIMARY KEY, seed_id TEXT UNIQUE, seed_rev INTEGER,
        version INTEGER NOT NULL, data TEXT NOT NULL,
        created REAL NOT NULL, created_by TEXT NOT NULL,
        updated REAL NOT NULL, updated_by TEXT NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0, deleted_at REAL, deleted_by TEXT,
        deleted_why TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS history(
        entry_id TEXT NOT NULL, version INTEGER NOT NULL, at REAL NOT NULL,
        author TEXT NOT NULL, action TEXT NOT NULL, data TEXT NOT NULL,
        note TEXT, PRIMARY KEY(entry_id, version, action))""")
    return con


# ------------------------------------------------------------------ validation

FIELDS = ("harness", "kind", "title", "body", "evidence", "status",
          "status_why", "proof", "where", "needs_operator", "target",
          "file_name", "skill_path", "download")


def _strings(x, path="") -> list[tuple[str, str]]:
    if isinstance(x, str):
        return [(path, x)]
    if isinstance(x, dict):
        return [s for k, v in x.items() for s in _strings(v, f"{path}.{k}" if path else str(k))]
    if isinstance(x, list):
        return [s for i, v in enumerate(x) for s in _strings(v, f"{path}[{i}]")]
    return []


def secrets_in(obj) -> list[dict]:
    """Every key-shaped string in `obj`: [{field, what}] -- never the value."""
    out = []
    for field, s in _strings(obj):
        for what, pat in SECRET_PATTERNS:
            if pat.search(s):
                out.append({"field": field or "(text)", "what": what})
    return out


def _skill_dir(rel: str) -> str | None:
    """The absolute folder of an allowed skill path, else None."""
    rel = (rel or "").replace("\\", "/").strip().strip("/")
    if not rel or ".." in rel.split("/"):
        return None
    if not any(rel == r or rel.startswith(r + "/") for r in SKILL_ROOTS):
        return None
    if rel in SKILL_ROOTS:
        return None
    p = os.path.normpath(os.path.join(ROOT, rel))
    return p if os.path.isfile(os.path.join(p, "SKILL.md")) else None


def normalise(fields: dict) -> dict:
    """The entry as stored: known fields only, defaults filled."""
    d = {k: fields.get(k) for k in FIELDS}
    for k in ("title", "body", "status_why", "target", "file_name", "skill_path"):
        d[k] = (d[k] or "") if isinstance(d[k], str) or d[k] is None else d[k]
        if isinstance(d[k], str):
            d[k] = d[k].strip() if k != "body" else d[k].rstrip()
    d["status"] = d["status"] or "trying"
    d["proof"] = d["proof"] or "unmeasured"
    d["where"] = d["where"] or "both"
    d["needs_operator"] = bool(d["needs_operator"])
    ev = d["evidence"] if isinstance(d["evidence"], list) else []
    d["evidence"] = [{"ref": str(e.get("ref") or "").strip(), "showed": str(e.get("showed") or "").strip()}
                     for e in ev if isinstance(e, dict)]
    dl = d["download"] if isinstance(d["download"], dict) else None
    if dl is not None:
        inst = dl.get("install") if isinstance(dl.get("install"), dict) else {}
        dl = {"source_url": str(dl.get("source_url") or "").strip(),
              "version": str(dl.get("version") or "").strip(),
              "hash": str(dl.get("hash") or "").strip(),
              "commit": str(dl.get("commit") or "").strip(),
              "licence": str(dl.get("licence") or "").strip(),
              "pin_missing_why": str(dl.get("pin_missing_why") or "").strip(),
              "verify": str(dl.get("verify") or "").strip(),
              "install": {os_: str(inst.get(os_) or "").strip() for os_ in OSES if str(inst.get(os_) or "").strip()}}
        if not any(v for k, v in dl.items() if k != "install") and not dl["install"]:
            dl = None
    d["download"] = dl
    return d


def validate(d: dict) -> list[dict]:
    """[{field, why}] for every rule the entry breaks; [] when it is good."""
    bad: list[dict] = []

    def no(field, why):
        bad.append({"field": field, "why": why})

    if d["harness"] not in HARNESS_IDS:
        no("harness", f"one of {', '.join(HARNESS_IDS)}")
    if d["kind"] not in KIND_IDS:
        no("kind", f"one of {', '.join(KIND_IDS)}")
    if d["status"] not in STATUSES:
        no("status", f"one of {', '.join(STATUSES)}")
    if d["proof"] not in PROOF_IDS:
        no("proof", f"one of {', '.join(PROOF_IDS)}")
    if d["where"] not in WHERE_IDS:
        no("where", f"one of {', '.join(WHERE_IDS)}")
    if not isinstance(d["title"], str) or not d["title"]:
        no("title", "required")
    for k in ("body", "status_why", "target", "file_name", "skill_path"):
        if not isinstance(d[k], str):
            no(k, "must be text")
    if d["status"] in ("trying", "rejected") and not d["status_why"]:
        no("status_why", f"a {d['status']} entry says why")
    # a verdict needs its evidence (AGENTS.md "Claims carry their evidence")
    if d["status"] in ("recommended", "rejected") and not d["evidence"]:
        no("evidence", f"a {d['status']} entry names its evidence: a doc path, run or link, and what it showed")
    for i, e in enumerate(d["evidence"]):
        if not e["ref"]:
            no(f"evidence[{i}].ref", "a doc path, run name or link")
        if not e["showed"]:
            no(f"evidence[{i}].showed", "what it showed")
    fn = d["file_name"]
    if isinstance(fn, str) and fn and not (fn == ".env" or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._\-]*", fn)):
        no("file_name", "a plain file name (letters, digits, . _ -)")
    if d["kind"] == "skill":
        sp = d["skill_path"]
        if sp:
            if _skill_dir(sp) is None:
                no("skill_path", f"a folder with a SKILL.md under {' or '.join(SKILL_ROOTS)}")
        elif not (isinstance(d["body"], str) and d["body"].lstrip().startswith("---")):
            no("body", "a skill is a repo skill folder (skill_path) or a SKILL.md text (starting with ---)")
    elif d["skill_path"]:
        no("skill_path", "only a skill entry has a skill folder")
    dl = d["download"]
    if d["kind"] == "download" and dl is None:
        no("download", "a download names its source, exact version, hash or commit, licence and install command")
    if dl is not None:
        if dl["source_url"] and not re.match(r"^https?://", dl["source_url"]):
            no("download.source_url", "an http(s) URL")
        if dl["version"] and not EXACT_VERSION.match(dl["version"]):
            no("download.version", "an exact version (no range, no 'latest')")
        if dl["hash"] and not (SHA256.match(dl["hash"]) or SRI512.match(dl["hash"])):
            no("download.hash", "sha256:<64 hex> or an npm integrity sha512-<base64>")
        if dl["commit"] and not COMMIT.match(dl["commit"]):
            no("download.commit", "a full 40-hex commit")
        for os_, cmd in dl["install"].items():
            if dl["version"] and dl["version"].lstrip("v") not in cmd:
                no(f"download.install.{os_}", f"the command names the exact version {dl['version']}")
        if d["status"] == "recommended":
            for k in ("source_url", "version", "licence"):
                if not dl[k]:
                    no(f"download.{k}", "required for a recommended download")
            if not dl["hash"] and not dl["commit"]:
                no("download.hash", "a recommended download is pinned by sha256/sha512 or a commit")
            if not dl["install"]:
                no("download.install", "a recommended download has an install command")
        elif not dl["hash"] and not dl["commit"] and not dl["pin_missing_why"]:
            no("download.pin_missing_why", "say why there is no hash or commit yet")
    for s in secrets_in(d):
        no(s["field"], f"looks like a secret ({s['what']}): keys never go in the kit; use an environment variable or {KEY_PLACEHOLDER}")
    return bad


# ------------------------------------------------------------------ reads

def _row(r: sqlite3.Row) -> dict:
    d = json.loads(r["data"])
    d.update({"id": r["id"], "seed_id": r["seed_id"], "version": r["version"],
              "created": r["created"], "created_by": r["created_by"],
              "updated": r["updated"], "updated_by": r["updated_by"],
              "deleted": bool(r["deleted"]), "deleted_at": r["deleted_at"],
              "deleted_by": r["deleted_by"], "deleted_why": r["deleted_why"]})
    return d


def entries(db: str | None = None, *, include_deleted: bool = False) -> list[dict]:
    con = _connect(db)
    try:
        q = "SELECT * FROM entries" + ("" if include_deleted else " WHERE deleted = 0")
        rows = [_row(r) for r in con.execute(q + " ORDER BY created, id")]
    finally:
        con.close()
    return rows


def get(entry_id: str, db: str | None = None) -> dict | None:
    con = _connect(db)
    try:
        r = con.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        if r is None:
            return None
        out = _row(r)
        out["history"] = [{"version": h["version"], "at": h["at"], "author": h["author"],
                           "action": h["action"], "note": h["note"], "data": json.loads(h["data"])}
                          for h in con.execute("SELECT * FROM history WHERE entry_id = ? "
                                               "ORDER BY version, at", (entry_id,))]
    finally:
        con.close()
    return out


def counts(rows: list[dict]) -> dict:
    out: dict = {"total": len(rows), "by_harness": {}, "by_kind": {}, "by_status": {},
                 "needs_operator": sum(1 for r in rows if r.get("needs_operator"))}
    for r in rows:
        for k, f in (("by_harness", "harness"), ("by_kind", "kind"), ("by_status", "status")):
            out[k][r[f]] = out[k].get(r[f], 0) + 1
    return out


# ------------------------------------------------------------------ writes

def _history(con, entry_id, version, author, action, data, note=None, at=None):
    con.execute("INSERT OR REPLACE INTO history VALUES (?,?,?,?,?,?,?)",
                (entry_id, version, at or time.time(), author, action,
                 json.dumps(data, sort_keys=True), note))


def create(fields: dict, author: str, db: str | None = None, *,
           seed_id: str | None = None, seed_rev: int | None = None) -> dict:
    d = normalise(fields)
    bad = validate(d)
    if bad:
        raise Refused(400, "entry refused: " + "; ".join(f"{b['field']}: {b['why']}" for b in bad), bad)
    now = time.time()
    eid = "hk-" + uuid.uuid4().hex[:12]
    with _LOCK:
        con = _connect(db)
        try:
            with con:
                con.execute("INSERT INTO entries(id, seed_id, seed_rev, version, data, created, created_by,"
                            " updated, updated_by) VALUES (?,?,?,?,?,?,?,?,?)",
                            (eid, seed_id, seed_rev, 1, json.dumps(d, sort_keys=True), now, author, now, author))
                _history(con, eid, 1, author, "seed" if seed_id else "create", d, at=now)
        finally:
            con.close()
    return get(eid, db)


def edit(entry_id: str, fields: dict, author: str, base_version: int | None,
         db: str | None = None, *, action: str = "edit") -> dict:
    with _LOCK:
        con = _connect(db)
        try:
            r = con.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if r is None:
                raise Refused(404, f"no entry {entry_id}")
            if r["deleted"]:
                raise Refused(409, f"entry {entry_id} is deleted; restore it first")
            if base_version is None or int(base_version) != r["version"]:
                raise Refused(409, f"entry {entry_id} is at version {r['version']}, the edit was made "
                                   f"against {base_version}: reload it and edit again")
            cur = json.loads(r["data"])
            merged = dict(cur)
            merged.update({k: v for k, v in fields.items() if k in FIELDS})
            d = normalise(merged)
            bad = validate(d)
            if bad:
                raise Refused(400, "edit refused: " + "; ".join(f"{b['field']}: {b['why']}" for b in bad), bad)
            if d == normalise(cur):
                return _row(r) | {"unchanged": True}
            v = r["version"] + 1
            now = time.time()
            with con:
                con.execute("UPDATE entries SET version=?, data=?, updated=?, updated_by=? WHERE id=?",
                            (v, json.dumps(d, sort_keys=True), now, author, entry_id))
                _history(con, entry_id, v, author, action, d, at=now)
        finally:
            con.close()
    return get(entry_id, db)


def set_deleted(entry_id: str, deleted: bool, author: str, why: str = "",
                db: str | None = None) -> dict:
    """Soft delete (or restore). The row and its history stay."""
    with _LOCK:
        con = _connect(db)
        try:
            r = con.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if r is None:
                raise Refused(404, f"no entry {entry_id}")
            if bool(r["deleted"]) == deleted:
                raise Refused(409, f"entry {entry_id} is already {'deleted' if deleted else 'live'}")
            if deleted and not (why or "").strip():
                raise Refused(400, "a delete says why", [{"field": "why", "why": "required"}])
            if secrets_in(why or ""):
                raise Refused(400, "the reason looks like a secret", [{"field": "why", "why": "looks like a secret"}])
            now = time.time()
            with con:
                if deleted:
                    con.execute("UPDATE entries SET deleted=1, deleted_at=?, deleted_by=?, deleted_why=? WHERE id=?",
                                (now, author, why.strip(), entry_id))
                else:
                    con.execute("UPDATE entries SET deleted=0, deleted_at=NULL, deleted_by=NULL, deleted_why=NULL"
                                " WHERE id=?", (entry_id,))
                _history(con, entry_id, r["version"], author, "delete" if deleted else "restore",
                         json.loads(r["data"]), note=(why or "").strip() or None, at=now)
        finally:
            con.close()
    return get(entry_id, db)


def seed(items: list[dict], db: str | None = None, author: str = "seed") -> dict:
    """Insert every seed item whose seed_id the store has never held (a
    deleted one counts as held: a delete is the operator's). An item whose
    `seed_rev` is newer than the stored one updates the entry ONLY while no
    one but the seed has written it; an entry a person edited is left alone
    and reported. Returns {added, updated, kept_edited}."""
    out = {"added": [], "updated": [], "kept_edited": []}
    con = _connect(db)
    try:
        held = {r["seed_id"]: r for r in con.execute("SELECT * FROM entries WHERE seed_id IS NOT NULL")}
        authors = {}
        for h in con.execute("SELECT entry_id, author FROM history"):
            authors.setdefault(h["entry_id"], set()).add(h["author"])
    finally:
        con.close()
    for it in items:
        sid, rev = it["seed_id"], int(it.get("seed_rev", 1))
        fields = {k: v for k, v in it.items() if k in FIELDS}
        if sid not in held:
            create(fields, author, db, seed_id=sid, seed_rev=rev)
            out["added"].append(sid)
            continue
        r = held[sid]
        if rev <= (r["seed_rev"] or 0):
            continue
        if authors.get(r["id"], set()) - {author} or r["deleted"]:
            out["kept_edited"].append(sid)
            continue
        edit(r["id"], fields, author, r["version"], db, action="reseed")
        con = _connect(db)
        try:
            with con:
                con.execute("UPDATE entries SET seed_rev=? WHERE id=?", (rev, r["id"]))
        finally:
            con.close()
        out["updated"].append(sid)
    return out


# ------------------------------------------------------------------ export

def public_base(env: dict | None = None, caddyfile: str | None = None) -> dict:
    """Where the remote machine reaches the proxy: {url, source}.
    YAMADORI_PUBLIC_BASE when set (mcp/images.py public_base reads the same
    variable; docs/IMAGEGEN.md sets it to the Caddy site), else the first
    site address of caddy/Caddyfile (Caddy terminates HTTPS for :1234 over
    the private ZeroTier network), else a placeholder the README names."""
    env = os.environ if env is None else env
    v = (env.get("YAMADORI_PUBLIC_BASE") or "").strip().rstrip("/")
    if v:
        return {"url": v, "source": "YAMADORI_PUBLIC_BASE"}
    path = caddyfile or os.path.join(ROOT, "caddy", "Caddyfile")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        text = ""
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9.\-]+\.[A-Za-z]{2,})(?::\d+)?\s*\{\s*$", line.strip())
        if m:
            return {"url": f"https://{m.group(1)}", "source": "caddy/Caddyfile site address"}
    return {"url": "https://<YOUR-YAMADORI-HOST>",
            "source": "none: set YAMADORI_PUBLIC_BASE on the server, or edit the URL in the kit"}


def templated_text(text: str, ctx: dict) -> str:
    for tok in TOKENS:
        text = text.replace(tok, str(ctx.get(tok.strip("{}"), tok)))
    return text


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "entry"


def _skill_files(entry: dict) -> dict[str, bytes]:
    """{relative path: bytes} of an entry's skill folder (repo folder, or its
    SKILL.md text). Only files under the folder; no symlinks followed out."""
    if entry.get("skill_path"):
        root = _skill_dir(entry["skill_path"])
        out = {}
        if root:
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d != "__pycache__" and not os.path.islink(os.path.join(dp, d))]
                for fn in sorted(fns):
                    p = os.path.join(dp, fn)
                    if os.path.islink(p):
                        continue
                    out[os.path.relpath(p, root).replace("\\", "/")] = open(p, "rb").read()
        return out
    return {"SKILL.md": (entry.get("body") or "").encode()}


def _skill_name(entry: dict) -> str:
    if entry.get("skill_path"):
        return os.path.basename(entry["skill_path"].replace("\\", "/").rstrip("/"))
    m = re.search(r"(?m)^name:\s*\"?([A-Za-z0-9._\-]+)", entry.get("body") or "")
    return m.group(1) if m else _slug(entry["title"])


def for_harness(rows: list[dict], harness: str) -> list[dict]:
    """The entries a harness's kit draws on: its own and any harness's."""
    return [r for r in rows if not r.get("deleted") and r["harness"] in (harness, "any")]


def export(harness: str, rows: list[dict], *, base: dict, window: int | None = None,
           max_output: int | None = None, now: float | None = None) -> dict:
    """The kit for `harness` as {files: {path: bytes}, manifest}. Nothing is
    fetched and nothing downloaded; `window`/`max_output` come from the
    caller (the proxy's /v1/models numbers), or stay as tokens the README
    explains. Raises Refused if a secret survived into any file."""
    import harness_kit_guide as guide
    if harness not in HARNESS_IDS:
        raise Refused(404, f"no harness {harness!r} (known: {', '.join(HARNESS_IDS)})")
    now = now or time.time()
    url = base["url"]
    ctx = {"PUBLIC_BASE": url, "API_BASE": url + "/v1", "TOOLS_MCP_URL": url + "/tools/mcp",
           "KEY_PLACEHOLDER": KEY_PLACEHOLDER}
    if window:
        ctx["CONTEXT_WINDOW"] = int(window)
        ctx["MAX_OUTPUT"] = int(max_output or 0) or ""
    top = f"yamadori-kit-{harness}"
    files: dict[str, bytes] = {}
    mine = for_harness(rows, harness)
    remote = [r for r in mine if r["where"] in ("remote", "both")]
    rec = [r for r in remote if r["status"] == "recommended"]
    trying = [r for r in remote if r["status"] == "trying"]
    manifest = {"kit": 1, "harness": harness, "generated": now,
                "public_base": base, "key": {"placeholder": KEY_PLACEHOLDER,
                                             "note": "no key is in this kit; see README.md 'The key'"},
                "window": {"context": window, "max_output": max_output,
                           "source": "the proxy's /v1/models (mcp/catalog.py)" if window else
                           "not read: fill in from GET {API_BASE}/models (README)"},
                "entries": [], "downloads": [], "skills": [], "prompts": [], "configs": [],
                "trying": [], "left_out": []}

    def put(path, data):
        files[f"{top}/{path}"] = data if isinstance(data, bytes) else data.encode()

    def ref(r):
        return {"id": r["id"], "version": r["version"], "title": r["title"], "kind": r["kind"],
                "harness": r["harness"], "status": r["status"], "proof": r["proof"],
                "where": r["where"], "needs_operator": r["needs_operator"],
                "evidence": r["evidence"]}

    used_names: set = set()

    def fname(r, default_ext):
        n = r.get("file_name") or (_slug(r["title"]) + default_ext)
        if n in used_names:
            n = f"{r['id']}-{n}"
        used_names.add(n)
        return n

    for group, lst in (("", rec), ("trying/", trying)):
        for r in lst:
            manifest["entries" if not group else "trying"].append(ref(r))
            if r["kind"] == "skill":
                name = _skill_name(r)
                for rel, data in _skill_files(r).items():
                    put(f"{group}skills/{name}/{rel}", data)
                (manifest["skills"] if not group else manifest["trying"][-1].setdefault("files", [])).append(
                    name if not group else f"{group}skills/{name}/")
            elif (r.get("body") or "").strip() and (r["kind"] in ("config", "prompt")
                                                     or (r["kind"] == "mcp_server" and r.get("file_name"))):
                sub = "prompts" if r["kind"] == "prompt" else "config"
                n = fname(r, ".md" if r["kind"] == "prompt" else ".txt")
                put(f"{group}{sub}/{n}", templated_text(r["body"], ctx) + "\n")
                item = {"file": f"{group}{sub}/{n}", "target": r.get("target") or "", "title": r["title"]}
                if not group:
                    manifest["prompts" if sub == "prompts" else "configs"].append(item)
                else:
                    manifest["trying"][-1].setdefault("files", []).append(item["file"])
            if r.get("download") and not group:
                dl = r["download"]
                manifest["downloads"].append({"title": r["title"], "source_url": dl["source_url"],
                                              "version": dl["version"], "hash": dl["hash"],
                                              "commit": dl["commit"], "licence": dl["licence"],
                                              "install": dl["install"], "verify": dl["verify"]})
    for r in mine:
        if r not in rec and r not in trying:
            manifest["left_out"].append({"id": r["id"], "title": r["title"], "status": r["status"],
                                         "where": r["where"], "why": r.get("status_why") or
                                         ("box only: the remote machine does not have this"
                                          if r["where"] == "box" else "")})
    put("README.md", guide.readme(harness, manifest, rec, trying, ctx))
    put("kit.json", json.dumps(manifest, indent=2, default=str) + "\n")
    leaks = []
    for path, data in files.items():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for s in secrets_in(text):
            leaks.append({"field": path, "why": s["what"]})
    if leaks:
        raise Refused(500, "the kit would carry a secret; nothing was exported", leaks)
    return {"files": files, "manifest": manifest, "top": top}


def zip_bytes(kit: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(kit["files"]):
            info = zipfile.ZipInfo(path, date_time=time.localtime(kit["manifest"]["generated"])[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, kit["files"][path])
    return buf.getvalue()
