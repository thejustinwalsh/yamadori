#!/usr/bin/env python
"""HARNESS TOOLS (mcp/harness_kit.py, mcp/dash_harness.py), asserted offline.

  the store    create / edit (a new version, history kept, a stale
               base_version refused) / soft delete and restore; validation
               of every field; no secret is ever stored
  the seed     every item valid, every item names evidence, the operator's
               three pending decisions are `trying` + needs_operator, the
               counts per harness and kind; idempotent; an edited entry is
               never overwritten by a later seed revision
  the API      through server.py's real app: 401 without a key, then the
               routes with a test key minted into a TEMP account registry;
               validation answers 400 with reasons, a stale edit 409
  the export   per harness: no key (the placeholder instead, and a key-shaped
               string anywhere refuses the export), every recommended
               download carries its exact version and hash into README and
               kit.json, the configs point at the public base, box-only and
               rejected entries are left out, the skill folders are whole,
               the zip opens

Every store is a temp path (mcp/offline_stores.py) set BEFORE any import; the
window the export would read from the model server is a fixed number; no
port is reached.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_harness_kit_")
os.environ["YAMADORI_ACCOUNTS_DIR"] = os.path.join(_TMP, "accounts")
os.environ.pop("YAMADORI_PUBLIC_BASE", None)

import dash_harness  # noqa: E402
import harness_kit  # noqa: E402
import harness_kit_seed  # noqa: E402

WINDOW, MAX_OUT = 163840, 32768
dash_harness.window_source = lambda: (WINDOW, MAX_OUT)

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def fresh_db() -> str:
    return os.path.join(tempfile.mkdtemp(dir=_TMP), "hk.sqlite3")


def good(**over) -> dict:
    d = {"harness": "pi", "kind": "note", "title": "a note", "body": "text",
         "status": "recommended", "proof": "verified", "where": "both",
         "evidence": [{"ref": "docs/HARNESSES.md", "showed": "it"}]}
    d.update(over)
    return d


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except harness_kit.Refused as e:
        return e
    return None


FAKE_KEY = "ym-" + "A1b2C3d4" * 5          # the shape accounts.create mints


# ------------------------------------------------------------------ store

def test_paths_are_temp():
    check(os.path.abspath(harness_kit.DB_PATH).startswith(os.path.abspath(_TMP)),
          "the store is the temp path, never index/harness_kit.sqlite3", harness_kit.DB_PATH)
    check("YAMADORI_HARNESS_KIT_DB" in offline_stores.STORES, "offline_stores moves the harness kit store")


def test_create_edit_history():
    db = fresh_db()
    e = harness_kit.create(good(), "operator:abc", db)
    check(e["version"] == 1 and e["id"].startswith("hk-"), "a new entry is version 1 with an hk- id", str(e.get("id")))
    check(e["created_by"] == "operator:abc" and e["history"][0]["action"] == "create", "the author and a create row are recorded")
    e2 = harness_kit.edit(e["id"], {"title": "a better note"}, "operator:def", 1, db)
    check(e2["version"] == 2 and e2["title"] == "a better note", "an edit is a new version", str(e2.get("version")))
    check([h["version"] for h in e2["history"]] == [1, 2] and e2["history"][0]["data"]["title"] == "a note",
          "history keeps every version whole")
    check(e2["updated_by"] == "operator:def" and e2["created_by"] == "operator:abc", "created_by stays, updated_by moves")
    r = refused(harness_kit.edit, e["id"], {"title": "stale"}, "operator:x", 1, db)
    check(r is not None and r.code == 409, "an edit against an old version is refused 409", r and r.message)
    r = refused(harness_kit.edit, e["id"], {"title": "no base"}, "operator:x", None, db)
    check(r is not None and r.code == 409, "an edit with no base_version is refused 409")
    same = harness_kit.edit(e["id"], {"title": "a better note"}, "operator:x", 2, db)
    check(same.get("unchanged") and same["version"] == 2, "an edit that changes nothing writes no version")
    r = refused(harness_kit.edit, "hk-nope", {"title": "x"}, "o", 1, db)
    check(r is not None and r.code == 404, "editing an unknown id is 404")


def test_soft_delete():
    db = fresh_db()
    e = harness_kit.create(good(), "operator:a", db)
    r = refused(harness_kit.set_deleted, e["id"], True, "operator:a", "", db)
    check(r is not None and r.code == 400, "a delete must say why")
    d = harness_kit.set_deleted(e["id"], True, "operator:a", "superseded", db)
    check(d["deleted"] and d["deleted_why"] == "superseded" and d["deleted_by"] == "operator:a",
          "a delete is soft: who, when and why kept")
    check(all(x["id"] != e["id"] for x in harness_kit.entries(db)), "a deleted entry leaves the live list")
    check(any(x["id"] == e["id"] for x in harness_kit.entries(db, include_deleted=True)), "and stays in the store")
    r = refused(harness_kit.edit, e["id"], {"title": "x"}, "o", 1, db)
    check(r is not None and r.code == 409, "a deleted entry cannot be edited until restored")
    back = harness_kit.set_deleted(e["id"], False, "operator:b", "", db)
    check(not back["deleted"] and [h["action"] for h in back["history"]] == ["create", "delete", "restore"],
          "restore brings it back; history shows create, delete, restore")


def test_validation():
    db = fresh_db()
    cases = [
        ("an unknown harness", good(harness="cursor"), "harness"),
        ("an unknown kind", good(kind="plugin"), "kind"),
        ("an unknown status", good(status="maybe"), "status"),
        ("an unknown proof", good(proof="vibes"), "proof"),
        ("an unknown where", good(where="cloud"), "where"),
        ("no title", good(title=""), "title"),
        ("a recommended entry with no evidence", good(evidence=[]), "evidence"),
        ("a rejected entry with no reason", good(status="rejected"), "status_why"),
        ("a trying entry with no reason", good(status="trying", evidence=[]), "status_why"),
        ("evidence with no 'showed'", good(evidence=[{"ref": "docs/X.md", "showed": ""}]), "evidence[0].showed"),
        ("a download with no download block", good(kind="download"), "download"),
        ("a recommended download with a range", good(kind="download", download={
            "source_url": "https://www.npmjs.com/package/x", "version": "^1.2.0", "hash": "sha256:" + "a" * 64,
            "licence": "MIT", "install": {"any": "npm i -g x@^1.2.0"}}), "download.version"),
        ("a recommended download with no hash or commit", good(kind="download", download={
            "source_url": "https://www.npmjs.com/package/x", "version": "1.2.0", "licence": "MIT",
            "install": {"any": "npm i -g x@1.2.0"}}), "download.hash"),
        ("an install command without the exact version", good(kind="download", download={
            "source_url": "https://www.npmjs.com/package/x", "version": "1.2.0", "hash": "sha256:" + "a" * 64,
            "licence": "MIT", "install": {"any": "npm i -g x"}}), "download.install.any"),
        ("a malformed hash", good(kind="download", download={
            "source_url": "https://x.test/", "version": "1.2.0", "hash": "md5:abc",
            "licence": "MIT", "install": {"any": "get x 1.2.0"}}), "download.hash"),
        ("a trying download with no pin and no reason", good(kind="download", status="trying", status_why="vetting",
            download={"source_url": "https://x.test/", "version": "1.2.0", "licence": "MIT"}), "download.pin_missing_why"),
        ("a skill folder outside the allowed roots", good(kind="skill", skill_path="index/accounts"), "skill_path"),
        ("a skill folder that escapes with ..", good(kind="skill", skill_path="bench/sandbox/harness_skills/../../../index"), "skill_path"),
        ("a skill with neither folder nor SKILL.md", good(kind="skill", body="just words"), "body"),
        ("a file name with a path", good(kind="config", file_name="../x.json"), "file_name"),
    ]
    for what, fields, field in cases:
        r = refused(harness_kit.create, fields, "o", db)
        ok = r is not None and r.code == 400 and any(x["field"] == field for x in r.reasons)
        check(ok, f"refused: {what} ({field})", r.message if r else "accepted")
    ok = harness_kit.create(good(kind="download", download={
        "source_url": "https://www.npmjs.com/package/x/v/1.2.0", "version": "1.2.0",
        "hash": "sha512-" + "A" * 86 + "==", "licence": "MIT",
        "install": {"windows": "npm install -g x@1.2.0", "unix": "npm install -g x@1.2.0"}}), "o", db)
    check(ok["download"]["version"] == "1.2.0", "a fully pinned recommended download is accepted")
    sk = harness_kit.create(good(kind="skill", skill_path="bench/sandbox/harness_skills/type-check"), "o", db)
    check(sk["skill_path"] == "bench/sandbox/harness_skills/type-check", "a skill folder under an allowed root is accepted")


def test_no_secret_is_stored():
    db = fresh_db()
    shapes = {"a yamadori key": FAKE_KEY, "an sk- key": "sk-" + "x" * 30,
              "a literal Bearer": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123",
              "a GitHub token": "ghp_" + "a" * 36, "a PEM key": "-----BEGIN RSA PRIVATE KEY-----"}
    for what, s in shapes.items():
        for field, fields in (("body", good(body=f"use {s}")),
                              ("evidence", good(evidence=[{"ref": "run x", "showed": s}]))):
            r = refused(harness_kit.create, fields, "o", db)
            check(r is not None and r.code == 400 and "secret" in r.message,
                  f"{what} in {field} is refused", r.message if r else "accepted")
            if r:
                check(s not in r.message and s not in json.dumps(r.reasons),
                      f"the refusal never echoes {what}")
    e = harness_kit.create(good(), "o", db)
    r = refused(harness_kit.edit, e["id"], {"body": FAKE_KEY}, "o", 1, db)
    check(r is not None and r.code == 400, "an edit cannot slip a key in either")
    refs = ["Bearer ${YAMADORI_API_KEY}", "Bearer $YAMADORI_API_KEY", "Bearer {env:YAMADORI_API_KEY}",
            f"Bearer {harness_kit.KEY_PLACEHOLDER}", "apiKey: $YAMADORI_PI_KEY"]
    check(not harness_kit.secrets_in(refs), "references and the placeholder are not secrets",
          json.dumps(harness_kit.secrets_in(refs)))
    raw = open(db, "rb").read()
    check(FAKE_KEY.encode() not in raw, "no refused key reached the database file")


# ------------------------------------------------------------------ seed

def test_seed_items():
    items = harness_kit_seed.items()
    ids = [i["seed_id"] for i in items]
    check(len(ids) == len(set(ids)), "seed ids are unique")
    bad = []
    for it in items:
        fields = {k: v for k, v in it.items() if k in harness_kit.FIELDS}
        b = harness_kit.validate(harness_kit.normalise(fields))
        if b:
            bad.append((it["seed_id"], b))
    check(not bad, f"every seed item passes validation ({len(items)} items)", json.dumps(bad)[:400])
    check(all(i.get("evidence") for i in items), "every seed item names its evidence",
          ", ".join(i["seed_id"] for i in items if not i.get("evidence")))
    check(all(e["ref"] and e["showed"] for i in items for e in i["evidence"]), "every evidence item has a ref and what it showed")
    need = {i["seed_id"] for i in items if i.get("needs_operator")}
    check(need == {"hermes.lsmcp", "hermes.lean-skills", "codex.package-api"},
          "the operator's three pending decisions carry needs_operator", str(sorted(need)))
    check(all(i["status"] == "trying" for i in items if i.get("needs_operator")), "each pending decision is `trying`")
    harn = {i["harness"] for i in items}
    check(harn == set(harness_kit.HARNESS_IDS), "every harness of the dropdown has seed entries", str(sorted(harn)))
    by = {i["seed_id"]: i for i in items}
    check(by["any.bridge.cclsp"]["status"] == "rejected" and by["any.bridge.isaacphi"]["status"] == "rejected"
          and by["any.bridge.serena"]["status"] == "trying" and by["hermes.lsmcp"]["download"]["version"] == "0.10.0",
          "the LSP bridges: lsmcp to vet (0.10.0), serena second, cclsp and isaacphi rejected")
    check("PI_CACHE_RETENTION=long" in by["pi.env"]["body"], "Pi's PI_CACHE_RETENTION=long is seeded")
    check("{{CONTEXT_WINDOW}}" in by["pi.models"]["body"], "Pi's window comes from /v1/models at export")
    check(by["hermes.lean"]["status"] == "recommended" and "10 tools" in by["hermes.lean"]["title"],
          "the Hermes lean arm (10 tools) is seeded")
    check("createWorld" in by["opencode.lsp-package-api"]["body"], "OpenCode's lsp on koota's createWorld is seeded")
    check(by["any.slow-webgl"]["where"] == "box" and by["any.slow-webgl"]["proof"] == "unmeasured",
          "the slow-WebGL fact is box-only and its effect unmeasured")
    skills = {i["skill_path"] for i in items if i.get("skill_path")}
    for sp in ("bench/sandbox/harness_skills/package-api", "bench/sandbox/harness_skills/page-check",
               "bench/sandbox/harness_skills/type-check", "bench/octopus/hermes_skills/look-at-a-screenshot",
               "bench/octopus/hermes_skills/type-check"):
        check(sp in skills, f"the harness skill {sp.rsplit('/', 1)[1]} ({sp.split('/')[1]}) is seeded")
    for i in items:
        dl = i.get("download")
        if dl and i["status"] == "recommended":
            check(harness_kit.SRI512.match(dl["hash"] or "") and dl["version"] in dl["install"]["windows"],
                  f"recommended download {i['seed_id']} is pinned by sha512 with its exact version in the command")
    counts = {}
    for i in items:
        counts.setdefault(i["harness"], {}).setdefault(i["kind"], 0)
        counts[i["harness"]][i["kind"]] += 1
    print("  seed counts per harness and kind: " + json.dumps(counts, sort_keys=True))


def test_seed_idempotent_and_respects_edits():
    db = fresh_db()
    items = harness_kit_seed.items()
    a = harness_kit.seed(items, db)
    check(len(a["added"]) == len(items), "the first seed adds every item", str(len(a["added"])))
    b = harness_kit.seed(items, db)
    check(not b["added"] and not b["updated"], "a second seed adds nothing")
    rows = {r["seed_id"]: r for r in harness_kit.entries(db)}
    target = rows["pi.env"]
    harness_kit.edit(target["id"], {"body": target["body"] + "\nPI_OFFLINE=1"}, "operator:me", target["version"], db)
    gone = rows["hermes.startup-probes"]
    harness_kit.set_deleted(gone["id"], True, "operator:me", "not useful", db)
    newer = [dict(i, seed_rev=2, title=i["title"] + " (rev 2)") if i["seed_id"] in ("pi.env", "pi.settings",
                                                                                     "hermes.startup-probes") else i
             for i in items]
    c = harness_kit.seed(newer, db)
    check(c["updated"] == ["pi.settings"], "a newer seed revision updates an entry only the seed wrote", str(c))
    check(sorted(c["kept_edited"]) == ["hermes.startup-probes", "pi.env"],
          "an entry a person edited or deleted is kept as they left it", str(c))
    after = {r["seed_id"]: r for r in harness_kit.entries(db, include_deleted=True)}
    check("PI_OFFLINE=1" in after["pi.env"]["body"] and after["hermes.startup-probes"]["deleted"],
          "the operator's edit and delete survive the reseed")
    h = harness_kit.get(after["pi.settings"]["id"], db)["history"]
    check([x["action"] for x in h] == ["seed", "reseed"], "a reseed is recorded in history", str([x["action"] for x in h]))


# ------------------------------------------------------------------ export

def _kit(harness, db, **kw):
    harness_kit.seed(harness_kit_seed.items(), db)
    return harness_kit.export(harness, harness_kit.entries(db),
                              base=kw.get("base", {"url": "https://ai.example.test", "source": "test"}),
                              window=kw.get("window", WINDOW), max_output=kw.get("max_output", MAX_OUT))


def test_export_per_harness():
    db = fresh_db()
    rows = None
    for h in harness_kit.HARNESS_IDS:
        kit = _kit(h, db)
        rows = rows or harness_kit.entries(db)
        files = kit["files"]
        text = {p: b.decode("utf-8", "replace") for p, b in files.items()}
        top = kit["top"]
        check(f"{top}/README.md" in files and f"{top}/kit.json" in files, f"[{h}] the kit has README.md and kit.json")
        allt = "\n".join(text.values())
        check(not harness_kit.secrets_in(allt), f"[{h}] nothing key-shaped anywhere in the kit")
        check(harness_kit.KEY_PLACEHOLDER in text[f"{top}/README.md"], f"[{h}] the README carries the key placeholder and says where")
        check("No key is in this kit" in text[f"{top}/README.md"], f"[{h}] the README says no key is in the kit")
        m = json.loads(text[f"{top}/kit.json"])
        check(m["key"]["placeholder"] == harness_kit.KEY_PLACEHOLDER, f"[{h}] kit.json names the placeholder")
        rec_dl = [r for r in harness_kit.for_harness(rows, h)
                  if r["kind"] == "download" and r["status"] == "recommended" and r["where"] in ("remote", "both")]
        check(len(m["downloads"]) == len(rec_dl), f"[{h}] every recommended download is in the kit ({len(rec_dl)})")
        for d in m["downloads"]:
            ok = (d["hash"] and d["hash"] in text[f"{top}/README.md"] and d["version"] in d["install"]["windows"]
                  and d["version"] in d["install"]["unix"] and d["licence"])
            check(ok, f"[{h}] {d['title'][:40]}: hash, exact version and licence carried")
        titles = {e["title"] for e in m["entries"]}
        box_only = [r["title"] for r in harness_kit.for_harness(rows, h) if r["where"] == "box"]
        check(not (titles & set(box_only)), f"[{h}] box-only entries are left out of the remote kit")
        rej = [r["title"] for r in harness_kit.for_harness(rows, h) if r["status"] == "rejected"]
        check(not (titles & set(rej)) and all(any(x["title"] == t for x in m["left_out"]) for t in rej),
              f"[{h}] rejected entries are left out and listed as such")
        leftover = [p for p, t in text.items() if "{{" in t and p.endswith((".json", ".toml", ".yaml", ".env"))]
        check(not leftover, f"[{h}] every token is filled when the window is known", str(leftover))
        z = zipfile.ZipFile(io.BytesIO(harness_kit.zip_bytes(kit)))
        check(sorted(z.namelist()) == sorted(files) and z.testzip() is None, f"[{h}] the zip opens and holds every file")


def test_export_contents():
    db = fresh_db()
    pi = _kit("pi", db)
    t = {p.split("/", 1)[1]: b for p, b in pi["files"].items()}
    models = json.loads(t["config/models.json"])
    prov = models["providers"]["yamadori"]
    check(prov["baseUrl"] == "https://ai.example.test/v1" and prov["apiKey"] == "$YAMADORI_PI_KEY",
          "Pi's models.json points at the public base and reads the key from the environment")
    check(prov["models"][0]["contextWindow"] == WINDOW and prov["models"][0]["maxTokens"] == MAX_OUT,
          "Pi's window and output ceiling are the proxy's")
    src = os.path.join(harness_kit.ROOT, "bench", "sandbox", "harness_skills", "package-api")
    check(t.get("skills/package-api/api.cjs") == open(os.path.join(src, "api.cjs"), "rb").read()
          and t.get("skills/package-api/SKILL.md") == open(os.path.join(src, "SKILL.md"), "rb").read(),
          "a skill folder is carried whole and byte for byte")
    check("skills/page-check/SKILL.md" not in t, "the box-only page-check skill stays out of the remote kit")
    cx = _kit("codex", db)
    t = {p.split("/", 1)[1]: b.decode() for p, b in cx["files"].items()}
    check('base_url = "https://ai.example.test/v1"' in t["config/config.toml"]
          and 'env_key = "YAMADORI_CODEX_KEY"' in t["config/config.toml"],
          "Codex's config.toml points at the public base and names the key's variable")
    cat = json.loads(t["config/yamadori-catalog.json"])
    check(cat["models"][0]["context_window"] == WINDOW, "Codex's catalog carries the proxy's window")
    check("trying/skills/package-api/SKILL.md" in t, "package-api for Codex is in trying/, not installed")
    check("NEEDS OPERATOR" in t["README.md"], "the README marks the pending decision")
    oc = _kit("opencode", db)
    t = {p.split("/", 1)[1]: b.decode() for p, b in oc["files"].items()}
    j = json.loads(t["config/opencode.json"])
    check(j["provider"]["yamadori"]["options"]["apiKey"] == "{env:YAMADORI_OPENCODE_KEY}"
          and j["provider"]["yamadori"]["options"]["baseURL"] == "https://ai.example.test/v1",
          "OpenCode's config reads the key from {env:...} and points at the public base")
    check(j["permission"]["task"] == "deny" and j["lsp"]["typescript-box"]["command"][0] == "typescript-language-server"
          and "--cdp-endpoint" not in json.dumps(j["mcp"]),
          "OpenCode's loadout is carried; the browser launches the machine's own (no sidecar CDP)")
    cc = _kit("claude-code", db)
    t = {p.split("/", 1)[1]: b.decode() for p, b in cc["files"].items()}
    check('"url": "https://ai.example.test/tools/mcp"' in t["config/mcp.json"]
          and "${YAMADORI_API_KEY}" in t["config/mcp.json"],
          "Claude Code's MCP config points at /tools/mcp and keeps the ${VAR} reference")
    cs = json.loads(t["config/settings.json"])
    check(cs["env"]["ANTHROPIC_BASE_URL"] == "https://ai.example.test"
          and cs["env"]["ANTHROPIC_MODEL"] == "yamadori" and cs["model"] == "yamadori"
          and cs["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == str(WINDOW)
          and cs["effortLevel"] == "medium"
          and not any("TOKEN" in k and "MAX" not in k for k in cs["env"]),
          "Claude Code's settings.json points ANTHROPIC_BASE_URL at the public base (Claude Code adds /v1/messages), "
          "pins the model, carries the proxy's window, and holds no key variable")
    ce = t["config/claude-code.env"]
    check("ANTHROPIC_AUTH_TOKEN=<PASTE-YOUR-YAMADORI-KEY-HERE>" in ce and "ANTHROPIC_AUTH_TOKEN" in t["README.md"]
          and "/v1/messages" in t["README.md"],
          "Claude Code's key is a placeholder only (claude-code.env), and the README names the variable")
    hm = _kit("hermes", db, window=None, max_output=None)
    t = {p.split("/", 1)[1]: b.decode() for p, b in hm["files"].items()}
    check('base_url: "https://ai.example.test/v1"' in t["config/hermes-provider.yaml"]
          and "key_env: YAMADORI_API_KEY" in t["config/hermes-provider.yaml"],
          "Hermes' provider points at the public base with key_env")
    check("NOT filled in" in t["README.md"], "with no window from the proxy the README says how to fill it in")
    check(not any(p.startswith("skills/") for p in t), "no box-only Hermes skill in the remote kit")


def test_export_refuses_a_leak():
    db = fresh_db()
    harness_kit.seed(harness_kit_seed.items(), db)
    rows = harness_kit.entries(db)
    rows[0] = dict(rows[0], body="token " + FAKE_KEY, harness="any", kind="note", where="both")
    r = refused(harness_kit.export, "pi", rows, base={"url": "https://ai.example.test", "source": "t"},
                window=WINDOW, max_output=MAX_OUT)
    check(r is not None and r.code == 500 and "secret" in r.message and FAKE_KEY not in r.message,
          "a key that got past the store still refuses the whole export")
    r = refused(harness_kit.export, "cursor", rows, base={"url": "x", "source": "t"})
    check(r is not None and r.code == 404, "an unknown harness is 404")


def test_public_base():
    b = harness_kit.public_base(env={"YAMADORI_PUBLIC_BASE": "https://ai.example.test/"})
    check(b == {"url": "https://ai.example.test", "source": "YAMADORI_PUBLIC_BASE"}, "YAMADORI_PUBLIC_BASE wins", str(b))
    cf = os.path.join(_TMP, "Caddyfile")
    open(cf, "w").write("{\n\temail a@b.c\n}\n\nai.example.test {\n\tlog\n}\n")
    b = harness_kit.public_base(env={}, caddyfile=cf)
    check(b["url"] == "https://ai.example.test" and "Caddyfile" in b["source"], "else the Caddy site address", str(b))
    b = harness_kit.public_base(env={}, caddyfile=os.path.join(_TMP, "none"))
    check(b["url"].startswith("https://<") and "none" in b["source"], "else a placeholder that says so")


# ------------------------------------------------------------------ API

def _client():
    from starlette.testclient import TestClient
    import accounts
    import server
    key = accounts.create("harness-kit-tests")      # a TEMP registry (YAMADORI_ACCOUNTS_DIR)
    return TestClient(server.app), {"Authorization": f"Bearer {key}"}, key


def test_api():
    c, auth, key = _client()
    base = "/dash/api/harness-kit"
    for method, path, body in (("get", base, None), ("get", base + "/export/pi", None),
                               ("post", base + "/entry", good())):
        r = c.get(path) if method == "get" else c.post(path, json=body)
        check(r.status_code == 401, f"{method.upper()} {path} without a key is 401", str(r.status_code))
        r = c.get(path, headers={"Authorization": "Bearer ym-wrongwrongwrongwrongwrong"}) if method == "get" else \
            c.post(path, json=body, headers={"Authorization": "Bearer ym-wrongwrongwrongwrongwrong"})
        check(r.status_code == 401, f"{method.upper()} {path} with an unknown key is 401", str(r.status_code))
    r = c.get(base, headers=auth)
    j = r.json()
    check(r.status_code == 200 and j["counts"]["total"] == len(harness_kit_seed.items()),
          "GET lists the seeded entries", str(r.status_code))
    check({h["id"] for h in j["harnesses"]} == set(harness_kit.HARNESS_IDS) and j["key_placeholder"] == harness_kit.KEY_PLACEHOLDER,
          "GET carries the dropdown's harnesses and the placeholder")
    r = c.post(base + "/entry", json={"fields": good(title="from the API")}, headers=auth)
    e = r.json().get("entry") or {}
    check(r.status_code == 200 and e.get("title") == "from the API", "POST entry creates", str(r.status_code))
    check(str(e.get("created_by", "")).startswith("operator:") and key not in json.dumps(e),
          "the author is the account id, never the key")
    r = c.post(base + "/entry", json={"fields": good(status="rejected")}, headers=auth)
    check(r.status_code == 400 and any(x["field"] == "status_why" for x in r.json()["reasons"]),
          "a bad entry is 400 with reasons", r.text[:200])
    r = c.post(base + "/entry", json={"fields": good(body=FAKE_KEY)}, headers=auth)
    check(r.status_code == 400 and FAKE_KEY not in r.text, "a key in a body is 400 and not echoed")
    r = c.post(base + "/entry/edit", json={"id": e["id"], "base_version": 1, "fields": {"title": "edited"}}, headers=auth)
    check(r.status_code == 200 and r.json()["entry"]["version"] == 2, "POST entry/edit makes version 2")
    r = c.post(base + "/entry/edit", json={"id": e["id"], "base_version": 1, "fields": {"title": "late"}}, headers=auth)
    check(r.status_code == 409, "a stale edit is 409", str(r.status_code))
    r = c.get(f"{base}/entry/{e['id']}", headers=auth)
    check(r.status_code == 200 and len(r.json()["entry"]["history"]) == 2, "GET entry/<id> returns the history")
    r = c.post(base + "/entry/delete", json={"id": e["id"], "why": "test"}, headers=auth)
    check(r.status_code == 200 and r.json()["entry"]["deleted"], "POST entry/delete soft-deletes")
    r = c.get(base + "/deleted", headers=auth)
    check(any(x["id"] == e["id"] for x in r.json()["entries"]), "GET deleted lists it")
    r = c.post(base + "/entry/restore", json={"id": e["id"]}, headers=auth)
    check(r.status_code == 200 and not r.json()["entry"]["deleted"], "POST entry/restore brings it back")
    r = c.get(base + "/export/codex", headers=auth)
    check(r.status_code == 200 and r.headers["content-type"].startswith("application/zip"), "GET export/<h> is a zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    blob = b"".join(z.read(n) for n in z.namelist())
    check(key.encode() not in blob and not harness_kit.secrets_in(blob.decode("utf-8", "replace")),
          "the exported zip holds neither the caller's key nor anything key-shaped")
    r = c.get(base + "/export/codex.json", headers=auth)
    check(r.status_code == 200 and "yamadori-kit-codex/README.md" in r.json()["files"], "GET export/<h>.json previews the kit")
    r = c.get(base + "/export/cursor", headers=auth)
    check(r.status_code == 404, "an unknown harness is 404")
    check(key not in open(harness_kit.DB_PATH, "rb").read().decode("latin-1"), "the key never reached the store file")


def test_server_routes_it():
    src = open(os.path.join(HERE, "server.py"), encoding="utf-8").read()
    check("import dash_harness" in src and "dash_harness.handle_get" in src and "dash_harness.handle_post" in src,
          "server.py dispatches the harness kit through the gated GET and POST chains")


def main() -> int:
    for fn in (test_paths_are_temp, test_create_edit_history, test_soft_delete, test_validation,
               test_no_secret_is_stored, test_seed_items, test_seed_idempotent_and_respects_edits,
               test_export_per_harness, test_export_contents, test_export_refuses_a_leak,
               test_public_base, test_api, test_server_routes_it):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised", traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            if not name:
                continue
            print(("  pass  " if ok else "  FAIL  ") + name + (f"   <- {detail}" if not ok and detail else ""))
    res = [r for r in _results if r[1]]
    passed = sum(1 for ok, _, _ in res if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(res)} checks passed")
    return 0 if passed == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
