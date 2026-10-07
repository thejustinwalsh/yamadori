#!/usr/bin/env python
"""PACKAGE ONBOARDING, offline: a prompt with links -> a held, tested
package, through the durable machine (docs/PACKAGE-ONBOARDING.md).

    python mcp/test_onboarding.py      -> "N/M checks passed"

No network, no GPU, no model: the registry, GitHub and the tarball are a
FAKE transport (package_net.TRANSPORT) over a tiny package, the skill
pipeline's document fetch reads the same fake, the embedder-backed index is
a fake deps.index_package, and the stack's idleness is a switch. Every store
is a temp path (offline_stores.isolate). The stages run through the real
worker (worker.run / run_one) and the real datasets rules.

  1. THE QUEUE. jobs.defer returns a running job without an attempt;
     claim() skips it until not_before; worker --once does not loop on it.
  2. IDLE. idle.stack_idle: a recent request (until = the request + the
     learner's IDLE_MINUTES), another gpu job, a generating slot, an
     unreadable /running -- each not idle; an idle-gated job is deferred by
     run_one BEFORE its handler, with no attempt spent.
  3. RATE LIMITS. A server's reset time defers the stage exactly that long.
  4. KIND package. The stage order, the per-kind questions, create().
  5. RESOLVE. Links, pins, the major and latest rules, gitHead, a GitHub
     repository link, siblings for two packages, the licence from verbatim
     quotes, the first resolution pins, replace vs alongside.
  6. CLARIFY. The licence NEVER blocks (operator, 2026-10-07): a quoted one
     is recorded and passes with no human; a restricted one is a note and
     holds nothing; none found is recorded as "not established" and passes.
  7. INDEX. The tarball verified against dist.integrity (a mismatch and an
     unpackedSize overrun are refused, unpacked nothing); LICENSE extracted;
     an indexed, healthy, embedded package is skipped.
  8. VOCAB. Promoted on a passing floor; HELD with the reason on a failing
     one (a blocker naming the rows); the operator's force + the sweep.
  9. SOURCES. Tier 1 (SKILL.md + references), the README lead only without
     a SKILL.md, llms.txt recorded (not ingested) beside a SKILL.md and
     ingested without one, prompt links, idempotency, the onboarding keys on
     every skill and the idle flag on its gpu stages.
 10. THE SKILLS JOIN. A skill in the pipeline blocks; an errored skill job
     blocks with its remedy; all stopped -> the sweep advances.
 11. REPLACE. A new version of the same major re-versions the earlier
     skill for the same path; a gone section is retired only at retire; a
     new major sits alongside with the risk recorded.
 12. A KILLED WORKER. A running stage job whose worker died is requeued with
     its attempts, and the stage re-runs idempotently.
 13. THE API. Submit (prompt shape), list, detail, review, promote, tier3;
     no filesystem path in any answer.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_onboarding_")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "packages")
os.environ["YAMADORI_PKG_HISTORY"] = os.path.join(_TMP, "packages",
                                                  "registry_history.json")
os.environ["YAMADORI_PKG_REGISTRY_DIR"] = os.path.join(_TMP, "packages")
os.environ["YAMADORI_PACKAGES_SRC"] = os.path.join(_TMP, "packages", "_src")
os.environ["YAMADORI_SKILL_CACHE_SECONDS"] = "0"
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")
os.environ["YAMADORI_POLL_SECONDS"] = "0.05"
os.environ["YAMADORI_BEAT_SECONDS"] = "3600"

import datasets  # noqa: E402
import deps  # noqa: E402
import idle  # noqa: E402
import jobs  # noqa: E402
import package_net  # noqa: E402
import skill_pipeline  # noqa: E402
import skills  # noqa: E402

FAILS: list[str] = []
N = [0]


def check(ok, name, detail=""):
    N[0] += 1
    if ok:
        print(f"  pass  {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f"\n        <- {str(detail)[:600]}"
                                   if detail != "" else ""))


# ===================================================================== fakes
C1 = "a" * 40            # widget 1.2.0's commit
C2 = "b" * 40            # widget 1.3.0's commit
C3 = "c" * 40            # widget 2.0.0's commit
CG = "d" * 40            # gizmo's commit


def tgz(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in files.items():
            data = text.encode("utf-8")
            ti = tarfile.TarInfo(f"package/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def sri(blob: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(blob).digest()).decode()


MIT = "MIT License\n\nCopyright (c) 2026 Acme\n\nPermission is hereby granted"
AGPL = ("GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3, 19 November 2007\n"
        "This program is licensed under the Affero General Public License.")
SKILL_MD = ("---\nname: widget\ndescription: Use when building with the "
            "widget package.\nlicense: MIT\n---\n# Widget\n\nUse makeWidget.")


def tarball_for(name: str, version: str, licence: str = MIT) -> bytes:
    return tgz({"package.json": json.dumps({"name": name, "version": version,
                                            "license": "MIT"}),
                "index.d.ts": "export declare function makeWidget(): void;\n"
                              "export declare function spinWidget(): void;\n",
                "LICENSE": licence, "README.md": "# widget\n",
                "bin/tool.exe": "MZ binary"})


BLOBS = {("acme-widget", "1.2.0"): tarball_for("acme-widget", "1.2.0"),
         ("acme-widget", "1.3.0"): tarball_for("acme-widget", "1.3.0"),
         ("acme-widget", "2.0.0"): tarball_for("acme-widget", "2.0.0"),
         ("@acme/gizmo", "0.1.0"): tarball_for("@acme/gizmo", "0.1.0"),
         ("agpl-thing", "1.0.0"): tarball_for("agpl-thing", "1.0.0", AGPL)}


def packument(name: str, versions: dict, latest: str, repo: str) -> dict:
    doc = {"name": name, "dist-tags": {"latest": latest}, "versions": {},
           "time": {}}
    for v, head in versions.items():
        blob = BLOBS.get((name, v), b"")
        vd = {"name": name, "version": v, "license": "MIT",
              "repository": {"type": "git", "url": f"git+https://github.com/"
                                                   f"{repo}.git"},
              "dist": {"tarball": f"https://registry.npmjs.org/{name}/-/"
                                  f"{name.split('/')[-1]}-{v}.tgz",
                       "integrity": sri(blob) if blob else None,
                       "unpackedSize": sum(len(x) for x in [blob]) * 20}}
        if head:
            vd["gitHead"] = head
        doc["versions"][v] = vd
        doc["time"][v] = "2026-09-01T00:00:00.000Z"
    return doc


PACKUMENTS = {
    "acme-widget": packument("acme-widget", {"1.2.0": C1, "1.3.0": C2,
                                             "2.0.0": C3, "2.1.0-beta.1": None},
                             "1.3.0", "acme/widget"),
    "@acme/gizmo": packument("@acme/gizmo", {"0.1.0": CG}, "0.1.0",
                             "acme/gizmo"),
    "agpl-thing": packument("agpl-thing", {"1.0.0": None}, "1.0.0",
                            "acme/agpl"),
}
PACKUMENTS["agpl-thing"]["versions"]["1.0.0"]["license"] = "AGPL-3.0"

# The repository at each commit: path -> text.
REPO = {
    ("acme/widget", C1): {
        "package.json": '{\n  "name": "acme-widget",\n  "version": "1.2.0",\n'
                        '  "license": "MIT"\n}\n',
        "LICENSE": MIT, "README.md": "# Widget\n\nThe widget package.",
        "skills/widget/SKILL.md": SKILL_MD,
        "skills/widget/references/api.md": "# API\n\nmakeWidget() makes one.",
        "skills/widget/references/old.md": "# Old\n\nThe old way.",
        "llms.txt": "# widget\n\n- [Guide](https://widget.dev/guide.md)\n",
        "examples/basic/main.ts": "import { makeWidget } from 'acme-widget'\n",
    },
    ("acme/widget", C2): {
        "package.json": '{\n  "name": "acme-widget",\n  "version": "1.3.0",\n'
                        '  "license": "MIT"\n}\n',
        "LICENSE": MIT, "README.md": "# Widget\n",
        "skills/widget/SKILL.md": SKILL_MD + "\n\nAlso spinWidget.",
        "skills/widget/references/api.md": "# API\n\nmakeWidget() makes one.",
    },
    ("acme/widget", C3): {
        "package.json": '{\n  "name": "acme-widget",\n  "version": "2.0.0",\n'
                        '  "license": "MIT"\n}\n',
        "LICENSE": MIT, "README.md": "# Widget 2\n",
        "skills/widget/SKILL.md": SKILL_MD,
    },
    ("acme/gizmo", CG): {
        "package.json": '{\n  "name": "@acme/gizmo",\n  "version": "0.1.0",\n'
                        '  "license": "MIT"\n}\n',
        "LICENSE": MIT, "README.md": "# Gizmo\n\nNo skill file here.",
        "llms.txt": "- [Gizmo guide](https://gizmo.dev/guide.md)\n",
    },
}
DOCS = {"https://widget.dev/guide.md": "# Guide\nMIT License\nUse widgets.",
        "https://gizmo.dev/guide.md": "# Gizmo guide\nMIT License\nGizmos.",
        "https://acme.dev/tips.md": "# Tips\nMIT License\nTips."}
RATE = {"on": False, "reset": None}
GETS: list[str] = []


def fake_transport(url, headers, max_bytes):
    GETS.append(url)
    R = package_net.Response
    assert headers.get("User-Agent") == package_net.UA
    assert "Authorization" not in headers and "Cookie" not in headers
    if RATE["on"] and "api.github.com" in url:
        return R(403, b"{}", {"x-ratelimit-remaining": "0",
                              "x-ratelimit-reset": str(RATE["reset"])}, url)
    if url.startswith("https://registry.npmjs.org/"):
        rest = url[len("https://registry.npmjs.org/"):]
        if "/-/" in rest:
            name, _, fn = rest.partition("/-/")
            for (n, v), blob in BLOBS.items():
                if n == name and fn.endswith(f"-{v}.tgz"):
                    return R(200, blob, {}, url)
            return R(404, b"", {}, url)
        name = rest.replace("%2F", "/")
        doc = PACKUMENTS.get(name)
        return R(200, json.dumps(doc).encode(), {}, url) if doc else \
            R(404, b'{"error":"Not found"}', {}, url)
    if url.startswith("https://api.github.com/repos/"):
        path = url[len("https://api.github.com/repos/"):]
        bits = path.split("/")
        repo = "/".join(bits[:2])
        tail = "/".join(bits[2:])
        if tail.startswith("git/trees/"):
            sha = tail[len("git/trees/"):].split("?")[0]
            files = REPO.get((repo, sha))
            if files is None:
                return R(404, b"{}", {}, url)
            ents = [{"path": p, "type": "blob", "sha": hashlib.sha1(
                t.encode()).hexdigest(), "size": len(t)} for p, t in
                files.items()]
            return R(200, json.dumps({"tree": ents, "truncated": False})
                     .encode(), {}, url)
        if tail.startswith("commits/"):
            ref = tail[len("commits/"):]
            sha = {"main": C2, "v1.3.0": C2}.get(ref, ref if len(ref) == 40
                                                 else None)
            return R(200, json.dumps({"sha": sha}).encode(), {}, url) if \
                sha else R(404, b"{}", {}, url)
        if tail == "" or tail == "/":
            return R(200, json.dumps({"default_branch": "main"}).encode(), {},
                     url)
        if tail.startswith("git/ref/tags/"):
            return R(404, b"{}", {}, url)
        return R(404, b"{}", {}, url)
    if url.startswith("https://raw.githubusercontent.com/"):
        bits = url[len("https://raw.githubusercontent.com/"):].split("/")
        repo, sha, path = "/".join(bits[:2]), bits[2], "/".join(bits[3:])
        text = (REPO.get((repo, sha)) or {}).get(path)
        return R(200, text.encode(), {}, url) if text is not None else \
            R(404, b"", {}, url)
    if url in DOCS:
        return R(200, DOCS[url].encode(), {"content-type": "text/markdown"},
                 url)
    return R(404, b"", {}, url)


package_net.TRANSPORT = fake_transport


def fake_fetch_source(url, beat=None):
    r = fake_transport(url, {"User-Agent": package_net.UA}, None)
    if r.status != 200:
        raise skill_pipeline.Refused(f"GET {url} answered HTTP {r.status}")
    return r.body, {"url": url, "final_url": url, "status": 200,
                    "content_type": "text/markdown", "charset": "utf-8",
                    "fetched_at": time.time(),
                    "robots": {"allowed": True, "why": "test"}}


skill_pipeline.fetch_source = fake_fetch_source

IDLE = {"on": True, "why": "test: idle", "skills": False}


def fake_idle(job_id=None):
    j = jobs.get(job_id) if job_id else None
    if j and str(j.get("queue") or "").startswith("skill."):
        # The skill model stages need a model: they stay deferred (the
        # tests end each version by hand, finish_skills).
        if not IDLE["skills"]:
            return {"idle": False, "why": "test: no model for skill stages",
                    "until": time.time() + 3600, "checks": {}}
    if IDLE["on"]:
        return {"idle": True, "why": IDLE["why"], "until": None, "checks": {}}
    return {"idle": False, "why": IDLE["why"], "until": time.time() + 3600,
            "checks": {}}


REAL_IDLE = idle.stack_idle
idle.stack_idle = fake_idle

INDEXED: list[tuple] = []


def fake_index_package(name, version, embed=False, db=None, src=None,
                       timeout=None, log=None, from_registry=None):
    INDEXED.append((name, version, embed, from_registry, bool(src)))
    return {"ok": True, "installed": True, "chunks": 7, "defs": 3,
            "code_chunks": 7, "zero_vectors": 0, "norm_ok": True,
            "files_selected": 2, "bytes_selected": 100}


deps.index_package = fake_index_package


# A fake registry (item F is mcp/package_registry.py, tested on its own in
# mcp/test_package_registry.py): here the floor's verdict is a switch.
try:
    import package_registry as _REAL_REGISTRY
except ImportError:
    _REAL_REGISTRY = None


class FakeRegistry(types.ModuleType):
    def __getattr__(self, name):
        # Everything the floor switch does not replace is the real module's
        # (skill_classify and skill_packages read it for the seed terms).
        if _REAL_REGISTRY is None:
            raise AttributeError(name)
        return getattr(_REAL_REGISTRY, name)

    def __init__(self):
        super().__init__("package_registry")
        self.FLOOR = {"passed": True, "rows": 74, "lost_tp": [],
                      "gained_fp": []}
        self.held = None
        self.entries = {}
        self.promoted = []
        self.next = None

    def entry(self, name):
        return self.entries.get(name)

    def load(self):
        return dict(self.entries)

    def held_manifest(self):
        return self.held

    def freeze_held(self):
        if self.held is None:
            self.held = {"dirs": [], "packages": {}}
        return self.held

    def add_package(self, name, *, version, aliases=(), built_on=(),
                    built_on_from="", onboarding=None, licence=None,
                    label=None):
        return {"name": name, "versions": [version], "aliases":
                list(aliases), "onboarding": onboarding, "built_on":
                list(built_on)}

    def build_candidate(self, add_dirs=(), remove_dirs=(), entries=()):
        dirs = sorted((set((self.held or {}).get("dirs") or [])
                       | set(add_dirs)) - set(remove_dirs))
        self.next = {"held": dirs, "entries": list(entries),
                     "signature": hashlib.sha256(json.dumps(
                         dirs).encode()).hexdigest()[:16],
                     "diff": {"per_package": {e["name"]: {"unique": 2}
                                              for e in entries},
                              "lost": {"koota": ["useQuery"]}
                              if not self.FLOOR["passed"] else {}}}
        return self.next

    def floor(self, cand):
        return dict(self.FLOOR)

    def promote(self, cand, *, by, forced=False, reason=""):
        if not self.FLOOR["passed"] and not forced:
            raise ValueError("floor failed")
        self.held = {"dirs": cand["held"], "packages": {}}
        for e in cand.get("entries") or []:
            have = self.entries.get(e["name"]) or {}
            e = dict(e, versions=sorted(set(have.get("versions") or [])
                                        | set(e.get("versions") or [])))
            self.entries[e["name"]] = e
        self.promoted.append((by, forced, cand["held"]))
        return self.held

    def next_candidate(self):
        return self.next


REG = FakeRegistry()
sys.modules["package_registry"] = REG


# Items H, I, K are their own modules with their own suites
# (mcp/test_example_knn.py, mcp/test_package_eval.py); here, recorders.
class _Rec(types.ModuleType):
    pass


EX = _Rec("package_examples")
EX.EXAMPLE_DIRS = ("examples",)
EX.calls = []
EX.collect = lambda did, pkg, t, linked_dirs=(), beat=None: (
    EX.calls.append((did, pkg["name"], tuple(linked_dirs or ())))
    or {"example_groups": 1, "example_files": 1, "example_dropped": 0})
KNN = _Rec("example_knn")
KNN.QUEUE = "package.example_knn_index"
KNN.FRESH = {"fresh": True, "why": "test"}
KNN.build = lambda beat=None: {"k": 1, "groups": 1, "signature": "x"}
KNN.index_state = lambda: dict(KNN.FRESH)
KNN.schedule = lambda: None
KNN.handle_build = lambda job, ctx=None: {"built": True}
EV = _Rec("package_eval")
EV.run = lambda did, pkg, beat=None: {"key": "k1", "in_sample": False,
                                      "summary": {"n": 1}}
sys.modules.update(package_examples=EX, example_knn=KNN, package_eval=EV)

import onboarding  # noqa: E402
import skill_match  # noqa: E402
import worker  # noqa: E402

skill_match.index_state = lambda pool=None: {"fresh": True, "why": "test"}


def drain() -> int:
    """Run every claimable job now (the real worker, --once)."""
    return worker.run(once=True)


def ds_of(did):
    return datasets.get(did)


def stage(did):
    return ds_of(did)["stage"]


def finish_skills(did, how="arm"):
    """The skill model stages need a model: end each onboarding skill's
    running version the way the pipeline would (armed / failed)."""
    for s in onboarding.skills_of(did):
        ver = skills.version(s["id"], s["latest_version"]) or {}
        if ver.get("state") != "running":
            continue
        for j in jobs.listing(dataset=f"skill:{s['id']}", state="queued"):
            con = jobs._db()
            con.execute("UPDATE jobs SET state='cancelled' WHERE id=?",
                        (j["id"],))
            con.close()
        if how == "arm":
            skills.update_version(s["id"], ver["version"], text=(
                "---\nname: x\ndescription: y\n---\n# X\n\n## Guidance\n\n"
                "- DO: z"), stage="arm")
            skills.arm(s["id"], ver["version"])
        else:
            skills.fail(s["id"], ver["version"], "test: stopped")


# ================================================================ 1. queue ==
def test_queue():
    jid = jobs.add("test.defer", {}, lane="cpu")
    j = jobs.claim("cpu", "t:1")
    check(j and j["id"] == jid and j["attempts"] == 1, "[queue] claimed once")
    jobs.defer(jid, time.time() + 3600, "test")
    row = jobs.get(jid)
    check(row["state"] == "queued" and row["attempts"] == 0
          and row["not_before"] > time.time()
          and row["progress"].startswith("waiting"),
          "[queue] defer: back to queued, the attempt given back, "
          "not_before set, the reason in progress", row)
    check(jobs.claim("cpu", "t:1") is None and jobs.claimable("cpu") == 0,
          "[queue] a deferred job is not claimable before not_before")
    con = jobs._db()
    con.execute("UPDATE jobs SET not_before=? WHERE id=?",
                (time.time() - 1, jid))
    con.close()
    j = jobs.claim("cpu", "t:1")
    check(j and j["id"] == jid and j["attempts"] == 1,
          "[queue] claimable again once not_before passed; attempts still 1")
    jobs.finish(jid, {})
    try:
        jobs.defer(jid, time.time(), "x")
        check(False, "[queue] deferring a job that is not running raises")
    except ValueError:
        check(True, "[queue] deferring a job that is not running raises")


# ================================================================= 2. idle ==
def test_idle():
    import corpus
    import skill_learn
    import sqlite3
    idle.stack_idle = REAL_IDLE
    try:
        p = os.path.abspath(corpus.CORPUS_DB)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        con = sqlite3.connect(p)
        con.execute("CREATE TABLE IF NOT EXISTS events(ts REAL, kind TEXT)")
        now = time.time()
        con.execute("DELETE FROM events")
        con.execute("INSERT INTO events VALUES(?, 'turn')", (now - 60,))
        con.commit()
        con.close()
        st = idle.stack_idle()
        want = now - 60 + skill_learn.IDLE_MINUTES * 60
        check(not st["idle"] and abs(st["until"] - want) < 2
              and "skill_learn.IDLE_MINUTES" in st["why"],
              "[idle] a request a minute ago: not idle, until = the request "
              "+ the learner's IDLE_MINUTES (the number's source named)", st)
        check("operator, 2026-09-27" in st["checks"]["idle_minutes_source"],
              "[idle] the value carries its source (the operator's keep)",
              st["checks"])
        con = sqlite3.connect(p)
        con.execute("UPDATE events SET ts=?", (now - 3600 * 5,))
        con.commit()
        con.close()
        import gpu_room
        real_running = gpu_room.running
        gpu_room.running = lambda up: None
        st = idle.stack_idle()
        check(not st["idle"] and "running" in st["why"],
              "[idle] llama-swap /running unreadable: not idle", st)
        gpu_room.running = lambda up: [{"model": "embeddings"}]
        st = idle.stack_idle()
        check(st["idle"] and "not loaded" in st["why"],
              "[idle] quiet, the main model not loaded (its /slots is never "
              "asked: that would load it): idle", st)
        real_slots = idle.slots_processing
        idle.slots_processing = lambda: ([2], "read")
        st = idle.stack_idle()
        check(not st["idle"] and "generating" in st["why"]
              and st["until"] > time.time(),
              "[idle] a slot generating: not idle, look again next tick", st)
        idle.slots_processing = real_slots
        jid = jobs.add("test.gpu", {}, lane="gpu")
        jobs.claim("gpu", "t:9")
        st = idle.stack_idle("other")
        check(not st["idle"] and "gpu-lane" in st["why"],
              "[idle] another gpu job running: not idle", st)
        jobs.finish(jid, {})
        gpu_room.running = real_running
    finally:
        idle.stack_idle = fake_idle
    # run_one defers an idle-gated job before its handler.
    ran = []
    worker.HANDLERS["test.idlejob"] = lambda job, ctx: ran.append(1) or {}
    IDLE.update(on=False, why="test: a request 2 min ago")
    jid = jobs.add("test.idlejob", {"idle": True}, lane="gpu")
    j = jobs.claim("gpu", "t:2")
    got = worker.run_one(j)
    row = jobs.get(jid)
    check(got == "queued" and not ran and row["attempts"] == 0
          and "idle stack" in (row["progress"] or ""),
          "[idle] run_one: not idle -> deferred before the handler, no "
          "attempt spent, the reason shown", row)
    n = drain()
    check(jobs.get(jid)["state"] == "queued",
          "[idle] worker --once returns, leaving the deferred job queued "
          f"(ran {n})")
    IDLE.update(on=True, why="test: idle")
    con = jobs._db()
    con.execute("UPDATE jobs SET not_before=NULL WHERE id=?", (jid,))
    con.close()
    drain()
    check(jobs.get(jid)["state"] == "done" and ran,
          "[idle] idle again: the same row runs and finishes")


# ========================================================== 3. rate limit ==
def test_rate_limit():
    RATE.update(on=True, reset=int(time.time()) + 900)
    worker.HANDLERS["test.ratelimited"] = onboarding._deferred_on_rate_limit(
        lambda job, ctx: package_net.gh("repos/acme/widget"))
    jid = jobs.add("test.ratelimited", {}, lane="net")
    got = worker.run_one(jobs.claim("net", "t:3"))
    row = jobs.get(jid)
    RATE.update(on=False)
    check(got == "queued" and abs(row["not_before"] - RATE["reset"]) < 1
          and row["attempts"] == 0,
          "[rate] GitHub's X-RateLimit-Reset defers the job exactly to the "
          "server's reset, no attempt spent", row)
    con = jobs._db()
    con.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (jid,))
    con.close()


# ======================================================= 4. kind package ==
def test_kind():
    check(datasets.stages_of({"kind": "package"}) == (
        "submitted", "resolve", "clarify", "index", "vocab", "examples",
        "knn", "sources", "skills", "retire", "rebuild", "evaluate",
        "complete"), "[kind] the stage order of the design")
    miss = [m["field"] for m in datasets.missing({"kind": "package"})]
    check(miss == ["source_url"],
          "[kind] a package asks only for the locator: never for a licence",
          miss)
    check(datasets.next_stage({"kind": "package", "stage": "vocab"})
          == "examples" and datasets.next_stage(
              {"kind": "recipes", "stage": "extract"}) == "index",
          "[kind] next_stage per kind; recipes unchanged")


# ============================================================= 5. resolve ==
def test_resolve_rules():
    import package_resolve as R
    c = R.classify_link("https://www.npmjs.com/package/@acme/gizmo/v/0.1.0")
    check(c["kind"] == "npm" and c["name"] == "@acme/gizmo"
          and c["version"] == "0.1.0", "[resolve] a scoped npm link with a "
          "version", c)
    c = R.classify_link("https://github.com/acme/widget/tree/main/skills/w")
    check(c["kind"] == "github_path" and c["is_dir"] and c["path"] ==
          "skills/w", "[resolve] a GitHub folder is a source", c)
    c = R.classify_link("https://github.com/acme/widget/releases/tag/v1.3.0")
    check(c["kind"] == "github_repo" and c["ref"] == "v1.3.0",
          "[resolve] a release tag is the repo at that ref", c)
    ch = R.choose_version(["1.0.0", "2.0.0-alpha.5", "2.0.0-alpha.12",
                           "1.1.0"], latest="1.1.0", want=None,
                          want_rule=None, major=2)
    check(ch["version"] == "2.0.0-alpha.12" and ch["rule"] == "major",
          "[resolve] a stated major with no stable release: the newest "
          "prerelease (numeric identifiers compared numerically)", ch)
    ch = R.choose_version(["1.0.0", "1.1.0", "1.2.0-rc.1"], latest="1.1.0",
                          want=None, want_rule=None, major=1)
    check(ch["version"] == "1.1.0", "[resolve] a stable release wins over "
          "a newer prerelease of the major", ch)
    check(R.major_after("build it with r3f v10 and koota", ["r3f"]) == 10,
          "[resolve] a major stated after the name or an alias")
    rr = R.replaces_rule("w", "1.3.0", ["1.2.0"])
    check(rr["mode"] == "replace" and rr["old"] == ["1.2.0"],
          "[resolve] the same major replaces (operator decision 2)", rr)
    rr = R.replaces_rule("w", "2.0.0", ["1.2.0"])
    check(rr["mode"] == "alongside" and "Right Family" in rr.get("risk", ""),
          "[resolve] a new major sits alongside, the risk recorded", rr)


def test_resolve_stage():
    GETS.clear()
    ds = onboarding.submit("Add the widget package "
                           "https://www.npmjs.com/package/acme-widget/v/1.2.0",
                           aliases=["widgets"], author="operator:test")
    did = ds["id"]
    check(ds["kind"] == "package" and ds["stage"] == "resolve"
          and ds["source_url"].endswith("/v/1.2.0"),
          "[resolve] submit: a package dataset at resolve, the first link "
          "its locator", ds)
    js = jobs.listing(dataset=did)
    check([j["queue"] for j in js] == ["package.resolve"],
          "[resolve] submit enqueues package.resolve only (no fetch, no "
          "assist)", [j["queue"] for j in js])
    drain()
    res = onboarding.resolution(did)
    pkg = res.get("package") or {}
    check(pkg.get("name") == "acme-widget" and pkg.get("version") == "1.2.0"
          and pkg.get("version_rule") == "link"
          and pkg.get("commit") == C1 and pkg.get("commit_rule") == "gitHead",
          "[resolve] the link's version; the commit npm recorded (gitHead)",
          {k: pkg.get(k) for k in ("name", "version", "version_rule",
                                   "commit", "commit_rule")})
    q = (onboarding.read_json(did, "licence.json") or {}).get("quotes") or []
    check(q and q[0]["spdx"] == "MIT" and q[0]["kind"] == "file"
          and any(x["kind"] == "manifest" and '"license": "MIT"' in
                  x["quote"] for x in q),
          "[resolve] the licence from verbatim quotes: the LICENSE file and "
          "the manifest line at the commit", q)
    d = ds_of(did)
    fs = {f["field"]: f for f in datasets.field_states(d)}
    check(d["licence"] == "MIT" and fs["licence"]["state"] == "evidence",
          "[resolve] the dataset's licence is evidence, with its quote",
          fs["licence"])
    check(d["stage"] == "index" or d["stage"] in ("vocab", "examples",
                                                  "knn", "sources",
                                                  "skills"),
          "[clarify] a quoted licence passes clarify with no human",
          d["stage"])
    check(all(u.startswith("https://") for u in GETS),
          "[resolve] GET only, https only", GETS[:3])
    # The first resolution pins: a newer `latest` must not move it.
    jid = jobs.add("package.resolve", {"dataset": did}, lane="net",
                   dataset=did, stage="resolve")
    worker.run_one(jobs.claim("net", "t:4"))
    check((jobs.get(jid)["result"] or {}).get("kept")
          and onboarding.resolution(did)["package"]["version"] == "1.2.0",
          "[resolve] a re-run keeps the first resolution",
          jobs.get(jid)["result"])
    return did


def test_resolve_group_and_rules():
    ds = onboarding.submit("widget v2 and the gizmo "
                           "https://www.npmjs.com/package/acme-widget "
                           "https://www.npmjs.com/package/@acme/gizmo "
                           "https://acme.dev/tips.md",
                           aliases=["widget=acme-widget"])
    did = ds["id"]
    IDLE.update(on=False, why="test: busy")      # stop at index
    drain()
    res = onboarding.resolution(did)
    check(res["package"]["version"] == "2.0.0"
          and res["package"]["version_rule"] == "major",
          "[resolve] 'widget v2': the newest stable of major 2 (not the "
          "2.1.0 beta)", res["package"])
    sibs = res.get("siblings") or []
    check(len(sibs) == 1 and sibs[0]["package"] == "@acme/gizmo@0.1.0",
          "[resolve] a second package: one sibling onboarding", sibs)
    sres = onboarding.resolution(sibs[0]["id"]) if sibs else {}
    check(sres.get("group") == did and sres.get("package", {}).get("name")
          == "@acme/gizmo" and res.get("group") == did,
          "[resolve] the sibling shares the group and resolves from the "
          "record (no second lookup)", {k: sres.get(k) for k in
                                        ("group", "how")})
    check(any(s.get("url") == "https://acme.dev/tips.md" and
              s.get("attach_to") == "acme-widget" for s in res["sources"]),
          "[resolve] a docs page attaches to the first package",
          res["sources"])
    check(stage(did) == "index" and jobs.listing(dataset=did,
                                                 state="queued"),
          "[index] a busy stack: the gpu stage waits (queued, deferred)",
          stage(did))
    q = [j for j in jobs.listing(dataset=did) if j["queue"] ==
         "package.index"][0]
    check(q["attempts"] == 0 and "idle stack" in (q["progress"] or ""),
          "[index] ... with no attempt spent and the reason on the row", q)
    IDLE.update(on=True)
    for j in jobs.listing(state="queued", limit=500):
        con = jobs._db()
        con.execute("UPDATE jobs SET not_before=NULL WHERE id=?", (j["id"],))
        con.close()
    return did, sibs[0]["id"] if sibs else None


def test_github_link_and_unresolved():
    import package_resolve as R
    res = R.resolve("the widget repo https://github.com/acme/widget and "
                    "https://www.npmjs.com/package/no-such-thing")
    names = [p["name"] for p in res["packages"]]
    check(names == ["acme-widget"] and res["packages"][0]["version"] ==
          "1.3.0" and res["packages"][0]["commit"] == C2,
          "[resolve] a GitHub repo link: package.json at the default "
          "branch's commit names it; its version is on the registry, so the "
          "registry artefact is indexed", res["packages"])
    check(any(u.get("name") == "no-such-thing" for u in res["unresolved"]),
          "[resolve] an unknown package is recorded unresolved, not guessed",
          res["unresolved"])


# ============================================================ 6. clarify ==
def test_clarify():
    ds = onboarding.submit("https://www.npmjs.com/package/agpl-thing")
    did = ds["id"]
    drain()
    d = ds_of(did)
    check(d["stage"] != "clarify" and d["licence"] == "AGPL-3.0"
          and onboarding.held_for_person(d) is None
          and any(w["kind"] == "licence" for w in datasets.warnings(d)),
          "[clarify] an AGPL licence found by evidence is recorded and shown "
          "as a note; nothing holds the onboarding for a person",
          {k: d.get(k) for k in ("stage", "licence")})
    # No licence at all: recorded as "not established", passes.
    PACKUMENTS["nolic"] = packument("nolic", {"1.0.0": None}, "1.0.0",
                                    "acme/nolic")
    PACKUMENTS["nolic"]["versions"]["1.0.0"].pop("license")
    ds = onboarding.submit("https://www.npmjs.com/package/nolic")
    did2 = ds["id"]
    IDLE.update(on=False, why="busy")
    drain()
    d = ds_of(did2)
    check(d["stage"] == "index" and d["licence"] == datasets.NOT_ESTABLISHED
          and not onboarding.blockers(dict(d, stage="clarify")),
          "[clarify] no quote: the licence is recorded as \"not "
          "established\" and the onboarding passes clarify with no human",
          {k: d.get(k) for k in ("stage", "licence")})
    IDLE.update(on=True)
    return did2


# ============================================================== 7. index ==
def test_fetch_verified():
    blob = BLOBS[("acme-widget", "1.2.0")]
    dest = os.path.join(_TMP, "fv", "w@1.2.0")
    got = deps.fetch_verified("acme-widget", "1.2.0", integrity=sri(b"x"),
                              dest=dest)
    check(not got["ok"] and "integrity" in got["error"]
          and not os.path.exists(dest),
          "[index] an integrity mismatch is refused, nothing unpacked", got)
    got = deps.fetch_verified("acme-widget", "1.2.0", integrity=sri(blob),
                              unpacked_size=10, dest=dest)
    check(not got["ok"] and "unpackedSize" in got["error"]
          and not os.path.exists(dest), "[index] more than the registry's "
          "own unpackedSize is refused", got)
    got = deps.fetch_verified("acme-widget", "1.2.0", integrity=sri(blob),
                              unpacked_size=10 ** 6, dest=dest)
    check(got["ok"] and got["licence_files"] == ["LICENSE"]
          and os.path.exists(os.path.join(dest, "LICENSE"))
          and os.path.exists(os.path.join(dest, "index.d.ts"))
          and not os.path.exists(os.path.join(dest, "bin", "tool.exe")),
          "[index] verified: LICENSE extracted beside the source; the "
          "extension filter holds", got)
    got2 = deps.fetch_verified("acme-widget", "1.2.0", integrity=sri(blob),
                               dest=dest)
    check(got2["ok"] and got2.get("reused"), "[index] a second run keeps "
          "the unpacked directory (verified again)", got2)


def test_index_stage(did):
    rec = onboarding.read_json(did, "index.json") or {}
    check((rec.get("fetch") or {}).get("ok") and INDEXED and
          INDEXED[0][:4] == ("acme-widget", "1.2.0", True, True),
          "[index] fetched and verified, indexed with embeddings, recorded "
          "as from the registry", {"rec": rec, "indexed": INDEXED[:1]})
    lic = onboarding.read_json(did, "licence.json") or {}
    check(any(q.get("kind") == "tarball" for q in lic.get("quotes") or [])
          and lic.get("agree"), "[index] the tarball's LICENSE is one more "
          "quote, and the quotes agree", lic)
    check(REG.held is not None, "[index] held.json is frozen before any "
          "fetch (a fetched package is not held until promoted)")


# ============================================================== 8. vocab ==
def test_vocab_promoted(did):
    v = onboarding.read_json(did, "vocab.json") or {}
    check(v.get("state") == "promoted" and REG.promoted and
          "acme-widget@1.2.0" in REG.promoted[-1][2],
          "[vocab] the floor passed: promoted", v.get("state"))
    check((v.get("entry") or {}).get("aliases") == ["widgets"],
          "[vocab] the registry entry carries the aliases the operator "
          "typed", v.get("entry"))


def test_vocab_held():
    REG.FLOOR = {"passed": False, "rows": 74,
                 "lost_tp": [{"id": "imp-query-nothing", "package": "koota"}],
                 "gained_fp": []}
    ds = onboarding.submit("https://www.npmjs.com/package/acme-widget/v/1.3.0")
    did = ds["id"]
    drain()
    d = ds_of(did)
    b = onboarding.blockers(d)
    check(d["stage"] == "vocab" and b and "HELD" in b[0]["what"]
          and "imp-query-nothing" in b[0]["what"]
          and "promote" in b[0]["remedy"],
          "[vocab] a failing floor HOLDS the vocabulary; the blocker names "
          "the rows and the remedy", b)
    onboarding.sweep()
    check(stage(did) == "vocab", "[vocab] the sweep does not pass a hold")
    onboarding.promote(did, author="operator:test")
    check(REG.promoted[-1][:2] == ("operator:test", True)
          and stage(did) != "vocab",
          "[vocab] the operator's force is recorded with the author; the "
          "sweep moves on", (REG.promoted[-1], stage(did)))
    REG.FLOOR = {"passed": True, "rows": 74, "lost_tp": [], "gained_fp": []}
    return did


# ============================================================ 9. sources ==
def test_sources(did):
    sk = onboarding.skills_of(did)
    urls = sorted(s["source_url"] for s in sk)
    base = f"https://raw.githubusercontent.com/acme/widget/{C1}/"
    check(base + "skills/widget/SKILL.md" in urls
          and base + "skills/widget/references/api.md" in urls
          and not any("README" in u for u in urls)
          and not any("widget.dev/guide" in u for u in urls),
          "[sources] tier 1: the SKILL.md and its references at the commit; "
          "no README lead and no llms.txt page beside a SKILL.md", urls)
    rec = onboarding.read_json(did, "sources.json") or {}
    check((rec.get("tier3") or {}).get("pages") and any(
        x.get("tier") == 3 for x in rec.get("skipped") or []),
          "[sources] the llms.txt pages are RECORDED (operator decision 3)",
          rec.get("skipped"))
    fr = [s for s in sk if s["source_url"].endswith("SKILL.md")][0]
    check(fr["source_kind"] == "frontier" and fr["meta"].get("package") ==
          "acme-widget" and fr["meta"].get("package_version") == "1.2.0"
          and fr["meta"].get("onboarding") == did
          and not fr.get("watch_seconds"),
          "[sources] the SKILL.md walks the frontier path with the "
          "onboarding keys; never watched (a commit-pinned URL)", fr["meta"])
    check(EX.calls and EX.calls[0][1] == "acme-widget",
          "[examples] the examples stage ran for the package", EX.calls)
    return sk


def test_sources_no_skill_md(gizmo):
    sk = onboarding.skills_of(gizmo)
    urls = sorted(s["source_url"] for s in sk)
    check(any(u.endswith("/README.md") for u in urls)
          and "https://gizmo.dev/guide.md" in urls,
          "[sources] no SKILL.md: the README yields the lead and the "
          "llms.txt pages are ingested", urls)
    lead = [s for s in sk if s["source_url"].endswith("/README.md")]
    check(lead and lead[0]["source_kind"] == "frontier"
          and lead[0]["meta"]["package"] == "@acme/gizmo",
          "[sources] the README lead is a frontier source declaring its "
          "package", lead[:1])


def test_skill_jobs_idle(sk):
    # screen runs (cpu, no model); screen_model is a gpu stage: idle flag.
    gpu = [j for s in sk for j in jobs.listing(dataset=f"skill:{s['id']}")
           if j["lane"] == "gpu"]
    check(gpu and all((j["payload"] or {}).get("idle") for j in gpu),
          "[skills] an onboarding skill's model stages carry the idle flag",
          [(j["queue"], j["payload"]) for j in gpu][:3])


# ============================================================ 10. the JOIN ==
def test_join(did):
    d = ds_of(did)
    check(d["stage"] == "skills", "[join] the onboarding waits at skills",
          d["stage"])
    b = onboarding.blockers(d)
    check(b and all("is at" in x["what"] for x in b),
          "[join] every skill still in the pipeline is a blocker", b[:2])
    s0 = onboarding.skills_of(did)[0]
    ver = skills.version(s0["id"], s0["latest_version"])
    jid = jobs.add("skill.screen_model", {"skill": s0["id"], "version":
                                          ver["version"],
                                          "stage": "screen_model"},
                   lane="gpu", dataset=f"skill:{s0['id']}",
                   stage="screen_model")
    con = jobs._db()
    con.execute("UPDATE jobs SET state='errored', error='boom' WHERE id=?",
                (jid,))
    con.close()
    finish_skills(did, "fail")
    b = onboarding.blockers(ds_of(did))
    check(len(b) == 1 and "errored" in b[0]["what"] and "rerun" in
          b[0]["remedy"], "[join] an errored skill job blocks with its "
          "remedy", b)
    con = jobs._db()
    con.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (jid,))
    con.close()
    onboarding.sweep()
    drain()
    onboarding.sweep()
    check(stage(did) == "complete",
          "[join] every skill stopped: retire, rebuild (fresh) and evaluate "
          "run, and the onboarding completes", stage(did))


# ============================================================ 11. REPLACE ==
def test_replace(first):
    # Arm v1.2.0's skills so the replacement can re-version them.
    old = {s["source_url"]: s for s in onboarding.skills_of(first)}
    ds = onboarding.submit("https://www.npmjs.com/package/acme-widget/v/1.3.0",
                           replaces=None)
    did = ds["id"]
    drain()
    res = onboarding.resolution(did)
    check((res.get("replaces") or {}).get("mode") == "replace",
          "[replace] 1.3.0 beside held 1.2.0: the same major replaces",
          res.get("replaces"))
    made = (onboarding.read_json(did, "sources.json") or {}).get("made") or {}
    rp = [x["id"] for x in made.get("repointed") or []]
    old_md = [s for u, s in old.items() if u.endswith("SKILL.md")][0]
    check(old_md["id"] in rp and skills.get(old_md["id"])["source_url"]
          .startswith(f"https://raw.githubusercontent.com/acme/widget/{C2}/"),
          "[replace] the SKILL.md skill gets a NEW VERSION at the new commit "
          "(repointed)", made)
    s = skills.get(old_md["id"])
    check(s["meta"]["onboarding"] == did and s["meta"]["package_version"] ==
          "1.3.0" and s["meta"].get("repointed"),
          "[replace] it now belongs to the new onboarding; the move is "
          "recorded", s["meta"])
    old_ref = [s for u, s in old.items() if u.endswith("references/old.md")]
    ret = [x["id"] for x in made.get("to_retire") or []]
    check(old_ref and old_ref[0]["id"] in ret and skills.get(
        old_ref[0]["id"])["status"] != "archived",
          "[replace] a source whose path is gone is marked to_retire, not "
          "archived yet", ret)
    finish_skills(did, "arm")
    onboarding.sweep()
    drain()
    onboarding.sweep()
    rr = onboarding.read_json(did, "retire.json") or {}
    check(old_ref and skills.get(old_ref[0]["id"])["status"] == "archived"
          and any(x["id"] == old_ref[0]["id"] for x in rr.get("archived")),
          "[replace] archived at the retire stage, after the JOIN", rr)
    return did


def test_decompose_to_retire():
    """handle_decompose: an onboarding source's vanished child is recorded
    to_retire, not archived at decompose time."""
    import skill_prompts
    src = skills.create(text=SKILL_MD, frontier=True,
                        meta={"onboarding": "dX", "package": "acme-widget",
                              "package_version": "1.3.0"},
                        enqueue_first=False)
    sid = src["id"]
    kid = skills.create_child(sid, 1, parsed={"name": "gone-child",
                                              "items": [{"form": "do",
                                                         "text": "x"}]},
                              section="Gone", enqueue_first=False)
    check(kid["meta"].get("onboarding") == "dX"
          and kid["meta"].get("package_version") == "1.3.0",
          "[decompose] create_child copies the onboarding keys", kid["meta"])
    v = skills.new_version(sid, "frontier_text")
    skills.store_source(sid, v, SKILL_MD.encode(), {"kind": "frontier"})
    real = skill_pipeline.ask_model
    skill_pipeline.ask_model = lambda *a, **k: ""
    real_parse = skill_prompts.parse_decompose
    skill_prompts.parse_decompose = lambda reply: [
        {"name": "new-child", "items": [{"form": "do", "text": "y"}],
         "section": "New", "lead": ""}]
    try:
        skills.update_version(sid, v, stage="decompose")
        out = skill_pipeline.handle_decompose(
            {"id": "t", "payload": {"skill": sid, "version": v,
                                    "stage": "decompose"}},
            skill_pipeline._InlineCtx(), inline=False)
    finally:
        skill_pipeline.ask_model = real
        skill_prompts.parse_decompose = real_parse
    check(kid["id"] in (out.get("to_retire") or [])
          and skills.get(kid["id"])["status"] != "archived"
          and any(x["id"] == kid["id"] for x in
                  skills.get(sid)["meta"].get("to_retire") or []),
          "[decompose] with meta.onboarding the gone child is to_retire on "
          "the source, still serving", out)


def test_alongside():
    import package_resolve as R
    rr = R.replaces_rule("acme-widget", "2.0.0", ["1.2.0", "1.3.0"])
    check(rr["mode"] == "alongside" and rr["old"] == ["1.2.0", "1.3.0"],
          "[alongside] 2.0.0 beside 1.x: alongside, the old kept", rr)
    rr = R.replaces_rule("acme-widget", "1.3.0", ["1.2.0"], "alongside")
    check(rr["mode"] == "alongside" and "operator" in rr["why"],
          "[alongside] the form's explicit choice wins and says so", rr)


def test_index_rebuild_is_idle_gated():
    real = skill_match.index_state
    skill_match.index_state = lambda pool=None: {"fresh": False,
                                                 "why": "test: stale"}
    try:
        jid = skill_match.schedule()
    finally:
        skill_match.index_state = real
    j = jobs.get(jid) if jid else {}
    check(j and (j.get("payload") or {}).get("idle") is True
          and j.get("lane") == "gpu_a4000",   # the embedder: the A4000's gpu scope (jobs.GPU_SCOPES)
          "[idle] the skill document index rebuild is idle-gated (operator "
          "decision 5)", j)
    if jid:
        con = jobs._db()
        con.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (jid,))
        con.close()


def test_no_package():
    """A prompt whose links resolve to no package: its sources run as
    ordinary skill submissions; the package stages record why they skip."""
    ds = onboarding.submit("a skill from these tips https://acme.dev/tips.md")
    did = ds["id"]
    drain()
    res = onboarding.resolution(did)
    check(res.get("package") is None and len(res.get("sources") or []) == 1,
          "[no package] resolved to no package; the page is a source", res)
    idx = onboarding.read_json(did, "index.json") or {}
    check("no package" in str(idx.get("skipped")),
          "[no package] index records why it is skipped (not errored)", idx)
    sk = onboarding.skills_of(did)
    check(len(sk) == 1 and sk[0]["source_url"] == "https://acme.dev/tips.md"
          and sk[0]["source_kind"] == "url" and "package" not in
          sk[0]["meta"], "[no package] one ordinary skill submission (distil "
          "path), no package keys", [s["meta"] for s in sk])
    check(stage(did) == "skills", "[no package] it waits at the skills JOIN "
          "like any onboarding", stage(did))
    finish_skills(did, "fail")


# ====================================================== 12. killed worker ==
def test_killed_worker():
    drain()
    ds = onboarding.submit("https://www.npmjs.com/package/@acme/gizmo")
    did = ds["id"]
    j = jobs.claim("net", "somehost:1:net0")
    check(j and j["queue"] == "package.resolve", "[killed] a stage job "
          "claimed by a worker that then dies", j)
    con = jobs._db()
    con.execute("UPDATE jobs SET heartbeat=? WHERE id=?",
                (time.time() - jobs.STALE_SECONDS - 5, j["id"]))
    con.close()
    n = jobs.reclaim()
    row = jobs.get(j["id"])
    check(n >= 1 and row["state"] == "queued" and row["attempts"] == 1,
          "[killed] reclaim returns it with its attempt count", row)
    IDLE.update(on=False, why="busy")
    drain()
    IDLE.update(on=True)
    check(stage(did) in ("index", "clarify") and onboarding.resolution(did)
          .get("package"), "[killed] the stage re-runs and the dataset "
          "moves on", stage(did))


# ================================================================ 13. API ==
def test_api(first):
    import dash_skills
    code, _ct, body = dash_skills.handle_post(
        "/dash/api/skill", {"prompt": "add gizmo "
                            "https://www.npmjs.com/package/@acme/gizmo",
                            "aliases": "gizmos, gz"}, who="abc123")
    got = json.loads(body)
    check(code == 200 and got["ok"] and got["onboarding"]["stage"] ==
          "resolve", "[api] POST /dash/api/skill with a prompt: an "
          "onboarding", got)
    nid = got["onboarding"]["id"]
    notes = datasets.package_notes(ds_of(nid))
    check(notes.get("aliases") == ["gizmos", "gz"] and notes.get("author")
          == "operator:abc123", "[api] the aliases as typed; the author "
          "recorded", notes)
    code, _ct, body = dash_skills.handle_post(
        "/dash/api/skill", {"prompt": "no links at all"}, who="x")
    check(code == 400, "[api] a prompt with no link is refused", body[:200])
    code, _ct, body = dash_skills.handle_get(
        "/dash/api/skill-factory/onboarding")
    rows = json.loads(body)["onboardings"]
    check(code == 200 and any(r["id"] == first and r["package"] ==
                              "acme-widget" for r in rows),
          "[api] the listing", rows[:2])
    code, _ct, body = dash_skills.handle_get(
        f"/dash/api/skill-factory/onboarding/{first}")
    d = json.loads(body)["onboarding"]
    check(code == 200 and d["resolution"]["package"]["name"] ==
          "acme-widget" and d["skills"] and d["stages"][0] == "submitted"
          and "job_rows" in d, "[api] the detail: resolution, skills, "
          "stages, job rows", list(d)[:12])
    blob = json.dumps(d)
    check(_TMP not in blob and _TMP.replace("\\", "/") not in blob
          and _TMP.replace("\\", "\\\\") not in blob,
          "[api] no filesystem path in the detail")
    code, _ct, body = dash_skills.handle_post(
        "/dash/api/skill-factory/onboarding/review",
        {"id": first, "stage": "sources", "note": "looks right"}, who="abc")
    got = json.loads(body)
    check(code == 200 and got["reviews"][-1]["note"] == "looks right"
          and got["reviews"][-1]["by"] == "operator:abc",
          "[api] a review note is stored after the fact", got)
    code, _ct, body = dash_skills.handle_post(
        "/dash/api/skill-factory/onboarding/promote", {"id": first}, who="a")
    check(code == 400 and "not held" in json.loads(body)["error"],
          "[api] promote refuses a vocabulary that is not held", body[:200])
    code, _ct, body = dash_skills.handle_get(
        "/dash/api/skill-factory/onboarding/nope")
    check(code == 404, "[api] an unknown onboarding is 404")
    code, _ct, body = dash_skills.handle_post(
        "/dash/api/skill-factory/onboarding/tier3", {"id": first}, who="a")
    got = json.loads(body)
    check(code == 200 and got.get("created"), "[api] tier3 ingests the "
          "recorded llms.txt pages on the operator's click", got)


def main() -> int:
    tests = [test_queue, test_idle, test_rate_limit, test_kind,
             test_resolve_rules]
    for t in tests:
        try:
            t()
        except Exception:                                        # noqa: BLE001
            check(False, f"{t.__name__} raised", traceback.format_exc())
    first = gizmo = None
    try:
        test_fetch_verified()
        first = test_resolve_stage()
        test_index_stage(first)
        test_vocab_promoted(first)
        sk = test_sources(first)
        test_skill_jobs_idle(sk)
        test_join(first)
    except Exception:                                            # noqa: BLE001
        check(False, "the first onboarding raised", traceback.format_exc())
    for t in (test_github_link_and_unresolved, test_clarify, test_alongside,
              test_no_package, test_index_rebuild_is_idle_gated,
              test_decompose_to_retire):
        try:
            t()
        except Exception:                                        # noqa: BLE001
            check(False, f"{t.__name__} raised", traceback.format_exc())
    try:
        _w2, gizmo = test_resolve_group_and_rules()
        drain()
        if gizmo:
            test_sources_no_skill_md(gizmo)
    except Exception:                                            # noqa: BLE001
        check(False, "the group raised", traceback.format_exc())
    try:
        if first:
            test_replace(first)
    except Exception:                                            # noqa: BLE001
        check(False, "replace raised", traceback.format_exc())
    for t in (test_vocab_held, test_killed_worker):
        try:
            t()
        except Exception:                                        # noqa: BLE001
            check(False, f"{t.__name__} raised", traceback.format_exc())
    try:
        if first:
            test_api(first)
    except Exception:                                            # noqa: BLE001
        check(False, "api raised", traceback.format_exc())
    print(f"\n  {N[0] - len(FAILS)}/{N[0]} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
