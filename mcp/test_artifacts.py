#!/usr/bin/env python
"""scripts/verify_artifacts.py and scripts/fetch_models.py, offline, on temp files.

    python mcp/test_artifacts.py      -> "N/M checks passed"

WHAT THIS GATES. The deploy check calls verify(config_path) to prove that
what llama-swap loads is what models/manifest.yaml records (docs/MODELS.md).
Each failure mode it claims to catch is produced here on purpose -- an
unrecorded file, a wrong size, a wrong hash, a missing file, a stale hash
cache, a malformed entry, a moved E1 head, a drifted lock, a dirty harness
tree -- and must be reported, or the check that it catches it could not fail.
A clean fixture must report nothing. The download path is exercised against
a local HTTP server: a good file lands, a bad one stays `.part`, an existing
different file is never overwritten.

Also, on the REAL tracked files (no hashing): the manifest has no schema
problems, and every model file config.example.yaml passes to a server is
recorded in it -- so the public manifest cannot drift from the public config.
"""
from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import yaml  # noqa: E402

import fetch_models as fm  # noqa: E402
import verify_artifacts as va  # noqa: E402

_results: list[tuple[bool, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
    return bool(ok)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Fixture:
    """A models dir, a config.yaml that loads two files (one via a macro chain,
    one sd-server style), a commented-out path, and a manifest recording them."""

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="yamadori-artifacts-")
        self.models = os.path.join(self.dir, "models").replace("\\", "/")
        os.makedirs(os.path.join(self.models, "img"))
        self.a = b"main model bytes" * 100
        self.b = b"vae bytes" * 50
        self.write("main.gguf", self.a)
        self.write("img/vae.safetensors", self.b)
        self.config = os.path.join(self.dir, "config.yaml")
        self.manifest = os.path.join(self.dir, "manifest.yaml")
        self.local = os.path.join(self.dir, "manifest.local.yaml")
        self.cache = os.path.join(self.dir, "hashes.json")
        self.write_config()
        self.write_manifest(self.artifacts())

    def write(self, rel: str, data: bytes) -> str:
        p = os.path.join(self.models, rel)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def write_config(self, extra: str = "") -> None:
        cfg = {
            "macros": {"server": '"C:/x/llama-server.exe"', "models": self.models,
                       "root": "${models}"},
            "models": {
                "main": {"cmd": "${server}\n--port ${PORT}\n"
                                "# -m ${models}/commented-out.gguf\n"
                                "-m ${root}/main.gguf\n-c 4096\n"},
                "img": {"cmd": "C:/x/sd-server.exe --listen-port ${PORT}\n"
                               "--vae ${models}/img/vae.safetensors --steps 4\n" + extra},
            },
        }
        with open(self.config, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

    def artifacts(self) -> list[dict]:
        prov = {"status": "hash_only"}
        return [
            {"id": "main", "role": "r", "status": "in_service",
             "path": "${models}/main.gguf", "size": len(self.a), "sha256": sha(self.a),
             "provenance": prov},
            {"id": "vae", "role": "r", "status": "in_service",
             "path": "${models}/img/vae.safetensors", "size": len(self.b),
             "sha256": sha(self.b), "provenance": prov},
        ]

    def write_manifest(self, arts: list[dict], runtimes: list | None = None,
                       path: str | None = None) -> None:
        with open(path or self.manifest, "w", encoding="utf-8") as f:
            yaml.safe_dump({"schema": 1, "artifacts": arts,
                            "runtimes": runtimes or []}, f)

    def verify(self, **kw) -> list:
        kw.setdefault("runtimes", False)
        kw.setdefault("hash_cache", None)
        return va.verify(self.config, self.manifest, self.local, **kw)

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def errors(probs: list) -> list:
    return [p for p in probs if p.severity == "error"]


def test_config_parse() -> None:
    print("config parsing")
    fx = Fixture()
    try:
        refs = va.config_files(fx.config)
        paths = sorted(p for _, _, p in refs)
        check(len(refs) == 2, "two model-file arguments found", str(refs))
        check(all("commented-out" not in p for p in paths), "a commented-out -m line is ignored")
        check(any(p.endswith("/main.gguf") and "${" not in p for p in paths),
              "a macro chain (${root} -> ${models}) is expanded")
        check(all(not p.endswith(".exe") for p in paths), "the binary is never taken as a model file")
        check({f for _, f, _ in refs} == {"-m", "--vae"}, "flags recorded")
    finally:
        fx.close()


def test_verify_files() -> None:
    print("verify: model files")
    fx = Fixture()
    try:
        check(fx.verify() == [], "a clean fixture reports nothing", str(fx.verify()))

        # unrecorded file
        fx.write("img/llm.gguf", b"encoder")
        fx.write_config("--llm ${models}/img/llm.gguf\n")
        p = errors(fx.verify())
        check(len(p) == 1 and "no manifest entry" in p[0].message and p[0].remedy,
              "an unrecorded file is an error with a remedy", str(p))

        # recorded in the LOCAL manifest -> fine
        fx.write_manifest([{"id": "llm", "role": "r", "status": "in_service",
                            "path": "${models}/img/llm.gguf", "size": 7,
                            "sha256": sha(b"encoder"), "provenance": {"status": "hash_only"}}],
                          path=fx.local)
        check(fx.verify() == [], "an entry in manifest.local.yaml records it", str(fx.verify()))
        os.remove(fx.local)
        fx.write_config()

        # wrong bytes, same size
        fx.write("main.gguf", b"X" * len(fx.a))
        p = errors(fx.verify())
        check(len(p) == 1 and "sha256" in p[0].message and p[0].artifact == "main",
              "same size, different bytes is caught by sha256", str(p))

        # wrong size
        fx.write("main.gguf", fx.a + b"!")
        p = errors(fx.verify())
        check(len(p) == 1 and "bytes; the manifest records" in p[0].message,
              "a size change is caught before hashing", str(p))

        # missing
        os.remove(os.path.join(fx.models, "main.gguf"))
        p = errors(fx.verify())
        check(len(p) == 1 and "does not exist" in p[0].message, "a missing file is an error", str(p))
        fx.write("main.gguf", fx.a)

        # in_service but not loaded -> warn only
        arts = fx.artifacts() + [{"id": "ghost", "role": "r", "status": "in_service",
                                  "path": "${models}/ghost.gguf", "size": 1, "sha256": "0" * 64,
                                  "provenance": {"status": "hash_only"}}]
        fx.write_manifest(arts)
        p = fx.verify()
        check(len(p) == 1 and p[0].severity == "warn" and p[0].artifact == "ghost",
              "an in-service entry config.yaml does not load is a warning", str(p))
        arts[-1]["status"] = "retired"
        fx.write_manifest(arts)
        check(fx.verify() == [], "a retired entry nothing loads is fine")
        fx.write_manifest(fx.artifacts())
    finally:
        fx.close()


def test_hash_cache() -> None:
    print("verify: hash cache")
    fx = Fixture()
    try:
        check(fx.verify(hash_cache=fx.cache) == [], "first run hashes and passes")
        cache = json.load(open(fx.cache, encoding="utf-8"))
        check(len(cache) == 2, "both files are cached", str(cache.keys()))
        # same size, same mtime, different bytes: the cache cannot see it...
        p = os.path.join(fx.models, "main.gguf")
        st = os.stat(p)
        with open(p, "wb") as f:
            f.write(b"Y" * len(fx.a))
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        check(fx.verify(hash_cache=fx.cache) == [],
              "the cache trusts (size, mtime) -- documented limit")
        # ...rehash can
        p2 = errors(fx.verify(hash_cache=fx.cache, rehash=True))
        check(len(p2) == 1 and p2[0].artifact == "main", "rehash=True ignores the cache", str(p2))
        # and an ordinary edit (new mtime) is caught through the cache
        time.sleep(0.01)
        with open(p, "wb") as f:
            f.write(b"Z" * len(fx.a))
        check(len(errors(fx.verify(hash_cache=fx.cache))) == 1, "a normal edit invalidates the cache")
    finally:
        fx.close()


def test_schema() -> None:
    print("verify: manifest schema")
    fx = Fixture()
    try:
        arts = fx.artifacts()
        arts[0]["provenance"] = {"status": "pinned", "repo": "a/b", "revision": "main",
                                 "filename": "x"}
        arts[1] = dict(arts[1], id="main")
        del arts[1]["role"]
        arts.append({"id": "z", "role": "r", "status": "live", "path": "${models}/z",
                     "size": 1, "sha256": "ABC", "provenance": {"status": "recipe"}})
        fx.write_manifest(arts)
        msgs = " | ".join(p.message for p in fx.verify())
        check("full 40-hex" in msgs, "a branch name is not a pinned revision", msgs)
        check("duplicate id" in msgs, "a duplicate id is caught", msgs)
        check("missing field `role`" in msgs, "a missing field is caught", msgs)
        check("status 'live'" in msgs, "an unknown status is caught", msgs)
        check("no `recipe`" in msgs, "a derived entry without a recipe is caught", msgs)
        check("64 lowercase hex" in msgs, "a malformed sha256 is caught", msgs)
    finally:
        fx.close()


def test_code_loaded() -> None:
    print("verify: artifacts loaded by code (E1 head)")
    fx = Fixture()
    try:
        heads = os.path.join(fx.dir, "e1", "route_in")
        os.makedirs(heads)
        raw = b"\x00\x01" * 64
        blob = {"artefact": "e1-head", "version": 1, "W_b64": base64.b64encode(raw).decode(),
                "W_sha256": sha(raw)}
        hp = os.path.join(heads, "v0001.json")
        data = json.dumps(blob).encode()
        open(hp, "wb").write(data)
        json.dump({"current": 1}, open(os.path.join(heads, "state.json"), "w"))
        head = {"id": "e1", "role": "r", "status": "in_service", "loaded_by": "code",
                "format": "e1-head", "severity": "warn",
                "path": hp.replace("\\", "/"), "size": len(data), "sha256": sha(data),
                "provenance": {"status": "recipe", "recipe": {"command": "x"}}}
        fx.write_manifest(fx.artifacts() + [head])
        check(fx.verify() == [], "a matching head passes", str(fx.verify()))
        json.dump({"current": 2}, open(os.path.join(heads, "state.json"), "w"))
        p = fx.verify()
        check(len(p) == 1 and p[0].severity == "warn" and "serves v2" in p[0].message,
              "a promoted version is a warning naming it", str(p))
        json.dump({"current": 1}, open(os.path.join(heads, "state.json"), "w"))
        bad = dict(blob, W_b64=base64.b64encode(b"\x09" * 128).decode())
        data2 = json.dumps(bad).encode()
        open(hp, "wb").write(data2)
        head.update(size=len(data2), sha256=sha(data2))
        fx.write_manifest(fx.artifacts() + [head])
        p = errors(fx.verify())
        check(len(p) == 1 and "W_sha256" in p[0].message,
              "weights that disagree with their own W_sha256 are an error", str(p))
        os.remove(hp)
        p = fx.verify()
        check(any("does not exist" in x.message for x in p), "a missing code artifact is reported", str(p))
    finally:
        fx.close()


def test_runtimes() -> None:
    print("verify: runtimes")
    fx = Fixture()
    try:
        lock = os.path.join(fx.dir, "req.lock.txt")
        rt = {"id": "py", "check": "lock", "severity": "error",
              "interpreter": sys.executable, "lock": lock, "freeze": "pip"}
        n = va.write_lock(lock, rt)
        check(n > 0, f"write_lock froze {n} packages")
        fx.write_manifest(fx.artifacts(), [rt])
        p = fx.verify(runtimes=True)
        check(p == [], "the interpreter matches the lock it just wrote", str(p))
        with open(lock, "a", encoding="utf-8") as f:
            f.write("not-a-real-package==0.0.1\n")
        p = errors(fx.verify(runtimes=True))
        check(len(p) == 1 and "-not-a-real-package==0.0.1" in p[0].message and "pip install -r" in p[0].remedy,
              "a package in the lock but not installed is an error with the restore command", str(p))
        text = open(lock, encoding="utf-8").read().replace(
            "# python: ", "# python: 2.7.0 was ", 1)
        open(lock, "w", encoding="utf-8").write(text)
        p = fx.verify(runtimes=True)
        check(any("python is" in x.message for x in p), "a Python version change is caught", str(p))
        hdr, reqs = va.read_lock(lock)
        check("#local: " not in "".join(reqs) and all(not r.startswith("#") for r in reqs),
              "#local lines are read back as requirements, not comments")

        # files + git checks
        f1 = os.path.join(fx.dir, "settings.template.yml")
        open(f1, "w").write("commit abc123\n")
        rt_files = {"id": "sx", "check": "files", "severity": "warn",
                    "files": [{"path": f1, "contains": "abc123", "sha256": va.sha256_file(f1)}]}
        repo = os.path.join(fx.dir, "harness")
        os.makedirs(repo)
        g = ["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(g + ["init", "-q"], check=True)
        open(os.path.join(repo, "a.txt"), "w", newline="\n").write("one\ntwo\n")
        subprocess.run(g + ["-c", "core.autocrlf=false", "add", "a.txt"], check=True)
        subprocess.run(g + ["commit", "-q", "-m", "x"], check=True)
        head = subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        rt_git = {"id": "hermes", "check": "git", "severity": "warn", "path": repo, "commit": head}
        fx.write_manifest(fx.artifacts(), [rt_files, rt_git])
        check(fx.verify(runtimes=True) == [], "files + git runtimes pass when recorded",
              str(fx.verify(runtimes=True)))
        open(os.path.join(repo, "a.txt"), "wb").write(b"one\r\ntwo\r\n")
        check(fx.verify(runtimes=True) == [], "line-ending-only changes are not drift")
        open(os.path.join(repo, "a.txt"), "w", newline="\n").write("one\nthree\n")
        open(f1, "w").write("commit zzz\n")
        p = fx.verify(runtimes=True)
        msgs = " | ".join(x.message for x in p)
        check("beyond line endings" in msgs, "a real edit in the harness tree is drift", msgs)
        check("does not contain 'abc123'" in msgs and "not the recorded bytes" in msgs,
              "a changed pinned file is drift", msgs)
        check(all(x.severity == "warn" for x in p), "runtime drift here is a warning, as configured")
        rt_git["commit"] = "0" * 40
        fx.write_manifest(fx.artifacts(), [rt_git])
        check(any("HEAD is" in x.message for x in fx.verify(runtimes=True)), "a moved commit is caught")
        fx.write_manifest(fx.artifacts(), [{"id": "old", "check": "lock", "status": "retired",
                                            "interpreter": "C:/nope/python.exe", "lock": "x"}])
        check(fx.verify(runtimes=True) == [], "a retired runtime is not checked")
    finally:
        fx.close()


def test_real_manifest() -> None:
    print("the tracked manifest vs config.example.yaml (no hashing)")
    m = va.load_manifest(va.MANIFEST, None)
    sp = va.schema_problems(m)
    check(sp == [], "models/manifest.yaml has no schema problems", "; ".join(map(str, sp)))
    # config.example.yaml's machine-specific values are @@PLACEHOLDERS@@ (scripts/make_config.py); render it with dummy
    # ones into a temp file (config.template.yaml was replaced by it, 2026-10-07).
    import tempfile
    import make_config as mc
    with open(os.path.join(ROOT, "config.example.yaml"), encoding="utf-8") as f:
        text = f.read()
    dummy = {n: "x" for n in mc.placeholders_in(text)}
    dummy.update({"MODELS_DIR": "/MODELS", "MAIN_MODEL": mc.DEFAULT_MAIN_MODEL})
    tmpdir = tempfile.mkdtemp()
    tmpl = os.path.join(tmpdir, "config.yaml")
    with open(tmpl, "w", encoding="utf-8", newline="\n") as f:
        f.write(mc.render(text, dummy))
    cfg = yaml.safe_load(open(tmpl, encoding="utf-8"))
    over = {"models": "/MODELS", "server": "srv", "server_mtp": "srv"}
    macros = va.config_macros(cfg, tmpl, over)
    recorded = {va.norm(va.resolve(str(a["path"]), macros)) for a in m["artifacts"]}
    refs = va.config_files(tmpl, over)
    missing = sorted({p for _, _, p in refs if va.norm(os.path.normpath(p)) not in recorded})
    check(len(refs) >= 10 and not missing,
          f"all {len(refs)} model-file arguments in config.example.yaml are recorded",
          "; ".join(missing))
    pinned = [a for a in m["artifacts"] if a["provenance"]["status"] == "pinned"]
    check(all(a["provenance"].get("checked") and a["provenance"].get("revision_basis")
              for a in pinned), "every pinned entry says when and why its revision")
    derived = [a for a in m["artifacts"] if a["provenance"]["status"] in ("reproduced", "recipe")]
    check(all((a["provenance"]["recipe"].get("inputs") or a["provenance"]["recipe"].get("command"))
              for a in derived), "every derived entry names inputs or a command")
    ids = {a["id"] for a in m["artifacts"]}
    dangling = [(a["id"], i["id"]) for a in derived
                for i in a["provenance"]["recipe"].get("inputs") or [] if "id" in i and i["id"] not in ids]
    check(not dangling, "every recipe input id is itself recorded", str(dangling))
    by_id = {a["id"]: a for a in m["artifacts"]}
    wrong = [(a["id"], i["id"]) for a in derived
             for i in a["provenance"]["recipe"].get("inputs") or []
             if "id" in i and i.get("sha256") != by_id.get(i["id"], {}).get("sha256")]
    check(not wrong, "every recipe input's sha256 is that artifact's sha256", str(wrong))
    for rt in m["runtimes"]:
        if rt.get("check") == "lock" and rt.get("status", "in_service") == "in_service":
            check(os.path.exists(os.path.join(ROOT, rt["lock"])), f"lock file {rt['lock']} exists")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D401
        pass


def test_download() -> None:
    print("fetch_models.download against a local server")
    d = tempfile.mkdtemp(prefix="yamadori-fetch-")
    srv = None
    try:
        src = os.path.join(d, "srv")
        os.makedirs(src)
        payload = os.urandom(200_000)
        open(os.path.join(src, "w.gguf"), "wb").write(payload)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                              functools.partial(_Quiet, directory=src))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/w.gguf"
        dest = os.path.join(d, "out", "w.gguf")
        with contextlib.redirect_stdout(io.StringIO()):
            got = fm.download(url, dest, size=len(payload), sha256=sha(payload))
        check(got == sha(payload) and open(dest, "rb").read() == payload,
              "a good download lands under its final name")
        check(not os.path.exists(dest + ".part"), "no .part is left behind")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fm.download(url, dest, size=len(payload), sha256=sha(payload))
        check("already present and verified" in out.getvalue(), "a present, verified file is left alone")
        bad = os.path.join(d, "out", "bad.gguf")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fm.download(url, bad, size=len(payload), sha256="0" * 64)
            ok = False
        except SystemExit as e:
            ok = "VERIFY FAILED" in str(e)
        check(ok and not os.path.exists(bad) and os.path.exists(bad + ".part"),
              "a hash mismatch keeps only the .part")
        open(bad, "wb").write(b"something else")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fm.download(url, bad, size=len(payload), sha256=sha(payload))
            ok = False
        except SystemExit as e:
            ok = "REFUSED" in str(e)
        check(ok and open(bad, "rb").read() == b"something else",
              "an existing different file is never overwritten")
        blob = os.path.join(d, "blob.txt")
        open(blob, "wb").write(b"hello\n")
        check(fm.git_blob_sha1(blob) == "ce013625030ba8dba906f756967f9e9ca394464a",
              "git blob id matches git's (hash-object of 'hello\\n')")
        check(fm.resolve_url("a/b", "0" * 40, "vae/x y.safetensors").endswith(
            "/a/b/resolve/" + "0" * 40 + "/vae/x%20y.safetensors"), "resolve URL is pinned and quoted")
    finally:
        if srv:
            srv.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def test_split_gguf_shards() -> None:
    """config.yaml names the first shard of a split GGUF (`-m <stem>-00001-of-
    00002.gguf`); llama.cpp loads the others itself. A manifest entry for a
    later shard is in service, not "loads no such file" (the warning
    flash-next-iq2xs-shard2-ngram drew, 2026-10-02) -- and, being loaded, it is
    checked like any file config.yaml names."""
    print("verify: split GGUF shards")
    got = [os.path.normpath(x) for x in va.other_shards("/m/x-00001-of-00003.gguf")]
    check(got == [os.path.normpath("/m/x-00002-of-00003.gguf"),
                  os.path.normpath("/m/x-00003-of-00003.gguf")],
          "other_shards names the rest of a split, same directory and stem", str(got))
    check(va.other_shards("/m/x.gguf") == [] and va.other_shards("/m/x-1-of-2.gguf") == [],
          "a plain GGUF, or a name not in the -NNNNN-of-NNNNN form, has no shards")
    fx = Fixture()
    try:
        s1, s2 = b"shard one bytes" * 20, b"shard two bytes" * 30
        fx.write("big-00001-of-00002.gguf", s1)
        fx.write("big-00002-of-00002.gguf", s2)
        fx.write_config("--llm ${models}/big-00001-of-00002.gguf\n")
        prov = {"status": "hash_only"}

        def arts(second: dict | None) -> list[dict]:
            a = fx.artifacts() + [{"id": "big-1", "role": "r", "status": "in_service",
                                   "path": "${models}/big-00001-of-00002.gguf",
                                   "size": len(s1), "sha256": sha(s1), "provenance": prov}]
            return a + ([second] if second else [])
        shard2 = {"id": "big-2", "role": "r", "status": "in_service",
                  "path": "${models}/big-00002-of-00002.gguf", "size": len(s2),
                  "sha256": sha(s2), "provenance": prov}
        fx.write_manifest(arts(shard2))
        check(fx.verify() == [], "a later shard of a split GGUF config.yaml names is "
              "in service: no warning", str(fx.verify()))

        fx.write("big-00002-of-00002.gguf", b"X" * len(s2))
        p = errors(fx.verify())
        check(len(p) == 1 and p[0].artifact == "big-2" and "sha256" in p[0].message,
              "and it is hashed: the wrong bytes in shard 2 are an error", str(p))
        os.remove(os.path.join(fx.models, "big-00002-of-00002.gguf"))
        p = errors(fx.verify())
        check(len(p) == 1 and p[0].artifact == "big-2" and "does not exist" in p[0].message,
              "and a missing shard 2 is an error", str(p))
        fx.write("big-00002-of-00002.gguf", s2)

        fx.write_manifest(arts(None))
        p = errors(fx.verify())
        check(len(p) == 1 and "no manifest entry" in p[0].message
              and p[0].artifact.endswith("big-00002-of-00002.gguf"),
              "a shard the loader will read that no manifest entry records is an error",
              str(p))

        # a shard entry whose first shard config.yaml does NOT name is still a ghost
        fx.write_config()
        fx.write_manifest(arts(shard2))
        p = fx.verify()
        check(sorted(x.artifact for x in p) == ["big-1", "big-2"]
              and all(x.severity == "warn" for x in p),
              "shards of a split config.yaml no longer names are both still warned about",
              str(p))
    finally:
        fx.close()


def main() -> int:
    for t in (test_config_parse, test_verify_files, test_split_gguf_shards, test_hash_cache,
              test_schema,
              test_code_loaded, test_runtimes, test_real_manifest, test_download):
        try:
            t()
        except Exception:
            traceback.print_exc()
            check(False, f"{t.__name__} raised")
    passed = sum(ok for ok, _ in _results)
    print(f"\n{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
