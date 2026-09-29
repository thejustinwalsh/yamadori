#!/usr/bin/env python
"""PACKAGE ONBOARDING through the REAL package registry, offline.

    python mcp/test_onboarding_registry.py      -> "N/M checks passed"

mcp/test_onboarding.py drives every stage with the registry's floor as a
switch; this suite runs resolve -> clarify -> index -> vocab of one small
package through the real mcp/package_registry.py, skill_packages and
skill_classify (a temp registry, a temp source tree, a fake transport, a
fake indexer), and checks what the rest of the stack then sees:

  1. held.json was frozen before the fetch, and the new package is held only
     after its vocabulary is promoted;
  2. packages.json has the entry: its term from the npm name, the operator's
     alias, its version, its onboarding;
  3. the taxonomy grows: skill_classify.taxonomy() lists the new framework
     term and asked_terms() finds it by the alias -- in this process, with
     no restart (refresh() on a stat);
  4. the detector finds the package from a unique exported symbol in code;
  5. the tag version records the catalogue (tag/6+<sha8> changed);
  6. the floor ran over the standing labels and passed, and the vocab record
     says so.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tarfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_onboarding_registry_")
_EMPTY = os.path.join(_TMP, "empty")
os.makedirs(_EMPTY, exist_ok=True)
os.environ.update({
    "YAMADORI_PKG_DIR": os.path.join(_TMP, "packages"),
    "YAMADORI_PKG_HISTORY": os.path.join(_TMP, "packages",
                                         "registry_history.json"),
    "YAMADORI_PKG_REGISTRY_DIR": os.path.join(_TMP, "packages"),
    "YAMADORI_PACKAGES_SRC": os.path.join(_TMP, "packages", "_src"),
    # The platform and prose lexicons are the TypeScript lib, @types/node
    # and installed READMEs: empty here, so the build is the fake package's
    # own (the real lexicons are mcp/test_skill_packages.py's).
    "YAMADORI_TS_LIB": _EMPTY, "YAMADORI_NODE_TYPES": _EMPTY,
    "YAMADORI_SKILL_CACHE_SECONDS": "0", "YAMADORI_POLL_SECONDS": "0.05",
    "YAMADORI_BEAT_SECONDS": "3600"})
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import deps  # noqa: E402
import idle  # noqa: E402
import package_net  # noqa: E402
import skill_packages  # noqa: E402

# A fixed English lexicon: the one-word rule is package_registry's own suite.
skill_packages.prose_words = lambda *a, **k: {"widget", "spin"}
skill_packages.tokenizer_words = lambda: set()

import onboarding  # noqa: E402
import package_registry  # noqa: E402
import skill_classify  # noqa: E402
import skill_prompts  # noqa: E402
import worker  # noqa: E402

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


C = "e" * 40
NAME, VER = "zorp-engine", "3.1.0"
MIT = "MIT License\n\nCopyright (c) 2026 Zorp\n\nPermission is hereby granted"


def _tgz(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for n, t in files.items():
            b = t.encode()
            ti = tarfile.TarInfo(f"package/{n}")
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    return buf.getvalue()


BLOB = _tgz({"package.json": json.dumps({"name": NAME, "version": VER}),
             "index.d.ts": "export declare function createZorpWorld(): "
                           "void;\nexport declare function spinZorpAxle(n: "
                           "number): void;\n",
             "LICENSE": MIT})
SRI = "sha512-" + base64.b64encode(hashlib.sha512(BLOB).digest()).decode()
PACKUMENT = {"name": NAME, "dist-tags": {"latest": VER},
             "versions": {VER: {"name": NAME, "version": VER, "license": "MIT",
                                "gitHead": C,
                                "repository": "github:zorp/engine",
                                "dist": {"tarball": f"https://registry.npmjs"
                                                    f".org/{NAME}/-/{NAME}-"
                                                    f"{VER}.tgz",
                                         "integrity": SRI,
                                         "unpackedSize": 10 ** 6}}},
             "time": {VER: "2026-09-20T00:00:00.000Z"}}
RAW = {"package.json": '{\n  "name": "zorp-engine",\n  "license": "MIT"\n}\n',
       "LICENSE": MIT}


def transport(url, headers, max_bytes):
    R = package_net.Response
    if url == f"https://registry.npmjs.org/{NAME}":
        return R(200, json.dumps(PACKUMENT).encode(), {}, url)
    if url.endswith(f"/{NAME}-{VER}.tgz"):
        return R(200, BLOB, {}, url)
    pre = f"https://raw.githubusercontent.com/zorp/engine/{C}/"
    if url.startswith(pre) and url[len(pre):] in RAW:
        return R(200, RAW[url[len(pre):]].encode(), {}, url)
    return R(404, b"", {}, url)


package_net.TRANSPORT = transport
idle.stack_idle = lambda job_id=None: {"idle": True, "why": "test",
                                       "until": None, "checks": {}}
deps.index_package = lambda *a, **k: {"ok": True, "installed": True,
                                      "chunks": 3, "files_selected": 1}


def main() -> int:
    try:
        tv0 = skill_prompts.tag_version()
        check(NAME not in skill_packages.vocabulary()["packages"],
              "[before] the package is not held")
        ds = onboarding.submit(f"add zorp https://www.npmjs.com/package/{NAME}",
                               aliases=["zorp"])
        did = ds["id"]
        # resolve -> clarify -> index -> vocab; stop before examples.
        for _ in range(8):
            ds = onboarding.datasets.get(did)
            if ds["stage"] in ("examples", "complete") or not \
                    worker.jobs.claimable(
                        worker.datasets.PACKAGE_ENQUEUE.get(
                            ds["stage"], ("", "net"))[1]):
                break
            lane = worker.datasets.PACKAGE_ENQUEUE[ds["stage"]][1]
            worker.run_one(worker.jobs.claim(lane, "t:1"))
        v = onboarding.read_json(did, "vocab.json") or {}
        check(v.get("state") == "promoted" and (v.get("floor") or {}).get(
            "passed") and (v.get("floor") or {}).get("rows"),
              "[vocab] the floor ran over the standing labels and passed; "
              "promoted", {k: v.get(k) for k in ("state", "why", "floor")})
        held = package_registry.held_manifest() or {}
        check(f"{NAME}@{VER}" in held.get("dirs", []) and held.get("by"),
              "[held] held.json names the new dir after promotion",
              held.get("dirs"))
        e = package_registry.entry(NAME) or {}
        check(e.get("term") == "zorp_engine" and "zorp" in e.get("aliases")
              and VER in e.get("versions") and e.get("onboarding") == did,
              "[registry] packages.json: the term from the npm name, the "
              "typed alias, the version, the onboarding", e)
        tax = [t["id"] for t in skill_classify.taxonomy()["framework"]]
        check("zorp_engine" in tax, "[taxonomy] a framework term for the "
              "package, in this process (refresh on a stat)", tax[-4:])
        asked = skill_classify.asked_terms("build the game with zorp")
        check("zorp_engine" in asked, "[taxonomy] asked_terms finds it by "
              "the operator's alias", asked)
        got = skill_packages.detect([{"role": "user", "content":
                                      "```ts\ncreateZorpWorld()\n```"}])
        check(NAME in got, "[detect] a unique exported symbol in code puts "
              "the package in play", got)
        check(skill_prompts.tag_version() != tv0
              and skill_prompts.tag_version().startswith("tag/6+"),
              "[tag] the recorded tag version names the new catalogue",
              (tv0, skill_prompts.tag_version()))
        check(onboarding.datasets.get(did)["stage"] == "examples",
              "[stage] the onboarding moved on to examples",
              onboarding.datasets.get(did)["stage"])
    except Exception:                                            # noqa: BLE001
        check(False, "the suite raised", traceback.format_exc())
    print(f"\n  {N[0] - len(FAILS)}/{N[0]} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
