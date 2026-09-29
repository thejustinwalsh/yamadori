#!/usr/bin/env python
"""yama_resolve_packages (mcp/npm_resolve.py and its driver,
mcp_servers/packagelens/resolve_driver.js). No Docker, no network, no GPU.

    python mcp/test_npm_resolve.py      -> "N/M checks passed"

GATED HERE:
  1. THE DRIVER, run by the host's node exactly as the container runs it
     (`node -`, the file on stdin, the request in YAMA_RESOLVE_INPUT) against
     a local registry (a thread here, 127.0.0.1) and a FAKE npm
     (mcp/fixtures/npm_resolve/fake_npm.js) whose canned outputs are npm
     11.19.0's own shapes, captured on 2026-09-29 (the ERESOLVE report's
     detail text verbatim, v3 lockfiles):
       - every npm call is `install --package-lock-only --ignore-scripts
         --no-audit --no-fund --json`, in the work directory;
       - a set npm accepts with a peer outside its range is caught by the
         EXACT run and fixed by moving the peer within what was asked
         (react ^19: 19.3.0 -> 19.2.8 for @react-three/fiber 10.0.0-alpha.5's
         `>=19.0 <19.3`), then the peer three and the type packages;
       - DefinitelyTyped's major.minor: react 19.2.8 -> @types/react 19.2.x,
         not 19.3.0; three 0.186.1 -> @types/three 0.186.0;
       - the optional peer npm left out (react-dom) listed with its ranges;
       - the prerelease policy: none without allow_prerelease (and no npm
         run), the channels said; with it, the version a dist-tag names, not
         a newer untagged per-commit build;
       - a conflict nothing asked can move: npm's words, the newest peer that
         fits, the newest dependent whose peer range admits the peer;
       - the dependent moved when the peer cannot be (drei 10.7.9 needs fiber
         ^9 -> drei 11.0.0-alpha.7, never drei 3.x, which does not declare
         the peer at all, nor 9.120.0, whose `>=8.0` a prerelease does not
         satisfy by npm's rule);
       - refused specs: words, a git URL, a missing package, a missing tag, a
         missing version, a duplicate -- each with its code, no npm run.
  2. THE MODULE: the container's command (pinned image, the gated network's
     proxy variables, uid 1000, no capabilities, no-new-privileges,
     read-only root, tmpfs /tmp, no volume, the driver on stdin); the
     arguments' shape; a fake container runner (the result line found among
     other output; no result -> RESOLVE_FAILED, retryable; docker missing ->
     MCP_SERVER_DOWN); the text (install lines first, package.json, each
     package and why, the moves, the conflict and what fits, the refusals
     with their remedies) passes the screen and the assured voice
     (skill_limits.doubt); the record carries no free text; the bound is
     npm's own (970 s).
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import npm_resolve  # noqa: E402
import skill_limits  # noqa: E402
import skill_screen  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="yamadori_test_npm_resolve_")
FAKE_NPM = os.path.join(HERE, "fixtures", "npm_resolve", "fake_npm.js")
NODE = shutil.which("node")

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ---------------------------------------------------------------- registry
FIBER = "@react-three/fiber"
DREI = "@react-three/drei"
P_ALPHA = {"react": ">=19.0 <19.3", "react-dom": ">=19.0 <19.3",
           "three": ">=0.185.0"}
P_CANARY = {"react": ">=19.0 <19.4", "react-dom": ">=19.0 <19.4",
            "three": ">=0.185.0"}
P_9 = {"react": ">=19 <19.4", "react-dom": ">=19 <19.4", "three": ">=0.156"}
OPT_DOM = {"react-dom": {"optional": True}}


def _pk(tags: dict, versions: dict) -> dict:
    return {"name": "x", "dist-tags": tags,
            "versions": {v: dict({"name": "x", "version": v}, **e)
                         for v, e in versions.items()}}


def _peers(p: dict, meta: dict | None = None) -> dict:
    return {"peerDependencies": p, **({"peerDependenciesMeta": meta} if meta else {})}


PACKUMENTS = {
    FIBER: _pk({"latest": "9.8.1", "alpha": "10.0.0-alpha.5",
                "canary": "10.0.0-canary.b9d5979"},
               {"9.8.1": _peers(P_9, OPT_DOM),
                "10.0.0-alpha.4": _peers(P_ALPHA, OPT_DOM),
                "10.0.0-alpha.5": _peers(P_ALPHA, OPT_DOM),
                "10.0.0-canary.b9d5979": _peers(P_CANARY, OPT_DOM),
                # Newer than b9d5979 by semver (f > b), and no dist-tag
                # names it: a per-commit build.
                "10.0.0-canary.f75fbf6": _peers(P_CANARY, OPT_DOM)}),
    "react": _pk({"latest": "19.3.0", "canary": "19.3.0-canary-d083ec1d-20260922"},
                 {"18.3.1": {}, "19.2.8": {}, "19.3.0": {},
                  "19.3.0-canary-d083ec1d-20260922": {}}),
    "react-dom": _pk({"latest": "19.3.0"},
                     {"19.2.8": _peers({"react": "^19.2.8"}),
                      "19.3.0": _peers({"react": "^19.3.0"})}),
    "three": _pk({"latest": "0.186.1"}, {"0.185.0": {}, "0.186.1": {}}),
    "@types/react": _pk({"latest": "19.3.0"}, {"19.2.18": {}, "19.3.0": {}}),
    "@types/react-dom": _pk({"latest": "19.3.0"}, {"19.2.7": {}, "19.3.0": {}}),
    "@types/three": _pk({"latest": "0.186.0"}, {"0.185.0": {}, "0.186.0": {}}),
    DREI: _pk({"latest": "10.7.9", "alpha": "11.0.0-alpha.7"},
              {"3.11.2": _peers({"react-three-fiber": ">=5.3.18",
                                 "react": ">=17.0", "three": ">=0.125.0"}),
               "9.120.0": _peers({FIBER: ">=8.0", "react": ">=18.0",
                                  "three": ">=0.137"}),
               "10.7.9": _peers({FIBER: "^9.0.0", "react": "^19",
                                 "three": ">=0.159"}),
               "11.0.0-alpha.7": _peers({FIBER: ">=10.0.0-0",
                                         "react": ">=19.0 <19.3",
                                         "three": ">=0.185"})}),
}
# The full version documents (for `types`): the r3f packages ship their own.
TYPED = {FIBER, DREI}


class Registry(http.server.BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        parts = self.path.split("?")[0].split("/")
        name = urllib.parse.unquote(parts[1]) if len(parts) > 1 else ""
        version = urllib.parse.unquote(parts[2]) if len(parts) > 2 else None
        Registry.hits.append({"name": name, "version": version,
                              "accept": self.headers.get("accept")})
        pk = PACKUMENTS.get(name)
        if pk is None or (version and version not in pk["versions"]):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"Not found"}')
            return
        if version:
            doc = dict(pk["versions"][version], name=name)
            if name in TYPED:
                doc["types"] = "dist/index.d.ts"
        else:
            doc = dict(pk, name=name)
        body = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Registry)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
REG = f"http://127.0.0.1:{_srv.server_address[1]}"


# ---------------------------------------------------------------- npm's shapes
def lock(root: dict, entries: dict, dev: dict | None = None) -> dict:
    """A v3 lockfile as npm 11 writes it (package-lock.json)."""
    pk = {"": {"name": "yama-resolve", "version": "0.0.0", "dependencies": root}}
    if dev:
        pk[""]["devDependencies"] = dev
    for n in sorted(entries):
        pk["node_modules/" + n] = dict(
            {"version": entries[n]["version"],
             "resolved": f"https://registry.npmjs.org/{n}/-/x.tgz",
             "integrity": "sha512-x", "license": "MIT"},
            **{k: v for k, v in entries[n].items() if k != "version"})
    return {"name": "yama-resolve", "version": "0.0.0", "lockfileVersion": 3,
            "requires": True, "packages": pk}


def eresolve(found: str, found_spec: str, peer: str, peer_range: str,
             frm: str, frm_spec: str) -> dict:
    """npm 11.19.0's --json ERESOLVE, the detail text as npm writes it
    (captured 2026-09-29: @react-three/fiber@10.0.0-alpha.5 with
    react@19.3.0)."""
    fname = found.rsplit("@", 1)[0]
    frname = frm.rsplit("@", 1)[0]
    detail = (f"While resolving: yama-resolve@0.0.0\nFound: {found}\n"
              f"node_modules/{fname}\n  {fname}@\"{found_spec}\" from the root "
              f"project\n\nCould not resolve dependency:\npeer {peer}@"
              f"\"{peer_range}\" from {frm}\nnode_modules/{frname}\n  "
              f"{frname}@\"{frm_spec}\" from the root project\n\nFix the "
              f"upstream dependency conflict, or retry this command with "
              f"--force or --legacy-peer-deps to accept an incorrect (and "
              f"potentially broken) dependency resolution.\n\n\nFor a full "
              f"report see:\n/tmp/.npm/_logs/2026-09-29T14_10_25_507Z-"
              f"eresolve-report.txt")
    return {"error": {"code": "ERESOLVE",
                      "summary": "unable to resolve dependency tree",
                      "detail": detail}}


OK_OUT = {"add": [], "added": 0, "audited": 0, "change": [], "changed": 0,
          "funding": 0, "remove": [], "removed": 0}
FIB_A = {"version": "10.0.0-alpha.5", "peerDependencies": P_ALPHA,
         "peerDependenciesMeta": OPT_DOM}
FIB_C = {"version": "10.0.0-canary.b9d5979", "peerDependencies": P_CANARY,
         "peerDependenciesMeta": OPT_DOM}
THREE_PEER = {"version": "0.186.1", "peer": True}


def run_driver(tag: str, packages: list, allow: bool, runs: list) -> tuple[dict, list]:
    """The driver as the container runs it: (its result, the npm calls)."""
    d = os.path.join(_TMP, tag)
    os.makedirs(d)
    scen, log = os.path.join(d, "scenario.json"), os.path.join(d, "npm.log")
    with open(scen, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f)
    open(log, "w").close()
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("http_proxy", "https_proxy", "no_proxy",
                                "node_use_env_proxy")}
    env.update(YAMA_RESOLVE_INPUT=json.dumps({
        "packages": packages, "allow_prerelease": allow, "registry": REG,
        "fetch_timeout_ms": npm_resolve.NPM_FETCH_TIMEOUT_MS}),
        YAMA_RESOLVE_NPM_CLI=FAKE_NPM, YAMA_RESOLVE_WORK=os.path.join(d, "work"),
        FAKE_NPM_SCENARIO=scen, FAKE_NPM_LOG=log)
    with open(npm_resolve.DRIVER, encoding="utf-8") as f:
        src = f.read()
    p = subprocess.run([NODE, "-"], input=src, capture_output=True, text=True,
                       env=env, timeout=120, encoding="utf-8")
    line = next((ln for ln in p.stdout.splitlines()
                 if ln.startswith(npm_resolve.RESULT_MARK)), None)
    assert line, f"no result line; rc {p.returncode}; stderr {p.stderr[-800:]}"
    calls = [json.loads(ln) for ln in open(log, encoding="utf-8") if ln.strip()]
    return json.loads(line[len(npm_resolve.RESULT_MARK):]), calls


FLAGS = ["install", "--package-lock-only", "--ignore-scripts", "--no-audit",
         "--no-fund", "--json"]
DRIVER_OUT: dict = {}


# --------------------------------------------------------------- 1. the driver
def test_a_peer_outside_its_range_is_moved_within_what_was_asked():
    L_a = lock({FIBER: "^10.0.0-alpha.5", "react": "^19.3.0"},
               {FIBER: FIB_A, "react": {"version": "19.3.0"},
                "three": THREE_PEER})
    L_b = lock({FIBER: "^10.0.0-alpha.5", "react": "19.2.8"},
               {FIBER: FIB_A, "react": {"version": "19.2.8"},
                "three": THREE_PEER})
    L_c = lock({FIBER: "^10.0.0-alpha.5", "react": "19.2.8"},
               {FIBER: FIB_A, "react": {"version": "19.2.8"},
                "three": THREE_PEER,
                "@types/react": {"version": "19.2.18", "dev": True},
                "@types/three": {"version": "0.186.0", "dev": True}},
               dev={"@types/react": "19.2.18", "@types/three": "0.186.0"})
    exact_a = [f"{FIBER}@10.0.0-alpha.5", "react@19.3.0", "three@0.186.1"]
    runs = [
        {"specs": [f"{FIBER}@alpha", "react@^19"], "stdout": OK_OUT, "lock": L_a},
        {"specs": exact_a, "exit": 1,
         "stdout": eresolve("react@19.3.0", "19.3.0", "react", ">=19.0 <19.3",
                            f"{FIBER}@10.0.0-alpha.5", "10.0.0-alpha.5")},
        {"specs": [f"{FIBER}@alpha", "react@19.2.8"], "stdout": OK_OUT, "lock": L_b},
        {"specs": [f"{FIBER}@10.0.0-alpha.5", "react@19.2.8", "three@0.186.1"],
         "stdout": OK_OUT, "lock": L_b},
        {"specs": ["@types/react@19.2.18", "@types/three@0.186.0"], "dev": True,
         "stdout": OK_OUT, "lock": L_c}]
    d, calls = run_driver("moved", [f"{FIBER}@alpha", "react@^19"], False, runs)
    DRIVER_OUT["moved"] = d
    check(len(calls) == 5 and all(c["argv"][:6] == FLAGS for c in calls)
          and all(os.path.basename(c["cwd"]) == "work" for c in calls),
          "every npm call is install --package-lock-only --ignore-scripts "
          "--no-audit --no-fund --json, in the work directory",
          json.dumps([c["argv"][:7] for c in calls])[:400])
    check([c["argv"][6:] for c in calls][1] == exact_a,
          "npm's accepted set is run again pinned exactly (the run that "
          "catches a peer outside its range)", json.dumps(calls[1]["argv"]))
    check(d["ok"] and d["verified"]
          and d["fixes"] == [{"name": "react", "from": "19.3.0", "to": "19.2.8",
                              "because": f"{FIBER}@10.0.0-alpha.5 needs react "
                                         f">=19.0 <19.3",
                              "asked": "react@^19"}],
          "the peer moves to the newest version the dependent's range and the "
          "caller's own range both admit (react ^19: 19.3.0 -> 19.2.8)",
          json.dumps(d["fixes"]))
    check(d["install"] == (f"npm install --save-exact {FIBER}@10.0.0-alpha.5 "
                           f"react@19.2.8 three@0.186.1")
          and [p["name"] for p in d["packages"]] == ["three"]
          and d["packages"][0]["needed_by"] == [
              {"name": FIBER, "version": "10.0.0-alpha.5",
               "range": ">=0.185.0", "optional": False}],
          "the install line: every package asked for, then the peer npm "
          "installed for them, exact, with who needs it", d["install"])
    check(d["install_dev"] == ("npm install --save-exact --save-dev "
                               "@types/react@19.2.18 @types/three@0.186.0")
          and [(t["for"], t["match"]) for t in d["types"]]
          == [("react", "major.minor"), ("three", "major.minor")]
          and not any(t["for"] == FIBER for t in d["types"]),
          "types: DefinitelyTyped's major.minor (react 19.2.8 -> @types/react "
          "19.2.18, not 19.3.0; three 0.186.1 -> 0.186.0); a package that "
          "ships its own gets none", d["install_dev"] or "")
    check([o["name"] for o in d["optional_peers"]] == ["react-dom"]
          and d["optional_peers"][0]["needed_by"][0]["range"] == ">=19.0 <19.3",
          "the optional peer npm left out, with the range the set declares",
          json.dumps(d["optional_peers"]))
    ac = [h["accept"] for h in Registry.hits if h["name"] == FIBER
          and not h["version"]]
    check(ac and all("application/vnd.npm.install-v1+json" in a for a in ac),
          "packuments are read as npm reads them (the abbreviated document)",
          json.dumps(ac))


def test_the_prerelease_policy():
    Registry.hits.clear()
    d, calls = run_driver("nopre", [f"{FIBER}@^10"], False, [])
    e = (d["errors"] or [{}])[0]
    check(not d["ok"] and not calls and e.get("code") == "ONLY_PRERELEASES"
          and [c["tag"] for c in e.get("channels") or []] == ["canary", "alpha"],
          "without allow_prerelease a range only prereleases satisfy is "
          "refused, the channels said, and npm is not run", json.dumps(d["errors"]))
    L = lock({FIBER: "10.0.0-canary.b9d5979"},
             {FIBER: FIB_C, "react": {"version": "19.3.0", "peer": True},
              "three": THREE_PEER})
    Ld = lock({FIBER: "10.0.0-canary.b9d5979"},
              {FIBER: FIB_C, "react": {"version": "19.3.0", "peer": True},
               "three": THREE_PEER,
               "@types/react": {"version": "19.3.0", "dev": True},
               "@types/three": {"version": "0.186.0", "dev": True}},
              dev={"@types/react": "19.3.0", "@types/three": "0.186.0"})
    exact = [f"{FIBER}@10.0.0-canary.b9d5979", "react@19.3.0", "three@0.186.1"]
    runs = [{"specs": [f"{FIBER}@10.0.0-canary.b9d5979"], "stdout": OK_OUT, "lock": L},
            {"specs": exact, "stdout": OK_OUT, "lock": L},
            {"specs": ["@types/react@19.3.0", "@types/three@0.186.0"],
             "dev": True, "stdout": OK_OUT, "lock": Ld}]
    d, calls = run_driver("pre", [f"{FIBER}@^10"], True, runs)
    DRIVER_OUT["pre"] = d
    r = (d["requested"] or [{}])[0]
    check(d["ok"] and r.get("version") == "10.0.0-canary.b9d5979"
          and r.get("how") == "a prerelease in ^10 (dist-tag canary)"
          and r.get("channels") == [{"tag": "alpha", "version": "10.0.0-alpha.5"}],
          "with allow_prerelease: the version a dist-tag names (canary "
          "b9d5979), not the newer untagged build f75fbf6; the other channel "
          "in range is said", json.dumps(r)[:400])
    check(d["install"] == ("npm install --save-exact "
                           f"{FIBER}@10.0.0-canary.b9d5979 react@19.3.0 "
                           "three@0.186.1"),
          "peers npm installed (react, three) join the line, exact", d["install"])


def test_a_conflict_nothing_asked_can_move():
    runs = [{"specs": [f"{FIBER}@alpha", "react@19.3.0"], "exit": 1,
             "stdout": eresolve("react@19.3.0", "19.3.0", "react",
                                ">=19.0 <19.3", f"{FIBER}@10.0.0-alpha.5",
                                "alpha")}]
    d, calls = run_driver("conflict", [f"{FIBER}@alpha", "react@19.3.0"],
                          False, runs)
    DRIVER_OUT["conflict"] = d
    c = (d["conflicts"] or [{}])[0]
    check(not d["ok"] and len(calls) == 1 and not d["fixes"]
          and c.get("edge") == {"type": "peer", "name": "react",
                                "range": ">=19.0 <19.3"}
          and c.get("found") == {"name": "react", "version": "19.3.0"}
          and c.get("from") == {"name": FIBER, "version": "10.0.0-alpha.5"}
          and c.get("peer_fits") == "19.2.8" and c.get("dependent_fits") == "9.8.1"
          and c.get("asked") == ["react@19.3.0", f"{FIBER}@alpha"],
          "exact asks are not moved: npm's report parsed, the newest react "
          "that fits (19.2.8), the newest fiber whose peer range admits react "
          "19.3.0 (9.8.1), what was asked", json.dumps(c)[:500])


def test_the_dependent_moves_when_the_peer_cannot():
    Lr = lock({FIBER: "10.0.0-canary.b9d5979", DREI: "11.0.0-alpha.7"},
              {DREI: {"version": "11.0.0-alpha.7",
                      "peerDependencies": {FIBER: ">=10.0.0-0",
                                           "react": ">=19.0 <19.3",
                                           "three": ">=0.185"}},
               FIBER: FIB_C, "react": {"version": "19.2.8", "peer": True},
               "three": THREE_PEER})
    Ld = json.loads(json.dumps(Lr))
    Ld["packages"][""]["devDependencies"] = {"@types/react": "19.2.18",
                                             "@types/three": "0.186.0"}
    Ld["packages"]["node_modules/@types/react"] = {"version": "19.2.18", "dev": True}
    Ld["packages"]["node_modules/@types/three"] = {"version": "0.186.0", "dev": True}
    two = [f"{FIBER}@10.0.0-canary.b9d5979", f"{DREI}@11.0.0-alpha.7"]
    runs = [{"specs": [f"{FIBER}@10.0.0-canary.b9d5979", DREI], "exit": 1,
             "stdout": eresolve(f"{FIBER}@10.0.0-canary.b9d5979",
                                "10.0.0-canary.b9d5979", FIBER, "^9.0.0",
                                f"{DREI}@10.7.9", "*")},
            {"specs": two, "stdout": OK_OUT, "lock": Lr},
            {"specs": two + ["react@19.2.8", "three@0.186.1"], "stdout": OK_OUT,
             "lock": Lr},
            {"specs": ["@types/react@19.2.18", "@types/three@0.186.0"],
             "dev": True, "stdout": OK_OUT, "lock": Ld}]
    d, calls = run_driver("drei", [f"{FIBER}@^10", DREI], True, runs)
    DRIVER_OUT["drei"] = d
    check(d["ok"] and d["fixes"] and d["fixes"][0]["name"] == DREI
          and d["fixes"][0]["from"] == "10.7.9"
          and d["fixes"][0]["to"] == "11.0.0-alpha.7",
          "the dependent moves to the newest version whose own peer range "
          "admits the peer (drei 11.0.0-alpha.7 for fiber's canary), never a "
          "version that does not declare the peer (drei 3.11.2) or whose range "
          "a prerelease does not satisfy by npm's rule (9.120.0: >=8.0)",
          json.dumps(d["fixes"]))
    check(d["install"] == ("npm install --save-exact "
                           f"{FIBER}@10.0.0-canary.b9d5979 "
                           f"{DREI}@11.0.0-alpha.7 react@19.2.8 three@0.186.1"),
          "the set after the move", d["install"] or json.dumps(d["errors"]))


def test_refused_specs():
    d, calls = run_driver("refused", [
        "pmndrs math", "git+https://example.com/x/y", "nosuch-yamadori-pkg",
        "three@nightly", "react@9.9.9", "react@18"], False, [])
    DRIVER_OUT["refused"] = d
    codes = {e["spec"]: e["code"] for e in d["errors"]}
    check(not d["ok"] and not calls and codes == {
        "pmndrs math": "NOT_A_PACKAGE_NAME",
        "git+https://example.com/x/y": "NOT_A_REGISTRY_SPEC",
        "nosuch-yamadori-pkg": "NOT_FOUND",
        "three@nightly": "NO_SUCH_TAG",
        "react@9.9.9": "NO_SUCH_VERSION",
        "react@18": "DUPLICATE"},
          "words, a git URL, a missing package, tag or version and a "
          "duplicate are refused by code, and npm is not run", json.dumps(codes))


# --------------------------------------------------------------- 2. the module
def test_the_container_command():
    argv = npm_resolve.run_argv("yamadori-mcp-packagelens:0.1.11",
                                "yama-mcp-net-packagelens", "yama-mcp-resolve-x",
                                '{"packages":["react"]}',
                                ["-e", "HTTPS_PROXY=http://egress:3128"])
    s = " ".join(argv)
    check(argv[:3] == ["docker", "run", "--rm"] and "-i" in argv
          and argv[-3:] == ["node", "yamadori-mcp-packagelens:0.1.11", "-"]
          and "--network yama-mcp-net-packagelens" in s
          and "-e HTTPS_PROXY=http://egress:3128" in s
          and "-e NODE_USE_ENV_PROXY=1" in s
          and "--user 1000:1000" in s and "--cap-drop ALL" in s
          and "--security-opt no-new-privileges" in s and "--read-only" in s
          and "--tmpfs /tmp" in s and "-v" not in argv and "--volume" not in s
          and "--mount" not in s and "TOKEN" not in s,
          "the container: the pinned image's node reading the driver from "
          "stdin, the gated network and its proxy, uid 1000, no capabilities, "
          "no-new-privileges, read-only root, tmpfs /tmp, no volume, no "
          "credential", s)
    argv = npm_resolve.run_argv("img", None, "n", "{}")
    check("--network none" in " ".join(argv), "no network given: none")
    check(npm_resolve.FETCH_WORST_S == 970.0,
          "the bound is npm's own for one fetch: 3 x 300 s + 10 s + 60 s")


def test_the_arguments():
    ca = npm_resolve.check_args
    check(ca({"packages": ["react", "three@^0.186"], "ecosystem": "npm"})
          == (["react", "three@^0.186"], False)
          and ca({"packages": "react, three", "ecosystem": "npm",
                  "allow_prerelease": "true"}) == (["react", "three"], True),
          "a list, or one string split as an npm command line; the flag as a "
          "boolean")
    for bad in ({"packages": ["react"], "ecosystem": "pypi"},
                {"packages": [], "ecosystem": "npm"},
                {"packages": [3], "ecosystem": "npm"},
                {"packages": ["react"], "ecosystem": "npm",
                 "allow_prerelease": "maybe"}):
        check(isinstance(ca(bad), str), f"refused: {json.dumps(bad)}", str(ca(bad)))


def test_the_runner_and_the_text():
    seen = {}

    def fake(argv, stdin_text, timeout_s, name):
        seen.update(argv=argv, stdin=stdin_text, timeout=timeout_s, name=name)
        return 0, ("npm notice something\n" + npm_resolve.RESULT_MARK
                   + json.dumps(DRIVER_OUT["moved"]) + "\n"), ""
    saved = npm_resolve.RUNNER_FN
    try:
        npm_resolve.RUNNER_FN = fake
        d = npm_resolve.resolve([f"{FIBER}@alpha", "react@^19"], False, "img",
                                "net", "packagelens", [], timeout_s=5.0)
        with open(npm_resolve.DRIVER, encoding="utf-8") as f:
            src = f.read()
        inp = next(a for a in seen["argv"] if a.startswith("YAMA_RESOLVE_INPUT="))
        check(d == DRIVER_OUT["moved"] and seen["stdin"] == src
              and json.loads(inp.split("=", 1)[1])["packages"]
              == [f"{FIBER}@alpha", "react@^19"]
              and seen["name"].startswith("yama-mcp-resolve-packagelens-")
              and seen["timeout"] == 5.0,
              "the runner gets the driver on stdin and the request in one "
              "variable; the result line is found among other output",
              json.dumps({k: v for k, v in seen.items() if k != "stdin"})[:300])
        npm_resolve.RUNNER_FN = lambda *a: (1, "", "docker: Error response\n")
        try:
            npm_resolve.resolve(["react"], False, "img", "net", "t", [])
            check(False, "no result line raises")
        except npm_resolve.ResolveError as e:
            check(e.code == "RESOLVE_FAILED" and e.retryable
                  and "Error response" in str(e),
                  "no result line: RESOLVE_FAILED, retryable, the stderr tail",
                  str(e))

        def missing(*a):
            raise FileNotFoundError("docker")
        npm_resolve.RUNNER_FN = missing
        try:
            npm_resolve.resolve(["react"], False, "img", "net", "t", [])
            check(False, "docker missing raises")
        except npm_resolve.ResolveError as e:
            check(e.code == "MCP_SERVER_DOWN", "docker missing: MCP_SERVER_DOWN",
                  str(e))
    finally:
        npm_resolve.RUNNER_FN = saved

    text, names = npm_resolve.render(DRIVER_OUT["moved"], {})
    lines = text.split("\n")
    check(lines[0].startswith("Resolved by npm ") and "no install scripts run"
          in lines[0]
          and lines[2] == (f"  npm install --save-exact {FIBER}@10.0.0-alpha.5 "
                           f"react@19.2.8 three@0.186.1")
          and lines[3] == ("  npm install --save-exact --save-dev "
                           "@types/react@19.2.18 @types/three@0.186.0"),
          "the install lines come first", "\n".join(lines[:5]))
    blk = text[text.index("package.json:\n") + len("package.json:\n"):]
    blk = json.loads(blk[:blk.index("\n}\n") + 2])
    check(blk == {"dependencies": {FIBER: "10.0.0-alpha.5", "react": "19.2.8",
                                   "three": "0.186.1"},
                  "devDependencies": {"@types/react": "19.2.18",
                                      "@types/three": "0.186.0"}},
          "package.json's blocks, exact", json.dumps(blk))
    check(f"- react 19.2.8 (asked react@^19; required by {FIBER} >=19.0 <19.3)"
          in text
          and "- three 0.186.1 (a peer dependency: required by "
              f"{FIBER} >=0.185.0)" in text
          and "- react 19.3.0 -> 19.2.8: " in text
          and f"- react-dom: {FIBER} >=19.0 <19.3 (optional)" in text
          and "- @types/react 19.2.18 (types for react 19.2.8, which ships "
              "none; DefinitelyTyped's version matches the library's "
              "major.minor)" in text,
          "each package and why; the move; the optional peer; the types", text)
    check(names == [FIBER, "react", "three", "@types/react", "@types/three"],
          "the names it returns (the probe's found names)", json.dumps(names))
    texts = {k: npm_resolve.render(v, {})[0] for k, v in DRIVER_OUT.items()}
    ct = texts["conflict"]
    check("These packages do not install together" in ct
          and f"- {FIBER}@10.0.0-alpha.5 needs react >=19.0 <19.3; the set has "
              "react 19.3.0" in ct
          and "- react 19.2.8 is the newest react that fits" in ct
          and f"- {FIBER} 9.8.1 is the newest {FIBER} whose peer range admits "
              "react 19.3.0." in ct
          and "Call it again with one of those versions" in ct,
          "a conflict: who needs what, and the newest of each side that fits",
          ct)
    rt = texts["refused"]
    check("- pmndrs math: " in rt and "call yama_find_package" in rt
          and "- react@18: react is asked for twice" in rt
          and "Could not resolve:" in rt,
          "each refusal with its remedy (words -> yama_find_package)", rt)
    for k, t in texts.items():
        v = skill_screen.screen_fetched(t, t, "text")
        check(v["ok"] and not v["stripped"],
              f"the {k} text passes the screen untouched",
              json.dumps(v["stripped"])[:200])
        # Line by line: the whole text names several releases as DATA (a
        # channel list beside a pick), which the rule's two-releases-in-one-
        # line test reads as history in prose; each line is one statement.
        bad = [(skill_limits.doubt(ln), ln) for ln in t.split("\n")
               if skill_limits.doubt(ln)]
        check(not bad,
              f"every line of the {k} text is in the assured voice "
              f"(skill_limits.doubt)", json.dumps(bad)[:300])
    rec = npm_resolve.record(DRIVER_OUT["moved"], [f"{FIBER}@alpha", "react@^19"],
                             False, 120)
    check(rec["pins"] == {FIBER: "10.0.0-alpha.5", "react": "19.2.8",
                          "three": "0.186.1", "@types/react": "19.2.18",
                          "@types/three": "0.186.0"}
          and rec["resolved"] == 5 and rec["fixes"] == 1
          and rec["conflicts"] == 0 and rec["errors"] == [] and rec["verified"]
          and rec["npm"]["runs"] == 5 and rec["runner"] == "npm_resolve",
          "the record: pins, counts, codes -- no free text", json.dumps(rec))
    rec = npm_resolve.record(DRIVER_OUT["refused"], ["pmndrs math"], False, 120)
    check(rec["resolved"] == 0 and rec["pins"] == {}
          and "NOT_A_PACKAGE_NAME" in rec["errors"],
          "a refused set records its codes and nothing resolved", json.dumps(rec))


def test_the_probe_reads_the_resolver():
    """bench/mcp/lookup_probe.py: the resolver is one of the tools a valid
    trial must have on main, and the install command's versions are
    compared with what it pinned (the command parsed per first command)."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "mcp"))
    import lookup_probe as L
    check("yama_resolve_packages" in L.TOOLS and L.RESOLVE in L.TOOLS
          and "yama_list_package_versions" in L.TOOLS
          and "yama_package_versions" not in L.TOOLS,
          "the probe's tools are the renamed four", json.dumps(L.TOOLS))
    cases = {
        "npm install three@0.186.1 react@19.2.8 2>&1 | tail -25":
            ["three@0.186.1", "react@19.2.8"],
        "cd app && npm i --prefix app -D @types/react@19.2.18\nnpm run dev":
            ["@types/react@19.2.18"],
        "npm install --save-exact @react-three/fiber@10.0.0-alpha.5 koota":
            ["@react-three/fiber@10.0.0-alpha.5", "koota"]}
    got = {c: L.install_packages(c) for c in cases}
    check(got == cases,
          "install names: a redirection, a flag's value and the next line "
          "are not packages", json.dumps(got))
    specs = {L._bare(p): L._spec_version(p) for p in
             cases["npm install --save-exact @react-three/fiber@10.0.0-alpha.5 koota"]}
    v = L.versions_vs_resolved(
        dict(specs, react="^19.2.8"),
        {"@react-three/fiber": "10.0.0-alpha.5", "koota": "0.6.6",
         "react": "19.3.0"})
    check(specs == {"@react-three/fiber": "10.0.0-alpha.5", "koota": None}
          and v == {"matched": ["@react-three/fiber"],
                    "differ": [{"name": "react", "named": "^19.2.8",
                                "resolved": "19.3.0"}],
                    "unpinned": ["koota"]},
          "versions named vs the resolver's pins: matched, differ, unpinned",
          json.dumps(v))


def main() -> int:
    if not NODE:
        check(False, "node is on PATH (the driver's offline tests run it)")
    tests = [test_a_peer_outside_its_range_is_moved_within_what_was_asked,
             test_the_prerelease_policy,
             test_a_conflict_nothing_asked_can_move,
             test_the_dependent_moves_when_the_peer_cannot,
             test_refused_specs,
             test_the_container_command,
             test_the_arguments,
             test_the_runner_and_the_text,
             test_the_probe_reads_the_resolver] if NODE else []
    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    _srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
