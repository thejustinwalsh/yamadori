#!/usr/bin/env python
"""Type-check a piece of code against a held package's types, in a
throwaway container. The PROVE stage's `types` check (mcp/skill_prove.py).

    python mcp/typecheck.py FILE.ts [pkg ...]    one check, printed as JSON

Operator, 2026-09-28: "wire the type check: tsc/pyright from the harness
box's tools volume (yamadori-typecheck-tools1, mounted read-only), run in a
throwaway container against the held package's types ... Record 'not run'
only when the package has no types."

HOW
  * The checkers are the harness box's pinned ones (typescript 5.9.3,
    pyright 1.1.414: bench/octopus/toolset_arms.py TOOLS_VOLUME, populated
    from the box image by `tools_volume(create=True)`), mounted read-only.
  * The container is the harness box image (node 24 + npm), `--rm`: the code,
    a tsconfig and a package.json go in on stdin (a tar), nothing is mounted
    from the host.
  * The packages are the HELD versions (index/packages/<name>@<version>.
    sqlite3, deps.db_path's naming): the skill's `package_version` when it
    has one, else the held version whose major the skill names (v10), else
    the newest held. A package whose own tarball carries no types gets its
    held `@types/<name>` too. npm installs them in the container
    (--ignore-scripts); the installed tree is kept in a named volume per
    package set (PACKAGE_VOLUME_PREFIX + a hash of the specs) so each set is
    installed once, and a later check mounts it.
  * RESULT {ran, ok, errors, why, seconds, packages, types}: `ran` False
    ("not run") only when every package the check names has no types (or
    the checker cannot run: no docker, the tools volume missing -- said so);
    a Python check has no package install here (the box image has no pip):
    a Python probe that names a package is "not run" with that reason, one
    that names none is checked by pyright against the standard library.
  * An offline suite never starts a container (YAMADORI_OFFLINE_GUARD=1 or
    YAMADORI_SKILL_TYPECHECK=0): `available()` says why.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PACKAGES_DIR = os.environ.get("YAMADORI_PKG_DIR") or os.path.join(
    ROOT, "index", "packages")
# bench/octopus/toolset_arms.py: the volume and where the terminal mounts it.
TOOLS_VOLUME = "yamadori-typecheck-tools1"
TOOLS_MOUNT = "/opt/yamadori-tools"
PACKAGE_VOLUME_PREFIX = "yamadori-prove-npm-"
TIMEOUT = int(os.environ.get("YAMADORI_TYPECHECK_TIMEOUT", "600"))
TS_LANGS = ("typescript", "tsx", "javascript", "jsx")
TSCONFIG = {"compilerOptions": {
    "target": "ES2022", "module": "ESNext", "moduleResolution": "Bundler",
    "jsx": "react-jsx", "strict": True, "noEmit": True,
    "skipLibCheck": True, "allowJs": True, "checkJs": True,
    "esModuleInterop": True, "lib": ["ES2022", "DOM", "DOM.Iterable"]}}
_DIAG = re.compile(r"^(?P<file>[^\s(]+)\((?P<line>\d+),(?P<col>\d+)\): "
                   r"error (?P<code>TS\d+): (?P<msg>.*)$")


def _box_image() -> str:
    sys.path.insert(0, os.path.join(ROOT, "bench", "sandbox"))
    import harness_box
    return harness_box.IMAGE


def _docker(args: list[str], *, stdin: bytes | None = None,
            timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=stdin,
                          capture_output=True, timeout=timeout)


def available() -> tuple[bool, str]:
    """(can a check run here, why not)."""
    if os.environ.get("YAMADORI_OFFLINE_GUARD") == "1" or \
            os.environ.get("YAMADORI_SKILL_TYPECHECK", "1") == "0":
        return False, ("an offline suite or YAMADORI_SKILL_TYPECHECK=0: no "
                       "container is started")
    try:
        v = _docker(["volume", "inspect", TOOLS_VOLUME], timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"docker is not usable here ({type(e).__name__})"
    if v.returncode != 0:
        return False, (f"the type-checker volume {TOOLS_VOLUME} is missing: "
                       "python bench/octopus/toolset_arms.py (tools_volume("
                       "create=True)) makes it from the harness box image")
    img = _docker(["image", "inspect", _box_image()], timeout=60)
    if img.returncode != 0:
        return False, (f"the harness box image {_box_image()} is not built "
                       "(bench/sandbox/harness_box.py build)")
    return True, ""


# ---------------------------------------------------------------------------
# The held versions.
# ---------------------------------------------------------------------------
def held_versions(name: str) -> list[str]:
    """The versions of `name` held in index/packages (deps.db_path naming:
    @scope/pkg -> scope__pkg)."""
    slug = name.lstrip("@").replace("/", "__")
    out = []
    try:
        files = os.listdir(PACKAGES_DIR)
    except OSError:
        return []
    for f in files:
        if f.startswith(slug + "@") and f.endswith(".sqlite3"):
            out.append(f[len(slug) + 1:-len(".sqlite3")])
    return sorted(out, key=_vkey)


def _vkey(v: str):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:-(.*))?$", v)
    if not m:
        return (0, 0, 0, 0, v)
    pre = m.group(4)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)),
            0 if pre else 1, pre or "")


def pinned(name: str, *, version: str | None = None,
           major: int | None = None) -> str | None:
    """The version to install: the given one, else the held one with the
    named major, else the newest held; None when nothing is held."""
    if version:
        return version
    held = held_versions(name)
    if major is not None:
        same = [v for v in held if v.split(".", 1)[0] == str(major)]
        if same:
            return same[-1]
    return held[-1] if held else None


def specs_for(packages: list[str], *, version: str | None = None,
              majors: dict | None = None) -> list[str]:
    """npm specs for the check: each package at its pinned version, plus
    its held @types package (installed beside it; a package that ships its
    own types ignores it). A package held at no version is installed at its
    latest (recorded)."""
    out = []
    for p in packages:
        v = pinned(p, version=version if len(packages) == 1 else None,
                   major=(majors or {}).get(p))
        out.append(f"{p}@{v}" if v else p)
        if not p.startswith("@types/"):
            t = "@types/" + p.lstrip("@").replace("/", "__")
            tv = pinned(t)
            if tv:
                out.append(f"{t}@{tv}")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# The check.
# ---------------------------------------------------------------------------
_SCRIPT = r"""
set -u
cd /work
if [ ! -f /deps/.installed ] && [ -s /work/specs ]; then
  cp /work/package.json /deps/package.json
  (cd /deps && npm install --ignore-scripts --no-audit --no-fund --loglevel=error $(cat /work/specs) >/deps/install.log 2>&1) \
    && touch /deps/.installed
fi
echo "@@INSTALL $( [ -f /deps/.installed ] && echo ok || echo failed )"
[ -f /deps/install.log ] && tail -c 1500 /deps/install.log | sed 's/^/@@LOG /'
ln -s /deps/node_modules /work/node_modules 2>/dev/null
node -e '
const fs=require("fs"),path=require("path");
const out={};
for (const p of JSON.parse(fs.readFileSync("/work/names.json","utf8"))) {
  const d=path.join("/work/node_modules",p);
  let t=false;
  try { const j=JSON.parse(fs.readFileSync(path.join(d,"package.json"),"utf8"));
        t=!!(j.types||j.typings||fs.existsSync(path.join(d,"index.d.ts"))
             ||JSON.stringify(j.exports||{}).includes("\"types\""));
  } catch(e) {}
  const at=path.join("/work/node_modules/@types",p.replace(/^@/,"").replace("/","__"));
  if (!t && fs.existsSync(at)) t=true;
  out[p]=t;
}
console.log("@@TYPES "+JSON.stringify(out));
'
echo "@@CHECK"
if [ "$LANG_KIND" = "python" ]; then
  /opt/yamadori-tools/bin/pyright --outputjson probe.py
else
  /opt/yamadori-tools/bin/tsc -p . --pretty false
fi
echo "@@RC $?"
"""


def _tar(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _volume_for(specs: list[str]) -> str:
    h = hashlib.sha256("\n".join(sorted(specs)).encode()).hexdigest()[:12]
    return PACKAGE_VOLUME_PREFIX + h


def check(code: str, lang: str | None, packages: list[str], *,
          version: str | None = None, majors: dict | None = None) -> dict:
    """{ran, ok, errors, why, seconds, packages, types}."""
    t0 = time.time()
    kind = "python" if (lang or "") == "python" else (
        "ts" if (lang or "typescript") in TS_LANGS else None)
    out: dict = {"ran": False, "ok": False, "errors": [], "why": "",
                 "packages": [], "types": {}}
    if kind is None:
        out["why"] = f"no type checker for {lang!r}"
        return out
    if kind == "python" and packages:
        out["why"] = ("a Python package's types are not installed here (the "
                      "checker image has no pip)")
        return out
    ok, why = available()
    if not ok:
        out["why"] = why
        return out
    specs = specs_for(packages, version=version, majors=majors) \
        if kind == "ts" else []
    out["packages"] = specs
    ext = {"typescript": "ts", "tsx": "tsx", "javascript": "js",
           "jsx": "jsx"}.get(lang or "typescript", "ts")
    files = {"names.json": json.dumps(list(packages)),
             "specs": " ".join(specs),
             "package.json": json.dumps({"name": "probe", "private": True,
                                         "type": "module"})}
    if kind == "python":
        files["probe.py"] = code
    else:
        files[f"probe.{ext}"] = code
        files["tsconfig.json"] = json.dumps(dict(
            TSCONFIG, include=[f"probe.{ext}"]))
    vol = _volume_for(specs) if specs else None
    args = ["run", "--rm", "-i", "--user", "node",
            "-e", f"LANG_KIND={kind}",
            "-v", f"{TOOLS_VOLUME}:{TOOLS_MOUNT}:ro"]
    if vol:
        args += ["-v", f"{vol}:/deps"]
    else:
        args += ["--network", "none", "--tmpfs", "/deps:uid=1000"]
    args += ["--entrypoint", "sh", _box_image(), "-c",
             "mkdir -p /work && cd /work && tar x && sh -c \"$0\"", _SCRIPT]
    if vol:
        # A fresh named volume is root-owned: hand it to the node user once.
        _docker(["run", "--rm", "--user", "0:0", "-v", f"{vol}:/deps",
                 "--entrypoint", "chown", _box_image(), "1000:1000", "/deps"],
                timeout=120)
    try:
        r = _docker(args, stdin=_tar(files), timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        out["why"] = f"the check ran past {TIMEOUT}s"
        out["seconds"] = round(time.time() - t0, 3)
        return out
    text = (r.stdout or b"").decode("utf-8", "replace")
    out["seconds"] = round(time.time() - t0, 3)
    m = re.search(r"^@@TYPES (.*)$", text, re.M)
    try:
        out["types"] = json.loads(m.group(1)) if m else {}
    except ValueError:
        out["types"] = {}
    inst = re.search(r"^@@INSTALL (\w+)", text, re.M)
    if specs and (not inst or inst.group(1) != "ok"):
        log = "\n".join(ln[6:] for ln in text.splitlines()
                        if ln.startswith("@@LOG "))
        out["why"] = ("not run: npm could not install "
                      + " ".join(specs) + ": " + log[-400:])
        return out
    if packages and kind == "ts" and not any(out["types"].values()):
        out["why"] = ("not run: " + ", ".join(packages) + " ships no types "
                      "and no @types package is held")
        return out
    body = text.split("@@CHECK", 1)[-1]
    rc = re.search(r"^@@RC (\d+)", body, re.M)
    errors = []
    if kind == "ts":
        for ln in body.splitlines():
            d = _DIAG.match(ln.strip())
            if d:
                errors.append(f"{d['file']}:{d['line']}:{d['col']} "
                              f"{d['code']} {d['msg']}"[:300])
    else:
        try:
            js = json.loads(body[:body.rfind("@@RC")].strip() or "{}")
            errors = [f"probe.py:{e['range']['start']['line'] + 1} "
                      f"{e.get('message', '')}"[:300]
                      for e in js.get("generalDiagnostics") or []
                      if e.get("severity") == "error"]
        except ValueError:
            pass
    if rc is None:
        out["why"] = "the checker did not finish: " + (r.stderr or b"").decode(
            "utf-8", "replace")[-300:]
        return out
    out.update(ran=True, ok=int(rc.group(1)) == 0 and not errors,
               errors=errors, why="types check" if not errors else
               f"{len(errors)} type error(s)")
    if int(rc.group(1)) != 0 and not errors:
        out["ok"] = False
        out["why"] = "the checker failed: " + body[-300:]
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path, pkgs = argv[0], argv[1:]
    with open(path, encoding="utf-8") as f:
        code = f.read()
    lang = "python" if path.endswith(".py") else (
        "tsx" if path.endswith(".tsx") else "typescript")
    print(json.dumps(check(code, lang, pkgs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
