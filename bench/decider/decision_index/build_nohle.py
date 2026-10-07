# Our driver (not part of the harness): normalize + freeze + v2 cut for everything except HLE (gated; needs the operator's HF login/terms).
# Windows fidelity shims (the harness was written on Linux): text files written with "\n" not CRLF, and provenance paths with "/".
import builtins, gzip, io, json, pathlib, shutil, sys
_open = builtins.open
def open_lf(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *a, **k):
    if "b" not in mode and any(c in mode for c in "wax+") and newline is None:
        newline = "\n"
    return _open(file, mode, buffering, encoding, errors, newline, *a, **k)
builtins.open = open_lf
io.open = open_lf
_pop = pathlib.Path.open
def path_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
    if "b" not in mode and any(c in mode for c in "wax+") and newline is None:
        newline = "\n"
    return _pop(self, mode, buffering, encoding, errors, newline)
pathlib.Path.open = path_open
_wt = pathlib.Path.write_text
def write_text(self, data, encoding=None, errors=None, newline=None):
    return _wt(self, data, encoding=encoding, errors=errors, newline=newline or "\n")
pathlib.Path.write_text = write_text
_rel=pathlib.PurePath.relative_to
def relative_to(self,*a,**k):
    r=_rel(self,*a,**k)
    return pathlib.PurePosixPath(r.as_posix())
pathlib.PurePath.relative_to=relative_to
from decision_index import constants as C
from decision_index.suite.build import rebuild, freeze, release_v2, layout as lay
lay.Layout.rel = lambda self, path: pathlib.Path(path).relative_to(self.root).as_posix()
from decision_index.suite.build.layout import Layout
from decision_index.suite.download import hub_file
log = lambda s: print(s, flush=True)
L = Layout("work")
import decision_index.suite.build.adapters_selection as _as
_bs = builtins.sorted
def csorted(it, **k):
    it = list(it)
    if it and isinstance(it[0], pathlib.PurePath) and "key" not in k:
        return _bs(it, key=lambda p: p.as_posix())   # Linux (case-sensitive) file order; Windows' Path order is case-insensitive
    return _bs(it, **k)
_as.sorted = csorted
nums = [n for n in sorted(rebuild.BUILDERS) if n != 45]
if len(sys.argv) > 1: nums = [int(x) for x in sys.argv[1:]]
rebuild.normalize(L, nums, log=log)
out = L.suite / "release-v1-rebuilt"
m = freeze.freeze(L, out, log=log)
rows = out / "selected-rows.jsonl"
with rows.open("rb") as s, gzip.open(out / C.SUITE_ROWS_FILE, "wb", compresslevel=9) as d:
    shutil.copyfileobj(s, d)
log(json.dumps({"event": "v1", "requests": m["requests"], "sha256": m["selected_rows_sha256"], "pinned": C.SUITE_ROWS_SHA256}))
v2 = release_v2.main(L, rows, hub_file("0.2", C.SUITE_EXCLUSIONS_FILE), skip_download=True, log=log)
log(json.dumps({"event": "v2", **{k: v for k, v in v2.items() if k != "added"}, "added": {k: v for k, v in (v2.get("added") or {}).items() if k != "benchmarks"}}))
log("BUILD DONE")
