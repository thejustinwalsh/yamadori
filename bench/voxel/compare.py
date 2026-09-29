#!/usr/bin/env python
"""The page the OPERATOR judges: every rep's screenshots side by side per arm,
with the objective stats, tokens and wall time. It scores nothing.

    python bench/voxel/compare.py --tag vx1            # results/vx1/index.html
    python bench/voxel/compare.py --tag vx1 --blind    # + index_blind.html

--blind shuffles the arms and labels them A, B, C ...; the blind page carries
no arm name, effort, model, token count, wall time or stack record (they
tell the arms apart), and its images and pages are COPIES under blind/ so no
path names an arm. The key goes to blind_key.json, a separate file: open it
after judging. The page is static and local (it references the gitignored
screenshots); the `page` links open model-written code in YOUR browser --
the screenshots were taken in the sandbox.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import statistics
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as runmod  # noqa: E402

DEFAULT_SHOTS = ("t05", "t10", "t15")
ORBIT_SHOTS = ("orbit_right", "orbit_high", "zoom_out")

CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#ffffff;--line:#deded8;--accent:#9a3b3b;
--ok:#2f6b3a;--bad:#9a3b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecece8;
--muted:#9c9c96;--card:#1f1f1e;--line:#33332f;--accent:#e08a8a;--ok:#8fcf98;--bad:#e08a8a}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecece8;--muted:#9c9c96;--card:#1f1f1e;--line:#33332f;
--accent:#e08a8a;--ok:#8fcf98;--bad:#e08a8a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1600px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}
.muted{color:var(--muted)}.prompt{background:var(--card);border:1px solid var(--line);
padding:10px 12px;border-radius:6px;white-space:pre-wrap}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);
padding:4px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
.grid{display:grid;gap:12px;grid-template-columns:repeat(var(--cols),minmax(280px,1fr));overflow-x:auto}
.cell{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;min-width:0}
.cell img{width:100%;height:auto;display:block;border-radius:4px;background:#000}
.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:0 10px;font-size:12px;margin-top:6px}
.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}
.head{font-weight:600;margin-bottom:6px}.bad{color:var(--bad)}.ok{color:var(--ok)}
code{font-size:12px}a{color:var(--accent)}.scroll{overflow-x:auto}
@media (max-width:700px){.grid{grid-template-columns:1fr}}
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def load(run_dir: str) -> tuple[dict, dict, dict]:
    """(manifest, (arm, rep) -> last summary row, (arm, rep, attempt) -> last check row)."""
    mp = os.path.join(run_dir, "manifest.json")
    manifest = json.load(open(mp, encoding="utf-8")) if os.path.isfile(mp) else {}
    rows = runmod.last_rows(runmod.read_rows(os.path.join(run_dir, "summary.jsonl")))
    checks = {}
    for c in runmod.read_rows(os.path.join(run_dir, "checks.jsonl")):
        checks[(c.get("arm"), c.get("rep"), c.get("attempt"))] = c
    return manifest, rows, checks


def arms_in_order(manifest: dict, rows: dict) -> list[str]:
    seen = list((manifest.get("arms") or {}).keys())
    for (arm, _rep) in rows:
        if arm not in seen:
            seen.append(arm)
    return [a for a in seen if any(k[0] == a for k in rows)]


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 1) if xs else None


def _fmt_bbox(b):
    return "x".join(f"{v:g}" for v in b) if isinstance(b, list) else None


def stat_pairs(check: dict | None, row: dict, blind: bool) -> list[tuple[str, str]]:
    c = check or {}
    out: list[tuple[str, str]] = []
    if not blind:
        u = row.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        x = row.get("x") or {}
        out += [("status", f"{row.get('status')} (finish {row.get('finish_reason')!s})"),
                ("wall", f"{row.get('wall_s')} s"),
                ("tokens", f"prompt {u.get('prompt_tokens')}, completion {u.get('completion_tokens')}"
                           f", reasoning {det.get('reasoning_tokens')}"),
                ("stack", f"tier {x.get('tier')}, route {x.get('route')}, fan-out {x.get('fanout_n')}"
                          f", deep {json.dumps(x.get('deep'))}, skills {len(x.get('skills') or [])}"
                          f", images {x.get('images')}")]
    ex = row.get("extract") or {}
    if "files_written" in ex:                  # task r3f-stack: a project, not a page
        cnt = ex.get("counts") or {}
        out.append(("project", f"{ex.get('files_written')} files ({ex.get('method')}), "
                               f"{ex.get('chars')} chars; layout {ex.get('layout')}"
                               + (f"; {cnt.get('refused')} path(s) refused" if cnt.get("refused") else "")
                               + (f"; {cnt.get('unclosed')} unclosed" if cnt.get("unclosed") else "")))
    else:
        out.append(("page", f"{ex.get('method')}, {ex.get('chars')} chars"
                            + ("" if ex.get("closed_fence") in (True, None) else ", UNCLOSED fence")
                            + ("" if ex.get("has_html_close", True) else ", no </html>")))
    if not check:
        out.append(("check", "not checked"))
        return out
    if c.get("error"):
        out.append(("check error", c.get("error")))
    if c.get("build"):
        b = c["build"]
        out.append(("build", f"{b.get('outcome')}: npm install rc {b.get('npm_install_rc')}, tsc rc "
                             f"{b.get('tsc_rc')} ({b.get('tsc_error_ts')} TS errors), build rc "
                             f"{b.get('build_rc')}"))
        for e in b.get("first_ts_errors") or []:
            out.append(("ts error", e))
    if c.get("stack"):
        s = c["stack"]
        out.append(("stack", f"{s.get('passed')}/{s.get('total')}: " + ", ".join(
            f"{k} {'yes' if v else 'no' if v is False else '?'}"
            for k, v in (s.get("checks") or {}).items())))
    if c.get("build") and not c.get("page_checked"):
        out.append(("render", "not run: no built dist/"))
        return out
    out += [("voxels", f"{c.get('voxels')} (box meshes {c.get('box_meshes')}, box instances "
                       f"{c.get('box_instances')}); non-box ~{c.get('cube_equiv_nonbox')} cube-equiv"),
            ("scene", f"{c.get('meshes')} meshes, {c.get('instanced_meshes')} instanced "
                      f"({c.get('instances')} inst), {c.get('triangles')} tris, "
                      f"{c.get('points')} point sets"),
            ("colours", f"{c.get('distinct_colors')} distinct in scene; {c.get('colours_q5_t15')} "
                        f"on screen (t15)"),
            ("extent", f"voxels {_fmt_bbox(c.get('bbox_voxels'))}; all {_fmt_bbox(c.get('bbox_all'))}"),
            ("render", f"non-blank {c.get('nonblank')}, first draw {c.get('first_draw_s')} s, "
                       f"{c.get('fps')} fps (software), shadows {c.get('shadows')}"),
            ("three.js", json.dumps(c.get("three")) if c.get("three") else "not seen"),
            ("orbit", f"{c.get('orbit')} (by {c.get('orbit_by')})"),
            ("errors", f"{c.get('page_errors')} page, {c.get('console_errors')} console"),
            ("network", f"cdn {', '.join(c.get('cdn_hosts') or []) or '-'}; blocked "
                        f"{', '.join(c.get('blocked_hosts') or []) or '-'}")]
    for e in c.get("first_errors") or []:
        out.append(("error", e))
    return out


def cell_html(label: str, shot_src, page_href: str | None, check: dict | None,
              row: dict, blind: bool, extra_links: list[tuple[str, str]] = ()) -> str:
    c = check or {}
    shots = [s for s in DEFAULT_SHOTS if shot_src(s)]
    if c.get("orbit"):
        shots += [s for s in ORBIT_SHOTS if shot_src(s)]
    hero = "t15" if "t15" in shots else (shots[0] if shots else None)
    parts = [f'<div class="cell"><div class="head">{esc(label)}</div>']
    if hero:
        parts.append(f'<a href="{esc(shot_src(hero))}"><img alt="{esc(label)} {hero}" '
                     f'src="{esc(shot_src(hero))}"></a>')
        rest = [s for s in shots if s != hero]
        if rest:
            parts.append('<div class="thumbs">' + "".join(
                f'<a href="{esc(shot_src(s))}" title="{s}"><img alt="{s}" '
                f'src="{esc(shot_src(s))}"></a>' for s in rest) + "</div>")
        if not c.get("orbit"):
            parts.append('<div class="muted">default view only (no orbit detected)</div>')
    else:
        parts.append(f'<div class="bad">no screenshot ({esc(row.get("status"))})</div>')
    parts.append('<dl class="kv">' + "".join(
        f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in stat_pairs(check, row, blind)) + "</dl>")
    links = ([("page (runs model code in your browser)", page_href)] if page_href else []) + list(extra_links)
    if links:
        parts.append('<div class="muted">' + " | ".join(
            f'<a href="{esc(h)}">{esc(t)}</a>' for t, h in links) + "</div>")
    parts.append("</div>")
    return "".join(parts)


def page_html(title: str, manifest: dict, sections: list[str], note: str) -> str:
    prompt = manifest.get("prompt") or runmod.PROMPT
    return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
            f"<h1>{esc(title)}</h1><p class=\"muted\">{note}</p>"
            f"<div class=\"prompt\">{esc(prompt)}</div>" + "".join(sections)
            + "<p class=\"muted\">Measured, not judged: bench/voxel/check.py (headless Chromium, "
              "SwiftShader software rendering -- fps ranks pages against each other, not a real "
              "card; Math.random seeded; CDN hosts only). Voxels = BoxGeometry meshes + "
              "BoxGeometry instances; non-box geometry is shown as triangles / 12.</p>"
              "</main></body></html>")


def build(run_dir: str, tag: str) -> str:
    manifest, rows, checks = load(run_dir)
    arms = arms_in_order(manifest, rows)
    reps = sorted({r for (_a, r) in rows})
    sections = []
    agg = ["<h2>Arms</h2><div class=\"scroll\"><table><tr><th>arm</th><th>sent</th><th>reps</th><th>not run</th>"
           "<th>stack errors</th><th>pages</th><th>non-blank</th><th>median wall s</th>"
           "<th>median completion tokens</th><th>median voxels</th></tr>"]
    for arm in arms:
        rs = [rows[(arm, r)] for r in reps if (arm, r) in rows]
        cs = [checks.get((arm, x["rep"], x["attempt"])) for x in rs]
        spec = (manifest.get("arms") or {}).get(arm) or {}
        agg.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
            arm, f"reasoning_effort={spec.get('effort')} model={spec.get('model')} "
                 f"features={json.dumps(spec.get('features'))}",
            len(rs), sum(x.get("status") == "not_run" for x in rs),
            sum(x.get("status") in runmod.ERRORS for x in rs),
            sum(bool(x.get("page")) for x in rs),
            sum(bool(c and c.get("nonblank")) for c in cs),
            _med([x.get("wall_s") for x in rs]),
            _med([(x.get("usage") or {}).get("completion_tokens") for x in rs]),
            _med([c.get("voxels") for c in cs if c]))) + "</tr>")
    agg.append("</table></div>")
    sections.append("".join(agg))
    sections.append("<h2>Side by side</h2>")
    one_arm = len(arms) == 1          # then the reps sit side by side in one row
    cells: list[str] = []
    for rep in reps:
        if not one_arm:
            cells = []
        for arm in arms:
            row = rows.get((arm, rep))
            if not row:
                cells.append(f'<div class="cell muted">{esc(arm)} r{rep}: not run yet</div>')
                continue
            d = row.get("dir", "")
            chk = checks.get((arm, rep, row.get("attempt")))

            def src(s, d=d):
                p = f"{d}/check/{s}.png"
                return p if os.path.isfile(os.path.join(run_dir, p)) else None
            links = [("response.json", f"{d}/response.json")]
            if row.get("project"):
                links.append(("project/", row["project"]))
                if os.path.isfile(os.path.join(run_dir, d, "build", "dist", "index.html")):
                    links.append(("built dist (runs model code in your browser; needs a "
                                  "server)", f"{d}/build/dist/index.html"))
            cells.append(cell_html(f"{arm} r{rep} (attempt {row.get('attempt')})", src,
                                   row.get("page"), chk, row, False, links))
        if not one_arm:
            sections.append(f'<div class="grid" style="--cols:{max(len(arms), 1)}">'
                            + "".join(cells) + "</div>")
    if one_arm:
        sections.append(f'<div class="grid" style="--cols:{max(min(len(reps), 3), 1)}">'
                        + "".join(cells) + "</div>")
    out = os.path.join(run_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page_html(f"Voxel scene: {tag}", manifest, sections,
                          f"Task {esc(manifest.get('task') or runmod.DEFAULT_TASK)}. "
                          "One prompt, single shot, through the proxy on :1234. "
                          + ("One arm: its reps side by side." if one_arm
                             else "Rows are reps, columns are arms.")))
    return out


def build_blind(run_dir: str, tag: str, seed: int | None = None) -> tuple[str, str]:
    manifest, rows, checks = load(run_dir)
    arms = arms_in_order(manifest, rows)
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    order = arms[:]
    random.Random(seed).shuffle(order)
    labels = {arm: string.ascii_uppercase[i] for i, arm in enumerate(order)}
    bdir = os.path.join(run_dir, "blind")
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(bdir)
    reps = sorted({r for (_a, r) in rows})
    sections = ["<h2>Side by side</h2>"]
    for rep in reps:
        cells = []
        for arm in order:
            lab = labels[arm]
            row = rows.get((arm, rep))
            if not row:
                cells.append(f'<div class="cell muted">{lab} r{rep}: not run</div>')
                continue
            d = os.path.join(run_dir, row.get("dir", ""))
            dst = os.path.join(bdir, f"{lab}-r{rep}")
            os.makedirs(dst)
            for s in DEFAULT_SHOTS + ORBIT_SHOTS:
                p = os.path.join(d, "check", f"{s}.png")
                if os.path.isfile(p):
                    shutil.copyfile(p, os.path.join(dst, f"{s}.png"))
            page = None
            if row.get("page") and os.path.isfile(os.path.join(run_dir, row["page"])):
                shutil.copyfile(os.path.join(run_dir, row["page"]), os.path.join(dst, "page.html"))
                page = f"blind/{lab}-r{rep}/page.html"

            def src(s, lab=lab, rep=rep):
                p = f"blind/{lab}-r{rep}/{s}.png"
                return p if os.path.isfile(os.path.join(run_dir, p)) else None
            cells.append(cell_html(f"{lab} r{rep}", src, page,
                                   checks.get((arm, rep, row.get("attempt"))), row, True))
        sections.append(f'<div class="grid" style="--cols:{max(len(order), 1)}">'
                        + "".join(cells) + "</div>")
    out = os.path.join(run_dir, "index_blind.html")
    blind_manifest = {"prompt": manifest.get("prompt")}
    with open(out, "w", encoding="utf-8") as f:
        f.write(page_html(f"Voxel scene: {tag} (blind)", blind_manifest, sections,
                          "Arms shuffled and labelled; the key is in blind_key.json. "
                          "Judge first, then open the key."))
    key = os.path.join(run_dir, "blind_key.json")
    with open(key, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "seed": seed, "labels": {v: k for k, v in labels.items()}}, f, indent=1)
    return out, key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="blind shuffle seed (default random)")
    a = ap.parse_args(argv)
    run_dir = os.path.join(runmod.RESULTS, a.tag)
    if not os.path.isfile(os.path.join(run_dir, "summary.jsonl")):
        print(f"  no summary.jsonl in {run_dir}")
        return 2
    print("  " + build(run_dir, a.tag))
    if a.blind:
        out, key = build_blind(run_dir, a.tag, a.seed)
        print(f"  {out}\n  key (open after judging): {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
