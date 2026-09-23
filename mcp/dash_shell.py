#!/usr/bin/env python
"""One chrome for every dashboard page, so three pages cannot drift apart.

WHAT THIS IS

`DESIGN.md` is the system; this file is the system compiled to CSS. Four
pages -- the dataset manager, corpus review, vitals, benchmark results -- are
written by different hands at different times, and the failure mode is not
that one of them looks wrong. It is that each looks fine alone and the set
looks like four products.
So the tokens, the document head and the navigation live in exactly one place
and the pages import them.

THE INTERFACE, WHICH IS DELIBERATELY FOUR NAMES

    TOKENS        the design system as custom properties on :root
    BASE          element and component styles that read those properties
    head(title)   doctype, meta, <title>, and <style>TOKENS + BASE</style>
    nav(active)   the header wordmark and the nav destinations in NAV

A page is `head("Vitals") + nav("vitals") + "<main>...</main>"`. Nothing else
is needed and nothing else is exported, because an interface a second author
has to read the source of is not an interface.

TWO MEASUREMENTS THAT CHANGED THE TOKENS

`--outline-variant: #516854` is the corrected border colour from DESIGN.md and
it measures **3.06:1 against the #0f131c ground** -- which is the whole margin
it has. Against the substrates that sit above the ground it does not pass:
2.81:1 on `#181c25`, 2.68:1 on `#1c2029`, 2.36:1 on `#262a34`. A panel edge
has the ground on one side and passes; a line *inside* a panel has neither
side on the ground and fails. So there are two border tokens, and which one
to use is decided by what the line sits on, not by taste:

    --border         panel edges, where the ground is one of the two sides
    --border-raised  any line on an elevated substrate (inputs, badges, cells)

Both are full-opacity colours. Never apply `opacity` to a border: composited,
`#516854` at 35% lands at 1.38:1 and the token stops meaning anything.

`--secondary-container: #d4004b` is a boundary colour, not a fill for text.
`#dfe2ef` on it is 4.17:1, under the 4.5:1 floor. As a 1px line or a state bar
it measures 3.45:1 on the ground and 3.17:1 on a panel and passes the 3:1 that
a non-text boundary owes. Destructive *text* is `--secondary #ffb2ba`.

THE FIVE STATES

Every panel ships loading, empty, ready, stale and error. `BASE` carries the
classes; the page supplies the words, because only the page knows which file
is missing or which endpoint failed:

    <div class="state">
      <p class="state-title">no benchmark results yet</p>
      <p class="state-detail">run bench/livecodebench.py --tiers</p>
    </div>

`.state.is-error` for a fault, `.state.is-loading` for the wait. Staleness is
`.is-stale` on the panel that went stale plus an `.age` readout carrying the
number of seconds, because a snapshot shown as current is the same failure as
an invented number.
"""
from __future__ import annotations

# Nav destinations, in the order an operator moves through them: what goes
# into the corpus, what the corpus says, whether the machine is alive, what
# the benchmark measured. DATA leads because it is where a corpus starts;
# CORPUS is still the review surface and is unchanged.
NAV = (
    ("data", "DATA", "/dash/data"),
    ("corpus", "CORPUS", "/dash"),
    ("vitals", "VITALS", "/dash/vitals"),
    ("results", "RESULTS", "/dash/results"),
)

# ---------------------------------------------------------------------------
# Tokens. Values are DESIGN.md's frontmatter verbatim; the only additions are
# the two border aliases and the motion duration, both explained above.
# ---------------------------------------------------------------------------
TOKENS = """
:root{
  color-scheme: dark;

  /* Substrate. The ground is #0f131c, never black: pure black crushes the
     elevation cues and leaves a shadow nowhere to go. Elevation is substrate
     and a 1px line, never a drop shadow. */
  --background: #0f131c;
  --surface-container-lowest: #0a0e17;
  --surface-container-low: #181c25;
  --surface-container: #1c2029;
  --surface-container-high: #262a34;
  --surface-container-highest: #31353f;

  /* Text. 14.39:1, 10.89:1 and 5.86:1 on the ground respectively. */
  --on-surface: #dfe2ef;
  --on-surface-variant: #b9cbb9;
  --outline: #849585;

  /* Lines, at full opacity always. See the module docstring for why there
     are two: #516854 passes 3:1 only where the ground is one of the sides. */
  --outline-variant: #516854;
  --border: var(--outline-variant);
  --border-raised: var(--outline);

  /* Four accents, each with one job. A hue without a job is paint.
       primary    alive, healthy, the primary action -- and nothing else
       tertiary   measurement in flight; also the focus ring
       secondary  destructive: -container is the line, plain is the text
       error      a fault the operator has to resolve */
  --primary-container: #00ff88;
  --tertiary-container: #5cf2ff;
  --secondary: #ffb2ba;
  --secondary-container: #d4004b;
  --error: #ffb4ab;

  /* Two families with one job each: Syne for headings so the display face is
     not also the data face, JetBrains Mono for machine output -- which here
     is honest rather than costume, because the content genuinely is machine
     output. Neither is fetched: a local instrument panel should not block on
     a font CDN, and the fallbacks are metrically close enough. */
  --font-display: 'Syne', 'Futura', 'Avenir Next', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono',
               Menlo, Consolas, monospace;

  /* The ladder. 9px is gone; 11px is the floor and it is a label, never body
     text. Body sits at 400 because light-on-dark needs the weight step. */
  --text-display: 28px;
  --text-heading: 18px;
  --text-title: 15px;
  --text-body: 14px;
  --text-small: 12px;
  --text-label: 11px;
  --track-display: -0.03em;
  --track-heading: 0em;
  --track-title: -0.01em;
  --track-body: -0.01em;
  --track-label: 0.06em;

  /* Baseline 4px. Unrelated groups are separated by space, not by a line;
     lines are the last resort, after space has failed. */
  --space-xs: 0.25rem;
  --space-sm: 0.5rem;
  --space-md: 0.75rem;
  --space-lg: 1.25rem;
  --space-xl: 2rem;

  --radius: 0.125rem;
  --radius-lg: 0.25rem;

  /* One authored moment per view, 120-200ms, ease-out. */
  --motion: 140ms;
  --ease: cubic-bezier(0.2, 0.8, 0.3, 1);
}
"""

# ---------------------------------------------------------------------------
# Base element and component styles. Everything here reads a token; no page
# should ever need a raw hex value.
# ---------------------------------------------------------------------------
BASE = """
*,*::before,*::after{box-sizing:border-box}

body{
  margin:0;
  padding:var(--space-md);
  background:var(--background);
  color:var(--on-surface);
  font-family:var(--font-mono);
  font-size:var(--text-body);
  font-weight:400;
  line-height:1.55;
  letter-spacing:var(--track-body);
  /* A changing number must not shift the column it lives in. */
  font-variant-numeric:tabular-nums;
  -webkit-font-smoothing:antialiased;
}

/* ---- type ladder --------------------------------------------------- */
.display{font-family:var(--font-display);font-size:var(--text-display);
  font-weight:800;letter-spacing:var(--track-display);line-height:1.1;margin:0}
.heading{font-family:var(--font-display);font-size:var(--text-heading);
  font-weight:700;letter-spacing:var(--track-heading);line-height:1.25;margin:0}
.title{font-size:var(--text-title);font-weight:600;
  letter-spacing:var(--track-title);margin:0}
.small{font-size:var(--text-small);line-height:1.5}
.label{font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);text-transform:uppercase;
  color:var(--outline);line-height:1.5}
.dim{color:var(--on-surface-variant)}
.num{font-size:var(--text-title);font-weight:600;color:var(--on-surface)}

a{color:var(--on-surface-variant);text-decoration-color:var(--outline);
  text-underline-offset:2px}
a:hover{color:var(--on-surface)}

/* ---- shell --------------------------------------------------------- */
.shell-head{display:flex;align-items:flex-end;justify-content:space-between;
  gap:var(--space-lg);flex-wrap:wrap;
  padding-bottom:var(--space-md);margin-bottom:var(--space-lg);
  border-bottom:1px solid var(--border)}
.shell-brand{display:flex;align-items:baseline;gap:var(--space-sm)}
.shell-mark{font-family:var(--font-display);font-size:var(--text-heading);
  font-weight:800;letter-spacing:0.02em;color:var(--on-surface)}
.shell-nav{display:flex;gap:var(--space-xs)}
.shell-nav a{display:inline-flex;align-items:center;min-height:32px;
  padding:0 var(--space-md);
  font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);text-transform:uppercase;
  color:var(--outline);text-decoration:none;
  border-bottom:2px solid transparent;
  transition:color var(--motion) var(--ease),
             border-color var(--motion) var(--ease)}
.shell-nav a:hover{color:var(--on-surface-variant);
  border-bottom-color:var(--border-raised)}
.shell-nav a[aria-current="page"]{color:var(--on-surface);
  border-bottom-color:var(--on-surface)}

/* ---- panels: substrate plus one full-opacity line ------------------ */
.panel{background:var(--surface-container-low);
  border:1px solid var(--border);border-radius:var(--radius);
  padding:var(--space-md) var(--space-lg)}
.panel + .panel{margin-top:var(--space-md)}
.group{margin-bottom:var(--space-lg)}
.panel-head{display:flex;align-items:baseline;justify-content:space-between;
  gap:var(--space-md);margin-bottom:var(--space-sm)}

/* ---- controls ------------------------------------------------------ */
input,select,textarea,button{font:inherit;letter-spacing:inherit;
  font-variant-numeric:tabular-nums;
  color:var(--on-surface);
  background:var(--surface-container);
  border:1px solid var(--border-raised);
  border-radius:var(--radius);
  padding:0 var(--space-sm);
  min-height:32px}
textarea{padding:var(--space-sm);line-height:1.55;resize:vertical;width:100%}
/* The checkbox takes the interaction hue, not the green: a filter toggle is
   not the primary action, and green that appears on ordinary controls stops
   marking the one control that is. */
input[type=checkbox]{min-height:0;width:16px;height:16px;padding:0;
  accent-color:var(--tertiary-container);vertical-align:-2px}
select:hover,input:hover,textarea:hover{border-color:var(--on-surface-variant)}
/* Placeholder is text and owes 4.5:1 like any other. The UA default is a
   mid grey that does not clear it, and Firefox dims it again with an opacity
   the token cannot see, so both are overridden. */
::placeholder{color:var(--outline);opacity:1}
button{cursor:pointer;padding:0 var(--space-md);
  transition:border-color var(--motion) var(--ease),
             background-color var(--motion) var(--ease)}
button:hover{border-color:var(--on-surface-variant)}

/* Green is the primary action and nothing else. */
.btn-primary{background:var(--primary-container);
  border-color:var(--primary-container);
  color:var(--surface-container-lowest);font-weight:600}
.btn-primary:hover{background:var(--primary-container);
  border-color:var(--on-surface)}
/* Destructive: the line carries #d4004b, the text carries #ffb2ba, because
   #dfe2ef on #d4004b is 4.17:1 and does not clear the body floor. */
.btn-danger{color:var(--secondary);border-color:var(--secondary-container)}
.btn-danger:hover{border-color:var(--secondary)}

/* Focus is always visible, 3:1 against its surround, and never removed. */
:focus-visible{outline:2px solid var(--tertiary-container);outline-offset:2px}

/* WCAG 2.5.8 puts the floor at 24x24; 44x44 is 2.5.5 at AAA and is what a
   finger actually needs. Coarse pointers get the AAA number. */
@media (pointer:coarse){
  button,select,input:not([type=checkbox]){min-height:44px}
  input[type=checkbox]{width:24px;height:24px}
  .shell-nav a{min-height:44px}
}

/* ---- badges and chips ---------------------------------------------- */
.badge,.chip{display:inline-flex;align-items:center;gap:var(--space-xs);
  font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);line-height:1.6;
  padding:1px var(--space-xs);border-radius:var(--radius);
  border:1px solid var(--border-raised);
  background:var(--surface-container-high);
  color:var(--on-surface-variant);white-space:nowrap}
.chip{background:transparent}
.chip-live{color:var(--primary-container);border-color:var(--primary-container)}
.chip-flight{color:var(--tertiary-container);border-color:var(--tertiary-container)}
.chip-cut{color:var(--secondary);border-color:var(--secondary-container)}
.chip-fault{color:var(--error);border-color:var(--error)}

/* ---- the five states ----------------------------------------------- */
.state{padding:var(--space-lg);border:1px solid var(--border);
  border-radius:var(--radius);background:var(--surface-container-low);
  margin-bottom:var(--space-md)}
.state-title{margin:0 0 var(--space-xs);font-size:var(--text-title);
  font-weight:600;letter-spacing:var(--track-title);color:var(--on-surface)}
.state-detail{margin:0;color:var(--on-surface-variant);
  font-size:var(--text-small)}
.state.is-error{border-color:var(--error)}
.state.is-error .state-title{color:var(--error)}
.state.is-loading .state-title{color:var(--on-surface-variant)}

/* Staleness is mandatory: every panel carries its age and greys past 30s. */
.age{font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);text-transform:uppercase;
  color:var(--outline)}
.is-stale{border-color:var(--border-raised)}
.is-stale .num,.is-stale .value{color:var(--on-surface-variant)}
.age.is-stale{color:var(--error)}

/* Pulses and scanlines are style, not law, and they go first. Nothing on
   any page depends on motion to be legible. */
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{
    animation-duration:0.001ms !important;animation-iteration-count:1 !important;
    transition-duration:0.001ms !important;scroll-behavior:auto !important}
}
"""


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def head(title: str) -> str:
    """Doctype, meta, title and the whole design system, as one string.

    Returns everything before the body content. Concatenate `nav(...)` and
    the page's own markup after it.
    """
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{_esc(title)} - yamadori</title>\n"
        f"<style>{TOKENS}{BASE}</style>\n"
    )


def nav(active: str) -> str:
    """The header: wordmark plus every destination in NAV.

    `active` is a key from NAV ("data", "corpus", "vitals", "results") or the path of
    the current page; anything unrecognised simply marks nothing active,
    which is better than marking the wrong thing.
    """
    key = (active or "").strip().lower().rstrip("/")
    links = []
    for name, label, href in NAV:
        here = key in (name, href, href.lstrip("/"))
        cur = ' aria-current="page"' if here else ""
        links.append(f'<a href="{href}"{cur}>{label}</a>')
    return (
        '<header class="shell-head">\n'
        '  <div class="shell-brand">'
        '<span class="shell-mark">YAMADORI</span>'
        '<span class="label">instruments</span></div>\n'
        '  <nav class="shell-nav" aria-label="sections">'
        + "".join(links) +
        "</nav>\n"
        "</header>\n"
    )



# ---------------------------------------------------------------------------
# The nav, standalone, for a page that already carries its own stylesheet.
#
# `dash_vitals` and `dash_results` were each built with a complete copy of the
# tokens, and their contrast pairs were measured against those copies. Pulling
# BASE in beside them would mean two stylesheets defining the same selectors,
# which is how a verified page quietly stops being the page that was verified.
#
# So the nav declares the four values it needs on `.shell-head` itself, which
# makes it independent of whether the host page calls its text `--on-surface`
# or `--fg`. The values are the same measured tokens: #849585 on the #0f131c
# ground is 6.05:1, #dfe2ef is 14.4:1, and the 1px rule is the panel-edge
# #516854 at 3.06:1, at full opacity like every other line in this system.
# ---------------------------------------------------------------------------
NAV_CSS = """
.shell-head{--nav-fg:#dfe2ef;--nav-dim:#849585;--nav-line:#516854;
  --nav-hover:#b9cbb9;
  display:flex;align-items:flex-end;justify-content:space-between;
  gap:1.25rem;flex-wrap:wrap;
  padding:0 0 0.75rem;margin:0 0 1.25rem;
  border-bottom:1px solid var(--nav-line)}
.shell-head .shell-brand{display:flex;align-items:baseline;gap:0.5rem}
.shell-head .shell-mark{font-family:Syne,ui-monospace,monospace;font-size:18px;
  font-weight:800;letter-spacing:0.02em;color:var(--nav-fg)}
.shell-head .label{font-size:11px;font-weight:500;letter-spacing:0.06em;
  text-transform:uppercase;color:var(--nav-dim)}
.shell-head .shell-nav{display:flex;gap:0.25rem}
.shell-head .shell-nav a{display:inline-flex;align-items:center;min-height:32px;
  padding:0 0.75rem;font-size:11px;font-weight:500;letter-spacing:0.06em;
  text-transform:uppercase;color:var(--nav-dim);text-decoration:none;
  border-bottom:2px solid transparent;
  transition:color 160ms ease-out,border-color 160ms ease-out}
.shell-head .shell-nav a:hover{color:var(--nav-hover);
  border-bottom-color:var(--nav-dim)}
.shell-head .shell-nav a[aria-current="page"]{color:var(--nav-fg);
  border-bottom-color:var(--nav-fg)}
.shell-head .shell-nav a:focus-visible{outline:2px solid #5cf2ff;
  outline-offset:2px}
@media (pointer:coarse){.shell-head .shell-nav a{min-height:44px}}
@media (prefers-reduced-motion:reduce){
  .shell-head .shell-nav a{transition:none}}
"""


def nav_block(active: str) -> str:
    """`nav(active)` plus only the CSS it needs, for a self-styled page."""
    return "<style>" + NAV_CSS + "</style>" + nav(active)

if __name__ == "__main__":
    print(head("Shell") + nav("corpus"))
