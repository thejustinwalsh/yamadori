---
name: Yamadori
surface:
  background: '#0f131c'
  surface-container-lowest: '#0a0e17'
  surface-container-low: '#181c25'
  surface-container: '#1c2029'
  surface-container-high: '#262a34'
  surface-container-highest: '#31353f'
text:
  on-surface: '#dfe2ef'
  on-surface-variant: '#b9cbb9'
  outline: '#849585'
line:
  outline-variant: '#516854'
accent:
  primary-container: '#00ff88'
  tertiary-container: '#5cf2ff'
  secondary: '#ffb2ba'
  secondary-container: '#d4004b'
  error: '#ffb4ab'
type:
  display: { family: Syne, size: 28px, weight: 800, tracking: -0.03em }
  heading: { family: Syne, size: 18px, weight: 700, tracking: 0em }
  title: { family: JetBrains Mono, size: 15px, weight: 600, tracking: -0.01em }
  body: { family: JetBrains Mono, size: 14px, weight: 400, tracking: -0.01em }
  small: { family: JetBrains Mono, size: 12px, weight: 400, tracking: 0em }
  label: { family: JetBrains Mono, size: 11px, weight: 500, tracking: 0.06em }
space: { xs: 0.25rem, sm: 0.5rem, md: 0.75rem, lg: 1.25rem, xl: 2rem }
radius: { DEFAULT: 0.125rem, lg: 0.25rem }
---

# Yamadori design system

A wild tree collected from a mountain and kept alive under instruments. Deep
ink substrates, monospaced telemetry, one living green. The aesthetic is
inherited from the Stitch exploration; what follows is what changed after the
rules were applied and the numbers were computed.

## The rule that outranks the rest

**Every number on screen is measured or it is not shown.**

The source mockup displayed `ABL: 99.4%`, `4x 4090`, `70.6B -> 40.4B ACTIVE`,
`1,429 bleached deadwood branches`, `99.82% recall @10`. None of it is this
system. We run one 27B at PTQ1_0 **on a single card** — `-dev CUDA0`, the
5060 Ti — with a 147,456-token pool. Two cards are present, but the second
carries retrieval and Laya, not half the generator: splitting the 27B across
both was tried and rejected, because every token then crosses PCIe onto the
slower card and buys context by spending throughput (`config.yaml`, the
comment above `-dev CUDA0`).
Invented telemetry in a mockup is a placeholder; invented telemetry in a
running dashboard is a lie told in a monospace font, which is the most
credible font there is.

A panel with no data renders the empty state and says what is missing. It does
not render a plausible number.

## Corrections made to the source, with the measurements behind them

**Border contrast failed.** `outline-variant #3b4b3d` measures **2.00:1**
against the `#0f131c` ground; non-text UI boundaries need **3.0:1**. Replaced
with `#516854` at **3.06:1**, same hue, ten percent more lightness.

**Opacity was the larger bug.** The mockup applied borders as
`border-outline-variant/35`. Composited, that is a different colour than the
token: `#3b4b3d` at 35% becomes `#1e2728` at **1.22:1**, and even the
corrected `#516854` lands at **1.38:1**. Opacity is not a styling shortcut on
anything with a contrast requirement. **Borders are set at full opacity, and
the token itself carries the value.** Use opacity for overlays and scrims,
never for a line that has to be seen.

**The frontmatter and the prose disagreed.** The source prose named
`#0a0c10`, `#ff3366`, `#00f0ff` and a weathered gold `#e5c07b`; its own
frontmatter said `#0a0e17`, `#d4004b`, `#5cf2ff` and had no gold at all. The
frontmatter wins, because it is the half that was measured and nine of its ten
pairs pass. The gold is dropped rather than invented -- a fifth accent hue
needs a job, and it had none.

**The primary colour was spent on decoration.** Green appeared on borders,
pills, meters, headings and buttons at once. It is now reserved for **live,
healthy, primary action**. When everything is the action colour, nothing is
findable.

## Colour, and what each hue is allowed to mean

Four accents, each with a stated job. A hue without a job is paint.

| token | job |
|---|---|
| `primary-container #00ff88` | alive, healthy, the primary action |
| `tertiary-container #5cf2ff` | throughput and measurement in flight |
| `secondary-container #d4004b` | destructive action, refused, cut |
| `error #ffb4ab` | a fault the operator must resolve |

Neutrals stay neutral. The corpus is explicit that reflexive tinting toward a
brand hue is a tic rather than a technique, and untinted grey is a legitimate
choice. These greys have a faint green cast already; that is enough.

**The ground is `#0f131c`, not black.** Pure black crushes elevation cues and
leaves shadows nowhere to go. Elevation is built by substrate, never by
drop shadow: `#0a0e17` floor, `#181c25` panel, `#1c2029` raised, `#262a34`
popover.

## Type

Two families, each with a stated job. Syne for headings, because the display
face should not also be the data face. **JetBrains Mono for everything that is
machine output** -- and here that is the legitimate use rather than costume,
because the content genuinely is machine output. Tabular figures throughout,
so a changing number does not shift the column.

The source ladder ran 9, 11, 12, 14, 15px. **Three of those steps are within
two pixels of a neighbour, which a reader cannot resolve**, so it buys
hierarchy nobody perceives. The 9px tier is gone; 11px is the floor, and it is
a label, never body text. Light-on-dark also needs compensation on three axes
at once -- slightly more line height, a touch more tracking, one weight step
down -- which is why body sits at 400 and not 300.

## Density

This is an instrument panel, and density is the point: an operator should see
GPU headroom, the context pool and the benchmark state without scrolling. But
density is earned by removing content, not by shrinking type below the
legible floor or removing the space between unrelated things.

Baseline 4px. Gutters `space-md`, outer margin `space-md`. Related metrics sit
in one bordered group; unrelated groups are separated by `space-lg`, not by a
divider line. Lines are the last resort, after space has failed.

## Shape and depth

Radius `0.125rem`, occasionally `0.25rem`. No pills. Taut corners read as
instrument, and the vocabulary is carving tools rather than soft product UI.

Elevation is **substrate and a 1px line at full opacity**. Glow is reserved
for the single live element in a view -- one focus, one running job -- and is
never ambient decoration. A dashboard where everything glows has no focus, and
the glow stops meaning anything.

## States, which are the part that gets skipped

Every panel ships five: **loading, empty, ready, stale, error**.

- **Empty** says what is missing and how to produce it, e.g. "no benchmark
  results yet -- run `bench/livecodebench.py --tiers`".
- **Stale** is mandatory and specific to this system. Telemetry is a snapshot;
  a number from four minutes ago shown as current is the same failure as an
  invented one. Every panel carries its age, and greys out past 30 seconds.
- **Error** names the failing component and what to check, never "something
  went wrong".

## Accessibility, as a constraint rather than a pass

- Body text 4.5:1, large text and controls 3:1, **computed, not eyeballed**,
  including every hover, focus, disabled and overlay state.
- Focus is always visible, 3:1 against its surround, and never removed.
- Touch targets **24x24 CSS px is the WCAG 2.2 AA floor** (SC 2.5.8). 44x44 is
  SC 2.5.5 at **AAA**, and Apple wants 44pt while Material wants 48dp. These
  are four different numbers from four authorities; anyone citing "44px, WCAG"
  is quoting AAA without knowing it. Target **44x44 on touch**, and treat
  24x24 as the floor a control may never go below.
- No meaning by colour alone: a red border carries a word too.
- Respect `prefers-reduced-motion`. Pulses and scanlines are the first things
  to go, and nothing depends on them to be legible.

## Motion

One authored moment per view at most. Transitions 120-200ms, ease-out.
Telemetry that pulses continuously is noise that trains the operator to stop
looking, which is the opposite of what an instrument is for. Animate to draw
attention to a change that happened; never to decorate a steady state.

## What is deliberately contested

The source's scanline overlay, continuous pulse animations and the ambient
glow are **style, not law**. They are cheap to remove and they cost contrast
and motion-sensitivity. They stay because the aesthetic is the point, but they
are gated behind `prefers-reduced-motion` and they never carry meaning.
