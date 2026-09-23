---
name: Yamadori Bonsai Engine
colors:
  surface: '#0f131c'
  surface-dim: '#0f131c'
  surface-bright: '#353943'
  surface-container-lowest: '#0a0e17'
  surface-container-low: '#181c25'
  surface-container: '#1c2029'
  surface-container-high: '#262a34'
  surface-container-highest: '#31353f'
  on-surface: '#dfe2ef'
  on-surface-variant: '#b9cbb9'
  inverse-surface: '#dfe2ef'
  inverse-on-surface: '#2c303a'
  outline: '#849585'
  outline-variant: '#3b4b3d'
  surface-tint: '#00e479'
  primary: '#f1ffef'
  on-primary: '#003919'
  primary-container: '#00ff88'
  on-primary-container: '#007139'
  inverse-primary: '#006d37'
  secondary: '#ffb2ba'
  on-secondary: '#670020'
  secondary-container: '#d4004b'
  on-secondary-container: '#ffe6e8'
  tertiary: '#eefeff'
  on-tertiary: '#00363a'
  tertiary-container: '#5cf2ff'
  on-tertiary-container: '#006d74'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#60ff99'
  primary-fixed-dim: '#00e479'
  on-primary-fixed: '#00210c'
  on-primary-fixed-variant: '#005228'
  secondary-fixed: '#ffd9dc'
  secondary-fixed-dim: '#ffb2ba'
  on-secondary-fixed: '#400011'
  on-secondary-fixed-variant: '#910030'
  tertiary-fixed: '#7df4ff'
  tertiary-fixed-dim: '#00dbe9'
  on-tertiary-fixed: '#002022'
  on-tertiary-fixed-variant: '#004f54'
  background: '#0f131c'
  on-background: '#dfe2ef'
  surface-variant: '#31353f'
typography:
  headline-xl:
    fontFamily: Syne
    fontSize: 36px
    fontWeight: '800'
    lineHeight: 42px
    letterSpacing: -0.04em
  headline-xl-mobile:
    fontFamily: Syne
    fontSize: 28px
    fontWeight: '800'
    lineHeight: 34px
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Syne
    fontSize: 26px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Syne
    fontSize: 18px
    fontWeight: '700'
    lineHeight: 24px
    letterSpacing: 0em
  title-md:
    fontFamily: JetBrains Mono
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: -0.01em
  body-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.06em
  label-xs:
    fontFamily: JetBrains Mono
    fontSize: 9px
    fontWeight: '700'
    lineHeight: 12px
    letterSpacing: 0.12em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.75rem
  margin: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system expresses a paradox: the ancient discipline of Japanese bonsai sculpture (*yamadori* — trees collected from the wild, shaped by harsh elements and carved into living artifacts) integrated with ruthless, post-alignment cyberpunk edge compute. It caters to machine learning researchers, self-hosters, and hardware minimalists who run uncensored, abliterated local weights on personal silicon.

The visual dialect rejects both corporate sterile dashboard tropes and juvenile neon clutter. Instead, it deploys a disciplined, high-density terminal style where deep *sumi-e* ink substrates meet surgical, monospaced diagnostic telemetry. The aesthetic balances natural weathered decay (*wabi-sabi* textures, dry-brushed dividers, muted deadwood slates) with intense, high-luminescence bioluminescent laser accents representing live weights, tensor flow, and directional vector ablation. Every screen feels like a sacred gardening instrument forged inside an orbital server silo.

## Colors

The palette is engineered strictly for high-fidelity dark mode execution on mobile OLED panels to ensure maximum black preservation and laser-sharp contrast.

- **Obsidian Substrates (`#0a0c10`, `#10141d`)**: Deep sumi-ink foundations. Base screens sit on `#0a0c10`, while functional panels, surface containers, and card layers elevate into rich slate-tinted charcoal (`#10141d` and `#161b26`).
- **Moss Bioluminescence / Primary (`#00ff88`)**: Represents vitality, active parameter retention, healthy quantization states, and high memory bandwidth efficiency.
- **Abliteration Crimson / Secondary (`#ff3366`)**: Represents severed refusal vectors, clipped residual connections, thermal throttling warnings, and deliberate unaligned tensor surgeries.
- **Electric Cyan / Tertiary (`#00f0ff`)**: Direct memory allocation metrics, token streaming throughput, and prompt-processing state.
- **Weathered Gold / Accent (`#e5c07b`)**: Evokes ancient brass and *jin* (carved deadwood), reserved for checkpoint snapshots, model hash validation, and active epoch markers.
- **Deadwood Slate (`#4a5568`, `#8892b0`)**: Structural framing, inactive parameter matrices, ghost gridlines, and low-priority system diagnostics.

## Typography

Typography establishes an intentional contrast between organic sculptural architecture and relentless machine precision:

1. **Syne** delivers sculptural, avant-garde elegance across primary displays, model signatures, and telemetry cluster titles. Its expansive glyph shapes echo modernist architectural stone and twisted branch forms.
2. **JetBrains Mono** governs all functional UI, runtime outputs, parameter values, layer trees, and memory offsets. Highly legible ligatures and rigid tabular proportions anchor rapid mobile triage without perceptual jitter during constant parameter streaming.

Labels utilize uppercase tracking (`letter-spacing: 0.06em` to `0.12em`) paired with microscopic badge styles to mimic flight recorder readouts and CNC carving blueprints.

## Layout & Spacing

Engineered specifically for handheld operations and terminal viewport constraints, the layout utilizes an ultra-dense 4-column fluid mobile grid anchored by consistent 16px (`1rem`) outer canvas margins and tight 12px (`0.75rem`) internal column gutters.

Vertical rhythm is governed by a baseline unit of 4px. Component spacing favors immediate data availability: compact paddings (`space-xs` and `space-sm`) allow dense clusters of telemetry cards, VRAM distribution charts, and tensor sparsity matrices to sit above the thumb fold without requiring vertical scrolling for routine diagnostics. Complex analytical structures—like layer ablation timelines—collapse along a central vertical axis resembling a disciplined tree trunk (*chokkan*).

## Elevation & Depth

Visual hierarchy rejects standard diffused drop shadows in favor of razor-thin structural barriers, tonal layering, and targeted bioluminescent emission:

- **Substrate Stacking**: True black floor (`#0a0c10`) &rarr; Container level 1 (`#10141d`) &rarr; Container level 2 (`#161b26`).
- **Low-Contrast Perimeter Etchings**: All cards and dialogs are bound by a 1px boundary of `#283244` or `#4a5568` at 35% opacity, evoking fine joinery.
- **Laser Edge Indicators**: Focused elements or critical warnings receive a unidirectional 1px glowing edge border (e.g., `border-left: 2px solid #00ff88` with a tight `0 0 10px rgba(0, 255, 136, 0.3)` back-glow).
- **Sub-Surface Mesh Blur**: Modal dialogs and floating action overlays apply `backdrop-filter: blur(12px)` coupled with an obsidian wash (`rgba(10, 12, 16, 0.85)`), allowing background telemetry indicators to faintly bleed through like streetlights through shoji paper.

## Shapes

The shape system adopts a minimal soft radius (`0.25rem` / `4px`) with occasional hard-chamfered geometric finishes (`0px`). Curves are kept exceptionally taut to reinforce cold hardware architecture and the sharp incisions of carving chisels (*jin tools*). Pill shapes are banned entirely; interactive buttons, status tags, and sparkline containers retain disciplined rectilinear silhouettes.

## Components

### Buttons & Surgical Triggers
- **Primary (Execute / Ablate)**: Jet-black background bordered by `#00ff88`, Syne/JetBrains typography rendered in `#00ff88`. Hover and active states fill the background with a 15% `#00ff88` flood with an immediate crisp inner border.
- **Destructive (Prune / Purge Layer)**: Bordered by `#ff3366`, typography in crimson, accompanied by a monospace warning icon `[ ! ]`.
- **Ghost Action**: Unbordered `#8892b0` label on `#161b26` background; responds to press with instantaneous flash to neutral silver.

### Telemetry Cards & Prune Meters
- **Container Structure**: 1px ghost perimeter (`#283244`), background `#10141d`, padded with `space-md`. Top-right corners contain monospaced index flags (e.g., `L-32 // ATTN_O`).
- **Prune & Sparsity Meter**: Horizontal segmented bar displaying weight reduction. Active surviving weights emit `#00ff88`; abliterated/culled parameter segments transition into hatch-patterned `#4a5568` or severed `#ff3366` ticks.

### Root Flare Graph (Weight Topology)
- Specialized tree-node network showing neural network depth and pruning cuts. Nodes are 6px squares: retained channels glow `#00f0ff`, deadwood nodes are dimmed to `#283244` crosses, and manipulated residual vectors pulse in crimson `#ff3366`.

### Inputs & Terminal Toggles
- **Hex/Weight Modifier Fields**: Recessed obsidian wells (`#0a0c10`) with inset top shadow and a left-aligned blinking caret `_`. Monospace text enters in `#00f0ff`.
- **Status Switches**: Rectangular toggle boxes; off state displays `[OFF]` in `#4a5568`, on state illuminates `[ACTIVE]` in `#00ff88` with a hard inner pip.

### Chips & Monospace Badges
- Ultra-compact tags (`label-xs`) built with a 1px border. Examples: `FP8_E4M3`, `VRAM: 14.2GB`, `ABL_VEC_7`. No rounded pills; pure architectural rectangles.