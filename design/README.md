# Yamadori design system

Source material for the dashboard rebuild. Everything here is DESIGN INTENT,
not a description of what is running.

| file | what it is |
|---|---|
| `DESIGN.md` | the design system: token frontmatter + prose rationale |
| `mockups/cockpit-desktop.html` | full cockpit, six panels, desktop |
| `mockups/cockpit-mobile.html` | the same system at phone width |
| `mockups/nebari-topology.html` | the NEBARI vector-memory view |

## READ THIS BEFORE BUILDING ANYTHING

### 1. The frontmatter is authoritative. The prose palette is stale.

`DESIGN.md` carries two different palettes and they do not agree. The
mockups settle it: they use the **frontmatter** tokens. Counted across all
three mockup files:

| prose "Colors" section | occurrences | frontmatter token | occurrences |
|---|---|---|---|
| `#0a0c10` obsidian base | **0** | `surface #0f131c` | 9 |
| `#10141d` panel | **0** | `surface-container-lowest #0a0e17` | 4 |
| `#161b26` panel 2 | **0** | `surface-container-low #181c25` | 3 |
| `#00f0ff` electric cyan | **0** | `tertiary-container #5cf2ff` | 4 |
| `#e5c07b` weathered gold | **0** | *no equivalent token exists* | — |
| `#8892b0`, `#4a5568` slate | **0** | `outline #849585` | — |
| `#ff3366` crimson | 2 | `secondary-container #d4004b` | — |

Seven of eight prose colours appear nowhere. Build from the frontmatter.
The prose is still worth reading for *rationale* — why crimson means a severed
refusal vector, why pills are banned — just not for hex values.

**Weathered gold has no token.** The prose reserves it for checkpoint
snapshots and hash validation, and the token set has no slot for it. Either
add a token or drop the concept; do not hardcode it.

### 2. Every number in the mockups is fabricated.

The mockups show `4x RTX 4090`, `96 GB VRAM pool`, a `70B` model, `8.4M`
embeddings, `128K` context, `1,180W`. **None of that is this stack.** Live
hardware is one 5060 Ti (bonsai, `-dev CUDA0`) and one A4000 (embeddings,
reranker, Laya, `-dev CUDA1`), running a 27B ternary model at `-c 147456`,
measured at 15.5-16.3 tok/s.

Wire these panels to `/dash/api/vitals` and friends, never to the mockup
values. A dashboard that displays numbers nobody measured is worse than no
dashboard: this repo has already spent a day attributing a benchmark result
to the wrong cause because a figure was read off the wrong axis.

### 3. Mockup names that do not exist in the code yet

`TAPROOT` / `BRANCH` / `SHOOT` (the three-tier memory strata in NEBARI) and
the group names `canopy` / `rootstock` / `graft` appear in the design and in
`AGENTS.md`'s naming table, but **not** in `config.yaml`, which uses
`primary` / `retrieval` / `ondemand`. They are aspirational. Treat the design
as the target and the config as the present tense; do not "fix" either to
match the other without deciding which one moves.

### 4. The mockups use zero CSS variables

All three are hardcoded hex. Extracting the frontmatter into tokens IS the
component-library work, not a preliminary to it.
