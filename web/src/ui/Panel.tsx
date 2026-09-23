// The telemetry card from the mockups: surface-container-low, a 1px
// outline-variant perimeter at 35%, kanji + romaji title, a tag chip, an
// index flag top-right, and an optional one-directional "laser edge".
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { colors, space } from '../tokens/tokens.stylex';
import { Chip, type Tone } from './primitives';
import { text } from './text';

const s = stylex.create({
  panel: {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceSm,
    padding: space.spaceMd,
    backgroundColor: colors.surfaceContainerLow,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 35%, transparent)`,
    minWidth: 0,
    // Corner ticks: CNC-blueprint registration marks, drawn with gradients.
    backgroundImage: `linear-gradient(${colors.outline}, ${colors.outline}), linear-gradient(${colors.outline}, ${colors.outline}), linear-gradient(${colors.outline}, ${colors.outline}), linear-gradient(${colors.outline}, ${colors.outline})`,
    backgroundSize: '8px 1px, 1px 8px, 8px 1px, 1px 8px',
    backgroundPosition: 'top left, top left, bottom right, bottom right',
    backgroundRepeat: 'no-repeat',
  },
  moss: { boxShadow: `inset 1px 0 0 ${colors.primaryContainer}, 0 0 12px color-mix(in srgb, ${colors.primaryContainer} 15%, transparent)` },
  cyan: { boxShadow: `inset 1px 0 0 ${colors.tertiaryContainer}, 0 0 12px color-mix(in srgb, ${colors.tertiaryContainer} 18%, transparent)` },
  crimson: { boxShadow: `inset 1px 0 0 ${colors.secondaryContainer}, 0 0 12px color-mix(in srgb, ${colors.secondaryContainer} 25%, transparent)` },
  rose: {},
  muted: {},
  stale: { opacity: 0.55, filter: 'saturate(0.3)' },
  // In a bento cell: take the remaining height, so the grid has no dead gap.
  fill: { flexGrow: 1 },
  head: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: space.spaceSm },
  titleRow: { display: 'flex', alignItems: 'center', gap: space.spaceXs, flexWrap: 'wrap', minWidth: 0 },
  title: { color: colors.primary, margin: 0 },
  kanji: { color: colors.primaryContainer, marginInlineEnd: '0.35em' },
  sub: { color: colors.onSurfaceVariant, margin: 0, marginTop: '2px' },
  flag: { color: colors.outline, whiteSpace: 'nowrap' },
});

export type PanelProps = {
  kanji?: string;
  title: string;
  tag?: ReactNode;
  tagTone?: Tone;
  sub?: ReactNode;
  flag?: ReactNode;
  edge?: Tone;
  stale?: boolean;
  children?: ReactNode;
  xstyle?: stylex.StyleXStyles;
  id?: string;
  /** grow to fill the bento cell (the last panel in a cell) */
  fill?: boolean;
};

export function Panel({ kanji, title, tag, tagTone = 'moss', sub, flag, edge, stale, children, xstyle, id, fill }: PanelProps) {
  return (
    <section id={id} aria-label={title} {...stylex.props(s.panel, edge && s[edge], stale && s.stale, fill && s.fill, xstyle)}>
      <header {...stylex.props(s.head)}>
        <div style={{ minWidth: 0 }}>
          <div {...stylex.props(s.titleRow)}>
            <h2 {...stylex.props(text.titleMd, s.title)}>
              {kanji && <span {...stylex.props(text.kanji, s.kanji)}>{kanji} ·</span>}
              {title}
            </h2>
            {tag && <Chip tone={tagTone}>{tag}</Chip>}
          </div>
          {sub && <p {...stylex.props(text.labelXs, s.sub)}>{sub}</p>}
        </div>
        {flag !== undefined && <span {...stylex.props(text.labelXs, s.flag)}>{flag}</span>}
      </header>
      {children}
    </section>
  );
}
