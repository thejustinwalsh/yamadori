// The mosaic every screen is laid out on: a 12-column grid with named areas
// per breakpoint. Cells stretch to their row, and the last panel in a cell
// takes `fill`, so a short panel grows to meet a tall neighbour instead of
// leaving a dead gap beside it.
//
// Each screen passes its own areas as a StyleX style (areas must be static
// for the compiler); `Cell` places a stack of panels into one named area.
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { space } from '../tokens/tokens.stylex';
import { MQ } from './breakpoints.stylex';

export { MQ };


const s = stylex.create({
  grid: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: {
      default: 'minmax(0, 1fr)',
      [MQ.tablet]: 'repeat(6, minmax(0, 1fr))',
      [MQ.desktop]: 'repeat(12, minmax(0, 1fr))',
    },
    alignItems: 'stretch',
    minWidth: 0,
  },
  cell: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  area: (name: string) => ({ gridArea: name }),
});

export function Bento({ areas, children }: { areas: stylex.StyleXStyles; children: ReactNode }) {
  return <div {...stylex.props(s.grid, areas)}>{children}</div>;
}

export function Cell({ area, children }: { area: string; children: ReactNode }) {
  return (
    <div data-area={area} {...stylex.props(s.cell, s.area(area))}>
      {children}
    </div>
  );
}
