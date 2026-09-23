import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { colors, space } from '../tokens/tokens.stylex';
import { text } from './text';

const s = stylex.create({
  wrap: { overflowX: 'auto', maxWidth: '100%' },
  // A long table scrolls inside its tile instead of stretching the page.
  tall: { maxHeight: { default: '520px', '@media (min-width: 1200px)': '640px' }, overflowY: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: {
    textAlign: 'left',
    color: colors.outline,
    paddingBlock: space.spaceXs,
    paddingInline: space.spaceXs,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: `color-mix(in srgb, ${colors.outlineVariant} 50%, transparent)`,
    whiteSpace: 'nowrap',
    // Stays visible when a long table scrolls inside its tile.
    position: 'sticky',
    top: 0,
    zIndex: 1,
    backgroundColor: colors.surfaceContainerLow,
  },
  td: {
    color: colors.onSurface,
    paddingBlock: space.spaceXs,
    paddingInline: space.spaceXs,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: `color-mix(in srgb, ${colors.outlineVariant} 18%, transparent)`,
    verticalAlign: 'top',
  },
  // Section headers do not stick: several would stack at top 0.
  thStatic: { position: 'static' },
  thNext: { paddingTop: space.spaceMd },
  num: { textAlign: 'right', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' },
  bad: { backgroundColor: `color-mix(in srgb, ${colors.secondaryContainer} 10%, transparent)` },
  void: { opacity: 0.55, textDecorationLine: 'line-through', textDecorationColor: colors.secondaryContainer },
});

export type Column<T> = {
  key: string;
  head: ReactNode;
  cell: (row: T) => ReactNode;
  num?: boolean;
};

export type Section<T> = {
  key: string;
  columns: Column<T>[];
  rows: T[];
  rowKey: (r: T, i: number) => string;
  bad?: (r: T) => boolean;
  /** shown in place of rows when `rows` is empty */
  empty?: ReactNode;
};

/**
 * Several row sets in ONE table, each under its own header row, so their
 * columns share widths. Two stacked <Table>s size their columns separately
 * and never line up. Every section must have the same number of columns.
 */
export function SectionedTable({ sections, caption }: { sections: Section<any>[]; caption?: string }) {
  const width = Math.max(...sections.map((x) => x.columns.length));
  return (
    <div {...stylex.props(s.wrap)}>
      <table {...stylex.props(s.table)}>
        {caption && <caption style={{ position: 'absolute', left: -9999 }}>{caption}</caption>}
        {sections.map((sec, si) => (
          <tbody key={sec.key}>
            <tr>
              {sec.columns.map((c) => (
                <th key={c.key} scope="col" {...stylex.props(text.labelXs, s.th, s.thStatic, si > 0 && s.thNext, c.num && s.num)}>
                  {c.head}
                </th>
              ))}
            </tr>
            {sec.rows.length === 0 && sec.empty ? (
              <tr>
                <td colSpan={width} {...stylex.props(text.bodySm, s.td)}>
                  {sec.empty}
                </td>
              </tr>
            ) : (
              sec.rows.map((r, i) => (
                <tr key={sec.rowKey(r, i)} {...stylex.props(sec.bad?.(r) && s.bad)}>
                  {sec.columns.map((c) => (
                    <td key={c.key} {...stylex.props(text.bodySm, s.td, c.num && s.num)}>
                      {c.cell(r)}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        ))}
      </table>
    </div>
  );
}

export function Table<T>({
  columns,
  rows,
  rowKey,
  bad,
  voided,
  caption,
  tall,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (r: T, i: number) => string;
  bad?: (r: T) => boolean;
  voided?: (r: T) => boolean;
  caption?: string;
  tall?: boolean;
}) {
  return (
    <div {...stylex.props(s.wrap, tall && s.tall)}>
      <table {...stylex.props(s.table)}>
        {caption && <caption style={{ position: 'absolute', left: -9999 }}>{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" {...stylex.props(text.labelXs, s.th, c.num && s.num)}>
                {c.head}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey(r, i)} {...stylex.props(bad?.(r) && s.bad)}>
              {columns.map((c) => (
                <td key={c.key} {...stylex.props(text.bodySm, s.td, c.num && s.num, voided?.(r) && c.num && s.void)}>
                  {c.cell(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
