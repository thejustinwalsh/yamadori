// What is running in each part of the KV bargraph right now, one line per part, from the lane view
// (mcp/lane_view.py via web/src/api/lanes.ts kvLive): the model and slot behind MAIN, CHILD and SECOND CONVERSATION,
// its state, rate and requests in flight. A part with nothing behind it says what it would be.
import * as stylex from '@stylexjs/stylex';
import { kvLive, laneView, type LaneView } from '../api/lanes';
import { colors, space } from '../tokens/tokens.stylex';
import { text } from './text';

const s = stylex.create({
  list: { display: 'flex', flexDirection: 'column', gap: '3px', minWidth: 0 },
  row: { display: 'grid', gridTemplateColumns: 'max-content minmax(0, 1fr)', columnGap: space.spaceXs, alignItems: 'baseline' },
  stacked: { display: 'flex', flexDirection: 'column', gap: 0 },
  part: { whiteSpace: 'nowrap' },
  words: { overflowWrap: 'anywhere' },
  moss: { color: colors.primaryContainer },
  cyan: { color: colors.tertiaryContainer },
  muted: { color: colors.outline },
});

const PART: Record<string, string> = { main: 'MAIN', child: 'CHILD', second: '2ND CONV' };

/** `stacked`: the part's name above its words, for a narrow card. */
export function KvLive({ view, noLane, stacked }: { view: LaneView | null | undefined; noLane: boolean; stacked?: boolean }) {
  const lv = laneView(view);
  if (!lv) return null;
  return (
    <div {...stylex.props(s.list)}>
      {kvLive(lv, noLane).map((l) => (
        <div key={l.part} {...stylex.props(text.labelXs, stacked ? s.stacked : s.row)}>
          <span {...stylex.props(s.part, s[l.tone])}>{PART[l.part]}</span>
          <span {...stylex.props(s.words, s.muted)}>{l.text.toUpperCase()}</span>
        </div>
      ))}
    </div>
  );
}
