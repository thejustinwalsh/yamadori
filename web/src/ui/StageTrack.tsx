// The pipeline's stages as a row of cells, one per stage, from the overview
// payload (datasets.STAGES). A cell is reached or not; it is not a progress
// bar, because nothing measures how far through a stage a job is.
import * as stylex from '@stylexjs/stylex';
import type { DatasetsOverview } from '../api/types';
import { n } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Label } from './primitives';
import { text } from './text';

const s = stylex.create({
  track: { display: 'flex', alignItems: 'stretch', gap: '2px', overflowX: 'auto' },
  stage: {
    flexGrow: 1,
    flexBasis: 0,
    minWidth: '84px',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    padding: space.spaceXs,
    backgroundColor: colors.surfaceContainerLowest,
    borderTopWidth: 2,
    borderTopStyle: 'solid',
    borderTopColor: colors.outlineVariant,
  },
  stageOn: { borderTopColor: colors.primaryContainer, boxShadow: `inset 0 12px 16px -12px color-mix(in srgb, ${colors.primaryContainer} 35%, transparent)` },
  stageHuman: { borderTopColor: colors.tertiaryContainer },
  stageOpt: { borderTopStyle: 'dashed' },
  count: { color: colors.primary },
});

export function StageTrack({ o, current, counts }: { o: DatasetsOverview; current?: string; counts?: Record<string, number> }) {
  return (
    <div {...stylex.props(s.track)} role="list" aria-label="pipeline stages">
      {o.stages.map((st, i) => {
        const human = o.human_stages.includes(st);
        const opt = o.optional_stages.includes(st);
        const q = o.enqueues[st];
        return (
          <div key={st} role="listitem" {...stylex.props(s.stage, human && s.stageHuman, opt && s.stageOpt, current === st && s.stageOn)}>
            <Label>{String(i + 1).padStart(2, '0')}</Label>
            <span {...stylex.props(text.labelMd, current === st ? text.moss : text.onSurface)}>{st}</span>
            <span {...stylex.props(text.labelXs, text.outline)}>
              {human ? (o.assist && st === 'clarify' ? 'ASSIST · HUMAN' : 'HUMAN') : q ? `${q.lane.toUpperCase()} LANE` : '—'}
              {opt ? ' · OPT' : ''}
            </span>
            {counts && <span {...stylex.props(text.titleMd, text.num, s.count)}>{n(counts[st] ?? 0)}</span>}
          </div>
        );
      })}
    </div>
  );
}
