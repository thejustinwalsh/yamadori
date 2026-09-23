import * as stylex from '@stylexjs/stylex';
import { PATHS, useShared } from '../api/data';
import type { Dataset } from '../api/types';
import { usePoll } from '../api/usePoll';
import { ago, n } from '../format';
import { href } from '../router';
import { colors, space } from '../tokens/tokens.stylex';
import { Guarded } from '../ui/ErrorBoundary';
import { NavLink } from '../ui/NavLink';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Row, Stat } from '../ui/primitives';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';
import { StageTrack } from '../ui/StageTrack';
import { ProvenanceBody } from './AddDataset';

const s = stylex.create({
  grid: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'minmax(0, 2fr) minmax(0, 1fr)' },
    alignItems: 'stretch',
  },
  col: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  back: { color: colors.tertiaryContainer, textDecoration: 'none' },
  kv: { display: 'grid', gridTemplateColumns: 'minmax(110px, auto) minmax(0, 1fr)', columnGap: space.spaceSm, rowGap: '4px' },
  v: { color: colors.onSurface, overflowWrap: 'anywhere' },
  unset: { color: colors.secondary },
  err: { color: colors.secondary, overflowWrap: 'anywhere' },
});

export function DatasetDetail({ id }: { id: string }) {
  const { datasets } = useShared();
  const d = usePoll<Dataset>(PATHS.dataset(id), 10000);
  const back = (
    <NavLink to={href.naedoko} {...stylex.props(text.labelMd, s.back)}>
      ← NAEDOKO
    </NavLink>
  );
  if (!d.data) {
    return (
      <div {...stylex.props(s.col)}>
        {back}
        <PollState path={PATHS.dataset(id)} failure={d.failure} what="dataset field" />
      </div>
    );
  }
  const x = d.data;
  const field = (k: keyof Dataset) => {
    const v = x[k];
    const empty = v === null || v === undefined || v === '' || (Array.isArray(v) && !v.length);
    return <span {...stylex.props(text.bodySm, empty ? s.unset : s.v)}>{empty ? 'not answered' : Array.isArray(v) ? v.join(', ') : String(v)}</span>;
  };
  return (
    <div {...stylex.props(s.col)}>
      {back}
      <div {...stylex.props(s.grid)}>
        <div {...stylex.props(s.col)}>
          <Guarded what="DATASET" source={PATHS.dataset(id)} render={() => (
            <Panel kanji="苗" title={x.name} tag={String(x.stage ?? '—').toUpperCase()} sub={x.prompt || undefined} flag={x.id} stale={d.stale} edge="moss">
              {datasets.data ? <StageTrack o={datasets.data} current={x.stage} /> : null}
              <div {...stylex.props(s.kv)}>
                {(['kind', 'source_name', 'source_url', 'licence', 'language', 'domains', 'notes', 'recipe_file'] as const).map((k) => (
                  <span key={k} style={{ display: 'contents' }}>
                    <Label>{k.replace('_', ' ')}</Label>
                    {field(k)}
                  </span>
                ))}
              </div>
              <Label>
                CREATED {ago(Date.now() / 1000 - x.created)} AGO · UPDATED {ago(Date.now() / 1000 - x.updated)} AGO · NEXT {x.next_stage ?? '—'}
              </Label>
            </Panel>
          )} />
          <Guarded what="CLARIFY" source={PATHS.dataset(id)} fill render={() => (
            <Panel fill
              kanji="問"
              title="CLARIFY · WHERE EACH ANSWER CAME FROM"
              tag={`${(x.field_states ?? []).filter((f) => f.state === 'needs_you').length} NEED YOU`}
              tagTone={(x.field_states ?? []).some((f) => f.state === 'needs_you') ? 'crimson' : 'moss'}
            >
              <ProvenanceBody d={x} reload={d.reload} />
            </Panel>
          )} />
        </div>
        <div {...stylex.props(s.col)}>
          <Guarded what="MISSING" source={PATHS.dataset(id)} render={() => (
            <Panel kanji="欠" title="MISSING" tag={`${x.missing?.length ?? 0}`} tagTone={x.missing?.length ? 'crimson' : 'moss'} edge={x.missing?.length ? 'crimson' : undefined}>
              {x.missing?.length ? (
                <div {...stylex.props(layout.stack)}>
                  {x.missing.map((m, i) => (
                    <Row key={i} bad>
                      <span {...stylex.props(s.err)}>{m.what}</span>
                    </Row>
                  ))}
                </div>
              ) : (
                <p {...stylex.props(text.bodySm, s.v)} style={{ margin: 0 }}>Every required field is answered.</p>
              )}
            </Panel>
          )} />
          <Guarded what="WARNINGS" source={PATHS.dataset(id)} render={() => (
            <Panel kanji="注" title="WARNINGS" tag={`${x.warnings?.length ?? 0}`} tagTone={x.warnings?.length ? 'rose' : 'moss'}>
              {x.warnings?.length ? x.warnings.map((w, i) => <Row key={i}><span>{w.what}</span></Row>) : <Label>NONE</Label>}
            </Panel>
          )} />
          <Guarded what="REVIEW" source={PATHS.dataset(id)} render={() => (
            <Panel kanji="審" title="REVIEW" tag="ROWS IN THE JSONL" tagTone="muted">
              {x.review ? (
                <div {...stylex.props(layout.grid2)}>
                  {Object.entries(x.review).map(([k, v]) => (
                    <Stat key={k} label={k} value={n(v)} />
                  ))}
                </div>
              ) : (
                <StateView kind="empty" title="no rows file" />
              )}
            </Panel>
          )} />
          <Guarded what="JOB COUNTS" source={PATHS.dataset(id)} render={() => (
            <Panel kanji="計" title="JOB COUNTS" tagTone="muted">
              <div {...stylex.props(layout.grid3)}>
                {Object.entries(x.jobs ?? {}).map(([k, v]) => (
                  <Stat key={k} label={k} value={n(v)} tone={k === 'errored' && v ? 'crimson' : undefined} />
                ))}
              </div>
            </Panel>
          )} />
          <Guarded what="JOBS" source={PATHS.dataset(id)} fill render={() => (
            <Panel fill kanji="務" title="JOBS" tag={`${x.job_rows?.length ?? 0} ROWS`} tagTone="cyan">
              {x.job_rows?.length ? (
                <Table
                  rows={x.job_rows}
                  rowKey={(j) => j.id}
                  bad={(j) => j.state === 'errored'}
                  columns={[
                    { key: 'q', head: 'queue', cell: (j) => j.queue },
                    { key: 'l', head: 'lane', cell: (j) => j.lane },
                    { key: 's', head: 'state', cell: (j) => <Chip tone={j.state === 'errored' ? 'crimson' : j.state === 'done' ? 'moss' : 'muted'}>{j.state}</Chip> },
                    { key: 'a', head: 'attempts', num: true, cell: (j) => `${j.attempts}/${j.max_attempts}` },
                    { key: 'p', head: 'progress / error', cell: (j) => <span {...stylex.props(j.error ? s.err : s.v)}>{j.error ?? j.progress ?? '—'}</span> },
                  ]}
                />
              ) : (
                <StateView kind="empty" title="no job rows" />
              )}
            </Panel>
          )} />
        </div>
      </div>
    </div>
  );
}
