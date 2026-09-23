// 苗床 NAEDOKO — the seedbed. Datasets moving through the pipeline, the
// durable job queue beneath them, and whether anything has ever worked it.
// New sources are added here: paste a URL or the text, and a live card shows
// what the model established, with its provenance, and what still needs you.
import * as stylex from '@stylexjs/stylex';
import { useState } from 'react';
import { useShared } from '../api/data';
import type { Dataset, DatasetsOverview } from '../api/types';
import { ago, n } from '../format';
import { href } from '../router';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { ErrorBoundary, need } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { NavLink } from '../ui/NavLink';
import { Chip, layout, Stat } from '../ui/primitives';
import { StageTrack } from '../ui/StageTrack';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';
import { AddDatasetPanel, DatasetCard } from './AddDataset';

// Kept for older imports: the track now lives in ui/StageTrack.tsx.
export { StageTrack };

const s = stylex.create({
  grid: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'minmax(0, 2fr) minmax(0, 1fr)' },
    alignItems: 'start',
  },
  col: { display: 'flex', flexDirection: 'column', gap: space.gutter, minWidth: 0 },
  warn: { margin: 0, color: colors.secondary },
  why: { margin: 0, color: colors.onSurfaceVariant, textTransform: 'none' },
  link: { color: colors.tertiaryContainer, textDecoration: 'none', ':hover': { textDecoration: 'underline' } },
});

// Which datasets this browser is watching as live cards. A per-viewer
// convenience only: losing it loses nothing, the table below still lists
// every dataset.
const WATCH_KEY = 'yamadori_naedoko_watch';

function readWatch(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(WATCH_KEY) ?? '[]');
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string').slice(0, 8) : [];
  } catch {
    return [];
  }
}

function writeWatch(ids: string[]): void {
  try {
    localStorage.setItem(WATCH_KEY, JSON.stringify(ids));
  } catch {
    /* storage blocked: the cards last until the page is reloaded */
  }
}

function DatasetList({ list }: { list: Dataset[] }) {
  if (!list.length) {
    return (
      <StateView kind="empty" title="no datasets" />
    );
  }
  return (
    <Table
      tall
      rows={list}
      rowKey={(d) => d.id}
      bad={(d) => (d.errored_jobs?.length ?? 0) > 0}
      columns={[
        {
          key: 'name',
          head: 'dataset',
          cell: (d) => (
            <NavLink to={href.dataset(d.id)} {...stylex.props(s.link)}>
              {d.name}
            </NavLink>
          ),
        },
        { key: 'stage', head: 'stage', cell: (d) => <Chip tone="moss">{d.stage}</Chip> },
        { key: 'next', head: 'next', cell: (d) => d.next_stage ?? '—' },
        { key: 'missing', head: 'needs you', num: true, cell: (d) => (d.missing?.length ? <Chip tone="crimson">{d.missing.length}</Chip> : 0) },
        { key: 'err', head: 'errored', num: true, cell: (d) => d.errored_jobs?.length ?? 0 },
        { key: 'upd', head: 'updated', num: true, cell: (d) => `${ago(Date.now() / 1000 - d.updated)} ago` },
      ]}
    />
  );
}

export function Naedoko() {
  const { datasets } = useShared();
  const [watch, setWatch] = useState<string[]>(readWatch);
  const o = datasets.data;
  const setAndStore = (ids: string[]) => {
    setWatch(ids);
    writeWatch(ids);
  };
  const add = (
    <AddDatasetPanel
      fill={watch.length === 0}
      onSubmitted={(id) => {
        setAndStore([id, ...watch.filter((x) => x !== id)].slice(0, 8));
        datasets.reload();
      }}
    />
  );
  const cards = watch.map((id) => <DatasetCard key={id} id={id} onClose={() => setAndStore(watch.filter((x) => x !== id))} />);
  if (!o) {
    return (
      <Bento areas={areas.naedoko}>
        <Cell area="add">
          {add}
          {cards}
        </Cell>
        <Cell area="queue">
          <Panel kanji="列" title="RETSU · JOB QUEUE" fill>
            <PollState path="/dash/api/datasets" failure={datasets.failure} />
          </Panel>
        </Cell>
      </Bento>
    );
  }
  const D = '/dash/api/datasets';
  const stale = datasets.stale;
  return (
    <Bento areas={areas.naedoko}>
      <Cell area="add">
        <ErrorBoundary what="ADD DATASET" source={D}>
          {add}
        </ErrorBoundary>
        {cards}
      </Cell>
      <Cell area="queue">
        <ErrorBoundary what="RETSU · JOB QUEUE" source={`${D} · queue`}>
          <QueuePanel o={o} stale={stale} />
        </ErrorBoundary>
        <ErrorBoundary what="WORKER" source={`${D} · worker`} fill>
          <WorkerPanel o={o} />
        </ErrorBoundary>
      </Cell>
      <Cell area="pipe">
        <ErrorBoundary what="NAEDOKO · PIPELINE" source={`${D} · stages`}>
          <PipelinePanel o={o} stale={stale} />
        </ErrorBoundary>
      </Cell>
      <Cell area="list">
        <ErrorBoundary what="DATASETS" source={`${D} · datasets`} fill>
          <Panel kanji="籍" title="DATASETS" tag="ALL" tagTone="muted" stale={stale} fill>
            <DatasetListOf o={o} />
          </Panel>
        </ErrorBoundary>
      </Cell>
      <Cell area="req">
        <ErrorBoundary what="REQUIRED BEFORE EXTRACT" source={`${D} · fields`} fill>
          <RequiredPanel o={o} />
        </ErrorBoundary>
      </Cell>
    </Bento>
  );
}

// Each panel body is its own component, so a malformed payload throws inside
// that panel's error boundary and nowhere else.
const D = '/dash/api/datasets';

function QueuePanel({ o, stale }: { o: DatasetsOverview; stale: boolean }) {
  const q = need(o, 'queue', D);
  return (
    <Panel kanji="列" title="RETSU · JOB QUEUE" tag="DURABLE" tagTone="cyan" flag={q.oldest_queued_age != null ? `OLDEST ${ago(q.oldest_queued_age)}` : undefined} stale={stale}>
      <div {...stylex.props(layout.grid3)}>
        {Object.entries(need(q, 'states', `${D} · queue`)).map(([k, v]) => (
          <Stat key={k} label={k} value={n(v)} tone={k === 'errored' && v ? 'crimson' : k === 'running' && v ? 'moss' : undefined} />
        ))}
      </div>
      <Table
        rows={Object.entries(q.lane_limits ?? {})}
        rowKey={([l]) => l}
        columns={[
          { key: 'l', head: 'lane', cell: ([l]) => l },
          { key: 'lim', head: 'limit', num: true, cell: ([, lim]) => lim },
          { key: 'run', head: 'running', num: true, cell: ([l]) => q.lanes?.[l]?.running ?? 0 },
          { key: 'que', head: 'queued', num: true, cell: ([l]) => q.lanes?.[l]?.queued ?? 0 },
        ]}
      />
    </Panel>
  );
}

function WorkerPanel({ o }: { o: DatasetsOverview }) {
  const w = need(o, 'worker', D);
  const never = w.ever_claimed === 0;
  return (
    <Panel kanji="働" title="WORKER" tag={never ? 'NEVER CLAIMED' : 'SEEN'} tagTone={never ? 'crimson' : 'moss'} edge={never ? 'crimson' : undefined} fill>
      {never ? (
        <p {...stylex.props(text.bodySm, s.warn)}>No job has ever been claimed.</p>
      ) : (
        <div {...stylex.props(layout.grid2)}>
          <Stat label="JOBS EVER CLAIMED" value={n(w.ever_claimed)} />
          <Stat label="LAST HEARTBEAT" value={w.last_heartbeat ? `${ago(Date.now() / 1000 - w.last_heartbeat)} ago` : '—'} />
        </div>
      )}
    </Panel>
  );
}

function PipelinePanel({ o, stale }: { o: DatasetsOverview; stale: boolean }) {
  const list = need(o, 'datasets', D, Array.isArray);
  const perStage: Record<string, number> = {};
  for (const d of list) perStage[d.stage] = (perStage[d.stage] ?? 0) + 1;
  need(o, 'stages', D, Array.isArray);
  return (
    <Panel kanji="苗床" title="NAEDOKO · PIPELINE" tag={`${list.length} DATASET${list.length === 1 ? '' : 'S'}`} stale={stale} edge="moss">
      <StageTrack o={o} counts={perStage} />
      <div {...stylex.props(layout.rowWrap, text.labelXs)}>
        <Chip tone="cyan">HUMAN STAGE</Chip>
        <Chip tone="muted">- - OPTIONAL</Chip>
      </div>
    </Panel>
  );
}

function DatasetListOf({ o }: { o: DatasetsOverview }) {
  return <DatasetList list={need(o, 'datasets', D, Array.isArray)} />;
}

function RequiredPanel({ o }: { o: DatasetsOverview }) {
  const fields = need(o, 'fields', D, Array.isArray);
  return (
    <Panel kanji="問" title="REQUIRED BEFORE EXTRACT" tag={`${fields.length} FIELDS`} tagTone="rose" fill>
      <div {...stylex.props(layout.stackSm)}>
        {fields.map((f) => (
          <div key={f.name}>
            <div {...stylex.props(layout.rowWrap)}>
              <span {...stylex.props(text.labelMd, text.primary)}>{f.label}</span>
              <Chip tone={f.severity === 'blocker' ? 'crimson' : 'muted'}>{f.severity}</Chip>
              {o.assist?.fields?.includes(f.name) ? (
                <Chip tone="cyan">{f.name === 'licence' ? 'ASSIST: QUOTE ONLY' : 'ASSIST MAY PROPOSE'}</Chip>
              ) : null}
            </div>
            <p {...stylex.props(text.bodySm, s.why)}>{f.why}</p>
          </div>
        ))}
      </div>
    </Panel>
  );
}

const areas = stylex.create({
  naedoko: {
    gridTemplateAreas: {
      default: '"add" "queue" "pipe" "list" "req"',
      [MQ.tablet]: '"add add add add add add" "pipe pipe pipe pipe pipe pipe" "queue queue queue req req req" "list list list list list list"',
      [MQ.desktop]:
        '"add add add add add add add queue queue queue queue queue" "pipe pipe pipe pipe pipe pipe pipe pipe pipe pipe pipe pipe" "list list list list list list list list req req req req"',
    },
  },
});
