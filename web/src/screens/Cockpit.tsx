import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { useShared } from '../api/data';
import { kvSplit } from '../api/kv';
import type { Endpoint, Listener, Tiers, Vitals } from '../api/types';
import { useNow } from '../api/usePoll';
import { Tokonoma } from '../bonsai/scene/Tokonoma';
import { gib, n } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Meter, Row, SplitBar, Stat, type Tone } from '../ui/primitives';
import { SeedPanel } from '../ui/SeedReadout';
import { PollState, StateView } from '../ui/StateView';
import { SectionedTable, Table } from '../ui/Table';
import { text } from '../ui/text';

const areas = stylex.create({
  cockpit: {
    gridTemplateAreas: {
      default: '"tree" "side" "stat" "tier" "proc"',
      [MQ.tablet]: '"tree tree tree tree tree tree" "side side side stat stat stat" "side side side tier tier tier" "proc proc proc proc proc proc"',
      [MQ.desktop]:
        '"tree tree tree tree tree tree tree tree side side side side" "stat stat stat stat proc proc proc proc proc tier tier tier"',
    },
  },
});

const s = stylex.create({
  gpu: { display: 'flex', flexDirection: 'column', gap: space.spaceXs, padding: space.spaceSm, backgroundColor: colors.surfaceContainerLowest, borderWidth: 1, borderStyle: 'solid', borderColor: `color-mix(in srgb, ${colors.outlineVariant} 25%, transparent)` },
  gpuTight: { borderColor: colors.secondaryContainer, boxShadow: `inset 2px 0 0 ${colors.secondaryContainer}` },
  big: { color: colors.primary, margin: 0 },
  unit: { color: colors.outline, marginInlineStart: '6px' },
  keyDot: { display: 'inline-block', width: '8px', height: '8px', marginInlineEnd: '6px' },
  kMain: { backgroundColor: colors.primaryContainer },
  kThink: { backgroundColor: colors.tertiaryContainer },
  kRes: { backgroundImage: `repeating-linear-gradient(-45deg, ${colors.surfaceContainerLow}, ${colors.surfaceContainerLow} 2px, ${colors.surfaceContainerHigh} 2px, ${colors.surfaceContainerHigh} 4px)` },
  note: { margin: 0, color: colors.onSurfaceVariant },
  warnItem: { color: colors.secondary },
  tier: { display: 'grid', gridTemplateColumns: '64px minmax(0,1fr)', gap: space.spaceSm, alignItems: 'start' },
  tierName: { color: colors.primary },
  tierOn: { color: colors.primaryContainer },
  why: { color: colors.onSurfaceVariant, margin: 0, textTransform: 'none', marginTop: '3px' },
});

/** One panel in its own error boundary: a bad payload fails that panel only. */
function Guard({ what, source, fill, children }: { what: string; source: string; fill?: boolean; children: ReactNode }) {
  return (
    <ErrorBoundary what={what} source={source} fill={fill}>
      {children}
    </ErrorBoundary>
  );
}

const V = '/dash/api/vitals';

export function Cockpit() {
  const { vitals, datasets, tiers } = useShared();
  const now = useNow(1000);
  const v = vitals.data;
  return (
    <Bento areas={areas.cockpit}>
      <Cell area="tree">
        <Guard what="TOKONOMA" source={V} fill>
          <Tokonoma vitals={v} datasets={datasets.data} fill />
        </Guard>
      </Cell>
      <Cell area="side">
        <Guard what="TANE · SEED" source={V}>
          <SeedPanel vitals={v} now={now} />
        </Guard>
        <Guard what="MIKI · KV POOL" source={V}>
          <KvPanel v={v} stale={vitals.stale} failure={vitals.failure} />
        </Guard>
        <Guard what="SILICON" source={V} fill>
          <SiliconPanel v={v} stale={vitals.stale} failure={vitals.failure} fill />
        </Guard>
      </Cell>
      <Cell area="stat">
        <Guard what="KEIHŌ · WARNINGS" source={V}>
          <WarningsPanel v={v} failure={vitals.failure} />
        </Guard>
        <Guard what="NE · SERVICES" source={V} fill>
          <ServicesPanel v={v} stale={vitals.stale} failure={vitals.failure} fill />
        </Guard>
      </Cell>
      <Cell area="proc">
        <Guard what="PROCESSES" source={V} fill>
          <ProcessesPanel v={v} stale={vitals.stale} failure={vitals.failure} fill />
        </Guard>
      </Cell>
      <Cell area="tier">
        <Guard what="DAN · EFFORT LADDER" source="/dash/api/tiers" fill>
          <TiersPanel t={tiers.data} failure={tiers.failure} fill />
        </Guard>
      </Cell>
    </Bento>
  );
}

type P = { v: Vitals | null; stale?: boolean; failure: ReturnType<typeof useShared>['vitals']['failure']; fill?: boolean };

export function KvPanel({ v, stale, failure, fill }: P) {
  const c = v?.context;
  const kv = kvSplit(c);
  const pct = (x: number) => `${((100 * x) / (kv?.pool || 1)).toFixed(1)}%`;
  return (
    <Panel kanji="幹" title="MIKI · KV POOL" tag={kv ? `${kv.helpers + 1} CONTEXTS` : 'TRUNK SPLIT'} tagTone="cyan" stale={stale} edge="cyan" fill={fill}>
      {!v ? (
        <PollState path={V} failure={failure} />
      ) : !kv ? (
        <StateView kind="error" title="no context budget" detail={c && 'error' in c ? c.error : undefined} />
      ) : (
        <>
          <div {...stylex.props(layout.grid4)}>
            <Stat label="POOL" value={n(kv.pool)} sub={`${kv.gib} GiB KV`} />
            <Stat label="MAIN" value={n(kv.main)} sub={pct(kv.main)} tone="moss" />
            <Stat
              label={`DEEP THINKING ×${kv.helpers}`}
              value={n(kv.helper * kv.helpers)}
              sub={`${n(kv.helper)} · ${pct(kv.helper)} each`}
              tone="cyan"
            />
            <Stat label="RESERVE" value={n(kv.reserve)} sub={pct(kv.reserve)} />
          </div>
          <SplitBar
            label={`main ${n(kv.main)}, ${kv.helpers} deep thinking contexts of ${n(kv.helper)}, reserve ${n(kv.reserve)}, of ${n(kv.pool)} tokens`}
            parts={[
              { value: kv.main, tone: 'moss' as Tone | 'hatch' },
              ...Array.from({ length: kv.helpers }, () => ({ value: kv.helper, tone: 'cyan' as Tone | 'hatch' })),
              ...(kv.reserve > 0 ? [{ value: kv.reserve, tone: 'hatch' as Tone | 'hatch' }] : []),
            ]}
          />
          <div {...stylex.props(layout.rowWrap, text.labelXs)} style={{ gap: 14 }}>
            <span><i {...stylex.props(s.keyDot, s.kMain)} />MAIN</span>
            <span><i {...stylex.props(s.keyDot, s.kThink)} />DEEP THINKING ×{kv.helpers}{kv.helpersReported ? '' : ' (DERIVED)'}</span>
            {kv.reserve > 0 && <span><i {...stylex.props(s.keyDot, s.kRes)} />RESERVE</span>}
          </div>
          {kv.pool === 131072 && <Chip tone="rose">POOL EQUALS THE FALLBACK VALUE · UNCONFIRMED</Chip>}
        </>
      )}
    </Panel>
  );
}

function SiliconPanel({ v, stale, failure, fill }: P) {
  const g = v?.gpus ?? [];
  return (
    <Panel kanji="鉄" title="SILICON" tag={v ? `${g.length} GPU` : '—'} flag="nvidia-smi" stale={stale} fill={fill}>
      {!v ? (
        <PollState path={V} failure={failure} />
      ) : !g.length ? (
        <StateView kind="empty" title="no GPU reported" />
      ) : (
        <div {...stylex.props(layout.stackSm)}>
          {g.map((x) => (
            <div key={x.index} {...stylex.props(s.gpu, x.tight && s.gpuTight)}>
              <div {...stylex.props(layout.between)}>
                <Label>GPU{x.index} · {x.name.replace(/^NVIDIA (GeForce )?/, '')}</Label>
                {x.tight ? <Chip tone="crimson">TIGHT</Chip> : <Chip tone="muted">UTIL {x.util}%</Chip>}
              </div>
              <p {...stylex.props(text.headlineSm, text.num, s.big)}>
                {n(x.free_mib)}
                <span {...stylex.props(text.labelXs, s.unit)}>MiB FREE</span>
              </p>
              <Meter value={x.pct / 100} tone={x.tight ? 'crimson' : 'moss'} label={`${n(x.used_mib)} of ${n(x.total_mib)} MiB used`} />
              <Label>USED {gib(x.used_mib)} / {gib(x.total_mib)} GiB · {x.pct}%</Label>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function WarningsPanel({ v, failure }: P) {
  const w = v?.warnings ?? [];
  return (
    <Panel kanji="警" title="KEIHŌ · WARNINGS" tag={v ? (w.length ? `${w.length} ACTIVE` : 'CLEAR') : '—'} tagTone={w.length ? 'crimson' : 'moss'} edge={w.length ? 'crimson' : undefined}>
      {!v ? (
        <PollState path={V} failure={failure} />
      ) : w.length ? (
        <div {...stylex.props(layout.stack)}>
          {w.map((x, i) => (
            <Row key={i} bad>
              <span {...stylex.props(s.warnItem)}>[ ! ] {x}</span>
            </Row>
          ))}
        </div>
      ) : (
        <p {...stylex.props(text.bodySm, s.note)}>No warnings in this snapshot.</p>
      )}
      {v?.duplicates?.length ? (
        <Label>DUPLICATE TREES · {v.duplicates.map((d) => `${d.what} (${d.pids.join(', ')})`).join('; ')}</Label>
      ) : null}
    </Panel>
  );
}

function ServicesPanel({ v, stale, failure, fill }: P) {
  return (
    <Panel kanji="根" title="NE · SERVICES" tag="PORTS" tagTone="cyan" stale={stale} fill={fill}>
      {!v ? (
        <PollState path={V} failure={failure} />
      ) : (
        // One table, two sections: the columns share widths and line up.
        // Each column keeps one alignment in both (text left, numbers right).
        <SectionedTable
          sections={[
            {
              key: 'endpoints',
              rows: v.endpoints ?? [],
              rowKey: (e: Endpoint) => e.name,
              bad: (e: Endpoint) => !e.ok,
              empty: <StateView kind="empty" title="no endpoint probed" />,
              columns: [
                { key: 'n', head: 'endpoint', cell: (e: Endpoint) => e.name },
                { key: 's', head: 'state', cell: (e: Endpoint) => (e.ok ? <Chip tone="moss">ANSWERING</Chip> : <Chip tone="crimson">DOWN</Chip>) },
                { key: 'h', head: 'http', num: true, cell: (e: Endpoint) => (e.code ? e.code : '—') },
                { key: 'm', head: 'ms', num: true, cell: (e: Endpoint) => n(e.ms) },
              ],
            },
            {
              key: 'listeners',
              rows: v.listeners ?? [],
              rowKey: (l: Listener) => `${l.port}-${l.pid}-${l.addr}`,
              bad: (l: Listener) => l.conflict,
              empty: <StateView kind="empty" title="nothing listening on a watched port" />,
              columns: [
                { key: 'r', head: 'role', cell: (l: Listener) => (l.conflict ? <>{l.role} <Chip tone="crimson">CONFLICT</Chip></> : l.role) },
                { key: 'a', head: 'address', cell: (l: Listener) => l.addr },
                { key: 'p', head: 'port', num: true, cell: (l: Listener) => l.port },
                { key: 'i', head: 'pid', num: true, cell: (l: Listener) => l.pid },
              ],
            },
          ]}
        />
      )}
    </Panel>
  );
}

function ProcessesPanel({ v, stale, failure, fill }: P) {
  const rows = v?.processes ?? [];
  return (
    <Panel kanji="動" title="PROCESSES" tag={v ? `${rows.length} MATCHING` : '—'} tagTone="muted" stale={stale} fill={fill}>
      {!v ? (
        <PollState path={V} failure={failure} />
      ) : !rows.length ? (
        <StateView kind="empty" title="no matching process" />
      ) : (
        <Table
          rows={rows}
          rowKey={(p, i) => `${p.pid}-${i}`}
          columns={[
            { key: 'pid', head: 'pid', num: true, cell: (p) => p.pid ?? '—' },
            { key: 'ppid', head: 'parent', num: true, cell: (p) => p.ppid ?? '—' },
            { key: 'what', head: 'what', cell: (p) => p.what },
            { key: 'port', head: 'port', num: true, cell: (p) => p.port || '—' },
          ]}
        />
      )}
    </Panel>
  );
}

const up = (x: unknown) => (typeof x === 'string' && x ? x.toUpperCase() : '—');

export function TiersPanel({ t, failure, fill }: { t: Tiers | null; failure: ReturnType<typeof useShared>['tiers']['failure']; fill?: boolean }) {
  const order = Array.isArray(t?.order) ? t.order : [];
  const table = t?.tiers && typeof t.tiers === 'object' ? t.tiers : {};
  return (
    <Panel kanji="段" title="DAN · EFFORT LADDER" tag={t ? `DEFAULT ${up(t.default)}` : '—'} tagTone="moss" fill={fill}>
      {!t ? (
        <PollState path="/dash/api/tiers" failure={failure} />
      ) : !order.length ? (
        <StateView kind="empty" title="/dash/api/tiers · no tier order" />
      ) : (
        <div {...stylex.props(layout.stackSm)}>
          {order.map((name) => {
            const x = table[name];
            if (!x) return null;
            const flags = [x.retrieval && 'search', x.hints && 'hints', x.fanout > 1 && `fan-out ${x.fanout}`, x.investigate && 'deep thinking'].filter(Boolean);
            // The effort the proxy sends (tiers.safe_effort) is what the model
            // sees. Until the server reports it, show what the tier asks for
            // and say that is all it is.
            const sent = typeof x.sent_effort === 'string' ? x.sent_effort : null;
            return (
              <div key={name} {...stylex.props(s.tier)}>
                <span {...stylex.props(text.labelMd, name === t.default ? s.tierOn : s.tierName)}>{name}</span>
                <div>
                  <div {...stylex.props(layout.rowWrap)}>
                    {sent ? (
                      <Chip tone="muted" title={sent !== x.effort ? `tier asks for ${x.effort}` : undefined}>
                        EFFORT {up(sent)}
                      </Chip>
                    ) : (
                      <Chip tone="muted" title="the value sent is not reported by this server">
                        ASKS {up(x.effort)} · SENT —
                      </Chip>
                    )}
                    {flags.map((f) => (
                      <Chip key={String(f)} tone={f === 'deep thinking' ? 'cyan' : 'moss'}>{String(f).toUpperCase()}</Chip>
                    ))}
                    {name === t.ceiling && <Chip tone="rose">CEILING</Chip>}
                  </div>
                  {x.why ? <p {...stylex.props(text.labelXs, s.why)}>{x.why}</p> : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
