// Parts the JJAVA and SOKUDO pages share, and the injector panel the Skills
// page carries too.
import * as stylex from '@stylexjs/stylex';
import { useState, type ReactNode } from 'react';
import {
  errorOf,
  injectorTotals,
  isWindow,
  msText,
  part,
  ranked,
  spreadText,
  WINDOWS,
  type Injector,
  type Jjava,
  type WindowName,
} from '../api/stats';
import { n } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Histogram } from './Charts';
import { Panel } from './Panel';
import { Chip, Label, layout, Stat, type Tone } from './primitives';
import { StateView } from './StateView';
import { Table } from './Table';
import { text } from './text';

const s = stylex.create({
  tabs: { display: 'flex', flexWrap: 'wrap', gap: '2px' },
  tab: {
    cursor: 'pointer',
    color: colors.outline,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 0,
    borderBottomWidth: 2,
    borderBottomStyle: 'solid',
    borderBottomColor: 'transparent',
    paddingInline: space.spaceSm,
    paddingBlock: space.spaceXs,
    minHeight: '32px',
    ':hover': { color: colors.primary },
  },
  tabOn: { color: colors.primaryContainer, borderBottomColor: colors.primaryContainer },
  kv: { display: 'flex', justifyContent: 'space-between', gap: space.spaceSm, minWidth: 0 },
  k: { color: colors.outline, whiteSpace: 'nowrap' },
  v: { color: colors.primary, textAlign: 'right', overflowWrap: 'anywhere', textTransform: 'none' },
  note: { margin: 0, color: colors.onSurfaceVariant, textTransform: 'none' },
  hists: { display: 'grid', gap: space.spaceSm, gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 220px), 1fr))' },
});

function readWindow(key: string): WindowName {
  let v: string | null = null;
  try {
    v = localStorage.getItem(key);
  } catch {
    v = null; // storage blocked: the default
  }
  return isWindow(v) ? v : '24h';
}

function writeWindow(key: string, w: WindowName): void {
  try {
    localStorage.setItem(key, w);
  } catch {
    /* storage blocked: the choice lasts this visit */
  }
}

/** The page's window, remembered in this browser only (a per-viewer convenience). */
export function useWindow(key: string): [WindowName, (w: WindowName) => void] {
  const [w, setW] = useState<WindowName>(() => readWindow(key));
  return [
    w,
    (x: WindowName) => {
      setW(x);
      writeWindow(key, x);
    },
  ];
}

export function WindowPicker({ value, onChange }: { value: WindowName; onChange: (w: WindowName) => void }) {
  return (
    <div role="group" aria-label="time window" {...stylex.props(s.tabs)}>
      {WINDOWS.map((w) => (
        <button key={w} type="button" aria-pressed={w === value} onClick={() => onChange(w)} {...stylex.props(text.labelMd, s.tab, w === value && s.tabOn)}>
          {w.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

export function KV({ k, v, title }: { k: ReactNode; v: ReactNode; title?: string }) {
  return (
    <div {...stylex.props(text.labelXs, s.kv)} title={title}>
      <span {...stylex.props(s.k)}>{k}</span>
      <span {...stylex.props(text.num, s.v)}>{v}</span>
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return <p {...stylex.props(text.labelXs, s.note)}>{children}</p>;
}

/** A count map as chips, largest first. */
export function Counts({ m, tone = 'muted', toneOf, empty = '—' }: { m: Record<string, number> | null | undefined; tone?: Tone; toneOf?: (k: string) => Tone; empty?: string }) {
  const rows = ranked(m);
  if (!rows.length) return <span {...stylex.props(text.labelXs, text.outline)}>{empty}</span>;
  return (
    <span {...stylex.props(layout.rowWrap)}>
      {rows.map(([k, c]) => (
        <Chip key={k} tone={toneOf ? toneOf(k) : tone}>
          <span style={{ textTransform: 'none' }}>{k}</span> {n(c)}
        </Chip>
      ))}
    </span>
  );
}

/** A section the server failed to build: its own error, never a guess. */
export function SectionError({ what, x }: { what: string; x: unknown }) {
  return <StateView kind="error" title={`${what} · the server's section failed`} detail={errorOf(x) ?? undefined} />;
}

/**
 * JJAVA · INJECTOR: the skills injector's decisions per model (stage 1's
 * candidates, stage 2's jjava Score per item, stage 3's noul, injected or
 * skipped, and why jjava could not answer). The JJAVA page and the Skills
 * page's SELECTIONS view both draw it from /dash/api/jjava.
 */
export function InjectorPanel({ data, fill, stale }: { data: Jjava | null; fill?: boolean; stale?: boolean }) {
  const inj = part<Injector>(data?.injector);
  const tot = injectorTotals(inj);
  const off = inj?.skills_by_tier ? Object.entries(inj.skills_by_tier).filter(([, on]) => on).map(([t]) => t) : null;
  return (
    <Panel
      kanji="注"
      title="JJAVA · SKILLS INJECTOR"
      tag={inj ? `${n(tot.runs)} RUNS · ${n(tot.injected)} INJECTED` : '—'}
      tagTone={tot.failed ? 'crimson' : 'cyan'}
      sub="stage 1 the mechanical filter · stage 2 a jjava Score per item (need) · stage 3 a jjava noul over the shortlist (docs/JJAVA.md 5.1)"
      stale={stale}
      fill={fill}
    >
      {!data ? (
        <StateView kind="loading" title="reading /dash/api/jjava" />
      ) : !inj ? (
        <SectionError what="injector" x={data.injector} />
      ) : (
        <>
          <div {...stylex.props(layout.grid4)}>
            <Stat label="REQUESTS RECORDED" value={n(inj.requests)} sub={`${n(inj.skills_on)} with skills on`} />
            <Stat label="DECIDER TURNS" value={n(inj.decider_turns)} sub={ranked(inj.decider_failures).map(([k, c]) => `${k} ${c}`).join(' · ') || 'no failure'} tone={Object.keys(inj.decider_failures).length ? 'rose' : undefined} />
            <Stat label="INJECTED" value={n(tot.injected)} sub={`of ${n(tot.runs)} injector runs`} tone="moss" />
            <Stat label="SKIPPED" value={n(tot.skipped)} sub={tot.failed ? `${n(tot.failed)} jjava could not answer` : 'the gate chose nothing'} tone={tot.failed ? 'rose' : undefined} />
          </div>
          {off && !off.length ? (
            <Chip tone="muted">SKILLS ARE OFF AT EVERY TIER · THE INJECTOR RUNS ONLY WHERE A HEADER FORCES SKILLS</Chip>
          ) : off ? (
            <Chip tone="moss">SKILLS ON AT {off.join(', ').toUpperCase()}</Chip>
          ) : null}
          {inj.models.length ? (
            <Table
              rows={inj.models}
              rowKey={(m) => m.model}
              bad={(m) => Object.keys(m.failures).length > 0}
              columns={[
                { key: 'm', head: 'model', cell: (m) => m.model },
                { key: 'r', head: 'runs', num: true, cell: (m) => n(m.runs) },
                { key: 'i', head: 'injected', num: true, cell: (m) => n(m.injected) },
                { key: 's1', head: 'stage 1 items', num: true, cell: (m) => m.stage1_items_mean ?? '—' },
                { key: 's2', head: 'stage 2 passed / asked', num: true, cell: (m) => `${m.stage2_passed_mean ?? '—'} / ${m.stage2_asked_mean ?? '—'}` },
                { key: 's3', head: 'stage 3 act', cell: (m) => <Counts m={m.stage3} toneOf={(k) => (k === 'all' ? 'moss' : k === 'sure_only' ? 'cyan' : 'muted')} /> },
                { key: 'f', head: 'jjava failed', cell: (m) => <Counts m={m.failures} tone="rose" empty="none" /> },
                { key: 'ms', head: 'ms', cell: (m) => spreadText(m.ms, msText) },
              ]}
            />
          ) : (
            <StateView kind="empty" title="no injector run in this window" detail={inj.note} />
          )}
          {inj.models.map((m) =>
            m.need_hist || m.noul_hist ? (
              <div key={m.model} {...stylex.props(layout.stack)}>
                <Label>{m.model} · beliefs</Label>
                <div {...stylex.props(s.hists)}>
                  {m.need_hist ? <Histogram bins={m.need_hist} label="stage 2 · need, P(top level)" slot={0} /> : null}
                  {m.noul_hist ? <Histogram bins={m.noul_hist} label="stage 3 · inject noul" slot={1} /> : null}
                </div>
                <KV k="STAGE 2 TIERS" v={ranked(m.item_tiers).map(([k, c]) => `${k} ${c}`).join(' · ') || '—'} />
                <KV k="STAGE 3 TIERS" v={ranked(m.stage3_tiers).map(([k, c]) => `${k} ${c}`).join(' · ') || '—'} />
                {ranked(m.why).length ? <KV k="WHY" v={ranked(m.why).slice(0, 3).map(([k, c]) => `${k} (${c})`).join(' · ')} /> : null}
              </div>
            ) : null,
          )}
          {inj.injector && !inj.injector.error ? (
            <KV
              k="INJECTOR"
              v={`${inj.injector.version ?? '—'} · questions ${inj.injector.questions ?? '—'} · at most ${inj.injector.max_skills ?? '—'} skills · thresholds ${typeof inj.injector.thresholds === 'string' ? inj.injector.thresholds : 'tuned rows present'}`}
            />
          ) : null}
          <Note>{inj.note}</Note>
        </>
      )}
    </Panel>
  );
}

/** A tone for a question set's tier. */
export const tierTone = (t: string): Tone => (t === 'high' ? 'moss' : t === 'medium' ? 'cyan' : t === 'low' ? 'rose' : 'muted');
