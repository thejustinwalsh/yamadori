// WHAT RUNS WHERE, in plain words (mcp/lane_view.py, carried as `cards` on /dash/api/vitals/pulse): the model on
// each card, a swap in progress and the last swaps, the worker's GPU job lanes, and what holds what and since when.
// Every figure is the server's; a part the server cannot read says so itself.
import * as stylex from '@stylexjs/stylex';
import { holdTimes, laneView, span, type Hold, type JobLane, type LaneCard, type LaneModel, type LaneView, type SwapDone } from '../../api/lanes';
import { gib } from '../../format';
import { colors, space } from '../../tokens/tokens.stylex';
import { Chip, Label } from '../../ui/primitives';
import { text } from '../../ui/text';
import { Card } from './Telemetry';

const s = stylex.create({
  block: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceXs,
    minWidth: 0,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: `color-mix(in srgb, ${colors.outlineVariant} 35%, transparent)`,
    paddingTop: space.spaceXs,
  },
  row: { display: 'grid', gridTemplateColumns: 'max-content max-content minmax(0, 1fr)', columnGap: space.spaceXs, alignItems: 'baseline' },
  nowrap: { whiteSpace: 'nowrap' },
  wrap: { overflowWrap: 'anywhere', textTransform: 'none' },
  soft: { color: colors.onSurfaceVariant },
  dim: { color: colors.outline },
  moss: { color: colors.primaryContainer },
  cyan: { color: colors.tertiaryContainer },
  rose: { color: colors.secondary },
  holdHead: { display: 'flex', gap: space.spaceXs, alignItems: 'baseline', minWidth: 0 },
});

const CARD_NAME: Record<string, string> = { main: '5060 TI', a4000: 'A4000' };
const cardName = (k: string, name?: string | null) => CARD_NAME[k] ?? name ?? k;

const KIND_LABEL: Record<string, string> = {
  swap: 'SWAP',
  model_state: 'LLAMA-SWAP',
  model_lease: 'REQUESTS',
  conversation: 'CONVERSATION',
  gpu_room_lease: 'A4000 LEASE',
  job_lane_paused: 'JOB LANE PAUSED',
  unknown: 'UNKNOWN',
};

function modelLine(m: LaneModel, now: number): string {
  const parts = [m.port ? `:${m.port}` : null];
  if (m.inflight > 0) parts.push(`${m.inflight} REQUEST${m.inflight === 1 ? '' : 'S'} IN FLIGHT${m.inflight_since ? ` SINCE ${span(now - m.inflight_since)} AGO` : ''}`);
  else if (m.state === 'ready') parts.push(m.slots?.slots.some((x) => x.state !== 'idle') ? 'WORKING' : 'IDLE');
  if (m.ttl) parts.push(`TTL ${m.ttl} S`);
  return parts.filter(Boolean).join(' · ');
}

function CardBlock({ c, now }: { c: LaneCard; now: number }) {
  return (
    <div {...stylex.props(s.section)}>
      <div {...stylex.props(text.labelXs, s.soft, s.nowrap)}>
        {cardName(c.key, c.name)}
        {c.used_mib != null && c.total_mib != null ? ` · VRAM ${gib(c.used_mib)} / ${gib(c.total_mib)} GIB` : ''}
        {c.util != null ? ` · ${c.util}% UTIL` : ''}
      </div>
      {c.models.length === 0 ? (
        <span {...stylex.props(text.labelXs, s.dim)}>nothing loaded by llama-swap</span>
      ) : (
        c.models.map((m) => (
          <div key={m.model} {...stylex.props(text.labelXs, s.row)}>
            <span {...stylex.props(m.state === 'ready' ? s.moss : s.cyan, s.nowrap)}>● {m.model.toUpperCase()}</span>
            <span {...stylex.props(s.soft, s.nowrap)}>{m.state.toUpperCase()}</span>
            <span {...stylex.props(s.dim, s.wrap)}>{modelLine(m, now)}</span>
          </div>
        ))
      )}
      {c.expected.map((e) => (
        <span key={e.model} {...stylex.props(text.labelXs, s.dim, s.wrap)}>
          ○ {e.model.toUpperCase()} not loaded · {e.for}
        </span>
      ))}
    </div>
  );
}

function SwapBlock({ lv }: { lv: NonNullable<ReturnType<typeof laneView>> }) {
  const now = lv.swap.now;
  const last: SwapDone[] = lv.swap.last;
  return (
    <div {...stylex.props(s.section)}>
      <Label>SWAPS</Label>
      {now ? (
        <span {...stylex.props(text.labelXs, s.rose, s.wrap)}>
          IN PROGRESS · {(now.from.length ? now.from.join(', ') : 'nothing').toUpperCase()} → {now.to.toUpperCase()} ·{' '}
          {now.phase === 'waiting' ? `WAITING FOR THE WORK ON ${now.from.join(', ').toUpperCase() || 'THE CARD'} TO END` : 'LLAMA-SWAP IS LOADING IT'} · {span(lv.at - now.since)} SO FAR
        </span>
      ) : (
        <span {...stylex.props(text.labelXs, s.dim)}>no swap in progress</span>
      )}
      {last.length === 0 ? (
        <span {...stylex.props(text.labelXs, s.dim)}>no swap recorded in the last 7 days</span>
      ) : (
        last.map((x) => (
          <span key={x.at} {...stylex.props(text.labelXs, s.soft, s.wrap)}>
            {span(lv.at - x.at)} AGO · {(x.from.length ? x.from.join(', ') : 'nothing').toUpperCase()} → {(x.to ?? '?').toUpperCase()}
            {x.load_s != null ? ` · ${x.load_s} S` : ''} · {x.ok ? 'OK' : 'FAILED'}
            {x.how ? ` · ${x.how}` : ''}
          </span>
        ))
      )}
    </div>
  );
}

function JobLanes({ jobs, now }: { jobs: JobLane[]; now: number }) {
  return (
    <div {...stylex.props(s.section)}>
      <Label>WORKER JOB LANES</Label>
      {jobs.map((j) => (
        <div key={j.lane} {...stylex.props(text.labelXs, s.row)}>
          <span {...stylex.props(s.soft, s.nowrap)}>{j.lane} · {cardName(j.card)}</span>
          <span {...stylex.props(j.paused ? s.rose : s.moss, s.nowrap)}>{j.paused ? 'PAUSED' : 'OPEN'}</span>
          <span {...stylex.props(s.dim, s.wrap)}>
            {j.error ? `jobs not read: ${j.error}` : `${j.running.length} RUNNING · ${j.queued ?? '—'} QUEUED`}
            {j.paused && j.paused.since ? ` · SINCE ${span(now - j.paused.since)} AGO` : ''}
            {j.paused && j.paused.until ? ` · UNTIL IN ${span(j.paused.until - now)}` : ''}
          </span>
        </div>
      ))}
    </div>
  );
}

function HoldRow({ h, now }: { h: Hold; now: number }) {
  const times = holdTimes(h, now);
  const who = [h.by ? `BY ${h.by}` : '', times, h.why ? `WHY: ${h.why}` : ''].filter((x) => x !== '').join(' · ');
  const tone = h.kind === 'job_lane_paused' ? 'rose' : h.kind === 'unknown' ? 'crimson' : 'cyan';
  return (
    <div {...stylex.props(s.block)}>
      <div {...stylex.props(s.holdHead)}>
        <Chip tone={tone}>{KIND_LABEL[h.kind] ?? h.kind.toUpperCase()}</Chip>
        <span {...stylex.props(text.labelXs, s.soft, s.wrap)}>{cardName(h.card)}</span>
      </div>
      <span {...stylex.props(text.labelXs, s.soft, s.wrap)}>{h.what}</span>
      <span {...stylex.props(text.labelXs, s.dim, s.wrap)}>
        {who === '' ? 'no owner or reason reported' : who}
      </span>
    </div>
  );
}

export function LoadoutCard({ view }: { view: LaneView | null | undefined }) {
  const lv = laneView(view);
  if (!lv) {
    return (
      <Card label="LOADOUT · WHAT HOLDS WHAT" wide>
        <span {...stylex.props(text.labelXs, s.dim, s.wrap)}>
          {view && 'error' in view && view.error ? `the lane view failed: ${view.error}` : 'not reported by this server (it predates the lane view; restart to load it)'}
        </span>
      </Card>
    );
  }
  const tag = lv.swap.now ? 'SWAPPING' : lv.holds.length ? `${lv.holds.length} HOLD${lv.holds.length === 1 ? '' : 'S'}` : 'NO HOLDS';
  return (
    <Card label="LOADOUT · WHAT HOLDS WHAT" right={tag} wide flashKey={lv.swap.now ? `s${lv.swap.now.to}` : null} tone="cyan">
      {lv.cards.map((c) => (
        <CardBlock key={c.key} c={c} now={lv.at} />
      ))}
      <SwapBlock lv={lv} />
      <JobLanes jobs={lv.jobs} now={lv.at} />
      <div {...stylex.props(s.section)}>
        <Label>HOLDS</Label>
        {lv.holds.length === 0 ? (
          <span {...stylex.props(text.labelXs, s.dim)}>nothing holds either card: no pause, no swap, no lease</span>
        ) : (
          lv.holds.map((h, i) => <HoldRow key={`${h.kind}-${h.card}-${i}`} h={h} now={lv.at} />)
        )}
        {!lv.in_proxy && (
          <span {...stylex.props(text.labelXs, s.dim, s.wrap)}>read outside the proxy: requests in flight, swaps in progress and conversation holds are not visible from here</span>
        )}
      </div>
    </Card>
  );
}
