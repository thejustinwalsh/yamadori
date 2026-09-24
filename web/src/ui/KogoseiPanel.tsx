// KŌGŌSEI · WATTS PER TOKEN (光合成, photosynthesis: how a tree turns light
// into growth; here, how the cards turn watts into tokens). The last 10
// minutes, polled every 2 s from /dash/api/power/series (mcp/power.py): each
// card's draw over the main model's generation and prompt-processing rates on
// one time axis, the idle baseline, energy per token on the main card, and
// the association between draw and rate. Hand-drawn SVG, design tokens only.
import * as stylex from '@stylexjs/stylex';
import type { Failure } from '../api/client';
import {
  bars,
  cardName,
  gaps,
  isSeries,
  joules,
  linePath,
  modelState,
  niceMax,
  rText,
  SERIES_EVERY_MS,
  SERIES_PATH,
  watts,
  type Series,
} from '../api/powerSeries';
import { usePoll } from '../api/usePoll';
import { colors, space } from '../tokens/tokens.stylex';
import { Panel } from './Panel';
import { Label } from './primitives';
import { PollState, StateView } from './StateView';
import { text } from './text';

const W = 480; // viewBox width; the SVG scales to the panel
const H_W = 56; // watts strip
const H_G = 34; // generation tok/s strip
const H_P = 22; // prompt-processing tok/s strip
const SC_W = 160;
const SC_H = 84;

const s = stylex.create({
  strip: {
    position: 'relative',
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`,
  },
  svg: { display: 'block', width: '100%' },
  tag: {
    position: 'absolute',
    top: '2px',
    left: '4px',
    right: '4px',
    display: 'flex',
    justifyContent: 'space-between',
    gap: space.spaceXs,
    pointerEvents: 'none',
    color: colors.outline,
  },
  charts: { display: 'flex', flexDirection: 'column', gap: '3px' },
  main: { fill: 'none', stroke: colors.primaryContainer, strokeWidth: 1.5 },
  other: { fill: 'none', stroke: colors.outline, strokeWidth: 1.2 },
  idle: { fill: 'none', stroke: colors.primaryContainer, strokeWidth: 1, strokeDasharray: '3 3', opacity: 0.55 },
  gen: { fill: colors.tertiaryContainer },
  pre: { fill: colors.tertiaryFixedDim, opacity: 0.6 },
  gap: { fill: colors.surfaceContainerHigh, opacity: 0.7 },
  dotGen: { fill: colors.tertiaryContainer },
  dotPre: { fill: colors.secondary, opacity: 0.8 },
  dotIdle: { fill: colors.outline, opacity: 0.6 },
  lower: { display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)', gap: space.spaceSm, alignItems: 'start' },
  readouts: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 },
  kv: { display: 'flex', justifyContent: 'space-between', gap: space.spaceXs, minWidth: 0 },
  k: { color: colors.outline, whiteSpace: 'nowrap' },
  v: { color: colors.primary, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  vMoss: { color: colors.primaryContainer },
  note: { margin: 0, color: colors.onSurfaceVariant },
  mainKey: { color: colors.primaryContainer },
  cyanKey: { color: colors.tertiaryContainer },
});

/** Polls /dash/api/power/series itself: it is the one panel that needs 2 s. */
export function KogoseiLive({ fill }: { fill?: boolean }) {
  const poll = usePoll<unknown>(SERIES_PATH, SERIES_EVERY_MS);
  return <KogoseiPanel d={isSeries(poll.data) ? poll.data : null} raw={poll.data} stale={poll.stale} failure={poll.failure} fill={fill} />;
}

type P = { d: Series | null; raw?: unknown; stale?: boolean; failure: Failure | null; fill?: boolean };

export function KogoseiPanel({ d, raw, stale, failure, fill }: P) {
  const st = d ? modelState(d) : null;
  return (
    <Panel
      kanji="光"
      title="KŌGŌSEI · WATTS PER TOKEN"
      tag={st ? st.tag : '—'}
      tagTone={st ? st.tone : 'muted'}
      flag={`${SERIES_EVERY_MS / 1000} s`}
      stale={stale}
      edge={st?.tone === 'moss' ? 'moss' : st?.tone === 'cyan' ? 'cyan' : undefined}
      fill={fill}
    >
      {!d ? <NoData raw={raw} failure={failure} /> : <Body d={d} note={st?.note ?? null} />}
    </Panel>
  );
}

function NoData({ raw, failure }: { raw: unknown; failure: Failure | null }) {
  if (raw !== null && raw !== undefined) return <StateView kind="error" title={`${SERIES_PATH} · not a series payload`} />;
  if (failure?.kind === 'http' && failure.status === 404)
    return <StateView kind="inert" title="this server predates the watts-vs-tokens ring" detail="Restart the proxy (mcp/server.py) to serve /dash/api/power/series." />;
  return <PollState path={SERIES_PATH} failure={failure} />;
}

function Body({ d, note }: { d: Series; note: string | null }) {
  const win = d.window_s;
  const mi = d.main_index;
  const main = mi === null ? null : d.gpus[mi];
  const others = d.gpus.map((g, i) => ({ g, i })).filter((x) => x.i !== mi);
  const wMax = niceMax(d.watts.flat(), 50);
  const gMax = niceMax(d.decode_tps, 10);
  const pMax = niceMax(d.prompt_tps, 100);
  const last = (xs: (number | null)[]) => [...xs].reverse().find((x) => typeof x === 'number') ?? null;
  const idleMain = mi === null ? null : (d.idle?.watts[mi] ?? null);
  const idleY = idleMain === null ? null : H_W - (Math.min(idleMain, wMax) / wMax) * H_W;
  const noRate = gaps(d.t, d.decode_tps, win, W, d.interval_s);
  const st = d.stats;
  return (
    <>
      <div {...stylex.props(s.charts)}>
        <div {...stylex.props(s.strip)}>
          <svg viewBox={`0 0 ${W} ${H_W}`} preserveAspectRatio="none" height={H_W} {...stylex.props(s.svg)} role="img" aria-label={`watts per card over the last ${Math.round(win / 60)} minutes`}>
            {others.map(({ g, i }) => (
              <path key={g.uuid ?? i} d={linePath(d.t, (d.watts[i] ?? []), win, wMax, W, H_W)} vectorEffect="non-scaling-stroke" {...stylex.props(s.other)} />
            ))}
            {idleY !== null && <line x1={0} x2={W} y1={idleY} y2={idleY} vectorEffect="non-scaling-stroke" {...stylex.props(s.idle)} />}
            {mi !== null && <path d={linePath(d.t, (d.watts[mi] ?? []), win, wMax, W, H_W)} vectorEffect="non-scaling-stroke" {...stylex.props(s.main)} />}
          </svg>
          <div {...stylex.props(text.labelXs, s.tag)}>
            <span>
              {main && <span {...stylex.props(s.mainKey)}>{cardName(main)} {watts(last(d.watts[mi as number] ?? []))}</span>}
              {others.map(({ g, i }) => (
                <span key={g.uuid ?? i}> · {cardName(g)} {watts(last(d.watts[i] ?? []))}</span>
              ))}
            </span>
            <span>{wMax} W</span>
          </div>
        </div>
        <div {...stylex.props(s.strip)}>
          <svg viewBox={`0 0 ${W} ${H_G}`} preserveAspectRatio="none" height={H_G} {...stylex.props(s.svg)} role="img" aria-label="generated tokens per second">
            {noRate.map((r, i) => <rect key={i} x={r.x} y={0} width={Math.max(r.w, 0.6)} height={H_G} {...stylex.props(s.gap)} />)}
            {bars(d.t, d.decode_tps, win, gMax, W, H_G, d.interval_s).map((b, i) => <rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} {...stylex.props(s.gen)} />)}
          </svg>
          <div {...stylex.props(text.labelXs, s.tag)}>
            <span {...stylex.props(s.cyanKey)}>GENERATION tok/s</span>
            <span>{gMax}</span>
          </div>
        </div>
        <div {...stylex.props(s.strip)}>
          <svg viewBox={`0 0 ${W} ${H_P}`} preserveAspectRatio="none" height={H_P} {...stylex.props(s.svg)} role="img" aria-label="prompt tokens processed per second">
            {noRate.map((r, i) => <rect key={i} x={r.x} y={0} width={Math.max(r.w, 0.6)} height={H_P} {...stylex.props(s.gap)} />)}
            {bars(d.t, d.prompt_tps, win, pMax, W, H_P, d.interval_s).map((b, i) => <rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} {...stylex.props(s.pre)} />)}
          </svg>
          <div {...stylex.props(text.labelXs, s.tag)}>
            <span>PROMPT tok/s</span>
            <span>{pMax}</span>
          </div>
        </div>
        <Label>
          −{Math.round(win / 60)} MIN · NOW · DASHED: IDLE {main ? cardName(main) : 'MAIN'} · SHADED: NO RATE
        </Label>
      </div>
      <div {...stylex.props(s.lower)}>
        <Scatter d={d} wMax={wMax} gMax={gMax} />
        <div {...stylex.props(s.readouts)}>
          <KV k="J / TOKEN" v={joules(st.j_per_token)} title={`main card only, ${st.j_n ?? 0} generating samples, ${st.tokens ?? 0} tokens`} moss />
          <KV k="OVER IDLE" v={joules(st.marginal_j_per_token)} title="main-card draw above its idle baseline, per token" />
          <KV k="J / GEN · PROMPT" v={`${joules(st.j_per_gen_token)} · ${joules(st.j_per_prompt_token)}`} title={`from ${st.gen_only_samples ?? 0} decode-only and ${st.prompt_only_samples ?? 0} prefill-only samples`} />
          <KV
            k="IDLE"
            v={d.idle ? d.gpus.map((g, i) => `${cardName(g)} ${watts(d.idle?.watts[i])}`).join(' · ') : 'NO IDLE SECOND YET'}
            title={d.idle ? `${d.idle.n} idle samples${d.idle.from_window ? '' : ', from an earlier window'}` : undefined}
          />
          <KV k="W ~ GEN tok/s" v={rText(st.r_decode, st.r_n, st.r_min, win)} />
        </div>
      </div>
      {note && <p {...stylex.props(text.labelXs, s.note)}>{note.toUpperCase()}</p>}
      <p {...stylex.props(text.labelXs, s.note)} title={`${d.source.tokens}; ${d.source.undercount}. Watts: ${d.source.watts}.`}>
        ASSOCIATION, NOT CAUSE · {main ? cardName(main).toUpperCase() : 'MAIN CARD'} TOKENS ONLY · RATES ARE A FLOOR
      </p>
    </>
  );
}

function KV({ k, v, title, moss }: { k: string; v: string; title?: string; moss?: boolean }) {
  return (
    <div {...stylex.props(text.labelXs, s.kv)} title={title}>
      <span {...stylex.props(s.k)}>{k}</span>
      <span {...stylex.props(text.num, s.v, moss && s.vMoss)}>{v}</span>
    </div>
  );
}

/** Main-card W (y) against generated tok/s (x), one mark per answered second. */
function Scatter({ d, wMax, gMax }: { d: Series; wMax: number; gMax: number }) {
  const mi = d.main_index;
  const pts: { x: number; y: number; kind: 'gen' | 'pre' | 'idle' }[] = [];
  if (mi !== null) {
    for (let i = 0; i < d.t.length; i++) {
      const w = d.watts[mi]?.[i];
      const g = d.decode_tps[i];
      if (typeof w !== 'number' || typeof g !== 'number') continue;
      const pre = d.prompt_tps[i] ?? 0;
      pts.push({ x: (g / gMax) * (SC_W - 4) + 2, y: SC_H - 2 - (Math.min(w, wMax) / wMax) * (SC_H - 4), kind: pre > 0 ? 'pre' : g > 0 ? 'gen' : 'idle' });
    }
  }
  return (
    <div {...stylex.props(s.strip)}>
      <svg viewBox={`0 0 ${SC_W} ${SC_H}`} preserveAspectRatio="none" height={SC_H} {...stylex.props(s.svg)} role="img" aria-label={`main card watts against generated tokens per second, ${pts.length} seconds`}>
        {pts.map((p, i) => (
          <rect key={i} x={p.x - 1.2} y={p.y - 1.2} width={2.4} height={2.4} {...stylex.props(p.kind === 'gen' ? s.dotGen : p.kind === 'pre' ? s.dotPre : s.dotIdle)} />
        ))}
      </svg>
      <div {...stylex.props(text.labelXs, s.tag)}>
        <span>W ↑ · tok/s →</span>
        <span>{pts.length}</span>
      </div>
    </div>
  );
}
