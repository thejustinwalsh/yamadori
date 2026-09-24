// MIZU · TOKENS and SETSUYAKU · SAVINGS, from /dash/api/tokens
// (mcp/dash_tokens.py), polled every 5 s so the counters move without a
// reload. Tokens over ALL TIME / LAST 30 DAYS / THIS WEEK by kind and by
// role; GPU electricity over the same days; and what the same tokens would
// have cost on a hosted API, labelled an estimate that excludes hardware.
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { dollarsText } from '../api/power';
import {
  cacheNote,
  compact,
  historyLine,
  isTokens,
  perMillion,
  ROLE_NAMES,
  stamp,
  totalTokens,
  TOKENS_EVERY_MS,
  TOKENS_PATH,
  usd,
  type PriceListing,
  type TokenKinds,
  type TokensPayload,
  type TokenWindow,
} from '../api/tokens';
import type { Failure } from '../api/client';
import { usePoll } from '../api/usePoll';
import { n } from '../format';
import { colors } from '../tokens/tokens.stylex';
import { ErrorBoundary } from './ErrorBoundary';
import { Panel } from './Panel';
import { Label, layout, Stat } from './primitives';
import { PollState, StateView } from './StateView';
import { SectionedTable } from './Table';
import { text } from './text';

const s = stylex.create({
  note: { margin: 0, color: colors.onSurfaceVariant },
  err: { margin: 0, color: colors.secondary },
  good: { color: colors.primaryContainer },
  bad: { color: colors.secondary },
});

type P = { d: TokensPayload | null; raw?: unknown; stale?: boolean; failure: Failure | null; fill?: boolean };

/** One poll of /dash/api/tokens, for screens that place the two panels in
 *  different cells. */
export function useTokens() {
  const poll = usePoll<unknown>(TOKENS_PATH, TOKENS_EVERY_MS);
  return { d: isTokens(poll.data) ? poll.data : null, raw: poll.data, stale: poll.stale, failure: poll.failure };
}

/** Both panels on one poll. */
export function TokenPanels({ fill }: { fill?: boolean }) {
  const poll = usePoll<unknown>(TOKENS_PATH, TOKENS_EVERY_MS);
  const d = isTokens(poll.data) ? poll.data : null;
  return (
    <>
      <ErrorBoundary what="MIZU · TOKENS" source={TOKENS_PATH}>
        <TokensPanel d={d} raw={poll.data} stale={poll.stale} failure={poll.failure} />
      </ErrorBoundary>
      <ErrorBoundary what="SETSUYAKU · SAVINGS" source={TOKENS_PATH} fill={fill}>
        <SavingsPanel d={d} raw={poll.data} stale={poll.stale} failure={poll.failure} fill={fill} />
      </ErrorBoundary>
    </>
  );
}

function NoData({ raw, failure }: { raw: unknown; failure: Failure | null }) {
  if (raw !== null && raw !== undefined) return <StateView kind="error" title={`${TOKENS_PATH} · not a tokens payload`} />;
  if (failure?.kind === 'http' && failure.status === 404)
    return <StateView kind="inert" title="this server predates the token ledger" detail="Restart the proxy (mcp/server.py) to serve /dash/api/tokens." />;
  return <PollState path={TOKENS_PATH} failure={failure} />;
}

/** A window's column head. */
const head = (w: TokenWindow) => (w.key === 'week' ? 'WEEK' : w.key === 'd30' ? '30 D' : 'ALL');

type KindRow = { key: string; label: string; get: (k: TokenKinds) => number | null };

export function TokensPanel({ d, raw, stale, failure, fill }: P) {
  return (
    <Panel
      kanji="水"
      title="MIZU · TOKENS"
      tag={d ? (d.recording.enabled ? 'LIVE' : 'NOT RECORDING') : '—'}
      tagTone={d && d.recording.enabled ? 'moss' : 'rose'}
      flag={`token ledger · ${TOKENS_EVERY_MS / 1000} s`}
      stale={stale}
      edge="cyan"
      fill={fill}
    >
      {!d ? <NoData raw={raw} failure={failure} /> : <TokensBody d={d} />}
    </Panel>
  );
}

function TokensBody({ d }: { d: TokensPayload }) {
  const ws = d.windows;
  const all = ws.find((w) => w.key === 'all');
  const rr = all?.tokens.reasoning_reported ?? 0;
  const gens = all?.tokens.generations ?? 0;
  const kinds: KindRow[] = [
    { key: 'pp', label: 'PROMPT · PROCESSED', get: (k) => k.prompt_processed },
    { key: 'pc', label: 'PROMPT · FROM CACHE', get: (k) => k.prompt_cached },
    ...((all?.tokens.prompt_unsplit ?? 0) > 0 ? [{ key: 'pu', label: 'PROMPT · NO SPLIT', get: (k: TokenKinds) => k.prompt_unsplit }] : []),
    { key: 'c', label: 'COMPLETION', get: (k) => k.completion },
    { key: 'r', label: rr ? `· OF WHICH REASONING (${n(rr)}/${n(gens)})` : '· OF WHICH REASONING', get: (k) => (rr ? k.reasoning : null) },
    { key: 'g', label: 'GENERATIONS', get: (k) => k.generations },
  ];
  const roles = d.roles.filter((r) => ws.some((w) => totalTokens(w.by_role[r]) > 0));
  return (
    <>
      {d.recording.errors > 0 && (
        <p {...stylex.props(text.labelXs, s.err)}>
          [ ! ] {d.recording.errors} LEDGER WRITE{d.recording.errors === 1 ? '' : 'S'} FAILED{d.recording.last_error ? ` · ${d.recording.last_error}` : ''}
        </p>
      )}
      <div {...stylex.props(layout.grid3)}>
        {ws.map((w) => (
          <Stat
            key={w.key}
            label={w.label}
            value={<span title={`${n(totalTokens(w.tokens))} tokens`}>{n(totalTokens(w.tokens))}</span>}
            sub={w.key === 'week' ? `FROM ${w.start ?? '—'} · MON 00:00` : w.start ? `FROM ${w.start}` : `SINCE ${d.history.first_day ?? '—'}`}
            tone={w.key === 'all' ? 'moss' : w.key === 'week' ? 'cyan' : undefined}
          />
        ))}
      </div>
      {/* One table, two sections (by kind, by role), so the window columns
          line up down the panel. */}
      <SectionedTable
        caption="tokens by kind and by role, per window"
        sections={[
          {
            key: 'kind',
            rows: kinds,
            rowKey: (r: KindRow) => r.key,
            columns: [
              { key: 'k', head: 'tokens', cell: (r: KindRow) => r.label },
              ...ws.map((w) => ({
                key: w.key,
                head: head(w),
                num: true,
                cell: (r: KindRow) => {
                  const v = r.get(w.tokens);
                  return v === null ? '—' : <span title={n(v)}>{compact(v)}</span>;
                },
              })),
            ],
          },
          ...(roles.length
            ? [
                {
                  key: 'role',
                  rows: roles,
                  rowKey: (r: string) => r,
                  columns: [
                    { key: 'r', head: 'by role', cell: (r: string) => ROLE_NAMES[r] ?? r.toUpperCase() },
                    ...ws.map((w) => ({
                      key: w.key,
                      head: head(w),
                      num: true,
                      cell: (r: string) => {
                        const v = totalTokens(w.by_role[r]);
                        return <span title={n(v)}>{compact(v)}</span>;
                      },
                    })),
                  ],
                },
              ]
            : []),
        ]}
      />
      <div {...stylex.props(layout.stack)}>
        <p {...stylex.props(text.labelXs, s.note)}>{historyLine(d)}</p>
        <p {...stylex.props(text.labelXs, s.note)}>
          DAYS IN {d.timezone.toUpperCase()} · WEEK FROM {d.week_start}
          {rr ? '' : ' · REASONING NOT REPORTED BY THE SERVER (INSIDE COMPLETION)'}
          {d.roles.includes('warm') ? ' · WARM PREFILL COUNTED, NOT PRICED' : ''}
        </p>
      </div>
    </>
  );
}

type CostRow = { key: string; label: string; cell: (w: TokenWindow) => ReactNode };

function signed(x: number | null) {
  if (x === null) return '—';
  return <span {...stylex.props(x >= 0 ? s.good : s.bad)}>{usd(x)}</span>;
}

export function SavingsPanel({ d, raw, stale, failure, fill }: P) {
  const p = d?.prices;
  const date = p?.snapshot?.date;
  const old = typeof p?.age_days === 'number' && p.age_days > 2;
  return (
    <Panel
      kanji="節"
      title="SETSUYAKU · SAVINGS"
      tag="ESTIMATE"
      tagTone="rose"
      flag={date ? `openrouter · ${date}` : 'openrouter'}
      stale={stale}
      fill={fill}
    >
      {!d ? <NoData raw={raw} failure={failure} /> : <SavingsBody d={d} old={old} />}
    </Panel>
  );
}

function SavingsBody({ d, old }: { d: TokensPayload; old: boolean }) {
  const p = d.prices;
  const ws = d.windows;
  const lfl = p.like_for_like;
  const avg = p.average;
  const lflName = lfl?.name ?? lfl?.id ?? 'LIKE-FOR-LIKE';
  const rows: CostRow[] = [
    { key: 'l', label: `HOSTED · ${lflName.replace(/^Qwen: /, '')}`.toUpperCase(), cell: (w) => usd(w.hosted.like_for_like?.usd) },
    { key: 'a', label: `HOSTED · MEDIAN OF ${avg?.n ?? '—'}`, cell: (w) => usd(w.hosted.average?.usd) },
    {
      key: 'e',
      label: 'OUR GPU ELECTRICITY',
      cell: (w) => <span title={w.electricity.why ?? `${w.electricity.days_measured} day(s) measured from ${w.electricity.from ?? '—'}`}>{dollarsText(w.electricity.cents === null ? null : w.electricity.cents / 100)}</span>,
    },
    { key: 'sl', label: 'SAVED VS LIKE-FOR-LIKE', cell: (w) => signed(w.saved.like_for_like) },
    { key: 'sa', label: 'SAVED VS MEDIAN', cell: (w) => signed(w.saved.average) },
  ];
  const all = ws.find((w) => w.key === 'all');
  return (
    <>
      {p.error ? <StateView kind="error" title="no price snapshot" detail={p.error} /> : null}
      {all && !p.error ? (
        <div {...stylex.props(layout.grid2)}>
          <Stat label="SAVED · ALL TIME · LIKE-FOR-LIKE" value={signed(all.saved.like_for_like)} sub={`VS ${lflName.replace(/^Qwen: /, '').toUpperCase()}`} />
          <Stat label="SAVED · ALL TIME · MEDIAN" value={signed(all.saved.average)} sub={`VS MEDIAN OF ${avg?.n ?? '—'} HOSTED MODELS`} />
        </div>
      ) : null}
      {/* One table, two sections: costs per window, then the prices behind
          them. Two stacked tables size their columns separately and their
          rows never line up. */}
      <SectionedTable
        caption="hosted cost, electricity and saving per window; hosted prices in dollars per million tokens"
        sections={[
          {
            key: 'cost',
            rows,
            rowKey: (r: CostRow) => r.key,
            columns: [{ key: 'k', head: 'dollars', cell: (r: CostRow) => r.label }, ...ws.map((w) => ({ key: w.key, head: head(w), num: true, cell: (r: CostRow) => r.cell(w) }))],
          },
          ...(p.error
            ? []
            : [
                {
                  key: 'price',
                  rows: [lfl, avg ? { id: 'median', name: `MEDIAN OF ${avg.n}`, input: avg.input, cache_read: avg.cache_read, output: avg.output, cache_read_listed: avg.cache_read_listed === avg.n } : undefined].filter(Boolean) as PriceListing[],
                  rowKey: (r: PriceListing) => r.id,
                  columns: [
                    { key: 'm', head: '$ / M tokens', cell: (r: PriceListing) => (r.id === 'median' ? r.name : r.id) },
                    { key: 'i', head: 'input', num: true, cell: (r: PriceListing) => (r.missing ? 'NOT LISTED' : perMillion(r.input)) },
                    { key: 'c', head: 'cached', num: true, cell: (r: PriceListing) => (r.missing ? '—' : `${perMillion(r.cache_read)}${r.cache_read_listed ? '' : '*'}`) },
                    { key: 'o', head: 'output', num: true, cell: (r: PriceListing) => (r.missing ? '—' : perMillion(r.output)) },
                  ],
                },
              ]),
        ]}
      />
      <div {...stylex.props(layout.stack)}>
        {cacheNote(lfl) && <p {...stylex.props(text.labelXs, s.note)}>* {lfl?.id}: {cacheNote(lfl)}</p>}
        {avg && avg.cache_read_listed < avg.n && (
          <p {...stylex.props(text.labelXs, s.note)}>* MEDIAN: {avg.n - avg.cache_read_listed} OF {avg.n} LIST NO CACHE-READ PRICE; THEIR INPUT PRICE STANDS IN</p>
        )}
        {lfl?.why && <p {...stylex.props(text.labelXs, s.note)}>LIKE-FOR-LIKE: {lfl.why}</p>}
        {avg && (
          <p {...stylex.props(text.labelXs, s.note)}>
            MEDIAN: {avg.definition.toUpperCase()} · {avg.members.map((m) => m.id).join(', ')}
            {avg.missing_members.length ? ` · NOT IN THIS SNAPSHOT: ${avg.missing_members.join(', ')}` : ''}
          </p>
        )}
        {d.basis.map((line) => (
          <p key={line} {...stylex.props(text.labelXs, s.note)}>
            {line}
          </p>
        ))}
        <Label>
          PRICES: OPENROUTER MODEL LIST · SNAPSHOT {p.snapshot?.date ?? 'NONE'}
          {p.snapshot?.fetched_at ? ` · FETCHED ${stamp(p.snapshot.fetched_at)}` : ''}
          {old ? ' · OLDER THAN 2 DAYS' : ''}
        </Label>
      </div>
    </>
  );
}
