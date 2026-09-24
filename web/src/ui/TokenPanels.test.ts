// Static renders of MIZU · TOKENS and SETSUYAKU · SAVINGS (vitest runs in
// node: react-dom/server markup), plus the pure helpers behind them. The
// payload is shaped exactly as mcp/dash_tokens.py payload() returns it.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { cacheNote, compact, historyLine, isTokens, perMillion, totalTokens, usd, type TokenKinds, type TokensPayload, type TokenWindow } from '../api/tokens';
import { SavingsPanel, TokensPanel } from './TokenPanels';

const kinds = (over: Partial<TokenKinds> = {}): TokenKinds => ({
  generations: 0,
  prompt_processed: 0,
  prompt_cached: 0,
  prompt_unsplit: 0,
  completion: 0,
  reasoning: 0,
  reasoning_reported: 0,
  no_usage: 0,
  ...over,
});

const byRole = (main: Partial<TokenKinds>, second: Partial<TokenKinds> = {}) => ({
  main: kinds(main),
  second_brain: kinds(second),
  side_call: kinds(),
  internal: kinds(),
  warm: kinds(),
});

const win = (key: string, label: string, start: string | null, t: Partial<TokenKinds>, over: Partial<TokenWindow> = {}): TokenWindow => ({
  key,
  label,
  start,
  tokens: kinds(t),
  priced: kinds(t),
  by_role: byRole(t),
  electricity: { cents: 50, kwh: 2, days_measured: 2, measured_hours: 48, gap_hours: 0, from: '2026-09-22' },
  hosted: { like_for_like: { usd: 0.54, parts: { input: 0.02, cached_input: 0.2, output: 0.32 } }, average: { usd: 4.5, parts: { input: 0.2, cached_input: 0.3, output: 4 } } },
  saved: { like_for_like: 0.04, average: 4.0 },
  ...over,
});

const payload = (over: Partial<TokensPayload> = {}): TokensPayload => ({
  at: 1790200000,
  today: '2026-09-23',
  timezone: 'machine local time (America/Detroit), the power ledger\'s clock',
  week_start: 'MONDAY 00:00 LOCAL (AMERICA/DETROIT)',
  recording: { enabled: true, db: 'index/token_ledger.sqlite3', writes: 3, errors: 0, last_error: null },
  history: {
    first_at: 1790085600,
    first_day: '2026-09-22',
    last_at: 1790200000,
    days: 2,
    accounts: 1,
    gap: { source: 'index/corpus.sqlite3 (client turns through :1234)', turns: 1027, from: 1790002350, until: 1790085600 },
    backfill: 'none: no earlier record carries token counts',
  },
  roles: ['main', 'second_brain', 'side_call', 'internal', 'warm'],
  priced_roles: ['main', 'second_brain', 'side_call', 'internal'],
  windows: [
    win('all', 'ALL TIME', null, { generations: 2, prompt_processed: 100_000, prompt_cached: 1_000_000, completion: 200_000 }),
    win('d30', 'LAST 30 DAYS', '2026-08-25', { generations: 2, prompt_processed: 100_000, prompt_cached: 1_000_000, completion: 200_000 }),
    win('week', 'THIS WEEK', '2026-09-21', { generations: 1, prompt_processed: 1_000, prompt_cached: 0, completion: 500 }, { starts: 'MONDAY 00:00 LOCAL (AMERICA/DETROIT)', saved: { like_for_like: -0.3, average: 1.2 } }),
  ],
  prices: {
    like_for_like: { id: 'qwen/qwen3.5-27b', name: 'Qwen: Qwen3.5-27B', input: 1.95e-7, output: 1.56e-6, cache_read: 1.95e-7, cache_read_listed: false, why: 'Bonsai 2 27B is a Qwen3.5-27B-family model' },
    average: {
      definition: 'median of each price, taken separately, over a named set of 11 models',
      members: [{ id: 'anthropic/claude-sonnet-5' }, { id: 'openai/gpt-6-sol' }],
      missing_members: ['z-ai/glm-5.3'],
      n: 10,
      input: 1.75e-6,
      cache_read: 2e-7,
      output: 1e-5,
      cache_read_listed: 10,
    },
    snapshot: { date: '2026-09-24', fetched_at: 1790261633, source: 'https://openrouter.ai/api/v1/models', listings: 458 },
    age_days: 0.1,
  },
  basis: ['ESTIMATE, NOT A BILL', 'SAVED = HOSTED COST - OUR GPU ELECTRICITY OVER THE SAME DAYS. EXCLUDES HARDWARE PURCHASE AND DEPRECIATION'],
  estimate: true,
  ...over,
});

const tokens = (d: TokensPayload | null, failure: Parameters<typeof TokensPanel>[0]['failure'] = null, raw?: unknown) =>
  renderToStaticMarkup(createElement(TokensPanel, { d, failure, raw }));
const savings = (d: TokensPayload | null, failure: Parameters<typeof SavingsPanel>[0]['failure'] = null) =>
  renderToStaticMarkup(createElement(SavingsPanel, { d, failure }));

describe('TokensPanel', () => {
  it('shows the live total for each window, exact, with the week start', () => {
    const html = tokens(payload());
    expect(html).toContain('MIZU · TOKENS');
    expect(html).toContain('LIVE');
    expect(html).toContain('ALL TIME');
    expect(html).toContain('LAST 30 DAYS');
    expect(html).toContain('THIS WEEK');
    expect(html).toContain('>1,300,000<'); // 100k + 1M + 200k
    expect(html).toContain('>1,500<');
    expect(html).toContain('FROM 2026-09-21 · MON 00:00');
    expect(html).toContain('WEEK FROM MONDAY 00:00 LOCAL (AMERICA/DETROIT)');
  });
  it('breaks tokens down by kind and by role, compactly, exact in the title', () => {
    const html = tokens(payload());
    expect(html).toContain('PROMPT · PROCESSED');
    expect(html).toContain('PROMPT · FROM CACHE');
    expect(html).toContain('COMPLETION');
    expect(html).toContain('title="1,000,000">1.00M<');
    expect(html).toContain('MAIN');
    expect(html).not.toContain('SIDE CALLS'); // a role with nothing is not listed
    expect(html).not.toContain('PROMPT · NO SPLIT'); // shown only when some prompt had no split
  });
  it('says reasoning is not reported rather than showing zero', () => {
    const html = tokens(payload());
    expect(html).toContain('REASONING NOT REPORTED BY THE SERVER');
    const w = payload().windows.map((x) => ({ ...x, tokens: { ...x.tokens, reasoning: 900, reasoning_reported: 2 } }));
    const rep = tokens(payload({ windows: w }));
    expect(rep).toContain('OF WHICH REASONING (2/2)');
    expect(rep).not.toContain('REASONING NOT REPORTED');
  });
  it('labels where the history starts and the unrecorded gap before it', () => {
    const html = tokens(payload());
    expect(html).toContain('HISTORY FROM');
    expect(html).toContain('1,027 CLIENT TURNS IN THE CORPUS');
    expect(html).toContain('HAVE NO TOKEN RECORD');
  });
  it('a process that is not recording says so, and a failed write is loud', () => {
    const off = payload({ recording: { enabled: false, db: 'x', writes: 0, errors: 0, last_error: null }, history: { ...payload().history, first_at: null, first_day: null } });
    const html = tokens(off);
    expect(html).toContain('NOT RECORDING');
    expect(html).toContain('THE LEDGER STARTS WHEN THE PROXY, TOOLS API AND WORKER RESTART');
    const bad = tokens(payload({ recording: { enabled: true, db: 'x', writes: 1, errors: 2, last_error: 'OperationalError: disk I/O error' } }));
    expect(bad).toContain('2 LEDGER WRITES FAILED · OperationalError: disk I/O error');
  });
  it('an older server (404) asks for a restart; no data yet is a loading state', () => {
    expect(tokens(null, { kind: 'http', status: 404, message: 'not found' })).toContain('predates the token ledger');
    expect(tokens(null)).toContain('reading /dash/api/tokens');
    expect(tokens(null, null, { something: 'else' })).toContain('not a tokens payload');
  });
});

describe('SavingsPanel', () => {
  it('is labelled an estimate, names the snapshot date, and excludes hardware', () => {
    const html = savings(payload());
    expect(html).toContain('SETSUYAKU · SAVINGS');
    expect(html).toContain('ESTIMATE');
    expect(html).toContain('openrouter · 2026-09-24');
    expect(html).toContain('SNAPSHOT 2026-09-24');
    expect(html).toContain('EXCLUDES HARDWARE PURCHASE AND DEPRECIATION');
  });
  it('shows hosted cost, our electricity and the saving per window', () => {
    const html = savings(payload());
    expect(html).toContain('HOSTED · QWEN3.5-27B');
    expect(html).toContain('HOSTED · MEDIAN OF 10');
    expect(html).toContain('OUR GPU ELECTRICITY');
    expect(html).toContain('$0.540');
    expect(html).toContain('$4.50');
    expect(html).toContain('$0.500'); // 50 cents of electricity
    expect(html).toContain('−$0.300'); // a negative saving keeps its sign
  });
  it('prices input, cached input and output per million, and flags a missing cache-read price', () => {
    const html = savings(payload());
    expect(html).toContain('qwen/qwen3.5-27b');
    expect(html).toContain('$0.195');
    expect(html).toContain('$0.195*');
    expect(html).toContain('$1.56');
    expect(html).toContain('NO CACHE-READ PRICE LISTED · CACHED INPUT PRICED AS UNCACHED INPUT');
    expect(html).toContain('MEDIAN OF 10');
    expect(html).toContain('$1.75');
    expect(html).toContain('$10.00');
    expect(html).toContain('NOT IN THIS SNAPSHOT: z-ai/glm-5.3');
    expect(html).toContain('anthropic/claude-sonnet-5');
  });
  it('no snapshot is an error, and no saving is claimed', () => {
    const w = payload().windows.map((x) => ({ ...x, hosted: { like_for_like: null, average: null }, saved: { like_for_like: null, average: null } }));
    const html = savings(payload({ prices: { error: 'no price snapshot yet: the worker enqueues prices.refresh' }, windows: w }));
    expect(html).toContain('no price snapshot');
    expect(html).toContain('prices.refresh');
    expect(html).not.toContain('SAVED · ALL TIME');
  });
});

describe('token helpers', () => {
  it('isTokens needs windows, history and prices', () => {
    expect(isTokens(payload())).toBe(true);
    expect(isTokens({ windows: [] })).toBe(false);
    expect(isTokens(null)).toBe(false);
  });
  it('totals every prompt kind plus completion', () => {
    expect(totalTokens(kinds({ prompt_processed: 1, prompt_cached: 2, prompt_unsplit: 3, completion: 4, reasoning: 99 }))).toBe(10);
    expect(totalTokens(undefined)).toBe(0);
  });
  it('compacts counts', () => {
    expect(compact(812)).toBe('812');
    expect(compact(45_600)).toBe('45.6k');
    expect(compact(1_234_567)).toBe('1.23M');
    expect(compact(4_560_000_000)).toBe('4.56B');
    expect(compact(null)).toBe('—');
  });
  it('formats dollars and per-million prices', () => {
    expect(usd(-0.3)).toBe('−$0.300');
    expect(usd(1234.5)).toBe('$1235');
    expect(usd(0.0012)).toBe('$0.0012');
    expect(perMillion(1.95e-7)).toBe('$0.195');
    expect(perMillion(1e-5)).toBe('$10.00');
    expect(perMillion(null)).toBe('—');
  });
  it('the cache note appears only for a listing without a cache-read price', () => {
    expect(cacheNote({ id: 'a', cache_read_listed: false })).toContain('NO CACHE-READ PRICE');
    expect(cacheNote({ id: 'a', cache_read_listed: true })).toBeNull();
    expect(cacheNote(undefined)).toBeNull();
  });
  it('the history line with nothing recorded but recording on', () => {
    const p = payload({ history: { ...payload().history, first_at: null, gap: { ...payload().history.gap, turns: 0 } } });
    expect(historyLine(p)).toBe('RECORDING · NO GENERATION YET · NO EARLIER CLIENT TRAFFIC IN THE CORPUS');
  });
});
