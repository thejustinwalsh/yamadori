// Types and pure helpers over mcp/dash_tokens.py's payload (/dash/api/tokens):
// tokens per window from the token ledger, GPU electricity from the power
// ledger, and the hosted-API estimate from the stored OpenRouter snapshot.
// Kept beside the panel rather than in types.ts so the two stay one change.

export const TOKENS_PATH = '/dash/api/tokens';
export const TOKENS_EVERY_MS = 5000; // the vitals cadence: live without a reload

export type TokenKinds = {
  generations: number;
  prompt_processed: number;
  prompt_cached: number;
  prompt_unsplit: number;
  completion: number;
  reasoning: number;
  reasoning_reported: number;
  no_usage: number;
};

export type Hosted = { usd: number; parts: { input: number; cached_input: number; output: number } } | null;

export type Electricity = {
  cents: number | null;
  kwh: number | null;
  days_measured: number;
  measured_hours: number;
  gap_hours: number;
  from: string | null;
  why?: string;
};

export type TokenWindow = {
  key: 'all' | 'd30' | 'week' | string;
  label: string;
  start: string | null;
  starts?: string;
  tokens: TokenKinds;
  priced: TokenKinds;
  by_role: Record<string, TokenKinds>;
  electricity: Electricity;
  hosted: { like_for_like: Hosted; average: Hosted };
  saved: { like_for_like: number | null; average: number | null };
};

export type PriceListing = {
  id: string;
  name?: string | null;
  input?: number | null;
  output?: number | null;
  cache_read?: number | null;
  cache_read_listed?: boolean;
  overrides_ignored?: boolean;
  missing?: boolean;
  why?: string;
};

export type Prices = {
  error?: string;
  like_for_like?: PriceListing;
  average?: {
    definition: string;
    members: PriceListing[];
    missing_members: string[];
    n: number;
    input: number | null;
    cache_read: number | null;
    output: number | null;
    cache_read_listed: number;
  };
  snapshot?: { date: string | null; fetched_at: number | null; source: string; listings: number | null };
  age_days?: number | null;
};

export type TokensPayload = {
  at: number;
  today: string;
  timezone: string;
  week_start: string;
  recording: { enabled: boolean; db: string; writes: number; errors: number; last_error: string | null };
  history: {
    first_at: number | null;
    first_day: string | null;
    last_at: number | null;
    days: number;
    accounts: number;
    gap: { source: string; turns: number | null; from: number | null; until: number | null; error?: string };
    backfill: string;
  };
  roles: string[];
  priced_roles: string[];
  windows: TokenWindow[];
  prices: Prices;
  basis: string[];
  estimate: boolean;
};

export function isTokens(p: unknown): p is TokensPayload {
  return !!p && typeof p === 'object' && Array.isArray((p as TokensPayload).windows) && 'history' in p && 'prices' in p;
}

const ok = (x: number | null | undefined): x is number => typeof x === 'number' && Number.isFinite(x);

/** Every token the window saw: prompt (processed, cached, unsplit) plus completion. */
export const totalTokens = (k: TokenKinds | undefined): number =>
  k ? k.prompt_processed + k.prompt_cached + k.prompt_unsplit + k.completion : 0;

/** 812, 45.6k, 1.23M, 4.56B: short enough for a table cell; the exact figure goes in its title. */
export function compact(x: number | null | undefined): string {
  if (!ok(x)) return '—';
  const a = Math.abs(x);
  if (a < 1000) return String(Math.round(x));
  const [div, unit] = a >= 1e9 ? [1e9, 'B'] : a >= 1e6 ? [1e6, 'M'] : [1e3, 'k'];
  const v = x / div;
  return `${Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)}${unit}`;
}

/** Dollars with enough digits to be non-zero; a negative saving keeps its sign. */
export function usd(x: number | null | undefined): string {
  if (!ok(x)) return '—';
  const a = Math.abs(x);
  const dp = a >= 100 ? 0 : a >= 1 ? 2 : a >= 0.01 ? 3 : 4;
  return `${x < 0 ? '−' : ''}$${a.toFixed(dp)}`;
}

/** A per-token price as dollars per million tokens. */
export function perMillion(p: number | null | undefined): string {
  if (!ok(p)) return '—';
  const m = p * 1e6;
  return `$${m >= 10 ? m.toFixed(2) : m >= 0.1 ? m.toFixed(3) : m.toFixed(4)}`;
}

export const ROLE_NAMES: Record<string, string> = {
  main: 'MAIN',
  second_brain: 'SECOND BRAIN',
  side_call: 'SIDE CALLS',
  internal: 'INTERNAL',
  warm: 'WARM PREFILL',
};

/** "SEP 24 14:03" in the browser's clock, from an epoch. */
export function stamp(epoch: number | null | undefined): string {
  if (!ok(epoch)) return '—';
  const d = new Date(epoch * 1000);
  const mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][d.getMonth()];
  return `${mon} ${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/** What the history line says: where the record starts, and the gap before it. */
export function historyLine(p: TokensPayload): string {
  const h = p.history;
  const g = h.gap;
  const gapText =
    g && ok(g.turns) && g.turns > 0
      ? `${g.turns.toLocaleString('en-US')} CLIENT TURNS IN THE CORPUS FROM ${stamp(g.from)} HAVE NO TOKEN RECORD`
      : 'NO EARLIER CLIENT TRAFFIC IN THE CORPUS';
  if (!ok(h.first_at)) {
    return p.recording.enabled
      ? `RECORDING · NO GENERATION YET · ${gapText}`
      : `NOT RECORDING IN THIS PROCESS · THE LEDGER STARTS WHEN THE PROXY, TOOLS API AND WORKER RESTART · ${gapText}`;
  }
  return `HISTORY FROM ${stamp(h.first_at)} · BEFORE IT: ${gapText}`;
}

/** Why the cached-input price is what it is, for one listing. */
export function cacheNote(l: PriceListing | undefined): string | null {
  if (!l || l.missing) return null;
  return l.cache_read_listed ? null : 'NO CACHE-READ PRICE LISTED · CACHED INPUT PRICED AS UNCACHED INPUT';
}
