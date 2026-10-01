// Types and pure helpers over the dashboard's history: /dash/api/jjava
// (mcp/dash_jjava.py, the JJAVA page and the Skills page's injector panel)
// and /dash/api/perf (mcp/dash_perf.py, the SOKUDO page). Both routes are
// read-only and model-free; each takes a window: 1h | 6h | 24h | 7d | 30d.

export const WINDOWS = ['1h', '6h', '24h', '7d', '30d'] as const;
export type WindowName = (typeof WINDOWS)[number];
export const DEFAULT_WINDOW: WindowName = '24h';

export const jjavaPath = (w: WindowName = DEFAULT_WINDOW) => (w === DEFAULT_WINDOW ? '/dash/api/jjava' : `/dash/api/jjava/${w}`);
export const perfPath = (w: WindowName = DEFAULT_WINDOW) => (w === DEFAULT_WINDOW ? '/dash/api/perf' : `/dash/api/perf/${w}`);
export const JJAVA_PATH = jjavaPath();
export const PERF_PATH = perfPath();

export const isWindow = (x: unknown): x is WindowName => typeof x === 'string' && (WINDOWS as readonly string[]).includes(x);

export type Spread = { n: number; p50: number | null; p90: number | null; max: number | null };
export type Err = { error: string };
export const isErr = (x: unknown): x is Err => !!x && typeof x === 'object' && typeof (x as Err).error === 'string';

export type StatsWindow = { name: WindowName; since: number; until: number; bucket_s: number; buckets: number[]; names: string[] };

// ------------------------------------------------------------------- jjava --
export type QuestionSet = {
  name: string;
  type: 'noul' | 'choice' | 'score' | null;
  caller: string;
  n: number;
  picks: Record<string, number>;
  tiers: Record<string, number>;
  models: Record<string, number>;
  ties: number;
  orders_disagree: number;
  confidence_hist: number[] | null;
  confidence: Spread | null;
  noul_hist: number[] | null;
  noul_middle: number | null;
};

export type JevRoute = { method: string; route: string; requests: number; statuses: Record<string, number>; series: number[] };
export type JevRecent = {
  ts: number; route: string | null; status: number | null; requested: string | null; model: string | null;
  traffic: string | null; questions: number; ms: number | null; usage: { input_tokens?: number; output_tokens?: number } | null; error: string | null;
};
export type JevModelFacts = { model: string; available: boolean; why: string; priors_measured?: { letter_prior: boolean; label_bias: boolean } };

export type InjectorModel = {
  model: string; runs: number; injected: number; skipped: number;
  stage1_items_mean: number | null; stage1_skills_mean: number | null;
  stage2_asked_mean: number | null; stage2_passed_mean: number | null;
  stage3: Record<string, number>; chosen_mean: number | null;
  failures: Record<string, number>; why: Record<string, number>;
  need_hist: number[] | null; noul_hist: number[] | null;
  item_tiers: Record<string, number>; stage3_tiers: Record<string, number>;
  ms: Spread;
};

export type Injector = {
  models: InjectorModel[];
  question_sets: QuestionSet[];
  requests: number;
  skills_on: number;
  decider_turns: number;
  decider_failures: Record<string, number>;
  injector: { version?: string; questions?: string; max_skills?: number; thresholds?: unknown; levels?: string[]; profiles?: Record<string, { family: string; format: string; voice: string; max_items: number }>; error?: string } | null;
  skills_by_tier: Record<string, boolean> | null;
  note: string;
};

export type Priors = Record<string, {
  configured?: boolean; record?: boolean; measured?: string[]; unmeasured?: string[];
  priors?: { letter_prior: boolean; label_bias: boolean };
  tie_band?: { value: number; measured: boolean; source: string };
  error?: string;
}>;

export type Jjava = {
  at: number;
  window: StatsWindow;
  sources: {
    decisions: { path: string; exists: boolean; bytes_read: number; cut: boolean; rows: number; bad_lines: number; legacy_rows: number; with_latency: number; error?: string };
    jev_calls: { path: string; exists: boolean; rows: number; error?: string };
    stats: { enabled: boolean; queued: number; written: number; dropped: number; errors: number; last_error: string | null };
    requests: number;
  };
  usage: { by_question_set: Record<string, number[]>; by_model: Record<string, number[]>; by_caller: Record<string, number[]>; totals: Record<string, number>; decisions: number } | Err;
  latency: {
    per_read: Spread & { by_question_set: Record<string, Spread>; by_model: Record<string, Spread> };
    per_decision: Spread;
    per_burst: { decision_log: Spread; decider_turns: Spread; jev_calls: Spread };
    jev_per_read: Spread;
    series: { p50: (number | null)[]; p90: (number | null)[]; n: number[] };
    note: string;
  } | Err;
  question_sets: QuestionSet[] | Err;
  thresholds: { decide_turn: Record<string, unknown>; skill_inject: Record<string, unknown>; state: string; fired: Record<string, Record<string, number>>; rule: string } | Err;
  lane: {
    releases: number; by_why: Record<string, number>; lane_burst_ended: number[]; lane_kept: number[];
    recent: { ts: number; slot: number | null; why: string | null; by_why: string | null; released: number; skipped: string | null; cells_before: number | null; ms: number | null; method: string | null }[];
    recent_in_memory: unknown; lane: { slot: number | null; ranked: boolean; rank: number; kept: boolean } | null; note: string;
  } | Err;
  jev: {
    routes: JevRoute[]; unknown_routes: string[]; requests: number;
    requested: Record<string, number>; served: Record<string, number>; traffic: Record<string, number>; errors: Record<string, number>;
    usage: { input_tokens: number; output_tokens: number; series: number[] };
    latency: Spread; recent: JevRecent[];
    config: { accepted?: string[]; aliases?: string[]; named?: Record<string, string>; latest?: string; limits?: Record<string, unknown>; one_call_at_a_time?: boolean; request_id_header?: string; models?: Record<string, JevModelFacts>; error?: string };
    note: string;
  } | Err;
  priors: Priors | Err;
  injector: Injector | Err;
};

// -------------------------------------------------------------------- perf --
export type ModelPerf = {
  model: string;
  generations: number;
  roles: Record<string, number>;
  tokens: { completion: number; processed: number; reused: number };
  decode_tps: Spread;
  prompt_tps: Spread;
  decider_read_prompt_tps: Spread;
  series: { decode_p50: (number | null)[]; decode_p10: (number | null)[]; decode_p90: (number | null)[]; prompt_p50: (number | null)[]; n: number[] };
  by_ctx: { bin: string; n: number; decode: Spread; prompt: Spread }[];
  scatter: [number, number, string][];
};

export type GpuCard = {
  idx: number; name: string | null; uuid: string | null; total_mib: number | null; main?: boolean;
  series: { t: number[]; util_avg: (number | null)[]; util_max: (number | null)[]; used_max: (number | null)[]; watts_avg: (number | null)[]; temp_max: (number | null)[]; samples: number[] };
};

export type GateGroup = {
  arm: string; what: string; n: number;
  decode: { median: number | null; min: number | null; max: number | null; n: number };
  prompt: { median: number | null; min: number | null; max: number | null; n: number };
  prompt_n: number | null; ctx_used: number | null;
};
export type GateFile = {
  kind: string; source: string; file: string; started: string | null; finished: string | null;
  steps: string[] | null; arms: string[] | null; pass: Record<string, boolean>; groups: GateGroup[]; mtime: number; in_progress: boolean;
};

export type SwapRow = { ts: number; from: string[]; to: string | null; load_s: number | null; ok: boolean; how: string | null; left_loaded: string[] };

export type Perf = {
  at: number;
  window: StatsWindow;
  ctx_bins: string[];
  left_out: { graph: string; why: string }[];
  sources: { stats: Jjava['sources']['stats']; generations: number };
  models: ModelPerf[] | Err;
  gpus: { bucket_s: number; cards: GpuCard[]; rows: number } | Err;
  swaps: { rows: SwapRow[]; load_s: Record<string, Spread>; last_in_memory: Record<string, unknown> | null; table: Record<string, unknown> } | Err;
  gates: { files: GateFile[]; errors: { file: string; error: string }[]; globs: { kind: string; glob: string }[] } | Err;
};

// ----------------------------------------------------------------- helpers --
export const ok = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

/** A section, or null when the server said it failed (the panel prints the error). */
export function part<T>(x: T | Err | undefined | null): T | null {
  return x && !isErr(x) ? (x as T) : null;
}

export const errorOf = (x: unknown): string | null => (isErr(x) ? x.error : null);

/** The categorical slots, in fixed order (dataviz: never cycled). A fifth
 *  entity and beyond fold into OTHER. */
export const SLOTS = 4;

/**
 * The series to draw from `{name: values}`: the SLOTS largest by total in a
 * fixed order (largest first), the rest summed into "other". Colour follows
 * the rank of the TOTAL, so the same data always paints the same way.
 */
export function topSeries(by: Record<string, number[]>, slots = SLOTS): { name: string; values: number[]; total: number }[] {
  const rows = Object.entries(by).map(([name, values]) => ({ name, values, total: values.reduce((a, b) => a + (ok(b) ? b : 0), 0) }));
  rows.sort((a, b) => b.total - a.total || a.name.localeCompare(b.name));
  if (rows.length <= slots) return rows;
  const keep = rows.slice(0, slots - 1);
  const rest = rows.slice(slots - 1);
  const n = Math.max(...rest.map((r) => r.values.length));
  const values = Array.from({ length: n }, (_, i) => rest.reduce((a, r) => a + (ok(r.values[i]) ? (r.values[i] as number) : 0), 0));
  return [...keep, { name: `other (${rest.length})`, values, total: rest.reduce((a, r) => a + r.total, 0) }];
}

/** Fixed colour slots for the three main models, whatever their order in the data. */
export const MODEL_SLOT: Record<string, number> = { bonsai: 0, 'mirai-s': 1, 'flash-next': 2 };

/** The label under a bucket's start: HH:MM inside a day, MON 23 beyond. */
export function bucketLabel(ts: number, bucketS: number): string {
  const d = new Date(ts * 1000);
  const p2 = (k: number) => String(k).padStart(2, '0');
  if (bucketS >= 86400) return `${['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'][d.getDay()]} ${p2(d.getDate())}`;
  if (bucketS >= 6 * 3600) return `${p2(d.getDate())} ${p2(d.getHours())}:00`;
  return `${p2(d.getHours())}:${p2(d.getMinutes())}`;
}

export const msText = (x: number | null | undefined): string => (!ok(x) ? '—' : x >= 10000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`);
export const tpsText = (x: number | null | undefined): string => (!ok(x) ? '—' : x >= 100 ? `${Math.round(x)}` : x.toFixed(1));

/** "p50 420 ms · p90 610 ms · n 88", or "no measurement" when n is 0. */
export function spreadText(s: Spread | null | undefined, f: (x: number | null | undefined) => string = msText): string {
  if (!s || !s.n) return 'no measurement';
  return `p50 ${f(s.p50)} · p90 ${f(s.p90)} · n ${s.n}`;
}

/** Sum of a count series. */
export const total = (xs: (number | null)[] | undefined): number => (xs ?? []).reduce<number>((a, b) => a + (ok(b) ? b : 0), 0);

/** Sorted entries of a count map, largest first. */
export const ranked = (m: Record<string, number> | undefined | null): [string, number][] =>
  Object.entries(m ?? {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));

/** A 10-bin histogram's bin label: "0.0–0.1". */
export const binLabel = (i: number, bins = 10): string => `${(i / bins).toFixed(1)}–${((i + 1) / bins).toFixed(1)}`;

/** Status class of an HTTP status for its chip: moss 2xx, rose 4xx, crimson 5xx. */
export function statusTone(code: string | number): 'moss' | 'rose' | 'crimson' | 'muted' {
  const c = Number(code);
  if (c >= 200 && c < 300) return 'moss';
  if (c >= 400 && c < 500) return 'rose';
  if (c >= 500) return 'crimson';
  return 'muted';
}

/** The GPU card's short name: "5060 Ti", "A4000". */
export const shortCard = (name: string | null | undefined, idx: number): string => (name ?? `GPU${idx}`).replace(/^NVIDIA (GeForce )?(RTX )?/, '');

/** The injector's run counts, summed over models (the Skills page's headline). */
export function injectorTotals(inj: Injector | null): { runs: number; injected: number; skipped: number; failed: number } {
  const ms = inj?.models ?? [];
  return {
    runs: ms.reduce((a, m) => a + m.runs, 0),
    injected: ms.reduce((a, m) => a + m.injected, 0),
    skipped: ms.reduce((a, m) => a + m.skipped, 0),
    failed: ms.reduce((a, m) => a + total(Object.values(m.failures)), 0),
  };
}
