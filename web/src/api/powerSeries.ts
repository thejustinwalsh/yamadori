// Types and pure helpers over mcp/power.py series() (/dash/api/power/series):
// the last 10 minutes of per-card watts beside the main model's tokens per
// second, the idle baseline, energy per token and the correlation.

export const SERIES_PATH = '/dash/api/power/series';
export const SERIES_EVERY_MS = 2000;

export type SeriesGpu = { uuid: string | null; index: number | null; name: string | null; main: boolean };

export type SeriesStats = {
  samples: number;
  ok_samples: number;
  generating_samples: number;
  window_s: number;
  j_per_token?: number | null;
  j_n?: number;
  tokens?: number;
  marginal_j_per_token?: number | null;
  j_per_gen_token?: number | null;
  gen_only_samples?: number;
  j_per_prompt_token?: number | null;
  prompt_only_samples?: number;
  gen_watts_mean?: number | null;
  gen_decode_tps_mean?: number | null;
  gen_prompt_tps_mean?: number | null;
  r_decode?: number | null;
  r_prompt?: number | null;
  r_n?: number;
  r_min?: number;
};

export type Series = {
  running: boolean;
  window_s: number;
  interval_s: number;
  now: number;
  model: { name: string; state: string | null; why: string | null; busy: number | null };
  gpus: SeriesGpu[];
  main_index: number | null;
  t: number[];
  watts: (number | null)[][];
  decode_tps: (number | null)[];
  prompt_tps: (number | null)[];
  busy: (number | null)[];
  idle: { watts: (number | null)[]; n: number; at: number; from_window: boolean } | null;
  stats: SeriesStats;
  source: { watts: string; tokens: string; undercount: string };
};

export function isSeries(p: unknown): p is Series {
  return !!p && typeof p === 'object' && Array.isArray((p as Series).t) && Array.isArray((p as Series).watts) && 'stats' in p;
}

const ok = (x: number | null | undefined): x is number => typeof x === 'number' && Number.isFinite(x);

/** "5060 Ti", "A4000": the card name without the vendor prefix. */
export const cardName = (g: SeriesGpu): string => (g.name ?? `GPU${g.index ?? '?'}`).replace(/^NVIDIA (GeForce )?(RTX )?/, '');

/** The panel's state, in words: what the model is doing right now. */
export function modelState(d: Series): { tag: string; tone: 'moss' | 'cyan' | 'rose' | 'muted'; note: string | null } {
  const st = d.model.state;
  if (st === 'not_loaded') return { tag: `${d.model.name.toUpperCase()} NOT LOADED`, tone: 'rose', note: `llama-swap lists no ${d.model.name}; nothing is loaded to find out` };
  if (st === 'swap_down') return { tag: 'LLAMA-SWAP DOWN', tone: 'rose', note: d.model.why };
  if (st === 'no_answer' || st === 'error') return { tag: 'NO SLOT READING', tone: 'rose', note: d.model.why };
  if (st && st !== 'ok') return { tag: `${d.model.name.toUpperCase()} ${st.toUpperCase()}`, tone: 'muted', note: 'the model is not ready; no rate is read until it is' };
  const last = (xs: (number | null)[]) => {
    for (let i = xs.length - 1; i >= 0; i--) if (ok(xs[i])) return xs[i] as number;
    return null;
  };
  const dec = last(d.decode_tps) ?? 0;
  const pre = last(d.prompt_tps) ?? 0;
  if (pre > 0) return { tag: `PREFILL · ${Math.round(pre)} tok/s`, tone: 'cyan', note: null };
  if (dec > 0 || (d.model.busy ?? 0) > 0) return { tag: `GENERATING · ${Math.round(dec)} tok/s`, tone: 'moss', note: null };
  return { tag: 'IDLE', tone: 'muted', note: d.stats.generating_samples ? null : `no generation in the last ${Math.round(d.window_s / 60)} min` };
}

/** The top of a scale: a round number at or above the largest value. */
export function niceMax(xs: (number | null)[], floor: number): number {
  const m = Math.max(floor, ...xs.filter(ok));
  const p = 10 ** Math.floor(Math.log10(m));
  for (const k of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (k * p >= m) return k * p;
  return 10 * p;
}

/**
 * An SVG path for one series on a [0, w] x [0, h] box, time on x from
 * -window to 0. A null breaks the line: an unknown second is never drawn as
 * a value.
 */
export function linePath(t: number[], ys: (number | null)[], window: number, max: number, w: number, h: number): string {
  let d = '';
  let pen = false;
  for (let i = 0; i < t.length; i++) {
    const y = ys[i];
    if (!ok(y)) {
      pen = false;
      continue;
    }
    const x = (((t[i] ?? 0) + window) / window) * w;
    const yy = h - (Math.min(y, max) / max) * h;
    d += `${pen ? 'L' : 'M'}${x.toFixed(1)} ${yy.toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** Bars (x, width, height) for a rate series; nulls and zeros draw nothing. */
export function bars(t: number[], ys: (number | null)[], window: number, max: number, w: number, h: number, interval = 1) {
  const bw = Math.max(0.6, (interval / window) * w);
  const out: { x: number; y: number; w: number; h: number }[] = [];
  for (let i = 0; i < t.length; i++) {
    const y = ys[i];
    if (!ok(y) || y <= 0) continue;
    const bh = Math.max(1, (Math.min(y, max) / max) * h);
    out.push({ x: (((t[i] ?? 0) + window) / window) * w - bw, y: h - bh, w: bw, h: bh });
  }
  return out;
}

/** Spans where the model gave no rate (not loaded, not answering): shaded, not drawn as zero. */
export function gaps(t: number[], ys: (number | null)[], window: number, w: number, interval = 1) {
  const out: { x: number; w: number }[] = [];
  let start: number | null = null;
  const x = (i: number) => (((t[i] ?? 0) + window) / window) * w;
  for (let i = 0; i <= t.length; i++) {
    const missing = i < t.length && !ok(ys[i]);
    if (missing && start === null) start = i;
    if (!missing && start !== null) {
      const x0 = x(start) - (interval / window) * w;
      out.push({ x: Math.max(0, x0), w: x(i - 1) - Math.max(0, x0) });
      start = null;
    }
  }
  return out;
}

export const joules = (j: number | null | undefined): string => (!ok(j) ? '—' : j >= 10 ? `${j.toFixed(1)} J` : j >= 0.1 ? `${j.toFixed(2)} J` : `${(j * 1000).toFixed(0)} mJ`);

export const watts = (w: number | null | undefined): string => (ok(w) ? `${Math.round(w)} W` : '—');

/** "r 0.87 · n 587 · 10 min", or why there is none. */
export function rText(r: number | null | undefined, n: number | undefined, min: number | undefined, windowS: number): string {
  const win = `${Math.round(windowS / 60)} MIN`;
  if (!ok(r)) return (n ?? 0) < (min ?? 30) ? `n ${n ?? 0} < ${min ?? 30} · NO r` : 'NO VARIANCE · NO r';
  return `r ${r.toFixed(2)} · n ${n} · ${win}`;
}
