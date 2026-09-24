// Static renders of KŌGŌSEI · WATTS PER TOKEN (vitest runs in node) and the
// pure helpers behind it. The payload is shaped as mcp/power.py series_of()
// returns it.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { bars, cardName, gaps, isSeries, joules, linePath, modelState, niceMax, rText, type Series } from '../api/powerSeries';
import { KogoseiPanel } from './KogoseiPanel';

const MAIN = { uuid: 'GPU-de66', index: 0, name: 'NVIDIA GeForce RTX 5060 Ti', main: true };
const A4000 = { uuid: 'GPU-43e3', index: 1, name: 'NVIDIA RTX A4000', main: false };

/** 60 idle s, 30 generating s at 30 tok/s, 10 prefill s at 1000 tok/s. */
function series(over: Partial<Series> = {}): Series {
  const t: number[] = [];
  const w0: (number | null)[] = [];
  const w1: (number | null)[] = [];
  const dec: (number | null)[] = [];
  const pre: (number | null)[] = [];
  const busy: (number | null)[] = [];
  for (let i = 0; i < 100; i++) {
    t.push(i - 99);
    const phase = i < 60 ? 'idle' : i < 90 ? 'gen' : 'pre';
    w0.push(phase === 'idle' ? 20 : phase === 'gen' ? 130 : 150);
    w1.push(6);
    dec.push(phase === 'gen' ? 30 : 0);
    pre.push(phase === 'pre' ? 1000 : 0);
    busy.push(phase === 'idle' ? 0 : 1);
  }
  return {
    running: true,
    window_s: 600,
    interval_s: 1,
    now: 1000,
    model: { name: 'bonsai', state: 'ok', why: null, busy: 1 },
    gpus: [MAIN, A4000],
    main_index: 0,
    t,
    watts: [w0, w1],
    decode_tps: dec,
    prompt_tps: pre,
    busy,
    idle: { watts: [20, 6], n: 60, at: 1000, from_window: true },
    stats: {
      samples: 100,
      ok_samples: 100,
      generating_samples: 40,
      window_s: 600,
      j_per_token: 0.5,
      j_n: 40,
      tokens: 10900,
      marginal_j_per_token: 0.422,
      j_per_gen_token: 4.333,
      gen_only_samples: 30,
      j_per_prompt_token: 0.15,
      prompt_only_samples: 10,
      r_decode: 0.61,
      r_prompt: 0.55,
      r_n: 100,
      r_min: 30,
    },
    source: { watts: 'nvidia-smi', tokens: '/slots', undercount: 'floor' },
    ...over,
  };
}

const render = (d: Series | null, failure: Parameters<typeof KogoseiPanel>[0]['failure'] = null, raw?: unknown) =>
  renderToStaticMarkup(createElement(KogoseiPanel, { d, failure, raw }));

describe('KogoseiPanel', () => {
  it('draws watts per card over generation and prompt rates, and names its state', () => {
    const html = render(series());
    expect(html).toContain('KŌGŌSEI · WATTS PER TOKEN');
    expect(html).toContain('光');
    expect(html).toContain('PREFILL · 1000 tok/s'); // the newest second was prefill
    expect(html).toContain('5060 Ti 150 W');
    expect(html).toContain('A4000 6 W');
    expect(html).toContain('GENERATION tok/s');
    expect(html).toContain('PROMPT tok/s');
    expect(html.match(/<path /g)?.length).toBe(2); // one line per card
    expect(html).toContain('DASHED: IDLE 5060 Ti');
  });
  it('shows energy per token, over idle, per phase, the idle baseline and r with n and window', () => {
    const html = render(series());
    expect(html).toContain('0.50 J'); // J / token
    expect(html).toContain('0.42 J'); // over idle
    expect(html).toContain('4.33 J · 0.15 J'); // per generated · per prompt token
    expect(html).toContain('5060 Ti 20 W · A4000 6 W'); // idle
    expect(html).toContain('r 0.61 · n 100 · 10 MIN');
    expect(html).toContain('ASSOCIATION, NOT CAUSE');
    expect(html).toContain('5060 TI TOKENS ONLY · RATES ARE A FLOOR');
  });
  it('idle: says so, and no energy per token is invented', () => {
    const z = (xs: (number | null)[]) => xs.map(() => 0);
    const d = series();
    const html = render(
      series({
        decode_tps: z(d.decode_tps),
        prompt_tps: z(d.prompt_tps),
        model: { name: 'bonsai', state: 'ok', why: null, busy: 0 },
        stats: { ...d.stats, generating_samples: 0, j_per_token: null, marginal_j_per_token: null, j_per_gen_token: null, j_per_prompt_token: null, r_decode: null, r_n: 100 },
      }),
    );
    expect(html).toContain('>IDLE<');
    expect(html).toContain('NO GENERATION IN THE LAST 10 MIN');
    expect(html).toContain('>— · —<');
    expect(html).toContain('NO VARIANCE · NO r');
  });
  it('not loaded: named, with no rates drawn and the gap shaded', () => {
    const d = series();
    const nul = (xs: (number | null)[]) => xs.map(() => null);
    const html = render(series({ model: { name: 'bonsai', state: 'not_loaded', why: null, busy: null }, decode_tps: nul(d.decode_tps), prompt_tps: nul(d.prompt_tps), busy: nul(d.busy) }));
    expect(html).toContain('BONSAI NOT LOADED');
    expect(html).toContain('NOTHING IS LOADED TO FIND OUT');
    expect(html).not.toContain('GENERATING ·');
    expect(html).not.toContain('PREFILL ·');
    expect(html).toContain('SHADED: NO RATE');
  });
  it('an older server (404) asks for a restart; no data yet is a loading state', () => {
    expect(render(null, { kind: 'http', status: 404, message: 'not found' })).toContain('predates the watts-vs-tokens ring');
    expect(render(null)).toContain('reading /dash/api/power/series');
    expect(render(null, null, { nope: 1 })).toContain('not a series payload');
  });
});

describe('series helpers', () => {
  it('isSeries needs t, watts and stats', () => {
    expect(isSeries(series())).toBe(true);
    expect(isSeries({ t: [] })).toBe(false);
  });
  it('card names drop the vendor prefix', () => {
    expect(cardName(MAIN)).toBe('5060 Ti');
    expect(cardName(A4000)).toBe('A4000');
  });
  it('a null breaks the line instead of drawing zero', () => {
    expect(linePath([-2, -1, 0], [10, null, 20], 2, 20, 100, 10)).toBe('M0.0 5.0M100.0 0.0');
  });
  it('bars skip nulls and zeros; gaps cover the nulls', () => {
    expect(bars([-2, -1, 0], [null, 0, 5], 2, 10, 100, 10)).toEqual([{ x: 50, y: 5, w: 50, h: 5 }]);
    expect(gaps([-3, -2, -1, 0], [1, null, null, 1], 3, 300)).toEqual([{ x: 0, w: 200 }]);
  });
  it('scales round up', () => {
    expect(niceMax([137], 50)).toBe(150);
    expect(niceMax([null, 3], 10)).toBe(10);
    expect(niceMax([1234], 100)).toBe(1500);
  });
  it('formats joules and r', () => {
    expect(joules(4.333)).toBe('4.33 J');
    expect(joules(0.05)).toBe('50 mJ');
    expect(joules(12.34)).toBe('12.3 J');
    expect(joules(null)).toBe('—');
    expect(rText(null, 12, 30, 600)).toBe('n 12 < 30 · NO r');
  });
  it('model states', () => {
    expect(modelState(series({ model: { name: 'bonsai', state: 'swap_down', why: 'refused', busy: null } })).tag).toBe('LLAMA-SWAP DOWN');
    expect(modelState(series({ model: { name: 'bonsai', state: 'starting', why: null, busy: null } })).tag).toBe('BONSAI STARTING');
    const d = series();
    expect(modelState(series({ prompt_tps: d.prompt_tps.map(() => 0) })).tag).toBe('GENERATING · 0 tok/s'); // busy, last second decoded nothing
  });
});
