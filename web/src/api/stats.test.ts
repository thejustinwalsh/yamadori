// The JJAVA and SOKUDO pages' pure helpers and static renders (vitest runs in
// node: react-dom/server markup), on payloads shaped exactly as
// mcp/dash_jjava.py and mcp/dash_perf.py return them.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { href, legacyTarget, resolve } from '../router';
import { dataFor } from '../routes';
import { Histogram, LineChart, StackedBars } from '../ui/Charts';
import { TierPanel } from '../screens/Performance';
import { InjectorPanel } from '../ui/StatsParts';
import {
  bucketLabel,
  distText,
  injectorTotals,
  isTraffic,
  isWindow,
  jjavaPath,
  part,
  perfPath,
  ranked,
  spreadText,
  statusTone,
  tierLabel,
  topSeries,
  total,
  type Injector,
  type Jjava,
  type Perf,
} from './stats';

const parser = (p: string) => {
  const keys: string[] = [];
  const src = p.replace(/:(\w+)/g, (_, k: string) => {
    keys.push(k);
    return '([^/]+)';
  });
  return { pattern: new RegExp(`^${src}/?$`, 'i'), keys };
};

describe('routes', () => {
  it('the retired benchmark page (/results) lands on PERFORMANCE; JJAVA has its own', () => {
    expect(resolve(parser, '/results')).toEqual({ name: 'perf' });
    expect(resolve(parser, '/performance')).toEqual({ name: 'perf' });
    expect(resolve(parser, '/jjava')).toEqual({ name: 'jjava' });
    expect(href.perf).toBe('/performance');
    expect(href.jjava).toBe('/jjava');
  });
  it('the retired NEBARI screen (/nebari) lands on the Skills library, and the URL is replaced', () => {
    expect(resolve(parser, '/nebari')).toEqual({ name: 'skills', view: 'library' });
    expect(legacyTarget('/nebari')).toBe('/skills');
    expect(legacyTarget('/results/')).toBe('/performance');
    expect(legacyTarget('/skills')).toBeNull();
    expect(dataFor({ name: 'skills', view: 'library' })).toContain('/dash/api/skill-factory/library');
  });
  it('a traffic class other than client names its path, window first', () => {
    expect(perfPath('24h', 'all')).toBe('/dash/api/perf/24h/all');
    expect(perfPath('7d', 'test')).toBe('/dash/api/perf/7d/test');
    expect(perfPath('7d', 'client')).toBe('/dash/api/perf/7d');
    expect(isTraffic('all')).toBe(true);
    expect(isTraffic('both')).toBe(false);
  });
  it('each page warms its default window; the Skills selections view warms jjava', () => {
    expect(dataFor({ name: 'perf' })).toEqual(['/dash/api/perf']);
    expect(dataFor({ name: 'jjava' })).toEqual(['/dash/api/jjava']);
    expect(dataFor({ name: 'skills', view: 'selections' })).toContain('/dash/api/jjava');
  });
  it('a window names its path; the default is the bare route', () => {
    expect(jjavaPath('24h')).toBe('/dash/api/jjava');
    expect(jjavaPath('7d')).toBe('/dash/api/jjava/7d');
    expect(perfPath('1h')).toBe('/dash/api/perf/1h');
    expect(perfPath('24h', 'client')).toBe('/dash/api/perf');
    expect(isWindow('30d')).toBe(true);
    expect(isWindow('2d')).toBe(false);
  });
});

describe('helpers', () => {
  it('topSeries keeps the largest in a fixed order and folds the rest into other', () => {
    const got = topSeries({ a: [1, 1], b: [5, 0], c: [0, 3], d: [1, 0], e: [0, 0] });
    expect(got.map((x) => x.name)).toEqual(['b', 'c', 'a', 'other (2)']);
    expect(got[3]?.values).toEqual([1, 0]);
    expect(topSeries({ x: [1], y: [2] }).map((x) => x.name)).toEqual(['y', 'x']);
  });
  it('a section the server failed is null; the panel prints its error', () => {
    expect(part({ error: 'boom' })).toBeNull();
    expect(part({ n: 1 })).toEqual({ n: 1 });
    expect(part(undefined)).toBeNull();
  });
  it('spreads read as p50 / p90 / n, and nothing measured says so', () => {
    expect(spreadText({ n: 3, p50: 420, p90: 612.4, max: 700 })).toBe('p50 420 ms · p90 612 ms · n 3');
    expect(spreadText({ n: 0, p50: null, p90: null, max: null })).toBe('no measurement');
    expect(spreadText(null)).toBe('no measurement');
  });
  it('labels buckets by their size', () => {
    const ts = new Date(2026, 8, 30, 14, 5).getTime() / 1000;
    expect(bucketLabel(ts, 300)).toBe('14:05');
    expect(bucketLabel(ts, 86400)).toBe('WED 30');
  });
  it('status tones: 2xx moss, 4xx rose, 5xx crimson', () => {
    expect(statusTone('200')).toBe('moss');
    expect(statusTone(401)).toBe('rose');
    expect(statusTone(529)).toBe('crimson');
    expect(statusTone('?')).toBe('muted');
  });
  it('counts and ranks', () => {
    expect(total([1, null, 2])).toBe(3);
    expect(ranked({ a: 1, b: 3, c: 3 })).toEqual([['b', 3], ['c', 3], ['a', 1]]);
  });
});

describe('charts', () => {
  const t = [0, 3600, 7200];
  it('a line breaks on an unmeasured bucket and every bucket has a hover title', () => {
    const html = renderToStaticMarkup(
      createElement(LineChart, { t, bucketS: 3600, unit: 'tok/s', label: 'decode', series: [{ name: 'bonsai', values: [50, null, 60], slot: 0 }, { name: 'flash-next', values: [10, 12, 14], slot: 2 }] }),
    );
    expect((html.match(/<title>/g) ?? []).length).toBe(3);
    expect(html).toContain('bonsai —');
    expect(html).toContain('flash-next');
    expect(html).toContain('stroke-dasharray="2 3"'); // secondary encoding per slot
    expect(html).not.toContain('NO MEASUREMENT');
  });
  it('an empty line chart says so rather than drawing zero', () => {
    const html = renderToStaticMarkup(createElement(LineChart, { t, bucketS: 3600, unit: 'ms', label: 'p50', series: [{ name: 'p50', values: [null, null, null], slot: 1 }] }));
    expect(html).toContain('NO MEASUREMENT IN THIS WINDOW');
  });
  it('stacked bars total each series in the legend', () => {
    const html = renderToStaticMarkup(createElement(StackedBars, { t, bucketS: 3600, label: 'use', series: [{ name: 'phase', values: [1, 2, 0], slot: 0 }, { name: 'choose', values: [0, 1, 1], slot: 1 }] }));
    expect(html).toContain('phase · 3');
    expect(html).toContain('choose · 2');
  });
  it('a histogram carries its counts', () => {
    const html = renderToStaticMarkup(createElement(Histogram, { bins: [0, 0, 1, 0, 0, 0, 0, 0, 0, 4], label: 'confidence' }));
    expect(html).toContain('n 5');
    expect(html).toContain('0.9–1.0 · 4');
  });
});

const inj: Injector = {
  models: [
    {
      model: 'bonsai', runs: 3, injected: 1, skipped: 2, stage1_items_mean: 5, stage1_skills_mean: 2, stage2_asked_mean: 5, stage2_passed_mean: 1,
      stage3: { all: 1, none: 1 }, chosen_mean: 0.33, failures: { MODEL_UNREACHABLE: 1 }, why: { 'the gate chose nothing': 1 },
      need_hist: [1, 0, 0, 0, 0, 0, 0, 0, 1, 2], noul_hist: null, item_tiers: { untuned: 12 }, stage3_tiers: { untuned: 2 },
      ms: { n: 3, p50: 800, p90: 1200, max: 1300 },
    },
  ],
  question_sets: [],
  requests: 40,
  skills_on: 3,
  decider_turns: 3,
  decider_failures: {},
  injector: { version: 'inject/1', questions: 'inject-q/1', max_skills: 3, thresholds: 'untuned' },
  skills_by_tier: { minimal: false, low: false, medium: false, high: false, xhigh: false, max: false },
  note: 'the injector runs only where skills run',
};

describe('InjectorPanel (the Skills page and JJAVA)', () => {
  it('totals the runs across models, and names what jjava could not answer', () => {
    expect(injectorTotals(inj)).toEqual({ runs: 3, injected: 1, skipped: 2, failed: 1 });
    expect(injectorTotals(null)).toEqual({ runs: 0, injected: 0, skipped: 0, failed: 0 });
  });
  it('renders per model, with skills off said plainly', () => {
    const data = { injector: inj } as unknown as Jjava;
    const html = renderToStaticMarkup(createElement(InjectorPanel, { data }));
    expect(html).toContain('JJAVA · SKILLS INJECTOR');
    expect(html).toContain('3 RUNS · 1 INJECTED');
    expect(html).toContain('SKILLS ARE OFF AT EVERY TIER');
    expect(html).toContain('MODEL_UNREACHABLE');
    expect(html).toContain('stage 2 · need, P(top level)');
  });
  it("a section the server failed shows the server's error", () => {
    const html = renderToStaticMarkup(createElement(InjectorPanel, { data: { injector: { error: 'injector: KeyError' } } as unknown as Jjava }));
    expect(html).toContain('injector: KeyError');
  });
});

describe('tok/s by effort tier', () => {
  const dist = (n: number, p50: number | null, mean: number | null, p90: number | null) => ({ n, p50, mean, p90 });
  const none = dist(0, null, null, null);
  const perf = {
    at: 1,
    window: { name: '7d', since: 0, until: 1, bucket_s: 21600, buckets: [], names: [] },
    ctx_bins: [],
    left_out: [],
    sources: { stats: { enabled: true, queued: 0, written: 0, dropped: 0, errors: 0, last_error: null }, generations: 3 },
    models: [],
    gpus: { bucket_s: 60, cards: [], rows: 0 },
    swaps: { rows: [], load_s: {}, last_in_memory: null, table: {} },
    gates: { files: [], errors: [], globs: [] },
    by_tier: {
      traffic: 'client',
      traffic_names: ['client', 'test', 'all'],
      window: '7d',
      ctx_buckets: ['0-8K', '8-32K', '32-64K', '64K+'],
      prefill_min_processed: 2048,
      generations: { in_window: 5, client: 3, test: 1, unrecorded: 1 },
      left_out: { traffic: 1, role: 0 },
      rows: [
        {
          tier: 'max', model: 'flash-next', role: 'main', n: 3, decode: dist(3, 28.8, 29.1, 31.4), prefill_warm: dist(1, 410.5, 410.5, 410.5), prefill_cold: dist(1, 95.2, 95.2, 95.2),
          by_ctx: [{ bucket: '8-32K', n: 3, decode: dist(3, 28.8, 29.1, 31.4), prefill_warm: dist(1, 410.5, 410.5, 410.5), prefill_cold: none }],
        },
        { tier: null, model: 'bonsai', role: 'main', n: 2, decode: dist(2, 60, 60, 60), prefill_warm: none, prefill_cold: none },
      ],
      by_model: [],
      rules: 'decode: llama-server predicted_n / predicted_ms',
    },
  } as unknown as Perf;
  it('reads rates as p50 / mean / p90 and a tier recorded before the column as unrecorded', () => {
    expect(distText(dist(3, 28.8, 29.1, 31.4))).toBe('28.8 · 29.1 · 31.4');
    expect(distText(none)).toBe('—');
    expect(distText(null)).toBe('—');
    expect(tierLabel('max')).toBe('max');
    expect(tierLabel(null)).toBe('unrecorded');
  });
  it('renders the table: tier, model, decode, prefill warm and cold, n, and the context breakdown', () => {
    const html = renderToStaticMarkup(createElement(TierPanel, { data: perf, traffic: 'client', setTraffic: () => undefined }));
    expect(html).toContain('TOK/S BY EFFORT TIER');
    expect(html).toContain('flash-next');
    expect(html).toContain('28.8 · 29.1 · 31.4');
    expect(html).toContain('411 (n 1)');
    expect(html).toContain('95.2 (n 1)');
    expect(html).toContain('unrecorded');
    expect(html).toContain('8-32K');
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain('3 CLIENT · 1 TEST · 1 UNRECORDED');
  });
  it('a server that predates the table says so; no row says why', () => {
    const old = { ...perf, by_tier: undefined } as unknown as Perf;
    expect(renderToStaticMarkup(createElement(TierPanel, { data: old, traffic: 'client', setTraffic: () => undefined }))).toContain('predates the by-tier table');
    const empty = { ...perf, by_tier: { ...(perf.by_tier as object), rows: [] } } as unknown as Perf;
    expect(renderToStaticMarkup(createElement(TierPanel, { data: empty, traffic: 'client', setTraffic: () => undefined }))).toContain('no client generation with a tier');
  });
});
