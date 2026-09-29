import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ServingPanel } from '../ui/HostPanels';
import { SlotsCard } from '../bonsai/scene/Telemetry';
import { deepRuns, mcpState, servedOnCard, type Deep } from './host';
import { flareRoots, itemsOf, type NebariSkills } from './nebari';
import type { Serving, Vitals } from './types';

const serving = (over: Partial<Serving> = {}): Serving => ({
  enabled: true,
  main: 'bonsai',
  max: 'flash-next',
  max_tier: 'max',
  on_card: 'bonsai',
  max_active: false,
  inflight: { bonsai: 1, 'flash-next': 0 },
  switching_to: null,
  last_max_end: null,
  idle_s: null,
  loaded: [
    { model: 'bonsai', state: 'ready', port: 10001, gguf: 'Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf' },
    { model: 'embed', state: 'ready', port: 10006, gguf: 'Qwen3-Embedding-0.6B-Q8_0.gguf' },
  ],
  error: null,
  ...over,
});

describe('servedOnCard (the header model chip, max mode aware)', () => {
  it('names the loaded main model and its gguf', () => {
    expect(servedOnCard(serving())).toEqual({ model: 'bonsai', gguf: 'Ternary-Bonsai-2-27B-PTQ1_0-mtp', max: false });
  });
  it('says so when the max model holds the card', () => {
    const s = serving({ on_card: 'flash-next', loaded: [{ model: 'flash-next', state: 'ready', port: 10009, gguf: 'Flash-Next-IQ2_XS-00001-of-00002.gguf' }] });
    expect(servedOnCard(s)).toEqual({ model: 'flash-next', gguf: 'Flash-Next-IQ2_XS-00001-of-00002', max: true });
  });
  it('is null for an older server or nothing on the card', () => {
    expect(servedOnCard(undefined)).toBeNull();
    expect(servedOnCard(serving({ on_card: null }))).toBeNull();
  });
});

describe('ServingPanel', () => {
  const html = (v: Partial<Vitals> | null) =>
    renderToStaticMarkup(createElement(ServingPanel, { v: v as Vitals | null, tiers: null, failure: null }));
  it('an older server: says the field is missing, invents nothing', () => {
    const out = html({ warnings: [] });
    expect(out).toContain('serving not reported by this server');
  });
  it('max mode configured: the max tier and the loaded models', () => {
    const out = html({ warnings: [], serving: serving() });
    expect(out).toContain('MAX MODE CONFIGURED');
    expect(out).toContain('flash-next');
    expect(out).toContain('tier max · every other tier bonsai');
    expect(out).toContain('10001');
    expect(out).toContain('SWAP BACK ON THE NEXT NON-MAX REQUEST');
  });
  it('max mode off: one model for every tier', () => {
    const out = html({ warnings: [], serving: serving({ enabled: false, max: null, inflight: {} }) });
    expect(out).toContain('ONE MODEL');
    expect(out).toContain('YAMADORI_MAX_MODEL unset');
  });
  it('llama-swap unread: says so', () => {
    expect(html({ warnings: [], serving: serving({ loaded: null, on_card: null }) })).toContain('llama-swap /running not read');
  });
});

describe('mcpState', () => {
  it('reads the host status, else says it never started, and a disabled row is disabled', () => {
    expect(mcpState({ id: 'a', enabled: true, status: { state: 'ready' } })).toBe('ready');
    expect(mcpState({ id: 'a', enabled: true })).toBe('not started');
    expect(mcpState({ id: 'a', enabled: false, status: { state: 'ready' } })).toBe('disabled');
  });
});

describe('deepRuns', () => {
  const d: Deep = {
    thresholds: { struggle_threshold: { value: 3, source: 'default' } },
    per_day: [
      { day: '2026-09-20', requests: 5, runs: { kickoff: 9 }, labels: {} },
      { day: '2026-09-27', requests: 10, runs: { kickoff: 1, struggle: 2 }, labels: {} },
      { day: '2026-09-28', requests: 20, runs: { auto: 3, kickoff: 1 }, labels: {} },
    ],
  };
  it('sums the last N days by trigger, largest first', () => {
    const r = deepRuns(d, 2);
    expect(r.requests).toBe(30);
    expect(r.runs).toBe(7);
    expect(r.byTrigger[0]).toEqual(['auto', 3]);
    expect(r.days.map((x) => x.day)).toEqual(['2026-09-27', '2026-09-28']);
  });
  it('is empty, not a crash, for a missing payload', () => {
    expect(deepRuns(null).runs).toBe(0);
  });
});

describe('flareRoots (NEBARI)', () => {
  const s: NebariSkills = {
    counts: { armed: 3 },
    total: 3,
    served: 3,
    served_by: { framework: { r3f: 2, react: 5 }, language: { typescript: 7 } },
    labels: { framework: { r3f: 'React Three Fiber', react: 'React' }, language: { typescript: 'TypeScript' } },
  };
  it('one heavy root per framework, one thin per language, by count, with taxonomy names', () => {
    expect(flareRoots(s)).toEqual([
      { label: 'React', n: 5, heavy: true },
      { label: 'React Three Fiber', n: 2, heavy: true },
      { label: 'TypeScript', n: 7, heavy: false },
    ]);
  });
  it('no skills payload: no roots', () => {
    expect(flareRoots(null)).toEqual([]);
  });
  it('items are chunks plus definitions', () => {
    expect(itemsOf({ package: 'x', version: '1', chunks: 3, defs: null, files: null, embedded: null, complete: null, published: null, unseen: null })).toBe(3);
  });
});

describe('SlotsCard (the slot layout)', () => {
  const slot = (id: number, over: object = {}) => ({ id, state: 'idle' as const, n_ctx: 262144, ctx: 0, prompt: 0, processed: 0, decoded: 0, remain: 0, tps: 0, pps: 0, ...over });
  it('marks the child slot and the primary conversation', () => {
    const out = renderToStaticMarkup(
      createElement(SlotsCard, {
        slots: { ok: true, ms: 1, model: 'bonsai', slots: [slot(0, { role: 'conversation', pinned: true, primary: true }), slot(1, { role: 'conversation' }), slot(2, { role: 'child' })] },
        lanes: null,
        context: null,
        since: 0,
      }),
    );
    expect(out).toContain('S0 · PRIMARY');
    expect(out).toContain('S2 · CHILD');
    expect(out).toContain('SLOTS · BONSAI');
  });
  it('names the reason, not a bare timeout, when the model is off the card', () => {
    const out = renderToStaticMarkup(
      createElement(SlotsCard, { slots: { ok: false, ms: 1, slots: [], off_card: true, error: 'bonsai is off the card: max mode (flash-next) holds it' }, lanes: null, context: null, since: 0 }),
    );
    expect(out).toContain('max mode (flash-next) holds it');
    expect(out).not.toContain('/slots not answering');
  });
});
