import { describe, expect, it } from 'vitest';
import type { Pulse, Slot } from '../api/types';
import { IDLE, LiveBus, liveTargets, marksOf, newEvents, WIND_FLOOR, windOffset } from './live';

const slot = (id: number, state: Slot['state'], tps = 0): Slot => ({
  id, state, n_ctx: 163840, ctx: 1000, prompt: 1000, processed: 900, decoded: 100, remain: 10, tps, pps: 0,
});

const pulse = (slots: Slot[], extra: Partial<Pulse> = {}): Pulse => ({
  at: 0,
  gpus: [{ index: 0, name: 'g', used_mib: 1, total_mib: 2, free_mib: 1, pct: 50, tight: false, util: 90 }],
  slots: { ok: true, slots, ms: 1 },
  lanes: null,
  tools: null,
  queue: null,
  seed: null,
  strata: null,
  context: { error: 'n/a' },
  ...extra,
});

describe('liveTargets', () => {
  it('is the idle breath with no signal at all', () => {
    expect(liveTargets(null, null)).toEqual(IDLE);
  });

  it('is calm when every slot is idle, even if the GPU is busy with something else', () => {
    const t = liveTargets(pulse([slot(0, 'idle'), slot(1, 'idle')]), null);
    expect(t.busy).toBe(0);
    expect(t.activity).toBe(0);
    expect(t.gust).toBe(0);
    expect(t.wind).toBe(IDLE.wind);
  });

  it('rises with decode rate and with the number of busy slots', () => {
    const one = liveTargets(pulse([slot(0, 'decode', 42), slot(1, 'idle')]), null);
    const three = liveTargets(pulse([slot(0, 'decode', 30), slot(1, 'decode', 25), slot(2, 'prefill')]), null);
    expect(one.activity).toBeCloseTo(0.7, 5);
    expect(three.activity).toBeGreaterThan(0.9);
    expect(three.busy).toBe(3);
    expect(three.wind).toBeGreaterThan(one.wind);
    expect(three.gust).toBeGreaterThan(one.gust);
    expect(one.glow).toBeGreaterThan(IDLE.glow);
  });

  it('falls back to GPU0 utilisation when the slots are not readable', () => {
    const p = pulse([]);
    p.slots = { ok: false, slots: [], ms: 1000, error: 'down' };
    const t = liveTargets(p, null);
    expect(t.activity).toBeCloseTo(0.9, 5);
    expect(t.measured).toBe(true);
  });
});

describe('events', () => {
  const tools = (turn: number, tool: number) => ({
    last: { id: tool, name: 'find_by_meaning', at: 1 }, running: null, recent: [], calls_window: 0,
    turns_window: 0, answers_window: 0, window_seconds: 600, last_turn: { id: turn, at: 1, open: false },
  });

  it('the first poll is a baseline, never a burst', () => {
    expect(newEvents(null, marksOf(pulse([slot(0, 'decode')], { tools: tools(5, 6) })))).toEqual([]);
  });

  it('a new turn, a new tool call and a new seed each fire once', () => {
    const a = marksOf(pulse([slot(0, 'idle')], { tools: tools(5, 6) }));
    const b = marksOf(
      pulse([slot(0, 'idle')], {
        tools: tools(9, 10),
        seed: { word: 'harbor', u32: 1, hex: '0x1', token_id: null, where: 'fanout', at: 77 },
      }),
    );
    expect(newEvents(a, b)).toEqual(['request', 'tool', 'seed']);
    expect(newEvents(b, b)).toEqual([]);
  });

  it('a slot waking from idle is a request even when no turn was logged', () => {
    const a = marksOf(pulse([slot(0, 'idle'), slot(1, 'idle')]));
    const b = marksOf(pulse([slot(0, 'prefill'), slot(1, 'idle')]));
    expect(newEvents(a, b)).toEqual(['request']);
  });
});

describe('wind', () => {
  const w = { amp: 1, phase: 1.3, gust: 1, gphase: 1.2 };
  const at = (x: number, y: number, z: number, s = w) => {
    const o = [0, 0, 0];
    windOffset(x, y, z, s, o);
    return Math.hypot(o[0]!, o[1]!, o[2]!);
  };

  it('never moves the nebari: nothing at or below the floor sways', () => {
    for (const y of [-0.03, 0, 0.1, WIND_FLOOR]) expect(at(0.7, y, 0.2)).toBe(0);
  });

  it('grows with height and with amplitude, and is still at zero amplitude', () => {
    expect(at(0.3, 2.5, 0.1)).toBeGreaterThan(at(0.3, 1.2, 0.1));
    expect(at(0.3, 2.5, 0.1, { ...w, amp: 0 })).toBe(0);
    expect(at(0.3, 2.5, 0.1, { ...w, amp: 0.3 })).toBeLessThan(at(0.3, 2.5, 0.1));
  });

  it('is continuous: neighbouring points move together, so pads stay on branches', () => {
    expect(Math.abs(at(0.5, 2, 0.5) - at(0.51, 2.01, 0.5))).toBeLessThan(2e-3);
  });
});

describe('LiveBus', () => {
  it('eases toward the targets between polls instead of jumping', () => {
    const bus = new LiveBus();
    bus.push(pulse([slot(0, 'decode', 60)]), null);
    bus.tick(1 / 60, false);
    expect(bus.activity).toBeGreaterThan(0);
    expect(bus.activity).toBeLessThan(0.1);
    for (let i = 0; i < 600; i++) bus.tick(1 / 60, false);
    expect(bus.activity).toBeCloseTo(1, 2);
    expect(bus.lively()).toBe(true);
  });

  it('reduced motion stills the wind but keeps the tree and its glow', () => {
    const bus = new LiveBus();
    bus.push(pulse([slot(0, 'decode', 60), slot(1, 'decode', 60)]), null);
    for (let i = 0; i < 600; i++) bus.tick(1 / 60, true);
    expect(bus.windState(true).amp).toBe(0);
    expect(bus.glowNow(true)).toBeGreaterThan(0.9);
  });

  it('records when each event fired, on the bus clock', () => {
    const bus = new LiveBus();
    bus.push(pulse([slot(0, 'idle')]), null);
    for (let i = 0; i < 10; i++) bus.tick(0.1, false);
    bus.push(pulse([slot(0, 'prefill')]), null);
    expect(bus.fired.request).toBeCloseTo(1, 5);
    expect(bus.animating()).toBe(true);
  });
});
