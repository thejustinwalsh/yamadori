import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { LoadoutCard } from '../bonsai/scene/LoadoutCard';
import { SlotsCard } from '../bonsai/scene/Telemetry';
import { holdTimes, kvLive, laneView, slotDetail, slotName, span, type LaneSlot, type LaneView } from './lanes';

const slot = (id: number, over: Partial<LaneSlot> = {}): LaneSlot => ({
  id,
  state: 'idle',
  n_ctx: 262144,
  ctx: 0,
  prompt: 0,
  processed: 0,
  decoded: 0,
  remain: 0,
  tps: 0,
  pps: 0,
  ...over,
});

const NOW = 1_800_000_000;

const view = (over: Partial<Extract<LaneView, { cards: unknown }>> = {}): LaneView => ({
  at: NOW,
  in_proxy: true,
  running_read: true,
  cards: [
    {
      key: 'main',
      index: 0,
      name: 'NVIDIA GeForce RTX 5060 Ti',
      uuid: 'u0',
      util: 51,
      used_mib: 15320,
      total_mib: 16311,
      models: [
        {
          model: 'flash-next',
          state: 'ready',
          port: 10008,
          gguf: 'x.gguf',
          card: 'main',
          main: true,
          inflight: 1,
          inflight_since: NOW - 12,
          last_use: null,
          slots: {
            ok: true,
            slots: [
              slot(0, {
                state: 'prefill',
                ctx: 8192,
                pps: 1396,
                role: 'conversation',
                pinned: true,
                primary: true,
                use: 'conversation (primary)',
                conv: { conv: 'c1a2b3c4', busy: true, idle_s: 0, held: true, rest_s: null, hold_s: 60 },
              }),
            ],
          },
        },
      ],
      expected: [],
    },
    {
      key: 'a4000',
      index: 1,
      name: 'NVIDIA RTX A4000',
      uuid: 'u1',
      models: [
        {
          model: 'bonsai-a4000',
          state: 'ready',
          port: 10002,
          gguf: 'b.gguf',
          card: 'a4000',
          main: false,
          inflight: 0,
          inflight_since: null,
          last_use: null,
          slots: {
            ok: true,
            slots: [
              slot(0, { n_ctx: 141312, use: 'second conversation (the other card)' }),
              slot(1, { n_ctx: 141312, ctx: 142, use: 'jjava, side calls' }),
            ],
          },
        },
      ],
      expected: [],
    },
  ],
  jobs: [
    { lane: 'gpu', card: 'main', label: 'main card (5060 Ti)', paused: { lane: 'gpu', by: 'coordinator', why: 'max soak', since: NOW - 3 * 86400, until: NOW + 4 * 86400 }, running: [], queued: 1, error: null },
    { lane: 'gpu_a4000', card: 'a4000', label: 'A4000', paused: null, running: [], queued: 0, error: null },
  ],
  holds: [
    { kind: 'job_lane_paused', card: 'main', what: "the worker's gpu job lane (main card (5060 Ti)) is paused: no new job starts there. This is a pause on the worker's queue, not a hold on the card", by: 'coordinator', why: 'max soak', since: NOW - 3 * 86400, until: NOW + 4 * 86400 },
  ],
  swap: { now: null, last: [{ at: NOW - 7200, from: ['bonsai'], to: 'flash-next', load_s: 63.2, ok: true, how: 'loaded', left_loaded: [] }] },
  ...over,
});

describe('lane view readings', () => {
  it('span is one coarse unit', () => {
    expect(span(12)).toBe('12 s');
    expect(span(300)).toBe('5 min');
    expect(span(3 * 86400)).toBe('3 d');
  });

  it('a hold says since when and until when, in the server clock', () => {
    expect(holdTimes({ since: NOW - 41, until: NOW + 300 }, NOW)).toBe('since 41 s ago · until in 5 min');
    expect(holdTimes({ since: null, until: null }, NOW)).toBe('');
  });

  it('laneView is null for an error or an older server', () => {
    expect(laneView({ error: 'x' })).toBeNull();
    expect(laneView(undefined)).toBeNull();
    expect(laneView(view())).not.toBeNull();
  });

  it('names what a slot is for', () => {
    expect(slotName({ use: 'jjava lane (the decider, side calls)', role: 'child' })).toBe('JJAVA LANE');
    expect(slotName({ use: 'jjava, side calls' })).toBe('JJAVA · SIDE CALLS');
    expect(slotName({ use: 'second conversation (the other card), c1a2b3c4' })).toBe('SECOND CONVERSATION');
    expect(slotName({ role: 'conversation', primary: true })).toBe('PRIMARY CONVERSATION');
    expect(slotName({ role: 'conversation' })).toBe('CONVERSATION');
  });

  it('an idle slot says its conversation, its idleness and what is left of its hold', () => {
    const d = slotDetail(slot(0, { ctx: 8192, role: 'conversation', primary: true, conv: { conv: 'c1a2b3c4', busy: false, idle_s: 12, held: true, rest_s: 48, hold_s: 60 } }));
    expect(d).toEqual(['PRIMARY CONVERSATION', 'CONV c1a2b3c4', 'IDLE 12 s', 'HOLD 48 S LEFT']);
    const over = slotDetail(slot(0, { ctx: 8192, role: 'conversation', conv: { conv: 'c1a2b3c4', busy: false, idle_s: 90, held: false, rest_s: 0, hold_s: 60 } }));
    expect(over).toContain('HOLD OVER (60 S) · FREE TO TAKE');
    expect(slotDetail(slot(1, { role: 'conversation' }))).toContain('EMPTY');
  });

  it('a working slot says its rate', () => {
    expect(slotDetail(slot(0, { state: 'decode', tps: 23.44, role: 'conversation' }))).toContain('23.4 TOK/S');
    expect(slotDetail(slot(0, { state: 'prefill', pps: 1396, role: 'conversation' }))).toContain('1396 PREFILL TOK/S');
  });

  it('the bargraph lines follow the model on the card, whichever it is', () => {
    const lines = kvLive(laneView(view()), true);
    expect(lines[0]?.text).toContain('flash-next');
    expect(lines[0]?.text).toContain('1 in flight');
    expect(lines[1]?.text).toContain('no lane on this card');
    expect(lines[2]?.text).toContain('bonsai-a4000');
    expect(kvLive(laneView(view({ cards: [] })), true)[0]?.text).toBe('no main model is loaded');
  });
});

describe('SlotsCard with the lane view', () => {
  const ctx = { pool: 262144, main: 262144, helper: 0, helpers: 0, reserve: 0, gib: 1, layout: 'cap' as const, model: 'flash-next', slots: 1 };
  const render = (v: LaneView | null) =>
    renderToStaticMarkup(createElement(SlotsCard, { slots: null, lanes: { main: 0, helper: 0, main_lanes: 2, helper_lanes: 1, admitted: 1, queued: 0, refused: 0, in_proxy: true }, context: ctx, since: 0, view: v }));

  it('labels the in-flight count as requests, and lists both cards slots', () => {
    const out = render(view());
    expect(out).toContain('REQUESTS 0/2');
    expect(out).not.toContain('MAIN 0/2');
    expect(out).toContain('SLOTS · FLASH-NEXT');
    expect(out).toContain('5060 TI · FLASH-NEXT');
    expect(out).toContain('A4000 · BONSAI-A4000');
    expect(out).toContain('SECOND CONVERSATION');
    expect(out).toContain('JJAVA · SIDE CALLS');
    expect(out).toContain('CONV c1a2b3c4');
  });

  it('never gives a label a fixed 28px column to overflow', () => {
    const out = renderToStaticMarkup(createElement(SlotsCard, { slots: null, lanes: null, context: ctx, since: 0, view: view() }));
    expect(out).not.toContain('28px');
  });
});

describe('LoadoutCard', () => {
  it('says a job lane is paused, by whom, why and until when, and that it is not the card', () => {
    const out = renderToStaticMarkup(createElement(LoadoutCard, { view: view() }));
    expect(out).toContain('JOB LANE PAUSED');
    expect(out).toContain('BY coordinator');
    expect(out).toContain('WHY: max soak');
    expect(out).toContain('not a hold on the card');
    expect(out).toContain('since 3 d ago');
  });

  it('shows the model on each card, the swap in progress and the last swap', () => {
    const v = view({ swap: { now: { to: 'flash-next', from: ['bonsai'], phase: 'loading', since: NOW - 41 }, last: [{ at: NOW - 7200, from: ['bonsai'], to: 'flash-next', load_s: 63.2, ok: true, how: 'loaded', left_loaded: [] }] } });
    const out = renderToStaticMarkup(createElement(LoadoutCard, { view: v }));
    expect(out).toContain('FLASH-NEXT');
    expect(out).toContain('BONSAI-A4000');
    expect(out).toContain('IN PROGRESS');
    expect(out).toContain('BONSAI → FLASH-NEXT');
    expect(out).toContain('41 s SO FAR');
    expect(out).toContain('63.2 S');
  });

  it('is honest when the server predates the view', () => {
    expect(renderToStaticMarkup(createElement(LoadoutCard, { view: undefined }))).toContain('not reported by this server');
    expect(renderToStaticMarkup(createElement(LoadoutCard, { view: { error: 'boom' } }))).toContain('the lane view failed: boom');
  });
});
