// @vitest-environment happy-dom
//
// Error boundaries do not run during server rendering, so these mount into a
// DOM (happy-dom) with react-dom/client.
//
// The failure being gated: a /dash/api/tiers payload without `default` made
// the effort ladder throw `t.default.toUpperCase()` during render, uncaught,
// and the whole dashboard went white.
import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import type { Tiers, Vitals } from '../api/types';
import { KvPanel, TiersPanel } from '../screens/Cockpit';
import { ErrorBoundary, Guarded, need, PayloadError } from './ErrorBoundary';
import { StrataPanel } from './Strata';

beforeAll(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
});

let host: HTMLDivElement | null = null;
afterEach(() => {
  host?.remove();
  host = null;
  vi.restoreAllMocks();
});

function mount(el: ReactElement): string {
  host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(el));
  return host.textContent ?? '';
}

const TIER = { thinks: true, floor: 1500, effort: 'high', retrieval: true, hints: true, fanout: 3, investigate: false, why: 'w' };

describe('the effort ladder that blanked the app', () => {
  it('renders a tiers payload with no `default` instead of throwing', () => {
    const t = { order: ['high'], tiers: { high: TIER }, ceiling: 'max' } as unknown as Tiers;
    const out = mount(createElement(TiersPanel, { t, failure: null }));
    expect(out).toContain('DEFAULT —');
    expect(out).toContain('high');
  });

  it('renders a payload with no order or tiers as a one-line empty state', () => {
    const out = mount(createElement(TiersPanel, { t: {} as Tiers, failure: null }));
    expect(out).toContain('no tier order');
  });

  // Operator, 2026-09-23: the ladder shows tier names, not the effort string
  // mapped under the hood. Only thinking OFF gets a chip.
  it('never shows the effort mapped under the hood', () => {
    const t = { order: ['xhigh'], tiers: { xhigh: { ...TIER, effort: 'medium', sent_effort: 'medium' } }, default: 'medium' } as unknown as Tiers;
    const out = mount(createElement(TiersPanel, { t, failure: null }));
    expect(out).toContain('xhigh');
    expect(out).not.toMatch(/EFFORT (MEDIUM|XHIGH|HIGH)/);
    expect(out).not.toContain('SENT');
  });

  it('marks a tier whose thinking is off', () => {
    const t = { order: ['minimal'], tiers: { minimal: { ...TIER, thinks: false } }, default: 'medium' } as unknown as Tiers;
    const out = mount(createElement(TiersPanel, { t, failure: null }));
    expect(out).toContain('THINKING OFF');
  });
});

describe('one bad panel fails alone', () => {
  function Broken(): ReactElement {
    need({} as { default?: string }, 'default', '/dash/api/tiers');
    return createElement('p', null, 'unreachable');
  }

  it('shows that panel\'s own error, naming the payload and key, while its sibling renders', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const out = mount(
      createElement(
        'div',
        null,
        createElement(ErrorBoundary, { what: 'DAN · EFFORT LADDER', source: '/dash/api/tiers', children: createElement(Broken) }),
        createElement(ErrorBoundary, { what: 'SILICON', source: '/dash/api/vitals', children: createElement('p', null, 'sibling still here') }),
      ),
    );
    expect(out).toContain('RENDER FAILED');
    expect(out).toContain('DAN · EFFORT LADDER');
    expect(out).toContain('/dash/api/tiers');
    expect(out).toContain('key default');
    expect(out).toContain('sibling still here');
  });

  it('catches an unexpected TypeError too, and names the source it was reading', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const Bad = (): ReactElement => {
      const x = undefined as unknown as { y: string };
      return createElement('p', null, x.y.toUpperCase());
    };
    const out = mount(createElement(ErrorBoundary, { what: 'PROCESSES', source: '/dash/api/vitals', children: createElement(Bad) }));
    expect(out).toContain('PROCESSES · render failed');
    expect(out).toContain('/dash/api/vitals');
    expect(out).toContain('TypeError');
  });

  it('Guarded catches a payload access written in the PAGE, not just in a child component', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const x = {} as { missing?: { what: string }[] };
    const out = mount(
      createElement(
        'div',
        null,
        createElement(Guarded, {
          what: 'MISSING',
          source: '/dash/api/datasets/abc',
          render: () => createElement('ul', null, x.missing!.map((m) => createElement('li', { key: m.what }, m.what))),
        }),
        createElement(Guarded, { what: 'WARNINGS', render: () => createElement('p', null, 'other panel fine') }),
      ),
    );
    expect(out).toContain('MISSING · render failed');
    expect(out).toContain('/dash/api/datasets/abc');
    expect(out).toContain('other panel fine');
  });

  it('need() names source and key', () => {
    expect(() => need({ a: undefined } as { a?: number }, 'a', '/x')).toThrow(PayloadError);
    expect(() => need({ a: 'x' } as { a: unknown }, 'a', '/x', (v) => typeof v === 'number')).toThrow(/\/x has no usable `a` \(got string\)/);
  });
});

describe('KV pool split', () => {
  const vit = (context: Vitals['context']) => ({ context }) as unknown as Vitals;

  it('draws main plus one segment per deep-thinking context, from the API fields', () => {
    const out = mount(
      createElement(KvPanel, {
        v: vit({ pool: 147456, main: 73728, helper: 36864, helpers: 2, reserve: 0, gib: 6.19 }),
        failure: null,
      }),
    );
    expect(out).toContain('DEEP THINKING ×2');
    expect(out).toContain('73,728'); // main, and 2 × 36,864
    expect(out).not.toContain('(DERIVED)');
    const bar = host!.querySelector('[role="img"]')!;
    expect(bar.getAttribute('aria-label')).toContain('2 deep thinking contexts of 36,864');
    expect(bar.children.length).toBe(3); // main + 2 helpers, no reserve segment at 0
  });

  it('derives the count from a server that predates `helpers`, and says so', () => {
    const out = mount(createElement(KvPanel, { v: vit({ pool: 147456, main: 88473, helper: 36864, reserve: 22119, gib: 6.19 }), failure: null }));
    expect(out).toContain('DEEP THINKING ×1');
    expect(out).toContain('(DERIVED)');
    expect(host!.querySelector('[role="img"]')!.children.length).toBe(3); // main + 1 helper + reserve
  });
});

describe('tier strata', () => {
  const S = { taproot: 30, branch: 12, shoot: 8, searches: 20, window_seconds: 86400 };

  it('one line when the server does not report it', () => {
    expect(mount(createElement(StrataPanel, { strata: undefined, present: false }))).toContain('not reported by this server');
  });

  it('one line when there were no searches', () => {
    const out = mount(createElement(StrataPanel, { strata: { ...S, taproot: 0, branch: 0, shoot: 0, searches: 0 }, present: true }));
    expect(out).toContain('no searches in the last 24 h');
    expect(out).not.toContain('TAPROOT');
  });

  it('draws the three layers with counts and shares', () => {
    const out = mount(createElement(StrataPanel, { strata: S, present: true }));
    expect(out).toContain('20 SEARCHES · 24 h');
    expect(out).toContain('30 · 60.0%');
    expect(out).toContain('12 · 24.0%');
    expect(out).toContain('8 · 16.0%');
  });

  it('a malformed count fails the panel with the key named', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const out = mount(
      createElement(ErrorBoundary, {
        what: 'TIER STRATA',
        children: createElement(StrataPanel, { strata: { ...S, taproot: 'many' } as unknown as typeof S, present: true }),
      }),
    );
    expect(out).toContain('key taproot');
  });
});
