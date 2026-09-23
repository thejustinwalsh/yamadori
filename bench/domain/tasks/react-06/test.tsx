import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { renderToString } from 'react-dom/server';
import { hydrateRoot } from 'react-dom/client';
import { useMediaQuery } from './solution';
import { installMatchMedia } from './helpers';

const WIDE = '(min-width: 800px)';
const DARK = '(prefers-color-scheme: dark)';

let seen: boolean[] = [];

function Probe({ query, fallback }: { query: string; fallback?: boolean }) {
  // An omitted serverFallback and an explicit undefined both mean the default.
  const matches: boolean = useMediaQuery(query, fallback);
  seen.push(matches);
  return <output>{matches ? 'yes' : 'no'}</output>;
}

const shown = () => screen.getByRole('status').textContent;

describe('useMediaQuery', () => {
  beforeEach(() => {
    seen = [];
  });

  it('returns the current match on the very first render', () => {
    installMatchMedia({ [WIDE]: true });
    render(<Probe query={WIDE} />);
    expect(seen[0]).toBe(true);
    expect(shown()).toBe('yes');
  });

  it('updates on change events', () => {
    const mm = installMatchMedia({ [WIDE]: true });
    render(<Probe query={WIDE} />);
    mm.set(WIDE, false);
    expect(shown()).toBe('no');
    mm.set(WIDE, true);
    expect(shown()).toBe('yes');
  });

  it('switches its subscription when the query changes', () => {
    const mm = installMatchMedia({ [WIDE]: true, [DARK]: false });
    const { rerender } = render(<Probe query={WIDE} />);
    expect(mm.listeners(WIDE)).toBeGreaterThan(0);
    seen = [];
    rerender(<Probe query={DARK} />);
    expect(seen[0]).toBe(false);
    expect(shown()).toBe('no');
    expect(mm.listeners(WIDE)).toBe(0);
    expect(mm.listeners(DARK)).toBeGreaterThan(0);
    mm.set(DARK, true);
    expect(shown()).toBe('yes');
    mm.set(WIDE, false);
    expect(shown()).toBe('yes');
  });

  it('removes its listener on unmount', () => {
    const mm = installMatchMedia({ [WIDE]: true });
    const { unmount } = render(<Probe query={WIDE} />);
    unmount();
    expect(mm.listeners(WIDE)).toBe(0);
  });

  it('returns serverFallback during server rendering', () => {
    vi.stubGlobal('window', undefined);
    vi.stubGlobal('matchMedia', undefined);
    expect(renderToString(<Probe query={WIDE} />)).toContain('<output>no</output>');
    expect(renderToString(<Probe query={WIDE} fallback={true} />)).toContain(
      '<output>yes</output>',
    );
  });

  it('hydrates server HTML without a mismatch, then shows the real value', async () => {
    vi.stubGlobal('window', undefined);
    vi.stubGlobal('matchMedia', undefined);
    const html = renderToString(<Probe query={WIDE} fallback={false} />);
    vi.unstubAllGlobals();
    installMatchMedia({ [WIDE]: true });

    const container = document.createElement('div');
    container.innerHTML = html;
    document.body.appendChild(container);
    const errors: unknown[] = [];
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    let root: ReturnType<typeof hydrateRoot> | undefined;
    await act(async () => {
      root = hydrateRoot(container, <Probe query={WIDE} fallback={false} />, {
        onRecoverableError: (e) => errors.push(e),
      });
    });
    quiet.mockRestore();
    expect(errors).toEqual([]);
    expect(container.textContent).toBe('yes');
    act(() => root?.unmount());
  });
});
