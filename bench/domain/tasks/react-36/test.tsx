import { describe, it, expect } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SlowSearch } from './solution';
import { deferred, settle, type Deferred } from './helpers';

// One promise per query, as the prompt promises: calling getResults during
// render is stable. The test decides when each query's results arrive.
function source() {
  const byQuery = new Map<string, Deferred<string[]>>();
  const calls: string[] = [];
  const slot = (q: string) => {
    let d = byQuery.get(q);
    if (!d) {
      d = deferred<string[]>();
      byQuery.set(q, d);
    }
    return d;
  };
  const getResults = (q: string): Promise<string[]> => {
    calls.push(q);
    return slot(q).promise;
  };
  return {
    getResults,
    calls,
    resolve: (q: string, items: string[]) => settle(slot(q), items),
  };
}

const input = () => screen.getByLabelText('Search') as HTMLInputElement;
const items = () => screen.queryAllByRole('listitem').map((li) => li.textContent);
const loading = () => screen.queryByText('Loading…') !== null;
const updating = () =>
  screen.queryAllByRole('status').some((el) => (el.textContent ?? '').includes('Updating…'));

// Rendering and typing go through an awaited act(): a component that
// suspends inside a synchronous act() is never retried by React's test
// scheduler, so a correct answer would look stuck.
async function mount() {
  const src = source();
  await act(async () => {
    render(<SlowSearch getResults={src.getResults} />);
  });
  const user = userEvent.setup();
  return {
    ...src,
    type: (text: string) =>
      act(async () => {
        await user.type(input(), text);
      }),
    clear: () =>
      act(async () => {
        await user.clear(input());
      }),
  };
}

async function loaded() {
  const s = await mount();
  await s.resolve('', ['apple', 'banana', 'cherry']);
  return s;
}

describe('SlowSearch', () => {
  it('shows the fallback until the results for the empty query arrive', async () => {
    const s = await mount();
    expect(s.calls).toContain('');
    expect(loading()).toBe(true);
    expect(screen.queryByRole('list')).toBe(null);
    expect(input().value).toBe('');
    await s.resolve('', ['apple', 'banana', 'cherry']);
    expect(loading()).toBe(false);
    expect(screen.getByRole('list')).toBeTruthy();
    expect(items()).toEqual(['apple', 'banana', 'cherry']);
    expect(updating()).toBe(false);
  });

  it('updates the input immediately while the new results are still loading', async () => {
    const s = await loaded();
    await s.type('b');
    expect(input().value).toBe('b');
    await s.type('a');
    expect(input().value).toBe('ba');
  });

  it('keeps the old list on screen instead of the fallback while loading', async () => {
    const s = await loaded();
    await s.type('b');
    expect(loading()).toBe(false);
    expect(items()).toEqual(['apple', 'banana', 'cherry']);
  });

  it('shows the Updating… status only while new results are pending', async () => {
    const s = await loaded();
    expect(updating()).toBe(false);
    await s.type('b');
    expect(updating()).toBe(true);
    await s.resolve('b', ['banana']);
    expect(updating()).toBe(false);
  });

  it('shows the new results once they arrive', async () => {
    const s = await loaded();
    await s.type('c');
    await s.resolve('c', ['cherry', 'coconut']);
    expect(items()).toEqual(['cherry', 'coconut']);
    expect(s.calls).toContain('c');
  });

  it('lands on the results for the latest text after fast typing', async () => {
    const s = await loaded();
    await s.type('ch');
    expect(input().value).toBe('ch');
    await s.resolve('c', ['cherry', 'coconut']);
    await s.resolve('ch', ['cherry']);
    expect(items()).toEqual(['cherry']);
    expect(updating()).toBe(false);
    expect(loading()).toBe(false);
    expect(input().value).toBe('ch');
  });

  it('never shows the fallback again across several searches', async () => {
    const s = await loaded();
    const seen: boolean[] = [];
    await s.type('k');
    seen.push(loading());
    await s.resolve('k', ['kiwi']);
    seen.push(loading());
    expect(items()).toEqual(['kiwi']);
    await s.type('i');
    seen.push(loading());
    expect(items()).toEqual(['kiwi']);
    await s.resolve('ki', ['kiwi', 'kiwano']);
    seen.push(loading());
    expect(items()).toEqual(['kiwi', 'kiwano']);
    await s.clear();
    seen.push(loading());
    expect(seen).toEqual([false, false, false, false, false]);
    expect(items()).toEqual(['apple', 'banana', 'cherry']);
    expect(updating()).toBe(false);
  });
});
