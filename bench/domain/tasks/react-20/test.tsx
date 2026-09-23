import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { InfiniteList } from './solution';
import { deferred, settle, installIntersectionObserver, type Deferred } from './helpers';

type Page = { items: string[]; hasMore: boolean };

function setup() {
  const io = installIntersectionObserver();
  const pending: Deferred<Page>[] = [];
  const loadPage = vi.fn((page: number): Promise<Page> => {
    void page;
    const d = deferred<Page>();
    pending.push(d);
    return d.promise;
  });
  const utils = render(<InfiniteList loadPage={loadPage} />);
  const sentinel = () => screen.getByTestId('sentinel');
  const scroll = (on = true) => {
    const s = screen.queryByTestId('sentinel');
    if (s) io.trigger(s, on);
  };
  return { io, pending, loadPage, sentinel, scroll, ...utils };
}

const items = () => {
  const list = screen.queryByRole('list');
  return list ? within(list).queryAllByRole('listitem').map((li) => li.textContent) : [];
};
const loading = () => {
  const s = screen.queryByRole('status');
  return s !== null && (s.textContent ?? '').includes('Loading');
};

describe('InfiniteList', () => {
  it('loads page 0 on mount and shows a loading status until it resolves', async () => {
    const { loadPage, pending } = setup();
    expect(loadPage.mock.calls).toEqual([[0]]);
    expect(loading()).toBe(true);
    await settle(pending[0], { items: ['a', 'b'], hasMore: true });
    expect(items()).toEqual(['a', 'b']);
    expect(loading()).toBe(false);
    expect(screen.queryByText(/Loading/)).toBeNull();
  });

  it('observes the sentinel and appends the next page when it intersects', async () => {
    const { io, loadPage, pending, sentinel, scroll } = setup();
    await settle(pending[0], { items: ['a', 'b'], hasMore: true });
    expect(io.observed()).toContain(sentinel());
    scroll();
    expect(loadPage.mock.calls).toEqual([[0], [1]]);
    expect(loading()).toBe(true);
    await settle(pending[1], { items: ['c'], hasMore: true });
    expect(items()).toEqual(['a', 'b', 'c']);
    scroll();
    expect(loadPage.mock.calls).toEqual([[0], [1], [2]]);
    await settle(pending[2], { items: ['d', 'e'], hasMore: true });
    expect(items()).toEqual(['a', 'b', 'c', 'd', 'e']);
    expect(loading()).toBe(false);
  });

  it('ignores entries that are not intersecting', async () => {
    const { loadPage, pending, scroll } = setup();
    await settle(pending[0], { items: ['a'], hasMore: true });
    scroll(false);
    expect(loadPage).toHaveBeenCalledTimes(1);
  });

  it('never starts a second load while one is in flight', async () => {
    const { loadPage, pending, scroll } = setup();
    scroll();
    scroll();
    expect(loadPage.mock.calls).toEqual([[0]]);
    await settle(pending[0], { items: ['a'], hasMore: true });
    scroll();
    scroll();
    scroll();
    expect(loadPage.mock.calls).toEqual([[0], [1]]);
    await settle(pending[1], { items: ['b'], hasMore: true });
    scroll();
    expect(loadPage.mock.calls).toEqual([[0], [1], [2]]);
    await settle(pending[2], { items: ['c'], hasMore: true });
    expect(items()).toEqual(['a', 'b', 'c']);
  });

  it('stops for good after hasMore is false', async () => {
    const { loadPage, pending, scroll } = setup();
    await settle(pending[0], { items: ['a'], hasMore: true });
    scroll();
    await settle(pending[1], { items: ['b'], hasMore: false });
    expect(screen.getByText('No more items')).toBeTruthy();
    expect(items()).toEqual(['a', 'b']);
    scroll();
    scroll();
    expect(loadPage).toHaveBeenCalledTimes(2);
    expect(loading()).toBe(false);
  });

  it('a first page with hasMore false never loads again', async () => {
    const { loadPage, pending, scroll } = setup();
    await settle(pending[0], { items: ['only'], hasMore: false });
    scroll();
    expect(loadPage).toHaveBeenCalledTimes(1);
    expect(screen.getByText('No more items')).toBeTruthy();
  });

  it('stops observing on unmount', async () => {
    const { io, pending, unmount } = setup();
    await settle(pending[0], { items: ['a'], hasMore: true });
    expect(io.observed().length).toBeGreaterThan(0);
    unmount();
    expect(io.observed()).toEqual([]);
  });
});
