import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Pagination } from './solution';

const nav = () => screen.getByRole('navigation', { name: 'Pagination' });

// The displayed sequence, read off the text nodes inside the nav: page numbers
// and ellipses in document order ("Previous"/"Next" text is skipped).
function sequence(): string {
  const walker = document.createTreeWalker(nav(), NodeFilter.SHOW_TEXT);
  const out: string[] = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const t = (n.textContent ?? '').trim();
    if (/^\d+$/.test(t) || t === '…') out.push(t);
  }
  return out.join(' ');
}

// Page buttons (everything but Previous/Next), by accessible name.
function pageButtons(): string[] {
  return within(nav())
    .getAllByRole('button')
    .map((b) => b.getAttribute('aria-label') ?? (b.textContent ?? '').trim())
    .filter((name) => name !== 'Previous' && name !== 'Next');
}

function view(page: number, totalPages: number, siblings?: number) {
  const onPageChange = vi.fn<(page: number) => void>();
  const r = render(
    siblings === undefined ? (
      <Pagination page={page} totalPages={totalPages} onPageChange={onPageChange} />
    ) : (
      <Pagination page={page} totalPages={totalPages} onPageChange={onPageChange} siblings={siblings} />
    ),
  );
  return { onPageChange, ...r };
}

function seqFor(page: number, totalPages: number, siblings?: number): string {
  const r = view(page, totalPages, siblings);
  const s = sequence();
  const numbers = s.split(' ').filter((t) => t !== '…');
  // every displayed number is a button named by that number, and nothing else is
  expect(pageButtons()).toEqual(numbers);
  r.unmount();
  return s;
}

describe('Pagination', () => {
  it('shows first, last, current +/- 1 and ellipses for longer gaps', () => {
    expect(seqFor(5, 10)).toBe('1 … 4 5 6 … 10');
    expect(seqFor(1, 10)).toBe('1 2 … 10');
    expect(seqFor(10, 10)).toBe('1 … 9 10');
  });

  it('shows a lone skipped page instead of an ellipsis', () => {
    expect(seqFor(3, 10)).toBe('1 2 3 4 … 10');
    expect(seqFor(4, 10)).toBe('1 2 3 4 5 … 10');
    expect(seqFor(7, 10)).toBe('1 … 6 7 8 9 10');
    expect(seqFor(8, 10)).toBe('1 … 7 8 9 10');
    expect(seqFor(3, 5)).toBe('1 2 3 4 5');
  });

  it('handles very small page counts', () => {
    expect(seqFor(1, 1)).toBe('1');
    expect(seqFor(2, 2)).toBe('1 2');
    expect(seqFor(1, 3)).toBe('1 2 3');
  });

  it('honours the siblings prop', () => {
    expect(seqFor(6, 12, 2)).toBe('1 … 4 5 6 7 8 … 12');
    expect(seqFor(5, 10, 0)).toBe('1 … 5 … 10');
    expect(seqFor(3, 10, 0)).toBe('1 2 3 … 10');
    expect(seqFor(4, 12, 2)).toBe('1 2 3 4 5 6 … 12');
  });

  it('marks only the current page with aria-current="page"', () => {
    const { container } = view(5, 10);
    expect(within(nav()).getByRole('button', { name: '5' }).getAttribute('aria-current')).toBe('page');
    expect(container.querySelectorAll('[aria-current]').length).toBe(1);
  });

  it('disables Previous on the first page and Next on the last', () => {
    const { rerender, onPageChange } = view(1, 3);
    const prev = () => screen.getByRole('button', { name: 'Previous' }) as HTMLButtonElement;
    const next = () => screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement;
    expect(prev().disabled).toBe(true);
    expect(next().disabled).toBe(false);
    rerender(<Pagination page={2} totalPages={3} onPageChange={onPageChange} />);
    expect(prev().disabled).toBe(false);
    expect(next().disabled).toBe(false);
    rerender(<Pagination page={3} totalPages={3} onPageChange={onPageChange} />);
    expect(prev().disabled).toBe(false);
    expect(next().disabled).toBe(true);
    rerender(<Pagination page={1} totalPages={1} onPageChange={onPageChange} />);
    expect(prev().disabled).toBe(true);
    expect(next().disabled).toBe(true);
  });

  it('reports page changes and stays controlled', async () => {
    const user = userEvent.setup();
    const { onPageChange, rerender } = view(5, 10);
    await user.click(screen.getByRole('button', { name: 'Previous' }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.click(screen.getByRole('button', { name: '10' }));
    expect(onPageChange.mock.calls).toEqual([[4], [6], [10]]);
    // nothing changes until the parent passes a new page
    expect(sequence()).toBe('1 … 4 5 6 … 10');
    rerender(<Pagination page={10} totalPages={10} onPageChange={onPageChange} />);
    expect(sequence()).toBe('1 … 9 10');
    expect(screen.getByRole('button', { name: '10' }).getAttribute('aria-current')).toBe('page');
  });
});
