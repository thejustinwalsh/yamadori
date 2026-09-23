import { describe, it, expect } from 'vitest';
import { StrictMode } from 'react';
import { render, screen, within } from '@testing-library/react';
import { Leaderboard } from './solution';

type P = { name: string; score: number };

const rows = (list: HTMLElement = screen.getByRole('list')) =>
  within(list).queryAllByRole('listitem').map((li) => li.textContent);

const deepFreeze = (ps: P[]): readonly P[] => Object.freeze(ps.map((p) => Object.freeze({ ...p })));

// A fresh copy per use, so an answer that mutates its input cannot hide it
// by pre-sorting a shared fixture for the tests that follow.
const ties = (): P[] => [
  { name: 'Dee', score: 70 },
  { name: 'Cal', score: 90 },
  { name: 'Eli', score: 40 },
  { name: 'Ava', score: 90 },
  { name: 'Ben', score: 100 },
  { name: 'Fay', score: 70 },
  { name: 'Gus', score: 70 },
].map((p) => ({ ...p }));

const TIES_EXPECTED = [
  '1. Ben — 100',
  '2. Ava — 90',
  '2. Cal — 90',
  '4. Dee — 70',
  '4. Fay — 70',
  '4. Gus — 70',
  '7. Eli — 40',
];

describe('Leaderboard', () => {
  it('lists players highest score first in the stated format', () => {
    render(
      <Leaderboard
        players={[
          { name: 'Ava', score: 12 },
          { name: 'Ben', score: 30 },
          { name: 'Cal', score: 21 },
        ]}
      />,
    );
    expect(rows()).toEqual(['1. Ben — 30', '2. Cal — 21', '3. Ava — 12']);
  });

  it('gives ties the same rank, skips the next ranks and orders ties by name', () => {
    render(<Leaderboard players={ties()} />);
    expect(rows()).toEqual(TIES_EXPECTED);
  });

  it('renders an empty list for no players', () => {
    render(<Leaderboard players={[]} />);
    expect(rows()).toEqual([]);
  });

  it('works on a deeply frozen array', () => {
    render(<Leaderboard players={deepFreeze(ties())} />);
    expect(rows()).toEqual(TIES_EXPECTED);
  });

  it('leaves an ordinary input array and its players untouched', () => {
    const input = ties();
    const before = JSON.stringify(input);
    const { rerender } = render(<Leaderboard players={input} />);
    rerender(<Leaderboard players={input} />);
    expect(JSON.stringify(input)).toBe(before);
    expect(rows()).toEqual(TIES_EXPECTED);
  });

  it('renders the same under StrictMode double rendering', () => {
    render(
      <StrictMode>
        <Leaderboard players={ties()} />
      </StrictMode>,
    );
    expect(rows()).toEqual(TIES_EXPECTED);
  });

  it('renders the same on every re-render and after the props change', () => {
    const same = ties();
    const { rerender } = render(<Leaderboard players={same} />);
    rerender(<Leaderboard players={same} />);
    rerender(<Leaderboard players={ties()} />);
    expect(rows()).toEqual(TIES_EXPECTED);
    rerender(<Leaderboard players={[{ name: 'Zed', score: 5 }, { name: 'Yan', score: 5 }]} />);
    expect(rows()).toEqual(['1. Yan — 5', '1. Zed — 5']);
  });

  it('numbers two leaderboards on one page independently', () => {
    render(
      <StrictMode>
        <div>
          <Leaderboard players={[{ name: 'Ava', score: 3 }, { name: 'Ben', score: 2 }]} />
        </div>
        <div>
          <Leaderboard players={[{ name: 'Cal', score: 9 }, { name: 'Dee', score: 1 }]} />
        </div>
      </StrictMode>,
    );
    const [a, b] = screen.getAllByRole('list');
    expect(rows(a)).toEqual(['1. Ava — 3', '2. Ben — 2']);
    expect(rows(b)).toEqual(['1. Cal — 9', '2. Dee — 1']);
  });
});
