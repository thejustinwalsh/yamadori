import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { memo } from 'react';
import { useCounter } from './solution';

type Opts = { initial?: number; min?: number; max?: number; step?: number };

const renders: Record<string, number> = {};

const Action = memo(function Action({ label, run }: { label: string; run: () => void }) {
  renders[label] = (renders[label] ?? 0) + 1;
  return <button onClick={run}>{label}</button>;
});

function Counter({ opts }: { opts?: Opts }) {
  // A fresh options object on every render, on purpose.
  const c = useCounter(opts && { ...opts });
  const n: number = c.count;
  return (
    <div>
      <output>{String(n)}</output>
      <Action label="increment" run={c.increment} />
      <Action label="decrement" run={c.decrement} />
      <Action label="reset" run={c.reset} />
      <button
        onClick={() => {
          c.increment();
          c.increment();
        }}
      >
        twice
      </button>
      <button onClick={() => c.set(100)}>set 100</button>
      <button onClick={() => c.set(-100)}>set -100</button>
      <button onClick={() => c.set(2)}>set 2</button>
    </div>
  );
}

const shown = () => screen.getByRole('status').textContent;

function setup(opts?: Opts) {
  for (const k of Object.keys(renders)) delete renders[k];
  const user = userEvent.setup({ delay: null });
  render(<Counter opts={opts} />);
  const click = async (name: string, times = 1) => {
    for (let i = 0; i < times; i++) await user.click(screen.getByRole('button', { name }));
  };
  return { click };
}

describe('useCounter', () => {
  it('works with no options: starts at 0, steps by 1, unbounded', async () => {
    const { click } = setup();
    expect(shown()).toBe('0');
    await click('increment');
    expect(shown()).toBe('1');
    await click('decrement', 3);
    expect(shown()).toBe('-2');
    await click('set 100');
    expect(shown()).toBe('100');
  });

  it('uses initial and step', async () => {
    const { click } = setup({ initial: 10, step: 5 });
    expect(shown()).toBe('10');
    await click('increment');
    expect(shown()).toBe('15');
    await click('decrement', 2);
    expect(shown()).toBe('5');
  });

  it('clamps increment and decrement to [min, max]', async () => {
    const { click } = setup({ min: 0, max: 3, step: 2 });
    await click('increment', 3);
    expect(shown()).toBe('3');
    await click('decrement', 3);
    expect(shown()).toBe('0');
  });

  it('clamps set', async () => {
    const { click } = setup({ min: -5, max: 5 });
    await click('set 100');
    expect(shown()).toBe('5');
    await click('set -100');
    expect(shown()).toBe('-5');
    await click('set 2');
    expect(shown()).toBe('2');
  });

  it('clamps the initial value, and reset returns to it', async () => {
    const { click } = setup({ initial: 10, max: 7 });
    expect(shown()).toBe('7');
    await click('decrement', 2);
    expect(shown()).toBe('5');
    await click('reset');
    expect(shown()).toBe('7');
  });

  it('applies every call made in the same handler', async () => {
    const { click } = setup({ step: 3, max: 10 });
    await click('twice');
    expect(shown()).toBe('6');
    await click('twice');
    expect(shown()).toBe('10');
  });

  it('keeps the functions stable so memoised children do not re-render', async () => {
    const { click } = setup({ min: 0, max: 100, step: 1 });
    await click('increment', 3);
    await click('decrement');
    await click('set 100');
    await click('reset');
    expect(shown()).toBe('0');
    expect(renders).toEqual({ increment: 1, decrement: 1, reset: 1 });
  });
});
