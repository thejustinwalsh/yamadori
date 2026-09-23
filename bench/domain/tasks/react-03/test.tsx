import { describe, it, expect } from 'vitest';
import { useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { usePrevious } from './solution';

function Probe({ value, tick = 0 }: { value: string; tick?: number }) {
  const prev: string | undefined = usePrevious(value);
  return (
    <output data-tick={tick}>{prev === undefined ? '(none)' : prev}</output>
  );
}

function NumberProbe({ value }: { value: number }) {
  const prev: number | undefined = usePrevious(value);
  return <output>{prev === undefined ? '(none)' : prev.toFixed(1)}</output>;
}

const A = { name: 'first' };
const B = { name: 'second' };

function ObjectProbe({ value }: { value: { name: string } }) {
  const prev = usePrevious(value);
  return <output>{prev === undefined ? '(none)' : prev.name}</output>;
}

function Counter() {
  const [count, setCount] = useState(0);
  const [other, setOther] = useState(0);
  const prev = usePrevious(count);
  return (
    <div>
      <button onClick={() => setCount((c) => c + 1)}>increment</button>
      <button onClick={() => setOther((o) => o + 1)}>unrelated {other}</button>
      <output>{prev === undefined ? '(none)' : String(prev)}</output>
    </div>
  );
}

const shown = () => screen.getByRole('status').textContent;

describe('usePrevious', () => {
  it('returns undefined on the first render', () => {
    render(<Probe value="a" />);
    expect(shown()).toBe('(none)');
  });

  it('stays undefined while the value has never changed', () => {
    const { rerender } = render(<Probe value="a" />);
    rerender(<Probe value="a" tick={1} />);
    rerender(<Probe value="a" tick={2} />);
    expect(shown()).toBe('(none)');
  });

  it('returns the value from before the last change', () => {
    const { rerender } = render(<Probe value="a" />);
    rerender(<Probe value="b" />);
    expect(shown()).toBe('a');
    rerender(<Probe value="c" />);
    expect(shown()).toBe('b');
  });

  it('keeps the previous value across re-renders that do not change it', () => {
    const { rerender } = render(<Probe value="a" />);
    rerender(<Probe value="b" />);
    rerender(<Probe value="b" tick={1} />);
    expect(shown()).toBe('a');
    rerender(<Probe value="b" tick={2} />);
    expect(shown()).toBe('a');
  });

  it('follows the example sequence a, a, b, b, c, a', () => {
    const { rerender } = render(<Probe value="a" />);
    const out: (string | null)[] = [shown()];
    for (const [i, v] of ['a', 'b', 'b', 'c', 'a'].entries()) {
      rerender(<Probe value={v} tick={i + 1} />);
      out.push(shown());
    }
    expect(out).toEqual(['(none)', '(none)', 'a', 'a', 'b', 'c']);
  });

  it('is unaffected by state updates elsewhere in the component', async () => {
    const user = userEvent.setup();
    render(<Counter />);
    await user.click(screen.getByRole('button', { name: 'increment' }));
    expect(shown()).toBe('0');
    await user.click(screen.getByRole('button', { name: /unrelated/ }));
    await user.click(screen.getByRole('button', { name: /unrelated/ }));
    expect(shown()).toBe('0');
    await user.click(screen.getByRole('button', { name: 'increment' }));
    expect(shown()).toBe('1');
  });

  it('compares by identity and works for any value type', () => {
    const { rerender } = render(<ObjectProbe value={A} />);
    rerender(<ObjectProbe value={A} />);
    expect(shown()).toBe('(none)');
    rerender(<ObjectProbe value={B} />);
    expect(shown()).toBe('first');
    const n = render(<NumberProbe value={1} />);
    n.rerender(<NumberProbe value={2} />);
    expect(n.container.textContent).toBe('1.0');
  });
});
