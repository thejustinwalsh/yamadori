import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useThrottle } from './solution';
import { advance } from './helpers';

let seen: string[] = [];

function Probe({ value, interval }: { value: string; interval: number }) {
  const v: string = useThrottle(value, interval);
  seen.push(v);
  return <output>{v}</output>;
}

function NumberProbe({ value }: { value: number }) {
  const n: number = useThrottle(value, 100);
  return <output>{n.toFixed(1)}</output>;
}

const shown = () => screen.getByRole('status').textContent;

describe('useThrottle', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    seen = [];
  });

  it('returns the first value immediately', () => {
    render(<Probe value="a" interval={300} />);
    expect(shown()).toBe('a');
  });

  it('emits a change immediately once intervalMs has passed since the last update', () => {
    const { rerender } = render(<Probe value="a" interval={300} />);
    advance(300);
    rerender(<Probe value="b" interval={300} />);
    expect(shown()).toBe('b');
    advance(1000);
    rerender(<Probe value="c" interval={300} />);
    expect(shown()).toBe('c');
  });

  it('delivers an early change as a trailing update at last update + intervalMs', () => {
    const { rerender } = render(<Probe value="a" interval={300} />);
    advance(100);
    rerender(<Probe value="b" interval={300} />);
    expect(shown()).toBe('a');
    advance(199);
    expect(shown()).toBe('a');
    advance(1);
    expect(shown()).toBe('b');
  });

  it('measures the window from the last emitted update', () => {
    const { rerender } = render(<Probe value="a" interval={300} />);
    advance(1000);
    rerender(<Probe value="b" interval={300} />);
    expect(shown()).toBe('b');
    advance(100);
    rerender(<Probe value="c" interval={300} />);
    advance(199);
    expect(shown()).toBe('b');
    advance(1);
    expect(shown()).toBe('c');
  });

  it('the trailing update carries the latest value and skips superseded ones', () => {
    const { rerender } = render(<Probe value="a" interval={300} />);
    advance(50);
    rerender(<Probe value="b" interval={300} />);
    advance(50);
    rerender(<Probe value="c" interval={300} />);
    advance(150);
    rerender(<Probe value="d" interval={300} />);
    advance(49);
    expect(shown()).toBe('a');
    advance(1);
    expect(shown()).toBe('d');
    expect(seen).not.toContain('b');
    expect(seen).not.toContain('c');
  });

  it('keeps updating once per interval while the input changes continuously', () => {
    const { rerender } = render(<Probe value="0" interval={300} />);
    const after: string[] = [];
    for (let k = 1; k <= 6; k++) {
      advance(100);
      rerender(<Probe value={String(k)} interval={300} />);
      after.push(shown() ?? '');
    }
    expect(after).toEqual(['0', '0', '2', '2', '2', '5']);
    advance(300);
    expect(shown()).toBe('6');
  });

  it('works for any value type', () => {
    const { rerender } = render(<NumberProbe value={1} />);
    rerender(<NumberProbe value={2} />);
    advance(100);
    expect(shown()).toBe('2.0');
  });

  it('leaves no timer pending after unmount', () => {
    const { rerender, unmount } = render(<Probe value="a" interval={300} />);
    advance(100);
    rerender(<Probe value="b" interval={300} />);
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
