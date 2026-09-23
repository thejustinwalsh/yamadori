import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useDebounce } from './solution';
import { advance } from './helpers';

function Probe({ value, delay }: { value: string; delay: number }) {
  const v: string = useDebounce(value, delay);
  return <output>{v}</output>;
}

function NumberProbe({ value }: { value: number }) {
  const n: number = useDebounce(value, 100);
  return <output>{n.toFixed(1)}</output>;
}

const shown = () => screen.getByRole('status').textContent;

describe('useDebounce', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('returns the first value immediately', () => {
    render(<Probe value="a" delay={500} />);
    expect(shown()).toBe('a');
  });

  it('updates only after the value has been stable for delayMs', () => {
    const { rerender } = render(<Probe value="a" delay={500} />);
    rerender(<Probe value="b" delay={500} />);
    expect(shown()).toBe('a');
    advance(499);
    expect(shown()).toBe('a');
    advance(1);
    expect(shown()).toBe('b');
  });

  it('restarts the wait on every change and never shows an intermediate value', () => {
    const { rerender } = render(<Probe value="a" delay={500} />);
    rerender(<Probe value="b" delay={500} />);
    advance(300);
    rerender(<Probe value="c" delay={500} />);
    advance(300);
    expect(shown()).toBe('a');
    advance(199);
    expect(shown()).toBe('a');
    advance(1);
    expect(shown()).toBe('c');
  });

  it('restarts the wait when delayMs changes', () => {
    const { rerender } = render(<Probe value="a" delay={500} />);
    rerender(<Probe value="b" delay={500} />);
    advance(300);
    rerender(<Probe value="b" delay={1000} />);
    advance(400);
    expect(shown()).toBe('a');
    advance(600);
    expect(shown()).toBe('b');
  });

  it('works for any value type', () => {
    const { rerender } = render(<NumberProbe value={1} />);
    rerender(<NumberProbe value={2} />);
    advance(100);
    expect(shown()).toBe('2.0');
  });

  it('leaves no timer pending after unmount', () => {
    const { rerender, unmount } = render(<Probe value="a" delay={500} />);
    rerender(<Probe value="b" delay={500} />);
    rerender(<Probe value="c" delay={500} />);
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
