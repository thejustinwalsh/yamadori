import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { useInterval } from './solution';
import { advance } from './helpers';

let log: string[] = [];

function Ticker({ delay, label = 'x' }: { delay: number | null; label?: string }) {
  // A new inline callback on every render.
  const r: void = useInterval(() => {
    log.push(label);
  }, delay);
  void r;
  return <p>{label}</p>;
}

describe('useInterval', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    log = [];
  });

  it('calls the callback every delayMs, starting delayMs after mount', () => {
    render(<Ticker delay={1000} />);
    advance(999);
    expect(log.length).toBe(0);
    advance(1);
    expect(log.length).toBe(1);
    advance(1000);
    expect(log.length).toBe(2);
    advance(3000);
    expect(log.length).toBe(5);
  });

  it('does not restart when it re-renders with a new callback', () => {
    const { rerender } = render(<Ticker delay={1000} />);
    for (let i = 1; i <= 5; i++) {
      advance(400);
      rerender(<Ticker delay={1000} />);
    }
    // t = 2000
    expect(log.length).toBe(2);
  });

  it('calls the latest callback', () => {
    const { rerender } = render(<Ticker delay={1000} label="a" />);
    advance(500);
    rerender(<Ticker delay={1000} label="b" />);
    advance(500);
    expect(log).toEqual(['b']);
    rerender(<Ticker delay={1000} label="c" />);
    advance(1000);
    expect(log).toEqual(['b', 'c']);
  });

  it('restarts the interval when delayMs changes', () => {
    const { rerender } = render(<Ticker delay={1000} />);
    advance(600);
    rerender(<Ticker delay={300} />);
    advance(299);
    expect(log.length).toBe(0);
    advance(1);
    expect(log.length).toBe(1);
    advance(300);
    expect(log.length).toBe(2);
    expect(vi.getTimerCount()).toBe(1);
  });

  it('is paused while delayMs is null and resumes with a fresh interval', () => {
    const { rerender } = render(<Ticker delay={null} />);
    advance(5000);
    expect(log.length).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
    rerender(<Ticker delay={500} />);
    advance(500);
    expect(log.length).toBe(1);
    rerender(<Ticker delay={null} />);
    advance(5000);
    expect(log.length).toBe(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('leaves no timer pending after unmount', () => {
    const { unmount } = render(<Ticker delay={1000} />);
    advance(1500);
    unmount();
    expect(vi.getTimerCount()).toBe(0);
    advance(5000);
    expect(log.length).toBe(1);
  });
});
