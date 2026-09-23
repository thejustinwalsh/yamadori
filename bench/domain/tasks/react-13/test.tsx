import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { useIdle } from './solution';
import { advance } from './helpers';

function Probe({ ms }: { ms: number }) {
  const idle: boolean = useIdle(ms);
  return <output>{idle ? 'idle' : 'active'}</output>;
}

const shown = () => screen.getByRole('status').textContent;

const ACTIVITY = ['mousemove', 'mousedown', 'keydown', 'touchstart', 'scroll', 'wheel'] as const;

function activity(type: string) {
  act(() => {
    window.dispatchEvent(new Event(type));
  });
}

describe('useIdle', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('starts active and goes idle after timeoutMs without activity', () => {
    render(<Probe ms={1000} />);
    expect(shown()).toBe('active');
    advance(999);
    expect(shown()).toBe('active');
    advance(1);
    expect(shown()).toBe('idle');
  });

  it('restarts the countdown on activity', () => {
    render(<Probe ms={1000} />);
    advance(600);
    activity('mousemove');
    advance(600);
    expect(shown()).toBe('active');
    advance(399);
    expect(shown()).toBe('active');
    advance(1);
    expect(shown()).toBe('idle');
  });

  it('wakes up on activity and goes idle again timeoutMs later', () => {
    render(<Probe ms={500} />);
    advance(500);
    expect(shown()).toBe('idle');
    activity('keydown');
    expect(shown()).toBe('active');
    advance(499);
    expect(shown()).toBe('active');
    advance(1);
    expect(shown()).toBe('idle');
  });

  it('counts every listed event type as activity', () => {
    render(<Probe ms={300} />);
    for (const type of ACTIVITY) {
      advance(300);
      expect(shown(), `idle before ${type}`).toBe('idle');
      activity(type);
      expect(shown(), `woken by ${type}`).toBe('active');
      advance(200);
      activity(type);
      advance(200);
      expect(shown(), `countdown restarted by ${type}`).toBe('active');
    }
  });

  it('restarts the countdown when timeoutMs changes', () => {
    const { rerender } = render(<Probe ms={1000} />);
    advance(600);
    rerender(<Probe ms={2000} />);
    advance(1999);
    expect(shown()).toBe('active');
    advance(1);
    expect(shown()).toBe('idle');
  });

  it('leaves no timer pending after unmount', () => {
    const { unmount } = render(<Probe ms={1000} />);
    advance(400);
    activity('mousedown');
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('removes every listener it added to window on unmount', () => {
    const add = vi.spyOn(window, 'addEventListener');
    const remove = vi.spyOn(window, 'removeEventListener');
    const { unmount } = render(<Probe ms={1000} />);
    const added = add.mock.calls
      .filter((c) => (ACTIVITY as readonly string[]).includes(c[0]))
      .map((c) => [c[0], c[1]] as const);
    expect(added.length).toBeGreaterThanOrEqual(ACTIVITY.length);
    unmount();
    const removed = remove.mock.calls.map((c) => [c[0], c[1]] as const);
    for (const [type, fn] of added) {
      expect(
        removed.some(([t, f]) => t === type && f === fn),
        `${type} listener removed`,
      ).toBe(true);
    }
  });
});
