import { describe, it, expect, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import { useEventListener } from './solution';

function ResizeProbe({ onResize, target }: { onResize: (e: UIEvent) => void; target?: EventTarget | null }) {
  useEventListener('resize', onResize, target);
  return null;
}

function KeyProbe({ onKey }: { onKey: (e: KeyboardEvent) => void }) {
  useEventListener('keydown', (e) => onKey(e));
  return null;
}

function fire(target: EventTarget, type: string): Event {
  const ev = new Event(type);
  act(() => {
    target.dispatchEvent(ev);
  });
  return ev;
}

const addsFor = (spy: { mock: { calls: unknown[][] } }, type: string) =>
  spy.mock.calls.filter((c) => c[0] === type).length;

describe('useEventListener', () => {
  it('listens on window by default and passes the event through', () => {
    const h = vi.fn<(e: UIEvent) => void>();
    render(<ResizeProbe onResize={h} />);
    const ev = fire(window, 'resize');
    expect(h).toHaveBeenCalledTimes(1);
    expect(h.mock.calls[0][0]).toBe(ev);
  });

  it('types the event from WindowEventMap', () => {
    const seen: string[] = [];
    render(<KeyProbe onKey={(e) => seen.push(e.key)} />);
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(seen).toEqual(['Escape']);
  });

  it('always calls the handler from the latest render', () => {
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = render(<ResizeProbe onResize={first} />);
    rerender(<ResizeProbe onResize={second} />);
    fire(window, 'resize');
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it('does not re-subscribe when only the handler changes', () => {
    const add = vi.spyOn(window, 'addEventListener');
    const remove = vi.spyOn(window, 'removeEventListener');
    const { rerender } = render(<ResizeProbe onResize={() => {}} />);
    rerender(<ResizeProbe onResize={() => {}} />);
    rerender(<ResizeProbe onResize={() => {}} />);
    expect(addsFor(add, 'resize')).toBe(1);
    expect(addsFor(remove, 'resize')).toBe(0);
  });

  it('listens on an explicit target and moves when the target changes', () => {
    const a = document.createElement('div');
    const b = document.createElement('div');
    const h = vi.fn();
    const { rerender } = render(<ResizeProbe onResize={h} target={a} />);
    fire(window, 'resize');
    expect(h).not.toHaveBeenCalled();
    fire(a, 'resize');
    expect(h).toHaveBeenCalledTimes(1);
    rerender(<ResizeProbe onResize={h} target={b} />);
    fire(a, 'resize');
    expect(h).toHaveBeenCalledTimes(1);
    fire(b, 'resize');
    expect(h).toHaveBeenCalledTimes(2);
  });

  it('attaches nothing while target is null', () => {
    const h = vi.fn();
    const { rerender } = render(<ResizeProbe onResize={h} target={null} />);
    fire(window, 'resize');
    expect(h).not.toHaveBeenCalled();
    rerender(<ResizeProbe onResize={h} target={window} />);
    fire(window, 'resize');
    expect(h).toHaveBeenCalledTimes(1);
  });

  it('stops listening after unmount', () => {
    const h = vi.fn();
    const el = document.createElement('div');
    const { unmount } = render(
      <>
        <ResizeProbe onResize={h} />
        <ResizeProbe onResize={h} target={el} />
      </>,
    );
    unmount();
    fire(window, 'resize');
    fire(el, 'resize');
    expect(h).not.toHaveBeenCalled();
  });
});
