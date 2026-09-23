import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useRef, type RefObject } from 'react';
import { useClickOutside } from './solution';

type Handler = (event: MouseEvent | TouchEvent) => void;

function Panel({ onOutside, open = true }: { onOutside: Handler; open?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useClickOutside(ref, onOutside);
  return (
    <div>
      <button>outside</button>
      {open && (
        <div ref={ref} aria-label="panel" role="dialog">
          <p>
            <span>nested text</span>
          </p>
          <button>inside</button>
        </div>
      )}
    </div>
  );
}

const outside = () => screen.getByRole('button', { name: 'outside' });

describe('useClickOutside', () => {
  it('calls the handler on mousedown outside, with the event', () => {
    const h = vi.fn<Handler>();
    render(<Panel onOutside={h} />);
    fireEvent.mouseDown(outside());
    expect(h).toHaveBeenCalledTimes(1);
    const ev = h.mock.calls[0][0];
    expect(ev.type).toBe('mousedown');
    expect(ev.target).toBe(outside());
  });

  it('calls the handler on touchstart outside, including on the document body', () => {
    const h = vi.fn<Handler>();
    render(<Panel onOutside={h} />);
    fireEvent.touchStart(outside());
    fireEvent.mouseDown(document.body);
    expect(h).toHaveBeenCalledTimes(2);
    expect(h.mock.calls[0][0].type).toBe('touchstart');
  });

  it('ignores presses on the element and anything inside it', () => {
    const h = vi.fn<Handler>();
    render(<Panel onOutside={h} />);
    fireEvent.mouseDown(screen.getByRole('dialog', { name: 'panel' }));
    fireEvent.mouseDown(screen.getByRole('button', { name: 'inside' }));
    fireEvent.mouseDown(screen.getByText('nested text'));
    fireEvent.touchStart(screen.getByText('nested text'));
    expect(h).not.toHaveBeenCalled();
  });

  it('ignores other event types such as click', () => {
    const h = vi.fn<Handler>();
    render(<Panel onOutside={h} />);
    fireEvent.click(outside());
    fireEvent.mouseUp(outside());
    expect(h).not.toHaveBeenCalled();
  });

  it('reads the ref when the event happens', () => {
    const h = vi.fn<Handler>();
    const { rerender } = render(<Panel onOutside={h} open={false} />);
    fireEvent.mouseDown(outside());
    expect(h).not.toHaveBeenCalled();
    rerender(<Panel onOutside={h} open />);
    fireEvent.mouseDown(screen.getByRole('button', { name: 'inside' }));
    expect(h).not.toHaveBeenCalled();
    fireEvent.mouseDown(outside());
    expect(h).toHaveBeenCalledTimes(1);
  });

  it('uses the handler from the latest render', () => {
    const first = vi.fn<Handler>();
    const second = vi.fn<Handler>();
    const { rerender } = render(<Panel onOutside={first} />);
    rerender(<Panel onOutside={second} />);
    fireEvent.mouseDown(outside());
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it('stops listening after unmount', () => {
    // The watched element lives outside the component, so ref.current stays
    // set after the component using the hook has gone.
    const box = document.createElement('div');
    document.body.append(box);
    const ref: RefObject<HTMLDivElement | null> = { current: box };
    function Watcher({ onOutside }: { onOutside: Handler }) {
      useClickOutside(ref, onOutside);
      return null;
    }
    const h = vi.fn<Handler>();
    const { unmount } = render(<Watcher onOutside={h} />);
    fireEvent.mouseDown(document.body);
    expect(h).toHaveBeenCalledTimes(1);
    unmount();
    fireEvent.mouseDown(document.body);
    fireEvent.touchStart(document.body);
    box.remove();
    expect(h).toHaveBeenCalledTimes(1);
  });
});
