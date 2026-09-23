import { describe, it, expect } from 'vitest';
import { useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useInView } from './solution';
import { installIntersectionObserver } from './helpers';

function Box({ options }: { options?: IntersectionObserverInit }) {
  const r: { ref: (node: HTMLDivElement | null) => void; inView: boolean } =
    useInView<HTMLDivElement>(options);
  return (
    <div>
      <div ref={r.ref} data-testid="target">
        target
      </div>
      <output>{r.inView ? 'visible' : 'hidden'}</output>
    </div>
  );
}

function Later() {
  const [show, setShow] = useState(false);
  const { ref, inView } = useInView();
  return (
    <div>
      <button onClick={() => setShow(true)}>show</button>
      {show && (
        <p ref={ref} data-testid="late">
          late
        </p>
      )}
      <output>{inView ? 'visible' : 'hidden'}</output>
    </div>
  );
}

function Switcher() {
  const [which, setWhich] = useState<'first' | 'second' | 'none'>('first');
  const { ref } = useInView<HTMLElement>();
  return (
    <div>
      <button onClick={() => setWhich('second')}>second</button>
      <button onClick={() => setWhich('none')}>none</button>
      {which === 'first' && (
        <section ref={ref} data-testid="first">
          first
        </section>
      )}
      {which === 'second' && (
        <article ref={ref} data-testid="second">
          second
        </article>
      )}
    </div>
  );
}

const shown = () => screen.getByRole('status').textContent;

describe('useInView', () => {
  it('starts out of view and observes the attached element', () => {
    const io = installIntersectionObserver();
    render(<Box />);
    expect(shown()).toBe('hidden');
    expect(io.observed()).toContain(screen.getByTestId('target'));
  });

  it('follows the latest entry for the element', () => {
    const io = installIntersectionObserver();
    render(<Box />);
    const el = screen.getByTestId('target');
    io.trigger(el, true);
    expect(shown()).toBe('visible');
    io.trigger(el, false);
    expect(shown()).toBe('hidden');
    io.trigger(el, true);
    expect(shown()).toBe('visible');
  });

  it('passes options to the IntersectionObserver constructor', () => {
    const io = installIntersectionObserver();
    render(<Box options={{ rootMargin: '20px', threshold: 0.5 }} />);
    const el = screen.getByTestId('target');
    const watching = io.instances().filter((o) => o.elements.has(el));
    expect(watching.length).toBeGreaterThan(0);
    for (const o of watching) {
      expect(o.options?.rootMargin).toBe('20px');
      expect(o.options?.threshold).toBe(0.5);
    }
  });

  it('observes an element that mounts after the component', async () => {
    const io = installIntersectionObserver();
    const user = userEvent.setup();
    render(<Later />);
    expect(io.observed()).toEqual([]);
    await user.click(screen.getByRole('button', { name: 'show' }));
    const el = screen.getByTestId('late');
    expect(io.observed()).toContain(el);
    io.trigger(el, true);
    expect(shown()).toBe('visible');
  });

  it('stops observing the old element when the ref moves, and a removed one', async () => {
    const io = installIntersectionObserver();
    const user = userEvent.setup();
    render(<Switcher />);
    const first = screen.getByTestId('first');
    expect(io.observed()).toEqual([first]);
    await user.click(screen.getByRole('button', { name: 'second' }));
    const second = screen.getByTestId('second');
    expect(io.observed()).toEqual([second]);
    await user.click(screen.getByRole('button', { name: 'none' }));
    expect(io.observed()).toEqual([]);
  });

  it('observes nothing after unmount', () => {
    const io = installIntersectionObserver();
    const { unmount } = render(<Box />);
    expect(io.observed().length).toBe(1);
    unmount();
    expect(io.observed()).toEqual([]);
  });
});
