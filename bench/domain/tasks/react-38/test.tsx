import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CounterProvider, useCount, useCountActions } from './solution';

let controlRenders = 0;
let seenActions: object[] = [];

function CountView({ label = 'count' }: { label?: string }) {
  const n: number = useCount();
  return <output aria-label={label}>{n}</output>;
}

// Uses only the actions: it must not re-render when the count changes.
function Controls({ prefix = '' }: { prefix?: string }) {
  controlRenders += 1;
  const { increment, decrement, reset } = useCountActions();
  return (
    <div>
      <button onClick={() => increment()}>{prefix}Increment</button>
      <button onClick={() => decrement()}>{prefix}Decrement</button>
      <button onClick={() => reset()}>{prefix}Reset</button>
    </div>
  );
}

// Reads both, so it re-renders on every change and records the actions
// object it received each time.
function Both() {
  const n = useCount();
  const actions = useCountActions();
  seenActions.push(actions);
  return <span>both {n}</span>;
}

const count = (label = 'count') => screen.getByLabelText(label).textContent;
const click = (user: ReturnType<typeof userEvent.setup>, name: string) =>
  user.click(screen.getByRole('button', { name }));

describe('CounterProvider', () => {
  beforeEach(() => {
    controlRenders = 0;
    seenActions = [];
  });

  it('starts at 0 and increments, decrements and resets', async () => {
    render(
      <CounterProvider>
        <CountView />
        <Controls />
      </CounterProvider>,
    );
    const user = userEvent.setup({ delay: null });
    expect(count()).toBe('0');
    await click(user, 'Increment');
    await click(user, 'Increment');
    await click(user, 'Increment');
    expect(count()).toBe('3');
    await click(user, 'Decrement');
    expect(count()).toBe('2');
    await click(user, 'Reset');
    expect(count()).toBe('0');
    await click(user, 'Decrement');
    expect(count()).toBe('-1');
  });

  it('never re-renders a component that only uses the actions', async () => {
    render(
      <CounterProvider>
        <CountView />
        <Controls />
      </CounterProvider>,
    );
    const user = userEvent.setup({ delay: null });
    expect(controlRenders).toBe(1);
    await click(user, 'Increment');
    await click(user, 'Increment');
    await click(user, 'Decrement');
    await click(user, 'Reset');
    expect(count()).toBe('0');
    await click(user, 'Increment');
    expect(count()).toBe('1');
    expect(controlRenders).toBe(1);
  });

  it('hands out the same actions object for the whole lifetime', async () => {
    render(
      <CounterProvider>
        <Both />
        <Controls />
      </CounterProvider>,
    );
    const user = userEvent.setup({ delay: null });
    await click(user, 'Increment');
    await click(user, 'Increment');
    await click(user, 'Reset');
    expect(screen.getByText('both 0')).toBeTruthy();
    expect(seenActions.length).toBeGreaterThanOrEqual(4);
    expect(new Set(seenActions).size).toBe(1);
  });

  it('keeps separate providers independent', async () => {
    render(
      <>
        <CounterProvider>
          <CountView label="first" />
          <Controls prefix="first " />
        </CounterProvider>
        <CounterProvider>
          <CountView label="second" />
          <Controls prefix="second " />
        </CounterProvider>
      </>,
    );
    const user = userEvent.setup({ delay: null });
    await click(user, 'first Increment');
    await click(user, 'first Increment');
    await click(user, 'second Decrement');
    expect(count('first')).toBe('2');
    expect(count('second')).toBe('-1');
  });

  it('useCount throws outside a CounterProvider', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(() => render(<CountView />)).toThrow(Error);
  });

  it('useCountActions throws outside a CounterProvider', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(() => render(<Controls />)).toThrow(Error);
  });
});
