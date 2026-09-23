import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState, type ErrorInfo, type ReactNode } from 'react';
import { ErrorBoundary } from './solution';

const ctl = { throwing: true, thrown: [] as Error[] };

function Bomb() {
  if (ctl.throwing) {
    const e = new Error(`boom ${ctl.thrown.length + 1}`);
    ctl.thrown.push(e);
    throw e;
  }
  return <p>all good</p>;
}

function Counter() {
  const [n, setN] = useState(0);
  return <button onClick={() => setN(n + 1)}>count {n}</button>;
}

const fallback = (error: Error, reset: () => void): ReactNode => (
  <div role="alert">
    <p>{error.message}</p>
    <button onClick={reset}>Try again</button>
  </div>
);

function setup(keys?: readonly unknown[], child: ReactNode = <Bomb />) {
  const onError = vi.fn<(error: Error, info: ErrorInfo) => void>();
  const ui = (k?: readonly unknown[]) => (
    <ErrorBoundary fallback={fallback} onError={onError} resetKeys={k}>
      {child}
    </ErrorBoundary>
  );
  const { rerender } = render(ui(keys));
  return { onError, setKeys: (k: readonly unknown[]) => rerender(ui(k)) };
}

const alertText = () => screen.getByRole('alert').querySelector('p')?.textContent;

describe('ErrorBoundary', () => {
  beforeEach(() => {
    ctl.throwing = true;
    ctl.thrown = [];
    // React reports every caught error on console.error; keep output readable.
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
  });

  it('renders its children when nothing throws', () => {
    ctl.throwing = false;
    const { onError } = setup();
    expect(screen.getByText('all good')).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(onError).not.toHaveBeenCalled();
  });

  it('renders the fallback with the thrown error and reports it once with React error info', () => {
    const { onError } = setup();
    expect(screen.queryByText('all good')).toBeNull();
    const last = ctl.thrown[ctl.thrown.length - 1];
    expect(alertText()).toBe(last.message);
    expect(onError).toHaveBeenCalledTimes(1);
    const [error, info] = onError.mock.calls[0];
    expect(error).toBe(last);
    expect(typeof info.componentStack).toBe('string');
    expect(info.componentStack).toContain('Bomb');
  });

  it('reset() renders the children again once the cause is fixed', async () => {
    const user = userEvent.setup({ delay: null });
    const { onError } = setup();
    ctl.throwing = false;
    await user.click(screen.getByRole('button', { name: 'Try again' }));
    expect(screen.getByText('all good')).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it('reset() while still broken catches the new error and reports it again', async () => {
    const user = userEvent.setup({ delay: null });
    const { onError } = setup();
    const first = ctl.thrown.length;
    await user.click(screen.getByRole('button', { name: 'Try again' }));
    expect(ctl.thrown.length).toBeGreaterThan(first);
    expect(alertText()).toBe(ctl.thrown[ctl.thrown.length - 1].message);
    expect(onError).toHaveBeenCalledTimes(2);
    expect(onError.mock.calls[1][0]).toBe(ctl.thrown[ctl.thrown.length - 1]);
  });

  it('a changed resetKeys element resets automatically', () => {
    const { onError, setKeys } = setup([1, 'a']);
    ctl.throwing = false;
    setKeys([1, 'b']);
    expect(screen.getByText('all good')).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it('a change in resetKeys length resets automatically', () => {
    const { setKeys } = setup([1]);
    ctl.throwing = false;
    setKeys([1, 2]);
    expect(screen.getByText('all good')).toBeTruthy();
  });

  it('a new array with the same elements is not a change', () => {
    const { onError, setKeys } = setup([1, 'a']);
    ctl.throwing = false;
    setKeys([1, 'a']);
    expect(screen.getByRole('alert')).toBeTruthy();
    expect(screen.queryByText('all good')).toBeNull();
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it('a resetKeys change without an error leaves the children and their state alone', async () => {
    const user = userEvent.setup({ delay: null });
    const { onError, setKeys } = setup([1], <Counter />);
    await user.click(screen.getByRole('button', { name: 'count 0' }));
    await user.click(screen.getByRole('button', { name: 'count 1' }));
    setKeys([2]);
    expect(screen.getByRole('button', { name: 'count 2' })).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(onError).not.toHaveBeenCalled();
  });
});
