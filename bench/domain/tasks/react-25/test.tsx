import { describe, it, expect, vi } from 'vitest';
import { render, screen, act, within, fireEvent } from '@testing-library/react';
import { ToastProvider, useToast } from './solution';
import { advance } from './helpers';

// fireEvent rather than userEvent: RTL's async wrapper only advances *jest*
// fake timers, so userEvent under vitest fake timers never settles.

type Api = { show: (message: string) => number; dismiss: (id: number) => void };
let api: Api;

function Grab() {
  api = useToast();
  return <p>app</p>;
}

function mount(duration?: number) {
  vi.useFakeTimers();
  const r = render(
    duration === undefined ? (
      <ToastProvider>
        <Grab />
      </ToastProvider>
    ) : (
      <ToastProvider duration={duration}>
        <Grab />
      </ToastProvider>
    ),
  );
  return r;
}

const WORDS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon'];
const visible = () =>
  screen.queryAllByRole('status').map((s) => WORDS.find((w) => (s.textContent ?? '').includes(w)) ?? '?');

function show(message: string): number {
  let id = 0;
  act(() => {
    id = api.show(message);
  });
  return id;
}

function toastFor(message: string): HTMLElement {
  const el = screen.queryAllByRole('status').find((s) => (s.textContent ?? '').includes(message));
  if (!el) throw new Error(`no toast for ${message}`);
  return el;
}

describe('ToastProvider / useToast', () => {
  it('renders children and each toast as a status with a Dismiss button, in order', () => {
    mount(1000);
    expect(screen.getByText('app')).toBeTruthy();
    const a: number = show('alpha');
    const b = show('beta');
    expect(a).not.toBe(b);
    expect(visible()).toEqual(['alpha', 'beta']);
    expect(within(toastFor('alpha')).getByRole('button', { name: 'Dismiss' })).toBeTruthy();
  });

  it('expires each toast duration ms after its own show', () => {
    mount(3000);
    show('alpha');
    advance(1000);
    show('beta');
    advance(1999);
    expect(visible()).toEqual(['alpha', 'beta']);
    advance(1);
    expect(visible()).toEqual(['beta']);
    advance(999);
    expect(visible()).toEqual(['beta']);
    advance(1);
    expect(visible()).toEqual([]);
  });

  it('uses 3000 ms when no duration is given', () => {
    mount();
    show('alpha');
    advance(2999);
    expect(visible()).toEqual(['alpha']);
    advance(1);
    expect(visible()).toEqual([]);
  });

  it('removes a toast when its Dismiss button is clicked and leaves the others on schedule', () => {
    mount(2000);
    show('alpha');
    advance(500);
    show('beta');
    fireEvent.click(within(toastFor('alpha')).getByRole('button', { name: 'Dismiss' }));
    expect(visible()).toEqual(['beta']);
    advance(1500);
    expect(visible()).toEqual(['beta']);
    advance(499);
    expect(visible()).toEqual(['beta']);
    advance(1);
    expect(visible()).toEqual([]);
  });

  it('dismiss(id) removes that toast, ignores unknown ids, and leaves no timer behind', () => {
    mount(5000);
    const a = show('alpha');
    const b = show('beta');
    act(() => api.dismiss(a));
    expect(visible()).toEqual(['beta']);
    act(() => api.dismiss(a));
    act(() => api.dismiss(12345));
    expect(visible()).toEqual(['beta']);
    act(() => api.dismiss(b));
    expect(visible()).toEqual([]);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('shows at most 3 toasts, dropping the oldest', () => {
    mount(3000);
    show('alpha');
    advance(100);
    show('beta');
    advance(100);
    show('gamma');
    advance(100);
    show('delta');
    expect(visible()).toEqual(['beta', 'gamma', 'delta']);
    advance(2700);
    // alpha's deadline has passed: it must not take anything with it
    expect(visible()).toEqual(['beta', 'gamma', 'delta']);
    advance(99);
    expect(visible()).toEqual(['beta', 'gamma', 'delta']);
    advance(1);
    expect(visible()).toEqual(['gamma', 'delta']);
  });

  it('leaves no timer pending once the dropped and remaining toasts are gone', () => {
    mount(3000);
    for (const w of ['alpha', 'beta', 'gamma', 'delta', 'epsilon']) show(w);
    expect(visible()).toEqual(['gamma', 'delta', 'epsilon']);
    for (const w of ['gamma', 'delta', 'epsilon']) {
      fireEvent.click(within(toastFor(w)).getByRole('button', { name: 'Dismiss' }));
    }
    expect(visible()).toEqual([]);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('clears every timer on unmount', () => {
    const { unmount } = mount(3000);
    show('alpha');
    show('beta');
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
