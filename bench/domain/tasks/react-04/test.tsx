import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderToString } from 'react-dom/server';
import { useLocalStorage } from './solution';

let seen: number[] = [];

function Counter({ initial = 0 }: { initial?: number }) {
  const [count, setCount, reset]: [
    number,
    (next: number | ((prev: number) => number)) => void,
    () => void,
  ] = useLocalStorage('count', initial);
  seen.push(count);
  return (
    <div>
      <output>{String(count)}</output>
      <button onClick={() => setCount(10)}>set ten</button>
      <button
        onClick={() => {
          setCount((c) => c + 1);
          setCount((c) => c + 1);
        }}
      >
        add two
      </button>
      <button onClick={reset}>reset</button>
    </div>
  );
}

function UserBox() {
  const [user, setUser] = useLocalStorage<{ name: string }>('user', { name: 'anon' });
  return (
    <div>
      <output>{user.name}</output>
      <button onClick={() => setUser({ name: 'ada' })}>rename</button>
    </div>
  );
}

const shown = () => screen.getByRole('status').textContent;

function storageEvent(key: string, newValue: string | null) {
  act(() => {
    window.dispatchEvent(new StorageEvent('storage', { key, newValue }));
  });
}

describe('useLocalStorage', () => {
  beforeEach(() => {
    seen = [];
  });

  it('uses initialValue when nothing is stored', () => {
    render(<Counter initial={3} />);
    expect(shown()).toBe('3');
  });

  it('returns the stored value on the very first render', () => {
    localStorage.setItem('count', '5');
    render(<Counter />);
    expect(seen[0]).toBe(5);
    expect(shown()).toBe('5');
  });

  it('stores values as JSON under the key', async () => {
    const user = userEvent.setup();
    render(<UserBox />);
    expect(shown()).toBe('anon');
    await user.click(screen.getByRole('button', { name: 'rename' }));
    expect(shown()).toBe('ada');
    expect(JSON.parse(localStorage.getItem('user') ?? 'null')).toEqual({ name: 'ada' });
  });

  it('gives functional updates the latest value, even twice in one handler', async () => {
    const user = userEvent.setup();
    render(<Counter />);
    await user.click(screen.getByRole('button', { name: 'set ten' }));
    await user.click(screen.getByRole('button', { name: 'add two' }));
    expect(shown()).toBe('12');
    expect(localStorage.getItem('count')).toBe('12');
  });

  it('the third element removes the key and returns to initialValue', async () => {
    const user = userEvent.setup();
    render(<Counter initial={1} />);
    await user.click(screen.getByRole('button', { name: 'set ten' }));
    await user.click(screen.getByRole('button', { name: 'reset' }));
    expect(shown()).toBe('1');
    expect(localStorage.getItem('count')).toBeNull();
  });

  it('falls back to initialValue when the stored text is not JSON', () => {
    localStorage.setItem('count', '{not json');
    render(<Counter initial={4} />);
    expect(shown()).toBe('4');
  });

  it('falls back to initialValue when localStorage access throws', () => {
    localStorage.setItem('count', '5');
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('The operation is insecure.', 'SecurityError');
    });
    render(<Counter initial={6} />);
    expect(shown()).toBe('6');
  });

  it('renders initialValue on the server, where window does not exist', () => {
    vi.stubGlobal('window', undefined);
    vi.stubGlobal('localStorage', undefined);
    const html = renderToString(<Counter initial={8} />);
    expect(html).toContain('<output>8</output>');
  });

  it('follows storage events for the same key from other tabs', () => {
    render(<Counter initial={2} />);
    localStorage.setItem('count', '42');
    storageEvent('count', '42');
    expect(shown()).toBe('42');
    localStorage.setItem('other', '99');
    storageEvent('other', '99');
    expect(shown()).toBe('42');
    localStorage.removeItem('count');
    storageEvent('count', null);
    expect(shown()).toBe('2');
  });
});
