import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { useCopyToClipboard } from './solution';
import { advance } from './helpers';

let api!: { copied: boolean; copy: (text: string) => Promise<boolean> };

function Probe({ ms }: { ms?: number }) {
  const c = useCopyToClipboard(ms);
  api = c;
  const copied: boolean = c.copied;
  return <output>{copied ? 'copied' : 'idle'}</output>;
}

const shown = () => screen.getByRole('status').textContent;

let writeText: ReturnType<typeof vi.fn<(text: string) => Promise<void>>>;

beforeEach(() => {
  vi.useFakeTimers();
  writeText = vi.fn<(text: string) => Promise<void>>(() => Promise.resolve());
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText },
    configurable: true,
    writable: true,
  });
});

/** Call copy() inside act and wait for its promise to settle. */
async function copy(text: string): Promise<boolean> {
  let out: boolean | undefined;
  await act(async () => {
    out = await api.copy(text);
  });
  return out as boolean;
}

describe('useCopyToClipboard', () => {
  it('starts not copied and writes the text to the clipboard', async () => {
    render(<Probe />);
    expect(shown()).toBe('idle');
    const ok: boolean = await copy('hello');
    expect(ok).toBe(true);
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText).toHaveBeenCalledWith('hello');
    expect(shown()).toBe('copied');
  });

  it('resets after the default 2000 ms', async () => {
    render(<Probe />);
    await copy('a');
    advance(1999);
    expect(shown()).toBe('copied');
    advance(1);
    expect(shown()).toBe('idle');
  });

  it('honours a custom resetMs', async () => {
    render(<Probe ms={500} />);
    await copy('a');
    advance(499);
    expect(shown()).toBe('copied');
    advance(1);
    expect(shown()).toBe('idle');
  });

  it('restarts the countdown on a second successful copy', async () => {
    render(<Probe ms={1000} />);
    await copy('a');
    advance(700);
    await copy('b');
    advance(700);
    expect(shown()).toBe('copied');
    advance(299);
    expect(shown()).toBe('copied');
    advance(1);
    expect(shown()).toBe('idle');
  });

  it('resolves false and stays not copied when the write is rejected', async () => {
    writeText.mockImplementation(() => Promise.reject(new Error('denied')));
    render(<Probe />);
    const ok = await copy('secret');
    expect(ok).toBe(false);
    expect(writeText).toHaveBeenCalledWith('secret');
    expect(shown()).toBe('idle');
    advance(5000);
    expect(shown()).toBe('idle');
  });

  it('can copy again after the reset', async () => {
    render(<Probe ms={100} />);
    await copy('a');
    advance(100);
    expect(shown()).toBe('idle');
    expect(await copy('b')).toBe(true);
    expect(shown()).toBe('copied');
    expect(writeText).toHaveBeenCalledTimes(2);
  });

  it('leaves no timer pending after unmount', async () => {
    const { unmount } = render(<Probe />);
    await copy('a');
    await copy('b');
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
