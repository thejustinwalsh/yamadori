import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useSet } from './solution';

let seen: ReadonlySet<string>[] = [];
const latest = () => seen[seen.length - 1];
const contents = (s: ReadonlySet<string>) => [...s].sort().join(',');

function Tags({ initial }: { initial?: Iterable<string> }) {
  const s = useSet<string>(initial);
  const values: ReadonlySet<string> = s.values;
  seen.push(values);
  const hasX: boolean = s.has('x');
  return (
    <div>
      <output aria-label="values">{contents(values)}</output>
      <p>{hasX ? 'has x' : 'no x'}</p>
      <button onClick={() => s.add('x')}>add x</button>
      <button onClick={() => s.add('a')}>add a</button>
      <button onClick={() => s.remove('x')}>remove x</button>
      <button onClick={() => s.remove('zzz')}>remove zzz</button>
      <button onClick={() => s.toggle('x')}>toggle x</button>
      <button onClick={() => s.clear()}>clear</button>
      <button
        onClick={() => {
          s.add('p');
          s.add('q');
          s.toggle('a');
        }}
      >
        batch
      </button>
    </div>
  );
}

function NumberProbe() {
  const s = useSet<number>([3, 1, 2]);
  const total = [...s.values].reduce((a, b) => a + b, 0);
  return <output aria-label="total">{total}</output>;
}

const shown = () => screen.getByRole('status', { name: 'values' }).textContent;

function setup(initial?: Iterable<string>) {
  seen = [];
  const user = userEvent.setup({ delay: null });
  render(<Tags initial={initial} />);
  const click = (name: string) => user.click(screen.getByRole('button', { name }));
  return { click };
}

describe('useSet', () => {
  it('starts from the initial items, or empty', () => {
    setup(['b', 'a']);
    expect(shown()).toBe('a,b');
    render(<NumberProbe />);
    expect(screen.getByRole('status', { name: 'total' }).textContent).toBe('6');
  });

  it('starts empty without an initial iterable', () => {
    setup();
    expect(shown()).toBe('');
    expect(latest().size).toBe(0);
  });

  it('adds, removes, toggles and clears, and has() follows', async () => {
    const { click } = setup(['a']);
    expect(screen.getByText('no x')).toBeTruthy();
    await click('add x');
    expect(shown()).toBe('a,x');
    expect(screen.getByText('has x')).toBeTruthy();
    await click('remove x');
    expect(shown()).toBe('a');
    expect(screen.getByText('no x')).toBeTruthy();
    await click('toggle x');
    expect(shown()).toBe('a,x');
    await click('toggle x');
    expect(shown()).toBe('a');
    await click('clear');
    expect(shown()).toBe('');
  });

  it('never modifies a Set it has already returned, nor the initial iterable', async () => {
    const init = new Set(['a']);
    const { click } = setup(init);
    const first = latest();
    await click('add x');
    const second = latest();
    expect(second).not.toBe(first);
    expect(contents(first)).toBe('a');
    await click('remove x');
    await click('clear');
    expect(contents(first)).toBe('a');
    expect(contents(second)).toBe('a,x');
    expect(contents(latest())).toBe('');
    expect([...init]).toEqual(['a']);
  });

  it('keeps the same Set when a call changes nothing', async () => {
    const { click } = setup(['a']);
    const before = latest();
    await click('add a');
    await click('remove zzz');
    expect(latest()).toBe(before);
    await click('clear');
    const empty = latest();
    expect(empty).not.toBe(before);
    await click('clear');
    expect(latest()).toBe(empty);
  });

  it('applies every call made in the same handler', async () => {
    const { click } = setup(['a']);
    await click('batch');
    expect(shown()).toBe('p,q');
    await click('batch');
    expect(shown()).toBe('a,p,q');
  });
});
