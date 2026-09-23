import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useUndoRedo } from './solution';

function Editor({ values }: { values: string[] }) {
  const h = useUndoRedo<string>('a');
  const s: string = h.state;
  const flags: [boolean, boolean] = [h.canUndo, h.canRedo];
  return (
    <div>
      <output aria-label="state">{String(s)}</output>
      <output aria-label="flags">{`undo:${flags[0]} redo:${flags[1]}`}</output>
      {values.map((v) => (
        <button key={v} onClick={() => h.set(v)}>{`set ${v}`}</button>
      ))}
      <button onClick={() => h.undo()}>Undo</button>
      <button onClick={() => h.redo()}>Redo</button>
      <button onClick={() => h.reset('z')}>Reset</button>
    </div>
  );
}

type Box = { n: number };
const one: Box = { n: 1 };

function Boxes() {
  const h = useUndoRedo<Box>(one);
  return (
    <div>
      <output aria-label="state">{h.state === one ? 'original' : `copy ${h.state.n}`}</output>
      <output aria-label="flags">{`undo:${h.canUndo} redo:${h.canRedo}`}</output>
      <button onClick={() => h.set(one)}>same</button>
      <button onClick={() => h.set({ n: 1 })}>equal copy</button>
      <button onClick={() => h.undo()}>Undo</button>
    </div>
  );
}

const state = () => screen.getByRole('status', { name: 'state' }).textContent;
const flags = () => screen.getByRole('status', { name: 'flags' }).textContent;

function setup(values = ['b', 'c', 'd']) {
  const user = userEvent.setup({ delay: null });
  render(<Editor values={values} />);
  const click = (name: string) => user.click(screen.getByRole('button', { name }));
  return { click };
}

describe('useUndoRedo', () => {
  it('starts at the initial value with no history', () => {
    setup();
    expect(state()).toBe('a');
    expect(flags()).toBe('undo:false redo:false');
  });

  it('undoes and redoes through the whole history', async () => {
    const { click } = setup();
    await click('set b');
    await click('set c');
    expect(state()).toBe('c');
    expect(flags()).toBe('undo:true redo:false');
    await click('Undo');
    expect(state()).toBe('b');
    expect(flags()).toBe('undo:true redo:true');
    await click('Undo');
    expect(state()).toBe('a');
    expect(flags()).toBe('undo:false redo:true');
    await click('Redo');
    await click('Redo');
    expect(state()).toBe('c');
    expect(flags()).toBe('undo:true redo:false');
  });

  it('treats undo and redo at the ends of the history as no-ops', async () => {
    const { click } = setup();
    await click('Undo');
    expect(state()).toBe('a');
    expect(flags()).toBe('undo:false redo:false');
    await click('set b');
    await click('Undo');
    await click('Undo');
    expect(state()).toBe('a');
    expect(flags()).toBe('undo:false redo:true');
    await click('Redo');
    await click('Redo');
    expect(state()).toBe('b');
    expect(flags()).toBe('undo:true redo:false');
    await click('Undo');
    expect(state()).toBe('a');
  });

  it('set discards the redo history', async () => {
    const { click } = setup();
    await click('set b');
    await click('set c');
    await click('Undo');
    expect(flags()).toBe('undo:true redo:true');
    await click('set d');
    expect(state()).toBe('d');
    expect(flags()).toBe('undo:true redo:false');
    await click('Redo');
    expect(state()).toBe('d');
    await click('Undo');
    expect(state()).toBe('b');
    await click('Undo');
    expect(state()).toBe('a');
  });

  it('set with the current value does nothing, not even clear redo', async () => {
    const { click } = setup(['a', 'b']);
    await click('set b');
    await click('set b');
    await click('Undo');
    expect(state()).toBe('a');
    expect(flags()).toBe('undo:false redo:true');
    await click('set a');
    expect(flags()).toBe('undo:false redo:true');
    await click('Redo');
    expect(state()).toBe('b');
  });

  it('compares by Object.is only', async () => {
    const user = userEvent.setup({ delay: null });
    render(<Boxes />);
    await user.click(screen.getByRole('button', { name: 'same' }));
    expect(flags()).toBe('undo:false redo:false');
    await user.click(screen.getByRole('button', { name: 'equal copy' }));
    expect(state()).toBe('copy 1');
    expect(flags()).toBe('undo:true redo:false');
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(state()).toBe('original');
  });

  it('reset replaces the state and clears both histories', async () => {
    const { click } = setup();
    await click('set b');
    await click('set c');
    await click('Undo');
    await click('Reset');
    expect(state()).toBe('z');
    expect(flags()).toBe('undo:false redo:false');
    await click('Undo');
    await click('Redo');
    expect(state()).toBe('z');
    await click('set b');
    await click('Undo');
    expect(state()).toBe('z');
  });
});
