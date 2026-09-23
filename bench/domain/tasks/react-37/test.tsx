import { describe, it, expect } from 'vitest';
import { createRef } from 'react';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FancyInput, type FancyInputHandle } from './solution';

const box = (name: string) => screen.getByLabelText(name) as HTMLInputElement;
const length = () => screen.getByText(/^Length: \d+$/).textContent;

describe('FancyInput', () => {
  it('is a plain function component, not a forwardRef wrapper', () => {
    expect(typeof FancyInput).toBe('function');
  });

  it('renders a labelled text input with a live length counter', async () => {
    render(<FancyInput label="Nickname" />);
    expect(box('Nickname').value).toBe('');
    expect(length()).toBe('Length: 0');
    await userEvent.setup({ delay: null }).type(box('Nickname'), 'hello');
    expect(box('Nickname').value).toBe('hello');
    expect(length()).toBe('Length: 5');
  });

  it('exposes getValue() through the ref prop', async () => {
    const ref = createRef<FancyInputHandle>();
    render(<FancyInput label="City" ref={ref} />);
    expect(ref.current).not.toBe(null);
    expect(ref.current!.getValue()).toBe('');
    await userEvent.setup({ delay: null }).type(box('City'), 'Oslo');
    expect(ref.current!.getValue()).toBe('Oslo');
  });

  it('focus() moves focus to the input', () => {
    const ref = createRef<FancyInputHandle>();
    render(
      <>
        <button>elsewhere</button>
        <FancyInput label="City" ref={ref} />
      </>,
    );
    screen.getByRole('button', { name: 'elsewhere' }).focus();
    act(() => ref.current!.focus());
    expect(document.activeElement).toBe(box('City'));
  });

  it('clear() empties the input, the counter and getValue()', async () => {
    const ref = createRef<FancyInputHandle>();
    render(<FancyInput label="City" ref={ref} />);
    await userEvent.setup({ delay: null }).type(box('City'), 'Lima');
    act(() => ref.current!.clear());
    expect(box('City').value).toBe('');
    expect(length()).toBe('Length: 0');
    expect(ref.current!.getValue()).toBe('');
  });

  it('stays cleared across a parent re-render and keeps working afterwards', async () => {
    const ref = createRef<FancyInputHandle>();
    const { rerender } = render(<FancyInput label="City" ref={ref} />);
    const user = userEvent.setup({ delay: null });
    await user.type(box('City'), 'Lima');
    act(() => ref.current!.clear());
    rerender(<FancyInput label="Town" ref={ref} />);
    expect(box('Town').value).toBe('');
    expect(length()).toBe('Length: 0');
    await user.type(box('Town'), 'Rome');
    expect(box('Town').value).toBe('Rome');
    expect(length()).toBe('Length: 4');
    expect(ref.current!.getValue()).toBe('Rome');
  });

  it('works with a callback ref, which receives null on unmount', () => {
    const seen: (FancyInputHandle | null)[] = [];
    const { unmount } = render(
      <FancyInput
        label="City"
        ref={(h) => {
          seen.push(h);
        }}
      />,
    );
    const handle = seen[seen.length - 1];
    expect(handle).not.toBe(null);
    expect(typeof handle!.focus).toBe('function');
    expect(typeof handle!.clear).toBe('function');
    expect(handle!.getValue()).toBe('');
    unmount();
    expect(seen[seen.length - 1]).toBe(null);
  });
});
