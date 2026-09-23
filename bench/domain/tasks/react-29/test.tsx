import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { OtpInput } from './solution';

function setup(length = 4) {
  const onComplete = vi.fn<(code: string) => void>();
  render(<OtpInput length={length} onComplete={onComplete} />);
  const user = userEvent.setup({ delay: null });
  const box = (n: number) => screen.getByLabelText(`Digit ${n} of ${length}`) as HTMLInputElement;
  const values = () => Array.from({ length }, (_, i) => box(i + 1).value).join('|');
  const focused = () => Array.from({ length }, (_, i) => box(i + 1)).indexOf(document.activeElement as HTMLInputElement) + 1;
  return { onComplete, user, box, values, focused };
}

describe('OtpInput', () => {
  it('renders `length` empty numeric text boxes labelled "Digit i of N"', () => {
    const { box } = setup(5);
    expect(screen.getAllByRole('textbox').length).toBe(5);
    for (let i = 1; i <= 5; i++) {
      expect(screen.getByRole('textbox', { name: `Digit ${i} of 5` })).toBe(box(i));
      expect(box(i).value).toBe('');
      expect(box(i).getAttribute('inputmode')).toBe('numeric');
    }
  });

  it('moves focus forward as digits are typed and reports the code once', async () => {
    const { user, box, values, focused, onComplete } = setup(4);
    await user.click(box(1));
    await user.keyboard('12');
    expect(values()).toBe('1|2||');
    expect(focused()).toBe(3);
    expect(onComplete).not.toHaveBeenCalled();
    await user.keyboard('34');
    expect(values()).toBe('1|2|3|4');
    expect(focused()).toBe(4);
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenCalledWith('1234');
  });

  it('ignores anything that is not a digit', async () => {
    const { user, box, values, focused } = setup(4);
    await user.click(box(1));
    await user.keyboard('a');
    expect(values()).toBe('|||');
    expect(focused()).toBe(1);
    await user.keyboard('7');
    await user.keyboard('x- ');
    expect(values()).toBe('7|||');
    expect(focused()).toBe(2);
  });

  it('Backspace in an empty box clears the previous box and moves back to it', async () => {
    const { user, box, values, focused } = setup(4);
    await user.click(box(1));
    await user.keyboard('12');
    expect(focused()).toBe(3);
    await user.keyboard('{Backspace}');
    expect(values()).toBe('1|||');
    expect(focused()).toBe(2);
    await user.keyboard('{Backspace}');
    expect(values()).toBe('|||');
    expect(focused()).toBe(1);
    await user.keyboard('{Backspace}');
    expect(values()).toBe('|||');
    expect(focused()).toBe(1);
  });

  it('pasting a full code fills every box and completes once', async () => {
    const { user, box, values, focused, onComplete } = setup(6);
    await user.click(box(1));
    await user.paste('482913');
    expect(values()).toBe('4|8|2|9|1|3');
    expect(focused()).toBe(6);
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenCalledWith('482913');
  });

  it('pasting into a middle box fills from there, skipping non-digits and dropping overflow', async () => {
    const { user, box, values, focused, onComplete } = setup(6);
    await user.click(box(3));
    await user.paste('9-8');
    expect(values()).toBe('||9|8||');
    expect(focused()).toBe(5);
    await user.click(box(5));
    await user.paste('7 6 5 4');
    expect(values()).toBe('||9|8|7|6');
    expect(focused()).toBe(6);
    expect(onComplete).not.toHaveBeenCalled();
  });

  it('calls onComplete again when a complete code is changed and completed again', async () => {
    const { user, box, values, focused, onComplete } = setup(4);
    await user.click(box(1));
    await user.keyboard('1234');
    expect(onComplete).toHaveBeenLastCalledWith('1234');
    await user.keyboard('{Backspace}');
    expect(values()).toBe('1|2|3|');
    expect(focused()).toBe(4);
    await user.keyboard('9');
    expect(onComplete).toHaveBeenCalledTimes(2);
    expect(onComplete).toHaveBeenLastCalledWith('1239');
    await user.click(box(2));
    await user.paste('55');
    expect(values()).toBe('1|5|5|9');
    expect(onComplete).toHaveBeenCalledTimes(3);
    expect(onComplete).toHaveBeenLastCalledWith('1559');
  });
});
