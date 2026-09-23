import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Combobox } from './solution';

const FRUIT = ['Apple', 'Banana', 'Grape', 'Apricot', 'Cherry'];

function setup() {
  const onSelect = vi.fn<(value: string) => void>();
  const user = userEvent.setup({ delay: null });
  render(<Combobox label="Fruit" options={FRUIT} onSelect={onSelect} />);
  const input = screen.getByRole('combobox', { name: 'Fruit' }) as HTMLInputElement;
  return { onSelect, user, input };
}

const optionTexts = () => screen.queryAllByRole('option').map((o) => o.textContent);

function activeText(input: HTMLElement): string | null {
  const on = screen.queryAllByRole('option').filter((o) => o.getAttribute('aria-selected') === 'true');
  const ad = input.getAttribute('aria-activedescendant') ?? '';
  if (on.length === 0) {
    expect(ad).toBe('');
    return null;
  }
  expect(on.length).toBe(1);
  for (const o of screen.getAllByRole('option')) {
    if (o !== on[0]) expect(o.getAttribute('aria-selected')).toBe('false');
  }
  expect(on[0].id).not.toBe('');
  expect(ad).toBe(on[0].id);
  expect(document.getElementById(ad)).toBe(on[0]);
  return on[0].textContent;
}

describe('Combobox', () => {
  it('starts closed with the combobox attributes', () => {
    const { input } = setup();
    expect(input.tagName).toBe('INPUT');
    expect(input.getAttribute('aria-autocomplete')).toBe('list');
    expect(input.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('listbox')).toBeNull();
  });

  it('typing opens a listbox filtered case-insensitively', async () => {
    const { user, input } = setup();
    await user.type(input, 'ap');
    expect(input.getAttribute('aria-expanded')).toBe('true');
    const listbox = screen.getByRole('listbox');
    expect(listbox.id).not.toBe('');
    expect(input.getAttribute('aria-controls')).toBe(listbox.id);
    expect(optionTexts()).toEqual(['Apple', 'Grape', 'Apricot']);
    await user.type(input, 'R');
    expect(optionTexts()).toEqual(['Apricot']);
    expect(activeText(input)).toBeNull();
  });

  it('shows "No results" when nothing matches', async () => {
    const { user, input } = setup();
    await user.type(input, 'kiwi');
    expect(screen.getByText('No results')).toBeTruthy();
    expect(optionTexts()).toEqual([]);
  });

  it('ArrowDown / ArrowUp move the active option with wrap-around', async () => {
    const { user, input } = setup();
    await user.type(input, 'an');
    expect(optionTexts()).toEqual(['Banana']);
    await user.clear(input);
    await user.type(input, 'r');
    expect(optionTexts()).toEqual(['Grape', 'Apricot', 'Cherry']);
    expect(activeText(input)).toBeNull();
    await user.keyboard('{ArrowDown}');
    expect(activeText(input)).toBe('Grape');
    await user.keyboard('{ArrowDown}{ArrowDown}');
    expect(activeText(input)).toBe('Cherry');
    await user.keyboard('{ArrowDown}');
    expect(activeText(input)).toBe('Grape');
    await user.keyboard('{ArrowUp}');
    expect(activeText(input)).toBe('Cherry');
    expect(document.activeElement).toBe(input);
  });

  it('ArrowUp with no active option activates the last option', async () => {
    const { user, input } = setup();
    await user.type(input, 'a');
    await user.keyboard('{ArrowUp}');
    expect(activeText(input)).toBe('Apricot');
  });

  it('Enter picks the active option, closes the list and calls onSelect', async () => {
    const { user, input, onSelect } = setup();
    await user.type(input, 'AP');
    await user.keyboard('{ArrowDown}{ArrowDown}{Enter}');
    expect(input.value).toBe('Grape');
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith('Grape');
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(input.getAttribute('aria-expanded')).toBe('false');
  });

  it('Enter with no active option does nothing', async () => {
    const { user, input, onSelect } = setup();
    await user.type(input, 'ch{Enter}');
    expect(onSelect).not.toHaveBeenCalled();
    expect(input.value).toBe('ch');
    expect(optionTexts()).toEqual(['Cherry']);
  });

  it('clicking an option picks it', async () => {
    const { user, input, onSelect } = setup();
    await user.type(input, 'b');
    await user.click(screen.getByRole('option', { name: 'Banana' }));
    expect(input.value).toBe('Banana');
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith('Banana');
    expect(screen.queryByRole('listbox')).toBeNull();
  });

  it('Escape closes the list without selecting', async () => {
    const { user, input, onSelect } = setup();
    await user.type(input, 'ap');
    await user.keyboard('{ArrowDown}{Escape}');
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(input.getAttribute('aria-expanded')).toBe('false');
    expect(input.value).toBe('ap');
    expect(onSelect).not.toHaveBeenCalled();
  });
});
