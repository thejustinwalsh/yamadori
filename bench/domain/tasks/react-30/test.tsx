import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TodoApp } from './solution';

function setup() {
  render(<TodoApp />);
  return userEvent.setup({ delay: null });
}

const input = () => screen.getByLabelText('New todo') as HTMLInputElement;
const checkbox = (text: string) => screen.getByRole('checkbox', { name: text }) as HTMLInputElement;
const button = (name: string) => screen.getByRole('button', { name });
// The innermost element whose whole text is "N item(s) left" (markup inside it is allowed).
function counter(): string {
  const hits = Array.from(document.body.querySelectorAll('*')).filter((el) =>
    /^\d+ items? left$/.test((el.textContent ?? '').trim().replace(/\s+/g, ' ')),
  );
  if (hits.length === 0) throw new Error('no "N items left" element');
  return (hits[hits.length - 1].textContent ?? '').trim().replace(/\s+/g, ' ');
}

// The todo texts shown, in list order, read from each item's checkbox label.
function shown(): string[] {
  return screen.queryAllByRole('listitem').map((li) => {
    const del = within(li).getByRole('button', { name: /^Delete / });
    const name = del.getAttribute('aria-label') ?? del.textContent ?? '';
    return name.replace(/^Delete /, '');
  });
}

async function add(user: ReturnType<typeof userEvent.setup>, ...texts: string[]) {
  for (const t of texts) {
    await user.type(input(), t);
    await user.click(button('Add'));
  }
}

describe('TodoApp', () => {
  it('starts empty with All selected and 0 items left', () => {
    setup();
    expect(input().value).toBe('');
    expect(shown()).toEqual([]);
    expect(counter()).toBe('0 items left');
    expect(button('All').getAttribute('aria-pressed')).toBe('true');
    expect(button('Active').getAttribute('aria-pressed')).toBe('false');
    expect(button('Completed').getAttribute('aria-pressed')).toBe('false');
  });

  it('adds trimmed todos with the button or Enter, ignores blank input, and clears the field', async () => {
    const user = setup();
    await user.type(input(), '  Buy milk  ');
    await user.click(button('Add'));
    expect(input().value).toBe('');
    await user.type(input(), 'Walk dog{Enter}');
    expect(input().value).toBe('');
    await user.type(input(), '   ');
    await user.click(button('Add'));
    await user.click(button('Add'));
    expect(shown()).toEqual(['Buy milk', 'Walk dog']);
    expect(checkbox('Buy milk').checked).toBe(false);
    expect(counter()).toBe('2 items left');
  });

  it('counts active todos with correct pluralisation', async () => {
    const user = setup();
    await add(user, 'Alpha', 'Beta');
    expect(counter()).toBe('2 items left');
    await user.click(checkbox('Alpha'));
    expect(checkbox('Alpha').checked).toBe(true);
    expect(counter()).toBe('1 item left');
    await user.click(checkbox('Beta'));
    expect(counter()).toBe('0 items left');
    await user.click(checkbox('Alpha'));
    expect(counter()).toBe('1 item left');
  });

  it('deletes a todo with its Delete button', async () => {
    const user = setup();
    await add(user, 'Alpha', 'Beta', 'Gamma');
    await user.click(button('Delete Beta'));
    expect(shown()).toEqual(['Alpha', 'Gamma']);
    expect(counter()).toBe('2 items left');
  });

  it('keeps each checkbox with its own todo when an earlier todo is deleted', async () => {
    const user = setup();
    await add(user, 'Alpha', 'Beta', 'Gamma');
    await user.click(checkbox('Beta'));
    await user.click(button('Delete Alpha'));
    expect(shown()).toEqual(['Beta', 'Gamma']);
    expect(checkbox('Beta').checked).toBe(true);
    expect(checkbox('Gamma').checked).toBe(false);
    expect(counter()).toBe('1 item left');
  });

  it('filters by Active and Completed and marks the selected filter', async () => {
    const user = setup();
    await add(user, 'Alpha', 'Beta', 'Gamma');
    await user.click(checkbox('Beta'));
    await user.click(button('Active'));
    expect(button('Active').getAttribute('aria-pressed')).toBe('true');
    expect(button('All').getAttribute('aria-pressed')).toBe('false');
    expect(shown()).toEqual(['Alpha', 'Gamma']);
    await user.click(button('Completed'));
    expect(button('Completed').getAttribute('aria-pressed')).toBe('true');
    expect(button('Active').getAttribute('aria-pressed')).toBe('false');
    expect(shown()).toEqual(['Beta']);
    expect(checkbox('Beta').checked).toBe(true);
    expect(counter()).toBe('2 items left');
    await user.click(button('All'));
    expect(shown()).toEqual(['Alpha', 'Beta', 'Gamma']);
  });

  it('updates the filtered list when a todo changes state', async () => {
    const user = setup();
    await add(user, 'Alpha', 'Beta');
    await user.click(button('Active'));
    await user.click(checkbox('Alpha'));
    expect(shown()).toEqual(['Beta']);
    expect(counter()).toBe('1 item left');
    await user.click(button('Completed'));
    expect(shown()).toEqual(['Alpha']);
    expect(checkbox('Alpha').checked).toBe(true);
    await user.click(checkbox('Alpha'));
    expect(shown()).toEqual([]);
    expect(counter()).toBe('2 items left');
  });
});
