import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ContactManager } from './solution';

const PEOPLE = [
  { id: 'c1', name: 'Ann', email: 'ann@example.com' },
  { id: 'c2', name: 'Bo', email: 'bo@example.com' },
  { id: 'c3', name: 'Cy', email: 'cy@example.com' },
];

const TWINS = [
  { id: 't1', name: 'Sam', email: 'sam.one@example.com' },
  { id: 't2', name: 'Sam', email: 'sam.two@example.com' },
];

const nameBox = () => screen.getByLabelText('Name') as HTMLInputElement;
const emailBox = () => screen.getByLabelText('Email') as HTMLInputElement;
const editor = () => [nameBox().value, emailBox().value];
const pick = (user: ReturnType<typeof userEvent.setup>, label: string) =>
  user.click(screen.getByRole('button', { name: label }));
const save = (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole('button', { name: 'Save' }));

async function retype(user: ReturnType<typeof userEvent.setup>, el: HTMLInputElement, text: string) {
  await user.clear(el);
  await user.type(el, text);
}

describe('ContactManager', () => {
  it('lists every contact and starts on the first one', () => {
    render(<ContactManager contacts={PEOPLE} />);
    expect(screen.getByRole('button', { name: 'Ann (ann@example.com)' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Bo (bo@example.com)' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Cy (cy@example.com)' })).toBeTruthy();
    expect(editor()).toEqual(['Ann', 'ann@example.com']);
  });

  it('shows the saved values of whichever contact is selected', async () => {
    render(<ContactManager contacts={PEOPLE} />);
    const user = userEvent.setup({ delay: null });
    await pick(user, 'Bo (bo@example.com)');
    expect(editor()).toEqual(['Bo', 'bo@example.com']);
    await pick(user, 'Cy (cy@example.com)');
    expect(editor()).toEqual(['Cy', 'cy@example.com']);
    await pick(user, 'Ann (ann@example.com)');
    expect(editor()).toEqual(['Ann', 'ann@example.com']);
  });

  it('Save stores the draft and the list shows the saved values', async () => {
    render(<ContactManager contacts={PEOPLE} />);
    const user = userEvent.setup({ delay: null });
    await pick(user, 'Bo (bo@example.com)');
    await retype(user, nameBox(), 'Bob');
    await retype(user, emailBox(), 'bob@example.com');
    expect(screen.queryByRole('button', { name: 'Bob (bob@example.com)' })).toBe(null);
    await save(user);
    expect(screen.getByRole('button', { name: 'Bob (bob@example.com)' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Bo (bo@example.com)' })).toBe(null);
    expect(editor()).toEqual(['Bob', 'bob@example.com']);
    await pick(user, 'Ann (ann@example.com)');
    await pick(user, 'Bob (bob@example.com)');
    expect(editor()).toEqual(['Bob', 'bob@example.com']);
  });

  it('switching contacts discards the unsaved draft', async () => {
    render(<ContactManager contacts={PEOPLE} />);
    const user = userEvent.setup({ delay: null });
    await retype(user, nameBox(), 'Annabel');
    await pick(user, 'Cy (cy@example.com)');
    expect(editor()).toEqual(['Cy', 'cy@example.com']);
    await pick(user, 'Ann (ann@example.com)');
    expect(editor()).toEqual(['Ann', 'ann@example.com']);
    expect(screen.getByRole('button', { name: 'Ann (ann@example.com)' })).toBeTruthy();
  });

  it('keeps other contacts saved while another is being edited', async () => {
    render(<ContactManager contacts={PEOPLE} />);
    const user = userEvent.setup({ delay: null });
    await retype(user, emailBox(), 'ann@new.example.com');
    await save(user);
    await pick(user, 'Cy (cy@example.com)');
    await retype(user, nameBox(), 'Cyrus');
    await pick(user, 'Ann (ann@new.example.com)');
    expect(editor()).toEqual(['Ann', 'ann@new.example.com']);
    await pick(user, 'Cy (cy@example.com)');
    expect(editor()).toEqual(['Cy', 'cy@example.com']);
  });

  it('tells apart two contacts with the same name', async () => {
    render(<ContactManager contacts={TWINS} />);
    const user = userEvent.setup({ delay: null });
    expect(editor()).toEqual(['Sam', 'sam.one@example.com']);
    await pick(user, 'Sam (sam.two@example.com)');
    expect(editor()).toEqual(['Sam', 'sam.two@example.com']);
    await retype(user, emailBox(), 'draft@example.com');
    await pick(user, 'Sam (sam.one@example.com)');
    expect(editor()).toEqual(['Sam', 'sam.one@example.com']);
  });

  it('saves to the right one of two same-named contacts', async () => {
    render(<ContactManager contacts={TWINS} />);
    const user = userEvent.setup({ delay: null });
    await pick(user, 'Sam (sam.two@example.com)');
    await retype(user, emailBox(), 'sam.2@example.com');
    await save(user);
    expect(screen.getByRole('button', { name: 'Sam (sam.one@example.com)' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Sam (sam.2@example.com)' })).toBeTruthy();
    await pick(user, 'Sam (sam.one@example.com)');
    expect(editor()).toEqual(['Sam', 'sam.one@example.com']);
  });
});
