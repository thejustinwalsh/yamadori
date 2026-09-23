import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SignupForm } from './solution';

const EMAIL_MSG = 'Enter a valid email';
const PW_MSG = 'Password must be at least 8 characters';
const MATCH_MSG = 'Passwords do not match';

function setup() {
  const onSubmit = vi.fn<(values: { email: string; password: string }) => void>();
  const user = userEvent.setup({ delay: null });
  render(<SignupForm onSubmit={onSubmit} />);
  return {
    onSubmit,
    user,
    email: screen.getByLabelText('Email') as HTMLInputElement,
    password: screen.getByLabelText('Password') as HTMLInputElement,
    confirm: screen.getByLabelText('Confirm password') as HTMLInputElement,
    submit: screen.getByRole('button', { name: 'Sign up' }),
  };
}

const shown = () => [EMAIL_MSG, PW_MSG, MATCH_MSG].filter((m) => screen.queryByText(m) !== null);

function expectError(input: HTMLElement, msg: string) {
  expect(input.getAttribute('aria-invalid')).toBe('true');
  const ids = (input.getAttribute('aria-describedby') ?? '').split(/\s+/).filter(Boolean);
  const texts = ids.map((id) => document.getElementById(id)?.textContent ?? '');
  expect(texts.some((t) => t.includes(msg))).toBe(true);
}

function expectClean(input: HTMLElement) {
  expect(input.getAttribute('aria-invalid')).not.toBe('true');
}

describe('SignupForm', () => {
  it('shows no message while the user is still typing in an untouched field', async () => {
    const { user, email, password, confirm } = setup();
    await user.type(email, 'nope');
    expect(email.value).toBe('nope');
    expect(shown()).toEqual([]);
    expectClean(email);
    expectClean(password);
    expectClean(confirm);
  });

  it('shows a field message after blur, wired with aria-invalid / aria-describedby', async () => {
    const { user, email, password } = setup();
    await user.type(email, 'nope');
    await user.click(password);
    expect(shown()).toEqual([EMAIL_MSG]);
    expectError(email, EMAIL_MSG);
    expectClean(password);
  });

  it('clears the message as soon as the field is fixed', async () => {
    const { user, email, password } = setup();
    await user.type(email, 'ana@example');
    await user.click(password);
    expect(shown()).toEqual([EMAIL_MSG]);
    await user.type(email, '.com');
    // clicking back into Email blurred Password, which is now (rightly) flagged
    expect(shown()).toEqual([PW_MSG]);
    expectClean(email);
  });

  it('a failed submit shows every invalid message and does not call onSubmit', async () => {
    const { user, email, password, confirm, submit, onSubmit } = setup();
    await user.type(email, 'bad');
    await user.type(password, 'short');
    await user.type(confirm, 'other');
    // typing moved focus through the first two fields; the submit is what reveals all three
    await user.click(submit);
    expect(onSubmit).not.toHaveBeenCalled();
    expect(shown()).toEqual([EMAIL_MSG, PW_MSG, MATCH_MSG]);
    expectError(email, EMAIL_MSG);
    expectError(password, PW_MSG);
    expectError(confirm, MATCH_MSG);
  });

  it('never submits while a field is invalid', async () => {
    const { user, email, password, confirm, submit, onSubmit } = setup();
    await user.click(submit);
    expect(onSubmit).not.toHaveBeenCalled();
    await user.type(email, 'ana@example.com');
    await user.type(password, 'correct horse');
    await user.type(confirm, 'correct hors');
    await user.click(submit);
    expect(onSubmit).not.toHaveBeenCalled();
    expect(shown()).toEqual([MATCH_MSG]);
  });

  it('re-checks Confirm password when Password changes', async () => {
    const { user, password, confirm, email } = setup();
    await user.type(email, 'ana@example.com');
    await user.type(password, 'longenough1');
    await user.type(confirm, 'longenough1');
    await user.click(email);
    expect(shown()).toEqual([]);
    await user.type(password, 'x');
    expect(shown()).toEqual([MATCH_MSG]);
    expectError(confirm, MATCH_MSG);
    await user.type(password, '{Backspace}');
    expect(shown()).toEqual([]);
    expectClean(confirm);
  });

  it('submits the values once when everything is valid', async () => {
    const { user, email, password, confirm, submit, onSubmit } = setup();
    await user.type(email, 'ana@example.com');
    await user.type(password, 'correct horse');
    await user.type(confirm, 'correct horse');
    await user.click(submit);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({ email: 'ana@example.com', password: 'correct horse' });
    expect(shown()).toEqual([]);
  });
});
