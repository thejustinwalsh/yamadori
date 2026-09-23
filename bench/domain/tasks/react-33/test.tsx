import { describe, it, expect, afterEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SubmitButton } from './solution';
import { deferred, settle, flush, type Deferred } from './helpers';

// React 19 entangles every async action in flight: one left unresolved keeps
// later ones pending too. Settle whatever a test left open.
let open: Deferred<void>[] = [];
afterEach(async () => {
  const left = open;
  open = [];
  await act(async () => {
    for (const d of left) d.resolve();
    await Promise.allSettled(left.map((d) => d.promise));
  });
});

/** An async form action owned by the test: each call waits on its own deferred. */
function formAction() {
  const pending: Deferred<void>[] = [];
  const action = async (_data: FormData): Promise<void> => {
    const d = deferred<void>();
    pending.push(d);
    open.push(d);
    await d.promise;
  };
  return { pending, action };
}

async function finish(d: Deferred<void>) {
  await settle(d, undefined);
  await flush();
}

const button = (name: string) => screen.getByRole('button', { name }) as HTMLButtonElement;

describe('SubmitButton', () => {
  it('renders an enabled submit button showing its children', () => {
    const { action } = formAction();
    render(
      <form action={action}>
        <SubmitButton pendingText="Saving…">Save</SubmitButton>
      </form>,
    );
    const b = button('Save');
    expect(b.type).toBe('submit');
    expect(b.disabled).toBe(false);
  });

  it('is disabled and shows pendingText while the form action runs, then recovers', async () => {
    const user = userEvent.setup({ delay: null });
    const { pending, action } = formAction();
    render(
      <form action={action}>
        <SubmitButton pendingText="Saving…">Save</SubmitButton>
      </form>,
    );
    await user.click(button('Save'));
    expect(pending.length).toBe(1);
    expect(screen.queryByRole('button', { name: 'Save' })).toBeNull();
    expect(button('Saving…').disabled).toBe(true);
    await finish(pending[0]);
    expect(button('Save').disabled).toBe(false);
    expect(screen.queryByRole('button', { name: 'Saving…' })).toBeNull();
  });

  it('cannot submit the form a second time while pending', async () => {
    const user = userEvent.setup({ delay: null });
    const { pending, action } = formAction();
    render(
      <form action={action}>
        <SubmitButton pendingText="Saving…">Save</SubmitButton>
      </form>,
    );
    await user.click(button('Save'));
    await user.click(screen.getByRole('button'));
    expect(pending.length).toBe(1);
  });

  it('reflects a submission started by pressing Enter in an input', async () => {
    const user = userEvent.setup({ delay: null });
    const { pending, action } = formAction();
    render(
      <form action={action}>
        <label>
          Title <input name="title" />
        </label>
        <SubmitButton pendingText="Publishing…">Publish</SubmitButton>
      </form>,
    );
    await user.type(screen.getByLabelText('Title'), 'Hello{Enter}');
    expect(pending.length).toBe(1);
    expect(button('Publishing…').disabled).toBe(true);
    await finish(pending[0]);
    expect(button('Publish').disabled).toBe(false);
  });

  it('reflects a submission started by another submit button of the same form', async () => {
    const user = userEvent.setup({ delay: null });
    const { pending, action } = formAction();
    render(
      <form action={action}>
        <SubmitButton pendingText="Saving…">Save</SubmitButton>
        <button type="submit">Save a copy</button>
      </form>,
    );
    await user.click(button('Save a copy'));
    expect(pending.length).toBe(1);
    expect(button('Saving…').disabled).toBe(true);
    await finish(pending[0]);
    expect(button('Save').disabled).toBe(false);
  });

  it('only the submitting form goes pending', async () => {
    const user = userEvent.setup({ delay: null });
    const a = formAction();
    const b = formAction();
    render(
      <>
        <form action={a.action}>
          <SubmitButton pendingText="Saving A…">Save A</SubmitButton>
        </form>
        <form action={b.action}>
          <SubmitButton pendingText="Saving B…">Save B</SubmitButton>
        </form>
      </>,
    );
    await user.click(button('Save A'));
    expect(button('Saving A…').disabled).toBe(true);
    expect(button('Save B').disabled).toBe(false);
    await finish(a.pending[0]);
    await user.click(button('Save B'));
    expect(b.pending.length).toBe(1);
    expect(button('Save A').disabled).toBe(false);
    expect(button('Saving B…').disabled).toBe(true);
    await finish(b.pending[0]);
    expect(button('Save B').disabled).toBe(false);
  });

  it('renders rich children', () => {
    const { action } = formAction();
    render(
      <form action={action}>
        <SubmitButton pendingText="Sending…">
          <strong>Send</strong> now
        </SubmitButton>
      </form>,
    );
    expect(button('Send now').querySelector('strong')?.textContent).toBe('Send');
  });
});
