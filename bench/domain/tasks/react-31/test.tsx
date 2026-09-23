import { describe, it, expect, afterEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SubscribeForm } from './solution';
import { deferred, settle, fail, flush, type Deferred } from './helpers';

// React 19 entangles every async action in flight: one left unresolved keeps
// later actions pending too. Settle whatever a test left open.
let open: Deferred<void>[] = [];
afterEach(async () => {
  const left = open;
  open = [];
  await act(async () => {
    for (const d of left) d.resolve();
    await Promise.allSettled(left.map((d) => d.promise));
  });
});

function setup() {
  const calls: string[] = [];
  const pending: Deferred<void>[] = [];
  const subscribe = (email: string): Promise<void> => {
    calls.push(email);
    const d = deferred<void>();
    pending.push(d);
    open.push(d);
    return d.promise;
  };
  const user = userEvent.setup({ delay: null });
  render(<SubscribeForm subscribe={subscribe} />);
  return { calls, pending, user };
}

const input = () => screen.getByLabelText('Email') as HTMLInputElement;
const button = () => screen.getByRole('button') as HTMLButtonElement;

describe('SubscribeForm', () => {
  it('starts with an enabled Subscribe button and nothing announced', () => {
    setup();
    expect(button().textContent).toBe('Subscribe');
    expect(button().disabled).toBe(false);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.queryByRole('status')).toBeNull();
  });

  it('calls subscribe with the typed email and shows the pending state', async () => {
    const { calls, user } = setup();
    await user.type(input(), 'ada@example.com');
    await user.click(screen.getByRole('button', { name: 'Subscribe' }));
    expect(calls).toEqual(['ada@example.com']);
    expect(button().textContent).toBe('Subscribing…');
    expect(button().disabled).toBe(true);
  });

  it('submits when Enter is pressed in the field', async () => {
    const { calls, user } = setup();
    await user.type(input(), 'grace@example.com{Enter}');
    expect(calls).toEqual(['grace@example.com']);
    expect(button().textContent).toBe('Subscribing…');
  });

  it('on success announces the address, empties the field and re-enables the button', async () => {
    const { pending, user } = setup();
    await user.type(input(), 'ada@example.com');
    await user.click(button());
    await settle(pending[0], undefined);
    await flush();
    expect(screen.getByRole('status').textContent).toBe('Subscribed ada@example.com');
    expect(screen.queryByRole('alert')).toBeNull();
    expect(input().value).toBe('');
    expect(button().textContent).toBe('Subscribe');
    expect(button().disabled).toBe(false);
  });

  it('on failure shows the error message and keeps the submitted email', async () => {
    const { pending, user } = setup();
    await user.type(input(), 'ada@example.com');
    await user.click(button());
    await fail(pending[0], new Error('That address is already on the list'));
    await flush();
    expect(screen.getByRole('alert').textContent).toBe('That address is already on the list');
    expect(screen.queryByRole('status')).toBeNull();
    expect(input().value).toBe('ada@example.com');
    expect(button().textContent).toBe('Subscribe');
    expect(button().disabled).toBe(false);
  });

  it('a retry after a failure resends the same email and clears the alert on success', async () => {
    const { calls, pending, user } = setup();
    await user.type(input(), 'ada@example.com');
    await user.click(button());
    await fail(pending[0], new Error('Network down'));
    await flush();
    await user.click(screen.getByRole('button', { name: 'Subscribe' }));
    expect(calls).toEqual(['ada@example.com', 'ada@example.com']);
    await settle(pending[1], undefined);
    await flush();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByRole('status').textContent).toBe('Subscribed ada@example.com');
  });

  it('a second address after a success replaces the status', async () => {
    const { calls, pending, user } = setup();
    await user.type(input(), 'ada@example.com');
    await user.click(button());
    await settle(pending[0], undefined);
    await flush();
    expect(input().value).toBe('');
    // form.reset() bypasses the value setter user-event tracks; clear its
    // stale copy before typing again.
    await user.clear(input());
    await user.type(input(), 'grace@example.com');
    await user.click(button());
    expect(calls).toEqual(['ada@example.com', 'grace@example.com']);
    await settle(pending[1], undefined);
    await flush();
    expect(screen.getByRole('status').textContent).toBe('Subscribed grace@example.com');
    expect(input().value).toBe('');
  });
});
