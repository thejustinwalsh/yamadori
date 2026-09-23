import { describe, it, expect, afterEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Thread } from './solution';
import { deferred, settle, fail, flush, type Deferred } from './helpers';

// React 19 entangles every async action in flight: one left unresolved keeps
// later ones pending too. Settle whatever a test left open.
let open: Deferred<string>[] = [];
afterEach(async () => {
  const left = open;
  open = [];
  await act(async () => {
    for (const d of left) d.resolve('');
    await Promise.allSettled(left.map((d) => d.promise));
  });
});

function setup(initialMessages: string[] = ['Hi!', 'How are you?']) {
  const calls: string[] = [];
  const pending: Deferred<string>[] = [];
  const send = (text: string): Promise<string> => {
    calls.push(text);
    const d = deferred<string>();
    pending.push(d);
    open.push(d);
    return d.promise;
  };
  const user = userEvent.setup({ delay: null });
  render(<Thread initialMessages={initialMessages} send={send} />);
  const post = async (text: string) => {
    const input = screen.getByLabelText('Message');
    // The field may still hold the previous text, and a form reset bypasses
    // the value user-event tracks: clear it first.
    await user.clear(input);
    await user.type(input, text);
    await user.click(screen.getByRole('button', { name: 'Send' }));
  };
  return { calls, pending, post };
}

const items = () => screen.queryAllByRole('listitem').map((li) => li.textContent);

async function resolve(d: Deferred<string>, value: string) {
  await settle(d, value);
  await flush();
}

async function reject(d: Deferred<string>) {
  await fail(d, new Error('offline'));
  await flush();
}

describe('Thread', () => {
  it('lists the initial messages in order', () => {
    setup();
    expect(items()).toEqual(['Hi!', 'How are you?']);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('shows the message as sending immediately on submit', async () => {
    const { calls, post } = setup();
    await post('fine thanks');
    expect(calls).toEqual(['fine thanks']);
    expect(items()).toEqual(['Hi!', 'How are you?', 'fine thanks (sending…)']);
  });

  it('replaces the sending item with the server copy when send resolves', async () => {
    const { pending, post } = setup();
    await post('fine thanks');
    await resolve(pending[0], 'Fine, thanks.');
    expect(items()).toEqual(['Hi!', 'How are you?', 'Fine, thanks.']);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('removes the optimistic item and raises an alert when send rejects', async () => {
    const { pending, post } = setup();
    await post('fine thanks');
    await reject(pending[0]);
    expect(items()).toEqual(['Hi!', 'How are you?']);
    expect(screen.getByRole('alert').textContent).toBe('Failed to send: fine thanks');
  });

  it('does not queue sends: each submission shows its own sending item at once', async () => {
    const { calls, pending, post } = setup([]);
    await post('one');
    await post('two');
    await post('three');
    expect(calls).toEqual(['one', 'two', 'three']);
    expect(items()).toEqual(['one (sending…)', 'two (sending…)', 'three (sending…)']);
    await resolve(pending[0], 'ONE');
    await resolve(pending[1], 'TWO');
    await resolve(pending[2], 'THREE');
    expect(items()).toEqual(['ONE', 'TWO', 'THREE']);
  });

  it('appends confirmed messages in the order the sends resolve', async () => {
    const { pending, post } = setup(['start']);
    await post('first');
    await post('second');
    await resolve(pending[1], 'second!');
    await resolve(pending[0], 'first!');
    expect(items()).toEqual(['start', 'second!', 'first!']);
  });

  it('with one send failing and one succeeding, only the failed one is dropped', async () => {
    const { pending, post } = setup(['start']);
    await post('lost');
    await post('kept');
    expect(items()).toEqual(['start', 'lost (sending…)', 'kept (sending…)']);
    await reject(pending[0]);
    await resolve(pending[1], 'kept.');
    expect(items()).toEqual(['start', 'kept.']);
    expect(screen.getByRole('alert').textContent).toBe('Failed to send: lost');
  });
});
