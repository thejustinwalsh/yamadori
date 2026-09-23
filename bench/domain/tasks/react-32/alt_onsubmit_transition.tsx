import { useOptimistic, useRef, useState, useTransition, type FormEvent } from 'react';

// onSubmit + startTransition instead of a form action, a controlled input
// cleared on submit, and an optimistic reducer keyed by a client id.
type Entry = { id: number; body: string; pending?: true };
type Pending = { id: number; body: string };

export function Thread({
  initialMessages,
  send,
}: {
  initialMessages: string[];
  send: (text: string) => Promise<string>;
}) {
  const nextId = useRef(initialMessages.length);
  const [confirmed, setConfirmed] = useState<Entry[]>(() =>
    initialMessages.map((body, id) => ({ id, body })),
  );
  const [failure, setFailure] = useState('');
  const [draft, setDraft] = useState('');
  const [, startTransition] = useTransition();
  const [entries, addOptimistic] = useOptimistic(confirmed, (list: Entry[], p: Pending): Entry[] =>
    list.concat({ id: p.id, body: p.body, pending: true }),
  );

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const body = draft;
    const id = nextId.current++;
    setDraft('');
    startTransition(async () => {
      addOptimistic({ id, body });
      await send(body).then(
        (saved) => startTransition(() => setConfirmed((c) => [...c, { id, body: saved }])),
        () => startTransition(() => setFailure(body)),
      );
    });
  }

  return (
    <div>
      <ol aria-label="Messages">
        {entries.map((m) => (
          <li key={m.id}>
            {m.body}
            {m.pending ? ' (sending…)' : null}
          </li>
        ))}
      </ol>
      {failure ? <div role="alert">Failed to send: {failure}</div> : null}
      <form onSubmit={onSubmit}>
        <input aria-label="Message" value={draft} onChange={(e) => setDraft(e.target.value)} />
        <button>Send</button>
      </form>
    </div>
  );
}
