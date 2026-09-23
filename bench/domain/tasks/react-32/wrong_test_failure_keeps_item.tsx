// Wrong: hand-rolled optimism in plain state; a failed send only raises the alert, so its "(sending…)" item stays forever.
import { useRef, useState, type FormEvent } from 'react';

type Item = { id: number; text: string; sending: boolean };

export function Thread({
  initialMessages,
  send,
}: {
  initialMessages: string[];
  send: (text: string) => Promise<string>;
}) {
  const seq = useRef(0);
  const [items, setItems] = useState<Item[]>(() =>
    initialMessages.map((text) => ({ id: seq.current++, text, sending: false })),
  );
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState('');

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const text = draft;
    const id = seq.current++;
    setItems((list) => [...list, { id, text, sending: true }]);
    send(text).then(
      (saved) =>
        setItems((list) => [
          ...list.filter((i) => i.id !== id),
          { id, text: saved, sending: false },
        ]),
      () => setError(`Failed to send: ${text}`),
    );
  }

  return (
    <section>
      <ul>
        {items.map((m) => (
          <li key={m.id}>{m.sending ? `${m.text} (sending…)` : m.text}</li>
        ))}
      </ul>
      {error !== null && <p role="alert">{error}</p>}
      <form onSubmit={onSubmit}>
        <label>
          Message <input value={draft} onChange={(e) => setDraft(e.target.value)} />
        </label>
        <button type="submit">Send</button>
      </form>
    </section>
  );
}
