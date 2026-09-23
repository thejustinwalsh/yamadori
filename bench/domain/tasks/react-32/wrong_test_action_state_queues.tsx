// Wrong: routes sends through useActionState, which runs actions one at a time, so a second message waits for the first send to settle.
import { startTransition, useActionState, useOptimistic, useState } from 'react';

type Shown = { text: string; sending: boolean };

export function Thread({
  initialMessages,
  send,
}: {
  initialMessages: string[];
  send: (text: string) => Promise<string>;
}) {
  const [error, setError] = useState<string | null>(null);
  const [messages, sendAction] = useActionState<string[], FormData>(async (prev, formData) => {
    const text = String(formData.get('message') ?? '');
    addPending(text);
    try {
      const saved = await send(text);
      return [...prev, saved];
    } catch {
      startTransition(() => setError(`Failed to send: ${text}`));
      return prev;
    }
  }, initialMessages);
  const [shown, addPending] = useOptimistic<Shown[], string>(
    messages.map((text) => ({ text, sending: false })),
    (current, text) => [...current, { text, sending: true }],
  );

  return (
    <section>
      <ul>
        {shown.map((m, i) => (
          <li key={i}>{m.sending ? `${m.text} (sending…)` : m.text}</li>
        ))}
      </ul>
      {error !== null && <p role="alert">{error}</p>}
      <form action={sendAction}>
        <label>
          Message <input name="message" />
        </label>
        <button type="submit">Send</button>
      </form>
    </section>
  );
}
