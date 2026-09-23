// Wrong (strict): the optimistic reducer's action parameter is left unannotated, so useOptimistic infers it as unknown.
import { startTransition, useOptimistic, useState } from 'react';

export function Thread({
  initialMessages,
  send,
}: {
  initialMessages: string[];
  send: (text: string) => Promise<string>;
}) {
  const [messages, setMessages] = useState<string[]>(initialMessages);
  const [error, setError] = useState<string | null>(null);
  const [shown, addPending] = useOptimistic(
    messages.map((text) => ({ text, sending: false })),
    (current, text) => [...current, { text, sending: true }],
  );

  async function sendAction(formData: FormData) {
    const text = String(formData.get('message') ?? '');
    addPending(text);
    try {
      const saved = await send(text);
      startTransition(() => setMessages((m) => [...m, saved]));
    } catch {
      startTransition(() => setError(`Failed to send: ${text}`));
    }
  }

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
