// Wrong: no optimistic update at all; a message appears only once send resolves.
import { useState } from 'react';

export function Thread({
  initialMessages,
  send,
}: {
  initialMessages: string[];
  send: (text: string) => Promise<string>;
}) {
  const [messages, setMessages] = useState<string[]>(initialMessages);
  const [error, setError] = useState<string | null>(null);

  async function sendAction(formData: FormData) {
    const text = String(formData.get('message') ?? '');
    try {
      const saved = await send(text);
      setMessages((m) => [...m, saved]);
    } catch {
      setError(`Failed to send: ${text}`);
    }
  }

  return (
    <section>
      <ul>
        {messages.map((m, i) => (
          <li key={i}>{m}</li>
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
