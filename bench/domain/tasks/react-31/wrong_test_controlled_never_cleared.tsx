// Wrong: a controlled input survives the form reset, but nothing clears it after a success, so the old address stays in the field.
import { useActionState, useState } from 'react';

type State = { ok: true; email: string } | { ok: false; message: string } | null;

export function SubscribeForm({ subscribe }: { subscribe: (email: string) => Promise<void> }) {
  const [email, setEmail] = useState('');
  const [state, formAction, isPending] = useActionState<State, FormData>(async () => {
    try {
      await subscribe(email);
      return { ok: true, email };
    } catch (err) {
      return { ok: false, message: (err as Error).message };
    }
  }, null);

  return (
    <form action={formAction}>
      <label>
        Email <input value={email} onChange={(e) => setEmail(e.target.value)} />
      </label>
      <button type="submit" disabled={isPending}>
        {isPending ? 'Subscribing…' : 'Subscribe'}
      </button>
      {state && !state.ok && <p role="alert">{state.message}</p>}
      {state?.ok && <p role="status">Subscribed {state.email}</p>}
    </form>
  );
}
