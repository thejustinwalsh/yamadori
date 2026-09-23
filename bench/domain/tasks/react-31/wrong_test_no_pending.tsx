// Wrong: ignores the pending flag, so the button stays enabled and never says "Subscribing…" while the call is in flight.
import { useActionState } from 'react';

type State = { ok: true; email: string } | { ok: false; email: string; message: string } | null;

export function SubscribeForm({ subscribe }: { subscribe: (email: string) => Promise<void> }) {
  const [state, formAction] = useActionState<State, FormData>(async (_prev, formData) => {
    const email = String(formData.get('email') ?? '');
    try {
      await subscribe(email);
      return { ok: true, email };
    } catch (err) {
      return { ok: false, email, message: (err as Error).message };
    }
  }, null);

  return (
    <form action={formAction}>
      <label>
        Email <input name="email" defaultValue={state && !state.ok ? state.email : ''} />
      </label>
      <button type="submit">Subscribe</button>
      {state && !state.ok && <p role="alert">{state.message}</p>}
      {state?.ok && <p role="status">Subscribed {state.email}</p>}
    </form>
  );
}
