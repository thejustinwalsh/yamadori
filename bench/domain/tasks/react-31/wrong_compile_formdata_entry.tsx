// Wrong (strict): FormData.get returns FormDataEntryValue | null, which is passed straight on as a string.
import { useActionState } from 'react';

type State = { ok: true; email: string } | { ok: false; email: string; message: string } | null;

export function SubscribeForm({ subscribe }: { subscribe: (email: string) => Promise<void> }) {
  const [state, formAction, isPending] = useActionState<State, FormData>(async (_prev, formData) => {
    const email = formData.get('email');
    try {
      await subscribe(email);
      return { ok: true, email: String(email) };
    } catch (err) {
      return { ok: false, email: String(email), message: (err as Error).message };
    }
  }, null);

  return (
    <form action={formAction}>
      <label>
        Email <input name="email" defaultValue={state && !state.ok ? state.email : ''} />
      </label>
      <button type="submit" disabled={isPending}>
        {isPending ? 'Subscribing…' : 'Subscribe'}
      </button>
      {state && !state.ok && <p role="alert">{state.message}</p>}
      {state?.ok && <p role="status">Subscribed {state.email}</p>}
    </form>
  );
}
