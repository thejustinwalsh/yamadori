// Wrong: a plain uncontrolled input. React 19 resets the form after the action even when it failed, so the email is gone and a retry sends "".
import { useActionState } from 'react';

type State = { ok: true; email: string } | { ok: false; message: string } | null;

export function SubscribeForm({ subscribe }: { subscribe: (email: string) => Promise<void> }) {
  const [state, formAction, isPending] = useActionState<State, FormData>(async (_prev, formData) => {
    const email = String(formData.get('email') ?? '');
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
        Email <input name="email" />
      </label>
      <button type="submit" disabled={isPending}>
        {isPending ? 'Subscribing…' : 'Subscribe'}
      </button>
      {state && !state.ok && <p role="alert">{state.message}</p>}
      {state?.ok && <p role="status">Subscribed {state.email}</p>}
    </form>
  );
}
