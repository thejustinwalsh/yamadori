import { useActionState } from 'react';

type State =
  | { kind: 'idle' }
  | { kind: 'subscribed'; email: string }
  | { kind: 'failed'; email: string; message: string };

export function SubscribeForm({ subscribe }: { subscribe: (email: string) => Promise<void> }) {
  const [state, formAction, isPending] = useActionState<State, FormData>(
    async (_prev, formData) => {
      const email = String(formData.get('email') ?? '');
      try {
        await subscribe(email);
        return { kind: 'subscribed', email };
      } catch (err) {
        return {
          kind: 'failed',
          email,
          message: err instanceof Error ? err.message : String(err),
        };
      }
    },
    { kind: 'idle' },
  );

  // React resets the form after every action, failed or not. The failed
  // email goes back in as the input's default value, which the reset restores.
  return (
    <form action={formAction}>
      <label>
        Email{' '}
        <input name="email" type="text" defaultValue={state.kind === 'failed' ? state.email : ''} />
      </label>
      <button type="submit" disabled={isPending}>
        {isPending ? 'Subscribing…' : 'Subscribe'}
      </button>
      {state.kind === 'failed' && <p role="alert">{state.message}</p>}
      {state.kind === 'subscribed' && <p role="status">Subscribed {state.email}</p>}
    </form>
  );
}
