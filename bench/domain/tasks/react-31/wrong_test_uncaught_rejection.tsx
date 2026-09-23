// Wrong: the rejection is not caught inside the action, so it escapes useActionState as a render error and no alert is ever shown.
import { useActionState } from 'react';

export function SubscribeForm({ subscribe }: { subscribe: (email: string) => Promise<void> }) {
  const [subscribed, formAction, isPending] = useActionState<string | null, FormData>(
    async (_prev, formData) => {
      const email = String(formData.get('email') ?? '');
      await subscribe(email);
      return email;
    },
    null,
  );

  return (
    <form action={formAction}>
      <label>
        Email <input name="email" />
      </label>
      <button type="submit" disabled={isPending}>
        {isPending ? 'Subscribing…' : 'Subscribe'}
      </button>
      {subscribed !== null && <p role="status">Subscribed {subscribed}</p>}
    </form>
  );
}
