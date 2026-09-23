import { useActionState, useEffect, useState } from 'react';
import { useFormStatus } from 'react-dom';

// A controlled input (the form reset does not touch it) cleared once a
// success lands, and a child button that reads the pending state through
// useFormStatus instead of useActionState's flag.
function Submit() {
  const { pending } = useFormStatus();
  return (
    <button type="submit" disabled={pending}>
      {pending ? 'Subscribing…' : 'Subscribe'}
    </button>
  );
}

interface Result {
  ok: boolean;
  text: string;
  seq: number;
}

export function SubscribeForm({ subscribe }: { subscribe: (email: string) => Promise<void> }) {
  const [email, setEmail] = useState('');
  const [result, run] = useActionState(
    async (prev: Result | null, data: FormData): Promise<Result> => {
      const value = data.get('address');
      const address = typeof value === 'string' ? value : '';
      const seq = (prev?.seq ?? 0) + 1;
      return subscribe(address).then(
        () => ({ ok: true, text: `Subscribed ${address}`, seq }),
        (e: unknown) => ({ ok: false, text: (e as Error).message, seq }),
      );
    },
    null,
  );

  useEffect(() => {
    if (result?.ok) setEmail('');
  }, [result]);

  return (
    <div>
      <form action={run}>
        <label htmlFor="sub-email">Email</label>
        <input id="sub-email" name="address" value={email} onChange={(e) => setEmail(e.target.value)} />
        <Submit />
      </form>
      {result && <div role={result.ok ? 'status' : 'alert'}>{result.text}</div>}
    </div>
  );
}
