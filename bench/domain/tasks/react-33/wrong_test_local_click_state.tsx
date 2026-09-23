// Wrong: tracks "pending" in local state set on click; it cannot see when the form's action ends, and misses submissions it did not start.
import { useState, type ReactNode } from 'react';

export function SubmitButton({ children, pendingText }: { children: ReactNode; pendingText: string }) {
  const [pending, setPending] = useState(false);
  return (
    <button type="submit" disabled={pending} onClick={() => setPending(true)}>
      {pending ? pendingText : children}
    </button>
  );
}
