// Wrong: shows the pending text but leaves the button enabled, so the form can be submitted again mid-flight.
import type { ReactNode } from 'react';
import { useFormStatus } from 'react-dom';

export function SubmitButton({ children, pendingText }: { children: ReactNode; pendingText: string }) {
  const { pending } = useFormStatus();
  return (
    <button type="submit" aria-disabled={pending}>
      {pending ? pendingText : children}
    </button>
  );
}
