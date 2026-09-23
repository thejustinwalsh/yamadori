import type { ReactNode } from 'react';
import { useFormStatus } from 'react-dom';

export function SubmitButton({ children, pendingText }: { children: ReactNode; pendingText: string }) {
  const { pending } = useFormStatus();
  return (
    <button type="submit" disabled={pending}>
      {pending ? pendingText : children}
    </button>
  );
}
