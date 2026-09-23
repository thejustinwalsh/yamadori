// Wrong (strict): useFormStatus is exported by react-dom, not react.
import { useFormStatus, type ReactNode } from 'react';

export function SubmitButton({ children, pendingText }: { children: ReactNode; pendingText: string }) {
  const { pending } = useFormStatus();
  return (
    <button type="submit" disabled={pending}>
      {pending ? pendingText : children}
    </button>
  );
}
