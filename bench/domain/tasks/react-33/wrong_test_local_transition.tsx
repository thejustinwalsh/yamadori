// Wrong: tracks pending with its own transition started on click. React entangles it with the form action only when this button was clicked, so a submission started another way is missed.
import { useTransition, type ReactNode } from 'react';

export function SubmitButton({ children, pendingText }: { children: ReactNode; pendingText: string }) {
  const [isPending, startTransition] = useTransition();
  return (
    <button
      type="submit"
      disabled={isPending}
      onClick={() => {
        startTransition(() => {});
      }}
    >
      {isPending ? pendingText : children}
    </button>
  );
}
