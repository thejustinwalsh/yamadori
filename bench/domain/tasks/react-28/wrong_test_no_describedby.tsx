// Wrong: the tooltip is never linked to the button with aria-describedby.
import { useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';

export function Tooltip({ content, label, delay = 300 }: { content: string; label: string; delay?: number }) {
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const id = useId();

  const cancel = () => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  };

  const start = () => {
    cancel();
    timer.current = setTimeout(() => {
      timer.current = null;
      setOpen(true);
    }, delay);
  };

  const hide = () => {
    cancel();
    setOpen(false);
  };

  useEffect(() => cancel, []);

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === 'Escape') hide();
  };

  return (
    <span>
      <button
        type="button"
        onMouseEnter={start}
        onMouseLeave={hide}
        onFocus={start}
        onBlur={hide}
        onKeyDown={onKeyDown}
      >
        {label}
      </button>
      {open && (
        <span role="tooltip" id={id}>
          {content}
        </span>
      )}
    </span>
  );
}
