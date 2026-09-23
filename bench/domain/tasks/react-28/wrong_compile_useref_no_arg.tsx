// Wrong (strict, React 19 types): useRef<T>() without an initial value is no longer allowed.
import { useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';

export function Tooltip({ content, label, delay = 300 }: { content: string; label: string; delay?: number }) {
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const id = useId();

  const cancel = () => {
    clearTimeout(timer.current);
    timer.current = undefined;
  };

  const start = () => {
    cancel();
    timer.current = setTimeout(() => {
      timer.current = undefined;
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
        aria-describedby={open ? id : undefined}
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
