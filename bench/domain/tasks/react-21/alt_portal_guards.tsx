import { useLayoutEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import type * as React from 'react';

// Portal into document.body; the trap uses two focus guards around the dialog
// instead of intercepting Tab; Escape is a document listener that is removed
// when the dialog closes.
const SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

let counter = 0;

export function Modal(props: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}): React.ReactElement | null {
  const box = useRef<HTMLElement | null>(null);
  const headingId = useRef<string>('');
  if (headingId.current === '') headingId.current = `modal-title-${++counter}`;
  const onCloseRef = useRef(props.onClose);
  onCloseRef.current = props.onClose;

  const focusables = (): HTMLElement[] =>
    box.current ? Array.from(box.current.querySelectorAll<HTMLElement>(SELECTOR)) : [];

  useLayoutEffect(() => {
    if (!props.open) return;
    const restoreTo = document.activeElement as HTMLElement | null;
    const list = focusables();
    if (list.length > 0) list[0].focus();
    else box.current?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && box.current && box.current.contains(document.activeElement)) {
        onCloseRef.current();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      if (restoreTo && typeof restoreTo.focus === 'function') restoreTo.focus();
    };
  }, [props.open]);

  if (!props.open) return null;

  const wrapTo = (end: 'first' | 'last') => () => {
    const list = focusables();
    if (list.length === 0) {
      box.current?.focus();
      return;
    }
    (end === 'first' ? list[0] : list[list.length - 1]).focus();
  };

  return createPortal(
    <>
      <span tabIndex={0} onFocus={wrapTo('last')} />
      <section
        ref={box}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId.current}
        tabIndex={-1}
      >
        <h1 id={headingId.current}>{props.title}</h1>
        <div>{props.children}</div>
      </section>
      <span tabIndex={0} onFocus={wrapTo('first')} />
    </>,
    document.body,
  );
}
