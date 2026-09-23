import { useId, useImperativeHandle, useRef, useState } from 'react';
import type { Ref } from 'react';

export interface FancyInputHandle {
  focus(): void;
  clear(): void;
  getValue(): string;
}

// Uncontrolled input: the DOM owns the text, React only tracks its length.
// The handle is rebuilt every render and reads the DOM, so it never goes stale.
export function FancyInput(props: { label: string; ref?: Ref<FancyInputHandle> }) {
  const id = useId();
  const node = useRef<HTMLInputElement | null>(null);
  const [length, setLength] = useState(0);

  useImperativeHandle(props.ref, () => ({
    focus() {
      node.current?.focus();
    },
    clear() {
      if (node.current) node.current.value = '';
      setLength(0);
    },
    getValue() {
      return node.current ? node.current.value : '';
    },
  }));

  return (
    <>
      <label htmlFor={id}>{props.label}</label>
      <input
        id={id}
        ref={node}
        defaultValue=""
        onInput={(e) => setLength(e.currentTarget.value.length)}
      />
      <span>{`Length: ${length}`}</span>
    </>
  );
}
