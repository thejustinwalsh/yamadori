// Wrong: clear() only blanks the DOM node; React state keeps the old text, so the counter is stale and the text comes back on re-render.
import { useImperativeHandle, useRef, useState, type Ref } from 'react';

export interface FancyInputHandle {
  focus(): void;
  clear(): void;
  getValue(): string;
}

export function FancyInput({ label, ref }: { label: string; ref?: Ref<FancyInputHandle> }) {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useImperativeHandle(
    ref,
    () => ({
      focus: () => inputRef.current?.focus(),
      clear: () => {
        if (inputRef.current) inputRef.current.value = '';
      },
      getValue: () => inputRef.current?.value ?? '',
    }),
    [value],
  );

  return (
    <div>
      <label>
        {label}
        <input ref={inputRef} value={value} onChange={(e) => setValue(e.target.value)} />
      </label>
      <p>Length: {value.length}</p>
    </div>
  );
}
