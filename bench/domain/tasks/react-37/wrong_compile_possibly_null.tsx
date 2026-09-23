// Wrong: (strict) inputRef.current is possibly null.
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
      focus: () => inputRef.current.focus(),
      clear: () => setValue(''),
      getValue: () => value,
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
