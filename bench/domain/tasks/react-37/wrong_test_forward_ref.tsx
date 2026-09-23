// Wrong: pre-19 forwardRef wrapper; FancyInput is an object, not a plain function component.
import { forwardRef, useImperativeHandle, useRef, useState } from 'react';

export interface FancyInputHandle {
  focus(): void;
  clear(): void;
  getValue(): string;
}

export const FancyInput = forwardRef<FancyInputHandle, { label: string }>(function FancyInput(
  { label },
  ref,
) {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useImperativeHandle(
    ref,
    () => ({
      focus: () => inputRef.current?.focus(),
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
});
