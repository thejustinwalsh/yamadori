// Wrong: no paste handling and maxLength={1}, so a pasted code only lands its first character in the focused box.
import { useRef, useState } from 'react';
import type { ChangeEvent, ClipboardEvent, KeyboardEvent } from 'react';

export function OtpInput({ length, onComplete }: { length: number; onComplete: (code: string) => void }) {
  const [digits, setDigits] = useState<string[]>(() => Array.from({ length }, () => ''));
  const refs = useRef<(HTMLInputElement | null)[]>([]);

  const focusBox = (i: number) => refs.current[Math.max(0, Math.min(length - 1, i))]?.focus();

  void (() => _onPaste);
  const commit = (next: string[]) => {
    setDigits(next);
    if (next.every((d) => d !== '')) onComplete(next.join(''));
  };

  const onChange = (i: number) => (e: ChangeEvent<HTMLInputElement>) => {
    const typed = e.target.value.replace(digits[i], '');
    const ch = typed.slice(-1);
    if (e.target.value === '') {
      const next = [...digits];
      next[i] = '';
      commit(next);
      return;
    }
    if (!/^[0-9]$/.test(ch)) return;
    const next = [...digits];
    next[i] = ch;
    commit(next);
    if (i < length - 1) focusBox(i + 1);
  };

  const onKeyDown = (i: number) => (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Backspace' && digits[i] === '' && i > 0) {
      e.preventDefault();
      const next = [...digits];
      next[i - 1] = '';
      commit(next);
      focusBox(i - 1);
    }
  };

  const _onPaste = (i: number) => (e: ClipboardEvent<HTMLInputElement>) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData('text').replace(/[^0-9]/g, '');
    if (!pasted) return;
    const next = [...digits];
    let j = i;
    for (const d of pasted) {
      if (j >= length) break;
      next[j++] = d;
    }
    commit(next);
    focusBox(j);
  };

  return (
    <div>
      {digits.map((d, i) => (
        <input
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="text"
          inputMode="numeric"
          autoComplete="one-time-code"
          aria-label={`Digit ${i + 1} of ${length}`}
          value={d}
          onChange={onChange(i)}
          onKeyDown={onKeyDown(i)}
          maxLength={1}
        />
      ))}
    </div>
  );
}
