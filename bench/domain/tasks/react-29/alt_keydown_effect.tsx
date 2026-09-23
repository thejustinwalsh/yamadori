import { useEffect, useRef, useState } from 'react';

// Keyboard-driven: digits and Backspace are handled in keydown (default
// prevented), boxes carry maxLength=1, focus goes through a container query,
// and completion is detected in an effect that runs once per code change.
export function OtpInput(props: { length: number; onComplete: (code: string) => void }) {
  const { length } = props;
  const [code, setCode] = useState<string>(' '.repeat(length));
  const wrap = useRef<HTMLDivElement>(null);
  const latest = useRef(props.onComplete);
  latest.current = props.onComplete;
  const changed = useRef(false);

  useEffect(() => {
    if (!changed.current) return;
    changed.current = false;
    if (!code.includes(' ')) latest.current(code);
  }, [code]);

  const boxAt = (i: number): HTMLInputElement | undefined =>
    wrap.current?.querySelectorAll('input')[i] ?? undefined;

  const write = (from: number, digits: string) => {
    const chars = code.split('');
    let i = from;
    for (const d of digits) {
      if (i >= length) break;
      chars[i++] = d;
    }
    const next = chars.join('');
    if (next !== code) {
      changed.current = true;
      setCode(next);
    }
    return i;
  };

  const clear = (i: number) => write(i, ' ');

  return (
    <div ref={wrap} role="group" aria-label="One-time code">
      {code.split('').map((c, i) => (
        <input
          key={i}
          type="text"
          inputMode="numeric"
          maxLength={1}
          aria-label={`Digit ${i + 1} of ${length}`}
          value={c.trim()}
          onChange={(e) => {
            // autofill or IME fallback: take whatever digits arrived
            const digits = e.target.value.replace(/\D/g, '');
            if (digits) boxAt(Math.min(write(i, digits), length - 1))?.focus();
          }}
          onKeyDown={(e) => {
            if (e.ctrlKey || e.metaKey || e.altKey) return;
            if (/^[0-9]$/.test(e.key)) {
              e.preventDefault();
              write(i, e.key);
              if (i + 1 < length) boxAt(i + 1)?.focus();
            } else if (e.key === 'Backspace') {
              e.preventDefault();
              if (c !== ' ') clear(i);
              else if (i > 0) {
                clear(i - 1);
                boxAt(i - 1)?.focus();
              }
            } else if (e.key.length === 1) {
              e.preventDefault();
            }
          }}
          onPaste={(e) => {
            e.preventDefault();
            const digits = e.clipboardData.getData('text/plain').replace(/\D/g, '');
            if (!digits) return;
            const end = write(i, digits);
            boxAt(Math.min(end, length - 1))?.focus();
          }}
        />
      ))}
    </div>
  );
}
