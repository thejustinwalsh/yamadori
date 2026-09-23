// Wrong: arrow keys are not clamped, so onChange receives 0 or max + 1.
import type { KeyboardEvent } from 'react';

export function StarRating({
  value,
  onChange,
  max = 5,
  readOnly = false,
}: {
  value: number;
  onChange: (v: number) => void;
  max?: number;
  readOnly?: boolean;
}) {
  const set = (n: number) => {
    if (readOnly) return;
    onChange(n);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      set(value + 1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      set(value - 1);
    }
  };

  const stars = Array.from({ length: max }, (_, i) => i + 1);
  return (
    <div role="radiogroup" aria-label="Rating" tabIndex={0} onKeyDown={onKeyDown}>
      {stars.map((n) => (
        <span
          key={n}
          role="radio"
          aria-label={n === 1 ? '1 star' : `${n} stars`}
          aria-checked={n === value}
          aria-disabled={readOnly || undefined}
          onClick={() => set(n)}
        >
          {n <= value ? '★' : '☆'}
        </span>
      ))}
    </div>
  );
}
