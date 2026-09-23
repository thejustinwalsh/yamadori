import type { KeyboardEvent } from 'react';

// Wrong: stars are handled by their 0-based index, so clicks report n - 1 and
// the checked star is one to the right of `value`.
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
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (readOnly) return;
    if (e.key === 'ArrowRight' && value < max) onChange(value + 1);
    if (e.key === 'ArrowLeft' && value > 1) onChange(value - 1);
  };
  return (
    <div role="radiogroup" aria-label="Rating" tabIndex={0} onKeyDown={onKeyDown}>
      {Array.from({ length: max }, (_, i) => (
        <span
          key={i}
          role="radio"
          aria-label={i === 0 ? '1 star' : `${i + 1} stars`}
          aria-checked={i === value}
          aria-disabled={readOnly || undefined}
          onClick={() => !readOnly && onChange(i)}
        >
          {i < value ? '★' : '☆'}
        </span>
      ))}
    </div>
  );
}
