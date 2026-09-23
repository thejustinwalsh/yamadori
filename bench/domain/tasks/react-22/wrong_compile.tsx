import { useState } from 'react';

// Wrong (strict): hover state is inferred as `null` from its initial value,
// so it can never be set to a star number.
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
  const [hover, setHover] = useState(null);
  const shown = hover ?? value;
  return (
    <div
      role="radiogroup"
      aria-label="Rating"
      tabIndex={0}
      onKeyDown={(e) => {
        if (readOnly) return;
        if (e.key === 'ArrowRight' && value < max) onChange(value + 1);
        if (e.key === 'ArrowLeft' && value > 1) onChange(value - 1);
      }}
    >
      {Array.from({ length: max }, (_, i) => i + 1).map((n) => (
        <span
          key={n}
          role="radio"
          aria-label={n === 1 ? '1 star' : `${n} stars`}
          aria-checked={n === value}
          aria-disabled={readOnly || undefined}
          onMouseEnter={() => setHover(n)}
          onMouseLeave={() => setHover(null)}
          onClick={() => !readOnly && onChange(n)}
        >
          {n <= shown ? '★' : '☆'}
        </span>
      ))}
    </div>
  );
}
