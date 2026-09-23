import { useId } from 'react';
import type * as React from 'react';

// Stars are <button>s named by their (visually hidden) text; the group is
// named through aria-labelledby; readOnly is enforced by a guard wrapper.
export function StarRating(props: {
  value: number;
  onChange: (v: number) => void;
  max?: number;
  readOnly?: boolean;
}): React.ReactElement {
  const max = props.max ?? 5;
  const labelId = useId();
  const emit = props.readOnly ? () => {} : props.onChange;
  const clamp = (n: number) => (n >= 1 && n <= max ? emit(n) : undefined);

  const items: React.ReactElement[] = [];
  for (let n = 1; n <= max; n++) {
    items.push(
      <button
        key={n}
        type="button"
        role="radio"
        tabIndex={-1}
        aria-checked={props.value === n ? 'true' : 'false'}
        aria-disabled={props.readOnly ? 'true' : undefined}
        onClick={() => clamp(n)}
      >
        <span aria-hidden="true">{props.value >= n ? '★' : '☆'}</span>
        <span style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden' }}>
          {n} {n === 1 ? 'star' : 'stars'}
        </span>
      </button>,
    );
  }

  return (
    <div>
      <span id={labelId} hidden>
        Rating
      </span>
      <div
        role="radiogroup"
        aria-labelledby={labelId}
        tabIndex={0}
        onKeyDown={(e) => {
          const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
          if (step === 0) return;
          e.preventDefault();
          clamp(props.value + step);
        }}
      >
        {items}
      </div>
    </div>
  );
}
