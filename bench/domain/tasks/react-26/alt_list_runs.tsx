import { useMemo } from 'react';

// Marks every page as kept or not, then collapses each run of dropped pages:
// a run of one becomes that page, a longer run becomes an ellipsis. Rendered
// as a list, with icon-only Previous/Next buttons named by aria-label.
function layout(current: number, total: number, sib: number): (number | null)[] {
  const keep = (p: number) =>
    p === 1 || p === total || Math.abs(p - current) <= sib;
  const result: (number | null)[] = [];
  let p = 1;
  while (p <= total) {
    if (keep(p)) {
      result.push(p);
      p++;
      continue;
    }
    let end = p;
    while (end + 1 <= total && !keep(end + 1)) end++;
    result.push(end === p ? p : null);
    p = end + 1;
  }
  return result;
}

interface PaginationProps {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  siblings?: number;
}

export function Pagination(props: PaginationProps) {
  const { page, totalPages, onPageChange } = props;
  const sib = props.siblings ?? 1;
  const entries = useMemo(() => layout(page, totalPages, sib), [page, totalPages, sib]);
  const atStart = page === 1;
  const atEnd = page === totalPages;

  return (
    <nav aria-label="Pagination">
      <ul style={{ display: 'flex', listStyle: 'none', gap: 4 }}>
        <li>
          <button aria-label="Previous" disabled={atStart} onClick={() => onPageChange(page - 1)}>
            ‹
          </button>
        </li>
        {entries.map((n, i) =>
          n === null ? (
            <li key={`e${i}`} aria-hidden="true">
              …
            </li>
          ) : (
            <li key={n}>
              <button
                aria-current={n === page ? 'page' : undefined}
                onClick={n === page ? undefined : () => onPageChange(n)}
              >
                {String(n)}
              </button>
            </li>
          ),
        )}
        <li>
          <button aria-label="Next" disabled={atEnd} onClick={() => onPageChange(page + 1)}>
            ›
          </button>
        </li>
      </ul>
    </nav>
  );
}
