type Item = { kind: 'page'; n: number } | { kind: 'gap'; key: string };

function items(page: number, total: number, siblings: number): Item[] {
  const shown = new Set<number>([1, total]);
  for (let p = Math.max(1, page - siblings); p <= Math.min(total, page + siblings); p++) shown.add(p);
  const sorted = [...shown].sort((a, b) => a - b);
  const out: Item[] = [];
  let prev = 0;
  for (const n of sorted) {
    if (prev > 0) {
      const missing = n - prev - 1;
      if (missing === 1) out.push({ kind: 'page', n: prev + 1 });
      else if (missing > 1) out.push({ kind: 'gap', key: `gap-${prev}` });
    }
    out.push({ kind: 'page', n });
    prev = n;
  }
  return out;
}

export function Pagination({
  page,
  totalPages,
  onPageChange,
  siblings = 1,
}: {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  siblings?: number;
}) {
  return (
    <nav aria-label="Pagination">
      <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
        Previous
      </button>
      {items(page, totalPages, siblings).map((it) =>
        it.kind === 'gap' ? (
          <span key={it.key}>…</span>
        ) : (
          <button
            key={it.n}
            type="button"
            aria-current={it.n === page ? 'page' : undefined}
            onClick={() => {
              if (it.n !== page) onPageChange(it.n);
            }}
          >
            {it.n}
          </button>
        ),
      )}
      <button type="button" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
        Next
      </button>
    </nav>
  );
}
