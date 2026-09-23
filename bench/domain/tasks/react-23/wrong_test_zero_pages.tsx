// Wrong: an empty result has zero pages, so it reads Page 0 of 0.
import { useId, useMemo, useState } from 'react';

type Row = { id: number; name: string; age: number };
type Key = 'name' | 'age';
type Dir = 'ascending' | 'descending';

export function DataTable({ rows, pageSize }: { rows: readonly Row[]; pageSize: number }) {
  const filterId = useId();
  const [filter, setFilter] = useState('');
  const [sort, setSort] = useState<{ key: Key; dir: Dir } | null>(null);
  const [page, setPage] = useState(1);

  const visible = useMemo(() => {
    const needle = filter.toLowerCase();
    const kept = rows.filter((r) => r.name.toLowerCase().includes(needle));
    if (!sort) return kept;
    const sign = sort.dir === 'ascending' ? 1 : -1;
    // `filter` returned a fresh array, so sorting it leaves `rows` untouched.
    return kept.sort((a, b) => {
      const c = sort.key === 'name' ? a.name.localeCompare(b.name) : a.age - b.age;
      return c * sign;
    });
  }, [rows, filter, sort]);

  const pageCount = Math.ceil(visible.length / pageSize);
  const current = Math.min(page, pageCount);
  const shown = visible.slice((current - 1) * pageSize, current * pageSize);

  const toggle = (key: Key) =>
    setSort((s) =>
      s && s.key === key
        ? { key, dir: s.dir === 'ascending' ? 'descending' : 'ascending' }
        : { key, dir: 'ascending' },
    );
  const ariaSort = (key: Key) => (sort && sort.key === key ? sort.dir : 'none');

  return (
    <div>
      <label htmlFor={filterId}>Filter</label>
      <input
        id={filterId}
        type="text"
        value={filter}
        onChange={(e) => {
          setFilter(e.target.value);
          setPage(1);
        }}
      />
      <table>
        <thead>
          <tr>
            <th aria-sort={ariaSort('name')}>
              <button type="button" onClick={() => toggle('name')}>
                Name
              </button>
            </th>
            <th aria-sort={ariaSort('age')}>
              <button type="button" onClick={() => toggle('age')}>
                Age
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => (
            <tr key={r.id}>
              <td>{r.name}</td>
              <td>{r.age}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <button type="button" disabled={current <= 1} onClick={() => setPage(current - 1)}>
        Previous
      </button>
      <span>
        Page {current} of {pageCount}
      </span>
      <button type="button" disabled={current >= pageCount} onClick={() => setPage(current + 1)}>
        Next
      </button>
    </div>
  );
}
