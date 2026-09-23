import { useState } from 'react';

// Wrong (strict): indexes a row with a `'name' | 'age'` key and subtracts,
// but `row.name` is a string.
type Row = { id: number; name: string; age: number };

export function DataTable({ rows, pageSize }: { rows: readonly Row[]; pageSize: number }) {
  const [filter, setFilter] = useState('');
  const [sortKey, setSortKey] = useState<'name' | 'age' | null>(null);
  const [asc, setAsc] = useState(true);
  const [page, setPage] = useState(1);

  let visible = rows.filter((r) => r.name.toLowerCase().includes(filter.toLowerCase()));
  if (sortKey) {
    visible = [...visible].sort((a, b) => (asc ? a[sortKey] - b[sortKey] : b[sortKey] - a[sortKey]));
  }
  const pageCount = Math.max(1, Math.ceil(visible.length / pageSize));
  const shown = visible.slice((page - 1) * pageSize, page * pageSize);

  const click = (key: 'name' | 'age') => {
    if (sortKey === key) setAsc(!asc);
    else {
      setSortKey(key);
      setAsc(true);
    }
  };
  const ariaSort = (key: 'name' | 'age') => (sortKey !== key ? 'none' : asc ? 'ascending' : 'descending');

  return (
    <div>
      <label htmlFor="dt-filter">Filter</label>
      <input id="dt-filter" value={filter} onChange={(e) => { setFilter(e.target.value); setPage(1); }} />
      <table>
        <thead>
          <tr>
            <th aria-sort={ariaSort('name')}><button onClick={() => click('name')}>Name</button></th>
            <th aria-sort={ariaSort('age')}><button onClick={() => click('age')}>Age</button></th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => (
            <tr key={r.id}><td>{r.name}</td><td>{r.age}</td></tr>
          ))}
        </tbody>
      </table>
      <button disabled={page === 1} onClick={() => setPage(page - 1)}>Previous</button>
      <span>Page {page} of {pageCount}</span>
      <button disabled={page === pageCount} onClick={() => setPage(page + 1)}>Next</button>
    </div>
  );
}
