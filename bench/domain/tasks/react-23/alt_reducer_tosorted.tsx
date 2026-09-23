import { useReducer } from 'react';
import type * as React from 'react';

// One reducer for filter / sort / page; sorting uses the non-mutating
// `toSorted`, with an explicit original-index tiebreak; sorting resets the page.
type Row = { id: number; name: string; age: number };
type Col = 'name' | 'age';
type State = { q: string; col: Col | null; desc: boolean; page: number };
type Msg = { type: 'query'; q: string } | { type: 'sort'; col: Col } | { type: 'page'; to: number };

function update(s: State, m: Msg): State {
  if (m.type === 'query') return { ...s, q: m.q, page: 0 };
  if (m.type === 'sort')
    return s.col === m.col ? { ...s, desc: !s.desc, page: 0 } : { ...s, col: m.col, desc: false, page: 0 };
  return { ...s, page: m.to };
}

export function DataTable(props: {
  rows: readonly Row[];
  pageSize: number;
}): React.ReactElement {
  const [s, send] = useReducer(update, { q: '', col: null, desc: false, page: 0 });

  const indexed = props.rows.map((row, i) => ({ row, i }));
  const q = s.q.toLocaleLowerCase();
  let list = indexed.filter(({ row }) => row.name.toLocaleLowerCase().includes(q));
  const col = s.col;
  if (col !== null) {
    list = list.toSorted((a, b) => {
      const c = col === 'name' ? a.row.name.localeCompare(b.row.name) : a.row.age - b.row.age;
      return c !== 0 ? (s.desc ? -c : c) : a.i - b.i;
    });
  }
  const pages = Math.max(1, Math.ceil(list.length / props.pageSize));
  const page = Math.min(s.page, pages - 1);
  const slice = list.slice(page * props.pageSize, (page + 1) * props.pageSize);

  const sortAttr = (c: Col): React.AriaAttributes['aria-sort'] =>
    s.col !== c ? 'none' : s.desc ? 'descending' : 'ascending';

  return (
    <section>
      <label>
        Filter
        <input value={s.q} onChange={(e) => send({ type: 'query', q: e.currentTarget.value })} />
      </label>
      <table>
        <thead>
          <tr>
            {(['name', 'age'] as const).map((c) => (
              <th key={c} scope="col" aria-sort={sortAttr(c)}>
                <button onClick={() => send({ type: 'sort', col: c })}>{c === 'name' ? 'Name' : 'Age'}</button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {slice.map(({ row }) => (
            <tr key={row.id}>
              <td>{row.name}</td>
              <td>{String(row.age)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <nav>
        <button disabled={page === 0} onClick={() => send({ type: 'page', to: page - 1 })}>
          Previous
        </button>
        <p>{`Page ${page + 1} of ${pages}`}</p>
        <button disabled={page >= pages - 1} onClick={() => send({ type: 'page', to: page + 1 })}>
          Next
        </button>
      </nav>
    </section>
  );
}
