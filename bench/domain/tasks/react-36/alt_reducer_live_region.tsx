import { Suspense, use, useReducer, useTransition, type ChangeEvent } from 'react';

type State = { text: string; query: string };
type Action = { type: 'typed'; text: string } | { type: 'search'; query: string };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'typed':
      return { ...state, text: action.text };
    case 'search':
      return { ...state, query: action.query };
  }
}

// The child asks for the promise itself; the status region is always
// mounted and only its text changes.
function ResultList({
  query,
  getResults,
}: {
  query: string;
  getResults: (query: string) => Promise<string[]>;
}) {
  const results = use(getResults(query));
  return (
    <ul aria-label="Results">
      {results.map((r, i) => (
        <li key={`${i}:${r}`}>{r}</li>
      ))}
    </ul>
  );
}

export function SlowSearch({
  getResults,
}: {
  getResults: (query: string) => Promise<string[]>;
}) {
  const [state, dispatch] = useReducer(reducer, { text: '', query: '' });
  const [pending, startTransition] = useTransition();

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const text = e.currentTarget.value;
    dispatch({ type: 'typed', text });
    startTransition(() => {
      dispatch({ type: 'search', query: text });
    });
  }

  return (
    <section>
      <label htmlFor="slow-search-input">Search</label>
      <input id="slow-search-input" type="search" value={state.text} onChange={onChange} />
      <div role="status" aria-live="polite">
        {pending ? 'Updating…' : ''}
      </div>
      <Suspense fallback={<span>Loading…</span>}>
        <ResultList query={state.query} getResults={getResults} />
      </Suspense>
    </section>
  );
}
