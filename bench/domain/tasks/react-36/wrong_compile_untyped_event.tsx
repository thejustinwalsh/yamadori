// Wrong: (strict) the change handler parameter is an implicit any.
import { Suspense, use, useState, useTransition } from 'react';

function Results({ promise }: { promise: Promise<string[]> }) {
  const items = use(promise);
  return (
    <ul>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

export function SlowSearch({
  getResults,
}: {
  getResults: (query: string) => Promise<string[]>;
}) {
  const [text, setText] = useState('');
  const [query, setQuery] = useState('');
  const [isPending, startTransition] = useTransition();

  function onChange(e) {
    const next = e.target.value;
    setText(next);
    startTransition(() => setQuery(next));
  }

  return (
    <div>
      <label>
        Search
        <input
          value={text}
          onChange={onChange}
        />
      </label>
      {isPending && <p role="status">Updating…</p>}
      <Suspense fallback={<p>Loading…</p>}>
        <Results promise={getResults(query)} />
      </Suspense>
    </div>
  );
}
