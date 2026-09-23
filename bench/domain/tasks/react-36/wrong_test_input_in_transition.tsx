// Wrong: the input value is updated inside the transition too, so it lags while results load.
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

  return (
    <div>
      <label>
        Search
        <input
          value={text}
          onChange={(e) => {
            const next = e.target.value;
            startTransition(() => {
              setText(next);
              setQuery(next);
            });
          }}
        />
      </label>
      {isPending && <p role="status">Updating…</p>}
      <Suspense fallback={<p>Loading…</p>}>
        <Results promise={getResults(query)} />
      </Suspense>
    </div>
  );
}
