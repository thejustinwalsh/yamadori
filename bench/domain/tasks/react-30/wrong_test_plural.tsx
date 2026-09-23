// Wrong: the counter always says "items", so one active todo reads "1 items left".
import { useRef, useState } from 'react';
import type { FormEvent } from 'react';

type Todo = { id: number; text: string; done: boolean };
type Filter = 'All' | 'Active' | 'Completed';

const FILTERS: Filter[] = ['All', 'Active', 'Completed'];

export function TodoApp() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [draft, setDraft] = useState('');
  const [filter, setFilter] = useState<Filter>('All');
  const nextId = useRef(1);

  const add = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setTodos((prev) => [...prev, { id: nextId.current++, text, done: false }]);
    setDraft('');
  };

  const toggle = (id: number) =>
    setTodos((prev) => prev.map((t) => (t.id === id ? { ...t, done: !t.done } : t)));
  const remove = (id: number) => setTodos((prev) => prev.filter((t) => t.id !== id));

  const visible = todos.filter((t) =>
    filter === 'All' ? true : filter === 'Active' ? !t.done : t.done,
  );
  const left = todos.filter((t) => !t.done).length;

  return (
    <section>
      <form onSubmit={add}>
        <input aria-label="New todo" value={draft} onChange={(e) => setDraft(e.target.value)} />
        <button type="submit">Add</button>
      </form>
      <ul>
        {visible.map((t) => (
          <li key={t.id}>
            <label>
              <input type="checkbox" checked={t.done} onChange={() => toggle(t.id)} />
              {t.text}
            </label>
            <button type="button" aria-label={`Delete ${t.text}`} onClick={() => remove(t.id)}>
              ×
            </button>
          </li>
        ))}
      </ul>
      <p>{`${left} items left`}</p>
      <div>
        {FILTERS.map((f) => (
          <button key={f} type="button" aria-pressed={filter === f} onClick={() => setFilter(f)}>
            {f}
          </button>
        ))}
      </div>
    </section>
  );
}
