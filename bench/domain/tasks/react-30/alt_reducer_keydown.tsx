import { useReducer, useState } from 'react';

// Reducer-held list keyed by a generated string id, no <form> (Enter handled
// in keydown), aria-label on the checkbox instead of a wrapping label, a
// visible "Delete <text>" button text, and markup inside the counter.
type Todo = { key: string; text: string; completed: boolean };
type Action =
  | { type: 'add'; text: string }
  | { type: 'toggle'; key: string }
  | { type: 'delete'; key: string };

let seq = 0;

function todosReducer(list: Todo[], action: Action): Todo[] {
  switch (action.type) {
    case 'add':
      seq += 1;
      return list.concat({ key: `todo-${seq}`, text: action.text, completed: false });
    case 'toggle':
      return list.map((t) => (t.key === action.key ? { ...t, completed: !t.completed } : t));
    case 'delete':
      return list.filter((t) => t.key !== action.key);
  }
}

const views = {
  All: () => true,
  Active: (t: Todo) => !t.completed,
  Completed: (t: Todo) => t.completed,
} as const;
type View = keyof typeof views;

export function TodoApp() {
  const [list, dispatch] = useReducer(todosReducer, []);
  const [text, setText] = useState('');
  const [view, setView] = useState<View>('All');

  const submit = () => {
    const clean = text.trim();
    if (clean.length > 0) {
      dispatch({ type: 'add', text: clean });
      setText('');
    }
  };

  const remaining = list.reduce((n, t) => (t.completed ? n : n + 1), 0);

  return (
    <div>
      <label htmlFor="new-todo">New todo</label>
      <input
        id="new-todo"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') submit();
        }}
      />
      <button onClick={submit}>Add</button>
      <nav>
        {(Object.keys(views) as View[]).map((v) => (
          <button key={v} aria-pressed={v === view ? 'true' : 'false'} onClick={() => setView(v)}>
            {v}
          </button>
        ))}
      </nav>
      <ol>
        {list.filter(views[view]).map((t) => (
          <li key={t.key}>
            <input
              type="checkbox"
              aria-label={t.text}
              checked={t.completed}
              onChange={() => dispatch({ type: 'toggle', key: t.key })}
            />
            <span style={{ textDecoration: t.completed ? 'line-through' : 'none' }}>{t.text}</span>
            <button onClick={() => dispatch({ type: 'delete', key: t.key })}>Delete {t.text}</button>
          </li>
        ))}
      </ol>
      <footer>
        <span>
          <strong>{remaining}</strong> {remaining === 1 ? 'item' : 'items'} left
        </span>
      </footer>
    </div>
  );
}
