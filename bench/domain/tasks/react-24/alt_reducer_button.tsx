import { useReducer, useRef } from 'react';
import type { DragEvent } from 'react';

// A real <button> that opens the picker, a wrapping <label>, a reducer, and
// the checks written as a classifier returning a reason or null.
type State = { over: boolean; rejected: { name: string; reason: string }[] };
type Action =
  | { type: 'over'; value: boolean }
  | { type: 'batch'; rejected: { name: string; reason: string }[] };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'over':
      return state.over === action.value ? state : { ...state, over: action.value };
    case 'batch':
      return { over: false, rejected: action.rejected };
  }
}

export function FileDropzone(props: {
  accept: string[];
  maxSizeBytes: number;
  onFiles: (files: File[]) => void;
}) {
  const [state, dispatch] = useReducer(reducer, { over: false, rejected: [] });
  const inputRef = useRef<HTMLInputElement>(null);

  const reason = (f: File): string | null => {
    const typeOk = props.accept.some((pattern) => {
      const [major, minor] = pattern.split('/');
      if (minor === '*') return f.type.split('/')[0] === major && f.type.includes('/');
      return f.type === pattern;
    });
    if (!typeOk) return 'unsupported type';
    if (f.size > props.maxSizeBytes) return 'too large';
    return null;
  };

  const take = (files: FileList | File[]) => {
    const accepted: File[] = [];
    const rejected: { name: string; reason: string }[] = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const why = reason(f);
      if (why === null) accepted.push(f);
      else rejected.push({ name: f.name, reason: why });
    }
    dispatch({ type: 'batch', rejected });
    if (accepted.length) props.onFiles(accepted);
  };

  const over = (e: DragEvent) => {
    e.preventDefault();
    dispatch({ type: 'over', value: true });
  };

  return (
    <section>
      <button
        type="button"
        aria-label="Upload files"
        onClick={() => inputRef.current?.click()}
        onDragEnter={over}
        onDragOver={over}
        onDragLeave={() => dispatch({ type: 'over', value: false })}
        onDrop={(e) => {
          e.preventDefault();
          take(e.dataTransfer.files);
        }}
      >
        {state.over ? <strong>Drop files here</strong> : <span>Add files</span>}
      </button>
      <label style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden' }}>
        Choose files
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={props.accept.join(',')}
          onChange={(e) => {
            const files = e.currentTarget.files;
            if (files) take(Array.from(files));
          }}
        />
      </label>
      <ul>
        {state.rejected.map((r, i) => (
          <li key={`${r.name}-${i}`} role="alert">
            {r.name}: {r.reason}
          </li>
        ))}
      </ul>
    </section>
  );
}
