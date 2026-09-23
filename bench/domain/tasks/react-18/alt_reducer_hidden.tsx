import { useId, useReducer } from 'react';
import type * as React from 'react';

// A reducer owns the state; the active option is remembered by value, not by
// index; the listbox stays mounted with `hidden`; the name comes from
// aria-labelledby.
type State = { text: string; open: boolean; active: string | null };
type Action =
  | { kind: 'type'; text: string }
  | { kind: 'move'; delta: 1 | -1; list: string[] }
  | { kind: 'close' }
  | { kind: 'picked'; value: string };

function reduce(s: State, a: Action): State {
  switch (a.kind) {
    case 'type':
      return { text: a.text, open: true, active: null };
    case 'move': {
      if (!s.open || a.list.length === 0) return s;
      const at = s.active === null ? -1 : a.list.indexOf(s.active);
      let next: number;
      if (at < 0) next = a.delta === 1 ? 0 : a.list.length - 1;
      else next = (at + a.delta + a.list.length) % a.list.length;
      return { ...s, active: a.list[next] };
    }
    case 'close':
      return { ...s, open: false, active: null };
    case 'picked':
      return { text: a.value, open: false, active: null };
  }
}

export function Combobox(props: {
  label: string;
  options: string[];
  onSelect: (value: string) => void;
}): React.ReactElement {
  const uid = useId();
  const [s, dispatch] = useReducer(reduce, { text: '', open: false, active: null });
  const q = s.text.toLocaleLowerCase();
  const list = props.options.filter((o) => o.toLocaleLowerCase().indexOf(q) !== -1);
  const activeIndex = s.active === null ? -1 : list.indexOf(s.active);

  function choose(value: string) {
    dispatch({ kind: 'picked', value });
    props.onSelect(value);
  }

  return (
    <div>
      <span id={`${uid}lbl`}>{props.label}</span>
      <input
        role="combobox"
        aria-labelledby={`${uid}lbl`}
        aria-autocomplete="list"
        aria-expanded={s.open ? 'true' : 'false'}
        aria-controls={`${uid}lb`}
        aria-activedescendant={s.open && activeIndex >= 0 ? `${uid}o${activeIndex}` : ''}
        value={s.text}
        onChange={(e) => dispatch({ kind: 'type', text: e.currentTarget.value })}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            dispatch({ kind: 'move', delta: e.key === 'ArrowDown' ? 1 : -1, list });
          } else if (e.key === 'Enter' && s.open && activeIndex >= 0) {
            choose(list[activeIndex]);
          } else if (e.key === 'Escape') {
            dispatch({ kind: 'close' });
          }
        }}
      />
      <div role="listbox" id={`${uid}lb`} hidden={!s.open}>
        {list.map((o, i) => (
          <div
            key={o}
            role="option"
            id={`${uid}o${i}`}
            aria-selected={i === activeIndex ? 'true' : 'false'}
            onPointerDown={(e) => e.preventDefault()}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => choose(o)}
          >
            {o}
          </div>
        ))}
      </div>
      {s.open && list.length === 0 && <p>No results</p>}
    </div>
  );
}
