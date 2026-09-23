import { useState } from 'react';

// Wrong (strict): reads `.value` off `e.target` in a keyboard handler, where
// the target is only an EventTarget.
export function Combobox({
  label,
  options,
  onSelect,
}: {
  label: string;
  options: string[];
  onSelect: (value: string) => void;
}) {
  const [text, setText] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const matches = options.filter((o) => o.toLowerCase().includes(text.toLowerCase()));

  return (
    <div>
      <label htmlFor="combo-input">{label}</label>
      <input
        id="combo-input"
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls="combo-list"
        aria-activedescendant={active >= 0 ? `combo-opt-${active}` : undefined}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setOpen(true);
          setActive(-1);
        }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') setActive((a) => (a + 1) % matches.length);
          else if (e.key === 'ArrowUp') setActive((a) => (a <= 0 ? matches.length - 1 : a - 1));
          else if (e.key === 'Escape') setOpen(false);
          else if (e.key === 'Enter' && active >= 0) {
            const value = matches[active];
            setText(value);
            setOpen(false);
            onSelect(e.target.value === value ? value : value);
          }
        }}
      />
      {open && (
        <ul role="listbox" id="combo-list">
          {matches.length === 0 && <li>No results</li>}
          {matches.map((o, i) => (
            <li key={o} id={`combo-opt-${i}`} role="option" aria-selected={i === active}
              onClick={() => { setText(o); setOpen(false); onSelect(o); }}>
              {o}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
