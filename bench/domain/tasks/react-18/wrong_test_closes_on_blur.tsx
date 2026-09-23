// Wrong: the list closes on input blur, so the mousedown of a click on an option unmounts it before the click lands.
import { useId, useState, type KeyboardEvent } from 'react';

export function Combobox({
  label,
  options,
  onSelect,
}: {
  label: string;
  options: string[];
  onSelect: (value: string) => void;
}) {
  const id = useId();
  const inputId = `${id}-input`;
  const listId = `${id}-list`;
  const optionId = (i: number) => `${id}-opt-${i}`;

  const [text, setText] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);

  const needle = text.toLowerCase();
  const matches = options.filter((o) => o.toLowerCase().includes(needle));

  const pick = (value: string) => {
    setText(value);
    setOpen(false);
    setActive(-1);
    onSelect(value);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (!open) return;
    const n = matches.length;
    if (e.key === 'ArrowDown' && n > 0) {
      e.preventDefault();
      setActive((a) => (a + 1) % n);
    } else if (e.key === 'ArrowUp' && n > 0) {
      e.preventDefault();
      setActive((a) => (a <= 0 ? n - 1 : a - 1));
    } else if (e.key === 'Enter') {
      if (active >= 0 && active < n) {
        e.preventDefault();
        pick(matches[active]);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
      setActive(-1);
    }
  };

  return (
    <div>
      <label htmlFor={inputId}>{label}</label>
      <input
        id={inputId}
        type="text"
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls={listId}
        aria-activedescendant={open && active >= 0 ? optionId(active) : undefined}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setOpen(true);
          setActive(-1);
        }}
        onKeyDown={onKeyDown}
        onBlur={() => setOpen(false)}
      />
      {open && (
        <ul role="listbox" id={listId}>
          {matches.length === 0 ? (
            <li>No results</li>
          ) : (
            matches.map((o, i) => (
              <li
                key={o}
                id={optionId(i)}
                role="option"
                aria-selected={i === active}
                onClick={() => pick(o)}
              >
                {o}
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
