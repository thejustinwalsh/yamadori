import { useId, useState, type ReactNode } from 'react';

// Wrong (strict): `tabs.find` may return undefined, and its `.content` is read
// unguarded; the key handler's event parameter is also left implicitly `any`.
export function Tabs({
  tabs,
  defaultTabId,
}: {
  tabs: { id: string; label: string; content: ReactNode }[];
  defaultTabId?: string;
}) {
  const base = useId();
  const [selectedId, setSelectedId] = useState(defaultTabId ?? tabs[0].id);
  const selected = tabs.find((t) => t.id === selectedId);
  const index = tabs.indexOf(selected);

  function onKeyDown(e) {
    const n = tabs.length;
    let next = index;
    if (e.key === 'ArrowRight') next = (index + 1) % n;
    else if (e.key === 'ArrowLeft') next = (index - 1 + n) % n;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = n - 1;
    else return;
    setSelectedId(tabs[next].id);
    document.getElementById(`${base}-tab-${next}`).focus();
  }

  return (
    <div>
      <div role="tablist">
        {tabs.map((t, i) => (
          <button
            key={t.id}
            role="tab"
            id={`${base}-tab-${i}`}
            aria-controls={`${base}-panel-${i}`}
            aria-selected={i === index}
            tabIndex={i === index ? 0 : -1}
            onClick={() => setSelectedId(t.id)}
            onKeyDown={onKeyDown}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${base}-panel-${index}`} aria-labelledby={`${base}-tab-${index}`}>
        {selected.content}
      </div>
    </div>
  );
}
