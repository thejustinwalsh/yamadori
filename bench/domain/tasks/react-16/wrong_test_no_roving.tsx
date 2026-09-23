// Wrong: no roving tabindex -- every tab stays in the Tab order (tabIndex never set).
import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';

export function Tabs({
  tabs,
  defaultTabId,
}: {
  tabs: { id: string; label: string; content: ReactNode }[];
  defaultTabId?: string;
}) {
  const base = useId();
  const [selectedId, setSelectedId] = useState<string | undefined>(
    () => defaultTabId ?? tabs[0]?.id,
  );
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);

  const index = Math.max(0, tabs.findIndex((t) => t.id === selectedId));
  const selected = tabs[index];

  const tabId = (i: number) => `${base}-tab-${i}`;
  const panelId = (i: number) => `${base}-panel-${i}`;

  const go = (i: number) => {
    const n = tabs.length;
    const next = ((i % n) + n) % n;
    setSelectedId(tabs[next].id);
    buttons.current[next]?.focus();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    switch (e.key) {
      case 'ArrowRight':
        go(index + 1);
        break;
      case 'ArrowLeft':
        go(index - 1);
        break;
      case 'Home':
        go(0);
        break;
      case 'End':
        go(tabs.length - 1);
        break;
      default:
        return;
    }
    e.preventDefault();
  };

  return (
    <div>
      <div role="tablist">
        {tabs.map((t, i) => (
          <button
            key={t.id}
            ref={(el) => {
              buttons.current[i] = el;
            }}
            type="button"
            role="tab"
            id={tabId(i)}
            aria-controls={panelId(i)}
            aria-selected={i === index}
            onClick={() => setSelectedId(t.id)}
            onKeyDown={onKeyDown}
          >
            {t.label}
          </button>
        ))}
      </div>
      {selected && (
        <div role="tabpanel" id={panelId(index)} aria-labelledby={tabId(index)} tabIndex={0}>
          {selected.content}
        </div>
      )}
    </div>
  );
}
