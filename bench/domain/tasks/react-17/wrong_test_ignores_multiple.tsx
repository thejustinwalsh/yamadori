// Wrong: allowMultiple is ignored; opening a section always closes the others.
import { useId, useState, type ReactNode } from 'react';

export function Accordion({
  items,
  allowMultiple = false,
}: {
  items: { id: string; title: string; content: ReactNode }[];
  allowMultiple?: boolean;
}) {
  const base = useId();
  const [open, setOpen] = useState<ReadonlySet<string>>(() => new Set());

  const toggle = (id: string) =>
    setOpen((prev) => {
      if (prev.has(id)) {
        const next = new Set(prev);
        next.delete(id);
        return next;
      }
      return new Set([id]);
    });

  return (
    <div>
      {items.map((item, i) => {
        const isOpen = open.has(item.id);
        const headerId = `${base}-h${i}`;
        const panelId = `${base}-p${i}`;
        return (
          <div key={item.id}>
            <h3>
              <button
                type="button"
                id={headerId}
                aria-expanded={isOpen}
                aria-controls={panelId}
                onClick={() => toggle(item.id)}
              >
                {item.title}
              </button>
            </h3>
            {isOpen && (
              <div role="region" id={panelId} aria-labelledby={headerId}>
                {item.content}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
