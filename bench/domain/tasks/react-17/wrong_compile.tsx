import { useState, type ReactNode } from 'react';

// Wrong (strict): state typed `string` but initialised with null.
export function Accordion({
  items,
  allowMultiple = false,
}: {
  items: { id: string; title: string; content: ReactNode }[];
  allowMultiple?: boolean;
}) {
  const [openId, setOpenId] = useState<string>(null);
  const [openIds, setOpenIds] = useState<string[]>([]);

  const isOpen = (id: string) => (allowMultiple ? openIds.includes(id) : openId === id);
  const toggle = (id: string) => {
    if (allowMultiple) {
      setOpenIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]));
    } else {
      setOpenId((cur) => (cur === id ? null : id));
    }
  };

  return (
    <div>
      {items.map((item) => (
        <div key={item.id}>
          <button
            id={`acc-h-${item.id}`}
            aria-expanded={isOpen(item.id)}
            aria-controls={`acc-p-${item.id}`}
            onClick={() => toggle(item.id)}
          >
            {item.title}
          </button>
          {isOpen(item.id) && (
            <div role="region" id={`acc-p-${item.id}`} aria-labelledby={`acc-h-${item.id}`}>
              {item.content}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
