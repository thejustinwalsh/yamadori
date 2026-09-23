import { useId, useState } from 'react';
import type * as React from 'react';

// Panels are always mounted and toggled with `hidden`; state is a boolean per
// index rather than a set of ids.
export function Accordion(props: {
  items: { id: string; title: string; content: React.ReactNode }[];
  allowMultiple?: boolean;
}): React.ReactElement {
  const uid = useId();
  const [flags, setFlags] = useState<boolean[]>(() => props.items.map(() => false));

  function activate(index: number) {
    setFlags((cur) =>
      props.items.map((_, i) => {
        if (i === index) return !cur[i];
        return props.allowMultiple ? Boolean(cur[i]) : false;
      }),
    );
  }

  return (
    <dl>
      {props.items.map((item, i) => (
        <div key={item.id}>
          <dt>
            <button
              id={`${uid}btn${i}`}
              aria-controls={`${uid}pnl${i}`}
              aria-expanded={flags[i] ? 'true' : 'false'}
              onClick={() => activate(i)}
            >
              {item.title}
            </button>
          </dt>
          <dd role="region" id={`${uid}pnl${i}`} aria-labelledby={`${uid}btn${i}`} hidden={!flags[i]}>
            {item.content}
          </dd>
        </div>
      ))}
    </dl>
  );
}
