import { useReducer, useState } from 'react';

interface Contact {
  id: string;
  name: string;
  email: string;
}

type Draft = { name: string; email: string };

// No key: the draft lives in the manager and is replaced during render
// whenever the selected id changes (the "store the previous id" pattern).
export function ContactManager({ contacts }: { contacts: Contact[] }) {
  const [saved, save] = useReducer(
    (list: Contact[], c: Contact) => list.map((x) => (x.id === c.id ? c : x)),
    contacts,
  );
  const [selectedId, setSelectedId] = useState(contacts.length > 0 ? contacts[0].id : '');
  const [draftFor, setDraftFor] = useState(selectedId);
  const current = saved.find((c) => c.id === selectedId) ?? null;
  const [draft, setDraft] = useState<Draft>({
    name: current?.name ?? '',
    email: current?.email ?? '',
  });

  if (draftFor !== selectedId) {
    setDraftFor(selectedId);
    setDraft({ name: current?.name ?? '', email: current?.email ?? '' });
  }

  return (
    <main>
      <nav>
        {saved.map((c) => (
          <button key={c.id} onClick={() => setSelectedId(c.id)}>
            {`${c.name} (${c.email})`}
          </button>
        ))}
      </nav>
      {current !== null && (
        <section>
          <label htmlFor="cm-name">Name</label>
          <input
            id="cm-name"
            value={draft.name}
            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
          />
          <label htmlFor="cm-email">Email</label>
          <input
            id="cm-email"
            value={draft.email}
            onChange={(e) => setDraft({ ...draft, email: e.target.value })}
          />
          <button onClick={() => save({ id: current.id, ...draft })}>Save</button>
        </section>
      )}
    </main>
  );
}
