// Wrong: the editor is keyed by name, so two contacts with the same name share one draft.
import { useState } from 'react';

type Contact = { id: string; name: string; email: string };

function Editor({ contact, onSave }: { contact: Contact; onSave: (c: Contact) => void }) {
  const [name, setName] = useState(contact.name);
  const [email, setEmail] = useState(contact.email);
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSave({ id: contact.id, name, email });
      }}
    >
      <label>
        Name
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <label>
        Email
        <input value={email} onChange={(e) => setEmail(e.target.value)} />
      </label>
      <button type="submit">Save</button>
    </form>
  );
}

export function ContactManager({ contacts }: { contacts: Contact[] }) {
  const [saved, setSaved] = useState<Contact[]>(contacts);
  const [selectedId, setSelectedId] = useState<string | undefined>(contacts[0]?.id);
  const selected = saved.find((c) => c.id === selectedId);

  return (
    <div>
      <ul>
        {saved.map((c) => (
          <li key={c.id}>
            <button type="button" onClick={() => setSelectedId(c.id)}>
              {c.name} ({c.email})
            </button>
          </li>
        ))}
      </ul>
      {selected && (
        <Editor
          key={selected.name}
          contact={selected}
          onSave={(next) => setSaved((all) => all.map((c) => (c.id === next.id ? next : c)))}
        />
      )}
    </div>
  );
}
