import { useState } from 'react';
import type * as React from 'react';

// One useState per value, blur handled once at the form (focusout bubbles in
// React), labels wrap their inputs, messages are <span>s with static ids.
export function SignupForm(props: {
  onSubmit: (values: { email: string; password: string }) => void;
}): React.ReactElement {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [visited, setVisited] = useState<string[]>([]);
  const [tried, setTried] = useState(false);

  const problems: Record<string, string | undefined> = {
    email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) ? undefined : 'Enter a valid email',
    password: password.length < 8 ? 'Password must be at least 8 characters' : undefined,
    confirm: confirm !== password ? 'Passwords do not match' : undefined,
  };
  const visible = (name: string) => (tried || visited.includes(name) ? problems[name] : undefined);

  function input(name: string, label: string, value: string, set: (v: string) => void) {
    const msg = visible(name);
    return (
      <p>
        <label>
          {label}
          <input
            name={name}
            type={name === 'email' ? 'text' : 'password'}
            value={value}
            onChange={(e) => set(e.currentTarget.value)}
            aria-invalid={msg ? 'true' : 'false'}
            aria-describedby={msg ? `signup-${name}-msg` : undefined}
          />
        </label>
        {msg ? <span id={`signup-${name}-msg`}>{msg}</span> : null}
      </p>
    );
  }

  return (
    <form
      onBlur={(e) => {
        const t = e.target as EventTarget;
        const name = t instanceof HTMLInputElement ? t.name : '';
        if (name && !visited.includes(name)) setVisited([...visited, name]);
      }}
      onSubmit={(e) => {
        e.preventDefault();
        setTried(true);
        if (Object.values(problems).every((p) => p === undefined)) {
          props.onSubmit({ email, password });
        }
      }}
    >
      {input('email', 'Email', email, setEmail)}
      {input('password', 'Password', password, setPassword)}
      {input('confirm', 'Confirm password', confirm, setConfirm)}
      <button>Sign up</button>
    </form>
  );
}
