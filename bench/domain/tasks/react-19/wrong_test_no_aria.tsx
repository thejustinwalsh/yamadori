// Wrong: messages are shown but never wired to the inputs (no aria-invalid / aria-describedby).
import { useId, useState, type FormEvent } from 'react';

type Field = 'email' | 'password' | 'confirm';

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function validate(v: Record<Field, string>): Record<Field, string | null> {
  return {
    email: EMAIL_RE.test(v.email) ? null : 'Enter a valid email',
    password: v.password.length >= 8 ? null : 'Password must be at least 8 characters',
    confirm: v.confirm === v.password ? null : 'Passwords do not match',
  };
}

export function SignupForm({
  onSubmit,
}: {
  onSubmit: (values: { email: string; password: string }) => void;
}) {
  const id = useId();
  const [values, setValues] = useState<Record<Field, string>>({ email: '', password: '', confirm: '' });
  const [touched, setTouched] = useState<Record<Field, boolean>>({
    email: false,
    password: false,
    confirm: false,
  });
  const [submitted, setSubmitted] = useState(false);

  const errors = validate(values);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitted(true);
    if (errors.email || errors.password || errors.confirm) return;
    onSubmit({ email: values.email, password: values.password });
  };

  const field = (name: Field, label: string, type: string) => {
    const inputId = `${id}-${name}`;
    const errorId = `${id}-${name}-error`;
    const error = submitted || touched[name] ? errors[name] : null;
    return (
      <div>
        <label htmlFor={inputId}>{label}</label>
        <input
          id={inputId}
          type={type}
          value={values[name]}
          onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
          onBlur={() => setTouched((t) => ({ ...t, [name]: true }))}
        />
        {error && (
          <p id={errorId} role="alert">
            {error}
          </p>
        )}
      </div>
    );
  };

  return (
    <form noValidate onSubmit={handleSubmit}>
      {field('email', 'Email', 'email')}
      {field('password', 'Password', 'password')}
      {field('confirm', 'Confirm password', 'password')}
      <button type="submit">Sign up</button>
    </form>
  );
}
