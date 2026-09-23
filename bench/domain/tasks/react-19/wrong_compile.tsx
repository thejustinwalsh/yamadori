import { useState, type FormEvent } from 'react';

// Wrong (strict): the errors object is inferred as `{}`, so `errors.email`
// and friends do not exist on its type.
export function SignupForm({
  onSubmit,
}: {
  onSubmit: (values: { email: string; password: string }) => void;
}) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [errors, setErrors] = useState({});

  const check = () => {
    const next = {};
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) next.email = 'Enter a valid email';
    if (password.length < 8) next.password = 'Password must be at least 8 characters';
    if (confirm !== password) next.confirm = 'Passwords do not match';
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (check()) onSubmit({ email, password });
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="su-email">Email</label>
      <input id="su-email" value={email} onChange={(e) => setEmail(e.target.value)} onBlur={check}
        aria-invalid={errors.email ? true : undefined} aria-describedby={errors.email ? 'su-email-err' : undefined} />
      {errors.email && <p id="su-email-err">{errors.email}</p>}
      <label htmlFor="su-pw">Password</label>
      <input id="su-pw" type="password" value={password} onChange={(e) => setPassword(e.target.value)} onBlur={check}
        aria-invalid={errors.password ? true : undefined} aria-describedby={errors.password ? 'su-pw-err' : undefined} />
      {errors.password && <p id="su-pw-err">{errors.password}</p>}
      <label htmlFor="su-confirm">Confirm password</label>
      <input id="su-confirm" type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} onBlur={check}
        aria-invalid={errors.confirm ? true : undefined} aria-describedby={errors.confirm ? 'su-confirm-err' : undefined} />
      {errors.confirm && <p id="su-confirm-err">{errors.confirm}</p>}
      <button type="submit">Sign up</button>
    </form>
  );
}
