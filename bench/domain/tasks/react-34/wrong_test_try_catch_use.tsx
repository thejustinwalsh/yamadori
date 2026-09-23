// Wrong: wraps use() in try/catch to handle rejection, which also swallows React's suspension signal, so the card shows the alert while still loading.
import { Suspense, use } from 'react';

type User = { name: string };

export function UserName({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise);
  return <h2>{user.name}</h2>;
}

function SafeUserName({ userPromise }: { userPromise: Promise<User> }) {
  try {
    const user = use(userPromise);
    return <h2>{user.name}</h2>;
  } catch {
    return <p role="alert">Could not load user</p>;
  }
}

export function UserCard({ userPromise }: { userPromise: Promise<User> }) {
  return (
    <Suspense fallback={<p>Loading user…</p>}>
      <SafeUserName userPromise={userPromise} />
    </Suspense>
  );
}
