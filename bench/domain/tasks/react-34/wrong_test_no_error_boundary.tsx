// Wrong: UserCard handles loading but not failure; a rejection escapes to whatever boundary is above it.
import { Suspense, use } from 'react';

type User = { name: string };

export function UserName({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise);
  return <h2>{user.name}</h2>;
}

export function UserCard({ userPromise }: { userPromise: Promise<User> }) {
  return (
    <Suspense fallback={<p>Loading user…</p>}>
      <UserName userPromise={userPromise} />
    </Suspense>
  );
}
