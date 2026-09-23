// Wrong: pre-19 data loading with useEffect + then. UserName never suspends (no Suspense fallback shows) and a rejection never reaches an error boundary.
import { Component, Suspense, useEffect, useState, type ReactNode } from 'react';

type User = { name: string };

export function UserName({ userPromise }: { userPromise: Promise<User> }) {
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => {
    let live = true;
    userPromise.then(
      (u) => {
        if (live) setUser(u);
      },
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, [userPromise]);
  return user ? <h2>{user.name}</h2> : null;
}

class LoadBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? <p role="alert">Could not load user</p> : this.props.children;
  }
}

export function UserCard({ userPromise }: { userPromise: Promise<User> }) {
  return (
    <LoadBoundary>
      <Suspense fallback={<p>Loading user…</p>}>
        <UserName userPromise={userPromise} />
      </Suspense>
    </LoadBoundary>
  );
}
