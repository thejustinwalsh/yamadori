import { Component, Suspense, use, type ReactNode } from 'react';

type User = { name: string };

export function UserName({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise);
  return <h2>{user.name}</h2>;
}

class LoadBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  render() {
    if (this.state.failed) return <p role="alert">Could not load user</p>;
    return this.props.children;
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
