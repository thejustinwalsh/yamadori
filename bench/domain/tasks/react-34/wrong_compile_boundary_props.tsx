// Wrong (strict): the boundary extends Component with no props type, so this.props.children does not exist.
import { Component, Suspense, use } from 'react';

type User = { name: string };

export function UserName({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise);
  return <h2>{user.name}</h2>;
}

class LoadBoundary extends Component {
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
