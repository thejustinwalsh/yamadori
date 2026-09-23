import * as React from 'react';

// A generic boundary with a fallback prop, placed INSIDE the Suspense
// boundary, and use() called in a nested helper that UserName renders.
interface User {
  name: string;
}

interface CatchProps {
  fallback: React.ReactNode;
  children?: React.ReactNode;
}

class Catch extends React.Component<CatchProps> {
  override state: { error: unknown } = { error: null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error ?? new Error('rejected') };
  }

  override componentDidCatch(): void {
    // nothing to report; the fallback says it all
  }

  override render(): React.ReactNode {
    return this.state.error !== null ? this.props.fallback : this.props.children;
  }
}

function Title({ source }: { source: Promise<User> }) {
  const { name } = React.use(source);
  return <h3 className="user-name">{name}</h3>;
}

export function UserName(props: { userPromise: Promise<User> }) {
  return <Title source={props.userPromise} />;
}

export function UserCard(props: { userPromise: Promise<User> }) {
  return (
    <article>
      <React.Suspense fallback={<span aria-live="polite">Loading user…</span>}>
        <Catch fallback={<div role="alert">Could not load user</div>}>
          <UserName userPromise={props.userPromise} />
        </Catch>
      </React.Suspense>
    </article>
  );
}
