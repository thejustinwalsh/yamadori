import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { Component, Suspense, type ReactNode } from 'react';
import { UserName, UserCard } from './solution';
import { deferred, settle, fail, flush } from './helpers';

type User = { name: string };

// The test's own error boundary, above the component under test.
const outerCaught: unknown[] = [];
class OuterBoundary extends Component<{ children: ReactNode }, { error: unknown }> {
  state: { error: unknown } = { error: null };
  static getDerivedStateFromError(error: unknown) {
    return { error };
  }
  componentDidCatch(error: unknown) {
    outerCaught.push(error);
  }
  render() {
    if (this.state.error !== null) {
      return <p>outer error: {(this.state.error as Error).message}</p>;
    }
    return this.props.children;
  }
}

function Outer({ children }: { children: ReactNode }) {
  return (
    <OuterBoundary>
      <Suspense fallback={<p>outer fallback</p>}>{children}</Suspense>
    </OuterBoundary>
  );
}

const heading = () => screen.queryByRole('heading');

// A component that suspends on use() must be rendered inside an awaited
// act(); RTL's synchronous render leaves the suspended tree un-retried.
async function mount(ui: ReactNode) {
  await act(async () => {
    render(ui);
  });
}

describe('UserName / UserCard', () => {
  beforeEach(() => {
    outerCaught.length = 0;
    // React reports errors caught by boundaries on console.error; keep the
    // output readable.
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
  });

  it('UserName suspends to the enclosing Suspense boundary while pending', async () => {
    const d = deferred<User>();
    await mount(
      <Outer>
        <UserName userPromise={d.promise} />
      </Outer>,
    );
    await flush();
    expect(screen.getByText('outer fallback')).toBeTruthy();
    expect(heading()).toBeNull();
    await settle(d, { name: 'Ada Lovelace' });
  });

  it('UserName renders the name as a heading once resolved', async () => {
    const d = deferred<User>();
    await mount(
      <Outer>
        <UserName userPromise={d.promise} />
      </Outer>,
    );
    await settle(d, { name: 'Ada Lovelace' });
    await flush();
    expect(screen.getByRole('heading', { name: 'Ada Lovelace' }).textContent).toBe('Ada Lovelace');
    expect(screen.queryByText('outer fallback')).toBeNull();
  });

  it('UserName lets a rejection propagate to the error boundary above it', async () => {
    const d = deferred<User>();
    await mount(
      <Outer>
        <UserName userPromise={d.promise} />
      </Outer>,
    );
    await fail(d, new Error('404 user'));
    await flush();
    expect(screen.getByText('outer error: 404 user')).toBeTruthy();
    expect(heading()).toBeNull();
  });

  it('UserCard shows its own loading fallback, not the outer one', async () => {
    const d = deferred<User>();
    await mount(
      <Outer>
        <UserCard userPromise={d.promise} />
      </Outer>,
    );
    await flush();
    expect(screen.getByText('Loading user…')).toBeTruthy();
    expect(screen.queryByText('outer fallback')).toBeNull();
    expect(heading()).toBeNull();
    await settle(d, { name: 'Grace Hopper' });
  });

  it('UserCard shows the name once resolved', async () => {
    const d = deferred<User>();
    await mount(
      <Outer>
        <UserCard userPromise={d.promise} />
      </Outer>,
    );
    await settle(d, { name: 'Grace Hopper' });
    await flush();
    expect(screen.getByRole('heading', { name: 'Grace Hopper' })).toBeTruthy();
    expect(screen.queryByText('Loading user…')).toBeNull();
  });

  it('UserCard contains a rejection and shows an alert', async () => {
    const d = deferred<User>();
    await mount(
      <Outer>
        <UserCard userPromise={d.promise} />
      </Outer>,
    );
    await fail(d, new Error('500'));
    await flush();
    expect(screen.getByRole('alert').textContent).toBe('Could not load user');
    expect(outerCaught).toEqual([]);
    expect(screen.queryByText(/outer error/)).toBeNull();
    expect(screen.queryByText('Loading user…')).toBeNull();
    expect(heading()).toBeNull();
  });

  it('cards load and fail independently', async () => {
    const ok = deferred<User>();
    const bad = deferred<User>();
    await mount(
      <Outer>
        <UserCard userPromise={ok.promise} />
        <UserCard userPromise={bad.promise} />
      </Outer>,
    );
    await fail(bad, new Error('gone'));
    await flush();
    expect(screen.getByRole('alert').textContent).toBe('Could not load user');
    expect(screen.getByText('Loading user…')).toBeTruthy();
    await settle(ok, { name: 'Alan Turing' });
    await flush();
    expect(screen.getByRole('heading', { name: 'Alan Turing' })).toBeTruthy();
    expect(screen.getAllByRole('alert').length).toBe(1);
    expect(outerCaught).toEqual([]);
  });
});
