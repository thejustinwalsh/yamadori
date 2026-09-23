import * as React from 'react';

// Tracks the last-seen resetKeys in state and resets from
// getDerivedStateFromProps, so the fallback never renders with stale keys;
// the error is boxed so a thrown non-Error value would still count.
type Props = {
  fallback: (error: Error, reset: () => void) => React.ReactNode;
  onError?: (error: Error, info: React.ErrorInfo) => void;
  resetKeys?: readonly unknown[];
  children: React.ReactNode;
};

type State = {
  caught: { error: Error } | undefined;
  keys: readonly unknown[];
};

const same = (a: readonly unknown[], b: readonly unknown[]) =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { caught: undefined, keys: props.resetKeys ?? [] };
    this.clear = this.clear.bind(this);
  }

  static getDerivedStateFromError(thrown: unknown): Partial<State> {
    return { caught: { error: thrown instanceof Error ? thrown : new Error(String(thrown)) } };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    const keys = props.resetKeys ?? [];
    if (same(keys, state.keys)) return null;
    return state.caught ? { keys, caught: undefined } : { keys };
  }

  override componentDidCatch(error: Error, info: React.ErrorInfo) {
    if (this.props.onError) this.props.onError(error, info);
  }

  clear() {
    this.setState({ caught: undefined });
  }

  override render() {
    const { caught } = this.state;
    return caught ? this.props.fallback(caught.error, this.clear) : this.props.children;
  }
}
