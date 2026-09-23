// Wrong: reset() only forces a re-render; the error stays in state, so the fallback never goes away.
import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  fallback: (error: Error, reset: () => void) => ReactNode;
  onError?: (error: Error, info: ErrorInfo) => void;
  resetKeys?: readonly unknown[];
  children: ReactNode;
}

export class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.props.onError?.(error, info);
  }

  componentDidUpdate(prev: Props) {
    const a = prev.resetKeys ?? [];
    const b = this.props.resetKeys ?? [];
    if (this.state.error && (a.length !== b.length || a.some((v, i) => !Object.is(v, b[i])))) {
      this.reset();
    }
  }

  reset = () => {
    this.forceUpdate();
  };

  render() {
    return this.state.error ? this.props.fallback(this.state.error, this.reset) : this.props.children;
  }
}
