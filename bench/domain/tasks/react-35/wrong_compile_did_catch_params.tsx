// Wrong (strict): componentDidCatch's parameters are not typed; class methods are not contextually typed by the base class, so they are implicit any.
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

  componentDidCatch(error, info) {
    this.props.onError?.(error, info);
  }

  componentDidUpdate(prev: Props, prevState: { error: Error | null }) {
    const a = prev.resetKeys ?? [];
    const b = this.props.resetKeys ?? [];
    if (this.state.error && prevState.error && (a.length !== b.length || a.some((v, i) => !Object.is(v, b[i])))) {
      this.reset();
    }
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    return this.state.error ? this.props.fallback(this.state.error, this.reset) : this.props.children;
  }
}
