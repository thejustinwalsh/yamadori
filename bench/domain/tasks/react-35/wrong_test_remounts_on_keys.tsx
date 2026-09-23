// Wrong: keys the children on the joined resetKeys, so any resetKeys change remounts healthy children and throws away their state.
import { Component, Fragment, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  fallback: (error: Error, reset: () => void) => ReactNode;
  onError?: (error: Error, info: ErrorInfo) => void;
  resetKeys?: readonly unknown[];
  children: ReactNode;
}

interface State {
  error: Error | null;
  key: string;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, key: (this.props.resetKeys ?? []).map(String).join('|') };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    const key = (props.resetKeys ?? []).map(String).join('|');
    return key === state.key ? null : { key, error: null };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.props.onError?.(error, info);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    if (this.state.error) return this.props.fallback(this.state.error, this.reset);
    return <Fragment key={this.state.key}>{this.props.children}</Fragment>;
  }
}
