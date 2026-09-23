// A failure inside one part of the page is shown as that part's error --
// named, verbatim, with the payload it was reading -- and never unmounts the
// panels around it.
//
// Two failures made this necessary: one shader compile error blanked the whole
// dashboard, and later a /dash/api/tiers payload without `default` made
// `t.default.toUpperCase()` throw during render, uncaught, and the operator got
// a white screen with every other panel's data lost with it.
import { Component, type ReactNode } from 'react';
import { Panel } from './Panel';
import { StateView } from './StateView';

/**
 * A payload that lacks a key a panel cannot draw without. Thrown by `need()`
 * so the boundary can name the exact key, instead of a TypeError about
 * reading a property of undefined.
 */
export class PayloadError extends Error {
  constructor(
    readonly source: string,
    readonly key: string,
    readonly got: unknown,
  ) {
    super(`${source} has no usable \`${key}\` (got ${describe(got)})`);
    this.name = 'PayloadError';
  }
}

function describe(v: unknown): string {
  if (v === undefined) return 'nothing';
  if (v === null) return 'null';
  if (Array.isArray(v)) return `an array of ${v.length}`;
  return typeof v;
}

/** `obj[key]`, or a PayloadError naming the source and the key. */
export function need<T, K extends keyof T>(obj: T, key: K, source: string, check: (v: unknown) => boolean = (v) => v !== undefined && v !== null): NonNullable<T[K]> {
  const v = obj?.[key];
  if (!check(v)) throw new PayloadError(source, String(key), v);
  return v as NonNullable<T[K]>;
}

type Props = {
  /** what this is, for the error title: a panel name */
  what: string;
  /** the payload it reads, e.g. /dash/api/tiers */
  source?: string;
  /** render the error inside a panel frame (default) or as a bare state */
  bare?: boolean;
  fill?: boolean;
  children: ReactNode;
};
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    const e = this.state.error;
    if (!e) return this.props.children;
    const key = e instanceof PayloadError ? e.key : null;
    const source = e instanceof PayloadError ? e.source : this.props.source;
    const body = (
      <StateView
        kind="error"
        title={`${this.props.what} · render failed`}
        detail={[source, key ? `key ${key}` : null, `${e.name}: ${e.message}`].filter(Boolean).join(' · ')}
      />
    );
    if (this.props.bare) return body;
    return (
      <Panel title={this.props.what} tag="RENDER FAILED" tagTone="crimson" edge="crimson" fill={this.props.fill}>
        {body}
      </Panel>
    );
  }
}

function Deferred({ render }: { render: () => ReactNode }) {
  return <>{render()}</>;
}

/**
 * A panel whose JSX is built INSIDE its boundary. `render` runs in a child
 * component, so a payload access that throws (`x.missing.map` on a payload
 * without `missing`) is caught here rather than in the page that holds it.
 */
export function Guarded({ what, source, fill, render }: { what: string; source?: string; fill?: boolean; render: () => ReactNode }) {
  return (
    <ErrorBoundary what={what} source={source} fill={fill}>
      <Deferred render={render} />
    </ErrorBoundary>
  );
}
