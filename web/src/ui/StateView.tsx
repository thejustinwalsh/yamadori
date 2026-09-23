// The non-data states, each saying what is actually true. Never a spinner
// over plausible numbers, never a placeholder figure.
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { describeFailure, type Failure } from '../api/client';
import { colors, space } from '../tokens/tokens.stylex';
import { text } from './text';

const s = stylex.create({
  box: {
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceXs,
    padding: space.spaceSm,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 70%, transparent)`,
    backgroundColor: colors.surfaceContainerLowest,
  },
  error: {
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.secondaryContainer} 60%, transparent)`,
    boxShadow: `inset 2px 0 0 ${colors.secondaryContainer}`,
  },
  title: { margin: 0, color: colors.onSurface },
  titleErr: { color: colors.secondary },
  detail: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere' },
  loading: {
    animationName: stylex.keyframes({ '0%': { opacity: 0.45 }, '50%': { opacity: 1 }, '100%': { opacity: 0.45 } }),
    animationDuration: '1.6s',
    animationIterationCount: 'infinite',
  },
});

export function StateView({
  kind,
  title,
  detail,
  children,
}: {
  kind: 'loading' | 'empty' | 'error' | 'inert';
  title: ReactNode;
  detail?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} {...stylex.props(s.box, kind === 'error' && s.error)}>
      <p {...stylex.props(text.labelMd, s.title, kind === 'error' && s.titleErr, kind === 'loading' && s.loading)}>
        {kind === 'error' ? '[ ! ] ' : kind === 'inert' ? '[ — ] ' : ''}
        {title}
      </p>
      {detail && <p {...stylex.props(text.bodySm, s.detail)}>{detail}</p>}
      {children}
    </div>
  );
}

/** Loading / failure for a poll that has no data yet. */
export function PollState({ path, failure }: { path: string; failure: Failure | null; what?: string }) {
  if (failure) {
    const d = describeFailure(path, failure);
    return <StateView kind="error" title={d.title} detail={d.detail || undefined} />;
  }
  return <StateView kind="loading" title={`reading ${path}`} />;
}
