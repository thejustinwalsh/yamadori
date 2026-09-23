import * as stylex from '@stylexjs/stylex';
import { useState } from 'react';
import { describeFailure, writeKey, type Failure } from '../api/client';
import { colors, space } from '../tokens/tokens.stylex';
import { Panel } from './Panel';
import { text } from './text';

const s = stylex.create({
  form: { display: 'flex', gap: space.spaceSm, flexWrap: 'wrap', alignItems: 'stretch' },
  input: {
    flexGrow: 1,
    minWidth: '220px',
    backgroundColor: colors.surfaceContainerLowest,
    color: colors.tertiaryContainer,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.outlineVariant,
    boxShadow: `inset 0 2px 6px color-mix(in srgb, ${colors.surfaceContainerLowest} 80%, black)`,
    paddingInline: space.spaceSm,
    paddingBlock: space.spaceXs,
    outline: { default: 'none', ':focus-visible': `1px solid ${colors.tertiaryContainer}` },
  },
  button: {
    backgroundColor: { default: colors.surfaceContainerLowest, ':hover': `color-mix(in srgb, ${colors.primaryContainer} 15%, transparent)` },
    color: colors.primaryContainer,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.primaryContainer,
    paddingInline: space.spaceMd,
    paddingBlock: space.spaceXs,
    cursor: 'pointer',
  },
  p: { margin: 0, color: colors.onSurfaceVariant },
  err: { margin: 0, color: colors.secondary },
});

/**
 * The same key an editor sends. It is stored in this browser only, under
 * localStorage['yamadori_key'], exactly as the Python pages store it.
 */
export function SignIn({ failure }: { failure: Failure | null }) {
  const [key, setKey] = useState('');
  const why = failure && failure.kind !== 'nokey' ? describeFailure('/dash/api/vitals', failure) : null;
  return (
    <Panel kanji="鍵" title="SIGN IN" tag="BEARER KEY" tagTone="cyan" edge="cyan" flag="GATED">
      {why && (
        <p role="alert" {...stylex.props(text.bodySm, s.err)}>
          [ ! ] {why.title}{why.detail ? ` · ${why.detail}` : ''}
        </p>
      )}
      <form
        {...stylex.props(s.form)}
        onSubmit={(e) => {
          e.preventDefault();
          writeKey(key.trim());
        }}
      >
        <input
          type="password"
          autoComplete="off"
          aria-label="API key"
          placeholder="API key"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          {...stylex.props(text.bodySm, s.input)}
        />
        <button type="submit" {...stylex.props(text.labelMd, s.button)}>
          [ SIGN IN ]
        </button>
      </form>
    </Panel>
  );
}
