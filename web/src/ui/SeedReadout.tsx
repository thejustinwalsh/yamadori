// The concept seed most recently injected (vitals.seed, from
// mcp/concept_seed.py record()): word, hex, and the 32 bits that are the
// bonsai's genome. The dashboard recomputes FNV-1a from the word and says so
// when it disagrees with the server rather than trusting either silently.
import * as stylex from '@stylexjs/stylex';
import type { Seed, Vitals } from '../api/types';
import { bits32, fnv1a32, hex32 } from '../bonsai/prng';
import { ago, clock, n } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Panel } from './Panel';
import { Chip, Label } from './primitives';
import { StateView } from './StateView';
import { text } from './text';

const flicker = stylex.keyframes({
  '0%': { opacity: 0.2, transform: 'translateX(-2px)', filter: 'blur(1px)' },
  '12%': { opacity: 1, transform: 'translateX(1px)' },
  '20%': { opacity: 0.6 },
  '28%': { opacity: 1, transform: 'translateX(0)', filter: 'blur(0)' },
  '100%': { opacity: 1 },
});
const sweep = stylex.keyframes({ '0%': { transform: 'translateY(-100%)' }, '100%': { transform: 'translateY(300%)' } });

const s = stylex.create({
  wordBox: {
    position: 'relative',
    overflow: 'hidden',
    paddingBlock: space.spaceSm,
    paddingInline: space.spaceMd,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.primaryContainer} 35%, transparent)`,
    boxShadow: `inset 0 0 40px color-mix(in srgb, ${colors.primaryContainer} 8%, transparent)`,
  },
  scan: {
    position: 'absolute',
    insetInline: 0,
    top: 0,
    height: '40%',
    pointerEvents: 'none',
    backgroundImage: `linear-gradient(to bottom, transparent, color-mix(in srgb, ${colors.primaryContainer} 10%, transparent), transparent)`,
    animationName: sweep,
    animationDuration: '3.2s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  word: {
    margin: 0,
    color: colors.primaryContainer,
    textShadow: `0 0 18px color-mix(in srgb, ${colors.primaryContainer} 55%, transparent), 0 0 2px ${colors.primaryContainer}`,
    overflowWrap: 'anywhere',
    animationName: flicker,
    animationDuration: '900ms',
    animationTimingFunction: 'steps(12)',
  },
  hex: {
    color: colors.tertiaryContainer,
    textShadow: `0 0 10px color-mix(in srgb, ${colors.tertiaryContainer} 50%, transparent)`,
    letterSpacing: '0.12em',
  },
  bits: { display: 'grid', gridTemplateColumns: 'repeat(32, minmax(0, 1fr))', gap: '2px' },
  bit: { height: '14px', backgroundColor: colors.surfaceContainerHigh },
  bitOn: { backgroundColor: colors.primaryContainer, boxShadow: `0 0 6px color-mix(in srgb, ${colors.primaryContainer} 70%, transparent)` },
  nibbleGap: { marginInlineStart: '3px' },
  bitText: { color: colors.outline, overflowWrap: 'anywhere', letterSpacing: '0.08em' },
  meta: { display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: space.spaceSm, rowGap: '2px' },
  v: { color: colors.onSurface, overflowWrap: 'anywhere' },
});

export function SeedBits({ u32 }: { u32: number }) {
  const b = bits32(u32);
  return (
    <div role="img" aria-label={`32 bits ${b}`}>
      <div {...stylex.props(s.bits)}>
        {b.split('').map((c, i) => (
          <span key={i} {...stylex.props(s.bit, c === '1' && s.bitOn, i % 4 === 0 && i > 0 && s.nibbleGap)} />
        ))}
      </div>
      <div {...stylex.props(text.labelXs, s.bitText)}>{b.replace(/(.{4})/g, '$1 ').trim()}</div>
    </div>
  );
}

function SeedBody({ seed, now }: { seed: Seed; now: number }) {
  const local = fnv1a32(seed.word);
  const agrees = local === seed.u32 >>> 0;
  return (
    <>
      <div {...stylex.props(s.wordBox)}>
        <span aria-hidden {...stylex.props(s.scan)} />
        <Label>CONCEPT WORD</Label>
        <p key={seed.word} {...stylex.props(text.headlineXl, s.word)}>
          {seed.word}
        </p>
        <span {...stylex.props(text.titleMd, s.hex, text.num)}>{hex32(seed.u32)}</span>
      </div>
      <SeedBits u32={seed.u32} />
      <div {...stylex.props(text.labelXs, s.meta)}>
        <Label>FNV-1a</Label>
        <span {...stylex.props(s.v)}>
          {agrees ? (
            <Chip tone="moss">VERIFIED · recomputed {hex32(local)} in this browser</Chip>
          ) : (
            <Chip tone="crimson">MISMATCH · server {hex32(seed.u32)} ≠ browser {hex32(local)}</Chip>
          )}
        </span>
        <Label>TOKEN</Label>
        <span {...stylex.props(s.v, text.num)}>{seed.token_id === null ? 'not a single vocabulary token' : `#${n(seed.token_id)}`}</span>
        <Label>INJECTED</Label>
        <span {...stylex.props(s.v)}>{seed.where || '—'}</span>
        <Label>AT</Label>
        <span {...stylex.props(s.v, text.num)}>
          {clock(seed.at)} · {ago(now / 1000 - seed.at)} ago
        </span>
      </div>
    </>
  );
}

export function SeedPanel({ vitals, now }: { vitals: Vitals | null; now: number }) {
  const has = vitals !== null && 'seed' in vitals;
  const seed = vitals?.seed ?? null;
  return (
    <Panel
      kanji="種"
      title="TANE · SEED"
      tag={seed ? 'GENOME LIVE' : 'NO SEED'}
      tagTone={seed ? 'moss' : 'muted'}
      flag={seed ? hex32(seed.u32) : undefined}
      edge={seed ? 'moss' : undefined}
    >
      {!vitals ? (
        <StateView kind="loading" title="reading /dash/api/vitals" />
      ) : !has ? (
        <StateView kind="inert" title="seed not reported by this server" />
      ) : !seed ? (
        <StateView kind="inert" title="no seed injected yet" />
      ) : (
        <SeedBody seed={seed} now={now} />
      )}
    </Panel>
  );
}
