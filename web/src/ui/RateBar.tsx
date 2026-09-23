// A measured rate with its Wilson interval, on a fixed 0..1 axis so two bars
// are comparable by eye. The interval is drawn, never omitted: a rate from
// n=3 must look like a rate from n=3.
import * as stylex from '@stylexjs/stylex';
import { colors } from '../tokens/tokens.stylex';
import type { Tone } from './primitives';

const s = stylex.create({
  track: { position: 'relative', height: '12px', minWidth: '90px', backgroundColor: colors.surfaceContainerHigh },
  ci: { position: 'absolute', top: '5px', height: '2px', backgroundColor: colors.onSurfaceVariant, opacity: 0.8 },
  tick: { position: 'absolute', top: '1px', width: '1px', height: '10px', backgroundColor: colors.onSurfaceVariant },
  bar: { position: 'absolute', top: '3px', height: '6px', left: 0 },
  moss: { backgroundColor: colors.primaryContainer, boxShadow: `0 0 8px color-mix(in srgb, ${colors.primaryContainer} 40%, transparent)` },
  cyan: { backgroundColor: colors.tertiaryContainer },
  crimson: { backgroundColor: colors.secondaryContainer },
  rose: { backgroundColor: colors.secondary },
  muted: { backgroundColor: colors.outline },
  pos: (l: number, w: number) => ({ left: `${l}%`, width: `${w}%` }),
  at: (l: number) => ({ left: `${l}%` }),
  w: (w: number) => ({ width: `${w}%` }),
});

const pc = (x: number) => Math.max(0, Math.min(100, x * 100));

export function RateBar({ rate, lo, hi, tone = 'moss', label }: { rate: number | null; lo?: number; hi?: number; tone?: Tone; label: string }) {
  return (
    <div role="img" aria-label={label} {...stylex.props(s.track)}>
      {rate !== null && <span {...stylex.props(s.bar, s[tone], s.w(pc(rate)))} />}
      {lo !== undefined && hi !== undefined && (
        <>
          <span {...stylex.props(s.ci, s.pos(pc(lo), pc(hi) - pc(lo)))} />
          <span {...stylex.props(s.tick, s.at(pc(lo)))} />
          <span {...stylex.props(s.tick, s.at(pc(hi)))} />
        </>
      )}
    </div>
  );
}
