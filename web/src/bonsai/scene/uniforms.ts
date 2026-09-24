// Scene-wide uniforms the live driver writes once per frame (Scene.tsx
// LiveDriver). One tokonoma canvas exists at a time, so module scope is the
// simplest owner. Times are on the LiveBus clock, in seconds.
import { uniform } from 'three/tsl';

export const U = {
  time: uniform(0),
  /** ground and ring brightness, eased from activity, with the idle breath */
  glow: uniform(0.12),
  /** when each event kind last fired; far in the past = no pulse */
  request: uniform(-100),
  tool: uniform(-100),
  seed: uniform(-100),
  /** moss cover on the bark: index staleness, [0, 1]; 0 when inert (mapping.ts) */
  moss: uniform(0),
  /** how far live foliage is lit toward the primary green: 0 when inert */
  leaf: uniform(0),
};
