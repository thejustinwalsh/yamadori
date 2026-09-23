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
};
