// The pre-r185 name. Still exported as a warnOnce shim, so it imports fine.
import { directionToFaceDirection, normalView } from 'three/tsl';

export const twoSidedNormal = directionToFaceDirection(normalView);
