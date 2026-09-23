// A plausible-sounding helper three@0.185.1 does not export.
import { flipOnBackSide, normalView } from 'three/tsl';

export const twoSidedNormal = flipOnBackSide(normalView);
