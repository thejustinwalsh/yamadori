import { negateOnBackSide, normalView } from 'three/tsl';

export const twoSidedNormal = negateOnBackSide(normalView);
