// Invents a TAU constant; r185 does not export one.
import { TAU, uv } from 'three/tsl';

export const angle = uv().x.mul( TAU );
