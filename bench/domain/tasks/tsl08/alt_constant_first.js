import { TWO_PI, uv } from 'three/tsl';

const u = uv().x;

export const angle = TWO_PI.mul( u );
