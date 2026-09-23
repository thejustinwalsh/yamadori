import { Fn, vec3 } from 'three/tsl';

export const tint = Fn( ( [ color, amount ] ) => color.mul( amount ).add( vec3( 0.05 ) ) );
