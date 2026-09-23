// Reaches the removed name through the TSL namespace; TSL.tslFn is undefined in r185.
import { TSL } from 'three/webgpu';
import { vec3 } from 'three/tsl';

export const tint = TSL.tslFn( ( [ color, amount ] ) => color.mul( amount ).add( vec3( 0.05 ) ) );
