// The r165 name imported from the r185 entry point: tslFn is no longer exported.
import { tslFn, vec3 } from 'three/tsl';

export const tint = tslFn( ( [ color, amount ] ) => color.mul( amount ).add( vec3( 0.05 ) ) );
