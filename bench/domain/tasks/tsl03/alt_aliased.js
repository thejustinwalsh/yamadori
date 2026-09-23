import { float, oscSine, time as elapsed, uniform } from 'three/tsl';

export const speed = uniform( float( 0.5 ) );
export const seconds = elapsed;
export const wave = oscSine( seconds.mul( speed ) );
