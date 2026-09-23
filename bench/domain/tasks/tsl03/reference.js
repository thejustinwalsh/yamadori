import { oscSine, time, uniform } from 'three/tsl';

export const speed = uniform( 0.5 );
export const wave = oscSine( time.mul( speed ) );
export const seconds = time;
