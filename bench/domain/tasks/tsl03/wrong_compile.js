// The r165 timer factories imported from r185: neither is exported any more.
import { timerLocal, timerGlobal, oscSine, uniform } from 'three/tsl';

export const speed = uniform( 0.5 );
export const wave = oscSine( timerLocal().mul( speed ) );
export const seconds = timerGlobal();
