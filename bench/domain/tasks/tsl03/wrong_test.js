// Hand-rolls a per-frame time uniform, which the prompt rules out; never imports r185's time node.
import { oscSine, uniform } from 'three/tsl';

export const speed = uniform( 0.5 );
export const seconds = uniform( 0 ).onFrameUpdate( ( frame ) => frame.time );
export const wave = oscSine( seconds.mul( speed ) );
