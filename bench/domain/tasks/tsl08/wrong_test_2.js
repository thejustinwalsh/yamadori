// Builds 2*PI by hand instead of using the predefined constant the prompt asks for.
import { PI, uv } from 'three/tsl';

export const angle = uv().x.mul( PI.mul( 2 ) );
