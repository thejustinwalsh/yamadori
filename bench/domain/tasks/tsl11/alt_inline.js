import { float, normalView, normalWorld, positionViewDirection } from 'three/tsl';

export const rim = float( 1 ).sub( normalView.dot( positionViewDirection ).clamp() ).pow( 3 );
export const upFacing = normalWorld.y.max( 0 );
