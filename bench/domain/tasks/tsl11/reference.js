import { Fn, dot, float, normalView, normalWorld, positionViewDirection } from 'three/tsl';

export const rim = Fn( () => {

	const facing = dot( normalView, positionViewDirection ).clamp();
	return float( 1 ).sub( facing ).pow( 3 );

} )();

export const upFacing = normalWorld.y.max( 0 );
