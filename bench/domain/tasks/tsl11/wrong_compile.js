// Invents an accessor name r185 does not export.
import { Fn, dot, float, normalWorld, positionViewDirection, transformedNormal } from 'three/tsl';

export const rim = Fn( () => {

	const facing = dot( transformedNormal, positionViewDirection ).clamp();
	return float( 1 ).sub( facing ).pow( 3 );

} )();

export const upFacing = normalWorld.y.max( 0 );
