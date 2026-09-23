// Half-ported: upFacing still reads the deprecated transformedNormalWorld.
import { Fn, dot, float, normalView, positionViewDirection, transformedNormalWorld } from 'three/tsl';

export const rim = Fn( () => {

	const facing = dot( normalView, positionViewDirection ).clamp();
	return float( 1 ).sub( facing ).pow( 3 );

} )();

export const upFacing = transformedNormalWorld.y.max( 0 );
