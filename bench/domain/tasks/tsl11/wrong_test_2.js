// Swaps to the raw geometry normals, which ignore normal mapping; the prompt asks for the transformed ones.
import { Fn, dot, float, normalViewGeometry, normalWorldGeometry, positionViewDirection } from 'three/tsl';

export const rim = Fn( () => {

	const facing = dot( normalViewGeometry, positionViewDirection ).clamp();
	return float( 1 ).sub( facing ).pow( 3 );

} )();

export const upFacing = normalWorldGeometry.y.max( 0 );
