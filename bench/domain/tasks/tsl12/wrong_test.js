// Unported: .label() is an r179 deprecation shim in r185.
import { reference, renderGroup, uniform } from 'three/tsl';

export function createCascadeUniforms( owner ) {

	const cascades = reference( '_cascades', 'vec2', owner ).setGroup( renderGroup ).label( 'cascades' );
	const shadowFar = uniform( 'float' ).setGroup( renderGroup ).label( 'shadowFar' );

	return { cascades, shadowFar };

}
