import { reference, renderGroup, uniform } from 'three/tsl';

export function createCascadeUniforms( owner ) {

	const cascades = reference( '_cascades', 'vec2', owner ).setGroup( renderGroup ).setName( 'cascades' );
	const shadowFar = uniform( 'float' ).setGroup( renderGroup ).setName( 'shadowFar' );

	return { cascades, shadowFar };

}
