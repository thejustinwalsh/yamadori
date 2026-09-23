// Imports a naming helper r185 does not export.
import { reference, renderGroup, uniform, nodeName } from 'three/tsl';

export function createCascadeUniforms( owner ) {

	const cascades = nodeName( reference( '_cascades', 'vec2', owner ).setGroup( renderGroup ), 'cascades' );
	const shadowFar = nodeName( uniform( 'float' ).setGroup( renderGroup ), 'shadowFar' );

	return { cascades, shadowFar };

}
