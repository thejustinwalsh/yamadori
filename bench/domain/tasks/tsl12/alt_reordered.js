import { reference, renderGroup, uniform } from 'three/tsl';

export const createCascadeUniforms = ( owner ) => ( {
	cascades: reference( '_cascades', 'vec2', owner ).setName( 'cascades' ).setGroup( renderGroup ),
	shadowFar: uniform( 'float' ).setName( 'shadowFar' ).setGroup( renderGroup )
} );
