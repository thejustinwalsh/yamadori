// Everything ported except the lowercase loop, which r185 no longer exports.
import { Fn, deltaTime, float, instanceIndex, int, loop, mod, select, storage } from 'three/tsl';

export function createDriftCompute( buffer, count ) {

	const particles = storage( buffer, 'vec4', count ).setPBO( true );

	return Fn( () => {

		const p = particles.element( instanceIndex );
		const push = float( 0 ).toVar();

		loop( { start: 0, end: 4 }, ( { i } ) => {

			push.addAssign( float( mod( int( instanceIndex ).add( i ), int( 3 ) ) ) );

		} );

		p.y.assign( select( p.y.greaterThan( 1 ), float( 0 ), p.y.add( push.mul( deltaTime ) ) ) );

	} )().compute( count );

}
