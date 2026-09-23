// Keeps remainder as a chained method; r185 removed it (no mod anywhere).
import { Fn, Loop, deltaTime, float, instanceIndex, int, select, storage } from 'three/tsl';

export function createDriftCompute( buffer, count ) {

	const particles = storage( buffer, 'vec4', count ).setPBO( true );

	return Fn( () => {

		const p = particles.element( instanceIndex );
		const push = float( 0 ).toVar();

		Loop( { start: 0, end: 4 }, ( { i } ) => {

			push.addAssign( float( int( instanceIndex ).add( i ).remainder( int( 3 ) ) ) );

		} );

		p.y.assign( select( p.y.greaterThan( 1 ), float( 0 ), p.y.add( push.mul( deltaTime ) ) ) );

	} )().compute( count );

}
