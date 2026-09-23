// Only the import path was changed; tslFn, storageObject, loop, cond, remainder, timerDelta are gone in r185.
import { tslFn, storageObject, instanceIndex, loop, cond, remainder, timerDelta, float, int } from 'three/tsl';

export function createDriftCompute( buffer, count ) {

	const particles = storageObject( buffer, 'vec4', count );

	return tslFn( () => {

		const p = particles.element( instanceIndex );
		const push = float( 0 ).toVar();

		loop( { start: 0, end: 4 }, ( { i } ) => {

			push.addAssign( float( remainder( int( instanceIndex ).add( i ), int( 3 ) ) ) );

		} );

		p.y.assign( cond( p.y.greaterThan( 1 ), float( 0 ), p.y.add( push.mul( timerDelta() ) ) ) );

	} )().compute( count );

}
