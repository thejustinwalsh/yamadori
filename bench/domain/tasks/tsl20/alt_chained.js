import { Fn, Loop, deltaTime, float, instanceIndex, int, storage } from 'three/tsl';

export function createDriftCompute( buffer, count ) {

	const particles = storage( buffer, 'vec4', count );
	particles.setPBO( true );

	const drift = Fn( () => {

		const particle = particles.element( instanceIndex );
		const push = float( 0 ).toVar( 'push' );

		Loop( 4, ( { i } ) => {

			const cond = int( instanceIndex ).add( i ).mod( 3 );
			push.addAssign( float( cond ) );

		} );

		const next = particle.y.add( push.mul( deltaTime ) );
		particle.y.assign( particle.y.greaterThan( 1 ).select( float( 0 ), next ) );

	} );

	return drift().compute( count );

}
