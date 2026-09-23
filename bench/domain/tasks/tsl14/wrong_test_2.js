// Drops the stack pushes entirely, so the declaration-order nodes are never emitted.
import { Break, Fn, If, Loop, int } from 'three/tsl';

export function emitTileLoop( builder, lightCount, getTile, shadeLight ) {

	Fn( () => {

		Loop( lightCount, ( { i } ) => {

			const lightIndex = getTile( i );
			If( lightIndex.equal( int( 0 ) ), () => {

				Break();

			} );
			shadeLight( lightIndex.sub( 1 ) );

		} );

	} )();

}
