// Imports toStack as if it were a function export; r185 only has it as a chained method (and Stack).
import { Break, Fn, If, Loop, int, toStack } from 'three/tsl';

export function emitTileLoop( builder, lightCount, getTile, shadeLight ) {

	const lightingModel = builder.context.reflectedLight;
	toStack( lightingModel.directDiffuse );
	toStack( lightingModel.directSpecular );

	toStack( Fn( () => {

		Loop( lightCount, ( { i } ) => {

			const lightIndex = getTile( i );
			If( lightIndex.equal( int( 0 ) ), () => {

				Break();

			} );
			toStack( shadeLight( lightIndex.sub( 1 ) ) );

		} );

	} )() );

}
