// Unported: .append() is an r176 deprecation shim in r185.
import { Break, Fn, If, Loop, int } from 'three/tsl';

export function emitTileLoop( builder, lightCount, getTile, shadeLight ) {

	const lightingModel = builder.context.reflectedLight;
	lightingModel.directDiffuse.append();
	lightingModel.directSpecular.append();

	Fn( () => {

		Loop( lightCount, ( { i } ) => {

			const lightIndex = getTile( i );
			If( lightIndex.equal( int( 0 ) ), () => {

				Break();

			} );
			shadeLight( lightIndex.sub( 1 ) ).append();

		} );

	} )().append();

}
