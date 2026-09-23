import { Break, Fn, If, Loop, int } from 'three/tsl';

export function emitTileLoop( builder, lightCount, getTile, shadeLight ) {

	const lightingModel = builder.context.reflectedLight;

	// force declaration order, before the loop
	lightingModel.directDiffuse.toStack();
	lightingModel.directSpecular.toStack();

	Fn( () => {

		Loop( lightCount, ( { i } ) => {

			const lightIndex = getTile( i );

			If( lightIndex.equal( int( 0 ) ), () => {

				Break();

			} );

			shadeLight( lightIndex.sub( 1 ) ).toStack();

		} );

	} )().toStack();

}
