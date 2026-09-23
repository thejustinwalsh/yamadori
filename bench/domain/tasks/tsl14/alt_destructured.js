import { Break, Fn, If, Loop, int } from 'three/tsl';

export function emitTileLoop( builder, lightCount, getTile, shadeLight ) {

	const { directDiffuse, directSpecular } = builder.context.reflectedLight;
	directDiffuse.toStack();
	directSpecular.toStack();

	const tileLoop = Fn( () => {

		Loop( lightCount, ( { i } ) => {

			const index = getTile( i );
			If( index.equal( int( 0 ) ), () => Break() );
			shadeLight( index.sub( 1 ) ).toStack();

		} );

	} );

	tileLoop().toStack();

}
