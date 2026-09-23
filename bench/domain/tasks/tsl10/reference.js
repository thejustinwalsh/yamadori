import { mrt, normalView, output, packNormalToRGB, pass, sample, unpackRGBToNormal } from 'three/tsl';

export function createNormalPass( scene, camera ) {

	const scenePass = pass( scene, camera );
	scenePass.setMRT( mrt( { output, normal: packNormalToRGB( normalView ) } ) );

	const sceneNormal = sample( ( uvNode ) => unpackRGBToNormal( scenePass.getTextureNode( 'normal' ).sample( uvNode ) ) );

	return { scenePass, sceneNormal };

}
