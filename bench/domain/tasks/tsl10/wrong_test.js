// Half-ported: the decode side still calls the deprecated colorToDirection.
import { colorToDirection, mrt, normalView, output, packNormalToRGB, pass, sample } from 'three/tsl';

export function createNormalPass( scene, camera ) {

	const scenePass = pass( scene, camera );
	scenePass.setMRT( mrt( { output, normal: packNormalToRGB( normalView ) } ) );

	const sceneNormal = sample( ( uvNode ) => colorToDirection( scenePass.getTextureNode( 'normal' ).sample( uvNode ) ) );

	return { scenePass, sceneNormal };

}
