// Invents a decode helper name r185 does not export.
import { mrt, normalView, output, packNormalToRGB, pass, sample, unpackNormalFromRGB } from 'three/tsl';

export function createNormalPass( scene, camera ) {

	const scenePass = pass( scene, camera );
	scenePass.setMRT( mrt( { output, normal: packNormalToRGB( normalView ) } ) );

	const sceneNormal = sample( ( uvNode ) => unpackNormalFromRGB( scenePass.getTextureNode( 'normal' ).sample( uvNode ) ) );

	return { scenePass, sceneNormal };

}
