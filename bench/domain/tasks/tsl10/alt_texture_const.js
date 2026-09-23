import { mrt, normalView, output, packNormalToRGB, pass, sample, unpackRGBToNormal } from 'three/tsl';

export const createNormalPass = ( scene, camera ) => {

	const scenePass = pass( scene, camera );
	const outputs = mrt( {
		output: output,
		normal: packNormalToRGB( normalView )
	} );
	scenePass.setMRT( outputs );

	const normalTexture = scenePass.getTextureNode( 'normal' );
	const sceneNormal = sample( ( coord ) => unpackRGBToNormal( normalTexture.sample( coord ) ) );

	return { scenePass: scenePass, sceneNormal: sceneNormal };

};
