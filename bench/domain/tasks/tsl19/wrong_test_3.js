// Unported: the r180 original, three deprecated APIs.
import { WebGPURenderer, PostProcessing } from 'three/webgpu';
import { pass } from 'three/tsl';

export async function setupPost( canvas, scene, camera ) {

	const renderer = new WebGPURenderer( { canvas } );

	const scenePass = pass( scene, camera );
	scenePass.setResolution( 0.5 );

	const post = new PostProcessing( renderer );
	post.outputNode = scenePass;

	renderer.setAnimationLoop( async () => {

		await post.renderAsync();

	} );

	return { renderer, post, scenePass };

}
