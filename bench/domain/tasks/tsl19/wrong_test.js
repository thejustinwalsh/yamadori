// Renames the class but keeps the r181-deprecated async per-frame renderAsync().
import { RenderPipeline, WebGPURenderer } from 'three/webgpu';
import { pass } from 'three/tsl';

export async function setupPost( canvas, scene, camera ) {

	const renderer = new WebGPURenderer( { canvas } );
	await renderer.init();

	const scenePass = pass( scene, camera );
	scenePass.setResolutionScale( 0.5 );

	const post = new RenderPipeline( renderer );
	post.outputNode = scenePass;

	renderer.setAnimationLoop( async () => {

		await post.renderAsync();

	} );

	return { renderer, post, scenePass };

}
