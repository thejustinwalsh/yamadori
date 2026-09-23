// Keeps the r181-deprecated PassNode.setResolution().
import { RenderPipeline, WebGPURenderer } from 'three/webgpu';
import { pass } from 'three/tsl';

export async function setupPost( canvas, scene, camera ) {

	const renderer = new WebGPURenderer( { canvas } );
	await renderer.init();

	const scenePass = pass( scene, camera );
	scenePass.setResolution( 0.5 );

	const post = new RenderPipeline( renderer );
	post.outputNode = scenePass;

	renderer.setAnimationLoop( () => {

		post.render();

	} );

	return { renderer, post, scenePass };

}
