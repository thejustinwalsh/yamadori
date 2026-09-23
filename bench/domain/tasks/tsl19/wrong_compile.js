// Imports the WebGPU classes from the core entry point, which exports neither.
import { RenderPipeline, WebGPURenderer } from 'three';
import { pass } from 'three/tsl';

export async function setupPost( canvas, scene, camera ) {

	const renderer = new WebGPURenderer( { canvas } );
	await renderer.init();

	const scenePass = pass( scene, camera );
	scenePass.setResolutionScale( 0.5 );

	const post = new RenderPipeline( renderer );
	post.outputNode = scenePass;

	renderer.setAnimationLoop( () => {

		post.render();

	} );

	return { renderer, post, scenePass };

}
