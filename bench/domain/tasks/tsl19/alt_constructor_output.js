import { pass } from 'three/tsl';
import { RenderPipeline, WebGPURenderer } from 'three/webgpu';

export async function setupPost( canvas, scene, camera ) {

	const renderer = new WebGPURenderer( { canvas: canvas } );

	const scenePass = pass( scene, camera );
	scenePass.setResolutionScale( 1 / 2 );

	const pipeline = new RenderPipeline( renderer, scenePass );

	await renderer.init();
	renderer.setAnimationLoop( () => pipeline.render() );

	return { renderer, post: pipeline, scenePass };

}
