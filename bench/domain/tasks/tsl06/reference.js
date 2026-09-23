import { RenderPipeline } from 'three/webgpu';
import { pass } from 'three/tsl';

export function createPipeline( renderer, scene, camera ) {

	const pipeline = new RenderPipeline( renderer );
	pipeline.outputNode = pass( scene, camera );

	return pipeline;

}
