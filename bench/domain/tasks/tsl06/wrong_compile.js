// Imports the class from the core entry point, which does not export it.
import { RenderPipeline } from 'three';
import { pass } from 'three/tsl';

export function createPipeline( renderer, scene, camera ) {

	const pipeline = new RenderPipeline( renderer );
	pipeline.outputNode = pass( scene, camera );

	return pipeline;

}
