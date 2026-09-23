// The pre-r183 class name, still exported as a warnOnce shim.
import { PostProcessing } from 'three/webgpu';
import { pass } from 'three/tsl';

export function createPipeline( renderer, scene, camera ) {

	const postProcessing = new PostProcessing( renderer );
	postProcessing.outputNode = pass( scene, camera );

	return postProcessing;

}
