import { FloatType, RenderTarget } from 'three';
import { WebGPURenderer } from 'three/webgpu';

export const createRendererAndTarget = ( canvas, width, height ) => {

	const parameters = { canvas: canvas, outputBufferType: FloatType };
	const renderer = new WebGPURenderer( parameters );

	const type = renderer.getOutputBufferType();
	const target = new RenderTarget( width, height, { type } );

	return { renderer, target };

};
