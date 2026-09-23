import { FloatType, RenderTarget, WebGPURenderer } from 'three/webgpu';

export function createRendererAndTarget( canvas, width, height ) {

	const renderer = new WebGPURenderer( { canvas, outputBufferType: FloatType } );
	const target = new RenderTarget( width, height, { type: renderer.getOutputBufferType() } );

	return { renderer, target };

}
