// Imports the renderer from the core entry point, which does not export it.
import { FloatType, RenderTarget, WebGPURenderer } from 'three';

export function createRendererAndTarget( canvas, width, height ) {

	const renderer = new WebGPURenderer( { canvas, outputBufferType: FloatType } );
	const target = new RenderTarget( width, height, { type: renderer.getOutputBufferType() } );

	return { renderer, target };

}
