// Unported: the r181 option and getter.
import { FloatType, RenderTarget, WebGPURenderer } from 'three/webgpu';

export function createRendererAndTarget( canvas, width, height ) {

	const renderer = new WebGPURenderer( { canvas, colorBufferType: FloatType } );
	const target = new RenderTarget( width, height, { type: renderer.getColorBufferType() } );

	return { renderer, target };

}
