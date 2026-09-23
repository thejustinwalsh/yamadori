// Ports the warning getter but keeps colorBufferType, which r185 silently ignores (buffers stay half-float).
import { FloatType, RenderTarget, WebGPURenderer } from 'three/webgpu';

export function createRendererAndTarget( canvas, width, height ) {

	const renderer = new WebGPURenderer( { canvas, colorBufferType: FloatType } );
	const target = new RenderTarget( width, height, { type: renderer.getOutputBufferType() } );

	return { renderer, target };

}
