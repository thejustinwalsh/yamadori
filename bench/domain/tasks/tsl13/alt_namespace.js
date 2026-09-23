import { TSL } from 'three/webgpu';
import { uv } from 'three/tsl';

export function sampleWithOffset( textureNode, offset ) {

	const isolated = TSL.isolate( textureNode );

	return isolated.context( {
		getUV: ( texNode ) => ( texNode.uvNode || uv() ).add( offset ),
		forceUVContext: true
	} );

}
