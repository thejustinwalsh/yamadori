import { uv } from 'three/tsl';

export const sampleWithOffset = ( textureNode, offset ) =>
	textureNode.isolate().context( { getUV: ( texNode ) => ( texNode.uvNode || uv() ).add( offset ), forceUVContext: true } );
