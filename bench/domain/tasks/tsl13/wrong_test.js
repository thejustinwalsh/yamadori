// Unported: .cache() is an r181 deprecation shim in r185.
import { uv } from 'three/tsl';

export const sampleWithOffset = ( textureNode, offset ) =>
	textureNode.cache().context( { getUV: ( texNode ) => ( texNode.uvNode || uv() ).add( offset ), forceUVContext: true } );
