// isolate is not re-exported by r185's 'three/tsl' entry point (build/three.tsl.js).
import { isolate, uv } from 'three/tsl';

export const sampleWithOffset = ( textureNode, offset ) =>
	isolate( textureNode ).context( { getUV: ( texNode ) => ( texNode.uvNode || uv() ).add( offset ), forceUVContext: true } );
