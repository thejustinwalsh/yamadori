import { Fn, float, vec3 } from 'three/tsl';

export const tint = Fn( ( inputs ) => {

	const [ baseColor, strength ] = inputs;
	const offset = vec3( float( 0.05 ) );

	return baseColor.mul( strength ).add( offset );

} );
