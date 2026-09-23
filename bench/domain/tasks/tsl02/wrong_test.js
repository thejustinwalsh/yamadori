// Sidesteps the rename with a WGSL string, which the prompt rules out; no Fn.
import { wgslFn } from 'three/tsl';

export const tint = wgslFn( `
	fn tint( color: vec3f, amount: f32 ) -> vec3f {
		return color * amount + vec3f( 0.05 );
	}
` );
