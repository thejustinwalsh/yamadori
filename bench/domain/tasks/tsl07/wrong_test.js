// Unported: viewportResolution is an r169 deprecation shim in r185.
import { viewportResolution, vec2 } from 'three/tsl';

export const aspect = viewportResolution.x.div( viewportResolution.y );
export const texelSize = vec2( 1 ).div( viewportResolution );
