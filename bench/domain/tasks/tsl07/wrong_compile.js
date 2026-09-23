// Invents a name r185 does not export.
import { screenResolution, vec2 } from 'three/tsl';

export const aspect = screenResolution.x.div( screenResolution.y );
export const texelSize = vec2( 1 ).div( screenResolution );
