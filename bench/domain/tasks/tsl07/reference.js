import { screenSize, vec2 } from 'three/tsl';

export const aspect = screenSize.x.div( screenSize.y );
export const texelSize = vec2( 1 ).div( screenSize );
