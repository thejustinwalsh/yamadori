// viewportSize is the current viewport rectangle, not the render-target size the prompt asks for.
import { viewportSize, vec2 } from 'three/tsl';

export const aspect = viewportSize.x.div( viewportSize.y );
export const texelSize = vec2( 1 ).div( viewportSize );
