import { screenSize } from 'three/tsl';

const size = screenSize;

export const aspect = size.x.div( size.y );
export const texelSize = size.reciprocal();
