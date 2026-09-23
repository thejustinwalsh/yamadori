// Hands the array to uniform(), which does not make an array uniform; no uniformArray.
import { Color } from 'three';
import { uniform } from 'three/tsl';

export const palette = uniform( [ new Color( 0xff0000 ), new Color( 0x00ff00 ), new Color( 0x0000ff ), new Color( 0xffffff ) ], 'color' );
export const paletteIndex = uniform( 0, 'int' );
export const paletteColor = palette.element( paletteIndex );
