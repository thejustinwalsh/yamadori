import { Color } from 'three';
import { uniform, uniformArray } from 'three/tsl';

export const palette = uniformArray( [ new Color( 0xff0000 ), new Color( 0x00ff00 ), new Color( 0x0000ff ), new Color( 0xffffff ) ], 'color' );
export const paletteIndex = uniform( 0, 'int' );
export const paletteColor = palette.element( paletteIndex );
