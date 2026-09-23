import { Color } from 'three';
import { int, uniform, uniformArray } from 'three/tsl';

const colors = [ 0xff0000, 0x00ff00, 0x0000ff, 0xffffff ].map( ( hex ) => new Color( hex ) );

export const palette = uniformArray( colors, 'color' );
export const paletteIndex = uniform( int( 0 ) );
export const paletteColor = palette.element( paletteIndex );
