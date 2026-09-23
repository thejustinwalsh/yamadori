// The r165 name imported from r185: uniforms is no longer exported.
import { Color } from 'three';
import { uniforms, uniform } from 'three/tsl';

export const palette = uniforms( [ new Color( 0xff0000 ), new Color( 0x00ff00 ), new Color( 0x0000ff ), new Color( 0xffffff ) ], 'color' );
export const paletteIndex = uniform( 0, 'int' );
export const paletteColor = palette.element( paletteIndex );
