// Half-ported: the density fog still uses the pre-r171 densityFog helper.
import { color, densityFog, fog, rangeFogFactor } from 'three/tsl';

export function useRangeFog( scene ) {

	scene.fogNode = fog( color( 0x8899aa ), rangeFogFactor( 10, 80 ) );

}

export function useDensityFog( scene ) {

	scene.fogNode = densityFog( color( 0x8899aa ), 0.03 );

}
