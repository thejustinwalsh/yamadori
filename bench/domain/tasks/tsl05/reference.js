import { color, densityFogFactor, fog, rangeFogFactor } from 'three/tsl';

export function useRangeFog( scene ) {

	scene.fogNode = fog( color( 0x8899aa ), rangeFogFactor( 10, 80 ) );

}

export function useDensityFog( scene ) {

	scene.fogNode = fog( color( 0x8899aa ), densityFogFactor( 0.03 ) );

}
