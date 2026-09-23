// Invents a factor name r185 does not export (the linear factor is rangeFogFactor).
import { color, densityFogFactor, fog, linearFogFactor } from 'three/tsl';

export function useRangeFog( scene ) {

	scene.fogNode = fog( color( 0x8899aa ), linearFogFactor( 10, 80 ) );

}

export function useDensityFog( scene ) {

	scene.fogNode = fog( color( 0x8899aa ), densityFogFactor( 0.03 ) );

}
