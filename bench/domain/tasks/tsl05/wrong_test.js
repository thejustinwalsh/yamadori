// Unported: rangeFog/densityFog are still listed by three/tsl but undefined in r185.
import { rangeFog, densityFog, color } from 'three/tsl';

export function useRangeFog( scene ) {

	scene.fogNode = rangeFog( color( 0x8899aa ), 10, 80 );

}

export function useDensityFog( scene ) {

	scene.fogNode = densityFog( color( 0x8899aa ), 0.03 );

}
