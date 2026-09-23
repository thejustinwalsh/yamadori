import { color, densityFogFactor, float, fog, rangeFogFactor } from 'three/tsl';

const fogColor = color( 0x8899aa );

export const useRangeFog = ( scene ) => {

	const factor = rangeFogFactor( float( 10 ), float( 80 ) );
	scene.fogNode = fog( fogColor, factor );

};

export const useDensityFog = ( scene ) => {

	scene.fogNode = fog( fogColor, densityFogFactor( 0.03 ) );

};
