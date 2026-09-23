// Imports the node material from the core entry point, which does not export it.
import { Color, Line2NodeMaterial } from 'three';
import { uniform } from 'three/tsl';

export const lineTint = uniform( new Color( 0x44aaff ) );

export function createFatLineMaterial() {

	const material = new Line2NodeMaterial( { linewidth: 4, worldUnits: false } );
	material.colorNode = lineTint;

	return material;

}
