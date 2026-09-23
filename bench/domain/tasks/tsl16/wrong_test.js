// Unported: lineColorNode is an r185 deprecation shim.
import { Color } from 'three';
import { Line2NodeMaterial } from 'three/webgpu';
import { uniform } from 'three/tsl';

export const lineTint = uniform( new Color( 0x44aaff ) );

export function createFatLineMaterial() {

	const material = new Line2NodeMaterial( { linewidth: 4, worldUnits: false } );
	material.lineColorNode = lineTint;

	return material;

}
