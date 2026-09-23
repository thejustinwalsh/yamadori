import { Color, Line2NodeMaterial } from 'three/webgpu';
import { uniform } from 'three/tsl';

export const lineTint = uniform( new Color( 0x44aaff ) );

export const createFatLineMaterial = () => new Line2NodeMaterial( {
	linewidth: 4,
	worldUnits: false,
	colorNode: lineTint
} );
