// Imports the node material from the core entry point, which does not export it.
import { MeshStandardNodeMaterial } from 'three';
import { reflector } from 'three/tsl';

export function createMirrorFloor() {

	const reflection = reflector( { resolutionScale: 0.5 } );
	reflection.target.rotateX( - Math.PI / 2 );

	const material = new MeshStandardNodeMaterial();
	material.colorNode = reflection;

	return { reflection, material };

}
