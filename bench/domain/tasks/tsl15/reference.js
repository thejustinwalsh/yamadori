import { MeshStandardNodeMaterial } from 'three/webgpu';
import { reflector } from 'three/tsl';

export function createMirrorFloor() {

	const reflection = reflector( { resolutionScale: 0.5 } );
	reflection.target.rotateX( - Math.PI / 2 );

	const material = new MeshStandardNodeMaterial();
	material.colorNode = reflection;

	return { reflection, material };

}
