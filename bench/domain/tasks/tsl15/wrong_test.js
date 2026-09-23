// The pre-r180 option name; r185 only honours it through a warnOnce shim.
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { reflector } from 'three/tsl';

export function createMirrorFloor() {

	const reflection = reflector( { resolution: 0.5 } );
	reflection.target.rotateX( - Math.PI / 2 );

	const material = new MeshStandardNodeMaterial();
	material.colorNode = reflection;

	return { reflection, material };

}
