import { reflector } from 'three/tsl';
import { MeshStandardNodeMaterial } from 'three/webgpu';

const HALF = 0.5;

export const createMirrorFloor = () => {

	const options = { resolutionScale: HALF };
	const reflection = reflector( options );
	reflection.target.rotation.x = - Math.PI / 2;

	const material = new MeshStandardNodeMaterial( { colorNode: reflection } );

	return { reflection, material };

};
