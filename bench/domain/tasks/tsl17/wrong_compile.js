// Timer is exported by 'three' and 'three/webgpu', not by 'three/tsl'.
import { Timer } from 'three/tsl';

export function createTicker() {

	const timer = new Timer();

	return {
		tick( timestamp ) {

			timer.update( timestamp );
			return { delta: timer.getDelta(), elapsed: timer.getElapsed() };

		}
	};

}
