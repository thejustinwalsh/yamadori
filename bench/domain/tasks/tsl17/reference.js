import { Timer } from 'three';

export function createTicker() {

	const timer = new Timer();

	return {
		tick( timestamp ) {

			timer.update( timestamp );

			return { delta: timer.getDelta(), elapsed: timer.getElapsed() };

		}
	};

}
