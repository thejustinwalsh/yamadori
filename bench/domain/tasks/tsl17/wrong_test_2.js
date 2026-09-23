// Uses Timer but never calls update(), so getDelta()/getElapsed() stay at 0.
import { Timer } from 'three';

export function createTicker() {

	const timer = new Timer();

	return {
		tick( timestamp ) {

			return { delta: timer.getDelta(), elapsed: timer.getElapsed() };

		}
	};

}
