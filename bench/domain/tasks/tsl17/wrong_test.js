// Unported: Clock is an r183 deprecation in r185.
import { Clock } from 'three';

export function createTicker() {

	const clock = new Clock();

	return {
		tick( timestamp ) {

			return { delta: clock.getDelta(), elapsed: clock.getElapsedTime() };

		}
	};

}
