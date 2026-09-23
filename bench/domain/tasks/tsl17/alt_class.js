import { Timer } from 'three';

class Ticker {

	constructor() {

		this.timer = new Timer();

	}

	tick( timestamp ) {

		const { timer } = this;
		timer.update( timestamp );
		const delta = timer.getDelta();
		const elapsed = timer.getElapsed();
		return { delta, elapsed };

	}

}

export function createTicker() {

	return new Ticker();

}
