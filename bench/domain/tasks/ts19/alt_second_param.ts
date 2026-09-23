// The pre-NoInfer idiom: a second type parameter constrained by the first, and a class.
class StateMachine<S extends string> {
  #current: S;
  constructor(readonly states: readonly S[], initial: S) {
    this.#current = initial;
  }
  get current(): S {
    return this.#current;
  }
  go(to: S): void {
    if (this.states.indexOf(to) < 0) throw new RangeError('no such state');
    this.#current = to;
  }
}

export function createMachine<S extends string, I extends S>(config: { states: readonly S[]; initial: I }): StateMachine<S> {
  return new StateMachine<S>(config.states, config.initial);
}
