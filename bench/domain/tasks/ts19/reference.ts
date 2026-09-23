export interface Machine<S extends string> {
  readonly states: readonly S[];
  readonly current: S;
  go(to: S): void;
}

export function createMachine<S extends string>(config: { states: readonly S[]; initial: NoInfer<S> }): Machine<S> {
  let current: S = config.initial;
  return {
    states: config.states,
    get current() {
      return current;
    },
    go(to: S) {
      if (!config.states.includes(to)) throw new RangeError(`unknown state: ${to}`);
      current = to;
    },
  };
}
