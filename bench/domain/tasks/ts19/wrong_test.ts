// S is inferred from `initial` as well, so a typo in `initial` silently widens the state union.
export function createMachine<S extends string>(config: { states: readonly S[]; initial: S }) {
  let current: S = config.initial;
  return {
    states: config.states,
    get current() { return current; },
    go(to: S) {
      if (!config.states.includes(to)) throw new RangeError(`unknown state: ${to}`);
      current = to;
    },
  };
}
