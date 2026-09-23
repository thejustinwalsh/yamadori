// Types states as plain string[]: no literal union, so nothing is checked.
export function createMachine(config: { states: readonly string[]; initial: string }) {
  let current = config.initial;
  return {
    states: config.states,
    get current() { return current; },
    go(to: string) {
      if (!config.states.includes(to)) throw new RangeError(`unknown state: ${to}`);
      current = to;
    },
  };
}
