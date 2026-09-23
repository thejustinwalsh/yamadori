// `current` is captured once at creation: the getter-less snapshot never changes after go().
export function createMachine<S extends string>(config: { states: readonly S[]; initial: NoInfer<S> }) {
  const m = {
    states: config.states,
    current: config.initial as S,
    go(to: S) {
      if (!config.states.includes(to)) throw new RangeError(`unknown state: ${to}`);
      m.current = to;
    },
  };
  return { states: m.states, current: m.current, go: m.go };
}
