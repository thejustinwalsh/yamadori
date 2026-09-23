// Arrow export with an inferred return type and differently named type parameters.
export const partial =
  <Bound extends unknown[], Rest extends unknown[], Out>(fn: (...args: [...Bound, ...Rest]) => Out, ...bound: Bound) =>
  (...rest: Rest): Out =>
    fn(...bound, ...rest);
