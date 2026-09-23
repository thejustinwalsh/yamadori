// Type-level assertion helpers shared by every `ts` grader (copied into each
// sandbox next to test.ts). `Equal` is the invariant-identity trick: it is
// false for `any` against anything else, so an answer typed `any` cannot
// satisfy an exact-type assertion.
export type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends (<T>() => T extends Y ? 1 : 2) ? true : false;
export type Expect<T extends true> = T;
export type IsAny<T> = 0 extends 1 & T ? true : false;
