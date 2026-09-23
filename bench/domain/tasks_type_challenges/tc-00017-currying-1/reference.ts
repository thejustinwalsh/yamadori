// A constraint that admits every type but names the primitive kinds keeps literal
// return types (`() => true` stays `true`) instead of widening them.
type CurryingReturn = string | number | bigint | boolean | symbol | object | null | undefined | void

type CurriedChain<Args extends unknown[], R> =
  Args extends [infer Head, ...infer Tail]
    ? (arg: Head) => CurriedChain<Tail, R>
    : R

type Curried<Args extends unknown[], R> =
  Args extends [] ? () => R : CurriedChain<Args, R>

declare function Currying<Args extends unknown[], R extends CurryingReturn>(fn: (...args: Args) => R): Curried<Args, R>
