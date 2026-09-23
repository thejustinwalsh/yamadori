// Every non-empty prefix of a parameter list.
type CurryPrefixes<P extends unknown[]> =
  P extends [...infer F, unknown] ? P | CurryPrefixes<F> : never

type CurryDrop<P extends unknown[], A extends unknown[]> =
  P extends [...{ [K in keyof A]: unknown }, ...infer Rest] ? Rest : never

type CurryFn<P extends unknown[], R> = <A extends CurryPrefixes<P>>(
  ...args: A
) => CurryDrop<P, A> extends infer Rest extends unknown[]
  ? Rest extends [] ? R : CurryFn<Rest, R>
  : never

declare function DynamicParamsCurrying<P extends unknown[], R>(
  fn: (...args: P) => R,
): P extends [] ? R : CurryFn<P, R>
