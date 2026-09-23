// A parameter list described as [required elements, optional elements, rest ([] or [X])].
type IPDescribe<L extends readonly unknown[], Req extends unknown[] = [], Opt extends unknown[] = []> =
  L extends readonly [] ? [Req, Opt, []]
  : L extends readonly [infer H, ...infer R]
    ? IPDescribe<R, [...Req, H], Opt>
    : IPOptionalTail<Required<L>, Req, Opt>

// L is not empty and does not start with a required element: its fixed elements
// are all optional (Required<> strips the marker), followed by an optional rest.
type IPOptionalTail<R extends readonly unknown[], Req extends unknown[], Opt extends unknown[]> =
  R extends readonly [] ? [Req, Opt, []]
  : R extends readonly [infer H, ...infer T] ? IPOptionalTail<T, Req, [...Opt, H]>
  : [Req, Opt, [R[number]]]

type IPMergeObjects<A, B> = A & B extends infer I ? { [K in keyof I]: I[K] } : never

type IPIsPlainObject<T> =
  T extends readonly unknown[] ? false : T extends (...args: never[]) => unknown ? false : T extends object ? true : false

// Intersection of two slot types; [] means "no constraint from this side".
type IPIntersect<A extends unknown[], B extends unknown[]> =
  A extends [infer a]
    ? B extends [infer b]
      ? IPIsPlainObject<a> extends true
        ? IPIsPlainObject<b> extends true ? IPMergeObjects<a, b> : a & b
        : a & b
      : a
    : B extends [infer b] ? b : never

// Next slot of one side: ['req' | 'opt' | 'none', type-as-tuple, remaining description].
type IPNext<D> =
  D extends [[infer H, ...infer R], infer O, infer X] ? ['req', [H], [R, O, X]]
  : D extends [[], [infer H, ...infer R], infer X] ? ['opt', [H], [[], R, X]]
  : D extends [[], [], infer X] ? ['none', X, D]
  : never

type IPCombine<L, R, Req extends unknown[] = [], Opt extends unknown[] = []> =
  [IPNext<L>, IPNext<R>] extends [
    [infer LK, infer LT extends unknown[], infer LD],
    [infer RK, infer RT extends unknown[], infer RD],
  ]
    ? [LK, RK] extends ['none', 'none']
      ? IPAssemble<Req, Opt, LT extends [] ? RT : RT extends [] ? LT : [IPIntersect<LT, RT>]>
      : 'req' extends LK | RK
        ? IPCombine<LD, RD, [...Req, IPIntersect<LT, RT>], Opt>
        : IPCombine<LD, RD, Req, [...Opt, IPIntersect<LT, RT>]>
    : never

type IPAssemble<Req extends unknown[], Opt extends unknown[], Rest extends unknown[]> =
  Rest extends [infer X] ? [...Req, ...Partial<Opt>, ...X[]] : [...Req, ...Partial<Opt>]

type IntersectParameters<
    l extends readonly unknown[],
    r extends readonly unknown[],
> = IPCombine<IPDescribe<l>, IPDescribe<r>>
