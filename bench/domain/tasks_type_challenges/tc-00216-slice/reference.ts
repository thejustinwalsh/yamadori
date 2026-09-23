type SlTuple<N extends number, R extends unknown[] = []> =
  R['length'] extends N ? R : SlTuple<N, [...R, unknown]>

// Normalize a possibly negative index against the array length (clamped at 0).
type SlNorm<N extends number, A extends unknown[]> =
  `${N}` extends `-${infer P extends number}`
    ? A extends [...infer F, ...SlTuple<P>] ? F['length'] : 0
    : N

type SlTake<A extends unknown[], N extends number, R extends unknown[] = []> =
  R['length'] extends N ? R
  : A extends [infer H, ...infer T] ? SlTake<T, N, [...R, H]>
  : R

type SlDrop<A extends unknown[], N extends number, C extends unknown[] = []> =
  C['length'] extends N ? A
  : A extends [unknown, ...infer T] ? SlDrop<T, N, [...C, unknown]>
  : []

type Slice<
  Arr extends unknown[] = [],
  Start extends number = 0,
  End extends number = Arr['length'],
> = SlDrop<SlTake<Arr, SlNorm<End, Arr>>, SlNorm<Start, Arr>>
