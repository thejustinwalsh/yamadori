// M => minuend, S => subtrahend
type SubTuple<N extends number, R extends unknown[] = []> =
  R['length'] extends N ? R : SubTuple<N, [...R, unknown]>

type Subtract<M extends number, S extends number> =
  SubTuple<M> extends [...SubTuple<S>, ...infer Rest] ? Rest['length'] : never
