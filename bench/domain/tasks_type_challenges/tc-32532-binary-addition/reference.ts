type Bit = 1 | 0

// [carry, sum] for a + b + carry-in
type BATable = {
  '000': [0, 0], '001': [0, 1], '010': [0, 1], '011': [1, 0],
  '100': [0, 1], '101': [1, 0], '110': [1, 0], '111': [1, 1],
}
type BALast<T extends Bit[]> = T extends [...Bit[], infer L extends Bit] ? L : 0
type BAInit<T extends Bit[]> = T extends [...infer I extends Bit[], Bit] ? I : []

type BAStep<A extends Bit[], B extends Bit[], Carry extends Bit, Out extends Bit[]> =
  [A, B] extends [[], []]
    ? Carry extends 1 ? [1, ...Out] : Out
    : BATable[`${BALast<A>}${BALast<B>}${Carry}`] extends [infer C extends Bit, infer S extends Bit]
      ? BAStep<BAInit<A>, BAInit<B>, C, [S, ...Out]>
      : never

type BinaryAdd<A extends Bit[], B extends Bit[]> = BAStep<A, B, 0, []>
