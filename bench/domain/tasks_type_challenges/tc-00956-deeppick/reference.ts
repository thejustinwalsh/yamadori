type DeepPickPath<T, P extends string> =
  P extends keyof T
    ? { [K in P]: T[P] }
    : P extends `${infer Head}.${infer Rest}`
      ? Head extends keyof T
        ? { [K in Head]: DeepPickPath<T[Head], Rest> }
        : unknown
      : unknown

// Each path is picked on its own and the picks are intersected (a missing path contributes `unknown`).
type DeepPick<T, K extends string> =
  (K extends unknown ? (arg: DeepPickPath<T, K>) => void : never) extends (arg: infer I) => void
    ? I
    : never
