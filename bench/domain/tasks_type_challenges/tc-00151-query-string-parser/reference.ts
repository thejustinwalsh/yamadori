type QSPair<S extends string> = S extends `${infer K}=${infer V}` ? [K, V] : [S, true]

type QSSplit<S extends string> =
  S extends '' ? []
  : S extends `${infer H}&${infer R}`
    ? H extends '' ? QSSplit<R> : [QSPair<H>, ...QSSplit<R>]
    : [QSPair<S>]

type QSMergeValue<Old, V> =
  Old extends unknown[]
    ? (V extends Old[number] ? Old : [...Old, V])
    : (V extends Old ? Old : [Old, V])

type QSAdd<O, K extends string, V> = {
  [P in keyof O | K]: P extends K
    ? (P extends keyof O ? QSMergeValue<O[P], V> : V)
    : P extends keyof O ? O[P] : never
}

type QSBuild<Pairs extends unknown[], O = {}> =
  Pairs extends [[infer K extends string, infer V], ...infer R]
    ? QSBuild<R, QSAdd<O, K, V>>
    : O

type ParseQueryString<S extends string> = QSBuild<QSSplit<S>>
