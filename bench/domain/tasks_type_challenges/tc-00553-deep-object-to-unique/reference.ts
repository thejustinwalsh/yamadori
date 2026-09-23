declare const deepObjectUniqueBrand: unique symbol

// Every object is branded with its root type and the path that reached it, under an
// optional symbol key: string/number keys are untouched and the result stays mutually
// assignable with the original, yet distinct objects no longer compare identical.
type DeepObjectToUniqAt<O extends object, Root, Path extends PropertyKey[]> = {
  [K in keyof O]: O[K] extends object ? DeepObjectToUniqAt<O[K], Root, [...Path, K]> : O[K]
} & {
  readonly [deepObjectUniqueBrand]?: [Root, Path]
}

type DeepObjectToUniq<O extends object> = DeepObjectToUniqAt<O, O, []>
