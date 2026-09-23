// Replace the members of T that are mutually assignable with From.
type URStep<T, From, To> = T extends From ? ([From] extends [T] ? To : T) : T

type UnionReplace<T, U extends [any, any][]> =
  U extends [infer P extends [unknown, unknown], ...infer R extends [any, any][]]
    ? UnionReplace<URStep<T, P[0], P[1]>, R>
    : T
