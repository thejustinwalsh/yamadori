type IntersectionMembers<X> = X extends readonly unknown[] ? X[number] : X

type Intersection<T extends readonly unknown[]> =
  T extends readonly [infer F, ...infer R]
    ? R extends []
      ? IntersectionMembers<F>
      : Extract<IntersectionMembers<F>, Intersection<R>>
    : never
