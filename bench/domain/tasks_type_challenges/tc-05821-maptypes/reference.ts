type MapTypesTarget<V, R> = R extends { mapFrom: infer From, mapTo: infer To }
  ? [V] extends [From] ? To : never
  : never

type MapTypesHasMatch<V, R> = R extends { mapFrom: infer From }
  ? [V] extends [From] ? true : never
  : never

type MapTypes<T, R extends { mapFrom: unknown, mapTo: unknown }> = {
  [K in keyof T]: [MapTypesHasMatch<T[K], R>] extends [never] ? T[K] : MapTypesTarget<T[K], R>
}
