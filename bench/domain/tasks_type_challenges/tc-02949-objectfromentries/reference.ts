type ObjectFromEntries<T> = {
  [K in T extends [infer Key extends PropertyKey, unknown] ? Key : never]:
    T extends [K, infer V] ? V : never
}
