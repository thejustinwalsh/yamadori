type UnionToObjectFromKey<Union, Key> =
  Union extends unknown ? (Key extends keyof Union ? Union : never) : never
