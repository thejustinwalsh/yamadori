type Combination<T extends string[], All extends string = T[number], U extends string = All> =
  U extends U
    ? U | `${U} ${Combination<[], Exclude<All, U>>}`
    : never
