type FindAll<T extends string, P extends string, I extends unknown[] = [], Acc extends number[] = []> =
  P extends ''
    ? []
    : T extends `${infer _Head}${infer Rest}`
      ? FindAll<Rest, P, [...I, unknown], T extends `${P}${string}` ? [...Acc, I['length']] : Acc>
      : Acc
