type Reverse<T extends readonly unknown[]> = T extends readonly [infer F, ...infer R]
  ? [...Reverse<R>, F]
  : []
