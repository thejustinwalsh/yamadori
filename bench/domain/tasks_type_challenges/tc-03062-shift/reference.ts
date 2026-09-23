type Shift<T extends readonly unknown[]> = T extends readonly [unknown, ...infer R] ? R : []
