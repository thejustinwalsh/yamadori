// Tests whether the property TYPE admits undefined: `c: number | undefined` and `e: unknown` are misclassified.
export type OptionalKeys<T> = {
  [K in keyof T]-?: undefined extends T[K] ? K : never;
}[keyof T];
export type RequiredKeys<T> = Exclude<keyof T, OptionalKeys<T>>;
export type SetOptional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
