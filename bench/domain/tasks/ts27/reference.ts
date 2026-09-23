export type OptionalKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? K : never;
}[keyof T];

export type RequiredKeys<T> = Exclude<keyof T, OptionalKeys<T>>;

export type SetOptional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
