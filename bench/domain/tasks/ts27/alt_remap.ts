// Key remapping to filter, `keyof` of the filtered object, and a single mapped type for SetOptional.
type OnlyOptional<T> = { [K in keyof T as T extends Record<K, T[K]> ? never : K]: T[K] };

export type OptionalKeys<T> = keyof OnlyOptional<T>;
export type RequiredKeys<T> = keyof { [K in keyof T as T extends Record<K, T[K]> ? K : never]: 0 };

export type SetOptional<T, K extends keyof T> = {
  [P in keyof T as P extends K ? never : P]: T[P];
} & {
  [P in K]?: T[P];
};
