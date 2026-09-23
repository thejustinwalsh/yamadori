type OUFlatten<T> = { [K in keyof T]: T[K] }

type OptionalUndefined<T, Props extends PropertyKey = keyof T> = OUFlatten<
  & { [K in keyof T as K extends Props ? (undefined extends T[K] ? never : K) : K]: T[K] }
  & { [K in keyof T as K extends Props ? (undefined extends T[K] ? K : never) : never]?: T[K] }
>
