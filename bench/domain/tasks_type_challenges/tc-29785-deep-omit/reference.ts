type DeepOmit<T, P extends string> =
  P extends `${infer Head}.${infer Rest}`
    ? { [K in keyof T]: K extends Head ? DeepOmit<T[K], Rest> : T[K] }
    : { [K in keyof T as K extends P ? never : K]: T[K] }
