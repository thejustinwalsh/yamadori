type KebabCaseRest<S extends string> = S extends `${infer F}${infer R}`
  ? F extends Lowercase<F>
    ? `${F}${KebabCaseRest<R>}`
    : `-${Lowercase<F>}${KebabCaseRest<R>}`
  : S

type KebabCase<S extends string> = S extends `${infer F}${infer R}`
  ? `${Lowercase<F>}${KebabCaseRest<R>}`
  : S
