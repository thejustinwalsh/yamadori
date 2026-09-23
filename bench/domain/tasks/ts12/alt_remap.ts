// Key remapping over the union members instead of Extract, and a conditional for the result.
type Cases<U extends { readonly kind: string }> = { [M in U as M['kind']]: (member: M) => any };
type Out<H> = { [K in keyof H]: H[K] extends (...a: any[]) => infer R ? R : never }[keyof H];

export function match<U extends { readonly kind: string }, H extends Cases<U>>(value: U, handlers: H): Out<H> {
  const table = handlers as unknown as Record<string, (member: U) => Out<H>>;
  return table[value.kind]!(value);
}

export const assertNever = (x: never): never => {
  throw new TypeError('unhandled case: ' + String((x as { kind?: unknown })?.kind ?? x));
};
