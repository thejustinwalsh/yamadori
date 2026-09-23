// A separate R type parameter cannot be inferred through the mapped type: every call returns unknown.
export function match<U extends { kind: string }, R>(
  value: U,
  handlers: { [K in U['kind']]: (v: Extract<U, { kind: K }>) => R },
): R {
  return (handlers[value.kind as U['kind']] as (v: U) => R)(value);
}
export function assertNever(value: never): never {
  throw new Error(`unexpected: ${String(value)}`);
}
