// Every handler receives the whole union, so member-specific fields are not accessible.
export function match<U extends { kind: string }, R>(value: U, handlers: Record<U['kind'], (v: U) => R>): R {
  return handlers[value.kind as U['kind']](value);
}
export function assertNever(value: never): never {
  throw new Error(`unexpected: ${String(value)}`);
}
