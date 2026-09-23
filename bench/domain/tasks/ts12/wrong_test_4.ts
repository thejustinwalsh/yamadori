// assertNever takes `unknown`, so it no longer proves exhaustiveness.
export function match<U extends { kind: string }, H extends { [K in U['kind']]: (v: Extract<U, { kind: K }>) => unknown }>(
  value: U,
  handlers: H,
): ReturnType<H[U['kind']]> {
  return (handlers[value.kind as U['kind']] as unknown as (v: U) => any)(value);
}
export function assertNever(value: unknown): never {
  throw new Error(`unexpected: ${String(value)}`);
}
