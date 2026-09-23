// Handlers are optional, so a missing case compiles (and throws at runtime).
export function match<U extends { kind: string }, H extends { [K in U['kind']]?: (v: Extract<U, { kind: K }>) => unknown }>(
  value: U,
  handlers: H,
): ReturnType<NonNullable<H[U['kind']]>> {
  const h = handlers[value.kind as U['kind']] as ((v: U) => any) | undefined;
  if (!h) throw new Error('no handler');
  return h(value);
}
export function assertNever(value: never): never {
  throw new Error(`unexpected: ${String(value)}`);
}
