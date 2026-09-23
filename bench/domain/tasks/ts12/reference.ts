export type Handlers<U extends { kind: string }> = {
  [K in U['kind']]: (value: Extract<U, { kind: K }>) => unknown;
};

export function match<U extends { kind: string }, H extends Handlers<U>>(
  value: U,
  handlers: H,
): ReturnType<H[U['kind']]> {
  const handler = handlers[value.kind as U['kind']] as unknown as (v: U) => ReturnType<H[U["kind"]]>;
  return handler(value);
}

export function assertNever(value: never): never {
  throw new Error(`unexpected value: ${JSON.stringify(value)}`);
}
