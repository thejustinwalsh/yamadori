// Variadic-tuple inference via [...R] instead of a const type parameter; Extract-based helpers.
export type Result<T, E> = { readonly ok: true; readonly value: T } | { readonly ok: false; readonly error: E };

export function ok<T>(value: T): Result<T, never> { return { ok: true, value }; }
export function err<E>(error: E): Result<never, E> { return { ok: false, error }; }

export const map = <T, U, E>(r: Result<T, E>, f: (value: T) => U): Result<U, E> =>
  r.ok ? { ok: true, value: f(r.value) } : { ok: false, error: r.error };

export const andThen = <T, U, E, F>(r: Result<T, E>, f: (value: T) => Result<U, F>): Result<U, E | F> =>
  r.ok ? f(r.value) : { ok: false, error: r.error };

export function tryCatch<T>(f: () => T): Result<T, Error> {
  try { return { ok: true, value: f() }; }
  catch (thrown) { return { ok: false, error: thrown instanceof Error ? thrown : new Error(String(thrown)) }; }
}

type Values<R extends unknown[]> = { [K in keyof R]: Extract<R[K], { ok: true }> extends { value: infer V } ? V : never };
type Errors<R extends unknown[]> = Extract<R[number], { ok: false; error: unknown }>['error'];

export function all<R extends Result<unknown, unknown>[]>(results: [...R]): Result<Values<R>, Errors<R>> {
  const firstErr = results.find((r) => !r.ok);
  if (firstErr && !firstErr.ok) return { ok: false, error: firstErr.error as Errors<R> };
  return { ok: true, value: results.map((r) => (r as { value: unknown }).value) as Values<R> };
}
