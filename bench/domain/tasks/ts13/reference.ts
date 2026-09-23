export type Result<T, E> = { readonly ok: true; readonly value: T } | { readonly ok: false; readonly error: E };

export const ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
export const err = <E>(error: E): Result<never, E> => ({ ok: false, error });

export function map<T, U, E>(r: Result<T, E>, f: (value: T) => U): Result<U, E> {
  return r.ok ? ok(f(r.value)) : r;
}

export function andThen<T, U, E, F>(r: Result<T, E>, f: (value: T) => Result<U, F>): Result<U, E | F> {
  return r.ok ? f(r.value) : r;
}

export function tryCatch<T>(f: () => T): Result<T, Error> {
  try {
    return ok(f());
  } catch (e) {
    return err(e instanceof Error ? e : new Error(String(e)));
  }
}

type ValueOf<R> = R extends { ok: true; value: infer T } ? T : never;
type ErrorOf<R> = R extends { ok: false; error: infer E } ? E : never;

export function all<const R extends readonly Result<unknown, unknown>[]>(
  results: R,
): Result<{ -readonly [K in keyof R]: ValueOf<R[K]> }, ErrorOf<R[number]>> {
  const values: unknown[] = [];
  for (const r of results) {
    if (!r.ok) return r as Result<never, ErrorOf<R[number]>>;
    values.push(r.value);
  }
  return ok(values as { -readonly [K in keyof R]: ValueOf<R[K]> });
}
