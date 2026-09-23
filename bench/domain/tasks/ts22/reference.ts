export class ParseError extends Error {
  constructor(message: string, readonly path: (string | number)[] = []) {
    super(path.length ? `${path.join('.')}: ${message}` : message);
    this.name = 'ParseError';
  }
}

export interface Parser<T> {
  parse(input: unknown): T;
}

export type Infer<P> = P extends Parser<infer T> ? T : never;

const parser = <T>(parse: (input: unknown) => T): Parser<T> => ({ parse });

const nest = <T>(key: string | number, run: () => T): T => {
  try {
    return run();
  } catch (e) {
    if (e instanceof ParseError) throw new ParseError(e.message, [key, ...e.path]);
    throw e;
  }
};

export const string = (): Parser<string> =>
  parser((x) => {
    if (typeof x !== 'string') throw new ParseError('expected a string');
    return x;
  });

export const number = (): Parser<number> =>
  parser((x) => {
    if (typeof x !== 'number' || !Number.isFinite(x)) throw new ParseError('expected a finite number');
    return x;
  });

export const boolean = (): Parser<boolean> =>
  parser((x) => {
    if (typeof x !== 'boolean') throw new ParseError('expected a boolean');
    return x;
  });

export const literal = <const V extends readonly (string | number | boolean | null)[]>(...values: V): Parser<V[number]> =>
  parser((x) => {
    if (!values.includes(x as V[number])) throw new ParseError(`expected one of ${JSON.stringify(values)}`);
    return x as V[number];
  });

export const array = <T>(item: Parser<T>): Parser<T[]> =>
  parser((x) => {
    if (!Array.isArray(x)) throw new ParseError('expected an array');
    return x.map((el, i) => nest(i, () => item.parse(el)));
  });

export const optional = <T>(inner: Parser<T>): Parser<T | undefined> =>
  parser((x) => (x === undefined ? undefined : inner.parse(x)));

type Shape = Record<string, Parser<unknown>>;
type OptionalKeys<S extends Shape> = { [K in keyof S]: undefined extends Infer<S[K]> ? K : never }[keyof S];
type ObjectOf<S extends Shape> =
  { [K in Exclude<keyof S, OptionalKeys<S>>]: Infer<S[K]> } &
  { [K in OptionalKeys<S>]?: Infer<S[K]> };

export const object = <S extends Shape>(shape: S): Parser<ObjectOf<S>> =>
  parser((x) => {
    if (typeof x !== 'object' || x === null || Array.isArray(x)) throw new ParseError('expected an object');
    const src = x as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(shape)) {
      const value = nest(key, () => shape[key]!.parse(src[key]));
      if (value !== undefined || key in src) out[key] = value;
    }
    return out as ObjectOf<S>;
  });

export async function fetchJson<T>(url: string, p: Parser<T>): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed with HTTP ${res.status}`);
  return p.parse(await res.json());
}
