// number() only checks typeof, so NaN and Infinity are accepted.
export class ParseError extends Error {
  constructor(message: string, readonly path: (string | number)[] = []) { super(message); }
}
export interface Parser<T> { parse(input: unknown): T }
export type Infer<P> = P extends Parser<infer T> ? T : never;
const mk = <T>(parse: (input: unknown) => T): Parser<T> => ({ parse });
const nest = <T>(key: string | number, run: () => T): T => {
  try { return run(); } catch (e) {
    if (e instanceof ParseError) throw new ParseError(e.message, [key, ...e.path]);
    throw e;
  }
};
export const string = () => mk((x) => { if (typeof x !== 'string') throw new ParseError('string'); return x; });
export const number = () => mk((x) => { if (typeof x !== 'number') throw new ParseError('number'); return x; });
export const boolean = () => mk((x) => { if (typeof x !== 'boolean') throw new ParseError('boolean'); return x; });
export const literal = <const V extends readonly (string | number | boolean | null)[]>(...values: V) =>
  mk((x) => { if (!values.includes(x as V[number])) throw new ParseError('literal'); return x as V[number]; });
export const array = <T>(item: Parser<T>) =>
  mk((x) => { if (!Array.isArray(x)) throw new ParseError('array'); return x.map((el, i) => nest(i, () => item.parse(el))); });
export const optional = <T>(inner: Parser<T>) => mk((x) => (x === undefined ? undefined : inner.parse(x)));
type Opt<S> = { [K in keyof S]: undefined extends Infer<S[K]> ? K : never }[keyof S];
export const object = <S extends Record<string, Parser<unknown>>>(shape: S) =>
  mk((x) => {
    if (typeof x !== 'object' || x === null || Array.isArray(x)) throw new ParseError('object');
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(shape)) {
      const v = nest(k, () => shape[k]!.parse((x as Record<string, unknown>)[k]));
      if (v !== undefined) out[k] = v;
    }
    return out as { [K in Exclude<keyof S, Opt<S>>]: Infer<S[K]> } & { [K in Opt<S>]?: Infer<S[K]> };
  });
export async function fetchJson<T>(url: string, p: Parser<T>): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return p.parse(await res.json());
}
