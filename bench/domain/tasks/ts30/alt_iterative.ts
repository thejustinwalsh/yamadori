// Character-at-a-time CamelCase with an accumulator, a homomorphic array mapping, and a loop-based runtime.
type Camel<S extends string, Acc extends string = ''> =
  S extends `_${infer C}${infer Rest}` ? Camel<Rest, `${Acc}${Uppercase<C>}`>
  : S extends `${infer C}${infer Rest}` ? Camel<Rest, `${Acc}${C}`>
  : Acc;

export type CamelCase<S extends string> = Camel<S>;

type ObjectKeys<T> = { [K in keyof T as K extends string ? Camel<K> : K]: CamelCaseKeys<T[K]> };

export type CamelCaseKeys<T> =
  T extends string | number | boolean | bigint | symbol | null | undefined ? T
  : T extends readonly unknown[] ? { [I in keyof T]: CamelCaseKeys<T[I]> }
  : ObjectKeys<T>;

export function camelizeKeys<T>(value: T): CamelCaseKeys<T> {
  const walk = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(walk);
    if (typeof v !== 'object' || v === null) return v;
    const out: Record<string, unknown> = {};
    for (const key in v as Record<string, unknown>) {
      let k = '';
      for (let i = 0; i < key.length; i++) {
        if (key[i] === '_' && i + 1 < key.length) k += key[++i]!.toUpperCase();
        else k += key[i];
      }
      out[k] = walk((v as Record<string, unknown>)[key]);
    }
    return out;
  };
  return walk(value) as CamelCaseKeys<T>;
}
