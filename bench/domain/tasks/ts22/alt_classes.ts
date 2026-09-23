// Class-based parsers with an explicit optional marker type, a path threaded through
// an internal method, and optional keys computed with key remapping.
export class ParseError extends Error {
  readonly path: (string | number)[];
  constructor(message: string, path: (string | number)[]) {
    super(`${message} at ${path.length ? path.join('/') : '<root>'}`);
    this.path = path;
  }
}

export abstract class Parser<T> {
  declare readonly _out: T;
  abstract run(input: unknown, path: (string | number)[]): T;
  parse(input: unknown): T {
    return this.run(input, []);
  }
}

export type Infer<P> = P extends Parser<infer T> ? T : never;

class Prim<T> extends Parser<T> {
  constructor(private readonly check: (x: unknown) => x is T, private readonly what: string) { super(); }
  run(input: unknown, path: (string | number)[]): T {
    if (!this.check(input)) throw new ParseError(`expected ${this.what}`, path);
    return input;
  }
}

class Opt<T> extends Parser<T | undefined> {
  readonly optional = true;
  constructor(readonly inner: Parser<T>) { super(); }
  run(input: unknown, path: (string | number)[]): T | undefined {
    return input === undefined ? undefined : this.inner.run(input, path);
  }
}

class Arr<T> extends Parser<T[]> {
  constructor(private readonly item: Parser<T>) { super(); }
  run(input: unknown, path: (string | number)[]): T[] {
    if (!Array.isArray(input)) throw new ParseError('expected array', path);
    return input.map((x, i) => this.item.run(x, [...path, i]));
  }
}

type Fields = { [key: string]: Parser<any> };
type Out<F extends Fields> = {
  [K in keyof F as F[K] extends Opt<any> ? never : K]: Infer<F[K]>;
} & {
  [K in keyof F as F[K] extends Opt<any> ? K : never]?: Infer<F[K]>;
};

class Obj<F extends Fields> extends Parser<Out<F>> {
  constructor(private readonly fields: F) { super(); }
  run(input: unknown, path: (string | number)[]): Out<F> {
    if (input === null || typeof input !== 'object' || Array.isArray(input)) throw new ParseError('expected object', path);
    const result: { [k: string]: unknown } = {};
    for (const [k, p] of Object.entries(this.fields)) {
      const raw = (input as { [k: string]: unknown })[k];
      if (raw === undefined && p instanceof Opt) continue;
      result[k] = p.run(raw, [...path, k]);
    }
    return result as Out<F>;
  }
}

export const string = () => new Prim((x): x is string => typeof x === 'string', 'string');
export const number = () => new Prim((x): x is number => typeof x === 'number' && Number.isFinite(x), 'finite number');
export const boolean = () => new Prim((x): x is boolean => typeof x === 'boolean', 'boolean');
export function literal<V extends (string | number | boolean | null)[]>(...values: V): Parser<V[number]> {
  return new Prim((x): x is V[number] => values.some((v) => v === x), values.join(' | '));
}
export const array = <T>(item: Parser<T>) => new Arr(item);
export const optional = <T>(inner: Parser<T>) => new Opt(inner);
export const object = <F extends Fields>(fields: F) => new Obj(fields);

export async function fetchJson<T>(url: string, parser: Parser<T>): Promise<T> {
  const response = await fetch(url);
  if (response.status < 200 || response.status > 299) {
    throw new Error(`request failed: ${response.status} ${response.statusText}`);
  }
  const body: unknown = await response.json();
  return parser.parse(body);
}
