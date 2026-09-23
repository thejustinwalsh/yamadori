import type { Equal, Expect } from './assert_types.ts';
import type { OptionalKeys, RequiredKeys, SetOptional } from './solution.ts';

type Flat<T> = { [K in keyof T]: T[K] };

type T1 = {
  a: number;
  b?: string;
  c: number | undefined;
  readonly d?: boolean;
  e: unknown;
  f: any;
  g?: undefined;
};
type _o1 = Expect<Equal<OptionalKeys<T1>, 'b' | 'd' | 'g'>>;
type _r1 = Expect<Equal<RequiredKeys<T1>, 'a' | 'c' | 'e' | 'f'>>;

interface Iface { id: string; note?: string; [Symbol.iterator]?: () => Iterator<number> }
type _o2 = Expect<Equal<OptionalKeys<Iface>, 'note' | typeof Symbol.iterator>>;
type _r2 = Expect<Equal<RequiredKeys<Iface>, 'id'>>;

type _o3 = Expect<Equal<OptionalKeys<{}>, never>>;
type _o4 = Expect<Equal<OptionalKeys<{ x: 1 }>, never>>;
type _r4 = Expect<Equal<RequiredKeys<Partial<{ x: 1; y: 2 }>>, never>>;

type User = { id: number; name: string; email: string; readonly createdAt: Date };
type Draft = SetOptional<User, 'id' | 'createdAt'>;
type _s1 = Expect<Equal<Flat<Draft>, { name: string; email: string; id?: number; readonly createdAt?: Date }>>;
type _s2 = Expect<Equal<OptionalKeys<Draft>, 'id' | 'createdAt'>>;

const ok: Draft = { name: 'n', email: 'e' };
void ok;

function compileOnly(): void {
  // @ts-expect-error -- name is still required
  const missing: Draft = { email: 'e' };
  // @ts-expect-error -- only keys of T may be made optional
  type Bad = SetOptional<User, 'nope'>;
  void missing;
}
void compileOnly;
