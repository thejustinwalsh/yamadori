import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import {
  array, boolean, fetchJson, literal, number, object, optional, ParseError, string,
  type Infer, type Parser,
} from './solution.ts';

// Flattens intersections at every level so `{a} & {b?}` and `{a; b?}` compare equal.
type Deep<T> = T extends (infer E)[] ? Deep<E>[] : T extends object ? { [K in keyof T]: Deep<T[K]> } : T;

const Address = object({ city: string(), zip: optional(string()) });
const User = object({
  id: number(),
  name: string(),
  admin: boolean(),
  email: optional(string()),
  roles: array(literal('reader', 'writer', 'owner')),
  address: Address,
});
type U = Infer<typeof User>;
type _u = Expect<Equal<Deep<U>, {
  id: number;
  name: string;
  admin: boolean;
  email?: string | undefined;
  roles: ('reader' | 'writer' | 'owner')[];
  address: { city: string; zip?: string | undefined };
}>>;
const Lit = literal(1, 2, 'x', true, null);
type _lit = Expect<Equal<Infer<typeof Lit>, 1 | 2 | 'x' | true | null>>;
const Nums = array(number());
type _arr = Expect<Equal<Infer<typeof Nums>, number[]>>;
const MaybeStr = optional(string());
type _opt = Expect<Equal<Infer<typeof MaybeStr>, string | undefined>>;
assert.equal(Lit.parse(null), null);
assert.equal(Lit.parse('x'), 'x');
assert.equal(MaybeStr.parse(undefined), undefined);
const asParser: Parser<U> = User;
void asParser;

const good = {
  id: 7, name: 'ann', admin: false, roles: ['reader', 'owner'],
  address: { city: 'Oslo' }, extra: 'dropped',
};
const u = User.parse(good);
type _pu = Expect<Equal<typeof u, U>>;
assert.equal(u.id, 7);
assert.deepEqual(u.roles, ['reader', 'owner']);
assert.equal(u.address.city, 'Oslo');
assert.equal('extra' in u, false, 'unknown keys are dropped');
assert.equal('email' in u, false, 'an absent optional key stays absent');
assert.equal('zip' in u.address, false);
assert.equal(User.parse({ ...good, email: 'a@b.c' }).email, 'a@b.c');

function failsAt(p: Parser<unknown>, input: unknown, path: (string | number)[]): void {
  assert.throws(() => p.parse(input), (e: unknown) => {
    assert.ok(e instanceof ParseError, `expected a ParseError, got ${e}`);
    assert.ok(e instanceof Error, 'ParseError is an Error');
    assert.deepEqual(e.path, path);
    return true;
  });
}
failsAt(User, { ...good, name: undefined }, ['name']);
failsAt(User, { ...good, id: '7' }, ['id']);
failsAt(User, { ...good, id: Number.NaN }, ['id']);
failsAt(User, { ...good, id: Infinity }, ['id']);
failsAt(User, { ...good, email: 42 }, ['email']);
failsAt(User, { ...good, roles: ['reader', 'admin'] }, ['roles', 1]);
failsAt(User, { ...good, roles: 'reader' }, ['roles']);
failsAt(User, { ...good, address: { city: 1 } }, ['address', 'city']);
failsAt(User, { ...good, address: { city: 'x', zip: false } }, ['address', 'zip']);
failsAt(User, null, []);
failsAt(User, [good], []);
failsAt(array(array(number())), [[1], [2, 'x']], [1, 1]);
failsAt(boolean(), 'true', []);
failsAt(Lit, 3, []);
failsAt(Lit, false, []);

// fetchJson with a stubbed fetch.
const realFetch = globalThis.fetch;
const requested: string[] = [];
const routes: Record<string, () => Response> = {
  '/u/7': () => Response.json(good),
  '/u/8': () => Response.json({ ...good, roles: ['nope'] }),
  '/u/9': () => Response.json({ error: 'not found' }, { status: 404 }),
};
globalThis.fetch = (async (input: string | URL | Request) => {
  const url = String(input);
  requested.push(url);
  return routes[url]!();
}) as typeof fetch;
try {
  const fetched = fetchJson('/u/7', User);
  type _f = Expect<Equal<typeof fetched, Promise<U>>>;
  assert.equal((await fetched).name, 'ann');
  await assert.rejects(fetchJson('/u/8', User), (e: unknown) => e instanceof ParseError && e.path.join() === 'roles,0');
  await assert.rejects(fetchJson('/u/9', User), (e: unknown) =>
    e instanceof Error && !(e instanceof ParseError) && /404/.test(e.message));
  assert.deepEqual(requested, ['/u/7', '/u/8', '/u/9']);
} finally {
  globalThis.fetch = realFetch;
}

function compileOnly(): void {
  // @ts-expect-error -- `admin` is boolean, not string
  const s: string = u.admin;
  // @ts-expect-error -- email may be undefined
  const e: string = u.email;
  // @ts-expect-error -- roles are a closed set
  const r: ('reader' | 'writer')[] = u.roles;
  void s; void e; void r;
}
void compileOnly;
