import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { props } from './solution.ts';

type User = { id: number; name: string };
const delay = <T,>(ms: number, v: T) => new Promise<T>((r) => setTimeout(() => r(v), ms));
const fetchUser = async (): Promise<User> => ({ id: 1, name: 'ann' });

const p = props({
  user: fetchUser(),
  count: delay(20, 3),
  label: 'plain',
  nested: Promise.resolve(Promise.resolve(true)),
  maybe: null as Promise<number> | null,
});
type _p = Expect<Equal<typeof p, Promise<{
  user: User;
  count: number;
  label: string;
  nested: boolean;
  maybe: number | null;
}>>>;
const r = await p;
assert.deepEqual(r, { user: { id: 1, name: 'ann' }, count: 3, label: 'plain', nested: true, maybe: null });
assert.deepEqual(Object.keys(r), ['user', 'count', 'label', 'nested', 'maybe'], 'key order is preserved');

// Rejection: rejects with the first rejection, without waiting for slower keys.
const err = new Error('nope');
const settled = await new Promise((resolve) => {
  const timer = setTimeout(() => resolve('still pending after 2 s'), 2000);
  props({ slow: new Promise<number>(() => {}), bad: Promise.reject(err) }).then(
    (value) => { clearTimeout(timer); resolve({ fulfilled: value }); },
    (reason) => { clearTimeout(timer); resolve({ rejected: reason }); },
  );
});
assert.deepEqual(settled, { rejected: err }, 'must reject with the first rejection without waiting for pending keys');

// Thenables are unwrapped like await does.
const thenable = { then(res: (v: string) => void) { res('t'); } };
const th = await props({ th: thenable });
type _th = Expect<Equal<typeof th, { th: string }>>;
assert.deepEqual(th, { th: 't' });

const empty = await props({});
assert.deepEqual(empty, {});

interface Deps { db: Promise<'db'>; cache: 'cache' }
const deps: Deps = { db: Promise.resolve('db'), cache: 'cache' };
const d = await props(deps);
type _d = Expect<Equal<typeof d, { db: 'db'; cache: 'cache' }>>;
assert.deepEqual(d, { db: 'db', cache: 'cache' });
