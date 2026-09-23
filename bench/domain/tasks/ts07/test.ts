import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { defineRoutes, pathOf, type Method, type RouteDef } from './solution.ts';

const input = {
  home: { path: '/', method: 'GET', tags: ['public', 'cached'] },
  createUser: { path: '/users', method: 'POST' },
  deleteUser: { path: '/users/:id', method: 'DELETE', tags: [] },
} as const satisfies Record<string, RouteDef>;

const routes = defineRoutes({
  home: { path: '/', method: 'GET', tags: ['public', 'cached'] },
  createUser: { path: '/users', method: 'POST' },
  deleteUser: { path: '/users/:id', method: 'DELETE', tags: [] },
});

type _p = Expect<Equal<typeof routes.home.path, '/'>>;
type _m = Expect<Equal<typeof routes.createUser.method, 'POST'>>;
type _t = Expect<Equal<typeof routes.home.tags, readonly ['public', 'cached']>>;
type _t0 = Expect<Equal<typeof routes.deleteUser.tags, readonly []>>;
type _k = Expect<Equal<keyof typeof routes, 'home' | 'createUser' | 'deleteUser'>>;
type _meth = Expect<Equal<Method, 'GET' | 'POST' | 'PUT' | 'DELETE'>>;

const p = pathOf(routes, 'deleteUser');
type _po = Expect<Equal<typeof p, '/users/:id'>>;
assert.equal(p, '/users/:id');
assert.equal(pathOf(routes, 'home'), '/');
assert.deepEqual(routes, input);

const same = { a: { path: '/a', method: 'PUT' } } as const;
assert.equal(defineRoutes(same), same, 'returns its argument unchanged');

function compileOnly(): void {
  // @ts-expect-error -- paths must start with a slash
  defineRoutes({ bad: { path: 'users', method: 'GET' } });
  // @ts-expect-error -- unknown HTTP method
  defineRoutes({ bad: { path: '/x', method: 'FETCH' } });
  // @ts-expect-error -- tags must be strings
  defineRoutes({ bad: { path: '/x', method: 'GET', tags: [1] } });
  // @ts-expect-error -- method is required
  defineRoutes({ bad: { path: '/x' } });
  // @ts-expect-error -- not a route name
  pathOf(routes, 'nope');
}
void compileOnly;
