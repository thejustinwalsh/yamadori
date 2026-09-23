import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { buildPath, type PathParams } from './solution.ts';

// Flattens intersections so `{a} & {b}` and `{a; b}` compare equal.
type Flat<T> = { [K in keyof T]: T[K] };

type _two = Expect<Equal<Flat<PathParams<'/users/:id/posts/:postId'>>, { id: string; postId: string }>>;
type _trailing = Expect<Equal<Flat<PathParams<'/users/:id'>>, { id: string }>>;
type _first = Expect<Equal<Flat<PathParams<'/:org/repos'>>, { org: string }>>;
type _three = Expect<Equal<Flat<PathParams<'/:a/x/:b/y/:c'>>, { a: string; b: string; c: string }>>;
type _snake = Expect<Equal<Flat<PathParams<'/files/:file_name'>>, { file_name: string }>>;
type _none = Expect<Equal<Flat<PathParams<'/health'>>, {}>>;
type _root = Expect<Equal<Flat<PathParams<'/'>>, {}>>;

assert.equal(buildPath('/users/:id/posts/:postId', { id: '42', postId: 'a b/c' }), '/users/42/posts/a%20b%2Fc');
assert.equal(buildPath('/users/:id', { id: 'x' }), '/users/x');
assert.equal(buildPath('/:org/repos', { org: 'acme' }), '/acme/repos');
assert.equal(buildPath('/health', {}), '/health');
assert.equal(buildPath('/:a/x/:b/y/:c', { a: '1', b: '2', c: '3' }), '/1/x/2/y/3');
const ret = buildPath('/users/:id', { id: '1' });
type _ret = Expect<Equal<typeof ret, string>>;

function compileOnly(): void {
  // @ts-expect-error -- postId is missing
  buildPath('/users/:id/posts/:postId', { id: '42' });
  // @ts-expect-error -- id is missing (trailing parameter)
  buildPath('/users/:id', {});
  // @ts-expect-error -- values must be strings
  buildPath('/users/:id', { id: 42 });
  // @ts-expect-error -- wrong parameter name
  buildPath('/users/:id', { ID: '42' });
}
void compileOnly;
