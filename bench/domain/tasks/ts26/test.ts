import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { request, type RequestSpec } from './solution.ts';

const get = request().url('/users').method('GET').header('accept', 'application/json').build();
type _g = Expect<Equal<typeof get, RequestSpec>>;
assert.deepEqual(get, { url: '/users', method: 'GET', headers: { accept: 'application/json' } });
assert.equal('body' in get, false, 'no body key unless body() was called');

const post = request()
  .method('POST')
  .header('content-type', 'text/plain')
  .body('hello')
  .url('/notes')
  .header('x-trace', '1')
  .build();
assert.deepEqual(post, {
  url: '/notes', method: 'POST',
  headers: { 'content-type': 'text/plain', 'x-trace': '1' }, body: 'hello',
});

const overwrite = request().url('/a').method('PUT').header('h', '1').header('h', '2').body('x').body('y').build();
assert.deepEqual(overwrite, { url: '/a', method: 'PUT', headers: { h: '2' }, body: 'y' });

// Immutable: each step returns a new builder; earlier builders are unaffected.
const base = request().url('/items').header('a', '1');
const g = base.method('GET').build();
const d = base.method('DELETE').header('b', '2').body('why').build();
const g2 = base.method('GET').build();
assert.deepEqual(g, { url: '/items', method: 'GET', headers: { a: '1' } });
assert.deepEqual(d, { url: '/items', method: 'DELETE', headers: { a: '1', b: '2' }, body: 'why' });
assert.deepEqual(g2, g);
const spec = request().url('/x').method('GET').build();
spec.headers.z = '9';
assert.deepEqual(request().url('/x').method('GET').build().headers, {});

function compileOnly(): void {
  // @ts-expect-error -- method is missing
  request().url('/a').build();
  // @ts-expect-error -- url is missing
  request().method('GET').build();
  // @ts-expect-error -- neither is set
  request().header('a', 'b').build();
  // @ts-expect-error -- url set twice
  request().url('/a').url('/b');
  // @ts-expect-error -- method set twice
  request().method('GET').method('POST');
  // @ts-expect-error -- GET requests have no body
  request().url('/a').method('GET').body('x');
  // @ts-expect-error -- body before a method is chosen
  request().url('/a').body('x');
  // @ts-expect-error -- unknown method
  request().method('PATCH');
}
void compileOnly;
