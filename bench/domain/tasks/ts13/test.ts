import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { all, andThen, err, map, ok, tryCatch, type Result } from './solution.ts';

const one = ok(1);
type _ok = Expect<Equal<typeof one, Result<number, never>>>;
const bad = err('boom' as const);
type _err = Expect<Equal<typeof bad, Result<never, 'boom'>>>;
assert.deepEqual(one, { ok: true, value: 1 });
assert.deepEqual(bad, { ok: false, error: 'boom' });

function narrow(r: Result<number, string>): string {
  if (r.ok) { type _v = Expect<Equal<typeof r.value, number>>; return `v${r.value}`; }
  type _e = Expect<Equal<typeof r.error, string>>;
  return `e${r.error}`;
}
assert.equal(narrow(ok(2)), 'v2');
assert.equal(narrow(err('x')), 'ex');

const parse = (s: string): Result<number, 'nan'> => (Number.isNaN(Number(s)) ? err('nan') : ok(Number(s)));
const positive = (n: number): Result<number, 'negative'> => (n >= 0 ? ok(n) : err('negative'));

const doubled = map(parse('21'), (n) => n * 2);
type _map = Expect<Equal<typeof doubled, Result<number, 'nan'>>>;
assert.deepEqual(doubled, { ok: true, value: 42 });
let called = false;
assert.deepEqual(map(parse('x'), (n) => { called = true; return n; }), { ok: false, error: 'nan' });
assert.equal(called, false, 'map must not call f on an error');

const chained = andThen(parse('-3'), positive);
type _chain = Expect<Equal<typeof chained, Result<number, 'nan' | 'negative'>>>;
assert.deepEqual(chained, { ok: false, error: 'negative' });
assert.deepEqual(andThen(parse('5'), positive), { ok: true, value: 5 });
assert.deepEqual(andThen(parse('?'), positive), { ok: false, error: 'nan' });

const t1 = tryCatch(() => JSON.parse('{"a":1}') as { a: number });
type _t1 = Expect<Equal<typeof t1, Result<{ a: number }, Error>>>;
assert.deepEqual(t1, { ok: true, value: { a: 1 } });
const t2 = tryCatch(() => JSON.parse('{'));
assert.ok(!t2.ok && t2.error instanceof SyntaxError, 'Error instances pass through');
const t3 = tryCatch((): number => { throw 'plain string'; });
assert.ok(!t3.ok && t3.error instanceof Error && t3.error.message === 'plain string', 'non-Error throws are wrapped');

const a: Result<number, 'a'> = ok(1);
const b: Result<string, 'b'> = ok('s');
const c: Result<boolean, 'c'> = err('c');
const both = all([a, b]);
type _all = Expect<Equal<typeof both, Result<[number, string], 'a' | 'b'>>>;
assert.deepEqual(both, { ok: true, value: [1, 's'] });
const three = all([a, c, b, err('late' as const)]);
type _all3 = Expect<Equal<typeof three, Result<[number, boolean, string, never], 'a' | 'c' | 'b' | 'late'>>>;
assert.deepEqual(three, { ok: false, error: 'c' }, 'first error wins');
const empty = all([]);
type _empty = Expect<Equal<typeof empty, Result<[], never>>>;
assert.deepEqual(empty, { ok: true, value: [] });
