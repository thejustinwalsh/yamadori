import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { join, split, type Join, type Split } from './solution.ts';

type _s1 = Expect<Equal<Split<'a,b,c', ','>, ['a', 'b', 'c']>>;
type _s2 = Expect<Equal<Split<'abc', ','>, ['abc']>>;
type _s3 = Expect<Equal<Split<'', ','>, ['']>>;
type _s4 = Expect<Equal<Split<'a,,b', ','>, ['a', '', 'b']>>;
type _s5 = Expect<Equal<Split<',a,', ','>, ['', 'a', '']>>;
type _s6 = Expect<Equal<Split<'a::b::c', '::'>, ['a', 'b', 'c']>>;
type _s7 = Expect<Equal<Split<'2026-09-22', '-'>, ['2026', '09', '22']>>;
type _s8 = Expect<Equal<Split<string, ','>, string[]>>;
type _s9 = Expect<Equal<Split<'a,b', string>, string[]>>;

type _j1 = Expect<Equal<Join<['a', 'b', 'c'], '-'>, 'a-b-c'>>;
type _j2 = Expect<Equal<Join<[], '-'>, ''>>;
type _j3 = Expect<Equal<Join<['x'], ', '>, 'x'>>;
type _j4 = Expect<Equal<Join<['', ''], '/'>, '/'>>;
type _j5 = Expect<Equal<Join<readonly ['a', 'b'], ''>, 'ab'>>;
type _j6 = Expect<Equal<Join<string[], '-'>, string>>;
type _j7 = Expect<Equal<Join<['a', 'b'], string>, string>>;
type _rt = Expect<Equal<Join<Split<'a.b.c', '.'>, '/'>, 'a/b/c'>>;

const parts = split('x.y.z', '.');
type _p = Expect<Equal<typeof parts, ['x', 'y', 'z']>>;
assert.deepEqual(parts, ['x', 'y', 'z']);
assert.deepEqual(split('', ','), ['']);
assert.deepEqual(split(',a,', ','), ['', 'a', '']);
assert.deepEqual(split('a::b', '::'), ['a', 'b']);

const wide: string = 'p q';
const w = split(wide, ' ');
type _w = Expect<Equal<typeof w, string[]>>;
assert.deepEqual(w, ['p', 'q']);

const joined = join(['a', 'b', 'c'], '/');
type _jj = Expect<Equal<typeof joined, 'a/b/c'>>;
assert.equal(joined, 'a/b/c');
const none = join([], '+');
type _jn = Expect<Equal<typeof none, ''>>;
assert.equal(none, '');
const dyn: string[] = ['q', 'r'];
const jd = join(dyn, ',');
type _jd = Expect<Equal<typeof jd, string>>;
assert.equal(jd, 'q,r');

function compileOnly(): void {
  // @ts-expect-error -- join only takes strings
  join([1, 2], ',');
  // @ts-expect-error -- the literal result is exact
  const bad: 'a-b' = join(['a', 'b'], '+');
  void bad;
}
void compileOnly;
