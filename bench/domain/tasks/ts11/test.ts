import assert from 'node:assert/strict';
import type { Equal, Expect, IsAny } from './assert_types.ts';
import { toOrderId, toUserId, type OrderId, type UserId } from './solution.ts';

type _nu = Expect<Equal<IsAny<UserId>, false>>;
type _no = Expect<Equal<IsAny<OrderId>, false>>;
type _neq = Expect<Equal<Equal<UserId, OrderId>, false>>;

const u = toUserId('u_ab12');
const o = toOrderId('o_77');
type _tu = Expect<Equal<typeof u, UserId>>;
type _to = Expect<Equal<typeof o, OrderId>>;

const asString: string = u;           // a UserId is usable as a string
const len: number = o.length;         // ...and has string members
const lookup = new Map<UserId, number>([[u, 1]]);
function takesUser(id: UserId): string { return id; }
function takesOrder(id: OrderId): string { return id; }

assert.equal(typeof u, 'string');
assert.equal(u, 'u_ab12');
assert.equal(asString.toUpperCase(), 'U_AB12');
assert.equal(len, 4);
assert.equal(lookup.get(u), 1);
assert.equal(takesUser(u), 'u_ab12');
assert.equal(takesOrder(o), 'o_77');
assert.equal(JSON.stringify({ u }), '{"u":"u_ab12"}');

for (const bad of ['', 'u_', 'U_ab', 'x_1', 'u_a-b', ' u_a']) {
  assert.throws(() => toUserId(bad), TypeError, `toUserId(${JSON.stringify(bad)})`);
}
for (const bad of ['o_', 'o_a', 'u_1', 'o_1 ']) {
  assert.throws(() => toOrderId(bad), TypeError, `toOrderId(${JSON.stringify(bad)})`);
}

function compileOnly(): void {
  // @ts-expect-error -- a raw string is not a UserId
  const a: UserId = 'u_ab12';
  // @ts-expect-error -- a UserId is not an OrderId
  const b: OrderId = u;
  // @ts-expect-error -- an OrderId is not a UserId
  takesUser(o);
  // @ts-expect-error -- string concatenation yields a plain string
  takesUser(u + '');
  void a; void b;
}
void compileOnly;
