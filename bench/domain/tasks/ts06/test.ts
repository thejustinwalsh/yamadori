import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { groupByStatus, totalsByStatus, type Order } from './solution.ts';

const data: Order[] = [
  { id: 'a', status: 'paid', total: 10 },
  { id: 'b', status: 'pending', total: 5 },
  { id: 'c', status: 'paid', total: 2.5 },
  { id: 'd', status: 'pending', total: 1 },
];
function* once(xs: Order[]): Generator<Order> { yield* xs; }

const g = groupByStatus(once(data));
type _g = Expect<Equal<typeof g, Map<'pending' | 'paid' | 'shipped', Order[]>>>;
assert.deepEqual([...g.keys()], ['paid', 'pending'], 'keys in first-seen order');
assert.deepEqual(g.get('paid')?.map((o) => o.id), ['a', 'c']);
assert.deepEqual(g.get('pending')?.map((o) => o.id), ['b', 'd']);
assert.equal(g.get('paid')?.[0], data[0], 'groups hold the original objects');
assert.equal(g.has('shipped'), false);

const t = totalsByStatus(once(data));
type _t = Expect<Equal<typeof t, Partial<Record<'pending' | 'paid' | 'shipped', number>>>>;
assert.deepEqual({ ...t }, { paid: 12.5, pending: 6 });
assert.equal('shipped' in t, false, 'absent statuses are absent');
assert.equal(t.shipped, undefined);

assert.equal(groupByStatus([]).size, 0);
assert.deepEqual({ ...totalsByStatus(new Set<Order>()) }, {});
assert.equal(data.length, 4);
