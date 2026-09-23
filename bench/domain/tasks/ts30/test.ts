import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { camelizeKeys, type CamelCase, type CamelCaseKeys } from './solution.ts';

type _c1 = Expect<Equal<CamelCase<'user_id'>, 'userId'>>;
type _c2 = Expect<Equal<CamelCase<'street_line_1'>, 'streetLine1'>>;
type _c3 = Expect<Equal<CamelCase<'a_b_c'>, 'aBC'>>;
type _c4 = Expect<Equal<CamelCase<'already'>, 'already'>>;
type _c5 = Expect<Equal<CamelCase<'is_2fa_enabled'>, 'is2faEnabled'>>;

type Api = {
  user_id: number;
  display_name: string;
  home_address: { street_line_1: string; zip_code: string } | null;
  recent_orders: { order_id: string; line_items: { sku_code: string; qty: number }[] }[];
  tags: string[];
  is_admin?: boolean;
};

type _k = Expect<Equal<CamelCaseKeys<Api>, {
  userId: number;
  displayName: string;
  homeAddress: { streetLine1: string; zipCode: string } | null;
  recentOrders: { orderId: string; lineItems: { skuCode: string; qty: number }[] }[];
  tags: string[];
  isAdmin?: boolean;
}>>;
type _prim = Expect<Equal<CamelCaseKeys<string>, string>>;
type _top = Expect<Equal<CamelCaseKeys<{ a_b: 1 }[]>, { aB: 1 }[]>>;

const raw: Api = {
  user_id: 7,
  display_name: 'Ann',
  home_address: { street_line_1: '1 Main St', zip_code: '0150' },
  recent_orders: [
    { order_id: 'o_1', line_items: [{ sku_code: 'x_y', qty: 2 }] },
    { order_id: 'o_2', line_items: [] },
  ],
  tags: ['snake_case_value'],
};
const snapshot = JSON.stringify(raw);

const out = camelizeKeys(raw);
type _o = Expect<Equal<typeof out, CamelCaseKeys<Api>>>;
assert.deepEqual(out, {
  userId: 7,
  displayName: 'Ann',
  homeAddress: { streetLine1: '1 Main St', zipCode: '0150' },
  recentOrders: [
    { orderId: 'o_1', lineItems: [{ skuCode: 'x_y', qty: 2 }] },
    { orderId: 'o_2', lineItems: [] },
  ],
  tags: ['snake_case_value'],
});
assert.equal(JSON.stringify(raw), snapshot, 'the input is not modified');
assert.equal(out.recentOrders[0]!.lineItems[0]!.skuCode, 'x_y', 'values are never renamed');

assert.deepEqual(camelizeKeys({ ...raw, home_address: null }).homeAddress, null);
assert.deepEqual(camelizeKeys([{ a_b: 1 }, { a_b: 2 }]), [{ aB: 1 }, { aB: 2 }]);
assert.equal(camelizeKeys('user_id'), 'user_id');
assert.equal(camelizeKeys(null), null);

function compileOnly(): void {
  // @ts-expect-error -- the snake_case key no longer exists
  out.user_id;
  // @ts-expect-error -- nested keys are converted too
  out.recentOrders[0]!.order_id;
}
void compileOnly;
