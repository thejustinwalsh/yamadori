// A plain string-literal `__brand` property per type instead of a unique symbol.
export type UserId = string & { readonly __brand: 'UserId' };
export type OrderId = string & { readonly __brand: 'OrderId' };

const USER = /^u_[0-9a-z]+$/;
const ORDER = /^o_[0-9]+$/;

export const toUserId = (raw: string): UserId => {
  if (USER.test(raw)) return raw as UserId;
  throw new TypeError('invalid UserId');
};
export const toOrderId = (raw: string): OrderId => {
  if (ORDER.test(raw)) return raw as OrderId;
  throw new TypeError('invalid OrderId');
};
