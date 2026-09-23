declare const brand: unique symbol;
type Brand<T, B extends string> = T & { readonly [brand]: B };

export type UserId = Brand<string, 'UserId'>;
export type OrderId = Brand<string, 'OrderId'>;

export function toUserId(raw: string): UserId {
  if (!/^u_[0-9a-z]+$/.test(raw)) throw new TypeError(`not a user id: ${raw}`);
  return raw as UserId;
}

export function toOrderId(raw: string): OrderId {
  if (!/^o_[0-9]+$/.test(raw)) throw new TypeError(`not an order id: ${raw}`);
  return raw as OrderId;
}
