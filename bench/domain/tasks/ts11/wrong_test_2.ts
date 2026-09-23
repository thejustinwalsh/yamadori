// The same brand for both types, so a UserId is accepted where an OrderId is required.
type Branded<T> = T & { readonly __branded: true };
export type UserId = Branded<string>;
export type OrderId = Branded<string>;
export function toUserId(raw: string): UserId {
  if (!/^u_[0-9a-z]+$/.test(raw)) throw new TypeError('bad');
  return raw as UserId;
}
export function toOrderId(raw: string): OrderId {
  if (!/^o_[0-9]+$/.test(raw)) throw new TypeError('bad');
  return raw as OrderId;
}
