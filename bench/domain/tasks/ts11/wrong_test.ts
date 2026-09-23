// Plain aliases: no nominal distinction at all.
export type UserId = string;
export type OrderId = string;
export function toUserId(raw: string): UserId {
  if (!/^u_[0-9a-z]+$/.test(raw)) throw new TypeError('bad');
  return raw;
}
export function toOrderId(raw: string): OrderId {
  if (!/^o_[0-9]+$/.test(raw)) throw new TypeError('bad');
  return raw;
}
