// Wraps the id in an object: nominal, but no longer a string at runtime.
export class UserId { constructor(readonly value: string) {} private readonly kind = 'user'; toString() { return this.value; } }
export class OrderId { constructor(readonly value: string) {} private readonly kind = 'order'; toString() { return this.value; } }
export function toUserId(raw: string): UserId {
  if (!/^u_[0-9a-z]+$/.test(raw)) throw new TypeError('bad');
  return new UserId(raw);
}
export function toOrderId(raw: string): OrderId {
  if (!/^o_[0-9]+$/.test(raw)) throw new TypeError('bad');
  return new OrderId(raw);
}
