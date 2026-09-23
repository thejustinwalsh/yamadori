// Treats the Iterable as an array.
export type Order = { id: string; status: 'pending' | 'paid' | 'shipped'; total: number };
export function groupByStatus(orders: Iterable<Order>): Map<Order['status'], Order[]> {
  const m = new Map<Order['status'], Order[]>();
  orders.forEach((o) => m.set(o.status, [...(m.get(o.status) ?? []), o]));
  return m;
}
export function totalsByStatus(orders: Iterable<Order>): Partial<Record<Order['status'], number>> {
  return orders.reduce((acc, o) => ({ ...acc, [o.status]: (acc[o.status] ?? 0) + o.total }), {});
}
