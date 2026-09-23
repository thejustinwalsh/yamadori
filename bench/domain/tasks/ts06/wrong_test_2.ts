// Iterates the input twice; a one-shot iterable (a generator) is empty the second time.
export type Order = { id: string; status: 'pending' | 'paid' | 'shipped'; total: number };
export function groupByStatus(orders: Iterable<Order>): Map<Order['status'], Order[]> {
  const m = new Map<Order['status'], Order[]>();
  for (const o of orders) m.set(o.status, []);
  for (const o of orders) m.get(o.status)!.push(o);
  return m;
}
export function totalsByStatus(orders: Iterable<Order>): Partial<Record<Order['status'], number>> {
  const out: Partial<Record<Order['status'], number>> = {};
  for (const o of orders) out[o.status] = (out[o.status] ?? 0) + o.total;
  return out;
}
