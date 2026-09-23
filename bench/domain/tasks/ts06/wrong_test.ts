// Pre-fills every status with 0, so statuses with no orders are present.
export type Order = { id: string; status: 'pending' | 'paid' | 'shipped'; total: number };
export function groupByStatus(orders: Iterable<Order>): Map<Order['status'], Order[]> {
  return Map.groupBy(orders, (o) => o.status);
}
export function totalsByStatus(orders: Iterable<Order>): Partial<Record<Order['status'], number>> {
  const out = { pending: 0, paid: 0, shipped: 0 };
  for (const o of orders) out[o.status] += o.total;
  return out;
}
