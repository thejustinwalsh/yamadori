export type Order = { id: string; status: 'pending' | 'paid' | 'shipped'; total: number };

export function groupByStatus(orders: Iterable<Order>): Map<Order['status'], Order[]> {
  return Map.groupBy(orders, (o) => o.status);
}

export function totalsByStatus(orders: Iterable<Order>): Partial<Record<Order['status'], number>> {
  const out: Partial<Record<Order['status'], number>> = {};
  for (const o of orders) out[o.status] = (out[o.status] ?? 0) + o.total;
  return out;
}
