// A hand-written Map loop, and totals derived from Object.groupBy.
type Status = 'pending' | 'paid' | 'shipped';
export interface Order { id: string; status: Status; total: number }

export function groupByStatus(orders: Iterable<Order>): Map<Status, Order[]> {
  const m = new Map<Status, Order[]>();
  for (const o of orders) {
    const bucket = m.get(o.status);
    if (bucket) bucket.push(o);
    else m.set(o.status, [o]);
  }
  return m;
}

export function totalsByStatus(orders: Iterable<Order>): Partial<Record<Status, number>> {
  const groups = Object.groupBy(orders, (o) => o.status);
  const out: Partial<Record<Status, number>> = {};
  for (const [k, list] of Object.entries(groups) as [Status, Order[]][]) {
    out[k] = list.reduce((s, o) => s + o.total, 0);
  }
  return out;
}
