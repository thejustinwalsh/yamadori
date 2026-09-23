// The KV pool as the API reports it: one main context and N deep-thinking
// contexts of `helper` tokens each (mcp/budget.py budgets()). Pure, so the
// split the panel draws is testable.
import type { ContextPool } from './types';

export type KvSplit = {
  pool: number;
  main: number;
  /** tokens per deep-thinking context */
  helper: number;
  helpers: number;
  /** true when `helpers` came from the payload, false when derived from its own sums */
  helpersReported: boolean;
  reserve: number;
  gib: number;
};

const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

/**
 * null when the payload has no usable pool. `helpers` is read from the API;
 * a server that predates the field is answered from its own arithmetic
 * (pool - main - reserve) / helper, never from a constant.
 */
export function kvSplit(c: ContextPool | null | undefined): KvSplit | null {
  if (!c || !('pool' in c) || !finite(c.pool) || c.pool <= 0) return null;
  const { pool, main, helper, reserve } = c;
  if (![main, helper, reserve].every(finite)) return null;
  let helpers: number;
  let reported = false;
  if (finite(c.helpers) && c.helpers >= 0) {
    helpers = Math.floor(c.helpers);
    reported = true;
  } else {
    const rest = pool - main - reserve;
    helpers = helper > 0 && rest > 0 && rest % helper === 0 ? rest / helper : helper > 0 ? 1 : 0;
  }
  return { pool, main, helper, helpers, helpersReported: reported, reserve, gib: finite(c.gib) ? c.gib : 0 };
}
