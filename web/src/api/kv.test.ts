import { describe, expect, it } from 'vitest';
import { kvSplit } from './kv';

describe('kvSplit', () => {
  it('reads helpers from the payload (mcp/budget.py budgets(), 2026-09-22)', () => {
    const k = kvSplit({ pool: 147456, main: 73728, helper: 36864, helpers: 2, reserve: 0, gib: 6.19 })!;
    expect(k.helpers).toBe(2);
    expect(k.helpersReported).toBe(true);
    expect(k.main + k.helpers * k.helper + k.reserve).toBe(k.pool);
  });

  it('derives helpers from an older payload by its own sums, never a constant', () => {
    const k = kvSplit({ pool: 147456, main: 88473, helper: 36864, reserve: 22119, gib: 6.19 })!;
    expect(k.helpersReported).toBe(false);
    expect(k.helpers).toBe(1); // 147456 - 88473 - 22119 = 36864 = 1 x helper
    const two = kvSplit({ pool: 100, main: 50, helper: 25, reserve: 0, gib: 1 })!;
    expect(two.helpers).toBe(2); // same rule, a different shape
  });

  it('is null for an error or an unusable pool', () => {
    expect(kvSplit({ error: 'URLError' })).toBeNull();
    expect(kvSplit(null)).toBeNull();
    expect(kvSplit({ pool: 0, main: 0, helper: 0, reserve: 0, gib: 0 })).toBeNull();
    expect(kvSplit({ pool: 10, main: Number.NaN, helper: 1, reserve: 0, gib: 0 })).toBeNull();
  });
});
