import { describe, expect, it } from 'vitest';
import { kvNames, kvSplit } from './kv';

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

  it('reads the layout, the cap source and the served VRAM line (2026-09-28)', () => {
    const k = kvSplit({ pool: 262144, main: 141824, helper: 65536, helpers: 1, reserve: 54784, gib: 11, layout: 'cap', cap_source: 'YAMADORI_MAIN_CAP', vram_line: null })!;
    expect(k.layout).toBe('cap');
    expect(k.capSource).toBe('YAMADORI_MAIN_CAP');
    expect(k.vramLine).toBeNull();
    const old = kvSplit({ pool: 100, main: 50, helper: 25, reserve: 0, gib: 1 })!;
    expect(old.layout).toBeNull(); // a server older than the field reads as split
  });
});

describe('kvNames', () => {
  const child = { role: 'decider lane', tokens: 3072, serves: ['the decider', 'small side calls (titles)'] };
  const cap = kvSplit({ pool: 262144, main: 141824, helper: 65536, helpers: 1, reserve: 54784, gib: 11, layout: 'cap', cap_source: 'llama-server /props kv_vram_cells', child })!;
  const split = kvSplit({ pool: 147456, main: 73728, helper: 36864, helpers: 2, reserve: 0, gib: 6.19, layout: 'split', child })!;

  it("cap layout: never calls the rest a reserve, and the child is named by the server's role (layout v2)", () => {
    const nm = kvNames(cap);
    expect(nm.main).toBe('MAIN · VRAM LINE');
    expect(nm.mainWhy).toContain('llama-server /props kv_vram_cells');
    expect(nm.helper).toBe('CHILD · DECIDER LANE');
    expect(nm.helperWhy).toContain('the decider, small side calls (titles)');
    expect(nm.helper).not.toContain('DEEP');
    expect(nm.reserve).toBe('SECOND CONVERSATION');
    expect(nm.tag).toBe('CAP LAYOUT');
  });

  it('split layout: main, N helper contexts and the reserve', () => {
    const nm = kvNames(split);
    expect(nm.helper).toBe('DECIDER LANE ×2');
    expect(nm.reserve).toBe('RESERVE');
    expect(nm.tag).toBe('3 CONTEXTS');
  });

  it('a server older than budget.child() gets the neutral "helper", never "deep thinking"', () => {
    const old = kvSplit({ pool: 262144, main: 141824, helper: 65536, helpers: 1, reserve: 54784, gib: 11, layout: 'cap' })!;
    expect(kvNames(old).helper).toBe('CHILD · HELPER');
  });
});
