// The KV pool as the API reports it (mcp/budget.py budgets()). Two layouts:
//
//   split  the pool divided: one main context, N helper contexts of
//          `helper` tokens each, and whatever the shares leave (reserve).
//   cap    THE CAP LAYOUT (operator, 2026-09-28): main is the tiered cache's
//          VRAM line (every conversation's window), the child slot -- the
//          decider lane since layout v2 (budget.child() names its role and
//          what it serves) -- has its own `helper` tokens
//          and swaps into VRAM while its conversation pauses, and the rest
//          of the pool is a second concurrent conversation's room, which may
//          spill to host RAM (budget.py docstring, slots RANKS).
//
// Pure, so the split the panels draw and the names they give it are testable.
import type { ContextPool } from './types';

export type KvLayout = 'cap' | 'split';

export type KvSplit = {
  pool: number;
  main: number;
  /** tokens per helper context (the child slot in the cap layout) */
  helper: number;
  helpers: number;
  /** true when `helpers` came from the payload, false when derived from its own sums */
  helpersReported: boolean;
  reserve: number;
  gib: number;
  /** null: a server older than the `layout` field (read as split) */
  layout: KvLayout | null;
  /** where the main cap came from (YAMADORI_MAIN_CAP, the served VRAM line), or why there is none */
  capSource: string | null;
  /** /props kv_vram_cells as served (engine patch 0041); null when not reported */
  vramLine: number | null;
  /** budget.child() role ("decider lane"); null on a server older than the field */
  childRole: string | null;
  /** what the child slot serves, as the server lists it; empty when not reported */
  childServes: string[];
};

/** What each part of the split is called, and what it is for. */
export type KvNames = {
  main: string;
  mainWhy: string;
  helper: string;
  helperWhy: string;
  /** the helper context's role in lower case, for prose ("decider lane") */
  helperRole: string;
  reserve: string;
  reserveWhy: string;
  /** the panel's tag */
  tag: string;
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
  const layout: KvLayout | null = c.layout === 'cap' || c.layout === 'split' ? c.layout : null;
  return {
    pool,
    main,
    helper,
    helpers,
    helpersReported: reported,
    reserve,
    gib: finite(c.gib) ? c.gib : 0,
    layout,
    capSource: typeof c.cap_source === 'string' && c.cap_source ? c.cap_source : null,
    vramLine: finite(c.vram_line) ? c.vram_line : null,
    childRole: c.child && typeof c.child.role === 'string' && c.child.role ? c.child.role : null,
    childServes: Array.isArray(c.child?.serves) ? c.child!.serves!.filter((x) => typeof x === 'string') : [],
  };
}

/**
 * The names of the parts. The child slot's role is the server's own word
 * (budget.child(): "decider lane" since layout v2); a server that predates
 * the field gets the neutral "helper".
 */
export function kvNames(kv: KvSplit): KvNames {
  const role = kv.childRole ?? 'helper';
  const serves = kv.childServes.length ? kv.childServes.join(', ') : null;
  if (kv.layout === 'cap') {
    return {
      main: 'MAIN · VRAM LINE',
      mainWhy: `every conversation's window${kv.capSource ? ` · ${kv.capSource}` : ''}`,
      helper: `CHILD · ${role.toUpperCase()}`,
      helperWhy: `${serves ?? role}; its own window, swapped into VRAM while its conversation pauses`,
      helperRole: role,
      reserve: 'SECOND CONVERSATION',
      reserveWhy: 'the rest of the pool: room for a second concurrent conversation, in host RAM when it spills',
      tag: 'CAP LAYOUT',
    };
  }
  return {
    main: 'MAIN',
    mainWhy: 'the conversation',
    helper: `${role.toUpperCase()} ×${kv.helpers}`,
    helperWhy: `${serves ?? role}, one at a time`,
    helperRole: role,
    reserve: 'RESERVE',
    reserveWhy: 'whatever the shares leave unclaimed',
    tag: `${kv.helpers + 1} CONTEXTS`,
  };
}
