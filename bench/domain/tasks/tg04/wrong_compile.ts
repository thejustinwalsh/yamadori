// Uses d.atomicU32 / d.atomicI32 as schemas; in typegpu 0.12.5 those names are types only -- the schema is d.atomic(d.u32).
import * as d from 'typegpu/data';

export const Counters = d.struct({
  hits: d.atomicU32,
  misses: d.atomicU32,
  balance: d.atomicI32,
});

export type CountersOnHost = d.Infer<typeof Counters>;
export type CountersInShader = d.InferGPU<typeof Counters>;
