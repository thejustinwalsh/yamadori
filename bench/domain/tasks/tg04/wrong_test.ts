// Plain u32/i32 fields: WGSL atomicAdd needs atomic<u32>/atomic<i32> members, and the GPU-side type loses atomicity.
import * as d from 'typegpu/data';

export const Counters = d.struct({
  hits: d.u32,
  misses: d.u32,
  balance: d.i32,
});

export type CountersOnHost = d.Infer<typeof Counters>;
export type CountersInShader = d.InferGPU<typeof Counters>;
