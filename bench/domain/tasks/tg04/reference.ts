import * as d from 'typegpu/data';

export const Counters = d.struct({
  hits: d.atomic(d.u32),
  misses: d.atomic(d.u32),
  balance: d.atomic(d.i32),
});

export type CountersOnHost = d.Infer<typeof Counters>;
export type CountersInShader = d.InferGPU<typeof Counters>;
