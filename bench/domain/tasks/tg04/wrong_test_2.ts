// Uses the host-side inference helper for the shader-side type too, so atomics show up as plain numbers.
import * as d from 'typegpu/data';

export const Counters = d.struct({
  hits: d.atomic(d.u32),
  misses: d.atomic(d.u32),
  balance: d.atomic(d.i32),
});

export type CountersOnHost = d.Infer<typeof Counters>;
export type CountersInShader = d.Infer<typeof Counters>;
