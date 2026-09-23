import assert from 'node:assert/strict';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { Counters, type CountersInShader, type CountersOnHost } from './solution.ts';

type _schema = Expect<Equal<typeof Counters, d.WgslStruct<{
  hits: d.Atomic<d.U32>; misses: d.Atomic<d.U32>; balance: d.Atomic<d.I32>;
}>>>;
type _host = Expect<Equal<CountersOnHost, { hits: number; misses: number; balance: number }>>;
type _gpu = Expect<Equal<CountersInShader, { hits: d.atomicU32; misses: d.atomicU32; balance: d.atomicI32 }>>;
// @ts-expect-error -- inside a shader an atomic is not a plain number
const _n: number = null! as CountersInShader['hits'];

assert.equal(d.sizeOf(Counters), 12);
const wgsl = tgpu.resolve([Counters]);
assert.match(wgsl, /hits\s*:\s*atomic<\s*u32\s*>/, wgsl);
assert.match(wgsl, /misses\s*:\s*atomic<\s*u32\s*>/, wgsl);
assert.match(wgsl, /balance\s*:\s*atomic<\s*i32\s*>/, wgsl);
assert.match(wgsl, /hits[\s\S]*misses[\s\S]*balance/, 'field order');
