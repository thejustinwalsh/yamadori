import assert from 'node:assert/strict';
import tgpu, { type TgpuComputeFn } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { integrate, Particle, SimParams, simLayout } from './solution.ts';

const _fn: TgpuComputeFn = integrate;
type _keys = Expect<Equal<keyof (typeof simLayout)['entries'], 'params' | 'particles'>>;
type _par = Expect<Equal<(typeof simLayout)['$']['particles'], d.InferGPU<typeof Particle>[]>>;
type _prm = Expect<Equal<(typeof simLayout)['$']['params'], d.InferGPU<typeof SimParams>>>;
export const _neg = () => {
  // @ts-expect-error -- an entry point is not callable from JS like a plain function
  integrate();
};

assert.deepEqual([...integrate.shell.workgroupSize].concat([1, 1]).slice(0, 3), [64, 1, 1], 'workgroup size');

const wgsl = tgpu.resolve([integrate]);
const entry = /@compute\s+@workgroup_size\(\s*64\s*(?:,\s*1\s*){0,2}\)\s*fn\s+integrate\s*\(((?:[^()]|\([^()]*\))*)\)\s*\{([\s\S]*)\}\s*$/.exec(wgsl);
assert.ok(entry, 'expected `@compute @workgroup_size(64) fn integrate(...) { ... }` at the end\n' + wgsl);
const [, params, body] = entry;
assert.match(params!, /^\s*@builtin\(\s*global_invocation_id\s*\)\s*\w+\s*:\s*vec3u\s*$/, 'the only parameter is the global invocation id\n' + wgsl);
assert.doesNotMatch(body!, /@builtin|->/, 'body must not contain a second header\n' + wgsl);

assert.match(wgsl, /struct\s+Particle\s*\{\s*pos\s*:\s*vec3f\s*,\s*vel\s*:\s*vec3f\s*,?\s*\}/, wgsl);
assert.match(wgsl, /struct\s+SimParams\s*\{\s*dt\s*:\s*f32\s*,\s*count\s*:\s*u32\s*,?\s*\}/, wgsl);
assert.match(wgsl, /@group\(\s*\d+\s*\)\s*@binding\(\s*0\s*\)\s*var<\s*uniform\s*>\s*params\s*:\s*SimParams\s*;/, wgsl);
assert.match(wgsl, /@group\(\s*\d+\s*\)\s*@binding\(\s*1\s*\)\s*var<\s*storage\s*,\s*read_write\s*>\s*particles\s*:\s*array<\s*Particle\s*>\s*;/, wgsl);
assert.match(body!, /\bparams\.count\b/, 'reads params.count\n' + wgsl);
assert.match(body!, /\bparams\.dt\b/, 'reads params.dt\n' + wgsl);
assert.match(body!, /\bparticles\s*\[/, 'indexes particles\n' + wgsl);
assert.match(body!, /\breturn\b/, 'early return\n' + wgsl);
assert.doesNotMatch(body!, /layout\.|\$\./, 'no unresolved layout accessors left in the body\n' + wgsl);
