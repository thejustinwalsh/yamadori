import assert from 'node:assert/strict';
import tgpu, { type TgpuFn } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { accumulate, bumpCounter } from './solution.ts';

type Args<F> = F extends TgpuFn<(...args: infer A) => any> ? A : never;
type _acc = Expect<Equal<Args<typeof accumulate>, [d.Ptr<'function', d.Vec3f, 'read-write'>, d.Vec3f, d.F32]>>;
type _bump = Expect<Equal<Args<typeof bumpCounter>, [d.Ptr<'private', d.U32, 'read-write'>]>>;
export const _neg = () => {
  // @ts-expect-error -- the counter pointer is private-scoped, not function-scoped
  const _x: [d.Ptr<'function', d.U32, 'read-write'>] = null! as Args<typeof bumpCounter>;
};

const acc = tgpu.resolve([accumulate]);
assert.match(acc,
  /fn\s+accumulate\s*\(\s*acc\s*:\s*ptr<\s*function\s*,\s*vec3f\s*(?:,\s*read_write\s*)?>\s*,\s*sample\s*:\s*vec3f\s*,\s*weight\s*:\s*f32\s*\)\s*\{/,
  acc);
assert.doesNotMatch(acc, /accumulate\s*\([^)]*\)\s*->/, 'no return type\n' + acc);
assert.match(acc, /\*\s*acc/, 'writes through the pointer\n' + acc);

const bump = tgpu.resolve([bumpCounter]);
assert.match(bump, /fn\s+bumpCounter\s*\(\s*counter\s*:\s*ptr<\s*private\s*,\s*u32\s*(?:,\s*read_write\s*)?>\s*\)\s*\{/, bump);
assert.match(bump, /\*\s*counter/, 'writes through the pointer\n' + bump);

// Used from a caller: the pointer is passed with &.
const caller = tgpu.fn([], d.vec3f)`() -> vec3f {
  var total = vec3f(0.0);
  accumulate(&total, vec3f(1.0, 2.0, 3.0), 0.5);
  return total;
}`.$uses({ accumulate }).$name('caller');
const both = tgpu.resolve([caller]);
assert.match(both, /fn\s+accumulate\s*\(/, both);
assert.match(both, /accumulate\s*\(\s*&total\s*,/, both);
