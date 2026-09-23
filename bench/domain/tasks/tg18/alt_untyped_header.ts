import { tgpu, d } from 'typegpu';

// Explicit void return, and a header without types: typegpu fills them in from the shell.
export const accumulate = tgpu
  .fn([d.ptrFn(d.vec3f), d.vec3f, d.f32], d.Void)('(acc, sample, weight) { *acc = *acc + sample * weight; }')
  .$name('accumulate');

export const bumpCounter = tgpu.fn([d.ptrPrivate(d.u32)], d.Void)('(counter) { *counter += 1u; }');
bumpCounter.$name('bumpCounter');
