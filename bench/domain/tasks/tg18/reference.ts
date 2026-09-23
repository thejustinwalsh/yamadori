import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const accumulate = tgpu.fn([d.ptrFn(d.vec3f), d.vec3f, d.f32])`(acc: ptr<function, vec3f>, sample: vec3f, weight: f32) {
  *acc += sample * weight;
}`.$name('accumulate');

export const bumpCounter = tgpu.fn([d.ptrPrivate(d.u32)])`(counter: ptr<private, u32>) {
  *counter += 1u;
}`.$name('bumpCounter');
